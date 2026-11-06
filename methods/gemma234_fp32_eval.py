"""FP32 full-evaluation of one representative Gemma-3-4B-it checkpoint (experiment log 234 / 235): A-FP32 scorer (FP32 weights and adapter, TF32 off, eval, no_grad, sdpa, batch 1; the scorer functions esci_gemma_core.score_batch / gemma_r7_eval.score_batch of the original evaluations) over the COMPLETE evaluation sets: ESCI validation (200 queries / 4,047 pairs) + test300 (300 / 6,078); MS MARCO TREC-DL 2019 (42 / 1,102) + 2020 (54 / 1,432). No training, no threshold search.
Per pair it stores [label log-prob, <end_of_turn> log-prob] of Relevant and Irrelevant (ell = label + eot; m = ell1 - ell0), in the order of the evaluation pool.   --mode full: all pairs;  --mode verify: (i) the first 64 pairs of the 233 sample order per test set are compared with the stored 233 AF scores, (ii) ESCI validation: the first 64 pairs in the order of SHA256('231:' + canonical query id) are scored with A-FP32 and with the official forward (B-FP32);  --max_pairs N: timing test on the first N pairs of every set (nothing durable is claimed).
    CUDA_VISIBLE_DEVICES=0 python gemma234_fp32_eval.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds ESCI --T 1 --loss CE --mode full --out cpmr_experiment/gemma234/out"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
for rel in ("", "methods", "evaluation", "cpmr_experiment/scripts"):
    sys.path.insert(0, str(ROOT / rel))
import torch
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True); ap.add_argument("--ds", required=True, choices=["ESCI", "MS"]); ap.add_argument("--T", type=int, required=True); ap.add_argument("--loss", required=True, choices=["CE", "SOFT", "ANY"])
ap.add_argument("--mode", default="full", choices=["full", "verify"]); ap.add_argument("--out", required=True); ap.add_argument("--max_pairs", type=int, default=0)
ap.add_argument("--sets", default="", help="comma separated subset of ESCIval,ESCItest,MS2019,MS2020 (default: all of the dataset)")
args = ap.parse_args()
t0 = time.time()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
man = json.load(open(ROOT / args.manifest, encoding="utf-8"))
ck = next(c for c in man["checkpoints"] if c["dataset"] == args.ds and c["T"] == args.T and c["loss"] == args.loss)
from layer_drift_repr import load_adapter
from src.model import load_model_and_tokenizer

POOLS = {}
if args.ds == "ESCI":
    import esci_gemma_core as core_mod
    cfg = yaml.safe_load(open(ROOT / "configs/gemma3_4b_esci.yaml"))
    instr = {e["display_id"]: e["full"] for e in json.load(open(ROOT / "cpmr_experiment/esci_t/esci_t_manifest.json", encoding="utf-8"))["entries"]}[f"T{args.T}"]
    ML = 704
    for name, fn in (("ESCIval", "sel_rows.jsonl"), ("ESCItest", "test300_rows.jsonl")):
        rows = {}
        for line in open(ROOT / "cpmr_experiment/esci/audit" / fn, encoding="utf-8"):
            r = json.loads(line)
            e = rows.setdefault(r["nq"], {"query": r["query"], "docs": [], "text": {}, "gold": {}})
            e["docs"].append(r["product_id"]); e["text"][r["product_id"]] = r["product_text"]; e["gold"][r["product_id"]] = int(r["label"] == "Relevant")
        POOLS[name] = rows
else:
    import gemma_r7_eval as core_mod
    from src.eval_rerank import load_test_pool
    cfg = yaml.safe_load(open(ROOT / "configs/gemma3_4b_msm.yaml"))
    man_i = json.load(open(ROOT / "cpmr_experiment/instructions_v2/manifest/instructions_v2.json", encoding="utf-8"))
    ins = {k: v["body"] + man_i["output_suffix"] for k, v in man_i["entries"].items() if "body" in v}
    ins.update(json.load(open(ROOT / "methods/custom_instructions_T6_T7.json", encoding="utf-8")))
    instr = ins[f"T{args.T}"]
    ML = 384
    for y in ("2019", "2020"):
        queries, cand_ids, doc_text, qrels = load_test_pool(year=y)
        judged = {q: [d for d in c if qrels.get(q, {}).get(d, -1) >= 0] for q, c in cand_ids.items() if q in queries}
        judged = {q: v for q, v in judged.items() if v}
        POOLS[f"MS{y}"] = {str(q): {"query": queries[q], "docs": [str(d) for d in v], "text": {str(d): doc_text[d] for d in v}, "gold": {str(d): int(qrels[q][d] >= 2) for d in v}} for q, v in judged.items()}
build, score_batch = core_mod.build, core_mod.score_batch
EOT, PAD = 106, 0
want = [s for s in (args.sets.split(",") if args.sets else list(POOLS)) if s in POOLS]
lm = load_model_and_tokenizer(model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=0.0, target_modules=cfg["lora"]["target_modules"], gradient_checkpointing=False, local_files_only=True, dtype=torch.float32)
model, tok = lm.model, lm.tokenizer
ckpt_obj = load_adapter(model, ROOT / ck["path"])
model.eval()
device = next(model.parameters()).device
dtypes = sorted({str(p.dtype) for p in model.parameters()})
assert dtypes == ["torch.float32"], dtypes
inner = model.base_model.model
core_ = inner.model
norm = core_.language_model.norm
w_head = inner.lm_head.weight.detach().float()
store = {}
norm.register_forward_pre_hook(lambda m, a: store.__setitem__("h", a[0].detach()))
meta = dict(checkpoint=ck, ds=args.ds, T=args.T, loss=args.loss, mode=args.mode, instruction_sha256=hashlib.sha256(instr.encode()).hexdigest(), max_length=ML, dtypes=dtypes, tf32_matmul_allowed=torch.backends.cuda.matmul.allow_tf32, attn_implementation=model.config._attn_implementation,
            adapter_global_step=ckpt_obj.get("global_step"), code_sha256={f: hashlib.sha256(open(ROOT / f, "rb").read()).hexdigest() for f in ("methods/gemma234_fp32_eval.py", "methods/esci_gemma_core.py", "methods/gemma_r7_eval.py")}, torch=torch.__version__)


def score_pair(q, d, e):
    r, shr = score_batch(core_, norm, w_head, store, tok, instr, e["query"], [d], e["text"], ML, device)
    assert shr == 0
    return [r[d]["Relevant"]["label"], r[d]["Relevant"]["eot"], r[d]["Irrelevant"]["label"], r[d]["Irrelevant"]["eot"]]


def ell_official(q, d, e):
    out = []
    for label in ("Relevant", "Irrelevant"):
        pre, resp, _ = build(tok, instr, e["query"], e["text"][d], label, ML)
        s = pre + resp + [EOT]
        ids = torch.tensor([s], device=device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            st, n = len(pre), len(resp)
            lp = torch.log_softmax(logits[0, st - 1: st + n].float(), dim=-1)
            v = lp.gather(1, torch.tensor(s[st: st + n + 1], device=device).unsqueeze(-1)).squeeze(-1).cpu().tolist()
        out += [float(sum(v[:n])), float(v[n])]
    return out


Path(ROOT / args.out).mkdir(parents=True, exist_ok=True)
tag = ck["tag"]
if args.mode == "full":
    for name in want:
        pool = POOLS[name]
        pairs, vals = [], []
        ts = time.time()
        for q, e in pool.items():
            for d in e["docs"]:
                if args.max_pairs and len(pairs) >= args.max_pairs:
                    break
                pairs.append([str(q), str(d), e["gold"][d]]); vals.append(score_pair(q, d, e))
            if args.max_pairs and len(pairs) >= args.max_pairs:
                break
        dt = time.time() - ts
        print(f"[fp32] {args.ds} T{args.T} {args.loss} {name}: {len(pairs)} pairs, {2 * len(pairs)} sequences, {dt:.0f}s ({dt / (2 * len(pairs)):.3f} s/sequence)", flush=True)
        json.dump(dict(meta=meta, set=name, pairs=pairs, scores=vals, seconds=dt), open(ROOT / args.out / f"fp32_{tag}__{name}{'__debug' if args.max_pairs else ''}.json", "w"))
else:
    res = {}
    for name in want:
        pool = POOLS[name]
        if name == "ESCIval":
            qs = sorted(pool, key=lambda q: hashlib.sha256(("231:" + str(q)).encode("utf-8")).hexdigest())
            sub = [(q, d) for q in qs for d in pool[q]["docs"]][:64]
        else:
            sm = man["sets"]["ESCI" if name == "ESCItest" else name]["fp32_subset_first64"]
            sub = [(q, d) for q, d in sm]
        a_new = [score_pair(q, d, pool[q]) for q, d in sub]
        rec = dict(set=name, pairs=[[q, d] for q, d in sub], A_new=a_new)
        if name == "ESCIval":
            rec["B_official_fp32"] = [ell_official(q, d, pool[q]) for q, d in sub]
        else:
            old = json.load(open(ROOT / "cpmr_experiment/gemma231/out" / f"audit_{tag}__{'ESCI' if name == 'ESCItest' else name}.json"))
            rec["AF_233"] = old["paths"]["AF"]; rec["BF_233"] = old["paths"]["BF"]
            assert [[p[0], p[1]] for p in old["pairs"][:64]] == [[q, d] for q, d in sub], "233 first-64 order mismatch"
        res[name] = rec
        print(f"[fp32] verify {args.ds} T{args.T} {args.loss} {name} done {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(meta=meta, results=res), open(ROOT / args.out / f"verify_{tag}.json", "w"))
meta["seconds"] = time.time() - t0
print(f"[fp32] done {args.ds} T{args.T} {args.loss} {args.mode} {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
