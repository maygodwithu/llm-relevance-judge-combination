"""Final-checkpoint scorer audit of Gemma-3-4B-it (experiment log 231 / 232): one checkpoint, its evaluation set(s), on the fixed sample of audit/sample_manifest.json. No threshold search, no training.
Paths (all on the SAME trained adapter loaded with layer_drift_repr.load_adapter; eval mode; no_grad; sdpa):
  A4  original scorer function (esci_gemma_core.score_batch / gemma_r7_eval.score_batch), batch 4 documents per label, right padding, query-wise chunking of the original evaluation (reproduces the stored raw scores)
  A1  the same function, batch 1
  B4  official transformers forward (Gemma3ForConditionalGeneration logits, bf16 forward, float log_softmax), same batch composition / padding as A4
  B1  official forward, batch 1
  AF / BF  the A (batch 1) and B (batch 1) paths of an FP32 model (FP32 weights and adapter, TF32 off), on the first 64 pairs of the sample order
Every path returns for each pair and label: sum of the label-token log-probabilities, the log-probability of <end_of_turn>; ell = label + eot; m = ell(Relevant) - ell(Irrelevant).
Output: one JSON per (checkpoint, evaluation set) with per-row scores of every path, the stored scores of the original evaluation, hashes and settings.
    CUDA_VISIBLE_DEVICES=0 python gemma231_audit.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds ESCI --T 1 --loss CE --out cpmr_experiment/gemma231/out"""
import argparse
import gzip
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
ap.add_argument("--out", required=True); ap.add_argument("--max_pairs", type=int, default=0, help="debug only: use the first N pairs of every set (nothing durable is claimed)")
args = ap.parse_args()
t0 = time.time()
man = json.load(open(ROOT / args.manifest, encoding="utf-8"))
ck = next(c for c in man["checkpoints"] if c["dataset"] == args.ds and c["T"] == args.T and c["loss"] == args.loss)
from layer_drift_repr import load_adapter
from src.model import load_model_and_tokenizer

if args.ds == "ESCI":
    import esci_gemma_core as core_mod
    cfg = yaml.safe_load(open(ROOT / "configs/gemma3_4b_esci.yaml"))
    instr = {e["display_id"]: e["full"] for e in json.load(open(ROOT / "cpmr_experiment/esci_t/esci_t_manifest.json", encoding="utf-8"))["entries"]}[f"T{args.T}"]
    ML = 704
    rows = {}
    for line in open(ROOT / "cpmr_experiment/esci/audit/test300_rows.jsonl", encoding="utf-8"):
        r = json.loads(line)
        e = rows.setdefault(r["nq"], {"query": r["query"], "docs": []})
        e["docs"].append(r["product_id"]); e.setdefault("text", {})[r["product_id"]] = r["product_text"]; e.setdefault("gold", {})[r["product_id"]] = r["esci"]
    SETS = {"ESCI": dict(pool=rows, stored=ROOT / f"cpmr_experiment/esci_gemma/eval/esciGemmaEval_{ck['tag']}_step625__T{args.T}__test300.json.gz")}
else:
    import gemma_r7_eval as core_mod
    from src.eval_rerank import load_test_pool
    cfg = yaml.safe_load(open(ROOT / "configs/gemma3_4b_msm.yaml"))
    man_i = json.load(open(ROOT / "cpmr_experiment/instructions_v2/manifest/instructions_v2.json", encoding="utf-8"))
    ins = {k: v["body"] + man_i["output_suffix"] for k, v in man_i["entries"].items() if "body" in v}
    ins.update(json.load(open(ROOT / "methods/custom_instructions_T6_T7.json", encoding="utf-8")))
    instr = ins[f"T{args.T}"]
    ML = 384
    SETS = {}
    for y in ("2019", "2020"):
        queries, cand_ids, doc_text, qrels = load_test_pool(year=y)
        judged = {q: [d for d in c if qrels.get(q, {}).get(d, -1) >= 0] for q, c in cand_ids.items() if q in queries}
        judged = {q: v for q, v in judged.items() if v}
        pool = {str(q): {"query": queries[q], "docs": list(v), "text": {d: doc_text[d] for d in v}, "gold": {d: qrels[q][d] for d in v}, "_qkey": q} for q, v in judged.items()}
        SETS[f"MS{y}"] = dict(pool=pool, stored=ROOT / f"cpmr_experiment/msm_gemma/eval/msmGemmaEval_{ck['tag']}__T{args.T}__{y}.json.gz")
EOT, PAD = 106, 0
build, score_batch = core_mod.build, core_mod.score_batch


def load_model(dtype):
    lm = load_model_and_tokenizer(model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=0.0, target_modules=cfg["lora"]["target_modules"], gradient_checkpointing=False, local_files_only=True, dtype=dtype)
    m = lm.model
    ckpt = load_adapter(m, ROOT / ck["path"])
    m.eval()
    return m, lm.tokenizer, ckpt


def ell_official(model, tok, qtext, dtexts, batch_docs, ml, device, bf):
    """official forward logits -> (label sum, eot) per doc and label for the docs in batch_docs (one padded batch per label)"""
    out = {d: {} for d in batch_docs}
    for label in ("Relevant", "Irrelevant"):
        ex = [build(tok, instr, qtext, dtexts[d], label, ml) for d in batch_docs]
        seqs = [e[0] + e[1] + [EOT] for e in ex]
        mx = max(len(s) for s in seqs)
        ids = torch.tensor([s + [PAD] * (mx - len(s)) for s in seqs], device=device)
        att = torch.tensor([[1] * len(s) + [0] * (mx - len(s)) for s in seqs], device=device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits
            for j, (e, s) in enumerate(zip(ex, seqs)):
                st, n = len(e[0]), len(e[1])
                lp = torch.log_softmax(logits[j, st - 1: st + n].float(), dim=-1)           # positions predicting the n label tokens and <end_of_turn>
                tg = torch.tensor(s[st: st + n + 1], device=device)
                v = lp.gather(1, tg.unsqueeze(-1)).squeeze(-1).cpu().tolist()
                out[batch_docs[j]][label] = (float(sum(v[:n])), float(v[n]))
    return out


def run_paths(model, tok, pool, order, chunks, device, mode):
    """mode 'bf16': A4, A1, B4, B1 for all pairs; returns {path: {(q,d): [l1,e1,l0,e0]}}"""
    inner = model.base_model.model
    core_ = inner.model
    norm = core_.language_model.norm
    w_head = inner.lm_head.weight.detach().float()
    store = {}
    hk = norm.register_forward_pre_hook(lambda m, a: store.__setitem__("h", a[0].detach()))
    res = {p: {} for p in ("A4", "A1", "B4", "B1")} if mode == "bf16" else {p: {} for p in ("AF", "BF")}
    for q, e in pool.items():
        todo = [d for d in e["docs"] if (q, str(d)) in order]
        if not todo:
            continue
        docs = e["docs"]
        batches = [docs[i:i + 4] for i in range(0, len(docs), 4)]
        for b in batches:
            if mode == "bf16":
                r, _ = score_batch(core_, norm, w_head, store, tok, instr, e["query"], b, e["text"], ML, device)
                for d in b:
                    if (q, str(d)) in order:
                        res["A4"][(q, str(d))] = [r[d]["Relevant"]["label"], r[d]["Relevant"]["eot"], r[d]["Irrelevant"]["label"], r[d]["Irrelevant"]["eot"]]
                o = ell_official(model, tok, e["query"], e["text"], b, ML, device, True)
                for d in b:
                    if (q, str(d)) in order:
                        res["B4"][(q, str(d))] = [o[d]["Relevant"][0], o[d]["Relevant"][1], o[d]["Irrelevant"][0], o[d]["Irrelevant"][1]]
        for d in todo:
            if mode == "bf16":
                r, _ = score_batch(core_, norm, w_head, store, tok, instr, e["query"], [d], e["text"], ML, device)
                res["A1"][(q, str(d))] = [r[d]["Relevant"]["label"], r[d]["Relevant"]["eot"], r[d]["Irrelevant"]["label"], r[d]["Irrelevant"]["eot"]]
                o = ell_official(model, tok, e["query"], e["text"], [d], ML, device, True)
                res["B1"][(q, str(d))] = [o[d]["Relevant"][0], o[d]["Relevant"][1], o[d]["Irrelevant"][0], o[d]["Irrelevant"][1]]
            else:
                r, _ = score_batch(core_, norm, w_head, store, tok, instr, e["query"], [d], e["text"], ML, device)
                res["AF"][(q, str(d))] = [r[d]["Relevant"]["label"], r[d]["Relevant"]["eot"], r[d]["Irrelevant"]["label"], r[d]["Irrelevant"]["eot"]]
                o = ell_official(model, tok, e["query"], e["text"], [d], ML, device, False)
                res["BF"][(q, str(d))] = [o[d]["Relevant"][0], o[d]["Relevant"][1], o[d]["Irrelevant"][0], o[d]["Irrelevant"][1]]
    hk.remove()
    return res


device = "cuda"
sets_run = list(SETS)
mset = {"ESCI": "ESCI", "MS2019": "MS2019", "MS2020": "MS2020"}
meta = dict(checkpoint=ck, ds=args.ds, T=args.T, loss=args.loss, instruction_sha256=hashlib.sha256(instr.encode()).hexdigest(), max_length=ML, code_sha256=man["code_sha256"], torch=torch.__version__, sample_manifest_sha256={k: man["sets"][k]["manifest_sha256"] for k in sets_run})
# ---------------- bf16 model
model, tok, ckpt_obj = load_model(torch.bfloat16)
meta["dtypes_bf16"] = sorted({str(p.dtype) for p in model.parameters()})
meta["attn_implementation"] = model.config._attn_implementation
meta["adapter_global_step"] = ckpt_obj.get("global_step")
out = {name: dict(meta=meta, pairs=man["sets"][name]["pairs"]) for name in sets_run}
orders = {}
for name in sets_run:
    pairs = man["sets"][name]["pairs"][: args.max_pairs] if args.max_pairs else man["sets"][name]["pairs"]
    orders[name] = {(p[0], p[1]): i for i, p in enumerate(pairs)}
    out[name]["pairs"] = pairs
    pool = SETS[name]["pool"]
    res = run_paths(model, tok, pool, orders[name], None, device, "bf16")
    out[name]["paths"] = {p: [res[p][(q, d)] for q, d, _ in pairs] for p in res}
    # stored scores of the original evaluation
    st = json.load(gzip.open(SETS[name]["stored"], "rt", encoding="utf-8"))
    if args.ds == "ESCI":
        out[name]["stored"] = [[st["raw"][q][d][0], st["raw"][q][d][1], st["raw"][q][d][3], st["raw"][q][d][4]] for q, d, _ in pairs]
        out[name]["stored_kind"] = "label and eot log-probabilities of Relevant / Irrelevant [l1, e1, l0, e0]"
    else:
        out[name]["stored"] = [[st["run_fp32"][q][d]] for q, d, _ in pairs]
        out[name]["stored_kind"] = "margin m only"
    out[name]["stored_file_sha256"] = hashlib.sha256(open(SETS[name]["stored"], "rb").read()).hexdigest()
    print(f"[audit] {args.ds} T{args.T} {args.loss} {name}: bf16 paths done ({len(pairs)} pairs) {time.time() - t0:.0f}s", flush=True)
del model
torch.cuda.empty_cache()
# ---------------- FP32 reference (TF32 off)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
model, tok, _ = load_model(torch.float32)
meta["dtypes_fp32"] = sorted({str(p.dtype) for p in model.parameters()})
meta["tf32_matmul_allowed"] = torch.backends.cuda.matmul.allow_tf32
for name in sets_run:
    pairs = man["sets"][name]["pairs"][: args.max_pairs] if args.max_pairs else man["sets"][name]["pairs"]
    sub = pairs[:64]
    order = {(p[0], p[1]): i for i, p in enumerate(sub)}
    res = run_paths(model, tok, SETS[name]["pool"], order, None, device, "fp32")
    out[name]["paths"]["AF"] = [res["AF"][(q, d)] for q, d, _ in sub]
    out[name]["paths"]["BF"] = [res["BF"][(q, d)] for q, d, _ in sub]
    out[name]["fp32_subset_pairs"] = len(sub)
    print(f"[audit] {args.ds} T{args.T} {args.loss} {name}: FP32 paths done ({len(sub)} pairs) {time.time() - t0:.0f}s", flush=True)
meta["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
meta["seconds"] = time.time() - t0
Path(ROOT / args.out).mkdir(parents=True, exist_ok=True)
for name in sets_run:
    out[name]["meta"] = meta
    json.dump(out[name], open(ROOT / args.out / f"audit_{ck['tag']}__{name}{'__debug' if args.max_pairs else ''}.json", "w"))
print(f"[audit] done {args.ds} T{args.T} {args.loss} {time.time() - t0:.0f}s peak {meta['peak_memory_gb']:.1f} GB", flush=True)
