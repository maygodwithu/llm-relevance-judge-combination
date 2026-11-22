"""Label-wise re-scoring of the EXISTING Gemma-3-4B-it MS MARCO T1 checkpoints (Base, CE seeds 0-2, CE+soft kappa seeds 0-2; final checkpoints, no training) on val200 and TREC-DL 2019 / 2020.
Copy of methods/msm_val_eval_lg.py (--family gemma) + msm_gemma_eval.py: the SAME scorer (methods/gemma_r7_eval.py score_batch: BF16 blocks, FP32 final norm / head / log_softmax at the supervised positions, label tokens + <end_of_turn>), same tokenizer / template / instruction texts /
max_length 384 / batch 4 / attention backend. The only change: the per-label values the scorer already returns are stored:  raw[qid][docid] = [logP(label tokens | R), logP(<end_of_turn> | R), logP(label tokens | I), logP(<end_of_turn> | I), n_label_tokens(R), n_label_tokens(I)].
Pre-registered gate (fixed before the run, no relaxation): for every (checkpoint, set) the sum margin  m_sum = (R label + R eot) - (I label + I eot)  must equal the STORED margin of the existing evaluation file with max |difference| <= 1e-4; a violation writes ABORT.txt and stops.
    CUDA_VISIBLE_DEVICES=<gpu> python msm_gemma_label390.py --out_dir <dir> --local_files_only
Output: <out_dir>/msmGemmaLabel390_<tag>__T1__<set>.json.gz, <out_dir>/gate390.json"""
import argparse, glob, gzip, hashlib, json, sys, time
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
for rel in ("", "methods", "evaluation", "cpmr_experiment/scripts"):
    sys.path.insert(0, str(ROOT / rel))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.eval_rerank import load_test_pool  # noqa: E402
from gemma_r7_eval import score_batch  # noqa: E402

PASSAGE_PREFIX = "Passage: "
JOBS = [("Base", "BASE")] + [(f"gmq_B_T1_s{s}", f"cpmr_experiment/msm_gemma/train/checkpoints/gmq_B_T1_s{s}/ckpt_final.pt") for s in (0, 1, 2)] + [(f"gmq_E_T1_s{s}", f"cpmr_experiment/msm_gemma/train/checkpoints/gmq_E_T1_s{s}/ckpt_final.pt") for s in (0, 1, 2)]
TOL = 1e-4


def load_val_pairs(path):
    queries, judged, doc_text, qrels = {}, {}, {}, {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        t = r["product_text"]; assert t.startswith(PASSAGE_PREFIX), r["product_id"]
        q, d = r["query_id"], r["product_id"]
        queries[q] = r["query"]; judged.setdefault(q, []).append(d); doc_text[d] = t[len(PASSAGE_PREFIX):]; qrels.setdefault(q, {})[d] = 3 if r["label"] == "Relevant" else 0
    return queries, doc_text, qrels, judged


def stored_margins(set_name, tag):
    pat = f"msmValGemmaEval_{tag}__T1__val200.json.gz" if set_name == "val200" else f"msmGemmaEval_{tag}__T1__{set_name}.json.gz"
    fs = glob.glob(str(ROOT / "cpmr_experiment" / "**" / pat), recursive=True)
    assert fs, ("no stored file", pat)
    run = json.load(gzip.open(fs[0], "rt", encoding="utf-8"))["run_fp32"]
    for f in fs[1:]:
        r2 = json.load(gzip.open(f, "rt", encoding="utf-8"))["run_fp32"]
        assert all(abs(float(r2[q][d]) - float(run[q][d])) < 1e-9 for q in run for d in run[q]), ("stored copies differ", pat)
    return run, hashlib.sha256(open(fs[0], "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="cpmr_experiment/instructions_v2/manifest/instructions_v2.json")
    ap.add_argument("--val_pairs", default="data/msmarco_po/val200/pairs_text_basic_seed0.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()
    from layer_drift_repr import load_adapter
    from src.model import load_model_and_tokenizer
    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / "configs/gemma3_4b_msm.yaml"))
    loaded = load_model_and_tokenizer(model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"], gradient_checkpointing=False, local_files_only=args.local_files_only)
    model, tok = loaded.model, loaded.tokenizer
    model.eval()
    inner = model.base_model.model
    core = inner.model
    norm = core.language_model.norm
    w_head = inner.lm_head.weight.detach().float()
    device = next(model.parameters()).device
    store = {}
    norm.register_forward_pre_hook(lambda m, a: store.__setitem__("h", a[0].detach()))
    man = json.load(open(ROOT / args.manifest, encoding="utf-8"))
    text = {k: v["body"] + man["output_suffix"] for k, v in man["entries"].items() if "body" in v}["T1"]
    out = ROOT / args.out_dir; out.mkdir(parents=True, exist_ok=True)
    pools = {}
    q, dt, qr, jd = load_val_pairs(ROOT / args.val_pairs); pools["val200"] = (q, dt, qr, jd)
    for y in ("2019", "2020"):
        qs, cand, dt, qr = load_test_pool(year=y)
        jd = {q: [d for d in cids if qr.get(q, {}).get(d, -1) >= 0] for q, cids in cand.items() if q in qs}; jd = {q: v for q, v in jd.items() if v}
        pools[y] = (qs, dt, qr, jd)
    psha = {s: hashlib.sha256(json.dumps({q: sorted(v) for q, v in sorted(p[3].items())}).encode()).hexdigest() for s, p in pools.items()}
    print({s: (len(p[3]), sum(len(v) for v in p[3].values()), psha[s][:12]) for s, p in pools.items()}, flush=True)
    gate = []
    for tag, path in JOBS:
        if path == "BASE":
            assert max(float(p_.detach().abs().max()) for n_, p_ in model.named_parameters() if "lora_B" in n_) == 0.0, "adapter not zero: Base must come first"
            ck, ck_sha = {"global_step": 0, "seed": None}, "BASE"
        else:
            ck = load_adapter(model, ROOT / path); ck_sha = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            assert "gemma" in str(ck.get("model", "gemma")).lower(), ck.get("model")
        for sname, (queries, doc_text, qrels, judged) in pools.items():
            f = out / f"msmGemmaLabel390_{tag}__T1__{sname}.json.gz"
            if f.exists():
                print("skip", f.name, flush=True); continue
            t1 = time.time(); raw, n_shr, msum = {}, 0, {}
            with torch.no_grad():
                for qid, cids in judged.items():
                    res = {}
                    for i in range(0, len(cids), args.batch_size):
                        r, s = score_batch(core, norm, w_head, store, tok, text, queries[qid], cids[i:i + args.batch_size], doc_text, args.max_length, device)
                        res.update(r); n_shr += s
                    raw[qid] = {d: [res[d]["Relevant"]["label"], res[d]["Relevant"]["eot"], res[d]["Irrelevant"]["label"], res[d]["Irrelevant"]["eot"], res[d]["Relevant"]["n"], res[d]["Irrelevant"]["n"]] for d in cids}
                    msum[qid] = {d: raw[qid][d][0] + raw[qid][d][1] - raw[qid][d][2] - raw[qid][d][3] for d in cids}
            stored, ssha = stored_margins(sname, tag)
            diffs = np.array([abs(msum[q][d] - float(stored[q][d])) for q in msum for d in msum[q]])
            dec = int(sum((msum[q][d] > 0) != (float(stored[q][d]) > 0) for q in msum for d in msum[q]))
            g = dict(tag=tag, set=sname, n_pairs=int(len(diffs)), max_abs_diff=float(diffs.max()), mean_abs_diff=float(diffs.mean()), identical=int((diffs == 0).sum()), decision_disagreements_t0=dec, stored_file_sha256=ssha, passed=bool(diffs.max() <= TOL), criterion=TOL, seconds=time.time() - t1)
            gate.append(g); print("GATE", g, flush=True)
            grades = {q: {d: qrels[q][d] for d in raw[q]} for q in raw}
            rec = {"meta": {"tag": tag, "set": sname, "checkpoint": path, "checkpoint_sha256": ck_sha, "global_step": ck.get("global_step"), "seed": ck.get("seed"), "instruction": "T1", "instruction_sha256": hashlib.sha256(text.encode()).hexdigest(), "max_length": args.max_length,
                            "pool_ids_sha256": psha[sname], "n_pairs": int(len(diffs)), "n_passage_shortenings_counted_over_two_labels": n_shr, "scorer": "methods/gemma_r7_eval.py score_batch (BF16 blocks, FP32 norm/head/log_softmax at the supervised positions, label tokens + <end_of_turn> 106)",
                            "fields": "raw[qid][docid] = [lp_label(R), lp_eot(R), lp_label(I), lp_eot(I), n_label_tokens(R), n_label_tokens(I)]", "model_name": cfg["model_name"], "torch": torch.__version__}, "raw": raw, "grades": grades}
            tmp = f.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump(rec, fh)
            tmp.replace(f)
            json.dump(gate, open(out / "gate390.json", "w"), indent=1)
            if not g["passed"]:
                (out / "ABORT.txt").write_text(f"gate failed: {g}\n"); print("ABORT: gate failed", flush=True); sys.exit(2)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
