"""Validation-pair scoring for the MS MARCO separate-validation threshold (user choice A0 = the EXISTING val200 selection pairs, unchanged), Llama-3.1-8B-Instruct and Gemma-3-4B-it: msm_llama_eval.py / msm_gemma_eval.py (the main-experiment TREC-DL evaluators) with exactly these changes:
the evaluation pool is the frozen val200 pair file (200 queries x [1 positive, 1 uniform-drawn BM25 top-1000 negative]; labels Relevant / Irrelevant -> grade 3 / 0) instead of TREC-DL; same instruction texts, same FP32 scorer (llama_r7_eval / gemma_r7_eval score_batch), max_length 384, label + terminator, no threshold search; several custom instruction files may be given.
Gate: for the FIRST job of the process the same process also scores the first N TREC-DL 2019 queries of the same job and compares with the stored main-experiment evaluation file (max |difference|, identical count printed); shortened inputs are counted by the scorer; every score is checked finite.
    CUDA_VISIBLE_DEVICES=<gpu> python methods/msm_val_eval_lg.py --family llama|gemma --models tag:instr:checkpoint ... --out_dir <dir> --local_files_only [--repro_trec2019 N]
Output: <out_dir>/msmVal{Llama|Gemma}Eval_<tag>__<instr>__val200.json.gz"""
import argparse
import glob
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
for rel in ("", "methods", "evaluation", "cpmr_experiment/scripts"):
    sys.path.insert(0, str(ROOT / rel))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.eval_rerank import load_test_pool  # noqa: E402

PASSAGE_PREFIX = "Passage: "
FAM = {"llama": dict(config="configs/llama31_8b_msm.yaml", stored="msmLlamaEval_", out="msmValLlamaEval_"), "gemma": dict(config="configs/gemma3_4b_msm.yaml", stored="msmGemmaEval_", out="msmValGemmaEval_")}


def load_val_pairs(path):
    queries, judged, doc_text, qrels = {}, {}, {}, {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        t = r["product_text"]
        assert t.startswith(PASSAGE_PREFIX), r["product_id"]
        q, d = r["query_id"], r["product_id"]
        queries[q] = r["query"]
        judged.setdefault(q, []).append(d)
        doc_text[d] = t[len(PASSAGE_PREFIX):]
        qrels.setdefault(q, {})[d] = 3 if r["label"] == "Relevant" else 0
    return queries, doc_text, qrels, judged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAM))
    ap.add_argument("--models", nargs="+", required=True, help="tag:instr:checkpoint path (or tag:instr:BASE); Base jobs first")
    ap.add_argument("--manifest", default="cpmr_experiment/instructions_v2/manifest/instructions_v2.json")
    ap.add_argument("--custom_instruction_file", nargs="+", default=[])
    ap.add_argument("--val_pairs", default="data/msmarco_po/val200/pairs_text_basic_seed0.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--repro_trec2019", type=int, default=0)
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()
    fam = FAM[args.family]
    if args.family == "llama":
        from llama_r7_eval import score_batch
    else:
        from gemma_r7_eval import score_batch
    from layer_drift_repr import load_adapter
    from src.model import load_model_and_tokenizer
    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / fam["config"]))
    loaded = load_model_and_tokenizer(model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
                                      gradient_checkpointing=False, local_files_only=args.local_files_only)
    model, tok = loaded.model, loaded.tokenizer
    model.eval()
    inner = model.base_model.model
    core = inner.model
    norm = core.norm if args.family == "llama" else core.language_model.norm
    w_head = inner.lm_head.weight.detach().float()
    device = next(model.parameters()).device
    store = {}
    norm.register_forward_pre_hook(lambda m, a: store.__setitem__("h", a[0].detach()))
    man = json.load(open(ROOT / args.manifest, encoding="utf-8"))
    instr = {k: v["body"] + man["output_suffix"] for k, v in man["entries"].items() if "body" in v}
    for cf in args.custom_instruction_file:
        instr.update(json.load(open(ROOT / cf, encoding="utf-8")))
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    queries, doc_text, qrels, judged = load_val_pairs(ROOT / args.val_pairs)
    pairs_sha = hashlib.sha256(open(ROOT / args.val_pairs, "rb").read()).hexdigest()
    n_pairs = sum(len(v) for v in judged.values())
    print(f"[msm-val-{args.family}] val pairs: {len(judged)} queries, {n_pairs} pairs, file sha256 {pairs_sha[:16]}", flush=True)

    def score_pool(text, qs, jd, dt):
        run, n_shr = {}, 0
        for qid, cids in jd.items():
            res = {}
            for i in range(0, len(cids), args.batch_size):
                r, s = score_batch(core, norm, w_head, store, tok, text, qs[qid], cids[i:i + args.batch_size], dt, args.max_length, device)
                res.update(r); n_shr += s
            run[qid] = {d: res[d]["Relevant"]["label"] + res[d]["Relevant"]["eot"] - res[d]["Irrelevant"]["label"] - res[d]["Irrelevant"]["eot"] for d in cids}
        return run, n_shr

    repro_done = False
    for job in args.models:
        tag, ins_id, path = job.split(":", 2)
        text = instr[ins_id]
        f = out / f"{fam['out']}{tag}__{ins_id}__val200.json.gz"
        if f.exists():
            print(f"[msm-val-{args.family}] skip {tag} {ins_id}", flush=True)
            continue
        if path == "BASE":
            assert max(float(p_.detach().abs().max()) for n_, p_ in model.named_parameters() if "lora_B" in n_) == 0.0, "adapter not zero: Base jobs must come first"
            ck, ck_sha = {"global_step": 0, "seed": None}, "BASE"
        else:
            p = ROOT / path
            ck = load_adapter(model, p)
            ck_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        t1 = time.time()
        with torch.no_grad():
            run, n_shr = score_pool(text, queries, judged, doc_text)
        m_all = np.array([run[q][d] for q in run for d in run[q]])
        assert np.isfinite(m_all).all(), "nonfinite score"
        grades = {q: {d: qrels[q][d] for d in run[q]} for q in run}
        tp = fp = fn = tn = 0
        for q in run:
            for d, m in run[q].items():
                pr, tr = m > 0, grades[q][d] >= 2
                tp += pr and tr; fp += pr and not tr; fn += (not pr) and tr; tn += (not pr) and (not tr)
        po = (tp + tn) / n_pairs; pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n_pairs ** 2
        print(f"[msm-val-{args.family}] {tag} {ins_id} val200: {n_pairs} pairs, kappa(t=0) {(po - pe) / (1 - pe):.4f}, predicted Relevant {100 * (tp + fp) / n_pairs:.1f}%, shortened {n_shr}, nonfinite 0 ({time.time() - t1:.0f}s)", flush=True)
        rec = {"meta": {"tag": tag, "set": "val200", "checkpoint": path, "checkpoint_sha256": ck_sha, "global_step": ck.get("global_step"), "seed": ck.get("seed"), "instruction": ins_id,
                        "instruction_sha256": hashlib.sha256(text.encode()).hexdigest(), "max_length": args.max_length, "n_pairs": n_pairs, "val_pairs_sha256": pairs_sha, "n_passage_shortenings_counted_over_two_labels": n_shr,
                        "scorer": f"methods/{args.family}_r7_eval.py score_batch (FP32 final norm / head / log_softmax at the supervised positions, label tokens + terminator)", "model_name": cfg["model_name"], "torch": torch.__version__, "batch_size": args.batch_size},
               "run_fp32": run, "grades": grades}
        tmp = f.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(rec, fh)
        tmp.replace(f)
        if args.repro_trec2019 and not repro_done:
            repro_done = True
            tq, tcand, tdoc, tqrels = load_test_pool(year="2019")
            tj = {q: [d for d in cids if tqrels.get(q, {}).get(d, -1) >= 0] for q, cids in tcand.items() if q in tq}
            tj = {q: v for q, v in tj.items() if v}
            tj = dict(list(tj.items())[: args.repro_trec2019])
            stored = None
            for sf in glob.glob(str(ROOT / "cpmr_experiment" / "**" / f"{fam['stored']}{tag}__{ins_id}__2019.json.gz"), recursive=True):
                stored = json.load(gzip.open(sf, "rt", encoding="utf-8"))["run_fp32"]
                break
            assert stored is not None, "no stored TREC-DL 2019 file for this job"
            with torch.no_grad():
                new, _ = score_pool(text, tq, tj, tdoc)
            diffs = np.array([abs(new[q][d] - float(stored[q][d])) for q in tj for d in tj[q]])
            print(f"[msm-val-{args.family}] GATE reproduction {tag}: {len(diffs)} TREC-DL 2019 pairs, max |new - stored| {diffs.max():.3e}, identical {int((diffs == 0).sum())}/{len(diffs)}", flush=True)
    print(f"[msm-val-{args.family}] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
