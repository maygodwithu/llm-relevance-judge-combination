"""Judged-pair evaluation of the Llama-3.1-8B-Instruct MS MARCO T1-T7 experiment (experiment log 214): every judged candidate of TREC-DL 2019 (1,102 pairs) and 2020 (1,432 pairs), human grade >= 0, scored with the model's own instruction by llama_r7_eval.score_batch
(batch of 4 documents x 2 labels, bf16 blocks, FP32 final norm / untied lm_head / log_softmax at the supervised positions, label tokens + <|eot_id|>): m = [lp_label + lp_eot](Relevant) - [lp_label + lp_eot](Irrelevant). max_length 384 (no passage is shortened; the count is recorded).
A job is tag:instr:path (instr T1..T7; path BASE = no adapter: the fresh LoRA with B = 0 is the Base model, so Base jobs must come before adapter jobs in one process). No nDCG, no unjudged candidate, no threshold search (the decision is m > 0). Score file msmLlamaEval_<tag>__<instr>__<year>.json.gz with the same layout as the Qwen msmEval files (run_fp32, grades).
    CUDA_VISIBLE_DEVICES=0 python methods/msm_llama_eval.py --models mlq_B_T1_s0:T1:cpmr_experiment/msm_llama/train/checkpoints/mlq_B_T1_s0/ckpt_final.pt --out_dir cpmr_experiment/msm_llama/eval --local_files_only"""
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

import torch  # noqa: E402
import yaml  # noqa: E402

from llama_r7_eval import score_batch  # noqa: E402
from src.eval_rerank import load_test_pool  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/llama31_8b_msm.yaml")
    ap.add_argument("--models", nargs="+", required=True, help="tag:instr:checkpoint path (or tag:instr:BASE)")
    ap.add_argument("--manifest", default="cpmr_experiment/instructions_v2/manifest/instructions_v2.json")
    ap.add_argument("--custom_instruction_file", default=None)
    ap.add_argument("--years", nargs="+", default=["2019", "2020"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--max_queries", type=int, default=0, help="debug: nothing durable")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()
    from layer_drift_repr import load_adapter
    from src.model import load_model_and_tokenizer
    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / args.config))
    loaded = load_model_and_tokenizer(model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
                                      gradient_checkpointing=False, local_files_only=args.local_files_only)
    model, tok = loaded.model, loaded.tokenizer
    model.eval()
    inner = model.base_model.model
    core = inner.model
    norm = core.norm
    w_head = inner.lm_head.weight.detach().float()
    device = next(model.parameters()).device
    store = {}
    norm.register_forward_pre_hook(lambda m, a: store.__setitem__("h", a[0].detach()))
    man = json.load(open(ROOT / args.manifest, encoding="utf-8"))
    instr = {k: v["body"] + man["output_suffix"] for k, v in man["entries"].items() if "body" in v}
    if args.custom_instruction_file:
        instr.update(json.load(open(ROOT / args.custom_instruction_file, encoding="utf-8")))
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pools = {}
    for y in args.years:
        queries, cand_ids, doc_text, qrels = load_test_pool(year=y)
        judged = {q: [d for d in cids if qrels.get(q, {}).get(d, -1) >= 0] for q, cids in cand_ids.items() if q in queries}
        judged = {q: v for q, v in judged.items() if v}
        pools[y] = (queries, doc_text, qrels, judged)
        ids_sha = hashlib.sha256(json.dumps({q: sorted(v) for q, v in sorted(judged.items())}).encode()).hexdigest()
        print(f"[msm-llama-eval] TREC-DL {y}: {len(judged)} queries, {sum(len(v) for v in judged.values())} judged pairs, judged-id sha256 {ids_sha[:16]}", flush=True)
    for job in args.models:
        tag, ins_id, path = job.split(":", 2)
        text = instr[ins_id]
        todo = [y for y in args.years if args.max_queries or not (out / f"msmLlamaEval_{tag}__{ins_id}__{y}.json.gz").exists()]
        if not todo:
            print(f"[msm-llama-eval] skip {tag} {ins_id}", flush=True)
            continue
        if path == "BASE":
            assert max(float(p_.detach().abs().max()) for n_, p_ in model.named_parameters() if "lora_B" in n_) == 0.0, "adapter not zero: Base jobs must come first"
            ck, ck_sha = {"global_step": 0, "seed": None}, "BASE"
        else:
            p = ROOT / path
            ck = load_adapter(model, p)
            ck_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        for y in todo:
            f = out / f"msmLlamaEval_{tag}__{ins_id}__{y}.json.gz"
            queries, doc_text, qrels, judged = pools[y]
            t1 = time.time()
            run, n_shr = {}, 0
            for n, (qid, cids) in enumerate(judged.items()):
                if args.max_queries and n >= args.max_queries:
                    break
                res = {}
                for i in range(0, len(cids), args.batch_size):
                    r, s = score_batch(core, norm, w_head, store, tok, text, queries[qid], cids[i:i + args.batch_size], doc_text, args.max_length, device)
                    res.update(r); n_shr += s
                run[qid] = {d: res[d]["Relevant"]["label"] + res[d]["Relevant"]["eot"] - res[d]["Irrelevant"]["label"] - res[d]["Irrelevant"]["eot"] for d in cids}
            grades = {q: {d: qrels[q][d] for d in run[q]} for q in run}
            tp = fp = fn = tn = 0
            for q in run:
                for d, m in run[q].items():
                    pr, tr = m > 0, grades[q][d] >= 2
                    tp += pr and tr; fp += pr and not tr; fn += (not pr) and tr; tn += (not pr) and (not tr)
            n_ = tp + fp + fn + tn
            po = (tp + tn) / n_; pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n_ ** 2
            print(f"[msm-llama-eval] {tag} {ins_id} {y}: {n_} pairs, kappa {(po - pe) / (1 - pe):.4f}, Relevant {100 * (tp + fp) / n_:.1f}%, shortened {n_shr} ({time.time() - t1:.0f}s)", flush=True)
            if args.max_queries:
                continue
            rec = {"meta": {"tag": tag, "instruction": ins_id, "instruction_sha256": hashlib.sha256(text.encode()).hexdigest(), "year": y, "checkpoint": path, "checkpoint_sha256": ck_sha, "global_step": ck.get("global_step"), "seed": ck.get("seed"),
                            "max_length": args.max_length, "n_judged": n_, "n_passage_shortenings_counted_over_two_labels": n_shr, "scorer": "methods/llama_r7_eval.py score_batch (FP32 norm/untied head/log_softmax at the supervised positions, label tokens + <|eot_id|>)",
                            "model_name": cfg["model_name"], "torch": torch.__version__, "batch_size": args.batch_size},
                   "run_fp32": run, "grades": grades}
            tmp = f.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump(rec, fh)
            tmp.replace(f)
    print(f"[msm-llama-eval] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
