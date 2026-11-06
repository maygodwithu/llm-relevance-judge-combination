"""Judged-pair evaluation for the instructions T1-T7 (experiment log 211): msm_final_eval.py (experiment log 176) with exactly these changes: a job is tag:instr:path (instr T1..T7; path BASE = no adapter, the fresh LoRA with B = 0 is the Base model), the instruction text of the job is used instead of the fixed T1 (T6, T7 from --custom_instruction_file), and the score file is msmEval_<tag>__<instr>__<year>.json.gz. The scorer (evaluation/rescore_fp32.py score_both), batch composition, pools and max_length are unchanged.
Original docstring follows.
Final-checkpoint judged-pair evaluation of the MS MARCO four-loss pilot (experiment log 176): every judged candidate of TREC-DL 2019 (1,102 pairs) and 2020 (1,432 pairs), human grade >= 0, scored with the T1 instruction by the existing FP32 scorer
evaluation/rescore_fp32.py score_both (the same teacher-forced forward, bf16 blocks, FP32 final norm / unembedding / log_softmax at the supervised positions, batch of 4 documents x 2 labels, label + terminator): m = log P(Relevant+terminator) - log P(Irrelevant+terminator).
No nDCG, no unjudged candidate, no threshold search (the decision is m > 0). One process loads the model once and scores several checkpoints.
    CUDA_VISIBLE_DEVICES=0 python methods/msm_final_eval.py --models msm_A_s0:cpmr_experiment/msm_loss/checkpoints/msm_A_s0/ckpt_final.pt ... --out_dir cpmr_experiment/msm_loss/eval --local_files_only"""
import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
for rel in ("", "evaluation", "cpmr_experiment/scripts", "cpmr_experiment/instructions_v2/scripts"):
    sys.path.insert(0, str(ROOT / rel))

import torch  # noqa: E402
import yaml  # noqa: E402

import rescore_fp32 as R  # noqa: E402
from layer_drift_lib import locate_blocks  # noqa: E402
from layer_drift_repr import load_adapter  # noqa: E402
from src.eval_rerank import load_test_pool  # noqa: E402
from v2_input_path import load_v2_instructions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen3_4b_msm_loss.yaml")
    ap.add_argument("--models", nargs="+", required=True, help="tag:instr:checkpoint path (or tag:instr:BASE)")
    ap.add_argument("--custom_instruction_file", default=None)
    ap.add_argument("--manifest", default="cpmr_experiment/instructions_v2/manifest/instructions_v2.json")
    ap.add_argument("--years", nargs="+", default=["2019", "2020"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--max_queries", type=int, default=0, help="debug: nothing durable")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()
    from src.model import load_model_and_tokenizer
    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / args.config))
    loaded = load_model_and_tokenizer(model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
                                      gradient_checkpointing=False, local_files_only=args.local_files_only)
    model, tok = loaded.model, loaded.tokenizer
    model.eval()
    device = next(model.parameters()).device
    _, _, norm, lm_head = locate_blocks(model)
    store = {}
    norm.register_forward_pre_hook(lambda m, a: store.__setitem__("h", a[0].detach()))
    w_head = lm_head.weight.detach().float()
    instr, _, _ = load_v2_instructions(ROOT / args.manifest, roles=("TRAIN",))
    instr = dict(instr)
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
        print(f"[msm-eval] TREC-DL {y}: {len(judged)} queries, {sum(len(v) for v in judged.values())} judged pairs, judged-id sha256 {ids_sha[:16]}", flush=True)
    for job in args.models:
        tag, ins_id, path = job.split(":", 2)
        text = instr[ins_id]
        if path == "BASE":
            assert max(float(p_.detach().abs().max()) for n_, p_ in model.named_parameters() if "lora_B" in n_) == 0.0, "adapter not zero: load order"
            ck, ck_sha = {"global_step": 0, "seed": None}, "BASE"
        else:
            p = ROOT / path
            ck = load_adapter(model, p)
            ck_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        for y in args.years:
            f = out / f"msmEval_{tag}__{ins_id}__{y}.json.gz"
            if f.exists() and not args.max_queries:
                print(f"[msm-eval] skip {tag} {y}", flush=True)
                continue
            queries, doc_text, qrels, judged = pools[y]
            t1 = time.time()
            run32, runbf = {}, {}
            for n, (qid, cids) in enumerate(judged.items()):
                if args.max_queries and n >= args.max_queries:
                    break
                runbf[qid], run32[qid] = R.score_both(model, tok, text, queries[qid], cids, doc_text, args.max_length, device, norm, w_head, store, args.batch_size)
            grades = {q: {d: qrels[q][d] for d in run32[q]} for q in run32}
            tp = fp = fn = tn = 0
            for q in run32:
                for d, m in run32[q].items():
                    pr, tr = m > 0, grades[q][d] >= 2
                    tp += pr and tr; fp += pr and not tr; fn += (not pr) and tr; tn += (not pr) and (not tr)
            n_ = tp + fp + fn + tn
            po = (tp + tn) / n_; pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n_ ** 2
            print(f"[msm-eval] {tag} {y}: {n_} pairs, kappa {(po - pe) / (1 - pe):.4f}, Relevant {100 * (tp + fp) / n_:.1f}% ({time.time() - t1:.0f}s)", flush=True)
            if args.max_queries:
                continue
            rec = {"meta": {"tag": tag, "year": y, "checkpoint": path, "checkpoint_sha256": ck_sha, "global_step": ck.get("global_step"), "seed": ck.get("seed"), "instruction": ins_id, "instruction_sha256": hashlib.sha256(text.encode()).hexdigest(), "max_length": args.max_length,
                            "n_judged": n_, "scorer": "evaluation/rescore_fp32.py score_both (FP32 norm/head/log_softmax at the supervised positions, label + terminator)", "model_name": cfg["model_name"], "torch": torch.__version__},
                   "run_fp32": run32, "run_bf16": runbf, "grades": grades}
            tmp = f.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump(rec, fh)
            tmp.replace(f)
    print(f"[msm-eval] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
