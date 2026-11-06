"""Multi-5 with ONE instruction per query (K=1): every query group is trained under one instruction drawn at random from T1-T5,
both its documents under that same instruction, token CE only, 10,000 forwards. The plain K=1 T1 run (condition A of the APOCF-online
pilot) differs from this in nothing but the instruction schedule: same trainer code (methods/train_apocf_online.py build_context and
run_steps), same data, optimizer, LR schedule, batch (8 queries per step), LoRA settings and final evaluation (T1, TREC-DL 2019/2020).

Schedule: the 5,000 queries are assigned to T1..T5 exactly 1,000 each, shuffled by a seed-fixed RNG ("multi5-k1-schedule-<seed>"), so
the exposure is balanced like the K=2 Multi and the choice is random per query. No prefix/branch, no API call, no confusion use (the
online confusion counter of the shared step function runs but nothing reads it).

Checkpoints are written at the milestone steps 62, 156, 312, 469, 625 (tags _t10, _t25, _t50, _t75, _t100) like the K=2 runs.

    python methods/train_multi5_k1.py --tag m5k1_s0 --seed 0 --local_files_only
    python methods/train_multi5_k1.py --tag m5k1_smoke --seed 0 --local_files_only --debug_max_steps 25   # nothing durable is written
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
if ROOT is None:
    raise RuntimeError("no .repo-root marker found in any parent directory")
for rel in ("", "methods", "cpmr_experiment/scripts"):
    sys.path.insert(0, str(ROOT / rel))

import torch  # noqa: E402

import train_apocf_online as T  # noqa: E402
from apocf_online_lib import OnlineConfusion, write_json  # noqa: E402

IDS = ("T1", "T2", "T3", "T4", "T5")
FRACS = ((10, 0.10), (25, 0.25), (50, 0.50), (75, 0.75), (100, 1.00))
LF = chr(10)


def make_schedule(n_queries, seed, ids=IDS):
    """One instruction id per query position: exactly n/len(ids) of each, shuffled by a seed-fixed RNG."""
    if n_queries % len(ids):
        raise ValueError(f"{n_queries} queries cannot be split evenly over {len(ids)} instructions")
    sched = [i for i in ids for _ in range(n_queries // len(ids))]
    random.Random(f"multi5-k1-schedule-{seed}").shuffle(sched)
    return sched


def save_adapter(ctx, path, tag, step, seed, total_steps, counts):
    path.mkdir(parents=True, exist_ok=True)
    state = {n: p.detach().cpu().clone() for n, p in ctx["model"].named_parameters() if p.requires_grad}
    torch.save({"adapter_state": state, "tag": tag, "model": ctx["cfg"]["model_name"], "condition": "Multi5-K1-CE",
                "global_step": step, "frac": step / total_steps, "seed": seed,
                "instruction_builds_so_far": dict(counts)}, path / "ckpt_final.pt")


def main():
    ap = argparse.ArgumentParser()
    T.common_args(ap)
    ap.set_defaults(out_root="cpmr_experiment/pilot_multi5_k1")
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()
    args.branch_step = 1  # unused by this command; build_context only asserts 0 < branch_step < total_steps
    ctx = T.build_context(args)
    ids = [i for i in IDS]
    for i in ids:
        if i not in ctx["v2"].instructions:
            raise SystemExit(f"{i} is not in the TRAIN pool")
    total, n_q = ctx["total_steps"], ctx["n_q"]
    sched = make_schedule(n_q, args.seed)
    pos_of = {qid: k for k, qid in enumerate(ctx["order"])}
    counts = {}
    inner = ctx["builder"]
    by_text = {t: k for k, t in ctx["v2"].instructions.items()}

    def counting_builder(tokenizer, instruction, *a, **kw):
        key = by_text.get(instruction, "UNKNOWN")
        counts[key] = counts.get(key, 0) + 1
        return inner(tokenizer, instruction, *a, **kw)

    ctx["builder"] = counting_builder

    def instruction_for(step, j, qid):
        i = sched[pos_of[qid]]
        return 0, ctx["v2"].instructions[i]

    out = ROOT / args.out_root
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stream = log_dir / f"steps_{args.tag}.jsonl"
    if stream.exists() and not args.debug_max_steps:
        stream.unlink()
    print(f"[m5k1] {n_q} queries -> exactly {n_q // len(ids)} per instruction {ids}; first 10 draws {sched[:10]}")
    milestones = sorted({max(1, round(f * total)): (pct, f) for pct, f in FRACS}.items())
    last = total if not args.debug_max_steps else min(total, args.debug_max_steps)
    conf, fwd = OnlineConfusion(), [0]
    prev = 0
    for mstep, (pct, _) in milestones:
        upto = min(mstep, last)
        if upto > prev:
            T.run_steps(ctx, args, prev, upto, instruction_for, conf, None if args.debug_max_steps else stream, fwd, args.tag)
            prev = upto
        if args.debug_max_steps or upto != mstep:
            continue
        save_adapter(ctx, out / "checkpoints" / f"{args.tag}_t{pct}", f"{args.tag}_t{pct}", mstep, args.seed, total, counts)
        print(f"[m5k1] saved checkpoint {args.tag}_t{pct} step={mstep}/{total}", flush=True)
    print(f"[m5k1] training builds per instruction: {counts}; total forwards {fwd[0]}", flush=True)
    if args.debug_max_steps:
        return 0
    write_json(log_dir / f"m5k1_stats_{args.tag}.json", {
        "tag": args.tag, "seed": args.seed, "queries_per_instruction": {i: sched.count(i) for i in ids},
        "training_builds_per_instruction": counts, "total_forwards": fwd[0], "schedule_first_20": sched[:20],
        "note": "one instruction per query group (both documents under it); builds count training forwards only"})
    if not args.skip_eval:
        from src.eval_rerank_mistral import evaluate_rerank_mistral
        t1 = ctx["t1"]
        for year in ("2019", "2020"):
            res = evaluate_rerank_mistral(ctx["model"], ctx["tok"], t1, args.max_length, ctx["device"],
                                          save_path=str(log_dir / f"eval_{args.tag}_t100_T1_trec{year}_scores.json"),
                                          year=year, enable_thinking=False)
            print(f"[eval] {args.tag} T1 TREC-DL {year}: nDCG@10={res['ndcg@10']:.4f} MRR@10={res['mrr@10']:.4f} "
                  f"n_queries={res['n_queries']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
