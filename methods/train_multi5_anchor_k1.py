"""Multi-5 with a T1 anchor, K=1 (experiment log 109): alternate, in training order, a query group under T1 and a query group under one of
T2..T5 drawn at random. One instruction per query group (both documents under it), one gold-label CE forward per document, token CE only,
5,000 queries, 625 optimizer steps, 10,000 forwards.

Schedule (positions are the training order of the query groups, 1-based):
  odd positions   (2,500)  T1 -- fixed, so every optimizer step of 8 queries holds exactly 4 T1 groups
  even positions  (2,500)  T2, T3, T4, T5, exactly 625 each, shuffled by a seed-fixed RNG ("anchor-k1-schedule-<seed>")
Exposure: T1 50%, T2..T5 12.5% each (forwards: T1 5,000, the others 1,250 each). It is a balanced random assignment with the totals fixed,
not independent sampling. Nothing else differs from the uniform Multi-5 K=1 run (methods/train_multi5_k1.py): same trainer functions
(methods/train_apocf_online.py build_context and run_steps), data order, passage cache, optimizer, LR schedule, LoRA settings, milestone
checkpoints (t10/t25/t50/t75/t100) and final evaluation (T1, TREC-DL 2019 and 2020).

    python methods/train_multi5_anchor_k1.py --tag anchork1_s0 --seed 0 --local_files_only
    python methods/train_multi5_anchor_k1.py --tag anchork1_smoke --seed 0 --local_files_only --debug_max_steps 25   # nothing durable is written
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
if ROOT is None:
    raise RuntimeError("no .repo-root marker found in any parent directory")
for rel in ("", "methods", "cpmr_experiment/scripts"):
    sys.path.insert(0, str(ROOT / rel))

import train_apocf_online as T  # noqa: E402
from apocf_online_lib import OnlineConfusion, write_json  # noqa: E402
from train_multi5_k1 import FRACS, IDS, save_adapter  # noqa: E402

OTHERS = IDS[1:]


def make_anchor_schedule(n_queries, seed):
    """Position 0, 2, 4, ... (the 1st, 3rd, 5th query group) -> T1; the other positions -> T2..T5, equal counts, shuffled."""
    if n_queries % 2 or (n_queries // 2) % len(OTHERS):
        raise ValueError(f"{n_queries} queries cannot be split into a T1 half and an even T2-T5 half")
    others = [i for i in OTHERS for _ in range(n_queries // 2 // len(OTHERS))]
    random.Random(f"anchor-k1-schedule-{seed}").shuffle(others)
    it = iter(others)
    return ["T1" if k % 2 == 0 else next(it) for k in range(n_queries)]


def main():
    ap = argparse.ArgumentParser()
    T.common_args(ap)
    ap.set_defaults(out_root="cpmr_experiment/pilot_multi5_anchor_k1")
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()
    args.branch_step = 1  # unused here; build_context only asserts 0 < branch_step < total_steps
    ctx = T.build_context(args)
    for i in IDS:
        if i not in ctx["v2"].instructions:
            raise SystemExit(f"{i} is not in the TRAIN pool")
    total, n_q = ctx["total_steps"], ctx["n_q"]
    sched = make_anchor_schedule(n_q, args.seed)
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
        return 0, ctx["v2"].instructions[sched[pos_of[qid]]]

    out = ROOT / args.out_root
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stream = log_dir / f"steps_{args.tag}.jsonl"
    if stream.exists() and not args.debug_max_steps:
        stream.unlink()
    print(f"[anchor] {n_q} queries: T1 {sched.count('T1')}, " + ", ".join(f"{i} {sched.count(i)}" for i in OTHERS)
          + f"; first 12 draws {sched[:12]}")
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
        print(f"[anchor] saved checkpoint {args.tag}_t{pct} step={mstep}/{total}", flush=True)
    print(f"[anchor] training builds per instruction: {counts}; total forwards {fwd[0]}", flush=True)
    if args.debug_max_steps:
        return 0
    write_json(log_dir / f"anchor_stats_{args.tag}.json", {
        "tag": args.tag, "seed": args.seed, "queries_per_instruction": {i: sched.count(i) for i in IDS},
        "training_builds_per_instruction": counts, "total_forwards": fwd[0], "schedule_first_24": sched[:24],
        "note": "odd query-group positions T1, even positions T2-T5 (625 each, shuffled); one instruction per query group"})
    if not args.skip_eval:
        from src.eval_rerank_mistral import evaluate_rerank_mistral
        for year in ("2019", "2020"):
            res = evaluate_rerank_mistral(ctx["model"], ctx["tok"], ctx["t1"], args.max_length, ctx["device"],
                                          save_path=str(log_dir / f"eval_{args.tag}_t100_T1_trec{year}_scores.json"),
                                          year=year, enable_thinking=False)
            print(f"[eval] {args.tag} T1 TREC-DL {year}: nDCG@10={res['ndcg@10']:.4f} MRR@10={res['mrr@10']:.4f} "
                  f"n_queries={res['n_queries']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
