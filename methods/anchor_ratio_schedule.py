"""Instruction schedule with an anchor share p other than 20% (uniform) / 50% (alternating anchor) / 100% (single).

One instruction id per query position, in the training order (position k belongs to step k // batch). The anchor appears a fixed number of times in
every optimizer step (floor(p * batch) or one more, so the total is exactly round(p * n)); the other four instructions share the remaining positions
with counts that differ by at most one, the leftover rotated by the seed. Positions inside a step are shuffled. All randomness comes from a private
random.Random seeded by a string, so the global RNG stream (dropout, data order) is untouched.
"""
import random

IDS = ("T1", "T2", "T3", "T4", "T5")


def make_ratio_schedule(n_queries, seed, anchor, p, batch=8, ids=IDS):
    if n_queries % batch:
        raise ValueError(f"{n_queries} queries are not a multiple of the batch {batch}")
    if anchor not in ids:
        raise ValueError(f"unknown anchor {anchor}")
    steps = n_queries // batch
    total_anchor = round(p * n_queries)
    base, extra = divmod(total_anchor, steps)
    if base + (1 if extra else 0) > batch or total_anchor > n_queries:
        raise ValueError(f"anchor share {p} does not fit the batch")
    rng = random.Random(f"anchor-ratio-schedule-{anchor}-{p}-{seed}")
    per_step = [base] * steps
    for s in rng.sample(range(steps), extra):
        per_step[s] += 1
    others_ids = [i for i in ids if i != anchor]
    n_other = n_queries - total_anchor
    q, r = divmod(n_other, len(others_ids))
    rot = seed % len(others_ids)
    rotated = others_ids[rot:] + others_ids[:rot]
    pool = [i for k, i in enumerate(rotated) for _ in range(q + (1 if k < r else 0))]
    rng.shuffle(pool)
    it = iter(pool)
    sched = []
    for s in range(steps):
        slot = [anchor] * per_step[s] + [next(it) for _ in range(batch - per_step[s])]
        rng.shuffle(slot)
        sched.extend(slot)
    return sched
