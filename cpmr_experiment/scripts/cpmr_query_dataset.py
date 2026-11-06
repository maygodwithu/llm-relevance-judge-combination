"""
Query-grouped training data + K-prompt-without-replacement sampler for CPMR/Rank
conditions.

data/scenarios/msmarco_m1_b5000_train.jsonl has exactly 1 Relevant + 1 Irrelevant row
per query_id (confirmed: cpmr_experiment/logs/audit_prereqs_output.txt SECTION 3, all
5000 queries, adjacent-pair layout). This groups those into (query, pos_text, neg_text)
units and shuffles the QUERY order with the same order_rng=random.Random(seed)
convention used by every other trajectory dataset in this project -- shared identically
across FixedOnly-Rank / Multi5-Rank / CPMR-beta0.1 / CPMR-beta1.0 (all instantiate this
class with seed=0), satisfying "동일한 query ID... 및 query-group order 사용".
"""
import json
import random
from collections import Counter


def load_query_groups(jsonl_path, seed):
    rows = [json.loads(l) for l in open(jsonl_path, encoding="utf-8") if l.strip()]
    groups = {}
    order_first_seen = []
    for r in rows:
        qid = r["query_id"]
        if qid not in groups:
            groups[qid] = {"query": r["query"], "pos": None, "neg": None}
            order_first_seen.append(qid)
        if r["label"] == "Relevant":
            groups[qid]["pos"] = r["product_text"]
        else:
            groups[qid]["neg"] = r["product_text"]
    for qid, g in groups.items():
        assert g["pos"] is not None and g["neg"] is not None, \
            f"query {qid} missing pos or neg row"

    order_rng = random.Random(seed)
    query_order = list(order_first_seen)
    order_rng.shuffle(query_order)
    return groups, query_order


def build_balanced_pair_sequence(keys, n_steps, seed):
    """n_steps pairs of 2 DISTINCT keys, exact global balance (each key appears in
    exactly n_steps*2/len(keys) pairs). Greedy sampling from a shrinking balanced
    multiset -- deterministic given seed."""
    K = len(keys)
    total_slots = n_steps * 2
    assert total_slots % K == 0, f"n_steps*2={total_slots} must be divisible by K={K}"
    per_key = total_slots // K
    remaining = {k: per_key for k in keys}
    rng = random.Random(seed)
    pairs = []
    for step in range(n_steps):
        pool1 = [k for k in keys for _ in range(remaining[k])]
        rng.shuffle(pool1)
        k1 = pool1[0]
        remaining[k1] -= 1
        pool2 = [k for k in keys if k != k1 for _ in range(remaining[k])]
        assert pool2, f"infeasible balanced draw at step {step}, remaining={remaining}"
        rng.shuffle(pool2)
        k2 = pool2[0]
        remaining[k2] -= 1
        pairs.append((k1, k2))
    assert all(v == 0 for v in remaining.values()), f"leftover slots: {remaining}"
    return pairs


def exposure_counts(prompt_sequence):
    c = Counter()
    for item in prompt_sequence:
        keys = item if isinstance(item, (tuple, list)) else (item,)
        for k in keys:
            c[k] += 1
    return dict(c)
