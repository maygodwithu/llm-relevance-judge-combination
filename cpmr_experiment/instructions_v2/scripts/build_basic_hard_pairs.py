"""
Build the basic/hard negative pairs from one BM25 candidate file.

Difficulty is manipulated by the ORIGINAL rank band only. Both conditions come from the same
index, the same settings and the same query, so the pool is identical and only the band differs.

  basic : drawn from the whole top-K, minus every known positive of that query
  hard  : drawn from ORIGINAL rank 1-25, minus every known positive of that query

Rules that are easy to get wrong, so they are stated and enforced here:

  * ONE negative per query per condition. So the hard condition needs >= 1 surviving candidate
    in rank 1-25, NOT 25 of them. Only a query with 0 survivors is excluded, and the reason is
    recorded in the shared eligibility record.
  * NO backfill below rank 26. If rank 1-25 has no survivor, the query is dropped -- it is not
    quietly replaced by rank 26+, which would erase the difficulty contrast.
  * The two conditions share the same query AND the same positive. The positive is chosen from
    all known positives by a fixed rule (lowest doc id, so it does not depend on search output),
    and it may lie OUTSIDE the search top-K. Using such a positive as the training positive is
    not the same thing as force-inserting it into the candidate pool, which never happens.
  * "rank 1-25 with positives removed" is NOT "top 25 after removing positives". This code
    implements the first: the rank filter is applied to the ORIGINAL rank, then positives are
    removed, so a band can legitimately hold fewer than 25 candidates.
  * A basic negative can land in the top 25 by chance, or be the same document as the hard
    negative. That is NOT re-drawn -- re-drawing would silently redefine basic as rank 26-K.
    Both rates are reported instead.
  * Unlabelled documents are NOT treated as confirmed irrelevant. Every negative is emitted with
    relevance_class="unlabelled" for the separate qualitative review
    (schema/stage0_records.json, qrels_state).

Eligibility unit is the query_group: a query is usable only if BOTH conditions can be built,
because basic and hard differ in the negative and must be compared on the same queries.

CPU only, no model, no tokenizer.

Usage:
  python cpmr_experiment/instructions_v2/scripts/build_basic_hard_pairs.py \
      --candidates data/msmarco_train_v2/candidates_top1000.jsonl \
      --out_dir data/msmarco_train_v2 --seed 0 --hard_rank_max 25
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

POSITIVE_RULE = ("min(known_positive_doc_ids) by ascending string compare. Independent of search "
                 "output, so both conditions share it even when it is outside the search top-K.")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def rel(p):
    """Repo-relative path when the file is inside the repo, absolute otherwise (temp dirs)."""
    try:
        return str(Path(p).relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def draw(rng_seed, query_id, condition, pool):
    """Seed-fixed draw. The seed is derived per (query, condition) so that adding or removing
    queries does not shift the draw for the others."""
    key = f"{rng_seed}|{query_id}|{condition}"
    r = random.Random(hashlib.sha256(key.encode("utf-8")).hexdigest())
    return r.choice(pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hard_rank_max", type=int, default=25)
    ap.add_argument("--selection_json", default=None,
                    help="query_selection.json; used for the known-positive list. Defaults to "
                         "the sibling of --candidates.")
    ap.add_argument("--allow_overwrite", action="store_true")
    args = ap.parse_args()

    cand_path = Path(args.candidates)
    if not cand_path.is_absolute():
        cand_path = ROOT / cand_path
    sel_path = Path(args.selection_json) if args.selection_json else cand_path.parent / "query_selection.json"
    if not sel_path.is_absolute():
        sel_path = ROOT / sel_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sel = json.load(open(sel_path, encoding="utf-8"))
    positives = {q: sorted(set(v)) for q, v in sel["positives_per_selected_query"].items()}

    excluded, rows = [], []
    stats = {"n_queries_in_candidates": 0, "n_query_groups_built": 0,
             "n_basic_negative_in_hard_band": 0, "n_basic_equals_hard": 0,
             "n_positive_outside_topk": 0, "n_hard_band_smaller_than_rank_max": 0,
             "hard_pool_sizes": [], "basic_pool_sizes": [],
             "basic_negative_ranks": [], "hard_negative_ranks": []}

    with open(cand_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            qid = rec["query_id"]
            stats["n_queries_in_candidates"] += 1
            pos_all = positives.get(qid, [])
            if not pos_all:
                excluded.append({"query_id": qid, "reason": "no_known_positive"})
                continue
            chosen_pos = min(pos_all)
            pos_set = set(pos_all)
            cands = rec["candidates"]
            returned = {c["doc_id"] for c in cands}
            if chosen_pos not in returned:
                stats["n_positive_outside_topk"] += 1

            basic_pool = [c for c in cands if c["doc_id"] not in pos_set]
            # rank filter on the ORIGINAL rank, then remove positives
            hard_pool = [c for c in cands
                         if c["rank"] <= args.hard_rank_max and c["doc_id"] not in pos_set]
            if not basic_pool:
                excluded.append({"query_id": qid, "reason": "no_basic_candidate_after_positive_removal",
                                 "n_candidates": len(cands)})
                continue
            if not hard_pool:
                excluded.append({"query_id": qid,
                                 "reason": f"no_candidate_in_rank_1_{args.hard_rank_max}_after_positive_removal",
                                 "n_candidates_in_band": 0, "n_candidates": len(cands),
                                 "note": "not backfilled from rank 26+ by design"})
                continue
            if len(hard_pool) < args.hard_rank_max:
                stats["n_hard_band_smaller_than_rank_max"] += 1

            b = draw(args.seed, qid, "basic", basic_pool)
            h = draw(args.seed, qid, "hard", hard_pool)
            if b["rank"] <= args.hard_rank_max:
                stats["n_basic_negative_in_hard_band"] += 1
            if b["doc_id"] == h["doc_id"]:
                stats["n_basic_equals_hard"] += 1
            stats["hard_pool_sizes"].append(len(hard_pool))
            stats["basic_pool_sizes"].append(len(basic_pool))
            stats["basic_negative_ranks"].append(b["rank"])
            stats["hard_negative_ranks"].append(h["rank"])
            stats["n_query_groups_built"] += 1

            for cond, c in (("basic", b), ("hard", h)):
                rows.append({
                    "query_group_id": qid, "query_id": qid, "query": rec["query"],
                    "condition": cond,
                    "positive_doc_id": chosen_pos,
                    "positive_in_search_topk": chosen_pos in returned,
                    "negative_doc_id": c["doc_id"],
                    "negative_original_rank": c["rank"], "negative_bm25_score": c["score"],
                    "negative_relevance_class": "unlabelled",
                    "n_known_positives": len(pos_all),
                    "seed": args.seed,
                })

    pairs_path = out_dir / f"pairs_basic_hard_seed{args.seed}.jsonl"
    rep_path = out_dir / f"pairs_basic_hard_seed{args.seed}_report.json"
    for p in (pairs_path, rep_path):
        if p.exists() and not args.allow_overwrite:
            raise SystemExit(f"{p} exists; pass --allow_overwrite to replace it deliberately")
    with open(pairs_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = max(stats["n_query_groups_built"], 1)

    def med(xs):
        if not xs:
            return None
        s = sorted(xs)
        return s[len(s) // 2]

    report = {
        "generated_by": "cpmr_experiment/instructions_v2/scripts/build_basic_hard_pairs.py",
        "candidates_file": rel(cand_path),
        "candidates_sha256": sha256_file(cand_path),
        "query_selection_sha256": sha256_file(sel_path),
        "seed": args.seed, "hard_rank_max": args.hard_rank_max,
        "rules": {
            "eligibility_unit": "query_group (a query is kept only if BOTH conditions are built)",
            "negatives_per_query_per_condition": 1,
            "hard_band": f"ORIGINAL rank 1-{args.hard_rank_max}, then all known positives removed",
            "hard_band_is_not": f"top {args.hard_rank_max} after removing positives",
            "basic_band": "the whole top-K, then all known positives removed",
            "backfill_below_band": "never",
            "positive_selection_rule": POSITIVE_RULE,
            "positive_force_inserted_into_candidates": False,
            "redraw_on_basic_in_hard_band_or_duplicate": False,
            "unlabelled_negatives": "recorded as relevance_class=unlabelled, NOT confirmed irrelevant",
        },
        "counts": {
            "n_queries_in_candidates": stats["n_queries_in_candidates"],
            "n_query_groups_built": stats["n_query_groups_built"],
            "n_pairs": len(rows),
            "n_excluded": len(excluded),
            "n_positive_outside_search_topk": stats["n_positive_outside_topk"],
            "n_hard_band_smaller_than_rank_max": stats["n_hard_band_smaller_than_rank_max"],
        },
        "overlap": {
            "n_basic_negative_in_hard_band": stats["n_basic_negative_in_hard_band"],
            "rate_basic_negative_in_hard_band": round(stats["n_basic_negative_in_hard_band"] / n, 4),
            "n_basic_equals_hard": stats["n_basic_equals_hard"],
            "rate_basic_equals_hard": round(stats["n_basic_equals_hard"] / n, 4),
            "note": ("expected under a uniform draw over the whole top-K; not re-drawn, because "
                     "re-drawing would redefine basic as rank 26+"),
        },
        "pool_sizes": {
            "hard_median": med(stats["hard_pool_sizes"]),
            "hard_min": min(stats["hard_pool_sizes"]) if stats["hard_pool_sizes"] else None,
            "basic_median": med(stats["basic_pool_sizes"]),
            "basic_min": min(stats["basic_pool_sizes"]) if stats["basic_pool_sizes"] else None,
        },
        "negative_rank_distribution": {
            "basic_median": med(stats["basic_negative_ranks"]),
            "hard_median": med(stats["hard_negative_ranks"]),
        },
        "excluded": excluded[:200],
        "excluded_reason_counts": {r: sum(1 for e in excluded if e["reason"] == r)
                                   for r in sorted({e["reason"] for e in excluded})},
        "output": {"pairs": rel(pairs_path),
                   "pairs_sha256": sha256_file(pairs_path)},
        "pending": ["qualitative review of unlabelled negatives for possible relevance"],
    }
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[pairs] query_groups {stats['n_query_groups_built']}/{stats['n_queries_in_candidates']}, "
          f"pairs {len(rows)}, excluded {len(excluded)} {report['excluded_reason_counts']}")
    print(f"[pairs] basic-in-top-{args.hard_rank_max} {report['overlap']['rate_basic_negative_in_hard_band']}, "
          f"basic==hard {report['overlap']['rate_basic_equals_hard']}")
    print(f"[pairs] wrote {pairs_path} and {rep_path}")


if __name__ == "__main__":
    main()
