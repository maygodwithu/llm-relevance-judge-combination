"""
Attach corpus text to the already-fixed MS MARCO 100-query basic/hard pairs.

What this does NOT do, deliberately: no re-search, no re-selection
of queries, no re-draw of negatives. The query ids, the positive and both negatives are read from
`query_selection.json` and `pairs_basic_hard_seed0.jsonl` exactly as they were fixed, and only the
document text is added.

Two failure kinds are kept apart:
  * positive outside the search top-1000 -- expected, already counted at candidate time (18/100).
    It is still the training positive; it was never inserted into the candidate pool.
  * positive (or negative) not present in the corpus at all -- an ERROR. Such a row is reported
    with its ids and the script exits non-zero. It is not substituted and not silently dropped.

Output is one jsonl per condition, in the schema the existing pool and build_passage_cache.py
already read (`query_id`, `query`, `product_id`, `product_text`, `label`), so the difficulty
conditions become two `--pool_condition` runs that `intersect_eligibility` can compare over the
same query groups.

Document text uses the same join rule as the candidate build: "{title}\\n{text}" with title=""
when absent, then prefixed with "Passage: " because `data_mistral.build_example_mistral` composes
`f"{instruction}\\n\\nQuery: {query}\\n{ptext}"` and expects the marker to already be there.

Random access via `docs_store()` rather than `docs_iter()`: only ~300 documents are needed, and a
full pass over the 8.84M-document corpus cost 91.0 s in the candidate build.

CPU only.

Usage:
  python cpmr_experiment/instructions_v2/scripts/join_msmarco_pair_text.py \
      --pairs data/msmarco_train_v2/pairs_basic_hard_seed0.jsonl \
      --out_dir data/msmarco_train_v2
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

IRDS_ID = "msmarco-passage/train"
DOC_JOIN_RULE = '"{title}\\n{text}", title="" when absent, then prefixed with "Passage: "'


def sha256_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def lib_versions():
    out = {}
    for m in ("ir_datasets",):
        try:
            out[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            out[m] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--selection_json", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--allow_overwrite", action="store_true")
    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.is_absolute():
        pairs_path = ROOT / pairs_path
    sel_path = Path(args.selection_json) if args.selection_json else pairs_path.parent / "query_selection.json"
    if not sel_path.is_absolute():
        sel_path = ROOT / sel_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sel = json.load(open(sel_path, encoding="utf-8"))
    rows = [json.loads(l) for l in open(pairs_path, encoding="utf-8")]
    conditions = sorted({r["condition"] for r in rows})
    qids_in = sorted({r["query_id"] for r in rows})

    import ir_datasets
    ds = ir_datasets.load(IRDS_ID)
    store = ds.docs_store()

    wanted = sorted({r["positive_doc_id"] for r in rows} | {r["negative_doc_id"] for r in rows})
    texts, missing = {}, []
    for did in wanted:
        try:
            d = store.get(did)   # raises KeyError when the id is not in the corpus
        except Exception:        # noqa: BLE001 -- absence is the signal, not a reason to stop
            d = None
        if d is None:
            missing.append(did)
            continue
        body = f"{getattr(d, 'title', '') or ''}\n{d.text or ''}"
        texts[did] = "Passage: " + body

    # Which role each missing id had, so the report names the failure precisely.
    missing_detail = []
    for did in missing:
        as_pos = sorted({r["query_id"] for r in rows if r["positive_doc_id"] == did})
        as_neg = sorted({(r["query_id"], r["condition"]) for r in rows if r["negative_doc_id"] == did})
        missing_detail.append({"doc_id": did, "as_positive_for": as_pos,
                               "as_negative_for": [list(x) for x in as_neg]})

    written = {}
    per_condition_counts = {}
    if not missing:
        for cond in conditions:
            out_rows = []
            for r in [x for x in rows if x["condition"] == cond]:
                out_rows.append({
                    "query_id": r["query_id"], "query": r["query"],
                    "product_id": r["positive_doc_id"], "product_text": texts[r["positive_doc_id"]],
                    "label": "Relevant",
                    "doc_role": "positive", "condition": cond,
                    "query_group_id": r["query_group_id"],
                    "positive_in_search_topk": r["positive_in_search_topk"],
                    "n_known_positives": r["n_known_positives"],
                    "text_sha256": sha256_text(texts[r["positive_doc_id"]]),
                })
                out_rows.append({
                    "query_id": r["query_id"], "query": r["query"],
                    "product_id": r["negative_doc_id"], "product_text": texts[r["negative_doc_id"]],
                    "label": "Irrelevant",
                    "doc_role": "negative", "condition": cond,
                    "query_group_id": r["query_group_id"],
                    "negative_original_rank": r["negative_original_rank"],
                    "negative_bm25_score": r["negative_bm25_score"],
                    "negative_relevance_class": r["negative_relevance_class"],
                    "text_sha256": sha256_text(texts[r["negative_doc_id"]]),
                })
            p = out_dir / f"pairs_text_{cond}_seed{rows[0]['seed']}.jsonl"
            if p.exists() and not args.allow_overwrite:
                raise SystemExit(f"{p} exists; pass --allow_overwrite to replace it deliberately")
            with open(p, "w", encoding="utf-8") as f:
                for x in out_rows:
                    f.write(json.dumps(x) + "\n")
            written[cond] = {"path": str(p.relative_to(ROOT)).replace("\\", "/"),
                             "n_rows": len(out_rows), "bytes": p.stat().st_size,
                             "sha256": sha256_file(p)}
            per_condition_counts[cond] = {
                "n_query_groups": len({x["query_id"] for x in out_rows}),
                "n_positive_rows": sum(1 for x in out_rows if x["doc_role"] == "positive"),
                "n_negative_rows": sum(1 for x in out_rows if x["doc_role"] == "negative"),
            }

    # The same (query, document) must carry one text regardless of condition; the passage cache
    # keys on content, so this is checked here rather than assumed.
    shared_pos = {}
    for r in rows:
        k = (r["query_id"], r["positive_doc_id"])
        shared_pos.setdefault(k, set()).add(sha256_text(texts[r["positive_doc_id"]])
                                           if r["positive_doc_id"] in texts else "MISSING")
    pos_text_disagreements = [{"query_id": q, "doc_id": d} for (q, d), v in shared_pos.items()
                              if len(v) > 1]

    lengths = sorted(len(t) for t in texts.values())
    report = {
        "generated_by": "cpmr_experiment/instructions_v2/scripts/join_msmarco_pair_text.py",
        "ir_datasets_id": IRDS_ID, "library_versions": lib_versions(),
        "doc_join_rule": DOC_JOIN_RULE,
        "access_mode": "docs_store() random access (not a full docs_iter pass)",
        "inputs": {
            "pairs": str(pairs_path.relative_to(ROOT)).replace("\\", "/"),
            "pairs_sha256": sha256_file(pairs_path),
            "query_selection_sha256": sha256_file(sel_path),
            "selection_rule": sel.get("selection_rule"),
            "selection_seed": sel.get("seed"),
        },
        "preserved": {
            "n_query_ids_in_pairs": len(qids_in),
            "n_query_ids_in_selection": sel["n_queries_selected"],
            "query_ids_unchanged": qids_in == sorted(sel["query_ids"]),
            "n_pairs_in": len(rows),
            "conditions": conditions,
            "note": "no re-search, no re-selection of queries, no re-draw of negatives",
        },
        "join": {
            "n_distinct_doc_ids_needed": len(wanted),
            "n_joined": len(texts),
            "n_missing_from_corpus": len(missing),
            "missing_detail": missing_detail,
            "n_queries_with_positive_outside_search_topk":
                len({r["query_id"] for r in rows if not r["positive_in_search_topk"]}),
            "positive_outside_topk_is_not_an_error": True,
            "same_query_document_one_text": not pos_text_disagreements,
            "pos_text_disagreements": pos_text_disagreements,
        },
        "text_lengths_chars": {
            "min": lengths[0] if lengths else None,
            "median": lengths[len(lengths) // 2] if lengths else None,
            "p95": lengths[int(len(lengths) * 0.95)] if lengths else None,
            "max": lengths[-1] if lengths else None,
        },
        "per_condition": per_condition_counts,
        "output": written,
    }
    rp = out_dir / "pairs_text_join_report.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[join] doc ids needed {len(wanted)}, joined {len(texts)}, missing {len(missing)}")
    print(f"[join] text chars min/median/p95/max = "
          f"{report['text_lengths_chars']['min']}/{report['text_lengths_chars']['median']}/"
          f"{report['text_lengths_chars']['p95']}/{report['text_lengths_chars']['max']}")
    print(f"[join] query ids unchanged: {report['preserved']['query_ids_unchanged']}")
    for c, w in written.items():
        print(f"[join] {c}: {w['n_rows']} rows -> {w['path']} (sha256 {w['sha256'][:16]})")
    print(f"[join] wrote {rp}")
    if missing:
        print(f"[join] ERROR: {len(missing)} document id(s) absent from the corpus:")
        for m in missing_detail:
            print(f"        {m}")
        print("[join] not substituted and not dropped -- exiting non-zero")
        return 1
    if pos_text_disagreements:
        print(f"[join] ERROR: {len(pos_text_disagreements)} (query, doc) pairs got more than one text")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
