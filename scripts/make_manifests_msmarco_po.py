"""Author-side tool: write the ID manifests of the two additional MS MARCO sets that the protocol of the paper needs next to the 5,000 training queries:
  data/manifests/msmarco_val200_pairs.csv   the 200-query validation set (400 pairs; thresholds and PO selection)
  data/manifests/msmarco_diag200_pairs.csv  the 200-query PO diagnostic set (400 pairs; confusion-matrix feedback of APO-CF)
and add their entries to data/manifests/msmarco_manifest.json. Same record layout as msmarco_train_pairs.csv; no text is stored (query / passage text are restored from ir_datasets and checked by SHA-256).
    python scripts/make_manifests_msmarco_po.py --src <original data/msmarco_po directory>"""
import argparse, csv, hashlib, json
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", default="data/manifests"); a = ap.parse_args()
S, O = Path(a.src), Path(a.out)
sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
shf = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
SETS = {"val200": (S / "val200/pairs_text_basic_seed0.jsonl", "msmarco_val200_pairs.csv"), "diag200": (S / "diag200/pairs_text_basic_diag200.jsonl", "msmarco_diag200_pairs.csv")}
mm = json.load(open(O / "msmarco_manifest.json", encoding="utf-8"))
for name, (p, fn) in SETS.items():
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    with open(O / fn, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\r\n" if False else "\n"); w.writerow(["query_id", "passage_id", "label", "query_sha256", "text_sha256", "extra_json"])
        for r in rows:
            rj = {k: v for k, v in r.items() if k not in ("query_id", "query", "product_id", "product_text", "label", "text_sha256")}
            w.writerow([r["query_id"], r["product_id"], r["label"], sha(r["query"]), r["text_sha256"], json.dumps(rj)])
    mm[name] = {"pairs_file_sha256": shf(p), "rows": len(rows), "queries": len({r["query_id"] for r in rows}), "relevant": sum(r["label"] == "Relevant" for r in rows), "manifest": f"data/manifests/{fn}",
                "restored_file": "data/msmarco_po/val200/pairs_text_basic_seed0.jsonl" if name == "val200" else "data/msmarco_po/diag200/pairs_text_basic_diag200.jsonl"}
mm["val200"]["rule"] = ("200 queries of the MS MARCO passage training split, disjoint from the 5,000 SFT queries and from the diagnostic set. They were drawn from the 400 queries added when the 5,200-query selection (random.Random(0) over the training queries "
                       "that have a known positive) was extended to 5,600: 241 of them have a known positive in the BM25 top-100, and 200 of these were sampled with random.Random('t1-mipro-val-v1'). The pairs follow the same one-positive / one-unlabelled rule as the training pairs: "
                       "positive = the known positive with the smallest passage id (string order); negative = one passage drawn uniformly (seed 0) from the BM25 top-1,000 of the query after removing all known positives, labelled Irrelevant (unlabelled passage; never redrawn). "
                       "BM25 = the repository's own implementation (k1 1.2, b 0.75). No TREC-DL query is used; the set is used for instruction selection (PO) and for threshold selection.")
mm["diag200"]["rule"] = ("200 queries of the 5,200-query selection that are not used for SFT (the tail of the string-sorted selection); the same sampling rule and pair builder as the training pairs, rows identical to the corresponding rows of the 5,200-query pair file; "
                        "used only as the confusion-matrix feedback set of PO.")
(O / "msmarco_manifest.json").write_text(json.dumps(mm, indent=1), encoding="utf-8")
print(json.dumps({k: mm[k] for k in ("val200", "diag200")}, indent=1))
