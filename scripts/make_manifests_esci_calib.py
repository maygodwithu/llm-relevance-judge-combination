"""Author-side tool: write the ID manifest of the independent ESCI calibration pool (200 queries, 3,939 judged pairs) used by the independent-calibration check
(`analysis/supplementary/`), in the same layout as the other ESCI row manifests, and add the set `calib` to data/manifests/esci_manifest.json.
`scripts/restore_data.py esci` then rebuilds `cpmr_experiment/esci/audit/calib_rows.jsonl` from the official ESCI files and checks its SHA-256, like every other ESCI set.
    python scripts/make_manifests_esci_calib.py --rows <original calib_rows.jsonl>"""
import argparse, csv, hashlib, json
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--out", default="data/manifests"); a = ap.parse_args()
O = Path(a.out)
rows = [json.loads(l) for l in open(a.rows, encoding="utf-8")]
with open(O / "esci_rows_calib.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["example_id", "query_id", "product_id", "esci", "cut"])
    for r in rows:
        w.writerow([r["example_id"], r["query_id"], r["product_id"], r["esci"], int(r["cut"])])
man = json.load(open(O / "esci_manifest.json", encoding="utf-8"))
man["sets"]["calib"] = {"rows": len(rows), "queries": len({r["nq"] for r in rows}), "rows_file_sha256": hashlib.sha256(Path(a.rows).read_bytes()).hexdigest(), "E": sum(r["esci"] == "E" for r in rows)}
man["calib_rule"] = ("independent calibration pool, fixed before any score was computed: the queries of the official training split (normalized query text, not also in the test split) are sorted, permuted with numpy default_rng(20261031) (the seed of the splits), "
                     "the first 5,400 (the training, diagnostic and validation queries) are left out, queries that appear in the training file of an earlier project are skipped, and the next 200 queries are taken with every judged product (rows built exactly like the other sets; 512-token product budget); "
                     "disjoint by query from sft2, diag, sel and test300 (checked by scripts/verify_data.py)")
(O / "esci_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
print(json.dumps(man["sets"]["calib"]))
