"""Verify the restored experiment inputs against the manifests (CPU only, no model, no network):
  * ESCI row files: row / query counts, Relevant (E) counts and sha256 of every file equal the manifest; the normalized queries of the training, diagnostic, validation and test splits are pairwise disjoint;
    example_id / query_id / product_id / label of every row equal the manifest rows; the (query, product) pairs are unique inside a set.
  * MS MARCO: 10,000 training rows (5,000 queries, 5,000 Relevant / 5,000 Irrelevant, one positive and one negative per query) and the sha256 of the pair file; TREC-DL judged-pair counts and judged-id hash of the pool files.
    python scripts/verify_data.py [--skip-esci] [--skip-msmarco]"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data" / "manifests"
ap = argparse.ArgumentParser()
ap.add_argument("--esci-dir", default="cpmr_experiment/esci/audit")
ap.add_argument("--train-dir", default="data/msmarco_train_v2_5k")
ap.add_argument("--pool-dir", default="data/scenarios")
ap.add_argument("--skip-esci", action="store_true")
ap.add_argument("--skip-msmarco", action="store_true")
a = ap.parse_args()
bad = []


def check(name, ok, detail=""):
    print(("OK   " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not ok:
        bad.append(name)


if not a.skip_esci:
    man = json.load(open(MAN / "esci_manifest.json", encoding="utf-8"))
    nq = {}
    for s, meta in man["sets"].items():
        f = ROOT / a.esci_dir / f"{s}_rows.jsonl"
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        mrows = list(csv.DictReader(open(MAN / f"esci_rows_{s}.csv", encoding="utf-8")))
        check(f"esci {s}: rows", len(rows) == meta["rows"] == len(mrows), f"{len(rows)}")
        check(f"esci {s}: queries", len({r['nq'] for r in rows}) == meta["queries"], f"{len({r['nq'] for r in rows})}")
        check(f"esci {s}: Relevant (E) rows", sum(r["esci"] == "E" for r in rows) == meta["E"], str(meta["E"]))
        check(f"esci {s}: ids and labels equal the manifest", all(str(r["example_id"]) == m["example_id"] and r["product_id"] == m["product_id"] and r["esci"] == m["esci"] and str(r["query_id"]) == m["query_id"] for r, m in zip(rows, mrows)))
        check(f"esci {s}: label mapping", all((r["label"] == "Relevant") == (r["esci"] == "E") for r in rows))
        check(f"esci {s}: (query, product) pairs unique", len({(r["nq"], r["product_id"]) for r in rows}) == len(rows))
        check(f"esci {s}: sha256", hashlib.sha256(f.read_bytes()).hexdigest() == meta["rows_file_sha256"])
        nq[s] = {r["nq"] for r in rows}
    names = list(nq)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            check(f"esci normalized-query overlap {names[i]} / {names[j]} = 0", not (nq[names[i]] & nq[names[j]]), str(len(nq[names[i]] & nq[names[j]])))
if not a.skip_msmarco:
    mm = json.load(open(MAN / "msmarco_manifest.json", encoding="utf-8"))
    f = ROOT / a.train_dir / "pairs_text_basic_5000.jsonl"
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    qs = {}
    for r in rows:
        qs.setdefault(r["query_id"], []).append(r["label"])
    check("msmarco train: rows / queries", len(rows) == mm["train_rows"] and len(qs) == mm["train_queries"], f"{len(rows)} / {len(qs)}")
    check("msmarco train: one Relevant and one Irrelevant per query", all(sorted(v) == ["Irrelevant", "Relevant"] for v in qs.values()))
    check("msmarco train: file sha256", hashlib.sha256(f.read_bytes()).hexdigest() == mm["train_pairs_file_sha256"])
    for y, meta in mm["years"].items():
        pool = ROOT / a.pool_dir / f"msmarco_trec{y}_bm25top100_pool.jsonl"
        pr = [json.loads(l) for l in open(pool, encoding="utf-8")]
        judged = {}
        for r in pr:
            if r["relevance"] is not None:
                judged.setdefault(r["query_id"], []).append(r["doc_id"])
        ids = hashlib.sha256(json.dumps({q: sorted(v) for q, v in sorted(judged.items())}).encode()).hexdigest()
        check(f"trec-dl {y}: judged queries / pairs", len(judged) == meta["queries"] and sum(len(v) for v in judged.values()) == meta["judged_pairs"], f"{len(judged)} / {sum(len(v) for v in judged.values())}")
        check(f"trec-dl {y}: judged-id hash", ids == meta["judged_id_sha256"])
print("\nall checks passed" if not bad else f"\n{len(bad)} check(s) FAILED")
sys.exit(1 if bad else 0)
