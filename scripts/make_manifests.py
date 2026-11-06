"""Author-side tool (run once on the machine that holds the original experiment files; CPU only): turn the frozen experiment inputs into small ID manifests.
Outputs (under --out):
  esci_rows_<set>.csv        set in {sft2, diag, sel, test300}: example_id, query_id, product_id, esci, cut   (file order of the original <set>_rows.jsonl; sft2 = the 10,000 training rows, 2 products per query)
  esci_manifest.json         per set: rows, queries, sha256 of the original rows file; product budget, max_length, seed, split rule text
  msmarco_train_pairs.csv    the 10,000 MS MARCO training rows (5,000 queries x {positive, negative}) in file order: query_id, passage_id, label, query_sha256, text_sha256, extra_json (the other fields of the original record in key order; query and product_text are restored from the corpus)
  trec_dl_judged_pairs.csv   the judged pairs of TREC-DL 2019 / 2020 in evaluation order (original BM25-pool order, judged only): year, query_id, doc_id, grade, doc_text_sha256 ; query texts: trec_dl_queries.csv (year, query_id, query_sha256)
  msmarco_manifest.json      sha256 of the original train pair file, counts, judged-id sha256 as printed by the evaluation (sha256 of json.dumps({q: sorted(v)}) over queries sorted as strings)
    python make_manifests.py --root <project root> --out <dir>"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R, O = Path(a.root), Path(a.out)
O.mkdir(parents=True, exist_ok=True)
sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
shf = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

A = R / "cpmr_experiment/esci/audit"
spl = json.load(open(A / "splits.json", encoding="utf-8"))
man = {"seed": spl["seed"], "product_budget": spl["product_budget"], "max_length": spl["max_length"], "overlap_removed": spl["overlap_removed"], "label_mapping": "E -> Relevant; S, C, I -> Irrelevant", "sets": {}}
for s in ("sft2", "diag", "sel", "test300"):
    f = A / f"{s}_rows.jsonl"
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    with open(O / f"esci_rows_{s}.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["example_id", "query_id", "product_id", "esci", "cut"])
        for r in rows:
            w.writerow([r["example_id"], r["query_id"], r["product_id"], r["esci"], int(r["cut"])])
    man["sets"][s] = {"rows": len(rows), "queries": len({r["nq"] for r in rows}), "rows_file_sha256": shf(f), "E": sum(r["esci"] == "E" for r in rows)}
(O / "esci_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")

P = R / "data/msmarco_train_v2_5k/pairs_text_basic_5000.jsonl"
rows = [json.loads(l) for l in open(P, encoding="utf-8")]
with open(O / "msmarco_train_pairs.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["query_id", "passage_id", "label", "query_sha256", "text_sha256", "extra_json"])
    for r in rows:
        rj = {k: v for k, v in r.items() if k not in ("query_id", "query", "product_id", "product_text", "label", "text_sha256")}   # extra fields in original key order; the record is query_id, query, product_id, product_text, label, <extra>, text_sha256
        w.writerow([r["query_id"], r["product_id"], r["label"], sha(r["query"]), r["text_sha256"], json.dumps(rj)])
mm = {"train_pairs_file_sha256": shf(P), "train_rows": len(rows), "train_queries": len({r["query_id"] for r in rows}), "years": {}}
tq = []
with open(O / "trec_dl_judged_pairs.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["year", "query_id", "doc_id", "grade", "doc_text_sha256"])
    for y in ("2019", "2020"):
        pool = R / f"data/scenarios/msmarco_trec{y}_bm25top100_pool.jsonl"
        n = 0
        judged = {}
        seen_q = {}
        for l in open(pool, encoding="utf-8"):
            r = json.loads(l)
            seen_q[r["query_id"]] = r["query"]
            if r["relevance"] is not None:
                w.writerow([y, r["query_id"], r["doc_id"], r["relevance"], sha(r["doc_text"])]); n += 1
                judged.setdefault(r["query_id"], []).append(r["doc_id"])
        judged = {q: v for q, v in judged.items() if v}
        for q in judged:
            tq.append((y, q, sha(seen_q[q])))
        mm["years"][y] = {"queries": len(judged), "judged_pairs": n, "pool_file_sha256": shf(pool), "judged_id_sha256": hashlib.sha256(json.dumps({q: sorted(v) for q, v in sorted(judged.items())}).encode()).hexdigest()}
with open(O / "trec_dl_queries.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["year", "query_id", "query_sha256"])
    w.writerows(tq)
# digest of the Qwen passage cache entries used by the Qwen3-4B MS MARCO trainers (restored with build_passage_cache.py; the cache file itself embeds the passage texts and is not shipped)
cache = json.load(open(R / "cpmr_experiment/instructions_v2/cache_5k/msmarco5k_basic__cache.json", encoding="utf-8"))
want = {(r["query_id"], r["product_id"]) for r in rows}
ent = sorted((v["query_id"], v["doc_id"], sha(v["passage_fixed"]), v["chars_kept"], bool(v["was_truncated"])) for v in cache["entries"].values() if (v["query_id"], v["doc_id"]) in want)
mm["qwen_passage_cache"] = {"entries": len(ent), "content_sha256": sha(json.dumps(ent)), "max_length": cache["identity"]["max_length"], "tokenizer_revision": cache["identity"]["tokenizer_identity"]["requested_revision"], "enable_thinking": cache["identity"]["enable_thinking"],
                            "corpus_version": "ir_datasets:msmarco-passage@ir_datasets-0.6.3,docs_count=8841823", "selection_seed": 0}
(O / "msmarco_manifest.json").write_text(json.dumps(mm, indent=1), encoding="utf-8")
print(json.dumps({"esci": man["sets"], "msmarco": mm}, indent=1))
