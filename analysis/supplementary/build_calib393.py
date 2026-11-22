"""Freeze the independent ESCI calibration pool BEFORE any score exists. CPU only.
Rule (fixed before any score was computed): perm[5400:] of default_rng(20261031) over the sorted train-side normalised queries, skip queries present in data/scenarios/esci_E_p1min_b10000_train.jsonl, first 200.
Rows: exactly the rule of methods/esci_audit.py (every judged product of the query, drop_duplicates(product_id, esci_label), sorted by (nq); product text with the Qwen3 tokenizer and the frozen 512-token budget, same field order / cut order).
Asserts: the query list equals calibration_pool_proposal.json (hash cbdc83098f2d7927), 200 queries / 3,939 rows, no query in diag / sel / sft / test, and the rows file format equals sel_rows.jsonl.
    python build_calib393.py --root <project root> --out <dir>"""
import argparse, hashlib, json, sys, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--out", required=True); a = ap.parse_args()
ROOT = Path(a.root); R = ROOT / "cpmr_experiment"; OUT = ROOT / a.out; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "methods"))
import esci_audit as EA  # noqa: E402  (product_text, norm_q, ESCI, MODEL; main() is not run)
from transformers import AutoTokenizer  # noqa: E402

spl = json.load(open(R / "esci/audit/splits.json"))
prop = json.load(open(R / "audit392/calibration_pool_proposal.json"))
ex = pd.read_parquet(f"{EA.ESCI}/examples.parquet"); pr = pd.read_parquet(f"{EA.ESCI}/products.parquet")
ex = ex[(ex.product_locale == "us") & (ex.small_version == 1)].copy(); pr = pr[pr.product_locale == "us"]
ex = ex.merge(pr, on=["product_locale", "product_id"], how="left", indicator=True); ex = ex[ex._merge == "both"].drop(columns="_merge")
ex["nq"] = ex["query"].map(EA.norm_q)
trq, teq = set(ex[ex.split == "train"].nq), set(ex[ex.split == "test"].nq)
train = ex[(ex.split == "train") & ~ex.nq.isin(trq & teq)]
qs = sorted(train.nq.unique()); perm = [qs[i] for i in np.random.default_rng(spl["seed"]).permutation(len(qs))]
assert perm[:200] == spl["diag"] and perm[200:400] == spl["sel"] and perm[400:5400] == spl["sft"]
old = {EA.norm_q(json.loads(l)["query"]) for l in open(ROOT / "data/scenarios/esci_E_p1min_b10000_train.jsonl", encoding="utf-8")}
pool = [q for q in perm[5400:] if q not in old][:200]
assert pool == prop["pool_queries"], "pool differs from the frozen proposal"
assert hashlib.sha256(json.dumps(pool).encode()).hexdigest()[:16] == "cbdc83098f2d7927"
assert not (set(pool) & (set(spl["diag"]) | set(spl["sel"]) | set(spl["sft"]) | set(spl["test"]))), "overlap with a frozen set"
tok = AutoTokenizer.from_pretrained(EA.MODEL, local_files_only=True)
B = 512
d = train[train.nq.isin(set(pool))].sort_values(["product_id", "example_id"])
by = {q: g.drop_duplicates(["product_id", "esci_label"]) for q, g in d.groupby("nq")}
cache, recs = {}, []
for q in sorted(pool):
    for _, r in by[q].iterrows():
        key = r["product_id"]
        if key not in cache:
            cache[key] = EA.product_text(tok, r, B)
        txt, n, cut, nfull = cache[key]
        recs.append({"example_id": int(r["example_id"]), "query_id": int(r["query_id"]), "nq": r["nq"], "query": r["query"], "product_id": key, "esci": r["esci_label"], "label": "Relevant" if r["esci_label"] == "E" else "Irrelevant",
                     "product_text": txt, "product_tokens": n, "product_tokens_full": nfull, "cut": bool(cut)})
assert len(recs) == 3939 and len({x["nq"] for x in recs}) == 200, (len(recs))
sel0 = json.loads(open(R / "esci/audit/sel_rows.jsonl", encoding="utf-8").readline())
assert list(sel0) == list(recs[0]), (list(sel0), list(recs[0]))
f = OUT / "calib_rows.jsonl"
with open(f, "w", encoding="utf-8", newline="\n") as fh:
    for x in recs:
        fh.write(json.dumps(x, ensure_ascii=False) + "\n")
sha = hashlib.sha256(f.read_bytes()).hexdigest()
man = dict(rule=prop["pool_rule"], frozen_before_any_score=True, rows=len(recs), queries=200, relevant_share=float(np.mean([x["esci"] == "E" for x in recs])), query_list_sha256_16="cbdc83098f2d7927", rows_file_sha256=sha,
           product_budget=B, cut_rows=int(sum(x["cut"] for x in recs)), classes={c: int(sum(x["esci"] == c for x in recs)) for c in "ESCI"},
           overlap_with=dict(diag=0, sel=0, sft=0, test=0), product_ids_shared_with=dict(sel=len({x["product_id"] for x in recs} & {json.loads(l)["product_id"] for l in open(R / "esci/audit/sel_rows.jsonl", encoding="utf-8")}),
                                                                                       sft2=len({x["product_id"] for x in recs} & {json.loads(l)["product_id"] for l in open(R / "esci/audit/sft2_rows.jsonl", encoding="utf-8")}),
                                                                                       test300=len({x["product_id"] for x in recs} & {json.loads(l)["product_id"] for l in open(R / "esci/audit/test300_rows.jsonl", encoding="utf-8")})))
(OUT / "calib_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
print(json.dumps(man, indent=1))
