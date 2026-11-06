"""ESCI extension (experiment log 156; Qwen3-4B Base, second server) - CPU audit, split, row freeze, product-text cache, token audit. No model scores, no GPU.
Data: official ESCI (amazon-science/esci-data) examples/products parquet as stored in <server-home>/esci_official (checked against the official row counts), product_locale 'us', small_version == 1, official train / test.
Label: E (exact) = Relevant, S/C/I = Irrelevant; the original E/S/C/I label is kept in every row.
Split (frozen before any score): normalized query = NFKC, lower case, whitespace collapsed. Train-side normalized queries that also occur in the official test are removed before sampling (rule stated here). The remaining distinct
normalized queries are sorted, permuted with numpy default_rng(20261031); first 200 = diagnostic, next 200 = selection, next min(5000, rest) = SFT. Diagnostic / selection keep every judged product of their queries.
SFT keeps, per query, at most 20 judged products drawn uniformly (default_rng([20261031, 2]) permutation per query in sorted-query order; all if fewer); the sample is fixed before looking at labels or scores.
Test: 1,000 normalized queries drawn uniformly (default_rng([20261031, 3])) from the official test queries, every judged product of them (label balance untouched).
Product text (fixed field order, no generated content, empty fields omitted, whitespace collapsed): 'Product:\\nTitle: ..\\nBrand: ..\\nColor: ..\\nBullet points: ..\\nDescription: ..'; if the block exceeds --product_budget tokens (Qwen3 tokenizer)
the description is cut first (token prefix), then the bullet points, then the title; the cached text does not depend on any instruction.
    python methods/esci_audit.py --out cpmr_experiment/esci/audit --product_budget 0      (0 = distribution only, nothing written)
    python methods/esci_audit.py --out cpmr_experiment/esci/audit --product_budget 384"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261031
LOC = "us"
ESCI = os.environ.get("ESCI_DIR", "external/esci-data")        # directory with the official examples / products parquet files
MODEL = os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-4B")
SUFFIX = '\n\nOutput exactly one label: "Relevant" or "Irrelevant".'
P_TEXT = {
    "P0": "Judge the product as relevant if the provided product information explicitly confirms that it satisfies every requirement stated in the query, without relying on unstated assumptions. Otherwise, judge the product as irrelevant.",
    "P1": "Judge the product as relevant if it matches the requested product and satisfies the requirements stated in the query. Otherwise, judge the product as irrelevant.",
    "P2": "Judge the product as relevant if it fulfills the main shopping need expressed in the query, even if some requested attributes differ. Otherwise, judge the product as irrelevant.",
    "P3": "Judge the product as relevant if it matches the requested product or is a usable substitute for it. Otherwise, judge the product as irrelevant.",
    "P4": "Judge the product as relevant if it matches the requested product, is a usable substitute, or is a complementary product intended to be used with it. Otherwise, judge the product as irrelevant.",
}
ws = lambda s: re.sub(r"\s+", " ", s).strip()
norm_q = lambda q: ws(unicodedata.normalize("NFKC", q).lower())
sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()


def clean(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else ws(str(v))


def product_text(tok, r, budget):
    f = {"title": clean(r["product_title"]), "brand": clean(r["product_brand"]), "color": clean(r["product_color"]), "bullets": clean(r["product_bullet_point"]), "desc": clean(r["product_description"])}
    names = [("title", "Title"), ("brand", "Brand"), ("color", "Color"), ("bullets", "Bullet points"), ("desc", "Description")]

    def render(ff):
        return "Product:\n" + "\n".join(f"{lab}: {ff[k]}" for k, lab in names if ff[k])
    full = render(f)
    n_full = len(tok(full, add_special_tokens=False)["input_ids"])
    if not budget or n_full <= budget:
        return full, n_full, False, n_full
    g = dict(f)
    for k in ("desc", "bullets", "title"):                                     # description first, then bullets, then title
        ids = tok(g[k], add_special_tokens=False)["input_ids"]
        lo, hi = 0, len(ids)
        while lo < hi:                                                       # longest token prefix of the field that keeps the block within the budget
            mid = (lo + hi + 1) // 2
            g[k] = tok.decode(ids[:mid]).strip()
            if len(tok(render(g), add_special_tokens=False)["input_ids"]) <= budget:
                lo = mid
            else:
                hi = mid - 1
        g[k] = tok.decode(ids[:lo]).strip() if lo else ""
        out = render(g)
        n = len(tok(out, add_special_tokens=False)["input_ids"])
        if n <= budget:
            return out, n, True, n_full
    return out, n, True, n_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--product_budget", type=int, default=0)
    ap.add_argument("--n_sft", type=int, default=5000)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    L = []

    def P(s=""):
        L.append(s); print(s, flush=True)
    ex_all = pd.read_parquet(next(p for p in (f"{ESCI}/shopping_queries_dataset_examples.parquet", f"{ESCI}/examples.parquet") if os.path.exists(p)))
    pr_all = pd.read_parquet(next(p for p in (f"{ESCI}/shopping_queries_dataset_products.parquet", f"{ESCI}/products.parquet") if os.path.exists(p)))
    P(f"official files: examples {len(ex_all):,} rows (official 2,621,288), products {len(pr_all):,} rows (official 1,814,924); sha256 examples {'n/a'}.. (see SHA256SUMS in the folder)")
    ex = ex_all[(ex_all.product_locale == LOC) & (ex_all.small_version == 1)].copy()
    P(f"us small_version=1: train {int((ex.split == 'train').sum()):,} rows / {ex[ex.split == 'train'].query_id.nunique():,} query ids; test {int((ex.split == 'test').sum()):,} rows / {ex[ex.split == 'test'].query_id.nunique():,} query ids")
    pr = pr_all[pr_all.product_locale == LOC]
    assert pr.product_id.is_unique
    P(f"us products {len(pr):,} (product_id unique within locale: {pr_all[pr_all.product_locale == LOC].product_id.is_unique})")
    ex = ex.merge(pr, on=["product_locale", "product_id"], how="left", indicator=True)
    miss = ex[ex._merge == "left_only"]
    P(f"examples without product record: {len(miss)} (train {int((miss.split == 'train').sum())}, test {int((miss.split == 'test').sum())})")
    ex = ex[ex._merge == "both"].drop(columns="_merge")
    ex["nq"] = ex["query"].map(norm_q)
    for sp in ("train", "test"):
        d = ex[ex.split == sp]
        P(f"[{sp}] labels E/S/C/I = " + "/".join(str(int((d.esci_label == k).sum())) for k in "ESCI") + f"; distinct normalized queries {d.nq.nunique():,} (query ids {d.query_id.nunique():,}); mean products per query {len(d) / d.nq.nunique():.2f}")
        g = d.groupby("nq").query_id.nunique()
        P(f"[{sp}] normalized queries with several query ids: {int((g > 1).sum())}")
        dup = d.groupby(["nq", "product_id"]).esci_label.agg(["size", "nunique"])
        P(f"[{sp}] repeated (normalized query, product) pairs: {int((dup['size'] > 1).sum())}; with conflicting labels: {int((dup['nunique'] > 1).sum())}")
    trq, teq = set(ex[ex.split == "train"].nq), set(ex[ex.split == "test"].nq)
    ov = trq & teq
    P(f"normalized queries in train and test: {len(ov)} (train-side rows removed: {int(ex[(ex.split == 'train') & ex.nq.isin(ov)].shape[0])}); product ids shared between train and test: {len(set(ex[ex.split == 'train'].product_id) & set(ex[ex.split == 'test'].product_id)):,} of {ex[ex.split == 'test'].product_id.nunique():,} test products")
    for c in ("product_title", "product_brand", "product_color", "product_bullet_point", "product_description"):
        P(f"missing/empty {c}: train {100 * (ex[ex.split == 'train'][c].map(clean) == '').mean():.1f}%, test {100 * (ex[ex.split == 'test'][c].map(clean) == '').mean():.1f}%")
    # ----- split
    train = ex[(ex.split == "train") & ~ex.nq.isin(ov)]
    qs = sorted(train.nq.unique())
    perm = [qs[i] for i in np.random.default_rng(SEED).permutation(len(qs))]
    diag, sel = perm[:200], perm[200:400]
    sft = perm[400:400 + args.n_sft]
    P(f"split: {len(qs):,} train-side normalized queries after removing test overlap -> diagnostic {len(diag)}, selection {len(sel)}, SFT {len(sft)} (spare {len(qs) - 400 - len(sft)})")
    rng_sft = np.random.default_rng([SEED, 2])
    tq = sorted(teq)
    test_q = sorted(tq[i] for i in np.random.default_rng([SEED, 3]).permutation(len(tq))[:1000])
    byq = {sp: {q: d.sort_values(["product_id", "example_id"]) for q, d in ex[ex.split == sp].groupby("nq")} for sp in ("train", "test")}

    def rows_for(qlist, sp, cap=None, rng=None):
        out = []
        for q in sorted(qlist):
            d = byq[sp][q].drop_duplicates(["product_id", "esci_label"])
            if cap and len(d) > cap:
                d = d.iloc[sorted(rng.permutation(len(d))[:cap])]
            out.append(d)
        return pd.concat(out)
    sets = {"diag": rows_for(diag, "train"), "sel": rows_for(sel, "train"), "sft": rows_for(sft, "train", 20, rng_sft), "test": rows_for(test_q, "test")}
    P("")
    for k, d in sets.items():
        c = d.esci_label.value_counts()
        P(f"{k:5s}: queries {d.nq.nunique():,}, rows {len(d):,}, E/S/C/I = {c.get('E', 0)}/{c.get('S', 0)}/{c.get('C', 0)}/{c.get('I', 0)} (E {100 * c.get('E', 0) / len(d):.1f}%), queries with E and non-E {int((d.groupby('nq').esci_label.agg(lambda s: ('E' in set(s)) and (set(s) != {'E'}))).sum())}, queries without E {int((d.groupby('nq').esci_label.agg(lambda s: 'E' not in set(s))).sum())}, products {d.product_id.nunique():,}")
    P(f"products shared between SFT rows and test rows: {len(set(sets['sft'].product_id) & set(sets['test'].product_id)):,}; between SFT and selection {len(set(sets['sft'].product_id) & set(sets['sel'].product_id)):,}, SFT and diagnostic {len(set(sets['sft'].product_id) & set(sets['diag'].product_id)):,}")
    # ----- token audit
    full_n = {}
    sample = pd.concat([sets["sft"].sample(3000, random_state=0), sets["test"]]).drop_duplicates("example_id")
    ntok = np.array([len(tok(product_text(tok, r, 0)[0], add_special_tokens=False)["input_ids"]) for _, r in sample.iterrows()])
    P(f"\nproduct block tokens (no budget; {len(ntok):,} rows from SFT sample + test): mean {ntok.mean():.0f}, median {np.median(ntok):.0f}, p90 {np.percentile(ntok, 90):.0f}, p95 {np.percentile(ntok, 95):.0f}, p99 {np.percentile(ntok, 99):.0f}, max {ntok.max()}; share > 256/384/512/768: " + "/".join(f"{100 * (ntok > b).mean():.1f}%" for b in (256, 384, 512, 768)) + "")
    qt = np.array([len(tok(q, add_special_tokens=False)["input_ids"]) for d in sets.values() for q in d["query"].unique()])
    itok = {k: len(tok(v + SUFFIX, add_special_tokens=False)["input_ids"]) for k, v in P_TEXT.items()}
    P(f"query tokens: mean {qt.mean():.1f}, max {qt.max()}; instruction tokens (with suffix): {itok}; APO-CF body limit 128 + suffix ~{len(tok(SUFFIX, add_special_tokens=False)['input_ids'])}")
    if not args.product_budget:
        (Path(args.out)).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "esci_audit_distribution.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
        return
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    B = args.product_budget
    qmax = int(qt.max())
    max_length = B + qmax + 128 + 40 + 40 + 8                                   # product block + longest query + longest instruction (128-token body + suffix) + template/'Query:' markers + response
    P(f"\nproduct budget {B} tokens; max_length (frozen) = {B} + {qmax} (longest query) + 128 + 40 (suffix) + 40 (template, markers) + 8 (response) = {max_length}")
    cache, stat = {}, collections.Counter()
    meta = {}
    for k, d in sets.items():
        recs = []
        for _, r in d.iterrows():
            key = r["product_id"]
            if key not in cache:
                cache[key] = product_text(tok, r, B)
            txt, n, cut, nfull = cache[key]
            stat[(k, "rows")] += 1; stat[(k, "cut")] += cut
            recs.append({"example_id": int(r["example_id"]), "query_id": int(r["query_id"]), "nq": r["nq"], "query": r["query"], "product_id": key, "esci": r["esci_label"], "label": "Relevant" if r["esci_label"] == "E" else "Irrelevant",
                         "product_text": txt, "product_tokens": n, "product_tokens_full": nfull, "cut": bool(cut)})
        f = out / f"{k}_rows.jsonl"
        with open(f, "w", encoding="utf-8", newline="\n") as fh:
            for x in recs:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        meta[k] = {"rows": len(recs), "queries": len({x["nq"] for x in recs}), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        P(f"{k:5s}: rows {len(recs):,}, product texts cut to the budget {stat[(k, 'cut')]:,} ({100 * stat[(k, 'cut')] / len(recs):.1f}%), mean product tokens {np.mean([x['product_tokens'] for x in recs]):.0f} (full {np.mean([x['product_tokens_full'] for x in recs]):.0f})")
    (out / "splits.json").write_text(json.dumps({"seed": SEED, "rule": __doc__, "diag": diag, "sel": sel, "sft": sft, "test": test_q, "sets": meta, "product_budget": B, "max_length": max_length, "overlap_removed": len(ov)}, ensure_ascii=False, indent=1), encoding="utf-8")
    man = {"note": "ESCI-P0..P4 originals of experiment log 156, frozen before any score", "entries": [{"display_id": k, "source_id": k, "full": v + SUFFIX, "sha256_full": sha(v + SUFFIX)} for k, v in P_TEXT.items()]}
    (out / "esci_p_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    (out / "esci_audit.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    for k in meta:
        P(f"sha256 {k}_rows.jsonl {meta[k]['sha256']}")
    (out / "esci_audit.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
