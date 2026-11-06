"""Restore the exact experiment inputs from the public sources and the ID manifests in data/manifests/ (no sampling is repeated: every list is read from a manifest).
    python scripts/restore_data.py esci     --esci-dir <dir with the official ESCI parquet files> [--tokenizer Qwen/Qwen3-4B] [--out cpmr_experiment/esci/audit]
    python scripts/restore_data.py msmarco  [--train-out data/msmarco_train_v2_5k] [--pool-out data/scenarios]
    python scripts/restore_data.py msmarco-cache [--local-files-only]      (needed only by the Qwen3-4B MS MARCO trainers; needs the Qwen3-4B tokenizer, revision pinned in the manifest)
ESCI: official files of https://github.com/amazon-science/esci-data (shopping_queries_dataset_examples.parquet, shopping_queries_dataset_products.parquet; the renamed examples.parquet / products.parquet are accepted).
      Rows are rebuilt with methods/esci_audit.py::product_text (same field order, whitespace rule and 512-token budget of the Qwen3 tokenizer) and checked against the sha256 of the original row files.
MS MARCO / TREC-DL: through the `ir_datasets` package (msmarco-passage/train, msmarco-passage/trec-dl-2019/judged, msmarco-passage/trec-dl-2020/judged; downloads the public collection on first use).
      The training pair file is rebuilt byte-identically (checked by sha256). The TREC-DL pool files are rebuilt with the judged pairs only (the unjudged BM25 pool entries never enter an evaluation); every query / document text is checked against its recorded sha256 and the judged-id hash of the evaluation is compared."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data" / "manifests"
sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()


def esci(a):
    import pandas as pd
    sys.path.insert(0, str(ROOT / "methods"))
    from esci_audit import norm_q, product_text
    from transformers import AutoTokenizer
    man = json.load(open(MAN / "esci_manifest.json", encoding="utf-8"))
    d = Path(a.esci_dir)
    ex_f = next(p for p in (d / "shopping_queries_dataset_examples.parquet", d / "examples.parquet") if p.exists())
    pr_f = next(p for p in (d / "shopping_queries_dataset_products.parquet", d / "products.parquet") if p.exists())
    ex = pd.read_parquet(ex_f)
    ex = ex[(ex.product_locale == "us") & (ex.small_version == 1)]
    pr = pd.read_parquet(pr_f)
    pr = pr[pr.product_locale == "us"]
    ex = ex.merge(pr, on=["product_locale", "product_id"], how="left", indicator=True)
    ex = ex[ex._merge == "both"].drop(columns="_merge").set_index("example_id", drop=False)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    budget = man["product_budget"]
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    cache, bad = {}, 0
    for s, meta in man["sets"].items():
        recs = []
        for r in csv.DictReader(open(MAN / f"esci_rows_{s}.csv", encoding="utf-8")):
            row = ex.loc[int(r["example_id"])]
            assert row["product_id"] == r["product_id"] and int(row["query_id"]) == int(r["query_id"]) and row["esci_label"] == r["esci"], (s, r)
            key = row["product_id"]
            if key not in cache:
                cache[key] = product_text(tok, row, budget)
            txt, n, cut, nfull = cache[key]
            assert int(cut) == int(r["cut"])
            recs.append({"example_id": int(row["example_id"]), "query_id": int(row["query_id"]), "nq": norm_q(row["query"]), "query": row["query"], "product_id": key, "esci": row["esci_label"],
                         "label": "Relevant" if row["esci_label"] == "E" else "Irrelevant", "product_text": txt, "product_tokens": n, "product_tokens_full": nfull, "cut": bool(cut)})
        f = out / f"{s}_rows.jsonl"
        with open(f, "w", encoding="utf-8", newline="\n") as fh:
            for x in recs:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        ok = h == meta["rows_file_sha256"]
        bad += not ok
        print(f"{s:8s} rows {len(recs):6d} queries {len({r['nq'] for r in recs}):5d} sha256 {'OK' if ok else 'MISMATCH ' + h[:16]}")
    (out / "splits.json").write_text(json.dumps({"seed": man["seed"], "product_budget": budget, "max_length": man["max_length"], "overlap_removed": man["overlap_removed"], "sets": man["sets"],
                                                 "note": "restored; the query lists of the original splits.json are not needed by the trainers (they read max_length only)"}, indent=1), encoding="utf-8")
    sys.exit(1 if bad else 0)


def msmarco(a):
    import ir_datasets
    mm = json.load(open(MAN / "msmarco_manifest.json", encoding="utf-8"))
    rows = list(csv.DictReader(open(MAN / "msmarco_train_pairs.csv", encoding="utf-8")))
    ds = ir_datasets.load("msmarco-passage/train")
    store = ds.docs_store()
    want_q = {r["query_id"] for r in rows}
    qtext = {q.query_id: q.text for q in ds.queries_iter() if q.query_id in want_q}
    texts = {}
    for did in sorted({r["passage_id"] for r in rows}):
        d = store.get(did)
        texts[did] = "Passage: " + f"{getattr(d, 'title', '') or ''}\n{d.text or ''}"
    out = ROOT / a.train_out
    out.mkdir(parents=True, exist_ok=True)
    f = out / "pairs_text_basic_5000.jsonl"
    with open(f, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            assert sha(qtext[r["query_id"]]) == r["query_sha256"] and sha(texts[r["passage_id"]]) == r["text_sha256"], r["query_id"]
            x = {"query_id": r["query_id"], "query": qtext[r["query_id"]], "product_id": r["passage_id"], "product_text": texts[r["passage_id"]], "label": r["label"]}
            x.update(json.loads(r["extra_json"]))
            x["text_sha256"] = r["text_sha256"]
            fh.write(json.dumps(x) + "\n")
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    ok = h == mm["train_pairs_file_sha256"]
    print(f"train pairs {len(rows)} rows, {len(want_q)} queries, file sha256 {'OK' if ok else 'MISMATCH ' + h[:16]}")
    bad = not ok
    qexp = {(r["year"], r["query_id"]): r["query_sha256"] for r in csv.DictReader(open(MAN / "trec_dl_queries.csv", encoding="utf-8"))}
    pairs = list(csv.DictReader(open(MAN / "trec_dl_judged_pairs.csv", encoding="utf-8")))
    for y in ("2019", "2020"):
        d2 = ir_datasets.load(f"msmarco-passage/trec-dl-{y}/judged")
        qs = {q.query_id: q.text for q in d2.queries_iter()}
        st = d2.docs_store()
        P = [p for p in pairs if p["year"] == y]
        pool = ROOT / a.pool_out / f"msmarco_trec{y}_bm25top100_pool.jsonl"
        pool.parent.mkdir(parents=True, exist_ok=True)
        judged = {}
        with open(pool, "w", encoding="utf-8", newline="\n") as fh:
            for p in P:
                t = st.get(p["doc_id"]).text
                assert sha(qs[p["query_id"]]) == qexp[(y, p["query_id"])] and sha(t) == p["doc_text_sha256"], (y, p["query_id"], p["doc_id"])
                fh.write(json.dumps({"query_id": p["query_id"], "query": qs[p["query_id"]], "doc_id": p["doc_id"], "doc_text": t, "relevance": int(p["grade"])}, ensure_ascii=False) + "\n")
                judged.setdefault(p["query_id"], []).append(p["doc_id"])
        ids = hashlib.sha256(json.dumps({q: sorted(v) for q, v in sorted(judged.items())}).encode()).hexdigest()
        ok = ids == mm["years"][y]["judged_id_sha256"] and len(judged) == mm["years"][y]["queries"] and len(P) == mm["years"][y]["judged_pairs"]
        bad |= not ok
        print(f"TREC-DL {y}: {len(judged)} queries, {len(P)} judged pairs, judged-id sha256 {'OK' if ok else 'MISMATCH'}")
    sys.exit(1 if bad else 0)


def cache(a):
    """Rebuild the Qwen3-4B passage cache read by the Qwen MS MARCO trainers (instructions_v2/cache_5k) from the restored training pairs with the original builder and arguments, then compare the entry contents with the recorded digest."""
    import subprocess
    mm = json.load(open(MAN / "msmarco_manifest.json", encoding="utf-8"))["qwen_passage_cache"]
    c = ROOT / "cpmr_experiment/instructions_v2"
    cmd = [sys.executable, str(c / "scripts/build_passage_cache.py"), "--pairs", "data/msmarco_train_v2_5k/pairs_text_basic_5000.jsonl", "--role", "train", "--pool_condition", "basic", "--config", "configs/qwen3_4b_msm_loss.yaml",
           "--enable_thinking", mm["enable_thinking"], "--tokenizer_revision", mm["tokenizer_revision"], "--dataset", "msmarco-passage", "--split", "train", "--corpus_version", mm["corpus_version"], "--selection_seed", str(mm["selection_seed"]),
           "--cache_out", str(c / "cache_5k/msmarco5k_basic__cache.json"), "--audit_out", str(c / "audit_5k/msmarco5k_basic__invariance.json"), "--eligibility_out", str(c / "cache_5k/msmarco5k_basic__eligibility.json")]
    if a.local_files_only:
        cmd.append("--local_files_only")
    (c / "cache_5k").mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    d = json.load(open(c / "cache_5k/msmarco5k_basic__cache.json", encoding="utf-8"))
    ent = sorted((v["query_id"], v["doc_id"], sha(v["passage_fixed"]), v["chars_kept"], bool(v["was_truncated"])) for v in d["entries"].values())
    ok = len(ent) == mm["entries"] and sha(json.dumps(ent)) == mm["content_sha256"]
    print(f"passage cache: {len(ent)} entries, content digest {'OK' if ok else 'MISMATCH'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("esci")
    e.add_argument("--esci-dir", required=True)
    e.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    e.add_argument("--out", default="cpmr_experiment/esci/audit")
    m = sp.add_parser("msmarco")
    m.add_argument("--train-out", default="data/msmarco_train_v2_5k")
    m.add_argument("--pool-out", default="data/scenarios")
    k = sp.add_parser("msmarco-cache")
    k.add_argument("--local-files-only", action="store_true")
    a = ap.parse_args()
    {"esci": esci, "msmarco": msmarco, "msmarco-cache": cache}[a.cmd](a)
