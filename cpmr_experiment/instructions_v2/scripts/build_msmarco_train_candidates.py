"""
BM25 top-K candidate generation for the OFFICIAL MS MARCO passage train split.

Why build our own index instead of using precomputed hard-negative files: the two candidate
files on the Hub are undocumented (no generation code, no BM25 settings recorded), so their
relation to an original BM25 rank is unknown. Building the index ourselves makes the rank and
the score ours, which is what a "same pool, different rank band" difficulty manipulation
requires. See the experiment record.

`cpmr_experiment/beir_pilot/audit/build_beir_candidates.py` is NOT modified -- v1 evaluation
pools must stay reproducible. This is a separate file, and it changes three things
deliberately, each recorded in the output:

  1. TOP_K 100 -> 1000, and the ORIGINAL rank and score are stored per candidate.
  2. Ties are broken explicitly by ascending doc id. The v1 code used
     `np.argsort(-scores)`, whose default quicksort is NOT stable, so equal-score documents
     came out in an arbitrary order and the rank-25 boundary was not reproducible. That
     matters here precisely because the hard condition is defined by a rank cut.
  3. No document text is written. 100 queries x 1000 candidates is already 100k rows; text is
     joined later from the corpus.

Scoring details, recorded rather than assumed (this is NOT Lucene/ElasticSearch-equivalent):
  * k1 = 1.2, b = 0.75
  * idf(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1), clipped at 0 (Lucene-style form)
  * tokenizer: sklearn CountVectorizer, token_pattern=r"(?u)\\b\\w+\\b", lowercase, no
    stemming, no stopword removal
  * document string = "{title}\\n{text}" with title="" when absent (MS MARCO passages have no
    title, so this reduces to "\\n{text}")
  * QUERY TERM FREQUENCY IS IGNORED: query terms are de-duplicated into a set, so a term
    repeated in the query contributes once. Real BM25 weights by qtf. Kept identical to the v1
    implementation so that pools built by both are comparable.
  * only documents with score > 0 are returned; a query whose terms are all out of vocabulary
    returns zero candidates and is recorded, not skipped silently.
  * qrels positives are NEVER inserted into the result. What the search returns is what is
    stored.

CPU only. No GPU, no tokenizer, no model.

Usage (server, from project root):
  # 1. select the query sample and write the id list FIRST (no search yet)
  python cpmr_experiment/instructions_v2/scripts/build_msmarco_train_candidates.py \
      --stage select --n_queries 100 --seed 0 --out_dir data/msmarco_train_v2
  # 2. build the index and search only those queries
  python cpmr_experiment/instructions_v2/scripts/build_msmarco_train_candidates.py \
      --stage search --out_dir data/msmarco_train_v2 --top_k 1000
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

K1, B = 1.2, 0.75
TOKEN_PATTERN = r"(?u)\b\w+\b"
IRDS_ID = "msmarco-passage/train"
SELECTION_RULE = ("queries of msmarco-passage/train that have at least one qrels entry with "
                  "relevance >= 1; shuffled with random.Random(seed) and truncated. Model "
                  "scores are never consulted.")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def sha256_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def peak_rss_gb():
    try:
        import resource
        # ru_maxrss is KiB on Linux
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), 3)
    except Exception:  # noqa: BLE001
        return None


def lib_versions():
    out = {}
    for m in ("ir_datasets", "numpy", "scipy", "sklearn"):
        try:
            out[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            out[m] = None
    return out


def irds_home():
    """ir_datasets resolves its own home (a config file can move it, and on ad002 it points at
    /data/... not $HOME). Guessing $HOME/.ir_datasets would measure an empty directory and report
    a download volume of 0."""
    import ir_datasets.util
    return Path(ir_datasets.util.home_path())


def dir_bytes(p):
    p = Path(p)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def doc_string(title, text):
    return f"{title or ''}\n{text or ''}"


# ---------------------------------------------------------------- stage: select

def stage_select(args, out_dir):
    import ir_datasets
    cache = irds_home()
    before = dir_bytes(cache)
    t0 = time.time()
    ds = ir_datasets.load(IRDS_ID)

    qrels_pos = {}
    n_qrel_rows = 0
    for qr in ds.qrels_iter():
        n_qrel_rows += 1
        if qr.relevance >= 1:
            qrels_pos.setdefault(qr.query_id, []).append(qr.doc_id)
    t_qrels = time.time() - t0

    t1 = time.time()
    queries = {}
    for q in ds.queries_iter():
        if q.query_id in qrels_pos:
            queries[q.query_id] = q.text
    t_queries = time.time() - t1

    eligible = sorted(queries)
    rng = random.Random(args.seed)
    shuffled = list(eligible)
    rng.shuffle(shuffled)
    n_used = min(args.n_queries, len(shuffled))
    picked = sorted(shuffled[:n_used])
    after = dir_bytes(cache)

    sel = {
        "generated_by": "cpmr_experiment/instructions_v2/scripts/build_msmarco_train_candidates.py",
        "stage": "select",
        "ir_datasets_id": IRDS_ID,
        "library_versions": lib_versions(),
        "selection_rule": SELECTION_RULE,
        "seed": args.seed,
        "n_qrel_rows": n_qrel_rows,
        "n_queries_with_positive": len(eligible),
        "n_queries_requested": args.n_queries,
        "n_queries_selected": n_used,
        "used_all_available": n_used == len(eligible),
        "query_ids": picked,
        "positives_per_selected_query": {q: sorted(set(qrels_pos[q])) for q in picked},
        "n_positives_distribution": {
            "min": min(len(set(qrels_pos[q])) for q in picked),
            "max": max(len(set(qrels_pos[q])) for q in picked),
        },
        "timing_s": {"qrels_iter": round(t_qrels, 1), "queries_iter": round(t_queries, 1)},
        "ir_datasets_cache_bytes_before": before,
        "ir_datasets_cache_bytes_after": after,
        # NOT the network transfer: ir_datasets also extracts archives into this directory, so the
        # growth exceeds what was downloaded. The first run measured 64.6 MB of growth against
        # 29.5 MB of declared download size (qrels.train.tsv + queries.tar.gz).
        "ir_datasets_cache_growth_bytes": after - before,
        "peak_rss_gb": peak_rss_gb(),
        "note": ("query ids and the selection rule are written BEFORE any search, so the sample "
                 "cannot be influenced by retrieval or model output."),
    }
    p = out_dir / "query_selection.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=2)
    print(f"[select] eligible {len(eligible)} -> selected {n_used} (seed {args.seed})")
    print(f"[select] qrels rows {n_qrel_rows}, cache growth "
          f"{(after - before) / 1e6:.1f} MB (transfer + extraction), "
          f"peak RSS {sel['peak_rss_gb']} GB")
    print(f"[select] wrote {p} (sha256 {sha256_file(p)[:16]})")


# ---------------------------------------------------------------- stage: search

def build_index(doc_texts):
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    t0 = time.time()
    vec = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True, dtype=np.float32)
    tf = vec.fit_transform(doc_texts).tocsc()
    n_docs, n_vocab = tf.shape
    doc_len = np.asarray(tf.sum(axis=1)).ravel()
    avgdl = float(doc_len.mean())
    df = np.diff(tf.indptr)
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
    idf = np.clip(idf, 0.0, None)
    doc_texts.clear() if isinstance(doc_texts, list) else None
    return {"tf": tf, "vectorizer": vec, "doc_len": doc_len, "avgdl": avgdl, "idf": idf,
            "n_docs": n_docs, "n_vocab": int(n_vocab), "nnz": int(tf.nnz),
            "build_time_s": round(time.time() - t0, 1)}


def search(index, query_text, doc_ids, top_k):
    """Returns (ranked doc ids, scores, diagnostics). Ties broken by ascending doc id."""
    import numpy as np
    analyzer = index["vectorizer"].build_analyzer()
    vocab = index["vectorizer"].vocabulary_
    toks = analyzer(query_text)
    # Query term frequency is intentionally ignored (set), matching the v1 implementation.
    term_ids = sorted({vocab[t] for t in toks if t in vocab})
    diag = {"n_query_tokens": len(toks), "n_query_tokens_in_vocab": len(term_ids),
            "no_query_terms": not term_ids}
    if not term_ids:
        return [], [], diag

    tf, doc_len, avgdl, idf = index["tf"], index["doc_len"], index["avgdl"], index["idf"]
    scores = np.zeros(index["n_docs"], dtype=np.float32)
    denom = 1.0 - B + B * (doc_len / avgdl)
    for t in term_ids:
        s, e = tf.indptr[t], tf.indptr[t + 1]
        di = tf.indices[s:e]
        tv = tf.data[s:e]
        scores[di] += idf[t] * (tv * (K1 + 1)) / (tv + K1 * denom[di])

    nz = np.nonzero(scores)[0]
    diag["n_docs_with_positive_score"] = int(len(nz))
    diag["fewer_than_top_k"] = int(len(nz)) < top_k
    if len(nz) == 0:
        return [], [], diag

    # Deterministic order: score descending, then doc id ascending.
    # The doc-id comparison is confined to the rows that can reach the cutoff. A query holding a
    # common word puts millions of rows in nz, and sorting that many Python strings would dominate
    # the run time, so: everything strictly above the k-th largest score is ordered first, and the
    # tie group sitting exactly at that score is ordered by doc id to fill the remaining slots.
    sc = scores[nz]
    k = min(top_k, len(nz))
    kth = float(np.partition(sc, len(sc) - k)[len(sc) - k])
    hi = nz[sc > kth]
    tie = nz[sc == kth]
    diag["n_at_cutoff_score"] = int(len(tie))
    diag["cutoff_tie_decided_by_doc_id"] = bool(len(tie) > (k - len(hi)))
    sel = sorted(hi.tolist(), key=lambda i: (-float(scores[i]), doc_ids[i]))
    sel += sorted(tie.tolist(), key=lambda i: doc_ids[i])[: k - len(sel)]
    return [doc_ids[i] for i in sel], [float(scores[i]) for i in sel], diag


def stage_search(args, out_dir):
    import ir_datasets
    sel_path = out_dir / "query_selection.json"
    if not sel_path.exists():
        raise SystemExit(f"{sel_path} not found -- run --stage select first")
    sel = json.load(open(sel_path, encoding="utf-8"))
    qids = sel["query_ids"]
    positives = {q: set(v) for q, v in sel["positives_per_selected_query"].items()}

    cache = irds_home()
    before = dir_bytes(cache)
    ds = ir_datasets.load(IRDS_ID)

    t0 = time.time()
    doc_ids, doc_texts = [], []
    for d in ds.docs_iter():
        doc_ids.append(d.doc_id)
        doc_texts.append(doc_string(getattr(d, "title", "") or "", d.text))
    corpus_load_s = round(time.time() - t0, 1)
    print(f"[search] corpus loaded: {len(doc_ids)} docs in {corpus_load_s}s "
          f"(this time was NOT included in the v1 index timer)")

    dup_doc_ids = len(doc_ids) - len(set(doc_ids))
    index = build_index(doc_texts)
    print(f"[search] index: {index['n_docs']} docs, vocab {index['n_vocab']}, "
          f"avgdl {index['avgdl']:.1f}, nnz {index['nnz']}, build {index['build_time_s']}s, "
          f"peak RSS {peak_rss_gb()} GB")

    qtext = {q.query_id: q.text for q in ds.queries_iter() if q.query_id in set(qids)}
    missing_queries = [q for q in qids if q not in qtext]

    out_path = out_dir / f"candidates_top{args.top_k}.jsonl"
    if out_path.exists() and not args.allow_overwrite:
        raise SystemExit(f"{out_path} exists; pass --allow_overwrite to replace it deliberately")

    anomalies = {"no_query_terms": [], "fewer_than_top_k": [], "zero_candidates": [],
                 "missing_query_text": missing_queries, "duplicate_doc_ids_in_corpus": dup_doc_ids,
                 "positive_not_in_corpus": [], "positive_outside_top_k": []}
    corpus_ids = set(doc_ids)
    t1 = time.time()
    n_rows = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, qid in enumerate(qids):
            if qid not in qtext:
                continue
            ids, scores, diag = search(index, qtext[qid], doc_ids, args.top_k)
            if diag["no_query_terms"]:
                anomalies["no_query_terms"].append(qid)
            if not ids:
                anomalies["zero_candidates"].append(qid)
            if diag.get("fewer_than_top_k"):
                anomalies["fewer_than_top_k"].append({"query_id": qid, "n": len(ids)})
            pos = positives.get(qid, set())
            for p in pos:
                if p not in corpus_ids:
                    anomalies["positive_not_in_corpus"].append({"query_id": qid, "doc_id": p})
            got = set(ids)
            outside = sorted(p for p in pos if p in corpus_ids and p not in got)
            if outside:
                anomalies["positive_outside_top_k"].append({"query_id": qid, "doc_ids": outside})
            # positives are NOT inserted; ranks are the search's own 1-based ranks
            f.write(json.dumps({
                "query_id": qid, "query": qtext[qid],
                "candidates": [{"doc_id": d, "rank": r, "score": s, "is_known_positive": d in pos}
                               for r, (d, s) in enumerate(zip(ids, scores), start=1)],
            }) + "\n")
            n_rows += 1
            if (i + 1) % 20 == 0:
                print(f"[search] {i + 1}/{len(qids)} queries, {time.time() - t1:.0f}s")
    search_s = round(time.time() - t1, 1)
    after = dir_bytes(cache)

    report = {
        "generated_by": "cpmr_experiment/instructions_v2/scripts/build_msmarco_train_candidates.py",
        "stage": "search",
        "ir_datasets_id": IRDS_ID, "library_versions": lib_versions(),
        "query_selection_sha256": sha256_file(sel_path),
        "bm25": {"k1": K1, "b": B, "top_k": args.top_k,
                 "idf_formula": "log((N - df + 0.5)/(df + 0.5) + 1), clipped at 0",
                 "tokenizer": f"sklearn CountVectorizer token_pattern={TOKEN_PATTERN} lowercase=True",
                 "stemming": False, "stopwords": False,
                 "doc_string": "{title}\\n{text}, title='' when absent",
                 "query_term_frequency": "IGNORED (query terms de-duplicated into a set)",
                 "returned_docs": "only score > 0",
                 "tie_break": ("score descending, then doc_id ascending as a Python string "
                               "compare; the string compare is applied only to rows above the "
                               "k-th largest score and to the tie group at that score"),
                 "not_equivalent_to": "Lucene / ElasticSearch BM25"},
        "corpus": {"n_docs": index["n_docs"], "n_vocab": index["n_vocab"],
                   "avgdl": round(index["avgdl"], 2), "nnz": index["nnz"],
                   "duplicate_doc_ids": dup_doc_ids},
        "timing_s": {"corpus_load": corpus_load_s, "index_build": index["build_time_s"],
                     "search_total": search_s,
                     "search_per_query": round(search_s / max(n_rows, 1), 4)},
        "peak_rss_gb": peak_rss_gb(),
        "n_queries_searched": n_rows,
        "output": {"path": str(out_path.relative_to(ROOT)).replace("\\", "/"),
                   "bytes": out_path.stat().st_size, "sha256": sha256_file(out_path)},
        "ir_datasets_cache_bytes_before": before, "ir_datasets_cache_bytes_after": after,
        "ir_datasets_cache_growth_bytes": after - before,  # transfer + extraction, see stage select
        "anomalies": {k: (v if not isinstance(v, list) or len(v) <= 50 else v[:50])
                      for k, v in anomalies.items()},
        "anomaly_counts": {k: (len(v) if isinstance(v, list) else v) for k, v in anomalies.items()},
        "note": ("positives are never inserted into the result. positive_outside_top_k records "
                 "queries whose known positive the search did not return -- such a positive may "
                 "still be used as the training positive, which is different from forcing it "
                 "into the candidate pool."),
    }
    rp = out_dir / f"candidates_top{args.top_k}_report.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[search] {n_rows} queries in {search_s}s ({report['timing_s']['search_per_query']}s/query)")
    print(f"[search] anomalies: {report['anomaly_counts']}")
    print(f"[search] wrote {out_path} and {rp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["select", "search"])
    ap.add_argument("--out_dir", default="data/msmarco_train_v2")
    ap.add_argument("--n_queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=1000)
    ap.add_argument("--allow_overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (stage_select if args.stage == "select" else stage_search)(args, out_dir)


if __name__ == "__main__":
    main()
