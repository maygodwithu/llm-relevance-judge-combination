"""MS MARCO negative-sampling rank audit, pre-fixed spot-check sample, TREC-DL judged-coverage audit. CPU only, read-only, existing files only (no retrieval, no model call, no new ranking).
A  training (5,000 query / 10,000 pair), diag200 and val200 negatives: stored original BM25 rank of every sampled negative (negative_original_rank of the pair builder), bins, positive outside the search top-K, relevance class, collisions with the known positives
B  spot-check sample of training negatives: up to 10 at original rank 1-10 and up to 10 at rank 11-100, default_rng(397), no query twice (rows sorted by query id, permuted by the PRNG, first eligible rows taken); sample frozen before any text is read
C  TREC-DL 2019 / 2020: official judged-subset queries, official candidate set (scoreddocs), qrels, the stored BM25 top-100 pool file, the evaluation rows (judged pairs inside the pool), per-query coverage, excluded queries
    python audit397.py --root . --out <dir>"""
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--out", required=True); a = ap.parse_args()
ROOT = Path(a.root).resolve(); OUT = ROOT / a.out; OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
res = dict(sources={})
D5 = ROOT / "data/msmarco_train_v2_5k"; PO = ROOT / "data/msmarco_po"
pairs_all = jl(D5 / "pairs_basic_hard_seed0.jsonl")
res["sources"]["pairs_basic_hard_seed0"] = dict(path="data/msmarco_train_v2_5k/pairs_basic_hard_seed0.jsonl", sha256=sha(D5 / "pairs_basic_hard_seed0.jsonl"), rows=len(pairs_all))
sel5200 = json.load(open(D5 / "query_selection.json")); tsel = json.load(open(D5 / "train_set_5000_selection.json"))
pos5200 = sel5200["positives_per_selected_query"]
train_ids = list(tsel["kept_query_ids"]); train_set = set(train_ids); diag_set = set(sel5200["query_ids"]) - train_set
basic = {r["query_id"]: r for r in pairs_all if r["condition"] == "basic"}
assert len(basic) == 5200 and len(train_set) == 5000 and len(diag_set) == 200
# the files actually used by the experiments
tr_text = jl(D5 / "pairs_text_basic_5000.jsonl"); dg_text = jl(PO / "diag200/pairs_text_basic_diag200.jsonl"); va_text = jl(PO / "val200/pairs_text_basic_seed0.jsonl")
res["sources"].update(train_pairs_text=dict(path="data/msmarco_train_v2_5k/pairs_text_basic_5000.jsonl", sha256=sha(D5 / "pairs_text_basic_5000.jsonl"), rows=len(tr_text)),
                      diag_pairs_text=dict(path="data/msmarco_po/diag200/pairs_text_basic_diag200.jsonl", sha256=sha(PO / "diag200/pairs_text_basic_diag200.jsonl"), rows=len(dg_text)),
                      val_pairs_text=dict(path="data/msmarco_po/val200/pairs_text_basic_seed0.jsonl", sha256=sha(PO / "val200/pairs_text_basic_seed0.jsonl"), rows=len(va_text)),
                      val_pair_builder=dict(path="data/msmarco_po/val200/pairs_basic_hard_seed0.jsonl", sha256=sha(PO / "val200/pairs_basic_hard_seed0.jsonl")))
val_pairs = [r for r in jl(PO / "val200/pairs_basic_hard_seed0.jsonl") if r["condition"] == "basic"]
vsel = json.load(open(PO / "val200/query_selection.json"))
posv = vsel.get("positives_per_selected_query", {})
fields = sorted(tr_text[0]); res["sources"]["text_row_fields"] = fields
qk = "query_id" if "query_id" in tr_text[0] else None


def q_of(r):
    return str(r.get("query_id", r.get("query_group_id")))


tr_q = {q_of(r) for r in tr_text}; dg_q = {q_of(r) for r in dg_text}; va_q = {q_of(r) for r in va_text}
res["set_membership"] = dict(train_text_queries=len(tr_q), train_equals_kept5000=tr_q == train_set, diag_text_queries=len(dg_q), diag_equals_remaining200=dg_q == diag_set, val_text_queries=len(va_q), val_equals_selection=va_q == set(vsel["query_ids"]))
BINS = [(1, 10), (11, 100), (101, 1000)]


def rank_stats(rows, pos_map, name):
    rk = np.array([r["negative_original_rank"] for r in rows if r.get("negative_original_rank") is not None])
    missing = sum(1 for r in rows if r.get("negative_original_rank") is None)
    d = dict(set=name, n_rows=len(rows), n_with_rank=int(len(rk)), n_missing_rank=int(missing), min=int(rk.min()), q1=float(np.percentile(rk, 25)), median=float(np.median(rk)), q3=float(np.percentile(rk, 75)), max=int(rk.max()))
    for lo, hi in BINS:
        n = int(((rk >= lo) & (rk <= hi)).sum()); d[f"rank_{lo}_{hi}"] = dict(n=n, share=n / len(rk))
    d["positive_outside_search_topk"] = dict(n=int(sum(1 for r in rows if not r.get("positive_in_search_topk", True))), share=float(np.mean([not r.get("positive_in_search_topk", True) for r in rows])))
    d["relevance_class"] = {k: int(sum(1 for r in rows if r.get("negative_relevance_class") == k)) for k in sorted({r.get("negative_relevance_class") for r in rows}, key=str)}
    coll = [r["query_id"] for r in rows if pos_map and str(r["negative_doc_id"]) in set(map(str, pos_map.get(str(r["query_id"]), [])))]
    d["negative_equals_known_positive"] = dict(n=len(coll), checked=bool(pos_map))
    d["n_known_positives_per_query"] = {str(k): int(v) for k, v in zip(*np.unique([r.get("n_known_positives", 1) for r in rows], return_counts=True))}
    return d


train_rows = [basic[q] for q in train_ids]; diag_rows = [basic[q] for q in sorted(diag_set)]
res["negative_rank"] = [rank_stats(train_rows, pos5200, "train (5,000 queries)"), rank_stats(diag_rows, pos5200, "diag200"), rank_stats(val_pairs, {q: v for q, v in posv.items()} if posv else None, "val200")]
# every set also via the SHA-pinned text files: negative doc id of the text row = negative doc id of the pair builder row
def check_text(rows_text, builder):
    bad = 0
    for r in rows_text:
        if r.get("label") == "Irrelevant":
            b = builder.get(q_of(r))
            if b is None or str(b["negative_doc_id"]) != str(r.get("product_id", r.get("doc_id"))):
                bad += 1
    return bad
res["text_vs_builder_negative_id_mismatch"] = dict(train=check_text(tr_text, basic), diag=check_text(dg_text, basic), val=check_text(va_text, {r["query_id"]: r for r in val_pairs}))
res["sampling_rules_from_code"] = json.load(open(D5 / "pairs_basic_hard_seed0_report.json"))["rules"]
res["retrieval_run_report"] = {k: json.load(open(D5 / "candidates_top1000_report.json")).get(k) for k in ("ir_datasets_id", "bm25", "n_queries_searched", "output", "anomaly_counts")}
res["val_retrieval_report"] = {k: json.load(open(PO / "val200/candidates_top1000_report.json")).get(k) for k in ("ir_datasets_id", "bm25", "n_queries_searched", "output", "anomaly_counts")}
# ---------------------------------------------------------------- B spot check (pre-fixed rule)
text_by = {}
for r in tr_text:
    text_by.setdefault(q_of(r), {})[r["label"]] = r
rng = np.random.default_rng(397)
pick, used = {"rank_1_10": [], "rank_11_100": []}, set()
for name, lo, hi in (("rank_1_10", 1, 10), ("rank_11_100", 11, 100)):
    elig = sorted([q for q in train_ids if lo <= basic[q]["negative_original_rank"] <= hi])
    perm = [elig[i] for i in rng.permutation(len(elig))]
    for q in perm:
        if q in used:
            continue
        pick[name].append(q); used.add(q)
        if len(pick[name]) == 10:
            break
spot = []
for name in pick:
    for q in pick[name]:
        b = basic[q]; tx = text_by[q]
        spot.append(dict(stratum=name, query_id=q, query=b["query"], negative_doc_id=b["negative_doc_id"], negative_original_rank=b["negative_original_rank"], negative_bm25_score=round(b["negative_bm25_score"], 3), qrels_status="not in the query's known-positive list (unlabelled)" if str(b["negative_doc_id"]) not in set(map(str, pos5200.get(q, []))) else "KNOWN POSITIVE",
                         n_known_positives=b["n_known_positives"], positive_doc_id=b["positive_doc_id"], positive_text=tx["Relevant"]["product_text"][:700], negative_text=tx["Irrelevant"]["product_text"][:700]))
with open(OUT / "spot_check_sample.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(spot[0]), lineterminator="\n"); w.writeheader(); w.writerows(spot)
res["spot_check_sample"] = dict(rule="default_rng(397); strata rank 1-10 then rank 11-100; eligible queries of the 5,000 training queries sorted by id, permuted by the PRNG, first 10 per stratum, a query is taken at most once over both strata", eligible={"rank_1_10": int(sum(1 for q in train_ids if basic[q]["negative_original_rank"] <= 10)), "rank_11_100": int(sum(1 for q in train_ids if 11 <= basic[q]["negative_original_rank"] <= 100))}, n={k: len(v) for k, v in pick.items()})
# ---------------------------------------------------------------- C TREC-DL coverage
import ir_datasets
sys.path.insert(0, str(ROOT))
from src.eval_rerank import YEAR_POOL_PATH, YEAR_DATASET_ID  # noqa: E402
man = json.load(open(PO / "split_manifest.json"))
cov, perq_rows = [], []
for y in ("2019", "2020"):
    ds = ir_datasets.load(YEAR_DATASET_ID[y])
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    qrels = {}
    for qr in ds.qrels_iter():
        qrels.setdefault(qr.query_id, {})[qr.doc_id] = qr.relevance
    sc = {}
    for s in ds.scoreddocs_iter():
        sc.setdefault(s.query_id, []).append(s.doc_id)
    pool = jl(YEAR_POOL_PATH[y]); pool_sha = sha(YEAR_POOL_PATH[y])
    cand = {}; rel = {}
    for r in pool:
        cand.setdefault(r["query_id"], []).append(r["doc_id"])
        if r["relevance"] is not None:
            rel.setdefault(r["query_id"], {})[r["doc_id"]] = r["relevance"]
    # consistency of the pool file with the official qrels and candidate set
    pool_rel_mismatch = sum(1 for qq, d in rel.items() for p, g in d.items() if qrels.get(qq, {}).get(p) != g)
    pool_not_in_scored = sum(1 for qq, ids in cand.items() for p in ids if p not in set(sc.get(qq, [])))
    qall = sorted(queries, key=int)
    judged_in_pool = {qq: [p for p in cand.get(qq, []) if p in rel.get(qq, {})] for qq in qall}
    eval_q = [qq for qq in qall if judged_in_pool[qq]]
    excluded = []
    for qq in qall:
        if qq not in eval_q:
            reason = "no candidate in the stored pool" if not cand.get(qq) else ("no judged pair among the stored top-100 candidates" if not rel.get(qq) else "other")
            excluded.append(dict(query_id=qq, query=queries[qq], n_candidates=len(cand.get(qq, [])), n_qrels_total=len(qrels.get(qq, {})), reason=reason))
    grades = [rel[qq][p] for qq in eval_q for p in judged_in_pool[qq]]
    g = {int(k): int(v) for k, v in zip(*np.unique(grades, return_counts=True))}
    n_rel = int(sum(1 for x in grades if x >= 2)); n_ev = len(grades)
    cvq = np.array([len(judged_in_pool[qq]) / max(len(cand[qq]), 1) for qq in eval_q])
    single_class = [qq for qq in eval_q if len({rel[qq][p] >= 2 for p in judged_in_pool[qq]}) == 1]
    all_q_qrel = sum(len(v) for qq, v in qrels.items())
    qrels_in_pool = sum(len(judged_in_pool[qq]) for qq in qall)
    in_sc = sum(1 for qq in qall for p in qrels.get(qq, {}) if p in set(sc.get(qq, [])))
    cov.append(dict(year=y, dataset_id=YEAR_DATASET_ID[y], pool_file=str(YEAR_POOL_PATH[y].relative_to(ROOT)), pool_sha256=pool_sha, official_judged_queries=len(queries), queries_with_official_candidates=sum(1 for qq in qall if sc.get(qq)), candidates_per_query_official=dict(min=min(len(v) for v in sc.values()), median=float(np.median([len(v) for v in sc.values()])), max=max(len(v) for v in sc.values())),
                    queries_in_pool_file=len(cand), pool_candidates_per_query=sorted({len(v) for v in cand.values()}), queries_with_at_least_one_judged_pair=len(eval_q), evaluation_queries=len(eval_q),
                    total_pool_pairs=int(sum(len(v) for v in cand.values())), judged_pairs_in_pool=int(qrels_in_pool), judged_share_of_pool=qrels_in_pool / sum(len(v) for v in cand.values()), unjudged_pairs_excluded=int(sum(len(v) for v in cand.values()) - qrels_in_pool), evaluation_pairs=n_ev,
                    grade_counts_evaluation_pairs=g, binary_relevant=n_rel, binary_irrelevant=n_ev - n_rel, relevant_share=n_rel / n_ev, label_rule="grades 2 and 3 -> Relevant, 0 and 1 -> Irrelevant (src evaluation: grades >= 2)",
                    per_query_judged_share=dict(median=float(np.median(cvq)), q1=float(np.percentile(cvq, 25)), q3=float(np.percentile(cvq, 75)), min=float(cvq.min()), max=float(cvq.max())), single_class_queries=len(single_class), single_class_ids=single_class,
                    official_qrels_total=int(all_q_qrel), official_qrels_in_official_candidate_set=int(in_sc), official_qrels_in_stored_top100_pool=int(qrels_in_pool), pool_relevance_vs_official_qrels_mismatches=int(pool_rel_mismatch), pool_docs_not_in_official_candidates=int(pool_not_in_scored), excluded_queries=excluded))
    for qq in qall:
        perq_rows.append(dict(year=y, query_id=qq, n_official_candidates=len(sc.get(qq, [])), n_pool_candidates=len(cand.get(qq, [])), n_judged_in_pool=len(judged_in_pool[qq]), judged_share=len(judged_in_pool[qq]) / max(len(cand.get(qq, [])), 1), n_relevant=sum(1 for p in judged_in_pool[qq] if rel[qq][p] >= 2), n_qrels_total=len(qrels.get(qq, {})), evaluated=qq in eval_q))
res["trec_coverage"] = cov
res["split_manifest_trec_count"] = man["sizes"]["trec"]
with open(OUT / "trec_per_query_coverage.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(perq_rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(perq_rows)
with open(OUT / "negative_rank_per_row.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["set", "query_id", "negative_doc_id", "negative_original_rank", "negative_bm25_score", "negative_relevance_class", "positive_in_search_topk"])
    for nm, rows in (("train", train_rows), ("diag200", diag_rows), ("val200", val_pairs)):
        for r in rows:
            w.writerow([nm, r["query_id"], r["negative_doc_id"], r.get("negative_original_rank"), r.get("negative_bm25_score"), r.get("negative_relevance_class"), r.get("positive_in_search_topk")])
res["cpu_seconds"] = time.time() - t0
(OUT / "audit397_results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("set_membership", "negative_rank", "text_vs_builder_negative_id_mismatch", "spot_check_sample")}, indent=1, default=str)[:6000])
for c in cov:
    print(json.dumps({k: v for k, v in c.items() if k not in ("excluded_queries",)}, default=str)[:2600]); print("excluded:", c["excluded_queries"])
print("cpu", round(res["cpu_seconds"]), "s")
