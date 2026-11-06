"""ROC-AUC of the stored continuous margins m (no threshold, no rounding, no sigmoid) and the paired query-cluster bootstrap of one evaluation set (experiment log 229, chapter 6.1).
AUC of one (condition, instruction, seed) = Mann-Whitney AUC over ALL evaluated pairs of the evaluation set (Relevant = 1 against Irrelevant = 0, ties count 0.5), pooled over queries (not the mean of per-query AUCs). Scores of different instructions / seeds are never pooled.
Bootstrap: B = 10,000 resamples of the evaluation queries WITH replacement, np.random.default_rng(SEED).integers(0, n_queries, size=(B, n_queries)) -> multiplicity matrix W (B x n_queries); every pair of a query carries the multiplicity of its query in the resample (all pairs of a drawn query stay together); the SAME W is used for every condition / instruction / seed of the set. The AUC of a resample is the weighted Mann-Whitney AUC with these multiplicities. Resamples with only one class (denominator 0; the same for all conditions because the labels are shared) are excluded for all conditions and counted.
Output: one .npz-like pickle per evaluation set with the full-sample AUC and the B-vector of resample AUCs of every (condition, T, seed), the file manifest (relative path, SHA256, rows) and the checks (rank-based AUC == weighted AUC with unit weights).
    python auc61_boot.py --root . --mods analysis --set ESCI-Qwen --out cpmr_experiment/auc61/out"""
import argparse
import gzip
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auc61_data as D

B = 10000
SEED = 61
CHUNK = 500


def auc_rank(m, y):
    rk = rankdata(m)
    a, b = int(y.sum()), int((~y).sum())
    return float((rk[y].sum() - a * (a + 1) / 2) / (a * b))


def prep(m):
    order = np.argsort(m, kind="mergesort")
    ms = m[order]
    chg = np.concatenate([[True], ms[1:] != ms[:-1]])
    gid = np.cumsum(chg) - 1
    starts = np.flatnonzero(chg)
    ends = np.concatenate([starts[1:], [len(ms)]])
    return order, starts[gid], ends[gid]


def weighted_auc_numerator(pp, y, rowq, Wc):
    """sum over positive pairs i of w_i * (weight of strictly lower negatives + 0.5 * weight of tied negatives); Wc: (b, nq) query multiplicities"""
    order, st, en = pp
    b = Wc.shape[0]
    wp = Wc[:, rowq[order]]
    ys = y[order]
    cn = np.zeros((b, len(ys) + 1))
    np.cumsum(np.where(ys, 0.0, wp), axis=1, out=cn[:, 1:])
    below, upto = cn[:, st], cn[:, en]
    return (np.where(ys, wp, 0.0) * (below + 0.5 * (upto - below))).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--mods", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    D.init(a.root, a.mods)
    sid = a.set
    vecs = {}
    manifest = []
    ref = None
    for cond in D.CONDS:
        for T in D.TS:
            for s in D.seeds_of(cond):
                r = D.get(sid, cond, T, s)
                if r is None:
                    continue
                ks, m, y, path = r
                if ref is None:
                    ref = (ks, y)
                    qs = sorted({q for q, _ in ks})
                    qid = {q: i for i, q in enumerate(qs)}
                    rowq = np.array([qid[q] for q, _ in ks])
                    nq = len(qs)
                assert ks == ref[0] and (y == ref[1]).all() and np.isfinite(m).all()
                vecs[(cond, T, s)] = m
                meta = json.load(gzip.open(path, "rt", encoding="utf-8")).get("meta", {})
                manifest.append(dict(cond=cond, T=T, seed=s, path=str(Path(path).relative_to(D.ROOT)), sha256=D.sha256(path), rows=len(ks), model=meta.get("model", meta.get("tag")), instruction=meta.get("instruction"), instruction_sha256=meta.get("instruction_sha256"),
                                     rows_file_sha256=meta.get("rows_file_sha256"), checkpoint_sha256=meta.get("checkpoint_sha256"), train_seed=meta.get("seed"), global_step=meta.get("global_step"), max_length=meta.get("max_length")))
    y = ref[1]
    n = len(y)
    rng = np.random.default_rng(SEED)
    W = np.stack([np.bincount(i, minlength=nq) for i in rng.integers(0, nq, size=(B, nq))]).astype(np.float64)
    wpos = W @ np.bincount(rowq, weights=y.astype(float), minlength=nq)
    wneg = W @ np.bincount(rowq, weights=(~y).astype(float), minlength=nq)
    valid = (wpos > 0) & (wneg > 0)
    den = wpos * wneg
    rows_sha = hashlib.sha256(json.dumps([[str(q), str(d), bool(g)] for (q, d), g in zip(ref[0], ref[1])]).encode()).hexdigest()
    out = dict(rows_sha256=rows_sha, set=sid, B=B, seed=SEED, n_queries=nq, n_rows=n, n_pos=int(y.sum()), n_invalid_draws=int((~valid).sum()), valid=valid, full={}, boot={}, manifest=manifest, ties_in_scores={})
    maxdiff = 0.0
    for key, m in vecs.items():
        pp = prep(m)
        full_rank = auc_rank(m, y)
        full_w = float(weighted_auc_numerator(pp, y, rowq, np.ones((1, nq)))[0] / (int(y.sum()) * int((~y).sum())))
        maxdiff = max(maxdiff, abs(full_rank - full_w))
        out["full"][key] = full_rank
        out["ties_in_scores"][key] = int(n - len(np.unique(m)))
        num = np.empty(B)
        for i in range(0, B, CHUNK):
            num[i:i + CHUNK] = weighted_auc_numerator(pp, y, rowq, W[i:i + CHUNK])
        with np.errstate(invalid="ignore", divide="ignore"):
            out["boot"][key] = np.where(valid, num / den, np.nan)
    out["max_abs_rank_vs_weighted_full_sample"] = maxdiff
    out["seconds"] = time.time() - t0
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(Path(a.out) / f"boot_{sid}.pkl", "wb"))
    print(f"{sid}: {len(vecs)} score vectors, {nq} queries, {n} rows, invalid draws {out['n_invalid_draws']}, max |rank AUC - weighted AUC (unit weights)| {maxdiff:.2e}, {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
