"""Library of the chapter 6.3 threshold-stability analysis (experiment log 241): validation-query bootstrap thresholds applied to the FIXED test / CROSS set, from the EXISTING stored scores (no new inference).
Loading goes through err62_lib / auc61_data (the loaders of the original analyses). Weighted threshold re-selection = the algorithm of esci204_stability.best_weighted (resampled rows with positive multiplicity, candidates: 0, midpoints of consecutive distinct scores of the resampled rows, below-all, above-all; hard kappa maximum of the weighted counts; ties -> 0, then smaller |t|, then smaller t; decision m > t)."""
import json
import sys
from pathlib import Path

import numpy as np

import err62_lib as L

R_DRAWS = 500
TOL = 1e-12
CONDS = ("BASE", "CE", "SOFT", "ANY", "POBASE", "POCE", "POSOFT")
ALL7 = tuple(range(1, 8))


def init(root, mods, kappa_pkl):
    global SEL_GOLD, QC
    L.init(root, mods, kappa_pkl)
    SEL_GOLD = L.SEL_GOLD
    QC = L.D._st["QC"]


def seeds_of(c):
    return (0,) if c in ("BASE", "POBASE") else L.SEEDS


def other_year_sid(sid):
    mdl, yr = sid.rsplit("-", 1)
    return mdl + "-" + ("2020" if yr == "2019" else "2019")


def val_arrays(sid, cond, T, s):
    """validation / CROSS selection scores of the checkpoint: -> (ks, m, y). ESCI: stored validation-selection scores (as in the reports); MS MARCO: the other year's judged pairs (CROSS)."""
    if sid.startswith("MS"):
        r = L.load(other_year_sid(sid), cond, T, s)
        return None if r is None else (r["ks"], r["m"], r["y"])
    st = {"ESCI-Qwen": L.D._st["qs"], "ESCI-Llama": L.D._st["ls"], "ESCI-Gemma": L.D._st["gs"]}[sid]
    mn, ins = L.D._escid(sid, cond, T, s)
    alias = L.D.esci_is_alias(sid, cond, T)
    if cond in ("BASE", "POBASE"):
        if sid == "ESCI-Gemma":
            r = st.arrays(mn, ins, "sel")
            return r[0], r[1], r[2]
        ks, m = st.base_sel(T, cond == "POBASE" and not alias)
        return ks, m, np.array([SEL_GOLD[k] for k in ks])
    r = st.arrays(mn, ins, "sel")
    return None if r is None else (r[0], r[1], r[2])


def kap_vec(tp, fp, fn, tn):
    return QC.kap_vec(tp, fp, fn, tn)


class Val:
    """one checkpoint's validation scores sorted by score, with the canonical query index of every row"""
    def __init__(self, ks, m, y, qid):
        self.order = np.argsort(m, kind="mergesort")
        self.ms = m[self.order]; self.ys = y[self.order]
        self.rq = np.array([qid[q] for q, _ in ks])[self.order]
        brk = np.concatenate([[True], self.ms[1:] != self.ms[:-1]])
        self.gid = np.cumsum(brk) - 1
        self.G = int(self.gid[-1]) + 1
        self.u = self.ms[brk]
        self.sd = float(m.std())
        self.m, self.y = m, y

    def best(self, w_q):
        """threshold re-selected on the resample with query multiplicities w_q (array over canonical queries); nan when no candidate has a defined kappa"""
        w = w_q[self.rq]
        pos = np.bincount(self.gid, weights=w * self.ys, minlength=self.G)
        neg = np.bincount(self.gid, weights=w * (~self.ys), minlength=self.G)
        keep = (pos + neg) > 0
        if not keep.any():
            return float("nan")
        u = self.u[keep]; pos = pos[keep]; neg = neg[keep]
        cp, cn = np.cumsum(pos), np.cumsum(neg)
        P_, N_ = cp[-1], cn[-1]
        cpz = np.concatenate([[0.0], cp]); cnz = np.concatenate([[0.0], cn])
        tvals = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]])
        k = kap_vec(P_ - cpz, N_ - cnz, cpz, cnz)
        g0 = int(np.searchsorted(u, 0.0, side="right"))
        t_all = np.concatenate([tvals, [0.0]]); k_all = np.concatenate([k, [k[g0]]])
        ok = np.isfinite(k_all)
        if not ok.any():
            return float("nan")
        t_all, k_all = t_all[ok], k_all[ok]
        key = np.lexsort((t_all, np.abs(t_all), (t_all != 0.0).astype(int), -np.round(k_all, 12)))
        return float(t_all[key[0]])


class Test:
    """fixed test / CROSS-target scores sorted per class for fast kappa at many thresholds"""
    def __init__(self, m, y):
        self.m, self.y = m, y
        self.ps, self.ns = np.sort(m[y]), np.sort(m[~y])
        self.ms = np.sort(m)
        self.P, self.N = len(self.ps), len(self.ns)

    def kappa(self, ts):
        ts = np.asarray(ts, float)
        tp = self.P - np.searchsorted(self.ps, ts, side="right")
        fp = self.N - np.searchsorted(self.ns, ts, side="right")
        return kap_vec(tp, fp, self.P - tp, self.N - fp)

    def changed(self, ts, t_ref):
        """share of test decisions that differ between thresholds ts and t_ref (m > t): rows with m between the two thresholds"""
        ts = np.asarray(ts, float)
        lo, hi = np.minimum(ts, t_ref), np.maximum(ts, t_ref)
        return (np.searchsorted(self.ms, hi, side="right") - np.searchsorted(self.ms, lo, side="right")) / len(self.ms)


def make_draws(nq, seed, R=R_DRAWS):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nq, size=(R, nq))
    return np.stack([np.bincount(i, minlength=nq) for i in idx]).astype(float)


def pct(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def summ(x, ref=None):
    """summary of per-draw values x (valid draws only)"""
    x = np.asarray(x, float)
    d = dict(n=int(len(x)), mean=float(x.mean()), sd=float(x.std(ddof=0)), p2_5=pct(x)[0], p97_5=pct(x)[1], pos=float((x > TOL).mean()), tie=float((np.abs(x) <= TOL).mean()), neg=float((x < -TOL).mean()))
    if ref is not None:
        d["full"] = float(ref)
    return d
