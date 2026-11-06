"""Shared loaders / metrics of the ESCI systematic comparison (experiment log 204; Qwen3-4B stages 1-2).
Score layout: <root>/cpmr_experiment/**/esciEval_<model>_step625__<instr>__<tag>.json.gz  (tag test300 | sel), Base: esciEval_Base__<instr>__test300.json.gz,
Base validation: <root>/cpmr_experiment/esci_t/p1_T<k>/p1_raw_cand<i>.json.gz (+ p1_result.json: chosen candidate).
Margin m = logP(label + terminator)(Relevant) - logP(label + terminator)(Irrelevant); Relevant if m > t; ESCI E = Relevant, S / C / I = Irrelevant.
Threshold rule (169 / 175 / 193-203): hard-kappa maximum over all validation pairs of the model's own scores; candidates 0, midpoints of consecutive distinct scores, below-all, above-all;
ties -> 0, then smaller |t|, then smaller t; one independent threshold per (path, instruction, seed); the Base threshold is never transferred."""
import glob
import gzip
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ld = lambda p: json.load(gzip.open(p, "rt", encoding="utf-8"))
SEEDS = [0, 1, 2]
TS = list(range(1, 8))
CLS = {0: "E", 1: "S", 2: "C", 3: "I"}


def kap(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def kap_vec(tp, fp, fn, tn):
    tp, fp, fn, tn = [np.asarray(x, dtype=np.float64) for x in (tp, fp, fn, tn)]
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pe == 1, np.nan, (po - pe) / (1 - pe))


def best_threshold(m, y):
    u = np.unique(m)
    cands = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0], [0.0]])
    best = None
    for t in np.unique(cands):
        p = m > t
        k = kap(int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum()))
        if not np.isfinite(k):
            continue
        key = (-round(float(k), 12), 0 if t == 0.0 else 1, abs(t), t)
        if best is None or key < best[0]:
            best = (key, float(t), float(k))
    return best[1], best[2]


def auc(m, y):
    rk = rankdata(m)
    npos, nneg = y.sum(), (~y).sum()
    return float((rk[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.idx = {}
        for f in glob.glob(str(self.root / "cpmr_experiment" / "**" / "*.json.gz"), recursive=True):
            b = os.path.basename(f)
            if b.startswith("esciEval_"):
                self.idx.setdefault(b, f)
        self.cache = {}

    def path(self, model, instr, tag):
        b = f"esciEval_{model}_step625__{instr}__{tag}.json.gz" if model != "Base" else f"esciEval_Base__{instr}__{tag}.json.gz"
        return self.idx.get(b)

    def raw(self, model, instr, tag):
        p = self.path(model, instr, tag)
        if p is None:
            return None
        if p not in self.cache:
            self.cache[p] = ld(p)["raw"]
        return self.cache[p]

    def arrays(self, model, instr, tag):
        raw = self.raw(model, instr, tag)
        if raw is None:
            return None
        ks = [(q, d) for q in sorted(raw) for d in sorted(raw[q])]
        m = np.array([raw[q][d][0] + raw[q][d][1] - raw[q][d][3] - raw[q][d][4] for q, d in ks])
        y = np.array([raw[q][d][10] >= 2 for q, d in ks])
        c = np.array([int(raw[q][d][11]) for q, d in ks])
        return ks, m, y, c

    def base_sel(self, T, apocf):
        d = self.root / "cpmr_experiment" / "esci_t" / f"p1_T{T}"
        idx = json.load(open(d / "p1_result.json"))["selection"]["index"] if apocf else 0
        p1 = ld(d / f"p1_raw_cand{idx}.json.gz")["margins"]
        ks = [(q, k) for q in sorted(p1) for k in sorted(p1[q])]
        return ks, np.array([p1[q][k][0] + p1[q][k][1] - p1[q][k][2] - p1[q][k][3] for q, k in ks])


# condition -> (model name template, instruction label template)
COND = {
    "Q1": ("E4_T{T}_s{s}", "T{T}"),
    "Q2": ("EQ_mix_T{T}_s{s}", "T{T}"),
    "Q3": ("EQ_mixany_T{T}_s{s}", "T{T}"),
    "P1": ("E4_T{T}_APOCF_s{s}", "T{T}_APOCF"),
    "P2": ("EQ_mix_T{T}_APOCF_s{s}", "T{T}_APOCF"),
}
CONDS_ALL = ["Q0", "Q1", "Q2", "Q3", "P0", "P1", "P2"]


def model_name(c, T, s):
    if c in ("Q0", "P0"):
        return "Base"
    return COND[c][0].format(T=T, s=s)


def instr_name(c, T):
    if c == "Q0":
        return f"T{T}"
    if c == "P0":
        return f"T{T}_APOCF"
    return COND[c][1].format(T=T)


def load_cond(st, c, T, s, test_keys=None, sel_gold=None):
    """-> dict(test=(keys, m, y, code), sel=(m, y)) or None when a file is missing."""
    mn, ins = model_name(c, T, s), instr_name(c, T)
    te = st.arrays(mn, ins, "test300")
    if te is None:
        return None
    if c in ("Q0", "P0"):
        if sel_gold is None:
            return {"test": te, "sel": None}
        ks, m = st.base_sel(T, c == "P0")
        y = np.array([sel_gold[k] for k in ks])
        return {"test": te, "sel": (m, y, ks)}
    se = st.arrays(mn, ins, "sel")
    return {"test": te, "sel": None if se is None else (se[1], se[2], se[0])}


def build_all(root):
    """Every available (path, instruction, seed) with both test300 and validation scores -> D[(c, T, s)] with margins, labels, ESCI codes, threshold, per-query confusion counts at t=0 and at the validation threshold."""
    st = Store(root)
    sel_gold = {}
    for l in open(Path(root) / "cpmr_experiment/esci/audit/sel_rows.jsonl", encoding="utf-8"):
        r = json.loads(l); sel_gold[(r["query"], r["product_id"])] = r["label"] == "Relevant"
    test_q = None
    D = {}
    for c in CONDS_ALL:
        for T in TS:
            for s in ([0] if c in ("Q0", "P0") else SEEDS):
                r = load_cond(st, c, T, s, sel_gold=sel_gold)
                if r is None or r["sel"] is None:
                    continue
                ks, m, y, code = r["test"]
                if test_q is None:
                    test_q = list(ks); qid = {q: i for i, q in enumerate(sorted({q for q, _ in ks}))}
                    row_q = np.array([qid[q] for q, _ in ks]); nq = len(qid)
                assert ks == test_q, (c, T, s)
                sm, sy, sks = r["sel"]
                thr, vk = best_threshold(sm, sy)
                e = dict(sel_ks=sks, m=m, y=y, code=code, thr=thr, val_kappa=vk, val_rel=float((sm > thr).mean()), val_pos=float(sy.mean()), val_auc=auc(sm, sy), auc=auc(m, y), sel_m=sm, sel_y=sy)
                for mode, t in (("t0", 0.0), ("cal", thr)):
                    p = m > t
                    cnt = np.zeros((nq, 4))
                    for j, arr in enumerate((p & y, p & ~y, ~p & y, ~p & ~y)):
                        cnt[:, j] = np.bincount(row_q, weights=arr.astype(float), minlength=nq)
                    tot = cnt.sum(0)
                    e[mode] = dict(cnt=cnt, pred=p, kappa=kap(*tot), acc=float((tot[0] + tot[3]) / tot.sum()), rel=float(p.mean()), fnr=float(tot[2] / (tot[0] + tot[2])),
                                   fpr={CLS[k]: float((p & (code == k)).sum() / (code == k).sum()) for k in (1, 2, 3)})
                D[(c, T, s)] = e
    return D, nq, row_q, test_q
