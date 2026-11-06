"""Library of the chapter 6.2 error / decision-transition analysis (experiment log 237): loading of the stored scores through the loaders of the original analyses (auc61_data), selected thresholds (stored/re-derived by the original rule), per-pair predictions, group definitions, per-query count matrices and the paired query-cluster bootstrap.
Groups: ESCI gold E (Relevant) / S / C / I (Irrelevant), S+C+I, all; MS MARCO Relevant (grades 2/3) / Irrelevant (grades 0/1), all. A decision is right when prediction == gold; E errors are false negatives (predicted Irrelevant), S/C/I errors are false positives (predicted Relevant).
Transition of a pair of conditions (before -> after) on the SAME examples: W->R (wrong before, right after = corrected), R->W (right before, wrong after = new error), WW, RR; denominators of every rate are the group sizes N_g (not the number of previous errors)."""
import pickle
import sys
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)
TS = tuple(range(1, 8))
B_DRAWS, RNG_SEED = 10000, 61


def init(root, mods, kappa_pkl):
    global D, QC, LM, K, SEL_GOLD, ROOT
    ROOT = Path(root)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import auc61_data as D_
    D = D_
    D.init(root, mods)
    QC = D._st["QC"]; LM = D._st["LM"]
    K = pickle.load(open(kappa_pkl, "rb"))
    import json
    SEL_GOLD = {}
    for l in open(ROOT / "cpmr_experiment/esci/audit/sel_rows.jsonl", encoding="utf-8"):
        r = json.loads(l); SEL_GOLD[(r["query"], r["product_id"])] = r["label"] == "Relevant"


def groups_of(sid, code):
    """-> dict name -> boolean mask over rows (code = ESCI class code 0 E 1 S 2 C 3 I, or MS grades)"""
    if sid.startswith("ESCI"):
        return {"E": code == 0, "S": code == 1, "C": code == 2, "I": code == 3, "SCI": code > 0, "ALL": np.ones(len(code), bool)}
    return {"Rel": code >= 2, "Irr": code < 2, "ALL": np.ones(len(code), bool)}


def load(sid, cond, T, s):
    """-> dict(m, y, code, ks, path) or None"""
    key = (sid, cond, T, s)
    if sid.startswith("ESCI"):
        d = D._escid(sid, cond, T, s)
        if d is None:
            return None
        st = {"ESCI-Qwen": D._st["qs"], "ESCI-Llama": D._st["ls"], "ESCI-Gemma": D._st["gs"]}[sid]
        r = st.arrays(d[0], d[1], "test300")
        return None if r is None else dict(ks=r[0], m=r[1], y=r[2], code=r[3], path=st.path(d[0], d[1], "test300"))
    _, mdl, yr = sid.split("-")
    mod = {"Qwen": D._st["MQ"], "Llama": D._st["ML"], "Gemma": D._st["MG"]}[mdl]
    f = {"Qwen": D._st["mq"], "Llama": D._st["ml"], "Gemma": D._st["mg"]}[mdl]
    pth = {"BASE": "M0", "CE": "M1", "SOFT": "M2", "ANY": "M3"}.get(cond)
    if pth is None or (pth == "M0" and s != 0):
        return None
    tg = mod.tags(pth, T, s)
    if tg is None:
        return None
    r = f.get(tg, T, yr)
    return None if r is None else dict(ks=r[0], m=r[1], y=r[2], code=r[3], path=None)


def _best(sm, sy):
    return QC.best_threshold(sm, sy)


def sel_threshold(sid, cond, T, s):
    """selected threshold of the original analyses: ESCI validation hard-kappa maximum (re-derived from the stored validation scores), MS MARCO CROSS. Non-PO conditions: from kappa.pkl (verified against the reported kappa)."""
    s0 = 0 if cond in ("BASE", "POBASE") else s
    if cond in ("BASE", "CE", "SOFT", "ANY"):
        return K[(sid, cond, T, s0)][2]
    # PO conditions (ESCI Qwen / Llama)
    if D.esci_is_alias(sid, cond, T):
        return sel_threshold(sid, {"POBASE": "BASE", "POCE": "CE", "POSOFT": "SOFT"}[cond], T, s)
    st = D._st["qs"] if sid == "ESCI-Qwen" else D._st["ls"]
    mn, ins = D._escid(sid, cond, T, s0)
    if cond == "POBASE":
        ks, m = st.base_sel(T, True)
        return _best(m, np.array([SEL_GOLD[k] for k in ks]))[0]
    r = st.arrays(mn, ins, "sel")
    return _best(r[1], r[2])[0]


def predictions(rec, thr, mode):
    return rec["m"] > (0.0 if mode == "default" else thr)


def counts_by_group(pred, rec, gm):
    y = rec["y"]
    out = {}
    for g, mk in gm.items():
        n = int(mk.sum())
        wrong = (pred != y) & mk
        out[g] = dict(N=n, errors=int(wrong.sum()), rate=float(wrong.sum() / n) if n else float("nan"))
    tp, fp, fn, tn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum()), int((~pred & ~y).sum())
    n = tp + fp + fn + tn
    po = (tp + tn) / n; pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return dict(groups=out, tp=tp, fp=fp, fn=fn, tn=tn, accuracy=float((tp + tn) / n), pred_rel=float((tp + fp) / n), kappa=float((po - pe) / (1 - pe)) if pe != 1 else float("nan"))


def trans_counts(pa, pb, y, gm):
    ra, rb = pa == y, pb == y
    out = {}
    for g, mk in gm.items():
        out[g] = dict(N=int(mk.sum()), WR=int(((~ra) & rb & mk).sum()), RW=int((ra & ~rb & mk).sum()), WW=int((~ra & ~rb & mk).sum()), RR=int((ra & rb & mk).sum()))
    return out


class Boot:
    """paired query-cluster bootstrap of one evaluation set: one resample matrix W (B x n_queries) for every precision / loss / instruction / seed"""
    def __init__(self, ks, y, code, gm):
        qs = sorted({q for q, _ in ks}); qid = {q: i for i, q in enumerate(qs)}
        self.nq = len(qs); self.rowq = np.array([qid[q] for q, _ in ks])
        rng = np.random.default_rng(RNG_SEED)
        self.W = np.stack([np.bincount(i, minlength=self.nq) for i in rng.integers(0, self.nq, size=(B_DRAWS, self.nq))]).astype(np.float64)
        self.gm = gm
        self.Ngq = {g: np.bincount(self.rowq, weights=mk.astype(float), minlength=self.nq) for g, mk in gm.items()}
        self.Ng = {g: self.W @ v for g, v in self.Ngq.items()}
        self.valid = {g: self.Ng[g] > 0 for g in gm}

    def qc(self, pa, pb, y):
        ra, rb = pa == y, pb == y
        out = {}
        for g, mk in self.gm.items():
            for nm, arr in (("WR", (~ra) & rb), ("RW", ra & ~rb), ("WW", (~ra) & (~rb)), ("RR", ra & rb)):
                out[(g, nm)] = np.bincount(self.rowq, weights=(arr & mk).astype(float), minlength=self.nq)
        return out

    def agg(self, qcs, Ts, seeds):
        """qcs[(T, s)] -> per-query transition counts; returns point (unit weights) and draw vectors of the rates (count / N_g) averaged over the instructions Ts then over seeds, for every (group, kind) with kind in WR, RW, net, changed, WW, RR"""
        pt, bt = {}, {}
        for g in self.gm:
            for nm in ("WR", "RW", "WW", "RR"):
                p_s, b_s = [], []
                for s in seeds:
                    p_s.append(np.mean([qcs[(T, s)][(g, nm)].sum() / self.Ngq[g].sum() for T in Ts]))
                    b_s.append(np.mean([(self.W @ qcs[(T, s)][(g, nm)]) / self.Ng[g] for T in Ts], axis=0))
                pt[(g, nm)] = float(np.mean(p_s)); bt[(g, nm)] = np.mean(b_s, axis=0)
            pt[(g, "net")] = pt[(g, "WR")] - pt[(g, "RW")]; bt[(g, "net")] = bt[(g, "WR")] - bt[(g, "RW")]
            pt[(g, "changed")] = pt[(g, "WR")] + pt[(g, "RW")]; bt[(g, "changed")] = bt[(g, "WR")] + bt[(g, "RW")]
        return pt, bt

    def ci(self, vec, g):
        v = vec[self.valid[g]]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], int(self.valid[g].sum())
