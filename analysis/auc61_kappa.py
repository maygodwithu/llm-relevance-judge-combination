"""Cohen's kappa of the stored scores for the 6.1 AUC / kappa link table (experiment log 229): default kappa (decision m > 0) and selected kappa (ESCI: the per-model threshold re-derived from the stored validation-selection scores with the rule of 169 / 175 / 193-223; MS MARCO: CROSS threshold = hard-kappa maximum over ALL judged pairs of the other year, same rule) for every Base / CE / soft / AnyLoss model of the 9 evaluation sets. No threshold is re-optimised for the test data; the rule, the candidate set and the tie handling are those of the original reports (esci204_common.best_threshold == msm_t7_analysis.best_threshold). Used to verify the reported four-digit kappa values and to compute default / selected delta-kappa of the same scope next to delta-AUC.
    python auc61_kappa.py --root . --mods analysis --out cpmr_experiment/auc61/out"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auc61_data as D

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--mods", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
D.init(a.root, a.mods)
QC = D._st["QC"]
kap, best = QC.kap, QC.best_threshold
sel_gold = {}
for l in open(Path(a.root) / "cpmr_experiment/esci/audit/sel_rows.jsonl", encoding="utf-8"):
    r = json.loads(l); sel_gold[(r["query"], r["product_id"])] = r["label"] == "Relevant"


def k_at(m, y, t):
    p = m > t
    return float(kap(int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum())))


def esci_sel(sid, cond, T, s):
    """validation-selection scores (m, y) of the model, as in the reports"""
    if sid == "ESCI-Qwen":
        st = D._st["qs"]
    elif sid == "ESCI-Llama":
        st = D._st["ls"]
    else:
        st = D._st["gs"]
    mn, ins = D._escid(sid, cond, T, s)
    if cond == "BASE":
        if sid == "ESCI-Gemma":
            r = st.arrays(mn, ins, "sel")
            return r[1], r[2]
        ks, m = st.base_sel(T, False)
        return m, np.array([sel_gold[k] for k in ks])
    r = st.arrays(mn, ins, "sel")
    return r[1], r[2]


out = {}
for sid in D.SETS:
    for cond in ("BASE", "CE", "SOFT", "ANY"):
        for T in D.TS:
            for s in D.seeds_of(cond):
                r = D.get(sid, cond, T, s)
                if r is None:
                    continue
                _, m, y, _ = r
                k0 = k_at(m, y, 0.0)
                if sid.startswith("ESCI"):
                    sm, sy = esci_sel(sid, cond, T, s)
                    thr, vk = best(sm, sy)
                else:
                    year = sid.split("-")[2]
                    other = sid.rsplit("-", 1)[0] + "-" + ("2020" if year == "2019" else "2019")
                    ro = D.get(other, cond, T, s)
                    thr, vk = best(ro[1], ro[2])
                out[(sid, cond, T, s)] = (k0, k_at(m, y, thr), float(thr))
Path(a.out).mkdir(parents=True, exist_ok=True)
pickle.dump(out, open(Path(a.out) / "kappa.pkl", "wb"))
print("kappa cells", len(out))
