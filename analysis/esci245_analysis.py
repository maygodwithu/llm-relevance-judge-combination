"""Codex 245 analysis (CPU): Qwen3-4B, ESCI, T1 (trained and evaluated), seed 0, final update 625. Eight conditions, own-instruction scores on the frozen validation (sel: 200 queries / 4,047 pairs) and test300 (300 queries / 6,078 pairs).
  CE, SOFT1 (CE + soft kappa lam 1), ANY10 (CE + AnyLoss-like kappa lam 1 alpha 10)  -- reused (165 / 184 / 185)
  SOFT03, SOFT3 (soft kappa lam .3 / 3), ANY5, ANY20 (alpha 5 / 20, lam 1), BCE (CE + 1.0 label-margin BCE)  -- new (245)
Margin m = [lp_label + lp_eot](Relevant) - same(Irrelevant) (raw fields 0 + 1 - 3 - 4), decision m > t, gold = field 10 == 3 (E).
Threshold t*: maximises the hard kappa over the validation partitions (candidates: below / above all scores, midpoints of consecutive distinct scores, and 0), ties -> t = 0, then smaller |t|, then smaller t
(the algorithm of esci204_stability.best_weighted with unit weights). Test: both t = 0 and the validation-selected t* are fixed before the test scores are used.
Paired comparisons: query-cluster bootstrap over the 300 test queries, 2,000 resamples (rng seed 0), one resample matrix for every comparison; thresholds are NOT re-selected inside the bootstrap
(fixed trained models and thresholds). Exploratory single-seed pilot: the intervals contain test-query sampling only.
    python esci245_analysis.py --dir <dir with the 16 esciEval_*.json.gz files> --out <dir>"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

CONDS = [("CE", "E4_T1_s0", "CE (reused)"), ("SOFT1", "EQ_mix_T1_s0", "CE+soft kappa lam 1 (reused)"), ("ANY10", "EQ_mixany_T1_s0", "CE+Any kappa lam 1 alpha 10 (reused)"),
         ("SOFT03", "Q245S03_mix_T1_s0", "CE+soft kappa lam .3"), ("SOFT3", "Q245S3_mix_T1_s0", "CE+soft kappa lam 3"), ("ANY5", "Q245A5_mixany_T1_s0", "CE+Any kappa lam 1 alpha 5"),
         ("ANY20", "Q245A20_mixany_T1_s0", "CE+Any kappa lam 1 alpha 20"), ("BCE", "Q245B_mixbce_T1_s0", "CE+label-margin BCE lam 1")]
PAIRS = [("SOFT03", "SOFT1"), ("SOFT3", "SOFT1"), ("ANY5", "ANY10"), ("ANY20", "ANY10"), ("BCE", "CE"), ("BCE", "SOFT1"), ("SOFT1", "CE"), ("ANY10", "CE"), ("BCE", "ANY10")]


def load(path):
    z = json.load(gzip.open(path, "rt", encoding="utf-8"))["raw"]
    keys, m, y = [], [], []
    for q in sorted(z):
        for d in sorted(z[q]):
            r = z[q][d]
            keys.append((q, d)); m.append(r[0] + r[1] - r[3] - r[4]); y.append(r[10] == 3)
    return keys, np.array(m, float), np.array(y, bool)


def kap(tp, fp, fn, tn):
    tp, fp, fn, tn = (np.asarray(x, float) for x in (tp, fp, fn, tn))
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return (po - pe) / (1 - pe)


def best_threshold(m, y):
    o = np.argsort(m, kind="mergesort")
    ms, ys = m[o], y[o]
    brk = np.concatenate([[True], ms[1:] != ms[:-1]])
    gid = np.cumsum(brk) - 1
    G = gid[-1] + 1
    u = ms[brk]
    pos = np.bincount(gid, weights=ys.astype(float), minlength=G)
    neg = np.bincount(gid, weights=(~ys).astype(float), minlength=G)
    cp, cn = np.cumsum(pos), np.cumsum(neg)
    P_, N_ = cp[-1], cn[-1]
    cpz = np.concatenate([[0.0], cp]); cnz = np.concatenate([[0.0], cn])
    tv = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]])
    k = kap(P_ - cpz, N_ - cnz, cpz, cnz)
    g0 = int(np.searchsorted(u, 0.0, side="right"))
    t_all = np.concatenate([tv, [0.0]])
    k_all = np.concatenate([k, [k[g0]]])
    ok = np.isfinite(k_all)
    t_all, k_all = t_all[ok], k_all[ok]
    key = np.lexsort((t_all, np.abs(t_all), (t_all != 0.0).astype(int), -np.round(k_all, 12)))
    return float(t_all[key[0]])


def auc(m, y):
    r = rankdata(m)
    P, N = y.sum(), (~y).sum()
    return float((r[y].sum() - P * (P + 1) / 2) / (P * N))


def metrics(m, y, t):
    pred = m > t
    tp, fp, fn, tn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum()), int((~pred & ~y).sum())
    n = len(m)
    return dict(kappa=float(kap(tp, fp, fn, tn)), tp=tp, fp=fp, fn=fn, tn=tn, error_rate=(fp + fn) / n, fp_rate=fp / (fp + tn), fn_rate=fn / (fn + tp), pred_pos_rate=(tp + fp) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    D, out = Path(a.dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R, ref = {}, {}
    for c, tag, desc in CONDS:
        v = load(D / f"esciEval_{tag}_step625__T1__sel.json.gz")
        t = load(D / f"esciEval_{tag}_step625__T1__test300.json.gz")
        for nm, x in (("sel", v), ("test", t)):
            if nm not in ref:
                ref[nm] = (x[0], x[2])
            assert x[0] == ref[nm][0] and (x[2] == ref[nm][1]).all(), (c, nm)       # identical pairs, order and gold in every condition
        ts = best_threshold(v[1], v[2])
        R[c] = dict(desc=desc, tag=tag, tstar=ts, val_auc=auc(v[1], v[2]), test_auc=auc(t[1], t[2]), val_t0=metrics(v[1], v[2], 0.0), val_tstar=metrics(v[1], v[2], ts),
                    test_t0=metrics(t[1], t[2], 0.0), test_tstar=metrics(t[1], t[2], ts), val_margin_sd=float(v[1].std()), m_test=t[1], y_test=t[2])
        assert R[c]["val_tstar"]["kappa"] >= R[c]["val_t0"]["kappa"] - 1e-12
    keys = ref["test"][0]
    qs = sorted({q for q, _ in keys})
    qi = {q: i for i, q in enumerate(qs)}
    rq = np.array([qi[q] for q, _ in keys])
    nq = len(qs)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, nq, size=(a.nboot, nq))
    W = np.stack([np.bincount(i, minlength=nq) for i in idx]).astype(float)

    def qk(c, which):
        t = 0.0 if which == "t0" else R[c]["tstar"]
        pred, y = R[c]["m_test"] > t, R[c]["y_test"]
        cnt = np.stack([np.bincount(rq, weights=w.astype(float), minlength=nq) for w in (pred & y, pred & ~y, ~pred & y, ~pred & ~y)], 1)
        return kap(*(W @ cnt).T)
    KB = {(c, w): qk(c, w) for c, _, _ in CONDS for w in ("t0", "tstar")}
    boot = {}
    for x, y in PAIRS:
        for w in ("t0", "tstar"):
            d = KB[(x, w)] - KB[(y, w)]
            pt = R[x]["test_" + w]["kappa"] - R[y]["test_" + w]["kappa"]
            boot[f"{x}-{y}|{w}"] = dict(point=pt, lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), share_pos=float((d > 0).mean()))
    clean = {c: {k: v for k, v in r.items() if k not in ("m_test", "y_test")} for c, r in R.items()}
    json.dump(dict(conditions=clean, bootstrap=boot, nboot=a.nboot, n_test_queries=nq), open(out / "esci245_results.json", "w"), indent=1)
    L = ["## validation / test (T1, seed 0, final 625; decision m > t; t* = validation-kappa maximiser)",
         "| cond | description | val kappa t=0 | t* | val kappa t* | test kappa t=0 | test kappa t* | test AUC | FP/FN @0 | err% @0 | PPR% @0 | FP/FN @t* | err% @t* | PPR% @t* |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, _, _ in CONDS:
        r = R[c]
        a0, a1 = r["test_t0"], r["test_tstar"]
        L.append(f"| {c} | {r['desc']} | {r['val_t0']['kappa']:.4f} | {r['tstar']:+.3f} | {r['val_tstar']['kappa']:.4f} | {a0['kappa']:.4f} | {a1['kappa']:.4f} | {r['test_auc']:.4f} | "
                 f"{a0['fp']}/{a0['fn']} | {100 * a0['error_rate']:.2f} | {100 * a0['pred_pos_rate']:.1f} | {a1['fp']}/{a1['fn']} | {100 * a1['error_rate']:.2f} | {100 * a1['pred_pos_rate']:.1f} |")
    L += ["", f"## paired test-query bootstrap ({a.nboot} resamples, {nq} queries, 95% percentile interval; fixed models and thresholds; exploratory, one seed)",
          "| x - y | kappa diff at t=0 | kappa diff at t* (each own) |", "|---|---|---|"]
    for x, y in PAIRS:
        b0, b1 = boot[f"{x}-{y}|t0"], boot[f"{x}-{y}|tstar"]
        L.append(f"| {x} - {y} | {b0['point']:+.4f} [{b0['lo']:+.4f}, {b0['hi']:+.4f}] | {b1['point']:+.4f} [{b1['lo']:+.4f}, {b1['hi']:+.4f}] |")
    (out / "esci245_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
