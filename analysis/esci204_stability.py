"""Threshold stability under validation-sample variation (experiment log 204, stored-score item 3; Qwen3-4B stages 1-2; paths Q1 / Q2 / Q3 (and P1 / P2 where available), instructions T1-T7).
For every model the 200 validation (selection) queries are resampled with replacement (500 resamples, np.random.default_rng(1), ONE validation resample set for every path / instruction / seed), the threshold is re-selected with the same rule
(hard-kappa maximum of the weighted resample; candidates 0, midpoints of consecutive distinct scores of the resampled rows, below-all, above-all; ties -> 0, then smaller |t|, then smaller t) and applied to the FIXED test300 scores.
Reported: distribution of the test kappa over the resamples, its change relative to t=0 and to the full-validation threshold, the share of test decisions that change, the re-selected thresholds in units of the model's own validation margin SD
(raw thresholds of models with different score scales are not compared), and the near-maximum region of the full-validation kappa curve (candidates within 0.005 of the maximum).
This is a sensitivity of the fixed test result to validation-sample variation; it is separate from the test-query bootstrap interval and does not pick thresholds by test performance.
    python esci204_stability.py --root <dir> --out <dir>"""
import argparse
from pathlib import Path

import numpy as np

import esci204_common as C

R = 500
SETS = {"T2-T7 (main)": list(range(2, 8)), "T1-T7": list(range(1, 8))}


def kap_arr(tp, fp, fn, tn):
    return C.kap_vec(tp, fp, fn, tn)


def best_weighted(order, ms, ys, w):
    """ms ascending, ys bool, w weights (same order); rows with w == 0 are ignored. -> threshold"""
    keep = w > 0
    m, y, ww = ms[keep], ys[keep], w[keep]
    # group by distinct score
    brk = np.concatenate([[True], m[1:] != m[:-1]])
    gid = np.cumsum(brk) - 1
    G = gid[-1] + 1
    pos = np.bincount(gid, weights=ww * y, minlength=G); neg = np.bincount(gid, weights=ww * (~y), minlength=G)
    u = m[brk]
    cp, cn = np.cumsum(pos), np.cumsum(neg)
    P_, N_ = cp[-1], cn[-1]
    # cut g (0..G-1): groups <= g predicted Irrelevant. cut -1: all positive
    cpz = np.concatenate([[0.0], cp]); cnz = np.concatenate([[0.0], cn])           # index g+1
    tvals = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]])      # cuts -1 .. G-1
    tp = P_ - cpz; fp = N_ - cnz; fn = cpz; tn = cnz
    k = kap_arr(tp, fp, fn, tn)
    # candidate t = 0 : groups with value <= 0 predicted negative
    g0 = int(np.searchsorted(u, 0.0, side="right"))                                  # number of groups <= 0 -> cut index g0 - 1 -> array index g0
    t_all = np.concatenate([tvals, [0.0]]); k_all = np.concatenate([k, [k[g0]]])
    ok = np.isfinite(k_all)
    t_all, k_all = t_all[ok], k_all[ok]
    key = np.lexsort((t_all, np.abs(t_all), (t_all != 0.0).astype(int), -np.round(k_all, 12)))
    return float(t_all[key[0]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D, nq, row_q, test_q = C.build_all(a.root)
    vq = sorted({q for e in D.values() for q, _ in e["sel_ks"]})
    vid = {q: i for i, q in enumerate(vq)}
    rng = np.random.default_rng(1)
    vidx = rng.integers(0, len(vq), size=(R, len(vq)))
    VW = np.stack([np.bincount(i, minlength=len(vq)) for i in vidx]).astype(float)           # (R, nVq)
    res = {}
    for key, e in D.items():
        c, T, s = key
        if c not in ("Q1", "Q2", "Q3", "P1", "P2"):
            continue
        sm, sy = e["sel_m"], e["sel_y"]
        rq = np.array([vid[q] for q, _ in e["sel_ks"]])
        order = np.argsort(sm, kind="mergesort")
        ms, ys, rqs = sm[order], sy[order], rq[order]
        ts = np.array([best_weighted(order, ms, ys, VW[b][rqs]) for b in range(R)])
        mt, yt = e["m"], e["y"]
        pos_sorted, neg_sorted = np.sort(mt[yt]), np.sort(mt[~yt])
        P_, N_ = len(pos_sorted), len(neg_sorted)
        tp = P_ - np.searchsorted(pos_sorted, ts, side="right"); fn = P_ - tp
        fp = N_ - np.searchsorted(neg_sorted, ts, side="right"); tn = N_ - fp
        kb = kap_arr(tp, fp, fn, tn)
        n = len(mt)
        pred_b = None
        chg_full = np.array([(np.abs((mt > t) != e["cal"]["pred"])).mean() for t in ts])
        chg_zero = np.array([(np.abs((mt > t) != e["t0"]["pred"])).mean() for t in ts])
        # near-maximum region of the full-validation curve
        u = np.unique(sm); cand = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]])
        sp, sn = np.sort(sm[sy]), np.sort(sm[~sy])
        tpv = len(sp) - np.searchsorted(sp, cand, side="right"); fpv = len(sn) - np.searchsorted(sn, cand, side="right")
        kv = kap_arr(tpv, fpv, len(sp) - tpv, len(sn) - fpv)
        near = cand[kv >= np.nanmax(kv) - 0.005]
        sd = float(sm.std())
        res[key] = dict(kb=kb, ts=ts, chg_full=chg_full, chg_zero=chg_zero, k0=e["t0"]["kappa"], kc=e["cal"]["kappa"], thr=e["thr"], sd=sd, near_lo=float(near.min()), near_hi=float(near.max()),
                        zero_near=bool(near.min() <= 0.0 <= near.max()), n_near=int(len(near)))

    def seeds(c):
        return C.SEEDS

    L = []
    P = L.append
    P(f"Threshold stability under validation-sample variation (experiment log 204): {R} validation-query resamples of the {len(vq)} selection queries, fixed test300; thresholds re-selected with the same rule")
    for name, Ts in SETS.items():
        P(f"\n## {name} (equal-weight mean of per-instruction test kappa, then mean over seeds)")
        P("| path | kappa t=0 | kappa full-val thr | resampled-thr kappa: mean | SD | 2.5% | 97.5% | mean - t0 | share of resamples with kappa > t0 | decisions changed vs full-val thr | vs t=0 |")
        P("|---|---|---|---|---|---|---|---|---|---|---|")
        for c in ("Q1", "Q2", "Q3", "P1", "P2"):
            if not all((c, T, s) in res for T in Ts for s in seeds(c)):
                continue
            kb = np.mean([np.mean([res[(c, T, s)]["kb"] for T in Ts], axis=0) for s in seeds(c)], axis=0)
            k0 = np.mean([res[(c, T, s)]["k0"] for T in Ts for s in seeds(c)]); kc = np.mean([res[(c, T, s)]["kc"] for T in Ts for s in seeds(c)])
            cf = np.mean([res[(c, T, s)]["chg_full"].mean() for T in Ts for s in seeds(c)]); cz = np.mean([res[(c, T, s)]["chg_zero"].mean() for T in Ts for s in seeds(c)])
            P(f"| {c} | {k0:.4f} | {kc:.4f} | {kb.mean():.4f} | {kb.std():.4f} | {np.percentile(kb, 2.5):.4f} | {np.percentile(kb, 97.5):.4f} | {kb.mean() - k0:+.4f} | {(kb > k0).mean():.3f} | {100 * cf:.1f}% | {100 * cz:.1f}% |")
        P("\n(the SD above is the SD over the 500 validation resamples of the equal-weight seed-mean; per-model re-selected thresholds and the near-maximum region follow)")
    P("\n## per-path threshold dispersion (re-selected threshold SD in units of the model's own validation margin SD; mean over models) and near-maximum region of the full-validation curve")
    P("| path | models | mean (SD of t_b / validation margin SD) | mean |t_full| / margin SD | models whose near-max region contains t=0 | mean near-max width / margin SD | mean number of near-max candidate cuts |")
    P("|---|---|---|---|---|---|---|")
    for c in ("Q1", "Q2", "Q3", "P1", "P2"):
        ks = [k for k in res if k[0] == c]
        if not ks:
            continue
        P(f"| {c} | {len(ks)} | {np.mean([res[k]['ts'].std() / res[k]['sd'] for k in ks]):.3f} | {np.mean([abs(res[k]['thr']) / res[k]['sd'] for k in ks]):.3f} | {sum(res[k]['zero_near'] for k in ks)}/{len(ks)} | {np.mean([(res[k]['near_hi'] - res[k]['near_lo']) / res[k]['sd'] for k in ks]):.3f} | {np.mean([res[k]['n_near'] for k in ks]):.1f} |")
    P("\n## per-model (path, T, seed): full-val t, re-selected t 2.5 / 50 / 97.5 %, test kappa at t=0 / full-val t / resample mean (2.5%, 97.5%)")
    for k in sorted(res):
        r = res[k]
        P(f"{k[0]} T{k[1]} s{k[2]}: t_full {r['thr']:+.3f}, t_b {np.percentile(r['ts'], 2.5):+.3f} / {np.percentile(r['ts'], 50):+.3f} / {np.percentile(r['ts'], 97.5):+.3f}, kappa t0 {r['k0']:.4f} / full {r['kc']:.4f} / boot {r['kb'].mean():.4f} ({np.percentile(r['kb'], 2.5):.4f}, {np.percentile(r['kb'], 97.5):.4f}), val margin SD {r['sd']:.2f}")
    (out / "esci204_stability_output.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
