"""Analysis of experiment log 224 (Gemma-3-4B-it, MS MARCO original instructions T1-T7; own-instruction; final 625-update models; TREC-DL 2019 42 queries / 1,102 judged pairs and 2020 54 / 1,432; grades 2/3 = Relevant, 0/1 = Irrelevant).
Paths (labels GM0-GM3 of the request 224 are written M0-M3 in the code): M0 Base, M1 token CE only, M2 CE + 1.0 soft kappa, M3 CE + 1.0 AnyLoss-like kappa (alpha 10); all new Gemma models (no reuse), final 625 updates, own instruction.
Each model at t = 0 and with CROSS calibration: the threshold is the hard-kappa maximum over ALL judged pairs of the OTHER year (2020 -> applied to 2019, 2019 -> applied to 2020), candidates 0 / midpoints of consecutive distinct scores / below-all / above-all, ties -> 0, then smaller |t|, then smaller t; m > t; one threshold per (model, instruction, seed, year); never the same year's labels.
Aggregation: per-instruction kappa -> equal-weight mean over the instruction set -> mean over seeds (no pooled confusion matrix). Paired test-query-cluster bootstrap, 10,000 resamples per year, np.random.default_rng(<year>), one resample set per year for every path / instruction / seed; thresholds fixed (only test-query variation).
    python msm_gemma_analysis.py --root <dir with cpmr_experiment/ score tree> --out <dir>"""
import argparse
import glob
import gzip
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ld = lambda p: json.load(gzip.open(p, "rt", encoding="utf-8"))
YEARS = ("2019", "2020")
SEEDS = (0, 1, 2)
NB = 10000


def kap_vec(tp, fp, fn, tn):
    tp, fp, fn, tn = [np.asarray(x, dtype=np.float64) for x in (tp, fp, fn, tn)]
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pe == 1, np.nan, (po - pe) / (1 - pe))


def kap(tp, fp, fn, tn):
    return float(kap_vec(tp, fp, fn, tn))


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
    a, b = y.sum(), (~y).sum()
    return float((rk[y].sum() - a * (a + 1) / 2) / (a * b))


class Files:
    def __init__(self, root):
        self.idx = {}
        for f in glob.glob(str(Path(root) / "cpmr_experiment" / "**" / "msmGemmaEval_*.json.gz"), recursive=True):
            self.idx.setdefault(os.path.basename(f), f)

    def get(self, tag, T, y):
        f = self.idx.get(f"msmGemmaEval_{tag}__T{T}__{y}.json.gz")
        if f is None:
            return None
        r = ld(f)
        run, gr = r["run_fp32"], r["grades"]
        qs = sorted(run)
        ks = [(q, d) for q in qs for d in sorted(run[q])]
        return ks, np.array([run[q][d] for q, d in ks]), np.array([gr[q][d] >= 2 for q, d in ks]), np.array([gr[q][d] for q, d in ks])


def tags(path, T, s):
    """-> eval tag of the model (None if the path has no model for this instruction)"""
    if path == "M0":
        return "Base" if s == 0 else None
    if path == "M1":
        return f"gmq_B_T{T}_s{s}"
    if path == "M2":
        return f"gmq_E_T{T}_s{s}"
    if path == "M3":
        return f"gmq_M_T{T}_s{s}"
    if path == "A1":
        return None
    if path == "AF":
        return None
    if path == "AX":
        return None
    raise ValueError(path)


PATHS = ("M0", "M1", "M2", "M3", "A1", "AF", "AX")
LAB = {"M0": "M0 Base", "M1": "M1 CE", "M2": "M2 CE+soft k", "M3": "M3 CE+AnyLoss k", "A1": "(aux) A1 CE+ranking, retrained here, T1", "AF": "(aux) AF CE+ranking, first-server model, T1", "AX": "(aux) AX CE+ranking, first-server models, T2-T7"}
CORE = ("M0", "M1", "M2", "M3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    F = Files(a.root)
    D = {}
    qinfo = {}
    for p in PATHS:
        for T in range(1, 8):
            for s in SEEDS:
                tg = tags(p, T, s)
                if tg is None:
                    continue
                per = {}
                for y in YEARS:
                    r = F.get(tg, T, y)
                    if r is None:
                        per = None
                        break
                    per[y] = r
                if per is None:
                    continue
                e = {}
                for y in YEARS:
                    ks, m, yy, gr = per[y]
                    if y not in qinfo:
                        qs = sorted({q for q, _ in ks}); qinfo[y] = (ks, {q: i for i, q in enumerate(qs)}, len(qs))
                    assert ks == qinfo[y][0], (p, T, s, y)
                    e[y] = dict(m=m, y=yy, grade=gr)
                for y in YEARS:
                    o = YEARS[1 - YEARS.index(y)]
                    thr, vk = best_threshold(e[o]["m"], e[o]["y"])
                    e[y]["thr"] = thr; e[y]["thr_val_kappa"] = vk
                    qi = qinfo[y]
                    rq = np.array([qi[1][q] for q, _ in qi[0]])
                    for mode, t in (("t0", 0.0), ("cross", thr)):
                        pr = e[y]["m"] > t; yy = e[y]["y"]
                        cnt = np.zeros((qi[2], 4))
                        for j, arr in enumerate((pr & yy, pr & ~yy, ~pr & yy, ~pr & ~yy)):
                            cnt[:, j] = np.bincount(rq, weights=arr.astype(float), minlength=qi[2])
                        tot = cnt.sum(0)
                        e[y][mode] = dict(cnt=cnt, pred=pr, kappa=kap(*tot), acc=float((tot[0] + tot[3]) / tot.sum()), rel=float(pr.mean()), fnr=float(tot[2] / (tot[0] + tot[2])), fpr=float(tot[1] / (tot[1] + tot[3])), tot=[int(x) for x in tot])
                    e[y]["auc"] = auc(e[y]["m"], e[y]["y"])
                D[(p, T, s)] = e
    W = {}
    for y in YEARS:
        rng = np.random.default_rng(int(y))
        nq = qinfo[y][2]
        W[y] = np.stack([np.bincount(i, minlength=nq) for i in rng.integers(0, nq, size=(NB, nq))]).astype(np.float32)
    for e in D.values():
        for y in YEARS:
            for mode in ("t0", "cross"):
                tot = W[y].astype(np.float64) @ e[y][mode]["cnt"]
                e[y][mode]["boot"] = kap_vec(tot[:, 0], tot[:, 1], tot[:, 2], tot[:, 3])

    seeds = lambda p: [0] if p == "M0" else list(SEEDS)
    have = lambda p, Ts: all((p, T, s) in D for T in Ts for s in seeds(p))

    def agg(p, y, mode, Ts):
        ps, bs = [], []
        for s in seeds(p):
            ps.append(np.mean([D[(p, T, s)][y][mode]["kappa"] for T in Ts])); bs.append(np.mean([D[(p, T, s)][y][mode]["boot"] for T in Ts], axis=0))
        return float(np.mean(ps)), np.mean(bs, axis=0), ps, [float(np.mean([D[(p, T, s)][y][mode]["kappa"] for s in seeds(p)])) for T in Ts]

    def diff(p1, m1, p2, m2, y, Ts):
        x1, b1, s1, t1 = agg(p1, y, m1, Ts); x2, b2, s2, t2 = agg(p2, y, m2, Ts)
        n = max(len(s1), len(s2))
        return x1 - x2, b1 - b2, [s1[i if len(s1) > 1 else 0] - s2[i if len(s2) > 1 else 0] for i in range(n)], [u - v for u, v in zip(t1, t2)]

    ci = lambda v, lv: (float(np.nanpercentile(v, (100 - lv) / 2)), float(np.nanpercentile(v, 100 - (100 - lv) / 2)))
    L = []
    P = L.append
    P("Gemma-3-4B-it MS MARCO T1-T7 loss comparison (experiment log 224): TREC-DL 2019 (42 queries, 1,102 judged pairs) and 2020 (54 queries, 1,432); CROSS = threshold from the other year's judged pairs")
    ex = next(iter(D.values()))
    for y in YEARS:
        P(f"  {y}: Relevant (grade>=2) {100 * ex[y]['y'].mean():.1f}% of judged pairs")
    P("completeness (models with both years): " + "; ".join(f"{p}: " + " ".join(f"T{T}:{sum((p, T, s) in D for s in seeds(p))}/{len(seeds(p))}" for T in range(1, 8)) for p in PATHS))
    SETS = {"T2-T7 (main)": list(range(2, 8)), "T1-T7": list(range(1, 8)), "T1": [1]}
    js = []
    for y in YEARS:
        for name, Ts in SETS.items():
            P(f"\n## TREC-DL {y}, instruction set {name}: per-instruction kappa -> equal-weight mean -> mean over seeds (seed values in brackets, SD over 3 seeds)")
            P("| path | t=0 kappa | CROSS kappa | CROSS - t0 [95%] | AUC | acc t0 / CROSS | Relevant% t0 / CROSS (gold " + f"{100 * ex[y]['y'].mean():.1f}" + "%) | FNR t0 / CROSS | FPR t0 / CROSS |")
            P("|---|---|---|---|---|---|---|---|---|")
            for p in PATHS:
                if not have(p, Ts):
                    continue
                x0, b0, s0, _ = agg(p, y, "t0", Ts); xc, bc, sc, _ = agg(p, y, "cross", Ts)
                es = [D[(p, T, s)][y] for T in Ts for s in seeds(p)]
                f = lambda k, md: np.mean([e[md][k] for e in es])
                lo, hi = ci(bc - b0, 95)
                P(f"| {LAB[p]} | {x0:.4f}" + (f" [{', '.join(f'{v:.4f}' for v in s0)}] SD {np.std(s0, ddof=1):.4f}" if len(s0) > 1 else "") + f" | {xc:.4f}" + (f" [{', '.join(f'{v:.4f}' for v in sc)}] SD {np.std(sc, ddof=1):.4f}" if len(sc) > 1 else "") +
                  f" | {xc - x0:+.4f} [{lo:+.4f}, {hi:+.4f}] | {np.mean([e['auc'] for e in es]):.4f} | {f('acc', 't0'):.3f} / {f('acc', 'cross'):.3f} | {100 * f('rel', 't0'):.1f} / {100 * f('rel', 'cross'):.1f} | {f('fnr', 't0'):.3f} / {f('fnr', 'cross'):.3f} | {f('fpr', 't0'):.3f} / {f('fpr', 'cross'):.3f} |")
    MAIN = [("MAIN-A cross M2 - M1 (soft kappa extra gain)", "M2", "cross", "M1", "cross"), ("MAIN-B cross M3 - M2 (AnyLoss-like vs soft kappa)", "M3", "cross", "M2", "cross"), ("MAIN-C cross M1 - M0 (SFT extra gain)", "M1", "cross", "M0", "cross")]
    AUX = [("aux t0 M2 - M1", "M2", "t0", "M1", "t0"), ("aux t0 M3 - M2", "M3", "t0", "M2", "t0"), ("aux t0 M1 - M0", "M1", "t0", "M0", "t0"), ("aux cross M3 - M1", "M3", "cross", "M1", "cross"), ("aux t0 M3 - M1", "M3", "t0", "M1", "t0"),
           ("calibration gain M0 (cross - t0)", "M0", "cross", "M0", "t0"), ("calibration gain M1", "M1", "cross", "M1", "t0"), ("calibration gain M2", "M2", "cross", "M2", "t0"), ("calibration gain M3", "M3", "cross", "M3", "t0"),
           ("(aux CE+ranking, other loss/environment) cross AX - M1", "AX", "cross", "M1", "cross"), ("(aux CE+ranking) t0 AX - M1", "AX", "t0", "M1", "t0"), ("(aux CE+ranking) cross AX - M0", "AX", "cross", "M0", "cross"),
           ("(aux CE+ranking, T1) cross A1 - M1", "A1", "cross", "M1", "cross"), ("(aux CE+ranking, T1) cross AF - M1", "AF", "cross", "M1", "cross")]
    P("\n## comparisons (seed-mean kappa differences; test-query cluster paired bootstrap, " + str(NB) + " resamples per year, thresholds fixed). 95% for all; 99.1667% for the six main contrasts (3 comparisons x 2 years, Bonferroni 6)")
    P("| year | set | comparison | diff | 95% | 99.1667% | seed diffs | per-instruction diffs | min / max | #T positive |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for y in YEARS:
        for name, Ts in SETS.items():
            for lab, p1, m1, p2, m2 in MAIN + AUX:
                if not (have(p1, Ts) and have(p2, Ts)):
                    continue
                main = (p1, m1, p2, m2) in [(x[1], x[2], x[3], x[4]) for x in MAIN]
                pt, bt, sd, td = diff(p1, m1, p2, m2, y, Ts)
                lo, hi = ci(bt, 95); lo2, hi2 = ci(bt, 99.1667)
                P(f"| {y} | {name} | {lab} | {pt:+.4f} | [{lo:+.4f}, {hi:+.4f}] | " + (f"[{lo2:+.4f}, {hi2:+.4f}]" if main and name.startswith("T2-T7") else "") + f" | {' / '.join(f'{v:+.3f}' for v in sd)} | {' '.join(f'{v:+.3f}' for v in td)} | {min(td):+.3f} / {max(td):+.3f} | {sum(v > 0 for v in td)}/{len(td)} |")
                js.append(dict(year=y, set=name, comparison=lab, diff=pt, ci95=[lo, hi], ci99167=[lo2, hi2], seed_diffs=[float(v) for v in sd], per_T=[float(v) for v in td]))
    # per-instruction table
    for y in YEARS:
        P(f"\n## TREC-DL {y}: per-instruction seed-mean kappa t=0 / CROSS (CROSS threshold seed 0/1/2)")
        P("| path | " + " | ".join(f"T{T}" for T in range(1, 8)) + " |"); P("|---|" + "---|" * 7)
        for p in PATHS:
            row = []
            for T in range(1, 8):
                if not have(p, [T]):
                    row.append("-"); continue
                k0 = np.mean([D[(p, T, s)][y]["t0"]["kappa"] for s in seeds(p)]); kc = np.mean([D[(p, T, s)][y]["cross"]["kappa"] for s in seeds(p)])
                row.append(f"{k0:.3f} / {kc:.3f} (t " + "/".join(f"{D[(p, T, s)][y]['thr']:+.2f}" for s in seeds(p)) + ")")
            P(f"| {p} | " + " | ".join(row) + " |")
    # confusion counts and errors
    for y in YEARS:
        for name in ("T2-T7 (main)", "T1-T7"):
            Ts = SETS[name]
            P(f"\n## TREC-DL {y}, {name}: mean confusion counts per model (TP / FN / FP / TN; gold Relevant {int(ex[y]['y'].sum())} of {len(ex[y]['y'])})")
            P("| path | mode | TP | FN | FP | TN |"); P("|---|---|---|---|---|---|")
            for p in CORE:
                if not have(p, Ts):
                    continue
                for mode in ("t0", "cross"):
                    c = np.mean([D[(p, T, s)][y][mode]["tot"] for T in Ts for s in seeds(p)], axis=0)
                    P(f"| {p} | {mode} | {c[0]:.1f} | {c[2]:.1f} | {c[1]:.1f} | {c[3]:.1f} |")
    PAIRS = (("M1", "t0", "M2", "t0"), ("M1", "cross", "M2", "cross"), ("M2", "t0", "M3", "t0"), ("M2", "cross", "M3", "cross"), ("M0", "cross", "M1", "cross"), ("M1", "t0", "M1", "cross"), ("M2", "t0", "M2", "cross"), ("M3", "t0", "M3", "cross"), ("M0", "t0", "M0", "cross"))
    for y in YEARS:
        Ts = SETS["T2-T7 (main)"]
        P(f"\n## TREC-DL {y}, T2-T7: decision changes between matched models (mean count per instruction x seed cell; R->W right before, wrong after; W->R wrong before, right after; WW common errors)")
        P("| from -> to | gold | R->W | W->R | net | WW | RR | changed |"); P("|---|---|---|---|---|---|---|---|")
        for p1, m1, p2, m2 in PAIRS:
            if not (have(p1, Ts) and have(p2, Ts)):
                continue
            acc = {"Relevant": [], "Irrelevant": []}
            for T in Ts:
                for s in SEEDS:
                    e1 = D[(p1, T, 0 if p1 == "M0" else s)][y]; e2 = D[(p2, T, 0 if p2 == "M0" else s)][y]
                    yy = e1["y"]; r1 = e1[m1]["pred"] == yy; r2 = e2[m2]["pred"] == yy
                    for g, sel in (("Relevant", yy), ("Irrelevant", ~yy)):
                        acc[g].append(((r1 & ~r2 & sel).sum(), (~r1 & r2 & sel).sum(), (~r1 & ~r2 & sel).sum(), (r1 & r2 & sel).sum(), (e1[m1]["pred"] != e2[m2]["pred"])[sel].sum()))
            for g in acc:
                v = np.array(acc[g], dtype=float).mean(0)
                P(f"| {p1} {m1} -> {p2} {m2} | {g} | {v[0]:.1f} | {v[1]:.1f} | {v[1] - v[0]:+.1f} | {v[2]:.1f} | {v[3]:.1f} | {v[4]:.1f} |")
    P("\n## score (margin m) distribution per path, T2-T7 models pooled (mean over instructions x seeds): mean, SD, share m > 0 on gold Relevant / gold Irrelevant")
    for y in YEARS:
        for p in CORE:
            Ts = SETS["T2-T7 (main)"]
            if not have(p, Ts):
                continue
            es = [D[(p, T, s)][y] for T in Ts for s in seeds(p)]
            P(f"  {y} {p}: mean m {np.mean([e['m'].mean() for e in es]):+.2f}, SD {np.mean([e['m'].std() for e in es]):.2f}, m>0 on Relevant {np.mean([(e['m'][e['y']] > 0).mean() for e in es]):.3f}, on Irrelevant {np.mean([(e['m'][~e['y']] > 0).mean() for e in es]):.3f}, CROSS thr mean {np.mean([e['thr'] for e in es]):+.2f}")
    (out / "msm_gemma_output.txt").write_text("\n".join(L), encoding="utf-8")
    (out / "msm_gemma_results.json").write_text(json.dumps(js, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
