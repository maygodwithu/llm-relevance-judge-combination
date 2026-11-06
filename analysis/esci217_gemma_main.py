"""Analysis of experiment log 217 (Gemma-3-4B-it, ESCI T1-T7; seeds 0/1/2; final 625-update models; own instruction).
Paths: G0 Base, G1 token CE, G2 CE + 1.0 soft kappa, G3 CE + 1.0 AnyLoss-like kappa (alpha 10); training: Base start, same seed initial adapter and row order across the three losses, lambda 1.0 fixed beforehand.
Each path at t = 0 and with its own validation threshold (selection 200 queries / 4,047 rows, hard-kappa maximum over all distinct-score partitions, ties -> 0 then smaller |t|; one threshold per model; the Base threshold is never transferred). Test = frozen test300 (300 queries / 6,078 rows).
Order of aggregation: per-instruction kappa -> equal-weight mean over the instruction set -> mean over seeds. Paired test-query cluster bootstrap, 2,000 resamples, np.random.default_rng(0), one resample set for every path / instruction / seed (thresholds, texts and trained models fixed).
Main comparisons on T1-T7 (calibrated): G2 - G1, G3 - G2, G1 - G0; intervals 95% and 98.33% (Bonferroni, 3 comparisons). Auxiliary: t = 0 versions, G3 - G1, calibration gains; sets T1-T7, T2-T7, T4-T7, T1-T3. Threshold stability: 500 validation-query resamples, default_rng(1).
    python esci217_gemma_main.py --root <dir with cpmr_experiment/ score tree> --out <dir>"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import esci204_common as C
from esci204_stability import best_weighted

TS = list(range(1, 8))
B = 2000
R = 500


class GStore(C.Store):
    def __init__(self, root):
        self.root = Path(root); self.idx = {}; self.cache = {}
        for f in glob.glob(str(self.root / "cpmr_experiment" / "**" / "*.json.gz"), recursive=True):
            b = os.path.basename(f)
            if b.startswith("esciGemmaEval_"):
                self.idx.setdefault(b, f)

    def path(self, model, instr, tag):
        b = f"esciGemmaEval_Base__{instr}__{tag}.json.gz" if model == "Base" else f"esciGemmaEval_{model}_step625__{instr}__{tag}.json.gz"
        return self.idx.get(b)


GCOND = {"G1": "GQ_ce_T{T}_s{s}", "G2": "GQ_mix_T{T}_s{s}", "G3": "GQ_mixany_T{T}_s{s}"}


def load(st, c, T, s):
    mn = "Base" if c == "G0" else GCOND[c].format(T=T, s=s)
    te = st.arrays(mn, f"T{T}", "test300"); se = st.arrays(mn, f"T{T}", "sel")
    return None if te is None or se is None else {"test": te, "sel": (se[1], se[2], se[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    st = GStore(a.root)
    seeds = lambda c: [0] if c == "G0" else C.SEEDS
    D, test_q = {}, None
    PATHS = ("G0", "G1", "G2", "G3")
    for c in PATHS:
        for T in TS:
            for s in seeds(c):
                r = load(st, c, T, s)
                if r is None:
                    continue
                ks, m, y, code = r["test"]
                if test_q is None:
                    test_q = list(ks); qid = {q: i for i, q in enumerate(sorted({q for q, _ in ks}))}; row_q = np.array([qid[q] for q, _ in ks]); nq = len(qid)
                assert ks == test_q
                sm, sy, sks = r["sel"]
                thr, vk = C.best_threshold(sm, sy)
                e = dict(sel_ks=sks, m=m, y=y, code=code, thr=thr, val_kappa=vk, val_auc=C.auc(sm, sy), auc=C.auc(m, y), sel_m=sm, sel_y=sy)
                for mode, t in (("t0", 0.0), ("cal", thr)):
                    p = m > t
                    cnt = np.zeros((nq, 4))
                    for j, arr in enumerate((p & y, p & ~y, ~p & y, ~p & ~y)):
                        cnt[:, j] = np.bincount(row_q, weights=arr.astype(float), minlength=nq)
                    tot = cnt.sum(0)
                    e[mode] = dict(cnt=cnt, pred=p, kappa=C.kap(*tot), acc=float((tot[0] + tot[3]) / tot.sum()), rel=float(p.mean()), fnr=float(tot[2] / (tot[0] + tot[2])), fpr={C.CLS[k]: float((p & (code == k)).sum() / (code == k).sum()) for k in (1, 2, 3)})
                D[(c, T, s)] = e
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(i, minlength=nq) for i in rng.integers(0, nq, size=(B, nq))]).astype(float)
    for e in D.values():
        for mode in ("t0", "cal"):
            tot = W @ e[mode]["cnt"]; e[mode]["boot"] = C.kap_vec(tot[:, 0], tot[:, 1], tot[:, 2], tot[:, 3])
    have = lambda c, Ts: all((c, T, s) in D for T in Ts for s in seeds(c))

    def agg(c, mode, Ts):
        ps, bs = [], []
        for s in seeds(c):
            ps.append(np.mean([D[(c, T, s)][mode]["kappa"] for T in Ts])); bs.append(np.mean([D[(c, T, s)][mode]["boot"] for T in Ts], axis=0))
        return float(np.mean(ps)), np.mean(bs, axis=0), ps, [float(np.mean([D[(c, T, s)][mode]["kappa"] for s in seeds(c)])) for T in Ts]

    def diff(c1, m1, c2, m2, Ts):
        p1, b1, s1, t1 = agg(c1, m1, Ts); p2, b2, s2, t2 = agg(c2, m2, Ts)
        n = max(len(s1), len(s2))
        return p1 - p2, b1 - b2, [s1[i if len(s1) > 1 else 0] - s2[i if len(s2) > 1 else 0] for i in range(n)], [x - y for x, y in zip(t1, t2)]

    ci = lambda v, lv: (float(np.percentile(v, (100 - lv) / 2)), float(np.percentile(v, 100 - (100 - lv) / 2)))
    L = []
    P = L.append
    P("Gemma-3-4B-it ESCI study (experiment log 217); test300 300 queries / 6,078 rows; validation 200 queries / 4,047 rows")
    P("completeness (models with both test300 and validation scores):")
    for c in PATHS:
        P(f"  {c}: " + " ".join(f"T{T}:{sum((c, T, s) in D for s in seeds(c))}/{len(seeds(c))}" for T in TS))
    SETS = {"T1-T7 (main)": TS, "T2-T7": TS[1:], "T4-T7": [4, 5, 6, 7], "T1-T3": [1, 2, 3]}
    for name, Ts in SETS.items():
        P(f"\n## instruction set {name}: per-instruction kappa -> equal-weight mean -> mean over seeds (seed values in brackets; SD over the 3 seeds)")
        P("| path | t=0 kappa | calibrated kappa | cal - t0 [95%] | AUC | acc t0 / cal | Relevant% t0 / cal | E FNR t0 / cal | S/C/I FPR cal |")
        P("|---|---|---|---|---|---|---|---|---|")
        for c in PATHS:
            if not have(c, Ts):
                continue
            p0, b0, s0, _ = agg(c, "t0", Ts); pc, bc, sc, _ = agg(c, "cal", Ts)
            es = [D[(c, T, s)] for T in Ts for s in seeds(c)]
            f = lambda k, md: np.mean([e[md][k] for e in es]); fp = {k: np.mean([e["cal"]["fpr"][k] for e in es]) for k in "SCI"}
            lo, hi = ci(bc - b0, 95)
            P(f"| {c} | {p0:.4f}" + (f" [{', '.join(f'{x:.4f}' for x in s0)}] SD {np.std(s0, ddof=1):.4f}" if len(s0) > 1 else "") + f" | {pc:.4f}" + (f" [{', '.join(f'{x:.4f}' for x in sc)}] SD {np.std(sc, ddof=1):.4f}" if len(sc) > 1 else "") +
              f" | {pc - p0:+.4f} [{lo:+.4f}, {hi:+.4f}] | {np.mean([e['auc'] for e in es]):.4f} | {f('acc', 't0'):.3f} / {f('acc', 'cal'):.3f} | {100 * f('rel', 't0'):.1f} / {100 * f('rel', 'cal'):.1f} | {f('fnr', 't0'):.3f} / {f('fnr', 'cal'):.3f} | {fp['S']:.3f}/{fp['C']:.3f}/{fp['I']:.3f} |")
    COMP = [("MAIN-1 cal G2 - cal G1 (soft kappa extra gain)", "G2", "cal", "G1", "cal", "main"), ("MAIN-2 cal G3 - cal G2 (AnyLoss-like vs soft kappa)", "G3", "cal", "G2", "cal", "main"), ("MAIN-3 cal G1 - cal G0 (SFT extra gain)", "G1", "cal", "G0", "cal", "main"),
            ("aux t0 G2 - G1", "G2", "t0", "G1", "t0", "aux"), ("aux t0 G3 - G2", "G3", "t0", "G2", "t0", "aux"), ("aux t0 G1 - G0", "G1", "t0", "G0", "t0", "aux"),
            ("aux cal G3 - cal G1", "G3", "cal", "G1", "cal", "aux"), ("aux t0 G3 - G1", "G3", "t0", "G1", "t0", "aux"),
            ("calibration gain G0 (cal - t0)", "G0", "cal", "G0", "t0", "aux"), ("calibration gain G1", "G1", "cal", "G1", "t0", "aux"), ("calibration gain G2", "G2", "cal", "G2", "t0", "aux"), ("calibration gain G3", "G3", "cal", "G3", "t0", "aux")]
    P("\n## comparisons (seed-mean kappa differences; test-query cluster paired bootstrap, 2,000 resamples, one resample set; thresholds and texts fixed). 95% and, for the three main comparisons, 98.33%")
    P("| set | comparison | diff | 95% | 98.33% | seed diffs (0/1/2) | per-T diffs (T order of the set) | min / max per-T | #T positive |")
    P("|---|---|---|---|---|---|---|---|---|")
    js = []
    for name, Ts in SETS.items():
        for lab, c1, m1, c2, m2, kind in COMP:
            if not (have(c1, Ts) and have(c2, Ts)):
                continue
            pt, bt, sd, td = diff(c1, m1, c2, m2, Ts)
            lo, hi = ci(bt, 95); lo2, hi2 = ci(bt, 98.33)
            P(f"| {name} | {lab} | {pt:+.4f} | [{lo:+.4f}, {hi:+.4f}] | " + (f"[{lo2:+.4f}, {hi2:+.4f}]" if kind == "main" else "") + f" | {' / '.join(f'{x:+.3f}' for x in sd)} | {' '.join(f'{x:+.3f}' for x in td)} | {min(td):+.3f} / {max(td):+.3f} | {sum(x > 0 for x in td)}/{len(td)} |")
            js.append(dict(set=name, comparison=lab, diff=pt, ci95=[lo, hi], ci9833=[lo2, hi2], seed_diffs=[float(x) for x in sd], per_T=[float(x) for x in td]))
    P("\n## per-instruction seed-mean kappa (t=0 / calibrated) and thresholds (seed 0/1/2)")
    P("| path | " + " | ".join(f"T{T}" for T in TS) + " |"); P("|---|" + "---|" * len(TS))
    for c in PATHS:
        row = []
        for T in TS:
            if not have(c, [T]):
                row.append("-"); continue
            k0 = np.mean([D[(c, T, s)]["t0"]["kappa"] for s in seeds(c)]); kc = np.mean([D[(c, T, s)]["cal"]["kappa"] for s in seeds(c)])
            row.append(f"{k0:.3f} / {kc:.3f} (t " + "/".join(f"{D[(c, T, s)]['thr']:+.2f}" for s in seeds(c)) + ")")
        P(f"| {c} | " + " | ".join(row) + " |")
    P("\n## validation kappa at the chosen threshold / validation AUC / test AUC (mean over models)")
    for c in PATHS:
        es = [e for (cc, T, s), e in D.items() if cc == c]
        if es:
            P(f"  {c}: val kappa {np.mean([e['val_kappa'] for e in es]):.4f}, val AUC {np.mean([e['val_auc'] for e in es]):.4f}, test AUC {np.mean([e['auc'] for e in es]):.4f}, n {len(es)}")
    # ---------------- errors
    code0 = next(iter(D.values()))["code"]
    for name in ("T1-T7 (main)", "T4-T7"):
        Ts = SETS[name]
        P(f"\n## errors per path, {name} (mean row counts per instruction x seed; gold E {int((code0 == 0).sum())}, S {int((code0 == 1).sum())}, C {int((code0 == 2).sum())}, I {int((code0 == 3).sum())})")
        P("| path | mode | FN (E predicted Irrelevant) | FP S | FP C | FP I | total errors |"); P("|---|---|---|---|---|---|---|")
        for c in PATHS:
            if not have(c, Ts):
                continue
            for mode in ("t0", "cal"):
                fn, f1, f2, f3, tot = [], [], [], [], []
                for T in Ts:
                    for s in seeds(c):
                        e = D[(c, T, s)]; p = e[mode]["pred"]; cd = e["code"]
                        fn.append(((~p) & e["y"]).sum()); tot.append((p != e["y"]).sum()); f1.append((p & (cd == 1)).sum()); f2.append((p & (cd == 2)).sum()); f3.append((p & (cd == 3)).sum())
                P(f"| {c} | {mode} | {np.mean(fn):.1f} | {np.mean(f1):.1f} | {np.mean(f2):.1f} | {np.mean(f3):.1f} | {np.mean(tot):.1f} |")
    PAIRS = (("G1", "t0", "G2", "t0"), ("G1", "cal", "G2", "cal"), ("G2", "t0", "G3", "t0"), ("G2", "cal", "G3", "cal"), ("G1", "cal", "G3", "cal"), ("G0", "t0", "G1", "t0"), ("G0", "cal", "G1", "cal"),
             ("G1", "t0", "G1", "cal"), ("G2", "t0", "G2", "cal"), ("G3", "t0", "G3", "cal"), ("G0", "t0", "G0", "cal"))
    for name in ("T1-T7 (main)", "T4-T7", "T1-T3"):
        Ts = SETS[name]
        P(f"\n## decision changes between matched models, {name} (mean count per instruction x seed cell; R->W right before / wrong after; W->R wrong before / right after; WW common errors)")
        P("| from -> to | gold group | R->W | W->R | net | WW | RR | decisions changed |"); P("|---|---|---|---|---|---|---|---|")
        for c1, m1, c2, m2 in PAIRS:
            if not (have(c1, Ts) and have(c2, Ts)):
                continue
            acc = {g: [] for g in ("E", "S+C+I")}
            for T in Ts:
                for s in C.SEEDS:
                    e1 = D[(c1, T, 0 if c1 == "G0" else s)]; e2 = D[(c2, T, 0 if c2 == "G0" else s)]
                    y = e1["y"]; r1 = e1[m1]["pred"] == y; r2 = e2[m2]["pred"] == y; cd = e1["code"]
                    for g, sel in (("E", cd == 0), ("S+C+I", cd > 0)):
                        acc[g].append(((r1 & ~r2 & sel).sum(), (~r1 & r2 & sel).sum(), (~r1 & ~r2 & sel).sum(), (r1 & r2 & sel).sum(), (e1[m1]["pred"] != e2[m2]["pred"])[sel].sum()))
            for g in acc:
                v = np.array(acc[g], dtype=float).mean(0)
                P(f"| {c1} {m1} -> {c2} {m2} | {g} | {v[0]:.1f} | {v[1]:.1f} | {v[1] - v[0]:+.1f} | {v[2]:.1f} | {v[3]:.1f} | {v[4]:.1f} |")
    # ---------------- stability
    vq = sorted({q for e in D.values() for q, _ in e["sel_ks"]}); vid = {q: i for i, q in enumerate(vq)}
    VW = np.stack([np.bincount(i, minlength=len(vq)) for i in np.random.default_rng(1).integers(0, len(vq), size=(R, len(vq)))]).astype(float)
    res = {}
    for key, e in D.items():
        c, T, s = key
        if c not in ("G1", "G2", "G3"):
            continue
        sm, sy = e["sel_m"], e["sel_y"]; rq = np.array([vid[q] for q, _ in e["sel_ks"]]); order = np.argsort(sm, kind="mergesort")
        ts = np.array([best_weighted(order, sm[order], sy[order], VW[b][rq[order]]) for b in range(R)])
        mt, yt = e["m"], e["y"]; ps, ns = np.sort(mt[yt]), np.sort(mt[~yt])
        tp = len(ps) - np.searchsorted(ps, ts, side="right"); fp = len(ns) - np.searchsorted(ns, ts, side="right")
        kb = C.kap_vec(tp, fp, len(ps) - tp, len(ns) - fp)
        u = np.unique(sm); cand = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]])
        sp_, sn_ = np.sort(sm[sy]), np.sort(sm[~sy])
        tpv = len(sp_) - np.searchsorted(sp_, cand, side="right"); fpv = len(sn_) - np.searchsorted(sn_, cand, side="right")
        kv = C.kap_vec(tpv, fpv, len(sp_) - tpv, len(sn_) - fpv); near = cand[kv >= np.nanmax(kv) - 0.005]
        res[key] = dict(kb=kb, chg=np.array([(np.abs((mt > t) != e["cal"]["pred"])).mean() for t in ts]), chg0=np.array([(np.abs((mt > t) != e["t0"]["pred"])).mean() for t in ts]), ts=ts, sd=float(sm.std()), thr=e["thr"],
                        zero_near=bool(near.min() <= 0.0 <= near.max()), width=float(near.max() - near.min()))
    P(f"\n## threshold stability under validation-sample variation ({R} resamples of the 200 selection queries, thresholds re-selected with the same rule, fixed test300); the near-maximum region = candidates within 0.005 of the full-validation maximum")
    P("| path (instructions) | kappa t=0 | kappa full-val thr | re-selected-thr kappa mean +- SD [2.5%, 97.5%] | share of resamples with kappa > t=0 | decisions changed vs full-val thr | vs t=0 | SD of re-selected t / validation margin SD | models whose near-max region contains t=0 | near-max width / margin SD |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for c in ("G1", "G2", "G3"):
        for Ts in (TS, [4, 5, 6, 7]):
            if not all((c, T, s) in res for T in Ts for s in C.SEEDS):
                continue
            kb = np.mean([np.mean([res[(c, T, s)]["kb"] for T in Ts], axis=0) for s in C.SEEDS], axis=0)
            cells = [(T, s) for T in Ts for s in C.SEEDS]
            k0 = np.mean([D[(c, T, s)]["t0"]["kappa"] for T, s in cells]); kc = np.mean([D[(c, T, s)]["cal"]["kappa"] for T, s in cells])
            P(f"| {c} (T{','.join(map(str, Ts))}) | {k0:.4f} | {kc:.4f} | {kb.mean():.4f} +- {kb.std():.4f} [{np.percentile(kb, 2.5):.4f}, {np.percentile(kb, 97.5):.4f}] | {(kb > k0).mean():.3f} | {100 * np.mean([res[(c, T, s)]['chg'].mean() for T, s in cells]):.1f}% | {100 * np.mean([res[(c, T, s)]['chg0'].mean() for T, s in cells]):.1f}% | "
              f"{np.mean([res[(c, T, s)]['ts'].std() / res[(c, T, s)]['sd'] for T, s in cells]):.3f} | {sum(res[(c, T, s)]['zero_near'] for T, s in cells)}/{len(cells)} | {np.mean([res[(c, T, s)]['width'] / res[(c, T, s)]['sd'] for T, s in cells]):.3f} |")
    (out / "esci217_gemma_output.txt").write_text("\n".join(L), encoding="utf-8")
    (out / "esci217_gemma_results.json").write_text(json.dumps(js, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
