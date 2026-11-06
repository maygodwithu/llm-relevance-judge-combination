"""Analysis of experiment log 208 (Llama-3.1-8B-Instruct, ESCI T1-T7; seeds 0/1/2; final 625-update models; own instruction): Llama completion of the Qwen 204 comparison.
Paths: L0 Base original text, L1 original CE (168), L2 original CE + soft kappa (lambda 1; T1-T3 from 207, T4-T7 new), L3 APO-CF -> CE (168; frozen Llama APO-CF text: T2 shorter, T3 bare, T5 shorter, T7 similar; T1, T4, T6 kept the original text, so L3 = L1 there: same models, difference exactly 0, no double counting),
L4 original CE + AnyLoss-like kappa (lambda 1, alpha 10, new), L5 APO-CF -> CE + soft kappa (T2, T3 new; T1 kept the original text so L5 = L2 there; T1-T3 fixed subset, never averaged into the T4-T7 or T1-T7 sets), L0P Base with the frozen APO-CF text (reference).
Each path at t = 0 and with its own validation threshold (selection 200 queries / 4,047 rows, hard-kappa maximum; the Base threshold is never transferred). Test = frozen test300 (300 queries / 6,078 rows).
Order of aggregation: per-instruction kappa -> equal-weight mean over the instruction set -> mean over seeds. Paired test-query cluster bootstrap, 2,000 resamples, np.random.default_rng(0), one resample set for every path / instruction / seed (thresholds and texts fixed; selection, validation sample and training seed not included).
Intervals: 95% for everything and 98.33% for the three main comparisons on T4-T7 (cal L2 - L1, cal L4 - L2, cal L1 - L0). Error counts, decision changes and threshold stability (500 validation-query resamples, default_rng(1)) as in the Qwen analysis.
    python esci204_llama_main.py --root <dir with cpmr_experiment/ score tree> --out <dir>"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

import esci204_common as C
from esci204_stability import best_weighted

TS = list(range(1, 8))
B = 2000
R = 500


class LStore(C.Store):
    def __init__(self, root):
        self.root = Path(root); self.idx = {}; self.cache = {}
        for f in glob.glob(str(self.root / "cpmr_experiment" / "**" / "*.json.gz"), recursive=True):
            b = os.path.basename(f)
            if b.startswith("esciLlamaEval_"):
                self.idx.setdefault(b, f)

    def path(self, model, instr, tag):
        b = f"esciLlamaEval_Base__{instr}__{tag}.json.gz" if model == "Base" else f"esciLlamaEval_{model}_step625__{instr}__{tag}.json.gz"
        return self.idx.get(b)

    def base_sel(self, T, apocf):
        d = self.root / "cpmr_experiment" / "esci_llama" / f"p1_T{T}"
        idx = json.load(open(d / "p1_result.json"))["selection"]["index"] if apocf else 0
        p1 = C.ld(d / f"p1_raw_cand{idx}.json.gz")["margins"]
        ks = [(q, k) for q in sorted(p1) for k in sorted(p1[q])]
        return ks, np.array([p1[q][k][0] + p1[q][k][1] - p1[q][k][2] - p1[q][k][3] for q, k in ks])


LCOND = {"L1": ("EL_T{T}_s{s}", "T{T}"), "L2": ("ELQ_mix_T{T}_s{s}", "T{T}"), "L3": ("EL_T{T}_APOCF_s{s}", "T{T}_APOCF"), "L4": ("ELQ_mixany_T{T}_s{s}", "T{T}"), "L5": ("ELQ_mix_T{T}_APOCF_s{s}", "T{T}_APOCF")}
PO_KEPT = {1, 4, 6}                                                    # Llama instructions whose APO-CF selection kept the original text


def load(st, c, T, s, gold):
    if c == "L3" and T in PO_KEPT:
        return load(st, "L1", T, s, gold)
    if c == "L5" and T in PO_KEPT:
        return load(st, "L2", T, s, gold)
    if c == "L0P" and T in PO_KEPT:
        return load(st, "L0", T, 0, gold)
    if c in ("L0", "L0P"):
        ins = f"T{T}" + ("_APOCF" if c == "L0P" else "")
        te = st.arrays("Base", ins, "test300")
        ks, m = st.base_sel(T, c == "L0P")
        return None if te is None else {"test": te, "sel": (m, np.array([gold[k] for k in ks]), ks)}
    mn, ins = LCOND[c][0].format(T=T, s=s), LCOND[c][1].format(T=T)
    te = st.arrays(mn, ins, "test300"); se = st.arrays(mn, ins, "sel")
    return None if te is None or se is None else {"test": te, "sel": (se[1], se[2], se[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    st = LStore(a.root)
    gold = {}
    for l in open(Path(a.root) / "cpmr_experiment/esci/audit/sel_rows.jsonl", encoding="utf-8"):
        r = json.loads(l); gold[(r["query"], r["product_id"])] = r["label"] == "Relevant"
    seeds = lambda c: [0] if c in ("L0", "L0P") else C.SEEDS
    D, test_q = {}, None
    PATHS = ("L0", "L0P", "L1", "L2", "L3", "L4", "L5")
    for c in PATHS:
        for T in TS:
            for s in seeds(c):
                r = load(st, c, T, s, gold)
                if r is None or r["sel"] is None:
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
    P("Llama ESCI completion (experiment log 208), Llama-3.1-8B-Instruct; test300 300 queries / 6,078 rows; validation 200 queries / 4,047 rows")
    P("completeness (models with both test300 and validation scores; PO-kept instructions T1, T4, T6 reuse L1 / L2 / L0):")
    for c in PATHS:
        P(f"  {c}: " + " ".join(f"T{T}:{sum((c, T, s) in D for s in seeds(c))}/{len(seeds(c))}" for T in TS))
    SETS = {"T4-T7 (main)": [4, 5, 6, 7], "T1-T7": TS, "T1-T3": [1, 2, 3], "T2-T3 (PO changed the text, T1-T3 subset)": [2, 3], "T5,T7 (PO changed the text, T4-T7 subset)": [5, 7]}
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
    COMP = [("MAIN-1 cal L2 - cal L1 (soft kappa extra gain)", "L2", "cal", "L1", "cal", "main"), ("MAIN-2 cal L4 - cal L2 (AnyLoss-like vs soft kappa)", "L4", "cal", "L2", "cal", "main"), ("MAIN-3 cal L1 - cal L0 (SFT extra gain)", "L1", "cal", "L0", "cal", "main"),
            ("aux t0 L2 - L1", "L2", "t0", "L1", "t0", "aux"), ("aux t0 L4 - L2", "L4", "t0", "L2", "t0", "aux"), ("aux t0 L1 - L0", "L1", "t0", "L0", "t0", "aux"),
            ("aux cal L4 - cal L1", "L4", "cal", "L1", "cal", "aux"), ("aux t0 L4 - L1", "L4", "t0", "L1", "t0", "aux"),
            ("calibration gain L0 (cal - t0)", "L0", "cal", "L0", "t0", "aux"), ("calibration gain L1", "L1", "cal", "L1", "t0", "aux"), ("calibration gain L2", "L2", "cal", "L2", "t0", "aux"), ("calibration gain L4", "L4", "cal", "L4", "t0", "aux"),
            ("calibration gain L3", "L3", "cal", "L3", "t0", "aux"), ("calibration gain L5", "L5", "cal", "L5", "t0", "aux"),
            ("PO effect L3 cal - L1 cal", "L3", "cal", "L1", "cal", "aux"), ("PO effect L3 t0 - L1 t0", "L3", "t0", "L1", "t0", "aux"),
            ("PO effect L5 cal - L2 cal", "L5", "cal", "L2", "cal", "aux"), ("PO effect L5 t0 - L2 t0", "L5", "t0", "L2", "t0", "aux"),
            ("L5 cal - L3 cal", "L5", "cal", "L3", "cal", "aux"), ("L5 t0 - L3 t0", "L5", "t0", "L3", "t0", "aux"),
            ("L3 cal - L0 cal", "L3", "cal", "L0", "cal", "aux"), ("L0P cal - L0 cal (PO text on Base)", "L0P", "cal", "L0", "cal", "aux"), ("L0P t0 - L0 t0", "L0P", "t0", "L0", "t0", "aux"), ("L3 cal - L0P cal (SFT after PO)", "L3", "cal", "L0P", "cal", "aux")]
    P("\n## comparisons (seed-mean kappa differences; test-query cluster paired bootstrap, 2,000 resamples, one resample set; thresholds and texts fixed). 95% and, for the three main comparisons on T4-T7, 98.33%")
    P("| set | comparison | diff | 95% | 98.33% | seed diffs (0/1/2) | per-T diffs (T order of the set) | min / max per-T | #T positive |")
    P("|---|---|---|---|---|---|---|---|---|")
    js = []
    for name, Ts in SETS.items():
        for lab, c1, m1, c2, m2, kind in COMP:
            if not (have(c1, Ts) and have(c2, Ts)):
                continue
            pt, bt, sd, td = diff(c1, m1, c2, m2, Ts)
            lo, hi = ci(bt, 95); lo2, hi2 = ci(bt, 98.33)
            P(f"| {name} | {lab} | {pt:+.4f} | [{lo:+.4f}, {hi:+.4f}] | " + (f"[{lo2:+.4f}, {hi2:+.4f}]" if kind == "main" and name.startswith("T4-T7") else "") + f" | {' / '.join(f'{x:+.3f}' for x in sd)} | {' '.join(f'{x:+.3f}' for x in td)} | {min(td):+.3f} / {max(td):+.3f} | {sum(x > 0 for x in td)}/{len(td)} |")
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
    for name in ("T4-T7 (main)", "T1-T7"):
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
    PAIRS = (("L1", "t0", "L2", "t0"), ("L1", "cal", "L2", "cal"), ("L2", "t0", "L4", "t0"), ("L2", "cal", "L4", "cal"), ("L1", "cal", "L4", "cal"), ("L0", "t0", "L1", "t0"), ("L0", "cal", "L1", "cal"), ("L1", "cal", "L3", "cal"), ("L1", "t0", "L3", "t0"),
             ("L2", "cal", "L5", "cal"), ("L3", "cal", "L5", "cal"), ("L1", "t0", "L1", "cal"), ("L2", "t0", "L2", "cal"), ("L4", "t0", "L4", "cal"), ("L0", "t0", "L0", "cal"))
    for name in ("T4-T7 (main)", "T1-T7", "T1-T3"):
        Ts = SETS[name]
        P(f"\n## decision changes between matched models, {name} (mean count per instruction x seed cell; R->W right before / wrong after; W->R wrong before / right after; WW common errors)")
        P("| from -> to | gold group | R->W | W->R | net | WW | RR | decisions changed |"); P("|---|---|---|---|---|---|---|---|")
        for c1, m1, c2, m2 in PAIRS:
            if not (have(c1, Ts) and have(c2, Ts)):
                continue
            acc = {g: [] for g in ("E", "S+C+I")}
            for T in Ts:
                for s in C.SEEDS:
                    e1 = D[(c1, T, 0 if c1 in ("L0", "L0P") else s)]; e2 = D[(c2, T, 0 if c2 in ("L0", "L0P") else s)]
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
        if c not in ("L1", "L2", "L3", "L4", "L5") or (c == "L3" and T in PO_KEPT) or (c == "L5" and T in PO_KEPT):
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
    P("| path (instructions) | kappa t=0 | kappa full-val thr | re-selected-thr kappa mean +- SD [2.5%, 97.5%] | share of resamples with kappa > t=0 | decisions changed vs full-val thr | vs t=0 | SD of re-selected t / validation margin SD | models whose near-max region contains t=0 | near-max width / margin SD |"); P("|---|---|---|---|---|---|---|---|---|---|")
    for c, Ts in (("L1", [4, 5, 6, 7]), ("L2", [4, 5, 6, 7]), ("L4", [4, 5, 6, 7]), ("L3", [5, 7]), ("L1", TS), ("L2", TS), ("L4", TS), ("L3", [2, 3, 5, 7]), ("L5", [2, 3])):
        if not all((c, T, s) in res for T in Ts for s in C.SEEDS):
            continue
        kb = np.mean([np.mean([res[(c, T, s)]["kb"] for T in Ts], axis=0) for s in C.SEEDS], axis=0)
        cells = [(T, s) for T in Ts for s in C.SEEDS]
        k0 = np.mean([D[(c, T, s)]["t0"]["kappa"] for T, s in cells]); kc = np.mean([D[(c, T, s)]["cal"]["kappa"] for T, s in cells])
        P(f"| {c} (T{','.join(map(str, Ts))}) | {k0:.4f} | {kc:.4f} | {kb.mean():.4f} +- {kb.std():.4f} [{np.percentile(kb, 2.5):.4f}, {np.percentile(kb, 97.5):.4f}] | {(kb > k0).mean():.3f} | {100 * np.mean([res[(c, T, s)]['chg'].mean() for T, s in cells]):.1f}% | {100 * np.mean([res[(c, T, s)]['chg0'].mean() for T, s in cells]):.1f}% | "
          f"{np.mean([res[(c, T, s)]['ts'].std() / res[(c, T, s)]['sd'] for T, s in cells]):.3f} | {sum(res[(c, T, s)]['zero_near'] for T, s in cells)}/{len(cells)} | {np.mean([res[(c, T, s)]['width'] / res[(c, T, s)]['sd'] for T, s in cells]):.3f} |")
    (out / "esci208_llama_output.txt").write_text("\n".join(L), encoding="utf-8")
    (out / "esci208_llama_results.json").write_text(json.dumps(js, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
