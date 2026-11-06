"""Main analysis of experiment log 204, stages 1-2 (Qwen3-4B; ESCI T1-T7; seeds 0/1/2; final 625-update models; own instruction).
Paths: Q0 Base original text, Q1 original CE, Q2 original CE + soft kappa (lambda 1), Q3 original CE + AnyLoss-like kappa (lambda 1, alpha 10), P0 Base with the frozen APO-CF text, P1 APO-CF -> CE, P2 APO-CF -> CE + soft kappa (T1-T3 only).
Each path at t = 0 and with its own validation threshold (selection 200 queries / 4,047 rows, hard-kappa maximum; the Base threshold is never transferred). Test = frozen test300 (300 queries / 6,078 rows).
Order of aggregation: per-instruction kappa -> equal-weight mean over the instruction set -> mean over seeds (never a pooled confusion matrix).
Bootstrap: paired test-query-cluster, 2,000 resamples, np.random.default_rng(0), ONE resample set for every path / instruction / seed (thresholds and texts fixed; selection, validation-sample and training-seed uncertainty are not included).
    python esci204_main.py --root <dir with cpmr_experiment/ score tree> --out <dir> [--tsets main=2-7,all=1-7,t13=1-3]"""
import argparse
import json
from pathlib import Path

import numpy as np

import esci204_common as C

B = 2000


def cis(v, level):
    lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D, nq, row_q, test_q = C.build_all(a.root)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, nq, size=(B, nq))
    W = np.stack([np.bincount(i, minlength=nq) for i in idx]).astype(float)                    # (B, nq) multiplicities
    for e in D.values():
        for mode in ("t0", "cal"):
            tot = W @ e[mode]["cnt"]
            e[mode]["boot"] = C.kap_vec(tot[:, 0], tot[:, 1], tot[:, 2], tot[:, 3])

    def seeds_of(c):
        return [0] if c in ("Q0", "P0") else C.SEEDS

    def have(c, Ts):
        return all((c, T, s) in D for T in Ts for s in seeds_of(c))

    def agg(c, mode, Ts):
        """-> (point, boot (B,), per-seed points, per-T seed-mean points)"""
        per_seed_pt, per_seed_bt = [], []
        for s in seeds_of(c):
            per_seed_pt.append(np.mean([D[(c, T, s)][mode]["kappa"] for T in Ts]))
            per_seed_bt.append(np.mean([D[(c, T, s)][mode]["boot"] for T in Ts], axis=0))
        perT = [float(np.mean([D[(c, T, s)][mode]["kappa"] for s in seeds_of(c)])) for T in Ts]
        return float(np.mean(per_seed_pt)), np.mean(per_seed_bt, axis=0), per_seed_pt, perT

    def diff(c1, m1, c2, m2, Ts):
        """c1/m1 - c2/m2 over the instruction set Ts: point, boot, per-seed (Base paths have one 'seed'), per-T seed-mean differences."""
        p1, b1, s1, t1 = agg(c1, m1, Ts); p2, b2, s2, t2 = agg(c2, m2, Ts)
        n = max(len(s1), len(s2))
        sd = [s1[i if len(s1) > 1 else 0] - s2[i if len(s2) > 1 else 0] for i in range(n)]
        return p1 - p2, b1 - b2, sd, [x - y for x, y in zip(t1, t2)]

    TSETS = {"T2-T7 (main)": list(range(2, 8)), "T1-T7": list(range(1, 8)), "T1-T3": [1, 2, 3]}
    lines = []
    P = lines.append
    P("ESCI systematic comparison (experiment log 204), stages 1-2; Qwen3-4B; test300 300 queries / 6,078 rows; validation 200 queries / 4,047 rows")
    P("completeness (models available with both test300 and validation scores):")
    for c in C.CONDS_ALL:
        P(f"  {c}: " + " ".join(f"T{T}:{sum((c, T, s) in D for s in seeds_of(c))}/{len(seeds_of(c))}" for T in C.TS))
    results = {}
    # ---------------- per-path tables
    for name, Ts in TSETS.items():
        P(f"\n## instruction set {name}: equal-weight mean of per-instruction kappa, then mean over seeds (SD over seeds in brackets: seed values)")
        P("| path | t=0 kappa | calibrated kappa | cal - t0 [95%] | AUC | acc t0 / cal | Relevant% t0 / cal | E FNR t0 / cal | S/C/I FPR cal |")
        P("|---|---|---|---|---|---|---|---|---|")
        for c in C.CONDS_ALL:
            if not have(c, Ts):
                continue
            p0, b0, s0, _ = agg(c, "t0", Ts); pc, bc, sc, _ = agg(c, "cal", Ts)
            es = [D[(c, T, s)] for T in Ts for s in seeds_of(c)]
            f = lambda k, mode: np.mean([e[mode][k] for e in es])
            fp = {k: np.mean([e["cal"]["fpr"][k] for e in es]) for k in "SCI"}
            lo, hi = cis(bc - b0, 95)
            P(f"| {c} | {p0:.4f}" + (f" [{', '.join(f'{x:.4f}' for x in s0)}]" if len(s0) > 1 else "") + f" | {pc:.4f}" + (f" [{', '.join(f'{x:.4f}' for x in sc)}]" if len(sc) > 1 else "") +
              f" | {pc - p0:+.4f} [{lo:+.4f}, {hi:+.4f}] | {np.mean([e['auc'] for e in es]):.4f} | {f('acc', 't0'):.3f} / {f('acc', 'cal'):.3f} | {100 * f('rel', 't0'):.1f} / {100 * f('rel', 'cal'):.1f} | {f('fnr', 't0'):.3f} / {f('fnr', 'cal'):.3f} | {fp['S']:.3f}/{fp['C']:.3f}/{fp['I']:.3f} |")
        results[name] = {}
    # ---------------- comparisons
    COMP = [
        ("MAIN-1 cal Q2 - cal Q1 (soft kappa extra gain)", "Q2", "cal", "Q1", "cal", "main"),
        ("MAIN-2 cal Q1 - cal Q0 (SFT extra gain)", "Q1", "cal", "Q0", "cal", "main"),
        ("MAIN-3 cal Q3 - cal Q2 (AnyLoss-like vs soft kappa)", "Q3", "cal", "Q2", "cal", "main"),
        ("aux t0 Q2 - Q1", "Q2", "t0", "Q1", "t0", "aux"), ("aux t0 Q1 - Q0", "Q1", "t0", "Q0", "t0", "aux"), ("aux t0 Q3 - Q2", "Q3", "t0", "Q2", "t0", "aux"),
        ("aux Q3 cal - Q1 cal", "Q3", "cal", "Q1", "cal", "aux"), ("aux t0 Q3 - Q1", "Q3", "t0", "Q1", "t0", "aux"),
        ("calibration gain Q1 (cal - t0)", "Q1", "cal", "Q1", "t0", "aux"), ("calibration gain Q2 (cal - t0)", "Q2", "cal", "Q2", "t0", "aux"), ("calibration gain Q3 (cal - t0)", "Q3", "cal", "Q3", "t0", "aux"),
        ("calibration gain Q0 (cal - t0)", "Q0", "cal", "Q0", "t0", "aux"),
        ("PO effect P1 cal - Q1 cal", "P1", "cal", "Q1", "cal", "aux"), ("PO effect P1 t0 - Q1 t0", "P1", "t0", "Q1", "t0", "aux"),
        ("PO effect P2 cal - Q2 cal", "P2", "cal", "Q2", "cal", "aux"), ("PO effect P2 t0 - Q2 t0", "P2", "t0", "Q2", "t0", "aux"),
        ("P2 cal - P1 cal", "P2", "cal", "P1", "cal", "aux"), ("P2 t0 - P1 t0", "P2", "t0", "P1", "t0", "aux"),
        ("PO baseline P1 cal - P0 cal (SFT after PO)", "P1", "cal", "P0", "cal", "aux"), ("P0 cal - Q0 cal (PO text on Base)", "P0", "cal", "Q0", "cal", "aux"),
        ("P0 t0 - Q0 t0", "P0", "t0", "Q0", "t0", "aux"), ("P1 t0 - P0 t0", "P1", "t0", "P0", "t0", "aux"),
    ]
    P("\n## comparisons (seed-mean kappa differences; test-query cluster paired bootstrap, 2,000 resamples, one resample set; thresholds and texts fixed). 95% and, for the three Qwen main comparisons, 98.33% (Bonferroni 3)")
    P("| set | comparison | diff | 95% | 98.33% | seed diffs | per-T diffs (T order) | #T positive |")
    P("|---|---|---|---|---|---|---|---|")
    js = []
    for name, Ts in TSETS.items():
        for lab, c1, m1, c2, m2, kind in COMP:
            if not (have(c1, Ts) and have(c2, Ts)):
                continue
            if kind == "main" and name != "T2-T7 (main)" and name != "T1-T7":
                pass
            pt, bt, sd, td = diff(c1, m1, c2, m2, Ts)
            lo, hi = cis(bt, 95); lo2, hi2 = cis(bt, 98.33)
            tag = lab if kind == "aux" else lab
            P(f"| {name} | {tag} | {pt:+.4f} | [{lo:+.4f}, {hi:+.4f}] | " + (f"[{lo2:+.4f}, {hi2:+.4f}]" if kind == "main" and name == "T2-T7 (main)" else "") +
              f" | {' / '.join(f'{x:+.3f}' for x in sd)} | {' '.join(f'{x:+.3f}' for x in td)} | {sum(x > 0 for x in td)}/{len(td)} |")
            js.append(dict(set=name, comparison=lab, diff=pt, ci95=[lo, hi], ci9833=[lo2, hi2], seed_diffs=[float(x) for x in sd], per_T=[float(x) for x in td]))
    # ---------------- per-instruction detail
    P("\n## per-instruction seed-mean kappa (t=0 / calibrated) and thresholds (seed 0/1/2)")
    P("| path | " + " | ".join(f"T{T}" for T in C.TS) + " |")
    P("|---|" + "---|" * len(C.TS))
    for c in C.CONDS_ALL:
        row = []
        for T in C.TS:
            if not have(c, [T]):
                row.append("-"); continue
            k0 = np.mean([D[(c, T, s)]["t0"]["kappa"] for s in seeds_of(c)]); kc = np.mean([D[(c, T, s)]["cal"]["kappa"] for s in seeds_of(c)])
            th = "/".join(f"{D[(c, T, s)]['thr']:+.2f}" for s in seeds_of(c))
            row.append(f"{k0:.3f} / {kc:.3f} (t {th})")
        P(f"| {c} | " + " | ".join(row) + " |")
    P("\n## validation (selection) hard kappa at the chosen threshold, validation AUC, validation positive rate (seed mean over T1-T7 where complete)")
    for c in C.CONDS_ALL:
        es = [e for (cc, T, s), e in D.items() if cc == c]
        if es:
            P(f"  {c}: val kappa {np.mean([e['val_kappa'] for e in es]):.4f}, val AUC {np.mean([e['val_auc'] for e in es]):.4f}, val Relevant% at thr {100 * np.mean([e['val_rel'] for e in es]):.1f}, val E rate {100 * np.mean([e['val_pos'] for e in es]):.1f}, test AUC {np.mean([e['auc'] for e in es]):.4f}, n models {len(es)}")
    (out / "esci204_main_output.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / "esci204_main_results.json").write_text(json.dumps(js, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
