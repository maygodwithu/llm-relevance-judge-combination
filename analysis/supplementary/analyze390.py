"""Gemma-3-4B-it MS MARCO T1 label-length normalisation on the label-wise re-scored existing checkpoints (Base, CE seeds 0-2, CE+soft kappa seeds 0-2).
ell1 = lp_label(R) + lp_eot(R), ell0 = lp_label(I) + lp_eot(I), L1 = n_label_tokens(R) + 1, L0 = n_label_tokens(I) + 1 (stored values); m_sum = ell1 - ell0; m_norm = ell1 / L1 - ell0 / L0.
Per run and score type ONE threshold from val200 (hard kappa maximiser, submitted candidate partitions and tie rule) is applied unchanged to TREC-DL 2019 and 2020; kappa(0), kappa(t*), AUC, Relevant rate, share of decisions changed by the normalisation (t=0 and t*) per year;
seed means / sample SD; soft - CE differences; reproduction of the stored sum margins (gate390.json) and of the submitted appendix T1 values first. No test-based choice.
    python analyze390.py --root <project root> --label_dir <dir with msmGemmaLabel390_*> --out <dir>"""
import argparse, csv, gzip, hashlib, json, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--label_dir", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
R = Path(a.root); LD = R / a.label_dir; OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R / "analysis"))
import esci204_common as C  # noqa: E402

TAGS = [("Base", "Base", 0)] + [(f"gmq_B_T1_s{s}", "CE", s) for s in (0, 1, 2)] + [(f"gmq_E_T1_s{s}", "CE+soft kappa", s) for s in (0, 1, 2)]
# submitted appendix, Gemma T1 (seed means, default / selected kappa, 3 decimals): reproduction check only
APP = {("Base", 2019): (0.370, 0.354), ("CE", 2019): (0.319, 0.370), ("CE+soft kappa", 2019): (0.358, 0.417),
       ("Base", 2020): (0.208, 0.199), ("CE", 2020): (0.185, 0.243), ("CE+soft kappa", 2020): (0.221, 0.304)}


def load(tag, s):
    p = LD / f"msmGemmaLabel390_{tag}__T1__{s}.json.gz"
    d = json.load(gzip.open(p, "rt", encoding="utf-8")); raw, gr = d["raw"], d["grades"]
    ks = [(q, x) for q in sorted(raw) for x in sorted(raw[q])]
    v = np.array([raw[q][x] for q, x in ks], dtype=float)
    e1, e0 = v[:, 0] + v[:, 1], v[:, 2] + v[:, 3]
    L1, L0 = v[:, 4] + 1, v[:, 5] + 1
    return dict(msum=e1 - e0, mnorm=e1 / L1 - e0 / L0, y=np.array([gr[q][x] >= 2 for q, x in ks]), L=(sorted(set(L1)), sorted(set(L0))), sha=hashlib.sha256(p.read_bytes()).hexdigest())


def kat(m, y, t):
    p = m > t
    return float(C.kap(int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum())))


rows = []
for tag, obj, s in TAGS:
    V = load(tag, "val200"); T = {y: load(tag, y) for y in ("2019", "2020")}
    for d in [V] + list(T.values()):
        assert d["L"] == ([2.0], [3.0]), d["L"]
    thr = {k: C.best_threshold(V[k], V["y"]) for k in ("msum", "mnorm")}
    for y, d in T.items():
        r = dict(objective=obj, seed=s, tag=tag, year=int(y), L1=2, L0=3, val_sha256=V["sha"], test_sha256=d["sha"])
        for k, nm in (("msum", "sum"), ("mnorm", "norm")):
            t, vk = thr[k]
            r.update({f"val200_kappa_t0_{nm}": kat(V[k], V["y"], 0.0), f"val200_kappa_tstar_{nm}": vk, f"tstar_{nm}": t, f"k0_{nm}": kat(d[k], d["y"], 0.0), f"kstar_{nm}": kat(d[k], d["y"], t),
                      f"auc_{nm}": float(C.auc(d[k], d["y"])), f"rel_t0_{nm}": float((d[k] > 0).mean()), f"rel_tstar_{nm}": float((d[k] > t).mean())})
        r["decisions_changed_t0"] = float(((d["msum"] > 0) != (d["mnorm"] > 0)).mean())
        r["decisions_changed_tstar"] = float(((d["msum"] > thr["msum"][0]) != (d["mnorm"] > thr["mnorm"][0])).mean())
        rows.append(r)
with open(OUT / "msm_gemma_norm390_per_run.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
summ, repro, diffs = [], [], []
for obj in ("Base", "CE", "CE+soft kappa"):
    for y in (2019, 2020):
        rs = [r for r in rows if r["objective"] == obj and r["year"] == y]
        d = dict(objective=obj, year=y, n=len(rs))
        for nm in ("sum", "norm"):
            for k in ("k0", "kstar", "auc", "rel_t0", "rel_tstar", "val200_kappa_tstar"):
                v = [r[f"{k}_{nm}"] for r in rs]
                d[f"{k}_{nm}_mean"] = float(np.mean(v)); d[f"{k}_{nm}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
        for k in ("decisions_changed_t0", "decisions_changed_tstar"):
            d[k + "_mean"] = float(np.mean([r[k] for r in rs])); d[k + "_max"] = float(max(r[k] for r in rs))
        summ.append(d)
        a0, a1 = APP[(obj, y)]
        repro.append(dict(objective=obj, year=y, k0=round(d["k0_sum_mean"], 3), k0_table=a0, kstar=round(d["kstar_sum_mean"], 3), kstar_table=a1,
                          match=abs(round(d["k0_sum_mean"], 3) - a0) < 1.5e-3 and abs(round(d["kstar_sum_mean"], 3) - a1) < 1.5e-3))
for y in (2019, 2020):
    for nm in ("sum", "norm"):
        for k, lab in (("k0", "t=0"), ("kstar", "val200 t*")):
            ce = {r["seed"]: r[f"{k}_{nm}"] for r in rows if r["objective"] == "CE" and r["year"] == y}
            so = {r["seed"]: r[f"{k}_{nm}"] for r in rows if r["objective"] == "CE+soft kappa" and r["year"] == y}
            sd = [so[s] - ce[s] for s in (0, 1, 2)]
            diffs.append(dict(year=y, score=nm, threshold=lab, mean_diff=float(np.mean(sd)), seed_diffs=sd, sd=float(np.std(sd, ddof=1)), positive_seeds=int(sum(x > 0 for x in sd))))
gate = json.load(open(LD / "gate390.json"))
res = dict(summary=summ, soft_minus_ce=diffs, reproduction_vs_submitted_appendix_T1=repro, reproduction_all_match=all(x["match"] for x in repro),
           gate=dict(n_checks=len(gate), all_passed=all(g["passed"] for g in gate), max_abs_diff=max(g["max_abs_diff"] for g in gate), mean_abs_diff=float(np.mean([g["mean_abs_diff"] for g in gate])),
                     decision_disagreements_t0=int(sum(g["decision_disagreements_t0"] for g in gate)), total_pairs=int(sum(g["n_pairs"] for g in gate))))
(OUT / "msm_gemma_norm390_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
L = ["| objective | year | score | test k(0) | test k(t*) | AUC | Relevant % t=0 / t* | decisions changed t=0 / t* (mean, max) |", "|---|---|---|---|---|---|---|---|"]
for d in summ:
    for nm in ("sum", "norm"):
        f = lambda k: f"{d[f'{k}_{nm}_mean']:.4f}" + (f" ± {d[f'{k}_{nm}_sd']:.4f}" if d["n"] > 1 else "")
        extra = (f"{100 * d['decisions_changed_t0_mean']:.1f}% / {100 * d['decisions_changed_tstar_mean']:.1f}% (max {100 * d['decisions_changed_t0_max']:.1f}% / {100 * d['decisions_changed_tstar_max']:.1f}%)") if nm == "norm" else ""
        L.append(f"| {d['objective']} | {d['year']} | {nm} | {f('k0')} | {f('kstar')} | {f('auc')} | {100 * d[f'rel_t0_{nm}_mean']:.1f} / {100 * d[f'rel_tstar_{nm}_mean']:.1f} | {extra} |")
L += ["", "| year | score | threshold | soft - CE mean | seed diffs | positive seeds |", "|---|---|---|---|---|---|"]
L += [f"| {x['year']} | {x['score']} | {x['threshold']} | {x['mean_diff']:+.4f} | {', '.join(f'{v:+.4f}' for v in x['seed_diffs'])} | {x['positive_seeds']}/3 |" for x in diffs]
L += ["", f"gate: {res['gate']}", "reproduction of the submitted appendix T1 (sum score, seed means): " + ("all 6 cells match" if res["reproduction_all_match"] else "MISMATCH")]
L += [f"- {x['objective']} {x['year']}: {x['k0']} / {x['kstar']} vs table {x['k0_table']} / {x['kstar_table']} -> {x['match']}" for x in repro]
(OUT / "msm_gemma_norm390_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
