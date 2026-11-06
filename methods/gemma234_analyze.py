"""Analysis of the FP32 full re-evaluation of the 12 representative Gemma checkpoints (experiment log 234 / 235). Inputs: FP32 per-pair scores (gemma234_fp32_eval.py; ESCI validation + test300, MS MARCO 2019 / 2020), the stored original BF16 (A4) scores, the manifest thresholds.
Three comparisons kept apart: (1) BF16 original: AUC, kappa(t=0), original threshold t_B (ESCI: validation hard-kappa maximum of the BF16 validation scores; MS MARCO: CROSS from the other year's BF16 scores), kappa(t_B); (2) FP32 with the FIXED t_B: AUC, kappa(t=0), kappa(t_B); (3) FP32 with a RE-SELECTED threshold t_F (ESCI: FP32 validation; MS 2019: FP32 2020 all pairs; MS 2020: FP32 2019), same rule and tie handling (0 first, smaller |t|, smaller t), evaluated on the FP32 test scores. AUC from the continuous m.
Per set: score-difference distribution FP32 - BF16, decision disagreements, AUC / kappa changes. Loss contrasts (soft - CE, AnyLoss-like - soft) for T1, T7 and the equal-weight mean of the two; BF16 vs FP32 side by side; precision change of every contrast. Paired query-cluster bootstrap: 10,000 draws, np.random.default_rng(61).integers(0, n_queries, size=(10000, n_queries)) per evaluation set, the same resample for every precision / loss / instruction, thresholds fixed after selection, per-instruction AUC / kappa per draw then averaged; draws with a single class or an undefined kappa anywhere in the set are excluded for all conditions (counted). 95% percentile CIs.
    python gemma234_analyze.py --root . --fp32 cpmr_experiment/gemma234/out --manifest cpmr_experiment/gemma231/sample_manifest.json --mods cpmr_experiment/auc61/mods --out cpmr_experiment/gemma234/res"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--fp32", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--mods", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
t0_ = time.time()
root = Path(a.root)
sys.path.insert(0, a.mods); sys.path.insert(0, str(root / "cpmr_experiment/auc61"))
import esci204_common as QC
from auc61_boot import prep, weighted_auc_numerator

kap, kap_vec, best = QC.kap, QC.kap_vec, QC.best_threshold
man = json.load(open(root / a.manifest, encoding="utf-8"))
CK = {(c["dataset"], c["T"], c["loss"]): c for c in man["checkpoints"]}
B_DRAWS, SEED = 10000, 61
LOSSES = ("CE", "SOFT", "ANY")
SETS = {"ESCI": ["ESCItest"], "MS": ["MS2019", "MS2020"]}
ld = lambda p: json.load(gzip.open(p, "rt", encoding="utf-8"))


def load_fp32(tag, name):
    d = json.load(open(root / a.fp32 / f"fp32_{tag}__{name}.json"))
    pairs = [(p[0], p[1]) for p in d["pairs"]]
    s = np.array(d["scores"], dtype=float)
    return pairs, np.array([p[2] for p in d["pairs"]], dtype=bool), s, s[:, 0] + s[:, 1] - s[:, 2] - s[:, 3], d["meta"]


def load_bf16(ck, name, pairs):
    T = ck["T"]
    if ck["dataset"] == "ESCI":
        sub = "sel" if name == "ESCIval" else "test300"
        r = ld(root / f"cpmr_experiment/esci_gemma/eval/esciGemmaEval_{ck['tag']}_step625__T{T}__{sub}.json.gz")["raw"]
        v = np.array([[r[q][d][0], r[q][d][1], r[q][d][3], r[q][d][4]] for q, d in pairs], dtype=float)
        return v, v[:, 0] + v[:, 1] - v[:, 2] - v[:, 3]
    y = name[2:]
    r = ld(root / f"cpmr_experiment/msm_gemma/eval/msmGemmaEval_{ck['tag']}__T{T}__{y}.json.gz")["run_fp32"]
    m = np.array([r[q][d] for q, d in pairs], dtype=float)
    return None, m


def k_at(m, y, t):
    p = m > t
    return float(kap(int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum())))


def auc(m, y):
    rk = rankdata(m); pp, nn = int(y.sum()), int((~y).sum())
    return float((rk[y].sum() - pp * (pp + 1) / 2) / (pp * nn))


def stats(d):
    d = np.asarray(d, float); ad = np.abs(d)
    return dict(n=len(d), signed_mean=float(d.mean()), mean_abs=float(ad.mean()), median_abs=float(np.median(ad)), p95_abs=float(np.percentile(ad, 95)), max_abs=float(ad.max()))


# ----------------------------------------------------------------- load everything
DATA = {}      # (ds, T, loss, set) -> dict
for (ds, T, loss), ck in sorted(CK.items()):
    sets = (["ESCIval", "ESCItest"] if ds == "ESCI" else ["MS2019", "MS2020"])
    for name in sets:
        pairs, y, s_f, m_f, meta = load_fp32(ck["tag"], name)
        v_b, m_b = load_bf16(ck, name, pairs)
        DATA[(ds, T, loss, name)] = dict(pairs=pairs, y=y, F=m_f, Fs=s_f, B=m_b, Bs=v_b, meta=meta, ck=ck)
n_expected = {"ESCIval": 4047, "ESCItest": 6078, "MS2019": 1102, "MS2020": 1432}
for k, d in DATA.items():
    assert len(d["pairs"]) == n_expected[k[3]], k
    assert np.isfinite(d["F"]).all() and np.isfinite(d["B"]).all(), k
# ----------------------------------------------------------------- thresholds
R = {}
for ds in ("ESCI", "MS"):
    for T in (1, 7):
        for loss in LOSSES:
            ck = CK[(ds, T, loss)]
            if ds == "ESCI":
                v, t_ref = DATA[(ds, T, loss, "ESCIval")], ck["stored_threshold_validation"]
                tB, kvB = best(v["B"], v["y"]); tF, kvF = best(v["F"], v["y"])
                thr = {"ESCItest": dict(tB=tB, tF=tF, val_kappa_B=kvB, val_kappa_F=kvF, tB_manifest=t_ref)}
            else:
                thr = {}
                for name, other in (("MS2019", "MS2020"), ("MS2020", "MS2019")):
                    o = DATA[(ds, T, loss, other)]
                    tB, kvB = best(o["B"], o["y"]); tF, kvF = best(o["F"], o["y"])
                    thr[name] = dict(tB=tB, tF=tF, val_kappa_B=kvB, val_kappa_F=kvF, tB_manifest=ck["stored_threshold_cross"][name[2:]])
            R[(ds, T, loss)] = thr
# ----------------------------------------------------------------- metrics, score differences
out = dict(per_checkpoint=[], contrasts=[], checks=[], flags=[])
M = {}
for (ds, T, loss), thr in sorted(R.items()):
    for name, th in thr.items():
        d = DATA[(ds, T, loss, name)]; y = d["y"]; tB, tF = th["tB"], th["tF"]
        rec = dict(dataset=ds, set=name, T=T, loss=loss, tag=d["ck"]["tag"], n=len(y), n_rel=int(y.sum()), tB=tB, tF=tF, tB_manifest=th["tB_manifest"], tB_reproduced=bool(abs(tB - th["tB_manifest"]) < 1e-9),
                   val_kappa_B=th["val_kappa_B"], val_kappa_F=th["val_kappa_F"], dtypes=d["meta"]["dtypes"], adapter_sha256=d["ck"]["sha256"])
        rec["BF16"] = dict(auc=auc(d["B"], y), kappa_t0=k_at(d["B"], y, 0.0), kappa_tB=k_at(d["B"], y, tB))
        rec["FP32"] = dict(auc=auc(d["F"], y), kappa_t0=k_at(d["F"], y, 0.0), kappa_tB=k_at(d["F"], y, tB), kappa_tF=k_at(d["F"], y, tF))
        rec["diff_m"] = stats(d["F"] - d["B"])
        if d["Bs"] is not None:
            rec["diff_ell1"] = stats(d["Fs"][:, 0] + d["Fs"][:, 1] - d["Bs"][:, 0] - d["Bs"][:, 1]); rec["diff_ell0"] = stats(d["Fs"][:, 2] + d["Fs"][:, 3] - d["Bs"][:, 2] - d["Bs"][:, 3])
        def dec(t):
            pb, pf = d["B"] > t, d["F"] > t
            return dict(disagree=int((pb != pf).sum()), rate=float((pb != pf).mean()), FP_new=int((pf & ~pb & ~y).sum()), TP_new=int((pf & ~pb & y).sum()), FN_new=int((~pf & pb & y).sum()), TN_new=int((~pf & pb & ~y).sum()))
        rec["decisions"] = dict(t0=dec(0.0), tB=dec(tB))
        pbB, pfF = d["B"] > tB, d["F"] > tF
        rec["decisions"]["BF16_tB_vs_FP32_tF"] = dict(disagree=int((pbB != pfF).sum()), rate=float((pbB != pfF).mean()))
        out["per_checkpoint"].append(rec); M[(ds, T, loss, name)] = rec
# ----------------------------------------------------------------- bootstrap
BOOT = {}
for ds in ("ESCI", "MS"):
    for name in SETS[ds]:
        d0 = DATA[(ds, 1, "CE", name)]
        qs = sorted({q for q, _ in d0["pairs"]}); qid = {q: i for i, q in enumerate(qs)}; nq = len(qs)
        rowq = np.array([qid[q] for q, _ in d0["pairs"]])
        y = d0["y"]
        rng = np.random.default_rng(SEED)
        W = np.stack([np.bincount(i, minlength=nq) for i in rng.integers(0, nq, size=(B_DRAWS, nq))]).astype(np.float64)
        wpos, wneg = W @ np.bincount(rowq, weights=y.astype(float), minlength=nq), W @ np.bincount(rowq, weights=(~y).astype(float), minlength=nq)
        valid = (wpos > 0) & (wneg > 0)
        den = wpos * wneg

        def boot_auc(m):
            pp = prep(m); num = np.empty(B_DRAWS)
            for i in range(0, B_DRAWS, 500):
                num[i:i + 500] = weighted_auc_numerator(pp, y, rowq, W[i:i + 500])
            with np.errstate(invalid="ignore", divide="ignore"):
                return num / den

        def boot_kappa(m, t):
            p = m > t
            cnt = np.stack([np.bincount(rowq, weights=arr.astype(float), minlength=nq) for arr in (p & y, p & ~y, ~p & y, ~p & ~y)], axis=1)
            tot = W @ cnt
            return kap_vec(tot[:, 0], tot[:, 1], tot[:, 2], tot[:, 3])
        for T in (1, 7):
            for loss in LOSSES:
                d = DATA[(ds, T, loss, name)]; th = R[(ds, T, loss)][name]
                BOOT[(ds, T, loss, name)] = dict(B_auc=boot_auc(d["B"]), F_auc=boot_auc(d["F"]), B_k0=boot_kappa(d["B"], 0.0), B_ksel=boot_kappa(d["B"], th["tB"]), F_k0=boot_kappa(d["F"], 0.0), F_ksel=boot_kappa(d["F"], th["tF"]), F_kfix=boot_kappa(d["F"], th["tB"]))
        allv = np.stack([v for k, bb in BOOT.items() if k[3] == name for v in bb.values()])
        mask = valid & np.isfinite(allv).all(axis=0)
        for k in [k for k in BOOT if k[3] == name]:
            BOOT[k]["_mask"] = mask
        out["checks"].append(dict(name=f"bootstrap draws excluded (single class or undefined kappa) {name}", excluded=int((~mask).sum()), of=B_DRAWS))
# ----------------------------------------------------------------- contrasts
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
METS = (("dAUC", "B_auc", "F_auc", "auc"), ("dKappa_t0", "B_k0", "F_k0", "kappa_t0"), ("dKappa_sel", "B_ksel", "F_ksel", None), ("dKappa_fixedB", "B_ksel", "F_kfix", None))
for ds in ("ESCI", "MS"):
    for name in SETS[ds]:
        for lab, hi, lo in (("soft - CE", "SOFT", "CE"), ("AnyLoss-like - soft", "ANY", "SOFT")):
            for Tsel in ("T1", "T7", "mean"):
                Ts = (1, 7) if Tsel == "mean" else (int(Tsel[1:]),)
                mask = BOOT[(ds, 1, "CE", name)]["_mask"]
                rec = dict(dataset=ds, set=name, contrast=lab, instruction=Tsel, metrics={})
                def pt(T, loss, prec, key):
                    m = M[(ds, T, loss, name)]
                    if key == "auc": return m[prec]["auc"]
                    if key == "kappa_t0": return m[prec]["kappa_t0"]
                    if key == "sel": return m["BF16"]["kappa_tB"] if prec == "BF16" else m["FP32"]["kappa_tF"]
                    if key == "fixedB": return m["FP32"]["kappa_tB"]
                def bt(T, loss, k):
                    return BOOT[(ds, T, loss, name)][k][mask]
                for met, kb, kf, pk in METS:
                    if met == "dAUC": B_pt = np.mean([pt(T, hi, "BF16", "auc") - pt(T, lo, "BF16", "auc") for T in Ts]); F_pt = np.mean([pt(T, hi, "FP32", "auc") - pt(T, lo, "FP32", "auc") for T in Ts])
                    elif met == "dKappa_t0": B_pt = np.mean([pt(T, hi, "BF16", "kappa_t0") - pt(T, lo, "BF16", "kappa_t0") for T in Ts]); F_pt = np.mean([pt(T, hi, "FP32", "kappa_t0") - pt(T, lo, "FP32", "kappa_t0") for T in Ts])
                    elif met == "dKappa_sel": B_pt = np.mean([pt(T, hi, "BF16", "sel") - pt(T, lo, "BF16", "sel") for T in Ts]); F_pt = np.mean([pt(T, hi, "FP32", "sel") - pt(T, lo, "FP32", "sel") for T in Ts])
                    else: B_pt = np.mean([pt(T, hi, "BF16", "sel") - pt(T, lo, "BF16", "sel") for T in Ts]); F_pt = np.mean([pt(T, hi, "FP32", "fixedB") - pt(T, lo, "FP32", "fixedB") for T in Ts])
                    Bv = np.mean([bt(T, hi, kb) - bt(T, lo, kb) for T in Ts], axis=0); Fv = np.mean([bt(T, hi, kf) - bt(T, lo, kf) for T in Ts], axis=0)
                    rec["metrics"][met] = dict(bf16=float(B_pt), fp32=float(F_pt), change=float(F_pt - B_pt), ci_bf16=ci(Bv), ci_fp32=ci(Fv), ci_change=ci(Fv - Bv), sign_change=bool(np.sign(B_pt) != np.sign(F_pt)))
                out["contrasts"].append(rec)
                ch = rec["metrics"]
                if abs(ch["dAUC"]["change"]) > 0.001 or abs(ch["dKappa_t0"]["change"]) > 0.005 or abs(ch["dKappa_sel"]["change"]) > 0.005:
                    out["flags"].append(dict(where=f"{name} {lab} {Tsel}", dAUC=ch["dAUC"]["change"], dKappa_t0=ch["dKappa_t0"]["change"], dKappa_sel=ch["dKappa_sel"]["change"]))
out["checks"].append(dict(name="original thresholds t_B re-derived from the stored BF16 scores equal the manifest thresholds", passed=all(r["tB_reproduced"] for r in out["per_checkpoint"]), n=len(out["per_checkpoint"])))
out["meta"] = dict(B=B_DRAWS, seed=SEED, seconds=time.time() - t0_)
Path(root / a.out).mkdir(parents=True, exist_ok=True)
json.dump(out, open(root / a.out / "gemma234_results.json", "w"), indent=1)
# ----------------------------------------------------------------- markdown
o = []; P = o.append
P("# Gemma 12-checkpoint FP32 full re-evaluation (234)\n")
P("## 1. Per checkpoint and evaluation set (BF16 original / FP32 fixed t_B / FP32 re-selected t_F)\n")
P("| set | T | loss | t_B (reproduced) | t_F | AUC BF16 → FP32 | κ(t0) BF16 → FP32 | κ(t_B) BF16 → FP32 | κ_F(t_F) | m diff mean/p95/max abs | disagree t0 / t_B |\n|---|---|---|---|---|---|---|---|---|---|---|")
for r in out["per_checkpoint"]:
    P(f"| {r['set']} | T{r['T']} | {r['loss']} | {r['tB']:+.4f} ({r['tB_reproduced']}) | {r['tF']:+.4f} | {r['BF16']['auc']:.4f} → {r['FP32']['auc']:.4f} | {r['BF16']['kappa_t0']:.4f} → {r['FP32']['kappa_t0']:.4f} | {r['BF16']['kappa_tB']:.4f} → {r['FP32']['kappa_tB']:.4f} | {r['FP32']['kappa_tF']:.4f} | {r['diff_m']['mean_abs']:.3f} / {r['diff_m']['p95_abs']:.3f} / {r['diff_m']['max_abs']:.3f} | {r['decisions']['t0']['disagree']} ({100 * r['decisions']['t0']['rate']:.2f}%) / {r['decisions']['tB']['disagree']} ({100 * r['decisions']['tB']['rate']:.2f}%) |")
P("\n## 2. Loss contrasts: BF16 original vs FP32 (point [95% CI]); change = FP32 − BF16\n")
P("| set | contrast | instr. | metric | BF16 | FP32 | change [95% CI] | sign change |\n|---|---|---|---|---|---|---|---|")
for c in out["contrasts"]:
    for met, v in c["metrics"].items():
        P(f"| {c['set']} | {c['contrast']} | {c['instruction']} | {met} | {v['bf16']:+.4f} [{v['ci_bf16'][0]:+.4f}, {v['ci_bf16'][1]:+.4f}] | {v['fp32']:+.4f} [{v['ci_fp32'][0]:+.4f}, {v['ci_fp32'][1]:+.4f}] | {v['change']:+.4f} [{v['ci_change'][0]:+.4f}, {v['ci_change'][1]:+.4f}] | {v['sign_change']} |")
P("\n## 3. Follow-up flags (change > ΔAUC 0.001 or Δκ 0.005; expansion criteria, not equivalence tests)\n")
for f in out["flags"]:
    P("- " + json.dumps(f))
P("\n## 4. Checks\n")
for c in out["checks"]:
    P("- " + json.dumps(c))
open(root / a.out / "gemma234_results.md", "w", encoding="utf-8").write("\n".join(o))
print("written", len(out["per_checkpoint"]), "checkpoint-set records,", len(out["contrasts"]), "contrast records,", len(out["flags"]), "flags")
