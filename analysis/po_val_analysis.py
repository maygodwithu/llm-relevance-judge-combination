"""MS MARCO separate-validation threshold analysis (user choice A0): the threshold of every checkpoint is the hard-kappa maximiser over the 400 val200 pairs (the frozen PO selection set; ties -> 0, then smaller |t|, then smaller t; decision m > t) and the SAME threshold is applied to TREC-DL 2019 and 2020
(never re-selected per year; the test data play no role in the selection). Stored raw FP32 scores only, CPU, no new test inference. Original: Qwen3-4B, Llama-3.1-8B-Instruct, Gemma-3-4B-it x T1-T7 x Base / CE / CE+soft kappa / CE+AnyLoss-style kappa (seeds 0-2; Base seed-less);
PO (Qwen T1-T7, Llama: the changed T2/T3, the other five are aliases of Original): Base / CE / CE+soft kappa. Existing CROSS values are reproduced and kept as a separate reference.
Aggregation: per-instruction kappa -> equal-weight mean over T1-T7 -> mean over seeds (per-seed kappa from the full judged confusion, never a mean of per-query kappa). Paired test-query-cluster bootstrap: B = 10,000, default_rng(int(year)), float32 multiplicities, one draw set per year for every cell (thresholds fixed).
Validation variation: 500 resamples (default_rng(63)) of the 200 val queries, thresholds of every checkpoint re-selected on the weighted validation scores and applied to both years; the same draws for every cell.
    python po_val_analysis.py --root . --kappa cpmr_experiment/auc61/out/kappa.pkl --out cpmr_experiment/msm_val/summary"""
import argparse
import glob
import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--kappa", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mods", default="analysis", help="directory with msm_t7_analysis.py (the CROSS reference implementation and the shared kappa / threshold helpers)")
a = ap.parse_args()
R = Path(a.root)
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R / a.mods))
import msm_t7_analysis as MQ

kap_vec, best_threshold, auc_rank = MQ.kap_vec, MQ.best_threshold, MQ.auc
YEARS = ("2019", "2020")
OTHER = {"2019": "2020", "2020": "2019"}
SEEDS = (0, 1, 2)
NB, R63 = 10000, 500
MODES = ("t0", "sel", "cross")
man = json.load(open(R / "cpmr_experiment/msm_po/freeze/po_selection_manifest.json"))
K = pickle.load(open(a.kappa, "rb"))
idx = {}
for f in glob.glob(str(R / "cpmr_experiment/**/msm*Eval_*.json.gz"), recursive=True):
    idx.setdefault(os.path.basename(f), f)
ld = lambda p: json.load(gzip.open(p, "rt", encoding="utf-8"))
FAM = {"qwen": ("msmEval_", "msmValEval_", "MS-Qwen"), "llama": ("msmLlamaEval_", "msmValLlamaEval_", "MS-Llama"), "gemma": ("msmGemmaEval_", "msmValGemmaEval_", "MS-Gemma")}
LETTER = {"CE": "B", "SOFT": "E", "ANY": "M"}
CN = {"BASE": "Base", "CE": "CE", "SOFT": "CE+soft kappa", "ANY": "CE+AnyLoss-style kappa"}


def changed(fam, T):
    return fam != "gemma" and bool(man["models"][fam][f"T{T}"]["changed"])


def spec(fam, var, cond, T, s):
    alias = var == "PO" and not changed(fam, T)
    orig = var == "ORIG" or alias
    instr = f"T{T}" if orig else f"T{T}_APOCF"
    if cond == "BASE":
        return ("Base" if orig else f"BaseAPO_T{T}"), instr, alias
    l = LETTER[cond]
    if orig:
        if fam == "qwen":
            tag = f"msq_M_T{T}_s{s}" if cond == "ANY" else (f"msm_{l}_s{s}" if T == 1 else f"msq_{l}_T{T}_s{s}")
        elif fam == "llama":
            tag = f"mlq_{l}_T{T}_s{s}"
        else:
            tag = f"gmq_{l}_T{T}_s{s}"
    else:
        tag = f"{'msqpo' if fam == 'qwen' else 'mlpo'}_{l}_T{T}_s{s}"
    return tag, instr, alias


def stat_of(tot, st):
    tp, fp, fn, tn = (tot[..., i] for i in range(4))
    n = tp + fp + fn + tn
    with np.errstate(all="ignore"):
        if st == "kappa":
            return kap_vec(tp, fp, fn, tn)
        if st == "acc":
            return (tp + tn) / n
        if st == "pos":
            return (tp + fp) / n
        if st == "fnr":
            return fn / (tp + fn)
        if st == "fpr":
            return fp / (fp + tn)
    raise KeyError(st)


class Val:
    def __init__(self, ks, m, y, qid):
        self.order = np.argsort(m, kind="mergesort")
        self.ms = m[self.order]; self.ys = y[self.order]
        self.rq = np.array([qid[q] for q, _ in ks])[self.order]
        brk = np.concatenate([[True], self.ms[1:] != self.ms[:-1]])
        self.gid = np.cumsum(brk) - 1
        self.G = int(self.gid[-1]) + 1
        self.u = self.ms[brk]

    def best(self, w_q):
        w = w_q[self.rq]
        pos = np.bincount(self.gid, weights=w * self.ys, minlength=self.G)
        neg = np.bincount(self.gid, weights=w * (~self.ys), minlength=self.G)
        keep = (pos + neg) > 0
        if not keep.any():
            return float("nan")
        u = self.u[keep]; pos = pos[keep]; neg = neg[keep]
        cp, cn = np.cumsum(pos), np.cumsum(neg)
        P_, N_ = cp[-1], cn[-1]
        cpz = np.concatenate([[0.0], cp]); cnz = np.concatenate([[0.0], cn])
        tvals = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]])
        k = kap_vec(P_ - cpz, N_ - cnz, cpz, cnz)
        g0 = int(np.searchsorted(u, 0.0, side="right"))
        t_all = np.concatenate([tvals, [0.0]]); k_all = np.concatenate([k, [k[g0]]])
        ok = np.isfinite(k_all)
        if not ok.any():
            return float("nan")
        t_all, k_all = t_all[ok], k_all[ok]
        key = np.lexsort((t_all, np.abs(t_all), (t_all != 0.0).astype(int), -np.round(k_all, 12)))
        return float(t_all[key[0]])


class Test:
    def __init__(self, m, y):
        self.ps, self.ns = np.sort(m[y]), np.sort(m[~y])
        self.P, self.N = len(self.ps), len(self.ns)

    def kappa(self, ts):
        ts = np.asarray(ts, float)
        tp = self.P - np.searchsorted(self.ps, ts, side="right")
        fp = self.N - np.searchsorted(self.ns, ts, side="right")
        return kap_vec(tp, fp, self.P - tp, self.N - fp)


pct = lambda x, p=(2.5, 97.5): [float(np.nanpercentile(x, p[0])), float(np.nanpercentile(x, p[1]))]
t_start = time.time()
RES = dict(meta={}, checks=[], fam={})
CSV = ["model,variant,cond,T,seed,alias_of_original,year,threshold_val200,val_kappa,val_auc,thr_cross,kappa_t0,kappa_val,kappa_cross,auc,pos_rate_t0,pos_rate_val,fnr_val,fpr_val,tp_val,fp_val,fn_val,tn_val"]
NPZ = {}

for fam in ("qwen", "llama", "gemma"):
    t_m = time.time()
    pre, vpre, sid = FAM[fam]
    vars_ = ("ORIG",) if fam == "gemma" else ("ORIG", "PO")
    D, ref = {}, {}
    vref = {}
    for var in vars_:
        for cond in ("BASE", "CE", "SOFT", "ANY"):
            if cond == "ANY" and var == "PO":
                continue
            for T in range(1, 8):
                for s in ((0,) if cond == "BASE" else SEEDS):
                    tag, instr, alias = spec(fam, var, cond, T, s)
                    if alias:
                        D[(var, cond, T, s)] = D[("ORIG", cond, T, s)]
                        continue
                    e = {"alias": False}
                    for y in YEARS:
                        f = idx[f"{pre}{tag}__{instr}__{y}.json.gz"]
                        r = ld(f); run, gr = r["run_fp32"], r["grades"]
                        ks = [(q, d) for q in sorted(run) for d in sorted(run[q])]
                        if y not in ref:
                            qs = sorted({q for q, _ in ks})
                            ref[y] = dict(ks=ks, nq=len(qs), rowq=np.array([{q: i for i, q in enumerate(qs)}[q] for q, _ in ks]))
                        assert ks == ref[y]["ks"]
                        e[y] = dict(m=np.array([run[q][d] for q, d in ks]), y=np.array([gr[q][d] >= 2 for q, d in ks]))
                        assert np.isfinite(e[y]["m"]).all()
                    vf = idx_v = R / "cpmr_experiment/msm_val/eval" / f"{vpre}{tag}__{instr}__val200.json.gz"
                    vr = ld(vf); vrun, vgr = vr["run_fp32"], vr["grades"]
                    vks = [(q, d) for q in sorted(vrun) for d in sorted(vrun[q])]
                    if "ks" not in vref:
                        vref["ks"] = vks
                    assert vks == vref["ks"]
                    e["val"] = dict(m=np.array([vrun[q][d] for q, d in vks]), y=np.array([vgr[q][d] >= 2 for q, d in vks]), ks=vks)
                    assert np.isfinite(e["val"]["m"]).all()
                    assert vr["meta"]["val_pairs_sha256"].startswith("6f2aae0d163d7908")
                    thr_v, vk = best_threshold(e["val"]["m"], e["val"]["y"])
                    e["thr_val"], e["val_kappa"] = thr_v, vk
                    e["val_auc"] = auc_rank(e["val"]["m"], e["val"]["y"])
                    for y in YEARS:
                        thr_c, _ = best_threshold(e[OTHER[y]]["m"], e[OTHER[y]]["y"])
                        e[y]["thr_cross"] = thr_c
                        rowq, nq = ref[y]["rowq"], ref[y]["nq"]
                        for mode, t in (("t0", 0.0), ("sel", thr_v), ("cross", thr_c)):
                            pr = e[y]["m"] > t; yy = e[y]["y"]
                            cnt = np.stack([np.bincount(rowq, weights=arr.astype(float), minlength=nq) for arr in (pr & yy, pr & ~yy, ~pr & yy, ~pr & ~yy)], axis=1)
                            e[y][mode] = dict(cnt=cnt, tot=cnt.sum(0))
                        e[y]["auc"] = auc_rank(e[y]["m"], e[y]["y"])
                    D[(var, cond, T, s)] = e
    nq = {y: ref[y]["nq"] for y in YEARS}
    W = {}
    for y in YEARS:
        rng = np.random.default_rng(int(y))
        W[y] = np.stack([np.bincount(i, minlength=nq[y]) for i in rng.integers(0, nq[y], size=(NB, nq[y]))]).astype(np.float32).astype(np.float64)
    uniq = {id(e): e for e in D.values()}
    for e in uniq.values():
        for y in YEARS:
            for mode in ("t0", "sel"):
                e[y][mode]["tot_b"] = W[y] @ e[y][mode]["cnt"]
    print(fam, "loaded", len(uniq), "unique cells", f"{time.time() - t_m:.0f}s", flush=True)
    # ---- csv + reproduction of Original t=0 / CROSS
    dmax = 0.0; n = 0
    for (var, cond, T, s), e in sorted(D.items()):
        for y in YEARS:
            k0, kv, kc = (float(stat_of(e[y][m]["tot"], "kappa")) for m in ("t0", "sel", "cross"))
            ts = e[y]["sel"]["tot"]
            CSV.append(f"{fam},{var},{cond},{T},{s},{int(var == 'PO' and not changed(fam, T))},{y},{e['thr_val']:.6f},{e['val_kappa']:.6f},{e['val_auc']:.6f},{e[y]['thr_cross']:.6f},{k0:.6f},{kv:.6f},{kc:.6f},{e[y]['auc']:.6f},"
                       f"{float(stat_of(e[y]['t0']['tot'], 'pos')):.6f},{float(stat_of(ts, 'pos')):.6f},{float(stat_of(ts, 'fnr')):.6f},{float(stat_of(ts, 'fpr')):.6f},{int(ts[0])},{int(ts[1])},{int(ts[2])},{int(ts[3])}")
            if var == "ORIG":
                kk0, kkf, kthr = K[(f"{sid}-{y}", cond, T, s)]
                dmax = max(dmax, abs(kk0 - k0), abs(kkf - kc), abs(kthr - e[y]["thr_cross"])); n += 1
    RES["checks"].append(dict(model=fam, check="Original cells: kappa(t=0), kappa(CROSS), CROSS threshold == kappa.pkl of the AUC re-analysis (230; reproduces reports 213 / 216 / 228)", n_cells=n, max_abs_diff=dmax))
    assert dmax < 1e-9, (fam, dmax)
    cache = {}

    def pt(e, y, mode, st):
        return e[y]["auc"] if st == "auc" else float(stat_of(e[y][mode]["tot"], st))

    def bv(e, y, mode, st):
        k = (id(e), y, mode, st)
        if k not in cache:
            cache[k] = stat_of(e[y][mode]["tot_b"], st)
        return cache[k]

    def agg(var, cond, y, mode, st, Ts, boot=True):
        seeds = (0,) if cond == "BASE" else SEEDS
        ps, bs = [], []
        for s in seeds:
            es = [D[(var, cond, T, s)] for T in Ts]
            ps.append(float(np.mean([pt(e, y, mode, st) for e in es])))
            if boot and mode != "cross":
                bs.append(np.mean([bv(e, y, mode, st) for e in es], axis=0))
        tp_ = [float(np.mean([pt(D[(var, cond, T, s)], y, mode, st) for s in seeds])) for T in Ts]
        return float(np.mean(ps)), (np.mean(bs, axis=0) if bs else None), ps, tp_

    def diff(vA, cA, vB, cB, y, mode, st, Ts):
        x1, b1, s1, t1 = agg(vA, cA, y, mode, st, Ts); x2, b2, s2, t2 = agg(vB, cB, y, mode, st, Ts)
        n_ = max(len(s1), len(s2))
        return x1 - x2, b1 - b2, [s1[i if len(s1) > 1 else 0] - s2[i if len(s2) > 1 else 0] for i in range(n_)], [u - v for u, v in zip(t1, t2)]

    Ts7 = list(range(1, 8))
    F = RES["fam"].setdefault(fam, {})
    # ---- level table
    lv = []
    for var in vars_:
        for y in YEARS:
            for cond in ("BASE", "CE", "SOFT", "ANY"):
                if cond == "ANY" and var == "PO":
                    continue
                r = dict(variant=var, year=y, cond=cond)
                for mode in MODES:
                    x, _, sd_, _ = agg(var, cond, y, mode, "kappa", Ts7, boot=False)
                    r[mode] = dict(kappa=x, seeds=sd_)
                r["auc"] = agg(var, cond, y, "t0", "auc", Ts7, boot=False)[0]
                for st in ("pos", "fnr", "fpr", "acc"):
                    r[st] = {m: agg(var, cond, y, m, st, Ts7, boot=False)[0] for m in ("t0", "sel")}
                es = [D[(var, cond, T, s)] for T in Ts7 for s in ((0,) if cond == "BASE" else SEEDS)]
                r["thr_val_mean"] = float(np.mean([e["thr_val"] for e in es])); r["thr_val_sd"] = float(np.std([e["thr_val"] for e in es])); r["val_kappa_mean"] = float(np.mean([e["val_kappa"] for e in es]))
                lv.append(r)
    F["levels"] = lv
    # ---- Original contrasts (K12 family: 3 contrasts x 2 thresholds x 2 years) and PO - Original
    ct = []
    for y in YEARS:
        for mode in ("t0", "sel"):
            for nm, (cA, cB) in {"CE-BASE": ("CE", "BASE"), "SOFT-CE": ("SOFT", "CE"), "ANY-SOFT": ("ANY", "SOFT")}.items():
                d, bd, sd_, td = diff("ORIG", cA, "ORIG", cB, y, mode, "kappa", Ts7)
                ct.append(dict(contrast=nm, year=y, mode=mode, diff=d, ci95=pct(bd), ci_k12=pct(bd, (100 * 0.05 / 24, 100 - 100 * 0.05 / 24)), seed_diffs=sd_, instr_diffs=td, n_pos_instr=int(sum(x > 0 for x in td))))
                NPZ[f"{fam}|{nm}|{y}|{mode}"] = bd
    F["contrasts"] = ct
    if fam != "gemma":
        po = []
        for y in YEARS:
            for mode in ("t0", "sel"):
                for cond in ("BASE", "CE", "SOFT"):
                    d, bd, sd_, td = diff("PO", cond, "ORIG", cond, y, mode, "kappa", Ts7)
                    po.append(dict(cond=cond, year=y, mode=mode, diff=d, ci95=pct(bd), seed_diffs=sd_, instr_diffs=td, share_pos=float(np.mean(bd > 0))))
                    NPZ[f"{fam}|PO-ORIG|{cond}|{y}|{mode}"] = bd
                for nm, (cA, cB) in {"SOFT-CE": ("SOFT", "CE"), "CE-BASE": ("CE", "BASE")}.items():
                    dp = diff("PO", cA, "PO", cB, y, mode, "kappa", Ts7); do = diff("ORIG", cA, "ORIG", cB, y, mode, "kappa", Ts7)
                    po.append(dict(cond=nm + " (PO) vs Original", year=y, mode=mode, po=dp[0], orig=do[0], diff=dp[0] - do[0], ci95=pct(dp[1] - do[1]), ci95_po_only=pct(dp[1]), ci95_orig_only=pct(do[1])))
        F["po_minus_orig"] = po
    # ---- per-instruction (seed mean), PO and Original, selected threshold
    F["per_instruction"] = [dict(variant=var, cond=cond, year=y, T=T, t0=agg(var, cond, y, "t0", "kappa", [T], boot=False)[0], sel=agg(var, cond, y, "sel", "kappa", [T], boot=False)[0], cross=agg(var, cond, y, "cross", "kappa", [T], boot=False)[0])
                            for var in vars_ for cond in ("BASE", "CE", "SOFT", "ANY") if not (cond == "ANY" and var == "PO") for y in YEARS for T in Ts7]
    # ---- validation-query resampling (500, default_rng(63)); one draw set for every cell and both years
    vqs = sorted({q for q, _ in vref["ks"]}); vq = {q: i for i, q in enumerate(vqs)}
    rng = np.random.default_rng(63)
    W63 = np.stack([np.bincount(i, minlength=len(vqs)) for i in rng.integers(0, len(vqs), size=(R63, len(vqs)))]).astype(float)
    info = {}
    for e in uniq.values():
        va = Val(vref["ks"], e["val"]["m"], e["val"]["y"], vq)
        ts = np.array([va.best(W63[b]) for b in range(R63)])
        assert abs(va.best(np.ones(len(vqs))) - e["thr_val"]) < 1e-9
        info[id(e)] = dict(ts=ts, te={y: Test(e[y]["m"], e[y]["y"]) for y in YEARS})
    bad = np.zeros(R63, bool)
    for v in info.values():
        bad |= ~np.isfinite(v["ts"])
    vi = np.where(~bad)[0]
    for e in uniq.values():
        v = info[id(e)]
        v["kb"] = {y: np.where(np.isin(np.arange(R63), vi), 0.0, np.nan) for y in YEARS}
        for y in YEARS:
            kb = np.full(R63, np.nan); kb[vi] = v["te"][y].kappa(v["ts"][vi]); v["kb"][y] = kb
    # direct row-replication check on a subset
    dm = 0.0
    for key in [("ORIG", "CE", 1, 0), ("ORIG", "SOFT", 7, 0), ("ORIG", "BASE", 3, 0)]:
        e = D[key]
        for b in range(3):
            rep = np.repeat(np.arange(len(e["val"]["m"])), W63[b][np.array([vq[q] for q, _ in vref["ks"]])].astype(int))
            t_dir, _ = best_threshold(e["val"]["m"][rep], e["val"]["y"][rep])
            dm = max(dm, abs(t_dir - info[id(e)]["ts"][b]))
    RES["checks"].append(dict(model=fam, check="validation resample: weighted threshold == original best_threshold on replicated rows (3 draws, subset)", max_abs_diff=dm, invalid_draws=int(bad.sum())))
    assert dm < 1e-9

    def aggd(var, cond, y, mode, Ts):
        seeds = (0,) if cond == "BASE" else SEEDS
        vals = []
        for s in seeds:
            xs = []
            for T in Ts:
                e = D[(var, cond, T, s)]
                xs.append(float(stat_of(e[y]["t0"]["tot"], "kappa")) if mode == "t0" else (float(stat_of(e[y]["sel"]["tot"], "kappa")) if mode == "full" else info[id(e)]["kb"][y][vi]))
            vals.append(np.mean(xs, axis=0))
        r = np.mean(vals, axis=0)
        return r if np.ndim(r) else np.full(len(vi), float(r))

    def sm(x):
        x = np.asarray(x, float)
        return dict(mean=float(x.mean()), sd=float(x.std()), p2_5=pct(x)[0], p97_5=pct(x)[1], share_pos=float((x > 1e-12).mean()))

    st = []
    for y in YEARS:
        for var in vars_:
            for cond in ("BASE", "CE", "SOFT", "ANY"):
                if cond == "ANY" and var == "PO":
                    continue
                kb = aggd(var, cond, y, "tb", Ts7); k0 = aggd(var, cond, y, "t0", Ts7)[0]; kf = aggd(var, cond, y, "full", Ts7)[0]
                st.append(dict(variant=var, year=y, cond=cond, k0=float(k0), kfull=float(kf), kb=sm(kb), kb_minus_full=sm(kb - kf), kb_minus_t0=sm(kb - k0), n_valid=int(len(vi))))
        for nm, (vA, cA, vB, cB) in {"SOFT-CE (Original)": ("ORIG", "SOFT", "ORIG", "CE"), "CE-BASE (Original)": ("ORIG", "CE", "ORIG", "BASE"), "ANY-SOFT (Original)": ("ORIG", "ANY", "ORIG", "SOFT")}.items():
            st.append(dict(contrast=nm, year=y, kb_diff=sm(aggd(vA, cA, y, "tb", Ts7) - aggd(vB, cB, y, "tb", Ts7))))
        if fam != "gemma":
            for cond in ("BASE", "CE", "SOFT"):
                st.append(dict(contrast=f"PO-Original {cond}", year=y, kb_diff=sm(aggd("PO", cond, y, "tb", Ts7) - aggd("ORIG", cond, y, "tb", Ts7))))
            st.append(dict(contrast="(SOFT-CE)PO - (SOFT-CE)Original", year=y, kb_diff=sm((aggd("PO", "SOFT", y, "tb", Ts7) - aggd("PO", "CE", y, "tb", Ts7)) - (aggd("ORIG", "SOFT", y, "tb", Ts7) - aggd("ORIG", "CE", y, "tb", Ts7)))))
    F["val_resample"] = st
    print(fam, "done", f"{time.time() - t_m:.0f}s", flush=True)
    del D, uniq, cache, info

RES["meta"] = dict(NB=NB, R63=R63, seconds=time.time() - t_start, numpy=np.__version__, python=sys.version.split()[0])
json.dump(RES, open(OUT / "po_val_results.json", "w"), default=float, indent=1)
np.savez_compressed(OUT / "po_val_paired_kappa_draws.npz", **{k.replace("|", "__"): v for k, v in NPZ.items()})
(OUT / "po_val_checkpoint_metrics.csv").write_text("\n".join(CSV) + "\n", encoding="utf-8")
print("ANALYSIS DONE", f"{time.time() - t_start:.0f}s")
