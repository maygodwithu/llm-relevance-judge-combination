"""ESCI PO T1-T7 completion analysis (messages 268/269; 18 new PO CE+soft-kappa runs finished): Original vs PO-selected instruction x Base / CE / CE+soft kappa, Qwen3-4B and Llama-3.1-8B-Instruct, ESCI test300 (300 queries / 6,078 rows; E = Relevant, S / C / I = Irrelevant),
from the stored raw FP32 scores through the loaders of the ESCI analyses (auc61_data, err62_lib, err63_lib; no new inference). Llama: PO kept the original text for T1, T4, T6 -> aliases of the Original results (difference exactly 0).
t=0 and the validation-selected threshold t* of every checkpoint (hard-kappa maximiser over the 200-query / 4,047-row validation scores of that checkpoint, ties -> 0, then smaller |t|, then smaller t; the Base threshold is never transferred; test never selects).
Aggregation: per-instruction kappa -> equal-weight mean over the instruction set (main T1-T7) -> mean over seeds (Base seed 0).
Bootstrap: kappa contrasts B = 2,000, default_rng(0) over the 300 test queries (the draws of the existing ESCI analyses, one set for every cell); AUC / error profile / transitions B = 10,000, default_rng(61) (the draws of 238 / 236); validation-query threshold stability 500 draws of the 200 validation queries, default_rng(1) ('reuse_rng1', the draws of the existing ESCI stability analyses) and default_rng(63) ('new_rng63').
    python esci_po_analysis.py --root . --kappa cpmr_experiment/auc61/out/kappa.pkl --mods analysis --prev cpmr_experiment/err63/out/err63_results.json --out cpmr_experiment/esci_268/summary"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
for k in ("root", "kappa", "mods", "prev", "out"):
    ap.add_argument("--" + k, required=True)
a = ap.parse_args()
R = Path(a.root)
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R / a.mods))
import err62_lib as L
import err63_lib as E
from auc61_boot import prep, weighted_auc_numerator

E.init(a.root, a.mods, a.kappa)
QC = L.QC
kap_vec = QC.kap_vec
prev = json.load(open(a.prev))
K = L.K
SEEDS = (0, 1, 2)
TS7 = list(range(1, 8)); TS3 = [1, 2, 3]
CN = {"BASE": "Base", "CE": "CE", "SOFT": "CE+soft kappa"}
PO_OF = {"BASE": "POBASE", "CE": "POCE", "SOFT": "POSOFT"}
B2, B10 = 2000, 10000
t_start = time.time()
RES = dict(meta={}, checks=[], fam={})
CSV = ["model,variant,cond,T,seed,alias_of_original,threshold_val,val_kappa,kappa_t0,kappa_sel,auc,pos_rate_t0,pos_rate_sel,fnr_sel,fpr_S_sel,fpr_C_sel,fpr_I_sel,tp_sel,fp_sel,fn_sel,tn_sel"]
RAW = [["model", "variant", "cond", "T", "seed", "set", "query_id", "pair_id", "esci_class", "label_E", "margin_fp32"]]
NPZ = {}
pct = lambda x, p=(2.5, 97.5): [float(np.nanpercentile(x, p[0])), float(np.nanpercentile(x, p[1]))]


def stat_of(tot, st):
    tp, fp, fn, tn = (tot[..., i] for i in range(4))
    n = tp + fp + fn + tn
    with np.errstate(all="ignore"):
        return {"kappa": lambda: kap_vec(tp, fp, fn, tn), "acc": lambda: (tp + tn) / n, "pos": lambda: (tp + fp) / n, "fnr": lambda: fn / (tp + fn)}[st]()


for sid, mn in (("ESCI-Qwen", "qwen"), ("ESCI-Llama", "llama")):
    t_m = time.time()
    D = {}
    ref = None
    for var in ("ORIG", "PO"):
        for cond in ("BASE", "CE", "SOFT"):
            c_ = cond if var == "ORIG" else PO_OF[cond]
            for T in TS7:
                for s in ((0,) if cond == "BASE" else SEEDS):
                    alias = var == "PO" and L.D.esci_is_alias(sid, c_, T)
                    if alias:
                        D[(var, cond, T, s)] = D[("ORIG", cond, T, s)]
                        continue
                    r = L.load(sid, c_, T, s)
                    assert r is not None, (sid, c_, T, s)
                    if ref is None:
                        ref = dict(ks=r["ks"], code=r["code"], y=r["y"])
                        qs = sorted({q for q, _ in r["ks"]}); qid = {q: i for i, q in enumerate(qs)}
                        rowq = np.array([qid[q] for q, _ in r["ks"]]); nq = len(qs)
                    assert r["ks"] == ref["ks"] and (r["y"] == ref["y"]).all() and (r["code"] == ref["code"]).all() and np.isfinite(r["m"]).all()
                    thr = float(L.sel_threshold(sid, c_, T, s))
                    e = dict(m=r["m"], thr=thr)
                    for mode, t in (("t0", 0.0), ("sel", thr)):
                        p = r["m"] > t
                        cnt = np.stack([np.bincount(rowq, weights=arr.astype(float), minlength=nq) for arr in (p & ref["y"], p & ~ref["y"], ~p & ref["y"], ~p & ~ref["y"])], axis=1)
                        e[mode] = dict(pred=p, cnt=cnt, tot=cnt.sum(0))
                    e["auc"] = QC.auc(r["m"], ref["y"])
                    va = E.val_arrays(sid, c_, T, s)
                    e["val"] = va
                    e["val_kappa"] = QC.best_threshold(va[1], va[2])[1]
                    D[(var, cond, T, s)] = e
    code = ref["code"]; y = ref["y"]
    gm = L.groups_of(sid, code)
    W2 = np.stack([np.bincount(i, minlength=nq) for i in np.random.default_rng(0).integers(0, nq, size=(B2, nq))]).astype(float)
    uniq = {id(e): e for e in D.values()}
    for e in uniq.values():
        for mode in ("t0", "sel"):
            e[mode]["tot_b"] = W2 @ e[mode]["cnt"]
    print(sid, "loaded", len(uniq), "unique cells", f"{time.time() - t_m:.0f}s", flush=True)
    # ---------------------------------------------------------------- csv / raw / reproduction
    dmax = 0.0; n = 0
    for (var, cond, T, s), e in sorted(D.items()):
        alias = var == "PO" and L.D.esci_is_alias(sid, PO_OF[cond], T)
        ts = e["sel"]["tot"]
        pr = e["sel"]["pred"]
        k0, ks_ = float(stat_of(e["t0"]["tot"], "kappa")), float(stat_of(ts, "kappa"))
        fpr = {c: float((pr & (code == c)).sum() / (code == c).sum()) for c in (1, 2, 3)}
        CSV.append(f"{mn},{var},{cond},{T},{s},{int(alias)},{e['thr']:.6f},{e['val_kappa']:.6f},{k0:.6f},{ks_:.6f},{e['auc']:.6f},{float(stat_of(e['t0']['tot'], 'pos')):.6f},{float(stat_of(ts, 'pos')):.6f},{float(stat_of(ts, 'fnr')):.6f},{fpr[1]:.6f},{fpr[2]:.6f},{fpr[3]:.6f},{int(ts[0])},{int(ts[1])},{int(ts[2])},{int(ts[3])}")
        if not alias and (var, cond, T, s) in D:
            for (q, d), mm, yy, cc in zip(ref["ks"], e["m"], y, code):
                RAW.append([mn, var, cond, T, s, "test300", q, d, int(cc), int(yy), repr(float(mm))])
            vks, vm, vy = e["val"]
            for (q, d), mm, yy in zip(vks, vm, vy):
                RAW.append([mn, var, cond, T, s, "validation", q, d, "", int(yy), repr(float(mm))])
        if var == "ORIG":
            kk0, kkf, kthr = K[(sid, cond, T, s)]
            dmax = max(dmax, abs(kk0 - k0), abs(kkf - ks_), abs(kthr - e["thr"])); n += 1
    RES["checks"].append(dict(model=mn, check="Original cells: kappa(t=0), kappa(validation t*), threshold == kappa.pkl of the AUC re-analysis (230; reproduces the reports 206 / 210)", n_cells=n, max_abs_diff=dmax))
    assert dmax < 1e-9, (sid, dmax)
    cache = {}

    def pt(e, mode, st):
        return float(e["auc"]) if st == "auc" else float(stat_of(e[mode]["tot"], st))

    def bv(e, mode, st):
        k = (id(e), mode, st)
        if k not in cache:
            cache[k] = stat_of(e[mode]["tot_b"], st)
        return cache[k]

    def agg(var, cond, mode, st, Ts, boot=True):
        seeds = (0,) if cond == "BASE" else SEEDS
        ps, bs = [], []
        for s in seeds:
            es = [D[(var, cond, T, s)] for T in Ts]
            ps.append(float(np.mean([pt(e, mode, st) for e in es])))
            if boot:
                bs.append(np.mean([bv(e, mode, st) for e in es], axis=0))
        tp_ = [float(np.mean([pt(D[(var, cond, T, s)], mode, st) for s in seeds])) for T in Ts]
        return float(np.mean(ps)), (np.mean(bs, axis=0) if bs else None), ps, tp_

    def diff(vA, cA, vB, cB, mode, Ts, st="kappa"):
        x1, b1, s1, t1 = agg(vA, cA, mode, st, Ts); x2, b2, s2, t2 = agg(vB, cB, mode, st, Ts)
        n_ = max(len(s1), len(s2))
        return x1 - x2, b1 - b2, [s1[i if len(s1) > 1 else 0] - s2[i if len(s2) > 1 else 0] for i in range(n_)], [u - v for u, v in zip(t1, t2)]

    CH = [T for T in TS7 if not L.D.esci_is_alias(sid, "POCE", T)]
    SC = {"T1-T7": TS7, "T1-T3": TS3, "T4-T7": [4, 5, 6, 7]}
    if mn == "llama":
        SC["changed only (T2,T3,T5,T7)"] = CH
    F = RES["fam"].setdefault(mn, {"changed": CH})
    # ---------------------------------------------------------------- reproduction of the existing T1-T3 PO / Original values (err63 reuse_rng1 cond_stats)
    pv = prev["sets"][sid]["specs"]["reuse_rng1"]["cond_stats"]
    d13 = 0.0; k13 = 0
    for cond in ("BASE", "CE", "SOFT"):
        for var, cc in (("ORIG", cond), ("PO", PO_OF[cond])):
            rec = pv.get(f"T1-T3|{cc}")
            if rec is None:
                continue
            d13 = max(d13, abs(rec["k0"] - agg(var, cond, "t0", "kappa", TS3, boot=False)[0]), abs(rec["kfull"] - agg(var, cond, "sel", "kappa", TS3, boot=False)[0])); k13 += 1
    RES["checks"].append(dict(model=mn, check="T1-T3 aggregates (Original and PO, Base / CE / soft) equal the stored 242 / err63 values (kappa(t=0), kappa(validation t*))", n_conditions=k13, max_abs_diff=d13))
    assert d13 < 1e-9, d13
    # ---------------------------------------------------------------- main PO tables
    main = {}
    for sc, Ts in SC.items():
        rows = []
        for cond in ("BASE", "CE", "SOFT"):
            r = dict(cond=cond)
            for mode in ("t0", "sel"):
                xo, bo, so, to = agg("ORIG", cond, mode, "kappa", Ts); xp, bp, sp, tp_ = agg("PO", cond, mode, "kappa", Ts)
                d, bd, sd_, td = diff("PO", cond, "ORIG", cond, mode, Ts)
                r[mode] = dict(orig=xo, po=xp, diff=d, ci95=pct(bd), share_pos=float(np.mean(bd > 0)), seed_diffs=sd_, instr_diffs=td, orig_seeds=so, po_seeds=sp)
                if sc == "T1-T7":
                    NPZ[f"{mn}|{cond}|{mode}"] = bd
            rows.append(r)
        main[sc] = rows
    F["main"] = main
    # ---------------------------------------------------------------- objective contrasts (Original and PO), T1-T7; 95% and K6 (3 x 2 family of the earlier ESCI analyses: Bonferroni 6)
    obj = []
    for mode in ("t0", "sel"):
        r = dict(mode=mode)
        for var in ("ORIG", "PO"):
            v = {}
            for cond in ("BASE", "CE", "SOFT"):
                v[cond] = agg(var, cond, mode, "kappa", TS7, boot=False)[0]
            for nm, (cA, cB) in {"CE-BASE": ("CE", "BASE"), "SOFT-CE": ("SOFT", "CE")}.items():
                d, bd, sd_, td = diff(var, cA, var, cB, mode, TS7)
                v[nm] = dict(diff=d, ci95=pct(bd), ci_k6=pct(bd, (100 * 0.05 / 12, 100 - 100 * 0.05 / 12)), seed_diffs=sd_, n_pos_instr=int(sum(x > 0 for x in td)))
                NPZ[f"{mn}|{var}|{nm}|{mode}"] = bd
            r[var] = v
        dp = diff("PO", "SOFT", "PO", "CE", mode, TS7); do_ = diff("ORIG", "SOFT", "ORIG", "CE", mode, TS7)
        r["SOFTCE_PO_minus_ORIG"] = dict(diff=dp[0] - do_[0], ci95=pct(dp[1] - do_[1]), seed_diffs=[u - v for u, v in zip(dp[2], do_[2])])
        obj.append(r)
    F["objective"] = obj
    # ---------------------------------------------------------------- per instruction (seed mean)
    F["per_instruction"] = [dict(variant=var, cond=cond, T=T, t0=agg(var, cond, "t0", "kappa", [T], boot=False)[0], sel=agg(var, cond, "sel", "kappa", [T], boot=False)[0], alias=bool(var == "PO" and L.D.esci_is_alias(sid, PO_OF[cond], T)))
                            for var in ("ORIG", "PO") for cond in ("BASE", "CE", "SOFT") for T in TS7]
    # ---------------------------------------------------------------- AUC, error profile, transitions (B = 10,000, default_rng(61), the 238 procedure)
    Bt = L.Boot(ref["ks"], y, code, gm)
    W61 = Bt.W
    wpos = W61 @ np.bincount(rowq, weights=y.astype(float), minlength=nq); wneg = W61 @ np.bincount(rowq, weights=(~y).astype(float), minlength=nq)
    valid = (wpos > 0) & (wneg > 0)
    for e in uniq.values():
        pp = prep(e["m"]); num = np.empty(B10)
        for i in range(0, B10, 500):
            num[i:i + 500] = weighted_auc_numerator(pp, y, rowq, W61[i:i + 500])
        with np.errstate(all="ignore"):
            e["aucb"] = np.where(valid, num / (wpos * wneg), np.nan)
        for mode in ("t0", "sel"):
            e[mode]["cnt_cls"] = np.stack([np.bincount(rowq, weights=(e[mode]["pred"] & (code == c)).astype(float), minlength=nq) for c in range(4)], axis=1)   # predicted-Relevant count per query and gold class E,S,C,I
            e[mode]["tot61"] = W61 @ e[mode]["cnt"]; e[mode]["cls61"] = W61 @ e[mode]["cnt_cls"]
    Ncls = np.array([[int((code == c).sum())] for c in range(4)], float)
    Ncls_q = np.stack([np.bincount(rowq, weights=(code == c).astype(float), minlength=nq) for c in range(4)], axis=1)
    Ncls_b = W61 @ Ncls_q
    prof = []
    for cond in ("BASE", "CE", "SOFT"):
        for mode in ("t0", "sel"):
            rec = dict(cond=cond, mode=mode)

            def val(var, what, boot):
                seeds = (0,) if cond == "BASE" else SEEDS
                ps, bs = [], []
                for s in seeds:
                    es = [D[(var, cond, T, s)] for T in TS7]
                    if what == "auc":
                        ps.append(np.mean([e["auc"] for e in es])); bs.append(np.mean([e["aucb"] for e in es], axis=0))
                    elif what in ("pos", "fnr", "kappa"):
                        ps.append(np.mean([float(stat_of(e[mode]["tot"], what)) for e in es])); bs.append(np.mean([stat_of(e[mode]["tot61"], what) for e in es], axis=0))
                    else:                                       # FPR of class S / C / I
                        c = {"S": 1, "C": 2, "I": 3}[what]
                        ps.append(np.mean([float(e[mode]["cnt_cls"][:, c].sum() / Ncls_q[:, c].sum()) for e in es])); bs.append(np.mean([(e[mode]["cls61"][:, c] / Ncls_b[:, c]) for e in es], axis=0))
                return float(np.mean(ps)), np.mean(bs, axis=0)
            for what in (("auc",) if mode == "t0" else ()) + ("pos", "fnr", "S", "C", "I"):
                xo, bo = val("ORIG", what, True); xp, bp = val("PO", what, True)
                rec[what] = dict(orig=xo, po=xp, diff=xp - xo, ci95=pct(bp - bo))
            prof.append(rec)
    F["profile"] = prof
    # transitions Original -> PO (T1-T7), groups E / S / C / I / S+C+I / all
    trows = []
    for cond in ("BASE", "CE", "SOFT"):
        for mode in ("t0", "sel"):
            seeds = (0,) if cond == "BASE" else SEEDS
            qcs = {(T, s): Bt.qc(D[("ORIG", cond, T, s)][mode]["pred"], D[("PO", cond, T, s)][mode]["pred"], y) for T in TS7 for s in seeds}
            pt_, bt_ = Bt.agg(qcs, TS7, seeds)
            rec = dict(cond=cond, mode=mode)
            for g in gm:
                rec[g] = {kd: dict(rate=pt_[(g, kd)], ci95=[float(np.percentile(bt_[(g, kd)][Bt.valid[g]], 2.5)), float(np.percentile(bt_[(g, kd)][Bt.valid[g]], 97.5))]) for kd in ("WR", "RW", "net", "changed")}
            trows.append(rec)
    F["transitions"] = trows
    # ---------------------------------------------------------------- validation-query stability (500 draws; default_rng(1) = reuse_rng1 and default_rng(63))
    vks0 = next(iter(uniq.values()))["val"][0]
    vqs = sorted({q for q, _ in vks0}); vq = {q: i for i, q in enumerate(vqs)}
    info = {}
    for e in uniq.values():
        assert e["val"][0] == vks0
        va = E.Val(e["val"][0], e["val"][1], e["val"][2], vq)
        info[id(e)] = dict(va=va, te=E.Test(e["m"], y), k0=float(E.Test(e["m"], y).kappa([0.0])[0]), kfull=float(E.Test(e["m"], y).kappa([e["thr"]])[0]))
        assert abs(va.best(np.ones(len(vqs))) - e["thr"]) < 1e-9
    stab = {}
    for spec, seed in (("reuse_rng1", 1), ("new_rng63", 63)):
        Wv = E.make_draws(len(vqs), seed)
        for e in uniq.values():
            info[id(e)]["ts"] = np.array([info[id(e)]["va"].best(Wv[b]) for b in range(E.R_DRAWS)])
        bad = np.zeros(E.R_DRAWS, bool)
        for v in info.values():
            bad |= ~np.isfinite(v["ts"])
        vi = np.where(~bad)[0]
        for v in info.values():
            kb = np.full(E.R_DRAWS, np.nan); kb[vi] = v["te"].kappa(v["ts"][vi]); v["kb"] = kb
        dm = 0.0
        for key in [("ORIG", "CE", 1, 0), ("ORIG", "SOFT", 7, 0), ("ORIG", "BASE", 3, 0), ("PO", "CE", 5, 0), ("PO", "SOFT", 4, 0)]:
            e = D[key]
            for b in range(3):
                rep = np.repeat(np.arange(len(e["val"][1])), Wv[b][np.array([vq[q] for q, _ in vks0])].astype(int))
                dm = max(dm, abs(QC.best_threshold(e["val"][1][rep], e["val"][2][rep])[0] - info[id(e)]["ts"][b]))
        RES["checks"].append(dict(model=mn, check=f"validation resample {spec}: weighted threshold == original best_threshold on replicated rows (3 draws, subset)", max_abs_diff=dm, invalid_draws=int(bad.sum())))
        assert dm < 1e-9

        def aggd(var, cond, mode):
            seeds = (0,) if cond == "BASE" else SEEDS
            vals = []
            for s in seeds:
                xs = []
                for T in TS7:
                    e = D[(var, cond, T, s)]; v = info[id(e)]
                    xs.append(v["k0"] if mode == "t0" else (v["kfull"] if mode == "full" else v["kb"][vi]))
                vals.append(np.mean(xs, axis=0))
            r = np.mean(vals, axis=0)
            return r if np.ndim(r) else np.full(len(vi), float(r))

        def sm(x):
            x = np.asarray(x, float)
            return dict(mean=float(x.mean()), sd=float(x.std()), p2_5=pct(x)[0], p97_5=pct(x)[1], share_pos=float((x > 1e-12).mean()))
        st = []
        for var in ("ORIG", "PO"):
            for cond in ("BASE", "CE", "SOFT"):
                kb = aggd(var, cond, "tb"); k0 = aggd(var, cond, "t0")[0]; kf = aggd(var, cond, "full")[0]
                st.append(dict(variant=var, cond=cond, k0=float(k0), kfull=float(kf), kb=sm(kb), kb_minus_full=sm(kb - kf), kb_minus_t0=sm(kb - k0), n_valid=int(len(vi))))
        for cond in ("BASE", "CE", "SOFT"):
            st.append(dict(contrast=f"PO-Original {cond}", kb_diff=sm(aggd("PO", cond, "tb") - aggd("ORIG", cond, "tb"))))
        for var in ("ORIG", "PO"):
            st.append(dict(contrast=f"SOFT-CE ({var})", kb_diff=sm(aggd(var, "SOFT", "tb") - aggd(var, "CE", "tb"))))
        st.append(dict(contrast="(SOFT-CE)PO - (SOFT-CE)Original", kb_diff=sm((aggd("PO", "SOFT", "tb") - aggd("PO", "CE", "tb")) - (aggd("ORIG", "SOFT", "tb") - aggd("ORIG", "CE", "tb")))))
        stab[spec] = st
    F["stability"] = stab
    print(sid, "done", f"{time.time() - t_m:.0f}s", flush=True)
    del D, uniq, cache, info

RES["meta"] = dict(B_kappa=B2, B_auc_err=B10, R_val=E.R_DRAWS, seconds=time.time() - t_start, numpy=np.__version__, python=sys.version.split()[0])
json.dump(RES, open(OUT / "esci_po_results.json", "w"), default=float, indent=1)
np.savez_compressed(OUT / "esci_po_paired_kappa_draws.npz", **{k.replace("|", "__"): v for k, v in NPZ.items()})
(OUT / "esci_po_checkpoint_metrics.csv").write_text("\n".join(CSV) + "\n", encoding="utf-8")
import csv
import gzip
with gzip.open(OUT / "esci_po_raw_scores_fp32.csv.gz", "wt", encoding="utf-8", newline="") as fh:
    csv.writer(fh).writerows(RAW)
print("ANALYSIS DONE", f"{time.time() - t_start:.0f}s")
