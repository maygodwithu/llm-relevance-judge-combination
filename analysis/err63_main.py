"""Chapter 6.3 threshold stability under validation-sample variation (experiment log 241), nine evaluation protocols from the EXISTING stored scores: ESCI (validation 200 queries resampled, test300 fixed) for Qwen3-4B / Llama-3.1-8B-Instruct / Gemma-3-4B-it; MS MARCO (resample TREC-DL 2020, evaluate 2019; resample 2019, evaluate 2020) for the same models. Conditions BASE / CE / SOFT / ANY (original texts T1-T7, seeds 0/1/2, final 625) and PO (ESCI Qwen / Llama, T1-T3).
Each draw: multiplicities of the validation queries (the SAME matrix for every model / objective / instruction / seed / PO), threshold re-selected on the weighted validation scores of every checkpoint with the original rule, applied to the fixed test; aggregation per draw: per instruction -> equal-weight mean over instructions -> mean over seeds (Base seedless).
Draw sets: ESCI 'reuse_rng1' = np.random.default_rng(1) (the draws of the 204 / 208 / 217 stability analyses; one matrix for all models) and 'new_rng63' = default_rng(63); MS MARCO 'new_rng63' = default_rng(63) per resampled year.
    python err63_main.py --root . --mods analysis --kappa cpmr_experiment/auc61/out/kappa.pkl --out cpmr_experiment/err63/out"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import err63_lib as E
import err62_lib as L

ap = argparse.ArgumentParser()
for k in ("root", "mods", "kappa", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--ndirect", type=int, default=5)
a = ap.parse_args()
t_start = time.time()
E.init(a.root, a.mods, a.kappa)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
SETS = ["ESCI-Qwen", "ESCI-Llama", "ESCI-Gemma", "MS-Qwen-2019", "MS-Qwen-2020", "MS-Llama-2019", "MS-Llama-2020", "MS-Gemma-2019", "MS-Gemma-2020"]
TS7 = list(range(1, 8)); TS3 = [1, 2, 3]
OUT = dict(sets={}, checks=[], meta={}, repro_stability={})
DRAWS = {}
MATS = {}
CANON = {}


def sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def draw_specs(sid):
    return [("reuse_rng1", 1), ("new_rng63", 63)] if sid.startswith("ESCI") else [("new_rng63", 63)]


for sid in SETS:
    t0 = time.time()
    CK = {}
    qlist = None
    for c in E.CONDS:
        for T in E.ALL7:
            for s in E.seeds_of(c):
                r = L.load(sid, c, T, s)
                if r is None:
                    continue
                va = E.val_arrays(sid, c, T, s)
                if va is None:
                    continue
                ks, vm, vy = va
                ql = sorted({q for q, _ in ks})
                if qlist is None:
                    qlist = ql
                assert ql == qlist, (sid, c, T, s, "validation query set differs")
                CK[(c, T, s)] = dict(r=r, va=(ks, vm, vy))
    qid = {q: i for i, q in enumerate(qlist)}
    nq = len(qlist)
    CANON[sid] = qlist
    for key, d in CK.items():
        ks, vm, vy = d["va"]
        d["val"] = E.Val(ks, vm, vy, qid)
        d["test"] = E.Test(d["r"]["m"], d["r"]["y"])
        d["t_full"] = float(L.sel_threshold(sid, key[0], key[1], key[2]))
        d["k0"] = float(d["test"].kappa([0.0])[0]); d["kfull"] = float(d["test"].kappa([d["t_full"]])[0])
    # ---- check: weighted algorithm with unit multiplicities == original full-validation selection; kappa vs kappa.pkl
    ones = np.ones(nq)
    dt = max(abs(d["val"].best(ones) - d["t_full"]) for d in CK.values())
    dk = 0.0; nk = 0
    for (c, T, s), d in CK.items():
        if c in ("BASE", "CE", "SOFT", "ANY"):
            s0 = 0 if c == "BASE" else s
            k0r, kfr, tr = L.K[(sid, c, T, s0)]
            dk = max(dk, abs(k0r - d["k0"]), abs(kfr - d["kfull"]), abs(tr - d["t_full"])); nk += 1
    OUT["checks"].append(dict(set=sid, check="unit-weight threshold == original full-validation threshold; kappa(t=0), kappa(t_full), t_full == kappa.pkl (shared/230)", n_checkpoints=len(CK), n_val_queries=nq,
                              n_val_rows=len(CK[next(iter(CK))]["va"][2]), max_abs_diff_unit_weight_vs_original_threshold=dt, max_abs_diff_vs_kappa_pkl=dk, n_vs_kappa_pkl=nk))
    assert dt < 1e-9 and dk < 1e-9, (sid, dt, dk)
    S_ = dict(n_val_queries=nq, n_val_rows=len(CK[next(iter(CK))]["va"][2]), n_test_rows=len(CK[next(iter(CK))]["r"]["y"]), n_checkpoints=len(CK), specs={})
    DRAWS[sid] = {}
    for spec, seed in draw_specs(sid):
        W = E.make_draws(nq, seed)
        MATS[(sid, spec)] = W
        for key, d in CK.items():
            d["ts_" + spec] = np.array([d["val"].best(W[b]) for b in range(E.R_DRAWS)])
        # common validity mask: a draw is used only when every checkpoint of the set has a defined threshold
        bad = np.zeros(E.R_DRAWS, bool)
        for d in CK.values():
            bad |= ~np.isfinite(d["ts_" + spec])
        valid = ~bad
        n_bad_ck = {f"{k[0]} T{k[1]} s{k[2]}": int((~np.isfinite(d["ts_" + spec])).sum()) for k, d in CK.items() if (~np.isfinite(d["ts_" + spec])).any()}
        vi = np.where(valid)[0]
        for key, d in CK.items():
            ts = d["ts_" + spec]
            kb = np.full(E.R_DRAWS, np.nan); kb[valid] = d["test"].kappa(ts[valid])
            d["kb_" + spec] = kb
            d["chg_full_" + spec] = np.array([d["test"].changed(ts[b], d["t_full"]) if valid[b] else np.nan for b in range(E.R_DRAWS)])
            d["chg_t0_" + spec] = np.array([d["test"].changed(ts[b], 0.0) if valid[b] else np.nan for b in range(E.R_DRAWS)])

        def Xs(c, T, s, mode, spec=spec, vi=vi):
            d = CK[(c, T, s)]
            return d["k0"] if mode == "t0" else (d["kfull"] if mode == "full" else d["kb_" + spec][vi])

        def agg(c, mode, Ts, vi=vi):
            """per-draw value: mean over instructions Ts (equal weight), then mean over seeds; scalar broadcast to the valid draws for modes t0 / full"""
            v = np.mean([np.mean([Xs(c, T, s, mode) for T in Ts], axis=0) for s in E.seeds_of(c)], axis=0)
            return v if np.ndim(v) else np.full(len(vi), float(v))

        def have(c, Ts):
            return all((c, T, s) in CK for T in Ts for s in E.seeds_of(c))

        sp = dict(seed=seed, n_valid_draws=int(valid.sum()), n_invalid_draws=int((~valid).sum()), invalid_checkpoint_draws=n_bad_ck, matrix_sha256=sha(W), cond_stats={}, paired=[], per_instruction={}, per_seed={}, thr_var={})
        # ----- condition statistics (T1-T7; PO T1-T3 with the matching originals)
        for scope, Ts, conds in (("T1-T7", TS7, ("BASE", "CE", "SOFT", "ANY")), ("T1-T3", TS3, ("BASE", "CE", "SOFT", "ANY", "POBASE", "POCE", "POSOFT"))):
            for c in conds:
                if not have(c, Ts):
                    continue
                k0, kf, kb = agg(c, "t0", Ts), agg(c, "full", Ts), agg(c, "tb", Ts)
                sp["cond_stats"][f"{scope}|{c}"] = dict(scope=scope, cond=c, k0=float(k0[0]), kfull=float(kf[0]), kb=E.summ(kb), delta_tb_t0=E.summ(kb - k0, kf[0] - k0[0]), delta_tb_full=E.summ(kb - kf))

        # ----- paired comparisons (same draws)
        def pair(name, scope, Ts, A, B):
            """A, B = (cond, mode): difference A - B per draw"""
            (ca, ma), (cb, mb) = A, B
            if not (have(ca, Ts) and have(cb, Ts)):
                return
            d_ = agg(ca, ma, Ts) - agg(cb, mb, Ts)
            ref = float((agg(ca, "full" if ma == "tb" else ma, Ts) - agg(cb, "full" if mb == "tb" else mb, Ts))[0])
            sp["paired"].append(dict(name=name, scope=scope, A=f"{ca}({ma})", B=f"{cb}({mb})", **E.summ(d_, ref), draws=[float(x) for x in d_]))
        for nm, A, B in (("CE-BASE", ("CE", "tb"), ("BASE", "tb")), ("SOFT-CE", ("SOFT", "tb"), ("CE", "tb")), ("ANY-SOFT", ("ANY", "tb"), ("SOFT", "tb")),
                         ("SOFT(t0)-CE(t_b)", ("SOFT", "t0"), ("CE", "tb")), ("SOFT(t_b)-SOFT(t0)", ("SOFT", "tb"), ("SOFT", "t0")),
                         ("CE(t_b)-CE(t0)", ("CE", "tb"), ("CE", "t0")), ("BASE(t_b)-BASE(t0)", ("BASE", "tb"), ("BASE", "t0")), ("ANY(t_b)-ANY(t0)", ("ANY", "tb"), ("ANY", "t0"))):
            pair(nm, "T1-T7", TS7, A, B)
        if sid in ("ESCI-Qwen", "ESCI-Llama"):
            for nm, A, B in (("PO BASE-BASE", ("POBASE", "tb"), ("BASE", "tb")), ("PO->CE - CE", ("POCE", "tb"), ("CE", "tb")), ("PO->SOFT - SOFT", ("POSOFT", "tb"), ("SOFT", "tb")), ("PO->SOFT - PO->CE", ("POSOFT", "tb"), ("POCE", "tb")),
                             ("CE-BASE", ("CE", "tb"), ("BASE", "tb")), ("SOFT-CE", ("SOFT", "tb"), ("CE", "tb")), ("ANY-SOFT", ("ANY", "tb"), ("SOFT", "tb")), ("SOFT(t0)-CE(t_b)", ("SOFT", "t0"), ("CE", "tb")), ("SOFT(t_b)-SOFT(t0)", ("SOFT", "tb"), ("SOFT", "t0")),
                             ("PO->CE - PO BASE", ("POCE", "tb"), ("POBASE", "tb")), ("PO->SOFT(t0) - PO->CE(t_b)", ("POSOFT", "t0"), ("POCE", "tb"))):
                pair(nm, "T1-T3", TS3, A, B)
        # ----- per instruction (mean over seeds) and per seed (mean over instructions)
        for T in TS7:
            blk = {}
            for c in ("BASE", "CE", "SOFT", "ANY"):
                if have(c, [T]):
                    k0, kf, kb = agg(c, "t0", [T]), agg(c, "full", [T]), agg(c, "tb", [T])
                    blk[c] = dict(k0=float(k0[0]), kfull=float(kf[0]), kb=E.summ(kb), delta_tb_t0=E.summ(kb - k0, kf[0] - k0[0]))
            for nm, A, B in (("CE-BASE", "CE", "BASE"), ("SOFT-CE", "SOFT", "CE"), ("ANY-SOFT", "ANY", "SOFT")):
                if have(A, [T]) and have(B, [T]):
                    blk[nm] = E.summ(agg(A, "tb", [T]) - agg(B, "tb", [T]), float((agg(A, "full", [T]) - agg(B, "full", [T]))[0]))
            sp["per_instruction"][f"T{T}"] = blk
        for s in L.SEEDS:
            blk = {}

            def aggs(c, mode, s=s, vi=vi):
                s_ = 0 if c == "BASE" else s
                v = np.mean([Xs(c, T, s_, mode) for T in TS7], axis=0)
                return v if np.ndim(v) else np.full(len(vi), float(v))
            for c in ("BASE", "CE", "SOFT", "ANY"):
                if all((c, T, 0 if c == "BASE" else s) in CK for T in TS7):
                    k0, kf, kb = aggs(c, "t0"), aggs(c, "full"), aggs(c, "tb")
                    blk[c] = dict(k0=float(k0[0]), kfull=float(kf[0]), kb=E.summ(kb), delta_tb_t0=E.summ(kb - k0, kf[0] - k0[0]))
            for nm, A, B in (("CE-BASE", "CE", "BASE"), ("SOFT-CE", "SOFT", "CE"), ("ANY-SOFT", "ANY", "SOFT")):
                if A in blk and B in blk:
                    blk[nm] = E.summ(aggs(A, "tb") - aggs(B, "tb"), float((aggs(A, "full") - aggs(B, "full"))[0]))
            sp["per_seed"][f"s{s}"] = blk
        # ----- threshold variation in units of the checkpoint's own validation margin SD; share of test decisions changed
        for c in E.CONDS:
            ks_ = [k for k in CK if k[0] == c]
            if not ks_:
                continue
            sdr, absf, abs0, cf, c0, sdz, nsd0 = [], [], [], [], [], [], 0
            for k in ks_:
                d = CK[k]; ts = d["ts_" + spec][vi]
                if d["val"].sd == 0:
                    nsd0 += 1
                    continue
                sdr.append(ts.std() / d["val"].sd); absf.append(abs(d["t_full"]) / d["val"].sd); abs0.append(abs(float(np.mean(ts))) / d["val"].sd)
                cf.append(np.nanmean(d["chg_full_" + spec][vi])); c0.append(np.nanmean(d["chg_t0_" + spec][vi]))
                sdz.append(float(np.mean(ts == 0.0)))
            sp["thr_var"][c] = dict(n_checkpoints=len(ks_), n_sd_zero=nsd0, sd_tb_over_margin_sd=float(np.mean(sdr)), abs_tfull_over_margin_sd=float(np.mean(absf)), abs_mean_tb_over_margin_sd=float(np.mean(abs0)),
                                    share_decisions_changed_vs_full=float(np.mean(cf)), share_decisions_changed_vs_t0=float(np.mean(c0)), share_draws_tb_equals_0=float(np.mean(sdz)))
        # ----- figure data: kappa(t_b) - kappa(t=0) per draw (T1-T7), also kappa(t_b)
        DRAWS[sid][spec] = dict(valid_draw_index=[int(i) for i in vi], delta_tb_minus_t0={}, kappa_tb={}, kappa_t0={}, kappa_full={})
        for c in ("BASE", "CE", "SOFT", "ANY"):
            if have(c, TS7):
                DRAWS[sid][spec]["delta_tb_minus_t0"][c] = [float(x) for x in agg(c, "tb", TS7) - agg(c, "t0", TS7)]
                DRAWS[sid][spec]["kappa_tb"][c] = [float(x) for x in agg(c, "tb", TS7)]
                DRAWS[sid][spec]["kappa_t0"][c] = float(agg(c, "t0", TS7)[0]); DRAWS[sid][spec]["kappa_full"][c] = float(agg(c, "full", TS7)[0])
        S_["specs"][spec] = sp
        if spec == "reuse_rng1":
            for c in ("CE", "SOFT", "ANY", "POCE", "POSOFT"):
                Ts = TS7 if c in ("CE", "SOFT", "ANY") else TS3
                if have(c, Ts):
                    kb = agg(c, "tb", Ts); k0 = agg(c, "t0", Ts)[0]; kf = agg(c, "full", Ts)[0]
                    OUT["repro_stability"].setdefault(sid, {})[c] = dict(scope="T1-T7" if c in ("CE", "SOFT", "ANY") else "T1-T3", k0=float(k0), kfull=float(kf), mean=float(kb.mean()), sd=float(kb.std()),
                                                                          p2_5=float(np.percentile(kb, 2.5)), p97_5=float(np.percentile(kb, 97.5)), mean_minus_t0=float(kb.mean() - k0), share_gt_t0=float((kb > k0).mean()))
        print(sid, spec, "valid", int(valid.sum()), f"{time.time() - t0:.0f}s", flush=True)
    # ---- check: direct row replication vs weighted count (first draws of the new63 matrix) on a checkpoint subset
    W = MATS[(sid, "new_rng63")]
    nd = 0; dmax = 0.0; dec_mis = 0
    for (c, T, s), d in sorted(CK.items()):
        if s != 0 or T not in (1, 7) or (c.startswith("PO") and T != 1):
            continue
        ks, vm, vy = d["va"]
        qrow = np.array([qid[q] for q, _ in ks])
        for b in range(a.ndirect):
            rep = np.repeat(np.arange(len(vm)), W[b][qrow].astype(int))
            t_dir, _ = E.QC.best_threshold(vm[rep], vy[rep])
            t_w = d["ts_new_rng63"][b]
            dmax = max(dmax, abs(t_dir - t_w)); dec_mis += int(((d["r"]["m"] > t_dir) != (d["r"]["m"] > t_w)).sum()); nd += 1
    OUT["checks"].append(dict(set=sid, check="direct row replication (original best_threshold on replicated rows) vs weighted count; draws 0..%d of new_rng63; checkpoints T1/T7 seed 0 of every condition (PO: T1)" % (a.ndirect - 1),
                              n_compared=nd, max_abs_threshold_diff=dmax, test_decision_mismatches=dec_mis))
    assert dmax < 1e-9 and dec_mis == 0, (sid, dmax, dec_mis)
    OUT["sets"][sid] = S_
    print(sid, "done", f"{time.time() - t0:.0f}s", "unit-weight dt", dt, "kappa.pkl dk", dk, "direct", nd, dmax, flush=True)
OUT["meta"] = dict(R=E.R_DRAWS, tol=E.TOL, seconds=time.time() - t_start, numpy=np.__version__, python=sys.version.split()[0])
json.dump(OUT, open(out / "err63_results.json", "w"), default=float)
json.dump(DRAWS, open(out / "err63_draws.json", "w"))
np.savez_compressed(out / "err63_multiplicities.npz", **{f"{k[0]}__{k[1]}": v.astype(np.int16) for k, v in MATS.items()})
json.dump(CANON, open(out / "err63_val_query_ids.json", "w"))
print("done", f"{time.time() - t_start:.0f}s")
