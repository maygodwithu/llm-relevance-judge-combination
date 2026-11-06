"""Gemma precision auxiliary analysis for chapter 6.3 (experiment log 241 section 6), from the STORED scores of messages 230 / 236 only (no GPU): T1 and T7, seed 0, CE / CE+soft kappa / CE+AnyLoss-like kappa; ESCI validation (resampled) -> test300 (fixed) and MS MARCO 2020 (resampled) -> 2019, 2019 (resampled) -> 2020. BF16 = the original stored scores, FP32 = the FP32 scores of 236. Both precisions use the SAME validation-query multiplicity matrices (those of err63_main, saved in err63_multiplicities.npz) and select their threshold from their OWN scores of every draw. Not mixed with the T1-T7 x 3-seed analysis.
    python err63_gemma.py --root . --mods analysis --kappa cpmr_experiment/auc61/out/kappa.pkl --fp32 cpmr_experiment/gemma234/out --manifest cpmr_experiment/gemma231/sample_manifest.json --g234 cpmr_experiment/err62/gemma234_results.json --prev cpmr_experiment/err63/out --out cpmr_experiment/err63/out"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import err63_lib as E

ap = argparse.ArgumentParser()
for k in ("root", "mods", "kappa", "fp32", "manifest", "g234", "prev", "out"):
    ap.add_argument("--" + k, required=True)
a = ap.parse_args()
E.init(a.root, a.mods, a.kappa)
root = Path(a.root)
man = json.load(open(root / a.manifest, encoding="utf-8"))
CK = {(c["dataset"], c["T"], c["loss"]): c for c in man["checkpoints"]}
g234 = json.load(open(root / a.g234, encoding="utf-8"))
REC = {(r["dataset"], r["set"], r["T"], r["loss"]): r for r in g234["per_checkpoint"]}
MAT = np.load(root / a.prev / "err63_multiplicities.npz")
QID = json.load(open(root / a.prev / "err63_val_query_ids.json"))
ld = lambda p: json.load(gzip.open(p, "rt", encoding="utf-8"))
LOSSES = ("CE", "SOFT", "ANY"); TS = (1, 7)


def load_fp32(tag, name):
    d = json.load(open(root / a.fp32 / f"fp32_{tag}__{name}.json"))
    pairs = [(p[0], p[1]) for p in d["pairs"]]
    s = np.array(d["scores"], dtype=float)
    return pairs, np.array([p[2] for p in d["pairs"]], dtype=bool), s[:, 0] + s[:, 1] - s[:, 2] - s[:, 3]


def load_bf16(ck, name, pairs):
    T = ck["T"]
    if ck["dataset"] == "ESCI":
        sub = "sel" if name == "ESCIval" else "test300"
        r = ld(root / f"cpmr_experiment/esci_gemma/eval/esciGemmaEval_{ck['tag']}_step625__T{T}__{sub}.json.gz")["raw"]
        return np.array([r[q][d][0] + r[q][d][1] - r[q][d][3] - r[q][d][4] for q, d in pairs], dtype=float)
    r = ld(root / f"cpmr_experiment/msm_gemma/eval/msmGemmaEval_{ck['tag']}__T{T}__{name[2:]}.json.gz")["run_fp32"]
    return np.array([r[q][d] for q, d in pairs], dtype=float)


PROTOS = [("ESCI", "ESCI-Gemma", "ESCIval", "ESCItest", ["reuse_rng1", "new_rng63"]), ("MS", "MS-Gemma-2019", "MS2020", "MS2019", ["new_rng63"]), ("MS", "MS-Gemma-2020", "MS2019", "MS2020", ["new_rng63"])]
OUT = dict(protocols={}, checks=[])
DRAWS = {}
for ds, sid, vname, tname, specs in PROTOS:
    # data per (T, loss, precision): validation (pairs, y, m) and test
    D = {}
    qlist = None
    for T in TS:
        for loss in LOSSES:
            ck = CK[(ds, T, loss)]
            vp, vy, vF = load_fp32(ck["tag"], vname); tp, ty, tF = load_fp32(ck["tag"], tname)
            vB, tB_ = load_bf16(ck, vname, vp), load_bf16(ck, tname, tp)
            ql = sorted({q for q, _ in vp})
            qlist = qlist or ql
            assert ql == qlist and ql == QID[sid], (sid, T, loss)
            D[(T, loss)] = dict(vp=vp, vy=vy, v=dict(BF16=vB, FP32=vF), ty=ty, t=dict(BF16=tB_, FP32=tF))
    qid = {q: i for i, q in enumerate(qlist)}
    nq = len(qlist)
    for spec in specs:
        W = MAT[f"{sid}__{spec}"].astype(float)
        assert W.shape == (E.R_DRAWS, nq)
        KB, KF, K0, TT = {}, {}, {}, {}
        for (T, loss), d in D.items():
            for prec in ("BF16", "FP32"):
                val = E.Val(d["vp"], d["v"][prec], d["vy"], qid)
                te = E.Test(d["t"][prec], d["ty"])
                tfull = val.best(np.ones(nq))
                ts = np.array([val.best(W[b]) for b in range(E.R_DRAWS)])
                assert np.isfinite(ts).all()
                KB[(T, loss, prec)] = te.kappa(ts); KF[(T, loss, prec)] = float(te.kappa([tfull])[0]); K0[(T, loss, prec)] = float(te.kappa([0.0])[0]); TT[(T, loss, prec)] = (ts, tfull, val.sd)
                # check against the stored 236 values
                r = REC[(ds, tname, T, loss)]
                if prec == "BF16":
                    ref = (r["BF16"]["kappa_t0"], r["BF16"]["kappa_tB"], r["tB"]); mine = (K0[(T, loss, prec)], KF[(T, loss, prec)], tfull)
                else:
                    ref = (r["FP32"]["kappa_t0"], r["FP32"]["kappa_tF"], r["tF"]); mine = (K0[(T, loss, prec)], KF[(T, loss, prec)], tfull)
                OUT["checks"].append(dict(protocol=sid, spec=spec, T=T, loss=loss, precision=prec, max_abs_diff_vs_236=float(max(abs(x - y) for x, y in zip(ref, mine)))))
        assert max(c["max_abs_diff_vs_236"] for c in OUT["checks"]) < 1e-9
        agg = lambda dct, loss, prec: np.mean([dct[(T, loss, prec)] for T in TS], axis=0)
        P = {}
        for prec in ("BF16", "FP32"):
            blk = {}
            for loss in LOSSES:
                k0 = np.mean([K0[(T, loss, prec)] for T in TS]); kf = np.mean([KF[(T, loss, prec)] for T in TS]); kb = agg(KB, loss, prec)
                blk[loss] = dict(k0=float(k0), kfull=float(kf), kb=E.summ(kb), delta_tb_t0=E.summ(kb - k0, kf - k0),
                                 thr_sd_over_margin_sd={f"T{T}": float(TT[(T, loss, prec)][0].std() / TT[(T, loss, prec)][2]) for T in TS},
                                 tfull={f"T{T}": float(TT[(T, loss, prec)][1]) for T in TS}, kb_T={f"T{T}": E.summ(KB[(T, loss, prec)]) for T in TS})
            for nm, A, B in (("SOFT-CE", "SOFT", "CE"), ("ANY-SOFT", "ANY", "SOFT")):
                d_ = agg(KB, A, prec) - agg(KB, B, prec)
                ref = np.mean([KF[(T, A, prec)] - KF[(T, B, prec)] for T in TS])
                blk[nm] = E.summ(d_, ref)
                blk[nm]["by_T"] = {f"T{T}": E.summ(KB[(T, A, prec)] - KB[(T, B, prec)], KF[(T, A, prec)] - KF[(T, B, prec)]) for T in TS}
                blk[nm]["draws"] = [float(x) for x in d_]
            blk["SOFT(t0)-CE(t_b)"] = E.summ(np.mean([K0[(T, "SOFT", prec)] for T in TS]) - agg(KB, "CE", prec), np.mean([K0[(T, "SOFT", prec)] - KF[(T, "CE", prec)] for T in TS]))
            P[prec] = blk
        P["FP32_minus_BF16"] = {}
        for nm in ("SOFT-CE", "ANY-SOFT"):
            P["FP32_minus_BF16"][nm] = E.summ(np.array(P["FP32"][nm]["draws"]) - np.array(P["BF16"][nm]["draws"]), P["FP32"][nm]["full"] - P["BF16"][nm]["full"])
        OUT["protocols"][f"{sid}|{spec}"] = P
        DRAWS[f"{sid}|{spec}"] = {prec: {loss: [float(x) for x in agg(KB, loss, prec)] for loss in LOSSES} for prec in ("BF16", "FP32")}
        print(sid, spec, "done", flush=True)
json.dump(OUT, open(root / a.out / "err63_gemma_fp32.json", "w"), default=float)
json.dump(DRAWS, open(root / a.out / "err63_gemma_fp32_draws.json", "w"))
print("max diff vs 236:", max(c["max_abs_diff_vs_236"] for c in OUT["checks"]))
