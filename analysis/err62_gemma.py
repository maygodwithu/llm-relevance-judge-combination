"""Gemma precision auxiliary analysis for chapter 6.2 (experiment log 237 section 6), from the STORED results of message 236 only (no GPU): T1 and T7, seed 0, CE / CE+soft kappa / CE+AnyLoss-like kappa, the complete evaluation sets (ESCI test300, TREC-DL 2019 / 2020). BF16 = the original stored scores (A4) with the original threshold t_B; FP32 = the A-FP32 scores of 236 with the re-selected threshold t_F (selected mode) and, separately, with the fixed t_B (diagnostic column). Class error rates, W->R / R->W / net for CE->soft, soft->AnyLoss-like and the same-model t=0 -> selected change; T1, T7 and their equal-weight mean; paired query-cluster bootstrap (10,000, rng seed 61, thresholds fixed) of the net improvement in both precisions and of their difference (FP32 - BF16). Not mixed with the T1-T7 x 3-seed averages; no Base analysis (no Base FP32).
    python err62_gemma.py --root . --mods analysis --kappa cpmr_experiment/auc61/out/kappa.pkl --fp32 cpmr_experiment/gemma234/out --manifest cpmr_experiment/gemma231/sample_manifest.json --g234 cpmr_experiment/err62/gemma234_results.json --out cpmr_experiment/err62/out"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import err62_lib as L

ap = argparse.ArgumentParser()
for k in ("root", "mods", "kappa", "fp32", "manifest", "g234", "out"):
    ap.add_argument("--" + k, required=True)
a = ap.parse_args()
L.init(a.root, a.mods, a.kappa)
root = Path(a.root)
man = json.load(open(root / a.manifest, encoding="utf-8"))
g234 = json.load(open(root / a.g234, encoding="utf-8"))
THR = {(r["dataset"], r["set"], r["T"], r["loss"]): (r["tB"], r["tF"]) for r in g234["per_checkpoint"]}
TAG = {(c["dataset"], c["T"], c["loss"]): c["tag"] for c in man["checkpoints"]}
OUT = dict(sets={}, meta={})
for ds, setname, sid in (("ESCI", "ESCItest", "ESCI-Gemma"), ("MS", "MS2019", "MS-Gemma-2019"), ("MS", "MS2020", "MS-Gemma-2020")):
    ref = L.load(sid, "CE", 1, 0)
    codemap = {k: c for k, c in zip(ref["ks"], ref["code"])}
    gm_ref = L.groups_of(sid, ref["code"])
    DATA = {}
    for T in (1, 7):
        for loss in ("CE", "SOFT", "ANY"):
            d = json.load(open(root / a.fp32 / f"fp32_{TAG[(ds, T, loss)]}__{setname}.json"))
            keys = [(p[0], p[1]) for p in d["pairs"]]
            s = np.array(d["scores"]); mF = s[:, 0] + s[:, 1] - s[:, 2] - s[:, 3]
            r = L.load(sid, loss, T, 0)
            idx = {k: i for i, k in enumerate(r["ks"])}
            mB = np.array([r["m"][idx[k]] for k in keys])
            y = np.array([bool(p[2]) for p in d["pairs"]])
            code = np.array([codemap[k] for k in keys])
            assert (y == np.array([r["y"][idx[k]] for k in keys])).all()
            DATA[(T, loss)] = dict(mB=mB, mF=mF, y=y, code=code, tB=THR[(ds, setname, T, loss)][0], tF=THR[(ds, setname, T, loss)][1], keys=keys)
    keys0 = DATA[(1, "CE")]["keys"]
    assert all(v["keys"] == keys0 for v in DATA.values())
    y = DATA[(1, "CE")]["y"]; code = DATA[(1, "CE")]["code"]
    gm = L.groups_of(sid, code)
    boot = L.Boot(keys0, y, code, gm)
    P = lambda T, loss, prec, mode: (DATA[(T, loss)]["mB"] > (0.0 if mode == "default" else DATA[(T, loss)]["tB"])) if prec == "BF16" else (DATA[(T, loss)]["mF"] > (0.0 if mode == "default" else (DATA[(T, loss)]["tF"] if mode == "selected" else DATA[(T, loss)]["tB"])))
    S_ = dict(N={g: int(m.sum()) for g, m in gm.items()}, conditions=[], comparisons=[])
    for T in (1, 7):
        for loss in ("CE", "SOFT", "ANY"):
            for prec in ("BF16", "FP32"):
                for mode in (("default", "selected") if prec == "BF16" else ("default", "selected", "fixedB")):
                    p = P(T, loss, prec, mode)
                    cg = L.counts_by_group(p, dict(y=y), gm)
                    S_["conditions"].append(dict(T=T, loss=loss, precision=prec, mode=mode, threshold=(0.0 if mode == "default" else (DATA[(T, loss)]["tB"] if prec == "BF16" or mode == "fixedB" else DATA[(T, loss)]["tF"])), **cg))
    COMP = [("CE->SOFT", ("CE", None), ("SOFT", None)), ("SOFT->ANY", ("SOFT", None), ("ANY", None))]
    def qcs_for(kind, loss_a, loss_b, prec, mode_a, mode_b):
        out = {}
        for T in (1, 7):
            if kind == "thr":
                pa, pb = P(T, loss_a, prec, "default"), P(T, loss_a, prec, mode_b)
            else:
                pa, pb = P(T, loss_a, prec, mode_a), P(T, loss_b, prec, mode_b)
            out[(T, 0)] = (boot.qc(pa, pb, y), L.trans_counts(pa, pb, y, gm))
        return out
    jobs = []
    for name, (la, _), (lb, _) in COMP:
        for mode in ("default", "selected"):
            jobs.append((f"{name} [{mode}]", "loss", la, lb, mode, mode, mode))
        jobs.append((f"{name} [fixed t_B for FP32]", "loss", la, lb, "fixedB", "fixedB", "fixedB"))
    for loss in ("CE", "SOFT", "ANY"):
        jobs.append((f"{loss} t0->selected", "thr", loss, loss, "default", "selected", "selected"))
        jobs.append((f"{loss} t0->fixed t_B (FP32 diagnostic)", "thr", loss, loss, "default", "fixedB", "fixedB"))
    for name, kind, la, lb, ma, mb, mode in jobs:
        ent = dict(comparison=name, kind=kind, mode=mode, scopes={})
        for sc_name, Ts in (("T1", [1]), ("T7", [7]), ("mean T1,T7", [1, 7])):
            res = {}
            for prec in ("BF16", "FP32"):
                fix = (lambda m: "selected" if m == "fixedB" else m) if prec == "BF16" else (lambda m: m)     # BF16 side of the fixed-t_B diagnostic = its own selected (t_B) decision
                q = qcs_for(kind, la, lb, prec, fix(ma), fix(mb))
                pt, bt = boot.agg({k: v[0] for k, v in q.items()}, Ts, (0,))
                res[prec] = (pt, bt, {k: v[1] for k, v in q.items()})
            scd = {}
            for g in gm:
                gg = {}
                for kd in ("WR", "RW", "net", "changed"):
                    ptB, btB = res["BF16"][0][(g, kd)], res["BF16"][1][(g, kd)]
                    ptF, btF = res["FP32"][0][(g, kd)], res["FP32"][1][(g, kd)]
                    ciB, _ = boot.ci(btB, g); ciF, _ = boot.ci(btF, g); ciD, _ = boot.ci(btF - btB, g)
                    gg[kd] = dict(bf16=ptB, fp32=ptF, change=ptF - ptB, ci_bf16=ciB, ci_fp32=ciF, ci_change=ciD, sign_change=bool(np.sign(ptB) != np.sign(ptF)) if kd == "net" else None)
                scd[g] = gg
            ent["scopes"][sc_name] = scd
        S_["comparisons"].append(ent)
    OUT["sets"][setname] = S_
    print(setname, len(S_["comparisons"]), "comparisons", flush=True)
Path(root / a.out).mkdir(parents=True, exist_ok=True)
json.dump(OUT, open(root / a.out / "err62_gemma_fp32.json", "w"), default=float)
print("done")
