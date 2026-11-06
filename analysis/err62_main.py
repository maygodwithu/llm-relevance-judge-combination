"""Chapter 6.2 error-rate and decision-transition analysis of the stored scores (experiment log 237): nine evaluation sets (ESCI test300 of Qwen / Llama / Gemma; TREC-DL 2019 and 2020 of each model), original texts T1-T7, SFT seeds 0/1/2, final 625 updates, own instruction; PO (ESCI Qwen / Llama).
Decisions m > t with t = 0 (default) or the original selected threshold (ESCI validation hard-kappa maximum, MS MARCO CROSS). Per (instruction, seed): class error rates and counts; aggregation = equal-weight mean over the instructions of the scope, then mean over seeds (Base has no training seed). Transitions are counted on the same examples; every rate is divided by the group size N_g. Bootstrap: 10,000 paired query-cluster draws (np.random.default_rng(61)), the same draws for every comparison / instruction / seed of an evaluation set, thresholds fixed, group denominators recomputed per draw, draws with an empty group excluded for that group.
    python err62_main.py --root . --mods analysis --kappa cpmr_experiment/auc61/out/kappa.pkl --out cpmr_experiment/err62/out"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import err62_lib as L

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--mods", required=True); ap.add_argument("--kappa", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
t_start = time.time()
L.init(a.root, a.mods, a.kappa)
SETS = ["ESCI-Qwen", "ESCI-Llama", "ESCI-Gemma", "MS-Qwen-2019", "MS-Qwen-2020", "MS-Llama-2019", "MS-Llama-2020", "MS-Gemma-2019", "MS-Gemma-2020"]
ALL7 = list(range(1, 8))
AUX = {"ESCI-Qwen": list(range(2, 8)), "ESCI-Llama": list(range(4, 8))}
seeds_of = lambda c: (0,) if c in ("BASE", "POBASE") else L.SEEDS
CONDS = ("BASE", "CE", "SOFT", "ANY", "POBASE", "POCE", "POSOFT")
fails = []
OUT = dict(sets={}, checks=[], meta={})
for sid in SETS:
    t0 = time.time()
    REC, THR = {}, {}
    ref = None
    for c in CONDS:
        for T in ALL7:
            for s in seeds_of(c):
                r = L.load(sid, c, T, s)
                if r is None:
                    continue
                if ref is None:
                    ref = r
                assert r["ks"] == ref["ks"] and (r["y"] == ref["y"]).all() and (r["code"] == ref["code"]).all(), (sid, c, T, s)
                assert np.isfinite(r["m"]).all()
                REC[(c, T, s)] = r
                THR[(c, T, s)] = L.sel_threshold(sid, c, T, s)
    gm = L.groups_of(sid, ref["code"]); y = ref["y"]
    Ng = {g: int(mk.sum()) for g, mk in gm.items()}
    if sid.startswith("ESCI"):
        assert (Ng["E"], Ng["S"], Ng["C"], Ng["I"]) == (2857, 1954, 226, 1041), Ng
        assert Ng["S"] + Ng["C"] + Ng["I"] == Ng["SCI"] and Ng["E"] + Ng["SCI"] == Ng["ALL"]
    boot = L.Boot(ref["ks"], y, ref["code"], gm)
    pred = lambda c, T, s, mode: L.predictions(REC[(c, T, 0 if c in ("BASE", "POBASE") else s)], THR[(c, T, 0 if c in ("BASE", "POBASE") else s)], mode)
    S_ = dict(N=Ng, n_rows=len(y), n_queries=boot.nq, invalid_draws={g: int((~boot.valid[g]).sum()) for g in gm}, conditions=[], cells=[], comparisons=[], thr_direction=[])
    # ---------------- condition records
    for (c, T, s), r in sorted(REC.items()):
        for mode in ("default", "selected"):
            p = L.predictions(r, THR[(c, T, s)], mode)
            cg = L.counts_by_group(p, r, gm)
            S_["conditions"].append(dict(cond=c, T=T, seed=s, mode=mode, threshold=(0.0 if mode == "default" else float(THR[(c, T, s)])), **cg))
    # ---------------- comparisons
    comps = []
    for mode in ("default", "selected"):
        comps += [(f"BASE->CE [{mode}]", ("BASE", mode), ("CE", mode), "orig", mode), (f"CE->SOFT [{mode}]", ("CE", mode), ("SOFT", mode), "orig", mode), (f"SOFT->ANY [{mode}]", ("SOFT", mode), ("ANY", mode), "orig", mode)]
    for c in ("BASE", "CE", "SOFT", "ANY"):
        comps.append((f"{c} t0->selected", (c, "default"), (c, "selected"), "thr", "default->selected"))
    if sid in ("ESCI-Qwen", "ESCI-Llama"):
        for mode in ("default", "selected"):
            comps += [(f"BASE->POBASE [{mode}]", ("BASE", mode), ("POBASE", mode), "po", mode), (f"CE->POCE [{mode}]", ("CE", mode), ("POCE", mode), "po", mode), (f"SOFT->POSOFT [{mode}]", ("SOFT", mode), ("POSOFT", mode), "po", mode)]
    for name, A, Bc, kind, mode in comps:
        ca, cb = A[0], Bc[0]
        seeds = L.SEEDS if (len(seeds_of(ca)) > 1 or len(seeds_of(cb)) > 1) else (0,)
        Ts_all = [T for T in ALL7 if all((cc, T, ss) in REC for cc in (ca, cb) for ss in seeds_of(cc))]
        qcs = {}
        for T in Ts_all:
            for s in seeds:
                pa = pred(ca, T, s, A[1]); pb = pred(cb, T, s, Bc[1])
                tc = L.trans_counts(pa, pb, y, gm)
                ea = L.counts_by_group(pa, REC[(ca, T, 0 if ca in ("BASE", "POBASE") else s)], gm)["groups"]
                eb = L.counts_by_group(pb, REC[(cb, T, 0 if cb in ("BASE", "POBASE") else s)], gm)["groups"]
                for g, v in tc.items():                                               # identities (original precision of the integer counts)
                    ok = (v["RR"] + v["WW"] + v["WR"] + v["RW"] == v["N"] and ea[g]["errors"] == v["WW"] + v["WR"] and eb[g]["errors"] == v["WW"] + v["RW"] and ea[g]["errors"] - eb[g]["errors"] == v["WR"] - v["RW"])
                    if not ok:
                        fails.append((sid, name, T, s, g))
                if kind == "thr":                                                       # a threshold change is one-directional
                    t = float(THR[(ca, T, 0 if ca == "BASE" else s)])
                    wrong_dir = {"gold_pos": tc.get("E", tc.get("Rel")), "gold_neg": tc.get("SCI", tc.get("Irr"))}
                    if t > 0:
                        bad = wrong_dir["gold_pos"]["WR"] + wrong_dir["gold_neg"]["RW"]
                    elif t < 0:
                        bad = wrong_dir["gold_pos"]["RW"] + wrong_dir["gold_neg"]["WR"]
                    else:
                        bad = sum(v["WR"] + v["RW"] for v in tc.values())
                    S_["thr_direction"].append(dict(comparison=name, T=T, seed=s, threshold=t, violations=int(bad), sign="+" if t > 0 else ("-" if t < 0 else "0")))
                S_["cells"].append(dict(comparison=name, mode=mode, T=T, seed=s, groups=tc))
                qcs[(T, s)] = boot.qc(pa, pb, y)
        scopes = []
        if kind == "po":
            if sid == "ESCI-Qwen" or sid == "ESCI-Llama":
                scopes.append(("main T1-T3", [T for T in (1, 2, 3) if T in Ts_all]))
                if ca in ("BASE", "CE"):
                    scopes.append(("aux T1-T7", Ts_all))
        else:
            scopes.append(("main T1-T7", Ts_all))
            if sid in AUX:
                scopes.append((f"aux T{AUX[sid][0]}-T{AUX[sid][-1]}", [T for T in AUX[sid] if T in Ts_all]))
            elif sid.startswith("MS"):
                scopes.append(("aux T2-T7", [T for T in range(2, 8) if T in Ts_all]))
        for sname, Ts in scopes:
            if kind == "po" and sname.startswith("main") and len(Ts) != 3:
                continue
            if not Ts:
                continue
            pt, bt = boot.agg(qcs, Ts, seeds)
            rec = dict(comparison=name, mode=mode, kind=kind, scope=sname, instructions=Ts, seeds=list(seeds), groups={})
            for g in gm:
                gg = {}
                for kd in ("WR", "RW", "net", "changed", "WW", "RR"):
                    ci, nv = boot.ci(bt[(g, kd)], g)
                    gg[kd] = dict(point=pt[(g, kd)], ci95=ci)
                gg["valid_draws"] = int(boot.valid[g].sum())
                gg["N"] = Ng[g]
                rec["groups"][g] = gg
            S_["comparisons"].append(rec)
    # ---------------- aggregated condition tables (scope = instruction mean then seed mean)
    S_["cond_tables"] = []
    for c in CONDS:
        for mode in ("default", "selected"):
            for sname, Ts in (("T1-T7", ALL7), ("T1-T3", [1, 2, 3]), ("T2-T7", list(range(2, 8))), ("T4-T7", list(range(4, 8)))):
                if not all((c, T, s) in REC for T in Ts for s in seeds_of(c)):
                    continue
                def cell(T, s):
                    r = REC[(c, T, s)]
                    return L.counts_by_group(L.predictions(r, THR[(c, T, s)], mode), r, gm)
                agg = {}
                per = {s: [cell(T, s) for T in Ts] for s in seeds_of(c)}
                for g in gm:
                    agg[g] = dict(rate=float(np.mean([np.mean([x["groups"][g]["rate"] for x in per[s]]) for s in per])), errors=float(np.mean([np.mean([x["groups"][g]["errors"] for x in per[s]]) for s in per])), N=Ng[g])
                for kx in ("accuracy", "pred_rel", "kappa", "tp", "fp", "fn", "tn"):
                    agg[kx] = float(np.mean([np.mean([x[kx] for x in per[s]]) for s in per]))
                agg["total_errors"] = agg["fp"] + agg["fn"]
                S_["cond_tables"].append(dict(cond=c, mode=mode, scope=sname, **agg))
    OUT["sets"][sid] = S_
    print(f"{sid}: {len(REC)} score vectors, {len(S_['comparisons'])} comparison scopes, {time.time() - t0:.0f}s", flush=True)
OUT["checks"].append(dict(name="transition identities (RR+WW+WR+RW=N_g; errors_before=WW+WR; errors_after=WW+RW; difference=WR-RW) at the integer counts of every comparison x instruction x seed x group", failures=len(fails), items=[list(map(str, f)) for f in fails[:20]]))
tv = [(sid, v) for sid, S_ in OUT["sets"].items() for v in S_["thr_direction"]]
OUT["checks"].append(dict(name="a threshold change moves decisions in one direction only (t>0: Relevant->Irrelevant; t<0: reverse)", n=len(tv), violations=sum(v["violations"] for _, v in tv),
                          n_t_positive=sum(1 for _, v in tv if v["sign"] == "+"), n_t_negative=sum(1 for _, v in tv if v["sign"] == "-"), n_t_zero=sum(1 for _, v in tv if v["sign"] == "0")))
OUT["meta"] = dict(seconds=time.time() - t_start, B=L.B_DRAWS, rng_seed=L.RNG_SEED)
Path(a.out).mkdir(parents=True, exist_ok=True)
json.dump(OUT, open(Path(a.out) / "err62_results.json", "w"), default=float)
print("done", OUT["checks"], f"{time.time() - t_start:.0f}s")
