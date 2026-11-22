"""CPU re-aggregation of the STORED scores of the submitted main experiments (original instructions T1-T7; Qwen3-4B, Llama-3.1-8B-Instruct, Gemma-3-4B-it; ESCI test300, MS MARCO -> TREC-DL 2019 / 2020;
CE, CE+soft kappa, CE+AnyLoss-style kappa; final 625 updates; seeds 0-2; Base once).
Part 1  seed-level table: per seed the T1-T7 arithmetic mean of kappa(t=0) and of kappa(validation-selected t*), then mean / sample SD (ddof 1) / range over the three seed means, paired soft-CE and AnyLoss-soft differences. T x seed cells are NOT treated as independent repetitions.
Part 2  joint validation x test query bootstrap of the CE+soft kappa threshold gain  kappa_test(t*) - kappa_test(0)  only: 2,000 repetitions, one draw matrix of the validation queries per dataset (ESCI sel200; MS MARCO val200) shared by every model / instruction / seed, one independent draw matrix of the test queries
per evaluation set (ESCI test300; DL2019; DL2020) shared by every condition and by t=0 and t*; queries resampled with replacement to their original number with all their pairs (multiplicities, never pair-level); the threshold of every model x instruction x seed is re-selected on every resampled validation set
with the submitted rule (hard-kappa maximiser over the partitions that remain, 0 always a candidate, ties 0 -> smaller |t| -> smaller t; exact partitions, no grid); test scores never enter the selection; MS MARCO: one val200 threshold per run and repetition applied to both years (separate test draws).
Per repetition: gain per run, equal-weight mean over T1-T7, then over the 3 seeds. Instructions and seeds are not resampled. Undefined draws (no defined kappa on the resampled validation set, or a zero kappa denominator on the resampled test set) are counted and the repetition is dropped for that cell (never set to 0).
PRNG: numpy SeedSequence([395, set_index, role]) with set_index 0 = ESCI, 1 = MS MARCO val200 / DL2019 / DL2020 in the order val, 2019, 2020; role 0 = validation draw, 1 = test draw.
    python joint395.py --root . --mods cpmr_experiment/auc61/mods --kappa cpmr_experiment/auc61/out/kappa.pkl --out <dir>"""
import argparse, glob, gzip, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
for k in ("root", "mods", "kappa", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--nboot", type=int, default=2000)
a = ap.parse_args()
t_start = time.time()
ROOT = Path(a.root); R = ROOT / "cpmr_experiment"; OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R / "analysis"))
import esci204_common as C  # noqa: E402  (best_threshold, kap_vec; identical formulas to the loaders of the original analyses)
NB = a.nboot
SEEDS = (0, 1, 2); TS = tuple(range(1, 8))
CONDS = ("BASE", "CE", "SOFT", "ANY")
CN = {"BASE": "Base", "CE": "CE", "SOFT": "CE+soft kappa", "ANY": "CE+AnyLoss-style kappa"}
kap_vec = C.kap_vec
ld = lambda p: json.load(gzip.open(p, "rt", encoding="utf-8"))


class Val:
    """one checkpoint's validation scores sorted by score with the canonical query index of every row; weighted exact threshold re-selection (algorithm of esci204_stability.best_weighted / err63_lib.Val)"""
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


class E:  # namespace used further below
    Val, Test, QC = Val, Test, C


# ------------------------------------------------------------------ file index with MODEL IDENTITY check (the same tags exist for Qwen3-8B runs; a first-match lookup would mix model sizes)
FILES = {}
for f in glob.glob(str(R / "**" / "*Eval_*.json.gz"), recursive=True):
    FILES.setdefault(os.path.basename(f), []).append(f)
_mn = {}


def model_of(f):
    if f not in _mn:
        _mn[f] = str(ld(f)["meta"]["model_name"]).lower()
    return _mn[f]


def find(name, flt):
    c = [f for f in FILES.get(name, []) if flt in model_of(f)]
    assert c, name
    return c


import pickle
KP = pickle.load(open(a.kappa, "rb"))          # kappa(0), kappa(selected threshold), threshold of the submitted analyses, key (set, cond, T, seed)
DISAMBIGUATED = []


def groups_of_copies(name, flt, reader, tol):
    groups = []                                  # list of [result, [files]]
    for f in find(name, flt):
        res = reader(f)
        for g in groups:
            if g[0][0] == res[0] and np.allclose(g[0][1], res[1], atol=tol, rtol=0):
                g[1].append(f); break
        else:
            groups.append([res, [f]])
    return groups


def pick(groups, check, name):
    if len(groups) == 1:
        return groups[0][0], groups[0][1][0]
    ok = [g for g in groups if check(g[0])]
    assert len(ok) == 1, ("copies differ and the submitted numbers do not select exactly one", name, len(groups), len(ok))
    DISAMBIGUATED.append(dict(name=name, n_distinct_copies=len(groups), kept_dir=os.path.dirname(os.path.relpath(ok[0][1][0], R))))
    return ok[0][0], ok[0][1][0]


def _read_esci(f):
    raw = ld(f)["raw"]
    ks = [(q, x) for q in sorted(raw) for x in sorted(raw[q])]
    m = np.array([raw[q][x][0] + raw[q][x][1] - raw[q][x][3] - raw[q][x][4] for q, x in ks]); y = np.array([raw[q][x][10] >= 2 for q, x in ks])
    return ks, m, y


def esci_arrays(name, flt, key=None, kind="test"):
    def check(res):
        if key is None:
            return False
        ks, m, y = res
        if kind == "test":
            p = m > 0
            return abs(float(C.kap(int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum()))) - KP[key][0]) < 1e-9
        return abs(C.best_threshold(m, y)[0] - KP[key][2]) < 1e-9
    res, f = pick(groups_of_copies(name, flt, _read_esci, 1e-6), check, name)
    return res[0], res[1], res[2], f


SEL_GOLD = {}
for l in open(R / "esci/audit/sel_rows.jsonl", encoding="utf-8"):
    r = json.loads(l); SEL_GOLD[(r["nq"], r["product_id"])] = r["label"] == "Relevant"
ESCI = {"ESCI-Qwen": dict(pre="esciEval_", flt="qwen3-4b", ce="E4_T{T}_s{s}_step625", soft="EQ_mix_T{T}_s{s}_step625", any="EQ_mixany_T{T}_s{s}_step625", p1="esci_t"),
        "ESCI-Llama": dict(pre="esciLlamaEval_", flt="llama-3.1-8b", ce="EL_T{T}_s{s}_step625", soft="ELQ_mix_T{T}_s{s}_step625", any="ELQ_mixany_T{T}_s{s}_step625", p1="esci_llama"),
        "ESCI-Gemma": dict(pre="esciGemmaEval_", flt="gemma-3-4b", ce="GQ_ce_T{T}_s{s}_step625", soft="GQ_mix_T{T}_s{s}_step625", any="GQ_mixany_T{T}_s{s}_step625", p1=None)}
DATA = {}
t0 = time.time()
for sid, f in ESCI.items():
    cells, qlist = {}, None
    for c in CONDS:
        for T in TS:
            for s in ((0,) if c == "BASE" else SEEDS):
                if c == "BASE":
                    te = esci_arrays(f"{f['pre']}Base__T{T}__test300.json.gz", f["flt"], (sid, "BASE", T, 0))
                    if f["p1"] is None:
                        va = esci_arrays(f"{f['pre']}Base__T{T}__sel.json.gz", f["flt"], (sid, "BASE", T, 0), "sel")
                        vks, vm, vy = va[0], va[1], va[2]
                    else:
                        d = ld(R / f["p1"] / f"p1_T{T}" / "p1_raw_cand0.json.gz")
                        mm = d["margins"]
                        vks = [(q, x) for q in sorted(mm) for x in sorted(mm[q])]
                        vm = np.array([mm[q][x][0] + mm[q][x][1] - mm[q][x][2] - mm[q][x][3] for q, x in vks]); vy = np.array([SEL_GOLD[k] for k in vks])
                else:
                    nm = f[{"CE": "ce", "SOFT": "soft", "ANY": "any"}[c]].format(T=T, s=s)
                    te = esci_arrays(f"{f['pre']}{nm}__T{T}__test300.json.gz", f["flt"], (sid, c, T, s))
                    va = esci_arrays(f"{f['pre']}{nm}__T{T}__sel.json.gz", f["flt"], (sid, c, T, s), "sel")
                    vks, vm, vy = va[0], va[1], va[2]
                ql = sorted({q for q, _ in vks}); qlist = qlist or ql; assert ql == qlist and len(ql) == 200
                cells[(c, T, s)] = dict(ks=vks, vm=vm, vy=vy, tks=te[0], tm=te[1], ty=te[2], test_file=te[3])
    DATA[sid] = dict(qlist=qlist, cells=cells, kind="ESCI", draw_key="ESCI")
print("ESCI loaded", f"{time.time() - t0:.0f}s", flush=True)
# MS MARCO (val200 files of the 276 run; test files of the main experiments)
FAMM = {"qwen": ("msmEval_", "msmValEval_", "qwen3-4b"), "llama": ("msmLlamaEval_", "msmValLlamaEval_", "llama-3.1-8b"), "gemma": ("msmGemmaEval_", "msmValGemmaEval_", "gemma-3-4b")}
LETTER = {"CE": "B", "SOFT": "E", "ANY": "M"}


def msm_spec(fam, cond, T, s):
    if cond == "BASE":
        return "Base", f"T{T}"
    l = LETTER[cond]
    if fam == "qwen":
        tag = f"msq_M_T{T}_s{s}" if cond == "ANY" else (f"msm_{l}_s{s}" if T == 1 else f"msq_{l}_T{T}_s{s}")
    elif fam == "llama":
        tag = f"mlq_{l}_T{T}_s{s}"
    else:
        tag = f"gmq_{l}_T{T}_s{s}"
    return tag, f"T{T}"


def _read_msm(f):
    r = ld(f); run, gr = r["run_fp32"], r["grades"]
    ks = [(q, d) for q in sorted(run) for d in sorted(run[q])]
    return ks, np.array([run[q][d] for q, d in ks]), np.array([gr[q][d] >= 2 for q, d in ks])


def msm_run(name, flt, key=None):
    def check(res):
        if key is None:
            return False
        ks, m, y = res
        p = m > 0
        return abs(float(C.kap(int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum()))) - KP[key][0]) < 1e-9
    res, f = pick(groups_of_copies(name, flt, _read_msm, 1e-9), check, name)
    return res[0], res[1], res[2], f, ld(f)


for fam, mdl in (("qwen", "Qwen"), ("llama", "Llama"), ("gemma", "Gemma")):
    pre, vpre, flt = FAMM[fam]
    cells = {2019: {}, 2020: {}}; vqlist = None
    for c in CONDS:
        for T in TS:
            for s in ((0,) if c == "BASE" else SEEDS):
                tag, ins = msm_spec(fam, c, T, s)
                vks, vm, vy, vf, vr = msm_run(f"{vpre}{tag}__{ins}__val200.json.gz", flt)
                assert vr["meta"]["val_pairs_sha256"].startswith("6f2aae0d163d7908") and np.isfinite(vm).all()
                vq = sorted({q for q, _ in vks}); vqlist = vqlist or vq; assert vq == vqlist
                for y in (2019, 2020):
                    t_ = msm_run(f"{pre}{tag}__{ins}__{y}.json.gz", flt, (f"MS-{mdl}-{y}", c, T, s))
                    cells[y][(c, T, s)] = dict(ks=vks, vm=vm, vy=vy, tks=t_[0], tm=t_[1], ty=t_[2], test_file=t_[3])
    for y in (2019, 2020):
        DATA[f"MS-{mdl}-{y}"] = dict(qlist=vqlist, cells=cells[y], kind="MS", draw_key=f"MS-{y}")
print("MS MARCO loaded", f"{time.time() - t0:.0f}s", flush=True)

def best_full(m, y):
    return C.best_threshold(m, y)


# ------------------------------------------------------------------ full-validation thresholds, point kappas, checks
ND = {}
for sid, D in DATA.items():
    qid = {q: i for i, q in enumerate(D["qlist"])}
    tq = None
    for key, e in D["cells"].items():
        e["val"] = E.Val(e["ks"], e["vm"], e["vy"], qid)
        if tq is None:
            tq = sorted({q for q, _ in e["tks"]}); D["tqlist"] = tq; tqi = {q: i for i, q in enumerate(tq)}; D["tqi"] = tqi
        else:
            assert sorted({q for q, _ in e["tks"]}) == tq
        e["test"] = E.Test(e["tm"], e["ty"])
        # per-query sorted class scores of the test set for fast weighted kappa at many thresholds
        rowq = np.array([tqi[q] for q, _ in e["tks"]])
        e["rowq"] = rowq
        tf = best_full(e["vm"], e["vy"])[0]
        tw = e["val"].best(np.ones(len(D["qlist"])))
        assert abs(tf - tw) < 1e-9, (sid, key, tf, tw)          # weighted exact algorithm with unit multiplicities == the submitted selection
        e["t_full"] = tf
        e["k0"] = float(e["test"].kappa([0.0])[0]); e["kfull"] = float(e["test"].kappa([tf])[0])
    D["nq_val"], D["nq_test"] = len(D["qlist"]), len(D["tqlist"])
    print(sid, "val queries", D["nq_val"], "test queries", D["nq_test"], "cells", len(D["cells"]), flush=True)

# reproduction of the submitted tables (point estimates of the manuscript, seed-mean of T1-T7 mean)
TAB = {  # (sid, cond) -> (kappa(0), kappa(t*)) as printed in the submitted tables
    ("ESCI-Qwen", "BASE"): (0.3135, 0.3439), ("ESCI-Qwen", "CE"): (0.3501, 0.3579), ("ESCI-Qwen", "SOFT"): (0.3818, 0.3764), ("ESCI-Qwen", "ANY"): (0.3778, 0.3738),
    ("ESCI-Llama", "BASE"): (0.2230, 0.2976), ("ESCI-Llama", "CE"): (0.3365, 0.3324), ("ESCI-Llama", "SOFT"): (0.4032, 0.3996), ("ESCI-Llama", "ANY"): (0.3950, 0.3920),
    ("ESCI-Gemma", "BASE"): (0.3080, 0.3065), ("ESCI-Gemma", "CE"): (0.2981, 0.3018), ("ESCI-Gemma", "SOFT"): (0.3350, 0.3342), ("ESCI-Gemma", "ANY"): (0.3315, 0.3304),
    ("MS-Qwen-2019", "BASE"): (0.2676, 0.4326), ("MS-Qwen-2019", "CE"): (0.4910, 0.4668), ("MS-Qwen-2019", "SOFT"): (0.5151, 0.4769), ("MS-Qwen-2019", "ANY"): (0.5151, 0.4726),
    ("MS-Qwen-2020", "BASE"): (0.1631, 0.2824), ("MS-Qwen-2020", "CE"): (0.3720, 0.3403), ("MS-Qwen-2020", "SOFT"): (0.3898, 0.3473), ("MS-Qwen-2020", "ANY"): (0.3904, 0.3459),
    ("MS-Llama-2019", "BASE"): (0.3233, 0.3789), ("MS-Llama-2019", "CE"): (0.4343, 0.4436), ("MS-Llama-2019", "SOFT"): (0.4479, 0.4528), ("MS-Llama-2019", "ANY"): (0.4467, 0.4595),
    ("MS-Llama-2020", "BASE"): (0.2415, 0.3157), ("MS-Llama-2020", "CE"): (0.3223, 0.3311), ("MS-Llama-2020", "SOFT"): (0.3406, 0.3477), ("MS-Llama-2020", "ANY"): (0.3426, 0.3614),
    ("MS-Gemma-2019", "BASE"): (0.3302, 0.3375), ("MS-Gemma-2019", "CE"): (0.3133, 0.3568), ("MS-Gemma-2019", "SOFT"): (0.3578, 0.3725), ("MS-Gemma-2019", "ANY"): (0.3589, 0.3775),
    ("MS-Gemma-2020", "BASE"): (0.1970, 0.2033), ("MS-Gemma-2020", "CE"): (0.1852, 0.2359), ("MS-Gemma-2020", "SOFT"): (0.2281, 0.2559), ("MS-Gemma-2020", "ANY"): (0.2310, 0.2621)}
PART1, REPRO, CSVROWS = [], [], []
for sid, D in DATA.items():
    for c in CONDS:
        seeds = (0,) if c == "BASE" else SEEDS
        per_seed = {}
        for mode, key in (("t0", "k0"), ("tstar", "kfull")):
            per_seed[mode] = [float(np.mean([D["cells"][(c, T, s)][key] for T in TS])) for s in seeds]
        for T in TS:
            for s in seeds:
                e = D["cells"][(c, T, s)]
                CSVROWS.append(f"{sid},{c},{T},{s},{e['t_full']:.9f},{e['k0']:.9f},{e['kfull']:.9f}")
        k0m, ksm = float(np.mean(per_seed["t0"])), float(np.mean(per_seed["tstar"]))
        a0, a1 = TAB[(sid, c)]
        REPRO.append(dict(set=sid, cond=c, k0=round(k0m, 4), k0_table=a0, kstar=round(ksm, 4), kstar_table=a1, match=abs(round(k0m, 4) - a0) < 1.1e-4 and abs(round(ksm, 4) - a1) < 1.1e-4))
        PART1.append(dict(set=sid, cond=c, seeds=list(seeds), seed_means_t0=per_seed["t0"], seed_means_tstar=per_seed["tstar"], mean_t0=k0m, mean_tstar=ksm,
                          sd_t0=float(np.std(per_seed["t0"], ddof=1)) if len(seeds) > 1 else None, sd_tstar=float(np.std(per_seed["tstar"], ddof=1)) if len(seeds) > 1 else None,
                          range_t0=[min(per_seed["t0"]), max(per_seed["t0"])], range_tstar=[min(per_seed["tstar"]), max(per_seed["tstar"])]))
open(OUT / "per_instruction_seed_kappa.csv", "w").write("set,cond,T,seed,t_star_full_validation,kappa_t0,kappa_tstar\n" + "\n".join(CSVROWS) + "\n")
PAIRED = []
for sid in DATA:
    P = {p["cond"]: p for p in PART1 if p["set"] == sid}
    for nm, (cA, cB) in {"soft-CE": ("SOFT", "CE"), "AnyLoss-soft": ("ANY", "SOFT")}.items():
        for mode in ("t0", "tstar"):
            d = [x - y for x, y in zip(P[cA][f"seed_means_{mode}"], P[cB][f"seed_means_{mode}"])]
            PAIRED.append(dict(set=sid, contrast=nm, threshold="t=0" if mode == "t0" else "validation t*", mean=float(np.mean(d)), sd=float(np.std(d, ddof=1)), seed_diffs=d, range=[min(d), max(d)], positive_seeds=int(sum(x > 0 for x in d))))
print("part 1 done; reproduction all match:", all(r["match"] for r in REPRO), flush=True)
assert all(r["match"] for r in REPRO), [r for r in REPRO if not r["match"]]

# ------------------------------------------------------------------ Part 2: joint bootstrap of the CE+soft kappa threshold gain
def draws(set_index, role, n):
    ss = np.random.SeedSequence([395, set_index, role]); rng = np.random.default_rng(ss)
    return np.stack([np.bincount(i, minlength=n) for i in rng.integers(0, n, size=(NB, n))]).astype(float)


WV, WT = {}, {}
WV["ESCI"] = draws(0, 0, DATA["ESCI-Qwen"]["nq_val"]); WT["ESCI"] = draws(0, 1, DATA["ESCI-Qwen"]["nq_test"])
WV["MS"] = draws(1, 0, DATA["MS-Qwen-2019"]["nq_val"]); WT["MS-2019"] = draws(1, 1, DATA["MS-Qwen-2019"]["nq_test"]); WT["MS-2020"] = draws(1, 2, DATA["MS-Qwen-2020"]["nq_test"])
for d in ("ESCI", "MS"):
    assert WV[d].shape == (NB, 200)


def test_counts_at(e, ts, tqi_n):
    """per test query, tp / fp / fn / tn at every threshold of ts -> arrays (nq, len(ts)); exact (counts of rows with m > t)"""
    if "qs" not in e:
        rowq, m, y = e["rowq"], e["tm"], e["ty"]
        e["qs"] = [(np.sort(m[(rowq == q) & y]), np.sort(m[(rowq == q) & ~y])) for q in range(tqi_n)]
    ts = np.asarray(ts, float)
    tp = np.stack([len(ps) - np.searchsorted(ps, ts, side="right") for ps, ns in e["qs"]]); fp = np.stack([len(ns) - np.searchsorted(ns, ts, side="right") for ps, ns in e["qs"]])
    P = np.array([len(ps) for ps, ns in e["qs"]])[:, None]; N = np.array([len(ns) for ps, ns in e["qs"]])[:, None]
    return tp, fp, P - tp, N - fp


def gain_draws(D, wv, wt):
    """per (T, seed) of the SOFT condition: thresholds re-selected on the resampled validation, gain kappa_test(t_b) - kappa_test(0) on the resampled test; returns array (21, NB) and counts of undefined draws"""
    out, bad_val, bad_test = [], 0, 0
    for T in TS:
        for s in SEEDS:
            e = D["cells"][("SOFT", T, s)]
            tb = np.array([e["val"].best(wv[b]) for b in range(NB)])
            ok_v = np.isfinite(tb); bad_val += int((~ok_v).sum())
            tt = np.where(ok_v, tb, 0.0)
            tp, fp, fn, tn = test_counts_at(e, tt, D["nq_test"])
            tp0, fp0, fn0, tn0 = test_counts_at(e, [0.0], D["nq_test"])
            W_ = wt
            k_t = kap_vec(np.einsum("bq,qb->b", W_, tp), np.einsum("bq,qb->b", W_, fp), np.einsum("bq,qb->b", W_, fn), np.einsum("bq,qb->b", W_, tn))
            k_0 = kap_vec(W_ @ tp0[:, 0], W_ @ fp0[:, 0], W_ @ fn0[:, 0], W_ @ tn0[:, 0])
            g = k_t - k_0
            g[~ok_v] = np.nan
            bad_test += int((~np.isfinite(g) & ok_v).sum())
            out.append(g)
    return np.stack(out), bad_val, bad_test


# implementation check on a small synthetic dataset: explicit row replication versus the weighted pipeline
def selftest():
    rng = np.random.default_rng(7)
    nqv, nqt, rpq = 12, 9, 6
    def make(nq):
        ks, m, y = [], [], []
        for q in range(nq):
            for r in range(rpq):
                lab = rng.random() < 0.45
                ks.append((f"q{q:02d}", f"d{r}")); y.append(lab); m.append(float(np.round(rng.normal(0.9 if lab else -0.6, 1.4), 1)))   # rounded: ties on purpose
        return ks, np.array(m), np.array(y)
    vks, vm, vy = make(nqv); tks, tm, ty = make(nqt)
    qid = {q: i for i, q in enumerate(sorted({q for q, _ in vks}))}
    val = E.Val(vks, vm, vy, qid)
    tq = sorted({q for q, _ in tks}); tqi = {q: i for i, q in enumerate(tq)}
    e = dict(rowq=np.array([tqi[q] for q, _ in tks]), tm=tm, ty=ty)
    wv = np.stack([np.bincount(i, minlength=nqv) for i in rng.integers(0, nqv, size=(40, nqv))]).astype(float)
    wt = np.stack([np.bincount(i, minlength=nqt) for i in rng.integers(0, nqt, size=(40, nqt))]).astype(float)
    worst = 0.0
    for b in range(40):
        rows_v = np.repeat(np.arange(len(vm)), wv[b][np.array([qid[q] for q, _ in vks])].astype(int))
        t_dir, _ = E.QC.best_threshold(vm[rows_v], vy[rows_v])
        t_w = val.best(wv[b])
        worst = max(worst, abs(t_dir - t_w))
        rows_t = np.repeat(np.arange(len(tm)), wt[b][e["rowq"]].astype(int))
        def kd(t):
            p = tm[rows_t] > t; yy = ty[rows_t]
            return float(E.QC.kap(int((p & yy).sum()), int((p & ~yy).sum()), int((~p & yy).sum()), int((~p & ~yy).sum())))
        g_dir = kd(t_dir) - kd(0.0)
        tp, fp, fn, tn = test_counts_at(e, [t_w], nqt); tp0, fp0, fn0, tn0 = test_counts_at(e, [0.0], nqt)
        kt = float(kap_vec(wt[b] @ tp[:, 0], wt[b] @ fp[:, 0], wt[b] @ fn[:, 0], wt[b] @ tn[:, 0])); k0_ = float(kap_vec(wt[b] @ tp0[:, 0], wt[b] @ fp0[:, 0], wt[b] @ fn0[:, 0], wt[b] @ tn0[:, 0]))
        worst = max(worst, abs((kt - k0_) - g_dir))
    return worst


st_worst = selftest()
print("synthetic self test: max |difference| between explicit replication and the weighted pipeline", st_worst, flush=True)
assert st_worst < 1e-9

PART2, DRAWS = [], {}
for sid, D in DATA.items():
    wv = WV["ESCI"] if D["kind"] == "ESCI" else WV["MS"]
    wt = WT["ESCI"] if D["kind"] == "ESCI" else WT[D["draw_key"]]
    g, bad_val, bad_test = gain_draws(D, wv, wt)
    valid = np.isfinite(g).all(axis=0)
    gm = np.nanmean(g, axis=0) if False else g.mean(axis=0)        # (21 cells: T-major, seed-minor) -> equal-weight over T and seeds == mean over all 21 cells because the grid is rectangular
    gm = gm[valid]
    point = float(np.mean([[D["cells"][("SOFT", T, s)]["kfull"] - D["cells"][("SOFT", T, s)]["k0"] for s in SEEDS] for T in TS]))
    per_seed_point = [float(np.mean([D["cells"][("SOFT", T, s)]["kfull"] - D["cells"][("SOFT", T, s)]["k0"] for T in TS])) for s in SEEDS]
    PART2.append(dict(set=sid, point_estimate=point, seed_point_estimates=per_seed_point, joint_mean=float(gm.mean()), joint_sd=float(gm.std(ddof=1)), p2_5=float(np.percentile(gm, 2.5)), p97_5=float(np.percentile(gm, 97.5)), positive_draw_share=float((gm > 0).mean()),
                      n_draws=NB, n_valid=int(valid.sum()), undefined_validation_selections=bad_val, undefined_test_kappas=bad_test, dropped_draws=int((~valid).sum())))
    DRAWS[sid] = gm
    print(sid, PART2[-1], flush=True)
np.savez_compressed(OUT / "joint_draws.npz", **{k: v for k, v in DRAWS.items()})
# auxiliary intervals from existing analyses (same contrast, same protocol) when present
AUX = {}
try:
    pv = json.load(open(R / "msm_val/summary/po_val_results.json"))["fam"]
    for mdl, fam in (("Qwen", "qwen"), ("Llama", "llama"), ("Gemma", "gemma")):
        for rec in pv[fam]["val_resample"]:
            if rec.get("variant") == "ORIG" and rec.get("cond") == "SOFT" and "kb_minus_t0" in rec:
                AUX[f"MS-{mdl}-{rec['year']}"] = dict(validation_only_500=dict(mean=rec["kb_minus_t0"]["mean"], p2_5=rec["kb_minus_t0"]["p2_5"], p97_5=rec["kb_minus_t0"]["p97_5"], source="shared/276 val_resample, default_rng(63), 500 draws, thresholds re-selected, test fixed"))
except Exception as ex:
    AUX["msm_error"] = str(ex)
try:
    e63 = json.load(open(R / "err63/out/err63_results.json"))["sets"]
    for mdl in ("Qwen", "Llama", "Gemma"):
        s_ = e63[f"ESCI-{mdl}"]["specs"]["reuse_rng1"]["cond_stats"]["T1-T7|SOFT"]["delta_tb_t0"]
        AUX[f"ESCI-{mdl}"] = dict(validation_only_500=dict(mean=s_["mean"], p2_5=s_["p2_5"], p97_5=s_["p97_5"], source="shared/242 err63_results, default_rng(1), 500 draws, thresholds re-selected, test fixed"))
except Exception as ex:
    AUX["esci_error"] = str(ex)
# test-only intervals of the submitted Table D (soft kappa threshold gain, original instructions, exploratory 95%, validation threshold fixed): ESCI and DL2019 only
AUX_TESTONLY = {"ESCI-Qwen": [-0.0054, -0.0134, 0.0026], "ESCI-Llama": [-0.0037, -0.0068, -0.0006], "ESCI-Gemma": [-0.0008, -0.0031, 0.0014], "MS-Qwen-2019": [-0.0382, -0.0552, -0.0208], "MS-Llama-2019": [0.0049, 0.0003, 0.0091], "MS-Gemma-2019": [0.0147, 0.0011, 0.0266]}
for k, v in AUX_TESTONLY.items():
    AUX.setdefault(k, {})["test_only_submitted_table_D"] = dict(point=v[0], lo=v[1], hi=v[2])
res = dict(disambiguated_copies=DISAMBIGUATED, part1=PART1, paired=PAIRED, reproduction=REPRO, part2=PART2, aux=AUX, selftest_max_abs_diff=st_worst, prng="numpy SeedSequence([395, set_index, role]); set_index 0 ESCI, 1 MS MARCO; role 0 validation, 1 test (MS MARCO: val draw role 0, DL2019 role 1, DL2020 role 2)", nboot=NB,
           inputs=dict(esci_val=len(DATA["ESCI-Qwen"]["qlist"]), esci_test=DATA["ESCI-Qwen"]["nq_test"], msm_val=DATA["MS-Qwen-2019"]["nq_val"], dl2019=DATA["MS-Qwen-2019"]["nq_test"], dl2020=DATA["MS-Qwen-2020"]["nq_test"]), cpu_seconds=time.time() - t_start)
(OUT / "joint395_results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print("done", f"{time.time() - t_start:.0f}s")
