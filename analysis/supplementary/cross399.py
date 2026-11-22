"""Cross-year threshold calibration of the submitted MS MARCO main experiments, CPU re-aggregation of STORED scores only (no GPU / inference / training).
Original T1-T7, Qwen3-4B / Llama-3.1-8B-Instruct / Gemma-3-4B-it, Base / CE / CE+soft kappa / CE+AnyLoss-style kappa, final 625 updates, seeds 0-2 (Base once); TREC-DL 2019 (42 q / 1,102 pairs) and 2020 (54 q / 1,432 pairs), grades 2/3 = Relevant.
Three thresholds per run: t=0; val200 t* (submitted protocol, hard-kappa maximiser on the 400 val200 pairs of the same run, applied unchanged to both years); CROSS = hard-kappa maximiser over ALL judged pairs of the OTHER year of the same run
(2020 -> 2019 test, 2019 -> 2020 test; rule: all partitions + 0, ties 0 -> smaller |t| -> smaller t); the test year's labels/scores never enter its own threshold; the threshold is applied numerically (not re-tuned).
Aggregation as the submitted tables: per seed the T1-T7 mean, then mean / sample SD (ddof 1) over the 3 seeds (Base: seed 0). Reproduction: t=0 and val200 cells vs the submitted table (4 decimals) and CROSS kappa / threshold vs the stored kappa.pkl of the earlier CROSS analysis (230)."""
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



DATA = {}
t0 = time.time()
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


# ================================================================== analysis
import csv, re, unicodedata


def best_full(m, y):
    return C.best_threshold(m, y)[0]


class Tst:
    def __init__(self, m, y):
        self.ps, self.ns = np.sort(m[y]), np.sort(m[~y]); self.P, self.N = len(self.ps), len(self.ns)

    def kappa(self, t):
        tp = self.P - np.searchsorted(self.ps, t, side="right"); fp = self.N - np.searchsorted(self.ns, t, side="right")
        return float(kap_vec(np.array([tp]), np.array([fp]), np.array([self.P - tp]), np.array([self.N - fp]))[0])


MODELS = ("Qwen", "Llama", "Gemma")
ROWS = []            # one per (model, test year, cond, T, seed)
CHK = dict(n=0, k0_vs_pkl=0, kcross_vs_pkl=0, tcross_vs_pkl=0, mism=[])
EVALN = {}
for mdl in MODELS:
    for y, yo in ((2019, 2020), (2020, 2019)):
        cells, other = DATA[f"MS-{mdl}-{y}"]["cells"], DATA[f"MS-{mdl}-{yo}"]["cells"]
        for key, e in cells.items():
            o = other[key]
            c, T, s = key
            # identical checkpoint / identical val scores for both years
            assert np.array_equal(e["vm"], o["vm"]) and e["ks"] == o["ks"]
            assert sorted({q for q, _ in e["tks"]}) != sorted({q for q, _ in o["tks"]})
            EVALN[(mdl, y)] = (len({q for q, _ in e["tks"]}), len(e["tks"]))
            test, cal = Tst(e["tm"], e["ty"]), (o["tm"], o["ty"])
            tval = best_full(e["vm"], e["vy"])
            tcross = best_full(*cal)
            k0, kval, kcross = test.kappa(0.0), test.kappa(tval), test.kappa(tcross)
            pk = KP[(f"MS-{mdl}-{y}", c, T, s)]
            CHK["n"] += 1
            ok0, okk, okt = abs(k0 - pk[0]) < 1e-9, abs(kcross - pk[1]) < 1e-9, abs(tcross - pk[2]) < 1e-9
            CHK["k0_vs_pkl"] += ok0; CHK["kcross_vs_pkl"] += okk; CHK["tcross_vs_pkl"] += okt
            if not (ok0 and okk and okt):
                CHK["mism"].append(dict(model=mdl, year=y, cond=c, T=T, seed=s, k0=k0, k0_pkl=pk[0], kcross=kcross, kcross_pkl=pk[1], tcross=tcross, tcross_pkl=pk[2]))
            ROWS.append(dict(model=mdl, year=y, cond=c, T=T, seed=s, t_val=tval, t_cross=tcross, k0=k0, kval=kval, kcross=kcross,
                             pos_rate_val=float((e["tm"] > tval).mean()), pos_rate_cross=float((e["tm"] > tcross).mean()), pos_rate_0=float((e["tm"] > 0).mean()), gold_rel=float(e["ty"].mean())))
print("rows", len(ROWS), "check vs kappa.pkl:", {k: v for k, v in CHK.items() if k != "mism"}, flush=True)
assert EVALN[("Qwen", 2019)] == (42, 1102) and EVALN[("Qwen", 2020)] == (54, 1432)
assert all(v == EVALN[(m, y)] for (m, y), v in EVALN.items() if True) and all(EVALN[(m, 2019)] == (42, 1102) and EVALN[(m, 2020)] == (54, 1432) for m in MODELS)

# --- seed-level aggregation (per seed: T1-T7 mean; then mean / SD over the three seed means)
CONDS_ = ("BASE", "CE", "SOFT", "ANY")
AGG = {}
for mdl in MODELS:
    for y in (2019, 2020):
        for c in CONDS_:
            seeds = (0,) if c == "BASE" else SEEDS
            d = {}
            for k in ("k0", "kval", "kcross", "t_val", "t_cross", "pos_rate_0", "pos_rate_val", "pos_rate_cross"):
                d[k] = [float(np.mean([r[k] for r in ROWS if r["model"] == mdl and r["year"] == y and r["cond"] == c and r["seed"] == s])) for s in seeds]
            AGG[(mdl, y, c)] = d
# reproduction of the submitted tables (t=0 and val200 t*)
REP = []
for mdl in MODELS:
    for y in (2019, 2020):
        for c in CONDS_:
            d = AGG[(mdl, y, c)]; a0, a1 = TAB[(f"MS-{mdl}-{y}", c)]
            k0m, kvm = float(np.mean(d["k0"])), float(np.mean(d["kval"]))
            REP.append(dict(model=mdl, year=y, cond=c, k0=round(k0m, 4), k0_table=a0, kval=round(kvm, 4), kval_table=a1, match=abs(round(k0m, 4) - a0) < 1.1e-4 and abs(round(kvm, 4) - a1) < 1.1e-4))
print("submitted-table reproduction (t=0, val200) all match:", all(r["match"] for r in REP), flush=True)
assert all(r["match"] for r in REP), [r for r in REP if not r["match"]]


def ms(v):
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else None)


SUM = []
for mdl in MODELS:
    for y in (2019, 2020):
        for c in CONDS_:
            d = AGG[(mdl, y, c)]
            row = dict(model=mdl, test_year=y, cond=c, calibration_year=2020 if y == 2019 else 2019)
            for k in ("k0", "kval", "kcross"):
                row[k], row[k + "_sd"] = ms(d[k])
            for nm, (a_, b_) in {"cross_minus_val": ("kcross", "kval"), "cross_minus_0": ("kcross", "k0"), "val_minus_0": ("kval", "k0")}.items():
                diffs = [x - z for x, z in zip(d[a_], d[b_])]
                row[nm], row[nm + "_sd"] = ms(diffs); row[nm + "_pos_seeds"] = int(sum(v > 0 for v in diffs)); row[nm + "_n_seeds"] = len(diffs)
            row["t_val_mean"] = float(np.mean(d["t_val"])); row["t_cross_mean"] = float(np.mean(d["t_cross"]))
            row["pos_rate_0"], row["pos_rate_val"], row["pos_rate_cross"] = (float(np.mean(d[k])) for k in ("pos_rate_0", "pos_rate_val", "pos_rate_cross"))
            SUM.append(row)
PAIR = []
for mdl in MODELS:
    for y in (2019, 2020):
        for nm, (cA, cB) in {"soft-CE": ("SOFT", "CE"), "AnyLoss-soft": ("ANY", "SOFT")}.items():
            for k, lab in (("k0", "t=0"), ("kval", "val200 t*"), ("kcross", "CROSS")):
                dd = [x - z for x, z in zip(AGG[(mdl, y, cA)][k], AGG[(mdl, y, cB)][k])]
                PAIR.append(dict(model=mdl, test_year=y, contrast=nm, threshold=lab, mean=float(np.mean(dd)), sd=float(np.std(dd, ddof=1)), seed_diffs=dd, positive_seeds=int(sum(v > 0 for v in dd))))

# --- query overlap between the two TREC-DL years and with the training / selection queries (ids and normalised texts)
def norm(t):
    t = unicodedata.normalize("NFKC", t).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


import ir_datasets
Q = {}
for y in (2019, 2020):
    ds = ir_datasets.load(f"msmarco-passage/trec-dl-{y}/judged")
    qs = {q.query_id: q.text for q in ds.queries_iter()}
    keep = {q for q, _ in DATA[f"MS-Qwen-{y}"]["cells"][("BASE", 1, 0)]["tks"]}
    Q[y] = {q: qs[q] for q in keep}
trainQ = {}
for nm, rel in (("train5000", "data/msmarco_train_v2_5k/pairs_text_basic_5000.jsonl"), ("diag200", "data/msmarco_po/diag200/pairs_text_basic_diag200.jsonl"), ("val200", "data/msmarco_po/val200/pairs_text_basic_seed0.jsonl")):
    qq = {}
    for l in open(ROOT / rel, encoding="utf-8"):
        r = json.loads(l); qq[str(r["query_id"])] = r["query"]
    trainQ[nm] = qq


def overlap(A, B):
    ids = sorted(set(A) & set(B))
    nA = {}
    for q, t in A.items():
        nA.setdefault(norm(t), []).append(q)
    nB = {norm(t) for t in B.values()}
    txt = sorted(x for x in nA if x in nB)
    return dict(n_a=len(A), n_b=len(B), id_overlap=len(ids), id_overlap_ids=ids[:20], normalised_text_overlap=len(txt), text_overlap_examples=txt[:10])


OVL = {"2019_vs_2020": overlap(Q[2019], Q[2020])}
for y in (2019, 2020):
    for nm, qq in trainQ.items():
        OVL[f"{y}_vs_{nm}"] = overlap(Q[y], qq)
print("overlap:", {k: (v["id_overlap"], v["normalised_text_overlap"]) for k, v in OVL.items()}, flush=True)

# --- provenance of the used test files (meta)
used = {}
for mdl in MODELS:
    for y in (2019, 2020):
        for key, e in DATA[f"MS-{mdl}-{y}"]["cells"].items():
            used[e["test_file"]] = (mdl, y)
META = {}
for f, (mdl, y) in used.items():
    m = ld(f)["meta"]
    for k, v in m.items():
        if isinstance(v, (str, int, float, bool)) and len(str(v)) < 120:
            META.setdefault((mdl, k), set()).add(str(v))
META_OUT = {f"{mdl}|{k}": sorted(v)[:4] for (mdl, k), v in META.items() if len(v) <= 3 and k not in ("created", "time", "timestamp")}
FILE_LIST = sorted((mdl, y, os.path.relpath(f, R)) for f, (mdl, y) in used.items())
print("meta keys", sorted({k for (_, k) in META}), flush=True)

# --- write
ST = {"t0": "t=0", "val": "val200 t*", "cross": "CROSS"}
with open(OUT / "cross_per_instruction_seed.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "test_year", "cond", "T", "seed", "t_val200", "t_cross", "kappa_t0", "kappa_val200", "kappa_cross", "pos_rate_t0", "pos_rate_val200", "pos_rate_cross", "gold_relevant_share"])
    for r in ROWS:
        w.writerow([r["model"], r["year"], r["cond"], r["T"], r["seed"], f"{r['t_val']:.9f}", f"{r['t_cross']:.9f}", f"{r['k0']:.9f}", f"{r['kval']:.9f}", f"{r['kcross']:.9f}", f"{r['pos_rate_0']:.4f}", f"{r['pos_rate_val']:.4f}", f"{r['pos_rate_cross']:.4f}", f"{r['gold_rel']:.4f}"])
with open(OUT / "cross_summary.csv", "w", newline="") as f:
    w = csv.writer(f); keys = list(SUM[0].keys()); w.writerow(keys)
    for r in SUM:
        w.writerow([("" if r[k] is None else (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])) for k in keys])
with open(OUT / "cross_seed_values.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "test_year", "cond", "seed", "kappa_t0", "kappa_val200", "kappa_cross"])
    for (mdl, y, c), d in AGG.items():
        for i, s in enumerate((0,) if c == "BASE" else SEEDS):
            w.writerow([mdl, y, c, s, f"{d['k0'][i]:.6f}", f"{d['kval'][i]:.6f}", f"{d['kcross'][i]:.6f}"])
json.dump(dict(cpu_seconds=time.time() - t_start, n_rows=len(ROWS), check_vs_kappa_pkl=CHK, reproduction_submitted_val200_t0=REP, summary=SUM, paired=PAIR, overlap=OVL, meta_constant_keys=META_OUT, disambiguated_copies=DISAMBIGUATED,
               evaluation_sizes={f"{m}-{y}": v for (m, y), v in EVALN.items()}, used_test_files=FILE_LIST), open(OUT / "cross399_results.json", "w"), indent=1, default=str)
print("done, cpu", f"{time.time() - t_start:.0f}s")
