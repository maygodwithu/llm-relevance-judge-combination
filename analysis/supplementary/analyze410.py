"""Independent-calibration check for ESCI Llama-3.1-8B-Instruct, T1-T7 (the first instruction, T2, was checked earlier and is reused). Read-only on all stored scores.
Fixed: the selected PO texts (T1, T4, T6 kept the original text; T2, T3, T5, T7 changed), the SFT checkpoints (CE+soft kappa seeds 0-2, final 625 updates), the stored test300 scores. Only the threshold data change:
  t = 0;  t* from the shared validation (`sel`, 200 queries, the set that also selected the PO text);  t* from the independent pool (200 queries / 3,939 rows of the earlier T2 check, frozen before any score),
each applied to the stored test300 margins. Rule = the submitted one (hard-kappa maximiser; ties: 0, then smaller |t|, then smaller t). Per instruction x variant (original / PO) x objective (Base once, CE+soft kappa seeds 0-2).
Aggregates: SFT = per seed the mean over T1-T7, then mean +- sample SD over the 3 seeds; Base = mean over the 7 instructions. Contrasts: PO - original per regime, and independent - shared change of the PO gain;
paired test-query bootstrap (10,000 draws, default_rng(410), ONE draw set for every condition; per draw the mean over instructions (and seeds); conditional on the fixed models and thresholds; exploratory).
    python analyze410.py --root <project root> --out <dir> [--Ts 1,2,3,4,5,6,7]"""
import argparse, csv, glob, gzip, hashlib, json, os, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--out", required=True); ap.add_argument("--Ts", default="1,2,3,4,5,6,7"); a = ap.parse_args()
R = Path(a.root) / "cpmr_experiment"; OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); TS = [int(x) for x in a.Ts.split(",")]
sys.path.insert(0, str(R.parent / "analysis"))
import esci204_common as C  # noqa: E402

CHANGED = {2, 3, 5, 7}
_IDX = {}
for f in glob.glob(str(R / "**" / "esciLlamaEval_*.json.gz"), recursive=True):
    _IDX.setdefault(os.path.basename(f), []).append(f)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def load(p):
    return json.load(gzip.open(p, "rt", encoding="utf-8"))


def margins(path):
    raw = load(path)["raw"]
    ks = [(q, x) for q in sorted(raw) for x in sorted(raw[q])]
    m = np.array([raw[q][x][0] + raw[q][x][1] - raw[q][x][3] - raw[q][x][4] for q, x in ks])
    y = np.array([raw[q][x][10] >= 2 for q, x in ks])
    return ks, m, y


def kat(m, y, t):
    p = m > t
    return float(C.kap(int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum())))


def stored(name):
    fs = _IDX.get(name, [])
    assert fs, ("missing stored score file", name)
    ref = margins(fs[0])[1]
    for c in fs[1:]:
        assert np.allclose(margins(c)[1], ref, atol=1e-6, rtol=0), ("copies differ", name)
    return fs[0]


man = {}; res_p1 = {}
for k in TS:
    man[k] = {e["display_id"]: e for e in json.load(open(R / f"esci_llama/p1_T{k}/esci_apocf_manifest.json"))["entries"]}
    res_p1[k] = json.load(open(R / f"esci_llama/p1_T{k}/p1_result.json"))


def cand_margins(k, idx, ks_ref):
    c = load(R / f"esci_llama/p1_T{k}/p1_raw_cand{idx}.json.gz")
    mm = c["margins"]
    return np.array([float(mm[q][x][0] + mm[q][x][1] - mm[q][x][2] - mm[q][x][3]) for q, x in ks_ref]), c["full_sha256"]


def calib_path(k, job, ins):
    tag, d = ("calib393", "esci393") if k == 2 else ("calib410", "esci410")
    return str(R / d / "eval" / f"esciLlamaEval_{job}__{ins}__{tag}.json.gz")


# ------------------------------------------------------------------ build the run list
runs = []; ksel_ref = ks_cal_ref = ks_te_ref = None; checks = []
for k in TS:
    idx = res_p1[k]["selection"]["index"]; changed = k in CHANGED
    assert changed == (idx != 0), ("PO change status differs from the pre-set list", k, idx)
    for var in ("orig", "po"):
        ins = f"T{k}" if var == "orig" or not changed else f"T{k}_APOCF"
        reused_original = (var == "po" and not changed)
        el = f"ELQ_mix_T{k}" if var == "orig" or not changed else f"ELQ_mix_T{k}_APOCF"
        jobs = [("Base", 0, "Base")] + [("CE+soft kappa", s, f"{el}_s{s}_step625") for s in (0, 1, 2)]
        for obj, s, name in jobs:
            r = dict(T=k, variant=var, objective=obj, seed=s, name=name, instr=ins, reused_original_text=reused_original, po_selection=res_p1[k]["selection"]["variant"])
            r["test"] = stored(f"esciLlamaEval_{name}__{ins}__test300.json.gz"); r["calib"] = calib_path(k, name, ins)
            if obj == "Base" and var == "po" and changed:
                r["sel"] = None
            else:
                r["sel"] = stored(f"esciLlamaEval_{name}__{ins}__sel.json.gz")
            runs.append(r)

for r in runs:
    # instruction hash of every score file used (against the submitted manifest text hash)
    k = r["T"]; ents = man[k]
    want = ents[r["instr"]]["sha256_full"] if r["instr"] in ents else ents[f"T{k}"]["sha256_full"]
    for key in ("test", "calib") + (("sel",) if r["sel"] else ()):
        got = load(r[key])["meta"].get("instruction_sha256")
        assert got == want, ("instruction hash", r["name"], r["instr"], key, got, want)
    if r["sel"]:
        ks, m, y = margins(r["sel"]); ksel_ref = ksel_ref or ks; assert ks == ksel_ref
        r["sel_m"], r["sel_y"], r["sel_sha"], r["sel_src"] = m, y, sha(r["sel"]), os.path.basename(r["sel"])
for r in runs:
    if r["sel"]: continue
    k = r["T"]; idx = res_p1[k]["selection"]["index"]
    base_o = next(x for x in runs if x["T"] == k and x["variant"] == "orig" and x["objective"] == "Base")
    cm0, fs0 = cand_margins(k, 0, ksel_ref); cms, fss = cand_margins(k, idx, ksel_ref)
    assert float(np.abs(cm0 - base_o["sel_m"]).max()) < 1e-4, ("cand0 != stored Base original sel", k)
    assert fs0 == man[k][f"T{k}"]["sha256_full"] and fss == man[k][f"T{k}_APOCF"]["sha256_full"], ("candidate text hash", k)
    checks.append(dict(T=k, cand0_equals_stored_base_original_sel_max_diff=float(np.abs(cm0 - base_o["sel_m"]).max()), selected_candidate_text_sha_equals_PO_text=True))
    r["sel_m"], r["sel_y"], r["sel_sha"], r["sel_src"] = cms, base_o["sel_y"], sha(R / f"esci_llama/p1_T{k}/p1_raw_cand{idx}.json.gz"), f"p1_raw_cand{idx}"
for k in TS:   # unchanged PO text: the selected candidate 0 IS the original text; verify cand text hash for T1, T4, T6
    if k not in CHANGED:
        _, f0 = cand_margins(k, 0, ksel_ref); assert f0 == man[k][f"T{k}"]["sha256_full"] == res_p1[k]["original_sha256"]
        checks.append(dict(T=k, unchanged_PO_text_equals_original=True))
for r in runs:
    ks, m, y = margins(r["calib"]); ks_cal_ref = ks_cal_ref or ks; assert ks == ks_cal_ref
    r["cal_m"], r["cal_y"], r["cal_sha"] = m, y, sha(r["calib"])
    ks, m, y = margins(r["test"]); ks_te_ref = ks_te_ref or ks; assert ks == ks_te_ref
    r["te_m"], r["te_y"], r["te_sha"] = m, y, sha(r["test"])
    r["t_sel"], r["val_k_sel"] = C.best_threshold(r["sel_m"], r["sel_y"]); r["t_cal"], r["val_k_cal"] = C.best_threshold(r["cal_m"], r["cal_y"])
    r["k0"], r["k_shared"], r["k_indep"] = kat(r["te_m"], r["te_y"], 0.0), kat(r["te_m"], r["te_y"], r["t_sel"]), kat(r["te_m"], r["te_y"], r["t_cal"])
    r["auc"] = float(C.auc(r["te_m"], r["te_y"])); r["rel_t0"], r["rel_shared"], r["rel_indep"] = float((r["te_m"] > 0).mean()), float((r["te_m"] > r["t_sel"]).mean()), float((r["te_m"] > r["t_cal"]).mean())
    r["k_cal_at_t_sel"], r["k_sel_at_t_cal"] = kat(r["cal_m"], r["cal_y"], r["t_sel"]), kat(r["sel_m"], r["sel_y"], r["t_cal"])
    r["k_cal_at_0"], r["k_sel_at_0"] = kat(r["cal_m"], r["cal_y"], 0.0), kat(r["sel_m"], r["sel_y"], 0.0)

KEYS = ["T", "variant", "objective", "seed", "name", "instr", "reused_original_text", "po_selection", "t_sel", "t_cal", "val_k_sel", "val_k_cal", "k_sel_at_0", "k_cal_at_0", "k_cal_at_t_sel", "k_sel_at_t_cal", "k0", "k_shared", "k_indep", "auc", "rel_t0", "rel_shared", "rel_indep", "sel_src", "sel_sha", "cal_sha", "te_sha"]


def wcsv(name, rows, keys=None):
    keys = keys or list(rows[0])
    with open(OUT / name, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n"); w.writeheader(); [w.writerow({k: r[k] for k in keys}) for r in rows]


wcsv("calib410_per_run.csv", runs, KEYS)
REG = {"t=0": "k0", "shared validation t*": "k_shared", "independent calibration t*": "k_indep"}
cell = {(r["T"], r["variant"], r["objective"], r["seed"]): r for r in runs}
SEEDS = {"Base": (0,), "CE+soft kappa": (0, 1, 2)}


def msd(v):
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"))


# per-instruction summary (SFT: mean +- SD over seeds; Base single)
per_instr = []
for k in TS:
    for var in ("orig", "po"):
        for obj in SEEDS:
            rs = [cell[(k, var, obj, s)] for s in SEEDS[obj]]; d = dict(T=f"T{k}", variant=var, objective=obj, po_text_changed=(k in CHANGED), seeds=len(rs))
            for m_ in ("t_sel", "t_cal", "k0", "k_shared", "k_indep", "auc", "val_k_sel", "val_k_cal"):
                d[m_ + "_mean"], d[m_ + "_sd"] = msd([r[m_] for r in rs])
            per_instr.append(d)
wcsv("per_instruction_summary.csv", per_instr)
# overall: SFT = per seed mean over instructions, then mean/SD over seeds; Base = mean over instructions
overall = []; per_seed_T = []
for var in ("orig", "po"):
    for obj in SEEDS:
        d = dict(variant=var, objective=obj)
        for m_ in ("t_sel", "t_cal", "k0", "k_shared", "k_indep", "auc"):
            vs = [float(np.mean([cell[(k, var, obj, s)][m_] for k in TS])) for s in SEEDS[obj]]
            d[m_ + "_mean"], d[m_ + "_sd"] = msd(vs)
            if m_ in ("k0", "k_shared", "k_indep"):
                for s, v in zip(SEEDS[obj], vs): per_seed_T.append(dict(variant=var, objective=obj, seed=s, quantity=m_, T_mean=v))
        overall.append(d)
wcsv("overall_summary.csv", overall); wcsv("per_seed_T_mean.csv", per_seed_T)

# bootstrap
B = 10000
qs = sorted({q for q, _ in ks_te_ref}); qid = {q: i for i, q in enumerate(qs)}; rowq = np.array([qid[q] for q, _ in ks_te_ref]); nq = len(qs)
rng = np.random.default_rng(410)
W = np.stack([np.bincount(i, minlength=nq) for i in rng.integers(0, nq, size=(B, nq))]).astype(float)
_cache = {}


def kboot(r, t):
    key = (r["T"], r["variant"], r["objective"], r["seed"], t)
    if key in _cache: return _cache[key]
    pr = r["te_m"] > t; y = r["te_y"]
    cnt = np.stack([np.bincount(rowq, weights=x.astype(float), minlength=nq) for x in (pr & y, pr & ~y, ~pr & y, ~pr & ~y)], 1)
    tot = cnt.sum(0); bt = W @ cnt
    _cache[key] = (float(C.kap(*tot)), C.kap_vec(bt[:, 0], bt[:, 1], bt[:, 2], bt[:, 3])); return _cache[key]


tfun = {"t=0": lambda r: 0.0, "shared validation t*": lambda r: r["t_sel"], "independent calibration t*": lambda r: r["t_cal"]}
# one gain object: PO - original at a regime, for a set of instructions Ks, objective, per seed -> (point per seed, boot per seed)
def gain(reg, obj, Ks, s):
    p = [kboot(cell[(k, "po", obj, s)], tfun[reg](cell[(k, "po", obj, s)])) for k in Ks]; o = [kboot(cell[(k, "orig", obj, s)], tfun[reg](cell[(k, "orig", obj, s)])) for k in Ks]
    return np.mean([x[0] for x in p]) - np.mean([x[0] for x in o]), np.mean([x[1] for x in p], 0) - np.mean([x[1] for x in o], 0)


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


contrasts = []; delta = []
for obj in SEEDS:
    g = {reg: [gain(reg, obj, TS, s) for s in SEEDS[obj]] for reg in REG}
    for reg in REG:
        sd = [x[0] for x in g[reg]]; bs = np.mean([x[1] for x in g[reg]], 0)
        contrasts.append(dict(scope="all instructions", regime=reg, objective=obj, po_minus_original=float(np.mean(sd)), seed_diffs=sd, positive_seeds=int(sum(x > 0 for x in sd)), ci95=ci(bs)))
    sd = [g["independent calibration t*"][i][0] - g["shared validation t*"][i][0] for i in range(len(SEEDS[obj]))]
    bs = np.mean([g["independent calibration t*"][i][1] - g["shared validation t*"][i][1] for i in range(len(SEEDS[obj]))], 0)
    delta.append(dict(objective=obj, change_in_PO_gain_independent_minus_shared=float(np.mean(sd)), seed_diffs=sd, positive_seeds=int(sum(x > 0 for x in sd)), ci95=ci(bs)))
# per instruction (changed-text instructions are the only ones with a non-zero PO - original difference)
per_instr_c = []
for k in TS:
    for obj in SEEDS:
        for reg in REG:
            gs = [gain(reg, obj, [k], s) for s in SEEDS[obj]]; sd = [x[0] for x in gs]; bs = np.mean([x[1] for x in gs], 0)
            per_instr_c.append(dict(T=f"T{k}", po_text_changed=(k in CHANGED), objective=obj, regime=reg, po_minus_original=float(np.mean(sd)), seed_diffs=sd, positive_seeds=int(sum(x > 0 for x in sd)), ci95=ci(bs)))
with open(OUT / "po_minus_original_per_instruction.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["instruction", "po_text_changed", "objective", "regime", "diff_seed_mean", "seed_diffs", "positive_seeds", "ci95_low", "ci95_high"])
    for c in per_instr_c: w.writerow([c["T"], c["po_text_changed"], c["objective"], c["regime"], f"{c['po_minus_original']:.6f}", ";".join(f"{x:.6f}" for x in c["seed_diffs"]), c["positive_seeds"], f"{c['ci95'][0]:.6f}", f"{c['ci95'][1]:.6f}"])
# changed-text instructions only (T2, T3, T5, T7), for reference
chg = [k for k in TS if k in CHANGED]; chg_c = []
if chg:
    for obj in SEEDS:
        for reg in REG:
            gs = [gain(reg, obj, chg, s) for s in SEEDS[obj]]; sd = [x[0] for x in gs]; bs = np.mean([x[1] for x in gs], 0)
            chg_c.append(dict(scope="changed-text instructions only " + ",".join(f"T{k}" for k in chg), regime=reg, objective=obj, po_minus_original=float(np.mean(sd)), seed_diffs=sd, positive_seeds=int(sum(x > 0 for x in sd)), ci95=ci(bs)))
repro = None
res = dict(overall=overall, per_instruction=per_instr, contrasts_all=contrasts, change_in_po_gain=delta, contrasts_changed_only=chg_c, contrasts_per_instruction=per_instr_c, checks=checks, instructions=TS, po_text_changed=sorted(CHANGED & set(TS)),
           note="paired test-query bootstrap 10,000 draws default_rng(410), one draw set for all conditions; per draw the mean over instructions (and seeds); conditional on fixed models and thresholds; exploratory; instructions/seeds are not independent samples")
(OUT / "calib410_results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
f4 = lambda t: f"{t[0]:.4f}" + (f" ± {t[1]:.4f}" if not np.isnan(t[1]) else "")
L = ["## Overall (SFT: per seed the mean over instructions, then mean ± SD over 3 seeds; Base: mean over instructions)", "| variant | objective | t* shared | t* independent | test k(0) | test k(shared t*) | test k(independent t*) | AUC |", "|---|---|---|---|---|---|---|---|"]
for d in overall:
    L.append(f"| {'original' if d['variant'] == 'orig' else 'PO'} | {d['objective']} | {d['t_sel_mean']:+.3f} | {d['t_cal_mean']:+.3f} | {d['k0_mean']:.4f}" + (f" ± {d['k0_sd']:.4f}" if not np.isnan(d['k0_sd']) else "") + f" | {d['k_shared_mean']:.4f}" + (f" ± {d['k_shared_sd']:.4f}" if not np.isnan(d['k_shared_sd']) else "") + f" | {d['k_indep_mean']:.4f}" + (f" ± {d['k_indep_sd']:.4f}" if not np.isnan(d['k_indep_sd']) else "") + f" | {d['auc_mean']:.4f} |")
L += ["", "## PO - original (all instructions; T1, T4, T6 kept the original text so their difference is exactly 0)", "| regime | objective | PO - original | seed diffs | positive seeds | 95% CI |", "|---|---|---|---|---|---|"]
for c in contrasts: L.append(f"| {c['regime']} | {c['objective']} | {c['po_minus_original']:+.4f} | {', '.join(f'{x:+.4f}' for x in c['seed_diffs'])} | {c['positive_seeds']}/{len(c['seed_diffs'])} | [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] |")
L += ["", "## Change of the PO gain: independent t* minus shared t*", "| objective | change | seed diffs | positive seeds | 95% CI |", "|---|---|---|---|---|"]
for c in delta: L.append(f"| {c['objective']} | {c['change_in_PO_gain_independent_minus_shared']:+.4f} | {', '.join(f'{x:+.4f}' for x in c['seed_diffs'])} | {c['positive_seeds']}/{len(c['seed_diffs'])} | [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] |")
if chg_c:
    L += ["", "## PO - original, changed-text instructions only (" + ",".join(f"T{k}" for k in chg) + ")", "| regime | objective | PO - original | seed diffs | positive seeds | 95% CI |", "|---|---|---|---|---|---|"]
    for c in chg_c: L.append(f"| {c['regime']} | {c['objective']} | {c['po_minus_original']:+.4f} | {', '.join(f'{x:+.4f}' for x in c['seed_diffs'])} | {c['positive_seeds']}/{len(c['seed_diffs'])} | [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] |")
L += ["", "## Per instruction, CE+soft kappa PO - original (seed mean; unadjusted 95% CI)", "| T | text changed | t=0 | shared t* | independent t* |", "|---|---|---|---|---|"]
for k in TS:
    cs = {c["regime"]: c for c in per_instr_c if c["T"] == f"T{k}" and c["objective"] == "CE+soft kappa"}
    L.append(f"| T{k} | {'yes' if k in CHANGED else 'no'} | " + " | ".join(f"{cs[g]['po_minus_original']:+.4f} [{cs[g]['ci95'][0]:+.4f}, {cs[g]['ci95'][1]:+.4f}] ({cs[g]['positive_seeds']}/3)" for g in REG) + " |")
(OUT / "calib410_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L)); print(json.dumps(checks))
