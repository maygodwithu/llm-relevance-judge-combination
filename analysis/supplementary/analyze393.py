"""Threshold selection on an INDEPENDENT calibration pool instead of the validation set that also selected the PO instruction. ESCI Llama-3.1-8B, T2 (original) vs T2_APOCF (selected PO text).
Fixed: the PO text, the SFT checkpoints (CE+soft kappa seeds 0-2), the stored test300 scores. Only the threshold data change. Per run: t=0, t* from the shared validation (`sel`, 200 queries, the set that selected the PO text), t* from the new independent pool (200 queries, frozen before any score),
each applied to the stored test300 margins. Threshold rule = the submitted one (hard-kappa maximiser, ties: 0, then smaller |t|, then smaller t). Contrasts: PO - original per regime (paired test-query bootstrap, 10,000 draws, default_rng(393), conditional on the fixed models and thresholds)
and the sampling variation of the threshold selection (500 resamples of the selection queries of each set, default_rng(393), thresholds re-selected, applied to the fixed test scores). No test-based choice of anything.
    python analyze393.py --root <project root> --out <dir>"""
import argparse, csv, glob, gzip, hashlib, json, os, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--out", required=True); a = ap.parse_args()
R = Path(a.root) / "cpmr_experiment"; OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R / "analysis"))
import esci204_common as C  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def one(pattern):
    fs = sorted(glob.glob(str(R / pattern), recursive=True))
    assert fs, pattern
    return fs[0]


def margins(path):
    d = json.load(gzip.open(path, "rt", encoding="utf-8"))
    raw = d["raw"]
    ks = [(q, x) for q in sorted(raw) for x in sorted(raw[q])]
    m = np.array([raw[q][x][0] + raw[q][x][1] - raw[q][x][3] - raw[q][x][4] for q, x in ks])
    y = np.array([raw[q][x][10] >= 2 for q, x in ks])
    return ks, m, y


def kat(m, y, t):
    p = m > t
    return float(C.kap(int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum())))


def best_thr_w(m, y, w):
    """weighted version of esci204_common.best_threshold (same candidate partitions, same tie rule); w = query multiplicities per row"""
    o = np.argsort(m, kind="stable"); ms, ys, ws_ = m[o], y[o], w[o]
    u = np.unique(ms)
    cand = np.unique(np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0], [0.0]]))
    idx = np.searchsorted(ms, cand, side="right")                       # rows with m <= t are predicted negative
    cp = np.concatenate([[0.0], np.cumsum(ws_ * ys)]); cn = np.concatenate([[0.0], np.cumsum(ws_ * (~ys))])
    pos, neg = cp[-1], cn[-1]
    fn = cp[idx]; tn = cn[idx]; tp = pos - fn; fp = neg - tn
    k = C.kap_vec(tp, fp, fn, tn)
    best = None
    for t, kk in zip(cand, k):
        if not np.isfinite(kk):
            continue
        key = (-round(float(kk), 12), 0 if t == 0.0 else 1, abs(t), t)
        if best is None or key < best[0]:
            best = (key, float(t), float(kk))
    return best[1], best[2]


# ------------------------------------------------------------------ inputs
ID = {"orig": dict(instr="T2", soft="ELQ_mix_T2_s{s}_step625", base="Base", test_pre="esciLlamaEval_", ck="T2"), "po": dict(instr="T2_APOCF", soft="ELQ_mix_T2_APOCF_s{s}_step625", base="Base", test_pre="esciLlamaEval_", ck="T2_APOCF")}
meta_pool = json.load(open(R / "esci393/calib/calib_manifest.json"))
runs = []
selidx = json.load(open(R / "esci_llama/p1_T2/p1_result.json"))["selection"]["index"]
for var, d in ID.items():
    ins = d["instr"]
    jobs = [("Base", 0, d["base"], f"esciLlamaEval_Base__{ins}__calib393.json.gz", f"esci_llama/eval/esciLlamaEval_Base__{ins}__test300.json.gz")]
    jobs += [("CE+soft kappa", s, d["soft"].format(s=s), f"esciLlamaEval_{d['soft'].format(s=s)}__{ins}__calib393.json.gz", f"esci_204/eval/esciLlamaEval_{d['soft'].format(s=s)}__{ins}__test300.json.gz") for s in (0, 1, 2)]
    for obj, s, name, calib_f, test_f in jobs:
        runs.append(dict(variant=var, objective=obj, seed=s, name=name, instr=ins, calib=str(R / "esci393/eval" / calib_f), test=str(R / test_f)))
# shared validation (`sel`) scores: stored SFT files; Base from the stored files (original) and the PO candidate raw margins (selected PO text)
def sel_scores(r):
    if r["objective"] == "CE+soft kappa":
        return one(f"esci_204/eval/esciLlamaEval_{r['name']}__{r['instr']}__sel.json.gz")
    return None
base_orig_sel = one("t334/base_sel/esciLlamaEval_Base__T2__sel.json.gz")
ksel_ref, m_sel_orig_base, y_sel = margins(base_orig_sel)
praw = json.load(gzip.open(R / f"esci_llama/p1_T2/p1_raw_cand{selidx}.json.gz", "rt", encoding="utf-8")); raw0 = json.load(gzip.open(R / "esci_llama/p1_T2/p1_raw_cand0.json.gz", "rt", encoding="utf-8"))
def from_cand(c):
    mm = c["margins"]
    if isinstance(mm, dict) and isinstance(next(iter(mm.values())), dict):
        return np.array([float(mm[q][x][0] + mm[q][x][1] - mm[q][x][2] - mm[q][x][3]) for q, x in ksel_ref])      # fields: [lp_label(R), lp_eot(R), lp_label(I), lp_eot(I)]
    raise SystemExit("unexpected candidate margin format")
cand0 = from_cand(raw0); candsel = from_cand(praw)
chk = dict(cand0_equals_stored_base_original_sel=float(np.abs(cand0 - m_sel_orig_base).max()), selected_candidate_text_sha256_equals_PO_text=(praw["full_sha256"] == json.load(open(R / "esci_llama/p1_T2/esci_apocf_manifest.json"))["entries"][1]["sha256_full"] if "sha256_full" in json.load(open(R / "esci_llama/p1_T2/esci_apocf_manifest.json"))["entries"][1] else praw["full_sha256"][:12]))
assert chk["cand0_equals_stored_base_original_sel"] < 1e-4, chk
ks_cal_ref = None; ks_test_ref = None
for r in runs:
    if r["objective"] == "Base":
        ms = m_sel_orig_base if r["variant"] == "orig" else candsel; r["sel_src"] = "t334 Base__T2__sel" if r["variant"] == "orig" else f"p1_raw_cand{selidx}"
        r["sel_m"], r["sel_y"], r["sel_sha"] = ms, y_sel, sha(base_orig_sel)
    else:
        f = sel_scores(r); ks, m, y = margins(f); assert ks == ksel_ref
        r["sel_m"], r["sel_y"], r["sel_sha"], r["sel_src"] = m, y, sha(f), os.path.basename(f)
    ks, m, y = margins(r["calib"]); ks_cal_ref = ks_cal_ref or ks; assert ks == ks_cal_ref
    r["cal_m"], r["cal_y"], r["cal_sha"] = m, y, sha(r["calib"])
    ks, m, y = margins(r["test"]); ks_test_ref = ks_test_ref or ks; assert ks == ks_test_ref
    r["te_m"], r["te_y"], r["te_sha"] = m, y, sha(r["test"])
    # thresholds and results
    r["t_sel"], r["val_k_sel"] = C.best_threshold(r["sel_m"], r["sel_y"]); r["t_cal"], r["val_k_cal"] = C.best_threshold(r["cal_m"], r["cal_y"])
    r["k0"], r["k_shared"], r["k_indep"] = kat(r["te_m"], r["te_y"], 0.0), kat(r["te_m"], r["te_y"], r["t_sel"]), kat(r["te_m"], r["te_y"], r["t_cal"])
    r["auc"] = float(C.auc(r["te_m"], r["te_y"])); r["rel_t0"], r["rel_shared"], r["rel_indep"] = float((r["te_m"] > 0).mean()), float((r["te_m"] > r["t_sel"]).mean()), float((r["te_m"] > r["t_cal"]).mean())
    r["k_cal_at_t_sel"], r["k_sel_at_t_cal"] = kat(r["cal_m"], r["cal_y"], r["t_sel"]), kat(r["sel_m"], r["sel_y"], r["t_cal"])
    r["k_cal_at_0"], r["k_sel_at_0"] = kat(r["cal_m"], r["cal_y"], 0.0), kat(r["sel_m"], r["sel_y"], 0.0)

keys = ["variant", "objective", "seed", "name", "instr", "t_sel", "t_cal", "val_k_sel", "val_k_cal", "k_sel_at_0", "k_cal_at_0", "k_cal_at_t_sel", "k_sel_at_t_cal", "k0", "k_shared", "k_indep", "auc", "rel_t0", "rel_shared", "rel_indep", "sel_src", "sel_sha", "cal_sha", "te_sha"]
with open(OUT / "calib393_per_run.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n"); w.writeheader(); [w.writerow({k: r[k] for k in keys}) for r in runs]


def mean_sd(v):
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"))


summ = {}
for var in ("orig", "po"):
    for obj in ("Base", "CE+soft kappa"):
        rs = [r for r in runs if r["variant"] == var and r["objective"] == obj]
        d = {}
        for k in ("t_sel", "t_cal", "val_k_sel", "val_k_cal", "k_sel_at_0", "k_cal_at_0", "k_cal_at_t_sel", "k_sel_at_t_cal", "k0", "k_shared", "k_indep", "auc", "rel_t0", "rel_shared", "rel_indep"):
            d[k] = mean_sd([r[k] for r in rs])
        d["seeds"] = [r["seed"] for r in rs]; d["per_seed"] = {k: [r[k] for r in rs] for k in ("k0", "k_shared", "k_indep", "t_sel", "t_cal")}
        summ[f"{var}|{obj}"] = d

# paired test-query bootstrap of PO - original
B = 10000
kk = ks_test_ref; qs = sorted({q for q, _ in kk}); qid = {q: i for i, q in enumerate(qs)}; rowq = np.array([qid[q] for q, _ in kk]); nq = len(qs)
rng = np.random.default_rng(393)
W = np.stack([np.bincount(i, minlength=nq) for i in rng.integers(0, nq, size=(B, nq))]).astype(float)
def kboot(r, t):
    pr = r["te_m"] > t; y = r["te_y"]
    cnt = np.stack([np.bincount(rowq, weights=x.astype(float), minlength=nq) for x in (pr & y, pr & ~y, ~pr & y, ~pr & ~y)], 1)
    tot = cnt.sum(0); bt = W @ cnt
    return float(C.kap(*tot)), C.kap_vec(bt[:, 0], bt[:, 1], bt[:, 2], bt[:, 3])
reg = {"t=0": lambda r: 0.0, "shared validation t*": lambda r: r["t_sel"], "independent calibration t*": lambda r: r["t_cal"]}
contrasts = []
for rname, tf in reg.items():
    for obj in ("Base", "CE+soft kappa"):
        o = [r for r in runs if r["variant"] == "orig" and r["objective"] == obj]; p = [r for r in runs if r["variant"] == "po" and r["objective"] == obj]
        ko = [kboot(r, tf(r)) for r in o]; kp = [kboot(r, tf(r)) for r in p]
        d = float(np.mean([x[0] for x in kp]) - np.mean([x[0] for x in ko])); bs = np.mean([x[1] for x in kp], 0) - np.mean([x[1] for x in ko], 0)
        sd = [kp[i][0] - ko[i][0] for i in range(len(o))]
        contrasts.append(dict(regime=rname, objective=obj, po_minus_original=d, seed_diffs=sd, positive_seeds=int(sum(x > 0 for x in sd)), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]))
# independent minus shared threshold, same model
indep = []
for var in ("orig", "po"):
    for obj in ("Base", "CE+soft kappa"):
        rs = [r for r in runs if r["variant"] == var and r["objective"] == obj]
        d = [r["k_indep"] - r["k_shared"] for r in rs]
        indep.append(dict(variant=var, objective=obj, mean=float(np.mean(d)), seed_diffs=d))
# sampling variation of the threshold selection: resample the selection queries of each set, re-select thresholds, apply to the fixed test scores
def resample(setname, nres=500):
    rg = np.random.default_rng(393)
    first = runs[0]; ks = ksel_ref if setname == "sel" else ks_cal_ref
    qq = sorted({q for q, _ in ks}); qi = {q: i for i, q in enumerate(qq)}; rq = np.array([qi[q] for q, _ in ks]); n = len(qq)
    draws = rg.integers(0, n, size=(nres, n))
    out = {f"{r['variant']}|{r['objective']}|{r['seed']}": [] for r in runs}
    for b in range(nres):
        mult = np.bincount(draws[b], minlength=n).astype(float)[rq]
        for r in runs:
            m, y = (r["sel_m"], r["sel_y"]) if setname == "sel" else (r["cal_m"], r["cal_y"])
            t, _ = best_thr_w(m, y, mult)
            out[f"{r['variant']}|{r['objective']}|{r['seed']}"].append(kat(r["te_m"], r["te_y"], t))
    res = {}
    for var in ("orig", "po"):
        for obj in ("Base", "CE+soft kappa"):
            seeds = [r["seed"] for r in runs if r["variant"] == var and r["objective"] == obj]
            res[f"{var}|{obj}"] = np.mean([out[f"{var}|{obj}|{s}"] for s in seeds], 0)
    return res
variation = {}
for setname in ("sel", "cal"):
    rv = resample(setname)
    variation[setname] = {k: dict(mean=float(v.mean()), p2_5=float(np.percentile(v, 2.5)), p97_5=float(np.percentile(v, 97.5)), sd=float(v.std())) for k, v in rv.items()}
    for obj in ("Base", "CE+soft kappa"):
        dd = rv[f"po|{obj}"] - rv[f"orig|{obj}"]
        variation[setname][f"PO-original|{obj}"] = dict(mean=float(dd.mean()), p2_5=float(np.percentile(dd, 2.5)), p97_5=float(np.percentile(dd, 97.5)), share_positive=float((dd > 0).mean()))
res = dict(pool=meta_pool, shared_validation=dict(file="sel_rows.jsonl", queries=200, rows=int(len(y_sel)), relevant_share=float(y_sel.mean())), checks=chk, summary=summ, po_minus_original=contrasts, independent_minus_shared=indep, threshold_selection_variation=variation,
           note="paired test-query bootstrap 10,000 draws default_rng(393) conditional on fixed models and thresholds; threshold-selection variation 500 resamples default_rng(393); exploratory")
(OUT / "calib393_results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
f4 = lambda t: f"{t[0]:.4f}" + (f" ± {t[1]:.4f}" if not np.isnan(t[1]) else "")
L = ["| instruction | objective | seeds | t* shared | t* independent | test k(0) | test k(shared t*) | test k(independent t*) | val k(t*) on own set: shared / independent |", "|---|---|---|---|---|---|---|---|---|"]
for var, nm in (("orig", "original T2"), ("po", "PO T2")):
    for obj in ("Base", "CE+soft kappa"):
        d = summ[f"{var}|{obj}"]
        L.append(f"| {nm} | {obj} | {len(d['seeds'])} | {d['t_sel'][0]:+.3f} | {d['t_cal'][0]:+.3f} | {f4(d['k0'])} | {f4(d['k_shared'])} | {f4(d['k_indep'])} | {d['val_k_sel'][0]:.4f} / {d['val_k_cal'][0]:.4f} |")
L += ["", "| regime | objective | PO - original | seed diffs | positive seeds | 95% CI |", "|---|---|---|---|---|---|"]
for c in contrasts:
    L.append(f"| {c['regime']} | {c['objective']} | {c['po_minus_original']:+.4f} | {', '.join(f'{x:+.4f}' for x in c['seed_diffs'])} | {c['positive_seeds']}/{len(c['seed_diffs'])} | [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] |")
L += ["", "| set | quantity | mean | 2.5% | 97.5% |", "|---|---|---|---|---|"]
for setname, nm in (("sel", "shared validation (sel)"), ("cal", "independent pool")):
    for k, v in variation[setname].items():
        L.append(f"| {nm} | {k} | {v['mean']:+.4f} | {v['p2_5']:+.4f} | {v['p97_5']:+.4f} |")
(OUT / "calib393_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L)); print(json.dumps(chk)); print(json.dumps(indep))
