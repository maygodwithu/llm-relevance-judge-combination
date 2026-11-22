"""Supplementary analysis of the training budget (Qwen3-4B, ESCI T1): learning-rate / training-duration comparison and validation curves from saved checkpoints. Read-only on all stored scores; writes only into --out.
Stages:
  reuse  : ESCI T1 Qwen3-4B / Gemma-3-4B-it CE sensitivity (257/259) - every stored candidate x seed: validation kappa(0), kappa(t*), t*, test kappa(0), kappa(t*), AUC, training seconds; seed mean / sample SD.
           lr 1e-5 (seed 0 only) is listed separately and never enters a 3-seed row. No setting is selected here; no seed-wise test maximum is taken.
  curves : Qwen stored CE and CE+soft kappa (lr 2e-5, 625 updates), seeds 0-2: validation kappa(0) and kappa(t*) at update 63/157/313/469/625 (t* from the validation scores of that checkpoint),
           loss-component logs (CE per update; soft-kappa term U/D) as plot-ready CSV. Reproduction gate: step-625 validation scores recomputed here vs the stored ones.
  long   : new Qwen CE, lr 2e-5, N updates (frozen budget), seeds 0-2 vs stored CE+soft kappa and stored CE (625): kappa(0), kappa(t*), seed-wise paired differences, mean / SD,
           paired test-query bootstrap (B = 10,000, default_rng(380), seed means per draw, conditional on the fixed models and thresholds), training-time ratio.
    python report380.py --root <project root> --out <dir> --stage reuse|curves|long|all"""
import argparse, glob, gzip, hashlib, json, os, sys, statistics as stt
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import esci204_common as C  # noqa: E402  (kap, kap_vec, best_threshold, auc)

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--out", required=True); ap.add_argument("--stage", default="all", choices=["reuse", "curves", "long", "all"])
a = ap.parse_args()
ROOT = Path(a.root); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
IDX = {}                                   # basename -> every file of that name (other model sizes reuse the same tags, e.g. Qwen3-8B runs E4_T1_s0 ...)
for f in glob.glob(str(ROOT / "cpmr_experiment" / "**" / "*.json.gz"), recursive=True):
    bn = os.path.basename(f)
    if bn.startswith("esciEval_") or bn.startswith("esciGemmaEval_"):
        IDX.setdefault(bn, []).append(f)
STATS = {}
WANT = {"esciEval_": "Qwen3-4B", "esciGemmaEval_": "gemma-3-4b-it"}
for f in glob.glob(str(ROOT / "cpmr_experiment" / "**" / "stats_*_T1_s*.json"), recursive=True):
    d = json.load(open(f)); mn = os.path.basename(str(d.get("model", ""))).lower()
    if mn and not any(w.lower() in mn for w in WANT.values()):
        continue                                # a stats file of another model size
    STATS.setdefault(os.path.basename(f)[6:-5], []).append((f, mn))
_CACHE = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def arrays(path):
    raw = json.load(gzip.open(path, "rt", encoding="utf-8"))["raw"]
    ks = [(q, d) for q in sorted(raw) for d in sorted(raw[q])]
    m = np.array([raw[q][d][0] + raw[q][d][1] - raw[q][d][3] - raw[q][d][4] for q, d in ks])
    y = np.array([raw[q][d][10] >= 2 for q, d in ks])
    return ks, m, y


def kat(m, y, t):
    p = m > t
    return float(C.kap(int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum())))


def find(pre, model, tag):
    """the evaluation file of this tag for the right base model: files of another model size are dropped by meta.model_name; the remaining copies must carry identical scores"""
    key = (pre, model, tag)
    if key in _CACHE:
        return _CACHE[key]
    cands = IDX.get(f"{pre}{model}__T1__{tag}.json.gz", [])
    good = [c for c in cands if WANT[pre].lower() in os.path.basename(str(json.load(gzip.open(c, "rt", encoding="utf-8"))["meta"]["model_name"])).lower()]
    if not good:
        _CACHE[key] = None; return None
    ref = arrays(good[0])[1]
    for c in good[1:]:
        assert np.allclose(arrays(c)[1], ref, atol=1e-6, rtol=0), ("copies of the same tag differ", model, tag, c)
    _CACHE[key] = good[0]
    return good[0]


FAM = {"Qwen3-4B": dict(pre="esciEval_", ce="E4_T1_s{s}_step625", l5="E255L5_T1_s{s}_step625", ep2="E255EP2_T1_s{s}_step1250", soft="EQ_mix_T1_s{s}_step625", l1="E255L1_T1_s0_step625", long="E380_T1_s{s}_step{N}"),
       "Gemma-3-4B": dict(pre="esciGemmaEval_", ce="GQ_ce_T1_s{s}_step625", l5="G255L5_ce_T1_s{s}_step625", ep2="G255EP2_ce_T1_s{s}_step1250", soft="GQ_mix_T1_s{s}_step625", l1="G255L1_ce_T1_s0_step625")}
LABEL = {"ce": "CE lr 2e-5 / 625 (stored main setting)", "l5": "CE lr 5e-5 / 625", "ep2": "CE lr 2e-5 / 1250", "soft": "CE+soft kappa lambda 1, lr 2e-5 / 625 (stored main setting)", "l1": "CE lr 1e-5 / 625 (seed 0 only, auxiliary)",
         "long": "CE lr 2e-5 / N updates (new, comparable-training-cost run)"}
TRAIN_TAG = {"Qwen3-4B": dict(ce="E4_T1_s{s}", l5="E255L5_T1_s{s}", ep2="E255EP2_T1_s{s}", soft="EQ_mix_T1_s{s}", l1="E255L1_T1_s0", long="E380_T1_s{s}"),
             "Gemma-3-4B": dict(ce="GQ_ce_T1_s{s}", l5="G255L5_ce_T1_s{s}", ep2="G255EP2_ce_T1_s{s}", soft="GQ_mix_T1_s{s}", l1="G255L1_ce_T1_s0")}
N_LONG = json.load(open(ROOT / "cpmr_experiment/applsci380/selection/budget_manifest.json"))["N_updates"]


def cell(model, key, s):
    f = FAM[model]; pre = f["pre"]; nm = f[key].format(s=s, N=N_LONG)
    pv, pt = find(pre, nm, "sel"), find(pre, nm, "test300")
    if pv is None or pt is None:
        return None
    ks, mv, yv = arrays(pv); kt, mt, yt = arrays(pt)
    thr, vk = C.best_threshold(mv, yv)
    tag = TRAIN_TAG[model][key].format(s=s)
    if tag in STATS:
        assert len({json.load(open(p))["train_seconds"] for p, _ in STATS[tag]}) == 1, ("conflicting stats", tag)
    sec = json.load(open(STATS[tag][0][0]))["train_seconds"] if tag in STATS else None
    return dict(model=model, setting=key, label=LABEL[key], seed=s, val_kappa_t0=kat(mv, yv, 0.0), val_kappa_tstar=vk, tstar=thr, test_kappa_t0=kat(mt, yt, 0.0), test_kappa_tstar=kat(mt, yt, thr), test_auc=float(C.auc(mt, yt)),
                test_rel_t0=float((mt > 0).mean()), test_rel_tstar=float((mt > thr).mean()), train_seconds=sec, val_sha256=sha(pv), test_sha256=sha(pt), _mt=mt, _yt=yt, _kt=kt, _thr=thr)


def strip(c):
    return {k: v for k, v in c.items() if not k.startswith("_")}


def msd(v):
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"))


def stage_reuse():
    rows, summ = [], []
    for model in FAM:
        for key in ("ce", "l5", "ep2", "soft", "l1"):
            seeds = (0,) if key == "l1" else (0, 1, 2)
            cs = [cell(model, key, s) for s in seeds]
            assert all(c is not None for c in cs), (model, key)
            rows += [strip(c) for c in cs]
            d = dict(model=model, setting=key, label=LABEL[key], n_seeds=len(cs))
            for k in ("val_kappa_t0", "val_kappa_tstar", "test_kappa_t0", "test_kappa_tstar", "test_auc", "train_seconds"):
                v = [c[k] for c in cs]; d[k + "_mean"], d[k + "_sd"] = msd(v)
            summ.append(d)
    (OUT / "reuse_per_seed.json").write_text(json.dumps(dict(rows=rows, summary=summ), indent=1), encoding="utf-8")
    import csv
    with open(OUT / "reuse_per_seed.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); cols = [k for k in rows[0]]
        w.writerow(cols); [w.writerow([r[k] for k in cols]) for r in rows]
    L = ["| model | setting | seeds | val k(0) | val k(t*) | test k(0) | test k(t*) | AUC | train s |", "|---|---|---|---|---|---|---|---|---|"]
    for d in summ:
        f = lambda k: f"{d[k + '_mean']:.4f}" + (f" ± {d[k + '_sd']:.4f}" if d["n_seeds"] > 1 else "")
        L.append(f"| {d['model']} | {d['label']} | {d['n_seeds']} | {f('val_kappa_t0')} | {f('val_kappa_tstar')} | {f('test_kappa_t0')} | {f('test_kappa_tstar')} | {f('test_auc')} | {d['train_seconds_mean']:.0f} |")
    (OUT / "reuse_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def stage_curves():
    import csv
    R = ROOT / "cpmr_experiment"
    RUN = {"CE": ("E4_T1_s{s}", "esci_t"), "soft": ("EQ_mix_T1_s{s}", "esci_mix")}
    rows, gate = [], []
    for kind, (tmpl, d) in RUN.items():
        for s in (0, 1, 2):
            tag = tmpl.format(s=s)
            for step in (63, 157, 313, 469, 625):
                nm = f"{tag}_step{step}"
                pv = (R / "applsci380/curves" / f"esciEval_{nm}__T1__sel.json.gz") if step != 625 else (Path(find("esciEval_", nm, "sel")) if find("esciEval_", nm, "sel") else None)
                # step 625 always from the stored file; 63..469 from the new validation-only evaluation
                if step != 625 and not pv.exists():
                    continue
                ks, mv, yv = arrays(pv)
                thr, vk = C.best_threshold(mv, yv)
                rows.append(dict(objective=kind, seed=s, step=step, val_kappa_t0=kat(mv, yv, 0.0), val_kappa_tstar=vk, tstar=thr, val_rel_t0=float((mv > 0).mean()), val_rel_tstar=float((mv > thr).mean()), source=("stored" if step == 625 else "new validation-only evaluation"), file_sha256=sha(pv)))
            if s == 0:                      # reproduction gate of the evaluator: step-625 validation scores recomputed vs stored
                nm = f"{tag}_step625"; pn = R / "applsci380/curves" / f"esciEval_{nm}__T1__sel.json.gz"
                if pn.exists():
                    k1, m1, _ = arrays(pn); k2, m2, _ = arrays(Path(find("esciEval_", nm, "sel")))
                    assert k1 == k2
                    gate.append(dict(model=nm, max_abs_margin_diff=float(np.abs(m1 - m2).max()), decisions_equal_t0=bool(((m1 > 0) == (m2 > 0)).all()), criterion=1e-4, passed=bool(np.abs(m1 - m2).max() <= 1e-4)))
    (OUT / "curves_validation.json").write_text(json.dumps(dict(rows=rows, gate=gate), indent=1), encoding="utf-8")
    with open(OUT / "curves_validation.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); cols = list(rows[0]); w.writerow(cols); [w.writerow([r[k] for k in cols]) for r in rows]
    # loss-component logs, plot ready: one row per update (CE; soft kappa term U/D) and 63-update window means
    cols = ["objective", "seed", "step", "ce", "lr", "kappa_term_U_over_D", "grad_norm"]
    out = []
    for kind, (tmpl, d) in RUN.items():
        for s in (0, 1, 2):
            tag = tmpl.format(s=s)
            for line in open(R / d / "train/logs" / f"steps_{tag}.jsonl", encoding="utf-8"):
                r = json.loads(line)
                kt = (r["U"] / max(r["D"], 1e-6)) if "U" in r else ""
                out.append([kind, s, r["step"], r["ce"], r.get("lr", ""), kt, r.get("grad_norm", "")])
    with open(OUT / "loss_components_per_update.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(cols); w.writerows(out)
    print("curves rows", len(rows), "gate", gate)


def stage_long():
    B = 10000
    res = dict(N_updates=N_LONG, rows=[], pairs=[])
    model = "Qwen3-4B"
    cells = {k: [cell(model, k, s) for s in (0, 1, 2)] for k in ("ce", "soft", "long")}
    assert all(c is not None for v in cells.values() for c in v), "long CE evaluation files missing"
    for k, v in cells.items():
        res["rows"] += [strip(c) for c in v]
    ks = None
    # paired bootstrap on test queries: one draw set shared by every method and seed
    pt = {k: [find(FAM[model]["pre"], FAM[model][k].format(s=s, N=N_LONG), "test300") for s in (0, 1, 2)] for k in cells}
    kk, _, _ = arrays(pt["ce"][0]); qs = sorted({q for q, _ in kk}); qid = {q: i for i, q in enumerate(qs)}; rowq = np.array([qid[q] for q, _ in kk]); nq = len(qs)
    rng = np.random.default_rng(380)
    W = np.stack([np.bincount(i, minlength=nq) for i in rng.integers(0, nq, size=(B, nq))]).astype(float)

    def kb(c, mode):
        assert [q for q, _ in c["_kt"]] == [q for q, _ in kk]
        t = 0.0 if mode == "t0" else c["_thr"]; pr = c["_mt"] > t; y = c["_yt"]
        cnt = np.stack([np.bincount(rowq, weights=x.astype(float), minlength=nq) for x in (pr & y, pr & ~y, ~pr & y, ~pr & ~y)], 1)
        tot = cnt.sum(0); bt = W @ cnt
        return float(C.kap(*tot)), C.kap_vec(bt[:, 0], bt[:, 1], bt[:, 2], bt[:, 3])
    for mode, key in (("t0", "test_kappa_t0"), ("tstar", "test_kappa_tstar")):
        pt_k = {k: [kb(c, mode) for c in v] for k, v in cells.items()}
        mean_pt = {k: float(np.mean([x[0] for x in v])) for k, v in pt_k.items()}
        mean_bt = {k: np.mean([x[1] for x in v], 0) for k, v in pt_k.items()}
        for A_, B_ in (("long", "soft"), ("long", "ce"), ("soft", "ce")):
            sd = [pt_k[A_][i][0] - pt_k[B_][i][0] for i in range(3)]
            d = mean_pt[A_] - mean_pt[B_]; bs = mean_bt[A_] - mean_bt[B_]
            res["pairs"].append(dict(threshold=mode, contrast=f"{A_} - {B_}", diff=d, seed_diffs=sd, seed_mean=float(np.mean(sd)), seed_sd=float(np.std(sd, ddof=1)), n_positive_seeds=int(sum(x > 0 for x in sd)),
                                     ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], note="paired test-query bootstrap, 10,000 draws, default_rng(380), conditional on the fixed models and thresholds"))
    secs = {k: [c["train_seconds"] for c in v] for k, v in cells.items()}
    res["training_seconds"] = secs
    res["time_ratio_long_over_soft_median"] = float(np.median(secs["long"]) / np.median(secs["soft"]))
    res["time_ratio_long_over_ce_median"] = float(np.median(secs["long"]) / np.median(secs["ce"]))
    (OUT / "long_ce_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    L = ["| setting | seed | val k(0) | val k(t*) | t* | test k(0) | test k(t*) | AUC | train s |", "|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        L.append(f"| {r['setting']} | {r['seed']} | {r['val_kappa_t0']:.4f} | {r['val_kappa_tstar']:.4f} | {r['tstar']:+.3f} | {r['test_kappa_t0']:.4f} | {r['test_kappa_tstar']:.4f} | {r['test_auc']:.4f} | {r['train_seconds']:.0f} |")
    L += ["", "| threshold | contrast | diff | seed diffs | positive seeds | 95% CI |", "|---|---|---|---|---|---|"]
    for p in res["pairs"]:
        L.append(f"| {p['threshold']} | {p['contrast']} | {p['diff']:+.4f} | {', '.join(f'{x:+.4f}' for x in p['seed_diffs'])} | {p['n_positive_seeds']}/3 | [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] |")
    L.append(f"\ntime ratio long / soft (median) {res['time_ratio_long_over_soft_median']:.3f}; long / CE625 {res['time_ratio_long_over_ce_median']:.3f}")
    (OUT / "long_ce_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if a.stage in ("reuse", "all"):
    stage_reuse()
if a.stage in ("curves", "all"):
    stage_curves()
if a.stage in ("long", "all"):
    stage_long()
