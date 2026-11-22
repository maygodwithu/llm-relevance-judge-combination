"""BCE / MAE controls extended to ESCI Qwen3-4B T1-T7 (seeds 0-2, lr 2e-5 / 625 updates). Read-only on all stored scores.
Objectives: CE (stored E4), CE+soft kappa lambda 1 (stored EQ_mix), CE+2MAE (T1: Q384M, T2-T7: Q407M), CE+BCE beta 1 (T1: Q245B seed 0 / Q384B seeds 1-2, T2-T7: Q407B).
Per T x seed: validation / test kappa(0), kappa(t*) (t* from each run's own validation scores, existing rule), AUC, Relevant-prediction rate, training time.
Aggregation: per seed the mean over T1-T7, then mean +- sample SD over the 3 seeds (no pooled-raw kappa). Paired contrasts with a paired test-query bootstrap (10,000 draws, default_rng(407),
ONE draw set shared by all objectives / instructions / seeds, conditional on fixed models and thresholds, exploratory).
    python report407.py --root <project root> --out <dir>"""
import argparse, csv, glob, gzip, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--out", required=True); a = ap.parse_args()
ROOT = Path(a.root); R = ROOT / "cpmr_experiment"; OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R / "analysis"))
import esci204_common as C  # noqa: E402

IDX = {}
for f in glob.glob(str(R / "**" / "esciEval_*.json.gz"), recursive=True):
    IDX.setdefault(os.path.basename(f), []).append(f)
_man = json.load(open(R / "esci_t/esci_t_manifest.json"))["entries"]
MAN = {e["display_id"]: e for e in _man}
_STATS = {}
for f in glob.glob(str(R / "**" / "stats_*.json"), recursive=True):
    _STATS.setdefault(os.path.basename(f)[len("stats_"):-len(".json")], []).append(f)
_META = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def load(path):
    return json.load(gzip.open(path, "rt", encoding="utf-8"))


def arrays(path):
    raw = load(path)["raw"]
    ks = [(q, d) for q in sorted(raw) for d in sorted(raw[q])]
    m = np.array([raw[q][d][0] + raw[q][d][1] - raw[q][d][3] - raw[q][d][4] for q, d in ks])
    y = np.array([raw[q][d][10] >= 2 for q, d in ks])
    return ks, m, y


def find(tag, k, st):
    cands = IDX.get(f"esciEval_{tag}_step625__T{k}__{st}.json.gz", [])
    good = []
    for c in cands:
        mn = os.path.basename(str(load(c)["meta"]["model_name"])).lower()
        if "qwen3-4b" in mn:
            good.append(c)
    assert good, ("missing", tag, st)
    ref = arrays(good[0])[1]
    for c in good[1:]:
        assert np.allclose(arrays(c)[1], ref, atol=1e-6, rtol=0), ("copies differ", tag, st)
    return good[0]


def kat(m, y, t):
    p = m > t
    return float(C.kap(int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum())))


OBJS = ["CE", "CE+soft kappa", "CE+2MAE", "CE+BCE"]


def tag_of(obj, k, s):
    if obj == "CE":
        return f"E4_T{k}_s{s}"
    if obj == "CE+soft kappa":
        return f"EQ_mix_T{k}_s{s}"
    if obj == "CE+2MAE":
        return f"Q384M_mixmae_T1_s{s}" if k == 1 else f"Q407M_mixmae_T{k}_s{s}"
    if k == 1:
        return "Q245B_mixbce_T1_s0" if s == 0 else f"Q384B_mixbce_T1_s{s}"
    return f"Q407B_mixbce_T{k}_s{s}"


def source_of(obj, k, s):
    if obj in ("CE", "CE+soft kappa") or (obj == "CE+BCE" and k == 1 and s == 0):
        return "stored (reused)"
    return "new (msg 385)" if k == 1 else "new (msg 407)"


def stats_path(tag):
    g = _STATS.get(tag, [])
    # keep only Qwen3-4B ESCI stats: tag names are unique per model for these tags; take the first and report count
    return g[0] if g else None


Ks, Ss = range(1, 8), (0, 1, 2)
cells = {}; test_keys = None; val_keys = None
for k in Ks:
    ie = MAN[f"T{k}"]["sha256_full"]
    for obj in OBJS:
        for s in Ss:
            t = tag_of(obj, k, s); pv, pt = find(t, k, "sel"), find(t, k, "test300")
            kv, mv, yv = arrays(pv); kt, mt, yt = arrays(pt)
            if test_keys is None:
                test_keys, val_keys = kt, kv
            assert kt == test_keys and kv == val_keys, ("row set differs", t)
            mvm = load(pv)["meta"]; mtm = load(pt)["meta"]
            assert mvm["instruction_sha256"] == ie == mtm["instruction_sha256"], ("instruction hash", t)
            thr, vk = C.best_threshold(mv, yv)
            sp = stats_path(t); ts = json.load(open(sp)).get("train_seconds") if sp else None
            cells[(k, obj, s)] = dict(instruction=f"T{k}", objective=obj, seed=s, tag=t, source=source_of(obj, k, s), instruction_sha256=ie,
                                      val_kappa_t0=kat(mv, yv, 0.0), val_kappa_tstar=vk, tstar=thr, val_auc=float(C.auc(mv, yv)),
                                      test_kappa_t0=kat(mt, yt, 0.0), test_kappa_tstar=kat(mt, yt, thr), test_auc=float(C.auc(mt, yt)),
                                      test_rel_t0=float((mt > 0).mean()), test_rel_tstar=float((mt > thr).mean()), val_rel_t0=float((mv > 0).mean()), val_rel_tstar=float((mv > thr).mean()),
                                      train_seconds=ts, val_sha256=sha(pv), test_sha256=sha(pt), _mt=mt, _yt=yt, _kt=kt, _thr=thr)
strip = lambda c: {k_: v for k_, v in c.items() if not k_.startswith("_")}
rows = [strip(c) for c in cells.values()]


def wcsv(name, rs):
    with open(OUT / name, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rs[0]), lineterminator="\n"); w.writeheader(); w.writerows(rs)


wcsv("per_instruction_seed.csv", rows)

MET = ("val_kappa_t0", "val_kappa_tstar", "test_kappa_t0", "test_kappa_tstar", "test_auc", "test_rel_t0", "test_rel_tstar")
per_instr = []
for k in Ks:
    for obj in OBJS:
        d = dict(instruction=f"T{k}", objective=obj)
        for m_ in MET:
            v = [cells[(k, obj, s)][m_] for s in Ss]; d[m_ + "_mean"] = float(np.mean(v)); d[m_ + "_sd"] = float(np.std(v, ddof=1))
        per_instr.append(d)
wcsv("per_instruction_summary.csv", per_instr)
per_seed = []; summ = []
for obj in OBJS:
    for s in Ss:
        d = dict(objective=obj, seed=s)
        for m_ in MET:
            d[m_ + "_T_mean"] = float(np.mean([cells[(k, obj, s)][m_] for k in Ks]))
        per_seed.append(d)
    d = dict(objective=obj)
    for m_ in MET:
        v = [x[m_ + "_T_mean"] for x in per_seed if x["objective"] == obj]; d[m_ + "_mean"] = float(np.mean(v)); d[m_ + "_sd"] = float(np.std(v, ddof=1))
    summ.append(d)
wcsv("per_seed_T_mean.csv", per_seed); wcsv("overall_summary.csv", summ)

# paired bootstrap, ONE draw set for everything
B = 10000
qs = sorted({q for q, _ in test_keys}); qid = {q: i for i, q in enumerate(qs)}; rowq = np.array([qid[q] for q, _ in test_keys]); nq = len(qs)
rng = np.random.default_rng(407)
W = np.stack([np.bincount(i, minlength=nq) for i in rng.integers(0, nq, size=(B, nq))]).astype(float)


def kb(c, mode):
    t = 0.0 if mode == "t0" else c["_thr"]; pr = c["_mt"] > t; y = c["_yt"]
    cnt = np.stack([np.bincount(rowq, weights=x.astype(float), minlength=nq) for x in (pr & y, pr & ~y, ~pr & y, ~pr & ~y)], 1)
    tot = cnt.sum(0); bt = W @ cnt
    return float(C.kap(*tot)), C.kap_vec(bt[:, 0], bt[:, 1], bt[:, 2], bt[:, 3])


CONTR = [("CE+soft kappa", "CE+2MAE"), ("CE+soft kappa", "CE+BCE"), ("CE+2MAE", "CE"), ("CE+BCE", "CE"), ("CE+soft kappa", "CE")]
K = len(CONTR)
pairs_overall = []; pairs_instr = []
for mode, key in (("t0", "kappa(0)"), ("tstar", "kappa(t*)")):
    pk = {(k, o, s): kb(cells[(k, o, s)], mode) for k in Ks for o in OBJS for s in Ss}
    for A_, B_ in CONTR:
        sd = [float(np.mean([pk[(k, A_, s)][0] - pk[(k, B_, s)][0] for k in Ks])) for s in Ss]
        bs = np.mean([pk[(k, A_, s)][1] - pk[(k, B_, s)][1] for k in Ks for s in Ss], 0)
        cell_d = [pk[(k, A_, s)][0] - pk[(k, B_, s)][0] for k in Ks for s in Ss]
        pairs_overall.append(dict(threshold=key, contrast=f"{A_} - {B_}", diff=float(np.mean(sd)), seed_diffs=sd, n_positive_seeds=int(sum(x > 0 for x in sd)), seed_sd=float(np.std(sd, ddof=1)),
                                  n_positive_cells_of_21=int(sum(x > 0 for x in cell_d)), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                  ci_bonferroni_K5=[float(np.percentile(bs, 100 * 0.05 / K / 2)), float(np.percentile(bs, 100 - 100 * 0.05 / K / 2))]))
        for k in Ks:
            sdk = [pk[(k, A_, s)][0] - pk[(k, B_, s)][0] for s in Ss]; bk = np.mean([pk[(k, A_, s)][1] - pk[(k, B_, s)][1] for s in Ss], 0)
            pairs_instr.append(dict(threshold=key, contrast=f"{A_} - {B_}", instruction=f"T{k}", diff=float(np.mean(sdk)), seed_diffs=sdk, n_positive_seeds=int(sum(x > 0 for x in sdk)),
                                    ci95=[float(np.percentile(bk, 2.5)), float(np.percentile(bk, 97.5))]))
with open(OUT / "paired_contrasts_instruction.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["threshold", "contrast", "instruction", "diff_seed_mean", "diff_s0", "diff_s1", "diff_s2", "positive_seeds", "ci95_low", "ci95_high"])
    for p in pairs_instr:
        w.writerow([p["threshold"], p["contrast"], p["instruction"], f"{p['diff']:.6f}", *[f"{x:.6f}" for x in p["seed_diffs"]], p["n_positive_seeds"], f"{p['ci95'][0]:.6f}", f"{p['ci95'][1]:.6f}"])


# run audit of the 36 new runs of this message
def Ljs(p):
    return json.load(open(p))


def same(a0, b0):
    return set(a0) == set(b0) and all(torch.equal(a0[x], b0[x]) for x in a0)


audit = []
for k in range(2, 8):
    for obj, pre, term in (("CE+2MAE", "Q407M_mixmae", "mae_loss"), ("CE+BCE", "Q407B_mixbce", "bce_loss")):
        for s in Ss:
            t = f"{pre}_T{k}_s{s}"; st = Ljs(R / "esci407/train/logs" / f"stats_{t}.json")
            steps = [json.loads(l) for l in open(R / "esci407/train/logs" / f"steps_{t}.jsonl", encoding="utf-8")]
            ce_ref = [json.loads(l) for l in open(R / "esci_t/train/logs" / f"steps_E4_T{k}_s{s}.jsonl", encoding="utf-8")][:2]
            ce_stats = Ljs(R / "esci_t/train/logs" / f"stats_E4_T{k}_s{s}.json")
            soft_p = stats_path(f"EQ_mix_T{k}_s{s}"); soft_st = Ljs(soft_p) if soft_p else None
            comp = max(abs(r["loss"] - (r["ce"] + st["lam"] * r[term])) for r in steps)
            a0 = torch.load(R / "esci407/train/checkpoints" / t / "ckpt_step0.pt", map_location="cpu", weights_only=False)["adapter_state"]
            ref0 = torch.load(R / "esci_t/train/checkpoints" / f"E4_T{k}_s{s}" / "ckpt_step0.pt", map_location="cpu", weights_only=False)["adapter_state"]
            soft0 = None
            sg = glob.glob(str(R / "**" / f"EQ_mix_T{k}_s{s}" / "ckpt_step0.pt"), recursive=True)
            if sg:
                soft0 = same(a0, torch.load(sg[0], map_location="cpu", weights_only=False)["adapter_state"])
            audit.append(dict(tag=t, objective=obj, instruction=f"T{k}", seed=s, loss=st["loss"], coefficient=st["lam"], updates=len(steps), optimizer_updates=st["optimizer_updates"], skipped=st["skipped_updates"], nonfinite=st["nonfinite"],
                              inputs_shortened=st["inputs_shortened"], max_length=st["max_length"], micro_batch=st["micro_batch"],
                              rows_hash_equal_stored_CE=st["rows_file_sha256"] == ce_stats["rows_file_sha256"],
                              rows_hash_equal_stored_soft=(st["rows_file_sha256"] == soft_st["rows_file_sha256"]) if soft_st else None,
                              order_hash_equal_stored_soft=(st["order_sha256"] == soft_st.get("order_sha256")) if soft_st and soft_st.get("order_sha256") else None,
                              step0_adapter_equal_stored_CE=same(a0, ref0), step0_adapter_equal_stored_soft=soft0,
                              update1_ce_equal_stored_CE=steps[0]["ce"] == ce_ref[0]["ce"], update2_ce_abs_diff_vs_stored_CE=abs(steps[1]["ce"] - ce_ref[1]["ce"]),
                              max_abs_loss_composition_error=comp, sequences_forwarded=st["sequences_forwarded"], train_seconds=st["train_seconds"], peak_gb=st["peak_memory_gb"],
                              mean_r=float(np.mean([r["r"] for r in steps])), mean_D=float(np.mean([r["D"] for r in steps]))))


def okf(x):
    return (x["optimizer_updates"] == 625 and x["skipped"] == 0 and x["nonfinite"] == 0 and x["inputs_shortened"] == 0 and x["rows_hash_equal_stored_CE"] and x["step0_adapter_equal_stored_CE"]
            and x["max_abs_loss_composition_error"] < 1e-6 and x["order_hash_equal_stored_soft"] is not False and x["step0_adapter_equal_stored_soft"] is not False)


ok = all(okf(x) for x in audit)
wcsv("run_audit.csv", audit)
res = dict(overall=summ, per_instruction=per_instr, per_seed_T_mean=per_seed, pairs_overall=pairs_overall, pairs_instruction=pairs_instr, audit=audit, audit_passed=bool(ok),
           note="paired test-query bootstrap, 10,000 draws, default_rng(407), one shared draw set for all objectives/instructions/seeds; point estimate = mean over seeds of the T1-T7 mean of per-run kappa (no pooled raw data); CI conditional on fixed models and validation-selected thresholds; exploratory; Bonferroni K = 5 contrasts (overall only); per-instruction intervals unadjusted")
(OUT / "bce_mae_t1t7_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
f = lambda d, k_: f"{d[k_ + '_mean']:.4f} ± {d[k_ + '_sd']:.4f}"
Lm = ["## Overall (mean over seeds of the T1-T7 mean, ± sample SD over 3 seeds)", "| objective | val k(0) | val k(t*) | test k(0) | test k(t*) | test AUC | Relevant rate t=0 | Relevant rate t* |", "|---|---|---|---|---|---|---|---|"]
for d in summ:
    Lm.append(f"| {d['objective']} | " + " | ".join(f(d, m_) for m_ in ("val_kappa_t0", "val_kappa_tstar", "test_kappa_t0", "test_kappa_tstar", "test_auc", "test_rel_t0", "test_rel_tstar")) + " |")
Lm += ["", "## Paired contrasts, overall (T1-T7, 3 seeds)", "| threshold | contrast | diff | seed diffs | positive seeds | positive cells /21 | 95% CI | Bonferroni K=5 CI |", "|---|---|---|---|---|---|---|---|"]
for p in pairs_overall:
    Lm.append(f"| {p['threshold']} | {p['contrast']} | {p['diff']:+.4f} | {', '.join(f'{x:+.4f}' for x in p['seed_diffs'])} | {p['n_positive_seeds']}/3 | {p['n_positive_cells_of_21']} | [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] | [{p['ci_bonferroni_K5'][0]:+.4f}, {p['ci_bonferroni_K5'][1]:+.4f}] |")
for lab, key in (("test kappa(0)", "test_kappa_t0"), ("test kappa(t*)", "test_kappa_tstar")):
    Lm += ["", f"## Per instruction, {lab} mean ± SD over 3 seeds", "| T | " + " | ".join(OBJS) + " |", "|---|" + "---|" * len(OBJS)]
    for k in Ks:
        Lm.append(f"| T{k} | " + " | ".join(f(next(d for d in per_instr if d['instruction'] == f'T{k}' and d['objective'] == o), key) for o in OBJS) + " |")
Lm += ["", f"audit passed (36 new runs): {ok}"]
(OUT / "bce_mae_t1t7_results.md").write_text("\n".join(Lm) + "\n", encoding="utf-8")
print("\n".join(Lm))
