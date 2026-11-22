"""BCE / MAE controls on ESCI Qwen3-4B T1 (seeds 0-2, lr 2e-5 / 625 updates; superseded for the paper by report407.py, which covers T1-T7 and reproduces these T1 values). Read-only on all stored scores.
Objectives (same 3 seeds): CE (stored), CE+soft kappa lambda 1 (stored), CE+2MAE (new: Q384M mixmae lam 2), CE+BCE beta 1 (seed 0 stored 245 run Q245B, seeds 1-2 new Q384B).
Reports validation / test kappa(0), kappa(t*) (t* from the validation scores of each run), AUC, seed values, mean / sample SD, seed-wise paired differences and paired test-query bootstrap contrasts
(10,000 draws, default_rng(384), one draw set shared by every method and seed, seed means per draw, conditional on the fixed models and thresholds, exploratory 95% plus Bonferroni K = 5), and the run audit (hashes, adapter step 0, first updates, loss composition).
    python report384.py --root <project root> --out <dir>"""
import argparse, glob, gzip, hashlib, json, os, sys
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


def find(tag, st):
    cands = IDX.get(f"esciEval_{tag}_step625__T1__{st}.json.gz", [])
    good = [c for c in cands if "qwen3-4b" in os.path.basename(str(json.load(gzip.open(c, "rt", encoding="utf-8"))["meta"]["model_name"])).lower()]
    assert good, ("missing", tag, st)
    ref = arrays(good[0])[1]
    for c in good[1:]:
        assert np.allclose(arrays(c)[1], ref, atol=1e-6, rtol=0), ("copies differ", tag, st)
    return good[0]


def kat(m, y, t):
    p = m > t
    return float(C.kap(int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum())))


OBJ = {"CE": "E4_T1_s{s}", "CE+soft kappa": "EQ_mix_T1_s{s}", "CE+2MAE": "Q384M_mixmae_T1_s{s}", "CE+BCE": None}
def tag_of(obj, s):
    if obj == "CE+BCE":
        return "Q245B_mixbce_T1_s0" if s == 0 else f"Q384B_mixbce_T1_s{s}"
    return OBJ[obj].format(s=s)
REUSED = {("CE", s) for s in (0, 1, 2)} | {("CE+soft kappa", s) for s in (0, 1, 2)} | {("CE+BCE", 0)}

cells = {}
for obj in OBJ:
    for s in (0, 1, 2):
        t = tag_of(obj, s); pv, pt = find(t, "sel"), find(t, "test300")
        _, mv, yv = arrays(pv); kt, mt, yt = arrays(pt); thr, vk = C.best_threshold(mv, yv)
        cells[(obj, s)] = dict(objective=obj, seed=s, tag=t, source="stored (reused)" if (obj, s) in REUSED else "new", val_kappa_t0=kat(mv, yv, 0.0), val_kappa_tstar=vk, tstar=thr, test_kappa_t0=kat(mt, yt, 0.0), test_kappa_tstar=kat(mt, yt, thr), test_auc=float(C.auc(mt, yt)),
                               test_rel_t0=float((mt > 0).mean()), test_rel_tstar=float((mt > thr).mean()), val_sha256=sha(pv), test_sha256=sha(pt), _mt=mt, _yt=yt, _kt=kt, _thr=thr)
strip = lambda c: {k: v for k, v in c.items() if not k.startswith("_")}
rows = [strip(c) for c in cells.values()]
import csv
with open(OUT / "objectives_per_seed.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
summ = []
for obj in OBJ:
    d = dict(objective=obj)
    for k in ("val_kappa_t0", "val_kappa_tstar", "test_kappa_t0", "test_kappa_tstar", "test_auc"):
        v = [cells[(obj, s)][k] for s in (0, 1, 2)]; d[k + "_mean"] = float(np.mean(v)); d[k + "_sd"] = float(np.std(v, ddof=1))
    summ.append(d)

# paired contrasts
B = 10000
kk = cells[("CE", 0)]["_kt"]; qs = sorted({q for q, _ in kk}); qid = {q: i for i, q in enumerate(qs)}; rowq = np.array([qid[q] for q, _ in kk]); nq = len(qs)
rng = np.random.default_rng(384)
W = np.stack([np.bincount(i, minlength=nq) for i in rng.integers(0, nq, size=(B, nq))]).astype(float)
def kb(c, mode):
    assert [q for q, _ in c["_kt"]] == [q for q, _ in kk]
    t = 0.0 if mode == "t0" else c["_thr"]; pr = c["_mt"] > t; y = c["_yt"]
    cnt = np.stack([np.bincount(rowq, weights=x.astype(float), minlength=nq) for x in (pr & y, pr & ~y, ~pr & y, ~pr & ~y)], 1)
    tot = cnt.sum(0); bt = W @ cnt
    return float(C.kap(*tot)), C.kap_vec(bt[:, 0], bt[:, 1], bt[:, 2], bt[:, 3])
CONTR = [("CE+soft kappa", "CE+2MAE"), ("CE+soft kappa", "CE+BCE"), ("CE+2MAE", "CE"), ("CE+BCE", "CE"), ("CE+soft kappa", "CE")]
pairs = []
K = len(CONTR)
for mode, key in (("t0", "kappa(0)"), ("tstar", "kappa(t*)")):
    pk = {o: [kb(cells[(o, s)], mode) for s in (0, 1, 2)] for o in OBJ}
    mp = {o: float(np.mean([x[0] for x in pk[o]])) for o in OBJ}; mb = {o: np.mean([x[1] for x in pk[o]], 0) for o in OBJ}
    for A_, B_ in CONTR:
        sd = [pk[A_][i][0] - pk[B_][i][0] for i in range(3)]; bs = mb[A_] - mb[B_]
        pairs.append(dict(threshold=key, contrast=f"{A_} - {B_}", diff=mp[A_] - mp[B_], seed_diffs=sd, n_positive_seeds=int(sum(x > 0 for x in sd)), seed_sd=float(np.std(sd, ddof=1)),
                          ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], ci_bonferroni_K5=[float(np.percentile(bs, 100 * 0.05 / K / 2)), float(np.percentile(bs, 100 - 100 * 0.05 / K / 2))]))

# run audit
def L(p): return json.load(open(p))
audit = []
for obj, d, tg in (("CE+2MAE", "esci384", "Q384M_mixmae_T1_s{s}"), ("CE+BCE", "esci384", "Q384B_mixbce_T1_s{s}")):
    for s in ((0, 1, 2) if obj == "CE+2MAE" else (1, 2)):
        t = tg.format(s=s); st = L(R / d / "train/logs" / f"stats_{t}.json"); ref = L(R / "esci_mix/train/logs" / f"stats_EQ_mix_T1_s{s}.json")
        steps = [json.loads(l) for l in open(R / d / "train/logs" / f"steps_{t}.jsonl", encoding="utf-8")]
        ce_ref = [json.loads(l) for l in open(R / "esci_t/train/logs" / f"steps_E4_T1_s{s}.jsonl", encoding="utf-8")][:2]
        term = "mae_loss" if obj == "CE+2MAE" else "bce_loss"
        lam = st["lam"]
        comp = max(abs(r["loss"] - (r["ce"] + lam * r[term])) for r in steps)
        a0 = torch.load(R / d / "train/checkpoints" / t / "ckpt_step0.pt", map_location="cpu", weights_only=False)["adapter_state"]
        b0 = torch.load(R / "esci_mix/train/checkpoints" / f"EQ_mix_T1_s{s}" / "ckpt_step0.pt", map_location="cpu", weights_only=False)["adapter_state"]
        audit.append(dict(tag=t, objective=obj, loss=st["loss"], coefficient=lam, updates=len(steps), optimizer_updates=st["optimizer_updates"], skipped=st["skipped_updates"], nonfinite=st["nonfinite"], inputs_shortened=st["inputs_shortened"], max_length=st["max_length"], micro_batch=st["micro_batch"],
                          rows_hash_equal_stored=st["rows_file_sha256"] == ref["rows_file_sha256"], order_hash_equal_stored_soft_run=st["order_sha256"] == ref["order_sha256"], step0_adapter_equal_stored_soft_run=set(a0) == set(b0) and all(torch.equal(a0[k], b0[k]) for k in a0),
                          update1_ce_equal_stored_CE=steps[0]["ce"] == ce_ref[0]["ce"], update2_ce_abs_diff_vs_stored_CE=abs(steps[1]["ce"] - ce_ref[1]["ce"]), max_abs_loss_composition_error=comp, sequences_forwarded=st["sequences_forwarded"], train_seconds=st["train_seconds"], peak_gb=st["peak_memory_gb"],
                          mean_r=float(np.mean([r["r"] for r in steps])), mean_D=float(np.mean([r["D"] for r in steps]))))
ok = all(x["optimizer_updates"] == 625 and x["skipped"] == 0 and x["nonfinite"] == 0 and x["inputs_shortened"] == 0 and x["rows_hash_equal_stored"] and x["order_hash_equal_stored_soft_run"] and x["step0_adapter_equal_stored_soft_run"] and x["max_abs_loss_composition_error"] < 1e-6 for x in audit)
res = dict(summary=summ, per_seed=rows, pairs=pairs, audit=audit, audit_passed=bool(ok), note="paired test-query bootstrap, 10,000 draws, default_rng(384); conditional on the fixed trained models and validation-selected thresholds; exploratory; Bonferroni K = 5 contrasts")
(OUT / "bce_mae_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
Lm = ["| objective | val k(0) | val k(t*) | test k(0) | test k(t*) | AUC | per-seed test k(0) | per-seed test k(t*) |", "|---|---|---|---|---|---|---|---|"]
for d in summ:
    o = d["objective"]; f = lambda k: f"{d[k + '_mean']:.4f} ± {d[k + '_sd']:.4f}"
    Lm.append(f"| {o} | {f('val_kappa_t0')} | {f('val_kappa_tstar')} | {f('test_kappa_t0')} | {f('test_kappa_tstar')} | {f('test_auc')} | " + ", ".join(f"{cells[(o, s)]['test_kappa_t0']:.4f}" for s in (0, 1, 2)) + " | " + ", ".join(f"{cells[(o, s)]['test_kappa_tstar']:.4f}" for s in (0, 1, 2)) + " |")
Lm += ["", "| threshold | contrast | diff | seed diffs | positive seeds | 95% CI | Bonferroni K=5 CI |", "|---|---|---|---|---|---|---|"]
for p in pairs:
    Lm.append(f"| {p['threshold']} | {p['contrast']} | {p['diff']:+.4f} | {', '.join(f'{x:+.4f}' for x in p['seed_diffs'])} | {p['n_positive_seeds']}/3 | [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] | [{p['ci_bonferroni_K5'][0]:+.4f}, {p['ci_bonferroni_K5'][1]:+.4f}] |")
Lm += ["", f"audit passed: {ok}"] + [f"- {x['tag']}: updates {x['optimizer_updates']}, skipped {x['skipped']}, nonfinite {x['nonfinite']}, shortened {x['inputs_shortened']}, rows hash {x['rows_hash_equal_stored']}, order hash {x['order_hash_equal_stored_soft_run']}, step-0 adapter {x['step0_adapter_equal_stored_soft_run']}, update-1 CE equal {x['update1_ce_equal_stored_CE']}, update-2 CE diff {x['update2_ce_abs_diff_vs_stored_CE']:.2e}, loss composition error {x['max_abs_loss_composition_error']:.1e}, mean r {x['mean_r']:.3f}, mean D {x['mean_D']:.3f}, {x['train_seconds']:.0f} s" for x in audit]
(OUT / "bce_mae_results.md").write_text("\n".join(Lm) + "\n", encoding="utf-8")
print("\n".join(Lm))
