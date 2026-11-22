"""Label-length normalised scores on the STORED ESCI T1 scores (CPU only; no training, no inference).
Stored record per row: [lp_label(R), lp_eot(R), lp_eos(R), lp_label(I), lp_eot(I), lp_eos(I), lp_first(R), lp_first(I), n_label_tokens(R), n_label_tokens(I), ...]
  ell1 = lp_label(R) + lp_eot(R),  ell0 = lp_label(I) + lp_eot(I)  (the submitted sum score; terminator kept),  L1 = n_label_tokens(R) + 1,  L0 = n_label_tokens(I) + 1
  m_sum = ell1 - ell0 (submitted),  m_norm = ell1 / L1 - ell0 / L0.   Nothing is back-computed from m: ell1, ell0 and the lengths are stored values.
Models x {Base, CE, CE+soft kappa (lambda 1)} x seeds 0-2 (Base once), ESCI T1, validation (sel, 200 queries) and test300 as stored. Thresholds: for each score type the hard-kappa maximising validation threshold with the submitted rule
(ties: 0, then smaller |t|, then smaller t); the sum threshold is NOT applied to the normalised score. No test-based choice. Reproduction check first: the sum results against the stored per-instruction T1 tables.
    python norm388.py --root <project root> --out <dir>"""
import argparse, csv, glob, gzip, hashlib, json, os, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--out", required=True); a = ap.parse_args()
R = Path(a.root) / "cpmr_experiment"; OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R / "analysis"))
import esci204_common as C  # noqa: E402

IDX = {}
for f in glob.glob(str(R / "**" / "esci*Eval_*.json.gz"), recursive=True):  # (names with set sel, sel200, test300)
    IDX.setdefault(os.path.basename(f), []).append(f)
FAM = {"Qwen3-4B": dict(pre="esciEval_", flt="qwen3-4b", ce="E4_T1_s{s}_step625", soft="EQ_mix_T1_s{s}_step625"),
       "Llama-3.1-8B": dict(pre="esciLlamaEval_", flt="llama-3.1-8b", ce="EL_T1_s{s}_step625", soft="ELQ_mix_T1_s{s}_step625"),
       "Gemma-3-4B": dict(pre="esciGemmaEval_", flt="gemma-3-4b", ce="GQ_ce_T1_s{s}_step625", soft="GQ_mix_T1_s{s}_step625")}
# stored per-instruction T1 values of the submitted appendix table (default / selected kappa, 3 decimals) used only as a reproduction check
APP = {("Qwen3-4B", "Base"): (0.353, 0.360), ("Qwen3-4B", "CE"): (0.357, 0.366), ("Qwen3-4B", "CE+soft kappa"): (0.391, 0.380),
       ("Llama-3.1-8B", "Base"): (0.302, 0.328), ("Llama-3.1-8B", "CE"): (0.358, 0.337), ("Llama-3.1-8B", "CE+soft kappa"): (0.413, 0.411),
       ("Gemma-3-4B", "Base"): (0.307, 0.302), ("Gemma-3-4B", "CE"): (0.301, 0.301), ("Gemma-3-4B", "CE+soft kappa"): (0.336, 0.336)}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def find(pre, flt, name, st):
    # the Qwen3-4B Base validation file is stored under the set name 'sel200' (same 200 validation queries, 4,047 rows; row set asserted below)
    names = [f"{pre}{name}__T1__{st}.json.gz"] + ([f"{pre}{name}__T1__sel200.json.gz"] if (name == "Base" and st == "sel") else [])
    c = [f for nm in names for f in IDX.get(nm, []) if flt in str(json.load(gzip.open(f, "rt", encoding="utf-8"))["meta"]["model_name"]).lower()]
    assert c, (name, st)
    ref = load(c[0])["msum"]
    for f in c[1:]:
        assert np.allclose(load(f)["msum"], ref, atol=1e-6, rtol=0), ("copies differ", name, st)
    return c[0]


def load(path):
    raw = json.load(gzip.open(path, "rt", encoding="utf-8"))["raw"]
    ks = [(q, d) for q in sorted(raw) for d in sorted(raw[q])]
    v = np.array([raw[q][d] for q, d in ks], dtype=float)
    e1, e0 = v[:, 0] + v[:, 1], v[:, 3] + v[:, 4]; L1, L0 = v[:, 8] + 1, v[:, 9] + 1
    return dict(ks=ks, e1=e1, e0=e0, L1=L1, L0=L0, msum=e1 - e0, mnorm=e1 / L1 - e0 / L0, y=v[:, 10] >= 2)


def kat(m, y, t):
    p = m > t
    return float(C.kap(int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum())))


def metrics(val, tst, key):
    thr, vk = C.best_threshold(val[key], val["y"])
    return {f"val_k0_{key}": kat(val[key], val["y"], 0.0), f"val_kstar_{key}": vk, f"tstar_{key}": thr, f"test_k0_{key}": kat(tst[key], tst["y"], 0.0), f"test_kstar_{key}": kat(tst[key], tst["y"], thr), f"test_auc_{key}": float(C.auc(tst[key], tst["y"])),
            f"rel_t0_{key}": float((tst[key] > 0).mean()), f"rel_tstar_{key}": float((tst[key] > thr).mean())}


rows, lens, REF = [], {}, {}
for model, f in FAM.items():
    runs = [("Base", 0, "Base")] + [(o, s, f[k].format(s=s)) for o, k in (("CE", "ce"), ("CE+soft kappa", "soft")) for s in (0, 1, 2)]
    for obj, s, name in runs:
        pv, pt = find(f["pre"], f["flt"], name, "sel"), find(f["pre"], f["flt"], name, "test300")
        val, tst = load(pv), load(pt)
        REF.setdefault((model, "sel"), val["ks"]); REF.setdefault((model, "test"), tst["ks"])
        assert val["ks"] == REF[(model, "sel")] and tst["ks"] == REF[(model, "test")], ("row set differs", name)
        lens[model] = sorted({(int(a), int(b)) for d in (val, tst) for a, b in zip(d["L1"], d["L0"])})
        r = dict(model=model, objective=obj, seed=s, tag=name, val_sha256=sha(pv), test_sha256=sha(pt), L1=int(tst["L1"][0]), L0=int(tst["L0"][0]),
                 decisions_differ_t0_share=float(((tst["msum"] > 0) != (tst["mnorm"] > 0)).mean()))
        r.update(metrics(val, tst, "msum")); r.update(metrics(val, tst, "mnorm"))
        rows.append(r)
# lengths constant over every stored row?
assert all(len(v) == 1 for v in lens.values()), lens
with open(OUT / "norm388_per_run.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
summ = []
for model in FAM:
    for obj in ("Base", "CE", "CE+soft kappa"):
        rs = [r for r in rows if r["model"] == model and r["objective"] == obj]
        d = dict(model=model, objective=obj, n=len(rs), L1=rs[0]["L1"], L0=rs[0]["L0"])
        for key in ("msum", "mnorm"):
            for k in ("val_k0", "val_kstar", "test_k0", "test_kstar", "test_auc", "rel_t0", "rel_tstar"):
                v = [r[f"{k}_{key}"] for r in rs]; d[f"{k}_{key}_mean"] = float(np.mean(v)); d[f"{k}_{key}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
        d["decisions_differ_t0_share_mean"] = float(np.mean([r["decisions_differ_t0_share"] for r in rs]))
        summ.append(d)
# reproduction of the stored sum results (appendix table, T1, seed means, 3 decimals)
repro = []
for d in summ:
    a0, a1 = APP[(d["model"], d["objective"])]
    repro.append(dict(model=d["model"], objective=d["objective"], k0_sum=round(d["test_k0_msum_mean"], 3), k0_table=a0, kstar_sum=round(d["test_kstar_msum_mean"], 3), kstar_table=a1, match=abs(round(d["test_k0_msum_mean"], 3) - a0) < 1.5e-3 and abs(round(d["test_kstar_msum_mean"], 3) - a1) < 1.5e-3))
# soft - CE
diffs = []
for model in FAM:
    for key in ("msum", "mnorm"):
        for thr in ("k0", "kstar"):
            ce = [r[f"test_{thr}_{key}"] for r in rows if r["model"] == model and r["objective"] == "CE"]; so = [r[f"test_{thr}_{key}"] for r in rows if r["model"] == model and r["objective"] == "CE+soft kappa"]
            sd = [x - y for x, y in zip(so, ce)]
            diffs.append(dict(model=model, score="sum" if key == "msum" else "normalised", threshold="t=0" if thr == "k0" else "validation t*", mean_diff=float(np.mean(sd)), seed_diffs=sd, sd=float(np.std(sd, ddof=1)), positive_seeds=int(sum(x > 0 for x in sd))))
res = dict(label_lengths_constant_over_all_stored_rows=True, lengths={m: dict(L1=v[0][0], L0=v[0][1]) for m, v in lens.items()}, summary=summ, soft_minus_ce=diffs, reproduction_vs_submitted_table=repro, reproduction_all_match=all(x["match"] for x in repro))
(OUT / "norm388_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
L = ["| model | L1 / L0 | objective | score | test k(0) | test k(t*) | AUC | Relevant % t=0 | Relevant % t* | val k(t*) |", "|---|---|---|---|---|---|---|---|---|---|"]
for d in summ:
    for key, nm in (("msum", "sum"), ("mnorm", "normalised")):
        f = lambda k: f"{d[f'{k}_{key}_mean']:.4f}" + (f" ± {d[f'{k}_{key}_sd']:.4f}" if d["n"] > 1 else "")
        L.append(f"| {d['model']} | {d['L1']} / {d['L0']} | {d['objective']} | {nm} | {f('test_k0')} | {f('test_kstar')} | {f('test_auc')} | {100 * d[f'rel_t0_{key}_mean']:.1f} | {100 * d[f'rel_tstar_{key}_mean']:.1f} | {f('val_kstar')} |")
L += ["", "| model | score | threshold | soft - CE mean | seed diffs | positive seeds |", "|---|---|---|---|---|---|"]
for x in diffs:
    L.append(f"| {x['model']} | {x['score']} | {x['threshold']} | {x['mean_diff']:+.4f} | {', '.join(f'{v:+.4f}' for v in x['seed_diffs'])} | {x['positive_seeds']}/3 |")
L += ["", "reproduction of the submitted T1 table (sum score): " + ("all 9 cells match" if res["reproduction_all_match"] else "MISMATCH")] + [f"- {x['model']} {x['objective']}: {x['k0_sum']} / {x['kstar_sum']} vs table {x['k0_table']} / {x['kstar_table']} -> {x['match']}" for x in repro]
(OUT / "norm388_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
