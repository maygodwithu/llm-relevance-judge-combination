"""Aggregation, contrasts and tables of the chapter 6.1 AUC re-analysis (experiment log 229). Input: boot_<set>.pkl (auc61_boot.py), kappa.pkl (auc61_kappa.py), and the reported values of chapter5_results.json for verification.
Aggregation order (as in the 5th chapter): AUC (or kappa) per (instruction, seed) over all evaluated pairs -> equal-weight mean over the instructions of the scope -> mean over seeds. Base has no training seed: its instruction-mean is common to every seed (not an independent repetition).
Each bootstrap draw repeats this aggregation with the resample's AUC values; contrasts are differences of the aggregated per-draw values. Percentile CIs: 95% and the family-adjusted CI (ESCI per model: 3 contrasts -> 98.33%; MS MARCO per model: 3 contrasts x 2 years = 6 -> 99.1667%; PO T1-T3 per model: 3 contrasts -> 98.33%). Exploratory T1-T7 PO auxiliary CIs are 95% only.
    python auc61_report.py --pk <dir with boot_*.pkl and kappa.pkl> --ch5 <chapter5_results.json> --out <dir>"""
import argparse
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)
MODELS = {"Qwen": ("ESCI-Qwen", "Qwen3-4B", 206, 213), "Llama": ("ESCI-Llama", "Llama-3.1-8B", 210, 216), "Gemma": ("ESCI-Gemma", "Gemma-3-4B", 223, 228)}
CODE = {"Qwen": {"BASE": "Q0", "CE": "Q1", "SOFT": "Q2", "ANY": "Q3", "POBASE": "P0", "POCE": "P1", "POSOFT": "P2"},
        "Llama": {"BASE": "L0", "CE": "L1", "SOFT": "L2", "ANY": "L4", "POBASE": "L0P", "POCE": "L3", "POSOFT": "L5"},
        "Gemma": {"BASE": "G0", "CE": "G1", "SOFT": "G2", "ANY": "G3"}}
MSCODE = {"BASE": "M0", "CE": "M1", "SOFT": "M2", "ANY": "M3"}
LAB = {"BASE": "Base", "CE": "CE", "SOFT": "CE+soft κ", "ANY": "CE+AnyLoss-like κ", "POBASE": "PO Base", "POCE": "PO→CE", "POSOFT": "PO→CE+soft κ"}
CONTRASTS = [("CE", "BASE", "CE − Base"), ("SOFT", "CE", "(CE+soft κ) − CE"), ("ANY", "SOFT", "(CE+AnyLoss-like κ) − (CE+soft κ)")]


def seeds_of(c):
    return (0,) if c in ("BASE", "POBASE") else SEEDS


def pct(v, lv):
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, (100 - lv) / 2)), float(np.percentile(v, 100 - (100 - lv) / 2))]


class Ev:
    """one evaluation set: full-sample and bootstrap AUC per (cond, T, seed)"""
    def __init__(self, pk):
        self.d = pickle.load(open(pk, "rb"))
        self.full, self.boot = self.d["full"], self.d["boot"]

    def has(self, c, Ts):
        return all((c, T, s) in self.full for T in Ts for s in seeds_of(c))

    def cell(self, c, T, s, boot=False):
        return (self.boot if boot else self.full)[(c, T, 0 if c in ("BASE", "POBASE") else s)]

    def agg(self, c, Ts, boot=False):
        """-> point (or draw vector), list of per-seed values, list of per-instruction seed-mean values"""
        per_seed = [np.mean([self.cell(c, T, s, boot) for T in Ts], axis=0) for s in seeds_of(c)]
        per_T = [np.mean([self.cell(c, T, s, boot) for s in seeds_of(c)], axis=0) for T in Ts]
        return np.mean(per_seed, axis=0), per_seed, per_T


def kagg(K, sid, c, Ts, mode):
    i = 0 if mode == "default" else 1
    per_seed = [np.mean([K[(sid, c, T, 0 if c == "BASE" else s)][i] for T in Ts]) for s in seeds_of(c)]
    per_T = [np.mean([K[(sid, c, T, 0 if c == "BASE" else s)][i] for s in seeds_of(c)]) for T in Ts]
    return float(np.mean(per_seed)), per_seed, per_T


def contrast(ev, K, sid, hi, lo, Ts, adj):
    ph, sh, th = ev.agg(hi, Ts)
    pl, sl, tl = ev.agg(lo, Ts)
    bh, _, _ = ev.agg(hi, Ts, boot=True)
    bl, _, _ = ev.agg(lo, Ts, boot=True)
    vec = bh - bl
    nS = max(len(sh), len(sl))
    sd = [sh[i if len(sh) > 1 else 0] - sl[i if len(sl) > 1 else 0] for i in range(nS)]
    pT = [float(a - b) for a, b in zip(th, tl)]
    rec = dict(hi=hi, lo=lo, scope=f"T{Ts[0]}-T{Ts[-1]}" if len(Ts) > 1 and list(Ts) == list(range(Ts[0], Ts[-1] + 1)) else "T" + ",".join(map(str, Ts)), instructions=list(Ts),
               diff=float(ph - pl), ci95=pct(vec, 95), seed_diffs=[float(x) for x in sd], per_instruction=dict(zip([f"T{T}" for T in Ts], pT)), n_positive_instructions=int(sum(x > 0 for x in pT)), n_instructions=len(Ts),
               draws_used=int(np.isfinite(vec).sum()))
    if adj:
        rec["ci_adjusted"] = pct(vec, adj[0]); rec["adjusted_level"] = adj[0]; rec["family_size"] = adj[1]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pk", required=True); ap.add_argument("--ch5", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pk = Path(a.pk)
    SETS = ["ESCI-Qwen", "ESCI-Llama", "ESCI-Gemma", "MS-Qwen-2019", "MS-Qwen-2020", "MS-Llama-2019", "MS-Llama-2020", "MS-Gemma-2019", "MS-Gemma-2020"]
    EV = {s: Ev(pk / f"boot_{s}.pkl") for s in SETS}
    K = pickle.load(open(pk / "kappa.pkl", "rb"))
    ch5 = json.load(open(a.ch5, encoding="utf-8"))
    R = dict(meta=dict(), checks=[], panelA=[], contrasts=[], po=[], link=[], aucs=[])
    # ------------------------------------------------------------- scopes
    ALL = list(range(1, 8))
    def scopes(sid):
        if sid == "ESCI-Qwen":
            return [("main", ALL), ("aux", list(range(2, 8)))]
        if sid == "ESCI-Llama":
            return [("main", ALL), ("aux", list(range(4, 8)))]
        if sid == "ESCI-Gemma":
            return [("main", ALL)]
        return [("main", ALL), ("aux", list(range(2, 8)))]
    # ------------------------------------------------------------- verification against the reported values
    rep = {}
    for x in ch5["aggregates"]:
        rep[(x["source_id"], x["year"], x["scope"], x["code"])] = x
    def sc_name(Ts):
        return f"T{Ts[0]}-T{Ts[-1]}" if len(Ts) > 1 else f"T{Ts[0]}"
    auc_max = k_max = 0.0
    n_auc = n_k = 0
    bad = []
    for sid in SETS:
        ds, mdl = sid.split("-")[0], sid.split("-")[1]
        year = sid.split("-")[2] if ds == "MS" else None
        src = MODELS[mdl][2] if ds == "ESCI" else MODELS[mdl][3]
        codes = CODE[mdl] if ds == "ESCI" else MSCODE
        for scope in ({"T1-T3": [1, 2, 3], "T1-T7": ALL, "T2-T7": list(range(2, 8)), "T4-T7": list(range(4, 8)), "T1": [1]}).items():
            for cond, code in codes.items():
                x = rep.get((src, year, scope[0], code))
                if x is None or not EV[sid].has(cond, scope[1]):
                    continue
                mine = float(EV[sid].agg(cond, scope[1])[0])
                d = abs(mine - x["auc"]); auc_max = max(auc_max, d); n_auc += 1
                if d > 5.0001e-5:
                    bad.append(("AUC", sid, scope[0], code, mine, x["auc"]))
                if cond in ("BASE", "CE", "SOFT", "ANY"):
                    for mode, key in (("default", "default"), ("selected", "selected")):
                        km = kagg(K, sid, cond, scope[1], mode)[0]
                        dk = abs(km - x[key]); k_max = max(k_max, dk); n_k += 1
                        if dk > 5.0001e-5:
                            bad.append((mode + " kappa", sid, scope[0], code, km, x[key]))
    R["checks"].append(dict(name="reported four-digit AUC (chapter5_results.json aggregates) reproduced from the raw scores", n=n_auc, max_abs_diff=auc_max, tolerance=5e-5, passed=len([b for b in bad if b[0] == "AUC"]) == 0))
    R["checks"].append(dict(name="reported four-digit default / selected kappa reproduced from the raw scores and the re-derived thresholds", n=n_k, max_abs_diff=k_max, tolerance=5e-5, passed=len([b for b in bad if b[0] != "AUC"]) == 0))
    R["checks"].append(dict(name="mismatches (set, scope, code, mine, reported)", items=[list(map(str, b)) for b in bad]))
    R["checks"].append(dict(name="rank-based AUC == weighted AUC with unit weights (max abs diff over all vectors)", value=max(EV[s].d["max_abs_rank_vs_weighted_full_sample"] for s in SETS)))
    R["checks"].append(dict(name="bootstrap: draws excluded because only one class remained (per set)", value={s: EV[s].d["n_invalid_draws"] for s in SETS}))
    R["meta"] = dict(B=EV[SETS[0]].d["B"], rng="np.random.default_rng(61).integers(0, n_queries, size=(10000, n_queries)) per evaluation set", sets={s: dict(n_queries=EV[s].d["n_queries"], n_rows=EV[s].d["n_rows"], n_pos=EV[s].d["n_pos"]) for s in SETS},
                     score_vectors={s: len(EV[s].full) for s in SETS}, ties_in_any_score_vector={s: int(sum(1 for v in EV[s].d["ties_in_scores"].values() if v > 0)) for s in SETS})
    # ------------------------------------------------------------- Panel A
    for sid in SETS:
        ev = EV[sid]
        row = dict(set=sid, scope="T1-T7")
        for c in ("BASE", "CE", "SOFT", "ANY"):
            p, ps, pT = ev.agg(c, ALL)
            b, _, _ = ev.agg(c, ALL, boot=True)
            row[c] = dict(auc=float(p), seed_values=[float(x) for x in ps], seed_sd=(float(np.std(ps, ddof=1)) if len(ps) > 1 else None), per_instruction=dict(zip([f"T{T}" for T in ALL], map(float, pT))), ci95=pct(b, 95))
        R["panelA"].append(row)
    # ------------------------------------------------------------- contrasts (original text)
    for sid in SETS:
        ds = sid.split("-")[0]
        mdl = sid.split("-")[1]
        year = sid.split("-")[2] if ds == "MS" else None
        adj = (98.33, 3) if ds == "ESCI" else (99.1667, 6)
        for kind, Ts in scopes(sid):
            for hi, lo, name in CONTRASTS:
                rec = contrast(EV[sid], K, sid, hi, lo, Ts, adj)
                rec.update(set=sid, model=mdl, dataset=ds, year=year, kind=kind, comparison=name,
                           delta_kappa_default=float(kagg(K, sid, hi, Ts, "default")[0] - kagg(K, sid, lo, Ts, "default")[0]),
                           delta_kappa_selected=float(kagg(K, sid, hi, Ts, "selected")[0] - kagg(K, sid, lo, Ts, "selected")[0]))
                kdd = [a_ - b_ for a_, b_ in zip(kagg(K, sid, hi, Ts, "default")[2], kagg(K, sid, lo, Ts, "default")[2])]
                kds = [a_ - b_ for a_, b_ in zip(kagg(K, sid, hi, Ts, "selected")[2], kagg(K, sid, lo, Ts, "selected")[2])]
                rec["n_positive_instructions_kappa_default"] = int(sum(x > 0 for x in kdd)); rec["n_positive_instructions_kappa_selected"] = int(sum(x > 0 for x in kds))
                R["contrasts"].append(rec)
    # verification of delta-kappa against the full-precision reported contrasts
    ccheck = []
    for c in ch5["contrasts"]:
        txt = c["comparison"]
        codes = re.findall(r"\b([QPLGM]\d|L0P)\b", txt)
        if len(codes) < 2 or txt.startswith("calibration") or txt.startswith("(aux"):
            continue
        mode = "default" if txt.startswith("aux t0") else ("selected" if ("cal" in txt or "cross" in txt) else None)
        if mode is None:
            continue
        ccheck.append(((c["source_id"], c.get("year"), c["set"].split()[0], codes[0], codes[1], mode), c["diff"]))
    inv = {}
    for mdl, (sid_e, nm, se, sm) in MODELS.items():
        for c, code in CODE[mdl].items():
            inv[(se, code)] = c
        for c, code in MSCODE.items():
            inv[(sm, code)] = c
    n_ck = 0; max_ck = 0.0; ck_bad = []
    for (src, yr, scope, hi, lo, mode), diff in ccheck:
        ch_hi = inv.get((src, hi)); ch_lo = inv.get((src, lo))
        if ch_hi is None or ch_lo is None:
            continue
        mdl = [m for m, v in MODELS.items() if src in (v[2], v[3])][0]
        sid = MODELS[mdl][0] if src == MODELS[mdl][2] else f"MS-{mdl}-{yr}" if yr else None
        if sid is None or sid not in EV:
            continue
        if scope.startswith("T1-T7"):
            Ts = ALL
        elif scope.startswith("T2-T7"):
            Ts = list(range(2, 8))
        elif scope.startswith("T4-T7"):
            Ts = list(range(4, 8))
        elif scope.startswith("T1-T3"):
            Ts = [1, 2, 3]
        else:
            continue
        if not (EV[sid].has(ch_hi, Ts) and EV[sid].has(ch_lo, Ts)) or ch_hi.startswith("PO") or ch_lo.startswith("PO"):
            continue
        mine = kagg(K, sid, ch_hi, Ts, mode)[0] - kagg(K, sid, ch_lo, Ts, mode)[0]
        n_ck += 1; max_ck = max(max_ck, abs(mine - diff))
        if abs(mine - diff) > 1e-9:
            ck_bad.append((src, yr, scope, hi, lo, mode, mine, diff))
    R["checks"].append(dict(name="delta-kappa (default / selected) recomputed from raw scores vs the full-precision contrasts of the original reports", n=n_ck, max_abs_diff=max_ck, tolerance=1e-9, passed=len(ck_bad) == 0, mismatches=[list(map(str, b)) for b in ck_bad[:20]]))
    # ------------------------------------------------------------- PO
    for mdl in ("Qwen", "Llama"):
        sid = MODELS[mdl][0]
        ev = EV[sid]
        for kind, Ts, conds, adj in (("main T1-T3", [1, 2, 3], [("POBASE", "BASE"), ("POCE", "CE"), ("POSOFT", "SOFT")], (98.33, 3)), ("aux T1-T7 (95% exploratory)", ALL, [("POBASE", "BASE"), ("POCE", "CE")], None)):
            for hi, lo in conds:
                if not (ev.has(hi, Ts) and ev.has(lo, Ts)):
                    R["po"].append(dict(set=sid, model=mdl, kind=kind, hi=hi, lo=lo, available=False))
                    continue
                rec = contrast(ev, K, sid, hi, lo, Ts, adj)
                rec.update(set=sid, model=mdl, kind=kind, comparison=f"{LAB[hi]} − {LAB[lo]}", available=True, auc_original=float(ev.agg(lo, Ts)[0]), auc_po=float(ev.agg(hi, Ts)[0]),
                           seed_values_original=[float(x) for x in ev.agg(lo, Ts)[1]], seed_values_po=[float(x) for x in ev.agg(hi, Ts)[1]])
                if mdl == "Llama":
                    rec["identical_to_original_instructions"] = [f"T{T}" for T in Ts if T in (1, 4, 6)]
                R["po"].append(rec)
    import csv
    with open(out / "score_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["evaluation_set", "rows_sha256_of_(query,pair_id,gold)", "n_queries", "n_rows", "condition", "instruction_index", "train_seed", "model_tag", "instruction_label", "instruction_sha256", "rows_file_sha256", "adapter_checkpoint_sha256", "global_step", "max_length", "score_file_relative_path", "score_file_sha256"])
        for sid in SETS:
            d = EV[sid].d
            for r in d["manifest"]:
                w.writerow([sid, d["rows_sha256"], d["n_queries"], d["n_rows"], r["cond"], r["T"], r["seed"], r["model"], r["instruction"], r["instruction_sha256"], r["rows_file_sha256"], r["checkpoint_sha256"], r["global_step"], r["max_length"], r["path"], r["sha256"]])
    R["meta"]["rows_sha256"] = {s: EV[s].d["rows_sha256"] for s in SETS}
    R["meta"]["seconds_report"] = time.time() - t0
    json.dump(R, open(out / "auc61_results.json", "w"), indent=1, default=float)
    print("report written; checks:", [(c["name"][:60], c.get("passed"), c.get("max_abs_diff")) for c in R["checks"] if "passed" in c])
    return R


if __name__ == "__main__":
    main()
