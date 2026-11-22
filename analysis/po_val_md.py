"""Markdown rendering of po_val_results.json (separate-validation threshold analysis). Formatting only.
    python po_val_md.py po_val_results.json out.md"""
import json
import sys

res = json.load(open(sys.argv[1]))
F = res["fam"]
L = []
P = L.append
MODELS = (("qwen", "Qwen3-4B"), ("llama", "Llama-3.1-8B-Instruct"), ("gemma", "Gemma-3-4B-it"))
CN = {"BASE": "Base", "CE": "CE", "SOFT": "CE+soft kappa", "ANY": "CE+AnyLoss-style kappa"}
f4 = lambda x: f"{x:.4f}"
sg = lambda x: f"{x:+.4f}"
ci = lambda c: f"[{c[0]:+.4f}, {c[1]:+.4f}]"

P("# MS MARCO separate-validation threshold (A0 = existing val200 selection pairs): t=0 vs validation t* applied to TREC-DL 2019 and 2020")
P("")
P("Threshold per checkpoint = hard-kappa maximiser over the 400 val200 pairs (ties -> 0, then smaller |t|, then smaller t; m > t), one threshold applied to both years; CROSS (the other year's judged pairs) is kept as a separate reference. Aggregation: per-instruction kappa -> equal-weight mean over T1-T7 -> mean over 3 seeds (Base seed 0). "
  "Bootstrap: 10,000 paired test-query-cluster resamples (default_rng(year)), thresholds fixed; K12 intervals = 99.5833% (3 contrasts x 2 thresholds x 2 years, Bonferroni 12); PO-Original intervals are exploratory 95%.")
P("")
P("## 0. Checks")
for c in res["checks"]:
    P(f"- {c['model']}: {c['check']}: " + ", ".join(f"{k}={v}" for k, v in c.items() if k not in ("model", "check")))
P("")
P("## 1. Levels: t=0 / validation t* / CROSS (reference) per objective (T1-T7, seed mean; mean validation threshold in brackets)")
for m, mn in MODELS:
    for y in ("2019", "2020"):
        P(f"\n### {mn}, TREC-DL {y}")
        P("| variant | objective | t=0 | validation t* | t* - t=0 | CROSS (reference) | mean t* (SD over cells) | AUC | pred. Relevant t=0 / t* | FNR t=0 / t* | FPR t=0 / t* |")
        P("|---|---|---|---|---|---|---|---|---|---|---|")
        for x in F[m]["levels"]:
            if x["year"] != y:
                continue
            P(f"| {x['variant']} | {CN[x['cond']]} | {f4(x['t0']['kappa'])} | {f4(x['sel']['kappa'])} | {sg(x['sel']['kappa'] - x['t0']['kappa'])} | {f4(x['cross']['kappa'])} | {x['thr_val_mean']:+.2f} ({x['thr_val_sd']:.2f}) | {f4(x['auc'])} | {x['pos']['t0']:.3f} / {x['pos']['sel']:.3f} | {x['fnr']['t0']:.3f} / {x['fnr']['sel']:.3f} | {x['fpr']['t0']:.3f} / {x['fpr']['sel']:.3f} |")
P("\n## 2. Objective contrasts, Original T1-T7 (diff, K12 99.58% interval, 95% interval, seed diffs, #instructions with a positive difference)")
P("| model | year | threshold | contrast | diff | K12 interval | 95% | seed diffs | #T>0 |")
P("|---|---|---|---|---|---|---|---|---|")
for m, mn in MODELS:
    for c in F[m]["contrasts"]:
        P(f"| {mn} | {c['year']} | {'t=0' if c['mode'] == 't0' else 'val t*'} | {c['contrast']} | {sg(c['diff'])} | {ci(c['ci_k12'])} | {ci(c['ci95'])} | {', '.join(sg(x) for x in c['seed_diffs'])} | {c['n_pos_instr']}/7 |")
P("\n## 3. PO - Original (T1-T7; exploratory 95%; Llama: five kept-original instructions are aliases, difference 0)")
P("| model | year | threshold | objective | PO - Original | 95% | seed diffs | share of draws > 0 |")
P("|---|---|---|---|---|---|---|---|")
for m, mn in MODELS[:2]:
    for c in F[m]["po_minus_orig"]:
        if "seed_diffs" in c:
            P(f"| {mn} | {c['year']} | {'t=0' if c['mode'] == 't0' else 'val t*'} | {CN[c['cond']]} | {sg(c['diff'])} | {ci(c['ci95'])} | {', '.join(sg(x) for x in c['seed_diffs'])} | {c['share_pos']:.2f} |")
P("\n### 3b. Objective gap after PO: (contrast under PO) - (contrast under Original)")
P("| model | year | threshold | contrast | PO | Original | difference [95%] |")
P("|---|---|---|---|---|---|---|")
for m, mn in MODELS[:2]:
    for c in F[m]["po_minus_orig"]:
        if "po" in c:
            P(f"| {mn} | {c['year']} | {'t=0' if c['mode'] == 't0' else 'val t*'} | {c['cond']} | {sg(c['po'])} | {sg(c['orig'])} | {sg(c['diff'])} {ci(c['ci95'])} |")
P("\n## 4. Validation-query resampling (500 draws of the 200 val queries; thresholds of every checkpoint re-selected and applied to the fixed test year; same draws for all cells and both years; T1-T7 seed means)")
P("| model | year | variant | objective | kappa(t=0) | kappa(t* full) | kappa(t_b): mean [2.5, 97.5] | kappa(t_b) - kappa(t*) mean | kappa(t_b) - kappa(t=0): mean, share > 0 |")
P("|---|---|---|---|---|---|---|---|---|")
for m, mn in MODELS:
    for s in F[m]["val_resample"]:
        if "cond" in s:
            P(f"| {mn} | {s['year']} | {s['variant']} | {CN[s['cond']]} | {f4(s['k0'])} | {f4(s['kfull'])} | {f4(s['kb']['mean'])} [{f4(s['kb']['p2_5'])}, {f4(s['kb']['p97_5'])}] | {sg(s['kb_minus_full']['mean'])} | {sg(s['kb_minus_t0']['mean'])}, {s['kb_minus_t0']['share_pos']:.2f} |")
P("\n### 4b. Paired contrasts of kappa(t_b) across the 500 validation resamples")
P("| model | year | contrast | mean | [2.5, 97.5] | share > 0 |")
P("|---|---|---|---|---|---|")
for m, mn in MODELS:
    for s in F[m]["val_resample"]:
        if "contrast" in s:
            d = s["kb_diff"]
            P(f"| {mn} | {s['year']} | {s['contrast']} | {sg(d['mean'])} | [{sg(d['p2_5'])}, {sg(d['p97_5'])}] | {d['share_pos']:.2f} |")
P("\n## 5. Per-instruction kappa (seed mean): t=0 / validation t* / CROSS reference")
for m, mn in MODELS:
    for y in ("2019", "2020"):
        P(f"\n### {mn}, TREC-DL {y}")
        P("| variant | objective | " + " | ".join(f"T{t}" for t in range(1, 8)) + " |")
        P("|---|---|" + "---|" * 7)
        for var in ("ORIG", "PO"):
            for cond in ("BASE", "CE", "SOFT", "ANY"):
                rows = [x for x in F[m]["per_instruction"] if x["variant"] == var and x["cond"] == cond and x["year"] == y]
                if rows:
                    rows.sort(key=lambda x: x["T"])
                    P(f"| {var} | {CN[cond]} | " + " | ".join(f"{x['t0']:.3f} / {x['sel']:.3f} / {x['cross']:.3f}" for x in rows) + " |")
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(L) + "\n")
print("written", len(L))
