"""Markdown rendering of esci_po_results.json (ESCI PO T1-T7 analysis, messages 268/269). Formatting only.
    python esci_po_md.py esci_po_results.json audit268.json out.md"""
import json
import sys

res = json.load(open(sys.argv[1]))
aud = json.load(open(sys.argv[2]))
F = res["fam"]
L = []
P = L.append
MODELS = (("qwen", "Qwen3-4B"), ("llama", "Llama-3.1-8B-Instruct"))
CN = {"BASE": "Base", "CE": "CE", "SOFT": "CE+soft kappa"}
f4 = lambda x: f"{x:.4f}"
sg = lambda x: f"{x:+.4f}"
ci = lambda c: f"[{c[0]:+.4f}, {c[1]:+.4f}]"

P("# ESCI PO T1-T7 completion: Original vs PO-selected instruction x Base / CE / CE+soft kappa (Qwen3-4B, Llama-3.1-8B-Instruct)")
P("")
P("ESCI test300 (300 queries / 6,078 rows; E = Relevant, S / C / I = Irrelevant). kappa at t=0 and at the validation-selected threshold t* of each checkpoint (200-query / 4,047-row validation, hard-kappa maximiser, ties -> 0, then smaller |t|, then smaller t; Base threshold never transferred). "
  "Aggregation: per-instruction kappa -> equal-weight mean over the instruction set -> mean over 3 seeds (Base seed 0). kappa contrasts: paired test-query bootstrap, 2,000 draws, default_rng(0), one draw set for every cell, exploratory 95% (K6 = 99.17% for the objective contrasts, Bonferroni 6 of the earlier ESCI family); "
  "AUC / error profile / transitions: 10,000 draws, default_rng(61); validation stability: 500 draws of the 200 validation queries (default_rng(1) = reuse_rng1 and default_rng(63)). Llama: PO kept the original text for T1, T4, T6 (aliases of the Original, difference exactly 0).")
P("")
P("## 0. Checks")
P(f"- Run audit (pre-registered): {'PASS' if aud['passed'] else 'FAIL'}; 18 new PO CE+soft kappa runs, 36 evaluation files; first-update CE equal to the stored CE-only PO run in {sum(r['equal'] for r in aud['rows'])}/18 runs (exact), step-0 adapter tensors equal in {sum(bool(r['step0_adapter_equal']) for r in aud['rows'])}/18; order and rows hashes, global_step 625, nonfinite 0, input shortening 0, instruction sha256 = stored PO CE evaluations; failures {len(aud['failures'])}.")
for c in res["checks"]:
    P(f"- {c['model']}: {c['check']}: " + ", ".join(f"{k}={v}" for k, v in c.items() if k not in ("model", "check")))
P("")
P("## 1. Main table: Original vs PO, t=0 and validation t* (T1-T7; auxiliary scopes below, never mixed)")
for m, mn in MODELS:
    for sc in F[m]["main"]:
        P(f"\n### {mn}, scope {sc}")
        P("| objective | Orig t=0 | PO t=0 | PO - Orig t=0 [95%] | seed diffs t=0 | Orig t* | PO t* | PO - Orig t* [95%] | seed diffs t* | #instr. diff>0 (t=0 / t*) |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        for r in F[m]["main"][sc]:
            a, b = r["t0"], r["sel"]
            P(f"| {CN[r['cond']]} | {f4(a['orig'])} | {f4(a['po'])} | {sg(a['diff'])} {ci(a['ci95'])} | {', '.join(sg(x) for x in a['seed_diffs'])} | {f4(b['orig'])} | {f4(b['po'])} | {sg(b['diff'])} {ci(b['ci95'])} | {', '.join(sg(x) for x in b['seed_diffs'])} | {sum(x > 1e-12 for x in a['instr_diffs'])} / {sum(x > 1e-12 for x in b['instr_diffs'])} of {len(a['instr_diffs'])} |")
P("\n## 2. Objective contrasts under Original and PO (T1-T7; K6 interval = 99.17%)")
P("| model | threshold | Orig Base / CE / soft | PO Base / CE / soft | Orig CE-Base | PO CE-Base | Orig soft-CE [95%] (K6) | PO soft-CE [95%] (K6) | (soft-CE)PO - (soft-CE)Orig [95%] |")
P("|---|---|---|---|---|---|---|---|---|")
for m, mn in MODELS:
    for o in F[m]["objective"]:
        a, b = o["ORIG"], o["PO"]
        P(f"| {mn} | {'t=0' if o['mode'] == 't0' else 't*'} | {f4(a['BASE'])} / {f4(a['CE'])} / {f4(a['SOFT'])} | {f4(b['BASE'])} / {f4(b['CE'])} / {f4(b['SOFT'])} | {sg(a['CE-BASE']['diff'])} | {sg(b['CE-BASE']['diff'])} | {sg(a['SOFT-CE']['diff'])} {ci(a['SOFT-CE']['ci95'])} {ci(a['SOFT-CE']['ci_k6'])} | {sg(b['SOFT-CE']['diff'])} {ci(b['SOFT-CE']['ci95'])} {ci(b['SOFT-CE']['ci_k6'])} | {sg(o['SOFTCE_PO_minus_ORIG']['diff'])} {ci(o['SOFTCE_PO_minus_ORIG']['ci95'])} |")
P("\n## 3. AUC and error profile, Original -> PO (T1-T7; paired 10,000 draws; AUC is threshold-free and shown with the t=0 rows)")
P("| model | objective | threshold | AUC | predicted Relevant | FNR of E | FPR S | FPR C | FPR I |")
P("|---|---|---|---|---|---|---|---|---|")
for m, mn in MODELS:
    for p in F[m]["profile"]:
        cell = lambda k: (f"{f4(p[k]['orig'])} -> {f4(p[k]['po'])}, {sg(p[k]['diff'])} {ci(p[k]['ci95'])}" if k in p else "")
        P(f"| {mn} | {CN[p['cond']]} | {'t=0' if p['mode'] == 't0' else 't*'} | {cell('auc')} | {cell('pos')} | {cell('fnr')} | {cell('S')} | {cell('C')} | {cell('I')} |")
P("\n## 4. Original -> PO decision transitions (T1-T7; rate = share of the gold group; W->R corrected, R->W new error, net = W->R - R->W)")
P("| model | objective | threshold | group | W->R | R->W | net [95%] | changed |")
P("|---|---|---|---|---|---|---|---|")
for m, mn in MODELS:
    for t in F[m]["transitions"]:
        for g, gn in (("E", "E (Relevant, FN side)"), ("S", "S"), ("C", "C"), ("I", "I"), ("SCI", "S+C+I"), ("ALL", "all")):
            P(f"| {mn} | {CN[t['cond']]} | {'t=0' if t['mode'] == 't0' else 't*'} | {gn} | {t[g]['WR']['rate']:.4f} | {t[g]['RW']['rate']:.4f} | {sg(t[g]['net']['rate'])} {ci(t[g]['net']['ci95'])} | {t[g]['changed']['rate']:.4f} |")
for spec, sn in (("reuse_rng1", "default_rng(1) (the draws of the existing ESCI stability analyses)"), ("new_rng63", "default_rng(63)")):
    P(f"\n## 5. Validation-query stability, 500 draws of the 200 validation queries, {sn} (thresholds re-selected per draw; same draws for every cell; T1-T7 seed means)")
    P("| model | variant | objective | kappa(t=0) | kappa(t* full) | kappa(t_b): mean [2.5, 97.5] | kappa(t_b) - kappa(t*) | kappa(t_b) - kappa(t=0): mean, share > 0 |")
    P("|---|---|---|---|---|---|---|---|")
    for m, mn in MODELS:
        for s in F[m]["stability"][spec]:
            if "cond" in s:
                P(f"| {mn} | {s['variant']} | {CN[s['cond']]} | {f4(s['k0'])} | {f4(s['kfull'])} | {f4(s['kb']['mean'])} [{f4(s['kb']['p2_5'])}, {f4(s['kb']['p97_5'])}] | {sg(s['kb_minus_full']['mean'])} | {sg(s['kb_minus_t0']['mean'])}, {s['kb_minus_t0']['share_pos']:.2f} |")
    P("\n| model | paired contrast of kappa(t_b) | mean | [2.5, 97.5] | share > 0 |")
    P("|---|---|---|---|---|")
    for m, mn in MODELS:
        for s in F[m]["stability"][spec]:
            if "contrast" in s:
                d = s["kb_diff"]
                P(f"| {mn} | {s['contrast']} | {sg(d['mean'])} | [{sg(d['p2_5'])}, {sg(d['p97_5'])}] | {d['share_pos']:.2f} |")
P("\n## 6. Per-instruction kappa (seed mean): t=0 / t*")
for m, mn in MODELS:
    P(f"\n### {mn} (PO-changed instructions: {F[m]['changed']})")
    P("| variant | objective | " + " | ".join(f"T{t}" for t in range(1, 8)) + " |")
    P("|---|---|" + "---|" * 7)
    for var in ("ORIG", "PO"):
        for cond in ("BASE", "CE", "SOFT"):
            rows = sorted([x for x in F[m]["per_instruction"] if x["variant"] == var and x["cond"] == cond], key=lambda x: x["T"])
            P(f"| {var} | {CN[cond]} | " + " | ".join(f"{x['t0']:.3f} / {x['sel']:.3f}" + (" (=Orig)" if x["alias"] else "") for x in rows) + " |")
open(sys.argv[3], "w", encoding="utf-8").write("\n".join(L) + "\n")
print("written", len(L))
