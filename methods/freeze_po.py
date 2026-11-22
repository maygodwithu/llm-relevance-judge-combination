"""CPU: freezes the MS MARCO PO selection. Reads the 14 stage-1 results (qwen / llama x T1-T7), writes
  freeze/po_selection_manifest.json   per (model, instruction): candidate table (variant, body tokens, validity reasons, truncation counts, selection kappa), selected slot, changed flag, SHA-256 of original and selected full text
  freeze/custom_instructions_po_<model>.json   {"<Tk>_APOCF": full text} for the CHANGED instructions only (the file the trainers and evaluators read with --custom_instruction_file)
  freeze/po_selection_manifest.sha256
Never overwritten. Written BEFORE any TREC-DL score of a PO instruction exists.
    python freeze_po.py --root . --out cpmr_experiment/msm_po/freeze"""
import argparse
import hashlib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = Path(a.root)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
man = {"rule": "APO-CF 1 round (Base = judge = proposer; shorter / similar / bare + original; selection kappa on the 400 frozen selection pairs; original wins ties; no additional input truncation); TREC-DL not used", "models": {}}
custom = {"qwen": {}, "llama": {}}
for m in ("qwen", "llama"):
    man["models"][m] = {}
    for k in range(1, 8):
        p = R / f"cpmr_experiment/msm_po/{m}/p1_T{k}/p1_result.json"
        if not p.exists():
            raise SystemExit(f"missing stage-1 result: {p}")
        r = json.load(open(p, encoding="utf-8"))
        assert r["original_sha256"] == sha(r["original_full"]) and r["selected_sha256"] == sha(r["selected_full"])
        man["models"][m][f"T{k}"] = {"result_file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "original_sha256": r["original_sha256"], "selected_sha256": r["selected_sha256"], "selected_index": r["selection"]["index"],
                                     "selected_variant": r["selection"]["variant"], "selection_kappa": r["selection"]["kappa"], "changed": r["changed"], "n_valid": r["n_valid"],
                                     "candidates": [{kk: c.get(kk) for kk in ("index", "variant", "body_tokens", "valid", "violations", "shortened_diag_sel", "shortened_train", "kappa_selection", "rel_pct", "body_sha256")} for c in r["candidates"]],
                                     "diag_pairs_sha256": r["data"]["diag_pairs_sha256"], "sel_pairs_sha256": r["data"]["sel_pairs_sha256"]}
        if r["changed"]:
            custom[m][f"T{k}_APOCF"] = r["selected_full"]
for m in custom:
    cp = out / f"custom_instructions_po_{m}.json"
    assert not cp.exists(), f"{cp} exists"
    cp.write_text(json.dumps(custom[m], ensure_ascii=False, indent=1), encoding="utf-8")
mp = out / "po_selection_manifest.json"
assert not mp.exists(), f"{mp} exists"
mp.write_text(json.dumps(man, indent=1), encoding="utf-8")
(out / "po_selection_manifest.sha256").write_text(hashlib.sha256(mp.read_bytes()).hexdigest() + "  po_selection_manifest.json\n", encoding="utf-8")
for m in ("qwen", "llama"):
    print(m, "changed:", [k for k, v in man["models"][m].items() if v["changed"]], "kept original:", [k for k, v in man["models"][m].items() if not v["changed"]])
