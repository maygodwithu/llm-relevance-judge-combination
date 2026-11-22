"""Which data PO instruction selection and threshold selection actually share, and whether an independent calibration pool exists. CPU only, read-only, no scores computed.
A  ESCI: roles of the frozen sets (diag / sel / sft / sft2 / test300), exact query / row / pair overlaps, file hashes against the hashes recorded in every PO result and training / evaluation record, the unused ('spare') normalised queries of the frozen permutation and their usage history
B  MS MARCO: roles from the frozen split manifests, hashes recorded in every PO result, provenance of val200, size of the unused query pool
C  PO inventory per (dataset, model, T): original text, candidates (variant, validity, selection kappa), selected candidate, changed flag, stored per-candidate raw margins on the selection set, SFT instruction hash of the PO runs
D  representative condition by the pre-registered rule 'smallest T whose PO text differs from the original'
    python audit392.py --root <project root> --out <dir>"""
import argparse
import os, glob, gzip, hashlib, json, os, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--out", required=True); a = ap.parse_args()
ROOT = Path(a.root); R = ROOT / "cpmr_experiment"; OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


res = dict(esci={}, msmarco={}, po={}, representative={})
# ------------------------------------------------------------------ A ESCI
AUD = R / "esci/audit"
spl = json.load(open(AUD / "splits.json"))
files = {"diag": "diag_rows.jsonl", "sel": "sel_rows.jsonl", "sft": "sft_rows.jsonl", "sft2": "sft2_rows.jsonl", "test300": "test300_rows.jsonl", "test1000": "test_rows.jsonl"}
S = {k: jl(AUD / f) for k, f in files.items()}
H = {k: sha(AUD / f) for k, f in files.items()}
role = {}
for k, rows in S.items():
    qs = {r["nq"] for r in rows}; pairs = {(r["nq"], r["product_id"]) for r in rows}; pids = {r["product_id"] for r in rows}
    role[k] = dict(file=files[k], rows=len(rows), queries=len(qs), pairs=len(pairs), products=len(pids), relevant_share=float(np.mean([r["esci"] == "E" for r in rows])), sha256=H[k])
res["esci"]["sets"] = role
ov = {}
keys = ["diag", "sel", "sft", "sft2", "test300"]
for i, x in enumerate(keys):
    for y in keys[i + 1:]:
        qx, qy = {r["nq"] for r in S[x]}, {r["nq"] for r in S[y]}
        px, py = {(r["nq"], r["product_id"]) for r in S[x]}, {(r["nq"], r["product_id"]) for r in S[y]}
        dx, dy = {r["product_id"] for r in S[x]}, {r["product_id"] for r in S[y]}
        ov[f"{x}|{y}"] = dict(shared_queries=len(qx & qy), shared_query_product_pairs=len(px & py), shared_product_ids=len(dx & dy))
res["esci"]["overlaps"] = ov
res["esci"]["frozen_lists"] = dict(diag=len(spl["diag"]), sel=len(spl["sel"]), sft=len(spl["sft"]), test=len(spl["test"]),
                                   lists_match_row_files=dict(diag=set(spl["diag"]) == {r["nq"] for r in S["diag"]}, sel=set(spl["sel"]) == {r["nq"] for r in S["sel"]}, sft=set(spl["sft"]) == {r["nq"] for r in S["sft"]}, sft2_subset_of_sft=({r["nq"] for r in S["sft2"]} <= set(spl["sft"]))))
# hashes recorded in the PO results and in the evaluation / training records
rec = {}
for fam, base in (("qwen", R / "esci_t"), ("llama", R / "esci_llama")):
    shas = {}
    for T in range(1, 8):
        p = base / f"p1_T{T}" / "p1_result.json"
        if p.exists():
            d = json.load(open(p)); shas[f"T{T}"] = d["file_sha256"]
    rec[fam] = shas
same = {}
for fam, shas in rec.items():
    for key in ("diag", "sel", "sft"):
        vals = {v.get(key) for v in shas.values()}
        same[f"{fam}:{key}"] = dict(distinct_hashes_over_T1_T7=len(vals), equals_current_file=(vals == {H[key]}) if key in H else None)
res["esci"]["po_recorded_hashes"] = same
# the validation used for threshold selection = the same sel_rows file
ev = {}
for pat, nm in (("esci_t/eval/esciEval_E4_T1_s0_step625__T1__sel.json.gz", "qwen"), ("esci_loss/eval/esciEval_E4_T1_s0_step625__T1__sel.json.gz", "qwen_alt"), ("esci_llama/eval/esciLlamaEval_EL_T1_s0_step625__T1__sel.json.gz", "llama")):
    f = R / pat
    if f.exists():
        m = json.load(gzip.open(f, "rt", encoding="utf-8"))["meta"]; ev[nm] = dict(rows_file_sha256=m.get("rows_file_sha256"), n_rows=m.get("n_rows_scored"), equals_sel_rows=m.get("rows_file_sha256") == H["sel"])
res["esci"]["threshold_validation_files"] = ev
# unused ('spare') normalised queries of the frozen permutation
import pandas as pd
ESCI = os.environ.get("ESCI_DIR", "external/esci-data")        # directory with the official ESCI parquet files
sys.path.insert(0, str(ROOT / "methods"))
import unicodedata
def norm_q(s):
    return " ".join(unicodedata.normalize("NFKC", str(s)).lower().split())
ex = pd.read_parquet(f"{ESCI}/examples.parquet"); pr = pd.read_parquet(f"{ESCI}/products.parquet")
ex = ex[(ex.product_locale == "us") & (ex.small_version == 1)].copy(); pr = pr[pr.product_locale == "us"]
ex = ex.merge(pr, on=["product_locale", "product_id"], how="left", indicator=True); ex = ex[ex._merge == "both"].drop(columns="_merge")
ex["nq"] = ex["query"].map(norm_q)
trq, teq = set(ex[ex.split == "train"].nq), set(ex[ex.split == "test"].nq)
train = ex[(ex.split == "train") & ~ex.nq.isin(trq & teq)]
qs = sorted(train.nq.unique()); perm = [qs[i] for i in np.random.default_rng(spl["seed"]).permutation(len(qs))]
assert perm[:200] == spl["diag"] and perm[200:400] == spl["sel"] and perm[400:5400] == spl["sft"], "frozen lists are not reproduced by the stated rule"
spare = perm[5400:]
byq = {q: d.drop_duplicates(["product_id", "esci_label"]) for q, d in train[train.nq.isin(set(spare[:400]))].groupby("nq")}
cand = spare[:200]; rows_c = sum(len(byq[q]) for q in cand); e_c = sum(int((byq[q].esci_label == "E").sum()) for q in cand)
res["esci"]["spare_pool"] = dict(train_side_queries_after_test_overlap_removal=len(qs), diag=200, sel=200, sft=5000, spare_queries=len(spare), frozen_lists_reproduced=True,
                                 first200_spare=dict(rule="perm[5400:5600] of default_rng(20261031) over the sorted train-side normalised queries (never trained on, never PO diagnostic / selection, never validation, never test)", queries=len(cand), judged_rows=int(rows_c), relevant_share=float(e_c / rows_c),
                                                     queries_hash=hashlib.sha256(json.dumps(cand).encode()).hexdigest()),
                                 mean_rows_per_query_sel=float(role["sel"]["rows"] / 200))
# usage history: any stored row / manifest file anywhere in the project that contains a spare-pool query
cs = set(cand); big = set(spare); used = {}
cands = [p for p in glob.glob(str(R / "**" / "*.jsonl"), recursive=True) + glob.glob(str(ROOT / "data" / "**" / "*.jsonl"), recursive=True) if os.path.getsize(p) < 400 * 2 ** 20]
scanned = 0
for p in cands:
    try:
        with open(p, encoding="utf-8") as fh:
            first = fh.readline()
            if '"nq"' not in first:
                continue
            scanned += 1
            hit = sum(1 for l in fh if json.loads(l).get("nq") in cs) + (1 if json.loads(first).get("nq") in cs else 0)
        if hit:
            used[os.path.relpath(p, ROOT)] = hit
    except Exception:
        pass
res["esci"]["spare_usage_history"] = dict(jsonl_files_with_nq_scanned=scanned, files_containing_any_first200_spare_query=used, note="only files that carry a normalised query field `nq` are covered; JSON manifests of other sets and in-code query sampling were checked separately by text search")

# ------------------------------------------------------------------ B MS MARCO
sm = json.load(open(ROOT / "data/msmarco_po/split_manifest.json"))
res["msmarco"]["split_manifest"] = {k: sm[k] for k in ("sizes", "overlap_id", "overlap_normalized_text", "passage_overlap_sel_vs_diag", "passage_overlap_sel_vs_train", "sel", "diag")}
poh = {}
for fam in ("qwen", "llama"):
    shas = {}
    for T in range(1, 8):
        p = R / "msm_po" / fam / f"p1_T{T}" / "p1_result.json"
        if p.exists():
            d = json.load(open(p)); shas[f"T{T}"] = d["data"]
    poh[fam] = shas
res["msmarco"]["po_recorded_data_hashes"] = {f"{fam}:{k}": len({v.get(k) for v in shas.values()}) for fam, shas in poh.items() for k in ("diag_pairs_sha256", "sel_pairs_sha256", "train_pairs_sha256")}
res["msmarco"]["po_recorded_values"] = {fam: sorted({json.dumps(v, sort_keys=True) for v in shas.values()}) for fam, shas in poh.items()}
vq = json.load(open(ROOT / "data/msmarco_po/val200/query_selection.json"))
qsel = json.load(open(ROOT / "data/msmarco_train_v2_5k/query_selection.json"))
tsel = json.load(open(ROOT / "data/msmarco_train_v2_5k/train_set_5000_selection.json"))
val_ids, sel5200 = set(vq["query_ids"]), set(qsel["query_ids"])
res["msmarco"]["val200_provenance"] = dict(source=vq.get("purpose"), val_manifest_sha256=vq.get("val_manifest_sha256"), n_val_queries=len(val_ids), val_in_5200=len(val_ids & sel5200), train_pairs_file="data/msmarco_train_v2_5k", n_5200=len(sel5200), n_train5000=len(tsel["kept_query_ids"]), n_diag_200_from_5200=len(sel5200 - set(tsel["kept_query_ids"])))
res["msmarco"]["unused_pool"] = dict(ir_datasets_id=qsel.get("ir_datasets_id"), n_queries_with_positive=int(qsel["n_queries_with_positive"]), used_5200=len(sel5200), used_val200=len(val_ids), unused_at_least=int(qsel["n_queries_with_positive"]) - len(sel5200 | val_ids),
                                     selection_rule_of_the_5200=qsel.get("selection_rule"), seed=qsel.get("seed"), note="trec queries (97) are from a different collection; other revision / PAKDD query lists are not enumerated here")

# ------------------------------------------------------------------ C PO inventory
inv = []
def po_rows(ds, fam, base, T):
    p = Path(base) / f"p1_T{T}" / "p1_result.json"
    d = json.load(open(p))
    cands = d["candidates"]; sel = d["selection"]
    ok = [("valid" in c and bool(c.get("valid"))) for c in cands]
    raw = sorted(glob.glob(str(Path(base) / f"p1_T{T}" / "p1_raw_cand*.json.gz")))
    return dict(dataset=ds, model=fam, T=f"T{T}", original_sha256=d["original_sha256"], n_candidates=len(cands), n_valid=int(d.get("n_valid_unique", d.get("n_valid", sum(ok)))), selected_index=sel["index"], selected_variant=sel["variant"], selection_kappa=sel.get("kappa"),
                original_kappa=sel.get("original_kappa"), changed=bool(sel.get("replaced", d.get("changed"))), candidate_kappas=[round(float(c["kappa"]), 4) if c.get("kappa") is not None else None for c in cands], raw_selection_margin_files=len(raw),
                n_selection_rows=d.get("n_selection_rows"), diag_confusion=d.get("confusion"))
for T in range(1, 8):
    inv.append(po_rows("ESCI", "Qwen3-4B", R / "esci_t", T)); inv.append(po_rows("ESCI", "Llama-3.1-8B", R / "esci_llama", T))
    for fam, name in (("qwen", "Qwen3-4B"), ("llama", "Llama-3.1-8B")):
        p = R / "msm_po" / fam / f"p1_T{T}" / "p1_result.json"
        d = json.load(open(p)); sel = d["selection"]
        inv.append(dict(dataset="MS MARCO", model=name, T=f"T{T}", original_sha256=d["original_sha256"], n_candidates=len(d["candidates"]), n_valid=int(d.get("n_valid", 0)), selected_index=sel["index"], selected_variant=sel["variant"], selection_kappa=sel.get("kappa"), original_kappa=None,
                        changed=bool(d["changed"]), candidate_kappas=[round(float(c["kappa"]), 4) if c.get("kappa") is not None else None for c in d["candidates"]], raw_selection_margin_files=len(glob.glob(str(R / "msm_po" / fam / f"p1_T{T}" / "*raw*"))), n_selection_rows=400, diag_confusion=d.get("confusion")))
res["po"]["inventory"] = inv
# D representative condition
chg = {(r["dataset"], r["model"]): [r["T"] for r in inv if r["dataset"] == r["dataset"] and r["changed"] and r["dataset"] == r["dataset"]] for r in inv}
rep = {}
for ds in ("ESCI", "MS MARCO"):
    for m in ("Qwen3-4B", "Llama-3.1-8B"):
        ch = sorted(r["T"] for r in inv if r["dataset"] == ds and r["model"] == m and r["changed"])
        rep[f"{ds}|{m}"] = dict(changed_T=ch, smallest_changed_T=ch[0] if ch else None, T1_changed="T1" in ch)
res["representative"] = rep
# SFT instruction hash of the PO runs = selected candidate (ESCI Llama T2 example)
chk = {}
for T in (2, 3, 5, 7):
    mani = json.load(open(R / "esci_llama" / f"p1_T{T}" / "esci_apocf_manifest.json"))["entries"][0]["full"]
    st = list((R / "esci_204/train/logs").glob(f"stats_ELQ_mix_T{T}_APOCF_s0.json"))
    if st:
        chk[f"ESCI Llama T{T}"] = dict(selected_text_sha256_equals_PO_SFT_run_instruction_sha256=hashlib.sha256(mani.encode()).hexdigest() == json.load(open(st[0])).get("instruction_sha256"), po_sft_tags_for_T=sorted(os.path.basename(x)[6:-5] for x in (R / "esci_204/train/logs").glob(f"stats_ELQ_mix_T{T}_*_s0.json")))
res["po"]["sft_instruction_check"] = chk
(OUT / "audit392_results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("esci", "msmarco")}, indent=1, default=str)[:9000])
print(json.dumps(res["representative"], indent=1)); print(json.dumps(res["po"]["sft_instruction_check"], indent=1))
