# Differences from the first public version of this repository

The first public version of the repository was built from an earlier state of the experiments and did **not** follow the protocol of the revised manuscript in the points below. The earlier text and scripts should not be read as the protocol of the paper. This version replaces them; nothing in the earlier version should be assumed to describe the same protocol unless it is restated here.

| topic | first public version | this version (= revised manuscript) |
|---|---|---|
| MS MARCO threshold | README and scripts selected the threshold on **all judged pairs of the other TREC-DL year** and applied it to the evaluated year | threshold selected on the **train-derived validation set `val200`** (200 queries, 400 pairs) and applied unchanged to **both** TREC-DL years; the cross-year thresholds remain only as a **supplementary analysis** (Appendix C) |
| MS MARCO data manifests | training pairs and TREC-DL judged pairs only | additionally `val200` and the PO diagnostic set `diag200` (`data/manifests/msmarco_{val200,diag200}_pairs.csv`); `restore_data.py` and `verify_data.py` restore and check them (SHA-256, disjointness of all query sets) |
| MS MARCO validation scoring | not available | `methods/msm_val_eval.py`, `msm_val_eval_lg.py`; stage `val` of `experiments/make_commands.py` |
| MS MARCO analysis | cross-year (`msm_*_analysis.py`, and `err62` / `err63` / `auc61` with the cross-year threshold) | `analysis/po_val_analysis.py` (threshold levels, paired contrasts, 500-resample threshold variation, error counts at the `val200` threshold); `auc61_kappa.py` writes `kappa_val200.pkl`, which `err62_main.py`, `err63_main.py` and `auc61_report.py` use for the main protocol (`kappa.pkl` keeps the cross-year setting) |
| PO scope | ESCI, T1–T3 | **Qwen3-4B and Llama-3.1-8B-Instruct, ESCI and MS MARCO, T1–T7** (no Gemma; PO before SFT only); selections and records for all 28 model–dataset–instruction cells in `cpmr_experiment/{esci_t,esci_llama,msm_po}/` and `docs/PO_SELECTION.md` |
| PO commands | ESCI T1–T3 | `make_commands.py --paths po-base,po-ce,po-soft` covers the changed instructions of all four model–dataset combinations |
| APO-CF meta-prompt | product (ESCI) domain only | product and passage (MS MARCO) domains; all 84 recorded meta-prompts are rebuilt byte for byte (`checks/run_checks.py`) |
| supplementary analyses | Appendix B pilot only | `docs/ADDITIONAL_ANALYSES.md`, `analysis/supplementary/` |
| internal wording | internal experiment-log numbers in comments and docstrings | removed; no behaviour of the code changed (see `CHANGES_FROM_ORIGINAL.md`) |

The main MS MARCO tables of the revised manuscript use the `val200` threshold; the cross-year thresholds are reported only as the supplementary comparison of Appendix C.
