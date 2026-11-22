# PO (APO-CF) selection, T1-T7

Generated from the frozen `p1_result.json` files in this repository (the numbers are read from those files, not recomputed).

Scope of PO in the paper: **Qwen3-4B and Llama-3.1-8B-Instruct, on ESCI and MS MARCO, for each of the seven instructions T1-T7**. Gemma-3-4B-it has no PO. PO is applied to the Base model (the model is judge and proposer), one round, three generated candidates (shorter / similar / bare) plus the original; the candidate with the highest Cohen's kappa at the default threshold t = 0 on the selection set is kept (ties: the original). The selected instruction is then fixed and used for the SFT objectives (PO -> SFT); no PO is run after SFT.

* Diagnostic set (confusion matrix given to the proposer): ESCI `diag` (200 queries / 3,879 pairs), MS MARCO `diag200` (200 queries / 400 pairs).
* Selection set (the same set that later supplies the validation threshold): ESCI `sel` (200 queries / 4,047 pairs), MS MARCO `val200` (200 queries / 400 pairs).
* **Kept** = the original text was retained (selection index 0). A kept instruction is not a separate run: the PO configuration is the original configuration (same model, same predictions). Commands are generated only for the **changed** instructions (`experiments/make_commands.py`).
* The proposer samples at temperature 1.0: re-running the stage gives other candidate texts, so the frozen selections are the reference. The recorded meta-prompts are rebuilt byte for byte by `src/apocf_prompt.py` (`checks/run_checks.py`).

## Qwen3-4B on ESCI

| instruction | selected | status | selection kappa (original -> selected) | valid candidates |
|---|---|---|---|---|
| T1 | similar | changed | 0.3240 -> 0.3372 | 4 of 4 |
| T2 | shorter | changed | 0.3014 -> 0.3263 | 4 of 4 |
| T3 | bare | changed | 0.2574 -> 0.3075 | 3 of 4 |
| T4 | bare | changed | 0.2629 -> 0.2870 | 4 of 4 |
| T5 | bare | changed | 0.2622 -> 0.3200 | 4 of 4 |
| T6 | shorter | changed | 0.2934 -> 0.3172 | 4 of 4 |
| T7 | similar | changed | 0.3066 -> 0.3092 | 4 of 4 |

Changed: T1, T2, T3, T4, T5, T6, T7; kept: none.

## Llama-3.1-8B-Instruct on ESCI

| instruction | selected | status | selection kappa (original -> selected) | valid candidates |
|---|---|---|---|---|
| T1 | unchanged | kept | 0.2774 -> 0.2774 | 3 of 4 |
| T2 | shorter | changed | 0.1842 -> 0.2474 | 3 of 4 |
| T3 | bare | changed | 0.2026 -> 0.2484 | 3 of 4 |
| T4 | unchanged | kept | 0.2538 -> 0.2538 | 4 of 4 |
| T5 | shorter | changed | 0.1520 -> 0.2214 | 3 of 4 |
| T6 | unchanged | kept | 0.2825 -> 0.2825 | 1 of 4 |
| T7 | similar | changed | 0.2322 -> 0.2511 | 3 of 4 |

Changed: T2, T3, T5, T7; kept: T1, T4, T6.

## Qwen3-4B on MS MARCO

| instruction | selected | status | selection kappa (original -> selected) | valid candidates |
|---|---|---|---|---|
| T1 | shorter | changed | 0.6750 -> 0.7800 | 4 of 4 |
| T2 | similar | changed | 0.6750 -> 0.7850 | 3 of 4 |
| T3 | similar | changed | 0.7500 -> 0.8100 | 2 of 4 |
| T4 | shorter | changed | 0.7150 -> 0.7950 | 3 of 4 |
| T5 | shorter | changed | 0.6900 -> 0.7300 | 3 of 4 |
| T6 | similar | changed | 0.6650 -> 0.7850 | 3 of 4 |
| T7 | shorter | changed | 0.6200 -> 0.8000 | 3 of 4 |

Changed: T1, T2, T3, T4, T5, T6, T7; kept: none.

## Llama-3.1-8B-Instruct on MS MARCO

| instruction | selected | status | selection kappa (original -> selected) | valid candidates |
|---|---|---|---|---|
| T1 | unchanged | kept | 0.7350 -> 0.7350 | 1 of 4 |
| T2 | similar | changed | 0.6900 -> 0.7700 | 3 of 4 |
| T3 | similar | changed | 0.7500 -> 0.7600 | 3 of 4 |
| T4 | unchanged | kept | 0.7450 -> 0.7450 | 1 of 4 |
| T5 | unchanged | kept | 0.7750 -> 0.7750 | 2 of 4 |
| T6 | unchanged | kept | 0.7050 -> 0.7050 | 2 of 4 |
| T7 | unchanged | kept | 0.7350 -> 0.7350 | 2 of 4 |

Changed: T2, T3; kept: T1, T4, T5, T6, T7.

## Files

| content | ESCI | MS MARCO |
|---|---|---|
| selection result, confusion matrix, proposer prompts and replies | `cpmr_experiment/esci_t/p1_T*/` (Qwen), `cpmr_experiment/esci_llama/p1_T*/` (Llama): `p1_result.json`, `p1_confusion.json`, `p1_exchange.json` | `cpmr_experiment/msm_po/{qwen,llama}/p1_T*/`: the same three files, `po_instruction.json` |
| selected instruction texts | `cpmr_experiment/*/p1_T*/esci_apocf_manifest.json` (display ids `T{k}` and `T{k}_APOCF`) | `cpmr_experiment/msm_po/freeze/custom_instructions_po_{qwen,llama}.json`; `po_selection_manifest.json` records every candidate and the SHA-256 of each text |
| Base scores of all candidates on the selection set | `p1_raw_cand*.json.gz` | stored with the evaluation scores (not distributed); recomputed by the `val` stage of `experiments/make_commands.py` |
