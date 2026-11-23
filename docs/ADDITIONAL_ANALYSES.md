# Supplementary analyses and their code

The main experiments are described in the README (protocol, restore, commands). The analyses below are **supplementary checks** that appear in the appendices of the paper. They are kept apart from the main protocol: they reuse the trained models and their stored scores (or score a few additional conditions that are stated explicitly), and none of them changes a main result.

All scripts are in `analysis/supplementary/` unless a path is given. They read the per-example score files written by the evaluation commands of the README (`cpmr_experiment/**/*Eval_*.json.gz`; found by file name, so keep only the run directories of one study in the tree). **The score files are not distributed**; the commands that regenerate them are in `experiments/commands/`. Each script takes `--root <project root> --out <dir>` and prints the table it writes.

| paper (appendix) | check | code | additional computation | notes |
|---|---|---|---|---|
| B, *Learning Rate and Training Budget* | learning rate and training duration for CE vs. CE+soft κ (Qwen3-4B, ESCI T1), validation learning curves, measured training times | `methods/esci_ce_train_380.py` (CE trainer with an explicit update budget), `budget380.py` (fixes the budget before any evaluation), `run380.sh`, `unit380.py`, `report380.py` | new CE training runs of longer duration and their validation / test scoring | one instruction (T1), three seeds |
| B, *Auxiliary Classification Losses* | CE+2·MAE and CE+BCE controls against CE and CE+soft κ, **ESCI Qwen3-4B, T1–T7, seeds 0–2** | `methods/esci_loss_train_384.py` (`--loss mixmae`, `--loss mixbce`; library `methods/esci_loss_lib.py`), `unit384.py` (CPU loss and gradient checks), `micro386_diag.py` (micro-batch numerical diagnosis), `run407.sh` (queue for T2–T7), `report407.py` (tables and paired bootstrap) | new training for the controls: T1 (5 runs) and T2–T7 (36 runs); CE and CE+soft κ are the main-experiment runs | ESCI Qwen3-4B only; Llama, Gemma and MS MARCO controls are not run (for MS MARCO the soft-κ term equals twice the mean absolute error under balanced batches, see the paper) |
| B, *Label Length and Score Normalization* | label-length-normalized scores and the label lengths of the three models | `norm388.py` (ESCI T1, stored scores), `analyze390.py` and `msm_gemma_label390.py` (Gemma MS MARCO T1; label-wise re-scoring of existing checkpoints) | `msm_gemma_label390.py` re-scores seven existing checkpoints | no training |
| A, *Variation across Training Seeds*; C, *Validation and Test Sampling Variability* | per-seed results and the joint resampling of validation and test queries (2,000 paired draws) | `joint395.py` | none (stored scores) | covers the original instructions T1–T7 of all three models on both datasets |
| A, *Sampling and Evaluation Coverage* | rank distribution of the sampled MS MARCO negatives, judged coverage of the TREC-DL pools | `audit397.py` | none (existing files only; no retrieval, no model call) | the BM25 top-100 pools of TREC-DL and the negative-sampling rank audit |
| C, *Cross-Year Threshold Selection* | thresholds selected on all judged pairs of one TREC-DL year and applied to the other year, compared with the `val200` threshold | `cross399.py`; the original cross-year implementation is `analysis/msm_{t7,llama,gemma}_analysis.py` | none (stored scores) | supplementary setting; the main protocol uses `val200` |
| C, *Independent Calibration after Prompt Selection* | threshold selected on a new 200-query ESCI calibration pool (`calib`, 3,939 pairs; query-disjoint from training, diagnostic, validation and test) instead of the validation split that also selected the PO instruction; **ESCI, Llama-3.1-8B-Instruct, T1–T7**, original and PO-selected text, Base and CE+soft κ (3 seeds); the PO selection and the checkpoints are fixed | `run393.sh` (T2: 8 scoring jobs) and `run410.sh` (T1, T3–T7: 36 scoring jobs; includes a reproduction gate, `gate_compare410.py`), `analyze410.py`; pool: `data/manifests/esci_rows_calib.csv` (`restore_data.py esci`), builder `build_calib393.py`, audit `audit392.py` | 44 scoring jobs on the pool (no training); instructions whose PO kept the original text (T1, T4, T6) reuse the original scores | results in `results/independent_calibration/` |

## Status of each item in this repository

* The scripts are the code that produced the reported numbers (path and wording edits only). Their inputs are the per-example score files written by the evaluation commands; these files are not distributed. `analyze410.py` was re-run on the scores of the paper and gives the tables in `results/independent_calibration/` exactly.
* `analysis/po_val_analysis.py` (main MS MARCO protocol; not in this list) and `analysis/esci_po_analysis.py` were re-run on the stored score files and reproduce their earlier outputs; see the README, section 7.
* Statistics: paired test-query bootstraps keep all pairs of a sampled query, use one set of resampled queries for all compared conditions, and describe evaluation-query variation conditional on the trained models and selected thresholds; the interval type and the draws are stated at the top of each script.

## Independent calibration, T1–T7: how to run

```bash
python scripts/restore_data.py esci --esci-dir <esci-data>/shopping_queries_dataset    # also restores cpmr_experiment/esci/audit/calib_rows.jsonl
G0=<gpu0> G1=<gpu1> bash analysis/supplementary/run393.sh                               # T2 (8 jobs) on the pool
G0=<gpu0> G1=<gpu1> bash analysis/supplementary/run410.sh                               # T1, T3-T7 (36 jobs); stops if the reproduction gate fails
python analysis/supplementary/analyze410.py --root . --out results/independent_calibration
```

Thresholds follow the main rule (κ maximizer; ties: 0, then smaller |t|, then smaller t) and are selected per instruction, seed and original / PO condition, once on the validation split `sel` and once on `calib`, then applied to the same test300 scores. Aggregation, bootstrap (10,000 paired test-query draws, models and thresholds fixed) and the exact tables are described at the top of `analyze410.py`; the files in `results/independent_calibration/` are its outputs on the scores of the paper.
