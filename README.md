# Reproduction package: *Which Combination Makes a Better LLM Relevance Judge? Prompt Optimization, Fine-Tuning Objectives, and Decision Thresholds*

Jaekeol Choi, Hankuk University of Foreign Studies.

Code, frozen instructions and ID manifests that reproduce the experiments of the paper. This version follows the protocol of the revised manuscript; see **Protocol** below and [`docs/MIGRATION.md`](docs/MIGRATION.md) for what differs from the first public version.
This repository contains **no model weights, no adapters, no copy of the public datasets and no per-example score files of the fine-tuned models**. Everything that can be downloaded is restored from the public sources by ID manifests; everything that can be recomputed is recomputed.

The study compares prompt optimization (APO-CF, "PO"), supervised fine-tuning (SFT) objectives and decision thresholds for an LLM relevance judge on ESCI (product search) and MS MARCO / TREC-DL 2019 and 2020 (passage ranking), with Qwen3-4B, Llama-3.1-8B-Instruct and Gemma-3-4B-it:
conditions **Base**, **CE**, **CE+soft κ** ("S"), **CE+AnyLoss-style κ** ("A"); original instructions **T1–T7** of each dataset, training seeds 0/1/2, final (625-update) checkpoint; default threshold t = 0 and a threshold selected on a separate validation set.

## Protocol (main experiments)

* **Models**: Qwen3-4B, Llama-3.1-8B-Instruct, Gemma-3-4B-it. **Instructions**: seven per dataset (T1–T7). **Conditions**: Base, CE, CE+soft κ, CE+AnyLoss-style κ; **PO** (APO-CF) → SFT with CE and CE+soft κ. Seeds 0/1/2, final (625-update) checkpoint.
* **PO scope**: Qwen3-4B and Llama-3.1-8B-Instruct on **ESCI and MS MARCO, T1–T7** (not Gemma, and no PO after SFT). The Base model evaluates the original and up to three generated candidates at t = 0 on the selection set and the highest-κ candidate is kept (ties: the original); it is then fixed for all SFT objectives. Which instructions were changed or kept is listed in [`docs/PO_SELECTION.md`](docs/PO_SELECTION.md).
* **Thresholds** (default t = 0 and a validation-selected t*): the threshold of each model, instruction and seed is the κ maximizer on a **separate validation set** of the same data source, fixed before the test sets are scored. **ESCI**: the validation split `sel` (the same split that selects the PO instruction). **MS MARCO**: the **train-derived validation set `val200`** (200 queries, 400 pairs; same one-positive / one-unlabelled sampling as the training pairs; no TREC-DL data); **the same threshold is applied unchanged to TREC-DL 2019 and to TREC-DL 2020**.
* **Cross-year thresholds are a supplementary analysis, not the main protocol**: Appendix C of the paper also reports thresholds selected on all judged pairs of one TREC-DL year and applied to the other year. The scripts `analysis/msm_{t7,llama,gemma}_analysis.py` implement that cross-year setting (their output columns are named CROSS); the main MS MARCO protocol is `analysis/po_val_analysis.py` (see section 5).

## 1. Layout

| path | content |
|---|---|
| `methods/` | trainers, evaluators, data-preparation and PO scripts (the code that produced the results; see `CHANGES_FROM_ORIGINAL.md`). MS MARCO validation scoring: `msm_val_eval.py` (Qwen), `msm_val_eval_lg.py` (Llama, Gemma); MS MARCO PO: `msm_apocf_p1.py`, `freeze_po.py` |
| `src/`, `evaluation/`, `cpmr_experiment/scripts/`, `cpmr_experiment/instructions_v2/scripts/` | model loading, prompt builders, scoring helpers (imported by `methods/`; the directory name `cpmr_experiment/` is the working-directory name of the original scripts and is kept so that no path has to be edited); the MS MARCO sampling builders (`build_msmarco_train_candidates.py`, `build_basic_hard_pairs.py`, `join_msmarco_pair_text.py`) document how the training, diagnostic and validation pairs were drawn |
| `configs/` | one YAML per model and dataset (LoRA, optimizer, input limit) |
| `data/manifests/` | ID manifests: the exact ESCI rows (`esci_rows_*.csv`), the MS MARCO training pairs, the **validation (`msmarco_val200_pairs.csv`) and PO-diagnostic (`msmarco_diag200_pairs.csv`) pairs**, the judged TREC-DL pairs, checksums (`esci_manifest.json`, `msmarco_manifest.json`) |
| `cpmr_experiment/esci_t/`, `cpmr_experiment/esci_llama/`, `methods/custom_instructions_T6_T7.json`, `cpmr_experiment/instructions_v2/manifest/` | frozen original instructions T1–T7 (ESCI and MS MARCO) and the APO-CF selection results of ESCI (T1–T7, Qwen and Llama) |
| `cpmr_experiment/msm_po/` | the APO-CF selection results of MS MARCO (T1–T7, Qwen and Llama) and the frozen PO instruction files (`freeze/`) |
| `cpmr_experiment/gemma231/sample_manifest.json` | checkpoint list and fixed query sample of the Gemma FP32 appendix |
| `scripts/` | `restore_data.py` (rebuild inputs), `verify_data.py` (check them), `make_ckpt_map.py`, `make_manifests*.py` (author-side tools that wrote the manifests) |
| `experiments/make_commands.py` | prints every training / evaluation / validation-scoring command of the paper (one condition, or `--all` for the full study) |
| `analysis/` | evaluation → threshold → tables → statistics → figure panels |
| `analysis/supplementary/` | scripts of the supplementary analyses (section 7), kept apart from the main protocol |
| `docs/` | `PO_SELECTION.md` (PO result per model, dataset and instruction), `ADDITIONAL_ANALYSES.md` (supplementary analyses and their code), `MIGRATION.md` (differences from the first public version) |
| `checks/run_checks.py` | CPU-only smoke checks (no GPU, no model download) |

## 2. Setup

* Python 3.12, `pip install -r requirements.txt` (versions used for the paper are listed there).
* Models (Hugging Face): `Qwen/Qwen3-4B` (revision `1cfa9a7208912126459214e8b04321603b3df60c`), `meta-llama/Llama-3.1-8B-Instruct` (revision `0e9e39f249a16976918f6564b8830bc894c89659`, **gated: accept the Meta license**), `google/gemma-3-4b-it` (revision `093f9f388b31de276ce2de164bdc2081324b9767`, **gated: accept the Gemma terms**). The commands use `--local_files_only`; drop it if the models are not cached locally, and set `HF_HOME` / `HF_HUB_OFFLINE` as usual.
* **Training settings** (all models): LoRA r 8 / α 16 / dropout 0.05 on the decoder q, k, v, o projections, AdamW, learning rate 2e-5, 625 updates with an effective batch of 16 sequences (ESCI: 16 rows; MS MARCO: 8 positive–negative pairs), warm-up over the first 10% of the updates followed by linear decay to 0, seeds 0 / 1 / 2, final checkpoint. Warm-up is **62 updates for MS MARCO** (`round(0.1 × 625)`, from `warmup_ratio: 0.1` in `configs/`) and **63 for ESCI** (`ceil(0.1 × 625)`, set in the ESCI trainers); the generated commands use these values without further arguments. Agreement-loss weight λ = 1, AnyLoss-style slope α = 10, denominator floor 1e-6; maximum input length 384 tokens for MS MARCO and 766 / 981 / 704 for ESCI (Qwen3-4B / Llama-3.1-8B / Gemma-3-4B).
* Hardware used: NVIDIA RTX PRO 6000 Blackwell Server Edition (about 96 GB), BF16 blocks with FP32 final norm / head / log-softmax. Peak allocated memory of an ESCI κ run (Qwen3-4B, micro-batch 4) is about 42 GB; MS MARCO Qwen3-4B runs (gradient checkpointing) about 10 GB. Select the GPU with `CUDA_VISIBLE_DEVICES` (the generated commands take `--gpu`).
* Roughly (one RTX PRO 6000): Qwen3-4B ESCI CE 11 min, CE+κ 38 min per run; ESCI evaluation about 3–4.5 min per checkpoint and set (validation / test). The full study is several hundred training runs (see `experiments/commands/`).

## 3. Restore and verify the data

```bash
# ESCI: download https://github.com/amazon-science/esci-data (shopping_queries_dataset_{examples,products}.parquet), then
python scripts/restore_data.py esci --esci-dir <esci-data>/shopping_queries_dataset --tokenizer Qwen/Qwen3-4B
# MS MARCO training pairs, validation set val200, PO diagnostic set diag200 and TREC-DL 2019/2020 judged pairs (downloaded by the ir_datasets package)
python scripts/restore_data.py msmarco
# only for the Qwen3-4B MS MARCO trainers: rebuild the passage cache (needs the pinned Qwen3-4B tokenizer)
python scripts/restore_data.py msmarco-cache
python scripts/verify_data.py
```

`restore_data.py` does **no sampling**: it reads the row lists from `data/manifests/` and rebuilds the files the trainers read (`cpmr_experiment/esci/audit/{sft2,diag,sel,test300,calib}_rows.jsonl`, `data/msmarco_train_v2_5k/pairs_text_basic_5000.jsonl`, `data/msmarco_po/val200/pairs_text_basic_seed0.jsonl`, `data/msmarco_po/diag200/pairs_text_basic_diag200.jsonl`, `data/scenarios/msmarco_trec{2019,2020}_bm25top100_pool.jsonl`) and compares every file with the recorded SHA-256 (ESCI row files and the MS MARCO pair files (training, val200, diag200) are byte-identical to the originals; the TREC-DL pool files contain the judged pairs only, which are the only pairs that enter an evaluation; their texts are checked pair by pair).

| set | queries | pairs | role |
|---|---:|---:|---|
| ESCI training (`sft2`) | 5,000 | 10,000 (2 products per query; 4,249 Relevant) | SFT |
| ESCI diagnostic (`diag`) | 200 | 3,879 | PO confusion matrix |
| ESCI validation (`sel`) | 200 | 4,047 | PO selection and thresholds |
| ESCI test (`test300`) | 300 | 6,078 | evaluation |
| ESCI independent calibration (`calib`) | 200 | 3,939 | threshold selection in the independent-calibration check only (Appendix C) |
| MS MARCO training | 5,000 | 10,000 (1 positive, 1 negative per query) | SFT |
| MS MARCO PO diagnostic (`diag200`) | 200 | 400 (1 positive, 1 negative per query) | PO confusion matrix |
| MS MARCO validation (`val200`) | 200 | 400 (1 positive, 1 negative per query) | PO selection and thresholds |
| TREC-DL 2019 / 2020 (judged) | 42 / 54 | 1,102 / 1,432 | evaluation; the threshold selected on `val200` is applied to both years |

Label mapping: ESCI E → Relevant, S/C/I → Irrelevant; TREC-DL grades 2–3 → Relevant, 0–1 → Irrelevant. The splits are disjoint by NFKC-lower-cased whitespace-normalized query text (`verify_data.py` checks all pairs of splits). ESCI product text is the fixed concatenation Title / Brand / Color / Bullet points / Description, cut once to 512 Qwen tokens (description first); the cut flags are in the manifests.

The 400 `val200` pairs were drawn from the 400 queries that extended the 5,200-query selection of the training split to 5,600 (disjoint from the 5,000 SFT queries and from `diag200`; 241 of them have a known positive in the BM25 top-100 and 200 were sampled with a fixed seed). Positive = the known positive with the smallest passage id; negative = one passage drawn uniformly from the BM25 top-1,000 of the query after removing all known positives, labelled Irrelevant (an *unlabelled* passage, never redrawn). The sampling procedure and the other fixed parameters are recorded in `data/manifests/msmarco_manifest.json` (`val200`, `diag200`) and implemented by the builders listed in section 1. `verify_data.py` checks that the training, `val200`, `diag200` and TREC-DL queries are pairwise disjoint (query id and normalized text).

## 4. One condition end to end (Qwen3-4B, ESCI, T1, seed 0, CE + soft κ)

```bash
python experiments/make_commands.py --model qwen --dataset esci --paths soft --instr 1 --seeds 0 --gpu 0
```

prints, in order, the training command (`methods/esci_loss_train_245.py --loss mix --lam 1.0 --alpha 10 --final_only --instr T1 --seed 0 ...`), the checkpoint-map command and the evaluation commands (`methods/esci_eval.py --jobs EQ_mix_T1_s0_step625:T1 ...` on the validation set `sel` and on `test300`). Run them from the repository root. `--paths` accepts `base,ce,soft,any,po-base,po-ce,po-soft`; `--stage prep|train|eval|all`.

## 5. All experiments of the paper

`python experiments/make_commands.py --all --out experiments/commands` writes one script per model and dataset (`qwen_esci.sh`, `qwen_msmarco.sh`, `llama_*`, `gemma_*`), plus `pilot245.sh` and `gemma_fp32.sh`. Each script contains the training, evaluation and (MS MARCO) validation-scoring stages.

| paper | experiment | commands / code |
|---|---|---|
| §4, Table 1 | data and splits | `scripts/restore_data.py`, `scripts/verify_data.py` |
| §5.1–5.2, App. A | Base / CE / S / A, T1–T7, 3 seeds, t = 0 and validation threshold — **ESCI** | `experiments/commands/{qwen,llama,gemma}_esci.sh`; tables `analysis/esci204_main.py` (Qwen), `esci208_llama_main.py`, `esci217_gemma_main.py` (threshold: the `sel` split) |
| §5.1–5.2, App. A | same — **MS MARCO**: models scored on TREC-DL 2019/2020 **and on `val200`**; threshold from `val200`, applied to both years | `experiments/commands/{qwen,llama,gemma}_msmarco.sh` (stages `eval` and `val`); tables and statistics `analysis/po_val_analysis.py` (+ `po_val_md.py`) |
| §5.3, App. A | PO → CE / S, T1–T7, Qwen3-4B and Llama-3.1-8B-Instruct, **ESCI and MS MARCO** | `--paths po-base,po-ce,po-soft` (changed instructions only; kept instructions are the original runs, see `docs/PO_SELECTION.md`); ESCI tables `analysis/esci_po_analysis.py` (+ `esci_po_md.py`); MS MARCO `analysis/po_val_analysis.py` |
| §6.1, App. B | AUC and paired query bootstrap | `analysis/auc61_kappa.py` (writes `kappa.pkl` = ESCI validation split + MS MARCO cross-year reference, and `kappa_val200.pkl` = ESCI validation split + **MS MARCO `val200` threshold**), `auc61_boot.py`, `auc61_report.py` (reads `kappa_val200.pkl` when present; AUC itself does not depend on the threshold) |
| §6.2, App. B | error counts and item-level transitions (corrected / new errors on the same examples) | `analysis/err62_main.py --kappa .../kappa_val200.pkl` (ESCI and MS MARCO at the main-protocol thresholds), `errfig_plot.py`; MS MARCO error counts at the `val200` threshold are also in `po_val_checkpoint_metrics.csv` (`po_val_analysis.py`) |
| §6.3, App. B | threshold re-selection on 500 validation resamples and panels | `analysis/err63_main.py --kappa .../kappa_val200.pkl` (ESCI: the `sel` queries are resampled; MS MARCO: the 200 `val200` queries), `thrfig_plot.py`; the same MS MARCO resampling is in `po_val_analysis.py` |
| App. B | Gemma FP32 evaluation | `experiments/commands/gemma_fp32.sh`, `analysis/err62_gemma.py`, `err63_gemma.py` |
| App. B | ESCI objective pilot: λ ∈ {0.3, 1, 3}, α ∈ {5, 10, 20}, CE+label-margin BCE | `experiments/commands/pilot245.sh`, `analysis/esci245_analysis.py` |
| App. B, C and the supplementary analyses | training budget, BCE / MAE controls, label length, seed and joint bootstrap, sampling and coverage, cross-year thresholds, independent calibration (ESCI Llama-3.1-8B, T1–T7) | see section 7 and `docs/ADDITIONAL_ANALYSES.md` |

## 6. Order of an analysis

1. Train and evaluate (`experiments/commands/*.sh`): each evaluation writes one gzip JSON per (checkpoint, instruction, set) with the four log-probabilities of the label and the terminator for both labels; the margin is m = [log P(Relevant) + log P(terminator)] − [log P(Irrelevant) + log P(terminator)], decision m > t.
2. Threshold: `analysis/esci204_common.py::best_threshold` — hard-κ maximum over all validation pairs of the model's own scores; candidates are 0, the midpoints of consecutive distinct scores, and below/above all scores; ties → t = 0, then smaller |t|, then smaller t. One threshold per model, instruction and seed; Base thresholds are never transferred. **Validation set: ESCI `sel`; MS MARCO `val200` (scored by `methods/msm_val_eval*.py`), with the same threshold applied to both TREC-DL years.**
3. Tables and statistics: the scripts of section 5. Aggregation: κ per instruction and seed → equal-weight mean over the instruction subset → mean over seeds (Base: one score).
4. Panels: `errfig_plot.py`, `thrfig_plot.py` (one vector file per panel; the LaTeX layout is not part of this repository).

```bash
# MS MARCO main protocol (stored scores of the three models; val200 threshold applied to both years)
python analysis/auc61_kappa.py --root . --mods analysis --out cpmr_experiment/auc61/out    # kappa.pkl (cross-year reference) and kappa_val200.pkl (main protocol)
python analysis/po_val_analysis.py --root . --mods analysis --kappa cpmr_experiment/auc61/out/kappa.pkl --out cpmr_experiment/msm_val/summary
python analysis/po_val_md.py cpmr_experiment/msm_val/summary/po_val_results.json cpmr_experiment/msm_val/summary/po_val_results.md
# error transitions and threshold stability (ESCI and MS MARCO, main-protocol thresholds); pass kappa.pkl instead to get the supplementary cross-year setting for MS MARCO
python analysis/err62_main.py --root . --mods analysis --kappa cpmr_experiment/auc61/out/kappa_val200.pkl --out cpmr_experiment/err62/out
python analysis/err63_main.py --root . --mods analysis --kappa cpmr_experiment/auc61/out/kappa_val200.pkl --out cpmr_experiment/err63/out
```

`po_val_analysis.py` asserts before it writes anything that the t = 0 and cross-year values it recomputes equal those of `kappa.pkl` (difference < 1e-9). The score files are found by name under `cpmr_experiment/**`; keep only the run directories of one study in the tree (the same file name from another model or study would be read as well).

## 7. Scope and caveats

* **Own-instruction evaluation.** Each SFT model is evaluated with the instruction it was trained with; seed means average instruction-level κ with equal weights.
* **PO.** The frozen selections are the reference ([`docs/PO_SELECTION.md`](docs/PO_SELECTION.md)); the proposer samples at temperature 1.0, so re-running the PO stage gives other candidate texts.
* **Thresholds.** Main protocol: `val200` (MS MARCO) / `sel` (ESCI). The cross-year setting is supplementary; the kappa table passed with `--kappa` selects the mode (`kappa_val200.pkl` main, `kappa.pkl` cross-year), see section 6.
* **Supplementary analyses** (training budget, BCE / MAE controls, label length, seed and joint bootstrap, sampling audit, cross-year, independent calibration) are kept apart from the main protocol: [`docs/ADDITIONAL_ANALYSES.md`](docs/ADDITIONAL_ANALYSES.md).
* **Precision and determinism.** BF16 forward passes with FP32 head / log-softmax scoring. Training seeds, adapter initialization and example order are fixed, but GPU kernels are not bit-deterministic, so re-training changes the last digits; statistics on stored scores are deterministic.
* **Checks.** `checks/run_checks.py` (CPU only); the restore of the MS MARCO and ESCI inputs reproduces the original file hashes; the analysis scripts reproduce the paper's stored tables on the author's score files. The per-example scores are not distributed, so the analyses of stored scores start from the evaluation commands; a complete re-training from this repository has not been repeated.
* **Compute.** The full study consists of several hundred training runs; the 5-run objective pilot took about 3.9 GPU-hours.

## 8. Sources and licenses of data and models

See `DATA_LICENSES.md`. ESCI (Amazon Shopping Queries Dataset, Apache-2.0), MS MARCO and TREC Deep Learning (research use terms of the collection owners) and the three models are **not** redistributed here and keep their own licenses and access conditions.
