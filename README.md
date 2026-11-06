# Reproduction package: *Which Combination Makes a Better LLM Relevance Judge? Prompt Optimization, Fine-Tuning Objectives, and Decision Thresholds*

Jaekeol Choi, Hankuk University of Foreign Studies.

Code, frozen instructions and ID manifests that reproduce the experiments of the paper.
This repository contains **no model weights, no adapters, no copy of the public datasets and no per-example score files of the fine-tuned models**. Everything that can be downloaded is restored from the public sources by ID manifests; everything that can be recomputed is recomputed.

The study compares prompt optimization (APO-CF, "PO"), supervised fine-tuning (SFT) objectives and decision thresholds for an LLM relevance judge on ESCI (product search) and MS MARCO / TREC-DL 2019 and 2020 (passage ranking), with Qwen3-4B, Llama-3.1-8B-Instruct and Gemma-3-4B-it:
conditions **Base**, **CE**, **CE+soft κ** ("S"), **CE+AnyLoss-style κ** ("A"); original instructions **T1–T7** of each dataset, training seeds 0/1/2, final (625-update) checkpoint; default threshold t = 0 and a threshold selected on a separate validation set.

## 1. Layout

| path | content |
|---|---|
| `methods/` | trainers, evaluators, data-preparation and PO scripts (the code that produced the results; see `CHANGES_FROM_ORIGINAL.md`) |
| `src/`, `evaluation/`, `cpmr_experiment/scripts/`, `cpmr_experiment/instructions_v2/scripts/` | model loading, prompt builders, scoring helpers (imported by `methods/`; the directory name `cpmr_experiment/` is the working-directory name of the original scripts and is kept so that no path has to be edited) |
| `configs/` | one YAML per model and dataset (LoRA, optimizer, input limit) |
| `data/manifests/` | ID manifests: the exact ESCI rows (`esci_rows_*.csv`), the MS MARCO training pairs, the judged TREC-DL pairs, checksums (`esci_manifest.json`, `msmarco_manifest.json`) |
| `cpmr_experiment/esci_t/`, `cpmr_experiment/esci_llama/`, `methods/custom_instructions_T6_T7.json`, `cpmr_experiment/instructions_v2/manifest/` | frozen instructions T1–T7 (ESCI and MS MARCO), the APO-CF selection results (T1–T3) and the Base selection scores that the Base thresholds need |
| `cpmr_experiment/gemma231/sample_manifest.json` | checkpoint list and fixed query sample of the Gemma FP32 appendix |
| `scripts/` | `restore_data.py` (rebuild inputs), `verify_data.py` (check them), `make_ckpt_map.py`, `make_manifests.py` (author-side tool that wrote the manifests) |
| `experiments/make_commands.py` | prints every training / evaluation command of the paper (one condition, or `--all` for the full study) |
| `analysis/` | evaluation → threshold → tables → statistics → figure panels |
| `checks/run_checks.py` | CPU-only smoke checks (no GPU, no model download) |

## 2. Setup

* Python 3.12, `pip install -r requirements.txt` (versions used for the paper are listed there).
* Models (Hugging Face): `Qwen/Qwen3-4B` (revision `1cfa9a7208912126459214e8b04321603b3df60c`), `meta-llama/Llama-3.1-8B-Instruct` (revision `0e9e39f249a16976918f6564b8830bc894c89659`, **gated: accept the Meta license**), `google/gemma-3-4b-it` (revision `093f9f388b31de276ce2de164bdc2081324b9767`, **gated: accept the Gemma terms**). The commands use `--local_files_only`; drop it if the models are not cached locally, and set `HF_HOME` / `HF_HUB_OFFLINE` as usual.
* Hardware used: NVIDIA RTX PRO 6000 Blackwell Server Edition (about 96 GB), BF16 blocks with FP32 final norm / head / log-softmax. Peak allocated memory of an ESCI κ run (Qwen3-4B, micro-batch 4) is about 42 GB; MS MARCO Qwen3-4B runs (gradient checkpointing) about 10 GB. Select the GPU with `CUDA_VISIBLE_DEVICES` (the generated commands take `--gpu`).
* Roughly (one RTX PRO 6000): Qwen3-4B ESCI CE 11 min, CE+κ 38 min per run; ESCI evaluation about 3–4.5 min per checkpoint and set (validation / test). The full study is several hundred training runs (see `experiments/commands/`).

## 3. Restore and verify the data

```bash
# ESCI: download https://github.com/amazon-science/esci-data (shopping_queries_dataset_{examples,products}.parquet), then
python scripts/restore_data.py esci --esci-dir <esci-data>/shopping_queries_dataset --tokenizer Qwen/Qwen3-4B
# MS MARCO training pairs and TREC-DL 2019/2020 judged pairs (downloaded by the ir_datasets package)
python scripts/restore_data.py msmarco
# only for the Qwen3-4B MS MARCO trainers: rebuild the passage cache (needs the pinned Qwen3-4B tokenizer)
python scripts/restore_data.py msmarco-cache
python scripts/verify_data.py
```

`restore_data.py` does **no sampling**: it reads the row lists from `data/manifests/` and rebuilds the files the trainers read (`cpmr_experiment/esci/audit/{sft2,diag,sel,test300}_rows.jsonl`, `data/msmarco_train_v2_5k/pairs_text_basic_5000.jsonl`, `data/scenarios/msmarco_trec{2019,2020}_bm25top100_pool.jsonl`) and compares every file with the recorded SHA-256 (ESCI row files and the MS MARCO pair file are byte-identical to the originals; the TREC-DL pool files contain the judged pairs only, which are the only pairs that enter an evaluation; their texts are checked pair by pair).

| set | queries | pairs | role |
|---|---:|---:|---|
| ESCI training (`sft2`) | 5,000 | 10,000 (2 products per query; 4,249 Relevant) | SFT |
| ESCI diagnostic (`diag`) | 200 | 3,879 | PO confusion matrix |
| ESCI validation (`sel`) | 200 | 4,047 | PO selection and thresholds |
| ESCI test (`test300`) | 300 | 6,078 | evaluation |
| MS MARCO training | 5,000 | 10,000 (1 positive, 1 negative per query) | SFT |
| TREC-DL 2019 / 2020 (judged) | 42 / 54 | 1,102 / 1,432 | evaluation; thresholds are selected on the other year |

Label mapping: ESCI E → Relevant, S/C/I → Irrelevant; TREC-DL grades 2–3 → Relevant, 0–1 → Irrelevant. The splits are disjoint by NFKC-lower-cased whitespace-normalized query text (`verify_data.py` checks all pairs of splits). ESCI product text is the fixed concatenation Title / Brand / Color / Bullet points / Description, cut once to 512 Qwen tokens (description first); the cut flags are in the manifests.

## 4. One condition end to end (Qwen3-4B, ESCI, T1, seed 0, CE + soft κ)

```bash
python experiments/make_commands.py --model qwen --dataset esci --paths soft --instr 1 --seeds 0 --gpu 0
```

prints, in order, the training command (`methods/esci_loss_train_245.py --loss mix --lam 1.0 --alpha 10 --final_only --instr T1 --seed 0 ...`), the checkpoint-map command and the evaluation commands (`methods/esci_eval.py --jobs EQ_mix_T1_s0_step625:T1 ...` on the validation set `sel` and on `test300`). Run them from the repository root. `--paths` accepts `base,ce,soft,any,po-base,po-ce,po-soft`; `--stage prep|train|eval|all`.

## 5. All experiments of the paper

`python experiments/make_commands.py --all --out experiments/commands` writes one script per model and dataset (`qwen_esci.sh`, `qwen_msmarco.sh`, `llama_*`, `gemma_*`), plus `pilot245.sh` and `gemma_fp32.sh`.

| paper | experiment | commands |
|---|---|---|
| §4, Table 1 | data and splits | `scripts/restore_data.py`, `scripts/verify_data.py` |
| §5.1–5.2, App. A | Base / CE / S / A, T1–T7, 3 seeds, t = 0 and validation threshold | `experiments/commands/{qwen,llama,gemma}_{esci,msmarco}.sh` (paths `base,ce,soft,any`); tables: `analysis/esci204_main.py` (Qwen ESCI), `esci208_llama_main.py`, `esci217_gemma_main.py`, `msm_t7_analysis.py`, `msm_llama_analysis.py`, `msm_gemma_analysis.py` |
| §5.3, App. A | PO (APO-CF) → CE / S on ESCI, T1–T3, Qwen3-4B and Llama-3.1-8B | `--paths po-base,po-ce,po-soft` (Llama: T1 keeps the original text, so T1 is an alias of the original runs, difference 0) |
| §6.1, App. B | AUC and paired query bootstrap | `analysis/auc61_kappa.py`, `auc61_boot.py`, `auc61_report.py` |
| §6.2, App. B | error transitions and CE→S panels | `analysis/err62_main.py`, `errfig_plot.py` |
| §6.3, App. B | threshold re-selection on 500 validation resamples and panels | `analysis/err63_main.py`, `thrfig_plot.py` |
| App. B | Gemma FP32 evaluation | `experiments/commands/gemma_fp32.sh`, `analysis/err62_gemma.py`, `err63_gemma.py` |
| App. B | ESCI objective pilot: λ ∈ {0.3, 1, 3}, α ∈ {5, 10, 20}, CE+label-margin BCE | `experiments/commands/pilot245.sh`, `analysis/esci245_analysis.py` |

## 6. Order of an analysis

1. Train and evaluate (`experiments/commands/*.sh`): each evaluation writes one gzip JSON per (checkpoint, instruction, set) with the four log-probabilities of the label and the terminator for both labels; the margin is m = [log P(Relevant) + log P(terminator)] − [log P(Irrelevant) + log P(terminator)], decision m > t.
2. Threshold: `analysis/esci204_common.py::best_threshold` — hard-κ maximum over all validation pairs of the model's own scores; candidates are 0, the midpoints of consecutive distinct scores, and below/above all scores; ties → t = 0, then smaller |t|, then smaller t. One threshold per model, instruction and seed; Base thresholds are never transferred.
3. Tables and statistics: the `*_main.py` / `*_analysis.py` scripts of §5, then `auc61_*`, `err62_*`, `err63_*` (they read the score files by file name from `cpmr_experiment/**`, so the original run layout is not needed). Aggregation: κ per instruction and seed → equal-weight mean over the instruction subset → mean over seeds (Base: one score).
4. Panels: `errfig_plot.py`, `thrfig_plot.py` (one vector file per panel; the LaTeX layout is not part of this repository).

```bash
python analysis/auc61_kappa.py --root . --mods analysis --out cpmr_experiment/auc61/out
python analysis/err62_main.py --root . --mods analysis --kappa cpmr_experiment/auc61/out/kappa.pkl --out cpmr_experiment/err62/out
python analysis/err63_main.py --root . --mods analysis --kappa cpmr_experiment/auc61/out/kappa.pkl --out cpmr_experiment/err63/out
```

## 7. Reproduction scope and caveats

* **Own-instruction evaluation.** Every SFT model is evaluated with the instruction it was trained with; Base uses the same text. Seed means average instruction-level κ with equal weights.
* **PO subset.** APO-CF results exist for T1–T3 of ESCI with Qwen3-4B and Llama-3.1-8B-Instruct (not for Gemma, not for MS MARCO). The selected instructions are shipped (`cpmr_experiment/esci_t/p1_T*/esci_apocf_manifest.json`, `cpmr_experiment/esci_llama/p1_T*/`) together with the proposer prompts and replies (`p1_exchange.json`). The proposer samples at temperature 1.0, so **re-running the PO stage does not reproduce the same candidates**; the frozen selections are the reference and the commands that regenerate the stage are commented out in the generated scripts. The meta-prompts shipped in `src/apocf_prompt.py` were checked to rebuild all 18 recorded meta-prompts (T1–T3, three variants, both models) byte for byte.
* **Precision.** Forward passes use BF16 blocks with FP32 head / log-softmax scoring. The Gemma FP32 appendix re-evaluates 12 representative checkpoints with FP32 weights; BF16/FP32 differences are reported in the paper, not hidden.
* **Determinism.** Training uses fixed seeds, adapter initialization and example order (hashes recorded in the logs), but GPU kernels and dropout streams are not bit-deterministic; re-training gives scores that differ in the last digits, not the stored ones. Statistics on stored scores (bootstraps, thresholds) are deterministic.
* **Checked without a GPU** (`checks/run_checks.py`, and restore on a CPU machine): imports/syntax, configs, manifest schemas and checksums, the tie rule of the threshold search, κ/AUC metrics on synthetic data, the command generator, path handling. **Not part of the checks:** an end-to-end training run from this repository on a clean machine (the GPU runs of the paper used the original working tree; the restored inputs and prompt builder were verified against it byte for byte, see `CHANGES_FROM_ORIGINAL.md`).
* **Compute.** The 5-run objective pilot took about 3.9 GPU-hours (training 3.0, evaluation 0.6, checks 0.2); the full study consists of several hundred training runs.

## 8. Sources and licenses of data and models

See `DATA_LICENSES.md`. ESCI (Amazon Shopping Queries Dataset, Apache-2.0), MS MARCO and TREC Deep Learning (research use terms of the collection owners) and the three models are **not** redistributed here and keep their own licenses and access conditions.

The license of the code in this repository has **not been chosen yet**; until a `LICENSE` file is added, all rights are reserved by the author.
