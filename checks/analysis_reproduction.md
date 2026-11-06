# Analysis scripts run from this repository on the original score tree

The scripts in `analysis/` were run from a copy of this repository (CPU only) against the original raw score files and compared with the result files that were delivered when the analyses were first run.

| script | output compared | values compared | differences |
|---|---|---:|---|
| `analysis/auc61_kappa.py` | `kappa.pkl` (630 cells) | byte comparison | identical |
| `analysis/err62_main.py` (6.2 error transitions, T1–T7, three models, two datasets, query bootstrap 10,000) | `err62_results.json` | 112,427 numbers / strings | 0 (only the recorded run time differs) |
| `analysis/err63_main.py` (6.3 threshold re-selection, 500 resamples) | `err63_results.json` | 88,115 numbers / strings | 0 (only the recorded run time differs) |

The raw score files of the fine-tuned models are not part of this repository; they are produced by the commands in `experiments/commands/` (the per-example score files are large and are regenerated rather than distributed). The comparison therefore shows that the analysis code of the repository computes the reported numbers from the original scores; re-running the training yields new scores that differ slightly (non-deterministic GPU kernels), so reproduced tables agree with the paper up to this training noise.
