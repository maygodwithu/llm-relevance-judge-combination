# Independent calibration, ESCI Llama-3.1-8B-Instruct, T1–T7

Outputs of `analysis/supplementary/analyze410.py` on the scores of the paper (see `docs/ADDITIONAL_ANALYSES.md`).

| file | content |
|---|---|
| `calib410_results.md` / `.json` | overall table, PO − original per threshold regime, change of the PO gain (independent − shared), changed-text instructions only, per-instruction table |
| `calib410_per_run.csv` | one row per instruction × {original, PO} × {Base, CE+soft κ seed 0–2}: thresholds from the shared validation split and from the independent pool, κ(0), κ(shared t*), κ(independent t*), AUC, predicted-Relevant rates, and the file name and SHA-256 of every score file used |
| `per_instruction_summary.csv`, `per_seed_T_mean.csv`, `overall_summary.csv` | mean ± SD over seeds per instruction; per-seed means over T1–T7; the overall summary |
| `po_minus_original_per_instruction.csv` | PO − original per instruction, seed-wise differences and unadjusted paired test-query bootstrap intervals |

SFT values are the mean over T1–T7 within each seed, then the mean ± sample SD over three seeds; Base is the mean over the instructions. T1, T4 and T6 kept the original text under PO, so their PO − original difference is exactly 0. Bootstrap: 10,000 paired test-query draws, one draw set for all conditions, models and thresholds fixed (exploratory 95% intervals).
