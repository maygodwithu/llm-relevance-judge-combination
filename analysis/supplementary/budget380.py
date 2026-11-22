"""Freeze the compute budget N of the long CE runs BEFORE any evaluation of them.
N = round(625 x median(soft kappa training elapsed) / median(CE training elapsed)), both from the stored Qwen3-4B ESCI T1 lr 2e-5 / 625-update runs (seeds 0-2), `train_seconds` of stats_*.json
(training loop only, checkpoint saving included, evaluation excluded; same definition in both trainers). Writes the manifest with the SHA-256 of every input file.
    python budget380.py --root <project root> --out <dir>"""
import argparse, hashlib, json, statistics as st
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--out", required=True); a = ap.parse_args()
R = Path(a.root) / "cpmr_experiment"
ce = [R / f"esci_t/train/logs/stats_E4_T1_s{s}.json" for s in (0, 1, 2)]
so = [R / f"esci_mix/train/logs/stats_EQ_mix_T1_s{s}.json" for s in (0, 1, 2)]
rd = lambda p: json.load(open(p))
tc, ts = [rd(p)["train_seconds"] for p in ce], [rd(p)["train_seconds"] for p in so]
for p in ce + so:
    d = rd(p); assert d["save_steps"] == [63, 157, 313, 469, 625] and d.get("lr") is None and d["warmup_updates"] == 63 and d["max_length"] == 766 and d["micro_batch"] == 4 and d["inputs_shortened"] == 0, p
mc, ms = st.median(tc), st.median(ts)
N = round(625 * ms / mc)
man = dict(rule="N = round(625 x median(soft train_seconds) / median(CE train_seconds)); frozen before any evaluation of the long CE runs",
           ce_runs=[dict(tag=f"E4_T1_s{s}", train_seconds=t, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for s, t, p in zip((0, 1, 2), tc, ce)],
           soft_runs=[dict(tag=f"EQ_mix_T1_s{s}", train_seconds=t, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for s, t, p in zip((0, 1, 2), ts, so)],
           median_ce_s=mc, median_soft_s=ms, ratio=ms / mc, N_updates=N, warmup_updates=-(-N // 10), rows_per_update=16, rows_exposed=N * 16, exposures_per_row=N * 16 / 10000,
           comparability_notes=["same definition: training loop, 5 checkpoint saves (63/157/313/469/625) in both, evaluation excluded",
                                "all six runs: one job per GPU, three jobs concurrent on three different GPUs of the same type (scheduler logs); no log shows a difference in this condition",
                                "the new long CE runs save only step 0 and the final checkpoint (2 saves instead of 6): about 2 s per skipped save, recorded, not corrected"])
Path(a.out).mkdir(parents=True, exist_ok=True)
(Path(a.out) / "budget_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
print("N =", N, "| median CE", round(mc, 2), "soft", round(ms, 2), "ratio", round(ms / mc, 4))
