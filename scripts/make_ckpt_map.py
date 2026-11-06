"""Write the checkpoint map that the ESCI evaluation scripts read (--ckpt_map): {"<tag>_step<N>": "<path to ckpt_step<N>.pt>"} for every adapter found under the given training roots.
    python scripts/make_ckpt_map.py --train-root cpmr_experiment/esci_t/train [--train-root ...] --out cpmr_experiment/esci_t/ckpt_map.json"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--train-root", action="append", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
m = {}
for r in a.train_root:
    for p in sorted((ROOT / r).glob("checkpoints/*/ckpt_step*.pt")):
        step = re.fullmatch(r"ckpt_step(\d+)\.pt", p.name)
        if step and int(step.group(1)) > 0:
            m[f"{p.parent.name}_step{step.group(1)}"] = str(p.relative_to(ROOT)).replace("\\", "/")
(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
(ROOT / a.out).write_text(json.dumps(m, indent=1), encoding="utf-8")
print(len(m), "checkpoints ->", a.out)
