"""Inventory of the stored raw scores for the 6.1 AUC analysis (experiment log 229): per evaluation set, condition and instruction the number of seeds with a stored score file (and the pair count), plus consistency of row keys / labels across all conditions of a set.   python auc61_inventory.py --root . --mods analysis"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import auc61_data as D
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--mods", required=True); ap.add_argument("--out", default="")
a = ap.parse_args()
D.init(a.root, a.mods)
inv = {}
for sid in D.SETS:
    ref = None
    print(f"## {sid}")
    for cond in D.CONDS:
        row = []
        for T in D.TS:
            n = 0
            for s in D.seeds_of(cond):
                r = D.get(sid, cond, T, s)
                if r is None:
                    continue
                n += 1
                ks, m, y, path = r
                if ref is None:
                    ref = (ks, y)
                assert ks == ref[0] and (y == ref[1]).all(), (sid, cond, T, s, "row keys / labels differ")
                assert (m == m).all() and abs(m).max() < 1e9, (sid, cond, T, s, "non-finite")
            row.append(n)
            inv[f"{sid}|{cond}|T{T}"] = n
        print(f"  {cond:7s} seeds per T1..T7: {row}  (expected {len(D.seeds_of(cond))})")
    print(f"  rows {len(ref[0])}, positives {int(ref[1].sum())}, queries {len({q for q, _ in ref[0]})}")
if a.out:
    json.dump(inv, open(a.out, "w"), indent=1)
