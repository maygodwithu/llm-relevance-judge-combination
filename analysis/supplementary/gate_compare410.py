"""Stage-0 gate (pre-registered, criterion max |margin difference| <= 1e-4, no relaxation): the same evaluator re-scores the first 20 stored selection queries (593 rows) for five jobs;
the margins must equal the stored selection margins (stored SFT / Base score files; for the selected-PO-text Base job the PO-stage candidate raw margins). Writes gate410.json, exits 1 if any job fails.
    python gate_compare410.py   (from the project root)"""
import glob, gzip, json, os, sys
import numpy as np

R = "cpmr_experiment"
G = f"{R}/esci410/gate"


def load(p):
    return json.load(gzip.open(p, "rt", encoding="utf-8"))


def m_eval(path):
    raw = load(path)["raw"]
    return {(q, d): v[0] + v[1] - v[3] - v[4] for q in raw for d, v in raw[q].items()}


def m_cand(path):
    mm = load(path)["margins"]
    return {(q, d): float(v[0] + v[1] - v[2] - v[3]) for q in mm for d, v in mm[q].items()}


def stored(name):
    fs = sorted(glob.glob(f"{R}/**/{name}", recursive=True))
    assert fs, name
    return fs[0]


# (gate file, reference loader, reference path)
JOBS = [
    ("esciLlamaEval_Base__T5__gate.json.gz", m_eval, stored("esciLlamaEval_Base__T5__sel.json.gz")),
    ("esciLlamaEval_Base__T5_APOCF__gate.json.gz", m_cand, f"{R}/esci_llama/p1_T5/p1_raw_cand1.json.gz"),
    ("esciLlamaEval_ELQ_mix_T5_APOCF_s0_step625__T5_APOCF__gate.json.gz", m_eval, stored("esciLlamaEval_ELQ_mix_T5_APOCF_s0_step625__T5_APOCF__sel.json.gz")),
    ("esciLlamaEval_ELQ_mix_T3_APOCF_s0_step625__T3_APOCF__gate.json.gz", m_eval, stored("esciLlamaEval_ELQ_mix_T3_APOCF_s0_step625__T3_APOCF__sel.json.gz")),
    ("esciLlamaEval_ELQ_mix_T7_s0_step625__T7__gate.json.gz", m_eval, stored("esciLlamaEval_ELQ_mix_T7_s0_step625__T7__sel.json.gz")),
]
out = []
for gf, fn, ref in JOBS:
    a = m_eval(f"{G}/{gf}"); b = fn(ref)
    keys = list(a); miss = [k for k in keys if k not in b]
    d = np.array([abs(a[k] - b[k]) for k in keys if k in b])
    out.append(dict(job=gf.replace("esciLlamaEval_", "").replace(".json.gz", ""), reference=os.path.relpath(ref, R).replace("\\", "/"), n=len(keys), missing_in_reference=len(miss), max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()), identical=int((d == 0).sum()),
                    decision_disagreements=int(sum((a[k] > 0) != (b[k] > 0) for k in keys if k in b)), passed=bool(not miss and len(keys) == 593 and d.max() <= 1e-4), criterion=1e-4))
json.dump(out, open(f"{G}/gate410.json", "w"), indent=1); print(json.dumps(out, indent=1))
sys.exit(0 if all(o["passed"] for o in out) else 1)
