"""Data access for the chapter 6.1 AUC re-analysis (experiment log 229): reads the EXISTING stored raw scores of the ESCI test300 runs (Qwen3-4B 206, Llama-3.1-8B 210, Gemma-3-4B 223) and of the MS MARCO TREC-DL 2019 / 2020 judged pairs (Qwen 213, Llama 216, Gemma 228) through the loaders of the analyses that produced those reports (esci204_common.Store, esci208_llama_main.LStore, esci217_gemma_main.GStore, msm_t7_analysis.Files, msm_llama_analysis.Files, msm_gemma_analysis.Files), so that the file names, instruction labels, row order and label definitions are exactly those of the reports. No new inference.
Evaluation sets (9): ESCI-Qwen / ESCI-Llama / ESCI-Gemma (test300, 300 queries / 6,078 pairs, E = Relevant) and MS-Qwen / MS-Llama / MS-Gemma x 2019 (42 queries / 1,102 pairs) / 2020 (54 / 1,432) (grades 2/3 = Relevant).
Conditions: BASE, CE, SOFT (CE + 1.0 soft kappa), ANY (CE + 1.0 AnyLoss-like kappa), and for ESCI Qwen / Llama the prompt-optimised (PO, APO-CF) texts POBASE, POCE, POSOFT.   get(set, cond, T, seed) -> (keys, m, y) or None; m = logP(Relevant + terminator) - logP(Irrelevant + terminator).
    set id: 'ESCI-Qwen' | 'ESCI-Llama' | 'ESCI-Gemma' | 'MS-Qwen-2019' | ... | 'MS-Gemma-2020'"""
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = None
SEEDS = (0, 1, 2)
TS = tuple(range(1, 8))
SETS = ["ESCI-Qwen", "ESCI-Llama", "ESCI-Gemma", "MS-Qwen-2019", "MS-Qwen-2020", "MS-Llama-2019", "MS-Llama-2020", "MS-Gemma-2019", "MS-Gemma-2020"]
CONDS = ("BASE", "CE", "SOFT", "ANY", "POBASE", "POCE", "POSOFT")
_cache = {}
_st = {}


def init(root, mods):
    global ROOT
    ROOT = Path(root)
    sys.path.insert(0, str(mods))
    import esci204_common as QC
    import esci208_llama_main as LM
    import esci217_gemma_main as GM
    import msm_gemma_analysis as MG
    import msm_llama_analysis as ML
    import msm_t7_analysis as MQ
    _st["QC"], _st["LM"], _st["GM"], _st["MQ"], _st["ML"], _st["MG"] = QC, LM, GM, MQ, ML, MG
    _st["qs"], _st["ls"], _st["gs"] = QC.Store(root), LM.LStore(root), GM.GStore(root)
    _st["mq"], _st["ml"], _st["mg"] = MQ.Files(root), ML.Files(root), MG.Files(root)


def _escid(sid, cond, T, s):
    """-> (model name, instruction label) of the stored ESCI test300 file, or None when the condition is not defined for this set/T"""
    if sid == "ESCI-Qwen":
        QC = _st["QC"]
        c = {"BASE": "Q0", "CE": "Q1", "SOFT": "Q2", "ANY": "Q3", "POBASE": "P0", "POCE": "P1", "POSOFT": "P2"}[cond]
        return QC.model_name(c, T, s), QC.instr_name(c, T)
    if sid == "ESCI-Llama":
        LM = _st["LM"]
        c = {"BASE": "L0", "CE": "L1", "SOFT": "L2", "ANY": "L4", "POBASE": "L0P", "POCE": "L3", "POSOFT": "L5"}[cond]
        if T in LM.PO_KEPT and c in ("L3", "L5", "L0P"):                        # the PO selection kept the original text: same models / scores as L1 / L2 / L0 (as in the 210 analysis)
            c = {"L3": "L1", "L5": "L2", "L0P": "L0"}[c]
        if c in ("L0", "L0P"):
            return "Base", f"T{T}" + ("_APOCF" if c == "L0P" else "")
        return LM.LCOND[c][0].format(T=T, s=s), LM.LCOND[c][1].format(T=T)
    if sid == "ESCI-Gemma":
        if cond in ("POBASE", "POCE", "POSOFT"):
            return None
        c = {"BASE": "G0", "CE": "G1", "SOFT": "G2", "ANY": "G3"}[cond]
        return ("Base" if c == "G0" else _st["GM"].GCOND[c].format(T=T, s=s)), f"T{T}"
    raise KeyError(sid)


def esci_is_alias(sid, cond, T):
    """True when a PO condition is the original text of the same T (Llama PO kept the original text for T1, T4, T6): same stored scores, counted as difference 0, never as an independent run"""
    return sid == "ESCI-Llama" and cond in ("POBASE", "POCE", "POSOFT") and T in _st["LM"].PO_KEPT


def get(sid, cond, T, s):
    key = (sid, cond, T, s)
    if key in _cache:
        return _cache[key]
    out = None
    if sid.startswith("ESCI"):
        d = _escid(sid, cond, T, s)
        if d is not None:
            st = {"ESCI-Qwen": _st["qs"], "ESCI-Llama": _st["ls"], "ESCI-Gemma": _st["gs"]}[sid]
            r = st.arrays(d[0], d[1], "test300")
            if r is not None:
                out = (r[0], r[1], r[2], st.path(d[0], d[1], "test300"))
    else:
        _, mdl, yr = sid.split("-")
        mod = {"Qwen": _st["MQ"], "Llama": _st["ML"], "Gemma": _st["MG"]}[mdl]
        f = {"Qwen": _st["mq"], "Llama": _st["ml"], "Gemma": _st["mg"]}[mdl]
        pth = {"BASE": "M0", "CE": "M1", "SOFT": "M2", "ANY": "M3"}.get(cond)
        if pth is not None and not (pth == "M0" and s != 0):
            tg = mod.tags(pth, T, s)
            if tg is not None:
                r = f.get(tg, T, yr)
                if r is not None:
                    out = (r[0], r[1], r[2], f.idx.get(f"{ {'Qwen': 'msmEval_', 'Llama': 'msmLlamaEval_', 'Gemma': 'msmGemmaEval_'}[mdl] }{tg}__T{T}__{yr}.json.gz"))
    _cache[key] = out
    return out


def seeds_of(cond):
    return (0,) if cond in ("BASE", "POBASE") else SEEDS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
