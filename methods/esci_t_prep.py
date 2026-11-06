"""ESCI-T1..T7 preparation (experiment log 162; CPU only, no model score): frozen ESCI-T manifest (full text + common suffix, tokens, sha256), identity check of the frozen 159/161 row files, and an exact prompt-length check of every T on every row
(user turn = '<instruction>\\n\\nQuery: <query>\\n<product block>' through the official chat template with enable_thinking=False, + the longer answer 'Irrelevant' + <|im_end|>) against the frozen max_length 766.
    python methods/esci_t_prep.py cpmr_experiment/esci_t"""
import hashlib
import os
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "cpmr_experiment/esci/audit"
OUT = ROOT / sys.argv[1]
SUFFIX = '\n\nOutput exactly one label: "Relevant" or "Irrelevant".'
T = {
    "T1": "Determine whether the product is an exact match for the query, using the provided product information. Label it Relevant only if it matches the requested product and meets all requirements stated in the query. Label substitutes, complementary products, and unrelated products Irrelevant.",
    "T2": "Compare the product information with what the query asks for. A product is Relevant when it is the requested product and satisfies every stated requirement. A substitute, an item intended to accompany the requested product, or an unrelated item is Irrelevant.",
    "T3": "Decide whether the product fulfills the query exactly. Use Relevant for the requested product when all specified requirements are met. Use Irrelevant for substitutes, complementary items, unrelated items, and products that fail a specified requirement. Base the decision on the supplied product information.",
    "T4": "Assess the match between the query and the product described. To be Relevant, the product must match what is requested and meet all explicitly stated requirements. Products offered as alternatives or complements, as well as unrelated products, are Irrelevant.",
    "T5": "Using the product details, check whether this is the product sought by the query with all requested requirements satisfied. If so, classify it as Relevant. If it is a substitute, a complementary product, an unrelated product, or fails a requested requirement, classify it as Irrelevant.",
    "T6": "For this query–product pair, assign Relevant only to an exact match: the requested product satisfying all requirements expressed in the query. Assign Irrelevant to substitutes, complementary products, unrelated products, and matches that do not satisfy those requirements. Judge from the provided product information.",
    "T7": "Evaluate whether the supplied product information describes an exact match to the user's query. The requested product meeting every stated requirement is Relevant. A substitute, a product used alongside it, an unrelated product, or a product missing a required attribute is Irrelevant.",
}
sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
tok = AutoTokenizer.from_pretrained(os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-4B"))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "audit").mkdir(exist_ok=True)
lines = []


def P(s=""):
    lines.append(s); print(s, flush=True)
FROZEN = {"diag_rows.jsonl": "e9463d739ba244f0a4348569a48e235135049ac04bea264ebdac0d116556d665", "sel_rows.jsonl": "20a3f71fad9089487acf2e7f0aed6a21376620dfc90421920584fe5863c2f597",
          "sft2_rows.jsonl": "00d35b223469c5689b93586725c71cd52dc2debea1954b1a4611610d388497a9", "test300_rows.jsonl": "a6f83c1c9e2431eefb250145b3acaafcc35c989caea1d4815df0d7b25f0672a1"}
for f, h in FROZEN.items():
    got = hashlib.sha256((A / f).read_bytes()).hexdigest()
    P(f"frozen rows {f}: sha256 {got[:16]}.. {'== 159/161 hash' if got == h else 'DIFFERS from 159/161 hash ' + h[:16]}")
    assert got == h
entries = []
for k, v in T.items():
    full = v + SUFFIX
    n = len(tok(full, add_special_tokens=False)["input_ids"])
    entries.append({"display_id": k, "source_id": f"ESCI-{k}", "full": full, "sha256_full": sha(full), "tokens_with_suffix": n})
    P(f"ESCI-{k}: {n} tokens with suffix, sha256 {sha(full)[:16]}..")
(OUT / "esci_t_manifest.json").write_text(json.dumps({"note": "ESCI-T1..T7 of experiment log 162, frozen before any score; common suffix after a blank line", "entries": entries}, indent=1, ensure_ascii=False), encoding="utf-8")
max_len = json.load(open(A / "splits.json", encoding="utf-8"))["max_length"]
ans = tok("Irrelevant", add_special_tokens=False)["input_ids"]
P(f"\nframe max_length {max_len}; answer 'Irrelevant' = {len(ans)} tokens + <|im_end|>")
rows = {f: [json.loads(l) for l in open(A / f, encoding="utf-8")] for f in FROZEN}
worst = {}
for k, e in ((e["display_id"], e) for e in entries):
    mx = 0; over = 0; n = 0
    for f, rr in rows.items():
        for r in rr:
            t = tok.apply_chat_template([{"role": "user", "content": f"{e['full']}\n\nQuery: {r['query']}\n{r['product_text']}"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            L = len(tok(t, add_special_tokens=False)["input_ids"]) + len(ans) + 1
            mx = max(mx, L); over += L > max_len; n += 1
    worst[k] = (mx, over, n)
    P(f"{k}: longest prompt+answer {mx} tokens over {n} rows; rows above {max_len}: {over}")
(OUT / "audit" / "esci_t_prep.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
