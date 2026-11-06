"""ESCI-T Llama preparation (experiment log 166; CPU only, no model score): Llama-3.1-8B-Instruct tokenizer / chat-template audit of the frozen 159/161/165 rows (the product strings were cached with the Qwen tokenizer and are passed unchanged),
answer-token and terminator check, and the common max_length: the longest empty-instruction prompt over all frozen rows + the reserved instruction space (128 body tokens + suffix) + answer + <|eot_id|> + margin. Also the exact longest prompt of each of the seven ESCI-T texts.
    python methods/esci_llama_prep.py cpmr_experiment/esci_llama"""
import hashlib
import os
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "cpmr_experiment/esci/audit"
OUT = ROOT / sys.argv[1]
MODEL = os.environ.get("LLAMA_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
SUFFIX = '\n\nOutput exactly one label: "Relevant" or "Irrelevant".'
tok = AutoTokenizer.from_pretrained(MODEL)
(OUT / "audit").mkdir(parents=True, exist_ok=True)
lines = []


def P(s=""):
    lines.append(s); print(s, flush=True)
FROZEN = {"diag_rows.jsonl": "e9463d739ba244f0a4348569a48e235135049ac04bea264ebdac0d116556d665", "sel_rows.jsonl": "20a3f71fad9089487acf2e7f0aed6a21376620dfc90421920584fe5863c2f597",
          "sft2_rows.jsonl": "00d35b223469c5689b93586725c71cd52dc2debea1954b1a4611610d388497a9", "test300_rows.jsonl": "a6f83c1c9e2431eefb250145b3acaafcc35c989caea1d4815df0d7b25f0672a1"}
for f, h in FROZEN.items():
    got = hashlib.sha256((A / f).read_bytes()).hexdigest()
    assert got == h, f
    P(f"frozen rows {f}: sha256 {got[:16]}.. == 159/161/165 hash")
man = json.load(open(ROOT / "cpmr_experiment/esci_t/esci_t_manifest.json", encoding="utf-8"))["entries"]
for e in man:
    assert hashlib.sha256(e["full"].encode("utf-8")).hexdigest() == e["sha256_full"]
P("ESCI-T manifest of 163/165: all seven sha256 verified; Llama tokens (with suffix): " + ", ".join(f"{e['display_id']} {len(tok(e['full'], add_special_tokens=False)['input_ids'])}" for e in man))
P(f"Llama tokenizer {MODEL.split('/')[-1][:12]}; Relevant -> {tok('Relevant', add_special_tokens=False)['input_ids']}, Irrelevant -> {tok('Irrelevant', add_special_tokens=False)['input_ids']}, <|eot_id|> id {tok.convert_tokens_to_ids('<|eot_id|>')}, <|end_of_text|> id {tok.convert_tokens_to_ids('<|end_of_text|>')}, pad (finetune_right_pad) id {tok.convert_tokens_to_ids('<|finetune_right_pad_id|>')}")
sample = tok.apply_chat_template([{"role": "user", "content": "INSTR\n\nQuery: Q\nProduct: P"}], tokenize=False, add_generation_prompt=True)
P("chat template rendering of a toy prompt: " + repr(sample))
n_suffix = len(tok(SUFFIX, add_special_tokens=False)["input_ids"])
ans = max(len(tok(x, add_special_tokens=False)["input_ids"]) for x in ("Relevant", "Irrelevant"))
rows = {f: [json.loads(l) for l in open(A / f, encoding="utf-8")] for f in FROZEN}
mx0, mxrow = 0, None
tl = {}
for f, rr in rows.items():
    for r in rr:
        t = tok.apply_chat_template([{"role": "user", "content": f"\n\nQuery: {r['query']}\n{r['product_text']}"}], tokenize=False, add_generation_prompt=True)
        L = len(tok(t, add_special_tokens=False)["input_ids"])
        if L > mx0:
            mx0, mxrow = L, (f, r["example_id"])
P(f"longest prompt with an EMPTY instruction over {sum(len(v) for v in rows.values())} rows: {mx0} tokens ({mxrow}); suffix {n_suffix} tokens; longest answer {ans} tokens + <|eot_id|>")
reserve = 128 + n_suffix + 2
max_length = mx0 + reserve + ans + 1 + 8
P(f"max_length (common, frozen) = {mx0} + reserved instruction space (128 body + {n_suffix} suffix + 2) + answer {ans} + eot 1 + margin 8 = {max_length}")
res = {}
for e in man:
    mx = 0; over = 0
    for f, rr in rows.items():
        for r in rr:
            t = tok.apply_chat_template([{"role": "user", "content": f"{e['full']}\n\nQuery: {r['query']}\n{r['product_text']}"}], tokenize=False, add_generation_prompt=True)
            L = len(tok(t, add_special_tokens=False)["input_ids"]) + ans + 1
            mx = max(mx, L); over += L > max_length
    res[e["display_id"]] = (mx, over)
    P(f"{e['display_id']}: longest prompt+answer {mx} tokens; rows above {max_length}: {over}")
json.dump({"max_length": max_length, "empty_instruction_longest": mx0, "reserve": reserve, "per_T_longest": {k: v[0] for k, v in res.items()}, "model_snapshot": "0e9e39f249a16976918f6564b8830bc894c89659"}, open(OUT / "audit" / "max_length.json", "w"), indent=1)
(OUT / "audit" / "esci_llama_prep.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
