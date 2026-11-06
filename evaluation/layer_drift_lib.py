"""Shared pieces for the layer-wise drift diagnostic on the Single-CE-K2 checkpoints (experiment log 090).

Measure A (weights, CPU):   per transformer block, ||dW||_F / ||W_base||_F for the learned LoRA update.
Measure B (representation): same inputs with the adapter ON and OFF (eval mode, no grad), the relative L2 and
                            cosine distance of the residual stream after every block at two token positions.

Conventions fixed here, written down because a wrong one makes a fake layer pattern:

* LoRA effective update dW = (lora_alpha / r) * B @ A. PEFT's default scaling (use_rslora off). Its Frobenius
  norm is computed exactly from the two r x r Gram matrices, ||BA||_F^2 = tr((B^T B)(A A^T)), in float64, so the
  (out x in) matrix is never built. A and B norms are NOT used to compare layers.
* "layer 0" is the embedding output; "layer 1..L" are the OUTPUTS OF THE TRANSFORMER BLOCKS taken with a forward
  hook on each decoder layer, i.e. BEFORE the final RMSNorm. HF's output_hidden_states is not used: its last entry
  is after the final norm, which would mix two different kinds of points.
* Two positions, both inside the prefix only (no gold label, no EOS is ever fed):
    instruction : the token that contains the last character of the instruction text, located with tokenizer
                  offsets. It precedes the query and the passage, so under causal attention it cannot depend on
                  them; it is therefore (nearly) the same for every input and must not be counted as many
                  independent observations.
    answer      : the last prefix token, the one just before the model would write its answer.
* Per query, the two documents are averaged first, so every query has equal weight.
"""
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path

import torch

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
ADAPTER_KEY = "base_model.model.model.layers.{layer}.self_attn.{proj}.lora_{ab}.default.weight"
BASE_KEY = "model.layers.{layer}.self_attn.{proj}.weight"


# ------------------------------------------------------------------ measure A
def lora_scaling(cfg):
    return float(cfg["lora"]["alpha"]) / float(cfg["lora"]["r"])


def delta_fro(A, B, scaling):
    """||scaling * B @ A||_F, exact, float64, without building the matrix. A: (r, in), B: (out, r)."""
    A64, B64 = A.detach().double(), B.detach().double()
    gram = (B64.T @ B64) @ (A64 @ A64.T)  # (r, r)
    sq = float(torch.trace(gram))
    return scaling * math.sqrt(max(sq, 0.0))


def weight_rows(adapter_state, base_fro, scaling, n_layers):
    """One row per (layer, projection). base_fro[(layer, proj)] is ||W_base||_F."""
    rows = []
    for layer in range(n_layers):
        for proj in PROJ:
            a = adapter_state[ADAPTER_KEY.format(layer=layer, proj=proj, ab="A")]
            b = adapter_state[ADAPTER_KEY.format(layer=layer, proj=proj, ab="B")]
            d = delta_fro(a, b, scaling)
            w = base_fro[(layer, proj)]
            rows.append({"layer": layer + 1, "proj": proj, "dW_fro": d, "W_fro": w, "rel": d / w})
    return rows


def layer_summary(rows):
    """sqrt(sum_module ||dW||^2) / sqrt(sum_module ||W||^2), fixed by the request, per layer."""
    out = {}
    for layer in sorted({r["layer"] for r in rows}):
        rs = [r for r in rows if r["layer"] == layer]
        d = math.sqrt(sum(r["dW_fro"] ** 2 for r in rs))
        w = math.sqrt(sum(r["W_fro"] ** 2 for r in rs))
        out[layer] = {"dW_layer": d, "W_layer": w, "rel_layer": d / w}
    return out


def base_weight_norms(model_name, n_layers, local_files_only=True):
    """||W||_F of every q/k/v/o projection, read tensor by tensor from the safetensors shards (no model is
    instantiated), float32 -> float64 accumulation. Returns (dict, snapshot_dir)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    snap = Path(snapshot_download(model_name, local_files_only=local_files_only,
                                  allow_patterns=["*.safetensors", "*.json"]))
    want = {BASE_KEY.format(layer=l, proj=p): (l, p) for l in range(n_layers) for p in PROJ}
    norms = {}
    for shard in sorted(snap.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                if k in want:
                    norms[want[k]] = float(f.get_tensor(k).float().double().pow(2).sum().sqrt())
    missing = [v for v in want.values() if v not in norms]
    if missing:
        raise RuntimeError(f"{len(missing)} base projections not found in {snap}, e.g. {missing[0]}")
    return norms, snap


def sha256_head(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


# ------------------------------------------------------------------ measure B
def normalise_query(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def find_positions(tokenizer, instruction, query, passage_text, max_length=384, enable_thinking=False):
    """Prefix-only token ids and the two positions. passage_text is exactly what the builder receives
    ("Passage: ..." or the cached preserved passage). Uses the project's own builder for the ids so training,
    evaluation and this probe see byte-identical prefixes."""
    from src.data_mistral import build_example_mistral
    ex = build_example_mistral(tokenizer, instruction, query, passage_text, "Relevant", max_length,
                               enable_thinking=enable_thinking)
    first = next(i for i, t in enumerate(ex.labels) if t != -100)
    prefix_ids = ex.input_ids[:first]  # no label tokens, no EOS

    # locate the instruction inside the rendered template, on the UNTRUNCATED text (truncation only changes
    # tokens after the instruction)
    user_content = f"{instruction}\n\nQuery: {query}\n{passage_text}"
    kw = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kw["enable_thinking"] = enable_thinking
    text = tokenizer.apply_chat_template([{"role": "user", "content": user_content}], **kw)
    char_end = text.index(instruction) + len(instruction)  # exclusive
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    last_char = char_end - 1  # index of the instruction's last character
    pos_instr = next(i for i, (s, e) in enumerate(enc["offset_mapping"]) if s <= last_char < e)
    tok_s, tok_e = enc["offset_mapping"][pos_instr]
    if enc["input_ids"][:pos_instr + 1] != prefix_ids[:pos_instr + 1]:
        raise RuntimeError("the instruction tokens of the rendered text differ from the builder's prefix")
    query_char = text.index("Query:", char_end)
    return {
        "input_ids": prefix_ids,
        "pos_instr": pos_instr,
        "pos_answer": len(prefix_ids) - 1,
        "instr_token_text": tokenizer.decode([prefix_ids[pos_instr]]),
        "instr_token_spills_past_instruction": bool(tok_e > char_end),
        "instr_token_precedes_query": bool(tok_e <= query_char),
        "n_prefix_tokens": len(prefix_ids),
        "next_token_is_label": bool(ex.labels[first] != -100),
    }


def locate_blocks(model):
    """(embed_tokens module, list of decoder blocks, final norm, lm_head) for a PEFT-wrapped or plain causal LM."""
    inner = model
    if hasattr(inner, "base_model") and hasattr(inner.base_model, "model"):
        inner = inner.base_model.model
    core = inner.model
    return core.embed_tokens, list(core.layers), core.norm, inner.lm_head


@torch.no_grad()
def collect_hidden(model, input_ids, positions, device):
    """Residual stream at the given positions: tensor (n_positions, L+1, d) float32 on CPU. Index 0 is the
    embedding output, 1..L are block outputs (before the final norm). Batch size 1: no padding anywhere."""
    embed, blocks, _, _ = locate_blocks(model)
    store = {}

    def grab(idx):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            store[idx] = h[0, positions].float().cpu()
        return hook

    handles = [embed.register_forward_hook(grab(0))] + [
        blk.register_forward_hook(grab(i + 1)) for i, blk in enumerate(blocks)]
    try:
        ids = torch.tensor([input_ids], device=device)
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return torch.stack([store[i] for i in range(len(blocks) + 1)], dim=1)


def rel_l2(h_on, h_off, eps=1e-6):
    num = (h_on.double() - h_off.double()).norm(dim=-1)
    den = h_off.double().norm(dim=-1).clamp_min(eps)
    return num / den


def cos_dist(h_on, h_off, eps=1e-12):
    a, b = h_on.double(), h_off.double()
    cos = (a * b).sum(-1) / (a.norm(dim=-1).clamp_min(eps) * b.norm(dim=-1).clamp_min(eps))
    return 1.0 - cos


# ------------------------------------------------------------------ samples
def sample_train_queries(query_ids, n, tag="layer-drift-train-sample-v1"):
    """n query ids, without replacement, fixed by a string seed (not by the checkpoint seed)."""
    return random.Random(tag).sample(sorted(query_ids), n)


def select_trec(rows, train_query_texts, train_query_ids=()):
    """rows: dicts with query_id, query, doc_id, doc_text, relevance (0..3 or None). For every query not seen in
    training (query id or normalised text), the first doc (sorted by doc_id) with relevance >= 2 and the first
    with relevance == 0. relevance 1 and None are never turned into 0. Returns (picked, report)."""
    seen_text = {normalise_query(t) for t in train_query_texts}
    seen_id = set(map(str, train_query_ids))
    by_q = {}
    for r in rows:
        by_q.setdefault(str(r["query_id"]), []).append(r)
    picked, excluded_overlap, no_pair = [], [], []
    for qid in sorted(by_q, key=lambda x: (len(x), x)):
        rs = by_q[qid]
        q = rs[0]["query"]
        if qid in seen_id or normalise_query(q) in seen_text:
            excluded_overlap.append(qid)
            continue
        pos = sorted((r for r in rs if r["relevance"] is not None and r["relevance"] >= 2), key=lambda r: str(r["doc_id"]))
        neg = sorted((r for r in rs if r["relevance"] == 0), key=lambda r: str(r["doc_id"]))
        if not pos or not neg:
            no_pair.append(qid)
            continue
        picked.append({"query_id": qid, "query": q, "pos": pos[0], "neg": neg[0]})
    return picked, {"n_queries_in_pool": len(by_q), "n_excluded_train_overlap": len(excluded_overlap),
                    "excluded_overlap_ids": excluded_overlap, "n_without_pair": len(no_pair),
                    "without_pair_ids": no_pair, "n_used": len(picked)}


def per_query_mean(values_by_doc):
    """values_by_doc: list of tensors (one per document of the SAME query) -> their mean."""
    return torch.stack(list(values_by_doc)).mean(0)


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
