"""Instruction-only LoRA (experiment log 166): the LoRA branch is switched on only at the instruction token positions of the prompt, off everywhere else.
    y[b,t] = W x[b,t] + m[t] * scale * B(A(dropout(x[b,t])))        m[t] = 1 inside the instruction span, else 0
Implemented as a forward hook on every lora_B module (peft adds the scaled B(A(x)) to the base output, so scaling the lora_B output by m is exactly the formula; the base branch is untouched,
no no_grad / detach is used, so the answer loss still reaches the LoRA parameters through attention to the instruction K/V). The span is a constant [a, b) of token positions because the instruction
is the first thing after the three chat-header tokens and is always followed by the same text ("\n\nQuery: "); it is verified on every built example. A constant range is also stable under
gradient-checkpointing recomputation and right padding.
"""
import torch

HEAD = "<|im_start|>user\n"
STATE = {"range": None, "mask": None, "prefix_mask": None, "scale": None, "capture": None, "lora_scale": 2.0}      # None = LoRA on everywhere (ordinary LoRA); (a, b) = on only in [a, b); (0, 0) = off everywhere


def instruction_range(tok, builder_example_ids, instruction):
    """token positions [a, b) overlapping the instruction characters in the rendered prompt; returns (a, b, ids[a:b])."""
    first = len(builder_example_ids)
    text = tok.decode(builder_example_ids)
    assert text.startswith(HEAD) and text[len(HEAD):].startswith(instruction), "instruction is not at the expected place"
    c0, c1 = len(HEAD), len(HEAD) + len(instruction)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == list(builder_example_ids), "re-tokenisation differs from the built ids"
    on = [k for k, (s, e) in enumerate(enc["offset_mapping"]) if s < c1 and e > c0]
    assert on == list(range(on[0], on[-1] + 1)), "instruction tokens are not contiguous"
    return on[0], on[-1] + 1, list(builder_example_ids[on[0]:on[-1] + 1]), first


def _mask_hook(module, args, out):
    pm = STATE["prefix_mask"]
    if pm is not None:                            # 1-D prompt mask shared by every row of the batch (rows differ only after the prompt); prompt positions beyond it (label / EOS) OFF
        T = out.shape[-2]
        m = torch.zeros(T, 1, dtype=out.dtype, device=out.device)
        if T == 1 and len(pm) != 1:               # cached decoding step: a new answer position stays OFF
            return out * m
        if T < len(pm):
            raise RuntimeError(f"prompt mask of {len(pm)} positions does not fit a forward of {T} positions")
        m[:len(pm), 0] = torch.tensor(pm, dtype=out.dtype, device=out.device)
        return out * m
    if STATE["mask"] is not None:                 # per-example mask [B, T] (instruction + chat template mode)
        return out * STATE["mask"].to(device=out.device, dtype=out.dtype).unsqueeze(-1)
    r = STATE["range"]
    if r is None:
        return out
    a, b = r
    m = torch.zeros(out.shape[-2], 1, dtype=out.dtype, device=out.device)
    m[a:min(b, out.shape[-2])] = 1
    return out * m


NAMES = {}     # id(lora_B module) -> module name, filled by install()


def _hook(module, args, out):
    """position masks (if any) and the optional external scalar STATE['scale'] c (LoRA ramp): the applied branch is c * (alpha / r) * B A x, c is not learned."""
    out = _mask_hook(module, args, out)
    cap = STATE["capture"]
    sc = STATE["scale"]
    if sc is not None:
        out = out * sc
    if cap is not None:         # RMS of the raw branch (alpha/r) B A x and of the applied branch c (alpha/r) B A x, padding-free (batch 1)
        n = NAMES[id(module)]
        raw = out.detach().float() * STATE["lora_scale"] / (1.0 if sc is None else sc)
        e = cap.setdefault(n, [0.0, 0.0, 0])
        e[0] += float(raw.pow(2).sum())
        e[1] += float((raw * (1.0 if sc is None else sc)).pow(2).sum())
        e[2] += raw.numel()
    return out


def install(model):
    """hooks on every lora_B; behaviour is controlled by STATE['range']. Returns the handles."""
    hs = []
    for name, mod in model.named_modules():
        if name.endswith(".lora_B.default"):
            NAMES[id(mod)] = name[:-len(".lora_B.default")]
            hs.append(mod.register_forward_hook(_hook))
    if not hs:
        raise RuntimeError("no lora_B modules found")
    return hs


EOS = 151645          # <|im_end|>; the first one in a built prompt starts the 9-token chat suffix (im_end, newline, im_start, assistant, newline, think, blank line, /think, blank line)
SUFFIX_LEN = 9


def template_mask(ids_row, a, b):
    """ON: chat header + instruction [0, b) and the chat suffix (9 tokens from the first <|im_end|>, which ends with the position that predicts the first answer token).
    OFF: field markers, query, passage, and the answer label + EOS."""
    ids = list(ids_row)
    s = ids.index(EOS)
    m = [0.0] * len(ids)
    for k in range(0, b):
        m[k] = 1.0
    for k in range(s, min(len(ids), s + SUFFIX_LEN)):
        m[k] = 1.0
    return m


def install_template_mode(model, a, b):
    """pre-hook on the whole model: reads input_ids, builds the per-example mask used by the lora_B hooks (instruction + chat template mode). Right padding is safe (the first EOS is searched)."""
    def pre(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        STATE["mask"] = torch.tensor([template_mask(r.tolist(), a, b) for r in ids], dtype=torch.float32)
    return model.register_forward_pre_hook(pre, with_kwargs=True)


WORD_PHRASE = "A passage is relevant even if"       # second sentence of T1 (unique in T1); the ON word is one whole word of it (default: the lowercase 'relevant')
WORD = "relevant"


def word_range(tok, builder_example_ids, instruction, phrase=WORD_PHRASE, word=WORD):
    """token positions [a, b) overlapping the characters of `word` (one whole word, matched at word boundaries) inside the unique `phrase` of the instruction in the rendered prompt;
    returns (a, b, ids[a:b], char span). Matching is on characters of the rendered prompt, never on token ids."""
    import re
    assert instruction.count(phrase) == 1, "the phrase must occur exactly once in the instruction"
    hits = [m.start() for m in re.finditer(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])", phrase)]
    assert len(hits) == 1, f"the word {word!r} must occur exactly once as a whole word in the phrase (found {len(hits)})"
    text = tok.decode(builder_example_ids)
    assert text.startswith(HEAD) and text[len(HEAD):].startswith(instruction), "instruction is not at the expected place"
    c0 = len(HEAD) + instruction.index(phrase) + hits[0]
    c1 = c0 + len(word)
    assert text[c0:c1] == word
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == list(builder_example_ids), "re-tokenisation differs from the built ids"
    on = [k for k, (s, e) in enumerate(enc["offset_mapping"]) if s < c1 and e > c0]
    assert on == list(range(on[0], on[-1] + 1)), "word tokens are not contiguous"
    return on[0], on[-1] + 1, list(builder_example_ids[on[0]:on[-1] + 1]), (c0, c1)
