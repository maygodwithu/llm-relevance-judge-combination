"""Pieces for the APOCF-online pilot (experiment log 078/079/080). New file: no shared-core file is
edited (docs/WORKING_AGREEMENT.md section 1).

Four independent parts, each testable without a GPU:

  * forward_ce_and_pred      one teacher-forced forward that returns the SAME ce as
                             cpmr_losses.score_label_forward(ce_only=True) plus the model's own
                             prediction read from the logits already computed (no extra forward)
  * OnlineConfusion          TP/FP/FN/TN accumulator with per-step snapshots
  * save_run_state / load_run_state    adapter + optimizer + scheduler + RNG, for branching
  * validate_generated_instruction     the pre-fixed acceptance rules for the one APOCF output

Prediction definition (079 section 2). Both labels begin with a different token ("Re"=693,
"Ir"=48113), so the logits at the position just BEFORE the first supervised token compare the two
labels without having seen either: under causal attention that position cannot depend on the gold
label that follows it, so teacher forcing does not leak the answer into the prediction. This is the
first-token decision only. It is NOT the full-sequence label log-probability the reranking score
uses, it is taken with LoRA dropout on, and it mixes model states across steps. It is therefore an
"online proxy confusion", never a confusion of a fixed checkpoint.
"""
import importlib.util
import json
import random
import re
from pathlib import Path

import torch

LF = chr(10)
LABELS = ("Relevant", "Irrelevant")
DEFAULT_REWRITER_PATH = str(Path(__file__).resolve().parents[1] / "src" / "apocf_prompt.py")


def label_first_token_ids(tokenizer):
    """(relevant_id, irrelevant_id), derived from the tokenizer and required to differ."""
    rel = tokenizer("Relevant", add_special_tokens=False)["input_ids"][0]
    irr = tokenizer("Irrelevant", add_special_tokens=False)["input_ids"][0]
    if rel == irr:
        raise RuntimeError(f"the two labels share a first token ({rel}); the first-token "
                           "prediction is not defined for this tokenizer")
    return rel, irr


def forward_ce_and_pred(model, tokenizer, instruction, query_text, doc_text, gold_label,
                        max_length, device, rel_id, irr_id, enable_thinking=None,
                        build_example_fn=None):
    """Returns (ce, pred, margin, n_label_positions).

    ce is exactly what score_label_forward(..., ce_only=True) returns: the model's own out.loss
    over the supervised span (gold label tokens plus EOS), grad-enabled. pred is "Relevant",
    "Irrelevant", or None on an exact tie. margin is logit[Relevant first token] minus
    logit[Irrelevant first token] at the position before the first supervised token.
    """
    from src.data_mistral import build_example_mistral
    builder = build_example_fn or build_example_mistral
    ex = builder(tokenizer, instruction, query_text, doc_text, gold_label, max_length,
                 enable_thinking=enable_thinking)
    input_ids = torch.tensor([ex.input_ids], device=device)
    labels = torch.tensor([ex.labels], device=device)
    attn = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)

    supervised = [i for i, t in enumerate(ex.labels) if t != -100]
    first = supervised[0]
    if first == 0:
        raise RuntimeError("the first supervised position is 0: nothing precedes it to predict from")
    with torch.no_grad():
        row = out.logits[0, first - 1].float()
        margin = float(row[rel_id] - row[irr_id])
    pred = "Relevant" if margin > 0 else ("Irrelevant" if margin < 0 else None)
    return out.loss, pred, margin, len(supervised)


class OnlineConfusion:
    """Cumulative TP/FP/FN/TN with Relevant as the positive class, plus per-step snapshots.

    FP = gold Irrelevant predicted Relevant; FN = gold Relevant predicted Irrelevant.
    A tie (pred None) is counted in n_excluded and in neither cell. Every example is added once,
    by the caller, after its forward and before any update that uses it.
    """

    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0
        self.n_excluded = 0
        self.steps = []
        self._cur = None

    def begin_step(self, step):
        self._cur = {"step": step, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "n_excluded": 0}

    def add(self, gold, pred):
        if self._cur is None:
            raise RuntimeError("begin_step() was not called")
        if gold not in LABELS:
            raise ValueError(f"unknown gold label {gold!r}")
        if pred is None:
            self.n_excluded += 1
            self._cur["n_excluded"] += 1
            return
        cell = {("Relevant", "Relevant"): "tp", ("Irrelevant", "Relevant"): "fp",
                ("Relevant", "Irrelevant"): "fn", ("Irrelevant", "Irrelevant"): "tn"}[(gold, pred)]
        setattr(self, cell, getattr(self, cell) + 1)
        self._cur[cell] += 1

    def end_step(self):
        self.steps.append(self._cur)
        row, self._cur = self.steps[-1], None
        return row

    @property
    def n_valid(self):
        return self.tp + self.fp + self.fn + self.tn

    def counts(self):
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
                "n_valid": self.n_valid, "n_excluded": self.n_excluded}

    def state_dict(self):
        return {"counts": self.counts(), "steps": self.steps}

    @classmethod
    def from_state_dict(cls, d):
        c = cls()
        c.tp, c.fp, c.fn, c.tn = (d["counts"][k] for k in ("tp", "fp", "fn", "tn"))
        c.n_excluded = d["counts"]["n_excluded"]
        c.steps = list(d["steps"])
        return c


def _rng_state():
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python": random.getstate()}


def _set_rng_state(st):
    torch.set_rng_state(st["torch"])
    if st["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])
    random.setstate(st["python"])


def save_run_state(path, model, optimizer, scheduler, step, extra):
    """Everything a branch needs to continue as if the run had not stopped."""
    state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter_state": state, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "rng": _rng_state(), "step": step,
                "extra": extra}, path)


def load_run_state(path, model, optimizer, scheduler):
    """Restores adapter, optimizer, scheduler and RNG in place; returns (step, extra)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    named = dict(model.named_parameters())
    missing = [n for n in blob["adapter_state"] if n not in named]
    if missing:
        raise RuntimeError(f"{len(missing)} saved tensors have no matching parameter, e.g. {missing[0]}")
    with torch.no_grad():
        for n, t in blob["adapter_state"].items():
            named[n].copy_(t.to(named[n].device, dtype=named[n].dtype))
    optimizer.load_state_dict(blob["optimizer"])
    scheduler.load_state_dict(blob["scheduler"])
    _set_rng_state(blob["rng"])
    return blob["step"], blob["extra"]


def make_exposure_schedule(n_queries, seed):
    """One slot per post-branch query: exactly half 0 (original), half 1 (added), shuffled by a
    seed-fixed RNG. The same list is used for B and C, so they differ only in the added text."""
    if n_queries % 2:
        raise ValueError(f"{n_queries} post-branch queries cannot split 1:1")
    slots = [0] * (n_queries // 2) + [1] * (n_queries // 2)
    random.Random(f"apocf-online-schedule-{seed}").shuffle(slots)
    return slots


def normalise_ws(text):
    return re.sub(r"\s+", " ", text).strip()


def validate_generated_instruction(text, base_instruction, tokenizer, max_tokens):
    """The acceptance rules, fixed before any run. Returns (ok, reasons). A failure is reported
    and the run stops: no regeneration, no repair, no second call (078)."""
    reasons = []
    if not text or not text.strip():
        return False, ["empty"]
    if normalise_ws(text) == normalise_ws(base_instruction):
        reasons.append("identical to the base instruction after whitespace normalisation")
    for lab in LABELS:
        if lab not in text:
            reasons.append(f"does not mention the output label {lab!r}")
    n_tok = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if n_tok > max_tokens:
        reasons.append(f"{n_tok} tokens exceeds the {max_tokens}-token bound the passage cache was "
                       "built for")
    return (not reasons), reasons


def load_apocf_module(rewriter_path=DEFAULT_REWRITER_PATH):
    """Load the APO-CF prompt builder (src/apocf_prompt.py) by file path."""
    spec = importlib.util.spec_from_file_location("sftpo_rewriter", rewriter_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("build_apocf_prompt",):
        if not hasattr(mod, name):
            raise RuntimeError(f"{rewriter_path} has no {name}")
    return mod


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline=LF) as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write(LF)
