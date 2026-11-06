"""Core of the T1 instruction-only MIPROv2 experiment (experiment log 059/061/062). Torch-free, GPU-free, scorer / prompt LM / tokenizer are injected, so every rule here is unit-testable on the CPU.

Rules (frozen before any run):
  * one predictor, instructions = a BODY; the output suffix '\\n\\nOutput exactly one label: "Relevant" or "Irrelevant".' is fixed outside DSPy, demos = 0;
  * candidate 0 of a stage = the stage's START body (P1: original T1, P2: I1); N = 4 slots in total = 1 unchanged + 3 proposed; the unchanged candidate wins ties;
  * a candidate is INVALID (excluded, logged, never regenerated outside the budget) if: empty / more than one paragraph / contains the label line / more than T1-body tokens + 40 (REAL tokenizer) /
    its val (query, passage, label) inputs are truncated more than with the start body / its training inputs raise more truncation than the start body; duplicates (normalised text) are logged, only the
    first occurrence stays;
  * selection metric = mean nDCG@10 over the fixed validation queries (the stage's 50 queries); TREC is never consulted;
  * the final choice scores EVERY valid unique candidate on the whole validation set (cached) -> highest mean, ties -> candidate 0.
Cache: raw margins in a directory named by the hash of a provenance dict (model revision, adapter/weights hash, tokenizer+template+max_length+dtype+FP32 path settings, scorer code hash, data hash);
aggregated metric cache additionally keyed by qrels hash + metric code hash; every file is written atomically (tmp + os.replace) so parallel processes never see partial files."""
import hashlib
import json
import math
import os
import re
import threading
import uuid
from pathlib import Path

import dspy

SUFFIX = '\n\nOutput exactly one label: "Relevant" or "Irrelevant".'
EXTRA_BODY_TOKENS = 40
# Rule revised on 2026-10-23 after the three APO-CF candidate TEXTS were seen but before any validation SCORE of an APO-CF candidate was viewed (experiment log 065): label NAMES may appear in a body ("A passage is "Relevant" only if ..."); only an OUTPUT-FORMAT instruction is invalid:
# the fixed sentence itself, or a sentence in which a verb of answering/labelling is followed by both label names joined by or/and. None of the 16 MIPROv2 candidates contains a label word, so no MIPROv2 validity changes.
OUTPUT_FORMAT_SENTENCE = re.compile(r"Output exactly one label|\b(?:respond|answer|reply|output|return|classify|label)\w*\b[^.\n]{0,60}[\"']?(?:Relevant|Irrelevant)[\"']?[^.\n]{0,24}\b(?:or|and)\b[^.\n]{0,24}[\"']?(?:Irrelevant|Relevant)\b", re.IGNORECASE)
PREFIX_BUDGET = 381          # max_length 384 - 2 label tokens - 1 EOS (the evaluator's budget for the longer label build)


def sha(s):
    return hashlib.sha256(s.encode("utf-8") if isinstance(s, str) else s).hexdigest()


def norm_text(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def split_start(full):
    assert full.endswith(SUFFIX), "the start instruction does not end with the fixed suffix"
    return full[: -len(SUFFIX)]


# ------------------------------------------------------------------ validity
class BodyChecker:
    """real-tokenizer length limit + structure checks; `val_check(full)` / `train_check(full)` are injected callables returning the number of truncated inputs (None = not checked)."""

    def __init__(self, tok, start_body, val_check=None, train_check=None, max_body_tokens=None, val_tolerance=0, train_tolerance=0):
        """max_body_tokens: ABSOLUTE cap of the body in real tokenizer tokens (default: start tokens + 40, the T1 runs). val_tolerance / train_tolerance: how many MORE truncated validation / training inputs
        than the start instruction are tolerated (default 0 = any additional truncation invalidates the candidate)."""
        self.tok, self.start_body = tok, start_body
        self.t1_tokens = len(tok(start_body, add_special_tokens=False)["input_ids"])
        self.max_tokens = max_body_tokens if max_body_tokens is not None else self.t1_tokens + EXTRA_BODY_TOKENS
        self.val_tolerance, self.train_tolerance = val_tolerance, train_tolerance
        self.val_check, self.train_check = val_check, train_check
        self.base_val = val_check(start_body + SUFFIX) if val_check else None
        self.base_train = train_check(start_body + SUFFIX) if train_check else None
        self.cache = {}

    def n_tokens(self, body):
        return len(self.tok(body, add_special_tokens=False)["input_ids"])

    def violations(self, body):
        if body in self.cache:
            return self.cache[body]
        v = []
        if not body.strip():
            v.append("empty")
        if "\n" in body.strip():
            v.append("not a single paragraph")
        if OUTPUT_FORMAT_SENTENCE.search(body):
            v.append("mentions the fixed label line")
        n = self.n_tokens(body)
        if n > self.max_tokens:
            v.append(f"{n} tokens > {self.max_tokens}")
        if not v and body != self.start_body:               # the input-level length checks (the start body is the baseline by definition)
            if self.val_check:
                t = self.val_check(body.strip() + SUFFIX)
                if t > self.base_val + self.val_tolerance:
                    v.append(f"validation inputs truncated: {t} > start {self.base_val} + tolerance {self.val_tolerance}")
            if self.train_check:
                t = self.train_check(body.strip() + SUFFIX)
                if t > self.base_train + self.train_tolerance:
                    v.append(f"training inputs truncated: {t} > start {self.base_train} + tolerance {self.train_tolerance}")
        self.cache[body] = v
        return v


# ------------------------------------------------------------------ cache
def _atomic_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex[:8]}")      # unique per process AND thread
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, default=str)           # DSPy usage objects (e.g. CompletionTokensDetailsWrapper) are not json types
    os.replace(tmp, path)


class RawScoreStore:
    """(provenance, instruction sha256, query id) -> {doc_id: margin}. The directory name is the hash of the whole provenance dict, so a change of weights / data / dtype / scorer
    settings / code can never hit an old entry; each file repeats its own state hash and is verified on read."""

    REQUIRED = ("model_revision", "weights", "tokenizer_template_maxlen_dtype_fp32path", "scorer_code_sha256", "data_sha256")

    def __init__(self, root, provenance):
        missing = [k for k in self.REQUIRED if k not in provenance]
        assert not missing, f"provenance lacks {missing}"
        self.prov = provenance
        self.state = sha(json.dumps(provenance, sort_keys=True))
        self.dir = Path(root) / self.state[:20]
        pf = self.dir / "provenance.json"
        if pf.exists():
            assert json.load(open(pf, encoding="utf-8")) == provenance, "provenance file differs from the requested state"
        else:
            _atomic_write(pf, provenance)
        self.hits = self.misses = 0

    def _p(self, inst_sha, qid):
        return self.dir / inst_sha[:20] / f"{qid}.json"

    def get(self, inst_sha, qid):
        p = self._p(inst_sha, qid)
        if p.exists():
            d = json.load(open(p, encoding="utf-8"))
            if d.get("state") == self.state and d.get("inst") == inst_sha and d.get("qid") == qid:
                self.hits += 1
                return d["scores"]
        return None

    def put(self, inst_sha, qid, scores):
        self.misses += 1
        _atomic_write(self._p(inst_sha, qid), {"state": self.state, "inst": inst_sha, "qid": qid, "scores": scores})

    def get_or_compute(self, full, qid, fn):
        s = sha(full)
        got = self.get(s, qid)
        if got is not None:
            return got
        val = fn(full, qid)
        self.put(s, qid, val)
        return val


class MetricStore:
    """aggregated cache: raw state + qrels hash + metric code hash + the validation query set + the instruction."""

    def __init__(self, root, raw_state, qrels_sha256, metric_code_sha256, qset_sha256):
        self.key0 = sha(json.dumps([raw_state, qrels_sha256, metric_code_sha256, qset_sha256]))
        self.dir = Path(root) / "metric" / self.key0[:20]

    def get(self, full):
        p = self.dir / f"{sha(full)[:20]}.json"
        return json.load(open(p, encoding="utf-8")) if p.exists() else None

    def put(self, full, obj):
        _atomic_write(self.dir / f"{sha(full)[:20]}.json", obj)


# ------------------------------------------------------------------ DSPy program + metric
class JudgePassage(dspy.Signature):
    """placeholder, replaced by the start body"""
    query: str = dspy.InputField()
    passage: str = dspy.InputField()
    label: str = dspy.OutputField(desc="the relevance judgement")


class RerankJudge(dspy.Module):
    """Relevance judge for passage re-ranking. The only optimisable part is its instruction: ONE short paragraph (at most 50 words, one paragraph, no line breaks) that tells the judge how to decide
    whether a passage is relevant to a query. The instruction must NOT mention the answer labels or any answer format: the answer format is appended automatically after the instruction.
    forward(qid) scores the validation query's candidate passages with the CURRENT instruction through the injected scorer; invalid or duplicate instructions score 0 and are logged."""

    def __init__(self, start_body, scorer, checker, log, seen_norm):
        super().__init__()
        self.judge = dspy.Predict(JudgePassage.with_instructions(start_body))
        self._scorer, self._checker, self._log, self._seen = scorer, checker, log, seen_norm

    def forward(self, qid=None, query=None, passage=None, **_):
        if qid is None:        # a TRAINING example: MIPROv2 runs a bootstrap step over the trainset even with 0 demos -> nothing is scored, nothing is recorded as a trace
            return dspy.Prediction(scores=None, rejected=["training example (proposer context only)"])
        body = self.judge.signature.instructions
        bad = self._checker.violations(body)
        if not bad:
            n = norm_text(body)
            first = self._seen.setdefault(n, body)
            if first != body:
                bad = ["duplicate of an earlier candidate"]
        if bad:
            self._log.append({"event": "rejected_candidate", "body": body, "violations": bad})
            return dspy.Prediction(scores=None, rejected=bad)
        full = body.strip() + SUFFIX
        assert full.endswith(SUFFIX) and not self.judge.demos, "suffix or demos changed"
        return dspy.Prediction(scores=self._scorer(full, qid), rejected=None)


def ndcg10(scores, pos, n_all_relevant):
    """binary relevance, linear gain, log2 discount, ideal DCG over ALL qrels positives (the 055 definition; ties in the score are broken by dict order only for exactly equal floats)."""
    top = sorted(scores, key=lambda d: -scores[d])[:10]
    dcg = sum((1 if d in pos else 0) / math.log2(i + 2) for i, d in enumerate(top))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(10, n_all_relevant)))
    return dcg / idcg if idcg > 0 else 0.0


def make_metric(qrels):
    """qrels: {qid: (set(positive doc ids inside the candidate pool), number of all qrels positives)}"""
    def metric(example, pred, trace=None):
        if pred.scores is None or getattr(example, "qid", None) is None:
            return 0.0
        pos, n_all = qrels[example.qid]
        return ndcg10(pred.scores, pos, n_all)
    return metric


def select_best(candidates, val_qids, scorer, qrels, metric_store=None):
    """candidates: [(index, body, violations)] in slot order, index 0 = the start body. Every valid UNIQUE candidate is scored on every validation query (cached); highest mean wins, ties -> smaller index."""
    rows, seen = [], {}
    for i, body, bad in candidates:
        n = norm_text(body)
        if not bad and n in seen:
            bad = [f"duplicate of candidate {seen[n]}"]
        if not bad:
            seen[n] = i
        if bad:
            rows.append({"index": i, "body": body, "body_sha256": sha(body), "violations": bad, "mean_ndcg10": None})
            continue
        full = body.strip() + SUFFIX
        cached = metric_store.get(full) if metric_store else None
        if cached is None:
            per = {q: ndcg10(scorer(full, q), qrels[q][0], qrels[q][1]) for q in val_qids}
            cached = {"mean": sum(per.values()) / len(per), "per_query": per}
            if metric_store:
                metric_store.put(full, cached)
        rows.append({"index": i, "body": body, "body_sha256": sha(body), "violations": [], "mean_ndcg10": cached["mean"], "per_query": cached["per_query"]})
    valid = [r for r in rows if r["mean_ndcg10"] is not None]
    best = max(valid, key=lambda r: (round(r["mean_ndcg10"], 12), -r["index"]))
    return best, rows


def run_stage(start_body, scorer, tok, trainset, val_examples, qrels, prompt_lm, *, val_check=None, train_check=None, metric_store=None, num_candidates=4, num_trials=6, seed=0, lm_log=None):
    """one P stage: MIPROv2 (instruction proposal + trial loop, tips/few-shot off) followed by the deterministic scoring of every valid candidate. Returns a result dict."""
    checker = BodyChecker(tok, start_body, val_check, train_check)
    log, seen = [], {}
    program = RerankJudge(start_body, scorer, checker, log, seen)
    captured = {}
    orig = dspy.teleprompt.MIPROv2._propose_instructions

    def rec(self, *a, **k):
        r = orig(self, *a, **k)                      # DSPy has replaced candidate 0 by the unchanged start instruction here
        captured["cands"] = list(r[0])
        return r
    dspy.teleprompt.MIPROv2._propose_instructions = rec
    try:
        opt = dspy.teleprompt.MIPROv2(metric=make_metric(qrels), prompt_model=prompt_lm, task_model=prompt_lm, auto=None, num_candidates=num_candidates, max_bootstrapped_demos=0, max_labeled_demos=0,
                                      seed=seed, num_threads=1, verbose=False, track_stats=True)
        opt.compile(program, trainset=trainset, valset=val_examples, num_trials=num_trials, minibatch=False, max_bootstrapped_demos=0, max_labeled_demos=0, seed=seed, requires_permission_to_run=False,
                    program_aware_proposer=True, data_aware_proposer=True, tip_aware_proposer=False, fewshot_aware_proposer=False)
    finally:
        dspy.teleprompt.MIPROv2._propose_instructions = orig
    cands = captured["cands"]
    assert len(cands) == num_candidates and cands[0] == start_body, "candidate 0 must be the unchanged start body"
    rows_in = [(i, b, checker.violations(b)) for i, b in enumerate(cands)]
    best, rows = select_best(rows_in, [e.qid for e in val_examples], scorer, qrels, metric_store)
    n_unique_valid = sum(1 for r in rows if r["mean_ndcg10"] is not None)
    return {"start_body": start_body, "start_body_sha256": sha(start_body), "start_body_tokens": checker.t1_tokens, "token_limit": checker.max_tokens, "slots": num_candidates, "proposed_slots": num_candidates - 1,
            "unique_valid_candidates": n_unique_valid, "candidates": rows, "selected_index": best["index"], "selected_body": best["body"], "selected_body_sha256": best["body_sha256"],
            "selected_mean_ndcg10": best["mean_ndcg10"], "start_mean_ndcg10": rows[0]["mean_ndcg10"], "mipro_trials": num_trials, "mipro_rejected_log": log,
            "base_val_truncated_inputs": checker.base_val, "base_train_truncated_inputs": checker.base_train}
