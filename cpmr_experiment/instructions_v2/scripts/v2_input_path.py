"""
The v2 input path that real training and evaluation use: v2 manifest instructions plus passages
served from the passage cache instead of truncated per prompt at build time.

This is the piece that was 0% through messages 017-040. Everything before it verified the
*producer* (the cache was written correctly); this verifies and supplies the *consumer* (training
and evaluation actually read it).

The v1 path is untouched. `src/data_mistral.build_example_mistral` is not modified, and
`train_cpmr.py` / `eval_on_beir.py` fall back to their original behaviour whenever the new flags
are absent. What changes when the flags are present:

  * instructions come from `instructions_v2.json` (roles TRAIN/TEST, canonical = T1) instead of the
    hardcoded `INSTR_TEXT`;
  * a passage is looked up in the cache by (query, raw passage) and the cached `passage_fixed` is
    what reaches the builder.

Three things are failures, not warnings:

  1. **cache miss** -- a (query, passage) the cache does not hold. Never falls back to raw text,
     because raw text would be truncated differently per prompt and silently reintroduce the
     confound the cache exists to remove.
  2. **settings / manifest mismatch** -- `PassageCache.load` compares the full identity
     (tokenizer + framework + max_length + enable_thinking + manifest roll-up); a differing
     field aborts.
  3. **unexpected additional truncation** -- the cached passage is supposed to fit every
     (prompt, label) combination already. If the builder's shrink loop fires anyway, the token ids
     differ from the expected sequence and that is reported per combination. Length equality is
     checked separately from content equality: equal length does not imply equal tokens (017).

`verify_token_identity` from `build_passage_cache` is reused rather than reimplemented, so the
consumer is checked against the same comparison the producer used.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_passage_cache import (label_parts, prefix_ids_for,  # noqa: E402
                                 verify_token_identity)
from passage_cache_io import CacheError, PassageCache  # noqa: E402
from src.data_mistral import build_example_mistral  # noqa: E402

CANONICAL_TRAIN_ID = "T1"


class CacheMiss(KeyError):
    """A (query, passage) the cache does not hold. Never recoverable by falling back to raw
    text -- that would reintroduce per-prompt truncation."""


class UnexpectedTruncation(RuntimeError):
    """The cached passage did not fit after all, so the builder re-truncated it."""


class SettingsMismatch(RuntimeError):
    """The caller asked for max_length / enable_thinking the cache was not built with."""


def load_v2_instructions(manifest_path, roles=("TRAIN",)):
    """Returns ({prompt_id: text}, canonical_id, manifest_metadata).

    The roll-up is recomputed from the per-entry hashes, so a manifest edited without updating
    its roll-up is rejected here too, not only in the cache builder.
    """
    import hashlib
    man = json.load(open(manifest_path, encoding="utf-8"))
    recomputed = hashlib.sha256(json.dumps(
        {k: e["sha256_full"] for k, e in sorted(man["entries"].items())},
        sort_keys=True).encode("utf-8")).hexdigest()
    if recomputed != man.get("roll_up_sha256"):
        raise SystemExit(f"manifest roll-up mismatch: recomputed {recomputed} != "
                         f"recorded {man.get('roll_up_sha256')}")
    roles = tuple(roles)
    out = {k: e["full"] for k, e in man["entries"].items() if e["role"] in roles}
    if not out:
        raise SystemExit(f"no manifest entry has role in {roles}")
    canonical = CANONICAL_TRAIN_ID if CANONICAL_TRAIN_ID in out else sorted(out)[0]
    meta = {"manifest_path": str(manifest_path), "version": man.get("version"),
            "lock_status": man.get("lock_status"), "roll_up_sha256": man["roll_up_sha256"],
            "roll_up_recomputed": True, "roles_selected": list(roles),
            "prompt_ids": sorted(out), "canonical_prompt_id": canonical}
    return out, canonical, meta


class V2InputPath:
    """Supplies (instruction, query, passage) triples to the existing builder, with the passage
    taken from the cache. Counts every lookup so a run can state how many passages actually came
    from the cache rather than asserting it."""

    def __init__(self, cache_paths, manifest_path, max_length, enable_thinking,
                 roles=("TRAIN",), profile="diagnostic", expected_identity=None):
        self.caches, self.cache_paths = [], []
        for cp in ([cache_paths] if isinstance(cache_paths, (str, Path)) else list(cache_paths)):
            # PassageCache.load enforces identity, profile and file-hash agreement; a mismatch
            # raises here rather than producing quietly different inputs.
            self.caches.append(PassageCache.load(Path(cp), expected_identity=expected_identity,
                                                profile=profile))
            self.cache_paths.append(str(cp))
        self.instructions, self.canonical_id, self.manifest_meta = load_v2_instructions(
            manifest_path, roles)
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.stats = {"lookups": 0, "hits": 0, "misses": 0,
                      "passages_already_fitting": 0, "passages_truncated_by_cache": 0}
        self._checked = set()
        self._identity_consistent = self._check_identity_consistency()

    def _check_identity_consistency(self):
        """Several caches may be loaded at once (basic and hard conditions). They must agree on
        HOW passages were produced; WHICH documents they hold is allowed to differ."""
        ids = {c.identity_hash for c in self.caches}
        if len(ids) > 1:
            raise SystemExit(f"passage caches disagree on identity: {sorted(ids)}")
        return True

    @staticmethod
    def passage_key(raw):
        """The cache was keyed on the marker-prefixed passage, because that is the string
        `build_example_mistral` receives (`data_mistral` composes
        f"{instruction}

Query: {query}
{ptext}" and expects the marker to be present).

        Callers arrive with both forms -- the training pool rows already carry "Passage: ", the
        BEIR sample rows and `score_candidates_mistral`'s doc_text do not -- so normalising here
        rather than at each call site removes a way to produce a spurious cache miss."""
        return raw if raw.startswith("Passage:") else f"Passage: {raw}"

    def fixed_passage(self, query, passage_in):
        """The cached passage for this (query, passage). Raises CacheMiss; never falls back."""
        passage_in = self.passage_key(passage_in)
        self.stats["lookups"] += 1
        for c in self.caches:
            try:
                got = c.get(query, passage_in)
            except CacheError:      # PassageCache signals a miss with CacheError, not KeyError
                continue
            self.stats["hits"] += 1
            if got == passage_in:
                self.stats["passages_already_fitting"] += 1
            else:
                self.stats["passages_truncated_by_cache"] += 1
            return got
        self.stats["misses"] += 1
        raise CacheMiss(f"no cache entry for query={query[:60]!r} "
                        f"passage[:60]={passage_in[:60]!r}")

    def build_example(self, tokenizer, instruction, query, passage_in, label,
                      verify=False):
        """Drop-in for build_example_mistral on the v2 path: resolve the passage through the cache,
        then build. With verify=True the produced token ids and label mask are compared against
        rendering the SAME fixed passage directly -- the comparison basis is the preserved passage,
        not an independently truncated original."""
        fixed = self.fixed_passage(query, passage_in)
        # `fixed` is already the preserved, marker-prefixed passage; the builder receives it as-is
        ex = build_example_mistral(tokenizer, instruction, query, fixed, label,
                                   self.max_length, enable_thinking=self.enable_thinking)
        if verify:
            expected_prefix = prefix_ids_for(tokenizer, instruction, query, fixed,
                                             self.enable_thinking)
            response_ids, eos = label_parts(tokenizer, label)
            expected_ids = expected_prefix + response_ids + eos
            if ex.input_ids != expected_ids:
                raise UnexpectedTruncation(
                    f"token ids differ for label={label}: builder produced "
                    f"{len(ex.input_ids)} ids, expected {len(expected_ids)} "
                    f"(same length: {len(ex.input_ids) == len(expected_ids)})")
            expected_labels = [-100] * len(expected_ids)
            for i in range(len(expected_prefix), len(expected_ids)):
                expected_labels[i] = expected_ids[i]
            if ex.labels != expected_labels:
                raise UnexpectedTruncation(f"label mask differs for label={label}")
        return ex

    def as_builder(self, verify_every_call=False):
        """A callable with `build_example_mistral`'s exact signature, for the `build_example_fn`
        seam in cpmr_losses/slot_losses/pmr_losses.

        max_length and enable_thinking are compared against what the cache was built with rather
        than accepted: a caller asking for different settings would render different token ids
        from the same cached passage, which is a settings mismatch, not a fallback."""
        def builder(tokenizer, instruction, query, product_text, label, max_length=None,
                    orig_label=None, enable_thinking=None):
            if max_length is not None and max_length != self.max_length:
                raise SettingsMismatch(f"caller max_length={max_length} but the cache was built "
                                       f"with {self.max_length}")
            if enable_thinking != self.enable_thinking:
                raise SettingsMismatch(f"caller enable_thinking={enable_thinking!r} but the cache "
                                       f"was built with {self.enable_thinking!r}")
            return self.build_example(tokenizer, instruction, query, product_text, label,
                                      verify=verify_every_call)
        return builder

    def instruction_pool(self):
        """The dict shape `compute_cpmr_losses(instruction_pool, prompt_keys, ...)` expects."""
        return dict(self.instructions)

    def report(self):
        return {"manifest": self.manifest_meta,
                "caches": [{"path": pth, "identity_hash": c.identity_hash,
                            "n_entries": len(c.entries),
                            "input_manifest": c.input_manifest}
                           for pth, c in zip(self.cache_paths, self.caches)],
                "identity_consistent_across_caches": self._identity_consistent,
                "max_length": self.max_length, "enable_thinking": self.enable_thinking,
                "lookup_stats": dict(self.stats)}


def integration_check(input_path, tokenizer, rows, n_short, n_truncated, labels=("Relevant",
                                                                                "Irrelevant")):
    """Run real input preparation over both short and genuinely truncated documents.

    The two classes get separate budgets rather than an ordering, because a budget plus an ordered
    list lets whichever class comes first crowd the other out entirely -- that failure happened in
    both directions in the producer check (033 section 5, then 039 section 2.4).
    """
    short_rows, trunc_rows = [], []
    for r in rows:
        passage_in = input_path.passage_key(r["product_text"])
        try:
            fixed = input_path.fixed_passage(r["query"], passage_in)
        except CacheMiss:
            return {"ok": False, "error": "cache miss while classifying rows",
                    "query_id": r["query_id"], "doc_id": r.get("product_id")}
        (trunc_rows if fixed != passage_in else short_rows).append(r)

    picked = ([("truncated", r) for r in trunc_rows[:n_truncated]]
              + [("untruncated", r) for r in short_rows[:n_short]])
    problems, per_row = [], []
    for kind, r in picked:
        combos_ok = True
        lp, cp, per_combo = verify_token_identity(
            tokenizer, input_path.instruction_pool(), list(labels), r["query"],
            input_path.fixed_passage(r["query"], r["product_text"]),
            input_path.max_length, input_path.enable_thinking)
        if lp or cp:
            combos_ok = False
            problems.extend(f"{r['query_id']}::{r.get('product_id')}: {x}" for x in lp + cp)
        # and the path the trainer will actually call
        for label in labels:
            for pid in sorted(input_path.instruction_pool()):
                try:
                    input_path.build_example(tokenizer, input_path.instruction_pool()[pid],
                                             r["query"], r["product_text"], label, verify=True)
                except (CacheMiss, UnexpectedTruncation) as e:
                    combos_ok = False
                    problems.append(f"{r['query_id']}::{r.get('product_id')} {pid}|{label}: {e}")
        per_row.append({"kind": kind, "query_id": r["query_id"],
                        "doc_id": r.get("product_id"),
                        "n_combinations": len(per_combo), "ok": combos_ok})
    return {
        "ok": not problems,
        "n_rows_available": {"truncated": len(trunc_rows), "untruncated": len(short_rows)},
        "n_rows_checked": {"truncated": sum(1 for k, _ in picked if k == "truncated"),
                           "untruncated": sum(1 for k, _ in picked if k == "untruncated")},
        "both_classes_checked": (any(k == "truncated" for k, _ in picked)
                                 and any(k == "untruncated" for k, _ in picked)),
        "n_combinations_per_row": len(input_path.instruction_pool()) * len(labels),
        "rows": per_row,
        "problems": problems[:100],
        "n_problems": len(problems),
    }
