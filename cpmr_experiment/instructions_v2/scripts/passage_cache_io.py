"""
Persistent format, writer and strict loader for the common-passage cache.

Separation of concerns: the AUDIT report and the REUSABLE CACHE are different files. The
audit describes what happened; the cache is the data training and evaluation consume. A
cache is usable only when it is marked complete, its identity matches the consumer's
identity, and its stored content digest matches the entries actually present.

No silent fallback. A missing or stale key raises; it never falls back to the original
passage or to a differently-truncated one, because that would reintroduce exactly the
per-prompt variation the cache exists to remove.

Two profiles (message 019 section 2.3):
  * diagnostic -- may run with an unidentified tokenizer; results are provisional.
  * production -- requires a locked manifest AND a fingerprinted tokenizer. A locked
    manifest alone is not sufficient.

CPU only, no tokenizer, no network.
"""
import hashlib
import json
import os
from pathlib import Path

CACHE_FORMAT_VERSION = "passage-cache-3"
# Bump when anything here or in build_passage_cache.py changes how a cached string is
# derived. Recorded in every cache and checked on load.
PREPROCESSING_CODE_VERSION = "v2-prep-3"

# Fields that must agree before two eligibility files may be intersected. An id string
# alone does not make two sets comparable (019 section 2.1).
ELIGIBILITY_COMPAT_FIELDS = ["dataset", "split", "corpus_version", "selection_seed",
                             "positive_selection_manifest_hash", "role", "eligibility_unit"]


class CacheError(RuntimeError):
    pass


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


TOKENIZER_IDENTITY_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                            "merges.txt", "special_tokens_map.json", "added_tokens.json"]

# Fields that B1 (rendering audit) and B2 (passage cache) must agree on when they are meant
# to describe the same setup (019 section 2.3, 021 section 1). A differing value is a
# finding, not a pass.
RUNTIME_MATCH_FIELDS = ["model_name", "tokenizer_resolved_revision", "tokenizer_file_hashes",
                        "chat_template_sha256", "label_token_ids", "eos_token_id",
                        "enable_thinking", "max_length", "manifest_roll_up", "framework"]


def framework_identity():
    """The library versions that actually perform rendering and tokenization.

    Why this is part of cache identity: the chat template is applied by the transformers
    jinja engine, so two runs can share an identical chat_template_sha256 and identical
    tokenizer file hashes yet render differently under different transformers versions.
    Without this field a cache built on one machine can be consumed on another undetected.
    Observed gap in this project: local mirror transformers 5.16.1 / torch 2.11.0 vs GPU
    server (ad002) transformers 5.14.1 / torch 2.5.1 -- see
    cpmr_experiment/beir_pilot/audit/environment_canary_record.json."""
    out = {}
    for mod in ("transformers", "tokenizers", "torch"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out


def tokenizer_identity(tokenizer, model_name, requested_revision):
    """Pin the tokenizer beyond its name: the resolved snapshot when available, plus hashes
    of the actual vocabulary / special-token files. A bare revision string does not identify
    a local snapshot that was edited in place, and an equal chat-template hash does not imply
    an equal vocabulary.

    Shared by B1 and B2 so their records are directly comparable (021 section 1)."""
    resolved, file_hashes, src_dir = None, {}, None
    nvp = getattr(tokenizer, "name_or_path", None)
    if nvp and Path(nvp).is_dir():
        src_dir = Path(nvp)
    else:
        try:
            from transformers.utils import cached_file
            p = cached_file(model_name, "tokenizer_config.json", revision=requested_revision,
                            _raise_exceptions_for_missing_entries=False)
            if p:
                src_dir = Path(p).parent
                resolved = src_dir.name   # HF snapshot dirs are named by commit sha
        except Exception:  # noqa: BLE001
            src_dir = None
    if src_dir and src_dir.is_dir():
        for fn in TOKENIZER_IDENTITY_FILES:
            fp = src_dir / fn
            if fp.exists():
                file_hashes[fn] = sha256_file(fp)
    return {
        "requested_revision": requested_revision,
        "resolved_revision": resolved,
        "source_dir": str(src_dir) if src_dir else None,
        "file_hashes": file_hashes,
        "identified": bool(file_hashes) or bool(resolved) or bool(requested_revision),
    }


def runtime_fingerprint(*, model_name, tokenizer_id, chat_template_sha256, label_token_ids,
                        eos_token_id, enable_thinking, max_length, manifest_roll_up):
    """The block both B1 and B2 write, so the two can be compared field by field."""
    return {
        "model_name": model_name,
        "tokenizer_requested_revision": tokenizer_id.get("requested_revision"),
        "tokenizer_resolved_revision": tokenizer_id.get("resolved_revision"),
        "tokenizer_source_dir": tokenizer_id.get("source_dir"),
        "tokenizer_file_hashes": tokenizer_id.get("file_hashes"),
        "tokenizer_identified": tokenizer_id.get("identified"),
        "chat_template_sha256": chat_template_sha256,
        "label_token_ids": label_token_ids,
        "eos_token_id": eos_token_id,
        "enable_thinking": enable_thinking,
        "max_length": max_length,
        "manifest_roll_up": manifest_roll_up,
        "framework": framework_identity(),
    }


def compare_runtime_fingerprints(a, b, a_name="B1", b_name="B2"):
    """Returns (ok, differences). A difference means the two runs did NOT share a setup and
    must not be reported as a matching pair."""
    diffs = []
    for k in RUNTIME_MATCH_FIELDS:
        va, vb = a.get(k), b.get(k)
        if va != vb:
            diffs.append({"field": k, a_name: va, b_name: vb})
    return (not diffs), diffs


def identity_dict(*, model_name, model_revision, tokenizer_identity, chat_template_sha256,
                  label_token_ids, eos_token_id, max_length, enable_thinking,
                  manifest_roll_up, manifest_version, manifest_lock_status, roles_used,
                  framework=None):
    """Everything that, if changed, must invalidate a cached passage.

    model_revision and tokenizer identity are separate: a repo can move its tokenizer files
    without changing weights and vice versa. tokenizer_identity carries the resolved
    snapshot and the hashes of the actual vocabulary / special-token files, which a bare
    revision string does not guarantee for a local snapshot.

    NOTE: identity says HOW passages were derived. It does NOT say WHICH input rows were
    processed -- that is input_manifest_hash, recorded separately (019 section 2.2)."""
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "preprocessing_code_version": PREPROCESSING_CODE_VERSION,
        "model_name": model_name,
        "model_revision": model_revision,
        "tokenizer_identity": tokenizer_identity,
        "framework": framework if framework is not None else framework_identity(),
        "chat_template_sha256": chat_template_sha256,
        "label_token_ids": label_token_ids,
        "eos_token_id": eos_token_id,
        "max_length": max_length,
        "enable_thinking": enable_thinking,
        "manifest_version": manifest_version,
        "manifest_roll_up": manifest_roll_up,
        "manifest_lock_status": manifest_lock_status,
        "roles_used": sorted(roles_used),
    }


def identity_hash(identity: dict) -> str:
    return sha256_text(json.dumps(identity, sort_keys=True, ensure_ascii=False))


def content_digest(entries: dict) -> str:
    """Canonical digest over the cached passages themselves (019 section 2.2).

    Keyed by entry key -> sha256 of passage_fixed, so a silently edited passage, a dropped
    entry or an added one all change the digest."""
    return sha256_text(json.dumps(
        {k: sha256_text(v["passage_fixed"]) for k, v in sorted(entries.items())},
        sort_keys=True, ensure_ascii=False))


def entry_key(identity_hash_value: str, query: str, passage_original: str) -> str:
    return sha256_text(json.dumps({
        "identity": identity_hash_value,
        "query": query,
        "passage_original_sha256": sha256_text(passage_original),
    }, sort_keys=True, ensure_ascii=False))


def tokenizer_is_identified(identity: dict) -> bool:
    ti = identity.get("tokenizer_identity") or {}
    return bool(ti.get("file_hashes")) or bool(ti.get("resolved_revision")) \
        or bool(ti.get("requested_revision"))


def verify_manifest(manifest: dict) -> str:
    """Recompute the roll-up from the entries instead of trusting the stored value."""
    recomputed = sha256_text(json.dumps(
        {k: v["sha256_full"] for k, v in sorted(manifest["entries"].items())}, sort_keys=True))
    stored = manifest.get("roll_up_sha256")
    if stored != recomputed:
        raise CacheError(
            f"manifest roll_up mismatch: stored={stored} recomputed={recomputed} -- "
            f"the manifest file has been edited without regenerating it")
    for key, e in manifest["entries"].items():
        if sha256_text(e["full"]) != e["sha256_full"]:
            raise CacheError(f"manifest entry {key}: sha256_full does not match its text")
    return recomputed


def write_cache(path, *, identity, entries, status, stats, input_manifest, audit_path=None,
                allow_replace_complete=False):
    """Atomic write with an explicit overwrite policy (019 section 2.2).

    A complete cache on disk is never replaced by default. Re-running the same identity over
    the same inputs with the same result is an explicit no-op. A subset run or an invalid
    result can never take the place of a complete one."""
    path = Path(path)
    if status not in ("complete", "incomplete", "invalid"):
        raise CacheError(f"bad status {status!r}")
    ident_hash = identity_hash(identity)
    digest = content_digest(entries)

    if path.exists():
        try:
            existing = json.load(open(path, encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise CacheError(f"existing cache at {path} is unreadable: {exc}") from exc
        if existing.get("status") == "complete":
            same_identity = existing.get("identity_hash") == ident_hash
            same_content = existing.get("content_digest") == digest
            same_inputs = (existing.get("input_manifest", {}).get("input_manifest_hash")
                           == input_manifest.get("input_manifest_hash"))
            if same_identity and same_content and same_inputs and status == "complete":
                return {"path": str(path), "identity_hash": ident_hash,
                        "content_digest": digest, "file_sha256": sha256_file(path),
                        "status": "complete", "n_entries": len(entries), "action": "no-op"}
            if not allow_replace_complete:
                why = []
                if not same_identity:
                    why.append("different identity")
                if not same_content:
                    why.append("different content")
                if not same_inputs:
                    why.append("different input set")
                if status != "complete":
                    why.append(f"new status is {status}")
                raise CacheError(
                    f"refusing to replace the complete cache at {path} ({', '.join(why)}). "
                    f"Write to a new path, or pass allow_replace_complete=True deliberately.")

    payload = {
        "status": status,
        "identity": identity,
        "identity_hash": ident_hash,
        "content_digest": digest,
        "input_manifest": input_manifest,
        "stats": stats,
        "audit_report": audit_path,
        "n_entries": len(entries),
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return {"path": str(path), "identity_hash": ident_hash, "content_digest": digest,
            "file_sha256": sha256_file(path), "status": status, "n_entries": len(entries),
            "action": "written"}


class PassageCache:
    """Strict reader. Construct with the consumer's own identity; any mismatch fails."""

    def __init__(self, payload, expected_identity=None, profile="diagnostic",
                 expected_file_sha256=None, actual_file_sha256=None):
        if profile not in ("diagnostic", "production"):
            raise CacheError(f"bad profile {profile!r}")
        self.payload = payload
        self.profile = profile
        self.status = payload.get("status")
        self.identity = payload.get("identity", {})
        self.identity_hash = payload.get("identity_hash")
        self.entries = payload.get("entries", {})
        self.input_manifest = payload.get("input_manifest", {})

        if self.status != "complete":
            raise CacheError(f"cache status is {self.status!r}, refusing to use it")
        if identity_hash(self.identity) != self.identity_hash:
            raise CacheError("cache identity_hash does not match its identity block")

        # content integrity (019 section 2.2)
        if payload.get("n_entries") != len(self.entries):
            raise CacheError(f"n_entries {payload.get('n_entries')} != actual {len(self.entries)}")
        stored_digest = payload.get("content_digest")
        actual_digest = content_digest(self.entries)
        if stored_digest != actual_digest:
            raise CacheError(
                f"cache content digest mismatch: stored={stored_digest} actual={actual_digest} "
                f"-- entries were edited, dropped or added after the cache was written")
        if expected_file_sha256 is not None and actual_file_sha256 is not None \
                and expected_file_sha256 != actual_file_sha256:
            raise CacheError("cache file sha256 does not match the expected value recorded "
                             "by the producer")

        if expected_identity is not None:
            want = identity_hash(expected_identity)
            if want != self.identity_hash:
                diffs = sorted(k for k in set(expected_identity) | set(self.identity)
                               if expected_identity.get(k) != self.identity.get(k))
                raise CacheError(f"cache identity mismatch; differing fields: {diffs}")

        if profile == "production":
            if self.identity.get("manifest_lock_status") != "locked":
                raise CacheError("production profile requires a locked manifest; this cache "
                                 "was built from a draft manifest")
            if not tokenizer_is_identified(self.identity):
                raise CacheError(
                    "production profile requires a fingerprinted tokenizer (pinned revision "
                    "or hashed local snapshot); this cache records none. A locked manifest "
                    "does not substitute for tokenizer identification.")

    @classmethod
    def load(cls, path, expected_identity=None, profile="diagnostic", expected_file_sha256=None):
        path = Path(path)
        if not path.exists():
            raise CacheError(f"cache not found: {path}")
        payload = json.load(open(path, encoding="utf-8"))
        return cls(payload, expected_identity, profile,
                   expected_file_sha256, sha256_file(path) if expected_file_sha256 else None)

    def get(self, query: str, passage_original: str) -> str:
        key = entry_key(self.identity_hash, query, passage_original)
        rec = self.entries.get(key)
        if rec is None:
            raise CacheError(
                f"cache miss for key {key[:16]}... (query={query[:50]!r}). "
                f"No fallback is performed: regenerate the cache for this input set and "
                f"record the regeneration, or fix the identity mismatch.")
        return rec["passage_fixed"]


def intersect_eligibility(paths, out_path):
    """Build the COMMON eligibility set across pool conditions / arms (019 section 2.1).

    The unit is the QUERY GROUP, not the pair: basic and hard conditions share a query and
    a positive but differ in the negative, so their pair ids are different by construction.
    Files are only intersected when their comparison frame agrees on every field in
    ELIGIBILITY_COMPAT_FIELDS -- an identical id string does not make two sets comparable."""
    loaded = []
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        if "eligible_unit_ids" not in d:
            raise CacheError(f"{p}: not an eligibility file (no eligible_unit_ids); "
                             f"files written by an older version must be regenerated")
        loaded.append((str(p), d))

    frames = {}
    for p, d in loaded:
        frames[p] = {k: d.get(k) for k in ELIGIBILITY_COMPAT_FIELDS}
    ref_path, ref = loaded[0][0], frames[loaded[0][0]]
    for p, f in frames.items():
        bad = [k for k in ELIGIBILITY_COMPAT_FIELDS if f.get(k) != ref.get(k)]
        if bad:
            raise CacheError(
                f"incompatible eligibility files, refusing to intersect: {p} differs from "
                f"{ref_path} in {bad}. Different datasets, splits, corpus versions, selection "
                f"seeds or positive selections must not be combined.")
    conditions = [d.get("pool_condition") for _, d in loaded]
    if len(set(conditions)) != len(conditions):
        raise CacheError(f"duplicate pool_condition among inputs: {conditions}")

    per_file, all_units, sets = {}, set(), []
    for p, d in loaded:
        per_file[p] = {"pool_condition": d.get("pool_condition"),
                       "n_eligible": len(d["eligible_unit_ids"]),
                       "n_excluded": len(d.get("excluded", [])),
                       "identity_hash": d.get("identity_hash"),
                       "status": d.get("status")}
        s = set(d["eligible_unit_ids"])
        sets.append(s)
        all_units |= s | {e["eligibility_unit_id"] for e in d.get("excluded", [])}
    common = set.intersection(*sets) if sets else set()
    dropped = sorted(all_units - common)
    payload = {
        "rule": ("a query group eligible in every listed condition is eligible; excluded in "
                 "any condition means excluded everywhere"),
        "eligibility_unit": ref.get("eligibility_unit"),
        "comparison_frame": ref,
        "sources": per_file,
        "n_union": len(all_units),
        "n_common_eligible": len(common),
        "n_dropped_for_consistency": len(dropped),
        "common_eligible_unit_ids": sorted(common),
        "dropped_unit_ids": dropped,
        "note": ("These are query-group ids, not pair ids. A pair id additionally carries "
                 "positive_doc_id, negative_doc_id and the construction condition, and is "
                 "what difficulty and disagreement records key on."),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out_path) + ".tmp")
    json.dump(payload, open(tmp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)
    return payload


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="common-eligibility intersection helper")
    ap.add_argument("--intersect", nargs="+", required=True,
                    help="eligibility json files, one per pool condition / arm")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = intersect_eligibility(a.intersect, a.out)
    print(f"[eligibility] unit={r['eligibility_unit']} union={r['n_union']} "
          f"common={r['n_common_eligible']} dropped={r['n_dropped_for_consistency']}")
    print(f"[eligibility] wrote {a.out}")
