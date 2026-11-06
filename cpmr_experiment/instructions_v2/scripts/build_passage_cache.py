"""
Passage invariance: make every instruction, arm and label path see the SAME passage string
for the same (query, passage), and persist that choice as a reusable cache.

Why: v2 instructions are paraphrases of different lengths, unlike the v1 adverb set which
was token-length identical (model_tokenizer_prompt_audit.json: token_length_diffs_vs_only
all 0, prefix 65 tokens). src/data_mistral.py:38-42 truncates the PASSAGE (raw characters,
0.8x iteratively) whenever the rendered prefix exceeds the budget, so a longer instruction
leaves less passage. Without this step a prompt could score worse purely because it saw
less text. We vary the instruction wording; we hold the passage information fixed.

Corrections applied after the message-017 code review:
  * the cache is PERSISTED (passage_cache_io.write_cache) and consumed through a strict
    loader; the audit report and the reusable cache are separate files;
  * cache identity pins model AND tokenizer revisions plus tokenizer file hashes, label
    token ids, EOS, max_length, enable_thinking and the manifest roll-up, which is
    recomputed from the manifest entries rather than trusted;
  * verification compares FULL TOKEN ID SEQUENCES, not Example.product_text -- that field
    stores the builder's INPUT argument, not the internally truncated text
    (src/data_mistral.py:55), so comparing it could never detect re-truncation;
  * exclusion is decided per PAIR/GROUP, never per row, and an eligibility file is written
    so that arms and negative conditions can be intersected to a common set;
  * there is no "keep" escape hatch: an unresolvable overflow makes the run invalid.

No assumption that one globally longest prompt binds: every (prompt, label) combination is
measured on every shrink iteration.

Requires a tokenizer (CPU only, no GPU, no model weights). Does not run on the local
Windows mirror, which has no Qwen3-4B tokenizer cached and no network.

Usage (server, from project root):
  python cpmr_experiment/instructions_v2/scripts/build_passage_cache.py \
      --pairs data/scenarios/msmarco_m1_b5000_train.jsonl --role train \
      --pool_condition basic_top1000 \
      --config configs/qwen3_4b_cpmr_pilot.yaml --enable_thinking false \
      --tokenizer_revision <pinned-sha> --roles_used TRAIN,TEST
"""
import argparse
import json
from itertools import zip_longest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
V2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from src.data_mistral import build_example_mistral  # noqa: E402
from passage_cache_io import (  # noqa: E402
    CacheError, PassageCache, entry_key, identity_dict, identity_hash, runtime_fingerprint,
    sha256_file, sha256_text, tokenizer_identity, tokenizer_is_identified, verify_manifest,
    write_cache,
)


def prefix_ids_for(tokenizer, instruction, query, passage_text, enable_thinking):
    """Byte-identical construction to src/data_mistral.py build_example_mistral.make_prefix."""
    user_content = f"{instruction}\n\nQuery: {query}\n{passage_text}"
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    text = tokenizer.apply_chat_template([{"role": "user", "content": user_content}], **kwargs)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def label_parts(tokenizer, label):
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    response_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
    return response_ids, eos


def fits_all(tokenizer, instructions, labels, query, passage_text, max_length, enable_thinking):
    per_combo, ok = {}, True
    for key, instr in instructions.items():
        n_prefix = len(prefix_ids_for(tokenizer, instr, query, passage_text, enable_thinking))
        for label in labels:
            response_ids, eos = label_parts(tokenizer, label)
            budget = max_length - len(response_ids) - len(eos)
            fit = n_prefix <= budget
            ok = ok and fit
            per_combo[f"{key}|{label}"] = {"n_prefix": n_prefix, "budget": budget,
                                           "n_total": n_prefix + len(response_ids) + len(eos),
                                           "fits": fit}
    return ok, per_combo


def shrink_to_common(tokenizer, instructions, labels, query, passage_text, max_length,
                     enable_thinking, min_chars=20):
    text, n_shrinks = passage_text, 0
    ok, per_combo = fits_all(tokenizer, instructions, labels, query, text, max_length,
                             enable_thinking)
    while not ok and len(text) > min_chars:
        text = text[: int(len(text) * 0.8)]
        n_shrinks += 1
        ok, per_combo = fits_all(tokenizer, instructions, labels, query, text, max_length,
                                 enable_thinking)
    return text, n_shrinks, ok, per_combo


def verify_token_identity(tokenizer, instructions, labels, query, passage_fixed, max_length,
                          enable_thinking):
    """Compare FULL expected token id sequences against what the shipped builder produces.

    Length equality is reported separately from content equality, because equal length does
    not imply equal tokens (017 §2.3). Example.product_text is deliberately NOT used: it
    stores the builder's input argument, so it cannot reveal internal truncation."""
    length_problems, content_problems, per_combo = [], [], {}
    for key, instr in instructions.items():
        expected_prefix = prefix_ids_for(tokenizer, instr, query, passage_fixed, enable_thinking)
        for label in labels:
            response_ids, eos = label_parts(tokenizer, label)
            expected_ids = expected_prefix + response_ids + eos
            ex = build_example_mistral(tokenizer, instr, query, passage_fixed, label,
                                       max_length, enable_thinking=enable_thinking)
            combo = f"{key}|{label}"
            same_len = len(ex.input_ids) == len(expected_ids)
            same_ids = ex.input_ids == expected_ids
            within = len(ex.input_ids) <= max_length
            # label mask: the response span must be exactly the label + eos positions
            resp_start = len(expected_prefix)
            expected_labels = [-100] * len(expected_ids)
            for i in range(resp_start, resp_start + len(response_ids) + len(eos)):
                expected_labels[i] = expected_ids[i]
            same_mask = ex.labels == expected_labels
            per_combo[combo] = {
                "n_expected": len(expected_ids), "n_actual": len(ex.input_ids),
                "same_length": same_len, "same_token_ids": same_ids,
                "same_label_mask": same_mask, "within_max_length": within,
                "prefix_sha256": sha256_text(json.dumps(expected_prefix)),
            }
            if not same_len:
                length_problems.append(f"{combo}: length {len(ex.input_ids)} != {len(expected_ids)} "
                                       f"(builder re-truncated)")
            if not same_ids:
                content_problems.append(f"{combo}: token ids differ from the expected sequence")
            if not same_mask:
                content_problems.append(f"{combo}: label mask differs from the expected mask")
            if not within:
                length_problems.append(f"{combo}: final length {len(ex.input_ids)} > {max_length}")
    return length_problems, content_problems, per_combo


def group_rows(rows, text_field, unit):
    """The eligibility unit is the QUERY GROUP, not the pair.

    basic and hard conditions share a query and a positive but differ in the negative, so
    their pair ids differ by construction and cannot be intersected. Intersecting query
    groups is what makes the arms comparable over the same queries (019 section 2.1)."""
    if unit == "candidate":
        return [(f"{r['query_id']}::{r.get('doc_id') or r.get('product_id')}", [r]) for r in rows]
    groups = {}
    for r in rows:
        groups.setdefault(r["query_id"], []).append(r)
    return list(groups.items())


def pair_id_for(members, pool_condition, selection_seed):
    """The REAL pair id, which difficulty and disagreement records key on. Unlike the
    eligibility unit it carries both document ids and the construction condition."""
    pos = next((m.get("product_id") or m.get("doc_id") for m in members
                if m.get("label") == "Relevant"), None)
    neg = next((m.get("product_id") or m.get("doc_id") for m in members
                if m.get("label") == "Irrelevant"), None)
    if pos is None or neg is None:
        return None
    return sha256_text(json.dumps({"query_id": members[0]["query_id"], "positive_doc_id": pos,
                                   "negative_doc_id": neg, "condition": pool_condition,
                                   "selection_seed": selection_seed}, sort_keys=True))[:32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--role", required=True, choices=["train", "eval"])
    ap.add_argument("--pool_condition", required=True,
                    help="e.g. basic_top1000 | hard_rank1_25 | eval_bm25_top100")
    ap.add_argument("--manifest", default=str(V2 / "manifest" / "instructions_v2.json"))
    ap.add_argument("--roles_used", default="TRAIN,TEST")
    ap.add_argument("--config", default="configs/qwen3_4b_cpmr_pilot.yaml")
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--enable_thinking", default="false", choices=["unset", "true", "false"])
    ap.add_argument("--tokenizer_revision", default=None,
                    help="pin the tokenizer; recorded either way, and the resolved snapshot "
                         "plus file hashes are recorded too")
    ap.add_argument("--model_revision", default=None)
    ap.add_argument("--dataset", required=True,
                    help="comparison-frame field; eligibility files may only be intersected "
                         "when dataset/split/corpus_version/selection_seed agree")
    ap.add_argument("--split", default="train")
    ap.add_argument("--corpus_version", required=True)
    ap.add_argument("--selection_seed", type=int, required=True)
    ap.add_argument("--profile", default="diagnostic", choices=["diagnostic", "production"],
                    help="production requires a locked manifest AND a fingerprinted tokenizer")
    ap.add_argument("--allow_replace_complete", action="store_true",
                    help="deliberately replace an existing complete cache; off by default")
    ap.add_argument("--verify_consumer", type=int, default=5,
                    help="after writing, reload the cache from disk and re-render this many "
                         "entries through the real builder (0 disables)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--cache_out", default=None)
    ap.add_argument("--audit_out", default=None)
    ap.add_argument("--eligibility_out", default=None)
    args = ap.parse_args()

    unit = "candidate" if args.role == "eval" else "pair"
    enable_thinking = {"unset": None, "true": True, "false": False}[args.enable_thinking]
    cfg = yaml.safe_load(open(ROOT / args.config))
    max_length = args.max_length or cfg["data"]["max_length"]

    tag = f"{Path(args.pairs).stem}__{args.pool_condition}"
    cache_out = Path(args.cache_out or V2 / "cache" / f"passage_cache__{tag}.json")
    audit_out = Path(args.audit_out or V2 / "audit" / f"passage_invariance__{tag}.json")
    elig_out = Path(args.eligibility_out or V2 / "cache" / f"eligibility__{tag}.json")

    man = json.load(open(args.manifest, encoding="utf-8"))
    verify_manifest(man)          # recomputed, not trusted
    roles = {r.strip().upper() for r in args.roles_used.split(",") if r.strip()}
    instructions = {k: e["full"] for k, e in man["entries"].items() if e["role"] in roles}
    if not instructions:
        raise SystemExit(f"no instructions for roles {roles}")
    labels = [man["label_strings"]["positive"], man["label_strings"]["negative"]]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], revision=args.tokenizer_revision,
        local_files_only=args.local_files_only)
    tok_id = tokenizer_identity(tokenizer, cfg["model_name"], args.tokenizer_revision)
    label_ids = {lab: tokenizer(lab, add_special_tokens=False)["input_ids"] for lab in labels}
    fingerprint = runtime_fingerprint(
        model_name=cfg["model_name"], tokenizer_id=tok_id,
        chat_template_sha256=sha256_text(getattr(tokenizer, "chat_template", None) or ""),
        label_token_ids=label_ids, eos_token_id=tokenizer.eos_token_id,
        enable_thinking=args.enable_thinking, max_length=max_length,
        manifest_roll_up=man["roll_up_sha256"])
    if not tok_id["identified"]:
        print("[cache] WARNING: could not hash tokenizer identity files; "
              "the cache will record this as unidentified")

    identity = identity_dict(
        model_name=cfg["model_name"], model_revision=args.model_revision,
        tokenizer_identity=tok_id,
        chat_template_sha256=sha256_text(getattr(tokenizer, "chat_template", None) or ""),
        label_token_ids=label_ids,
        eos_token_id=tokenizer.eos_token_id, max_length=max_length,
        enable_thinking=args.enable_thinking, manifest_roll_up=man["roll_up_sha256"],
        manifest_version=man.get("version"), manifest_lock_status=man.get("lock_status"),
        roles_used=roles)
    ident_hash = identity_hash(identity)

    # production gate (019 section 2.3): a locked manifest is NOT sufficient on its own
    if args.profile == "production":
        if not tokenizer_is_identified(identity):
            raise SystemExit(
                "profile=production but the tokenizer could not be fingerprinted "
                "(no pinned revision, no resolved snapshot, no file hashes). "
                "Pin --tokenizer_revision or use a verified local snapshot.")
        if man.get("lock_status") != "locked":
            raise SystemExit(
                f"profile=production but the manifest lock_status is "
                f"{man.get('lock_status')!r}; lock the instruction set first.")

    pairs_path = Path(args.pairs) if Path(args.pairs).is_absolute() else ROOT / args.pairs
    rows = [json.loads(l) for l in open(pairs_path, encoding="utf-8") if l.strip()]
    text_field = "product_text" if rows and "product_text" in rows[0] else "doc_text"
    groups = group_rows(rows, text_field, unit)
    if args.limit:
        groups = groups[: args.limit]

    # WHICH rows were processed -- distinct from identity, which says HOW (019 section 2.2)
    input_manifest = {
        "pairs_file": str(args.pairs),
        "pairs_file_sha256": sha256_file(pairs_path),
        "n_rows_in_file": len(rows),
        "limit": args.limit,
        "n_groups_processed": len(groups),
        "unit": unit,
        "processed_unit_ids_sha256": sha256_text(json.dumps([g for g, _ in groups],
                                                            sort_keys=False)),
        "is_subset": bool(args.limit),
    }
    input_manifest["input_manifest_hash"] = sha256_text(
        json.dumps(input_manifest, sort_keys=True, ensure_ascii=False))

    cache_entries, records, excluded, length_problems, content_problems = {}, [], [], [], []
    eligible_unit_ids = []

    for gi, (gid, members) in enumerate(groups):
        member_recs, group_ok = [], True
        for r in members:
            raw = r[text_field]
            passage_in = raw if raw.startswith("Passage:") else f"Passage: {raw}"
            fixed, n_shrinks, ok, per_combo = shrink_to_common(
                tokenizer, instructions, labels, r["query"], passage_in, max_length,
                enable_thinking)
            rec = {
                "eligibility_unit_id": gid,
                "pair_id": pair_id_for(members, args.pool_condition, args.selection_seed),
                "query_id": r["query_id"],
                "doc_id": r.get("doc_id") or r.get("product_id"),
                "row_label": r.get("label"),
                "cache_key": entry_key(ident_hash, r["query"], passage_in),
                "passage_original_sha256": sha256_text(passage_in),
                "passage_fixed_sha256": sha256_text(fixed),
                "chars_original": len(passage_in), "chars_kept": len(fixed),
                "chars_kept_frac": round(len(fixed) / len(passage_in), 6) if passage_in else 1.0,
                "n_shrinks": n_shrinks, "was_truncated": n_shrinks > 0,
                "fits_all_combinations": ok,
                "n_prefix_min": min(c["n_prefix"] for c in per_combo.values()),
                "n_prefix_max": max(c["n_prefix"] for c in per_combo.values()),
                "binding_combination": max(per_combo, key=lambda k: per_combo[k]["n_prefix"]),
                "_query": r["query"], "_passage_in": passage_in, "_fixed": fixed,
            }
            if not ok:
                group_ok = False
                rec["status"] = "overflow"
            member_recs.append(rec)

        if group_ok:
            for rec in member_recs:
                lp, cp, per_combo = verify_token_identity(
                    tokenizer, instructions, labels, rec["_query"], rec["_fixed"],
                    max_length, enable_thinking)
                rec["length_problems"], rec["content_problems"] = lp, cp
                rec["verification"] = per_combo
                length_problems.extend(f"{rec['eligibility_unit_id']}: {p}" for p in lp)
                content_problems.extend(f"{rec['eligibility_unit_id']}: {p}" for p in cp)
                rec["status"] = "ok" if not (lp or cp) else "verification_failed"
                if rec["status"] == "ok":
                    cache_entries[rec["cache_key"]] = {
                        "passage_fixed": rec["_fixed"],
                        "query_id": rec["query_id"], "doc_id": rec["doc_id"],
                        "passage_original_sha256": rec["passage_original_sha256"],
                        "chars_kept": rec["chars_kept"], "was_truncated": rec["was_truncated"],
                    }
            if all(r["status"] == "ok" for r in member_recs):
                eligible_unit_ids.append(gid)
        else:
            # PAIR-level exclusion: if any member overflows, the whole group goes, so a
            # query never contributes a half pair (017 section 2.4).
            reasons = [r["doc_id"] for r in member_recs if r.get("status") == "overflow"]
            excluded.append({"eligibility_unit_id": gid, "query_id": members[0]["query_id"],
                             "pair_id": pair_id_for(members, args.pool_condition,
                                                    args.selection_seed),
                             "reason": "budget_overflow", "offending_doc_ids": reasons,
                             "n_members_dropped": len(member_recs)})
            for rec in member_recs:
                rec["status"] = rec.get("status", "dropped_with_group")

        for rec in member_recs:
            for k in ("_query", "_passage_in", "_fixed"):
                rec.pop(k, None)
            records.append(rec)
        if (gi + 1) % 500 == 0:
            print(f"[cache] {gi + 1}/{len(groups)} groups")

    n_trunc = sum(1 for r in records if r.get("was_truncated"))
    eval_overflow = args.role == "eval" and excluded
    status = "complete"
    if length_problems or content_problems:
        status = "invalid"
    elif eval_overflow:
        status = "invalid"
    elif excluded and args.role == "train":
        status = "complete"   # pair-level exclusions are recorded, not silent

    stats = {
        "role": args.role, "unit": unit, "pool_condition": args.pool_condition,
        "n_groups": len(groups), "n_rows": len(records),
        "n_truncated_rows": n_trunc,
        "truncation_rate_rows": round(n_trunc / len(records), 6) if records else None,
        "n_excluded_units": len(excluded),
        "n_eligible_units": len(eligible_unit_ids),
        "n_length_problems": len(length_problems),
        "n_content_problems": len(content_problems),
    }

    audit = {
        "generated_by": "cpmr_experiment/instructions_v2/scripts/build_passage_cache.py",
        "pairs_file": str(args.pairs), "identity": identity, "identity_hash": ident_hash,
        "manifest_roll_up_recomputed": True,
        "n_instructions": len(instructions), "labels": labels,
        "status": status, "stats": stats,
        "length_problems": length_problems[:200],
        "content_problems": content_problems[:200],
        "excluded_pairs": excluded,
        "input_manifest": input_manifest,
        "profile": args.profile,
        "runtime_fingerprint": fingerprint,
        "caveat": ("chars_kept is bounded by the binding (prompt, label) combination, so "
                   "shorter instructions give up context they could otherwise have used. "
                   "That loss is the price of holding passage information fixed; it is "
                   "reported, not hidden. Excluding a pair changes the training sample "
                   "composition -- use the eligibility intersection so every arm and "
                   "negative condition trains on the same pairs."),
        "records": records,
    }
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(audit, open(audit_out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    elig = {
        "eligibility_unit": "query_group",
        "dataset": args.dataset, "split": args.split,
        "corpus_version": args.corpus_version, "selection_seed": args.selection_seed,
        "positive_selection_manifest_hash": sha256_text(json.dumps(
            {"rule": "one pre-selected positive per query; all known positives excluded "
                     "from the negative pool", "seed": args.selection_seed}, sort_keys=True)),
        "pool_condition": args.pool_condition, "role": args.role,
        "identity_hash": ident_hash, "status": status,
        "input_manifest_hash": input_manifest["input_manifest_hash"],
        "eligible_unit_ids": eligible_unit_ids, "excluded": excluded,
        "note": ("query-group ids, not pair ids: basic and hard share the query and the "
                 "positive but differ in the negative, so their pair ids differ by "
                 "construction and cannot be intersected"),
    }
    elig_out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(elig, open(elig_out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    written = write_cache(cache_out, identity=identity, entries=cache_entries, status=status,
                          stats=stats, input_manifest=input_manifest,
                          audit_path=str(audit_out),
                          allow_replace_complete=args.allow_replace_complete)

    # CONSUMER ROUND TRIP (019 section 3): reload the file we just wrote through the strict
    # loader and re-render through the real builder, so the smoke run actually demonstrates
    # the consumer path rather than only the producer.
    consumer = {"ran": False}
    if args.verify_consumer and status == "complete" and cache_entries:
        problems = []
        cache = PassageCache.load(cache_out, expected_identity=identity,
                                  profile=args.profile,
                                  expected_file_sha256=written["file_sha256"])
        # Both classes, deliberately. Two opposite mistakes are possible here and both have been
        # made: with the default ordering the budget lands entirely on short documents and the
        # truncation path is never re-checked through the loader (033 section 5); with truncated
        # records simply placed first, a budget smaller than the truncated count excludes short
        # documents entirely (observed on nfcorpus/boundary: 20 truncated / 0 untruncated).
        # So the budget is split between the two classes, and whichever class is short gives its
        # unused share to the other.
        ok_records = [r for r in records if r.get("status") == "ok"]
        tr = [r for r in ok_records if r.get("was_truncated")]
        un = [r for r in ok_records if not r.get("was_truncated")]
        n_trunc_avail = len(tr)
        budget = args.verify_consumer
        n_tr = min(len(tr), (budget + 1) // 2)
        n_un = min(len(un), budget - n_tr)
        n_tr = min(len(tr), budget - n_un)          # give back the share un could not use
        picked = tr[:n_tr] + un[:n_un]
        trunc_first = [r for pair in zip_longest(tr[:n_tr], un[:n_un]) for r in pair
                       if r is not None]           # interleaved, deterministic
        assert len(trunc_first) == len(picked)
        checked = 0
        checked_truncated = 0
        checked_untruncated = 0
        for rec in trunc_first:
            if checked >= args.verify_consumer:
                break
            src = next((r for r in rows
                        if r["query_id"] == rec["query_id"]
                        and (r.get("doc_id") or r.get("product_id")) == rec["doc_id"]), None)
            if src is None:
                continue
            raw = src[text_field]
            passage_in = raw if raw.startswith("Passage:") else f"Passage: {raw}"
            got = cache.get(src["query"], passage_in)
            if sha256_text(got) != rec["passage_fixed_sha256"]:
                problems.append(f"{rec['eligibility_unit_id']}: loaded passage differs")
            lp, cp, _ = verify_token_identity(tokenizer, instructions, labels, src["query"],
                                              got, max_length, enable_thinking)
            problems.extend(f"{rec['eligibility_unit_id']}: consumer {x}" for x in lp + cp)
            checked += 1
            if rec.get("was_truncated"):
                checked_truncated += 1
            else:
                checked_untruncated += 1
        consumer = {"ran": True, "n_checked": checked,
                    "n_truncated_available": n_trunc_avail,
                    "n_checked_truncated": checked_truncated,
                    "n_checked_untruncated": checked_untruncated,
                    "truncation_path_exercised": checked_truncated > 0,
                    "n_untruncated_available": len(un),
                    "both_classes_checked": checked_truncated > 0 and checked_untruncated > 0,
                    "note": ("the budget is split between truncated and untruncated records so "
                             "neither class can be crowded out; a zero count means that class was "
                             "absent from this input set, not that the check was skipped"),
                    "problems": problems,
                    "cache_file_sha256": written["file_sha256"],
                    "content_digest": written["content_digest"],
                    "profile": args.profile, "ok": not problems}
        if problems:
            status = "invalid"
            audit["status"] = status
            print(f"[cache] CONSUMER ROUND TRIP FAILED: {len(problems)} problems")
    audit["consumer_round_trip"] = consumer
    audit["cache_file"] = {"path": written["path"], "file_sha256": written["file_sha256"],
                           "content_digest": written["content_digest"],
                           "status": written["status"], "action": written["action"]}
    json.dump(audit, open(audit_out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"[cache] groups={len(groups)} rows={len(records)} truncated={n_trunc}")
    print(f"[cache] excluded_units={len(excluded)} eligible_units={len(eligible_unit_ids)}")
    print(f"[cache] length_problems={len(length_problems)} content_problems={len(content_problems)}")
    print(f"[cache] status={status} identity={ident_hash[:16]}...")
    print(f"[cache] cache  -> {written['path']} (sha256 {written['file_sha256'][:16]}...)")
    print(f"[cache] audit  -> {audit_out}")
    print(f"[cache] eligib -> {elig_out}")
    if consumer["ran"]:
        print("[cache] consumer round trip: checked={} (truncated {} / untruncated {}, "
              "truncated available {}) problems={} profile={}".format(
                  consumer["n_checked"], consumer["n_checked_truncated"],
                  consumer["n_checked_untruncated"], consumer["n_truncated_available"],
                  len(consumer["problems"]), consumer["profile"]))
    if status != "complete":
        print("[cache] STATUS IS NOT COMPLETE -- this cache must not be consumed")
        sys.exit(1)


if __name__ == "__main__":
    main()
