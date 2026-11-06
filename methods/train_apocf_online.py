"""APOCF-online pilot (experiment log 078, decisions in 080): does adding ONE instruction found by
APOCF part-way through training improve retrieval under the ORIGINAL instruction?

Three commands, run in this order. Nothing here is started by importing the module.

  prefix   steps 0..branch_step-1, original instruction (T1) only, token-only CE, K=1. Accumulates
           the online proxy confusion from every one of those forwards, then saves the full run
           state (adapter, optimizer, scheduler, RNG). branch_step defaults to the END OF WARMUP,
           round(warmup_ratio * total_steps) = 62 of 625 (user decision, 080).
  apocf    ONE call to the rewriter model with the accumulated confusion. Writes the prompt, the raw
           reply, the cleaned instruction and the acceptance verdict. No regeneration: a reply that
           fails the rules is reported and the C branch refuses to run.
  branch   steps branch_step..total_steps-1 from the saved state, for one condition:
             A  Single      original instruction only
             B  Fixed-Multi original + fixed paraphrase T2, exposure 1:1 per query
             C  APOCF-Multi original + the APOCF instruction, exposure 1:1 per query
           B and C use the SAME exposure schedule; only the added text differs. Final evaluation
           uses the original instruction (T1) for every condition.

Unit: one query group = (positive, negative), both under the same instruction; loss per query is
the mean gold-label CE over the two, 8 queries per optimizer step, exactly as the K=2 token-only
trainer weights them, but with ONE instruction per forward (K=1), so 10,000 forwards per condition.

Limitation recorded here because it cannot be removed: the RNG state is restored identically at the
branch, but dropout consumes it per token, and the added instruction changes sequence lengths, so
B and C (and A) draw different dropout masks afterwards. The branch removes the prefix noise; it
does not make the three post-branch runs bitwise parallel.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
if ROOT is None:
    raise RuntimeError("no .repo-root marker found in any parent directory")
for rel in ("", "methods", "cpmr_experiment/scripts"):
    sys.path.insert(0, str(ROOT / rel))

import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import get_linear_schedule_with_warmup  # noqa: E402

from apocf_online_lib import (OnlineConfusion, forward_ce_and_pred,  # noqa: E402
                              label_first_token_ids, load_apocf_module, load_run_state,
                              make_exposure_schedule, save_run_state,
                              validate_generated_instruction, write_json)
from cpmr_query_dataset import load_query_groups  # noqa: E402

LF = chr(10)
EFFECTIVE_BATCH = 8
DEFAULT_KEY_PATH = "apikey.txt"      # not used by any experiment of the paper (online APO-CF training of an earlier phase)
DEFAULT_TRAIN_PAIRS = "data/msmarco_train_v2_5k/pairs_text_basic_5000.jsonl"
DEFAULT_CACHE = "cpmr_experiment/instructions_v2/cache_5k/msmarco5k_basic__cache.json"
DEFAULT_MANIFEST = "cpmr_experiment/instructions_v2/manifest/instructions_v2.json"


def common_args(ap):
    ap.add_argument("--config", default="configs/qwen3_4b_pilot_2x2.yaml")
    ap.add_argument("--tag", required=True, help="run family name; branches add _A/_B/_C")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_root", default="cpmr_experiment/pilot_apocf_online")
    ap.add_argument("--train_pairs", default=DEFAULT_TRAIN_PAIRS,
                    help="v2 basic pool. The v1 default of train_cpmr.py is deliberately NOT used")
    ap.add_argument("--passage_cache", action="append", default=None)
    ap.add_argument("--instructions_manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--branch_step", type=int, default=None,
                    help="default: end of warmup, round(warmup_ratio*total_steps)")
    ap.add_argument("--log_interval", type=int, default=25)
    ap.add_argument("--debug_max_steps", type=int, default=None,
                    help="smoke only: stop after this many optimizer steps, write nothing durable")
    ap.add_argument("--local_files_only", action="store_true")


def build_context(args, need_model=True):
    """Everything the three commands share. Prints the arithmetic so a log states it."""
    cfg = yaml.safe_load(open(ROOT / args.config))
    caches = args.passage_cache or [DEFAULT_CACHE]
    sys.path.insert(0, str(ROOT / "cpmr_experiment" / "instructions_v2" / "scripts"))
    from v2_input_path import V2InputPath
    v2 = V2InputPath(cache_paths=[ROOT / p for p in caches],
                     manifest_path=ROOT / args.instructions_manifest, max_length=args.max_length,
                     enable_thinking=False, roles=("TRAIN",), profile="diagnostic")
    groups, order = load_query_groups(ROOT / args.train_pairs, args.seed)
    n_q = len(order)
    assert n_q % EFFECTIVE_BATCH == 0, f"{n_q} queries not divisible by {EFFECTIVE_BATCH}"
    total_steps = n_q // EFFECTIVE_BATCH
    num_warmup = round(cfg["optim"]["warmup_ratio"] * total_steps)
    branch_step = args.branch_step if args.branch_step is not None else num_warmup
    assert 0 < branch_step < total_steps
    print(f"[apocf] {n_q} queries, {total_steps} steps, warmup {num_warmup}, branch at step "
          f"{branch_step}; post-branch queries {(total_steps - branch_step) * EFFECTIVE_BATCH}")
    ctx = {"cfg": cfg, "v2": v2, "groups": groups, "order": order, "n_q": n_q,
           "total_steps": total_steps, "num_warmup": num_warmup, "branch_step": branch_step,
           "t1": v2.instructions[v2.canonical_id], "canonical_id": v2.canonical_id,
           "builder": v2.as_builder()}
    if need_model:
        from src.model import load_model_and_tokenizer
        torch.manual_seed(args.seed)
        loaded = load_model_and_tokenizer(
            model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
            gradient_checkpointing=cfg.get("gradient_checkpointing", False),
            local_files_only=args.local_files_only)
        model, tok = loaded.model, loaded.tokenizer
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg["optim"]["lr"])
        sched = get_linear_schedule_with_warmup(opt, num_warmup, total_steps)
        ctx.update(model=model, tok=tok, opt=opt, sched=sched,
                   device=next(model.parameters()).device,
                   rel_id=None, irr_id=None)
        ctx["rel_id"], ctx["irr_id"] = label_first_token_ids(tok)
    return ctx


def run_steps(ctx, args, first_step, last_step, instruction_for, confusion, stream_path,
              total_fwd, tag):
    """Optimizer steps first_step..last_step-1. instruction_for(step, j, qid) -> (slot, text)."""
    model, tok, opt, sched = ctx["model"], ctx["tok"], ctx["opt"], ctx["sched"]
    t0 = time.time()
    for step in range(first_step, last_step):
        opt.zero_grad(set_to_none=True)
        qids = ctx["order"][step * EFFECTIVE_BATCH:(step + 1) * EFFECTIVE_BATCH]
        confusion.begin_step(step + 1)
        ces, slot_n = [], {0: 0, 1: 0}
        for j, qid in enumerate(qids):
            g = ctx["groups"][qid]
            slot, text = instruction_for(step, j, qid)
            slot_n[slot] += 1
            q_ces = []
            for doc, gold in ((g["pos"], "Relevant"), (g["neg"], "Irrelevant")):
                ce, pred, _m, _n = forward_ce_and_pred(
                    model, tok, text, g["query"], doc, gold, args.max_length, ctx["device"],
                    ctx["rel_id"], ctx["irr_id"], enable_thinking=False,
                    build_example_fn=ctx["builder"])
                confusion.add(gold, pred)
                q_ces.append(ce)
                total_fwd[0] += 1
            loss = torch.stack(q_ces).mean()
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step} qid {qid}; stopping, no changes")
            (loss / EFFECTIVE_BATCH).backward()
            ces.append(float(loss.detach()))
        opt.step()
        sched.step()
        row = confusion.end_step()
        row.update({"ce": sum(ces) / len(ces), "lr": sched.get_last_lr()[0],
                    "n_slot0": slot_n[0], "n_slot1": slot_n[1], "total_forward_passes": total_fwd[0]})
        if stream_path is not None:
            with open(stream_path, "a", encoding="utf-8", newline=LF) as f:
                f.write(json.dumps(row) + LF)
        if (step + 1) % args.log_interval == 0 or step + 1 == last_step:
            print(f"[apocf] {tag} step={step + 1}/{ctx['total_steps']} ce={row['ce']:.4f} "
                  f"lr={row['lr']:.2e} fwd={total_fwd[0]} elapsed={time.time() - t0:.0f}s", flush=True)


def cmd_prefix(args):
    ctx = build_context(args)
    out = ROOT / args.out_root
    ckdir = out / "checkpoints" / f"{args.tag}_prefix"
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stream = log_dir / f"steps_{args.tag}_prefix.jsonl"
    if stream.exists():
        stream.unlink()
    last = ctx["branch_step"]
    if args.debug_max_steps:
        last = min(last, args.debug_max_steps)
        print(f"[apocf][DEBUG] capped to {last} steps; nothing durable is written")
    conf, fwd = OnlineConfusion(), [0]
    run_steps(ctx, args, 0, last, lambda s, j, q: (0, ctx["t1"]), conf, stream, fwd,
              f"{args.tag}_prefix")
    print(f"[apocf] prefix confusion {conf.counts()}")
    if args.debug_max_steps:
        return 0
    sched = make_exposure_schedule((ctx["total_steps"] - last) * EFFECTIVE_BATCH, args.seed)
    extra = {"confusion": conf.state_dict(), "branch_step": last, "seed": args.seed,
             "exposure_schedule": sched, "total_forward_passes": fwd[0],
             "train_pairs": args.train_pairs, "canonical_id": ctx["canonical_id"],
             "t1_sha256": hashlib.sha256(ctx["t1"].encode("utf-8")).hexdigest(),
             "scope": "online proxy confusion: first-token decision, LoRA dropout on, mixed "
                      "model states from step 0 to the branch; not a fixed-checkpoint confusion"}
    save_run_state(ckdir / "prefix_state.pt", ctx["model"], ctx["opt"], ctx["sched"], last, extra)
    write_json(out / "results" / f"{args.tag}_prefix_confusion.json",
               {"tag": args.tag, "branch_step": last, **conf.counts(), "scope": extra["scope"]})
    print(f"[apocf] saved {ckdir / 'prefix_state.pt'}")
    return 0


def load_prefix_confusion(out, tag):
    p = out / "results" / f"{tag}_prefix_confusion.json"
    return json.load(open(p, encoding="utf-8"))


class CountingClient:
    """Wraps the OpenAI client to count every request and keep the usage, so 'called once' is a
    measured fact in the result file and not a claim. The rewriter helper re-sends once without
    `temperature` if the model rejects it; that second request would show up here."""

    def __init__(self, client):
        self._client, self.n_requests, self.usages = client, 0, []
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.n_requests += 1
                resp = outer._client.chat.completions.create(**kw)
                u = getattr(resp, "usage", None)
                if u is not None:
                    outer.usages.append({"prompt_tokens": u.prompt_tokens,
                                         "completion_tokens": u.completion_tokens})
                return resp

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def cmd_apocf(args):
    from transformers import AutoTokenizer
    cfg = yaml.safe_load(open(ROOT / args.config))
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], local_files_only=args.local_files_only)
    v2_ctx = build_context(args, need_model=False)
    out = ROOT / args.out_root
    conf = load_prefix_confusion(out, args.tag)
    mod = load_apocf_module(args.rewriter_path)
    prompt = mod.build_apocf_prompt(v2_ctx["t1"], conf["tn"], conf["fp"], conf["fn"], conf["tp"],
                                    domain="passage", level="full", constraint_variant="similar")
    res = {"tag": args.tag, "model": args.model, "temperature": args.temperature,
           "confusion": {k: conf[k] for k in ("tp", "fp", "fn", "tn", "n_valid", "n_excluded")},
           "branch_step": conf["branch_step"], "scope": conf["scope"],
           "rewriter_path": args.rewriter_path,
           "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
    print(prompt)
    if args.dry_run:
        print("[apocf] --dry_run: no API call was made")
        return 0
    if (out / "results" / f"{args.tag}_apocf_result.json").exists() and not args.overwrite:
        raise SystemExit("an APOCF result already exists for this tag; refusing a second call "
                         "(078: one call only). Pass --overwrite only on explicit instruction.")
    client = CountingClient(mod.load_openai_client(args.api_key_path))
    t0 = time.time()
    try:
        raw = mod._call_openai_chat(client, args.model, prompt, args.max_tokens,
                                    temperature=args.temperature)
    except Exception as e:  # report, never retry
        res.update({"error": f"{type(e).__name__}: {e}", "n_requests": client.n_requests})
        write_json(out / "results" / f"{args.tag}_apocf_result.json", res)
        raise SystemExit(f"API call failed ({res['error']}); not retried")
    text = mod._clean_revision(raw)
    max_tokens = max(len(tok(e["full"], add_special_tokens=False)["input_ids"])
                     for e in json.load(open(ROOT / args.instructions_manifest,
                                             encoding="utf-8"))["entries"].values())
    ok, reasons = validate_generated_instruction(text, v2_ctx["t1"], tok, max_tokens)
    trunc = {"checked": False}
    if ok and not args.skip_truncation_check:
        from v2_input_path import UnexpectedTruncation
        bad = 0
        for qid in v2_ctx["order"]:
            g = v2_ctx["groups"][qid]
            for doc, gold in ((g["pos"], "Relevant"), (g["neg"], "Irrelevant")):
                try:
                    v2_ctx["v2"].build_example(tok, text, g["query"], doc, gold, verify=True)
                except UnexpectedTruncation:
                    bad += 1
        trunc = {"checked": True, "n_examples": 2 * len(v2_ctx["order"]), "n_extra_truncated": bad}
        if bad:
            ok, reasons = False, reasons + [f"{bad} examples would be truncated further by the "
                                           "new instruction"]
    res.update({"raw_response": raw, "instruction": text, "accepted": ok, "reasons": reasons,
                "token_bound": max_tokens, "truncation_check": trunc,
                "n_requests": client.n_requests, "usage": client.usages,
                "seconds": round(time.time() - t0, 2)})
    write_json(out / "results" / f"{args.tag}_apocf_result.json", res)
    write_json(out / "results" / f"{args.tag}_apocf_instruction.json",
               {"text": text, "accepted": ok, "reasons": reasons})
    print(f"[apocf] accepted={ok} reasons={reasons} requests={client.n_requests}")
    print(f"[apocf] instruction: {text}")
    return 0


def cmd_branch(args):
    ctx = build_context(args)
    out = ROOT / args.out_root
    t1 = ctx["t1"]
    if args.condition == "A":
        added = None
    elif args.condition == "B":
        added = ctx["v2"].instructions[args.fixed_paraphrase_id]
    else:
        p = out / "results" / f"{args.tag}_apocf_instruction.json"
        if not p.exists():
            raise SystemExit(f"{p} missing: run the apocf command first")
        d = json.load(open(p, encoding="utf-8"))
        if not d["accepted"] and not args.override_rejection:
            raise SystemExit(f"the APOCF instruction was not accepted ({d['reasons']}); condition C "
                             "does not run (078: no regeneration, no repair). A user decision to run it "
                             "anyway is recorded with --override_rejection '<reason>'")
        added = d["text"]
        if not d["accepted"]:
            # the verdict file is left exactly as written; the override is a separate record
            write_json(out / "results" / f"{args.tag}_C_override.json",
                       {"tag": args.tag, "instruction_verdict": "not accepted",
                        "rejection_reasons": d["reasons"], "override_reason": args.override_rejection,
                        "instruction": added, "regenerated": False, "repaired": False})
            print(f"[apocf] OVERRIDE: running C although the instruction was not accepted "
                  f"{d['reasons']}; reason recorded: {args.override_rejection}")
    ckdir = out / "checkpoints" / f"{args.tag}_prefix"
    step, extra = load_run_state(ckdir / "prefix_state.pt", ctx["model"], ctx["opt"], ctx["sched"])
    if step != ctx["branch_step"] or extra["branch_step"] != step:
        raise SystemExit(f"saved state is at step {step}, expected branch step {ctx['branch_step']}")
    sched_slots = extra["exposure_schedule"]
    print(f"[apocf] branch {args.condition} from step {step}; added instruction: "
          f"{'none' if added is None else added!r}")
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    btag = f"{args.tag}_{args.condition}"
    stream = log_dir / f"steps_{btag}.jsonl"
    if stream.exists():
        stream.unlink()

    def instruction_for(s, j, qid):
        k = (s - step) * EFFECTIVE_BATCH + j
        slot = sched_slots[k] if added is not None else 0
        return slot, (added if slot == 1 else t1)

    last = ctx["total_steps"]
    if args.debug_max_steps:
        last = min(last, step + args.debug_max_steps)
        print(f"[apocf][DEBUG] capped to {last}; nothing durable is written")
    conf, fwd = OnlineConfusion(), [extra["total_forward_passes"]]
    run_steps(ctx, args, step, last, instruction_for, conf, None if args.debug_max_steps else stream,
              fwd, btag)
    if args.debug_max_steps:
        return 0
    ck = out / "checkpoints" / f"{btag}_t100"
    ck.mkdir(parents=True, exist_ok=True)
    state = {n: p.detach().cpu().clone() for n, p in ctx["model"].named_parameters()
             if p.requires_grad}
    torch.save({"adapter_state": state, "tag": f"{btag}_t100", "model": ctx["cfg"]["model_name"],
                "condition": f"APOCF-online-{args.condition}", "global_step": last,
                "frac": 1.0, "seed": args.seed, "branch_step": step,
                "added_instruction": added, "total_forward_passes_so_far": fwd[0],
                "rejection_overridden": bool(args.condition == "C" and args.override_rejection),
                "override_reason": args.override_rejection},
               ck / "ckpt_final.pt")
    print(f"[apocf] saved {ck / 'ckpt_final.pt'}; total forwards {fwd[0]}")
    if not args.skip_eval:
        from src.eval_rerank_mistral import evaluate_rerank_mistral
        for year in ("2019", "2020"):
            res = evaluate_rerank_mistral(
                ctx["model"], ctx["tok"], t1, args.max_length, ctx["device"],
                save_path=str(log_dir / f"eval_{btag}_t100_T1_trec{year}_scores.json"),
                year=year, enable_thinking=False)
            print(f"[eval] {btag} T1 TREC-DL {year}: nDCG@10={res['ndcg@10']:.4f} "
                  f"MRR@10={res['mrr@10']:.4f} n_queries={res['n_queries']}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prefix")
    common_args(p)
    p = sub.add_parser("apocf")
    common_args(p)
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=400)
    p.add_argument("--api_key_path", default=DEFAULT_KEY_PATH)
    p.add_argument("--rewriter_path", default=None)
    p.add_argument("--dry_run", action="store_true", help="print the prompt, make no API call")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_truncation_check", action="store_true")
    p = sub.add_parser("branch")
    common_args(p)
    p.add_argument("--condition", required=True, choices=["A", "B", "C"])
    p.add_argument("--fixed_paraphrase_id", default="T2")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--override_rejection", default=None,
                   help="run C although the APOCF instruction failed the acceptance rules; the "
                        "reason is written to results/<tag>_C_override.json and the checkpoint")
    args = ap.parse_args()
    if args.cmd == "apocf" and args.rewriter_path is None:
        from apocf_online_lib import DEFAULT_REWRITER_PATH
        args.rewriter_path = DEFAULT_REWRITER_PATH
    return {"prefix": cmd_prefix, "apocf": cmd_apocf, "branch": cmd_branch}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
