"""K=1 token CE + a triplet ranking loss built from the SAME forwards: 10,000 forwards, exactly the cost of plain K=1 CE.

Each query group is (query, positive document, negative document). Each document is forwarded ONCE with its gold label teacher-forced (as in plain K=1 CE),
which gives its token CE and, at the position before the first label token, the first-token margin
    m = logit(Relevant first token) - logit(Irrelevant first token)       (= log P(Relevant) - log P(Irrelevant) at that position)
The loss of a query group is
    L = mean(CE_pos, CE_neg) + alpha * softplus( -(m_pos - m_neg) )        alpha = 1, as in the existing FixedOnly-Rank condition
so the only change with respect to plain K=1 CE is the second term, and no forward is added. The existing ranking loss (cpmr_losses.compute_cpmr_losses) scores a document with
a full-sequence log-probability difference over BOTH labels, which needs two forwards per document (20,000 forwards); this is the closest single-forward variant, not the same
loss. Trainer functions, data, order, optimizer, LR schedule and LoRA settings are those of plain K=1 CE (methods/train_apocf_online.py build_context), the final evaluation is
T1 on TREC-DL 2019 and 2020, milestone checkpoints t10/t25/t50/t75/t100 are saved.

Instruction schedule (--schedule): single (one fixed instruction --train_prompt, default T1: the run above), uniform (a random one of T1-T5 per query group, 1,000 each, the schedule of
methods/train_multi5_k1.py), anchor (T1 on every other query group, T2-T5 on the rest, 625 each, the schedule of methods/train_multi5_anchor_k1.py). Everything else is identical.

    python methods/train_rank_margin_k1.py --tag rankm_k1_s0 --seed 0 --local_files_only
    python methods/train_rank_margin_k1.py --tag rankm_k1_smoke --seed 0 --local_files_only --debug_max_steps 25      # nothing durable is written
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
if ROOT is None:
    raise RuntimeError("no .repo-root marker found in any parent directory")
for rel in ("", "methods", "cpmr_experiment/scripts"):
    sys.path.insert(0, str(ROOT / rel))

import random  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_apocf_online as T  # noqa: E402
from apocf_online_lib import write_json  # noqa: E402
from anchor_ratio_schedule import make_ratio_schedule  # noqa: E402
from train_multi5_anchor_k1 import make_anchor_schedule  # noqa: E402
from train_multi5_k1 import FRACS, IDS, make_schedule  # noqa: E402

EFFECTIVE_BATCH = T.EFFECTIVE_BATCH
LF = chr(10)


def make_anchor_schedule_for(n_queries, seed, anchor):
    """The anchor schedule of train_multi5_anchor_k1 with any instruction as the anchor: positions 0, 2, 4, ... -> anchor, the others -> the four other
    instructions in equal counts, shuffled by the seed-fixed RNG. anchor T1 reproduces make_anchor_schedule exactly."""
    others_ids = [i for i in IDS if i != anchor]
    if n_queries % 2 or (n_queries // 2) % len(others_ids):
        raise ValueError(f"{n_queries} queries cannot be split into an anchor half and an even half over {len(others_ids)} instructions")
    others = [i for i in others_ids for _ in range(n_queries // 2 // len(others_ids))]
    random.Random(f"anchor-k1-schedule-{seed}").shuffle(others)
    it = iter(others)
    return [anchor if k % 2 == 0 else next(it) for k in range(n_queries)]


def margin_from_logits(logits_row, rel_id, irr_id):
    """First-token margin from the logits at the position before the first label token (float32, keeps the graph)."""
    row = logits_row.float()
    return row[rel_id] - row[irr_id]


def triplet_loss(ce_pos, ce_neg, m_pos, m_neg, alpha):
    """mean CE of the two documents + alpha * softplus(-(m_pos - m_neg)); returns (loss, rank_term)."""
    rank = F.softplus(-(m_pos - m_neg))
    return 0.5 * (ce_pos + ce_neg) + alpha * rank, rank


def forward_ce_and_margin(model, tokenizer, text, query, doc, gold, max_length, device, rel_id, irr_id, builder):
    ex = builder(tokenizer, text, query, doc, gold, max_length, enable_thinking=False)
    ids = torch.tensor([ex.input_ids], device=device)
    labels = torch.tensor([ex.labels], device=device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
    first = next(i for i, t in enumerate(ex.labels) if t != -100)
    if first == 0:
        raise RuntimeError("the first supervised position is 0")
    return out.loss, margin_from_logits(out.logits[0, first - 1], rel_id, irr_id)


def save_adapter(ctx, path, tag, step, seed, total_steps, alpha):
    path.mkdir(parents=True, exist_ok=True)
    state = {n: p.detach().cpu().clone() for n, p in ctx["model"].named_parameters() if p.requires_grad}
    torch.save({"adapter_state": state, "tag": tag, "model": ctx["cfg"]["model_name"], "condition": "Rank-margin-K1", "global_step": step,
                "frac": step / total_steps, "seed": seed, "alpha": alpha}, path / "ckpt_final.pt")


def save_full_state(ctx, args, path, step, fwd):
    """complete training state for an exact continuation: trainable weights, AdamW moments + step, LR scheduler, torch CPU / CUDA / python RNG, forward counter and the data-order signature
    (the data cursor is the optimizer step itself: update s consumes ctx['order'][s*8:(s+1)*8])."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "seed": args.seed, "tag": args.tag, "adapter": {n: p.detach().cpu().clone() for n, p in ctx["model"].named_parameters() if p.requires_grad},
                "opt": ctx["opt"].state_dict(), "sched": ctx["sched"].state_dict(), "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(), "py_random": random.getstate(),
                "fwd": fwd[0], "order_sig": [len(ctx["order"]), list(ctx["order"][:16]), list(ctx["order"][-16:])], "args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))}}, path)


def load_full_state(ctx, args, path):
    st = torch.load(path, map_location="cpu", weights_only=False)
    assert st["seed"] == args.seed, f"state seed {st['seed']} != {args.seed}"
    assert st["order_sig"] == [len(ctx["order"]), list(ctx["order"][:16]), list(ctx["order"][-16:])], "data order differs from the saved state"
    names = {n for n, p in ctx["model"].named_parameters() if p.requires_grad}
    assert names == set(st["adapter"]), "trainable parameter names differ"
    for n, p in ctx["model"].named_parameters():
        if p.requires_grad:
            p.data.copy_(st["adapter"][n].to(p.device, p.dtype))
    ctx["opt"].load_state_dict(st["opt"])
    ctx["sched"].load_state_dict(st["sched"])
    torch.set_rng_state(st["rng_cpu"])
    torch.cuda.set_rng_state_all(st["rng_cuda"])
    random.setstate(st["py_random"])
    return st


def ramp_c(args, total_steps, step):
    """LoRA strength of the optimizer update `step` (0-based): c = c0 + (1 - c0) * step / (T - 1); the same c for every microbatch and pos/neg forward of the step."""
    return args.lora_ramp_start + (1.0 - args.lora_ramp_start) * step / (total_steps - 1)


def diag_pass(ctx, args, tag, steps_done, c_last, c_next, out_path):
    """training-dynamics diagnostic on a fixed small set of TRAINING queries (not a generalisation set): eval mode, dropout off, no optimizer / scheduler / data-order change, RNG states restored.
    Each checkpoint is evaluated with the strength it was last trained with (c_last; 0.1 at step 0) and with c = 1; records CE / ranking loss / first-token margins and the RMS of the raw branch
    (alpha/r) B A x, the applied branch c (alpha/r) B A x and the base branch W x per module, and ||BA||_F."""
    import instruction_only_lora as IOL
    model, tok = ctx["model"], ctx["tok"]
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    was_training = model.training
    model.eval()
    rec = {"tag": tag, "steps_done": steps_done, "c_last": c_last, "c_next": c_next, "n_queries": len(ctx["diag_set"]), "settings": {}}
    final = steps_done == ctx["total_steps"] or c_last == 1.0
    for name, c in ((("c_last", c_last),) if final else (("c_last", c_last), ("c_1", 1.0))):
        bcap = {}
        hooks = []
        for n, m in model.named_modules():
            if n.endswith(".base_layer"):
                def mk(key):
                    def f(mod, a_, out):
                        e = bcap.setdefault(key, [0.0, 0])
                        e[0] += float(out.detach().float().pow(2).sum())
                        e[1] += out.numel()
                    return f
                hooks.append(m.register_forward_hook(mk(n[:-len(".base_layer")])))
        IOL.STATE["scale"] = c
        IOL.STATE["capture"] = {}
        ces, ranks, mps, mns = [], [], [], []
        with torch.no_grad():
            for qid in ctx["diag_set"]:
                g = ctx["groups"][qid]
                text = ctx["instr_for"](qid)
                ce_p, m_p = forward_ce_and_margin(model, tok, text, g["query"], g["pos"], "Relevant", args.max_length, ctx["device"], ctx["rel_id"], ctx["irr_id"], ctx["builder"])
                ce_n, m_n = forward_ce_and_margin(model, tok, text, g["query"], g["neg"], "Irrelevant", args.max_length, ctx["device"], ctx["rel_id"], ctx["irr_id"], ctx["builder"])
                ces.append(float(0.5 * (ce_p + ce_n)))
                ranks.append(float(F.softplus(-(m_p - m_n))))
                mps.append(float(m_p))
                mns.append(float(m_n))
        cap = IOL.STATE["capture"]
        for h in hooks:
            h.remove()
        rms = {k: [(v[0] / v[2]) ** 0.5, (v[1] / v[2]) ** 0.5, (bcap[k][0] / bcap[k][1]) ** 0.5] for k, v in cap.items()}      # raw, applied, base
        n_ = len(ces)
        rec["settings"][name] = {"c": c, "ce": sum(ces) / n_, "rank": sum(ranks) / n_, "loss": sum(ces) / n_ + args.alpha * sum(ranks) / n_, "margin_pos": sum(mps) / n_, "margin_neg": sum(mns) / n_,
                                 "gap": (sum(mps) - sum(mns)) / n_, "rms_raw_applied_base": rms}
    IOL.STATE["capture"] = None
    fro = {}
    for n, m in model.named_modules():
        if n.endswith(".lora_B.default"):
            B = m.weight.detach().float()
            A = model.get_submodule(n.replace("lora_B", "lora_A")).weight.detach().float()
            fro[n[:-len(".lora_B.default")]] = float(((B.T @ B) * (A @ A.T)).sum().clamp_min(0).sqrt())
    rec["fro_BA"] = fro
    if ctx.get("ba_ref") is not None:             # cosine of BA with the BA at the branch point (the A/B factors themselves are reparametrisation-dependent, BA is not)
        cos = {}
        for n, m in model.named_modules():
            if n.endswith(".lora_B.default"):
                key = n[:-len(".lora_B.default")]
                B = m.weight.detach().float()
                A = model.get_submodule(n.replace("lora_B", "lora_A")).weight.detach().float()
                A0, B0 = ctx["ba_ref"][key]
                dot = ((B.T @ B0) * (A0 @ A.T).T).sum()
                n1 = ((B.T @ B) * (A @ A.T)).sum().clamp_min(0).sqrt()
                n0 = ((B0.T @ B0) * (A0 @ A0.T)).sum().clamp_min(0).sqrt()
                cos[key] = float(dot / (n1 * n0 + 1e-30))
        rec["cos_BA_vs_branch"] = cos
    rec["lora_scale"] = IOL.STATE["lora_scale"]
    IOL.STATE["scale"] = c_next if c_next is not None else 1.0
    if was_training:
        model.train()
    torch.set_rng_state(rng_cpu)
    torch.cuda.set_rng_state_all(rng_cuda)
    with open(out_path, "a", encoding="utf-8", newline=LF) as f:
        f.write(json.dumps(rec) + LF)
    s1 = rec["settings"]["c_last"]
    print(f"[diag] {tag} steps_done={steps_done} c_last={c_last:.4f} c_next={c_next}: loss(c_last) {s1['loss']:.4f} ce {s1['ce']:.4f} rank {s1['rank']:.4f} gap {s1['gap']:.2f}"
          + ("" if final else f" | at c=1: loss {rec['settings']['c_1']['loss']:.4f} gap {rec['settings']['c_1']['gap']:.2f}"), flush=True)


def run_steps(ctx, args, first_step, last_step, stream_path, total_fwd, tag):
    model, tok, opt, sched = ctx["model"], ctx["tok"], ctx["opt"], ctx["sched"]
    t0 = time.time()
    for step in range(first_step, last_step):
        opt.zero_grad(set_to_none=True)
        c_now = None
        if ctx.get("ramp") is not None:
            import instruction_only_lora as _IOL
            c_now = ctx["ramp"](step)
            _IOL.STATE["scale"] = c_now
        qids = ctx["order"][step * EFFECTIVE_BATCH:(step + 1) * EFFECTIVE_BATCH]
        ces, ranks, gaps = [], [], []
        for qid in qids:
            g = ctx["groups"][qid]
            text = ctx["instr_for"](qid)
            ce_p, m_p = forward_ce_and_margin(model, tok, text, g["query"], g["pos"], "Relevant", args.max_length, ctx["device"], ctx["rel_id"], ctx["irr_id"], ctx["builder"])
            ce_n, m_n = forward_ce_and_margin(model, tok, text, g["query"], g["neg"], "Irrelevant", args.max_length, ctx["device"], ctx["rel_id"], ctx["irr_id"], ctx["builder"])
            total_fwd[0] += 2
            loss, rank = triplet_loss(ce_p, ce_n, m_p, m_n, args.alpha)
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step} qid {qid}; stopping, no changes")
            (loss / EFFECTIVE_BATCH).backward()
            ces.append(float(0.5 * (ce_p + ce_n)))
            ranks.append(float(rank))
            gaps.append(float(m_p - m_n))
        lr_used = opt.param_groups[0]["lr"]
        if ctx.get("lr_mult") is not None:          # LR control: the LR actually used by this update = multiplier * the scheduler's LR eta_s (the scheduler recomputes eta from its base after sched.step(), no accumulation)
            eta = lr_used
            planned = ctx["lr_plan"]["eta"][step]
            assert abs(eta - planned) <= 1e-12 + 1e-9 * abs(planned), f"scheduler LR {eta} differs from the planned eta_s {planned} at update {step}"
            lr_used = eta * ctx["lr_mult"](step)
            for g_ in opt.param_groups:
                g_["lr"] = lr_used
        opt.step()
        sched.step()
        row = {"step": step + 1, "ce": sum(ces) / len(ces), "rank": sum(ranks) / len(ranks), "gap": sum(gaps) / len(gaps), "lr": sched.get_last_lr()[0],
               "total_forward_passes": total_fwd[0], "c": c_now, "lr_used": lr_used, "loss": sum(ces) / len(ces) + args.alpha * sum(ranks) / len(ranks)}
        if stream_path is not None:
            with open(stream_path, "a", encoding="utf-8", newline=LF) as f:
                f.write(json.dumps(row) + LF)
        if (step + 1) % args.log_interval == 0 or step + 1 == last_step:
            print(f"[rankm] {tag} step={step + 1}/{ctx['total_steps']} ce={row['ce']:.4f} rank={row['rank']:.4f} gap={row['gap']:.2f} lr={row['lr']:.2e} "
                  f"fwd={total_fwd[0]} elapsed={time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    T.common_args(ap)
    ap.set_defaults(out_root="cpmr_experiment/pilot_rank_margin_k1")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--schedule", choices=["single", "uniform", "anchor", "ratio"], default="single")
    ap.add_argument("--anchor_ratio", type=float, default=None, help="anchor share of --schedule ratio (anchor = --anchor_prompt), e.g. 0.35 or 0.75")
    ap.add_argument("--train_prompt", default="T1", choices=list(IDS), help="the fixed instruction of --schedule single")
    ap.add_argument("--anchor_prompt", default="T1", choices=list(IDS), help="the anchor instruction of --schedule anchor")
    ap.add_argument("--eval_prompts", nargs="+", default=["T1"], choices=list(IDS) + ["R", "C"], help="instructions the final model is evaluated with")
    ap.add_argument("--custom_id", default=None, choices=["R", "C", "I1"], help="id of a custom instruction given with --custom_instruction (schedule single, all queries use it)")
    ap.add_argument("--custom_instruction_file", default=None, help="json {id: text}; the text of --custom_id is used (avoids shell quoting)")
    ap.add_argument("--custom_instruction", default=None, help="full instruction text trained and evaluated under --custom_id; a passage that does not fit raises in the v2 builder")
    ap.add_argument("--eval_years", nargs="+", default=["2019", "2020"])
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--lora_mask", choices=["none", "instruction", "instruction_template", "relevant_word"], default="none", help="instruction: LoRA branch on only at the instruction tokens; instruction_template: also on at the chat-template tokens (header and suffix); training and evaluation")
    ap.add_argument("--word", default=None, help="with --lora_mask relevant_word: the single whole word of the T1 phrase to switch ON (default relevant); e.g. is, even")
    ap.add_argument("--lora_ramp_start", type=float, default=None, help="LoRA strength ramp c: c0 at the first update to 1 at the last (external scalar on the branch output, not learned)")
    ap.add_argument("--diag_queries", type=int, default=64)
    ap.add_argument("--lr_mode", choices=["none", "ramp", "const", "norm"], default="none", help="LR control (experiment log 184/188): ramp = c_s * eta_s, const = k * eta_s with k = sum(c_s eta_s) / sum(eta_s), norm = (c_s / k) * eta_s (same LR sum as the existing schedule); forward strength stays 1")
    ap.add_argument("--lora_ramp_start_for_lr", type=float, default=0.1, help="c0 of c_s used inside --lr_mode (c_s = c0 + (1 - c0) s/(T-1))")
    ap.add_argument("--save_state_at", type=int, default=None, help="save the complete training state after this many completed updates (branch point)")
    ap.add_argument("--state_out", default=None)
    ap.add_argument("--stop_after", type=int, default=None, help="prefix run: stop after this many completed updates (after the state is saved and diagnosed), no final evaluation")
    ap.add_argument("--resume_state", default=None, help="continue from a complete saved state")
    ap.add_argument("--branch_c", choices=["ramp", "one"], default="ramp", help="after a resume: continue the ramp, or set the extra coefficient to 1 from the next update on")
    ap.add_argument("--diag_steps", type=int, nargs="*", default=[], help="extra completed-update counts at which the fixed-sample diagnostic is run")
    ap.add_argument("--const_prefix", choices=["C", "D"], default=None, help="experiment log 195: constant extra coefficient for the first --const_until updates, then 1; C = c0 (0.1), D = the mean of the ramp coefficients of those updates")
    ap.add_argument("--const_until", type=int, default=156)
    ap.add_argument("--no_grad_ckpt", action="store_true", help="disable gradient checkpointing (needed by the per-example mask: recomputation would read the wrong mask)")
    args = ap.parse_args()
    args.branch_step = 1
    ctx = T.build_context(args)
    total = ctx["total_steps"]
    n_q = ctx["n_q"]
    if args.custom_instruction_file is not None:
        file_texts = json.load(open(ROOT / args.custom_instruction_file, encoding="utf-8"))
        args.custom_instruction = file_texts[args.custom_id]
        for k_, v_ in file_texts.items():                # every id in the file can be used as an evaluation instruction, not only the trained one
            ctx["v2"].instructions.setdefault(k_, v_.replace(chr(92) + "n", LF))
    if args.custom_instruction is not None:
        if args.custom_id is None or args.schedule != "single":
            raise SystemExit("--custom_instruction needs --custom_id and --schedule single")
        ctx["v2"].instructions[args.custom_id] = args.custom_instruction.replace(chr(92) + "n", LF)
    if args.schedule == "single" and args.custom_id is not None and args.custom_instruction is not None:
        sched = [args.custom_id] * n_q
    elif args.schedule == "single":
        sched = [args.train_prompt] * n_q
    elif args.schedule == "uniform":
        sched = make_schedule(n_q, args.seed)
    elif args.schedule == "ratio":
        if args.anchor_ratio is None:
            raise SystemExit("--schedule ratio needs --anchor_ratio")
        sched = make_ratio_schedule(n_q, args.seed, args.anchor_prompt, args.anchor_ratio, batch=EFFECTIVE_BATCH)
    else:
        sched = make_anchor_schedule_for(n_q, args.seed, args.anchor_prompt)
        if args.anchor_prompt == "T1":
            assert sched == make_anchor_schedule(n_q, args.seed)
    pos_of = {qid: k for k, qid in enumerate(ctx["order"])}
    inst_span = None
    if args.lora_mask in ("instruction", "instruction_template", "relevant_word"):
        if args.schedule != "single":
            raise SystemExit("--lora_mask instruction is defined for --schedule single")
        import instruction_only_lora as IOL
        instr_text = ctx["v2"].instructions[args.train_prompt]
        g0 = ctx["groups"][ctx["order"][0]]
        ex0 = ctx["builder"](ctx["tok"], instr_text, g0["query"], g0["pos"], "Relevant", args.max_length, enable_thinking=False)
        first0 = next(i for i, t in enumerate(ex0.labels) if t != -100)
        a_, b_, ref_ids, _ = IOL.instruction_range(ctx["tok"], ex0.input_ids[:first0], instr_text)
        if args.lora_mask == "relevant_word":
            a_, b_, ref_ids, cs_ = IOL.word_range(ctx["tok"], ex0.input_ids[:first0], instr_text, word=args.word or IOL.WORD)
            print(f"[rankm] single-word LoRA: one word '{args.word or IOL.WORD}' (chars {cs_}), {b_ - a_} token(s) at position(s) {list(range(a_, b_))}: ids {ref_ids} = {ctx['tok'].convert_ids_to_tokens(ref_ids)}", flush=True)
        inst_span = (a_, b_, ref_ids)
        IOL.STATE["range"] = (a_, b_)
        IOL.install(ctx["model"])
        if args.lora_mask == "instruction_template":
            if not args.no_grad_ckpt:
                raise SystemExit("--lora_mask instruction_template needs --no_grad_ckpt")
            IOL.STATE["range"] = None
            IOL.install_template_mode(ctx["model"], a_, b_)
            print("[rankm] instruction + chat-template mode: ON = positions [0, %d) and the 9 suffix tokens from the first <|im_end|>" % b_, flush=True)
        if args.no_grad_ckpt:
            ctx["model"].gradient_checkpointing_disable()
            print("[rankm] gradient checkpointing disabled", flush=True)
        print(f"[rankm] instruction-only LoRA: ON token positions [{a_}, {b_}) = {b_ - a_} tokens: {ctx['tok'].convert_ids_to_tokens(ref_ids)}", flush=True)
    ctx["ramp"] = None
    if args.lora_ramp_start is not None:
        if args.lora_mask != "none" or args.schedule != "single":
            raise SystemExit("--lora_ramp_start is defined for --schedule single without a position mask")
        import instruction_only_lora as IOL
        IOL.STATE["lora_scale"] = ctx["cfg"]["lora"]["alpha"] / ctx["cfg"]["lora"]["r"]
        IOL.install(ctx["model"])
        if args.const_prefix is not None:
            ramp_vals = [ramp_c(args, total, s_) for s_ in range(args.const_until)]
            c_bar = sum(ramp_vals) / len(ramp_vals)                    # computed from the actual coefficients, not hard-coded
            c_pre = args.lora_ramp_start if args.const_prefix == "C" else c_bar
            ctx["ramp"] = lambda step: c_pre if step < args.const_until else 1.0
            ctx["const_plan"] = {"condition": args.const_prefix, "c_pre": c_pre, "c_bar": c_bar, "until": args.const_until, "sum_ramp_prefix": sum(ramp_vals), "sum_this_prefix": c_pre * args.const_until,
                                 "c": [ctx["ramp"](s_) for s_ in range(total)], "ramp_c": [ramp_c(args, total, s_) for s_ in range(total)]}
            print(f"[rankm] constant-prefix coefficient {args.const_prefix}: c = {c_pre:.12f} for updates 0..{args.const_until - 1}, then 1 for {args.const_until}..{total - 1}; ramp mean over the prefix c_bar = {c_bar:.12f}; "
                  f"coefficient sum over the prefix: ramp {sum(ramp_vals):.9f}, this run {c_pre * args.const_until:.9f}; effective native scale {2 * c_pre:.6f} -> 2", flush=True)
        elif args.resume_state is not None and args.branch_c == "one":
            _bs = torch.load(args.resume_state, map_location="cpu", weights_only=False)["step"]
            ctx["ramp"] = lambda step: ramp_c(args, total, step) if step < _bs else 1.0
        else:
            ctx["ramp"] = lambda step: ramp_c(args, total, step)
        order0 = T.load_query_groups(ROOT / args.train_pairs, 0)[1]          # the diagnostic queries are drawn from the seed-0 order, so every seed uses the same 64 queries
        ctx["diag_set"] = random.Random(20261018).sample(list(order0), args.diag_queries)
        print(f"[rankm] LoRA ramp: c = {args.lora_ramp_start} + {1 - args.lora_ramp_start:.2f} * s/({total}-1), s = 0..{total - 1}; native alpha/r = {IOL.STATE['lora_scale']} stays; c(0) = {ramp_c(args, total, 0):.4f}, c({total - 1}) = {ramp_c(args, total, total - 1):.4f}; "
              f"{len(ctx['diag_set'])} fixed diagnostic training queries", flush=True)
    ctx["lr_mult"] = None
    if args.lr_mode != "none":
        if args.lora_ramp_start is not None or args.lora_mask != "none" or args.schedule != "single":
            raise SystemExit("--lr_mode is defined for --schedule single without a forward ramp or a position mask")
        import instruction_only_lora as IOL
        IOL.STATE["lora_scale"] = ctx["cfg"]["lora"]["alpha"] / ctx["cfg"]["lora"]["r"]
        IOL.install(ctx["model"])
        from transformers import get_linear_schedule_with_warmup
        dummy = torch.zeros(1, requires_grad=True)
        o2 = torch.optim.AdamW([dummy], lr=ctx["cfg"]["optim"]["lr"])
        s2 = get_linear_schedule_with_warmup(o2, ctx["num_warmup"], total)
        eta = []
        for s_ in range(total):                         # eta_s = the LR the optimizer holds at update s (before the scheduler step of that update)
            eta.append(o2.param_groups[0]["lr"])
            o2.step()
            s2.step()
        cs = [args.lora_ramp_start_for_lr + (1 - args.lora_ramp_start_for_lr) * s_ / (total - 1) for s_ in range(total)]
        k = sum(c_ * e_ for c_, e_ in zip(cs, eta)) / sum(eta)
        mult = {"ramp": (lambda st: cs[st]), "const": (lambda st: k), "norm": (lambda st: cs[st] / k)}[args.lr_mode]
        ctx["lr_mult"] = mult
        used = [e_ * mult(i_) for i_, e_ in enumerate(eta)]
        ctx["lr_plan"] = {"eta": eta, "c": cs, "k": k, "mode": args.lr_mode, "sum_eta": sum(eta), "sum_c_eta": sum(c_ * e_ for c_, e_ in zip(cs, eta)), "sum_used": sum(used), "peak_eta": max(eta), "peak_used": max(used),
                          "eta_0": eta[0], "eta_last": eta[-1], "warmup_steps": ctx["num_warmup"], "optimizer": "AdamW(default betas, eps, weight_decay=0.01), no gradient clipping"}
        ctx["diag_set"] = random.Random(20261018).sample(list(T.load_query_groups(ROOT / args.train_pairs, 0)[1]), args.diag_queries)
        print(f"[rankm] LR control {args.lr_mode}: eta_s from the scheduler (peak {max(eta):.3e} at warmup end, eta_0 = {eta[0]:.3e}, eta_624 = {eta[-1]:.3e}); sum eta = {sum(eta):.6e}; sum c_s*eta_s = {sum(c_ * e_ for c_, e_ in zip(cs, eta)):.6e}; "
              f"k = {k:.6f}; planned sum of used LR = {sum(used):.6e}; forward strength 1 (no ramp)", flush=True)
    counts = {}
    by_text = {t: k for k, t in ctx["v2"].instructions.items()}
    inner = ctx["builder"]

    def counting_builder(tokenizer, instruction, *a, **kw):
        key = by_text.get(instruction, "UNKNOWN")
        counts[key] = counts.get(key, 0) + 1
        ex = inner(tokenizer, instruction, *a, **kw)
        if inst_span is not None:
            assert ex.input_ids[inst_span[0]:inst_span[1]] == inst_span[2], "instruction span differs in a built example"
        return ex
    ctx["builder"] = counting_builder
    ctx["instr_for"] = lambda qid: ctx["v2"].instructions[sched[pos_of[qid]]]
    print(f"[rankm] schedule {args.schedule}: " + ", ".join(f"{i} {sched.count(i)}" for i in list(IDS) + ["R", "C"] if sched.count(i)))
    out = ROOT / args.out_root
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stream = log_dir / f"steps_{args.tag}.jsonl"
    if stream.exists() and not args.debug_max_steps:
        stream.unlink()
    print(f"[rankm] K=1 CE + triplet loss from the same forwards, alpha={args.alpha}, {ctx['n_q']} query groups, {total} steps")
    milestones = sorted({max(1, round(f * total)): (pct, f) for pct, f in FRACS}.items())
    last = total if not args.debug_max_steps else min(total, args.debug_max_steps)
    fwd, prev = [0], 0
    diag_path = log_dir / f"diag_{args.tag}.jsonl"
    cfun = ctx["ramp"] if ctx["ramp"] is not None else (lambda st: 1.0)
    if args.resume_state is not None:
        st_ = load_full_state(ctx, args, args.resume_state)
        prev = st_["step"]
        fwd[0] = st_["fwd"]
        ctx["ba_ref"] = {}
        for n, m in ctx["model"].named_modules():
            if n.endswith(".lora_B.default"):
                ctx["ba_ref"][n[:-len(".lora_B.default")]] = (ctx["model"].get_submodule(n.replace("lora_B", "lora_A")).weight.detach().float().clone(), m.weight.detach().float().clone())
        print(f"[rankm] resumed the complete state after {prev} updates ({args.resume_state}); branch {args.branch_c}: next update {prev} uses c = {cfun(prev):.5f}; AdamW step counter "
              f"{[int(v['step']) for v in ctx['opt'].state.values()][:1]}, scheduler last_epoch {ctx['sched'].last_epoch}, lr {ctx['sched'].get_last_lr()[0]:.4e}", flush=True)
    if ctx.get("diag_set") is not None and not args.debug_max_steps:
        if args.lr_mode != "none":
            write_json(log_dir / f"lr_plan_{args.tag}.json", ctx["lr_plan"])
        if ctx.get("const_plan") is not None:
            write_json(log_dir / f"const_plan_{args.tag}.json", ctx["const_plan"])
        if diag_path.exists():
            diag_path.unlink()
        if args.resume_state is not None:
            diag_pass(ctx, args, f"{args.tag}_branch{prev}", prev, cfun(prev - 1), cfun(prev), diag_path)    # the same weights evaluated with c_last and with c = 1
        else:
            save_adapter(ctx, out / "checkpoints" / f"{args.tag}_t0", f"{args.tag}_t0", 0, args.seed, total, args.alpha)
            diag_pass(ctx, args, f"{args.tag}_t0", 0, cfun(0), cfun(0), diag_path)          # step 0: strength 0.1
    stops = {m_: pct_ for m_, (pct_, _) in milestones}
    for extra in list(args.diag_steps) + ([args.save_state_at] if args.save_state_at else []):
        stops.setdefault(extra, None)
    for mstep, pct in sorted(stops.items()):
        if mstep <= prev:
            continue
        upto = min(mstep, last)
        if upto > prev:
            run_steps(ctx, args, prev, upto, None if args.debug_max_steps else stream, fwd, args.tag)
            prev = upto
        if args.debug_max_steps or upto != mstep:
            continue
        if args.save_state_at == mstep:
            save_full_state(ctx, args, ROOT / args.state_out, mstep, fwd)
            print(f"[rankm] saved the complete training state after {mstep} updates -> {args.state_out}", flush=True)
        if pct is None:
            if ctx.get("diag_set") is not None:
                diag_pass(ctx, args, f"{args.tag}_u{mstep}", mstep, cfun(mstep - 1), cfun(mstep) if mstep < total else None, diag_path)
            if args.stop_after == mstep:
                return 0
            continue
        save_adapter(ctx, out / "checkpoints" / f"{args.tag}_t{pct}", f"{args.tag}_t{pct}", mstep, args.seed, total, args.alpha)
        print(f"[rankm] saved checkpoint {args.tag}_t{pct} step={mstep}/{total}", flush=True)
        if ctx.get("diag_set") is not None:
            diag_pass(ctx, args, f"{args.tag}_t{pct}", mstep, cfun(mstep - 1), cfun(mstep) if mstep < total else None, diag_path)
        if args.stop_after == mstep:
            return 0
    print(f"[rankm] total forwards {fwd[0]}; training builds per instruction {counts}", flush=True)
    if args.debug_max_steps:
        return 0
    write_json(log_dir / f"rankm_stats_{args.tag}.json", {"tag": args.tag, "seed": args.seed, "alpha": args.alpha, "total_forwards": fwd[0],
                                                          "schedule": args.schedule, "train_prompt": args.train_prompt, "custom_id": args.custom_id, "custom_instruction": args.custom_instruction, "anchor_prompt": args.anchor_prompt, "anchor_ratio": args.anchor_ratio, "training_builds_per_instruction": counts, "lora_mask": args.lora_mask, "word": args.word or (IOL.WORD if args.lora_mask == "relevant_word" else None), "instruction_on_span": None if inst_span is None else [inst_span[0], inst_span[1]], "lora_ramp_start": args.lora_ramp_start, "lr_mode": args.lr_mode, "lr_k": (ctx["lr_plan"]["k"] if ctx.get("lr_mult") is not None else None),
                                                          "note": "one forward per document; first-token margin triplet loss from the same forwards"})
    if ctx["ramp"] is not None:
        import instruction_only_lora as IOL
        IOL.STATE["scale"] = 1.0                 # final evaluation at c = 1 (the strength of the last update)
        print("[rankm] final evaluation at c = 1.0", flush=True)
    if not args.skip_eval:
        from src.eval_rerank_mistral import evaluate_rerank_mistral
        for pid in args.eval_prompts:
            for year in args.eval_years:
                res = evaluate_rerank_mistral(ctx["model"], ctx["tok"], ctx["v2"].instructions[pid], args.max_length, ctx["device"],
                                              save_path=str(log_dir / f"eval_{args.tag}_t100_{pid}_trec{year}_scores.json"), year=year, enable_thinking=False)
                print(f"[eval] {args.tag} {pid} TREC-DL {year}: nDCG@10={res['ndcg@10']:.4f} MRR@10={res['mrr@10']:.4f} n_queries={res['n_queries']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
