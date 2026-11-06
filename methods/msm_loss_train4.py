"""MS MARCO CE / CE + soft kappa / CE + AnyLoss-like kappa trainer for the instructions T1-T7 (experiment log 211, Qwen3-4B, second server). Derived from msm_loss_train2.py (experiment log 178; imported helpers unchanged) with exactly these changes:
(1) --instr T1..T7 (T1-T5 from instructions_v2.json role TRAIN; T6, T7 = the custom instructions of experiment log 055 read from --custom_instruction_file), the instruction text is passed to the unchanged builder; (2) --loss B = L_CE only (0.5 (CE_pos + CE_neg) per group, the loss of condition B of 176/177, computed by the same ce_rank_part code with the ranking term off),
--loss E = L_CE + lam L_K (soft kappa, unchanged), --loss M = L_CE + lam L_AK (AnyLoss-like kappa, q = sigmoid(10 (p - .5)), alpha 10 fixed, lam = 1.0); (3) the tag / output names carry the instruction. Everything else (data, order, model, LoRA, optimizer, scheduler, input builder, gradient checkpointing, exact two-pass kappa part with dropout RNG restore) is the code of 177/178.
Original docstring of msm_loss_train2.py follows.
MS MARCO additional conditions of the four-loss pilot (experiment log 178; same server, data, model, order, optimizer, config as msm_loss_train.py of experiment log 177, which is imported and not modified):
--loss E : L_CE + lam * L_K            (L_CE = mean over the 8 groups of 0.5 (CE_pos + CE_neg), the loss of condition B)
--loss F : L_CE + L_rank + lam * L_K   (L_CE + L_rank is the loss of condition A; L_rank = mean over the 8 groups of softplus(-(m_pos - m_neg)), first-token margins, coefficient 1)
--loss G : L_AK alone (AnyLoss-like kappa, alpha 10: q = sigmoid(10 (p - .5)), p = sigmoid(m); the same formula as condition D with q instead of p)
L_K = U / clamp_min(V, 1e-6) over the 16 sequences of the update, p = sigmoid(m), m = log P(Relevant + terminator) - log P(Irrelevant + terminator) (teacher forced, FP32 head), denominator gradient kept; lam = 1.0 fixed beforehand (no search, no normalisation by the number of terms).
Per update: (1) the CE (and ranking) part with the gold-label forwards of conditions A / B, backward per group; (2) the kappa part by the exact two-pass scheme (pass 1 without grad -> 16 margins -> dL_K/dm, pass 2 per group with the dropout RNG state restored); the CE forwards are independent forwards with their own
dropout draws, and after pass 2 the RNG is set to the state after pass 1 so that the stream continues as if pass 1 had run once. The two parts are added in the gradient buffer exactly once each.
    CUDA_VISIBLE_DEVICES=0 python methods/msm_loss_train2.py --loss F --seed 0 --tag msm_F_s0 --config configs/qwen3_4b_msm_loss.yaml --out_root cpmr_experiment/msm_loss2 --local_files_only [--check N] [--lam 1.0]"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
for rel in ("", "methods", "evaluation", "cpmr_experiment/scripts", "cpmr_experiment/instructions_v2/scripts"):
    sys.path.insert(0, str(ROOT / rel))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_apocf_online as T  # noqa: E402
from esci_loss_lib import batch_diagnostics, get_rng, grad_wrt_m, set_rng  # noqa: E402
from msm_loss_train import EB, group_margins  # noqa: E402
from train_rank_margin_k1 import forward_ce_and_margin  # noqa: E402


def ce_rank_part(ctx, groups, text, with_rank, scale=1.0):
    """backward of scale * (L_CE [+ L_rank]); returns the component values (floats) of the update"""
    model, tok, device = ctx["model"], ctx["tok"], ctx["device"]
    ces, ranks, gaps = [], [], []
    for g in groups:
        ce_p, m_p = forward_ce_and_margin(model, tok, text, g["query"], g["pos"], "Relevant", 384, device, ctx["rel_id"], ctx["irr_id"], ctx["builder"])
        ce_n, m_n = forward_ce_and_margin(model, tok, text, g["query"], g["neg"], "Irrelevant", 384, device, ctx["rel_id"], ctx["irr_id"], ctx["builder"])
        rank = F.softplus(-(m_p - m_n))
        loss = 0.5 * (ce_p + ce_n) + (1.0 * rank if with_rank else 0.0)
        if not torch.isfinite(loss):
            raise SystemExit("non-finite CE/rank loss")
        (scale * loss / EB).backward()
        ces.append(float(0.5 * (ce_p + ce_n))); ranks.append(float(rank)); gaps.append(float(m_p - m_n))
    return {"ce": sum(ces) / EB, "rank": sum(ranks) / EB, "gap": sum(gaps) / EB}


def kappa_part(ctx, groups, text, kind, lam, core, norm, w_head, store, pad_id):
    """backward of lam * L_K (kind softk) or of L_AK (kind anyk) by the exact two-pass scheme; returns (value of the kappa loss, diagnostics dict, number of sequences forwarded)"""
    device = ctx["device"]
    y = torch.tensor([1.0, 0.0] * EB, device=device)
    states, ms = [], []
    with torch.no_grad():
        for g in groups:
            states.append(get_rng())
            ms.append(group_margins(ctx, core, norm, w_head, store, text, g, pad_id))
    end_state = get_rng()
    mvec = torch.cat(ms)
    diag = batch_diagnostics("softk" if kind == "softk" else "anyk", mvec, y)
    assert abs(diag["r"] - 0.5) < 1e-9, f"batch with r = {diag['r']}"
    gvec, lval = grad_wrt_m(kind, mvec, y)
    p = torch.sigmoid(mvec)
    qv = p if kind == "softk" else torch.sigmoid(10.0 * (p - 0.5))
    diag["loss_simplified"] = float(2 * (1 - (y * qv + (1 - y) * (1 - qv)).mean()))            # r = .5: U / V = 2 (1 - soft accuracy of q)
    diag["q_mid_frac"] = float(((qv > 0.1) & (qv < 0.9)).float().mean())
    diag["q_outer_frac"] = float(((qv < 0.1) | (qv > 0.9)).float().mean())
    dmax = 0.0
    for k, (g, st) in enumerate(zip(groups, states)):
        set_rng(st)
        m2 = group_margins(ctx, core, norm, w_head, store, text, g, pad_id)
        dmax = max(dmax, float((m2.detach() - mvec[2 * k:2 * k + 2]).abs().max()))
        (lam * gvec[2 * k:2 * k + 2] * m2).sum().backward()
    set_rng(end_state)
    diag["max_pass_diff"] = dmax
    return lval, diag, mvec


def main():
    ap = argparse.ArgumentParser()
    T.common_args(ap)
    ap.set_defaults(out_root="cpmr_experiment/msm_t7")
    ap.add_argument("--loss", required=True, choices=["B", "E", "M", "G"])                       # G = AnyLoss-like kappa alone (the loss G of 178), gate checks only
    ap.add_argument("--instr", default="T1")
    ap.add_argument("--custom_instruction_file", default=None)
    ap.add_argument("--dump_grad", default="", help="test only (with --check 1): save the flattened gradient of the first update and stop before the optimizer step")
    ap.add_argument("--test_b_std", type=float, default=0.0, help="test only: non-zero LoRA B (normal, this std) so that every parameter receives a gradient")
    ap.add_argument("--check_log", default="", help="check mode only: append the full-precision row of every checked update to this file")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--check", type=int, default=0, help="N updates, nothing durable")
    args = ap.parse_args()
    args.branch_step = 1
    ctx = T.build_context(args)
    if args.test_b_std > 0:
        torch.manual_seed(123)
        for n_, p_ in ctx["model"].named_parameters():
            if "lora_B" in n_:
                torch.nn.init.normal_(p_, std=args.test_b_std)
        torch.manual_seed(args.seed)
    total, model, tok, opt, sched, device = ctx["total_steps"], ctx["model"], ctx["tok"], ctx["opt"], ctx["sched"], ctx["device"]
    assert total == 625 and EB == 8, (total, EB)
    if args.custom_instruction_file:
        for k_, v_ in json.load(open(ROOT / args.custom_instruction_file, encoding="utf-8")).items():
            ctx["v2"].instructions[k_] = v_
    text = ctx["v2"].instructions[args.instr]
    inner = model.base_model.model
    core, norm = inner.model, inner.model.norm
    store = {}
    norm.register_forward_pre_hook(lambda mod, a: store.__setitem__("h", a[0]))
    w_head = inner.lm_head.weight.detach().float()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tag = args.tag
    out = ROOT / args.out_root
    logd, ckd = out / "logs", out / "checkpoints" / tag
    if not args.check:
        logd.mkdir(parents=True, exist_ok=True); ckd.mkdir(parents=True, exist_ok=True)
        assert not (ckd / "ckpt_final.pt").exists(), f"{tag} already done"
    steps_f = logd / f"steps_{tag}.jsonl"
    if steps_f.exists() and not args.check:
        steps_f.unlink()

    def save(step, name):
        st = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        torch.save({"adapter_state": st, "tag": tag, "model": ctx["cfg"]["model_name"], "condition": f"msm-loss-{args.loss}", "instruction": args.instr, "global_step": step, "frac": step / total, "seed": args.seed, "loss": args.loss, "lam": args.lam}, ckd / name)
    print(f"[msm-loss2] {tag}: loss {args.loss}, lam {args.lam}, {ctx['n_q']} query groups, {total} updates, warm-up {ctx['num_warmup']}, {args.instr} sha {hashlib.sha256(text.encode()).hexdigest()[:12]}", flush=True)
    if not args.check:
        save(0, "ckpt_step0.pt")
    t0 = time.time()
    n_seq = n_nonfinite = 0
    kind = "anyk" if args.loss in ("M", "G") else "softk"
    for step in range(args.check if args.check else total):
        opt.zero_grad(set_to_none=True)
        qids = ctx["order"][step * EB:(step + 1) * EB]
        groups = [ctx["groups"][q] for q in qids]
        row = {"step": step + 1}
        parts = {"ce": 0.0, "rank": 0.0, "gap": 0.0}
        parts = {"ce": 0.0, "rank": 0.0, "gap": 0.0}
        if args.loss != "G":
            parts = ce_rank_part(ctx, groups, text, with_rank=False)
            n_seq += 2 * EB
        if args.loss == "B":
            lk, diag = 0.0, {}
        else:
            lk, diag, mvec = kappa_part(ctx, groups, text, kind, args.lam, core, norm, w_head, store, pad_id)
            n_seq += 4 * EB * 2
        row.update(diag)
        row.update({"ce": parts["ce"], "rank": parts["rank"], "gap": parts["gap"], "kappa_loss": lk})
        row["loss"] = parts["ce"] + (0.0 if args.loss == "B" else args.lam * lk)
        if not math.isfinite(row["loss"]):
            n_nonfinite += 1
            raise SystemExit(f"non-finite loss at update {step + 1}")
        params = [p for p in model.parameters() if p.requires_grad]
        gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)))
        if not math.isfinite(gn):
            n_nonfinite += 1
            raise SystemExit(f"non-finite gradient norm at update {step + 1}")
        if args.dump_grad:
            torch.save({"grad": torch.cat([p.grad.detach().float().flatten() if p.grad is not None else torch.zeros(p.numel(), device=p.device) for p in params]).cpu(), "row": row}, args.dump_grad)
            print(f"[dump] {args.loss}: gradient norm {gn:.6e} -> {args.dump_grad}")
            return 0
        row["lr_used"] = opt.param_groups[0]["lr"]
        opt.step(); sched.step()
        row.update({"lr": sched.get_last_lr()[0], "grad_norm": gn, "sequences_forwarded": n_seq})
        if not args.check:
            with open(steps_f, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row) + "\n")
        if args.check and args.check_log:
            with open(args.check_log, "a", encoding="utf-8", newline="\n") as f_:
                f_.write(json.dumps(row) + "\n")
        if (step + 1) % 25 == 0 or args.check or step + 1 == total:
            print(f"[msm-loss2] {tag} update {step + 1}/{total} loss {row['loss']:.4f} (ce {row['ce']:.3f} rank {row['rank']:.3f} kappa {lk:.3f}) grad_norm {gn:.3f} p_sat {row.get('p_sat', float('nan')):.2f} max_pass_diff {row.get('max_pass_diff', 0.0):.1e} elapsed {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
    dt = time.time() - t0
    if args.check:
        bmax = max(float(p.detach().abs().max()) for n, p in model.named_parameters() if "lora_B" in n)
        print(f"[check] {args.loss}: {args.check} updates, max |lora_B| {bmax:.3e}, {dt / args.check:.2f} s/update, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
        return 0
    save(total, "ckpt_final.pt")
    stats = {"tag": tag, "seed": args.seed, "loss": args.loss, "lam": args.lam, "updates": total, "sequences_forwarded": n_seq, "nonfinite": n_nonfinite, "train_seconds": dt, "peak_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
             "order_sha256": hashlib.sha256(json.dumps(list(map(str, ctx["order"]))).encode()).hexdigest(), "instruction": args.instr, "instruction_sha256": hashlib.sha256(text.encode()).hexdigest(), "torch": torch.__version__, "gradient_checkpointing": ctx["cfg"].get("gradient_checkpointing")}
    (logd / f"stats_{tag}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"[msm-loss2] {tag} done {dt:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
