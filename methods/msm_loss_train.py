"""MS MARCO four-loss pilot trainer (experiment log 176; Qwen3-4B, MS MARCO T1, seeds 0/1/2; second server). Same data, order, model, LoRA, optimizer, scheduler, input builder (V2 path, max_length 384) and gradient-checkpointing config as the original
K=1 trainer methods/train_rank_margin_k1.py (schedule single, T1 for every query group): 5,000 frozen query groups (one positive, one negative passage), 10,000 sequences, 8 groups = 16 sequences per update, 625 updates, warm-up round(.1 * 625) = 62, AdamW lr 2e-5, no clipping.
--loss A : the original loss, unchanged: 0.5 (CE_pos + CE_neg) + 1.0 * softplus(-(m_pos - m_neg)), m = first-token logit margin; forwards: the gold label of each document only (HF loss, bf16 logits).
--loss B : 0.5 (CE_pos + CE_neg) per group (ranking term removed, same token CE reduction).
--loss C : label BCE: mean over the 16 sequences of softplus(m) - y m, m = s_Relevant - s_Irrelevant, s_label = sum of log P(label tokens + terminator) (teacher forced, both labels, FP32 final norm / head / log_softmax of the evaluation path).
--loss D : soft kappa only: p = sigmoid(m); r = mean(y), s = mean(p), a = mean(y p), U = r + s - 2a, V = r + s - 2 r s, loss = U / clamp_min(V, 1e-6) over the 16 sequences of the update (exact two-pass computation, dropout RNG restored; the denominator keeps its gradient).
Every update has 8 positive and 8 negative sequences, so r = 0.5 and V = 0.5 exactly: D equals 2 (1 - soft accuracy) (audited per update: r_audit).
    CUDA_VISIBLE_DEVICES=0 python methods/msm_loss_train.py --loss D --seed 0 --local_files_only [--check N]"""
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
from esci_loss_lib import batch_diagnostics, get_rng, grad_wrt_m, loss_from_m, set_rng  # noqa: E402
from train_rank_margin_k1 import forward_ce_and_margin, triplet_loss  # noqa: E402

EB = T.EFFECTIVE_BATCH           # 8 query groups per update


def seq_logprob_sums(core, norm, w_head, store, seqs, pad_id, device):
    """seqs: list of (input_ids, labels) of the V2 builder (labels -100 outside the supervised span = label tokens + terminator); returns sum of log P over the supervised tokens (FP32 norm / head / log_softmax at the supervised positions)."""
    mx = max(len(i) for i, _ in seqs)
    ids = torch.tensor([i + [pad_id] * (mx - len(i)) for i, _ in seqs], device=device)
    att = torch.tensor([[1] * len(i) + [0] * (mx - len(i)) for i, _ in seqs], device=device)
    core(input_ids=ids, attention_mask=att, use_cache=False)           # the pre-norm hook fills store["h"] (graph kept)
    H = store["h"]
    out = []
    for j, (i_, l_) in enumerate(seqs):
        pos = [p for p, t in enumerate(l_) if t != -100]
        y = norm(H[j, [p - 1 for p in pos]].float())
        lp = torch.log_softmax(y @ w_head.T, dim=-1)
        tgt = torch.tensor([i_[p] for p in pos], device=device)
        out.append(lp.gather(1, tgt.unsqueeze(-1)).squeeze(-1).sum())
    return torch.stack(out)


def group_margins(ctx, core, norm, w_head, store, text, g, pad_id):
    """(m_pos, m_neg) of one query group: both labels teacher forced for the positive and for the negative document"""
    b = ctx["builder"]
    seqs = []
    for doc in (g["pos"], g["neg"]):
        for label in ("Relevant", "Irrelevant"):
            ex = b(ctx["tok"], text, g["query"], doc, label, 384, enable_thinking=False)
            seqs.append((list(ex.input_ids), list(ex.labels)))
    s = seq_logprob_sums(core, norm, w_head, store, seqs, pad_id, ctx["device"])
    return torch.stack([s[0] - s[1], s[2] - s[3]])


def main():
    ap = argparse.ArgumentParser()
    T.common_args(ap)
    ap.set_defaults(out_root="cpmr_experiment/msm_loss")
    ap.add_argument("--loss", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--check", type=int, default=0, help="N updates, nothing durable")
    args = ap.parse_args()
    args.branch_step = 1
    ctx = T.build_context(args)
    total, model, tok, opt, sched, device = ctx["total_steps"], ctx["model"], ctx["tok"], ctx["opt"], ctx["sched"], ctx["device"]
    assert total == 625 and EB == 8, (total, EB)
    text = ctx["v2"].instructions["T1"]
    inner = model.base_model.model
    core, norm = inner.model, inner.model.norm
    store = {}
    norm.register_forward_pre_hook(lambda mod, a: store.__setitem__("h", a[0]))
    w_head = inner.lm_head.weight.detach().float()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tag = args.tag if getattr(args, "tag", None) else f"msm_{args.loss}_s{args.seed}"
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
        torch.save({"adapter_state": st, "tag": tag, "model": ctx["cfg"]["model_name"], "condition": f"msm-loss-{args.loss}", "global_step": step, "frac": step / total, "seed": args.seed, "loss": args.loss}, ckd / name)
    print(f"[msm-loss] {tag}: loss {args.loss}, {ctx['n_q']} query groups, {total} updates, warm-up {ctx['num_warmup']}, T1 sha {hashlib.sha256(text.encode()).hexdigest()[:12]}, gradient checkpointing {ctx['cfg'].get('gradient_checkpointing')}", flush=True)
    if not args.check:
        save(0, "ckpt_step0.pt")
    t0 = time.time()
    n_fwd = n_skip = n_nonfinite = 0
    for step in range(args.check if args.check else total):
        opt.zero_grad(set_to_none=True)
        qids = ctx["order"][step * EB:(step + 1) * EB]
        groups = [ctx["groups"][q] for q in qids]
        row = {"step": step + 1}
        if args.loss in ("A", "B"):
            ces, ranks, gaps = [], [], []
            for g in groups:
                ce_p, m_p = forward_ce_and_margin(model, tok, text, g["query"], g["pos"], "Relevant", args.max_length, device, ctx["rel_id"], ctx["irr_id"], ctx["builder"])
                ce_n, m_n = forward_ce_and_margin(model, tok, text, g["query"], g["neg"], "Irrelevant", args.max_length, device, ctx["rel_id"], ctx["irr_id"], ctx["builder"])
                n_fwd += 2
                if args.loss == "A":
                    loss, rank = triplet_loss(ce_p, ce_n, m_p, m_n, args.alpha)
                else:
                    loss, rank = 0.5 * (ce_p + ce_n), F.softplus(-(m_p - m_n))
                if not torch.isfinite(loss):
                    raise SystemExit(f"non-finite loss at update {step + 1}")
                (loss / EB).backward()
                ces.append(float(0.5 * (ce_p + ce_n))); ranks.append(float(rank)); gaps.append(float(m_p - m_n))
            row.update({"ce": sum(ces) / EB, "rank": sum(ranks) / EB, "gap": sum(gaps) / EB})
            row["loss"] = row["ce"] + (args.alpha * row["rank"] if args.loss == "A" else 0.0)
        elif args.loss == "C":
            ms = []
            for g in groups:
                m = group_margins(ctx, core, norm, w_head, store, text, g, pad_id)
                yy = torch.tensor([1.0, 0.0], device=device)
                ((F.softplus(m) - yy * m).sum() / (2 * EB)).backward()
                ms.append(m.detach()); n_fwd += 4
            mvec = torch.cat(ms)
            y = torch.tensor([1.0, 0.0] * EB, device=device)
            row["loss"] = float((F.softplus(mvec) - y * mvec).mean())
            row.update(batch_diagnostics("softk", mvec, y))
        else:
            y = torch.tensor([1.0, 0.0] * EB, device=device)
            states, ms = [], []
            with torch.no_grad():
                for g in groups:
                    states.append(get_rng())
                    ms.append(group_margins(ctx, core, norm, w_head, store, text, g, pad_id)); n_fwd += 4
            mvec = torch.cat(ms)
            row.update(batch_diagnostics("softk", mvec, y))
            assert abs(row["r"] - 0.5) < 1e-9, f"batch with r = {row['r']}"
            gvec, lval = grad_wrt_m("softk", mvec, y)
            row["loss"] = lval
            row["loss_simplified"] = float(2 * (1 - (y * torch.sigmoid(mvec) + (1 - y) * (1 - torch.sigmoid(mvec))).mean()))        # 2 (1 - soft accuracy) at r = .5
            dmax = 0.0
            for k, (g, st) in enumerate(zip(groups, states)):
                set_rng(st)
                m2 = group_margins(ctx, core, norm, w_head, store, text, g, pad_id)
                dmax = max(dmax, float((m2.detach() - mvec[2 * k:2 * k + 2]).abs().max()))
                (gvec[2 * k:2 * k + 2] * m2).sum().backward(); n_fwd += 4
            row["max_pass_diff"] = dmax
        if not math.isfinite(row["loss"]):
            n_nonfinite += 1
            raise SystemExit(f"non-finite loss at update {step + 1}")
        params = [p for p in model.parameters() if p.requires_grad]
        gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)))
        if not math.isfinite(gn):
            n_nonfinite += 1
            raise SystemExit(f"non-finite gradient norm at update {step + 1}")
        row["lr_used"] = opt.param_groups[0]["lr"]
        opt.step(); sched.step()
        row.update({"lr": sched.get_last_lr()[0], "grad_norm": gn, "sequences_forwarded": n_fwd})
        if not args.check:
            with open(steps_f, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row) + "\n")
        if (step + 1) % 25 == 0 or args.check or step + 1 == total:
            print(f"[msm-loss] {tag} update {step + 1}/{total} loss {row['loss']:.4f} grad_norm {gn:.3f} " + (f"p_sat {row.get('p_sat', float('nan')):.2f} " if 'p_sat' in row else f"rank {row.get('rank', 0):.3f} gap {row.get('gap', 0):.2f} ") + f"max_pass_diff {row.get('max_pass_diff', 0):.1e} elapsed {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
    dt = time.time() - t0
    if args.check:
        bmax = max(float(p.detach().abs().max()) for n, p in model.named_parameters() if "lora_B" in n)
        print(f"[check] {args.loss}: {args.check} updates, max |lora_B| {bmax:.3e}, {dt / args.check:.2f} s/update, sequences {n_fwd}, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
        return 0
    save(total, "ckpt_final.pt")
    stats = {"tag": tag, "seed": args.seed, "loss": args.loss, "updates": total, "sequences_forwarded": n_fwd, "nonfinite": n_nonfinite, "train_seconds": dt, "peak_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "order_sha256": hashlib.sha256(json.dumps(list(map(str, ctx["order"]))).encode()).hexdigest(),
             "instruction_sha256": hashlib.sha256(text.encode()).hexdigest(), "torch": torch.__version__, "gradient_checkpointing": ctx["cfg"].get("gradient_checkpointing"), "train_pairs": args.train_pairs}
    (logd / f"stats_{tag}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"[msm-loss] {tag} done {dt:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
