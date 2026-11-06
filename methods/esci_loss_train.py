"""Four-loss pilot trainer (experiment log 169; Qwen3-4B Base, ESCI-T1, seed 0; same recipe as esci_train.py of 165): --loss ce | bce | softk | anyk.
ce   : esci_core.ce_rows (identical to esci_train.py: token mean of the correct label + <|im_end|>, mean over the rows).
bce  : mean over the 16 rows of softplus(m) - y m, m = s_plus - s_minus (sums of log P(label + <|im_end|>), both labels teacher forced, FP32 norm / head / log_softmax).
softk / anyk : one loss over the 16 rows (loss_from_m); exact two-pass computation (pass 1 no grad -> 16 margins -> dL/dm; pass 2 per micro-batch with grad and the restored dropout RNG state). Single-class updates (r = 0 or 1): loss / gradient 0, optimizer.step skipped
(scheduler still advances: the 625 slots are fixed beforehand); the rows count as exposed.
All else as 165: 10,000 frozen rows, 1 epoch, effective batch 16 rows, B = 625, warm-up 63, checkpoints 63/157/313/469/625 (+0), AdamW lr 2e-5, LoRA r8 / alpha 16 / dropout .05 on q/k/v/o, torch.manual_seed(seed) before the model is built, order default_rng(seed).permutation.
    CUDA_VISIBLE_DEVICES=0 python methods/esci_loss_train.py --loss softk --instr T1 --seed 0 --instr_file <manifest> --out_root <dir> --local_files_only [--check N]"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
for rel in ("", "methods", "evaluation", "cpmr_experiment/scripts"):
    sys.path.insert(0, str(ROOT / rel))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import get_linear_schedule_with_warmup  # noqa: E402

from esci_core import EOT, PAD, REL_FIRST, IRR_FIRST, build, ce_rows  # noqa: E402
from esci_loss_lib import batch_diagnostics, get_rng, grad_wrt_m, loss_from_m, margins, set_rng  # noqa: E402

EFFECTIVE_BATCH = 16
STEPS_FRAC = (0.1, 0.25, 0.5, 0.75, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", required=True, choices=["ce", "bce", "softk", "anyk"])
    ap.add_argument("--instr", default="T1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instr_file", default="cpmr_experiment/esci_t/esci_t_manifest.json")
    ap.add_argument("--rows", default="cpmr_experiment/esci/audit/sft2_rows.jsonl")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--config", default="configs/qwen3_4b_esci.yaml")
    ap.add_argument("--max_length", type=int, default=0)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--tag_prefix", default="EQ")
    ap.add_argument("--check", type=int, default=0, help="N updates, nothing durable")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()
    from src.model import load_model_and_tokenizer
    cfg = yaml.safe_load(open(ROOT / args.config))
    text = {e["display_id"]: e["full"] for e in json.load(open(ROOT / args.instr_file, encoding="utf-8"))["entries"]}[args.instr]
    rows = [json.loads(l) for l in open(ROOT / args.rows, encoding="utf-8")]
    max_length = args.max_length or json.load(open(ROOT / "cpmr_experiment/esci/audit/splits.json"))["max_length"]
    M = len(rows)
    B = math.ceil(M / EFFECTIVE_BATCH)
    warm = math.ceil(0.1 * B)
    save_steps = sorted({math.ceil(f * B) for f in STEPS_FRAC})
    order = [int(i) for i in np.random.default_rng(args.seed).permutation(M)]
    torch.manual_seed(args.seed)
    lm = load_model_and_tokenizer(model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
                                  gradient_checkpointing=False, local_files_only=args.local_files_only)
    model, tok = lm.model, lm.tokenizer
    model.train()
    inner = model.base_model.model
    core, head = inner.model, inner.lm_head
    norm = core.norm
    device = next(model.parameters()).device
    assert tok("Relevant", add_special_tokens=False)["input_ids"][0] == REL_FIRST and tok("Irrelevant", add_special_tokens=False)["input_ids"][0] == IRR_FIRST
    params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    mods = sorted({n.split(".lora_")[0] for n, p in model.named_parameters() if "lora_" in n})
    assert n_tr == 5898240 and len(mods) == 144, (n_tr, len(mods))
    store = {}
    norm.register_forward_pre_hook(lambda mod, a: store.__setitem__("h", a[0]))          # graph kept (not detached): training path
    w_head = head.weight.detach().float()                                                  # tied embedding matrix in FP32 (frozen)
    opt = torch.optim.AdamW(params, lr=cfg["optim"]["lr"])
    sched = get_linear_schedule_with_warmup(opt, warm, B)
    tag = f"{args.tag_prefix}_{args.loss}_{args.instr}_s{args.seed}"
    out = ROOT / args.out_root
    ckd, logd = out / "checkpoints" / tag, out / "logs"
    if not args.check:
        ckd.mkdir(parents=True, exist_ok=True); logd.mkdir(parents=True, exist_ok=True)
        assert not (ckd / "done.json").exists(), f"{tag} already done"

    def save(step):
        st = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save({"adapter_state": st, "tag": tag, "model": cfg["model_name"], "global_step": step, "frac": step / B, "seed": args.seed, "instruction": args.instr, "loss": args.loss, "alpha": 1.0}, ckd / f"ckpt_step{step}.pt")

    def seqs_of(mb):
        sp, sn = [], []
        shr = 0
        for r in mb:
            pre, resp, s1 = build(tok, text, r["query"], r["product_text"], "Relevant", max_length)
            pre2, resp2, s2 = build(tok, text, r["query"], r["product_text"], "Irrelevant", max_length)
            sp.append((pre, resp, EOT)); sn.append((pre2, resp2, EOT)); shr += int(s1) + int(s2)
        return sp, sn, shr
    print(f"[loss-train] {tag}: {M} rows, {B} updates (last batch {M - EFFECTIVE_BATCH * (B - 1)} rows), warm-up {warm}, checkpoints {save_steps}, max_length {max_length}, micro {args.micro}, loss {args.loss}, parameters {n_tr}, text sha {hashlib.sha256(text.encode()).hexdigest()[:12]}", flush=True)
    if not args.check:
        save(0)
    steps_f = logd / f"steps_{tag}.jsonl"
    if steps_f.exists() and not args.check:
        steps_f.unlink()
    t0 = time.time()
    n_seq = n_tok = n_shr = n_skip = n_clamp = n_nonfinite = n_applied = 0
    for step in range(args.check if args.check else B):
        opt.zero_grad(set_to_none=True)
        idx = order[step * EFFECTIVE_BATCH:(step + 1) * EFFECTIVE_BATCH]
        ng = len(idx)
        y = torch.tensor([1.0 if rows[i]["label"] == "Relevant" else 0.0 for i in idx], device=device)
        mbs = [[rows[j] for j in idx[i:i + args.micro]] for i in range(0, ng, args.micro)]
        row = {"step": step + 1, "rows": ng, "n_relevant": int(y.sum().item())}
        skipped = False
        if args.loss == "ce":
            ces = []
            for mb in mbs:
                ce, shr, ntk = ce_rows(core, head, tok, text, [(r["query"], r["product_text"], r["label"]) for r in mb], max_length, device)
                (ce.sum() / ng).backward(); ces += [float(x) for x in ce]
                n_seq += len(mb); n_tok += ntk; n_shr += shr
            row["loss"] = sum(ces) / ng
        elif args.loss == "bce":
            ms, off = [], 0
            for mb in mbs:
                sp, sn, shr = seqs_of(mb)
                m, _, _ = margins(core, norm, w_head, store, sp, sn, PAD, device)
                yy = y[off:off + len(mb)]
                ((torch.nn.functional.softplus(m) - yy * m).sum() / ng).backward()
                ms.append(m.detach()); n_seq += 2 * len(mb); n_shr += shr; off += len(mb)
            mvec = torch.cat(ms)
            row["loss"] = float((torch.nn.functional.softplus(mvec) - y * mvec).mean())
            row.update(batch_diagnostics("softk", mvec, y))
        else:
            states, ms = [], []
            with torch.no_grad():
                for mb in mbs:
                    sp, sn, shr = seqs_of(mb)
                    states.append(get_rng())
                    m, _, _ = margins(core, norm, w_head, store, sp, sn, PAD, device)
                    ms.append(m); n_seq += 2 * len(mb); n_shr += shr
            mvec = torch.cat(ms)
            row.update(batch_diagnostics(args.loss, mvec, y))
            if row["single_class"]:
                skipped = True                                       # r = 0 or 1: loss / gradient 0, no optimizer step
                row["loss"] = float(loss_from_m(args.loss, mvec, y))
            else:
                g, lval = grad_wrt_m(args.loss, mvec, y)
                row["loss"] = lval
                if row["clamped"]:
                    n_clamp += 1
                dmax, off = 0.0, 0
                for mb, st in zip(mbs, states):
                    sp, sn, _ = seqs_of(mb)
                    set_rng(st)
                    m2, _, _ = margins(core, norm, w_head, store, sp, sn, PAD, device)
                    dmax = max(dmax, float((m2.detach() - mvec[off:off + len(mb)]).abs().max()))
                    (g[off:off + len(mb)] * m2).sum().backward()
                    off += len(mb); n_seq += 2 * len(mb)
                row["max_pass_diff"] = dmax
        if not math.isfinite(row["loss"]):
            n_nonfinite += 1
            raise SystemExit(f"non-finite loss at update {step + 1}")
        lr_used = opt.param_groups[0]["lr"]
        if skipped:
            n_skip += 1; gn = 0.0
        else:
            gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)))
            if not math.isfinite(gn):
                n_nonfinite += 1
                raise SystemExit(f"non-finite gradient norm at update {step + 1}")
            opt.step(); n_applied += 1
        sched.step()
        row.update({"skipped": skipped, "lr_used": lr_used, "lr": sched.get_last_lr()[0], "grad_norm": gn})
        if not args.check:
            with open(steps_f, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row) + "\n")
        if (step + 1) % 25 == 0 or args.check or step + 1 == B:
            print(f"[loss-train] {tag} update {step + 1}/{B} loss {row['loss']:.4f} grad_norm {gn:.3f} " + (f"p_mean {row.get('p_mean', float('nan')):.3f} q_sat {row.get('q_sat', float('nan')):.2f} " if 'p_mean' in row else "") +
                  f"max_pass_diff {row.get('max_pass_diff', 0):.1e} skipped {skipped} elapsed {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
        if not args.check and (step + 1) in save_steps:
            t1 = time.time(); save(step + 1); print(f"[loss-train] saved update {step + 1} ({time.time() - t1:.1f}s)", flush=True)
    dt = time.time() - t0
    n_up = args.check if args.check else B
    if args.check:
        bmax = max(float(p.detach().abs().max()) for n, p in model.named_parameters() if "lora_B" in n)
        print(f"[check] {args.loss}: after {n_up} updates max |lora_B| = {bmax:.3e}, {dt / n_up:.2f} s/update, sequences scored {n_seq}, shortened {n_shr}, skipped {n_skip}, clamped {n_clamp}, peak memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
        return 0
    stats = {"tag": tag, "seed": args.seed, "instruction": args.instr, "loss": args.loss, "updates_slots": B, "optimizer_updates": n_applied, "skipped_updates": n_skip, "clamped_updates": n_clamp, "nonfinite": n_nonfinite,
             "rows_exposed": M, "sequences_forwarded": n_seq, "inputs_shortened": n_shr, "train_seconds": dt, "peak_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "save_steps": save_steps, "warmup_updates": warm, "max_length": max_length, "micro_batch": args.micro,
             "rows_file_sha256": hashlib.sha256((ROOT / args.rows).read_bytes()).hexdigest(), "order_sha256": hashlib.sha256(np.array(order, dtype=np.int64).tobytes()).hexdigest(), "torch": torch.__version__}
    (logd / f"stats_{tag}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    (ckd / "done.json").write_text(json.dumps({"done": True, "seconds": dt}), encoding="utf-8")
    print(f"[loss-train] {tag} done {dt:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
