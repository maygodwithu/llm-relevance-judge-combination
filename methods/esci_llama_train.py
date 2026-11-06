"""ESCI-T SFT (experiment log 166; Llama-3.1-8B-Instruct, second server), CE-ONLY, one fixed relevance instruction (original P0-P4 or a selected APO-CF instruction).
Loss (all instructions, all seeds): CE = token mean over the supervised tokens (answer 'Relevant' / 'Irrelevant' tokens + <|eot_id|>) of ONE row; a batch loss is the mean of the CE of its rows (equal weight per row). No ranking term: the frozen rows are a variable-size
set of E / S / C / I products per query, the existing trainer takes forced positive-negative groups, and building pairs would need a new pair rule / weighting (156: CE-only is the default in that case).
Rows: every row of cpmr_experiment/esci/audit/sft_rows.jsonl (<= 20 products of each of 5,000 frozen queries), each exposed ONCE (1 epoch) in a seed-specific order (numpy default_rng(seed) permutation of all rows, shared by every instruction of that seed),
effective batch 16 rows = ceil(M/16) = B updates (the last update has M - 16 (B-1) rows, loss averaged over the actual rows, no duplicated or dropped row); warm-up ceil(0.1 B) updates, linear decay to 0; checkpoints at ceil(f B), f = .1 .25 .5 .75 1 (+ step 0).
Micro-batches of --micro rows (right padded, same attention mask) are accumulated to the 16-row update. AdamW (default betas / eps, weight decay 0.01) lr 2e-5, no clipping, bf16 base, LoRA r8 / alpha 16 / dropout .05 on q/k/v/o (128 modules, 6,815,744 parameters),
torch.manual_seed(seed) before the model is built -> the same LoRA A initialisation for every instruction of a seed.
    CUDA_VISIBLE_DEVICES=0 python methods/esci_train.py --instr P1 --seed 0 --instr_file <manifest> --out_root <dir> --local_files_only [--check]"""
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

from esci_llama_core import ce_rows, REL_FIRST, IRR_FIRST  # noqa: E402

EFFECTIVE_BATCH = 16
STEPS_FRAC = (0.1, 0.25, 0.5, 0.75, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instr", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--instr_file", required=True)
    ap.add_argument("--rows", default="cpmr_experiment/esci/audit/sft2_rows.jsonl")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--config", default="configs/llama31_8b_esci.yaml")
    ap.add_argument("--max_length", type=int, default=0, help="0 = read from cpmr_experiment/esci_llama/audit/max_length.json")
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--tag_prefix", default="EL")
    ap.add_argument("--check", type=int, default=0, help="N updates, nothing durable")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()
    from src.model import load_model_and_tokenizer
    cfg = yaml.safe_load(open(ROOT / args.config))
    ins = {e["display_id"]: e["full"] for e in json.load(open(ROOT / args.instr_file, encoding="utf-8"))["entries"]}
    text = ins[args.instr]
    rows = [json.loads(l) for l in open(ROOT / args.rows, encoding="utf-8")]
    max_length = args.max_length or json.load(open(ROOT / "cpmr_experiment/esci_llama/audit/max_length.json"))["max_length"]
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
    device = next(model.parameters()).device
    assert tok("Relevant", add_special_tokens=False)["input_ids"][0] == REL_FIRST and tok("Irrelevant", add_special_tokens=False)["input_ids"][0] == IRR_FIRST
    params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    mods = sorted({n.split(".lora_")[0] for n, p in model.named_parameters() if "lora_" in n})
    assert n_tr == 6815744 and len(mods) == 128, (n_tr, len(mods))
    opt = torch.optim.AdamW(params, lr=cfg["optim"]["lr"])
    sched = get_linear_schedule_with_warmup(opt, warm, B)
    tag = f"{args.tag_prefix}_{args.instr}_s{args.seed}"
    out = ROOT / args.out_root
    ckd, logd = out / "checkpoints" / tag, out / "logs"
    if not args.check:
        ckd.mkdir(parents=True, exist_ok=True); logd.mkdir(parents=True, exist_ok=True)
        assert not (ckd / "done.json").exists(), f"{tag} already done"

    def save(step):
        st = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save({"adapter_state": st, "tag": tag, "model": cfg["model_name"], "global_step": step, "frac": step / B, "seed": args.seed, "instruction": args.instr, "alpha": 1.0}, ckd / f"ckpt_step{step}.pt")
    print(f"[esci-train] {tag}: {M} rows, {B} updates (last batch {M - EFFECTIVE_BATCH * (B - 1)} rows), warm-up {warm}, checkpoints {save_steps}, max_length {max_length}, micro {args.micro}, trainable parameters {n_tr}, {len(mods)} LoRA modules, text sha {hashlib.sha256(text.encode()).hexdigest()[:12]}", flush=True)
    if not args.check:
        save(0)
    steps_f = logd / f"steps_{tag}.jsonl"
    if steps_f.exists() and not args.check:
        steps_f.unlink()
    t0 = time.time()
    n_fwd = n_tok = n_shr = 0
    for step in range(args.check if args.check else B):
        opt.zero_grad(set_to_none=True)
        idx = order[step * EFFECTIVE_BATCH:(step + 1) * EFFECTIVE_BATCH]
        ng = len(idx)
        ces = []
        for i in range(0, ng, args.micro):
            mb = [rows[j] for j in idx[i:i + args.micro]]
            ce, shr, ntk = ce_rows(core, head, tok, text, [(r["query"], r["product_text"], r["label"]) for r in mb], max_length, device)
            if not torch.isfinite(ce).all():
                raise SystemExit(f"non-finite loss at update {step + 1}")
            (ce.sum() / ng).backward()
            ces += [float(x) for x in ce]
            n_fwd += len(mb); n_tok += ntk; n_shr += shr
        gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)))
        lr_used = opt.param_groups[0]["lr"]
        opt.step(); sched.step()
        row = {"step": step + 1, "rows": ng, "ce": sum(ces) / ng, "lr_used": lr_used, "lr": sched.get_last_lr()[0], "grad_norm": gn, "total_rows": n_fwd, "tokens": n_tok}
        if not args.check:
            with open(steps_f, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row) + "\n")
        if (step + 1) % 100 == 0 or args.check or step + 1 == B:
            print(f"[esci-train] {tag} update {step + 1}/{B} ce {row['ce']:.4f} lr {row['lr']:.2e} grad_norm {gn:.3f} rows {n_fwd} elapsed {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
        if not args.check and (step + 1) in save_steps:
            t1 = time.time(); save(step + 1); print(f"[esci-train] saved update {step + 1} ({time.time() - t1:.1f}s)", flush=True)
    if args.check:
        bmax = max(float(p.detach().abs().max()) for n, p in model.named_parameters() if "lora_B" in n)
        print(f"[check] after {args.check} updates: max |lora_B| = {bmax:.3e} (was 0), seconds per update {(time.time() - t0) / args.check:.2f} ({(time.time() - t0) / n_fwd:.3f} s/row, {n_tok / n_fwd:.0f} tokens/row), shortened {n_shr}, peak memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
        return 0
    stats = {"tag": tag, "seed": args.seed, "instruction": args.instr, "instruction_sha256": hashlib.sha256(text.encode()).hexdigest(), "updates": B, "rows_exposed": M, "unique_queries": len({rows[i]["nq"] for i in order}), "loss": "CE only (row mean)",
             "total_rows": n_fwd, "tokens": n_tok, "inputs_shortened": n_shr, "train_seconds": time.time() - t0, "peak_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "save_steps": save_steps, "warmup_updates": warm, "max_length": max_length, "micro_batch": args.micro,
             "optimizer": "AdamW lr 2e-5 warmup ceil(0.1B) linear decay, no clipping", "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "modules": len(mods), "parameters": n_tr}, "model": cfg["model_name"],
             "rows_file_sha256": hashlib.sha256((ROOT / args.rows).read_bytes()).hexdigest(), "chat_template": "Llama-3.1 official (default system paragraph with the fixed date)", "torch": torch.__version__}
    (logd / f"stats_{tag}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    (ckd / "done.json").write_text(json.dumps({"done": True, "seconds": time.time() - t0}), encoding="utf-8")
    print(f"[esci-train] {tag} done {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
