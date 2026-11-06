"""Llama-3.1-8B-Instruct SFT with one fixed relevance instruction (experiment log 139). Same recipe as the Gemma runs (shared/138 gemma_r7_train.py) and the Qwen R runs:
5,000 training query groups (one Relevant + one Irrelevant passage each, every query exposed once, order = load_query_groups(seed)), 625 optimizer updates x 8 query groups (the loss of a group is divided by 8, 16 sequences per update),
loss of a group L = 0.5 (CE_pos + CE_neg) + 1.0 * softplus(-(m_pos - m_neg)); CE = token-mean cross entropy of the supervised tokens of one document (answer tokens + <|eot_id|>); m = logit(first token of 'Relevant') - logit(first token of 'Irrelevant')
at the position before the first answer token (float32) = the training ranking score (the evaluation sequence-margin is a different quantity); AdamW (default betas / eps, weight decay 0.01) lr 2e-5, linear warm-up 62 updates (10%) then linear decay to 0, no gradient clipping,
bf16 base weights, LoRA r 8 / alpha 16 / dropout 0.05 on the decoder q/k/v/o projections only (128 modules, 6,815,744 parameters), torch seed fixed before the model is built -> the seven instructions of one seed start from the same adapter initialisation (A random, B = 0) and see the same data order.
Differences to the Gemma runs: Llama chat template and tokenizer (answer + <|eot_id|>), UNTIED lm_head, padding token 128004 (unused in training: one sequence per forward), gradient checkpointing is available (--grad_ckpt 1) but OFF by default, as in the Gemma runs: the pilot measured a worst-case peak of 18.9 GB without it on a 24 GB card;
the decoder (LlamaModel) is called directly and only the supervised positions pass through the vocabulary head. Passage strings are the training pool strings (identical to the Qwen / Gemma strings). Adapter checkpoints at the completed updates 62, 156, 312, 469, 625 (+ step 0), real global step stored inside.
    CUDA_VISIBLE_DEVICES=0 python methods/llama_r7_train.py --instr R1 --seed 0 --out_root cpmr_experiment/llama_r7/train --local_files_only [--check]"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
for rel in ("", "methods", "evaluation", "cpmr_experiment/scripts"):
    sys.path.insert(0, str(ROOT / rel))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from transformers import get_linear_schedule_with_warmup  # noqa: E402

from cpmr_query_dataset import load_query_groups  # noqa: E402
from llama_r7_eval import EOT, build  # noqa: E402

EFFECTIVE_BATCH = 8
SAVE_STEPS = [62, 156, 312, 469, 625]
REL_FIRST, IRR_FIRST = 697, 49213


def doc_forward(core, head, tok, text, query, ptext, label, max_length, device, store=None):
    pre, resp, shr = build_from_text(tok, text, query, ptext, label, max_length)
    ids = torch.tensor([pre + resp + [EOT]], device=device)
    out = core(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    h = out.last_hidden_state[0]                                   # after the final norm (bf16)
    st = len(pre)
    rows = torch.arange(st - 1, st + len(resp), device=device)       # positions predicting the answer tokens and <|eot_id|>
    logits = head(h[rows]).float()                                 # only the supervised rows pass through the vocabulary head
    tgt = ids[0, st:st + len(resp) + 1]
    ce = F.cross_entropy(logits, tgt)                              # token mean over the supervised tokens of this document
    m = logits[0, REL_FIRST] - logits[0, IRR_FIRST]
    return ce, m, shr, len(resp) + 1


def build_from_text(tok, instruction, query, ptext, label, max_length):
    """same as gemma_r7_eval.build but the passage already carries its 'Passage:' marker (training pool strings)."""
    resp = tok(label, add_special_tokens=False)["input_ids"]
    budget = max_length - len(resp) - 1

    def prefix(pt):
        t = tok.apply_chat_template([{"role": "user", "content": f"{instruction}\n\nQuery: {query}\n{pt}"}], tokenize=False, add_generation_prompt=True)
        return tok(t, add_special_tokens=False)["input_ids"]
    text = ptext if ptext.startswith("Passage:") else f"Passage: {ptext}"
    pre, shrunk = prefix(text), False
    while len(pre) > budget and len(text) > 20:
        text = text[: int(len(text) * 0.8)]
        pre = prefix(text)
        shrunk = True
    return pre, resp, shrunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instr", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--instr_file", default="cpmr_experiment/llama_r7/llama_r7_manifest.json")
    ap.add_argument("--train_pairs", default="data/msmarco_train_v2_5k/pairs_text_basic_5000.jsonl")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--config", default="configs/llama31_8b.yaml")
    ap.add_argument("--max_length", type=int, default=500)
    ap.add_argument("--grad_ckpt", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="4 updates, nothing durable")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()
    from src.model import load_model_and_tokenizer
    cfg = yaml.safe_load(open(ROOT / args.config))
    ins = {e["display_id"]: e["full"] for e in json.load(open(ROOT / args.instr_file, encoding="utf-8"))["entries"]}
    text = ins[args.instr]
    groups, order = load_query_groups(ROOT / args.train_pairs, args.seed)
    n_q = len(order)
    total_steps = n_q // EFFECTIVE_BATCH
    assert n_q % EFFECTIVE_BATCH == 0 and total_steps == 625
    warm = round(cfg["optim"]["warmup_ratio"] * total_steps)
    torch.manual_seed(args.seed)                                   # before the model / LoRA is built (as build_context of the Qwen runs)
    lm = load_model_and_tokenizer(model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
                                  gradient_checkpointing=bool(args.grad_ckpt), local_files_only=args.local_files_only)
    model, tok = lm.model, lm.tokenizer
    model.train()
    inner = model.base_model.model
    core, head = inner.model, inner.lm_head
    device = next(model.parameters()).device
    assert tok("Relevant", add_special_tokens=False)["input_ids"][0] == REL_FIRST and tok("Irrelevant", add_special_tokens=False)["input_ids"][0] == IRR_FIRST
    params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    mods = sorted({n.split(".lora_")[0] for n, p in model.named_parameters() if "lora_" in n})
    assert n_tr == 6815744 and len(mods) == 128
    opt = torch.optim.AdamW(params, lr=cfg["optim"]["lr"])
    sched = get_linear_schedule_with_warmup(opt, warm, total_steps)
    tag = f"L_{args.instr}_s{args.seed}"
    out = ROOT / args.out_root
    ckd = out / "checkpoints" / tag
    logd = out / "logs"
    if not args.check:
        ckd.mkdir(parents=True, exist_ok=True); logd.mkdir(parents=True, exist_ok=True)
        assert not (ckd / "done.json").exists(), f"{tag} already done"

    def save(step):
        st = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save({"adapter_state": st, "tag": tag, "model": cfg["model_name"], "global_step": step, "frac": step / total_steps, "seed": args.seed, "instruction": args.instr, "alpha": 1.0}, ckd / f"ckpt_step{step}.pt")
    print(f"[llama-train] {tag}: {n_q} query groups, {total_steps} updates, warm-up {warm}, trainable parameters {n_tr}, {len(mods)} LoRA modules, text sha {__import__('hashlib').sha256(text.encode()).hexdigest()[:12]}", flush=True)
    if not args.check:
        save(0)
    steps_f = logd / f"steps_{tag}.jsonl"
    if steps_f.exists() and not args.check:
        steps_f.unlink()
    t0 = time.time()
    n_fwd, n_tok, n_shr = 0, 0, 0
    last = 4 if args.check else total_steps
    for step in range(last):
        opt.zero_grad(set_to_none=True)
        qids = order[step * EFFECTIVE_BATCH:(step + 1) * EFFECTIVE_BATCH]
        ces, ranks, gaps = [], [], []
        for qid in qids:
            g = groups[qid]
            ce_p, m_p, s1, n1 = doc_forward(core, head, tok, text, g["query"], g["pos"], "Relevant", args.max_length, device)
            ce_n, m_n, s2, n2 = doc_forward(core, head, tok, text, g["query"], g["neg"], "Irrelevant", args.max_length, device)
            rank = F.softplus(-(m_p - m_n))
            loss = 0.5 * (ce_p + ce_n) + 1.0 * rank
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at update {step + 1} query {qid}")
            (loss / EFFECTIVE_BATCH).backward()
            ces.append(float(0.5 * (ce_p + ce_n))); ranks.append(float(rank)); gaps.append(float(m_p - m_n))
            n_fwd += 2; n_tok += n1 + n2; n_shr += int(s1) + int(s2)
        gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)))
        lr_used = opt.param_groups[0]["lr"]
        opt.step(); sched.step()
        row = {"step": step + 1, "ce": sum(ces) / 8, "rank": sum(ranks) / 8, "gap": sum(gaps) / 8, "lr_used": lr_used, "lr": sched.get_last_lr()[0], "grad_norm": gn, "total_forward_passes": n_fwd, "supervised_tokens": n_tok,
               "loss": sum(ces) / 8 + sum(ranks) / 8}
        if not args.check:
            with open(steps_f, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row) + "\n")
        if (step + 1) % 25 == 0 or args.check:
            print(f"[llama-train] {tag} update {step + 1}/{total_steps} ce {row['ce']:.4f} rank {row['rank']:.4f} gap {row['gap']:.2f} lr {row['lr']:.2e} grad_norm {gn:.3f} fwd {n_fwd} elapsed {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
        if not args.check and (step + 1) in SAVE_STEPS:
            t1 = time.time(); save(step + 1); print(f"[llama-train] saved update {step + 1} ({time.time() - t1:.1f}s)", flush=True)
    if args.check:
        bmax = max(float(p.detach().abs().max()) for n, p in model.named_parameters() if "lora_B" in n)
        print(f"[check] after 4 updates: max |lora_B| = {bmax:.3e} (was 0), seconds per update {(time.time() - t0) / 4:.2f}, shortened passages {n_shr}, peak memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
        return 0
    stats = {"tag": tag, "seed": args.seed, "instruction": args.instr, "instruction_sha256": __import__("hashlib").sha256(text.encode()).hexdigest(), "updates": total_steps, "query_groups_exposed": total_steps * EFFECTIVE_BATCH, "unique_queries": len(set(order)),
             "total_forward_passes": n_fwd, "supervised_tokens": n_tok, "passages_shortened": n_shr, "train_seconds": time.time() - t0, "peak_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "save_steps": SAVE_STEPS, "max_length": args.max_length,
             "optimizer": "AdamW lr 2e-5 warmup 62 linear decay, no clipping", "grad_ckpt": args.grad_ckpt, "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "modules": len(mods), "parameters": n_tr}, "torch": torch.__version__}
    (logd / f"stats_{tag}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    (ckd / "done.json").write_text(json.dumps({"done": True, "seconds": time.time() - t0}), encoding="utf-8")
    print(f"[llama-train] {tag} done {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
