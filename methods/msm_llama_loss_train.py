"""Llama-3.1-8B-Instruct MS MARCO CE / CE + soft kappa / CE + AnyLoss-like kappa trainer for the original instructions T1-T7 (experiment log 214; second server).
Data, order, optimizer, schedule and LoRA are those of the Llama R-series trainer llama_r7_train.py (experiment log 139) and of the Qwen experiment of 211/213 (msm_loss_train4.py): 5,000 frozen query groups of data/msmarco_train_v2_5k/pairs_text_basic_5000.jsonl in the order of load_query_groups(seed), one positive and one negative passage per group, 8 groups = 16 sequences per update, 625 updates, warm-up 62, AdamW lr 2e-5 (no clipping), LoRA r8 / alpha16 / dropout .05 on q/k/v/o (128 modules, 6,815,744 parameters), torch.manual_seed(seed) before the model is built.
Input = llama_r7_train.build_from_text (official Llama-3.1 chat template with its default system paragraph; user turn '<instruction>\\n\\nQuery: <q>\\nPassage: <p>'; answer tokens of the label + <|eot_id|>); max_length 384 (the longest sequence over T1-T7, training and TREC-DL is 365 tokens, so no passage is shortened).
--loss B : L_CE = mean over the 8 groups of 0.5 (CE_pos + CE_neg), CE of one document = token mean over its supervised tokens (label tokens + <|eot_id|>) -- the CE part of llama_r7_train.py (llama_r7_train.doc_forward, unchanged), no ranking term.
--loss E : L_CE + lam * L_K, L_K = U / clamp_min(V, 1e-6) over the 16 sequences, p = sigmoid(m), m = log P(Relevant + <|eot_id|>) - log P(Irrelevant + <|eot_id|>) (sum over the label tokens, teacher forced, FP32 final norm / untied lm_head / log_softmax), denominator gradient kept.
--loss M : L_CE + lam * L_AK (q = sigmoid(10 (p - .5)), alpha 10 fixed).   --loss G : L_AK alone (gate checks only).   lam = 1.0 fixed beforehand.
The kappa part is the exact two-pass scheme of the Qwen trainer (pass 1 no grad -> 16 margins -> dL_K/dm, pass 2 per group with the dropout RNG state restored; afterwards the RNG is set to the state after pass 1); the CE forwards are independent forwards done first.
Every update has 8 positive and 8 negative sequences, so r = .5 and V = .5 (L_K = 1 + 2s - 4a), audited per update.
    CUDA_VISIBLE_DEVICES=0 python methods/msm_llama_loss_train.py --loss E --instr T1 --seed 0 --tag mlq_E_T1_s0 --out_root cpmr_experiment/msm_llama/train --local_files_only [--check N]"""
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

import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import get_linear_schedule_with_warmup  # noqa: E402

from cpmr_query_dataset import load_query_groups  # noqa: E402
from esci_loss_lib import batch_diagnostics, get_rng, grad_wrt_m, margins, set_rng  # noqa: E402
from llama_r7_train import build_from_text, doc_forward, REL_FIRST, IRR_FIRST  # noqa: E402

EB = 8
EOT, PAD = 128009, 128004


def group_seqs(tok, text, g, max_length):
    """[(prefix, answer, eot)] of the Relevant-label and the Irrelevant-label sequences of the positive and the negative document of one group"""
    sp, sn, shr = [], [], 0
    for doc in (g["pos"], g["neg"]):
        pre, resp, s1 = build_from_text(tok, text, g["query"], doc, "Relevant", max_length)
        pre2, resp2, s2 = build_from_text(tok, text, g["query"], doc, "Irrelevant", max_length)
        sp.append((pre, resp, EOT)); sn.append((pre2, resp2, EOT)); shr += int(s1) + int(s2)
    return sp, sn, shr


def group_margins(core, norm, w_head, store, tok, text, g, max_length, device):
    sp, sn, _ = group_seqs(tok, text, g, max_length)
    m, _, _ = margins(core, norm, w_head, store, sp, sn, PAD, device)
    return m                                             # [m_pos, m_neg]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", required=True, choices=["B", "E", "M", "G", "S"])        # G = AnyLoss-like kappa alone, S = soft kappa alone: gate checks only
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--instr", default="T1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--config", default="configs/llama31_8b_msm.yaml")
    ap.add_argument("--train_pairs", default="data/msmarco_train_v2_5k/pairs_text_basic_5000.jsonl")
    ap.add_argument("--manifest", default="cpmr_experiment/instructions_v2/manifest/instructions_v2.json")
    ap.add_argument("--custom_instruction_file", default=None)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--grad_ckpt", type=int, default=0)
    ap.add_argument("--lora_dropout", type=float, default=None, help="test only: override the config dropout")
    ap.add_argument("--check", type=int, default=0, help="N updates, nothing durable")
    ap.add_argument("--check_log", default="")
    ap.add_argument("--dump_grad", default="", help="test only (with --check 1): save the flattened gradient of the first update and stop before the optimizer step")
    ap.add_argument("--test_b_std", type=float, default=0.0, help="test only: non-zero LoRA B (normal, this std) so that every parameter receives a gradient")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()
    from src.model import load_model_and_tokenizer
    cfg = yaml.safe_load(open(ROOT / args.config))
    man = json.load(open(ROOT / args.manifest, encoding="utf-8"))
    instr = {k: v["body"] + man["output_suffix"] for k, v in man["entries"].items() if "body" in v}
    if args.custom_instruction_file:
        instr.update(json.load(open(ROOT / args.custom_instruction_file, encoding="utf-8")))
    text = instr[args.instr]
    groups, order = load_query_groups(ROOT / args.train_pairs, args.seed)
    n_q = len(order)
    total = n_q // EB
    assert n_q % EB == 0 and total == 625, (n_q, total)
    warm = round(cfg["optim"]["warmup_ratio"] * total)
    torch.manual_seed(args.seed)
    lm = load_model_and_tokenizer(model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"] if args.lora_dropout is None else args.lora_dropout,
                                  target_modules=cfg["lora"]["target_modules"], gradient_checkpointing=bool(args.grad_ckpt), local_files_only=args.local_files_only)
    model, tok = lm.model, lm.tokenizer
    model.train()
    if args.test_b_std > 0:
        torch.manual_seed(123)
        for n_, p_ in model.named_parameters():
            if "lora_B" in n_:
                torch.nn.init.normal_(p_, std=args.test_b_std)
        torch.manual_seed(args.seed)
    inner = model.base_model.model
    core, head = inner.model, inner.lm_head
    norm = core.norm
    device = next(model.parameters()).device
    assert tok("Relevant", add_special_tokens=False)["input_ids"][0] == REL_FIRST and tok("Irrelevant", add_special_tokens=False)["input_ids"][0] == IRR_FIRST
    params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    mods = sorted({n.split(".lora_")[0] for n, p in model.named_parameters() if "lora_" in n})
    assert n_tr == 6815744 and len(mods) == 128, (n_tr, len(mods))
    store = {}
    norm.register_forward_pre_hook(lambda mod, a: store.__setitem__("h", a[0]))          # graph kept (training path)
    w_head = head.weight.detach().float()                                                  # untied lm_head in FP32 (frozen)
    opt = torch.optim.AdamW(params, lr=cfg["optim"]["lr"])
    sched = get_linear_schedule_with_warmup(opt, warm, total)
    tag = args.tag
    out = ROOT / args.out_root
    ckd, logd = out / "checkpoints" / tag, out / "logs"
    if not args.check:
        ckd.mkdir(parents=True, exist_ok=True); logd.mkdir(parents=True, exist_ok=True)
        assert not (ckd / "ckpt_final.pt").exists(), f"{tag} already done"
    steps_f = logd / f"steps_{tag}.jsonl"
    if steps_f.exists() and not args.check:
        steps_f.unlink()

    def save(step, name):
        st = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        torch.save({"adapter_state": st, "tag": tag, "model": cfg["model_name"], "condition": f"msm-llama-{args.loss}", "instruction": args.instr, "global_step": step, "frac": step / total, "seed": args.seed, "loss": args.loss, "lam": args.lam}, ckd / name)
    print(f"[msm-llama] {tag}: loss {args.loss}, lam {args.lam}, {n_q} query groups, {total} updates, warm-up {warm}, max_length {args.max_length}, grad_ckpt {args.grad_ckpt}, {args.instr} sha {hashlib.sha256(text.encode()).hexdigest()[:12]}", flush=True)
    if not args.check:
        save(0, "ckpt_step0.pt")
    kind = "anyk" if args.loss in ("M", "G") else "softk"
    t0 = time.time()
    n_seq = n_shr = n_nonfinite = 0
    for step in range(args.check if args.check else total):
        opt.zero_grad(set_to_none=True)
        qids = order[step * EB:(step + 1) * EB]
        gs = [groups[q] for q in qids]
        row = {"step": step + 1}
        ces = []
        if args.loss not in ("G", "S"):
            for g in gs:                                                            # (1) the CE part (llama_r7_train.doc_forward, one document per forward), backward per group
                ce_p, _, s1, _ = doc_forward(core, head, tok, text, g["query"], g["pos"], "Relevant", args.max_length, device)
                ce_n, _, s2, _ = doc_forward(core, head, tok, text, g["query"], g["neg"], "Irrelevant", args.max_length, device)
                loss = 0.5 * (ce_p + ce_n)
                if not torch.isfinite(loss):
                    raise SystemExit(f"non-finite CE at update {step + 1}")
                (loss / EB).backward()
                ces.append(float(loss)); n_seq += 2; n_shr += int(s1) + int(s2)
        ce = sum(ces) / EB if ces else 0.0
        lk = 0.0
        if args.loss != "B":
            y = torch.tensor([1.0, 0.0] * EB, device=device)
            states, ms = [], []
            with torch.no_grad():                                                   # (2) the kappa part, pass 1
                for g in gs:
                    states.append(get_rng())
                    ms.append(group_margins(core, norm, w_head, store, tok, text, g, args.max_length, device)); n_seq += 4
            end_state = get_rng()
            mvec = torch.cat(ms)
            diag = batch_diagnostics(kind, mvec, y)
            assert abs(diag["r"] - 0.5) < 1e-9, f"batch with r = {diag['r']}"
            gvec, lk = grad_wrt_m(kind, mvec, y)
            p = torch.sigmoid(mvec)
            qv = p if kind == "softk" else torch.sigmoid(10.0 * (p - 0.5))
            diag["loss_simplified"] = float(2 * (1 - (y * qv + (1 - y) * (1 - qv)).mean()))        # r = .5: U / V = 2 (1 - soft accuracy of q)
            diag["q_mid_frac"] = float(((qv > 0.1) & (qv < 0.9)).float().mean()); diag["q_outer_frac"] = float(((qv < 0.1) | (qv > 0.9)).float().mean())
            dmax = 0.0
            for k, (g, st) in enumerate(zip(gs, states)):                           # pass 2 with the dropout RNG restored
                set_rng(st)
                m2 = group_margins(core, norm, w_head, store, tok, text, g, args.max_length, device)
                dmax = max(dmax, float((m2.detach() - mvec[2 * k:2 * k + 2]).abs().max()))
                (args.lam * gvec[2 * k:2 * k + 2] * m2).sum().backward(); n_seq += 4
            set_rng(end_state)
            diag["max_pass_diff"] = dmax
            row.update(diag)
        row.update({"ce": ce, "kappa_loss": lk})
        row["loss"] = ce + (0.0 if args.loss == "B" else args.lam * lk)
        if not math.isfinite(row["loss"]):
            n_nonfinite += 1
            raise SystemExit(f"non-finite loss at update {step + 1}")
        gn = float(torch.sqrt(sum((p_.grad.float() ** 2).sum() for p_ in params if p_.grad is not None)))
        if not math.isfinite(gn):
            n_nonfinite += 1
            raise SystemExit(f"non-finite gradient norm at update {step + 1}")
        if args.dump_grad:
            torch.save({"grad": torch.cat([p_.grad.detach().float().flatten() if p_.grad is not None else torch.zeros(p_.numel(), device=p_.device) for p_ in params]).cpu(), "row": row}, args.dump_grad)
            print(f"[dump] {args.loss}: gradient norm {gn:.6e} -> {args.dump_grad}")
            return 0
        row["lr_used"] = opt.param_groups[0]["lr"]
        opt.step(); sched.step()
        row.update({"lr": sched.get_last_lr()[0], "grad_norm": gn, "sequences_forwarded": n_seq})
        if not args.check:
            with open(steps_f, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row) + "\n")
        elif args.check_log:
            with open(args.check_log, "a", encoding="utf-8", newline="\n") as f_:
                f_.write(json.dumps(row) + "\n")
        if (step + 1) % 25 == 0 or args.check or step + 1 == total:
            print(f"[msm-llama] {tag} update {step + 1}/{total} loss {row['loss']:.4f} (ce {ce:.3f} kappa {lk:.3f}) grad_norm {gn:.3f} p_sat {row.get('p_sat', float('nan')):.2f} max_pass_diff {row.get('max_pass_diff', 0.0):.1e} "
                  f"elapsed {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
    dt = time.time() - t0
    if args.check:
        bmax = max(float(p_.detach().abs().max()) for n, p_ in model.named_parameters() if "lora_B" in n)
        print(f"[check] {args.loss}: {args.check} updates, max |lora_B| {bmax:.3e}, {dt / args.check:.2f} s/update, shortened {n_shr}, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
        return 0
    save(total, "ckpt_final.pt")
    stats = {"tag": tag, "seed": args.seed, "instruction": args.instr, "loss": args.loss, "lam": args.lam, "updates": total, "sequences_forwarded": n_seq, "inputs_shortened": n_shr, "nonfinite": n_nonfinite, "train_seconds": dt,
             "peak_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "order_sha256": hashlib.sha256(json.dumps(list(map(str, order))).encode()).hexdigest(), "instruction_sha256": hashlib.sha256(text.encode()).hexdigest(),
             "max_length": args.max_length, "grad_ckpt": args.grad_ckpt, "model": cfg["model_name"], "torch": torch.__version__}
    (logd / f"stats_{tag}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"[msm-llama] {tag} done {dt:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
