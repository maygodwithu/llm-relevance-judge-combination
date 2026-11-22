"""Minimal numerical diagnosis of the micro-batch sensitivity of the CE+2MAE gradient (ESCI Qwen3-4B, instruction T1, seed 0, the FIRST update batch = 16 rows of the real order).
No optimizer step, no training, no evaluation. Dropout 0, model.train() (as in the 385 gate). For ONE weights condition per process:
    --weights artificial : the 385 gate weights (torch.manual_seed(0) at model build, LoRA B ~ N(0, 0.02^2) with torch.manual_seed(123), exactly as esci_loss_train_384.py --test_b_std 0.02)
    --weights final      : the real adapter of the new CE+2MAE seed-0 run (ckpt_step625.pt of Q384M_mixmae_T1_s0)
Four conditions per process, in this order: BF16 micro 4, BF16 micro 2 (the training precision: bf16 base blocks, FP32 head / log_softmax), then the SAME in-memory weights cast to FP32 (model.float(), TF32 off): FP32 micro 4, FP32 micro 2.
Per condition (fixed 16 rows, effective batch 16, every term divided by 16): per-row label scores l_R, l_I, margin m, u = sigmoid(m), analytic dL/dm of 2 x mean|y-u|, per-row CE, MAE loss, and the LoRA gradients of the CE part and of the MAE part computed SEPARATELY (zero_grad between them, no subtraction of totals).
Decomposition: G(micro2) - G(micro4) = J2^T (c2 - c4) + (J2 - J4)^T c4, where c = dL/dm of the MAE part: H = J2^T c4 is obtained by one backward of sum_i c4_i m_i in the micro-2 forward.
    python micro386_diag.py --root <project root> --weights artificial|final --out <dir>"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--weights", required=True, choices=["artificial", "final"]); ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default="cpmr_experiment/esci384/train/checkpoints/Q384M_mixmae_T1_s0/ckpt_step625.pt")
a = ap.parse_args()
ROOT = Path(a.root).resolve()
for rel in ("", "methods", "evaluation", "cpmr_experiment/scripts"):
    sys.path.insert(0, str(ROOT / rel))
from esci_core import EOT, PAD, REL_FIRST, IRR_FIRST, build, ce_rows  # noqa: E402
from esci_loss_lib import margins  # noqa: E402
from layer_drift_repr import load_adapter  # noqa: E402
from src.model import load_model_and_tokenizer  # noqa: E402

OUT = ROOT / a.out; OUT.mkdir(parents=True, exist_ok=True)
EB, LAM = 16, 2.0
cfg = yaml.safe_load(open(ROOT / "configs/qwen3_4b_esci.yaml"))
text = {e["display_id"]: e["full"] for e in json.load(open(ROOT / "cpmr_experiment/esci_t/esci_t_manifest.json", encoding="utf-8"))["entries"]}["T1"]
rows = [json.loads(l) for l in open(ROOT / "cpmr_experiment/esci/audit/sft2_rows.jsonl", encoding="utf-8")]
max_length = json.load(open(ROOT / "cpmr_experiment/esci/audit/splits.json"))["max_length"]
order = [int(i) for i in np.random.default_rng(0).permutation(len(rows))]
idx = order[:EB]
torch.manual_seed(0)
lm = load_model_and_tokenizer(model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=0.0, target_modules=cfg["lora"]["target_modules"], gradient_checkpointing=False, local_files_only=True)
model, tok = lm.model, lm.tokenizer
model.train()
if a.weights == "artificial":
    torch.manual_seed(123)
    for n_, p_ in model.named_parameters():
        if "lora_B" in n_:
            torch.nn.init.normal_(p_, std=0.02)
    adapter_note = "LoRA B ~ N(0, 0.02^2) (artificial, as in the 385 gate); LoRA A as built with torch.manual_seed(0)"
else:
    ck = load_adapter(model, ROOT / a.ckpt)
    adapter_note = f"adapter of {Path(a.ckpt).parent.name} step {ck.get('global_step')}"
inner = model.base_model.model
core, head = inner.model, inner.lm_head
norm = core.norm
device = next(model.parameters()).device
params = [p for p in model.parameters() if p.requires_grad]
store = {}
norm.register_forward_pre_hook(lambda mod, args: store.__setitem__("h", args[0]))
meta = dict(weights=a.weights, adapter=adapter_note, rows="first 16 rows of the seed-0 order", param_dtypes=sorted({str(p.dtype) for p in params}), base_dtype=str(getattr(getattr(inner.model.layers[0].self_attn.q_proj, "base_layer", inner.model.layers[0].self_attn.q_proj), "weight").dtype),
            attn_implementation=str(getattr(model.config, "_attn_implementation", "?")), tf32_matmul_default=bool(torch.backends.cuda.matmul.allow_tf32), torch=torch.__version__, dropout=0.0, optimizer_step=False, clipping=False)
y = torch.tensor([1.0 if rows[i]["label"] == "Relevant" else 0.0 for i in idx], device=device)
RWS = [rows[i] for i in idx]


def seqs_of(mb):
    sp, sn = [], []
    for r in mb:
        pre, resp, _ = build(tok, text, r["query"], r["product_text"], "Relevant", max_length)
        pre2, resp2, _ = build(tok, text, r["query"], r["product_text"], "Irrelevant", max_length)
        sp.append((pre, resp, EOT)); sn.append((pre2, resp2, EOT))
    return sp, sn


def flat():
    return torch.cat([p.grad.detach().double().flatten().cpu() if p.grad is not None else torch.zeros(p.numel(), dtype=torch.float64) for p in params])


def zero():
    for p in params:
        p.grad = None


def one_condition(prec, micro, c_for_H=None):
    w_head = head.weight.detach().float()
    mbs = [RWS[i:i + micro] for i in range(0, EB, micro)]
    res = dict(precision=prec, micro=micro)
    zero(); ces = []
    for mb in mbs:                                              # CE part (as the trainer: backward per micro-batch, /16)
        ce, _, _ = ce_rows(core, head, tok, text, [(r["query"], r["product_text"], r["label"]) for r in mb], max_length, device)
        (ce.sum() / EB).backward(); ces += [float(x) for x in ce]
    gCE = flat(); zero()
    ms, sR, sI = [], [], []
    off = 0
    for mb in mbs:                                              # MAE part (as the trainer: lam x sum over the micro-batch / 16)
        sp, sn = seqs_of(mb)
        m, s_r, s_i = margins(core, norm, w_head, store, sp, sn, PAD, device)
        yy_mb = y[off:off + len(mb)]
        (LAM * (yy_mb - torch.sigmoid(m)).abs().sum() / EB).backward(); off += len(mb)
        ms.append(m.detach().float().cpu()); sR.append(s_r.detach().float().cpu()); sI.append(s_i.detach().float().cpu())
    gMAE = flat(); zero()
    m = torch.cat(ms).double(); u = torch.sigmoid(m); yy = y.cpu().double()
    c = (-(LAM / EB) * torch.sign(yy - u) * u * (1 - u))        # analytic dL/dm_i of lam x mean|y-u|
    res.update(l_R=torch.cat(sR).double().tolist(), l_I=torch.cat(sI).double().tolist(), m=m.tolist(), u=u.tolist(), dLdm=c.tolist(), ce_rows=ces, mae_loss=float((yy - u).abs().mean()), loss_mae_term=float(LAM * (yy - u).abs().mean()),
               ce_mean=float(np.mean(ces)), gCE_norm=float(gCE.norm()), gMAE_norm=float(gMAE.norm()))
    if c_for_H is not None:                                     # H = J^T c_ref in THIS forward
        mbs2 = mbs; zero()
        off = 0
        for mb in mbs2:
            sp, sn = seqs_of(mb)
            mm, _, _ = margins(core, norm, w_head, store, sp, sn, PAD, device)
            (mm * torch.tensor(c_for_H[off:off + len(mb)], device=mm.device, dtype=mm.dtype)).sum().backward(); off += len(mb)
        res["_H"] = flat(); zero()
    res["_gCE"], res["_gMAE"] = gCE, gMAE
    return res


def cmp(A, B, key):
    ga, gb = A[key], B[key]
    d = ga - gb
    return dict(abs_diff_norm=float(d.norm()), rel_to_norm_first=float(d.norm() / max(float(ga.norm()), 1e-30)), rel_to_norm_second=float(d.norm() / max(float(gb.norm()), 1e-30)), cosine=float((ga @ gb) / max(float(ga.norm() * gb.norm()), 1e-30)), norm_first=float(ga.norm()), norm_second=float(gb.norm()))


out = dict(meta=meta, conditions={}, comparisons={})
t0 = time.time()
for prec in ("bf16", "fp32"):
    if prec == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        model.float(); meta["fp32_param_dtypes"] = sorted({str(p.dtype) for p in params}); params = [p for p in model.parameters() if p.requires_grad]
    try:
        r4 = one_condition(prec, 4)
        r2 = one_condition(prec, 2, c_for_H=r4["dLdm"])
    except torch.cuda.OutOfMemoryError as e:
        out["conditions"][prec] = dict(status="OOM", message=str(e)[:200]); torch.cuda.empty_cache(); continue
    d = {}
    d["margin_max_abs_diff"] = float(np.abs(np.array(r4["m"]) - np.array(r2["m"])).max())
    d["l_R_max_abs_diff"] = float(np.abs(np.array(r4["l_R"]) - np.array(r2["l_R"])).max()); d["l_I_max_abs_diff"] = float(np.abs(np.array(r4["l_I"]) - np.array(r2["l_I"])).max())
    d["mae_loss_diff"] = r2["mae_loss"] - r4["mae_loss"]; d["ce_mean_diff"] = r2["ce_mean"] - r4["ce_mean"]
    d["ce_gradient"] = cmp(r4, r2, "_gCE"); d["mae_gradient"] = cmp(r4, r2, "_gMAE")
    G4, G2 = r4["_gCE"] + r4["_gMAE"], r2["_gCE"] + r2["_gMAE"]
    d["total_gradient"] = cmp(dict(g=G4), dict(g=G2), "g")
    H = r2["_H"]; gm4, gm2 = r4["_gMAE"], r2["_gMAE"]
    d["mae_gradient_decomposition"] = dict(norm_G2_minus_G4=float((gm2 - gm4).norm()), norm_J2T_dc=float((gm2 - H).norm()), norm_dJT_c4=float((H - gm4).norm()), note="G2-G4 = J2^T (c2-c4) + (J2-J4)^T c4; c = dL/dm of the MAE part")
    dc = np.array(r2["dLdm"]) - np.array(r4["dLdm"]); order_rows = np.argsort(-np.abs(dc))
    d["rows"] = [dict(row=int(i), label=float(y[i]), m4=r4["m"][i], m2=r2["m"][i], dm=r2["m"][i] - r4["m"][i], u4=r4["u"][i], c4=r4["dLdm"][i], c2=r2["dLdm"][i], dc=float(dc[i])) for i in range(EB)]
    d["share_of_sum_abs_dc_in_top3_rows"] = float(np.abs(dc)[order_rows[:3]].sum() / max(np.abs(dc).sum(), 1e-30))
    d["saturated_rows_u_lt_1e-3_or_gt_1-1e-3"] = int(sum(1 for u_ in r4["u"] if u_ < 1e-3 or u_ > 1 - 1e-3))
    d["near_zero_note"] = "relative differences use the norm of the first (micro 4) gradient as denominator and, separately, of the second; values with a denominator < 1e-6 are not interpretable"
    out["conditions"][prec] = dict(status="ok", micro4={k: v for k, v in r4.items() if not k.startswith("_")}, micro2={k: v for k, v in r2.items() if not k.startswith("_")}, comparison_micro4_vs_micro2=d)
    torch.save({k: r4[k] for k in ("_gCE", "_gMAE")} | {"micro2_" + k: r2[k] for k in ("_gCE", "_gMAE", "_H")}, OUT / f"grads_{a.weights}_{prec}.pt")
    out["comparisons"][prec + "_mem_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
out["seconds"] = time.time() - t0
json.dump(out, open(OUT / f"micro386_{a.weights}.json", "w"), indent=1)
print(json.dumps({p: ({k: v for k, v in c["comparison_micro4_vs_micro2"].items() if k not in ("rows",)} if c.get("status") == "ok" else c) for p, c in out["conditions"].items()}, indent=1)[:6000])
