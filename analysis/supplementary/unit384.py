"""CPU unit checks for the 384 objectives (no GPU, no model): the loss terms of mixmae / mixbce as the trainer computes them (per micro-batch of 4 rows, sum / 16) against the library definitions and full-batch autograd.
  mixmae: lam x mean|y - sigmoid(m)| with lam = 2      mixbce: lam x mean BCEWithLogits(m, y) with lam = 1
Checks: (1) micro-batch decomposition equals the full-batch loss; (2) the gradient accumulated over micro-batches equals the full-batch gradient; (3) MAE term equals U of the library's soft-kappa statistics (U = r + s - 2a); (4) on balanced batches 2 x MAE equals the library soft-kappa loss and its gradient; (5) BCE term equals library 'bce'.
    CUDA_VISIBLE_DEVICES= python unit384.py <project root>"""
import sys, json
import torch
sys.path.insert(0, sys.argv[1] + "/methods")
import esci_loss_lib as LIB
torch.manual_seed(384)
res = dict(n=0, micro_vs_full_loss=0.0, micro_vs_full_grad=0.0, U_vs_MAE=0.0, balanced_2mae_vs_softk_loss=0.0, balanced_2mae_vs_softk_grad=0.0, bce_vs_lib=0.0)
for dtype in (torch.float64, torch.float32):
    for _ in range(300):
        res["n"] += 1
        m = (torch.randn(16, dtype=dtype) * torch.empty(16, dtype=dtype).uniform_(0.1, 8)).requires_grad_(True)
        npos = int(torch.randint(1, 16, (1,))) if _ % 2 else 8
        y = torch.zeros(16, dtype=dtype); y[torch.randperm(16)[:npos]] = 1.0
        ng = 16; lam_mae, lam_bce = 2.0, 1.0
        # trainer form: sum over micro-batches of 4 rows, each term sum / ng
        mae_micro = sum(lam_mae * (y[i:i + 4] - torch.sigmoid(m[i:i + 4])).abs().sum() / ng for i in range(0, 16, 4))
        bce_micro = sum(lam_bce * torch.nn.functional.binary_cross_entropy_with_logits(m[i:i + 4], y[i:i + 4], reduction="sum") / ng for i in range(0, 16, 4))
        mae_full = lam_mae * (y - torch.sigmoid(m)).abs().mean(); bce_full = lam_bce * torch.nn.functional.binary_cross_entropy_with_logits(m, y, reduction="mean")
        g1, = torch.autograd.grad(mae_micro, m, retain_graph=True); g2, = torch.autograd.grad(mae_full, m, retain_graph=True)
        h1, = torch.autograd.grad(bce_micro, m, retain_graph=True); h2, = torch.autograd.grad(bce_full, m, retain_graph=True)
        res["micro_vs_full_loss"] = max(res["micro_vs_full_loss"], float((mae_micro - mae_full).abs()), float((bce_micro - bce_full).abs()))
        res["micro_vs_full_grad"] = max(res["micro_vs_full_grad"], float((g1 - g2).abs().max()), float((h1 - h2).abs().max()))
        u = torch.sigmoid(m.detach()); r, s, a, D, U = LIB.kappa_stats(y, u)
        res["U_vs_MAE"] = max(res["U_vs_MAE"], float((U - (y - u).abs().mean()).abs()))
        res["bce_vs_lib"] = max(res["bce_vs_lib"], float((LIB.loss_from_m("bce", m.detach(), y) - bce_full.detach() / lam_bce).abs()))
        if npos == 8:
            gk, lk = LIB.grad_wrt_m("softk", m.detach(), y)
            res["balanced_2mae_vs_softk_loss"] = max(res["balanced_2mae_vs_softk_loss"], abs(lk - float(mae_full)))
            res["balanced_2mae_vs_softk_grad"] = max(res["balanced_2mae_vs_softk_grad"], float((gk - g2).abs().max()))
tol = {"float64 and float32 pooled": 1e-5}
res["criterion_max_abs_diff"] = 1e-5
res["passed"] = all(res[k] <= 1e-5 for k in ("micro_vs_full_loss", "micro_vs_full_grad", "U_vs_MAE", "balanced_2mae_vs_softk_loss", "balanced_2mae_vs_softk_grad", "bce_vs_lib"))
print(json.dumps(res, indent=1))
