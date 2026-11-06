"""Loss library of the ESCI four-loss pilot (experiment log 169; Qwen3-4B, ESCI-T1, seed 0). Pure torch; no tokenizer needed for the tested parts.
A  token CE     : esci_core.ce_rows (unchanged, the 165 implementation)
B  label BCE    : mean over rows of softplus(m) - y m, m = s_plus - s_minus, s_label = sum of log P(label tokens + terminator) (teacher forced, FP32 head / log_softmax of the evaluation path)
C  soft kappa   : q = p = sigmoid(m);  D  AnyLoss-like kappa: q = sigmoid(alpha (p - 0.5)), alpha = 10 (fixed beforehand)
C / D: r = mean(y), s = mean(q), a = mean(y q), D = r + s - 2 r s, U = r + s - 2 a, loss = U / clamp_min(D, eps) (= 1 - soft kappa for D > eps), eps = 1e-6; the denominator is NOT detached; one loss over the 16 rows of the update (never an average of micro-batch kappas).
Because the 16-row graph does not fit in memory, C / D use an exact two-pass scheme: pass 1 (no grad) computes the 16 margins, loss_from_m gives g = dL/dm in FP32, pass 2 recomputes each micro-batch with grad (same parameters, same dropout RNG state) and back-propagates sum_i g_i m_i,
which is the chain rule of the full-batch loss."""
import torch
import torch.nn.functional as F

EPS = 1e-6
ALPHA = 10.0


def kappa_stats(y, q):
    r, s, a = y.mean(), q.mean(), (y * q).mean()
    return r, s, a, r + s - 2 * r * s, r + s - 2 * a


def loss_from_m(kind, m, y, alpha=ALPHA, eps=EPS):
    """m, y: 1-d FP32/FP64 tensors of the same length (the whole update). Returns the scalar loss."""
    if kind == "bce":
        return (F.softplus(m) - y * m).mean()
    p = torch.sigmoid(m)
    q = p if kind == "softk" else torch.sigmoid(alpha * (p - 0.5))
    _, _, _, D, U = kappa_stats(y, q)
    return U / D.clamp_min(eps)


def grad_wrt_m(kind, m, y):
    """dL/dm (FP32) of the full-batch loss; autograd on the margin vector only."""
    mm = m.detach().clone().requires_grad_(True)
    loss = loss_from_m(kind, mm, y)
    (g,) = torch.autograd.grad(loss, mm)
    return g, float(loss.detach())


def batch_diagnostics(kind, m, y, alpha=ALPHA, eps=EPS):
    p = torch.sigmoid(m)
    q = p if kind == "softk" else torch.sigmoid(alpha * (p - 0.5))
    r, s, a, D, U = kappa_stats(y, q)
    return {"r": float(r), "s": float(s), "a": float(a), "D": float(D), "U": float(U), "clamped": bool(D <= eps), "single_class": bool(float(r) in (0.0, 1.0)),
            "p_sat": float(((p < 1e-3) | (p > 1 - 1e-3)).float().mean()), "q_sat": float(((q < 1e-3) | (q > 1 - 1e-3)).float().mean()), "p_mean": float(p.mean()), "m_mean": float(m.mean()),
            "hard_pos_rate": float((m > 0).float().mean())}


def seq_logprob_sums(core, norm, w_head, store, seqs, pad, device):
    """seqs: list of (prefix_ids, response_ids, terminator_id); returns a tensor [len(seqs)] = sum of log P over the response tokens + the terminator (FP32 norm / head / log_softmax at the supervised positions)."""
    full = [list(p) + list(r) + [t] for p, r, t in seqs]
    mx = max(len(s) for s in full)
    ids = torch.tensor([s + [pad] * (mx - len(s)) for s in full], device=device)
    att = torch.tensor([[1] * len(s) + [0] * (mx - len(s)) for s in full], device=device)
    core(input_ids=ids, attention_mask=att, use_cache=False)          # forward pre-hook of the final norm fills store["h"] (not detached: the graph is kept)
    H = store["h"]
    out = []
    for j, (p, r, t) in enumerate(seqs):
        st, n = len(p), len(r)
        pos = torch.arange(st - 1, st + n, device=device)
        y = norm(H[j, pos].float())
        lp = torch.log_softmax(y @ w_head.T, dim=-1)
        tgt = ids[j, st:st + n + 1]
        out.append(lp.gather(1, tgt.unsqueeze(-1)).squeeze(-1).sum())
    return torch.stack(out)


def margins(core, norm, w_head, store, seq_pos, seq_neg, pad, device):
    """m = s_plus - s_minus for n rows (the n Relevant-label and the n Irrelevant-label sequences are scored in one batch of 2n)."""
    n = len(seq_pos)
    s = seq_logprob_sums(core, norm, w_head, store, list(seq_pos) + list(seq_neg), pad, device)
    return s[:n] - s[n:], s[:n], s[n:]


def get_rng():
    st = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state()
    return st


def set_rng(st):
    torch.set_rng_state(st["cpu"])
    if "cuda" in st:
        torch.cuda.set_rng_state(st["cuda"])
