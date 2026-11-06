"""Shared pieces of the ESCI extension (experiment log 156; Qwen3-4B, second server): prompt builder, label+<|im_end|> scorer (same path as qwen8_r7_eval.score_batch / antique_eval: BF16 blocks, FP32 final norm / head / log_softmax at the supervised
positions, template with enable_thinking=False, right padding) and the CE row forward used by the trainer. The product block is the cached text of methods/esci_audit.py (does not depend on the instruction); it already carries its 'Product:' marker.
Prompt: '<instruction>\\n\\nQuery: <query>\\n<product block>' as the user turn of the official chat template; answer 'Relevant' / 'Irrelevant' + <|im_end|>."""
import torch
import torch.nn.functional as F

EOT, EOS, PAD = 151645, 151643, 151643
REL_FIRST, IRR_FIRST = 693, 48113


def build(tok, instruction, query, ptext, label, max_length):
    resp = tok(label, add_special_tokens=False)["input_ids"]
    budget = max_length - len(resp) - 1

    def prefix(pt):
        t = tok.apply_chat_template([{"role": "user", "content": f"{instruction}\n\nQuery: {query}\n{pt}"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tok(t, add_special_tokens=False)["input_ids"]
    text, pre, shrunk = ptext, None, False
    pre = prefix(text)
    while len(pre) > budget and len(text) > 20:                       # never expected to trigger (max_length reserves the whole instruction space); counted and reported
        text = text[: int(len(text) * 0.8)]
        pre = prefix(text)
        shrunk = True
    return pre, resp, shrunk


@torch.no_grad()
def score_batch(core, norm, w_head, store, tok, instruction, query, batch_docs, doc_text, max_length, device):
    out = {d: {} for d in batch_docs}
    shr = 0
    for label in ("Relevant", "Irrelevant"):
        ex = [build(tok, instruction, query, doc_text[d], label, max_length) for d in batch_docs]
        shr += sum(e[2] for e in ex)
        seqs = [e[0] + e[1] + [EOT] for e in ex]
        mx = max(len(s) for s in seqs)
        ids = torch.tensor([s + [PAD] * (mx - len(s)) for s in seqs], device=device)
        att = torch.tensor([[1] * len(s) + [0] * (mx - len(s)) for s in seqs], device=device)
        core(input_ids=ids, attention_mask=att, use_cache=False)          # the pre-norm hook fills store["h"]
        H = store["h"]
        rows, tgt, owner = [], [], []
        for j, (e, s) in enumerate(zip(ex, seqs)):
            st, n = len(e[0]), len(e[1])
            for k in range(n + 1):
                rows.append((j, st + k - 1)); tgt.append(s[st + k]); owner.append((j, k, n))
        jj = torch.tensor([r[0] for r in rows], device=device); pp = torch.tensor([r[1] for r in rows], device=device)
        y = norm(H[jj, pp].float())
        lp = torch.log_softmax(y @ w_head.T, dim=-1)
        tg = torch.tensor(tgt, device=device)
        lp_t = lp.gather(1, tg.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        lp_eos = lp[:, EOS].cpu().numpy()
        for idx, (j, k, n) in enumerate(owner):
            rec = out[batch_docs[j]].setdefault(label, {"label": 0.0, "first": None, "eot": None, "eos": None, "n": n})
            if k < n:
                rec["label"] += float(lp_t[idx])
                if k == 0:
                    rec["first"] = float(lp_t[idx])
            else:
                rec["eot"] = float(lp_t[idx]); rec["eos"] = float(lp_eos[idx])
    return out, shr


def ce_rows(core, head, tok, instruction, rows, max_length, device):
    """CE of one micro-batch of rows (list of (query, product block, label)); returns the per-row CE (token mean over the supervised tokens: answer tokens + <|im_end|>) and the number of shortened inputs."""
    ex = [build(tok, instruction, q, p, lab, max_length) for q, p, lab in rows]
    seqs = [e[0] + e[1] + [EOT] for e in ex]
    mx = max(len(s) for s in seqs)
    ids = torch.tensor([s + [PAD] * (mx - len(s)) for s in seqs], device=device)
    att = torch.tensor([[1] * len(s) + [0] * (mx - len(s)) for s in seqs], device=device)
    h = core(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state          # after the final norm (bf16)
    ces = []
    for j, (e, s) in enumerate(zip(ex, seqs)):
        st, n = len(e[0]), len(e[1])
        pos = torch.arange(st - 1, st + n, device=device)
        logits = head(h[j, pos]).float()
        ces.append(F.cross_entropy(logits, ids[j, st:st + n + 1]))
    return torch.stack(ces), sum(int(e[2]) for e in ex), sum(len(s) for s in seqs)
