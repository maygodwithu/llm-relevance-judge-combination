"""Gemma-3-4B-IT evaluation of the modified R7 relevance instructions (experiment log 135): TREC-DL 2019 / 2020, every original candidate pair (judged and unjudged), label + end-of-turn scoring in FP32.
Model: google/gemma-3-4b-it (local snapshot, revision recorded), Gemma3ForConditionalGeneration, text path only; bf16 transformer blocks, the final RMSNorm (applied to float32 input -> float32 output), the tied unembedding and log_softmax are evaluated in float32 at the supervised positions only
(same computation scheme as loss_score_fp32.score_both_ext of the Qwen experiments; the vocabulary is 262,208, no final logit soft-capping in this model).
Prompt: official chat template of the tokenizer (<bos><start_of_turn>user\n{instruction}\n\nQuery: {query}\nPassage: {passage}<end_of_turn>\n<start_of_turn>model\n), tokenised with add_special_tokens=False, followed by the answer tokens of the label and the turn terminator.
Terminator: <end_of_turn> (id 106, the token Gemma-IT emits after an answer; the Qwen analogue is <|im_end|>) is the PRIMARY 'EOS'. The same forward also gives log P(<eos> (id 1) | label) and the label-only sum, so the alternative scorings are stored without extra cost.
Per document and label: lp_label (sum of the answer tokens), lp_eot, lp_eos, lp_first (first answer token), n_label_tokens. Primary margin m = [lp_label + lp_eot](Relevant) - [lp_label + lp_eot](Irrelevant); Relevant if m > 0.
    CUDA_VISIBLE_DEVICES=0 python methods/gemma_r7_eval.py --jobs Base:R0 Base:R1 ... --instr_file <json> --out_dir <dir> --local_files_only"""
import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
for rel in ("", "methods", "evaluation", "cpmr_experiment/scripts"):
    sys.path.insert(0, str(ROOT / rel))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.eval_rerank import load_test_pool  # noqa: E402

EOT, EOS = 106, 1


def build(tok, instruction, query, doc, label, max_length):
    resp = tok(label, add_special_tokens=False)["input_ids"]
    budget = max_length - len(resp) - 1

    def prefix(ptext):
        t = tok.apply_chat_template([{"role": "user", "content": f"{instruction}\n\nQuery: {query}\n{ptext}"}], tokenize=False, add_generation_prompt=True)
        return tok(t, add_special_tokens=False)["input_ids"]
    text = f"Passage: {doc}"
    pre, shrunk = prefix(text), False
    while len(pre) > budget and len(text) > 20:
        text = text[: int(len(text) * 0.8)]
        pre = prefix(text)
        shrunk = True
    return pre, resp, shrunk


@torch.no_grad()
def score_batch(core, norm, w_head, store, tok, instruction, query, batch_docs, doc_text, max_length, device):
    pad = tok.pad_token_id
    out = {d: {} for d in batch_docs}
    shr = 0
    for label in ("Relevant", "Irrelevant"):
        ex = [build(tok, instruction, query, doc_text[d], label, max_length) for d in batch_docs]
        shr += sum(e[2] for e in ex)
        seqs = [e[0] + e[1] + [EOT] for e in ex]
        mx = max(len(s) for s in seqs)
        ids = torch.tensor([s + [pad] * (mx - len(s)) for s in seqs], device=device)
        att = torch.tensor([[1] * len(s) + [0] * (mx - len(s)) for s in seqs], device=device)
        core(input_ids=ids, attention_mask=att, use_cache=False)          # the pre-norm hook fills store["h"]
        H = store["h"]
        rows, tgt, owner = [], [], []
        for j, (e, s) in enumerate(zip(ex, seqs)):
            st, n = len(e[0]), len(e[1])
            for k in range(n + 1):                                        # n answer tokens + the terminator
                rows.append((j, st + k - 1)); tgt.append(s[st + k]); owner.append((j, k, n))
        jj = torch.tensor([r[0] for r in rows], device=device); pp = torch.tensor([r[1] for r in rows], device=device)
        hs = H[jj, pp].float()
        y = norm(hs)                                                      # Gemma3RMSNorm: float32 in -> float32 out
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


def ndcg10(margins, rel):
    import math
    top = sorted(margins, key=lambda d: -margins[d])[:10]
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(top))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(rel.values(), reverse=True)[:10]))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", required=True, help="model:Ri; Base or a checkpoint tag (path via --ckpt_map)")
    ap.add_argument("--instr_file", required=True)
    ap.add_argument("--ckpt_map", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="configs/gemma3_4b.yaml")
    ap.add_argument("--max_length", type=int, default=500)
    ap.add_argument("--years", nargs="+", default=["2019", "2020"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_queries", type=int, default=0, help="debug: nothing durable is written")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()
    from layer_drift_repr import load_adapter
    from src.model import load_model_and_tokenizer
    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / args.config))
    loaded = load_model_and_tokenizer(model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
                                      gradient_checkpointing=False, local_files_only=args.local_files_only)
    model, tok = loaded.model, loaded.tokenizer
    model.eval()
    inner = model.base_model.model
    core = inner.model                       # Gemma3Model: the language model is called without the vocabulary head
    norm = core.language_model.norm
    w_head = inner.lm_head.weight.detach().float()
    device = next(model.parameters()).device
    store = {}
    norm.register_forward_pre_hook(lambda m, a: store.__setitem__("h", a[0].detach()))
    ins = json.load(open(ROOT / args.instr_file, encoding="utf-8"))
    ins = {k: v["full"] for k, v in ins["entries"].items()}
    cmap = json.load(open(ROOT / args.ckpt_map)) if args.ckpt_map else {}
    pools = {y: load_test_pool(year=y) for y in args.years}
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    loaded_tag, ck = None, {}
    revision = getattr(inner.config, "_commit_hash", None)
    for job in args.jobs:
        mname, ri = job.split(":")
        text = ins[ri]
        for year in args.years:
            f = out / f"gemmaeval_{mname}__{ri}__{year}.json.gz"
            if f.exists() and not args.max_queries:
                print(f"[gemma] skip {job} {year}", flush=True)
                continue
            if loaded_tag != mname:
                if mname == "Base":
                    assert loaded_tag in (None, "Base")
                    assert max(float(p.detach().abs().max()) for n, p in model.named_parameters() if "lora_B" in n) == 0.0
                    ck = {"seed": None, "global_step": 0}
                else:
                    ck = load_adapter(model, ROOT / cmap[mname])
                loaded_tag = mname
            queries, cand_ids, doc_text, qrels = pools[year]
            raw, t1, n_shr, n_docs = {}, time.time(), 0, 0
            for n, (qid, cids) in enumerate(cand_ids.items()):
                if args.max_queries and n >= args.max_queries:
                    break
                if qid not in queries or not cids:
                    continue
                res = {}
                for i in range(0, len(cids), args.batch_size):
                    r, s = score_batch(core, norm, w_head, store, tok, text, queries[qid], cids[i:i + args.batch_size], doc_text, args.max_length, device)
                    res.update(r); n_shr += s
                g = qrels.get(qid)
                raw[qid] = {d: [res[d]["Relevant"]["label"], res[d]["Relevant"]["eot"], res[d]["Relevant"]["eos"], res[d]["Irrelevant"]["label"], res[d]["Irrelevant"]["eot"], res[d]["Irrelevant"]["eos"],
                                res[d]["Relevant"]["first"], res[d]["Irrelevant"]["first"], res[d]["Relevant"]["n"], res[d]["Irrelevant"]["n"], (-1 if (g is None or d not in g) else int(g[d]))] for d in cids}
                n_docs += len(cids)
            eval_q = [q for q in raw if q in qrels]
            nd = float(np.mean([ndcg10({d: v[0] + v[1] - v[3] - v[4] for d, v in raw[q].items()}, {d: int(x) for d, x in qrels[q].items()}) for q in eval_q]))
            m_all = np.array([v[0] + v[1] - v[3] - v[4] for q in raw for v in raw[q].values()])
            jj = [(v[0] + v[1] - v[3] - v[4], v[10]) for q in raw for v in raw[q].values() if v[10] >= 0]
            mj = np.array([a for a, b in jj]); hj = np.array([b >= 2 for a, b in jj])
            pr = mj > 0
            po = (pr == hj).mean(); pe = pr.mean() * hj.mean() + (1 - pr.mean()) * (1 - hj.mean())
            print(f"[gemma] {job} TREC-DL {year}: {len(raw)} queries, {n_docs} pairs, {len(jj)} judged; kappa {(po - pe) / (1 - pe):.4f}, nDCG@10 {nd:.4f}, Rel (all) {100 * (m_all > 0).mean():.1f}%, ties {(m_all == 0).sum()}, shortened passages {n_shr} "
                  f"({time.time() - t1:.0f}s)", flush=True)
            if not args.max_queries:
                meta = {"model": mname, "instruction": ri, "instruction_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "year": year, "max_length": args.max_length, "n_queries_scored": len(raw), "n_pairs_scored": n_docs,
                        "n_judged": len(jj), "n_passage_shortenings_counted_over_two_labels": n_shr, "model_name": cfg["model_name"], "model_revision": revision, "terminator_primary": "<end_of_turn> (106)", "terminator_alt": "<eos> (1)",
                        "fields": "raw[qid][doc] = [lp_label(R), lp_eot(R), lp_eos(R), lp_label(I), lp_eot(I), lp_eos(I), lp_first(R), lp_first(I), n_label_tokens(R), n_label_tokens(I), grade or -1]", "ndcg10_primary": nd, "seed": ck.get("seed"),
                        "global_step": ck.get("global_step"), "torch": torch.__version__, "batch_size": args.batch_size, "dtype_forward": "bfloat16", "head_norm": "float32"}
                tmp = f.with_suffix(".tmp")
                with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                    json.dump({"meta": meta, "raw": raw}, fh)
                tmp.replace(f)
    print(f"[gemma] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
