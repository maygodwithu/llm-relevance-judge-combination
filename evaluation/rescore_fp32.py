"""Re-score the TREC-DL 2019 candidates of final checkpoints with the final norm and the LM head in float32 (experiment log 130).

The reported score of a candidate is s = log P(Relevant) - log P(Irrelevant) (src/eval_rerank_mistral.py), computed from bf16 logits; the stored margins sit on a coarse
grid (steps of about 0.25) and documents within ~0.004 of each other are ordered by rounding noise. Here the SAME teacher-forced forward is used, the pre-norm hidden
state of the last block is captured, and at the supervised positions the final RMSNorm, the unembedding and log_softmax are evaluated in float32. For every candidate
both scores are produced from one forward pass: `run_bf16` (exactly the evaluator's computation) and `run_fp32`. The upstream block computations stay in bf16 (not changed).

    CUDA_VISIBLE_DEVICES=0 python evaluation/rescore_fp32.py --ckpt_root DIR --tags tagA tagB --ref_dir cpmr_experiment/pilot_rank_margin_k1/logs --out_dir DIR --local_files_only
The instruction text and the saved bf16 run of each tag are read from <ref_dir>/eval_<tag-without-_t100>_t100_T1_trec2019_scores.json; run_bf16 is compared with it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
if ROOT is None:
    raise RuntimeError("no .repo-root marker found in any parent directory")
for rel in ("", "evaluation", "cpmr_experiment/scripts", "cpmr_experiment/instructions_v2/scripts"):
    sys.path.insert(0, str(ROOT / rel))

import torch  # noqa: E402
import yaml  # noqa: E402
from ranx import Qrels, Run, evaluate  # noqa: E402

from layer_drift_lib import locate_blocks  # noqa: E402
from layer_drift_repr import load_adapter  # noqa: E402
from src.data_mistral import build_example_mistral  # noqa: E402
from src.eval_rerank import load_test_pool  # noqa: E402


@torch.no_grad()
def score_both(model, tok, instruction, query_text, cand_ids, doc_text, max_length, device, norm, w_head, store, batch_size):
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    s_bf, s_32 = {}, {}
    for i in range(0, len(cand_ids), batch_size):
        batch = cand_ids[i:i + batch_size]
        acc = {pid: {} for pid in batch}
        for label in ("Relevant", "Irrelevant"):
            ex = [build_example_mistral(tok, instruction, query_text, f"Passage: {doc_text[pid]}", label, max_length, enable_thinking=False) for pid in batch]
            mx = max(len(e.input_ids) for e in ex)
            ids = torch.tensor([e.input_ids + [pad_id] * (mx - len(e.input_ids)) for e in ex], device=device)
            lab = torch.tensor([e.labels + [-100] * (mx - len(e.labels)) for e in ex], device=device)
            att = torch.tensor([[1] * len(e.input_ids) + [0] * (mx - len(e.input_ids)) for e in ex], device=device)
            out = model(input_ids=ids, attention_mask=att, use_cache=False)
            # the evaluator's computation, bf16 logits
            lp = torch.log_softmax(out.logits[:, :-1], dim=-1)
            tl = torch.gather(lp, 2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            mask = (lab[:, 1:] != -100).float()
            seq_bf = (tl * mask).sum(dim=1)
            # float32 final norm + unembedding at the supervised positions only
            h = store["h"][:, :-1].float()                                 # (B, T-1, d) pre-norm output of the last block
            sel = mask.bool()
            hs = h[sel]                                                     # (n_sup, d)
            y = hs * torch.rsqrt(hs.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon) * norm.weight.float()
            logit = y @ w_head.T                                            # (n_sup, V) float32
            lp32 = torch.log_softmax(logit, dim=-1)
            tgt = ids[:, 1:][sel]
            tok_lp = lp32.gather(1, tgt.unsqueeze(-1)).squeeze(-1)
            rows = sel.nonzero()[:, 0]
            seq32 = torch.zeros(len(batch), device=device, dtype=torch.float32).index_add_(0, rows, tok_lp)
            for j, pid in enumerate(batch):
                acc[pid][label] = (seq_bf[j].item(), seq32[j].item())
        for pid in batch:
            s_bf[pid] = acc[pid]["Relevant"][0] - acc[pid]["Irrelevant"][0]
            s_32[pid] = acc[pid]["Relevant"][1] - acc[pid]["Irrelevant"][1]
    return s_bf, s_32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen3_4b_pilot_2x2.yaml")
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--tags", nargs="+", default=[], help="checkpoint tags scored with T1 (output rescore_<tag>.json)")
    ap.add_argument("--jobs", nargs="+", default=[], help="tag:Ti pairs scored with instruction Ti (output rescore_<tag>__<Ti>.json); a finished output is skipped")
    ap.add_argument("--manifest", default="cpmr_experiment/instructions_v2/manifest/instructions_v2.json")
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--ref_dir2", default="cpmr_experiment/eval_own_prompt", help="second place of saved own-instruction scores: <tag>__<Ti>_trec<year>_scores.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--year", default="2019")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--max_queries", type=int, default=0, help="debug: only the first n queries, nothing durable is compared")
    args = ap.parse_args()
    from src.model import load_model_and_tokenizer

    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / args.config))
    loaded = load_model_and_tokenizer(model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
                                      target_modules=cfg["lora"]["target_modules"], gradient_checkpointing=False, local_files_only=args.local_files_only)
    model, tok = loaded.model, loaded.tokenizer
    model.eval()
    device = next(model.parameters()).device
    _, _, norm, lm_head = locate_blocks(model)
    store = {}
    norm.register_forward_pre_hook(lambda m, a: store.__setitem__("h", a[0].detach()))
    w_head = lm_head.weight.detach().float()
    queries, cand_ids, doc_text, qrels = load_test_pool(year=args.year)
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(ROOT / args.manifest, encoding="utf-8"))["entries"]
    work = [(tag, "T1", False) for tag in args.tags] + [(j.split(":")[0], j.split(":")[1], True) for j in args.jobs]
    loaded_tag = None
    for tag, pid, new_name in work:
        out_file = out / (f"rescore_{tag}__{pid}.json" if new_name else f"rescore_{tag}.json")
        if out_file.exists() and not args.max_queries:
            print(f"[rescore] skip {tag} {pid} (exists)", flush=True)
            continue
        if loaded_tag != tag:
            ck = load_adapter(model, ROOT / args.ckpt_root / tag / "ckpt_final.pt")
            loaded_tag = tag
        instruction = manifest[pid]["full"]
        refs = [ROOT / args.ref_dir / f"eval_{tag}_{pid}_trec{args.year}_scores.json", ROOT / args.ref_dir2 / f"{tag}__{pid}_trec{args.year}_scores.json"]
        ref = next((r for r in refs if r.exists()), None)
        refd = json.load(open(ref, encoding="utf-8")) if ref else None
        if refd is not None and refd["instruction"] != instruction:
            print(f"[rescore] WARNING {tag} {pid}: the manifest instruction differs from the saved run's", flush=True)
        run_bf, run_32 = {}, {}
        for n, (qid, cids) in enumerate(cand_ids.items()):
            if args.max_queries and n >= args.max_queries:
                break
            if qid not in queries or not cids:
                continue
            run_bf[qid], run_32[qid] = score_both(model, tok, instruction, queries[qid], cids, doc_text, args.max_length, device, norm, w_head, store, args.batch_size)
        qg = {q: g for q, g in qrels.items() if q in run_bf}
        res = {}
        for name, run in (("bf16", run_bf), ("fp32", run_32)):
            res[name] = float(evaluate(Qrels(qg), Run(run), "ndcg@10", make_comparable=True))
        n_all = sum(len(v) for v in run_bf.values())
        if refd is not None:
            dmax = max(abs(run_bf[q][d] - refd["run"][q][d]) for q in run_bf for d in run_bf[q] if q in refd["run"] and d in refd["run"][q])
            n_equal = sum(1 for q in run_bf for d in run_bf[q] if run_bf[q][d] == refd["run"][q][d])
        else:
            dmax, n_equal = None, None
        print(f"[rescore] {tag} {pid}: nDCG@10 bf16 {res['bf16']:.4f} fp32 {res['fp32']:.4f}; run_bf16 vs saved run: " +
              (f"max |diff| {dmax:.3e}, identical {n_equal}/{n_all}" if dmax is not None else "no saved run") + f" ({time.time() - t0:.0f}s)", flush=True)
        if not args.max_queries:
            with open(out_file, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"tag": tag, "prompt": pid, "seed": ck.get("seed"), "instruction": instruction, "run_fp32": run_32, "run_bf16": run_bf, "qrels_graded": qg,
                           "ndcg10_bf16": res["bf16"], "ndcg10_fp32": res["fp32"], "max_abs_diff_vs_saved": dmax, "n_identical_vs_saved": n_equal, "n_candidates": n_all}, f)
    print(f"[rescore] done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
