"""
Mistral variant of src/eval_rerank.py's score_candidates/evaluate_rerank: identical
teacher-forced conditional log-likelihood scoring (s = log P(Relevant) - log P(Irrelevant),
no generation), just built via build_example_mistral (the model's official chat template)
instead of build_example. Reuses load_test_pool from src.eval_rerank unchanged -- the fixed
BM25-top-100 candidate pool is model-independent.
"""
import json
from pathlib import Path

import torch
from ranx import Qrels, Run, evaluate

from src.data_mistral import build_example_mistral
from src.eval_rerank import load_test_pool, CANDIDATE_CAP, CANDIDATE_SEED


@torch.no_grad()
def score_candidates_mistral(model, tokenizer, instruction, query_text, cand_ids, doc_text,
                              max_length, device, batch_size: int = 16, enable_thinking: bool = None,
                              build_example_fn=None):
    """build_example_fn: left at None this calls build_example_mistral exactly as before.
    The v2 evaluation path passes a builder that resolves the passage through the passage cache
    first, so evaluation sees the same preserved passage training saw
    (cpmr_experiment/instructions_v2/scripts/v2_input_path.py)."""
    was_training = model.training
    model.eval()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    scores = {pid: {} for pid in cand_ids}
    for i in range(0, len(cand_ids), batch_size):
        batch_ids = cand_ids[i:i + batch_size]
        for cand_label in ("Relevant", "Irrelevant"):
            builder = build_example_fn or build_example_mistral
            examples = [
                builder(tokenizer, instruction, query_text,
                        f"Passage: {doc_text[pid]}", cand_label, max_length,
                        enable_thinking=enable_thinking)
                for pid in batch_ids
            ]
            max_len = max(len(e.input_ids) for e in examples)
            input_ids, labels, attn = [], [], []
            for e in examples:
                pad_n = max_len - len(e.input_ids)
                input_ids.append(e.input_ids + [pad_id] * pad_n)
                labels.append(e.labels + [-100] * pad_n)
                attn.append([1] * len(e.input_ids) + [0] * pad_n)
            input_ids = torch.tensor(input_ids, device=device)
            labels_t = torch.tensor(labels, device=device)
            attn_t = torch.tensor(attn, device=device)

            out = model(input_ids=input_ids, attention_mask=attn_t, use_cache=False)
            logprobs = torch.log_softmax(out.logits[:, :-1], dim=-1)
            target = input_ids[:, 1:]
            token_logprob = torch.gather(logprobs, 2, target.unsqueeze(-1)).squeeze(-1)
            mask = (labels_t[:, 1:] != -100).float()
            seq_logprob = (token_logprob * mask).sum(dim=1)

            for j, pid in enumerate(batch_ids):
                scores[pid][cand_label] = seq_logprob[j].item()

    if was_training:
        model.train()
    return {pid: s["Relevant"] - s["Irrelevant"] for pid, s in scores.items()}


def evaluate_rerank_mistral(model, tokenizer, instruction: str, max_length: int, device,
                             cap: int = CANDIDATE_CAP, seed: int = CANDIDATE_SEED,
                             batch_size: int = 4, save_path: str = None, year: str = "2019",
                             enable_thinking: bool = None) -> dict:
    queries, cand_ids, doc_text, qrels = load_test_pool(cap=cap, seed=seed, year=year)

    run_dict = {}
    for qid, cids in cand_ids.items():
        if qid not in queries or not cids:
            continue
        run_dict[qid] = score_candidates_mistral(model, tokenizer, instruction, queries[qid],
                                                   cids, doc_text, max_length, device, batch_size,
                                                   enable_thinking=enable_thinking)

    qrels_graded = {qid: g for qid, g in qrels.items() if qid in run_dict}
    qrels_binary = {
        qid: {pid: (1 if rel >= 2 else 0) for pid, rel in g.items()}
        for qid, g in qrels_graded.items()
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"instruction": instruction, "run": run_dict, "qrels_graded": qrels_graded},
                      f, ensure_ascii=False)

    run = Run(run_dict)
    ndcg = evaluate(Qrels(qrels_graded), run, ["ndcg@5", "ndcg@10"], make_comparable=True)
    mrr = evaluate(Qrels(qrels_binary), run, ["mrr@5", "mrr@10"], make_comparable=True)
    return {
        "ndcg@5": float(ndcg["ndcg@5"]), "ndcg@10": float(ndcg["ndcg@10"]),
        "mrr@5": float(mrr["mrr@5"]), "mrr@10": float(mrr["mrr@10"]),
        "n_queries": len(run_dict),
    }
