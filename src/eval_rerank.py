"""
TREC DL 2019 held-out test set에서 nDCG@10 / MRR@10을 계산한다 (WSDM 논문의 MS MARCO
평가 지표). ESCI의 confusion-matrix 기반 binary accuracy와 달리, 여기서는 query당
후보 passage 여러 개를 모델 스코어로 재순위화한 뒤 실제 랭킹 지표를 낸다.

2026-08-08 수정 히스토리 (원인: "qrels 판정 문서는 후보 pool에 없더라도 항상
포함시킨다"고 문서화했지만 실제로는 BM25 top-1000과 qrels를 합친 뒤 cap=100으로
균등 무작위 샘플링해서 qrels도 다른 후보와 똑같이 탈락시켰던 버그. 원래 코드는
src/eval_rerank_cap100_backup.py로 보존):
  1. 전체 union(BM25 top-1000 ∪ qrels, 캡 없음): 쿼리당 평균 ~1063개, 호출 10배
     넘게 증가 -- 사용자 요청으로 폐기.
  2. relevance>=2 강제포함 + cap=100까지 negative로 채움: relevant 비율이 평균
     ~51%까지 치솟아 MRR@10이 세 시드에서 전부 정확히 0.9264로 포화 -- 폐기.
  3. relevance>=2 강제포함 + negative 고정 100/500개 추가: 여전히 MRR@10이
     0.90+ 수준으로 포화, 또는(500개) 계산량이 원본 논문 관행(top-1000 전체)에
     맞먹을 정도로 커짐 -- 폐기.
  4. qrels 판정 문서만 그대로 pool로 사용(BM25 후보 안 섞음): 포화는 안 됐지만
     밀도가 여전히 높고(~평균 27%), "top-1000에서 top-100"이라는 원래 설계
     의도와는 다른 성격의(pool 기반) 평가가 됨.

**핵심 재확인**: `ds.scoreddocs_iter()`가 주는 BM25 top-1000은 score가 전부
0.0으로 캐시돼 있어 원래 BM25 순위를 복원할 수 없다(직접 확인). 기존 코드는
이 때문에 "top-100"을 무작위 샘플링으로 근사했었는데, 이는 진짜 BM25 top-100보다
훨씬 노이즈가 심한(운으로만 결정되는) 근사다.

최종 방식(사용자 결정, 2026-08-08): query마다 그 query의 BM25 top-1000(scoreddocs,
41/43개 쿼리는 정확히 1000개) 후보 문서 텍스트로 **직접 `rank_bm25`(BM25Okapi)
인덱스를 만들고, 그 query로 검색해서 진짜 top-100을 뽑는다** -- 43번의 저렴한
로컬 BM25 검색으로 캐시에서 유실된 원래 순위를 사실상 복원하는 셈이다. 결과:
쿼리당 정확히 100개(1000개 미만인 2개 쿼리는 그 이하), 총 4,300쌍(원래 cap=100과
동일한 규모) -- 무작위가 아니라 진짜 BM25 순위 기반이라 relevant 문서가 자연스러운
(운이 아닌) 확률로 살아남는다.

2026-08-08 추가 수정: 위 BM25 인덱스를 매 `load_test_pool()` 호출마다(즉 매 평가마다)
즉석으로 다시 만들던 걸, 재현성을 위해 **한 번 계산해서 고정 파일로 저장**하고 이후엔
그 파일만 읽도록 변경(사용자 지적: "즉석에서 계산하게 해. 차이가 생길수도있는데"). 고정
파일: data/scenarios/msmarco_trec2019_bm25top100_pool.jsonl (4,142줄, query_id/query/
doc_id/doc_text/relevance 필드). 파일이 없으면 최초 1회 생성 스크립트
(scripts/build_msmarco_bm25_pool.py 또는 이 파일의 __main__)로 만들어야 한다.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import ir_datasets
import torch
import yaml
from ranx import Qrels, Run, evaluate
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import build_example
from src.model import load_model_and_tokenizer

ROOT = Path(__file__).resolve().parent.parent
DATASET_ID = "msmarco-passage/trec-dl-2019/judged"
BM25_POOL_PATH = ROOT / "data" / "scenarios" / "msmarco_trec2019_bm25top100_pool.jsonl"
CANDIDATE_CAP = 100  # 하위 호환(옛 --cap 인자 기본값용) -- load_test_pool은 더 이상 사용 안 함
CANDIDATE_SEED = 0

# 2026-08-20(사용자 요청): "2019 결과의 독립 재현 확인"을 위해 TREC DL 2020(judged)도
# 같은 방식(공식 scoreddocs 풀에서 진짜 BM25 top-100 복원, qrels 강제포함 없음 --
# FiQA와 달리 이 공식 풀은 이미 밀도가 높아서(2019: query당 평균 10.86개 relevant,
# zero-relevant query 16%) 강제포함이 굳이 필요없고, 두 연도를 정확히 같은 방법론으로
# 비교하기 위해 일부러 안 바꾼다)으로 pool을 만든다.
YEAR_DATASET_ID = {
    "2019": "msmarco-passage/trec-dl-2019/judged",
    "2020": "msmarco-passage/trec-dl-2020/judged",
}
YEAR_POOL_PATH = {
    "2019": BM25_POOL_PATH,
    "2020": ROOT / "data" / "scenarios" / "msmarco_trec2020_bm25top100_pool.jsonl",
}

WELL_SPECIFIED_INSTRUCTION = (
    "You are a passage relevance judge. Given a search query and a "
    "candidate passage, decide whether the passage is Relevant or Irrelevant.\n"
    "Definition: label the passage Relevant ONLY if it directly and "
    "substantially answers the query's information need with a clear, "
    "dedicated answer — not merely a passage that touches on the same topic.\n"
    "A passage that is only loosely or tangentially related to the query's "
    "subject, or that mentions the topic without actually answering it, "
    "must be labeled Irrelevant.\n"
    "Respond with exactly one word: \"Relevant\" or \"Irrelevant\"."
)

UNDER_SPECIFIED_INSTRUCTION = (
    "You are a passage relevance judge. Given a search query and a "
    "candidate passage, decide whether the passage answers the query.\n"
    "Respond with exactly one word: \"Relevant\" or \"Irrelevant\"."
)


def _tokenize(text: str):
    return text.lower().split()


def build_bm25_pool_file(cap: int = CANDIDATE_CAP, out_path: Path = None, year: str = "2019"):
    """query마다 BM25 top-1000 후보로 BM25Okapi 인덱스를 만들어 그 query로 검색,
    상위 cap(=100)개를 뽑아 out_path(기본 BM25_POOL_PATH)에 저장한다. 재현성을
    위해 이 함수는 파일을 새로 만들 때만(최초 1회, 또는 의도적 재생성 시) 호출하고,
    평가 시엔 항상 load_test_pool()이 저장된 파일을 읽는다. year="2019"|"2020"."""
    import json
    out_path = out_path or YEAR_POOL_PATH[year]
    ds = ir_datasets.load(YEAR_DATASET_ID[year])
    queries = {q.query_id: q.text for q in ds.queries_iter()}

    qrels = {}
    for qr in ds.qrels_iter():
        qrels.setdefault(qr.query_id, {})[qr.doc_id] = qr.relevance

    scoreddocs = {}
    for sd in ds.scoreddocs_iter():
        scoreddocs.setdefault(sd.query_id, set()).add(sd.doc_id)

    store = ds.docs_store()
    all_pool_ids = sorted({pid for ids in scoreddocs.values() for pid in ids})
    all_doc_text = {doc_id: doc.text for doc_id, doc in store.get_many(all_pool_ids).items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for qid in sorted(queries, key=lambda q: int(q)):
            ids = sorted(scoreddocs.get(qid, []))
            texts = [all_doc_text[pid] for pid in ids]
            bm25 = BM25Okapi([_tokenize(t) for t in texts])
            scores = bm25.get_scores(_tokenize(queries[qid]))
            ranked = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
            top_ids = sorted(ids[i] for i in ranked[:cap])
            for pid in top_ids:
                f.write(json.dumps({
                    "query_id": qid, "query": queries[qid], "doc_id": pid,
                    "doc_text": all_doc_text[pid],
                    "relevance": qrels.get(qid, {}).get(pid),
                }, ensure_ascii=False))
                f.write(chr(10))
    return out_path


def load_test_pool(cap: int = CANDIDATE_CAP, seed: int = CANDIDATE_SEED, year: str = "2019"):
    """반환: queries(qid->text), cand_ids(qid->[pid,...]), doc_text(pid->text),
    qrels(qid->{pid: 0~3 등급}). YEAR_POOL_PATH[year]에 미리 저장해둔 고정 pool
    파일을 읽는다(재현성을 위해 매 호출마다 재계산하지 않음 -- 파일이 없으면
    build_bm25_pool_file()로 먼저 생성해야 함). cap/seed 인자는 하위 호환용."""
    import json
    pool_path = YEAR_POOL_PATH[year]
    if not pool_path.exists():
        raise SystemExit(
            f"{pool_path} not found; run build_bm25_pool_file(year='{year}') once to create it"
        )
    queries, cand_ids, doc_text, qrels = {}, {}, {}, {}
    with open(pool_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qid, pid = r["query_id"], r["doc_id"]
            queries[qid] = r["query"]
            cand_ids.setdefault(qid, []).append(pid)
            doc_text[pid] = r["doc_text"]
            if r["relevance"] is not None:
                qrels.setdefault(qid, {})[pid] = r["relevance"]

    return queries, cand_ids, doc_text, qrels


@torch.no_grad()
def score_candidates(model, tokenizer, instruction, query_text, cand_ids, doc_text,
                      max_length, device, batch_size: int = 16, example_builder=None):
    """query 하나의 후보 pid마다 log P(Relevant) - log P(Irrelevant) 랭킹 스코어를 계산.
    rewriter.predict_label과 동일한 pointwise scoring 원리를 배치 처리로 확장한 것
    (query 1000개 x pid 최대 100개를 순차 처리하면 너무 느려 배치가 필수).
    example_builder: build_example과 동일한 시그니처의 대체 함수(기본값 build_example) --
    sftreg의 no-instruction semantic-responsiveness 조건처럼 프롬프트 구성 자체를
    바꿔야 할 때 사용."""
    example_builder = example_builder or build_example
    was_training = model.training
    model.eval()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    scores = {pid: {} for pid in cand_ids}
    for i in range(0, len(cand_ids), batch_size):
        batch_ids = cand_ids[i:i + batch_size]
        for cand_label in ("Relevant", "Irrelevant"):
            examples = [
                example_builder(tokenizer, instruction, query_text,
                                 f"Passage: {doc_text[pid]}", cand_label, max_length)
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
            # vocab이 커서(15만+) 전체 시퀀스 길이 x float32로 log_softmax를 만들면
            # 메모리가 급증한다(4B 모델 학습 직후라 이미 메모리 여유가 적음) — 상대
            # 비교만 필요하므로 float32 캐스팅 없이 원래 dtype(bf16)로 계산한다.
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


def evaluate_rerank(model, tokenizer, instruction: str, max_length: int, device,
                     cap: int = CANDIDATE_CAP, seed: int = CANDIDATE_SEED,
                     batch_size: int = 4, save_path: str = None, year: str = "2019",
                     example_builder=None) -> dict:
    queries, cand_ids, doc_text, qrels = load_test_pool(cap=cap, seed=seed, year=year)

    run_dict = {}
    for qid, cids in cand_ids.items():
        if qid not in queries or not cids:
            continue
        run_dict[qid] = score_candidates(model, tokenizer, instruction, queries[qid],
                                          cids, doc_text, max_length, device, batch_size,
                                          example_builder=example_builder)

    qrels_graded = {qid: g for qid, g in qrels.items() if qid in run_dict}
    qrels_binary = {
        qid: {pid: (1 if rel >= 2 else 0) for pid, rel in g.items()}
        for qid, g in qrels_graded.items()
    }

    if save_path:
        # 후보별 원점수를 남겨두면, 나중에 다른 cutoff(@3, @20 등)를 보고 싶을 때
        # 모델을 다시 학습하지 않고 이 파일만 다시 읽어서 ranx로 재계산할 수 있다.
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"instruction": instruction, "run": run_dict, "qrels_graded": qrels_graded},
                      f, ensure_ascii=False)

    run = Run(run_dict)
    # 후보 스코어는 이미 다 계산돼 있으니 cutoff별 재계산은 공짜 — @5/@10 둘 다 낸다.
    ndcg = evaluate(Qrels(qrels_graded), run, ["ndcg@5", "ndcg@10"], make_comparable=True)
    mrr = evaluate(Qrels(qrels_binary), run, ["mrr@5", "mrr@10"], make_comparable=True)
    return {
        "ndcg@5": float(ndcg["ndcg@5"]), "ndcg@10": float(ndcg["ndcg@10"]),
        "mrr@5": float(mrr["mrr@5"]), "mrr@10": float(mrr["mrr@10"]),
        "n_queries": len(run_dict),
    }


def recompute_from_saved(save_path: str, ks=(5, 10)) -> dict:
    """evaluate_rerank(save_path=...)가 남긴 원점수 파일을 다시 읽어, 재학습 없이
    임의의 cutoff로 nDCG/MRR을 다시 계산한다."""
    with open(save_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    run = Run(data["run"])
    qrels_graded = data["qrels_graded"]
    qrels_binary = {qid: {pid: (1 if rel >= 2 else 0) for pid, rel in g.items()}
                     for qid, g in qrels_graded.items()}
    ndcg_metrics = [f"ndcg@{k}" for k in ks]
    mrr_metrics = [f"mrr@{k}" for k in ks]
    ndcg = evaluate(Qrels(qrels_graded), run, ndcg_metrics, make_comparable=True)
    mrr = evaluate(Qrels(qrels_binary), run, mrr_metrics, make_comparable=True)
    result = {m: float(ndcg[m]) for m in ndcg_metrics}
    result.update({m: float(mrr[m]) for m in mrr_metrics})
    result["n_queries"] = len(data["run"])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--cap", type=int, default=CANDIDATE_CAP)
    ap.add_argument("--seed", type=int, default=CANDIDATE_SEED)
    ap.add_argument("--batch_size", type=int, default=4)  # 8B 모델은 16이면 OOM(실측) — 4로 통일
    ap.add_argument("--save_dir", default=None,
                     help="후보별 원점수를 저장할 디렉토리 (재학습 없이 다른 cutoff 재분석용)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    loaded = load_model_and_tokenizer(
        model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
    )
    model, tokenizer = loaded.model, loaded.tokenizer
    device = next(model.parameters()).device
    max_length = cfg["data"]["max_length"]

    for name, instruction in [("well-specified", WELL_SPECIFIED_INSTRUCTION),
                               ("under-specified", UNDER_SPECIFIED_INSTRUCTION)]:
        save_path = str(Path(args.save_dir) / f"zeroshot_{name}.json") if args.save_dir else None
        result = evaluate_rerank(model, tokenizer, instruction, max_length, device,
                                  cap=args.cap, seed=args.seed, batch_size=args.batch_size,
                                  save_path=save_path)
        print(f"=== Zero-Shot, {name} instruction ===")
        print(f"nDCG@5={result['ndcg@5']:.4f}  nDCG@10={result['ndcg@10']:.4f}  "
              f"MRR@5={result['mrr@5']:.4f}  MRR@10={result['mrr@10']:.4f}  "
              f"(n_queries={result['n_queries']})")


if __name__ == "__main__":
    main()
