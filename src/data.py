"""
Instruction / task-input / response 세그먼트를 분리 토크나이즈하여
정확한 토큰 위치(span)를 추적하는 데이터 유틸.

논문 3.1.1/3.1.2의 전제조건: I_{b,t}(instruction 토큰 위치), X_{b,t}(task-input
토큰 위치)를 padding/response/제어토큰과 분리해서 알아야 gradient sensitivity를
계산할 수 있다. 전체 문자열을 한 번에 토크나이즈한 뒤 substring으로 위치를
역산하면 subword 경계에서 어긋날 수 있으므로, 세그먼트별로 따로 토크나이즈해서
이어붙이는 방식을 쓴다.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import torch


INSTR_PREFIX = "### Instruction:\n"
INSTR_SUFFIX = "\n\n"


def wrap_instruction(instruction: str) -> str:
    """instruction 세그먼트 조립 시 실제로 토크나이즈되는 문자열.
    src/rewriter.py가 gradient-highlight span의 문자 오프셋을 이 문자열의
    토큰 오프셋과 맞추기 위해 동일한 래핑을 재사용한다."""
    return f"{INSTR_PREFIX}{instruction}{INSTR_SUFFIX}"


def _encode_segment(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@dataclass
class Example:
    input_ids: list
    labels: list
    instr_span: tuple  # (start, end) exclusive
    input_span: tuple  # (start, end) exclusive
    label_str: str  # "Relevant"/"Irrelevant" — in-step confusion tracking용
    orig_label_str: str  # confusion matrix 카테고리 키 (ESCI E/S/C/I 등, 없으면 label_str과 동일)
    query: str = None  # 2026-08-06 추가 -- rewrite 프롬프트에 오분류 예시 원문을 보여주기 위해 보존
    product_text: str = None  # 위와 동일 목적(ESCI는 product, MSMARCO는 passage 텍스트)


def build_example(tokenizer, instruction: str, query: str, product_text: str,
                   label: str, max_length: int = 512, orig_label: str = None) -> Example:
    instr_text = wrap_instruction(instruction)
    input_text = f"### Input:\nQuery: {query}\n{product_text}\n\n### Response:\n"
    response_text = label

    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    instr_ids = _encode_segment(tokenizer, instr_text)
    input_ids_seg = _encode_segment(tokenizer, input_text)
    response_ids = _encode_segment(tokenizer, response_text)

    input_ids = bos + instr_ids + input_ids_seg + response_ids + eos

    instr_start = len(bos)
    instr_end = instr_start + len(instr_ids)
    input_start = instr_end
    input_end = input_start + len(input_ids_seg)
    response_start = input_end
    response_end = response_start + len(response_ids) + len(eos)

    if len(input_ids) > max_length:
        # product_text가 비정상적으로 길 때만 발생. input 세그먼트를 뒤에서 잘라
        # response/instruction 영역은 보존한다.
        overflow = len(input_ids) - max_length
        cut = max(0, len(input_ids_seg) - overflow)
        input_ids_seg = input_ids_seg[:cut]
        input_ids = bos + instr_ids + input_ids_seg + response_ids + eos
        input_end = input_start + len(input_ids_seg)
        response_start = input_end
        response_end = response_start + len(response_ids) + len(eos)

    labels = [-100] * len(input_ids)
    for i in range(response_start, response_end):
        labels[i] = input_ids[i]

    return Example(
        input_ids=input_ids,
        labels=labels,
        instr_span=(instr_start, instr_end),
        input_span=(input_start, input_end),
        label_str=label,
        orig_label_str=orig_label if orig_label is not None else label,
        query=query,
        product_text=product_text,
    )


class PairDataset(torch.utils.data.Dataset):
    """jsonl (query, product_text, label, instruction) pair 파일을 로드."""

    def __init__(self, tokenizer, jsonl_path, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        # 3.2: rewriting이 채택되면 이후 배치는 새 instruction을 써야 하므로,
        # jsonl에 저장된 instruction 대신 이 값을 우선 사용할 수 있게 한다.
        self.instruction_override = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        instruction = self.instruction_override or r["instruction"]
        return build_example(
            self.tokenizer, instruction, r["query"], r["product_text"],
            r["label"], self.max_length, orig_label=r.get("orig_label"),
        )


def collate(tokenizer, examples):
    max_len = max(len(e.input_ids) for e in examples)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    input_ids, labels, attn = [], [], []
    instr_spans, input_spans = [], []
    label_strs, orig_label_strs = [], []
    queries, product_texts = [], []
    for e in examples:
        n = len(e.input_ids)
        pad_n = max_len - n
        input_ids.append(e.input_ids + [pad_id] * pad_n)
        labels.append(e.labels + [-100] * pad_n)
        attn.append([1] * n + [0] * pad_n)
        instr_spans.append(e.instr_span)
        input_spans.append(e.input_span)
        label_strs.append(e.label_str)
        orig_label_strs.append(e.orig_label_str)
        queries.append(e.query)
        product_texts.append(e.product_text)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "instr_spans": instr_spans,  # list[(start,end)], padding 로직상 그대로 유효
        "input_spans": input_spans,
        "label_strs": label_strs,  # in-step confusion tracking용, 텐서화 안 함
        "orig_label_strs": orig_label_strs,
        "queries": queries,  # 2026-08-06 추가 -- 오분류 예시 원문(rewrite 프롬프트용), 텐서화 안 함
        "product_texts": product_texts,
    }


def make_collate_fn(tokenizer):
    def _fn(examples):
        return collate(tokenizer, examples)
    return _fn
