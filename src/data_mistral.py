"""
Mistral variant of src/data.py's build_example: uses the model's official chat template
([INST]...[/INST]) instead of the generic "### Instruction:\n...\n\n### Response:\n" format
used for Qwen, since an Instruct-tuned model evaluated off its trained template risks a
degenerate Pre-SFT baseline that would confound "model family" with "wrong prompt format".
Instruction text, query/passage content, and output labels (Relevant/Irrelevant) are otherwise
byte-identical to the Qwen pipeline -- only the wrapping syntax differs.

instr_span/input_span are not meaningfully separable once instruction+query+passage share one
chat turn (unlike the Qwen segment-tracking format) and are not used by any of the actual
training/eval scripts here (only input_ids/labels/attention_mask are read) -- set to placeholder
spans, not used downstream.
"""
from src.data import Example, collate, make_collate_fn  # noqa: F401 (re-exported)
import torch


def build_example_mistral(tokenizer, instruction: str, query: str, product_text: str,
                           label: str, max_length: int = 384, orig_label: str = None,
                           enable_thinking: bool = None) -> Example:
    """enable_thinking: passed through to apply_chat_template only when not None (default),
    so existing call sites that don't pass it keep the exact prior rendered output --
    Gemma's template has no `enable_thinking` variable and ignores the extra kwarg either
    way. Qwen call sites pass enable_thinking=False explicitly."""
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    response_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
    budget = max_length - len(response_ids) - len(eos)

    def make_prefix(ptext):
        user_content = f"{instruction}\n\nQuery: {query}\n{ptext}"
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        prefix_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}], **template_kwargs)
        return tokenizer(prefix_text, add_special_tokens=False)["input_ids"]

    text = product_text
    prefix_ids = make_prefix(text)
    while len(prefix_ids) > budget and len(text) > 20:
        text = text[: int(len(text) * 0.8)]
        prefix_ids = make_prefix(text)

    input_ids = prefix_ids + response_ids + eos
    response_start = len(prefix_ids)
    response_end = response_start + len(response_ids) + len(eos)
    labels = [-100] * len(input_ids)
    for i in range(response_start, response_end):
        labels[i] = input_ids[i]

    return Example(
        input_ids=input_ids, labels=labels,
        instr_span=(0, 0), input_span=(0, len(prefix_ids)),
        label_str=label, orig_label_str=orig_label if orig_label is not None else label,
        query=query, product_text=product_text,
    )


class PairDatasetMistral(torch.utils.data.Dataset):
    """Same jsonl schema as src.data.PairDataset, built via build_example_mistral."""

    def __init__(self, tokenizer, jsonl_path, max_length: int = 384):
        import json
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.instruction_override = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        instruction = self.instruction_override or r["instruction"]
        return build_example_mistral(
            self.tokenizer, instruction, r["query"], r["product_text"],
            r["label"], self.max_length, orig_label=r.get("orig_label"),
        )
