"""
Qwen3-4B + LoRA 로딩과, 입력 임베딩 활성값(activation)의 그래디언트를 가로채는
훅. 논문 3.1.1: LoRA 파라미터만 학습되고 임베딩 행렬(weight) 자체는 고정되지만,
"임베딩 활성값(activation)"에 대한 그래디언트는 진단 신호로 그대로 사용한다.

주의: peft로 LoRA를 적용하면 embed_tokens.weight.requires_grad=False가 되므로
nn.Embedding의 forward 출력 텐서도 기본적으로는 requires_grad=False(즉 grad_fn이
없는 leaf 텐서)이다. 이 훅은 forward 직후 그 출력 텐서에 대해
`requires_grad_(True)`를 걸어 leaf-with-grad로 승격시킨다. 이렇게 하면 이후
연산(attention 등)이 이 텐서를 시작점으로 그래프를 쌓게 되고, backward() 후
PyTorch가 leaf 텐서의 .grad를 자동으로 채워준다 — 추가 backward pass 없이
표준 SFT backward 한 번으로 활성값 그래디언트를 얻는다는 논문의 설명과 일치한다.
"""
from dataclasses import dataclass

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


class EmbeddingGradCapture:
    """input embedding의 forward 출력을 grad 추적 가능하게 만들고,
    backward 이후 그 activation과 .grad를 보관한다."""

    def __init__(self, model):
        embed = model.get_input_embeddings()
        self._activation = None
        self.handle = embed.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        output.requires_grad_(True)
        output.retain_grad()
        self._activation = output
        return output

    @property
    def grad(self):
        if self._activation is None:
            return None
        return self._activation.grad

    def clear(self):
        self._activation = None

    def remove(self):
        self.handle.remove()


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    embed_grad: EmbeddingGradCapture


def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen3-4B",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules=None,
    dtype=torch.bfloat16,
    device: str = "cuda",
    gradient_checkpointing: bool = False,
    local_files_only: bool = False,
    model_kwargs: dict = None,
) -> LoadedModel:
    target_modules = target_modules or DEFAULT_TARGET_MODULES

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype,
                                                        local_files_only=local_files_only, **(model_kwargs or {}))
    base_model.to(device)
    print(f"[load_model_and_tokenizer] model_name={model_name} local_files_only={local_files_only} "
          f"resolved_commit_hash={getattr(base_model.config, '_commit_hash', None)} "
          f"model_class={type(base_model).__name__}")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    lora_module_names = sorted({name.rsplit(".", 1)[0] for name, p in model.named_parameters()
                                 if p.requires_grad})
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[load_model_and_tokenizer] n_lora_modules={len(lora_module_names)} "
          f"n_trainable_params={n_trainable}")
    print(f"[load_model_and_tokenizer] lora_module_names={lora_module_names}")

    if gradient_checkpointing:
        # 8B급 이상 모델은 batch=2/seq=384만으로도 24GB 카드 한계에 거의 닿는다
        # (실측: reserved 24.64GB) — activation을 저장 안 하고 backward 때
        # 재계산해서 ~6GB 여유를 만든다(실측: 18.66GB). embed_grad 훅은 임베딩
        # 레이어 자체(체크포인팅 대상이 아닌 첫 레이어)의 forward hook이라
        # 이 옵션과 무관하게 그대로 동작한다 — 별도로 enable_input_require_grads()를
        # 부를 필요 없음(이미 embed_grad._hook이 동일한 역할을 함).
        model.gradient_checkpointing_enable()

    embed_grad = EmbeddingGradCapture(model)

    return LoadedModel(model=model, tokenizer=tokenizer, embed_grad=embed_grad)
