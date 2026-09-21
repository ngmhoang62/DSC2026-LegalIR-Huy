"""Fine-tune infgrad/Prism-Qwen3.5-Reranker-2B as a CANDIDATE for the `jina` channel.

Same generative yes/no cross-encoder recipe as finetune_qwen3_reranker.py --
LoRA on a frozen fp16 base, score = logit(yes) - logit(no) at the next-token
position, substituted into the LTR fusion under the `jina` channel key (see
that file's module docstring for why this can only ever stand in for `jina`,
never `dense`/aiteamvn). This file only differs in HF_REPO, the default model
folder, and the system prompt Prism's own model card documents (it omits the
'answer can only be "yes" or "no"' sentence Qwen3-Reranker's prompt has, and
warns inputs longer than its ~10K-token training cap may degrade -- irrelevant
here since --max-length defaults to 1024, well under that).

run/train_epoch/evaluate/sanity_check/lora_state_dict are imported rather
than duplicated -- everything about the training loop, LTR substitution, and
checkpoint bookkeeping is identical (and a Colab notebook driving this
cell-by-cell calls train_epoch/evaluate/sanity_check/lora_state_dict directly
on this module, so they need to be real attributes here, not just used
internally by run()); only load_model() (the HF repo id) and the prompt text
change in this file.

Usage
    python finetune_prism_reranker.py --epochs 3
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch_common as tc
from finetune_qwen3_reranker import (evaluate, lora_state_dict, run,   # noqa: F401
                                     sanity_check, train_epoch)

TAG = "prism_reranker"
DEFAULT_MODEL = "models/prism-qwen3.5-reranker-2b"
HF_REPO = "infgrad/Prism-Qwen3.5-Reranker-2B"
SYSTEM_PROMPT = ('Judge whether the Document meets the requirements based on the '
                 'Query and the Instruct provided.')


def load_model(root, model_path, gradient_checkpointing, lora_r, lora_alpha,
               lora_dropout, lora_target_modules):
    source = tc.resolve_model_source(root, model_path, HF_REPO)
    tokenizer = AutoTokenizer.from_pretrained(source, padding_side="left")
    base_model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float16)

    from peft import LoraConfig, TaskType, get_peft_model

    try:
        import peft.tuners.lora.torchao as _lora_torchao
        _lora_torchao.is_torchao_available = lambda: False
    except ImportError:
        pass

    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=lora_target_modules, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    trainable = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
            trainable += param.numel()
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable/1e6:.1f}M / {total/1e6:.0f}M trainable "
          f"({trainable/max(total,1):.2%})", flush=True)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print("Gradient checkpointing enabled", flush=True)

    model._tokenizer = tokenizer
    return model, tokenizer


def main():
    run(TAG, DEFAULT_MODEL, HF_REPO, SYSTEM_PROMPT, load_model)


if __name__ == "__main__":
    main()
