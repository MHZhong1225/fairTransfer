"""Same native BF16 judge loaders with an explicit server-local model directory."""
import os
from pathlib import Path


def load(judge):
    import torch
    root=Path(os.environ['UMM_JUDGE_MODEL_ROOT'])
    if judge=='qwen':
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        path=root/'Qwen3-VL-8B-Instruct'
        processor=AutoProcessor.from_pretrained(path,local_files_only=True)
        model=Qwen3VLForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,
            device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
        processor.tokenizer.padding_side='left'
        return model,processor
    import ministral_judge
    ministral_judge.MODEL=root/'Ministral-3-8B-Instruct-2512-BF16'
    return ministral_judge.load()
