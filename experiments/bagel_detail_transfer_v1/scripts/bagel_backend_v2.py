"""Same BAGEL computation with repeatable FlashAttention backward accumulation."""
from contextlib import contextmanager
from bagel_backend import (DetailBagel as OriginalDetailBagel, GENERATION,
    PREPROCESSING, U_SITES, frozen_versions, check_frozen, tensor_hash)


@contextmanager
def deterministic_attention_backward():
    import modeling.bagel.qwen2_navit as qwen
    previous = qwen.flash_attn_varlen_func
    def attention(*args, **kwargs):
        kwargs['deterministic'] = True
        return previous(*args, **kwargs)
    qwen.flash_attn_varlen_func = attention
    try:
        yield
    finally:
        qwen.flash_attn_varlen_func = previous


class DetailBagel(OriginalDetailBagel):
    def caption_logits(self, row):
        with deterministic_attention_backward():
            return super().caption_logits(row)
