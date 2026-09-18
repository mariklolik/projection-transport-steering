from types import FunctionType

import torch
from torch.nn.attention.flex_attention import create_block_mask
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.flex_attention import flex_attention_forward
from transformers.masking_utils import flex_attention_mask


def _eager_create_block_mask(**kwargs):
    if kwargs.get("_compile") is not True:
        raise ValueError("unexpected native compile request")
    kwargs["_compile"] = False
    return create_block_mask(**kwargs)


if flex_attention_mask.__globals__["create_block_mask"] is not create_block_mask:
    raise ValueError("unexpected native mask builder binding")

eager_flex_attention_mask = FunctionType(
    flex_attention_mask.__code__,
    {**flex_attention_mask.__globals__, "create_block_mask": _eager_create_block_mask},
    "eager_flex_attention_mask",
    flex_attention_mask.__defaults__,
    flex_attention_mask.__closure__,
)
eager_flex_attention_mask.__kwdefaults__ = flex_attention_mask.__kwdefaults__
eager_flex_attention_mask.__annotations__ = flex_attention_mask.__annotations__


def fp32_prefill_flex(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    dtype = query.dtype
    if query.shape[-2] > 1:
        query, key, value = query.float(), key.float(), value.float()
    output, lse = flex_attention_forward(module, query, key, value, attention_mask, **kwargs)
    return output.to(dtype), lse.to(dtype) if lse is not None else None


def fp32_prefill_flex_v4(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if 1 < query.shape[-2] < 128:
        kwargs["kernel_options"] = {
            **(kwargs.get("kernel_options") or {}),
            "FORCE_USE_FLEX_ATTENTION": True,
        }
    return fp32_prefill_flex(module, query, key, value, attention_mask, **kwargs)


def register_fp32_prefill_flex() -> None:
    torch.set_float32_matmul_precision("highest")
    AttentionInterface.register("fp32_prefill_flex", fp32_prefill_flex)
    AttentionMaskInterface.register("fp32_prefill_flex", flex_attention_mask)
    AttentionInterface.register("fp32_prefill_flex_v4", fp32_prefill_flex_v4)
    AttentionMaskInterface.register("fp32_prefill_flex_v4", flex_attention_mask)
    AttentionInterface.register("fp32_prefill_flex_eager", fp32_prefill_flex_v4)
    AttentionMaskInterface.register("fp32_prefill_flex_eager", eager_flex_attention_mask)
