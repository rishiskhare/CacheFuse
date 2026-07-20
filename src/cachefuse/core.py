# SPDX-License-Identifier: Apache-2.0
"""Model loading, KV geometry, and the `DynamicCache` utilities (prefill, move, build).

`same_kv_shape` gates the direct transfer: the sender and receiver must share KV
geometry and depth for the sharer's per-layer cache to slot into the receiver.
"""

# Standard
import logging
import os

# Third Party
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

# User token for gated models/datasets (optional).
HF_TOKEN = os.environ.get("HF_TOKEN") or None


def head_dim(config: PretrainedConfig) -> int:
    """Per-head dimension of `config`, falling back to ``hidden_size // n_heads``.

    Some architectures (Llama, Mistral, Qwen2) do not set an explicit ``head_dim``.
    """
    return getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)


def num_kv_heads(config: PretrainedConfig) -> int:
    """Number of key/value heads of `config` (equals attention heads if not GQA)."""
    return getattr(config, "num_key_value_heads", config.num_attention_heads)


def same_kv_shape(sender: PretrainedConfig, receiver: PretrainedConfig) -> bool:
    """Whether two configs share KV-cache shape (kv-heads, head-dim, depth).

    When ``True`` the sharer's per-layer cache slots directly into the receiver at
    identity layer alignment, so a direct transfer is well-defined.
    """
    return (
        num_kv_heads(sender) == num_kv_heads(receiver)
        and head_dim(sender) == head_dim(receiver)
        and sender.num_hidden_layers == receiver.num_hidden_layers
    )


def load_models(
    sender: str, receiver: str, device: str, sender_device: str | None = None
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel, PreTrainedModel]:
    """Load the receiver's tokenizer and both models in bf16, eval mode.

    The receiver loads on `device`; the sender on `sender_device` if given, else on
    `device` too (the single-GPU default). When the two differ, the sender's prefilled
    cache is later moved to the receiver's `device` for fusion, so a pair can span two
    GPUs. A KV geometry/depth mismatch is logged as a warning here (the transfer build
    then rejects it). Returns ``(tokenizer, sender_model, receiver_model)``.
    """
    sdev = sender_device or device
    trust = os.environ.get("TRUST_REMOTE_CODE") == "1"  # for custom archs
    tok = AutoTokenizer.from_pretrained(
        receiver, token=HF_TOKEN, trust_remote_code=trust
    )
    sender_model = (
        AutoModelForCausalLM.from_pretrained(
            sender, dtype=torch.bfloat16, token=HF_TOKEN, trust_remote_code=trust
        )
        .to(sdev)
        .eval()
    )
    receiver_model = (
        AutoModelForCausalLM.from_pretrained(
            receiver, dtype=torch.bfloat16, token=HF_TOKEN, trust_remote_code=trust
        )
        .to(device)
        .eval()
    )
    sc, rc = sender_model.config, receiver_model.config
    if not same_kv_shape(sc, rc):
        logger.warning(
            "sender/receiver KV shape differs: kv_heads=%d/%d, head_dim=%d/%d, "
            "layers=%d/%d -- the direct transfer needs an exact match and rejects "
            "this pair",
            num_kv_heads(sc),
            num_kv_heads(rc),
            head_dim(sc),
            head_dim(rc),
            sc.num_hidden_layers,
            rc.num_hidden_layers,
        )
    logger.info(
        "sender@%s kv_heads=%d head_dim=%d layers=%d; "
        "receiver@%s kv_heads=%d head_dim=%d layers=%d",
        sdev,
        num_kv_heads(sc),
        head_dim(sc),
        sc.num_hidden_layers,
        device,
        num_kv_heads(rc),
        head_dim(rc),
        rc.num_hidden_layers,
    )
    return tok, sender_model, receiver_model


@torch.no_grad()
def move_cache(cache: DynamicCache, device: str) -> DynamicCache:
    """Move every layer's keys/values of `cache` to `device` in place; return it."""
    for layer in cache.layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
        layer.device = torch.device(device)
    return cache


def make_cache(
    keys: list[torch.Tensor], values: list[torch.Tensor], dev: str
) -> DynamicCache:
    """Build a bf16 `DynamicCache` on `dev` from per-layer key/value tensors."""
    cache = DynamicCache()
    for li in range(len(keys)):
        cache.update(
            keys[li].to(dev, torch.bfloat16), values[li].to(dev, torch.bfloat16), li
        )
    return cache


@torch.no_grad()
def prefill(model: PreTrainedModel, input_ids: torch.Tensor) -> DynamicCache:
    """Prefill `model` on `input_ids` `[1, S]` and return its `DynamicCache`.

    `logits_to_keep=1` avoids materializing full-sequence logits.
    """
    return model(
        input_ids=input_ids,
        use_cache=True,
        past_key_values=DynamicCache(),
        logits_to_keep=1,
    ).past_key_values
