# SPDX-License-Identifier: Apache-2.0
"""The sender -> receiver direct KV-cache transfer.

A *transfer* turns sharer prompt token ids into a receiver-geometry `DynamicCache` the
receiver can decode from. The direct transfer copies the sharer's cache straight into
the receiver at identity layer alignment -- valid only when the two models share KV
geometry and depth (`core.same_kv_shape`).
"""

# Standard
import logging
from collections.abc import Callable

# Third Party
import torch
from transformers import (
    DynamicCache,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# First Party
from cachefuse import core
from cachefuse.config import Settings

logger = logging.getLogger(__name__)

# A transfer: sender token ids (on the shared device) -> receiver-geometry cache.
CacheTransfer = Callable[[torch.Tensor], DynamicCache]


@torch.no_grad()
def build_transfer(
    cfg: Settings,
    sender: PreTrainedModel,
    receiver: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
) -> CacheTransfer:
    """Build the direct transfer: prefill the sharer, move its cache to the receiver.

    `tok` is unused here; it is kept for a uniform transfer-builder signature.

    Raises:
        ValueError: if the sender and receiver KV geometry or depth differ.
    """
    if not core.same_kv_shape(sender.config, receiver.config):
        raise ValueError(
            "the direct transfer requires identical KV geometry and depth; this pair "
            "differs"
        )
    dev = cfg.device
    logger.info("direct transfer (cache copy)")

    def direct_transfer(sids: torch.Tensor) -> DynamicCache:
        return core.move_cache(core.prefill(sender, sids), dev)

    return direct_transfer
