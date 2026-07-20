# SPDX-License-Identifier: Apache-2.0
"""CacheFuse -- training-free cross-model KV-cache reuse (direct transfer).

A *sender* LLM prefills a context and a *receiver* LLM reuses that KV cache directly
instead of recomputing its own prefill. Run the evaluation with
``python -m cachefuse.evaluate``.

Hugging Face caches follow the usual env vars (`HF_HOME`, `HF_HUB_CACHE`,
`HF_DATASETS_CACHE`); unset, libraries use ``~/.cache/huggingface``.
"""

# First Party
from cachefuse.config import BENCHMARKS, Settings
from cachefuse.core import (
    head_dim,
    load_models,
    make_cache,
    move_cache,
    num_kv_heads,
    prefill,
    same_kv_shape,
)
from cachefuse.transfers import CacheTransfer, build_transfer

__all__ = [
    "BENCHMARKS",
    "CacheTransfer",
    "Settings",
    "build_transfer",
    "head_dim",
    "load_models",
    "make_cache",
    "move_cache",
    "num_kv_heads",
    "prefill",
    "same_kv_shape",
]
