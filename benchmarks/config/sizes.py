# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Standard size configurations for benchmarks.

Two regimes:
- CPU-bound (small): Expose host overhead, kernel launch latency, recipe management.
- GPU-bound (large): Saturate GPU compute and memory bandwidth.

Each entry is a tuple: (shape_tuple, regime_label).
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class SizeConfig:
    """A benchmark size configuration with a regime label."""
    shape: tuple
    regime: str  # "cpu_bound" or "gpu_bound"

    @property
    def id(self) -> str:
        """Human-readable ID for pytest parametrize."""
        shape_str = "x".join(str(s) for s in self.shape)
        return f"{self.regime}_{shape_str}"


def _make_sizes(shapes_small, shapes_large) -> List[SizeConfig]:
    sizes = []
    for s in shapes_small:
        sizes.append(SizeConfig(shape=s, regime="cpu_bound"))
    for s in shapes_large:
        sizes.append(SizeConfig(shape=s, regime="gpu_bound"))
    return sizes


# ---------------------------------------------------------------------------
# Activation: (rows, cols)
# ---------------------------------------------------------------------------
ACTIVATION_SIZES = _make_sizes(
    shapes_small=[
        (1, 128),
        (4, 256),
        (16, 64),
        (32, 128),
    ],
    shapes_large=[
        (2048, 4096),
        (8192, 4096),
        (8192, 12288),
        (16384, 12288),
    ],
)

# Gated activations need even col count (split in half): (rows, cols)
# cols must be even since the input is split into two halves
GATED_ACTIVATION_SIZES = _make_sizes(
    shapes_small=[
        (1, 256),
        (4, 512),
        (16, 128),
        (32, 256),
    ],
    shapes_large=[
        (2048, 8192),
        (8192, 8192),
        (8192, 24576),
        (16384, 24576),
    ],
)

# ---------------------------------------------------------------------------
# Normalization: (rows, hidden_size)
# ---------------------------------------------------------------------------
NORMALIZATION_SIZES = _make_sizes(
    shapes_small=[
        (1, 64),
        (4, 128),
        (16, 256),
        (32, 64),
    ],
    shapes_large=[
        (2048, 4096),
        (8192, 4096),
        (8192, 12288),
        (16384, 12288),
    ],
)

# ---------------------------------------------------------------------------
# Cast / Quantize: (rows, cols)
# ---------------------------------------------------------------------------
CAST_SIZES = _make_sizes(
    shapes_small=[
        (1, 128),
        (4, 256),
        (16, 64),
        (32, 128),
    ],
    shapes_large=[
        (2048, 4096),
        (8192, 4096),
        (8192, 12288),
        (16384, 12288),
    ],
)

# ---------------------------------------------------------------------------
# Transpose: (rows, cols)
# ---------------------------------------------------------------------------
TRANSPOSE_SIZES = _make_sizes(
    shapes_small=[
        (16, 16),
        (32, 64),
        (64, 128),
    ],
    shapes_large=[
        (2048, 4096),
        (4096, 4096),
        (8192, 12288),
    ],
)

# ---------------------------------------------------------------------------
# GEMM: (M, K, N)
# ---------------------------------------------------------------------------
GEMM_SIZES = _make_sizes(
    shapes_small=[
        (1, 64, 64),
        (4, 128, 128),
        (16, 64, 64),
        (32, 128, 256),
    ],
    shapes_large=[
        (2048, 4096, 4096),
        (4096, 4096, 4096),
        (8192, 4096, 16384),
        (16384, 4096, 4096),
    ],
)

# ---------------------------------------------------------------------------
# Linear (PyTorch/JAX): (batch_seq, in_features, out_features)
# Same format as GEMM
# ---------------------------------------------------------------------------
LINEAR_SIZES = GEMM_SIZES

# ---------------------------------------------------------------------------
# Grouped Linear: (batch_seq, in_features, out_features, num_gemms)
# ---------------------------------------------------------------------------
GROUPED_LINEAR_SIZES = _make_sizes(
    shapes_small=[
        (16, 128, 128, 4),
        (32, 128, 256, 8),
    ],
    shapes_large=[
        (4096, 4096, 4096, 4),
        (4096, 4096, 4096, 8),
        (8192, 4096, 16384, 4),
        (8192, 4096, 16384, 16),
    ],
)

# ---------------------------------------------------------------------------
# Attention: (batch, num_heads, num_gqa_groups, head_dim, seq_q, seq_kv)
# ---------------------------------------------------------------------------
ATTENTION_SIZES = _make_sizes(
    shapes_small=[
        (1, 4, 4, 64, 32, 32),
        (1, 8, 8, 64, 64, 64),
        (2, 4, 4, 64, 128, 128),
    ],
    shapes_large=[
        (2, 16, 16, 64, 512, 512),
        (2, 16, 16, 128, 2048, 2048),
        (2, 32, 4, 128, 2048, 2048),
        (2, 32, 4, 128, 8192, 8192),
    ],
)

# ---------------------------------------------------------------------------
# Softmax: (batch, heads, seq_q, seq_kv)
# ---------------------------------------------------------------------------
SOFTMAX_SIZES = _make_sizes(
    shapes_small=[
        (1, 4, 32, 32),
        (2, 8, 64, 64),
    ],
    shapes_large=[
        (2, 16, 512, 512),
        (2, 32, 2048, 2048),
    ],
)

# ---------------------------------------------------------------------------
# RoPE: (seq_len, batch, num_heads, head_dim)
# ---------------------------------------------------------------------------
ROPE_SIZES = _make_sizes(
    shapes_small=[
        (32, 1, 4, 64),
        (64, 2, 8, 64),
    ],
    shapes_large=[
        (2048, 2, 32, 128),
        (8192, 2, 32, 128),
    ],
)

# ---------------------------------------------------------------------------
# TransformerLayer: (batch, seq_len, hidden_size, num_heads, ffn_hidden_size)
# ---------------------------------------------------------------------------
TRANSFORMER_LAYER_SIZES = _make_sizes(
    shapes_small=[
        (1, 32, 256, 4, 1024),
        (2, 64, 512, 8, 2048),
    ],
    shapes_large=[
        (2, 512, 4096, 32, 16384),
        (2, 2048, 4096, 32, 16384),
        (1, 2048, 12288, 96, 49152),
    ],
)

# ---------------------------------------------------------------------------
# Multi-tensor ops: list of (num_tensors, tensor_size)
# ---------------------------------------------------------------------------
MULTI_TENSOR_SIZES = _make_sizes(
    shapes_small=[
        (4, 1024),
        (8, 4096),
    ],
    shapes_large=[
        (32, 1048576),
        (64, 4194304),
        (128, 1048576),
    ],
)

# ---------------------------------------------------------------------------
# Hadamard: (rows, cols) -- cols must be power of 2
# ---------------------------------------------------------------------------
HADAMARD_SIZES = _make_sizes(
    shapes_small=[
        (16, 64),
        (32, 128),
        (64, 256),
    ],
    shapes_large=[
        (2048, 4096),
        (8192, 4096),
        (16384, 4096),
    ],
)

# ---------------------------------------------------------------------------
# Swizzle: (rows, cols)
# ---------------------------------------------------------------------------
SWIZZLE_SIZES = _make_sizes(
    shapes_small=[
        (16, 128),
        (32, 256),
    ],
    shapes_large=[
        (2048, 4096),
        (8192, 4096),
    ],
)

# ---------------------------------------------------------------------------
# Router / MoE: (num_tokens, num_experts, topk)
# ---------------------------------------------------------------------------
ROUTER_SIZES = _make_sizes(
    shapes_small=[
        (32, 8, 2),
        (64, 16, 2),
    ],
    shapes_large=[
        (4096, 64, 2),
        (8192, 128, 2),
        (16384, 64, 4),
    ],
)

# ---------------------------------------------------------------------------
# Optimizer parameter groups: (num_params, param_size)
# ---------------------------------------------------------------------------
OPTIMIZER_SIZES = _make_sizes(
    shapes_small=[
        (4, 4096),
        (8, 16384),
    ],
    shapes_large=[
        (32, 1048576),
        (64, 4194304),
    ],
)


def filter_sizes(sizes: List[SizeConfig], regime: str = "all") -> List[SizeConfig]:
    """Filter sizes by regime."""
    if regime == "all":
        return sizes
    return [s for s in sizes if s.regime == regime]
