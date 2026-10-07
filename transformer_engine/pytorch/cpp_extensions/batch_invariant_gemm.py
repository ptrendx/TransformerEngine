# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Batch-invariant BF16 GEMM forward path.

The general :func:`general_gemm` path delegates kernel selection to cuBLASLt, whose
heuristic depends on the full problem shape. That makes a given input row's output
depend on how many *other* rows are present in the batch. This module provides an
opt-in forward path with a reduction order that only depends on the tile
coordinates, so a row's result is bitwise stable across batch composition.
On Blackwell, large aligned problems use a fixed-tile CuTeDSL tensor-core
kernel; the Triton kernel handles other configurations.

Supported: BF16, ``Y = X @ W.T``, contiguous 2-D operands, no bias.
Unsupported combinations raise instead of silently falling back.
"""

from functools import lru_cache
from importlib.util import find_spec
from typing import Literal, Optional

import torch
import triton
import triton.language as tl

__all__ = ["batch_invariant_gemm", "is_supported"]

# Tile geometry. These are fixed on purpose: making them depend on M would
# reintroduce the batch dependence this path exists to remove.
TRITON_BLOCK_M = 64
TRITON_BLOCK_N = 64
TRITON_BLOCK_K = 64


@triton.jit
def _bi_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C[m, n] = sum_k A[m, k] * B[n, k], accumulated in a single BLOCK_K loop."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Fixed-trip sequential reduction: the order is a function of K and BLOCK_K
    # only, so it cannot vary with M or with the tile's position in the batch.
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_rem)
        b_mask = (offs_k[:, None] < k_rem) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


def is_supported(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> bool:
    """Whether ``batch_invariant_gemm`` can run this configuration."""
    try:
        _check(a, b, out)
    except ValueError:
        return False
    return True


def _check(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor]) -> None:
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError(
            f"batch_invariant_gemm supports bfloat16 inputs only, got {a.dtype} and {b.dtype}."
        )
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(
            f"batch_invariant_gemm requires 2-D operands, got {a.dim()}-D and {b.dim()}-D."
        )
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("batch_invariant_gemm requires contiguous operands.")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"K mismatch: A has {a.shape[1]} columns, B has {b.shape[1]}.")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise ValueError("batch_invariant_gemm requires operands on the same CUDA device.")
    if out is not None and (out.dim() != 2 or not out.is_contiguous()):
        raise ValueError("batch_invariant_gemm requires a contiguous 2-D out tensor.")
    if out is not None and (out.dtype != torch.bfloat16 or out.device != a.device):
        raise ValueError("batch_invariant_gemm requires a BF16 out tensor on the input device.")


def _can_use_cutedsl(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Blackwell TMA needs 16-byte aligned rows; keep dispatch independent of M."""
    n, k = b.shape
    return (
        a.device.type == "cuda"
        and torch.cuda.get_device_capability(a.device)[0] == 10
        and n >= 8
        and k >= 8
        and n % 8 == 0
        and k % 8 == 0
        and a.data_ptr() % 16 == 0
        and b.data_ptr() % 16 == 0
    )


@lru_cache(maxsize=1)
def _cutedsl_dependencies_available() -> bool:
    """Check optional CuTeDSL dependencies once for automatic dispatch."""
    return all(find_spec(name) is not None for name in ("cutlass", "cuda", "tvm_ffi"))


@lru_cache(maxsize=32)
def _get_cutedsl_gemm(device_index: int, n: int, k: int, a_row_stride: int):
    """Compile for the requested GPU once and cache the direct TVM-FFI callable."""
    try:
        from cuda.bindings.driver import CUstream
        from transformer_engine.common.CuTeDSL.gemm.batch_invariant import (
            compile_batch_invariant_gemm,
        )
    except ImportError as exc:
        raise RuntimeError("CuTeDSL batch-invariant GEMM requires cutlass and cuda-python") from exc
    with torch.cuda.device(device_index):
        return compile_batch_invariant_gemm(n, k, device_index, a_row_stride), CUstream


def _batch_invariant_gemm_cutedsl(
    a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor]
) -> torch.Tensor:
    """Run the same compiled CuTeDSL kernel for every batch size."""
    # The Blackwell TMA epilogue faults for a one-row output. Give it two
    # logical rows while aliasing the input, then return only the first result.
    # No device copy is needed for the input and the reduction order is unchanged.
    single_row = a.shape[0] == 1
    if single_row:
        a = a.expand(2, -1)
        padded_out = torch.empty((2, b.shape[0]), dtype=a.dtype, device=a.device)
    elif out is not None and out.data_ptr() % 16 == 0:
        padded_out = out
    else:
        padded_out = torch.empty((a.shape[0], b.shape[0]), dtype=a.dtype, device=a.device)

    compiled, stream_type = _get_cutedsl_gemm(a.device.index, b.shape[0], b.shape[1], a.stride(0))
    stream = stream_type(torch.cuda.current_stream(device=a.device).cuda_stream)
    compiled(a, b, padded_out, stream)

    if out is None:
        return padded_out[:1] if single_row else padded_out
    if padded_out is not out:
        out.copy_(padded_out[:1] if single_row else padded_out)
    return out


def batch_invariant_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    backend: Literal["auto", "triton", "cutedsl"] = "auto",
) -> torch.Tensor:
    """Compute ``Y = A @ B.T`` with a batch-composition-independent reduction order.

    Parameters
    ----------
    a : torch.Tensor
        BF16 activation, shape ``[M, K]``, contiguous. Row ``m`` of ``a`` always
        produces the same output bits for a given ``b``, independent of ``M`` and of
        the row's position.
    b : torch.Tensor
        BF16 weight, shape ``[N, K]``, contiguous.
    out : torch.Tensor, optional
        BF16 destination of shape ``[M, N]``. A fresh tensor is allocated when absent.
    backend : {"auto", "triton", "cutedsl"}
        ``"auto"`` uses CuTeDSL on Blackwell for aligned N and K of at least
        512, and Triton otherwise. Selection is independent of M.

    Returns
    -------
    torch.Tensor
        The ``[M, N]`` result. Same layout contract as :func:`general_gemm`'s output.
    """
    _check(a, b, out)
    if backend not in ("auto", "triton", "cutedsl"):
        raise ValueError(f"Unknown batch_invariant_gemm backend: {backend}.")
    m, k = a.shape
    n = b.shape[0]
    use_cutedsl = backend == "cutedsl" or (
        backend == "auto"
        and n >= 512
        and k >= 512
        and _can_use_cutedsl(a, b)
        and _cutedsl_dependencies_available()
    )
    if backend == "cutedsl" and not _can_use_cutedsl(a, b):
        raise ValueError(
            "CuTeDSL requires Blackwell, aligned input pointers, and N,K divisible by 8"
            " and at least 8."
        )
    if out is not None and out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}.")
    if m == 0 or n == 0:
        return out if out is not None else torch.empty((m, n), dtype=a.dtype, device=a.device)

    if use_cutedsl:
        return _batch_invariant_gemm_cutedsl(a, b, out)

    if out is None:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)

    grid = (triton.cdiv(m, TRITON_BLOCK_M), triton.cdiv(n, TRITON_BLOCK_N))
    _bi_gemm_kernel[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=TRITON_BLOCK_M,
        BLOCK_N=TRITON_BLOCK_N,
        BLOCK_K=TRITON_BLOCK_K,
    )
    return out
