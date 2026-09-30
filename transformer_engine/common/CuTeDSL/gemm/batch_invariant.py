# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compile the fixed-tile Blackwell BF16 batch-invariant GEMM.

The MMA tile, cluster, and K loop do not depend on M. A compiled function accepts
any M with the same N and K, including batches formed by slicing the input.
"""

from functools import lru_cache

import cutlass
from cutlass import cute
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
import cuda.bindings.driver as cuda

from ._blackwell_dense_gemm import DenseGemmKernel


@cute.jit
def _gemm_2d(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: cuda.CUstream,  # pylint: disable=c-extension-no-member
):
    """Give the dense GEMM's tensors a unit batch mode without PyTorch views."""
    a3 = cute.make_tensor(
        a.iterator,
        cute.make_layout((a.shape[0], a.shape[1], 1), stride=(a.layout.stride[0], 1, 1)),
    )
    b3 = cute.make_tensor(
        b.iterator,
        cute.make_layout((b.shape[0], b.shape[1], 1), stride=(b.layout.stride[0], 1, 1)),
    )
    c3 = cute.make_tensor(
        c.iterator,
        cute.make_layout((c.shape[0], c.shape[1], 1), stride=(c.layout.stride[0], 1, 1)),
    )
    DenseGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=True,
        mma_tiler_mn=(128, 256),
        cluster_shape_mn=(2, 1),
        use_tma_store=True,
    )(a3, b3, c3, stream)


@lru_cache(maxsize=32)
def compile_batch_invariant_gemm(n: int, k: int, device_index: int, a_row_stride: int):
    """Return a CuTeDSL/TVM-FFI callable for contiguous BF16 ``[M,K] @ [N,K].T``.

    The device index is part of the cache key because CuTe compiles for the active
    device architecture. The caller must make that device current before calling.
    """
    del device_index
    m = cute.sym_int32()
    a = make_fake_tensor(cutlass.BFloat16, (m, k), stride=(a_row_stride, 1), assumed_align=16)
    b = make_fake_tensor(cutlass.BFloat16, (n, k), stride=(k, 1), assumed_align=16)
    c = make_fake_tensor(cutlass.BFloat16, (m, n), stride=(n, 1), assumed_align=16)
    return cute.compile(_gemm_2d, a, b, c, make_fake_stream(), options="--enable-tvm-ffi")
