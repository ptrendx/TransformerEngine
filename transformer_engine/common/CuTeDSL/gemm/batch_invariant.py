# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compile the persistent Blackwell BF16 batch-invariant GEMM.

Output tiling adapts to M without changing the cluster or sequential K reduction.
A compiled function accepts any M with the same N and K, including sliced inputs.
"""

from functools import lru_cache

import cutlass
from cutlass import cute, utils
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
import cuda.bindings.driver as cuda

from ._blackwell_dense_gemm import DenseGemmKernel


@cute.jit
def _gemm_2d(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: cuda.CUstream,  # pylint: disable=c-extension-no-member
    max_active_clusters: cutlass.Constexpr,
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
    # All variants use the same K=64 tile and ordered tcgen05 MMA instructions.
    # Choose output geometry in the compiled host function, with no Python dispatch.
    if a.shape[0] <= 256:
        DenseGemmKernel(cutlass.Float32, True, (128, 128), (2, 1), True)(
            a3, b3, c3, max_active_clusters, stream
        )
    elif a.shape[0] <= 1024:
        DenseGemmKernel(cutlass.Float32, True, (128, 256), (2, 1), True)(
            a3, b3, c3, max_active_clusters, stream
        )
    else:
        DenseGemmKernel(cutlass.Float32, True, (256, 256), (2, 1), True)(
            a3, b3, c3, max_active_clusters, stream
        )


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
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(2)
    return cute.compile(
        _gemm_2d, a, b, c, make_fake_stream(), max_active_clusters, options="--enable-tvm-ffi"
    )


@lru_cache(maxsize=32)
def register_native_gemm(n: int, k: int, device_index: int, a_row_stride: int):
    """Register the compiled native entrypoint once for C++ dispatch."""
    import tvm_ffi

    compiled = compile_batch_invariant_gemm(n, k, device_index, a_row_stride)
    native = getattr(compiled, "__tvm_ffi_object__", lambda: None)()
    if native is None:
        if isinstance(compiled, tvm_ffi.Function):
            native = compiled
        else:
            raise RuntimeError(
                "CuTeDSL GEMM compilation did not expose a native TVM FFI entrypoint"
            )
    name = f"nvte.batch_invariant.{device_index}.{n}.{k}.{a_row_stride}"
    tvm_ffi.register_global_func(name, native, override=True)
    return name
