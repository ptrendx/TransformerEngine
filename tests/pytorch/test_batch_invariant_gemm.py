# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""A row's output must not depend on which other rows were in the batch.

The general GEMM path lets cuBLASLt pick a kernel from the full problem shape, so
the same row can produce different bits depending on the batch it arrives in.
``batch_invariant_gemm`` fixes the tile geometry and the reduction order so the
result is a function of the row and the weight only.
"""

import pytest
import torch

from transformer_engine.pytorch.cpp_extensions.batch_invariant_gemm import (
    batch_invariant_gemm,
    is_supported,
)

M, N, K = 256, 192, 128
SLICES = [(0, 1), (0, 7), (5, 29), (128, 256), (0, 256)]


@pytest.fixture(params=["triton", "cutedsl"])
def backend(request):
    if request.param == "cutedsl":
        pytest.importorskip("cutlass")
        pytest.importorskip("tvm_ffi")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            pytest.skip("CuTeDSL batch-invariant GEMM requires Blackwell")
    return request.param


@pytest.fixture(scope="module")
def operands():
    if not torch.cuda.is_available():
        pytest.skip("batch_invariant_gemm requires a GPU")
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    return a, b


def test_rows_are_bitwise_stable_across_batch_composition(operands, backend):
    a, b = operands
    full = batch_invariant_gemm(a, b, backend=backend)
    for lo, hi in SLICES:
        part = batch_invariant_gemm(a[lo:hi].contiguous(), b, backend=backend)
        assert torch.equal(part, full[lo:hi]), f"rows {lo}:{hi} changed when the batch was sliced"


def test_matches_torch_reference(operands, backend):
    a, b = operands
    got = batch_invariant_gemm(a, b, backend=backend)
    ref = (a.float() @ b.float().T).to(torch.bfloat16)
    torch.testing.assert_close(got, ref, rtol=0.02, atol=0.5)


def test_out_parameter_is_written(operands, backend):
    a, b = operands
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    assert batch_invariant_gemm(a, b, out=out, backend=backend) is out
    assert torch.equal(out, batch_invariant_gemm(a, b, backend=backend))


def test_cutedsl_one_row_out_and_auto_dispatch():
    pytest.importorskip("cutlass")
    pytest.importorskip("tvm_ffi")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("CuTeDSL batch-invariant GEMM requires Blackwell")
    torch.manual_seed(1)
    a = torch.randn(128, 512, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    full = batch_invariant_gemm(a, b, backend="cutedsl")
    one_row = torch.empty(1, 512, dtype=torch.bfloat16, device="cuda")
    assert batch_invariant_gemm(a[:1], b, out=one_row, backend="cutedsl") is one_row
    assert torch.equal(one_row, full[:1])
    assert torch.equal(batch_invariant_gemm(a, b), full)
    assert torch.equal(batch_invariant_gemm(a[:1], b), full[:1])
    out_storage = torch.empty(128 * 512 + 1, dtype=torch.bfloat16, device="cuda")
    unaligned_out = out_storage[1:].view(128, 512)
    assert unaligned_out.data_ptr() % 16 != 0
    assert batch_invariant_gemm(a, b, out=unaligned_out, backend="cutedsl") is unaligned_out
    assert torch.equal(unaligned_out, full)
    a_storage = torch.empty(128 * 512 + 1, dtype=torch.bfloat16, device="cuda")
    unaligned_a = a_storage[1:].view(128, 512)
    unaligned_a.copy_(a)
    assert torch.equal(
        batch_invariant_gemm(unaligned_a, b),
        batch_invariant_gemm(unaligned_a, b, backend="triton"),
    )


def test_unsupported_combinations_raise():
    if not torch.cuda.is_available():
        pytest.skip("batch_invariant_gemm requires a GPU")
    a = torch.randn(8, 16, dtype=torch.float16, device="cuda")
    b = torch.randn(8, 16, dtype=torch.float16, device="cuda")
    assert not is_supported(a, b)
    with pytest.raises(ValueError, match="bfloat16"):
        batch_invariant_gemm(a, b)
    with pytest.raises(ValueError, match="Unknown"):
        batch_invariant_gemm(a.to(torch.bfloat16), b.to(torch.bfloat16), backend="invalid")
