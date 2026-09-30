# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""A row's output must not depend on which other rows were in the batch.

The general GEMM path lets cuBLASLt pick a kernel from the full problem shape, so
the same row can produce different bits depending on the batch it arrives in.
``batch_invariant_gemm`` fixes the K reduction order for a given weight shape
so the result is a function of the row and the weight only.
"""

import pytest
import torch

from transformer_engine.pytorch.cpp_extensions.batch_invariant_gemm import (
    batch_invariant_gemm,
    is_supported,
)
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm

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


def test_triton_tiles_are_bitwise_stable_across_batch_sizes():
    """On SM100, M/N tiling changes must leave each row's K reduction unchanged."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("adaptive Triton tiling requires Blackwell")
    torch.manual_seed(2026)
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    full = batch_invariant_gemm(a, b, backend="triton")
    for lo, hi in ((0, 1), (0, 64), (0, 256), (128, 1152), (128, 2304)):
        part = batch_invariant_gemm(a[lo:hi], b, backend="triton")
        assert torch.equal(part, full[lo:hi]), f"rows {lo}:{hi} changed across Triton tiles"


@pytest.mark.parametrize(
    "shape", [(4096, 4096, 4096), (4097, 1032, 520), (1025, 520, 72), (1025, 8, 8)]
)
def test_cutedsl_persistent_tiles_are_bitwise_stable(shape):
    """Cover tile transitions, multiple persistent iterations, and N/K tails."""
    pytest.importorskip("cutlass")
    pytest.importorskip("tvm_ffi")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("CuTeDSL batch-invariant GEMM requires Blackwell")
    m, n, k = shape
    torch.manual_seed(2026)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    full = batch_invariant_gemm(a, b, backend="cutedsl")
    ref = (a.float() @ b.float().T).to(torch.bfloat16)
    torch.testing.assert_close(full, ref, rtol=0.008, atol=0.03125)
    for lo, rows in ((0, 1), (5, 7), (0, 256), (5, 257), (0, 1024), (0, 1025)):
        part = batch_invariant_gemm(a[lo : lo + rows], b, backend="cutedsl")
        assert torch.equal(part, full[lo : lo + rows]), f"{rows} rows changed across CuTe tiles"
    perm = torch.randperm(m, device="cuda")
    reordered = batch_invariant_gemm(a[perm], b, backend="cutedsl")
    assert torch.equal(reordered, full[perm]), "rows changed when moved to different tiles"


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


@pytest.mark.parametrize("m", [0, 1, 17, 257, 1025])
@pytest.mark.parametrize("output_kind", ["allocate", "preallocated", "unaligned"])
def test_general_gemm_batch_invariant_native(m, output_kind, monkeypatch):
    """Exercise the native allocation, padding, copy, and cached TVM FFI paths."""
    pytest.importorskip("cutlass")
    pytest.importorskip("tvm_ffi")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("native CuTeDSL GEMM requires compute capability 10.x")
    import importlib

    module = importlib.import_module(
        "transformer_engine.pytorch.cpp_extensions.batch_invariant_gemm"
    )
    if not module._native_cutedsl_available():
        pytest.skip("the extension was built without TVM FFI headers")
    a = torch.randn(m, 520, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(512, 520, dtype=torch.bfloat16, device="cuda")
    expected = module._batch_invariant_gemm_cutedsl_python(a, b, None) if m else a[:, :512]
    if output_kind == "allocate":
        out = None
    elif output_kind == "preallocated":
        out = torch.empty(m, 512, dtype=torch.bfloat16, device="cuda")
    else:
        out = torch.empty(m * 512 + 1, dtype=torch.bfloat16, device="cuda")[1:].view(m, 512)

    def unexpected_python_launch(*args, **kwargs):
        raise AssertionError("native dispatch re-entered the Python kernel launcher")

    monkeypatch.setattr(module, "_batch_invariant_gemm_cutedsl_python", unexpected_python_launch)
    result, bias_grad, gelu_input, extra = general_gemm(b, a, out=out, batch_invariant=True)
    assert torch.equal(result, expected)
    assert result.shape == (m, 512)
    assert bias_grad is gelu_input is extra is None
    if out is not None:
        assert result is out
    torch.testing.assert_close(
        result, (a.float() @ b.float().T).bfloat16(), rtol=0.008, atol=0.03125
    )
    if m:
        assert torch.equal(general_gemm(b, a[:1], batch_invariant=True)[0], result[:1])


@pytest.mark.parametrize(
    "options",
    [
        {"layout": "NN"},
        {"out_dtype": torch.float32},
        {"alpha": 2.0},
        {"beta": 1.0},
        {"accumulate": True},
        {"gelu": True},
        {"grad": True},
        {"use_split_accumulator": True},
        {"bulk_overlap": True},
        {"bias": "provided"},
        {"quantization_params": "provided"},
    ],
)
def test_general_gemm_batch_invariant_rejects_options(operands, options):
    a, b = operands
    with pytest.raises(ValueError, match="unscaled BF16 TN forward"):
        general_gemm(b, a, batch_invariant=True, **options)


def test_general_gemm_batch_invariant_triton_fallback(operands):
    a, b = operands
    assert torch.equal(
        general_gemm(b, a, batch_invariant=True)[0], batch_invariant_gemm(a, b, backend="triton")
    )


def test_native_cutedsl_nondefault_stream_and_graph(backend):
    if backend != "cutedsl":
        pytest.skip("native dispatch is specific to CuTeDSL")
    a = torch.randn(17, 512, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expected = general_gemm(b, a, batch_invariant=True)[0]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = general_gemm(b, a, batch_invariant=True)[0]
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(result, expected)


def test_cutedsl_python_launcher_when_native_unavailable(backend, operands, monkeypatch):
    """Dependencies installed after a build without TVM headers still work."""
    if backend != "cutedsl":
        pytest.skip("Python launcher fallback is specific to CuTeDSL")
    import importlib

    module = importlib.import_module(
        "transformer_engine.pytorch.cpp_extensions.batch_invariant_gemm"
    )
    a, b = operands
    expected = batch_invariant_gemm(a, b, backend="cutedsl")
    monkeypatch.setattr(module, "_native_cutedsl_available", lambda: False)
    assert torch.equal(batch_invariant_gemm(a, b, backend="cutedsl"), expected)


@pytest.mark.parametrize("kind", ["shape", "dtype", "layout", "device"])
def test_native_cutedsl_rejects_invalid_output(kind, backend):
    if backend != "cutedsl":
        pytest.skip("native output validation is specific to CuTeDSL")
    a = torch.randn(17, 512, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    if kind == "shape":
        out = torch.empty(16, 512, dtype=torch.bfloat16, device="cuda")
    elif kind == "dtype":
        out = torch.empty(17, 512, dtype=torch.float32, device="cuda")
    elif kind == "layout":
        out = torch.empty(512, 17, dtype=torch.bfloat16, device="cuda").T
    else:
        out = torch.empty(17, 512, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="out"):
        batch_invariant_gemm(a, b, out=out, backend="cutedsl")


def test_native_cutedsl_device_guard(backend):
    if backend != "cutedsl" or torch.cuda.device_count() < 2:
        pytest.skip("device-guard test requires CuTeDSL and two GPUs")
    device = (torch.cuda.current_device() + 1) % torch.cuda.device_count()
    a = torch.randn(17, 512, dtype=torch.bfloat16, device=device)
    b = torch.randn(512, 512, dtype=torch.bfloat16, device=device)
    result = general_gemm(b, a, batch_invariant=True)[0]
    assert result.device.index == device
    torch.testing.assert_close(
        result, (a.float() @ b.float().T).bfloat16(), rtol=0.008, atol=0.03125
    )
