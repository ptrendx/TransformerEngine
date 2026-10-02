# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Eager layout metadata, copying and Python fallback contracts."""

import warnings
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from transformer_engine.pytorch.attention.dot_product_attention import utils


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Run metadata contracts without requiring a GPU for the CPU cases."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return request.param


def _outcome(function, tensors, fmt):
    try:
        return function(*tensors, qkv_format=fmt), None
    except (AssertionError, RuntimeError, IndexError, ZeroDivisionError) as error:
        return None, (type(error), str(error))


def _compare(tensors, fmt):
    """Compare classifications, replacements, aliases and exception contracts."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        actual, error = _outcome(utils.get_qkv_layout, tensors, fmt)
        reference, ref_error = _outcome(utils._get_qkv_layout_python, tensors, fmt)
    assert error == ref_error
    if error is not None:
        return None
    assert (actual[0], actual[-2:]) == (reference[0], reference[-2:])
    for original, result, expected in zip(tensors, actual[1:4], reference[1:4]):
        assert (result is original) == (expected is original)
        assert result.shape == expected.shape
        assert result.stride() == expected.stride()
        assert result.storage_offset() == expected.storage_offset()
        assert result.requires_grad == expected.requires_grad
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    for i in range(1, 4):
        for j in range(1, 4):
            assert (
                actual[i].untyped_storage().data_ptr() == actual[j].untyped_storage().data_ptr()
            ) == (
                reference[i].untyped_storage().data_ptr()
                == reference[j].untyped_storage().data_ptr()
            )
    return actual


@pytest.mark.parametrize(
    "fmt,shape", [("bshd", (2, 8, 4, 16)), ("sbhd", (8, 2, 4, 16)), ("thd", (16, 4, 16))]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.uint8])
@pytest.mark.parametrize(
    "kind",
    [
        "separate",
        "3hd",
        "h3d",
        "2hd",
        "h2d",
        "adjacent",
        "qk",
        "kv",
        "copy",
        "gqa",
        "unequal_v",
        "offset",
        "empty",
        "singleton",
        "bad_stride",
    ],
)
def test_layout_contract(device, fmt, shape, dtype, kind):
    """Cover dense FP8 backing storage as well as ordinary floating tensors."""

    def make(size):
        return torch.ones(size, dtype=dtype, device=device)

    tensors = [make(shape) for _ in range(3)]
    expected = "_".join([fmt] * 3)
    if kind in ("3hd", "h3d", "2hd", "h2d"):
        count = 3 if "3" in kind else 2
        axis = -3 if kind[0].isdigit() else -2
        packed_shape = list(shape)
        packed_shape.insert(len(shape) + axis + 1, count)
        parts = list(make(packed_shape).unbind(axis))
        tensors = parts if count == 3 else [tensors[0], *parts]
        suffix = (
            fmt[:-2] + str(count) + fmt[-2:] if axis == -3 else fmt[:-1] + str(count) + fmt[-1:]
        )
        expected = suffix if count == 3 else fmt + "_" + suffix
    elif kind in ("adjacent", "qk", "kv"):
        n = tensors[0].numel()
        buf = make((3 * n,))
        parts = [buf[i * n : (i + 1) * n].view(shape) for i in range(3)]
        tensors = (
            parts
            if kind == "adjacent"
            else ([*parts[:2], tensors[2]] if kind == "qk" else [tensors[0], *parts[:2]])
        )
    elif kind == "copy":
        tensors[1] = make((*shape[:-2], shape[-2] * 2, shape[-1]))[..., ::2, :]
    elif kind == "gqa":
        tensors[1:] = [make((*shape[:-2], 2, shape[-1])) for _ in range(2)]
    elif kind == "unequal_v":
        tensors[2] = make((*shape[:-1], 24))
    elif kind == "offset":
        tensors = [make((t.numel() + 7,))[7:].view(shape) for t in tensors]
    elif kind == "empty":
        tensors = [make((0, *shape[1:])) for _ in range(3)]
    elif kind == "singleton":
        tensors = [make((1,) * len(shape)) for _ in range(3)]
    elif kind == "bad_stride":
        tensors[0] = make((*shape[:-1], shape[-1] * 2))[..., ::2]
    result = _compare(tensors, fmt)
    if kind not in ("offset", "bad_stride"):
        assert result[0] == expected
    if kind == "copy":
        assert result[2] is not tensors[1]


@pytest.mark.parametrize(
    "fmt,qshape,kshape",
    [
        ("sbhd_2bshd", (8, 2, 4, 16), (2, 8, 4, 16)),
        ("bshd_2sbhd", (2, 8, 4, 16), (8, 2, 4, 16)),
        ("thd_2bshd", (16, 4, 16), (2, 8, 4, 16)),
        ("thd_2sbhd", (16, 4, 16), (8, 2, 4, 16)),
    ],
)
def test_mixed_formats(device, fmt, qshape, kshape):
    """Mixed cache formats keep independently declared Q and KV formats."""
    result = _compare([torch.ones(s, device=device) for s in (qshape, kshape, kshape)], fmt)
    qfmt, kvfmt = fmt.split("_2")
    assert result[0] == f"{qfmt}_{kvfmt}_{kvfmt}"


def test_copy_gradients_and_versions(device):
    """ATen copies preserve gradients; a no-copy path preserves view versions."""
    base = torch.randn(2, 8, 8, 16, device=device, requires_grad=True)
    tensors = [
        torch.randn(2, 8, 4, 16, device=device, requires_grad=True),
        base[..., ::2, :],
        torch.randn(2, 8, 4, 16, device=device, requires_grad=True),
    ]
    result = _compare(tensors, "bshd")
    reference = utils._get_qkv_layout_python(*tensors, "bshd")
    grads = torch.autograd.grad(
        sum(t.square().sum() for t in result[1:4]), tensors, retain_graph=True
    )
    ref_grads = torch.autograd.grad(sum(t.square().sum() for t in reference[1:4]), tensors)
    for actual, expected in zip(grads, ref_grads):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert result[1] is tensors[0]
    version = tensors[0]._version
    with torch.no_grad():
        result[1].add_(1)
    assert tensors[0]._version == version + 1


class _Mode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize(
    "kind",
    ["parameter", "subclass", "meta", "dispatch", "func", "export", "compile", "custom_format"],
)
def test_python_fallback(kind):
    """Never inspect ordinary TensorImpl metadata across active transforms."""
    tensors = [torch.randn(2, 8, 4, 16) for _ in range(3)]
    if kind == "parameter":
        tensors = [torch.nn.Parameter(t) for t in tensors]
    elif kind == "subclass":

        class Subclass(torch.Tensor):
            pass

        tensors = [t.as_subclass(Subclass) for t in tensors]
    elif kind == "meta":
        tensors = [t.to("meta") for t in tensors]
    with patch.object(
        utils.tex, "get_qkv_layout", side_effect=AssertionError("unexpected native entry")
    ):
        if kind == "dispatch":
            with _Mode():
                utils.get_qkv_layout(*tensors, "bshd")
        elif kind == "func":
            # The reference currently cannot read a GradTrackingTensor's storage.
            # Verify delegation, including its established exception behavior.
            with patch.object(
                utils,
                "_get_qkv_layout_python",
                return_value=("bshd_bshd_bshd", *tensors, "bshd", "bshd"),
            ) as fallback:
                torch.func.grad(
                    lambda q: utils.get_qkv_layout(q, tensors[1], tensors[2], "bshd")[1].sum()
                )(tensors[0])
                fallback.assert_called_once()
        elif kind == "export":
            with patch.object(utils, "is_in_onnx_export_mode", return_value=True):
                utils.get_qkv_layout(*tensors, "bshd")
        elif kind == "compile":

            def function(q, k, v):
                return sum(utils.get_qkv_layout(q, k, v, "bshd")[1:4])

            compiled = torch.compile(function, backend="eager", fullgraph=True, dynamic=True)
            for shape in [(2, 8, 4, 16), (3, 5, 4, 16)]:
                inputs = [torch.randn(shape) for _ in range(3)]
                torch.testing.assert_close(compiled(*inputs), sum(inputs))
        elif kind == "meta":
            # Eligibility is decided inside the native entry for metadata-only
            # tensors; this guard intentionally exercises the reference directly.
            utils._get_qkv_layout_python(*tensors, "bshd")
        else:
            utils.get_qkv_layout(*tensors, "custom" if kind == "custom_format" else "bshd")


def test_warning_and_paged_prefix(device):
    """Preserve warning text, caller location, CPU offload suppression and prefix."""
    q, k, v = torch.ones(2, 8, 3, 4, 16, device=device).unbind(2)
    with pytest.warns(DeprecationWarning, match="pointer-based detection") as recorded:
        result = utils.get_qkv_layout(q, k, v, "bshd", SimpleNamespace(is_paged=True))
    assert recorded[0].filename == __file__
    assert result[0] == "paged_kv_bs3hd"
    with patch.object(utils, "is_cpu_offload_enabled", return_value=True), warnings.catch_warnings(
        record=True
    ) as recorded:
        warnings.simplefilter("always")
        utils.get_qkv_layout(q, k, v, "bshd")
    assert not recorded


@pytest.mark.parametrize("shape", [(), (1,), (2, 8, 4, 0), (2, 8, 4)])
def test_exception_contract(device, shape):
    """Malformed ranks and zero head dimension preserve Python exceptions."""
    _compare([torch.ones(shape, device=device) for _ in range(3)], "bshd")


def test_hidden_functional_wrapper():
    """Exact Python Tensor type does not guarantee ordinary native storage."""
    tensors = [torch._to_functional_tensor(torch.ones(2, 8, 4, 16)) for _ in range(3)]
    assert all(type(t) is torch.Tensor for t in tensors)
    assert utils.tex.get_qkv_layout(*tensors, "bshd") is None
    _compare(tensors, "bshd")


def test_native_meta_delegation():
    """The installed binding delegates meta storage without reading pointers."""
    tensors = [torch.ones(2, 8, 4, 16, device="meta") for _ in range(3)]
    assert utils.tex.get_qkv_layout(*tensors, "bshd") is None
    result = utils.get_qkv_layout(*tensors, "bshd")
    assert result[1] is tensors[0]


def test_large_integer_stride_normalization():
    """Preserve Python's exact integer-ratio rounding for empty tensor strides."""
    q = torch.empty(0, 2, 4, 3)
    k = torch.empty_strided((0, 2, 4, 3), (9007199254740993, 9007199254740993, 3, 1))
    v = torch.empty_strided((0, 2, 4, 5), (15011998757901655, 15011998757901655, 5, 1))
    assert utils.tex.get_qkv_layout(q, k, v, "bshd") is None
    assert _compare([q, k, v], "bshd")[0] == "bshd_bshd_bshd"
