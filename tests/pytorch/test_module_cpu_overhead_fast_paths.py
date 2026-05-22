# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from contextlib import contextmanager
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

import transformer_engine  # noqa: F401
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch.module import layernorm_linear as te_layernorm_linear_module
from transformer_engine.pytorch.module import layernorm_mlp as te_layernorm_mlp_module
from transformer_engine.pytorch.module import linear as te_linear_module
from transformer_engine.pytorch.quantization import FP8GlobalStateManager


_HIDDEN_SIZE = 32
_FFN_HIDDEN_SIZE = 128
_BATCH_SIZE = 32
_MODULE_NAMES = ("linear", "layernorm_linear", "layernorm_mlp")
_CACHE_KEYS = {
    "linear": ("weight",),
    "layernorm_linear": ("weight",),
    "layernorm_mlp": ("fc1_weight", "fc2_weight"),
}
_BF16_TOLS = {"rtol": 0.0, "atol": 0.0}
_MXFP8_TOLS = {"rtol": 2.0e-2, "atol": 2.0e-2}


bf16_available, reason_for_no_bf16 = te.is_bf16_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)


@pytest.fixture(autouse=True)
def _reset_fp8_global_state():
    yield
    FP8GlobalStateManager.reset()


def _reset_rng(seed: int = 1234) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _make_module(module_name: str) -> torch.nn.Module:
    if module_name == "linear":
        return te.Linear(
            _HIDDEN_SIZE,
            _HIDDEN_SIZE,
            bias=True,
            params_dtype=torch.bfloat16,
            device="cuda",
        )
    if module_name == "layernorm_linear":
        return te.LayerNormLinear(
            _HIDDEN_SIZE,
            _HIDDEN_SIZE,
            bias=True,
            params_dtype=torch.bfloat16,
            device="cuda",
        )
    if module_name == "layernorm_mlp":
        return te.LayerNormMLP(
            _HIDDEN_SIZE,
            _FFN_HIDDEN_SIZE,
            bias=True,
            params_dtype=torch.bfloat16,
            device="cuda",
        )
    raise ValueError(f"Unsupported module {module_name!r}")


def _make_input(seed: int = 2345) -> torch.Tensor:
    _reset_rng(seed)
    return torch.randn(
        _BATCH_SIZE,
        _HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )


def _clone_parameters(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone().requires_grad_(param.requires_grad)
        for name, param in module.named_parameters()
    }


def _reference_forward(
    module_name: str,
    module: torch.nn.Module,
    params: dict[str, torch.Tensor],
    inp: torch.Tensor,
) -> torch.Tensor:
    if module_name == "linear":
        return F.linear(inp, params["weight"], params["bias"])

    ln_out = F.layer_norm(
        inp,
        (_HIDDEN_SIZE,),
        params["layer_norm_weight"],
        params["layer_norm_bias"],
        module.eps,
    )
    if module_name == "layernorm_linear":
        return F.linear(ln_out, params["weight"], params["bias"])

    if module_name == "layernorm_mlp":
        hidden = F.linear(ln_out, params["fc1_weight"], params["fc1_bias"])
        hidden = F.gelu(hidden, approximate="tanh")
        return F.linear(hidden, params["fc2_weight"], params["fc2_bias"])

    raise ValueError(f"Unsupported module {module_name!r}")


@contextmanager
def _fail_if_te_autograd_path_is_used(module_name: str):
    autograd_classes = {
        "linear": te_linear_module._Linear,
        "layernorm_linear": te_layernorm_linear_module._LayerNormLinear,
        "layernorm_mlp": te_layernorm_mlp_module._LayerNormMLP,
    }
    with mock.patch.object(
        autograd_classes[module_name],
        "apply",
        side_effect=AssertionError(f"{module_name} did not take the torch BF16 fast path"),
    ):
        yield


def _disable_mxfp8_deferred_backward(module: torch.nn.Module) -> None:
    module._should_defer_fp8_backward_tensors = lambda debug, is_grad_enabled: False


def _assert_tensors_close(actual: torch.Tensor, expected: torch.Tensor, tols: dict[str, float]):
    torch.testing.assert_close(
        actual,
        expected,
        check_dtype=False,
        check_device=True,
        **tols,
    )


def _assert_grads_close(
    actual_module: torch.nn.Module,
    expected_module: torch.nn.Module,
    tols: dict[str, float],
) -> None:
    for (actual_name, actual_param), (expected_name, expected_param) in zip(
        actual_module.named_parameters(), expected_module.named_parameters()
    ):
        assert actual_name == expected_name
        assert actual_param.grad is not None, f"{actual_name} grad was not computed"
        assert expected_param.grad is not None, f"{expected_name} grad was not computed"
        _assert_tensors_close(actual_param.grad, expected_param.grad, tols)


def _assert_parameter_grads_match_reference(
    module: torch.nn.Module,
    ref_params: dict[str, torch.Tensor],
    tols: dict[str, float],
) -> None:
    for name, param in module.named_parameters():
        assert param.grad is not None, f"{name} grad was not computed"
        assert ref_params[name].grad is not None, f"reference {name} grad was not computed"
        _assert_tensors_close(param.grad, ref_params[name].grad, tols)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not bf16_available, reason=reason_for_no_bf16)
@pytest.mark.parametrize("module_name", _MODULE_NAMES)
def test_module_cpu_overhead_bf16_fast_path_matches_torch_reference(module_name):
    _reset_rng()
    module = _make_module(module_name)
    inp = _make_input()
    ref_inp = inp.detach().clone().requires_grad_(True)
    ref_params = _clone_parameters(module)

    with _fail_if_te_autograd_path_is_used(module_name):
        out = module(inp)
    ref_out = _reference_forward(module_name, module, ref_params, ref_inp)

    _assert_tensors_close(out, ref_out, _BF16_TOLS)

    _reset_rng(3456)
    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    ref_out.backward(grad_output.detach().clone())

    assert inp.grad is not None
    assert ref_inp.grad is not None
    _assert_tensors_close(inp.grad, ref_inp.grad, _BF16_TOLS)
    _assert_parameter_grads_match_reference(module, ref_params, _BF16_TOLS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not bf16_available, reason=reason_for_no_bf16)
@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("module_name", _MODULE_NAMES)
def test_module_cpu_overhead_mxfp8_deferred_backward_matches_non_deferred(module_name):
    _reset_rng()
    deferred_module = _make_module(module_name)
    reference_module = _make_module(module_name)
    reference_module.load_state_dict(deferred_module.state_dict())
    _disable_mxfp8_deferred_backward(reference_module)

    inp = _make_input()
    ref_inp = inp.detach().clone().requires_grad_(True)
    fp8_recipe = MXFP8BlockScaling()

    with te.autocast(enabled=True, recipe=fp8_recipe):
        out = deferred_module(inp)
        ref_out = reference_module(ref_inp)

    for cache_key in _CACHE_KEYS[module_name]:
        assert cache_key in deferred_module._fp8_workspaces
    assert not reference_module._fp8_workspaces
    _assert_tensors_close(out, ref_out, _MXFP8_TOLS)

    _reset_rng(4567)
    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    ref_out.backward(grad_output.detach().clone())

    assert inp.grad is not None
    assert ref_inp.grad is not None
    _assert_tensors_close(inp.grad, ref_inp.grad, _MXFP8_TOLS)
    _assert_grads_close(deferred_module, reference_module, _MXFP8_TOLS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not bf16_available, reason=reason_for_no_bf16)
@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("module_name", _MODULE_NAMES)
def test_module_cpu_overhead_mxfp8_weight_update_refreshes_cached_workspace(module_name):
    _reset_rng()
    module = _make_module(module_name)
    fp8_recipe = MXFP8BlockScaling()

    inp = _make_input()
    with te.autocast(enabled=True, recipe=fp8_recipe):
        out = module(inp)
    out.sum().backward()

    cache_key = _CACHE_KEYS[module_name][0]
    weight = getattr(module, cache_key)
    workspace = module._fp8_workspaces[cache_key]
    cached_rowwise_data = workspace._rowwise_data.detach().clone()
    cached_version = module._fp8_workspace_versions[cache_key]
    assert cached_version == weight._version

    module.zero_grad(set_to_none=True)
    inp.grad = None
    with torch.no_grad():
        weight.add_(0.5)
    assert weight._version != cached_version

    inp = _make_input(seed=5678)
    with te.autocast(enabled=True, recipe=fp8_recipe):
        out = module(inp)
    out.sum().backward()

    refreshed_workspace = module._fp8_workspaces[cache_key]
    assert module._fp8_workspace_versions[cache_key] == weight._version
    assert not torch.equal(cached_rowwise_data, refreshed_workspace._rowwise_data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not bf16_available, reason=reason_for_no_bf16)
@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
def test_module_cpu_overhead_linear_mxfp8_input_workspace_matches_uncached_path():
    _reset_rng()
    module = _make_module("linear")
    reference_module = _make_module("linear")
    reference_module.load_state_dict(module.state_dict())
    reference_module._get_fp8_transient_workspace = lambda *args, **kwargs: None
    fp8_recipe = MXFP8BlockScaling()

    inp = _make_input()
    ref_inp = inp.detach().clone().requires_grad_(True)
    with te.autocast(enabled=True, recipe=fp8_recipe):
        out = module(inp)
        ref_out = reference_module(ref_inp)

    workspace = module._fp8_transient_workspaces["linear_input"]
    weight_workspace = module._fp8_workspaces["weight"]
    cached_rowwise_data = workspace._rowwise_data.detach().clone()
    assert not reference_module._fp8_transient_workspaces
    assert workspace._with_gemm_swizzled_scales
    assert weight_workspace._with_gemm_swizzled_scales
    _assert_tensors_close(out, ref_out, _MXFP8_TOLS)

    with mock.patch.object(
        te_linear_module.tex,
        "quantize",
        wraps=te_linear_module.tex.quantize,
    ) as quantize_mock, mock.patch.object(
        te_linear_module.tex,
        "mxfp8_gemm_tn",
        wraps=te_linear_module.tex.mxfp8_gemm_tn,
    ) as fast_gemm_mock:
        with te.autocast(enabled=True, recipe=fp8_recipe):
            out = module(inp)
    assert quantize_mock.call_count == 0
    assert fast_gemm_mock.call_count == 1
    ref_inp = inp.detach().clone().requires_grad_(True)
    with te.autocast(enabled=True, recipe=fp8_recipe):
        ref_out = reference_module(ref_inp)
    assert module._fp8_transient_workspaces["linear_input"] is workspace
    assert module._fp8_workspaces["weight"] is weight_workspace
    torch.testing.assert_close(
        cached_rowwise_data,
        workspace._rowwise_data,
        check_dtype=True,
        check_device=True,
    )
    _assert_tensors_close(out, ref_out, _MXFP8_TOLS)

    _reset_rng(7890)
    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    ref_out.backward(grad_output.detach().clone())
    assert inp.grad is not None
    assert ref_inp.grad is not None
    _assert_tensors_close(inp.grad, ref_inp.grad, _MXFP8_TOLS)
    _assert_grads_close(module, reference_module, _MXFP8_TOLS)
    module.zero_grad(set_to_none=True)
    reference_module.zero_grad(set_to_none=True)
    inp.grad = None

    new_inp = _make_input(seed=6789)
    new_ref_inp = new_inp.detach().clone().requires_grad_(True)
    with te.autocast(enabled=True, recipe=fp8_recipe):
        out = module(new_inp)
        ref_out = reference_module(new_ref_inp)

    assert module._fp8_transient_workspaces["linear_input"] is workspace
    assert not torch.equal(cached_rowwise_data, workspace._rowwise_data)
    _assert_tensors_close(out, ref_out, _MXFP8_TOLS)
