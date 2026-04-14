# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmarks for te.LayerNormMLP."""

from contextlib import nullcontext

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.quantization import autocast

from benchmarks.config.sizes import LINEAR_SIZES
from benchmarks.pytorch.conftest import skip_if_recipe_unavailable, get_recipe

RECIPES = ["bf16", "fp8_block", "mxfp8", "nvfp4"]


def shape_id(size_config):
    return size_config.id


@pytest.mark.parametrize("size_cfg", LINEAR_SIZES, ids=shape_id)
@pytest.mark.parametrize("recipe_name", RECIPES)
@pytest.mark.parametrize("direction", ["fwd_only", "fwd_bwd"])
def test_bench_layernorm_mlp(
    size_cfg, recipe_name, direction, benchmark_config, benchmark_timer, benchmark_reporter
):
    skip_if_recipe_unavailable(recipe_name)

    m, k, n = size_cfg.shape
    needs_grad = direction == "fwd_bwd"

    layer = te.LayerNormMLP(
        hidden_size=k,
        ffn_hidden_size=n,
        bias=False,
        params_dtype=torch.bfloat16,
    ).cuda()

    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", requires_grad=needs_grad)

    recipe = get_recipe(recipe_name)

    def run_step():
        ctx = autocast(enabled=True, recipe=recipe) if recipe is not None else nullcontext()
        with ctx:
            if direction == "fwd_only":
                with torch.no_grad():
                    layer(x)
            else:
                x.grad = None
                out = layer(x)
                out.backward(torch.ones_like(out))

    result = benchmark_timer.measure(
        stmt="run_step()",
        globals_dict={"run_step": run_step},
        label="LayerNormMLP",
        sub_label=f"{m}x{k}x{n}",
        name="LayerNormMLP",
        category="module",
        shape=size_cfg.shape,
        dtype="bf16",
        recipe=recipe_name,
        direction=direction,
        regime=size_cfg.regime,
    )
    benchmark_reporter.report(result)
