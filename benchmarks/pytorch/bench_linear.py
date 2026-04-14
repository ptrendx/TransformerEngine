# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmarks for te.Linear module."""

import pytest
import torch
from contextlib import nullcontext

import transformer_engine.pytorch as te
from transformer_engine.pytorch.quantization import autocast

from benchmarks.config.sizes import LINEAR_SIZES, filter_sizes
from benchmarks.pytorch.conftest import skip_if_recipe_unavailable, get_recipe


def shape_id(size_config):
    return size_config.id


RECIPES = ["bf16", "fp8_block", "mxfp8", "nvfp4"]


@pytest.mark.parametrize("size_cfg", LINEAR_SIZES, ids=shape_id)
@pytest.mark.parametrize("recipe_name", RECIPES)
@pytest.mark.parametrize("direction", ["fwd_only", "fwd_bwd"])
def test_bench_linear(
    size_cfg, recipe_name, direction, benchmark_config, benchmark_timer, benchmark_reporter
):
    if (
        size_cfg.regime not in (benchmark_config.size_filter, "all")
        and benchmark_config.size_filter != "all"
    ):
        pytest.skip(f"Skipping {size_cfg.regime} sizes")

    skip_if_recipe_unavailable(recipe_name)
    recipe = get_recipe(recipe_name)

    m, k, n = size_cfg.shape
    layer = te.Linear(k, n, bias=False, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(
        m, k, dtype=torch.bfloat16, device="cuda", requires_grad=(direction == "fwd_bwd")
    )
    grad = (
        torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        if direction == "fwd_bwd"
        else None
    )

    def run_step():
        ctx = autocast(enabled=True, recipe=recipe) if recipe is not None else nullcontext()
        with ctx:
            if direction == "fwd_only":
                with torch.no_grad():
                    layer(x)
            else:
                layer.zero_grad()
                x.grad = None
                y = layer(x)
                y.backward(grad)

    result = benchmark_timer.measure(
        stmt="run_step()",
        globals_dict={"run_step": run_step},
        label="te.Linear",
        sub_label=f"{m}x{k}x{n}_{recipe_name}_{direction}",
        name="te.Linear",
        category="linear",
        shape=size_cfg.shape,
        recipe=recipe_name,
        direction=direction,
        regime=size_cfg.regime,
        total_flops=2 * m * k * n * (2 if direction == "fwd_bwd" else 1),
    )
    benchmark_reporter.report(result)
