# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmarks for te.GroupedLinear module."""

import pytest
import torch
from contextlib import nullcontext

import transformer_engine.pytorch as te
from transformer_engine.pytorch.quantization import autocast

from benchmarks.config.sizes import GROUPED_LINEAR_SIZES, filter_sizes
from benchmarks.pytorch.conftest import skip_if_recipe_unavailable, get_recipe


def shape_id(size_config):
    return size_config.id


RECIPES = ["bf16", "fp8_block", "mxfp8", "nvfp4"]


@pytest.mark.parametrize("size_cfg", GROUPED_LINEAR_SIZES, ids=shape_id)
@pytest.mark.parametrize("recipe_name", RECIPES)
@pytest.mark.parametrize("direction", ["fwd_only", "fwd_bwd"])
def test_bench_grouped_linear(
    size_cfg, recipe_name, direction, benchmark_config, benchmark_timer, benchmark_reporter
):
    if (
        size_cfg.regime not in (benchmark_config.size_filter, "all")
        and benchmark_config.size_filter != "all"
    ):
        pytest.skip(f"Skipping {size_cfg.regime} sizes")

    skip_if_recipe_unavailable(recipe_name)
    recipe = get_recipe(recipe_name)

    m, k, n, num_gemms = size_cfg.shape
    layer = te.GroupedLinear(
        num_gemms=num_gemms,
        in_features=k,
        out_features=n,
        bias=False,
        params_dtype=torch.bfloat16,
    ).cuda()
    x = torch.randn(
        m, k, dtype=torch.bfloat16, device="cuda", requires_grad=(direction == "fwd_bwd")
    )
    m_splits = [m // num_gemms] * num_gemms
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
                    layer(x, m_splits=m_splits)
            else:
                layer.zero_grad()
                x.grad = None
                y = layer(x, m_splits=m_splits)
                y.backward(grad)

    result = benchmark_timer.measure(
        stmt="run_step()",
        globals_dict={"run_step": run_step},
        label="te.GroupedLinear",
        sub_label=f"{m}x{k}x{n}x{num_gemms}_{recipe_name}_{direction}",
        name="te.GroupedLinear",
        category="grouped_linear",
        shape=size_cfg.shape,
        recipe=recipe_name,
        direction=direction,
        regime=size_cfg.regime,
        total_flops=2 * m * k * n * (2 if direction == "fwd_bwd" else 1),
    )
    benchmark_reporter.report(result)
