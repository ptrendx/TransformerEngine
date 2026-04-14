# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmarks for standalone quantization recipe overhead."""

import pytest
import torch
from contextlib import nullcontext

import transformer_engine.pytorch as te
from transformer_engine.pytorch.quantization import autocast

from benchmarks.config.sizes import CAST_SIZES, filter_sizes
from benchmarks.pytorch.conftest import skip_if_recipe_unavailable, get_recipe


def shape_id(size_config):
    return size_config.id


RECIPES = ["fp8_block", "mxfp8", "nvfp4"]


@pytest.mark.parametrize("size_cfg", CAST_SIZES, ids=shape_id)
@pytest.mark.parametrize("recipe_name", RECIPES)
def test_bench_quantize_autocast(
    size_cfg, recipe_name, benchmark_config, benchmark_timer, benchmark_reporter,
):
    """Benchmark the overhead of the autocast context manager + a simple Linear forward.

    This measures the combined cost of recipe management (amax history,
    scale computation) and the quantized operation itself, isolating the
    quantization overhead from a pure bf16 baseline.
    """
    if benchmark_config.size_filter != "all" and size_cfg.regime != benchmark_config.size_filter:
        pytest.skip(f"Skipping {size_cfg.regime} sizes")

    skip_if_recipe_unavailable(recipe_name)
    recipe = get_recipe(recipe_name)

    rows, cols = size_cfg.shape
    layer = te.Linear(cols, cols, bias=False, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")

    def run_step():
        with autocast(enabled=True, recipe=recipe):
            with torch.no_grad():
                layer(x)

    result = benchmark_timer.measure(
        stmt="run_step()",
        globals_dict={"run_step": run_step},
        label="autocast+Linear",
        sub_label=f"{rows}x{cols}_{recipe_name}",
        name=f"autocast+Linear({recipe_name})",
        category="quantization",
        shape=size_cfg.shape,
        recipe=recipe_name,
        direction="fwd",
        regime=size_cfg.regime,
    )
    benchmark_reporter.report(result)


@pytest.mark.parametrize("size_cfg", CAST_SIZES, ids=shape_id)
def test_bench_linear_bf16_baseline(
    size_cfg, benchmark_config, benchmark_timer, benchmark_reporter,
):
    """bf16 baseline for comparison with quantized versions."""
    if benchmark_config.size_filter != "all" and size_cfg.regime != benchmark_config.size_filter:
        pytest.skip(f"Skipping {size_cfg.regime} sizes")

    rows, cols = size_cfg.shape
    layer = te.Linear(cols, cols, bias=False, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")

    def run_step():
        with torch.no_grad():
            layer(x)

    result = benchmark_timer.measure(
        stmt="run_step()",
        globals_dict={"run_step": run_step},
        label="Linear(bf16)",
        sub_label=f"{rows}x{cols}_bf16",
        name="Linear(bf16_baseline)",
        category="quantization",
        shape=size_cfg.shape,
        recipe="bf16",
        direction="fwd",
        regime=size_cfg.regime,
    )
    benchmark_reporter.report(result)
