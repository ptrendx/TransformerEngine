# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmarks for te.RMSNorm module."""

import pytest
import torch

import transformer_engine.pytorch as te

from benchmarks.config.sizes import NORMALIZATION_SIZES, filter_sizes


def shape_id(size_config):
    return size_config.id


@pytest.mark.parametrize("size_cfg", NORMALIZATION_SIZES, ids=shape_id)
@pytest.mark.parametrize("direction", ["fwd_only", "fwd_bwd"])
def test_bench_rmsnorm(
    size_cfg, direction, benchmark_config, benchmark_timer, benchmark_reporter
):
    if (
        size_cfg.regime not in (benchmark_config.size_filter, "all")
        and benchmark_config.size_filter != "all"
    ):
        pytest.skip(f"Skipping {size_cfg.regime} sizes")

    rows, hidden_size = size_cfg.shape
    layer = te.RMSNorm(hidden_size, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(
        rows, hidden_size, dtype=torch.bfloat16, device="cuda",
        requires_grad=(direction == "fwd_bwd"),
    )
    grad = (
        torch.randn(rows, hidden_size, dtype=torch.bfloat16, device="cuda")
        if direction == "fwd_bwd"
        else None
    )

    def run_step():
        if direction == "fwd_only":
            with torch.no_grad():
                layer(x)
        else:
            layer.zero_grad()
            x.grad = None
            y = layer(x)
            y.backward(grad)

    # Total bytes: read input + write output (fwd), doubled for bwd
    total_bytes = (
        2 * rows * hidden_size * 2 * (2 if direction == "fwd_bwd" else 1)
    )

    result = benchmark_timer.measure(
        stmt="run_step()",
        globals_dict={"run_step": run_step},
        label="te.RMSNorm",
        sub_label=f"{rows}x{hidden_size}_bf16_{direction}",
        name="te.RMSNorm",
        category="normalization",
        shape=size_cfg.shape,
        recipe="bf16",
        direction=direction,
        regime=size_cfg.regime,
        total_bytes=total_bytes,
    )
    benchmark_reporter.report(result)
