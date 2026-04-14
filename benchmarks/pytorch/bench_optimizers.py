# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmarks for te.optimizers.FusedAdam and te.optimizers.FusedSGD."""

import pytest
import torch

from transformer_engine.pytorch.optimizers import FusedAdam, FusedSGD

from benchmarks.config.sizes import OPTIMIZER_SIZES


def shape_id(size_config):
    return size_config.id


@pytest.mark.parametrize("size_cfg", OPTIMIZER_SIZES, ids=shape_id)
def test_bench_fused_adam(
    size_cfg, benchmark_config, benchmark_timer, benchmark_reporter
):
    num_params, param_size = size_cfg.shape

    params = [
        torch.randn(param_size, dtype=torch.float32, device="cuda", requires_grad=True)
        for _ in range(num_params)
    ]

    # Create gradients
    for p in params:
        p.grad = torch.randn_like(p)

    optimizer = FusedAdam(params, lr=1e-3)

    def run_step():
        optimizer.step()

    result = benchmark_timer.measure(
        stmt="run_step()",
        globals_dict={"run_step": run_step},
        label="FusedAdam",
        sub_label=f"{num_params}x{param_size}",
        name="FusedAdam",
        category="optimizer",
        shape=size_cfg.shape,
        dtype="fp32",
        recipe="bf16",
        direction="fwd_only",
        regime=size_cfg.regime,
    )
    benchmark_reporter.report(result)


@pytest.mark.parametrize("size_cfg", OPTIMIZER_SIZES, ids=shape_id)
def test_bench_fused_sgd(
    size_cfg, benchmark_config, benchmark_timer, benchmark_reporter
):
    num_params, param_size = size_cfg.shape

    params = [
        torch.randn(param_size, dtype=torch.float32, device="cuda", requires_grad=True)
        for _ in range(num_params)
    ]

    # Create gradients
    for p in params:
        p.grad = torch.randn_like(p)

    optimizer = FusedSGD(params, lr=1e-2, momentum=0.9)

    def run_step():
        optimizer.step()

    result = benchmark_timer.measure(
        stmt="run_step()",
        globals_dict={"run_step": run_step},
        label="FusedSGD",
        sub_label=f"{num_params}x{param_size}",
        name="FusedSGD",
        category="optimizer",
        shape=size_cfg.shape,
        dtype="fp32",
        recipe="bf16",
        direction="fwd_only",
        regime=size_cfg.regime,
    )
    benchmark_reporter.report(result)
