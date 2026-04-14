# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor parallel benchmarks for te.Linear with column/row parallel modes."""

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
from transformer_engine.pytorch.quantization import autocast
from contextlib import nullcontext

from benchmarks.config.sizes import LINEAR_SIZES, filter_sizes
from benchmarks.config.benchmark_config import BenchmarkConfig
from benchmarks.common.distributed_utils import distributed_measure, init_nccl_warmup
from benchmarks.common.reporter import BenchmarkReporter
from benchmarks.common.result_types import collect_metadata
from benchmarks.pytorch.conftest import skip_if_recipe_unavailable, get_recipe


def run(config: BenchmarkConfig, reporter: BenchmarkReporter):
    """Run tensor parallel benchmarks. Called from run_distributed.py."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    init_nccl_warmup()

    sizes = filter_sizes(LINEAR_SIZES, config.size_filter)
    # Only GPU-bound sizes make sense for distributed benchmarks
    sizes = [s for s in sizes if s.regime == "gpu_bound"]

    for recipe_name in config.recipes:
        recipe = get_recipe(recipe_name)

        for parallel_mode in ["column", "row"]:
            for size_cfg in sizes:
                m, k, n = size_cfg.shape

                # For TP, output dimension is split across ranks
                layer = te.Linear(
                    k, n,
                    bias=False,
                    params_dtype=torch.bfloat16,
                    tp_group=dist.group.WORLD,
                    tp_size=world_size,
                    parallel_mode=parallel_mode,
                ).cuda()

                if parallel_mode == "column":
                    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda",
                                    requires_grad=True)
                    grad = torch.randn(m, n // world_size, dtype=torch.bfloat16,
                                       device="cuda")
                else:
                    x = torch.randn(m, k // world_size, dtype=torch.bfloat16,
                                    device="cuda", requires_grad=True)
                    grad = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")

                def run_step():
                    ctx = autocast(enabled=True, recipe=recipe) if recipe else nullcontext()
                    with ctx:
                        layer.zero_grad()
                        x.grad = None
                        y = layer(x)
                        y.backward(grad)

                result = distributed_measure(
                    run_step,
                    config=config,
                    name=f"te.Linear(TP-{parallel_mode})",
                    category="tensor_parallel",
                    shape=size_cfg.shape,
                    recipe=recipe_name,
                    direction="fwd_bwd",
                    regime=size_cfg.regime,
                    total_flops=2 * m * k * n * 2,  # fwd + bwd
                )

                if result is not None:
                    reporter.report(result)
