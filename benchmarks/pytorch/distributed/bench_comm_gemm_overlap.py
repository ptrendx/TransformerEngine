# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Communication-GEMM overlap benchmarks."""

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
from transformer_engine.pytorch.quantization import autocast
from contextlib import nullcontext

from benchmarks.config.sizes import LINEAR_SIZES, filter_sizes
from benchmarks.config.benchmark_config import BenchmarkConfig
from benchmarks.common.distributed_utils import distributed_measure, init_nccl_warmup
from benchmarks.common.reporter import BenchmarkReporter
from benchmarks.pytorch.conftest import get_recipe


def run(config: BenchmarkConfig, reporter: BenchmarkReporter):
    """Run comm-GEMM overlap benchmarks. Called from run_distributed.py."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    init_nccl_warmup()

    sizes = filter_sizes(LINEAR_SIZES, config.size_filter)
    sizes = [s for s in sizes if s.regime == "gpu_bound"]

    for recipe_name in config.recipes:
        recipe = get_recipe(recipe_name)

        for size_cfg in sizes:
            m, k, n = size_cfg.shape

            # Column-parallel with all-gather overlap
            try:
                layer = te.Linear(
                    k, n,
                    bias=False,
                    params_dtype=torch.bfloat16,
                    tp_group=dist.group.WORLD,
                    tp_size=world_size,
                    parallel_mode="column",
                    ub_overlap_ag=True,
                ).cuda()
            except (RuntimeError, ValueError) as e:
                # Userbuffers may not be available
                if rank == 0:
                    print(f"Skipping comm_gemm_overlap: {e}")
                return

            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda",
                            requires_grad=True)
            grad = torch.randn(m, n // world_size, dtype=torch.bfloat16,
                               device="cuda")

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
                name="te.Linear(AG-overlap)",
                category="comm_gemm_overlap",
                shape=size_cfg.shape,
                recipe=recipe_name,
                direction="fwd_bwd",
                regime=size_cfg.regime,
            )

            if result is not None:
                reporter.report(result)
