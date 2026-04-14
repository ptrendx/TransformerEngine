# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Context parallel benchmarks for DotProductAttention with CP."""

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te

from benchmarks.config.sizes import ATTENTION_SIZES, filter_sizes
from benchmarks.config.benchmark_config import BenchmarkConfig
from benchmarks.common.distributed_utils import distributed_measure, init_nccl_warmup
from benchmarks.common.reporter import BenchmarkReporter


def run(config: BenchmarkConfig, reporter: BenchmarkReporter):
    """Run context parallel attention benchmarks. Called from run_distributed.py."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    init_nccl_warmup()

    sizes = filter_sizes(ATTENTION_SIZES, config.size_filter)
    # Only GPU-bound and seqlen divisible by world_size
    sizes = [s for s in sizes if s.regime == "gpu_bound" and s.shape[4] % world_size == 0]

    for size_cfg in sizes:
        batch, num_heads, num_gqa_groups, head_dim, seq_q, seq_kv = size_cfg.shape

        attn = te.DotProductAttention(
            num_attention_heads=num_heads,
            kv_channels=head_dim,
            num_gqa_groups=num_gqa_groups,
            attention_dropout=0.0,
            attn_mask_type="causal",
            cp_group=dist.group.WORLD,
            cp_global_ranks=list(range(world_size)),
            cp_stream=torch.cuda.Stream(),
        ).cuda()

        # With CP, sequence is split across ranks
        local_seq_q = seq_q // world_size
        local_seq_kv = seq_kv // world_size

        q = torch.randn(batch, local_seq_q, num_heads, head_dim,
                         dtype=torch.bfloat16, device="cuda", requires_grad=True)
        k = torch.randn(batch, local_seq_kv, num_gqa_groups, head_dim,
                         dtype=torch.bfloat16, device="cuda", requires_grad=True)
        v = torch.randn(batch, local_seq_kv, num_gqa_groups, head_dim,
                         dtype=torch.bfloat16, device="cuda", requires_grad=True)

        def run_step():
            q.grad = k.grad = v.grad = None
            out = attn(q, k, v)
            out.backward(torch.ones_like(out))

        result = distributed_measure(
            run_step,
            config=config,
            name="DotProductAttention(CP)",
            category="context_parallel",
            shape=size_cfg.shape,
            recipe="bf16",
            direction="fwd_bwd",
            regime=size_cfg.regime,
        )

        if result is not None:
            reporter.report(result)
