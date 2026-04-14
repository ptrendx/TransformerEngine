# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmarks for te.DotProductAttention."""

import pytest
import torch
import transformer_engine.pytorch as te

from benchmarks.config.sizes import ATTENTION_SIZES


def shape_id(size_config):
    return size_config.id


@pytest.mark.parametrize("size_cfg", ATTENTION_SIZES, ids=shape_id)
@pytest.mark.parametrize("direction", ["fwd_only", "fwd_bwd"])
def test_bench_attention(
    size_cfg, direction, benchmark_config, benchmark_timer, benchmark_reporter
):
    batch, num_heads, num_gqa_groups, head_dim, seq_q, seq_kv = size_cfg.shape
    needs_grad = direction == "fwd_bwd"

    attn = te.DotProductAttention(
        num_attention_heads=num_heads,
        kv_channels=head_dim,
        num_gqa_groups=num_gqa_groups,
        attention_dropout=0.0,
        attn_mask_type="causal",
    ).cuda()

    q = torch.randn(
        batch, seq_q, num_heads, head_dim,
        dtype=torch.bfloat16, device="cuda", requires_grad=needs_grad,
    )
    k = torch.randn(
        batch, seq_kv, num_gqa_groups, head_dim,
        dtype=torch.bfloat16, device="cuda", requires_grad=needs_grad,
    )
    v = torch.randn(
        batch, seq_kv, num_gqa_groups, head_dim,
        dtype=torch.bfloat16, device="cuda", requires_grad=needs_grad,
    )

    def run_step():
        if direction == "fwd_only":
            with torch.no_grad():
                attn(q, k, v)
        else:
            q.grad = k.grad = v.grad = None
            out = attn(q, k, v)
            out.backward(torch.ones_like(out))

    result = benchmark_timer.measure(
        stmt="run_step()",
        globals_dict={"run_step": run_step},
        label="DotProductAttention",
        sub_label=f"b{batch}_h{num_heads}_g{num_gqa_groups}_d{head_dim}_sq{seq_q}_skv{seq_kv}",
        name="DotProductAttention",
        category="attention",
        shape=size_cfg.shape,
        dtype="bf16",
        recipe="bf16",
        direction=direction,
        regime=size_cfg.regime,
    )
    benchmark_reporter.report(result)
