# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmarks for te.MultiheadAttention."""

from contextlib import nullcontext

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.quantization import autocast

from benchmarks.config.sizes import ATTENTION_SIZES
from benchmarks.pytorch.conftest import skip_if_recipe_unavailable, get_recipe

RECIPES = ["bf16", "fp8_block"]


def shape_id(size_config):
    return size_config.id


@pytest.mark.parametrize("size_cfg", ATTENTION_SIZES, ids=shape_id)
@pytest.mark.parametrize("recipe_name", RECIPES)
@pytest.mark.parametrize("direction", ["fwd_only", "fwd_bwd"])
def test_bench_multi_head_attention(
    size_cfg, recipe_name, direction, benchmark_config, benchmark_timer, benchmark_reporter
):
    skip_if_recipe_unavailable(recipe_name)

    batch, num_heads, num_gqa_groups, head_dim, seq_q, seq_kv = size_cfg.shape
    hidden_size = num_heads * head_dim
    needs_grad = direction == "fwd_bwd"

    mha = te.MultiheadAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_gqa_groups=num_gqa_groups,
        attention_dropout=0.0,
        bias=False,
        attn_mask_type="causal",
        params_dtype=torch.bfloat16,
        input_layernorm=False,
    ).cuda()

    # Input is (seq_len, batch, hidden_size) for SBHD layout
    x = torch.randn(
        seq_q, batch, hidden_size,
        dtype=torch.bfloat16, device="cuda", requires_grad=needs_grad,
    )

    recipe = get_recipe(recipe_name)

    def run_step():
        ctx = autocast(enabled=True, recipe=recipe) if recipe is not None else nullcontext()
        with ctx:
            if direction == "fwd_only":
                with torch.no_grad():
                    mha(x)
            else:
                x.grad = None
                out = mha(x)
                out.backward(torch.ones_like(out))

    result = benchmark_timer.measure(
        stmt="run_step()",
        globals_dict={"run_step": run_step},
        label="MultiheadAttention",
        sub_label=f"b{batch}_h{num_heads}_g{num_gqa_groups}_d{head_dim}_sq{seq_q}_skv{seq_kv}",
        name="MultiheadAttention",
        category="attention",
        shape=size_cfg.shape,
        dtype="bf16",
        recipe=recipe_name,
        direction=direction,
        regime=size_cfg.regime,
    )
    benchmark_reporter.report(result)
