# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmarks for te.TransformerLayer."""

from contextlib import nullcontext

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.quantization import autocast

from benchmarks.config.sizes import TRANSFORMER_LAYER_SIZES
from benchmarks.pytorch.conftest import skip_if_recipe_unavailable, get_recipe

RECIPES = ["bf16", "fp8_block", "mxfp8", "nvfp4"]


def shape_id(size_config):
    return size_config.id


@pytest.mark.parametrize("size_cfg", TRANSFORMER_LAYER_SIZES, ids=shape_id)
@pytest.mark.parametrize("recipe_name", RECIPES)
@pytest.mark.parametrize("direction", ["fwd_only", "fwd_bwd"])
def test_bench_transformer_layer(
    size_cfg, recipe_name, direction, benchmark_config, benchmark_timer, benchmark_reporter
):
    skip_if_recipe_unavailable(recipe_name)

    batch, seq_len, hidden_size, num_heads, ffn_hidden_size = size_cfg.shape
    needs_grad = direction == "fwd_bwd"

    layer = te.TransformerLayer(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_heads,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bias=False,
        attn_mask_type="causal",
        params_dtype=torch.bfloat16,
    ).cuda()

    # Input: (seq_len, batch, hidden_size)
    x = torch.randn(
        seq_len, batch, hidden_size,
        dtype=torch.bfloat16, device="cuda", requires_grad=needs_grad,
    )

    recipe = get_recipe(recipe_name)

    def run_step():
        ctx = autocast(enabled=True, recipe=recipe) if recipe is not None else nullcontext()
        with ctx:
            if direction == "fwd_only":
                with torch.no_grad():
                    layer(x)
            else:
                x.grad = None
                out = layer(x)
                out.backward(torch.ones_like(out))

    result = benchmark_timer.measure(
        stmt="run_step()",
        globals_dict={"run_step": run_step},
        label="TransformerLayer",
        sub_label=f"b{batch}_s{seq_len}_h{hidden_size}_nh{num_heads}_ffn{ffn_hidden_size}",
        name="TransformerLayer",
        category="module",
        shape=size_cfg.shape,
        dtype="bf16",
        recipe=recipe_name,
        direction=direction,
        regime=size_cfg.regime,
    )
    benchmark_reporter.report(result)
