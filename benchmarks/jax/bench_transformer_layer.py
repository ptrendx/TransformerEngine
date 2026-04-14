# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmarks for te.jax.flax.TransformerLayer."""

import pytest
import jax
import jax.numpy as jnp

from transformer_engine.jax.flax import TransformerLayer

from benchmarks.config.sizes import TRANSFORMER_LAYER_SIZES


def shape_id(size_config):
    return size_config.id


@pytest.mark.parametrize("size_cfg", TRANSFORMER_LAYER_SIZES, ids=shape_id)
@pytest.mark.parametrize("direction", ["fwd_only", "fwd_bwd"])
def test_bench_transformer_layer(
    size_cfg, direction, benchmark_config, benchmark_timer, benchmark_reporter,
):
    if benchmark_config.size_filter != "all" and size_cfg.regime != benchmark_config.size_filter:
        pytest.skip(f"Skipping {size_cfg.regime} sizes")

    batch, seq_len, hidden_size, num_heads, ffn_hidden_size = size_cfg.shape

    model = TransformerLayer(
        hidden_size=hidden_size,
        mlp_hidden_size=ffn_hidden_size,
        num_attention_heads=num_heads,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        dtype=jnp.bfloat16,
    )
    key = jax.random.PRNGKey(0)
    # Input: (batch, seq_len, hidden_size)
    x = jax.random.normal(key, (batch, seq_len, hidden_size), dtype=jnp.bfloat16)
    # Attention mask: (batch, 1, seq_len, seq_len)
    mask = jnp.ones((batch, 1, seq_len, seq_len), dtype=jnp.uint8)

    variables = model.init(key, x, attention_mask=mask)

    if direction == "fwd_only":
        @jax.jit
        def run_step(variables, x, mask):
            return model.apply(variables, x, attention_mask=mask)

        result = benchmark_timer.measure(
            run_step, args=(variables, x, mask),
            label="TransformerLayer",
            sub_label=f"b{batch}_s{seq_len}_h{hidden_size}_fwd",
            name="te.jax.TransformerLayer",
            category="transformer",
            shape=size_cfg.shape,
            direction="fwd",
            regime=size_cfg.regime,
        )
    else:
        @jax.jit
        def run_step(variables, x, mask):
            def loss_fn(variables, x, mask):
                return jnp.sum(model.apply(variables, x, attention_mask=mask))
            return jax.grad(loss_fn)(variables, x, mask)

        result = benchmark_timer.measure(
            run_step, args=(variables, x, mask),
            label="TransformerLayer",
            sub_label=f"b{batch}_s{seq_len}_h{hidden_size}_fwd_bwd",
            name="te.jax.TransformerLayer",
            category="transformer",
            shape=size_cfg.shape,
            direction="fwd_bwd",
            regime=size_cfg.regime,
        )

    benchmark_reporter.report(result)
