# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmarks for te.jax.flax.DotProductAttention."""

import pytest
import jax
import jax.numpy as jnp

from transformer_engine.jax.flax import DotProductAttention

from benchmarks.config.sizes import ATTENTION_SIZES


def shape_id(size_config):
    return size_config.id


@pytest.mark.parametrize("size_cfg", ATTENTION_SIZES, ids=shape_id)
@pytest.mark.parametrize("direction", ["fwd_only", "fwd_bwd"])
def test_bench_attention(
    size_cfg, direction, benchmark_config, benchmark_timer, benchmark_reporter,
):
    if benchmark_config.size_filter != "all" and size_cfg.regime != benchmark_config.size_filter:
        pytest.skip(f"Skipping {size_cfg.regime} sizes")

    batch, num_heads, num_gqa_groups, head_dim, seq_q, seq_kv = size_cfg.shape

    model = DotProductAttention(
        head_dim=head_dim,
        num_attention_heads=num_heads,
        num_gqa_groups=num_gqa_groups,
        attn_mask_type="causal",
        dtype=jnp.bfloat16,
    )
    key = jax.random.PRNGKey(0)

    # Q: (batch, seq_q, num_heads, head_dim)
    # K, V: (batch, seq_kv, num_gqa_groups, head_dim)
    q = jax.random.normal(key, (batch, seq_q, num_heads, head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(key, (batch, seq_kv, num_gqa_groups, head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(key, (batch, seq_kv, num_gqa_groups, head_dim), dtype=jnp.bfloat16)

    variables = model.init(key, q, k, v)

    if direction == "fwd_only":
        @jax.jit
        def run_step(variables, q, k, v):
            return model.apply(variables, q, k, v)

        result = benchmark_timer.measure(
            run_step, args=(variables, q, k, v),
            label="DotProductAttention",
            sub_label=f"b{batch}_h{num_heads}_d{head_dim}_s{seq_q}_fwd",
            name="te.jax.DotProductAttention",
            category="attention",
            shape=size_cfg.shape,
            direction="fwd",
            regime=size_cfg.regime,
        )
    else:
        @jax.jit
        def run_step(variables, q, k, v):
            def loss_fn(variables, q, k, v):
                return jnp.sum(model.apply(variables, q, k, v))
            return jax.grad(loss_fn)(variables, q, k, v)

        result = benchmark_timer.measure(
            run_step, args=(variables, q, k, v),
            label="DotProductAttention",
            sub_label=f"b{batch}_h{num_heads}_d{head_dim}_s{seq_q}_fwd_bwd",
            name="te.jax.DotProductAttention",
            category="attention",
            shape=size_cfg.shape,
            direction="fwd_bwd",
            regime=size_cfg.regime,
        )

    benchmark_reporter.report(result)
