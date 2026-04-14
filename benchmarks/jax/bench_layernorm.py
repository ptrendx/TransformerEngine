# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmarks for te.jax.flax.LayerNorm."""

import pytest
import jax
import jax.numpy as jnp

from transformer_engine.jax.flax import LayerNorm

from benchmarks.config.sizes import NORMALIZATION_SIZES


def shape_id(size_config):
    return size_config.id


@pytest.mark.parametrize("size_cfg", NORMALIZATION_SIZES, ids=shape_id)
@pytest.mark.parametrize("direction", ["fwd_only", "fwd_bwd"])
def test_bench_layernorm(
    size_cfg, direction, benchmark_config, benchmark_timer, benchmark_reporter,
):
    if benchmark_config.size_filter != "all" and size_cfg.regime != benchmark_config.size_filter:
        pytest.skip(f"Skipping {size_cfg.regime} sizes")

    rows, hidden_size = size_cfg.shape

    model = LayerNorm(dtype=jnp.bfloat16)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (rows, hidden_size), dtype=jnp.bfloat16)
    variables = model.init(key, x)

    if direction == "fwd_only":
        @jax.jit
        def run_step(variables, x):
            return model.apply(variables, x)

        result = benchmark_timer.measure(
            run_step, args=(variables, x),
            label="LayerNorm",
            sub_label=f"{rows}x{hidden_size}_fwd",
            name="te.jax.LayerNorm",
            category="normalization",
            shape=size_cfg.shape,
            direction="fwd",
            regime=size_cfg.regime,
            total_bytes=rows * hidden_size * 2 * 2,  # read + write bf16
        )
    else:
        @jax.jit
        def run_step(variables, x):
            def loss_fn(variables, x):
                return jnp.sum(model.apply(variables, x))
            return jax.grad(loss_fn)(variables, x)

        result = benchmark_timer.measure(
            run_step, args=(variables, x),
            label="LayerNorm",
            sub_label=f"{rows}x{hidden_size}_fwd_bwd",
            name="te.jax.LayerNorm",
            category="normalization",
            shape=size_cfg.shape,
            direction="fwd_bwd",
            regime=size_cfg.regime,
            total_bytes=rows * hidden_size * 2 * 4,  # fwd+bwd read+write
        )

    benchmark_reporter.report(result)
