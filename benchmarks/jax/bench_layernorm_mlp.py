# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmarks for te.jax.flax.LayerNormMLP."""

import pytest
import jax
import jax.numpy as jnp

from transformer_engine.jax.flax import LayerNormMLP

from benchmarks.config.sizes import LINEAR_SIZES


def shape_id(size_config):
    return size_config.id


@pytest.mark.parametrize("size_cfg", LINEAR_SIZES, ids=shape_id)
@pytest.mark.parametrize("direction", ["fwd_only", "fwd_bwd"])
def test_bench_layernorm_mlp(
    size_cfg, direction, benchmark_config, benchmark_timer, benchmark_reporter,
):
    if benchmark_config.size_filter != "all" and size_cfg.regime != benchmark_config.size_filter:
        pytest.skip(f"Skipping {size_cfg.regime} sizes")

    m, k, n = size_cfg.shape

    model = LayerNormMLP(
        intermediate_dim=n,
        activations=("gelu",),
        dtype=jnp.bfloat16,
    )
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (m, k), dtype=jnp.bfloat16)
    variables = model.init(key, x)

    if direction == "fwd_only":
        @jax.jit
        def run_step(variables, x):
            return model.apply(variables, x)

        result = benchmark_timer.measure(
            run_step, args=(variables, x),
            label="LayerNormMLP",
            sub_label=f"{m}x{k}x{n}_fwd",
            name="te.jax.LayerNormMLP",
            category="mlp",
            shape=size_cfg.shape,
            direction="fwd",
            regime=size_cfg.regime,
            total_flops=2 * m * k * n + 2 * m * n * k,  # two GEMMs
        )
    else:
        @jax.jit
        def run_step(variables, x):
            def loss_fn(variables, x):
                return jnp.sum(model.apply(variables, x))
            return jax.grad(loss_fn)(variables, x)

        result = benchmark_timer.measure(
            run_step, args=(variables, x),
            label="LayerNormMLP",
            sub_label=f"{m}x{k}x{n}_fwd_bwd",
            name="te.jax.LayerNormMLP",
            category="mlp",
            shape=size_cfg.shape,
            direction="fwd_bwd",
            regime=size_cfg.regime,
            total_flops=(2 * m * k * n + 2 * m * n * k) * 2,
        )

    benchmark_reporter.report(result)
