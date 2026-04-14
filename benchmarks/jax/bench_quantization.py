# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmarks for JAX quantization primitives."""

import pytest
import jax
import jax.numpy as jnp

from benchmarks.config.sizes import CAST_SIZES


def shape_id(size_config):
    return size_config.id


@pytest.mark.parametrize("size_cfg", CAST_SIZES, ids=shape_id)
def test_bench_quantize_fp8(
    size_cfg, benchmark_config, benchmark_timer, benchmark_reporter,
):
    """Benchmark FP8 quantization via the JAX quantize module."""
    if benchmark_config.size_filter != "all" and size_cfg.regime != benchmark_config.size_filter:
        pytest.skip(f"Skipping {size_cfg.regime} sizes")

    try:
        from transformer_engine.jax.quantize import quantize, dequantize
    except ImportError:
        pytest.skip("JAX quantize module not available")

    rows, cols = size_cfg.shape
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (rows, cols), dtype=jnp.bfloat16)

    @jax.jit
    def run_quantize(x):
        return quantize(x, q_dtype=jnp.float8_e4m3fn)

    result = benchmark_timer.measure(
        run_quantize, args=(x,),
        label="quantize_fp8",
        sub_label=f"{rows}x{cols}",
        name="te.jax.quantize(fp8)",
        category="quantization",
        shape=size_cfg.shape,
        recipe="fp8",
        direction="fwd",
        regime=size_cfg.regime,
        total_bytes=rows * cols * (2 + 1),  # bf16 input + fp8 output
    )
    benchmark_reporter.report(result)
