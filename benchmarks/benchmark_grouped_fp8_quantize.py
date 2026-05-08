#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark grouped FP8 tensor-scaling quantize."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch

import transformer_engine  # noqa: F401
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tensors", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("ORCHESTRA_BENCHMARK_RAW_REPORT", "grouped_fp8_quantize.json")),
    )
    return parser.parse_args()


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Grouped FP8 quantize benchmark requires CUDA.")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    first_dims = torch.full((args.num_tensors,), args.rows, dtype=torch.int64, device=device)
    logical_rows = args.rows * args.num_tensors
    logical_elements = logical_rows * args.cols

    x = torch.randn((logical_rows, args.cols), dtype=dtype, device=device)
    scales = torch.full((args.num_tensors,), 1.0, dtype=torch.float32, device=device)
    amax = torch.zeros((args.num_tensors,), dtype=torch.float32, device=device)
    quantizer = Float8Quantizer(scales, amax, tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)

    for _ in range(args.warmup):
        tex.group_quantize(x, quantizer, args.num_tensors, first_dims)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = None
    for _ in range(args.iters):
        out = tex.group_quantize(x, quantizer, args.num_tensors, first_dims)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / args.iters
    logical_input_bytes = logical_elements * _dtype_nbytes(dtype)
    logical_output_bytes = logical_elements
    logical_scale_bytes = args.num_tensors * _dtype_nbytes(torch.float32) * 2
    logical_total_bytes = logical_input_bytes + logical_output_bytes + logical_scale_bytes
    allocation_input_elements = x.numel()
    allocation_output_elements = out.rowwise_data.numel()
    allocation_input_bytes = allocation_input_elements * _dtype_nbytes(dtype)
    allocation_output_bytes = allocation_output_elements
    bandwidth_gbps = logical_total_bytes / (elapsed_ms / 1.0e3) / 1.0e9

    report: Dict[str, Any] = {
        "schema_version": "grouped_fp8_quantize_benchmark/v1",
        "benchmark": "grouped_fp8_tensor_scaling_quantize",
        "num_tensors": args.num_tensors,
        "rows_per_tensor": args.rows,
        "cols": args.cols,
        "logical_elements": logical_elements,
        "allocation_input_elements": allocation_input_elements,
        "allocation_output_elements": allocation_output_elements,
        "logical_input_bytes": logical_input_bytes,
        "logical_output_bytes": logical_output_bytes,
        "logical_scale_bytes": logical_scale_bytes,
        "logical_total_bytes": logical_total_bytes,
        "allocation_input_bytes": allocation_input_bytes,
        "allocation_output_bytes": allocation_output_bytes,
        "elapsed_ms": elapsed_ms,
        "bandwidth_GBps_actual_bytes": bandwidth_gbps,
        "dtype": str(dtype),
        "fp8_dtype": "kFloat8E4M3",
        "rowwise": True,
        "columnwise": False,
        "cuda_device": torch.cuda.get_device_name(device),
        "warmup": args.warmup,
        "iters": args.iters,
        "notes": [
            "Bandwidth denominator uses logical bytes.",
            (
                "The Python grouped quantize API allocates output at logical size; allocation "
                "fields are included so capacity-backed benchmark variants can be audited with "
                "the same schema."
            ),
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
