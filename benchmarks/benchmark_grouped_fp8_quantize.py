# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark grouped FP8 current-scaling quantization."""

import argparse
import json
import os
import statistics
from typing import List

import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import Float8CurrentScalingQuantizer
import transformer_engine_torch as tex


def _parse_first_dims(value: str) -> List[int]:
    dims = [int(item) for item in value.split(",") if item]
    if not dims:
        raise argparse.ArgumentTypeError("at least one first dimension is required")
    if any(dim < 0 for dim in dims):
        raise argparse.ArgumentTypeError("first dimensions must be non-negative")
    return dims


def _benchmark_once(
    grouped_input: torch.Tensor,
    quantizer: Float8CurrentScalingQuantizer,
    first_dims: torch.Tensor,
    warmup: int,
    iterations: int,
) -> List[float]:
    num_tensors = first_dims.numel()

    for _ in range(warmup):
        tex.group_quantize(grouped_input, quantizer, num_tensors, first_dims)
    torch.cuda.synchronize()

    timings_us = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        start.record()
        tex.group_quantize(grouped_input, quantizer, num_tensors, first_dims)
        end.record()
        end.synchronize()
        timings_us.append(start.elapsed_time(end) * 1000.0)
    return timings_us


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--first-dims",
        type=_parse_first_dims,
        default=_parse_first_dims("8192,4096,12288,6144,10240,8192,7168,9216"),
        help="Comma-separated per-tensor first dimensions.",
    )
    parser.add_argument("--hidden", type=int, default=8192, help="Common hidden dimension.")
    parser.add_argument("--padding-rows", type=int, default=4096, help="Unused tail rows.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--iterations", type=int, default=100, help="Measured iterations.")
    parser.add_argument(
        "--output",
        default=os.environ.get("ORCHESTRA_BENCHMARK_RAW_REPORT", "grouped_fp8_quantize.json"),
        help="Path for the JSON benchmark report.",
    )
    args = parser.parse_args()

    fp8_available, reason = te.is_fp8_available(return_reason=True)
    if not fp8_available:
        raise RuntimeError(f"FP8 is not available: {reason}")
    if args.hidden <= 0:
        raise ValueError("--hidden must be positive")
    if args.padding_rows < 0:
        raise ValueError("--padding-rows must be non-negative")

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    num_tensors = len(args.first_dims)
    actual_rows = sum(args.first_dims)
    allocated_rows = actual_rows + args.padding_rows
    input_dtype = torch.bfloat16

    actual_input = torch.randn((actual_rows, args.hidden), dtype=input_dtype, device="cuda")
    if args.padding_rows:
        sentinel = torch.full(
            (args.padding_rows, args.hidden),
            1.0e6,
            dtype=input_dtype,
            device="cuda",
        )
        grouped_input = torch.cat([actual_input, sentinel], dim=0)
    else:
        grouped_input = actual_input

    first_dims = torch.tensor(args.first_dims, dtype=torch.int64, device="cuda")
    quantizer = Float8CurrentScalingQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        device="cuda",
        rowwise=True,
        columnwise=False,
    )

    output = tex.group_quantize(grouped_input, quantizer, num_tensors, first_dims)
    expected_actual_elements = actual_rows * args.hidden
    if output.tensor_offsets[-1].item() != expected_actual_elements:
        raise RuntimeError("Grouped tensor offsets do not match actual element count")
    if args.padding_rows and torch.any(output.amax >= 1.0e6):
        raise RuntimeError("Sentinel tail values affected grouped FP8 amax")
    torch.cuda.synchronize()

    timings_us = _benchmark_once(
        grouped_input,
        quantizer,
        first_dims,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    median_us = statistics.median(timings_us)

    actual_elements = expected_actual_elements
    allocated_elements = allocated_rows * args.hidden
    tail_elements = allocated_elements - actual_elements
    actual_input_bytes = actual_elements * grouped_input.element_size()
    actual_output_bytes = actual_elements
    metadata_bytes = num_tensors * 3 * 4
    memory_traffic_bytes = 2 * actual_input_bytes + actual_output_bytes + metadata_bytes
    median_seconds = median_us / 1.0e6
    bandwidth_tbps = memory_traffic_bytes / median_seconds / 1.0e12

    report = {
        "benchmark": "grouped_fp8_current_scaling_quantize",
        "hardware_target": "GB200",
        "num_tensors": num_tensors,
        "first_dims": args.first_dims,
        "hidden": args.hidden,
        "padding_rows": args.padding_rows,
        "input_dtype": str(grouped_input.dtype),
        "fp8_dtype": "kFloat8E4M3",
        "actual_elements": actual_elements,
        "allocated_elements": allocated_elements,
        "tail_elements": tail_elements,
        "actual_input_bytes": actual_input_bytes,
        "actual_output_bytes": actual_output_bytes,
        "metadata_bytes": metadata_bytes,
        "memory_traffic_bytes": memory_traffic_bytes,
        "allocated_tail_bytes_excluded": True,
        "byte_accounting_note": (
            "memory_traffic_bytes counts two reads of actual input payload, one write of actual "
            "FP8 output payload, and per-tensor amax/scale/scale_inv metadata; allocated tail "
            "capacity is excluded."
        ),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "median_us": median_us,
        "min_us": min(timings_us),
        "max_us": max(timings_us),
        "bandwidth_TBps": bandwidth_tbps,
        "bandwidth_GBps": bandwidth_tbps * 1000.0,
    }

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
