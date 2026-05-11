#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark grouped FP8 current-scaling quantization."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import List

import torch

from transformer_engine.pytorch import Float8CurrentScalingQuantizer
import transformer_engine_torch as tex


def _parse_first_dims(value: str) -> List[int]:
    dims = [int(dim) for dim in value.split(",") if dim]
    if not dims or any(dim < 0 for dim in dims):
        raise argparse.ArgumentTypeError("first dims must be a comma-separated list of >=0 ints")
    return dims


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tensors", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument(
        "--first-dims",
        type=_parse_first_dims,
        default=_parse_first_dims("16384,12288,8192,20480,4096,24576,6144,10240"),
    )
    parser.add_argument("--allocated-first-dim", type=int, default=114688)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--require-min-tbps", type=float, default=6.0)
    parser.add_argument("--peak-bandwidth-tbps", type=float, default=8.0)
    parser.add_argument("--tail-sentinel", type=float, default=4096.0)
    parser.add_argument("--columnwise", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ["ORCHESTRA_BENCHMARK_RAW_REPORT"])
        if "ORCHESTRA_BENCHMARK_RAW_REPORT" in os.environ
        else None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_tensors != len(args.first_dims):
        raise ValueError("--num-tensors must match the number of --first-dims entries")
    if args.hidden_size <= 0:
        raise ValueError("--hidden-size must be positive")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    actual_first_dim = sum(args.first_dims)
    if args.allocated_first_dim < actual_first_dim:
        raise ValueError("--allocated-first-dim must be >= sum(first_dims)")

    device = torch.device("cuda")
    input_dtype = torch.bfloat16
    output_element_size = 1
    output_count = 1 + int(args.columnwise)
    input_element_size = _dtype_nbytes(input_dtype)

    torch.manual_seed(1234)
    x = torch.empty(
        (args.allocated_first_dim, args.hidden_size), dtype=input_dtype, device=device
    )
    x[:actual_first_dim].normal_(mean=0.0, std=0.5)
    if args.allocated_first_dim > actual_first_dim:
        x[actual_first_dim:].fill_(args.tail_sentinel)

    first_dims = torch.tensor(args.first_dims, dtype=torch.int64, device=device)
    quantizer = Float8CurrentScalingQuantizer(
        tex.DType.kFloat8E4M3,
        device=device,
        rowwise=True,
        columnwise=args.columnwise,
    )

    for _ in range(args.warmup):
        tex.group_quantize(x, quantizer, args.num_tensors, first_dims)
    torch.cuda.synchronize()

    times_sec = []
    for _ in range(args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = tex.group_quantize(x, quantizer, args.num_tensors, first_dims)
        end.record()
        end.synchronize()
        times_sec.append(start.elapsed_time(end) / 1000.0)

    torch.cuda.synchronize()

    actual_numel = actual_first_dim * args.hidden_size
    allocated_numel = args.allocated_first_dim * args.hidden_size
    tail_numel = allocated_numel - actual_numel
    logical_payload_bytes = actual_numel * (
        input_element_size + output_count * output_element_size
    )
    estimated_relevant_traffic_bytes = actual_numel * (
        2 * input_element_size + output_count * output_element_size
    )

    median_sec = statistics.median(times_sec)
    mean_sec = statistics.fmean(times_sec)
    min_sec = min(times_sec)
    bandwidth_tbps = estimated_relevant_traffic_bytes / median_sec / 1.0e12
    peak_fraction = bandwidth_tbps / args.peak_bandwidth_tbps

    result = {
        "benchmark": "grouped_fp8_current_scaling",
        "device_name": torch.cuda.get_device_name(device),
        "num_tensors": args.num_tensors,
        "hidden_size": args.hidden_size,
        "first_dims": args.first_dims,
        "actual_first_dim": actual_first_dim,
        "allocated_first_dim": args.allocated_first_dim,
        "actual_numel": actual_numel,
        "allocated_numel": allocated_numel,
        "tail_numel": tail_numel,
        "tail_excluded_from_byte_accounting": True,
        "tail_sentinel": args.tail_sentinel,
        "rowwise_output": True,
        "columnwise_output": args.columnwise,
        "input_dtype": str(input_dtype),
        "fp8_dtype": "kFloat8E4M3",
        "input_element_size": input_element_size,
        "output_element_size_per_present_output": output_element_size,
        "logical_payload_bytes": logical_payload_bytes,
        "estimated_relevant_traffic_bytes": estimated_relevant_traffic_bytes,
        "elapsed_sec_median": median_sec,
        "elapsed_sec_mean": mean_sec,
        "elapsed_sec_min": min_sec,
        "bandwidth_TBps_actual_bytes": bandwidth_tbps,
        "gb200_peak_bandwidth_TBps_expectation": args.peak_bandwidth_tbps,
        "fraction_of_gb200_peak_bandwidth": peak_fraction,
        "required_min_TBps": args.require_min_tbps,
        "meets_required_min_TBps": bandwidth_tbps >= args.require_min_tbps,
        "output_rowwise_numel": output.rowwise_data.numel(),
        "output_tensor_offsets_last": int(output.tensor_offsets[-1].item()),
    }

    report = json.dumps(result, indent=2, sort_keys=True)
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
