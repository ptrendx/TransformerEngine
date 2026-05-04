# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark grouped FP8 tensor-scaling quantize.

The reported bandwidth is computed from the actual logical tensor sizes in the
group. Allocation/capacity sizes are emitted separately so benchmark analysis can
audit that padding is excluded from throughput.
"""

import argparse
import json
import os
from typing import List

import torch
import torch.utils.benchmark as benchmark

import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer


def _parse_first_dims(value: str) -> List[int]:
    dims = [int(x) for x in value.split(",") if x]
    if not dims:
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    if any(dim < 0 for dim in dims):
        raise argparse.ArgumentTypeError("first dimensions must be non-negative")
    return dims


def _make_quantizer(fp8_dtype: tex.DType) -> Float8Quantizer:
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    amax = torch.zeros(1, dtype=torch.float32, device="cuda")
    quantizer = Float8Quantizer(scale=scale, amax=amax, fp8_dtype=fp8_dtype)
    quantizer.set_usage(rowwise=True, columnwise=True)
    return quantizer


def run_case(args: argparse.Namespace) -> dict:
    first_dims = _parse_first_dims(args.first_dims)
    num_tensors = len(first_dims)
    actual_first_dim = sum(first_dims)
    allocation_first_dim = actual_first_dim + args.padding_rows
    actual_total_elements = actual_first_dim * args.last_dim
    allocation_elements = allocation_first_dim * args.last_dim

    fp8_dtype = getattr(tex.DType, args.fp8_dtype)
    input_dtype = getattr(torch, args.input_dtype)
    input_tensor = torch.randn(
        allocation_first_dim,
        args.last_dim,
        dtype=input_dtype,
        device="cuda",
    )
    if args.padding_rows > 0:
        input_tensor[actual_first_dim:, :].fill_(1234.0)

    first_dims_tensor = torch.tensor(first_dims, dtype=torch.int64, device="cuda")
    quantizer = _make_quantizer(fp8_dtype)

    def quantize_once():
        return tex.group_quantize(input_tensor, quantizer, num_tensors, first_dims_tensor)

    for _ in range(args.warmup):
        quantize_once()
    torch.cuda.synchronize()

    timer = benchmark.Timer(
        stmt="quantize_once()",
        globals={"quantize_once": quantize_once},
        num_threads=1,
    )
    timing = timer.blocked_autorange(min_run_time=args.min_run_time)
    torch.cuda.synchronize()

    input_bytes_per_element = torch.empty((), dtype=input_dtype).element_size()
    fp8_bytes_per_element = 1
    metadata_bytes = num_tensors * 3 * 4
    actual_input_bytes = actual_total_elements * input_bytes_per_element
    actual_output_bytes = actual_total_elements * fp8_bytes_per_element * 2
    allocation_input_bytes = allocation_elements * input_bytes_per_element
    allocation_output_bytes = allocation_elements * fp8_bytes_per_element * 2
    actual_total_bytes = actual_input_bytes + actual_output_bytes + metadata_bytes
    median_seconds = timing.median
    bandwidth_gbps = actual_total_bytes / median_seconds / 1.0e9

    result = {
        "benchmark": "grouped_fp8_tensor_scaling_quantize",
        "num_tensors": num_tensors,
        "first_dims": first_dims,
        "logical_last_dim": args.last_dim,
        "padding_rows": args.padding_rows,
        "actual_total_elements": actual_total_elements,
        "allocation_elements": allocation_elements,
        "actual_input_bytes": actual_input_bytes,
        "actual_output_bytes": actual_output_bytes,
        "actual_metadata_bytes": metadata_bytes,
        "actual_total_bytes": actual_total_bytes,
        "allocation_input_bytes": allocation_input_bytes,
        "allocation_output_bytes": allocation_output_bytes,
        "allocation_total_data_bytes": allocation_input_bytes + allocation_output_bytes,
        "input_dtype": args.input_dtype,
        "fp8_dtype": args.fp8_dtype,
        "scaling_mode": "NVTE_DELAYED_TENSOR_SCALING",
        "rowwise": True,
        "columnwise": True,
        "median_seconds": median_seconds,
        "median_us": median_seconds * 1.0e6,
        "bandwidth_gbps": bandwidth_gbps,
        "bandwidth_formula": (
            "(actual_input_bytes + actual_output_bytes + actual_metadata_bytes) / median_seconds"
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-dims", default="4096,3072,2048,1024", type=str)
    parser.add_argument("--last-dim", default=4096, type=int)
    parser.add_argument("--padding-rows", default=1024, type=int)
    parser.add_argument("--input-dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--fp8-dtype", default="kFloat8E4M3", choices=["kFloat8E4M3", "kFloat8E5M2"])
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--min-run-time", default=2.0, type=float)
    parser.add_argument("--output", default=os.environ.get("ORCHESTRA_BENCHMARK_RAW_REPORT"))
    args = parser.parse_args()

    if args.last_dim <= 0:
        raise ValueError("--last-dim must be positive")
    if args.padding_rows < 0:
        raise ValueError("--padding-rows must be non-negative")

    result = run_case(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")


if __name__ == "__main__":
    main()
