# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
import json
import os
from pathlib import Path

import torch
import torch.utils.benchmark as benchmark

import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer


def run_case(num_tensors, cols, first_dims, allocation_rows, min_run_time):
    device = torch.device("cuda")
    actual_rows = sum(first_dims)
    actual_elements = actual_rows * cols
    allocation_elements = allocation_rows * cols
    if allocation_elements < actual_elements:
        raise ValueError("allocation_rows must provide at least the logical grouped total")

    x = torch.randn((allocation_rows, cols), dtype=torch.bfloat16, device=device)
    first_dims_tensor = torch.tensor(first_dims, dtype=torch.int64, device=device)
    scale = torch.ones((1,), dtype=torch.float32, device=device)
    amax = torch.zeros((num_tensors,), dtype=torch.float32, device=device)
    quantizer = Float8Quantizer(
        scale=scale,
        amax=amax,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )

    def kernel():
        return tex.group_quantize(x, quantizer, num_tensors, first_dims_tensor)

    kernel()
    torch.cuda.synchronize()
    timing = benchmark.Timer(stmt="kernel()", globals={"kernel": kernel}, num_threads=1)
    result = timing.blocked_autorange(min_run_time=min_run_time)
    torch.cuda.synchronize()

    time_s = result.median
    actual_input_bytes = actual_elements * x.element_size()
    actual_output_bytes = actual_elements
    actual_scale_bytes = num_tensors * 3 * torch.tensor([], dtype=torch.float32).element_size()
    actual_total_bytes = actual_input_bytes + actual_output_bytes + actual_scale_bytes
    allocation_input_bytes = allocation_elements * x.element_size()
    allocation_output_bytes = allocation_elements
    allocation_total_bytes = allocation_input_bytes + allocation_output_bytes + actual_scale_bytes

    return {
        "kernel": "tex.group_quantize_fp8_tensor_scaling",
        "num_tensors": num_tensors,
        "cols": cols,
        "first_dims": first_dims,
        "actual_total_elements": actual_elements,
        "allocation_elements": allocation_elements,
        "actual_input_bytes": actual_input_bytes,
        "actual_output_bytes": actual_output_bytes,
        "actual_scale_amax_scale_inv_bytes": actual_scale_bytes,
        "actual_total_bytes": actual_total_bytes,
        "allocation_input_bytes": allocation_input_bytes,
        "allocation_output_bytes": allocation_output_bytes,
        "allocation_total_bytes": allocation_total_bytes,
        "time_us": time_s * 1.0e6,
        "bandwidth_GBps_actual_bytes": actual_total_bytes / time_s / 1.0e9,
        "bandwidth_GBps_allocation_bytes_for_audit_only": allocation_total_bytes / time_s / 1.0e9,
        "min_run_time_s": min_run_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.environ.get("ORCHESTRA_BENCHMARK_RAW_REPORT"))
    parser.add_argument("--min-run-time", type=float, default=2.0)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--first-dims", default="512,1024,256,768")
    parser.add_argument("--allocation-extra-rows", type=int, default=512)
    args = parser.parse_args()

    if args.output is None:
        raise ValueError("--output or ORCHESTRA_BENCHMARK_RAW_REPORT is required")

    first_dims = [int(x) for x in args.first_dims.split(",") if x]
    allocation_rows = sum(first_dims) + args.allocation_extra_rows
    record = run_case(
        num_tensors=len(first_dims),
        cols=args.cols,
        first_dims=first_dims,
        allocation_rows=allocation_rows,
        min_run_time=args.min_run_time,
    )

    payload = {
        "benchmark": "grouped_fp8_tensor_scaling_quantize",
        "records": [record],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
