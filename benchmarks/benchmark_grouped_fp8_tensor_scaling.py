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

import transformer_engine  # pylint: disable=unused-import
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import Float8CurrentScalingQuantizer


GB200_PEAK_BANDWIDTH_TBPS = 8.0


def _parse_first_dims(value: str, num_tensors: int) -> List[int]:
    if value:
        first_dims = [int(dim) for dim in value.split(",")]
    else:
        first_dims = [12288] * num_tensors
    if len(first_dims) != num_tensors:
        raise ValueError(f"Expected {num_tensors} first dims, got {len(first_dims)}")
    if any(dim < 0 for dim in first_dims):
        raise ValueError("first dims must be non-negative")
    return first_dims


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _make_input(first_dims: List[int], hidden_dim: int, tail_rows: int, dtype: torch.dtype):
    actual_rows = sum(first_dims)
    allocated_rows = actual_rows + tail_rows
    grouped_input = torch.empty(allocated_rows, hidden_dim, dtype=dtype, device="cuda")
    if actual_rows > 0:
        grouped_input[:actual_rows].normal_(mean=0.0, std=0.5)
    if tail_rows > 0:
        grouped_input[actual_rows:].fill_(1000)
    first_dims_tensor = torch.tensor(first_dims, dtype=torch.int64, device="cuda")
    return grouped_input, first_dims_tensor


def _run_once(grouped_input, quantizer, num_tensors, first_dims_tensor):
    return tex.group_quantize(grouped_input, quantizer, num_tensors, first_dims_tensor)


def _make_runner(args, grouped_input, quantizer, first_dims_tensor):
    if not args.cuda_graph:
        return lambda: _run_once(grouped_input, quantizer, args.num_tensors, first_dims_tensor)

    graph_output = {}
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        graph_output["output"] = _run_once(
            grouped_input, quantizer, args.num_tensors, first_dims_tensor
        )

    def _replay():
        graph.replay()
        return graph_output["output"]

    return _replay


def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    fp8_available, reason = te.is_fp8_available(return_reason=True)
    if not fp8_available:
        raise RuntimeError(f"FP8 is not available: {reason}")

    torch.manual_seed(args.seed)
    dtype = _dtype_from_name(args.dtype)
    first_dims = _parse_first_dims(args.first_dims, args.num_tensors)
    grouped_input, first_dims_tensor = _make_input(
        first_dims, args.hidden_dim, args.tail_rows, dtype
    )
    quantizer = Float8CurrentScalingQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        device="cuda",
        rowwise=True,
        columnwise=False,
        force_pow_2_scales=args.force_pow_2_scales,
        amax_epsilon=args.amax_epsilon,
    )

    for _ in range(args.warmup_iters):
        _run_once(grouped_input, quantizer, args.num_tensors, first_dims_tensor)
    torch.cuda.synchronize()

    run_measured = _make_runner(args, grouped_input, quantizer, first_dims_tensor)
    torch.cuda.synchronize()

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()

    times_us = []
    for _ in range(args.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            output = run_measured()
        end.record()
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0 / args.iters)

    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()

    actual_elements = sum(first_dims) * args.hidden_dim
    allocated_elements = grouped_input.numel()
    tail_elements = allocated_elements - actual_elements
    input_element_size = torch.empty((), dtype=dtype, device="cuda").element_size()
    fp8_element_size = 1
    metadata_bytes = args.num_tensors * 3 * 4
    processed_bytes = actual_elements * (2 * input_element_size + fp8_element_size) + metadata_bytes
    median_us = statistics.median(times_us)
    bandwidth_tbps_actual_bytes = processed_bytes / (median_us * 1.0e-6) / 1.0e12
    peak_fraction = bandwidth_tbps_actual_bytes / GB200_PEAK_BANDWIDTH_TBPS

    output_path = args.output or os.environ.get(
        "ORCHESTRA_BENCHMARK_RAW_REPORT", "grouped_fp8_tensor_scaling_report.json"
    )
    report = {
        "benchmark": "grouped_fp8_current_scaling_quantize",
        "num_tensors": args.num_tensors,
        "first_dims": first_dims,
        "hidden_dim": args.hidden_dim,
        "tail_rows": args.tail_rows,
        "dtype": args.dtype,
        "fp8_dtype": "kFloat8E4M3",
        "warmup_iters": args.warmup_iters,
        "iters": args.iters,
        "repeats": args.repeats,
        "cuda_graph_replay": args.cuda_graph,
        "actual_elements": actual_elements,
        "allocated_elements": allocated_elements,
        "tail_elements": tail_elements,
        "processed_bytes": processed_bytes,
        "byte_formula": (
            "actual_elements * (2 * input_element_size + fp8_element_size) "
            "+ num_tensors * 3 * sizeof(float)"
        ),
        "median_us": median_us,
        "times_us": times_us,
        "bandwidth_TBps_actual_bytes": bandwidth_tbps_actual_bytes,
        "gb200_peak_bandwidth_TBps": GB200_PEAK_BANDWIDTH_TBPS,
        "peak_fraction": peak_fraction,
        "scale_inv_sample": output.scale_inv[: min(args.num_tensors, 8)].float().tolist(),
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(
        "actual_elements={actual_elements} allocated_elements={allocated_elements} "
        "tail_elements={tail_elements} processed_bytes={processed_bytes} "
        "median_us={median_us:.3f} "
        "bandwidth_TBps_actual_bytes={bandwidth_TBps_actual_bytes:.3f} "
        "peak_fraction={peak_fraction:.3f}".format(**report)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tensors", type=int, default=8)
    parser.add_argument(
        "--first-dims",
        type=str,
        default="15296,8960,14656,14784,11712,7936,14080,10880",
        help="Comma-separated first dimensions for each grouped tensor.",
    )
    parser.add_argument("--hidden-dim", type=int, default=7168)
    parser.add_argument("--tail-rows", type=int, default=4096)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force-pow-2-scales", action="store_true")
    parser.add_argument("--amax-epsilon", type=float, default=0.0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--no-cuda-graph",
        action="store_false",
        dest="cuda_graph",
        help="Measure eager public API calls instead of CUDA graph replay.",
    )
    parser.add_argument("--output", type=str, nargs="?", default=None)
    parser.set_defaults(cuda_graph=True)
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
