# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark grouped FP8 tensor-scaling quantize."""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import Float8CurrentScalingQuantizer, Float8Quantizer
import transformer_engine_torch as tex


FP8_METADATA_TENSORS = ("scale", "amax", "scale_inv")


@dataclass(frozen=True)
class BenchmarkCase:
    """One grouped quantize shape case."""

    name: str
    splits: List[int]
    cols: int
    allocation_rows: int

    @property
    def num_tensors(self) -> int:
        return len(self.splits)

    @property
    def actual_rows(self) -> int:
        return sum(self.splits)

    @property
    def actual_total_elements(self) -> int:
        return self.actual_rows * self.cols

    @property
    def logical_shapes(self) -> List[List[int]]:
        return [[rows, self.cols] for rows in self.splits]

    @property
    def has_capacity_slack(self) -> bool:
        return self.allocation_rows > self.actual_rows


def parse_csv_ints(value: str) -> List[int]:
    """Parse comma-separated positive integers."""
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def parse_csv_strings(value: str, choices: Iterable[str]) -> List[str]:
    """Parse comma-separated strings and validate them against choices."""
    valid = set(choices)
    values = [part.strip() for part in value.split(",") if part.strip()]
    invalid = sorted(set(values) - valid)
    if invalid:
        raise argparse.ArgumentTypeError(
            f"invalid values {invalid}; expected comma-separated values from {sorted(valid)}"
        )
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def make_cases(args: argparse.Namespace) -> Dict[str, BenchmarkCase]:
    """Build balanced, unbalanced, and capacity-slack grouped cases."""
    balanced_splits = [args.rows_per_tensor] * args.num_tensors

    if args.unbalanced_splits is None:
        large = args.rows_per_tensor * 2
        small = max(args.rows_per_tensor // 2, 1)
        unbalanced_splits = [large, args.rows_per_tensor]
        unbalanced_splits.extend([small] * (args.num_tensors - 2))
        if len(unbalanced_splits) < args.num_tensors:
            unbalanced_splits.extend([args.rows_per_tensor] * (args.num_tensors - 2))
    else:
        unbalanced_splits = args.unbalanced_splits
        if len(unbalanced_splits) != args.num_tensors:
            raise ValueError(
                f"--unbalanced-splits must contain {args.num_tensors} entries, "
                f"got {len(unbalanced_splits)}"
            )

    unbalanced_rows = sum(unbalanced_splits)
    slack_rows = max(args.slack_rows, int(unbalanced_rows * args.slack_fraction))
    if slack_rows <= 0:
        slack_rows = max(args.rows_per_tensor, 1)

    return {
        "balanced": BenchmarkCase(
            name="balanced",
            splits=balanced_splits,
            cols=args.cols,
            allocation_rows=sum(balanced_splits),
        ),
        "unbalanced": BenchmarkCase(
            name="unbalanced",
            splits=unbalanced_splits,
            cols=args.cols,
            allocation_rows=unbalanced_rows,
        ),
        "capacity_slack": BenchmarkCase(
            name="capacity_slack",
            splits=unbalanced_splits,
            cols=args.cols,
            allocation_rows=unbalanced_rows + slack_rows,
        ),
    }


def make_quantizer(mode: str, num_tensors: int, device: torch.device):
    """Create a rowwise FP8 tensor-scaling quantizer."""
    if mode == "delayed":
        return Float8Quantizer(
            scale=torch.ones(num_tensors, dtype=torch.float32, device=device),
            amax=torch.zeros(num_tensors, dtype=torch.float32, device=device),
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
        )
    if mode == "current":
        return Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device=device,
            rowwise=True,
            columnwise=False,
            force_pow_2_scales=True,
            amax_epsilon=0.0,
        )
    raise ValueError(f"Unknown mode {mode!r}")


def make_input(case: BenchmarkCase, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Create logical input as a contiguous view over a possibly larger allocation."""
    backing = torch.empty((case.allocation_rows, case.cols), dtype=dtype, device=device)
    backing.uniform_(-1.0, 1.0)
    if case.has_capacity_slack:
        backing[case.actual_rows :, :].fill_(1024.0)
    logical_input = backing[: case.actual_rows, :]
    if not logical_input.is_contiguous():
        raise RuntimeError("logical grouped input view must be contiguous")
    return logical_input


def first_dims_tensor(case: BenchmarkCase, device: torch.device) -> Optional[torch.Tensor]:
    """Return first-dimension metadata only when the grouped shapes differ."""
    if all(split == case.splits[0] for split in case.splits):
        return None
    return torch.tensor(case.splits, dtype=torch.int64, device=device)


def storage_nbytes(tensor: torch.Tensor) -> int:
    """Return backing storage size in bytes."""
    return tensor.untyped_storage().nbytes()


def metadata_nbytes(output) -> int:
    """Return FP8 grouped tensor-scaling metadata bytes."""
    total = 0
    for name in FP8_METADATA_TENSORS:
        tensor = getattr(output, name, None)
        if tensor is not None:
            total += tensor.numel() * tensor.element_size()
    return total


def run_group_quantize(
    input_tensor: torch.Tensor,
    quantizer,
    case: BenchmarkCase,
    first_dims: Optional[torch.Tensor],
):
    """Dispatch through the PyTorch extension grouped quantize entry point."""
    return tex.group_quantize(input_tensor, quantizer, case.num_tensors, first_dims)


def time_case(
    input_tensor: torch.Tensor,
    quantizer,
    case: BenchmarkCase,
    first_dims: Optional[torch.Tensor],
    warmup: int,
    iterations: int,
):
    """Measure average CUDA stream elapsed time for grouped quantize."""
    output = None
    for _ in range(warmup):
        output = run_group_quantize(input_tensor, quantizer, case, first_dims)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        output = run_group_quantize(input_tensor, quantizer, case, first_dims)
    end.record()
    end.synchronize()

    latency_ms = start.elapsed_time(end) / iterations
    return latency_ms, output


def validate_capacity_slack(case: BenchmarkCase, output) -> None:
    """Check that slack sentinels did not contribute to per-tensor amax."""
    if not case.has_capacity_slack:
        return
    max_amax = float(output.amax.max().item())
    if max_amax > 16.0:
        raise RuntimeError(
            f"capacity slack appears to affect amax for {case.name}: max amax {max_amax}"
        )


def benchmark_one(
    mode: str,
    case: BenchmarkCase,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    """Run and report one grouped FP8 quantize benchmark case."""
    input_tensor = make_input(case, args.dtype, device)
    first_dims = first_dims_tensor(case, device)
    quantizer = make_quantizer(mode, case.num_tensors, device)

    latency_ms, output = time_case(
        input_tensor=input_tensor,
        quantizer=quantizer,
        case=case,
        first_dims=first_dims,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    validate_capacity_slack(case, output)

    actual_input_bytes = case.actual_total_elements * input_tensor.element_size()
    actual_output_bytes = output.rowwise_data.numel() * output.rowwise_data.element_size()
    scale_metadata_bytes = metadata_nbytes(output)
    input_allocation_bytes = storage_nbytes(input_tensor)
    output_allocation_bytes = storage_nbytes(output.rowwise_data)
    allocation_bytes = input_allocation_bytes + output_allocation_bytes
    actual_payload_bytes = actual_input_bytes + actual_output_bytes + scale_metadata_bytes
    input_read_multiplier = 2 if mode == "current" else 1
    bandwidth_numerator_bytes = (
        input_read_multiplier * actual_input_bytes + actual_output_bytes + scale_metadata_bytes
    )
    latency_s = latency_ms / 1000.0
    bandwidth_gb_s = bandwidth_numerator_bytes / latency_s / 1.0e9
    formula = (
        f"({input_read_multiplier} * actual_input_bytes + actual_output_bytes + "
        "scale_metadata_bytes) / latency_seconds / 1e9"
    )

    result = {
        "mode": mode,
        "case": case.name,
        "num_tensors": case.num_tensors,
        "splits": case.splits,
        "logical_shapes": case.logical_shapes,
        "logical_shape": [case.actual_rows, case.cols],
        "actual_total_elements": case.actual_total_elements,
        "allocation_elements": case.allocation_rows * case.cols,
        "input_allocation_elements": case.allocation_rows * case.cols,
        "output_allocation_elements": output.rowwise_data.numel(),
        "actual_input_bytes": actual_input_bytes,
        "actual_output_bytes": actual_output_bytes,
        "scale_metadata_bytes": scale_metadata_bytes,
        "actual_payload_bytes": actual_payload_bytes,
        "input_allocation_bytes": input_allocation_bytes,
        "output_allocation_bytes": output_allocation_bytes,
        "allocation_bytes": allocation_bytes,
        "allocation_bytes_differ": allocation_bytes
        != actual_input_bytes + actual_output_bytes,
        "input_read_multiplier": input_read_multiplier,
        "bandwidth_numerator_bytes": bandwidth_numerator_bytes,
        "latency_ms": latency_ms,
        "latency_us": latency_ms * 1000.0,
        "bandwidth_gb_s": bandwidth_gb_s,
        "bandwidth_formula": formula,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "input_dtype": str(input_tensor.dtype),
        "output_dtype": "fp8_e4m3",
    }

    allocation_note = (
        f" allocation_bytes={allocation_bytes}"
        if result["allocation_bytes_differ"]
        else " allocation_bytes=same_as_actual_payload_allocation"
    )
    print(
        "grouped_fp8_quantize "
        f"mode={mode} case={case.name} num_tensors={case.num_tensors} "
        f"splits={case.splits} logical_shapes={case.logical_shapes} "
        f"actual_total_elements={case.actual_total_elements} "
        f"allocation_elements={result['allocation_elements']} "
        f"actual_input_bytes={actual_input_bytes} "
        f"actual_output_bytes={actual_output_bytes} "
        f"scale_metadata_bytes={scale_metadata_bytes}"
        f"{allocation_note} "
        f"latency_us={result['latency_us']:.3f} "
        f"bandwidth_gb_s={bandwidth_gb_s:.3f} "
        f"bandwidth_formula=\"{formula}\""
    )
    return result


def write_report(path: str, report: Dict[str, object]) -> None:
    """Write a JSON benchmark report."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse benchmark command-line arguments."""
    default_output = os.environ.get(
        "ORCHESTRA_BENCHMARK_RAW_REPORT",
        "benchmark_grouped_fp8_quantize.json",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=default_output, help="JSON report path")
    parser.add_argument("--num-tensors", type=int, default=4, help="Number of grouped tensors")
    parser.add_argument(
        "--rows-per-tensor",
        type=int,
        default=4096,
        help="Rows per tensor for the balanced case",
    )
    parser.add_argument("--cols", type=int, default=4096, help="Common last dimension")
    parser.add_argument(
        "--unbalanced-splits",
        type=parse_csv_ints,
        default=None,
        help="Comma-separated row splits for unbalanced and capacity-slack cases",
    )
    parser.add_argument(
        "--slack-fraction",
        type=float,
        default=0.25,
        help="Extra input allocation rows as a fraction of logical rows",
    )
    parser.add_argument(
        "--slack-rows",
        type=int,
        default=0,
        help="Minimum extra input allocation rows for the capacity-slack case",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Timed iterations")
    parser.add_argument(
        "--modes",
        type=lambda value: parse_csv_strings(value, ("delayed", "current")),
        default=["delayed", "current"],
        help="Comma-separated modes: delayed,current",
    )
    parser.add_argument(
        "--cases",
        type=lambda value: parse_csv_strings(
            value, ("balanced", "unbalanced", "capacity_slack")
        ),
        default=["balanced", "unbalanced", "capacity_slack"],
        help="Comma-separated cases: balanced,unbalanced,capacity_slack",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Input dtype",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use small dimensions and few iterations for smoke testing",
    )
    args = parser.parse_args()

    if args.num_tensors < 2:
        raise ValueError("--num-tensors must be at least 2")
    if args.rows_per_tensor <= 0:
        raise ValueError("--rows-per-tensor must be positive")
    if args.cols <= 0:
        raise ValueError("--cols must be positive")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.slack_fraction < 0:
        raise ValueError("--slack-fraction must be non-negative")
    if args.slack_rows < 0:
        raise ValueError("--slack-rows must be non-negative")

    if args.quick:
        args.rows_per_tensor = min(args.rows_per_tensor, 64)
        args.cols = min(args.cols, 128)
        args.warmup = min(args.warmup, 2)
        args.iterations = min(args.iterations, 5)
        args.slack_rows = max(args.slack_rows, 16)

    args.dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    return args


def main() -> None:
    """Run grouped FP8 quantize benchmarks."""
    args = parse_args()
    fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
    if not fp8_available:
        raise RuntimeError(f"FP8 is not available: {reason_for_no_fp8}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for grouped FP8 quantize benchmarking")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    cases = make_cases(args)
    selected_cases = [cases[name] for name in args.cases]

    results = []
    start_time = time.time()
    for mode in args.modes:
        for case in selected_cases:
            results.append(benchmark_one(mode, case, args, device))

    report = {
        "schema_version": "grouped_fp8_quantize_benchmark/v1",
        "benchmark": "grouped_fp8_quantize",
        "summary": (
            "Grouped FP8 tensor-scaling quantize through "
            "transformer_engine_torch.group_quantize."
        ),
        "device": torch.cuda.get_device_name(device),
        "cuda_device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "modes": args.modes,
        "cases": args.cases,
        "result_count": len(results),
        "elapsed_wall_time_sec": time.time() - start_time,
        "results": results,
    }
    write_report(args.output, report)
    print(f"Wrote grouped FP8 quantize benchmark report to {args.output}")


if __name__ == "__main__":
    main()
