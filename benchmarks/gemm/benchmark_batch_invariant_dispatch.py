# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Measure complete PyTorch host submission and profile native CuTeDSL dispatch.

Pin the process to a CPU core for host comparisons. Inputs/weights are reused;
``alloc`` calls include warmed output allocation. Synchronization is outside the
CPU timer. ``--profile`` records NVTX ranges for individual Nsight kernel timings.
"""

import argparse
import gc
import json
import statistics
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path

import torch

from benchmark_batch_invariant_gemm import parse_shape
from transformer_engine.pytorch.cpp_extensions.batch_invariant_gemm import (
    _batch_invariant_gemm_cutedsl_python,
    _can_use_cutedsl,
    _check,
    batch_invariant_gemm,
)
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm


def python_cutedsl(a, b, out=None):
    """Previous complete BF16 public-call validation and Python launch path."""
    _check(a, b, out)
    if not _can_use_cutedsl(a, b):
        raise ValueError("Python CuTeDSL baseline requires aligned SM100 BF16 operands")
    if out is not None and out.shape != (a.shape[0], b.shape[0]):
        raise ValueError("Output must have shape [M,N]")
    return _batch_invariant_gemm_cutedsl_python(a, b, out)


def make_calls(a, b, allocate):
    """Create public calls with independent outputs, or allocation inside each call."""
    out = lambda: (
        None
        if allocate
        else torch.empty(  # pylint: disable=unnecessary-lambda-assignment
            a.shape[0], b.shape[0], dtype=a.dtype, device=a.device
        )
    )
    return {
        "python": partial(python_cutedsl, a, b, out=out()),
        "native": partial(batch_invariant_gemm, a, b, out=out(), backend="cutedsl"),
        "general_bi": partial(general_gemm, b, a, out=out(), batch_invariant=True),
        "triton": partial(batch_invariant_gemm, a, b, out=out(), backend="triton"),
        "cublas": partial(general_gemm, b, a, out=out(), out_dtype=torch.bfloat16),
    }


def result_tensor(result):
    """Unwrap general_gemm's four outputs for correctness checks."""
    return result[0] if isinstance(result, (tuple, list)) else result


def cpu_times(calls, trials, batch):
    """Rotate path order and measure complete asynchronous calls, including allocation."""
    samples = {name: [] for name in calls}
    names = list(calls)
    enabled = gc.isenabled()
    gc.disable()
    try:
        for trial in range(trials):
            offset = trial % len(names)
            for name in names[offset:] + names[:offset]:
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                for _ in range(batch):
                    calls[name]()
                elapsed = time.perf_counter_ns() - start
                torch.cuda.synchronize()
                samples[name].append(elapsed / 1000 / batch)
    finally:
        if enabled:
            gc.enable()
    stats = {}
    for name, values in samples.items():
        ordered = sorted(values)
        stats[name] = {
            "median_us": statistics.median(values),
            "p10_us": ordered[int(0.1 * (trials - 1))],
            "p90_us": ordered[int(0.9 * (trials - 1))],
        }
    return stats


def make_graphs(calls, nodes):
    """Build and warm graphs before the profiler starts."""
    graphs = {}
    for name, call in calls.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(nodes):
                call()
        for _ in range(3):
            graph.replay()
        graphs[name] = graph
    torch.cuda.synchronize()
    return graphs


def profile_calls(shape, calls, graphs, nodes):
    """Capture warmed graph/eager NVTX ranges for individual kernel timing."""
    for mode in ("graph", "eager"):
        for name, call in calls.items():
            for _ in range(10):
                graphs[name].replay()
            torch.cuda.synchronize()
            with torch.cuda.nvtx.range(f"BI/{mode}/{'x'.join(map(str, shape))}/{name}"):
                if mode == "graph":
                    for _ in range(3):
                        graphs[name].replay()
                else:
                    for _ in range(nodes):
                        call()
                torch.cuda.synchronize()


def main():
    """Compare previous Python launch with the production native integration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=parse_shape,
        default=[(m, 4096, 4096) for m in (1, 16, 256, 1024, 4096, 8192)] + [(8192, 8192, 8192)],
    )
    parser.add_argument("--trials", type=int, default=101)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.trials, args.batch, args.warmup, args.nodes) < 1:
        parser.error("Trial, batch, warmup, and node counts must be positive")
    torch.manual_seed(2026)
    environment = {
        "gpu": torch.cuda.get_device_name(),
        "cc": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cutlass": version("nvidia-cutlass-dsl"),
        "triton": version("triton"),
        "seed": 2026,
        "warmup": args.warmup,
        "trials": args.trials,
        "batch": args.batch,
        "nodes": args.nodes,
    }
    print(json.dumps(environment), flush=True)
    records = []
    profile_cases = []
    for shape in args.shapes:
        m, n, k = shape
        a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        reference = (a.float() @ b.float().T).bfloat16()
        calls = make_calls(a, b, False)
        calls.update({name + "_alloc": fn for name, fn in make_calls(a, b, True).items()})
        for _ in range(args.warmup):
            for call in calls.values():
                call()
        torch.cuda.synchronize()
        expected = python_cutedsl(a, b)
        for name, call in calls.items():
            actual = result_tensor(call())
            torch.testing.assert_close(actual, reference, rtol=0.008, atol=0.03125)
            if name.startswith("native") or name.startswith("general_bi") and min(n, k) >= 512:
                assert torch.equal(actual, expected), name
        if args.profile:
            calls = {name: fn for name, fn in calls.items() if not name.endswith("_alloc")}
            profile_cases.append((shape, calls, make_graphs(calls, args.nodes), reference))
        else:
            record = {"shape": shape, "stats": cpu_times(calls, args.trials, args.batch)}
            records.append(record)
            print(json.dumps(record), flush=True)
            if args.output:
                args.output.write_text(
                    json.dumps({"environment": environment, "records": records}, indent=2)
                )

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
        for shape, calls, graphs, reference in profile_cases:
            profile_calls(shape, calls, graphs, args.nodes)
            for call in calls.values():
                torch.testing.assert_close(
                    result_tensor(call()), reference, rtol=0.008, atol=0.03125
                )
        torch.cuda.cudart().cudaProfilerStop()
        print("Profile complete; reference checks passed.", flush=True)


if __name__ == "__main__":
    main()
