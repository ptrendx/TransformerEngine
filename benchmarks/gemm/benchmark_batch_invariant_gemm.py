# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compare BF16 batch-invariant GEMM backends with ``general_gemm``.

Example:
    CUDA_VISIBLE_DEVICES=0 python benchmarks/gemm/benchmark_batch_invariant_gemm.py

The CUDA graph timing measures steady-state GPU work. The eager timing includes
Python dispatch, output tensors are preallocated in both cases, and compilation
is excluded by warmup.
"""

import argparse
import statistics
import time

import torch

from transformer_engine.pytorch.cpp_extensions.batch_invariant_gemm import batch_invariant_gemm
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm


def graph_us(fn, nodes: int, trials: int) -> float:
    """Median GPU microseconds per invocation of an unrolled CUDA graph."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(nodes):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / nodes)
    return statistics.median(samples)


def eager_us(fn, iterations: int, trials: int) -> float:
    """Median synchronized wall-clock microseconds per eager invocation."""
    samples = []
    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / iterations)
    return statistics.median(samples)


def parse_shape(value: str) -> tuple[int, int, int]:
    """Parse MxNxK dimensions."""
    try:
        shape = tuple(int(x) for x in value.lower().split("x"))
        if len(shape) != 3 or min(shape) < 1:
            raise ValueError
        return shape
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be MxNxK with positive integers") from exc


def main() -> None:
    """Run the comparison on the selected CUDA device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        type=parse_shape,
        nargs="+",
        default=[(256, 192, 128)] + [(m, 4096, 4096) for m in (1, 16, 64, 256, 1024, 4096)],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--graph-nodes", type=int, default=100)
    parser.add_argument("--graph-trials", type=int, default=9)
    parser.add_argument("--eager-iterations", type=int, default=200)
    parser.add_argument("--eager-trials", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")
    torch.manual_seed(2026)
    print(f"GPU: {torch.cuda.get_device_name()} | PyTorch: {torch.__version__}")
    print("M,N,K | graph Triton/CuTe/general (us) | eager Triton/CuTe/general (us)")
    for m, n, k in args.shapes:
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        outputs = {
            name: torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            for name in ("triton", "cutedsl", "general")
        }
        calls = {
            name: (lambda name=name: batch_invariant_gemm(a, b, out=outputs[name], backend=name))
            for name in ("triton", "cutedsl")
        }
        calls["general"] = lambda: general_gemm(
            b, a, out_dtype=torch.bfloat16, out=outputs["general"], layout="TN"
        )
        for _ in range(args.warmup):
            for fn in calls.values():
                fn()
        torch.cuda.synchronize()
        reference = (a.float() @ b.float().T).to(torch.bfloat16)
        for name, result in outputs.items():
            torch.testing.assert_close(result, reference, rtol=0.02, atol=0.5, msg=name)
        graph = {
            name: graph_us(fn, args.graph_nodes, args.graph_trials) for name, fn in calls.items()
        }
        eager = {
            name: eager_us(fn, args.eager_iterations, args.eager_trials)
            for name, fn in calls.items()
        }
        print(
            f"{m},{n},{k} | "
            f"{graph['triton']:.3f}/{graph['cutedsl']:.3f}/{graph['general']:.3f} | "
            f"{eager['triton']:.3f}/{eager['cutedsl']:.3f}/{eager['general']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
