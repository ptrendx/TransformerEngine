# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Paired host submission benchmark for the installed DotProductAttention.

Run in separate processes for each backend/mode/case and repeat the command.
Compare paired CPU deltas, not GPU acceleration inferred from host timings.
The existing benchmark_attention.py measures complementary GPU workloads.
"""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from types import FunctionType, SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["fused", "flash"], default="fused")
    parser.add_argument("--case", choices=["short", "prefill", "gqa"], default="short")
    parser.add_argument("--mode", choices=["inference", "train"], default="inference")
    parser.add_argument(
        "--layout",
        choices=[
            "separate",
            "packed_qkv",
            "packed_kv",
            "declared",
            "sbhd",
            "copy",
            "parameter",
            "thd",
            "ragged",
        ],
        default="separate",
    )
    parser.add_argument("--maxima", choices=["both", "q", "kv", "neither"], default="both")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--detector-only", action="store_true")
    parser.add_argument("--allow-nondeterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--baseline", type=Path, help="Matched prior result for a CPU regression gate"
    )
    parser.add_argument("--max-regression-percent", type=float, default=5.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for key, value in {
        "NVTE_FUSED_ATTN": args.backend == "fused",
        "NVTE_FLASH_ATTN": args.backend == "flash",
        "NVTE_UNFUSED_ATTN": 0,
        "NVTE_FLASH_ATTN_V2": 1,
        "NVTE_FLASH_ATTN_V3": 0,
        "NVTE_FLASH_ATTN_V4": 0,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": args.allow_nondeterministic,
        "NVTE_NVTX_ENABLED": 0,
    }.items():
        os.environ[key] = str(int(value))
    import torch
    import transformer_engine
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.attention.dot_product_attention import (
        dot_product_attention as dpa,
    )
    from transformer_engine.pytorch.attention.dot_product_attention import utils

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    b, s, h, g, d = {
        "short": (2, 128, 16, 16, 64),
        "prefill": (2, 2048, 16, 16, 128),
        "gqa": (2, 512, 32, 4, 128),
    }[args.case]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    training = args.mode == "train"
    detector_only = args.detector_only or args.layout == "parameter"
    fmt = (
        "sbhd" if args.layout == "sbhd" else ("thd" if args.layout in ("thd", "ragged") else "bshd")
    )
    lengths = [s, s // 2] if args.layout == "ragged" else [s] * b

    def shape(heads):
        return (
            (s, b, heads, d)
            if fmt == "sbhd"
            else ((sum(lengths), heads, d) if fmt == "thd" else (b, s, heads, d))
        )

    def rand(size):
        return torch.randn(size, device="cuda", dtype=dtype, requires_grad=training)

    q, k, v = [rand(shape(heads)) for heads in (h, g, g)]
    kwargs = {}
    if args.layout in ("packed_qkv", "declared", "packed_kv"):
        count = 2 if args.layout == "packed_kv" else 3
        if count == 3 and h != g:
            parser.error("Packed QKV requires equal Q/KV head counts")
        size = list(shape(g))
        size.insert(-2, count)
        packed = rand(size)
        if count == 3:
            q, k, v = packed.unbind(-3)
        else:
            k, v = packed.unbind(-3)
        if args.layout == "declared":
            kwargs.update(qkv_layer=packed, qkv_interleave_dim=-3)
    elif args.layout == "copy":
        k = rand((*shape(g)[:-2], 2 * g, d))[..., ::2, :]
    elif args.layout == "parameter":
        q, k, v = [torch.nn.Parameter(t, requires_grad=training) for t in (q, k, v)]
    if fmt == "thd":
        cu = torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32)
        kwargs.update(cu_seqlens_q=cu, cu_seqlens_kv=cu)
        if args.maxima in ("both", "q"):
            kwargs["max_seqlen_q"] = s
        if args.maxima in ("both", "kv"):
            kwargs["max_seqlen_kv"] = s
    module = (
        te.DotProductAttention(
            h,
            d,
            num_gqa_groups=g,
            qkv_format=fmt,
            attn_mask_type="padding_causal" if fmt == "thd" else "causal",
            attention_dropout=0,
        )
        .cuda()
        .train(training)
    )
    ctx = contextlib.nullcontext if training else torch.no_grad
    native = utils.get_qkv_layout
    reference = utils._get_qkv_layout_python
    # Isolated globals preserve the NumPy oracle without mutating NumPy itself.
    # torch.Size.numel is the shape-only control; the installed native detector
    # reads Tensor.numel directly without building Python shape objects.
    numel_globals = dict(reference.__globals__, np=SimpleNamespace(prod=lambda size: size.numel()))
    numel = FunctionType(
        reference.__code__,
        numel_globals,
        reference.__name__,
        reference.__defaults__,
        reference.__closure__,
    )
    routes = {"python": reference, "numel": numel, "native": native}

    def legacy_maxima(cuq, cuk, maxq, maxk):
        if maxq is None:
            maxq = (cuq[1:] - cuq[:-1]).max().item()
        if maxk is None:
            maxk = (cuk[1:] - cuk[:-1]).max().item()
        return maxq, maxk

    @contextlib.contextmanager
    def route_context(name):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(utils, "get_qkv_layout", routes[name]))
            if fmt == "thd" and name != "native" and hasattr(utils, "get_max_seqlens"):
                stack.enter_context(patch.object(utils, "get_max_seqlens", legacy_maxima))
            yield

    grad = None

    def step():
        if detector_only:
            return utils.get_qkv_layout(q, k, v, fmt)[1:4]
        result = (
            module(None, None, None, **kwargs)
            if args.layout == "declared"
            else module(q, k, v, **kwargs)
        )
        if training:
            return (
                result,
                *torch.autograd.grad(
                    result, (packed,) if args.layout == "declared" else (q, k, v), grad
                ),
            )
        return (result,)

    if not detector_only:
        selected = []
        handle = getattr(
            module, "fused_attention" if args.backend == "fused" else "flash_attention"
        ).register_forward_pre_hook(lambda *_args: selected.append(True))
        with ctx():
            output = (
                module(None, None, None, **kwargs)
                if args.layout == "declared"
                else module(q, k, v, **kwargs)
            )
            grad = torch.randn_like(output)
        handle.remove()
        assert selected, "Requested backend was not invoked"

        # Independent FP32 reference, including GQA, ragged sequences and gradients.
        def dense(t):
            return t.transpose(0, 1) if fmt == "sbhd" else t

        refs = [dense(t).detach().float().requires_grad_(training) for t in (q, k, v)]
        outputs = []
        offset = 0
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            for index, length in enumerate(lengths):
                ts = [
                    t[offset : offset + length] if fmt == "thd" else t[index, :length] for t in refs
                ]
                qr, kr, vr = [t.transpose(0, 1).unsqueeze(0) for t in ts]
                result = torch.nn.functional.scaled_dot_product_attention(
                    qr,
                    kr.repeat_interleave(h // g, 1),
                    vr.repeat_interleave(h // g, 1),
                    is_causal=True,
                )
                outputs.append(result.squeeze(0).transpose(0, 1))
                offset += length
        ref_output = torch.cat(outputs) if fmt == "thd" else torch.stack(outputs)
        if fmt == "sbhd":
            ref_output = ref_output.transpose(0, 1)
        ref_output = ref_output.flatten(-2)
        expected = (ref_output,)
        if training:
            ref_grads = torch.autograd.grad(ref_output, refs, grad.float())
            ref_grads = [t.transpose(0, 1) if fmt == "sbhd" else t for t in ref_grads]
            expected += (
                (torch.stack(ref_grads, -3),) if args.layout == "declared" else tuple(ref_grads)
            )
    else:
        expected = tuple(t.detach().float() for t in (q, k, v))
    tol = dict(atol=0.025, rtol=0.025) if dtype == torch.bfloat16 else dict(atol=0.005, rtol=0.005)
    errors = {}
    records = []
    gpu_times = {}
    with ctx():
        baseline = step()
        for name in routes:
            with route_context(name):
                actual = step()
                for x, y, z in zip(actual, baseline, expected):
                    torch.testing.assert_close(
                        x, y, **(tol if args.allow_nondeterministic else dict(atol=0, rtol=0))
                    )
                    torch.testing.assert_close(x.float(), z, **tol)
                errors[name] = [
                    float((x.detach().float() - y.detach()).abs().max())
                    for x, y in zip(actual, expected)
                ]
                for _ in range(args.warmup):
                    step()
                torch.cuda.synchronize()
        for repeat in range(args.repeats):
            rows = {name: [] for name in routes}
            for batch in range(args.batches):
                names = list(routes)
                rotation = (batch + repeat) % len(names)
                for name in names[rotation:] + names[:rotation]:
                    with route_context(name):
                        torch.cuda.synchronize()
                        wall = time.perf_counter_ns()
                        process = time.process_time_ns()
                        thread = time.thread_time_ns()
                        for _ in range(args.batch_size):
                            step()
                        thread_elapsed = time.thread_time_ns() - thread
                        process_elapsed = time.process_time_ns() - process
                        host_elapsed = time.perf_counter_ns() - wall
                        torch.cuda.synchronize()
                        complete_elapsed = time.perf_counter_ns() - wall
                    rows[name].append(
                        {
                            key: value / 1000 / args.batch_size
                            for key, value in dict(
                                thread_cpu_us=thread_elapsed,
                                process_cpu_us=process_elapsed,
                                host_us=host_elapsed,
                                complete_us=complete_elapsed,
                            ).items()
                        }
                    )
            records.append(rows)
        if not detector_only:
            # Device events on a captured workload separate GPU execution from host
            # submission gaps. Capture, warmup and input copies are outside timing.
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                # Use independent graph leaves: retaining the eager result can retain
                # AccumulateGrad nodes bound to the submitting stream on newer PyTorch.
                with torch.no_grad():
                    if args.layout in ("packed_qkv", "packed_kv", "declared"):
                        packed = packed.detach().clone().requires_grad_(training)
                        if args.layout == "packed_kv":
                            q = q.detach().clone().requires_grad_(training)
                        if args.layout == "declared":
                            kwargs["qkv_layer"] = packed
                    else:
                        clones = [
                            torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
                            .copy_(t)
                            .requires_grad_(training)
                            for t in (q, k, v)
                        ]
                        q, k, v = (
                            [torch.nn.Parameter(t, requires_grad=training) for t in clones]
                            if args.layout == "parameter"
                            else clones
                        )
                # Recreate views with recording enabled for training, after copying.
                if args.layout == "packed_kv":
                    k, v = packed.unbind(-3)
                elif args.layout in ("packed_qkv", "declared"):
                    q, k, v = packed.unbind(-3)
            graph = torch.cuda.CUDAGraph()
            capture_kwargs = kwargs.copy()
            if fmt == "thd":
                kwargs.update(max_seqlen_q=s, max_seqlen_kv=s)
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    step()
            with torch.cuda.graph(graph, stream=capture_stream):
                graph_results = step()
            torch.cuda.current_stream().wait_stream(capture_stream)
            kwargs.clear()
            kwargs.update(capture_kwargs)
            graph.replay()
            for actual, reference_result in zip(graph_results, baseline):
                torch.testing.assert_close(actual, reference_result, **tol)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(20):
                graph.replay()
            start.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            gpu_times["graph_replay_us"] = start.elapsed_time(end) * 1000 / 100
    summary = {
        name: {
            key: [statistics.median(row[key] for row in repeat[name]) for repeat in records]
            for key in records[0][name][0]
        }
        for name in routes
    }
    deltas = {
        name: {
            key: [
                statistics.median(p[key] - c[key] for p, c in zip(repeat["python"], repeat[name]))
                for repeat in records
            ]
            for key in records[0][name][0]
        }
        for name in ("numel", "native")
    }
    metadata = dict(
        args={
            **vars(args),
            "out": str(args.out),
            "baseline": str(args.baseline) if args.baseline else None,
        },
        command=os.sys.argv,
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        device_count=torch.cuda.device_count(),
        topology=subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        )
        .strip()
        .splitlines()[0],
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        nccl=list(torch.cuda.nccl.version()),
        extension=tex.__file__,
        extension_sha256=hashlib.sha256(Path(tex.__file__).read_bytes()).hexdigest(),
        source=transformer_engine.__file__,
        source_sha256={
            path: hashlib.sha256(
                (Path(transformer_engine.__file__).parent.parent / path).read_bytes()
            ).hexdigest()
            for path in (
                "transformer_engine/pytorch/attention/dot_product_attention/utils.py",
                "transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py",
                "transformer_engine/pytorch/attention/dot_product_attention/backends.py",
                "transformer_engine/pytorch/csrc/extensions/attention_layout.cpp",
                "benchmarks/attention/benchmark_attention_cpu.py",
            )
        },
        backend={key: str(value) for key, value in dpa._attention_backends.items()},
        shape=[b, s, h, g, d],
        strides=[list(t.stride()) for t in (q, k, v)],
        distribution="standard normal",
        scope="layout detector" if detector_only else "DotProductAttention forward/backward",
        maxima_routes=(
            "Python/numel use legacy scalar readbacks; native uses batched missing maxima"
        ),
        affinity=sorted(os.sched_getaffinity(0)),
        env={key: value for key, value in os.environ.items() if key.startswith("NVTE_")},
        synchronization="before batches and after submission; final wait excluded from CPU/host",
        correctness=errors,
        records=records,
        summary=summary,
        paired_savings_us=deltas,
        gpu=gpu_times,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.baseline:
        previous = json.loads(args.baseline.read_text())
        for key in (
            "device",
            "capability",
            "device_count",
            "topology",
            "shape",
            "strides",
            "affinity",
            "torch",
            "cuda",
            "cudnn",
            "nccl",
            "driver",
            "scope",
            "backend",
            "env",
        ):
            if metadata[key] != previous[key]:
                raise ValueError(f"Regression baseline mismatch for {key}")
        for key in (
            "backend",
            "case",
            "mode",
            "layout",
            "dtype",
            "maxima",
            "seed",
            "batch_size",
            "warmup",
            "batches",
            "repeats",
            "allow_nondeterministic",
            "detector_only",
        ):
            if metadata["args"][key] != previous["args"][key]:
                raise ValueError(f"Regression baseline configuration mismatch for {key}")
        current_cpu = statistics.median(summary["native"]["process_cpu_us"])
        previous_cpu = statistics.median(previous["summary"]["native"]["process_cpu_us"])
        metadata["regression_percent"] = 100 * (current_cpu / previous_cpu - 1)
    args.out.write_text(json.dumps(metadata, indent=2) + "\n")
    if args.baseline and metadata["regression_percent"] > args.max_regression_percent:
        raise RuntimeError(
            f"CPU regression {metadata['regression_percent']:.2f}% exceeds configured gate"
        )
    print(json.dumps(dict(summary=summary, paired_savings_us=deltas, gpu=gpu_times), indent=2))


if __name__ == "__main__":
    main()
