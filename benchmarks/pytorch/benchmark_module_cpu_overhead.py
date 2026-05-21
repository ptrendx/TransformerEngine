# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark CPU-side invocation overhead for tiny PyTorch TE modules."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError:
    torch = None


MODULES = ("linear", "layernorm_linear", "layernorm_mlp")
MODES = ("bf16", "mxfp8")
DIRECTIONS = ("forward", "backward")
GOAL_RATIOS = {"bf16": 2.0, "mxfp8": 4.0}


@dataclass(frozen=True)
class BenchmarkConfig:
    modules: Tuple[str, ...]
    modes: Tuple[str, ...]
    directions: Tuple[str, ...]
    batch_size: int
    hidden_size: int
    ffn_hidden_size: int
    warmup: int
    iterations: int
    json_output: Optional[Path]
    report_output: Optional[Path]
    profile: bool
    profile_module: str
    profile_mode: str
    profile_direction: str
    dry_run: bool


if torch is not None:

    class TorchLayerNormLinear(torch.nn.Module):
        """Torch-native BF16 baseline for TE LayerNormLinear."""

        def __init__(self, hidden_size: int, out_features: int, device: str) -> None:
            super().__init__()
            self.layer_norm = torch.nn.LayerNorm(hidden_size, device=device, dtype=torch.bfloat16)
            self.linear = torch.nn.Linear(
                hidden_size, out_features, bias=True, device=device, dtype=torch.bfloat16
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(self.layer_norm(x))

    class TorchLayerNormMLP(torch.nn.Module):
        """Torch-native BF16 baseline for TE LayerNormMLP."""

        def __init__(self, hidden_size: int, ffn_hidden_size: int, device: str) -> None:
            super().__init__()
            self.layer_norm = torch.nn.LayerNorm(hidden_size, device=device, dtype=torch.bfloat16)
            self.fc1 = torch.nn.Linear(
                hidden_size, ffn_hidden_size, bias=True, device=device, dtype=torch.bfloat16
            )
            self.fc2 = torch.nn.Linear(
                ffn_hidden_size, hidden_size, bias=True, device=device, dtype=torch.bfloat16
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.layer_norm(x)
            x = self.fc1(x)
            x = torch.nn.functional.gelu(x, approximate="tanh")
            return self.fc2(x)


def _parse_csv(value: str, choices: Sequence[str], name: str) -> Tuple[str, ...]:
    if value == "all":
        return tuple(choices)
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected) - set(choices))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown {name}: {', '.join(unknown)}")
    if not selected:
        raise argparse.ArgumentTypeError(f"At least one {name} must be selected")
    return selected


def _parse_args(argv: Optional[Sequence[str]] = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", default="all", help="Comma list or all")
    parser.add_argument("--modes", default="bf16,mxfp8", help="Comma list or all")
    parser.add_argument("--directions", default="forward,backward", help="Comma list or all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--ffn-hidden-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-module", choices=MODULES, default="linear")
    parser.add_argument("--profile-mode", choices=MODES, default="mxfp8")
    parser.add_argument("--profile-direction", choices=DIRECTIONS, default="forward")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit the case matrix without importing Transformer Engine or using CUDA.",
    )
    args = parser.parse_args(argv)

    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.batch_size <= 0 or args.hidden_size <= 0 or args.ffn_hidden_size <= 0:
        parser.error("Shape sizes must be positive")

    modules = _parse_csv(args.modules, MODULES, "module")
    modes = _parse_csv(args.modes, MODES, "mode")
    directions = _parse_csv(args.directions, DIRECTIONS, "direction")
    if args.profile and (
        args.profile_module not in modules
        or args.profile_mode not in modes
        or args.profile_direction not in directions
    ):
        parser.error("Profile selector must be included in --modules, --modes, and --directions")

    return BenchmarkConfig(
        modules=modules,
        modes=modes,
        directions=directions,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        warmup=args.warmup,
        iterations=args.iterations,
        json_output=args.json_output,
        report_output=args.report_output,
        profile=args.profile,
        profile_module=args.profile_module,
        profile_mode=args.profile_mode,
        profile_direction=args.profile_direction,
        dry_run=args.dry_run,
    )


def _import_te():
    if torch is None:
        raise RuntimeError("PyTorch is required for module CPU-overhead benchmarking")
    import transformer_engine  # noqa: F401
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import MXFP8BlockScaling

    return te, MXFP8BlockScaling


def _synchronize() -> None:
    torch.cuda.synchronize()


def _clear_grads(module: torch.nn.Module, inp: torch.Tensor) -> None:
    module.zero_grad(set_to_none=True)
    inp.grad = None


@contextmanager
def _profile_capture(enabled: bool, label: str):
    if not enabled:
        yield
        return
    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()


def _mode_context(te_module, mode: str, mxfp8_recipe):
    if mode == "mxfp8":
        return te_module.autocast(enabled=True, recipe=mxfp8_recipe)
    return nullcontext()


def _make_input(config: BenchmarkConfig, device: str) -> torch.Tensor:
    return torch.randn(
        config.batch_size,
        config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )


def _make_modules(module_name: str, config: BenchmarkConfig, te_module, device: str):
    if module_name == "linear":
        te_mod = te_module.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            params_dtype=torch.bfloat16,
            device=device,
        )
        torch_mod = torch.nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            device=device,
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            torch_mod.weight.copy_(te_mod.weight)
            torch_mod.bias.copy_(te_mod.bias)
        return te_mod, torch_mod

    if module_name == "layernorm_linear":
        te_mod = te_module.LayerNormLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            params_dtype=torch.bfloat16,
            device=device,
        )
        torch_mod = TorchLayerNormLinear(config.hidden_size, config.hidden_size, device)
        with torch.no_grad():
            torch_mod.layer_norm.weight.copy_(te_mod.layer_norm_weight)
            torch_mod.layer_norm.bias.copy_(te_mod.layer_norm_bias)
            torch_mod.linear.weight.copy_(te_mod.weight)
            torch_mod.linear.bias.copy_(te_mod.bias)
        return te_mod, torch_mod

    if module_name == "layernorm_mlp":
        te_mod = te_module.LayerNormMLP(
            config.hidden_size,
            config.ffn_hidden_size,
            bias=True,
            activation="gelu",
            params_dtype=torch.bfloat16,
            device=device,
        )
        torch_mod = TorchLayerNormMLP(config.hidden_size, config.ffn_hidden_size, device)
        with torch.no_grad():
            torch_mod.layer_norm.weight.copy_(te_mod.layer_norm_weight)
            torch_mod.layer_norm.bias.copy_(te_mod.layer_norm_bias)
            torch_mod.fc1.weight.copy_(te_mod.fc1_weight)
            torch_mod.fc1.bias.copy_(te_mod.fc1_bias)
            torch_mod.fc2.weight.copy_(te_mod.fc2_weight)
            torch_mod.fc2.bias.copy_(te_mod.fc2_bias)
        return te_mod, torch_mod

    raise ValueError(f"Unhandled module {module_name}")


def _run_forward(module: torch.nn.Module, inp: torch.Tensor) -> torch.Tensor:
    out = module(inp)
    if isinstance(out, tuple):
        out = out[0]
    return out


def _measure_forward(
    module: torch.nn.Module,
    inp: torch.Tensor,
    warmup: int,
    iterations: int,
    profile: bool,
    label: str,
) -> List[float]:
    for _ in range(warmup):
        _run_forward(module, inp)
    _synchronize()

    samples_us: List[float] = []
    with _profile_capture(profile, label):
        for _ in range(iterations):
            _synchronize()
            start = time.perf_counter_ns()
            _run_forward(module, inp)
            end = time.perf_counter_ns()
            _synchronize()
            samples_us.append((end - start) / 1000.0)
    return samples_us


def _measure_backward(
    module: torch.nn.Module,
    inp: torch.Tensor,
    warmup: int,
    iterations: int,
    profile: bool,
    label: str,
) -> List[float]:
    for _ in range(warmup):
        _clear_grads(module, inp)
        out = _run_forward(module, inp)
        _synchronize()
        _clear_grads(module, inp)
        out.backward(torch.ones_like(out))
    _synchronize()

    samples_us: List[float] = []
    with _profile_capture(profile, label):
        for _ in range(iterations):
            _clear_grads(module, inp)
            out = _run_forward(module, inp)
            grad_output = torch.ones_like(out)
            _synchronize()
            _clear_grads(module, inp)
            start = time.perf_counter_ns()
            out.backward(grad_output)
            end = time.perf_counter_ns()
            _synchronize()
            samples_us.append((end - start) / 1000.0)
    return samples_us


def _measure_invocations(
    module: torch.nn.Module,
    inp: torch.Tensor,
    direction: str,
    warmup: int,
    iterations: int,
    profile: bool,
    label: str,
) -> List[float]:
    if direction == "forward":
        return _measure_forward(module, inp, warmup, iterations, profile, label)
    if direction == "backward":
        return _measure_backward(module, inp, warmup, iterations, profile, label)
    raise ValueError(f"Unhandled direction {direction}")


def _summarize_samples(samples_us: Sequence[float]) -> Dict[str, float]:
    return {
        "median_us": statistics.median(samples_us),
        "mean_us": statistics.fmean(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


def _case_id(module_name: str, mode: str, direction: str) -> str:
    return f"{module_name}:{mode}:{direction}"


def _skip_case(
    module_name: str,
    mode: str,
    direction: str,
    reason: str,
    config: BenchmarkConfig,
) -> Dict[str, object]:
    return {
        "case_id": _case_id(module_name, mode, direction),
        "module": module_name,
        "mode": mode,
        "direction": direction,
        "status": "skipped",
        "skip_reason": reason,
        "input_shape": [config.batch_size, config.hidden_size],
        "hidden_size": config.hidden_size,
        "ffn_hidden_size": config.ffn_hidden_size if module_name == "layernorm_mlp" else None,
        "te_us": None,
        "torch_us": None,
        "ratio": None,
        "goal_ratio": GOAL_RATIOS[mode],
        "passed": None,
        "samples": 0,
        "te_samples_us": [],
        "torch_samples_us": [],
    }


def _run_case(
    module_name: str,
    mode: str,
    direction: str,
    config: BenchmarkConfig,
    te_module,
    mxfp8_recipe,
    device: str,
) -> Dict[str, object]:
    te_mod, torch_mod = _make_modules(module_name, config, te_module, device)
    te_mod.train()
    torch_mod.train()

    torch_inp = _make_input(config, device)
    te_inp = torch_inp.detach().clone().requires_grad_(True)

    torch_samples = _measure_invocations(
        torch_mod,
        torch_inp,
        direction,
        config.warmup,
        config.iterations,
        profile=False,
        label=f"torch_{_case_id(module_name, 'bf16', direction)}",
    )

    profile_this_case = (
        config.profile
        and module_name == config.profile_module
        and mode == config.profile_mode
        and direction == config.profile_direction
    )
    with _mode_context(te_module, mode, mxfp8_recipe):
        te_samples = _measure_invocations(
            te_mod,
            te_inp,
            direction,
            config.warmup,
            config.iterations,
            profile=profile_this_case,
            label=f"te_{_case_id(module_name, mode, direction)}",
        )

    torch_summary = _summarize_samples(torch_samples)
    te_summary = _summarize_samples(te_samples)
    te_us = te_summary["median_us"]
    torch_us = torch_summary["median_us"]
    ratio = te_us / torch_us if torch_us > 0 else float("inf")
    goal_ratio = GOAL_RATIOS[mode]

    return {
        "case_id": _case_id(module_name, mode, direction),
        "module": module_name,
        "mode": mode,
        "direction": direction,
        "status": "measured",
        "skip_reason": None,
        "input_shape": [config.batch_size, config.hidden_size],
        "hidden_size": config.hidden_size,
        "ffn_hidden_size": config.ffn_hidden_size if module_name == "layernorm_mlp" else None,
        "te_us": te_us,
        "torch_us": torch_us,
        "ratio": ratio,
        "goal_ratio": goal_ratio,
        "passed": ratio <= goal_ratio,
        "samples": config.iterations,
        "te_summary": te_summary,
        "torch_summary": torch_summary,
        "te_samples_us": te_samples,
        "torch_samples_us": torch_samples,
    }


def _metadata(config: BenchmarkConfig) -> Dict[str, object]:
    cuda_available = torch is not None and torch.cuda.is_available()
    device_name = torch.cuda.get_device_name() if cuda_available else None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if torch is not None else None,
        "cuda_device_name": device_name,
        "pid": os.getpid(),
        "config": {
            "modules": list(config.modules),
            "modes": list(config.modes),
            "directions": list(config.directions),
            "batch_size": config.batch_size,
            "hidden_size": config.hidden_size,
            "ffn_hidden_size": config.ffn_hidden_size,
            "warmup": config.warmup,
            "iterations": config.iterations,
            "profile": config.profile,
            "profile_module": config.profile_module,
            "profile_mode": config.profile_mode,
            "profile_direction": config.profile_direction,
            "dry_run": config.dry_run,
        },
    }


def run_benchmark(config: BenchmarkConfig) -> Dict[str, object]:
    metadata = _metadata(config)
    cases: List[Dict[str, object]] = []

    if config.dry_run:
        for module_name in config.modules:
            for mode in config.modes:
                for direction in config.directions:
                    cases.append(_skip_case(module_name, mode, direction, "dry_run", config))
        return {"schema_version": "te_module_cpu_overhead/v1", "metadata": metadata, "cases": cases}

    if torch is None:
        raise RuntimeError("PyTorch is required for module CPU-overhead benchmarking")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for module CPU-overhead benchmarking")

    te_module, recipe_cls = _import_te()
    mxfp8_available, reason_for_no_mxfp8 = te_module.is_mxfp8_available(return_reason=True)
    metadata["transformer_engine_version"] = getattr(te_module, "__version__", None)
    metadata["mxfp8_available"] = mxfp8_available
    metadata["mxfp8_unavailable_reason"] = reason_for_no_mxfp8
    mxfp8_recipe = recipe_cls()
    device = "cuda"

    for module_name in config.modules:
        for mode in config.modes:
            for direction in config.directions:
                if mode == "mxfp8" and not mxfp8_available:
                    cases.append(
                        _skip_case(
                            module_name,
                            mode,
                            direction,
                            f"MXFP8 unavailable: {reason_for_no_mxfp8}",
                            config,
                        )
                    )
                    continue
                cases.append(
                    _run_case(
                        module_name,
                        mode,
                        direction,
                        config,
                        te_module,
                        mxfp8_recipe,
                        device,
                    )
                )

    return {"schema_version": "te_module_cpu_overhead/v1", "metadata": metadata, "cases": cases}


def _format_us(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}x"


def make_report(result: Dict[str, object]) -> str:
    lines = [
        "# Transformer Engine Tiny Module CPU Overhead",
        "",
        "| Case | TE us/invocation | Torch BF16 us/invocation | Ratio | Goal | Status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in result["cases"]:
        if case["status"] == "skipped":
            status = f"SKIP: {case['skip_reason']}"
        else:
            status = "PASS" if case["passed"] else "FAIL"
        lines.append(
            "| "
            f"{case['case_id']} | "
            f"{_format_us(case['te_us'])} | "
            f"{_format_us(case['torch_us'])} | "
            f"{_format_ratio(case['ratio'])} | "
            f"{case['goal_ratio']:.1f}x | "
            f"{status} |"
        )
    lines.append("")
    lines.append(
        "Timings measure CPU wall-clock enqueue overhead. CUDA synchronization happens "
        "before and after each sample but is outside the timed interval."
    )
    return "\n".join(lines)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = _parse_args(argv)
    result = run_benchmark(config)
    report = make_report(result)

    if config.json_output is not None:
        _write_text(config.json_output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    if config.report_output is not None:
        _write_text(config.report_output, report + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
