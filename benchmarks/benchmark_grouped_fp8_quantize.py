# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

te = None
tex = None
torch = None
benchmark = None
Float8Quantizer = None


def _prepend_repo_to_sys_path():
    repo_root = str(_REPO_ROOT)
    if sys.path[0] != repo_root:
        try:
            sys.path.remove(repo_root)
        except ValueError:
            pass
        sys.path.insert(0, repo_root)


def _is_under_repo(path):
    if path is None:
        return False
    try:
        Path(path).resolve().relative_to(_REPO_ROOT)
    except ValueError:
        return False
    return True


def _candidate_paths_ok(paths):
    return _is_under_repo(paths.get("transformer_engine")) and _is_under_repo(
        paths.get("transformer_engine_torch")
    )


def _format_paths(paths):
    return ", ".join(
        f"{name}={paths.get(name, '<unavailable>')}"
        for name in ("transformer_engine", "transformer_engine_torch")
    )


def _probe_transformer_engine_paths():
    # Probe in a child process so a stale extension is never loaded in the benchmark process.
    env = os.environ.copy()
    env["NVTE_FRAMEWORK"] = "pytorch"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    code = f"""
import json
import sys
sys.path.insert(0, {str(_REPO_ROOT)!r})
import transformer_engine
import transformer_engine_torch as tex
print("NVTE_BENCHMARK_IMPORT_PATHS=" + json.dumps({{
    "transformer_engine": getattr(transformer_engine, "__file__", None),
    "transformer_engine_torch": getattr(tex, "__file__", None),
}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("NVTE_BENCHMARK_IMPORT_PATHS="):
            return json.loads(line.split("=", 1)[1]), None
    error = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
    return {}, error


def _install_candidate_checkout():
    env = os.environ.copy()
    env["NVTE_FRAMEWORK"] = "pytorch"
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "-v",
        "-e",
        str(_REPO_ROOT),
    ]
    subprocess.run(command, cwd=_REPO_ROOT, env=env, check=True)


def _ensure_candidate_checkout_extension():
    os.environ["NVTE_FRAMEWORK"] = "pytorch"
    _prepend_repo_to_sys_path()
    paths, error = _probe_transformer_engine_paths()
    if _candidate_paths_ok(paths):
        return

    if os.getenv("NVTE_BENCHMARK_SKIP_CANDIDATE_INSTALL", "0") == "1":
        details = error if error is not None else _format_paths(paths)
        raise RuntimeError(
            "Grouped FP8 benchmark is not using the candidate Transformer Engine checkout "
            f"{_REPO_ROOT}: {details}"
        )

    _install_candidate_checkout()
    paths, error = _probe_transformer_engine_paths()
    if not _candidate_paths_ok(paths):
        details = error if error is not None else _format_paths(paths)
        raise RuntimeError(
            "Candidate Transformer Engine install completed, but the grouped FP8 benchmark "
            f"still is not using modules from {_REPO_ROOT}: {details}"
        )


def _load_transformer_engine_modules():
    global te, tex, torch, benchmark, Float8Quantizer

    _ensure_candidate_checkout_extension()

    import torch as torch_module
    import torch.utils.benchmark as benchmark_module

    import transformer_engine as te_module
    import transformer_engine_torch as tex_module
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer as Quantizer

    paths = {
        "transformer_engine": getattr(te_module, "__file__", None),
        "transformer_engine_torch": getattr(tex_module, "__file__", None),
    }
    if not _candidate_paths_ok(paths):
        raise RuntimeError(
            "Grouped FP8 benchmark imported Transformer Engine modules outside the candidate "
            f"checkout {_REPO_ROOT}: {_format_paths(paths)}"
        )

    te = te_module
    tex = tex_module
    torch = torch_module
    benchmark = benchmark_module
    Float8Quantizer = Quantizer


def _module_path_metadata():
    return {
        "candidate_repo": str(_REPO_ROOT),
        "transformer_engine_path": str(Path(te.__file__).resolve()),
        "transformer_engine_torch_path": str(Path(tex.__file__).resolve()),
    }


def run_case(num_tensors, cols, first_dims, allocation_rows, min_run_time):
    if tex is None:
        _load_transformer_engine_modules()

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
        **_module_path_metadata(),
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
