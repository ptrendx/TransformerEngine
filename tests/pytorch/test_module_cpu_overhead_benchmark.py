# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = REPO_ROOT / "benchmarks" / "pytorch" / "benchmark_module_cpu_overhead.py"


def _run_benchmark(args):
    return subprocess.run(
        [sys.executable, str(BENCHMARK), *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def test_module_cpu_overhead_benchmark_dry_run_outputs_case_matrix(tmp_path):
    json_output = tmp_path / "result.json"
    report_output = tmp_path / "result.md"

    _run_benchmark(
        [
            "--dry-run",
            "--modules",
            "linear",
            "--modes",
            "bf16",
            "--directions",
            "forward,backward",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--json-output",
            str(json_output),
            "--report-output",
            str(report_output),
        ]
    )

    data = json.loads(json_output.read_text(encoding="utf-8"))
    assert data["schema_version"] == "te_module_cpu_overhead/v1"
    assert [case["case_id"] for case in data["cases"]] == [
        "linear:bf16:forward",
        "linear:bf16:backward",
    ]
    assert all(case["status"] == "skipped" for case in data["cases"])
    assert "linear:bf16:forward" in report_output.read_text(encoding="utf-8")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_module_cpu_overhead_benchmark_cuda_bf16_smoke(tmp_path):
    json_output = tmp_path / "result.json"

    _run_benchmark(
        [
            "--modules",
            "all",
            "--modes",
            "bf16",
            "--directions",
            "forward,backward",
            "--batch-size",
            "32",
            "--hidden-size",
            "32",
            "--ffn-hidden-size",
            "128",
            "--warmup",
            "1",
            "--iterations",
            "1",
            "--json-output",
            str(json_output),
        ]
    )

    data = json.loads(json_output.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in data["cases"]}
    assert set(cases) == {
        "linear:bf16:forward",
        "linear:bf16:backward",
        "layernorm_linear:bf16:forward",
        "layernorm_linear:bf16:backward",
        "layernorm_mlp:bf16:forward",
        "layernorm_mlp:bf16:backward",
    }
    for case in cases.values():
        assert case["status"] == "measured"
        assert case["samples"] == 1
        assert case["te_us"] > 0
        assert case["torch_us"] > 0
