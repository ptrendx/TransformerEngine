# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import json
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None


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


def _assert_raw_report(data, expected_case_ids=None):
    assert data["schema_version"] == "benchmark_raw_report/v1"
    required_keys = {
        "command",
        "working_directory",
        "repo_ref",
        "cluster_name",
        "gpu_type",
        "benchmark_output_path",
        "exit_code",
        "measurements",
    }
    optional_keys = {"environment_artifact_uri", "stdout_uri", "stderr_uri"}
    assert required_keys <= set(data)
    assert set(data) <= required_keys | optional_keys
    assert data["exit_code"] == 0

    measurement_keys = {
        "case_id",
        "metric",
        "value",
        "unit",
        "iteration",
        "higher_is_better",
    }
    for measurement in data["measurements"]:
        assert set(measurement) == measurement_keys
        assert isinstance(measurement["value"], (int, float))

    if expected_case_ids is not None:
        measured_case_ids = {
            measurement["case_id"]
            for measurement in data["measurements"]
            if measurement["metric"] == "te_us"
        }
        assert measured_case_ids == set(expected_case_ids)


def test_module_cpu_overhead_benchmark_dry_run_outputs_case_matrix(tmp_path):
    raw_output = tmp_path / "raw.json"
    detail_output = tmp_path / "detail.json"
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
            str(raw_output),
            "--detail-json-output",
            str(detail_output),
            "--report-output",
            str(report_output),
        ]
    )

    raw_data = json.loads(raw_output.read_text(encoding="utf-8"))
    _assert_raw_report(raw_data)
    assert raw_data["measurements"] == []

    data = json.loads(detail_output.read_text(encoding="utf-8"))
    assert data["schema_version"] == "te_module_cpu_overhead/v1"
    assert [case["case_id"] for case in data["cases"]] == [
        "linear:bf16:forward",
        "linear:bf16:backward",
    ]
    assert all(case["status"] == "skipped" for case in data["cases"])
    assert "linear:bf16:forward" in report_output.read_text(encoding="utf-8")


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="CUDA is required")
def test_module_cpu_overhead_benchmark_cuda_bf16_smoke(tmp_path):
    raw_output = tmp_path / "raw.json"
    detail_output = tmp_path / "detail.json"

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
            str(raw_output),
            "--detail-json-output",
            str(detail_output),
        ]
    )

    data = json.loads(detail_output.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in data["cases"]}
    expected_cases = {
        "linear:bf16:forward",
        "linear:bf16:backward",
        "layernorm_linear:bf16:forward",
        "layernorm_linear:bf16:backward",
        "layernorm_mlp:bf16:forward",
        "layernorm_mlp:bf16:backward",
    }
    assert set(cases) == expected_cases
    for case in cases.values():
        assert case["status"] == "measured"
        assert case["samples"] == 1
        assert case["te_us"] > 0
        assert case["torch_us"] > 0

    raw_data = json.loads(raw_output.read_text(encoding="utf-8"))
    _assert_raw_report(raw_data, expected_cases)
    assert {
        (measurement["case_id"], measurement["metric"])
        for measurement in raw_data["measurements"]
    } >= {(case_id, "te_vs_torch_ratio") for case_id in expected_cases}


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="CUDA is required")
def test_module_cpu_overhead_benchmark_cuda_mxfp8_smoke(tmp_path):
    import transformer_engine.pytorch as te

    mxfp8_available, reason = te.is_mxfp8_available(return_reason=True)
    if not mxfp8_available:
        pytest.skip(f"MXFP8 unavailable: {reason}")

    raw_output = tmp_path / "raw.json"
    detail_output = tmp_path / "detail.json"

    _run_benchmark(
        [
            "--modules",
            "all",
            "--modes",
            "mxfp8",
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
            str(raw_output),
            "--detail-json-output",
            str(detail_output),
        ]
    )

    data = json.loads(detail_output.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in data["cases"]}
    expected_cases = {
        "linear:mxfp8:forward",
        "linear:mxfp8:backward",
        "layernorm_linear:mxfp8:forward",
        "layernorm_linear:mxfp8:backward",
        "layernorm_mlp:mxfp8:forward",
        "layernorm_mlp:mxfp8:backward",
    }
    assert set(cases) == expected_cases
    for case in cases.values():
        assert case["status"] == "measured"
        assert case["samples"] == 1
        assert case["te_us"] > 0
        assert case["torch_us"] > 0

    raw_data = json.loads(raw_output.read_text(encoding="utf-8"))
    _assert_raw_report(raw_data, expected_cases)
