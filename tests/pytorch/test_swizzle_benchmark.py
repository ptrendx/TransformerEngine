# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import sys

import pytest

import benchmarks.swizzle.benchmark_swizzle as benchmark_swizzle


class _FakeDevice:
    def __init__(self, device_type, index=None):
        self.type = device_type
        self.index = index

    def __str__(self):
        if self.index is None:
            return self.type
        return f"{self.type}:{self.index}"


class _FakeCuda:
    def __init__(self, current_device):
        self._current_device = current_device
        self.set_device_calls = []

    def current_device(self):
        return self._current_device

    def set_device(self, device):
        self.set_device_calls.append(device)


class _FakeTorch:
    def __init__(self, current_device=0):
        self.cuda = _FakeCuda(current_device)

    def device(self, device_arg, index=None):
        if isinstance(device_arg, _FakeDevice):
            return device_arg
        if index is not None:
            return _FakeDevice(device_arg, index)
        if ":" in device_arg:
            device_type, device_index = device_arg.split(":", 1)
            return _FakeDevice(device_type, int(device_index))
        return _FakeDevice(device_arg)


def test_parse_args_defaults_to_indexed_cuda_device(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark_swizzle.py"])

    args = benchmark_swizzle.parse_args()

    assert args.device == "cuda:0"


@pytest.mark.parametrize(
    ("device_arg", "current_device", "expected_index"),
    [
        ("cuda", 2, 2),
        ("cuda:1", 0, 1),
        (_FakeDevice("cuda", 4), 0, 4),
        ("0", 3, 0),
        (0, 3, 0),
    ],
)
def test_resolve_cuda_device_returns_indexed_device(
    monkeypatch, device_arg, current_device, expected_index
):
    fake_torch = _FakeTorch(current_device)
    monkeypatch.setattr(benchmark_swizzle, "torch", fake_torch, raising=False)

    device = benchmark_swizzle.resolve_cuda_device(device_arg)

    assert device.type == "cuda"
    assert device.index == expected_index
    assert fake_torch.cuda.set_device_calls == [expected_index]


def test_resolve_cuda_device_handles_bare_torch_device(monkeypatch):
    fake_torch = _FakeTorch(current_device=1)
    monkeypatch.setattr(benchmark_swizzle, "torch", fake_torch, raising=False)

    device = benchmark_swizzle.resolve_cuda_device(_FakeDevice("cuda"))

    assert device.type == "cuda"
    assert device.index == 1
    assert fake_torch.cuda.set_device_calls == [1]


def test_resolve_cuda_device_rejects_non_cuda_device(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(benchmark_swizzle, "torch", fake_torch, raising=False)

    with pytest.raises(ValueError, match="CUDA device is required"):
        benchmark_swizzle.resolve_cuda_device("cpu")

    assert fake_torch.cuda.set_device_calls == []


def test_build_cases_includes_grouped_combined_modes():
    cases = {case.name: case for case in benchmark_swizzle.build_cases()}

    assert cases["grouped_uniform_rowwise_columnwise_small"].direction == (
        benchmark_swizzle.COMBINED_DIRECTION
    )
    assert cases["grouped_uniform_rowwise_columnwise_large"].direction == (
        benchmark_swizzle.COMBINED_DIRECTION
    )
    assert cases["grouped_variable_rowwise_columnwise_small"].direction == (
        benchmark_swizzle.COMBINED_DIRECTION
    )
    assert cases["grouped_variable_rowwise_columnwise_large"].direction == (
        benchmark_swizzle.COMBINED_DIRECTION
    )
    assert benchmark_swizzle.scale_directions(benchmark_swizzle.COMBINED_DIRECTION) == (
        "rowwise",
        "columnwise",
    )


@pytest.mark.parametrize(
    ("case_name", "expected_classification"),
    [
        ("grouped_uniform_rowwise_columnwise_small", "small"),
        ("grouped_uniform_rowwise_columnwise_large", "large"),
        ("grouped_variable_rowwise_columnwise_small", "small"),
        ("grouped_variable_rowwise_columnwise_large", "large"),
    ],
)
def test_combined_grouped_case_byte_accounting(case_name, expected_classification):
    cases = {case.name: case for case in benchmark_swizzle.build_cases()}
    spec = cases[case_name]

    input_bytes, output_bytes, processed_bytes = benchmark_swizzle.case_scale_bytes(spec)
    row_bytes = benchmark_swizzle.case_direction_scale_bytes(spec, "rowwise")
    column_bytes = benchmark_swizzle.case_direction_scale_bytes(spec, "columnwise")

    assert input_bytes == row_bytes[0] + column_bytes[0]
    assert output_bytes == row_bytes[1] + column_bytes[1]
    assert processed_bytes == row_bytes[2] + column_bytes[2]
    assert (output_bytes > benchmark_swizzle.SMALL_OUTPUT_LIMIT_BYTES) == (
        expected_classification == "large"
    )


def test_small_cases_use_expanded_footprint():
    cases = {case.name: case for case in benchmark_swizzle.build_cases()}

    small_case_names = [case_name for case_name in cases if case_name.endswith("_small")]
    for case_name in small_case_names:
        _, output_bytes, _ = benchmark_swizzle.case_scale_bytes(cases[case_name])
        assert 16 * 1024 * 1024 <= output_bytes <= benchmark_swizzle.SMALL_OUTPUT_LIMIT_BYTES
