# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

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
        if index is not None:
            return _FakeDevice(device_arg, index)
        if ":" in device_arg:
            device_type, device_index = device_arg.split(":", 1)
            return _FakeDevice(device_type, int(device_index))
        return _FakeDevice(device_arg)


@pytest.mark.parametrize(
    ("device_arg", "current_device", "expected_index"),
    [
        ("cuda", 2, 2),
        ("cuda:1", 0, 1),
        ("0", 3, 0),
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


def test_resolve_cuda_device_rejects_non_cuda_device(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(benchmark_swizzle, "torch", fake_torch, raising=False)

    with pytest.raises(ValueError, match="CUDA device is required"):
        benchmark_swizzle.resolve_cuda_device("cpu")

    assert fake_torch.cuda.set_device_calls == []
