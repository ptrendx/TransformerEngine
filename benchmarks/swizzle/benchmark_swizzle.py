#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark nvte swizzle and grouped swizzle scale kernels.

The benchmark calls the public C API through ctypes so timing covers the same
entry points used by framework wrappers while reusing preallocated scale buffers.
Only scale buffers are allocated at full size; data tensors are represented by a
one-byte dummy allocation plus the logical shape metadata because swizzle kernels
only inspect data shapes.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


K_NVTE_FLOAT8_E4M3 = 7
K_NVTE_FLOAT8_E8M0 = 9
K_NVTE_INT64 = 3
NVTE_MXFP8_1D_SCALING = 1

K_NVTE_ROWWISE_DATA = 0
K_NVTE_COLUMNWISE_DATA = 1
K_NVTE_ROWWISE_SCALE_INV = 4
K_NVTE_COLUMNWISE_SCALE_INV = 5
K_NVTE_WITH_GEMM_SWIZZLED_SCALES = 7

K_NVTE_GROUPED_ROWWISE_DATA = 0
K_NVTE_GROUPED_COLUMNWISE_DATA = 1
K_NVTE_GROUPED_ROWWISE_SCALE_INV = 4
K_NVTE_GROUPED_COLUMNWISE_SCALE_INV = 5
K_NVTE_GROUPED_FIRST_DIMS = 7
K_NVTE_GROUPED_TENSOR_OFFSETS = 9
K_NVTE_GROUPED_WITH_GEMM_SWIZZLED_SCALES = 10

MXFP8_BLOCK_SIZE = 32
SWIZZLE_TILE_M = 128
SWIZZLE_TILE_K = 4
DEFAULT_CUDA_DEVICE = "cuda:0"
ROWWISE_DIRECTION = "rowwise"
COLUMNWISE_DIRECTION = "columnwise"
COMBINED_DIRECTION = "rowwise+columnwise"
SCALE_DIRECTIONS = (ROWWISE_DIRECTION, COLUMNWISE_DIRECTION)


class NVTEShape(ctypes.Structure):
    _fields_ = [("data", ctypes.c_size_t * 15), ("ndim", ctypes.c_size_t)]


class NVTEBasicTensor(ctypes.Structure):
    _fields_ = [
        ("data_ptr", ctypes.c_void_p),
        ("dtype", ctypes.c_int),
        ("shape", NVTEShape),
    ]


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def make_shape(dims: Iterable[int]) -> NVTEShape:
    shape = NVTEShape()
    dims = list(dims)
    if len(dims) > len(shape.data):
        raise ValueError(f"NVTEShape supports at most {len(shape.data)} dims, got {len(dims)}")
    shape.ndim = len(dims)
    for idx, dim in enumerate(dims):
        shape.data[idx] = int(dim)
    return shape


def make_basic_tensor(ptr: int | None, dtype: int, dims: Iterable[int]) -> NVTEBasicTensor:
    return NVTEBasicTensor(ctypes.c_void_p(ptr or 0), dtype, make_shape(dims))


class NvteAPI:
    def __init__(self) -> None:
        import transformer_engine.common as te_common

        self.lib = te_common._load_core_library()
        self.lib.nvte_create_tensor.argtypes = [ctypes.c_int]
        self.lib.nvte_create_tensor.restype = ctypes.c_void_p
        self.lib.nvte_destroy_tensor.argtypes = [ctypes.c_void_p]
        self.lib.nvte_destroy_tensor.restype = None
        self.lib.nvte_set_tensor_param_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.nvte_set_tensor_param_v2.restype = None
        self.lib.nvte_swizzle_scaling_factors.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.nvte_swizzle_scaling_factors.restype = None

        self.lib.nvte_create_grouped_tensor.argtypes = [
            ctypes.c_int,
            ctypes.c_size_t,
            NVTEShape,
        ]
        self.lib.nvte_create_grouped_tensor.restype = ctypes.c_void_p
        self.lib.nvte_destroy_grouped_tensor.argtypes = [ctypes.c_void_p]
        self.lib.nvte_destroy_grouped_tensor.restype = None
        self.lib.nvte_set_grouped_tensor_param.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.nvte_set_grouped_tensor_param.restype = None
        self.lib.nvte_swizzle_grouped_scaling_factors.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.nvte_swizzle_grouped_scaling_factors.restype = None

    def create_tensor(self) -> ctypes.c_void_p:
        return self.lib.nvte_create_tensor(NVTE_MXFP8_1D_SCALING)

    def set_tensor_basic(
        self, handle: ctypes.c_void_p, param: int, ptr: int | None, dtype: int, dims: Iterable[int]
    ) -> None:
        tensor = make_basic_tensor(ptr, dtype, dims)
        self.lib.nvte_set_tensor_param_v2(
            handle, param, ctypes.byref(tensor), ctypes.sizeof(tensor)
        )

    def set_tensor_bool(self, handle: ctypes.c_void_p, param: int, value: bool) -> None:
        raw = ctypes.c_uint8(1 if value else 0)
        self.lib.nvte_set_tensor_param_v2(handle, param, ctypes.byref(raw), ctypes.sizeof(raw))

    def create_grouped_tensor(
        self, num_tensors: int, logical_shape: Iterable[int]
    ) -> ctypes.c_void_p:
        return self.lib.nvte_create_grouped_tensor(
            NVTE_MXFP8_1D_SCALING, num_tensors, make_shape(logical_shape)
        )

    def set_grouped_basic(
        self, handle: ctypes.c_void_p, param: int, ptr: int | None, dtype: int, dims: Iterable[int]
    ) -> None:
        tensor = make_basic_tensor(ptr, dtype, dims)
        self.lib.nvte_set_grouped_tensor_param(
            handle, param, ctypes.byref(tensor), ctypes.sizeof(tensor)
        )

    def set_grouped_bool(self, handle: ctypes.c_void_p, param: int, value: bool) -> None:
        raw = ctypes.c_uint8(1 if value else 0)
        self.lib.nvte_set_grouped_tensor_param(
            handle, param, ctypes.byref(raw), ctypes.sizeof(raw)
        )

    def swizzle(self, input_handle: ctypes.c_void_p, output_handle: ctypes.c_void_p) -> None:
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self.lib.nvte_swizzle_scaling_factors(input_handle, output_handle, stream)

    def grouped_swizzle(
        self, input_handle: ctypes.c_void_p, output_handle: ctypes.c_void_p
    ) -> None:
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self.lib.nvte_swizzle_grouped_scaling_factors(input_handle, output_handle, stream)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    api: str
    direction: str
    data_shapes: tuple[tuple[int, int], ...]
    variable_first_dims: bool = False

    @property
    def num_tensors(self) -> int:
        return len(self.data_shapes)


@dataclass
class BenchBuffer:
    input_handle: ctypes.c_void_p
    output_handle: ctypes.c_void_p
    tensors: list[torch.Tensor]


def scale_directions(direction: str) -> tuple[str, ...]:
    if direction == COMBINED_DIRECTION:
        return SCALE_DIRECTIONS
    if direction in SCALE_DIRECTIONS:
        return (direction,)
    raise ValueError(f"Unsupported scale direction: {direction}")


def regular_scale_shape(shape: tuple[int, int], direction: str) -> tuple[int, int]:
    m, k = shape
    if direction == ROWWISE_DIRECTION:
        return round_up(m, SWIZZLE_TILE_M), round_up(
            math.ceil(k / MXFP8_BLOCK_SIZE), SWIZZLE_TILE_K
        )
    if direction != COLUMNWISE_DIRECTION:
        raise ValueError(f"Regular swizzle supports one scale direction, got {direction}")
    return round_up(math.ceil(m / MXFP8_BLOCK_SIZE), SWIZZLE_TILE_K), round_up(k, SWIZZLE_TILE_M)


def grouped_output_scale_elems_for_direction(
    shapes: tuple[tuple[int, int], ...], direction: str
) -> int:
    elems = 0
    for shape in shapes:
        scale_shape = regular_scale_shape(shape, direction)
        elems += product(scale_shape)
    return elems


def grouped_output_scale_elems(shapes: tuple[tuple[int, int], ...], direction: str) -> int:
    return sum(
        grouped_output_scale_elems_for_direction(shapes, scale_direction)
        for scale_direction in scale_directions(direction)
    )


def grouped_uniform_input_scale_elems_for_direction(
    num_tensors: int, shape: tuple[int, int], direction: str
) -> int:
    m, k = shape
    if direction == ROWWISE_DIRECTION:
        padded_k = round_up(math.ceil(k / MXFP8_BLOCK_SIZE), SWIZZLE_TILE_K)
        return round_up(num_tensors * m, SWIZZLE_TILE_M) * padded_k
    if direction != COLUMNWISE_DIRECTION:
        raise ValueError(f"Grouped uniform swizzle supports one scale direction, got {direction}")
    padded_m = round_up(k, SWIZZLE_TILE_M)
    k_scales = math.ceil(m / MXFP8_BLOCK_SIZE)
    return round_up(num_tensors * k_scales, SWIZZLE_TILE_K) * padded_m


def grouped_uniform_input_scale_elems(
    num_tensors: int, shape: tuple[int, int], direction: str
) -> int:
    return sum(
        grouped_uniform_input_scale_elems_for_direction(num_tensors, shape, scale_direction)
        for scale_direction in scale_directions(direction)
    )


def case_direction_scale_bytes(spec: CaseSpec, direction: str) -> tuple[int, int, int]:
    output_elems = grouped_output_scale_elems_for_direction(spec.data_shapes, direction)
    if spec.api == "regular":
        input_elems = output_elems
    elif spec.api == "grouped_uniform":
        input_elems = grouped_uniform_input_scale_elems_for_direction(
            spec.num_tensors, spec.data_shapes[0], direction
        )
    else:
        input_elems = output_elems
    return input_elems, output_elems, input_elems + output_elems


def case_scale_bytes(spec: CaseSpec) -> tuple[int, int, int]:
    input_elems = 0
    output_elems = 0
    processed_elems = 0
    for direction in scale_directions(spec.direction):
        direction_input, direction_output, direction_processed = case_direction_scale_bytes(
            spec, direction
        )
        input_elems += direction_input
        output_elems += direction_output
        processed_elems += direction_processed
    return input_elems, output_elems, processed_elems


def make_regular_buffer(api: NvteAPI, spec: CaseSpec, device: torch.device) -> BenchBuffer:
    directions = scale_directions(spec.direction)
    if len(directions) != 1:
        raise ValueError("Regular swizzle benchmark cases must use exactly one scale direction")
    direction = directions[0]
    shape = spec.data_shapes[0]
    scale_shape = regular_scale_shape(shape, direction)
    dummy_data = torch.empty(1, dtype=torch.uint8, device=device)
    input_scale = torch.empty(scale_shape, dtype=torch.uint8, device=device)
    output_scale = torch.empty(scale_shape, dtype=torch.uint8, device=device)

    input_handle = api.create_tensor()
    output_handle = api.create_tensor()
    data_param = K_NVTE_ROWWISE_DATA if direction == ROWWISE_DIRECTION else K_NVTE_COLUMNWISE_DATA
    scale_param = (
        K_NVTE_ROWWISE_SCALE_INV
        if direction == ROWWISE_DIRECTION
        else K_NVTE_COLUMNWISE_SCALE_INV
    )
    api.set_tensor_basic(input_handle, data_param, dummy_data.data_ptr(), K_NVTE_FLOAT8_E4M3, shape)
    api.set_tensor_basic(
        output_handle, data_param, dummy_data.data_ptr(), K_NVTE_FLOAT8_E4M3, shape
    )
    api.set_tensor_basic(
        input_handle, scale_param, input_scale.data_ptr(), K_NVTE_FLOAT8_E8M0, scale_shape
    )
    api.set_tensor_basic(
        output_handle, scale_param, output_scale.data_ptr(), K_NVTE_FLOAT8_E8M0, scale_shape
    )
    api.set_tensor_bool(input_handle, K_NVTE_WITH_GEMM_SWIZZLED_SCALES, False)
    api.set_tensor_bool(output_handle, K_NVTE_WITH_GEMM_SWIZZLED_SCALES, True)
    return BenchBuffer(input_handle, output_handle, [dummy_data, input_scale, output_scale])


def make_grouped_buffer(api: NvteAPI, spec: CaseSpec, device: torch.device) -> BenchBuffer:
    num_tensors = spec.num_tensors
    first_dims = [shape[0] for shape in spec.data_shapes]
    last_dims = [shape[1] for shape in spec.data_shapes]
    if len(set(last_dims)) != 1:
        raise ValueError("This benchmark driver expects grouped cases with a common last dim")
    logical_shape = (sum(first_dims), last_dims[0])
    dummy_data = torch.empty(1, dtype=torch.uint8, device=device)
    tensors = [dummy_data]

    first_dims_tensor = None
    tensor_offsets = None
    if spec.variable_first_dims:
        first_dims_tensor = torch.tensor(first_dims, dtype=torch.int64, device=device)
        offsets = [0]
        for m, k in spec.data_shapes:
            offsets.append(offsets[-1] + m * k)
        tensor_offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
        tensors.append(first_dims_tensor)
        tensors.append(tensor_offsets)

    input_handle = api.create_grouped_tensor(num_tensors, logical_shape)
    output_handle = api.create_grouped_tensor(num_tensors, logical_shape)

    flat_data_shape = (sum(m * k for m, k in spec.data_shapes),)
    for direction in scale_directions(spec.direction):
        input_elems, output_elems, _ = case_direction_scale_bytes(spec, direction)
        input_scale = torch.empty(input_elems, dtype=torch.uint8, device=device)
        output_scale = torch.empty(output_elems, dtype=torch.uint8, device=device)
        tensors.extend([input_scale, output_scale])

        data_param = (
            K_NVTE_GROUPED_ROWWISE_DATA
            if direction == ROWWISE_DIRECTION
            else K_NVTE_GROUPED_COLUMNWISE_DATA
        )
        scale_param = (
            K_NVTE_GROUPED_ROWWISE_SCALE_INV
            if direction == ROWWISE_DIRECTION
            else K_NVTE_GROUPED_COLUMNWISE_SCALE_INV
        )
        api.set_grouped_basic(
            input_handle, data_param, dummy_data.data_ptr(), K_NVTE_FLOAT8_E4M3, flat_data_shape
        )
        api.set_grouped_basic(
            output_handle, data_param, dummy_data.data_ptr(), K_NVTE_FLOAT8_E4M3, flat_data_shape
        )
        api.set_grouped_basic(
            input_handle, scale_param, input_scale.data_ptr(), K_NVTE_FLOAT8_E8M0, (input_elems,)
        )
        api.set_grouped_basic(
            output_handle,
            scale_param,
            output_scale.data_ptr(),
            K_NVTE_FLOAT8_E8M0,
            (output_elems,),
        )
    if first_dims_tensor is not None:
        assert tensor_offsets is not None
        api.set_grouped_basic(
            input_handle,
            K_NVTE_GROUPED_FIRST_DIMS,
            first_dims_tensor.data_ptr(),
            K_NVTE_INT64,
            (num_tensors,),
        )
        api.set_grouped_basic(
            output_handle,
            K_NVTE_GROUPED_FIRST_DIMS,
            first_dims_tensor.data_ptr(),
            K_NVTE_INT64,
            (num_tensors,),
        )
        api.set_grouped_basic(
            input_handle,
            K_NVTE_GROUPED_TENSOR_OFFSETS,
            tensor_offsets.data_ptr(),
            K_NVTE_INT64,
            (num_tensors + 1,),
        )
        api.set_grouped_basic(
            output_handle,
            K_NVTE_GROUPED_TENSOR_OFFSETS,
            tensor_offsets.data_ptr(),
            K_NVTE_INT64,
            (num_tensors + 1,),
        )
    api.set_grouped_bool(input_handle, K_NVTE_GROUPED_WITH_GEMM_SWIZZLED_SCALES, False)
    api.set_grouped_bool(output_handle, K_NVTE_GROUPED_WITH_GEMM_SWIZZLED_SCALES, True)
    return BenchBuffer(input_handle, output_handle, tensors)


def make_case_buffer(api: NvteAPI, spec: CaseSpec, device: torch.device) -> BenchBuffer:
    if spec.api == "regular":
        return make_regular_buffer(api, spec, device)
    return make_grouped_buffer(api, spec, device)


def build_cases() -> list[CaseSpec]:
    uniform_combined_small = ((16384, 1024),) * 8
    uniform_combined_large = ((65536, 4096),) * 8
    variable_small = (
        (16384, 1024),
        (24576, 1024),
        (32768, 1024),
        (40960, 1024),
        (49152, 1024),
        (57344, 1024),
        (16384, 1024),
        (24576, 1024),
    )
    variable_combined_small = tuple((m // 2, k) for m, k in variable_small)
    variable_large = tuple((m * 8, k) for m, k in variable_small)
    return [
        CaseSpec("regular_rowwise_small", "regular", "rowwise", ((262144, 1024),)),
        CaseSpec("regular_rowwise_large", "regular", "rowwise", ((524288, 4096),)),
        CaseSpec("regular_columnwise_small", "regular", "columnwise", ((262144, 1024),)),
        CaseSpec("regular_columnwise_large", "regular", "columnwise", ((4096, 524288),)),
        CaseSpec(
            "grouped_uniform_rowwise_small", "grouped_uniform", "rowwise", ((32768, 1024),) * 8
        ),
        CaseSpec(
            "grouped_uniform_rowwise_large", "grouped_uniform", "rowwise", ((65536, 4096),) * 8
        ),
        CaseSpec(
            "grouped_uniform_columnwise_small",
            "grouped_uniform",
            "columnwise",
            ((32768, 1024),) * 8,
        ),
        CaseSpec(
            "grouped_uniform_columnwise_large",
            "grouped_uniform",
            "columnwise",
            ((4096, 65536),) * 8,
        ),
        CaseSpec(
            "grouped_uniform_rowwise_columnwise_small",
            "grouped_uniform",
            COMBINED_DIRECTION,
            uniform_combined_small,
        ),
        CaseSpec(
            "grouped_uniform_rowwise_columnwise_large",
            "grouped_uniform",
            COMBINED_DIRECTION,
            uniform_combined_large,
        ),
        CaseSpec(
            "grouped_variable_rowwise_small",
            "grouped_variable",
            "rowwise",
            variable_small,
            variable_first_dims=True,
        ),
        CaseSpec(
            "grouped_variable_rowwise_large",
            "grouped_variable",
            "rowwise",
            variable_large,
            variable_first_dims=True,
        ),
        CaseSpec(
            "grouped_variable_columnwise_small",
            "grouped_variable",
            "columnwise",
            variable_small,
            variable_first_dims=True,
        ),
        CaseSpec(
            "grouped_variable_columnwise_large",
            "grouped_variable",
            "columnwise",
            variable_large,
            variable_first_dims=True,
        ),
        CaseSpec(
            "grouped_variable_rowwise_columnwise_small",
            "grouped_variable",
            COMBINED_DIRECTION,
            variable_combined_small,
            variable_first_dims=True,
        ),
        CaseSpec(
            "grouped_variable_rowwise_columnwise_large",
            "grouped_variable",
            COMBINED_DIRECTION,
            variable_large,
            variable_first_dims=True,
        ),
    ]


def select_cases(
    cases: list[CaseSpec], case_filter: str, profile_case: str | None
) -> list[CaseSpec]:
    selected = (
        cases if case_filter == "all" else [case for case in cases if case.name == case_filter]
    )
    if profile_case:
        selected = [case for case in selected if case.name == profile_case]
    if not selected:
        known = ", ".join(case.name for case in cases)
        raise ValueError(f"No benchmark cases selected. Known cases: {known}")
    return selected


def buffer_count_for_case(
    processed_bytes: int, min_buffers: int, max_buffers: int, footprint_mib: int
) -> int:
    requested = math.ceil(footprint_mib * 1024 * 1024 / max(processed_bytes, 1))
    return min(max(min_buffers, requested), max_buffers)


def launch(api: NvteAPI, spec: CaseSpec, buf: BenchBuffer) -> None:
    if spec.api == "regular":
        api.swizzle(buf.input_handle, buf.output_handle)
    else:
        api.grouped_swizzle(buf.input_handle, buf.output_handle)


def check_cuda_status(status: Any, name: str) -> None:
    code = status[0] if isinstance(status, tuple) else status
    if code != 0:
        raise RuntimeError(f"{name} failed with CUDA status {code}")


def resolve_cuda_device(device_arg: Any) -> torch.device:
    if isinstance(device_arg, int):
        device = torch.device("cuda", device_arg)
    elif isinstance(device_arg, str) and device_arg.isdigit():
        device = torch.device("cuda", int(device_arg))
    else:
        device = torch.device(device_arg)
    if device.type != "cuda":
        raise ValueError(f"CUDA device is required, got {device}")

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    return torch.device("cuda", device_index)


def run_case(
    api: NvteAPI,
    spec: CaseSpec,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    input_bytes, output_bytes, processed_bytes = case_scale_bytes(spec)
    count = buffer_count_for_case(
        processed_bytes, args.min_buffers, args.max_buffers, args.buffer_footprint_mib
    )
    buffers = [make_case_buffer(api, spec, device) for _ in range(count)]

    for i in range(args.warmup_iterations):
        launch(api, spec, buffers[i % count])
    torch.cuda.synchronize()

    launches_per_sample = args.launches_per_sample
    timed_launches = args.measurement_iterations * launches_per_sample
    samples_us: list[float] = []
    if args.profile:
        check_cuda_status(torch.cuda.cudart().cudaProfilerStart(), "cudaProfilerStart")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(timed_launches):
            launch(api, spec, buffers[i % count])
        end.record()
        end.synchronize()
        check_cuda_status(torch.cuda.cudart().cudaProfilerStop(), "cudaProfilerStop")
        samples_us.append(start.elapsed_time(end) * 1000.0 / timed_launches)
    else:
        for i in range(args.measurement_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            launch_base = i * launches_per_sample
            for j in range(launches_per_sample):
                launch(api, spec, buffers[(launch_base + j) % count])
            end.record()
            end.synchronize()
            samples_us.append(start.elapsed_time(end) * 1000.0 / launches_per_sample)

    median_us = statistics.median(samples_us)
    bandwidth_tbps = processed_bytes / (median_us * 1.0e-6) / 1.0e12
    props = torch.cuda.get_device_properties(device)

    case_report: dict[str, Any] = {
        "case_name": spec.name,
        "api": spec.api,
        "direction": spec.direction,
        "scale_directions": list(scale_directions(spec.direction)),
        "num_tensors": spec.num_tensors,
        "input_scale_bytes": input_bytes,
        "output_scale_bytes": output_bytes,
        "processed_bytes": processed_bytes,
        "per_direction_scale_bytes": {
            direction: {
                "input_scale_bytes": direction_input,
                "output_scale_bytes": direction_output,
                "processed_bytes": direction_processed,
            }
            for direction in scale_directions(spec.direction)
            for direction_input, direction_output, direction_processed in [
                case_direction_scale_bytes(spec, direction)
            ]
        },
        "buffer_count": count,
        "classification": "large" if output_bytes > 10 * 1024 * 1024 else "small",
        "warmup_iterations": args.warmup_iterations,
        "measurement_iterations": args.measurement_iterations,
        "launches_per_sample": launches_per_sample,
        "timed_launches": timed_launches,
        "median_us": median_us,
        "bandwidth_TBps": bandwidth_tbps,
        "device_name": props.name,
        "sm_arch": props.major * 10 + props.minor,
    }
    if spec.num_tensors == 1:
        case_report["M"] = spec.data_shapes[0][0]
        case_report["K"] = spec.data_shapes[0][1]
    else:
        case_report["shapes"] = [list(shape) for shape in spec.data_shapes]
    return case_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="all", help="Case name to run, or 'all'.")
    parser.add_argument(
        "--output",
        default=os.getenv("ORCHESTRA_BENCHMARK_RAW_REPORT", "swizzle_benchmark_report.json"),
    )
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--measurement-iterations", type=int, default=100)
    parser.add_argument(
        "--launches-per-sample",
        type=int,
        default=20,
        help="Swizzle launches timed inside each CUDA-event measurement window.",
    )
    parser.add_argument("--min-buffers", type=int, default=8)
    parser.add_argument("--max-buffers", type=int, default=128)
    parser.add_argument("--buffer-footprint-mib", type=int, default=512)
    parser.add_argument(
        "--device",
        default=DEFAULT_CUDA_DEVICE,
        help=(
            "CUDA device to use, e.g. 'cuda:0' or '0'. Bare 'cuda' resolves to the "
            "current device."
        ),
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-case", default=None)
    args = parser.parse_args()
    if args.profile and not args.profile_case:
        parser.error("--profile requires --profile-case so profiler output maps to one case")
    if (
        args.warmup_iterations <= 0
        or args.measurement_iterations <= 0
        or args.launches_per_sample <= 0
    ):
        parser.error(
            "Warmup iterations, measurement iterations, and launches per sample must be positive"
        )
    return args


def main() -> None:
    args = parse_args()
    global torch  # pylint: disable=global-variable-not-assigned,invalid-name
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for swizzle benchmarking")
    device = resolve_cuda_device(args.device)
    api = NvteAPI()
    selected = select_cases(build_cases(), args.case, args.profile_case)
    results = [run_case(api, case, args, device) for case in selected]
    props = torch.cuda.get_device_properties(device)
    report = {
        "schema_version": "swizzle_benchmark/v1",
        "device_name": props.name,
        "sm_arch": props.major * 10 + props.minor,
        "profile_enabled": bool(args.profile),
        "profile_after_warmup": bool(args.profile),
        "cases": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
