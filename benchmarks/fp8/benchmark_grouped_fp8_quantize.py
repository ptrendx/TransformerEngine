# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark grouped FP8 tensor-scaling quantize with padded backing allocation."""

import argparse
import ctypes
import glob
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex


NVTE_DELAYED_TENSOR_SCALING = 0
kNVTEInt64 = 3
kNVTEFloat32 = 4
kNVTEBFloat16 = 6
kNVTEFloat8E4M3 = 7

kNVTEGroupedRowwiseData = 0
kNVTEGroupedScale = 2
kNVTEGroupedAmax = 3
kNVTEGroupedRowwiseScaleInv = 4
kNVTEGroupedFirstDims = 7
kNVTEGroupedTensorOffsets = 9


class NVTEShape(ctypes.Structure):
    _fields_ = [("data", ctypes.c_size_t * 15), ("ndim", ctypes.c_size_t)]


class NVTEBasicTensor(ctypes.Structure):
    _fields_ = [
        ("data_ptr", ctypes.c_void_p),
        ("dtype", ctypes.c_int),
        ("shape", NVTEShape),
    ]


def make_shape(dims: List[int]) -> NVTEShape:
    shape = NVTEShape()
    shape.ndim = len(dims)
    for i, dim in enumerate(dims):
        shape.data[i] = dim
    return shape


def load_nvte_library() -> ctypes.CDLL:
    candidates = [Path(tex.__file__)]
    candidates.extend(Path(p) for p in glob.glob(str(Path(tex.__file__).parent / "*.so")))
    for candidate in candidates:
        try:
            lib = ctypes.CDLL(str(candidate))
            getattr(lib, "nvte_group_quantize")
            return lib
        except (OSError, AttributeError):
            continue
    lib = ctypes.CDLL(None)
    getattr(lib, "nvte_group_quantize")
    return lib


def configure_library(lib: ctypes.CDLL) -> None:
    lib.nvte_create_grouped_tensor.argtypes = [ctypes.c_int, ctypes.c_size_t, NVTEShape]
    lib.nvte_create_grouped_tensor.restype = ctypes.c_void_p
    lib.nvte_destroy_grouped_tensor.argtypes = [ctypes.c_void_p]
    lib.nvte_destroy_grouped_tensor.restype = None
    lib.nvte_set_grouped_tensor_param.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.nvte_set_grouped_tensor_param.restype = None
    lib.nvte_group_quantize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.nvte_group_quantize.restype = None


def set_grouped_tensor_param(
    lib: ctypes.CDLL, grouped_tensor: ctypes.c_void_p, param: int, basic_tensor: NVTEBasicTensor
) -> None:
    lib.nvte_set_grouped_tensor_param(
        grouped_tensor,
        param,
        ctypes.byref(basic_tensor),
        ctypes.sizeof(NVTEBasicTensor),
    )


def make_basic_tensor(tensor: torch.Tensor, nvte_dtype: int, dims: List[int]) -> NVTEBasicTensor:
    return NVTEBasicTensor(
        ctypes.c_void_p(tensor.data_ptr()),
        nvte_dtype,
        make_shape(dims),
    )


def build_grouped_handles(
    lib: ctypes.CDLL,
    first_dims: torch.Tensor,
    offsets: torch.Tensor,
    input_backing: torch.Tensor,
    output_backing: torch.Tensor,
    scale: torch.Tensor,
    scale_inv: torch.Tensor,
    amax: torch.Tensor,
    logical_shape: List[int],
):
    num_tensors = int(first_dims.numel())
    input_group = lib.nvte_create_grouped_tensor(
        NVTE_DELAYED_TENSOR_SCALING, num_tensors, make_shape(logical_shape)
    )
    output_group = lib.nvte_create_grouped_tensor(
        NVTE_DELAYED_TENSOR_SCALING, num_tensors, make_shape(logical_shape)
    )

    first_dims_basic = make_basic_tensor(first_dims, kNVTEInt64, [num_tensors])
    offsets_basic = make_basic_tensor(offsets, kNVTEInt64, [num_tensors + 1])
    input_basic = make_basic_tensor(input_backing, kNVTEBFloat16, [input_backing.numel()])
    output_basic = make_basic_tensor(output_backing, kNVTEFloat8E4M3, [output_backing.numel()])
    scale_basic = make_basic_tensor(scale, kNVTEFloat32, [scale.numel()])
    scale_inv_basic = make_basic_tensor(scale_inv, kNVTEFloat32, [scale_inv.numel()])
    amax_basic = make_basic_tensor(amax, kNVTEFloat32, [amax.numel()])

    set_grouped_tensor_param(lib, input_group, kNVTEGroupedRowwiseData, input_basic)
    set_grouped_tensor_param(lib, input_group, kNVTEGroupedFirstDims, first_dims_basic)
    set_grouped_tensor_param(lib, input_group, kNVTEGroupedTensorOffsets, offsets_basic)

    set_grouped_tensor_param(lib, output_group, kNVTEGroupedRowwiseData, output_basic)
    set_grouped_tensor_param(lib, output_group, kNVTEGroupedScale, scale_basic)
    set_grouped_tensor_param(lib, output_group, kNVTEGroupedAmax, amax_basic)
    set_grouped_tensor_param(lib, output_group, kNVTEGroupedRowwiseScaleInv, scale_inv_basic)
    set_grouped_tensor_param(lib, output_group, kNVTEGroupedFirstDims, first_dims_basic)
    set_grouped_tensor_param(lib, output_group, kNVTEGroupedTensorOffsets, offsets_basic)

    return input_group, output_group


def run_case(lib: ctypes.CDLL, mode: str, iterations: int, warmup: int) -> Dict[str, object]:
    assert mode in ("delayed", "current")
    device = torch.device("cuda")
    first_dims_list = [1536, 896, 2048, 512]
    last_dim = 4096
    padding_after = [1024, 2048, 512, 4096]

    offsets_list = [0]
    for rows, padding in zip(first_dims_list, padding_after):
        offsets_list.append(offsets_list[-1] + rows * last_dim + padding)
    allocation_elements = offsets_list[-1]
    logical_total_elements = sum(rows * last_dim for rows in first_dims_list)
    logical_shape = [sum(first_dims_list), last_dim]

    input_backing = torch.empty(allocation_elements, dtype=torch.bfloat16, device=device)
    output_backing = torch.empty(allocation_elements, dtype=torch.uint8, device=device)
    first_dims = torch.tensor(first_dims_list, dtype=torch.int64, device=device)
    offsets = torch.tensor(offsets_list, dtype=torch.int64, device=device)
    num_tensors = len(first_dims_list)

    torch.manual_seed(1234)
    for tensor_id, rows in enumerate(first_dims_list):
        start = offsets_list[tensor_id]
        end = start + rows * last_dim
        input_backing[start:end].normal_()

    scale_elements = 1 if mode == "delayed" else num_tensors
    scale = torch.ones(scale_elements, dtype=torch.float32, device=device)
    scale_inv = torch.empty(num_tensors, dtype=torch.float32, device=device)
    amax = torch.zeros(num_tensors, dtype=torch.float32, device=device)

    input_group, output_group = build_grouped_handles(
        lib, first_dims, offsets, input_backing, output_backing, scale, scale_inv, amax, logical_shape
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    try:
        for _ in range(warmup):
            lib.nvte_group_quantize(input_group, output_group, None, stream)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iterations):
            lib.nvte_group_quantize(input_group, output_group, None, stream)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event) / iterations
    finally:
        lib.nvte_destroy_grouped_tensor(input_group)
        lib.nvte_destroy_grouped_tensor(output_group)

    input_dtype_bytes = torch.finfo(torch.bfloat16).bits // 8
    output_dtype_bytes = 1
    logical_input_bytes = logical_total_elements * input_dtype_bytes
    logical_output_bytes = logical_total_elements * output_dtype_bytes
    allocation_input_bytes = allocation_elements * input_dtype_bytes
    allocation_output_bytes = allocation_elements * output_dtype_bytes
    metadata_bytes = num_tensors * 2 * 4 + scale_elements * 4
    logical_data_bytes = logical_input_bytes + logical_output_bytes
    bandwidth_gbps = logical_data_bytes / (elapsed_ms / 1e3) / 1e9

    return {
        "mode": mode,
        "num_tensors": num_tensors,
        "first_dims": first_dims_list,
        "last_dim": last_dim,
        "padding_after_elements": padding_after,
        "logical_total_elements": logical_total_elements,
        "allocation_elements": allocation_elements,
        "allocation_differs_from_logical": allocation_elements != logical_total_elements,
        "input_dtype": "bfloat16",
        "output_dtype": "float8_e4m3",
        "input_dtype_bytes": input_dtype_bytes,
        "output_dtype_bytes": output_dtype_bytes,
        "logical_input_bytes": logical_input_bytes,
        "logical_output_bytes": logical_output_bytes,
        "allocation_input_bytes": allocation_input_bytes,
        "allocation_output_bytes": allocation_output_bytes,
        "metadata_bytes": metadata_bytes,
        "bandwidth_denominator_bytes": logical_data_bytes,
        "bandwidth_denominator": "logical input bytes + logical output bytes",
        "iterations": iterations,
        "warmup": warmup,
        "elapsed_ms_per_iter": elapsed_ms,
        "logical_bandwidth_gbps": bandwidth_gbps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=os.environ.get(
            "ORCHESTRA_BENCHMARK_RAW_REPORT", "benchmark_grouped_fp8_quantize_report.json"
        ),
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--modes", nargs="+", default=["delayed", "current"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Grouped FP8 quantize benchmark requires CUDA.")
    fp8_available, reason = te.is_fp8_available(return_reason=True)
    if not fp8_available:
        raise RuntimeError(f"FP8 is not available on this system: {reason}")

    lib = load_nvte_library()
    configure_library(lib)
    results = [run_case(lib, mode, args.iterations, args.warmup) for mode in args.modes]
    report = {
        "benchmark": "grouped_fp8_tensor_scaling_quantize",
        "success": True,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
