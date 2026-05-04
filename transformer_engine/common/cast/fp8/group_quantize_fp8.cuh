/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_fp8.cuh
 *  \brief CUDA kernels to quantize grouped tensors to tensor-scaled FP8.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <cstddef>
#include <cstdint>

#include "../../common.h"
#include "../../util/math.h"
#include "../../utils.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace group_quantize_kernel {

constexpr size_t THREADS_PER_BLOCK = 256;
constexpr size_t ELEMENTS_PER_THREAD = 4;

__device__ __forceinline__ size_t get_grouped_logical_total(
    const ShapeRepresentation shape_rep, const size_t num_tensors, const size_t fallback_total,
    const int64_t *const __restrict__ offsets_ptr) {
  if (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS || offsets_ptr == nullptr) {
    return fallback_total;
  }
  return static_cast<size_t>(offsets_ptr[num_tensors]);
}

__device__ __forceinline__ size_t get_tensor_id_from_offset(
    const ShapeRepresentation shape_rep, const size_t num_tensors, const size_t logical_offset,
    const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ offsets_ptr) {
  if (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS) {
    const size_t tensor_size = first_logical_dim * last_logical_dim / num_tensors;
    return logical_offset / tensor_size;
  }

  size_t low = 1;
  size_t high = num_tensors;
  while (low < high) {
    const size_t mid = low + (high - low) / 2;
    const size_t mid_offset = static_cast<size_t>(offsets_ptr[mid]);
    if (mid_offset <= logical_offset) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low - 1;
}

__device__ __forceinline__ size_t get_tensor_rows(
    const ShapeRepresentation shape_rep, const size_t tensor_id, const size_t first_logical_dim,
    const size_t num_tensors, const int64_t *const __restrict__ first_dims_ptr) {
  if (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS) {
    return first_logical_dim / num_tensors;
  }
  if (shape_rep == ShapeRepresentation::VARYING_LAST_DIM) {
    return first_logical_dim;
  }
  return static_cast<size_t>(first_dims_ptr[tensor_id]);
}

__device__ __forceinline__ size_t get_tensor_cols(
    const ShapeRepresentation shape_rep, const size_t tensor_id, const size_t last_logical_dim,
    const int64_t *const __restrict__ last_dims_ptr) {
  if (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS ||
      shape_rep == ShapeRepresentation::VARYING_FIRST_DIM) {
    return last_logical_dim;
  }
  return static_cast<size_t>(last_dims_ptr[tensor_id]);
}

__device__ __forceinline__ size_t get_tensor_start_offset(
    const ShapeRepresentation shape_rep, const size_t tensor_id, const size_t first_logical_dim,
    const size_t last_logical_dim, const size_t num_tensors,
    const int64_t *const __restrict__ offsets_ptr) {
  if (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS) {
    return tensor_id * (first_logical_dim * last_logical_dim / num_tensors);
  }
  return static_cast<size_t>(offsets_ptr[tensor_id]);
}

template <typename IType, typename OType>
__global__ void init_group_quantize_fp8_metadata_kernel(
    const float *const scale_ptr, float *const scale_inv_ptr, float *const columnwise_scale_inv_ptr,
    const float *const noop_ptr, const size_t num_tensors, const bool scale_is_per_tensor,
    const bool rowwise, const bool columnwise) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) return;

  const size_t tensor_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (tensor_id >= num_tensors) return;

  const size_t scale_idx = scale_is_per_tensor ? tensor_id : 0;
  const float scale = (scale_ptr != nullptr) ? scale_ptr[scale_idx] : 1.0f;
  if (rowwise && scale_inv_ptr != nullptr) {
    reciprocal<float>(&scale_inv_ptr[tensor_id], scale);
  }
  if (columnwise && columnwise_scale_inv_ptr != nullptr) {
    reciprocal<float>(&columnwise_scale_inv_ptr[tensor_id], scale);
  }
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          typename OType>
__global__ void group_quantize_fp8_kernel(
    const IType *const input_ptr, OType *const output_ptr, OType *const columnwise_output_ptr,
    float *const amax_ptr, const float *const scale_ptr, const float *const noop_ptr,
    const ShapeRepresentation shape_rep, const size_t num_tensors, const size_t first_logical_dim,
    const size_t last_logical_dim, const size_t launch_elems,
    const int64_t *const __restrict__ offsets_ptr,
    const int64_t *const __restrict__ first_dims_ptr,
    const int64_t *const __restrict__ last_dims_ptr, const bool scale_is_per_tensor) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) return;

  const size_t fallback_total = first_logical_dim * last_logical_dim;
  const size_t logical_total =
      get_grouped_logical_total(shape_rep, num_tensors, fallback_total, offsets_ptr);
  const size_t thread_stride = blockDim.x * gridDim.x;
  size_t logical_offset = blockIdx.x * blockDim.x + threadIdx.x;

  for (; logical_offset < launch_elems; logical_offset += thread_stride) {
    if (logical_offset >= logical_total) {
      continue;
    }

    const size_t tensor_id = get_tensor_id_from_offset(
        shape_rep, num_tensors, logical_offset, first_logical_dim, last_logical_dim, offsets_ptr);
    const size_t rows =
        get_tensor_rows(shape_rep, tensor_id, first_logical_dim, num_tensors, first_dims_ptr);
    const size_t cols = get_tensor_cols(shape_rep, tensor_id, last_logical_dim, last_dims_ptr);
    if (rows == 0 || cols == 0) {
      continue;
    }

    const size_t tensor_start = get_tensor_start_offset(
        shape_rep, tensor_id, first_logical_dim, last_logical_dim, num_tensors, offsets_ptr);
    const size_t tensor_offset = logical_offset - tensor_start;
    if (tensor_offset >= rows * cols) {
      continue;
    }

    const size_t row = tensor_offset / cols;
    const size_t col = tensor_offset - row * cols;
    const size_t scale_idx = scale_is_per_tensor ? tensor_id : 0;
    const float scale = (scale_ptr != nullptr) ? scale_ptr[scale_idx] : 1.0f;

    float elt = static_cast<float>(input_ptr[logical_offset]);
    if constexpr (IS_ACT) {
      elt = OP(elt, {});
    }
    if (amax_ptr != nullptr) {
      atomicMaxFloat(&amax_ptr[tensor_id], fabsf(elt));
    }

    const OType out = static_cast<OType>(elt * scale);
    if (output_ptr != nullptr) {
      output_ptr[logical_offset] = out;
    }
    if (columnwise_output_ptr != nullptr) {
      columnwise_output_ptr[tensor_start + col * rows + row] = out;
    }
  }
}

}  // namespace group_quantize_kernel

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void group_quantize(const GroupedTensor *input, const Tensor *noop, GroupedTensor *output,
                    const QuantizationConfig *quant_config, cudaStream_t stream) {
  using namespace group_quantize_kernel;

  CheckNoopTensor(*noop, "cast_noop");

  NVTE_CHECK(input != nullptr, "Input grouped tensor must be allocated.");
  NVTE_CHECK(output != nullptr, "Output grouped tensor must be allocated.");
  CheckGroupedTensorShapeArrays(*input, "group_quantize_input");
  CheckGroupedTensorShapeArrays(*output, "group_quantize_output");
  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->logical_shape.ndim == 2 && output->logical_shape.ndim == 2,
             "Input and output grouped tensors must have 2D logical shapes.");
  NVTE_CHECK(input->logical_shape.data[0] == output->logical_shape.data[0] &&
                 input->logical_shape.data[1] == output->logical_shape.data[1],
             "Input and output grouped tensor logical shapes must match.");
  NVTE_CHECK(input->has_data(), "Cannot quantize grouped tensor without rowwise input data.");
  NVTE_CHECK(!is_fp8_dtype(input->dtype()), "Input grouped tensor must be in higher precision.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output grouped tensor must have FP8 type.");

  const bool rowwise = output->has_data();
  const bool columnwise = output->has_columnwise_data();
  NVTE_CHECK(rowwise || columnwise, "Either rowwise or columnwise output data need to be allocated.");

  ShapeRepresentation shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  if (output->all_same_shape()) {
    shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  } else if (output->all_same_first_dim()) {
    shape_rep = ShapeRepresentation::VARYING_LAST_DIM;
  } else if (output->all_same_last_dim()) {
    shape_rep = ShapeRepresentation::VARYING_FIRST_DIM;
  } else if (output->varying_both_dims()) {
    shape_rep = ShapeRepresentation::VARYING_BOTH_DIMS;
  }

  const size_t num_tensors = output->num_tensors;
  const size_t first_logical_dim = output->logical_shape.data[0];
  const size_t last_logical_dim = output->logical_shape.data[1];
  const size_t launch_elems = first_logical_dim * last_logical_dim;

  if (launch_elems == 0) {
    return;
  }

  const int64_t *const offsets_ptr = reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  const int64_t *const first_dims_ptr = reinterpret_cast<const int64_t *>(output->first_dims.dptr);
  const int64_t *const last_dims_ptr = reinterpret_cast<const int64_t *>(output->last_dims.dptr);

  const size_t scale_numel = output->scale.numel();
  NVTE_CHECK(output->scale.has_data(), "Grouped FP8 output scale must be allocated.");
  NVTE_CHECK(output->scale.dtype == DType::kFloat32, "Grouped FP8 output scale must be Float32.");
  NVTE_CHECK(scale_numel == 1 || scale_numel == num_tensors,
             "Grouped FP8 output scale must have 1 or num_tensors entries.");
  const bool scale_is_per_tensor = (scale_numel == num_tensors);

  if (output->amax.has_data()) {
    NVTE_CHECK(output->amax.dtype == DType::kFloat32, "Grouped FP8 output amax must be Float32.");
    NVTE_CHECK(output->amax.numel() == num_tensors,
               "Grouped FP8 output amax must have num_tensors entries.");
  }
  if (rowwise) {
    NVTE_CHECK(output->scale_inv.has_data(),
               "Grouped FP8 rowwise output scale_inv must be allocated.");
    NVTE_CHECK(output->scale_inv.dtype == DType::kFloat32,
               "Grouped FP8 rowwise output scale_inv must be Float32.");
    NVTE_CHECK(output->scale_inv.numel() == num_tensors,
               "Grouped FP8 rowwise output scale_inv must have num_tensors entries.");
  }
  if (columnwise) {
    NVTE_CHECK(output->columnwise_scale_inv.has_data(),
               "Grouped FP8 columnwise output scale_inv must be allocated.");
    NVTE_CHECK(output->columnwise_scale_inv.dtype == DType::kFloat32,
               "Grouped FP8 columnwise output scale_inv must be Float32.");
    NVTE_CHECK(output->columnwise_scale_inv.numel() == num_tensors,
               "Grouped FP8 columnwise output scale_inv must have num_tensors entries.");
  }

  const float *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  const float *scale_ptr = reinterpret_cast<const float *>(output->scale.dptr);
  float *amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  float *columnwise_scale_inv_ptr = reinterpret_cast<float *>(output->columnwise_scale_inv.dptr);

  if (quant_config != nullptr) {
    NVTE_CHECK(!quant_config->stochastic_rounding,
               "Stochastic rounding is not supported for grouped FP8 tensor scaling.");
  }

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          const IType *input_ptr = reinterpret_cast<const IType *>(input->data.dptr);
          OType *output_ptr = rowwise ? reinterpret_cast<OType *>(output->data.dptr) : nullptr;
          OType *columnwise_output_ptr =
              columnwise ? reinterpret_cast<OType *>(output->columnwise_data.dptr) : nullptr;

          const dim3 metadata_block(THREADS_PER_BLOCK);
          const dim3 metadata_grid(DIVUP(num_tensors, static_cast<size_t>(THREADS_PER_BLOCK)));
          init_group_quantize_fp8_metadata_kernel<IType, OType>
              <<<metadata_grid, metadata_block, 0, stream>>>(
                  scale_ptr, scale_inv_ptr, columnwise_scale_inv_ptr, noop_ptr, num_tensors,
                  scale_is_per_tensor, rowwise, columnwise);

          const size_t blocks =
              DIVUP(launch_elems, THREADS_PER_BLOCK * ELEMENTS_PER_THREAD);
          const dim3 block(THREADS_PER_BLOCK);
          const dim3 grid(blocks);
          group_quantize_fp8_kernel<IS_ACT, ParamOP, OP, IType, OType>
              <<<grid, block, 0, stream>>>(
                  input_ptr, output_ptr, columnwise_output_ptr, amax_ptr, scale_ptr, noop_ptr,
                  shape_rep, num_tensors, first_logical_dim, last_logical_dim, launch_elems,
                  offsets_ptr, first_dims_ptr, last_dims_ptr, scale_is_per_tensor);
          NVTE_CHECK_CUDA(cudaGetLastError()););  // NOLINT(*)
  );                                             // NOLINT(*)
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
