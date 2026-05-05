/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_fp8.cuh
 *  \brief CUDA kernels to quantize grouped tensors to FP8 with tensor scaling.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_

#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <algorithm>
#include <cstddef>
#include <limits>

#include "../../common.h"
#include "../../recipe/recipe_common.cuh"
#include "../../util/math.h"
#include "../../util/vectorized_pointwise.h"
#include "../../utils.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace group_quantize_kernel {

constexpr size_t THREADS_PER_BLOCK = 256;

namespace detail {
using Empty = transformer_engine::Empty;
__device__ inline float identity(float value, const Empty &) { return value; }
}  // namespace detail

__host__ inline ShapeRepresentation get_shape_representation(const GroupedTensor &tensor) {
  if (tensor.all_same_shape()) {
    return ShapeRepresentation::SAME_BOTH_DIMS;
  }
  if (tensor.all_same_first_dim()) {
    return ShapeRepresentation::VARYING_LAST_DIM;
  }
  if (tensor.all_same_last_dim()) {
    return ShapeRepresentation::VARYING_FIRST_DIM;
  }
  return ShapeRepresentation::VARYING_BOTH_DIMS;
}

__device__ __forceinline__ size_t tensor_rows(const ShapeRepresentation shape_rep,
                                              const size_t tensor_id,
                                              const size_t num_tensors,
                                              const size_t first_logical_dim,
                                              const int64_t *const first_dims) {
  switch (shape_rep) {
    case ShapeRepresentation::SAME_BOTH_DIMS:
      return first_logical_dim / num_tensors;
    case ShapeRepresentation::VARYING_FIRST_DIM:
    case ShapeRepresentation::VARYING_BOTH_DIMS:
      return static_cast<size_t>(first_dims[tensor_id]);
    case ShapeRepresentation::VARYING_LAST_DIM:
      return first_logical_dim;
  }
  return 0;
}

__device__ __forceinline__ size_t tensor_cols(const ShapeRepresentation shape_rep,
                                              const size_t tensor_id,
                                              const size_t last_logical_dim,
                                              const int64_t *const last_dims) {
  switch (shape_rep) {
    case ShapeRepresentation::SAME_BOTH_DIMS:
    case ShapeRepresentation::VARYING_FIRST_DIM:
      return last_logical_dim;
    case ShapeRepresentation::VARYING_LAST_DIM:
    case ShapeRepresentation::VARYING_BOTH_DIMS:
      return static_cast<size_t>(last_dims[tensor_id]);
  }
  return 0;
}

struct LogicalIndex {
  size_t tensor_id = 0;
  size_t local_idx = 0;
  size_t compact_base = 0;
  size_t rows = 0;
  size_t cols = 0;
};

__device__ __forceinline__ LogicalIndex decode_logical_index(
    const size_t compact_idx, const ShapeRepresentation shape_rep, const size_t num_tensors,
    const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const first_dims, const int64_t *const last_dims) {
  size_t compact_base = 0;
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    const size_t rows =
        tensor_rows(shape_rep, tensor_id, num_tensors, first_logical_dim, first_dims);
    const size_t cols = tensor_cols(shape_rep, tensor_id, last_logical_dim, last_dims);
    const size_t numel = rows * cols;
    if (compact_idx < compact_base + numel) {
      return {tensor_id, compact_idx - compact_base, compact_base, rows, cols};
    }
    compact_base += numel;
  }
  return {};
}

__device__ __forceinline__ size_t physical_base(const int64_t *const offsets,
                                                const size_t tensor_id,
                                                const size_t compact_base) {
  return offsets == nullptr ? compact_base : static_cast<size_t>(offsets[tensor_id]);
}

__device__ __forceinline__ size_t grouped_meta_index(const size_t tensor_id,
                                                     const size_t meta_numel) {
  return meta_numel == 1 ? 0 : tensor_id;
}

__global__ void zero_grouped_amax_kernel(float *amax, const size_t amax_numel,
                                         const float *noop) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < amax_numel;
       idx += gridDim.x * blockDim.x) {
    amax[idx] = 0.0f;
  }
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType>
__global__ void grouped_amax_kernel(const IType *input, float *amax, const size_t amax_numel,
                                    const ShapeRepresentation shape_rep,
                                    const size_t num_tensors, const size_t first_logical_dim,
                                    const size_t last_logical_dim,
                                    const int64_t *const first_dims,
                                    const int64_t *const last_dims,
                                    const int64_t *const input_offsets,
                                    const size_t logical_total, const float *noop) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  constexpr float (*UnaryOP)(float, const ParamOP &) = (OP == nullptr) ? detail::identity : OP;

  for (size_t compact_idx = blockIdx.x * blockDim.x + threadIdx.x; compact_idx < logical_total;
       compact_idx += gridDim.x * blockDim.x) {
    const LogicalIndex logical =
        decode_logical_index(compact_idx, shape_rep, num_tensors, first_logical_dim,
                             last_logical_dim, first_dims, last_dims);
    const size_t input_idx =
        physical_base(input_offsets, logical.tensor_id, logical.compact_base) + logical.local_idx;
    float value = static_cast<float>(input[input_idx]);
    if constexpr (IS_ACT) {
      value = UnaryOP(value, {});
    }
    atomicMaxFloat(&amax[grouped_meta_index(logical.tensor_id, amax_numel)], fabsf(value));
  }
}

template <typename OType>
__global__ void grouped_compute_scale_kernel(
    const float *amax, float *scale, float *scale_inv, float *columnwise_scale_inv,
    const size_t num_tensors, const float max_fp8, const bool force_pow_2_scales,
    const float epsilon, const float *noop) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  for (size_t tensor_id = blockIdx.x * blockDim.x + threadIdx.x; tensor_id < num_tensors;
       tensor_id += gridDim.x * blockDim.x) {
    const float scale_value = compute_scale_from_amax(
        amax[tensor_id], max_fp8, force_pow_2_scales, epsilon, std::numeric_limits<float>::max());
    scale[tensor_id] = scale_value;
    if (scale_inv != nullptr) {
      reciprocal<float>(&scale_inv[tensor_id], scale_value);
    }
    if (columnwise_scale_inv != nullptr) {
      reciprocal<float>(&columnwise_scale_inv[tensor_id], scale_value);
    }
  }
}

template <bool IS_ACT, bool CURRENT_SCALING, typename ParamOP,
          float (*OP)(float, const ParamOP &), typename IType, typename OType>
__global__ void grouped_quantize_fp8_kernel(
    const IType *input, OType *output, OType *columnwise_output, const float *scale, float *amax,
    float *scale_inv, float *columnwise_scale_inv, const size_t scale_numel,
    const size_t amax_numel, const ShapeRepresentation shape_rep, const size_t num_tensors,
    const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const first_dims, const int64_t *const last_dims,
    const int64_t *const input_offsets, const int64_t *const output_offsets,
    const size_t logical_total, const float *noop) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  constexpr float (*UnaryOP)(float, const ParamOP &) = (OP == nullptr) ? detail::identity : OP;

  for (size_t compact_idx = blockIdx.x * blockDim.x + threadIdx.x; compact_idx < logical_total;
       compact_idx += gridDim.x * blockDim.x) {
    const LogicalIndex logical =
        decode_logical_index(compact_idx, shape_rep, num_tensors, first_logical_dim,
                             last_logical_dim, first_dims, last_dims);
    const size_t input_base = physical_base(input_offsets, logical.tensor_id, logical.compact_base);
    const size_t output_base =
        physical_base(output_offsets, logical.tensor_id, logical.compact_base);
    const size_t row = logical.local_idx / logical.cols;
    const size_t col = logical.local_idx - row * logical.cols;

    const size_t scale_idx = grouped_meta_index(logical.tensor_id, scale_numel);
    const float scale_value = scale == nullptr ? 1.0f : scale[scale_idx];

    float value = static_cast<float>(input[input_base + logical.local_idx]);
    if constexpr (IS_ACT) {
      value = UnaryOP(value, {});
    }
    if constexpr (!CURRENT_SCALING) {
      if (amax != nullptr) {
        atomicMaxFloat(&amax[grouped_meta_index(logical.tensor_id, amax_numel)], fabsf(value));
      }
      if (logical.local_idx == 0) {
        if (scale_inv != nullptr) {
          reciprocal<float>(&scale_inv[logical.tensor_id], scale_value);
        }
        if (columnwise_scale_inv != nullptr) {
          reciprocal<float>(&columnwise_scale_inv[logical.tensor_id], scale_value);
        }
      }
    }

    const OType out = static_cast<OType>(value * scale_value);
    if (output != nullptr) {
      output[output_base + logical.local_idx] = out;
    }
    if (columnwise_output != nullptr) {
      columnwise_output[output_base + col * logical.rows + row] = out;
    }
  }
}

}  // namespace group_quantize_kernel

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void group_quantize(const GroupedTensor *input, const Tensor *noop, GroupedTensor *output,
                    const QuantizationConfig *quant_config, cudaStream_t stream) {
  using namespace group_quantize_kernel;

  CheckNoopTensor(*noop, "cast_noop");

  const bool use_rowwise = output->has_data();
  const bool use_columnwise = output->has_columnwise_data();
  NVTE_CHECK(use_rowwise || use_columnwise,
             "Either rowwise or columnwise output data need to be allocated.");
  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->has_data(), "Cannot quantize grouped tensor without rowwise input data.");
  NVTE_CHECK(!is_fp8_dtype(input->dtype()), "Input must be in higher precision.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  if (use_columnwise) {
    NVTE_CHECK(output->columnwise_data.dtype == output->dtype(),
               "Rowwise and columnwise FP8 output types must match.");
  }
  if (use_rowwise) {
    NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
  }
  if (use_columnwise) {
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
               "Columnwise scaling tensor must be allocated.");
  }
  NVTE_CHECK(input->logical_shape.ndim == 2 && output->logical_shape.ndim == 2,
             "Grouped FP8 quantize expects 2D logical shapes.");
  NVTE_CHECK(input->logical_shape.data[0] == output->logical_shape.data[0] &&
                 input->logical_shape.data[1] == output->logical_shape.data[1],
             "Input and output grouped logical shapes must match.");

  const size_t num_tensors = output->num_tensors;
  const size_t first_logical_dim = output->logical_shape.data[0];
  const size_t last_logical_dim = output->logical_shape.data[1];
  const size_t logical_total = first_logical_dim * last_logical_dim;
  if (logical_total == 0) {
    return;
  }

  const ShapeRepresentation shape_rep = get_shape_representation(*output);
  const int64_t *const first_dims = output->first_dims.dptr != nullptr
                                        ? reinterpret_cast<const int64_t *>(output->first_dims.dptr)
                                        : reinterpret_cast<const int64_t *>(input->first_dims.dptr);
  const int64_t *const last_dims = output->last_dims.dptr != nullptr
                                       ? reinterpret_cast<const int64_t *>(output->last_dims.dptr)
                                       : reinterpret_cast<const int64_t *>(input->last_dims.dptr);
  const int64_t *const input_offsets =
      input->tensor_offsets.dptr == nullptr
          ? nullptr
          : reinterpret_cast<const int64_t *>(input->tensor_offsets.dptr);
  const int64_t *const output_offsets =
      output->tensor_offsets.dptr == nullptr
          ? nullptr
          : reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);

  if (shape_rep == ShapeRepresentation::VARYING_FIRST_DIM ||
      shape_rep == ShapeRepresentation::VARYING_BOTH_DIMS) {
    NVTE_CHECK(first_dims != nullptr, "Grouped FP8 quantize requires first_dims metadata.");
  }
  if (shape_rep == ShapeRepresentation::VARYING_LAST_DIM ||
      shape_rep == ShapeRepresentation::VARYING_BOTH_DIMS) {
    NVTE_CHECK(last_dims != nullptr, "Grouped FP8 quantize requires last_dims metadata.");
  }

  const float *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  float *amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *scale_ptr = reinterpret_cast<float *>(output->scale.dptr);
  float *scale_inv_ptr =
      use_rowwise ? reinterpret_cast<float *>(output->scale_inv.dptr) : nullptr;
  float *columnwise_scale_inv_ptr =
      use_columnwise ? reinterpret_cast<float *>(output->columnwise_scale_inv.dptr) : nullptr;

  const size_t scale_numel = output->scale.numel();
  const size_t amax_numel = output->amax.numel();
  const bool current_scaling =
      scale_ptr != nullptr && amax_ptr != nullptr && scale_numel == num_tensors &&
      amax_numel == num_tensors;

  if (current_scaling) {
    NVTE_CHECK(scale_inv_ptr != nullptr || columnwise_scale_inv_ptr != nullptr,
               "Current scaling requires scale inverse output.");
  } else {
    NVTE_CHECK(scale_ptr == nullptr || scale_numel == 1,
               "Grouped delayed FP8 quantize expects a scalar scale tensor. A per-tensor scale "
               "buffer is interpreted as current scaling metadata.");
  }

  constexpr size_t max_blocks = 65535;
  const size_t num_blocks = std::min(DIVUP(logical_total, THREADS_PER_BLOCK), max_blocks);
  const size_t meta_blocks =
      std::min(DIVUP(std::max(num_tensors, amax_numel), THREADS_PER_BLOCK), max_blocks);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          const IType *input_ptr = reinterpret_cast<const IType *>(input->data.dptr);
          OType *output_ptr =
              use_rowwise ? reinterpret_cast<OType *>(output->data.dptr) : nullptr;
          OType *columnwise_output_ptr =
              use_columnwise ? reinterpret_cast<OType *>(output->columnwise_data.dptr) : nullptr;

          if (current_scaling) {
            zero_grouped_amax_kernel<<<meta_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                amax_ptr, amax_numel, noop_ptr);
            NVTE_CHECK_CUDA(cudaGetLastError());

            grouped_amax_kernel<IS_ACT, ParamOP, OP, IType>
                <<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    input_ptr, amax_ptr, amax_numel, shape_rep, num_tensors, first_logical_dim,
                    last_logical_dim, first_dims, last_dims, input_offsets, logical_total,
                    noop_ptr);
            NVTE_CHECK_CUDA(cudaGetLastError());

            grouped_compute_scale_kernel<OType>
                <<<meta_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    amax_ptr, scale_ptr, scale_inv_ptr, columnwise_scale_inv_ptr, num_tensors,
                    Quantized_Limits<OType>::max_norm, quant_config->force_pow_2_scales,
                    quant_config->amax_epsilon, noop_ptr);
            NVTE_CHECK_CUDA(cudaGetLastError());

            grouped_quantize_fp8_kernel<IS_ACT, true, ParamOP, OP, IType, OType>
                <<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    input_ptr, output_ptr, columnwise_output_ptr, scale_ptr, amax_ptr,
                    scale_inv_ptr, columnwise_scale_inv_ptr, scale_numel, amax_numel, shape_rep,
                    num_tensors, first_logical_dim, last_logical_dim, first_dims, last_dims,
                    input_offsets, output_offsets, logical_total, noop_ptr);
          } else {
            grouped_quantize_fp8_kernel<IS_ACT, false, ParamOP, OP, IType, OType>
                <<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    input_ptr, output_ptr, columnwise_output_ptr, scale_ptr, amax_ptr,
                    scale_inv_ptr, columnwise_scale_inv_ptr, scale_numel, amax_numel, shape_rep,
                    num_tensors, first_logical_dim, last_logical_dim, first_dims, last_dims,
                    input_offsets, output_offsets, logical_total, noop_ptr);
          }
          NVTE_CHECK_CUDA(cudaGetLastError());
      );  // NOLINT(*)
  );      // NOLINT(*)
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
