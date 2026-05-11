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
#include <limits>

#include "../../common.h"
#include "../../recipe/recipe_common.cuh"
#include "../../util/math.h"
#include "../../utils.cuh"
#include "../core/common.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace group_quantize_current_scaling_kernel {

constexpr size_t THREADS_PER_BLOCK = 256;
constexpr size_t ELEMS_PER_THREAD = 8;
constexpr size_t ELEMS_PER_BLOCK = THREADS_PER_BLOCK * ELEMS_PER_THREAD;

__device__ __forceinline__ bool noop_enabled(const float *noop_ptr) {
  return noop_ptr != nullptr && noop_ptr[0] == 1.0f;
}

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ size_t get_tensor_rows(
    const size_t tensor_id, const size_t first_logical_dim,
    const int64_t *const __restrict__ first_dims_ptr, const size_t num_tensors) {
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
    return first_logical_dim / num_tensors;
  } else {
    return static_cast<size_t>(first_dims_ptr[tensor_id]);
  }
}

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ size_t get_tensor_offset(
    const size_t tensor_id, const size_t rows, const size_t cols,
    const int64_t *const __restrict__ offsets_ptr) {
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
    return tensor_id * rows * cols;
  } else {
    return static_cast<size_t>(offsets_ptr[tensor_id]);
  }
}

template <typename T = float>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    initialize_amax_kernel(float *amax_ptr, const size_t num_tensors, const float *noop_ptr) {
  if (noop_enabled(noop_ptr)) {
    return;
  }

  for (size_t tensor_id = blockIdx.x * blockDim.x + threadIdx.x; tensor_id < num_tensors;
       tensor_id += gridDim.x * blockDim.x) {
    amax_ptr[tensor_id] = 0.0f;
  }
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          ShapeRepresentation SHAPE_REP>
__global__ void __launch_bounds__(THREADS_PER_BLOCK) compute_amax_kernel(
    const IType *const __restrict__ input_ptr, float *const __restrict__ amax_ptr,
    const size_t num_tensors, const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ offsets_ptr, const int64_t *const __restrict__ first_dims_ptr,
    const float *noop_ptr) {
  if (noop_enabled(noop_ptr)) {
    return;
  }

  const size_t tensor_id = blockIdx.y;
  if (tensor_id >= num_tensors) {
    return;
  }

  const size_t rows =
      get_tensor_rows<SHAPE_REP>(tensor_id, first_logical_dim, first_dims_ptr, num_tensors);
  const size_t cols = last_logical_dim;
  const size_t numel = rows * cols;
  const size_t block_base = blockIdx.x * ELEMS_PER_BLOCK;
  if (block_base >= numel) {
    return;
  }

  const size_t tensor_offset = get_tensor_offset<SHAPE_REP>(tensor_id, rows, cols, offsets_ptr);
  const IType *const tensor_input = input_ptr + tensor_offset;

  float thread_amax = 0.0f;
  for (size_t base = block_base + threadIdx.x * ELEMS_PER_THREAD; base < numel;
       base += gridDim.x * ELEMS_PER_BLOCK) {
#pragma unroll
    for (size_t i = 0; i < ELEMS_PER_THREAD; ++i) {
      const size_t idx = base + i;
      if (idx < numel) {
        float value = static_cast<float>(tensor_input[idx]);
        if constexpr (IS_ACT) {
          value = OP(value, {});
        }
        thread_amax = fmaxf(thread_amax, fabsf(value));
      }
    }
  }

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  thread_amax = reduce_max<THREADS_PER_BLOCK / THREADS_PER_WARP>(thread_amax, warp_id);
  if (threadIdx.x == 0) {
    atomicMaxFloat(amax_ptr + tensor_id, thread_amax);
  }
}

template <typename OType>
__global__ void __launch_bounds__(THREADS_PER_BLOCK) compute_scale_kernel(
    const float *const __restrict__ amax_ptr, float *const __restrict__ scale_ptr,
    float *const __restrict__ scale_inv_ptr, float *const __restrict__ columnwise_scale_inv_ptr,
    const size_t num_tensors, const bool force_pow_2_scales, const float epsilon,
    const float *noop_ptr) {
  if (noop_enabled(noop_ptr)) {
    return;
  }

  for (size_t tensor_id = blockIdx.x * blockDim.x + threadIdx.x; tensor_id < num_tensors;
       tensor_id += gridDim.x * blockDim.x) {
    constexpr float max_fp8 = TypeInfo<OType>::max_finite_value;
    const float scale = compute_scale_from_amax(amax_ptr[tensor_id], max_fp8, force_pow_2_scales,
                                                epsilon, std::numeric_limits<float>::max());
    scale_ptr[tensor_id] = scale;
    if (scale_inv_ptr != nullptr) {
      reciprocal<float>(scale_inv_ptr + tensor_id, scale);
    }
    if (columnwise_scale_inv_ptr != nullptr) {
      reciprocal<float>(columnwise_scale_inv_ptr + tensor_id, scale);
    }
  }
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          typename OType, ShapeRepresentation SHAPE_REP>
__global__ void __launch_bounds__(THREADS_PER_BLOCK) cast_fp8_kernel(
    const IType *const __restrict__ input_ptr, OType *const __restrict__ output_ptr,
    OType *const __restrict__ columnwise_output_ptr, const float *const __restrict__ scale_ptr,
    const size_t num_tensors, const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ offsets_ptr, const int64_t *const __restrict__ first_dims_ptr,
    const float *noop_ptr) {
  if (noop_enabled(noop_ptr)) {
    return;
  }

  const size_t tensor_id = blockIdx.y;
  if (tensor_id >= num_tensors) {
    return;
  }

  const size_t rows =
      get_tensor_rows<SHAPE_REP>(tensor_id, first_logical_dim, first_dims_ptr, num_tensors);
  const size_t cols = last_logical_dim;
  const size_t numel = rows * cols;
  const size_t block_base = blockIdx.x * ELEMS_PER_BLOCK;
  if (block_base >= numel) {
    return;
  }

  const size_t tensor_offset = get_tensor_offset<SHAPE_REP>(tensor_id, rows, cols, offsets_ptr);
  const IType *const tensor_input = input_ptr + tensor_offset;
  OType *const tensor_output = output_ptr != nullptr ? output_ptr + tensor_offset : nullptr;
  OType *const tensor_columnwise_output =
      columnwise_output_ptr != nullptr ? columnwise_output_ptr + tensor_offset : nullptr;
  const float scale = scale_ptr[tensor_id];

  for (size_t base = block_base + threadIdx.x * ELEMS_PER_THREAD; base < numel;
       base += gridDim.x * ELEMS_PER_BLOCK) {
#pragma unroll
    for (size_t i = 0; i < ELEMS_PER_THREAD; ++i) {
      const size_t idx = base + i;
      if (idx < numel) {
        float value = static_cast<float>(tensor_input[idx]);
        if constexpr (IS_ACT) {
          value = OP(value, {});
        }
        const OType quantized = static_cast<OType>(value * scale);
        if (tensor_output != nullptr) {
          tensor_output[idx] = quantized;
        }
        if (tensor_columnwise_output != nullptr) {
          const size_t row = idx / cols;
          const size_t col = idx - row * cols;
          tensor_columnwise_output[col * rows + row] = quantized;
        }
      }
    }
  }
}

}  // namespace group_quantize_current_scaling_kernel

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void group_quantize_current_scaling(const GroupedTensor *input, const Tensor *noop,
                                    GroupedTensor *output,
                                    const QuantizationConfig *quant_config,
                                    cudaStream_t stream) {
  using namespace group_quantize_current_scaling_kernel;

  CheckNoopTensor(*noop, "cast_noop");
  CheckInputGroupedTensor(*input, "cast_input");
  CheckOutputGroupedTensor(*output, "cast_output", false);

  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->logical_shape.ndim == 2 && output->logical_shape.ndim == 2,
             "Input and output grouped tensor logical shapes must be 2D.");
  NVTE_CHECK(input->logical_shape.data[0] == output->logical_shape.data[0] &&
                 input->logical_shape.data[1] == output->logical_shape.data[1],
             "Input and output grouped tensor logical shapes must match.");
  NVTE_CHECK(input->has_data(), "Cannot quantize grouped tensor without rowwise input data.");
  NVTE_CHECK(!is_fp8_dtype(input->dtype()), "Input must be in higher precision.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");
  NVTE_CHECK(output->scale.has_data(), "Grouped FP8 current scaling requires scale tensor.");
  NVTE_CHECK(output->amax.has_data(), "Grouped FP8 current scaling requires amax tensor.");
  NVTE_CHECK(output->scale.dtype == DType::kFloat32,
             "Grouped FP8 current scaling requires FP32 scale tensor.");
  NVTE_CHECK(output->amax.dtype == DType::kFloat32,
             "Grouped FP8 current scaling requires FP32 amax tensor.");
  if (output->has_data()) {
    NVTE_CHECK(output->scale_inv.has_data(),
               "Grouped FP8 current scaling requires rowwise scale_inv tensor.");
    NVTE_CHECK(output->scale_inv.dtype == DType::kFloat32,
               "Grouped FP8 current scaling requires FP32 rowwise scale_inv tensor.");
  }
  if (output->has_columnwise_data()) {
    NVTE_CHECK(output->columnwise_scale_inv.has_data(),
               "Grouped FP8 current scaling requires columnwise scale_inv tensor.");
    NVTE_CHECK(output->columnwise_scale_inv.dtype == DType::kFloat32,
               "Grouped FP8 current scaling requires FP32 columnwise scale_inv tensor.");
  }

  ShapeRepresentation shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  if (output->all_same_shape()) {
    NVTE_CHECK(output->logical_shape.data[0] % output->num_tensors == 0,
               "Grouped FP8 current scaling requires logical first dimension to be divisible by "
               "num_tensors when first_dims is not provided.");
    shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  } else if (output->all_same_last_dim()) {
    shape_rep = ShapeRepresentation::VARYING_FIRST_DIM;
  } else {
    NVTE_ERROR("Grouped FP8 current scaling only supports a common last dimension.");
  }

  const size_t num_tensors = output->num_tensors;
  const size_t first_logical_dim = output->logical_shape.data[0];
  const size_t last_logical_dim = output->logical_shape.data[1];
  const size_t total_elements = first_logical_dim * last_logical_dim;

  const int64_t *const offsets_ptr = reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  const int64_t *const first_dims_ptr = reinterpret_cast<const int64_t *>(output->first_dims.dptr);
  if (shape_rep == ShapeRepresentation::VARYING_FIRST_DIM) {
    NVTE_CHECK(offsets_ptr != nullptr,
               "Grouped FP8 current scaling requires tensor_offsets for varying first dims.");
    NVTE_CHECK(first_dims_ptr != nullptr,
               "Grouped FP8 current scaling requires first_dims for varying first dims.");
  }

  const float *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  float *const columnwise_scale_inv_ptr =
      reinterpret_cast<float *>(output->columnwise_scale_inv.dptr);

  const bool force_pow_2_scales =
      quant_config != nullptr ? quant_config->force_pow_2_scales : false;
  const float epsilon = quant_config != nullptr ? quant_config->amax_epsilon : 0.0f;

  const size_t init_blocks = std::max<size_t>(1, DIVUP(num_tensors, THREADS_PER_BLOCK));
  initialize_amax_kernel<float>
      <<<init_blocks, THREADS_PER_BLOCK, 0, stream>>>(amax_ptr, num_tensors, noop_ptr);
  NVTE_CHECK_CUDA(cudaGetLastError());

  const size_t average_tensor_elements = DIVUP(total_elements, num_tensors);
  const size_t blocks_per_tensor =
      std::max<size_t>(1, std::min<size_t>(DIVUP(average_tensor_elements, ELEMS_PER_BLOCK), 65535));
  const dim3 grouped_grid(blocks_per_tensor, num_tensors);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          TRANSFORMER_ENGINE_GROUP_TENSOR_SHAPE_REPRESENTATION_SWITCH(
              shape_rep, SHAPE_REP,
              compute_amax_kernel<IS_ACT, ParamOP, OP, IType, SHAPE_REP>
              <<<grouped_grid, THREADS_PER_BLOCK, 0, stream>>>(
                  reinterpret_cast<const IType *>(input->data.dptr), amax_ptr, num_tensors,
                  first_logical_dim, last_logical_dim, offsets_ptr, first_dims_ptr, noop_ptr);
              NVTE_CHECK_CUDA(cudaGetLastError());

              const size_t scale_blocks =
                  std::max<size_t>(1, DIVUP(num_tensors, THREADS_PER_BLOCK));
              compute_scale_kernel<OType><<<scale_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                  amax_ptr, scale_ptr, scale_inv_ptr, columnwise_scale_inv_ptr, num_tensors,
                  force_pow_2_scales, epsilon, noop_ptr);
              NVTE_CHECK_CUDA(cudaGetLastError());

              OType *const output_ptr =
                  output->has_data() ? reinterpret_cast<OType *>(output->data.dptr) : nullptr;
              OType *const columnwise_output_ptr = output->has_columnwise_data()
                                                       ? reinterpret_cast<OType *>(
                                                             output->columnwise_data.dptr)
                                                       : nullptr;
              cast_fp8_kernel<IS_ACT, ParamOP, OP, IType, OType, SHAPE_REP>
              <<<grouped_grid, THREADS_PER_BLOCK, 0, stream>>>(
                  reinterpret_cast<const IType *>(input->data.dptr), output_ptr,
                  columnwise_output_ptr, scale_ptr, num_tensors, first_logical_dim,
                  last_logical_dim, offsets_ptr, first_dims_ptr, noop_ptr);
              NVTE_CHECK_CUDA(cudaGetLastError()););););  // NOLINT(*)
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
