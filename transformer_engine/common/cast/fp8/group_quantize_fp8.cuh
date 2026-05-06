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

#include <cfloat>
#include <cstddef>
#include <cstdint>

#include "../../common.h"
#include "../../recipe/recipe_common.cuh"
#include "../../util/cuda_runtime.h"
#include "../../utils.cuh"
#include "../core/common.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace group_quantize_kernel {

constexpr size_t THREADS_PER_BLOCK = 256;

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ size_t get_tensor_rows(
    const size_t tensor_id, const size_t num_tensors, const size_t first_logical_dim,
    const int64_t *const __restrict__ first_dims_ptr) {
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
    return first_logical_dim / num_tensors;
  } else if constexpr (SHAPE_REP == ShapeRepresentation::VARYING_LAST_DIM) {
    return first_logical_dim;
  } else {
    return static_cast<size_t>(first_dims_ptr[tensor_id]);
  }
}

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ size_t get_tensor_cols(
    const size_t tensor_id, const size_t last_logical_dim,
    const int64_t *const __restrict__ last_dims_ptr) {
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS ||
                SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM) {
    return last_logical_dim;
  } else {
    return static_cast<size_t>(last_dims_ptr[tensor_id]);
  }
}

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ size_t get_tensor_id(
    const size_t logical_idx, const size_t num_tensors, const size_t first_logical_dim,
    const size_t last_logical_dim, const int64_t *const __restrict__ offsets_ptr) {
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
    const size_t elems_per_tensor = (first_logical_dim / num_tensors) * last_logical_dim;
    return logical_idx / elems_per_tensor;
  } else {
    size_t low = 1;
    size_t high = num_tensors;
    while (low < high) {
      const size_t mid = low + (high - low) / 2;
      if (static_cast<size_t>(offsets_ptr[mid]) <= logical_idx) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    return low - 1;
  }
}

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ size_t get_tensor_base(
    const size_t tensor_id, const size_t first_logical_dim, const size_t last_logical_dim,
    const size_t num_tensors, const int64_t *const __restrict__ offsets_ptr) {
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
    return tensor_id * (first_logical_dim / num_tensors) * last_logical_dim;
  } else {
    return static_cast<size_t>(offsets_ptr[tensor_id]);
  }
}

__device__ __forceinline__ size_t per_tensor_or_broadcast_index(const size_t tensor_id,
                                                                const size_t numel) {
  return (numel == 1) ? 0 : tensor_id;
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &),
          ShapeRepresentation SHAPE_REP, typename IType>
__global__ void __launch_bounds__(THREADS_PER_BLOCK) grouped_amax_kernel(
    const IType *const __restrict__ input, float *const __restrict__ amax,
    const float *const __restrict__ noop, const size_t total_elements, const size_t num_tensors,
    const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ offsets_ptr, const int64_t *const __restrict__ first_dims_ptr,
    const int64_t *const __restrict__ last_dims_ptr) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements;
       idx += gridDim.x * blockDim.x) {
    const size_t tensor_id =
        get_tensor_id<SHAPE_REP>(idx, num_tensors, first_logical_dim, last_logical_dim, offsets_ptr);
    const size_t tensor_base =
        get_tensor_base<SHAPE_REP>(tensor_id, first_logical_dim, last_logical_dim, num_tensors,
                                   offsets_ptr);
    const size_t local_idx = idx - tensor_base;
    const size_t rows = get_tensor_rows<SHAPE_REP>(tensor_id, num_tensors, first_logical_dim,
                                                   first_dims_ptr);
    const size_t cols = get_tensor_cols<SHAPE_REP>(tensor_id, last_logical_dim, last_dims_ptr);
    if (local_idx >= rows * cols) {
      continue;
    }

    float value = static_cast<float>(input[idx]);
    if constexpr (IS_ACT) {
      value = OP(value, {});
    }
    atomicMaxFloat(&amax[tensor_id], fabsf(value));
  }
}

__launch_bounds__(THREADS_PER_BLOCK) __global__ static void grouped_init_amax_kernel(
    float *const __restrict__ amax, const float *const __restrict__ noop,
    const size_t num_tensors) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_tensors;
       idx += gridDim.x * blockDim.x) {
    amax[idx] = 0.0f;
  }
}

template <typename OType>
__global__ void __launch_bounds__(THREADS_PER_BLOCK) grouped_compute_scale_kernel(
    const float *const __restrict__ amax, float *const __restrict__ scale,
    float *const __restrict__ scale_inv, float *const __restrict__ columnwise_scale_inv,
    const float *const __restrict__ noop, const size_t num_tensors, const bool force_pow_2_scales,
    const float epsilon) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  for (size_t tensor_id = blockIdx.x * blockDim.x + threadIdx.x; tensor_id < num_tensors;
       tensor_id += gridDim.x * blockDim.x) {
    const float scale_value =
        compute_scale_from_amax(amax[tensor_id], Quantized_Limits<OType>::max_norm,
                                force_pow_2_scales, epsilon, FLT_MAX);
    scale[tensor_id] = scale_value;
    if (scale_inv != nullptr) {
      reciprocal<float>(&scale_inv[tensor_id], scale_value);
    }
    if (columnwise_scale_inv != nullptr) {
      reciprocal<float>(&columnwise_scale_inv[tensor_id], scale_value);
    }
  }
}

__launch_bounds__(THREADS_PER_BLOCK) __global__ static void grouped_update_scale_inv_kernel(
    const float *const __restrict__ scale, float *const __restrict__ scale_inv,
    float *const __restrict__ columnwise_scale_inv, const float *const __restrict__ noop,
    const size_t num_tensors, const size_t scale_numel, const size_t scale_inv_numel,
    const size_t columnwise_scale_inv_numel) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  for (size_t tensor_id = blockIdx.x * blockDim.x + threadIdx.x; tensor_id < num_tensors;
       tensor_id += gridDim.x * blockDim.x) {
    const float scale_value = scale[per_tensor_or_broadcast_index(tensor_id, scale_numel)];
    if (scale_inv != nullptr) {
      reciprocal<float>(
          &scale_inv[per_tensor_or_broadcast_index(tensor_id, scale_inv_numel)], scale_value);
    }
    if (columnwise_scale_inv != nullptr) {
      reciprocal<float>(&columnwise_scale_inv[per_tensor_or_broadcast_index(
                             tensor_id, columnwise_scale_inv_numel)],
                        scale_value);
    }
  }
}

template <bool UPDATE_AMAX, bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &),
          ShapeRepresentation SHAPE_REP, typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_BLOCK) grouped_cast_fp8_kernel(
    const IType *const __restrict__ input, OType *const __restrict__ output,
    OType *const __restrict__ columnwise_output, float *const __restrict__ amax,
    const float *const __restrict__ scale, const float *const __restrict__ noop,
    const size_t total_elements, const size_t num_tensors, const size_t first_logical_dim,
    const size_t last_logical_dim, const int64_t *const __restrict__ offsets_ptr,
    const int64_t *const __restrict__ first_dims_ptr, const int64_t *const __restrict__ last_dims_ptr,
    const size_t scale_numel) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements;
       idx += gridDim.x * blockDim.x) {
    const size_t tensor_id =
        get_tensor_id<SHAPE_REP>(idx, num_tensors, first_logical_dim, last_logical_dim, offsets_ptr);
    const size_t tensor_base =
        get_tensor_base<SHAPE_REP>(tensor_id, first_logical_dim, last_logical_dim, num_tensors,
                                   offsets_ptr);
    const size_t local_idx = idx - tensor_base;
    const size_t rows = get_tensor_rows<SHAPE_REP>(tensor_id, num_tensors, first_logical_dim,
                                                   first_dims_ptr);
    const size_t cols = get_tensor_cols<SHAPE_REP>(tensor_id, last_logical_dim, last_dims_ptr);
    if (local_idx >= rows * cols) {
      continue;
    }

    float value = static_cast<float>(input[idx]);
    if constexpr (IS_ACT) {
      value = OP(value, {});
    }

    if constexpr (UPDATE_AMAX) {
      atomicMaxFloat(&amax[tensor_id], fabsf(value));
    }

    const float scale_value = scale[per_tensor_or_broadcast_index(tensor_id, scale_numel)];
    const OType out_value = static_cast<OType>(value * scale_value);
    if (output != nullptr) {
      output[idx] = out_value;
    }
    if (columnwise_output != nullptr) {
      const size_t row = local_idx / cols;
      const size_t col = local_idx - row * cols;
      columnwise_output[tensor_base + col * rows + row] = out_value;
    }
  }
}

inline ShapeRepresentation get_shape_representation(const GroupedTensor *output) {
  if (output->all_same_shape()) {
    return ShapeRepresentation::SAME_BOTH_DIMS;
  }
  if (output->all_same_first_dim()) {
    return ShapeRepresentation::VARYING_LAST_DIM;
  }
  if (output->all_same_last_dim()) {
    return ShapeRepresentation::VARYING_FIRST_DIM;
  }
  if (output->varying_both_dims()) {
    return ShapeRepresentation::VARYING_BOTH_DIMS;
  }
  NVTE_ERROR("Unsupported grouped tensor shape representation.");
}

inline size_t grouped_actual_total_elements(const GroupedTensor *output, cudaStream_t stream) {
  if (!output->tensor_offsets.has_data()) {
    return output->logical_shape.data[0] * output->logical_shape.data[1];
  }

  int64_t total_elements = 0;
  const int64_t *const offsets = reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  NVTE_CHECK_CUDA(cudaMemcpyAsync(&total_elements, offsets + output->num_tensors,
                                  sizeof(total_elements), cudaMemcpyDeviceToHost, stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
  NVTE_CHECK(total_elements >= 0, "Grouped tensor offsets contain a negative total element count.");
  return static_cast<size_t>(total_elements);
}

inline void check_scale_numel(const SimpleTensor &tensor, const size_t num_tensors,
                              const char *name) {
  NVTE_CHECK(tensor.numel() == 1 || tensor.numel() == num_tensors, name,
             " must have either one entry or one entry per grouped tensor (got ", tensor.shape,
             ")");
}

}  // namespace group_quantize_kernel

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void group_quantize(const GroupedTensor *input, const Tensor *noop, GroupedTensor *output,
                    const QuantizationConfig *quant_config, cudaStream_t stream) {
  using namespace group_quantize_kernel;

  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");
  CheckInputGroupedTensor(*input, "cast_input");
  CheckOutputGroupedTensor(*output, "cast_output", false);

  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->logical_shape.ndim == output->logical_shape.ndim,
             "Grouped FP8 input and output logical shapes must have the same rank.");
  for (size_t dim = 0; dim < input->logical_shape.ndim; ++dim) {
    NVTE_CHECK(input->logical_shape.data[dim] == output->logical_shape.data[dim],
               "Grouped FP8 input and output logical shapes must match.");
  }
  NVTE_CHECK(input->all_same_first_dim() == output->all_same_first_dim() &&
                 input->all_same_last_dim() == output->all_same_last_dim(),
             "Grouped FP8 input and output shape metadata must have the same representation.");
  NVTE_CHECK(input->has_data(), "Grouped FP8 quantize requires rowwise input data.");
  NVTE_CHECK(!is_fp8_dtype(input->dtype()), "Input must be in higher precision.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  NVTE_CHECK(output->scale.has_data(), "Grouped FP8 output scale must be allocated.");
  NVTE_CHECK(output->scale.dtype == DType::kFloat32, "Grouped FP8 output scale must be Float32.");
  NVTE_CHECK(output->amax.has_data(), "Grouped FP8 output amax must be allocated.");
  NVTE_CHECK(output->amax.dtype == DType::kFloat32, "Grouped FP8 output amax must be Float32.");
  NVTE_CHECK(output->amax.numel() == output->num_tensors,
             "Grouped FP8 output amax must have one entry per grouped tensor.");

  const bool use_rowwise = output->has_data();
  const bool use_columnwise = output->has_columnwise_data();
  NVTE_CHECK(use_rowwise || use_columnwise,
             "Either rowwise or columnwise grouped FP8 output data need to be allocated.");

  check_scale_numel(output->scale, output->num_tensors, "Grouped FP8 output scale");
  if (use_rowwise) {
    check_scale_numel(output->scale_inv, output->num_tensors,
                      "Grouped FP8 output rowwise scale_inv");
  }
  if (use_columnwise) {
    check_scale_numel(output->columnwise_scale_inv, output->num_tensors,
                      "Grouped FP8 output columnwise scale_inv");
  }

  const ShapeRepresentation shape_rep = get_shape_representation(output);
  const size_t total_elements = grouped_actual_total_elements(output, stream);
  NVTE_CHECK(input->data.numel() >= total_elements, "Grouped FP8 input data has ",
             input->data.numel(), " elements, but ", total_elements, " are required.");
  if (use_rowwise) {
    NVTE_CHECK(output->data.numel() >= total_elements, "Grouped FP8 rowwise output data has ",
               output->data.numel(), " elements, but ", total_elements, " are required.");
  }
  if (use_columnwise) {
    NVTE_CHECK(output->columnwise_data.numel() >= total_elements,
               "Grouped FP8 columnwise output data has ", output->columnwise_data.numel(),
               " elements, but ", total_elements, " are required.");
  }
  if (total_elements == 0) {
    return;
  }

  const size_t num_tensors = output->num_tensors;
  const size_t first_logical_dim = output->logical_shape.data[0];
  const size_t last_logical_dim = output->logical_shape.data[1];
  const int64_t *const offsets_ptr = reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  const int64_t *const first_dims_ptr = reinterpret_cast<const int64_t *>(output->first_dims.dptr);
  const int64_t *const last_dims_ptr = reinterpret_cast<const int64_t *>(output->last_dims.dptr);
  const float *const noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);

  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);
  float *const scale_inv_ptr =
      use_rowwise ? reinterpret_cast<float *>(output->scale_inv.dptr) : nullptr;
  float *const columnwise_scale_inv_ptr =
      use_columnwise ? reinterpret_cast<float *>(output->columnwise_scale_inv.dptr) : nullptr;

  const bool compute_scale_from_amax =
      quant_config != nullptr && quant_config->compute_scale_from_amax;
  if (compute_scale_from_amax) {
    NVTE_CHECK(output->scale.numel() == num_tensors,
               "Grouped FP8 current scaling requires one scale entry per grouped tensor.");
    if (use_rowwise) {
      NVTE_CHECK(output->scale_inv.numel() == num_tensors,
                 "Grouped FP8 current scaling requires one rowwise scale_inv entry per grouped "
                 "tensor.");
    }
    if (use_columnwise) {
      NVTE_CHECK(output->columnwise_scale_inv.numel() == num_tensors,
                 "Grouped FP8 current scaling requires one columnwise scale_inv entry per grouped "
                 "tensor.");
    }
  }

  const dim3 element_grid(DIVUP(total_elements, THREADS_PER_BLOCK));
  const dim3 tensor_grid(DIVUP(num_tensors, THREADS_PER_BLOCK));
  const dim3 block(THREADS_PER_BLOCK);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          const IType *input_ptr = reinterpret_cast<const IType *>(input->data.dptr);
          OType *output_ptr =
              use_rowwise ? reinterpret_cast<OType *>(output->data.dptr) : nullptr;
          OType *columnwise_output_ptr =
              use_columnwise ? reinterpret_cast<OType *>(output->columnwise_data.dptr) : nullptr;

          TRANSFORMER_ENGINE_GROUP_TENSOR_SHAPE_REPRESENTATION_SWITCH(
              shape_rep, SHAPE_REP,
              {
                if (compute_scale_from_amax) {
                  grouped_init_amax_kernel<<<tensor_grid, block, 0, stream>>>(
                      amax_ptr, noop_ptr, num_tensors);
                  grouped_amax_kernel<IS_ACT, ParamOP, OP, SHAPE_REP, IType>
                      <<<element_grid, block, 0, stream>>>(
                          input_ptr, amax_ptr, noop_ptr, total_elements, num_tensors,
                          first_logical_dim, last_logical_dim, offsets_ptr, first_dims_ptr,
                          last_dims_ptr);
                  grouped_compute_scale_kernel<OType><<<tensor_grid, block, 0, stream>>>(
                      amax_ptr, scale_ptr, scale_inv_ptr, columnwise_scale_inv_ptr, noop_ptr,
                      num_tensors, quant_config->force_pow_2_scales, quant_config->amax_epsilon);
                  grouped_cast_fp8_kernel</*UPDATE_AMAX=*/false, IS_ACT, ParamOP, OP, SHAPE_REP,
                                          IType, OType>
                      <<<element_grid, block, 0, stream>>>(
                          input_ptr, output_ptr, columnwise_output_ptr, amax_ptr, scale_ptr,
                          noop_ptr, total_elements, num_tensors, first_logical_dim,
                          last_logical_dim, offsets_ptr, first_dims_ptr, last_dims_ptr,
                          output->scale.numel());
                } else {
                  grouped_init_amax_kernel<<<tensor_grid, block, 0, stream>>>(
                      amax_ptr, noop_ptr, num_tensors);
                  grouped_update_scale_inv_kernel<<<tensor_grid, block, 0, stream>>>(
                      scale_ptr, scale_inv_ptr, columnwise_scale_inv_ptr, noop_ptr, num_tensors,
                      output->scale.numel(), use_rowwise ? output->scale_inv.numel() : 0,
                      use_columnwise ? output->columnwise_scale_inv.numel() : 0);
                  grouped_cast_fp8_kernel</*UPDATE_AMAX=*/true, IS_ACT, ParamOP, OP, SHAPE_REP,
                                          IType, OType>
                      <<<element_grid, block, 0, stream>>>(
                          input_ptr, output_ptr, columnwise_output_ptr, amax_ptr, scale_ptr,
                          noop_ptr, total_elements, num_tensors, first_logical_dim,
                          last_logical_dim, offsets_ptr, first_dims_ptr, last_dims_ptr,
                          output->scale.numel());
                }
              }););  // NOLINT(*)
  );                  // NOLINT(*)

  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
