/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_fp8.cuh
 *  \brief CUDA kernels to quantize grouped tensors to FP8 tensor scaling.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_

#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <cfloat>
#include <cstdint>

#include "../../common.h"
#include "../../recipe/recipe_common.cuh"
#include "../../utils.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {

static constexpr const char *kGroupedFP8TensorScalingColumnwiseUnsupported =
    "Grouped FP8 tensor-scaling quantize currently supports rowwise output only; columnwise output "
    "is not implemented.";

static constexpr const char *kGroupedFP8TensorScalingFusedActivationUnsupported =
    "Grouped FP8 tensor-scaling quantize does not support fused grouped activation paths.";

static constexpr const char *kGroupedFP8TensorScalingFusedBackwardUnsupported =
    "Grouped FP8 tensor-scaling quantize does not support fused grouped backward/dbias/dact paths.";

namespace group_quantize_kernel {

constexpr size_t THREADS_PER_BLOCK = 256;

__device__ __forceinline__ bool should_skip(const float *const noop) {
  return noop != nullptr && noop[0] == 1.0f;
}

__device__ __forceinline__ size_t bounded_total_elements(
    const int64_t *const __restrict__ offsets, const size_t num_tensors,
    const size_t actual_total) {
  if (offsets == nullptr) {
    return actual_total;
  }

  const int64_t offsets_total = offsets[num_tensors];
  if (offsets_total <= 0) {
    return 0;
  }

  const size_t offsets_total_size = static_cast<size_t>(offsets_total);
  return offsets_total_size < actual_total ? offsets_total_size : actual_total;
}

__device__ __forceinline__ size_t find_tensor_id(
    const size_t logical_offset, const size_t num_tensors, const size_t same_shape_tensor_elements,
    const int64_t *const __restrict__ offsets) {
  if (offsets == nullptr) {
    const size_t tensor_id = logical_offset / same_shape_tensor_elements;
    return tensor_id < num_tensors ? tensor_id : num_tensors - 1;
  }

  size_t low = 0;
  size_t high = num_tensors;
  while (low + 1 < high) {
    const size_t mid = low + (high - low) / 2;
    const int64_t mid_offset = offsets[mid];
    if (mid_offset <= static_cast<int64_t>(logical_offset)) {
      low = mid;
    } else {
      high = mid;
    }
  }
  return low;
}

__global__ void validate_offsets_kernel(const int64_t *const __restrict__ offsets,
                                        const size_t num_tensors, const size_t actual_total,
                                        const float *const noop) {
  if (should_skip(noop)) {
    return;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (offsets[0] != 0 || offsets[num_tensors] < 0 ||
        static_cast<size_t>(offsets[num_tensors]) != actual_total) {
      NVTE_DEVICE_ERROR("Grouped FP8 tensor_offsets do not match logical_shape actual total.");
    }
  }

  for (size_t tensor_id = blockIdx.x * blockDim.x + threadIdx.x; tensor_id < num_tensors;
       tensor_id += gridDim.x * blockDim.x) {
    const int64_t start = offsets[tensor_id];
    const int64_t end = offsets[tensor_id + 1];
    if (start < 0 || end < 0 || start > end || static_cast<size_t>(end) > actual_total) {
      NVTE_DEVICE_ERROR("Grouped FP8 tensor_offsets must be monotonic within logical_shape.");
    }
  }
}

__global__ void init_delayed_metadata_kernel(float *const amax, float *const scale_inv,
                                             const float *const scale, const size_t num_tensors,
                                             const float *const noop) {
  if (should_skip(noop)) {
    return;
  }

  for (size_t tensor_id = blockIdx.x * blockDim.x + threadIdx.x; tensor_id < num_tensors;
       tensor_id += gridDim.x * blockDim.x) {
    amax[tensor_id] = 0.0f;
    reciprocal<float>(scale_inv + tensor_id, scale[tensor_id]);
  }
}

__global__ void init_current_amax_kernel(float *const amax, const size_t num_tensors,
                                         const float *const noop) {
  if (should_skip(noop)) {
    return;
  }

  for (size_t tensor_id = blockIdx.x * blockDim.x + threadIdx.x; tensor_id < num_tensors;
       tensor_id += gridDim.x * blockDim.x) {
    amax[tensor_id] = 0.0f;
  }
}

template <typename OType>
__global__ void compute_current_scale_kernel(const float *const amax, float *const scale,
                                             float *const scale_inv, const size_t num_tensors,
                                             const bool force_pow_2_scales,
                                             const float amax_epsilon, const float *const noop) {
  if (should_skip(noop)) {
    return;
  }

  for (size_t tensor_id = blockIdx.x * blockDim.x + threadIdx.x; tensor_id < num_tensors;
       tensor_id += gridDim.x * blockDim.x) {
    const float scale_value = compute_scale_from_amax(
        amax[tensor_id], Quantized_Limits<OType>::max_norm, force_pow_2_scales, amax_epsilon,
        FLT_MAX);
    scale[tensor_id] = scale_value;
    reciprocal<float>(scale_inv + tensor_id, scale_value);
  }
}

template <typename IType, typename OType, bool UPDATE_AMAX>
__global__ void grouped_cast_kernel(const IType *const __restrict__ input,
                                    OType *const __restrict__ output, float *const amax,
                                    const float *const __restrict__ scale,
                                    const int64_t *const __restrict__ offsets,
                                    const size_t num_tensors,
                                    const size_t same_shape_tensor_elements,
                                    const size_t actual_total, const float *const noop) {
  if (should_skip(noop)) {
    return;
  }

  const size_t block_start = blockIdx.x * blockDim.x;
  const size_t bounded_total = bounded_total_elements(offsets, num_tensors, actual_total);
  if (block_start >= bounded_total) {
    return;
  }

  const size_t thread_offset = block_start + threadIdx.x;
  const bool thread_has_work = thread_offset < bounded_total;
  const size_t block_end = block_start + blockDim.x - 1;
  const size_t block_last = block_end < bounded_total ? block_end : bounded_total - 1;
  const size_t first_tensor_id =
      find_tensor_id(block_start, num_tensors, same_shape_tensor_elements, offsets);
  const size_t last_tensor_id =
      find_tensor_id(block_last, num_tensors, same_shape_tensor_elements, offsets);
  const bool block_has_single_tensor = first_tensor_id == last_tensor_id;

  float local_amax = 0.0f;
  if (thread_has_work) {
    const size_t tensor_id =
        block_has_single_tensor
            ? first_tensor_id
            : find_tensor_id(thread_offset, num_tensors, same_shape_tensor_elements, offsets);
    const float input_value = static_cast<float>(input[thread_offset]);
    output[thread_offset] = static_cast<OType>(input_value * scale[tensor_id]);

    if constexpr (UPDATE_AMAX) {
      const float abs_value = fabsf(input_value);
      local_amax = isnan(abs_value) ? 0.0f : abs_value;
      if (!block_has_single_tensor) {
        atomicMaxFloat(amax + tensor_id, local_amax);
      }
    }
  }

  if constexpr (UPDATE_AMAX) {
    if (block_has_single_tensor) {
      const int warp_id = threadIdx.x / THREADS_PER_WARP;
      local_amax = reduce_max<THREADS_PER_BLOCK / THREADS_PER_WARP>(local_amax, warp_id);
      if (threadIdx.x == 0) {
        atomicMaxFloat(amax + first_tensor_id, local_amax);
      }
    }
  }
}

template <typename IType>
__global__ void grouped_amax_kernel(const IType *const __restrict__ input, float *const amax,
                                    const int64_t *const __restrict__ offsets,
                                    const size_t num_tensors,
                                    const size_t same_shape_tensor_elements,
                                    const size_t actual_total, const float *const noop) {
  if (should_skip(noop)) {
    return;
  }

  const size_t block_start = blockIdx.x * blockDim.x;
  const size_t bounded_total = bounded_total_elements(offsets, num_tensors, actual_total);
  if (block_start >= bounded_total) {
    return;
  }

  const size_t thread_offset = block_start + threadIdx.x;
  const bool thread_has_work = thread_offset < bounded_total;
  const size_t block_end = block_start + blockDim.x - 1;
  const size_t block_last = block_end < bounded_total ? block_end : bounded_total - 1;
  const size_t first_tensor_id =
      find_tensor_id(block_start, num_tensors, same_shape_tensor_elements, offsets);
  const size_t last_tensor_id =
      find_tensor_id(block_last, num_tensors, same_shape_tensor_elements, offsets);
  const bool block_has_single_tensor = first_tensor_id == last_tensor_id;

  float local_amax = 0.0f;
  if (thread_has_work) {
    const size_t tensor_id =
        block_has_single_tensor
            ? first_tensor_id
            : find_tensor_id(thread_offset, num_tensors, same_shape_tensor_elements, offsets);
    const float abs_value = fabsf(static_cast<float>(input[thread_offset]));
    local_amax = isnan(abs_value) ? 0.0f : abs_value;
    if (!block_has_single_tensor) {
      atomicMaxFloat(amax + tensor_id, local_amax);
    }
  }

  if (block_has_single_tensor) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    local_amax = reduce_max<THREADS_PER_BLOCK / THREADS_PER_WARP>(local_amax, warp_id);
    if (threadIdx.x == 0) {
      atomicMaxFloat(amax + first_tensor_id, local_amax);
    }
  }
}

inline size_t actual_total_from_logical_shape(const GroupedTensor &tensor) {
  NVTE_CHECK(tensor.logical_shape.ndim == 2, "Grouped FP8 tensor logical_shape must be 2D.");
  return tensor.logical_shape.data[0] * tensor.logical_shape.data[1];
}

inline size_t launch_blocks(const size_t elements) {
  return DIVUP(elements, THREADS_PER_BLOCK);
}

inline size_t metadata_blocks(const size_t num_tensors) {
  constexpr size_t max_metadata_blocks = 65535;
  const size_t blocks = DIVUP(num_tensors, THREADS_PER_BLOCK);
  return blocks > max_metadata_blocks ? max_metadata_blocks : blocks;
}

}  // namespace group_quantize_kernel

inline void group_quantize(const GroupedTensor *input, const Tensor *noop, GroupedTensor *output,
                           const QuantizationConfig *quant_config, cudaStream_t stream) {
  using namespace group_quantize_kernel;

  NVTE_CHECK(input != nullptr, "Invalid grouped FP8 quantize input (got NULL)");
  NVTE_CHECK(noop != nullptr, "Invalid grouped FP8 quantize noop tensor (got NULL)");
  NVTE_CHECK(output != nullptr, "Invalid grouped FP8 quantize output (got NULL)");

  CheckNoopTensor(*noop, "cast_noop");
  CheckInputGroupedTensor(*input, "group_cast_input");
  CheckOutputGroupedTensor(*output, "group_cast_output");

  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->logical_shape.ndim == output->logical_shape.ndim,
             "Input and output grouped tensor logical shapes need to match.");
  for (size_t i = 0; i < input->logical_shape.ndim; ++i) {
    NVTE_CHECK(input->logical_shape.data[i] == output->logical_shape.data[i],
               "Input and output grouped tensor logical shapes need to match.");
  }
  NVTE_CHECK(input->has_data(), "Cannot quantize grouped FP8 tensor without rowwise input data.");
  NVTE_CHECK(!is_fp8_dtype(input->dtype()), "Input must be in higher precision.");
  NVTE_CHECK(output->scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Grouped FP8 tensor-scaling quantize expects NVTE_DELAYED_TENSOR_SCALING output.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  NVTE_CHECK(output->has_data(), "Grouped FP8 tensor-scaling quantize requires rowwise output.");
  NVTE_CHECK(!output->has_columnwise_data(), kGroupedFP8TensorScalingColumnwiseUnsupported);

  NVTE_CHECK(input->all_same_first_dim() == output->all_same_first_dim() &&
                 input->all_same_last_dim() == output->all_same_last_dim(),
             "Input and output grouped tensor shape metadata need to match.");

  const size_t actual_total = actual_total_from_logical_shape(*output);
  NVTE_CHECK(actual_total == actual_total_from_logical_shape(*input),
             "Input and output grouped tensor logical sizes need to match.");
  NVTE_CHECK(output->all_same_shape() || output->tensor_offsets.dptr != nullptr,
             "Grouped FP8 varying-shape quantize requires tensor_offsets.");

  const size_t num_tensors = output->num_tensors;
  size_t same_shape_tensor_elements = 0;
  if (output->all_same_shape()) {
    NVTE_CHECK(actual_total % num_tensors == 0,
               "Grouped FP8 same-shape logical size must be divisible by num_tensors.");
    same_shape_tensor_elements = actual_total / num_tensors;
  }

  const bool use_current_scaling = quant_config != nullptr && quant_config->compute_scale_from_amax;
  const bool force_pow_2_scales = quant_config != nullptr && quant_config->force_pow_2_scales;
  const float amax_epsilon = quant_config != nullptr ? quant_config->amax_epsilon : 0.0f;

  const int64_t *const offsets =
      output->all_same_shape() ? nullptr
                               : reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  const float *const noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  const dim3 metadata_grid(metadata_blocks(num_tensors));
  const dim3 metadata_block(THREADS_PER_BLOCK);
  const dim3 data_grid(launch_blocks(actual_total));
  const dim3 data_block(THREADS_PER_BLOCK);

  if (offsets != nullptr) {
    validate_offsets_kernel<<<metadata_grid, metadata_block, 0, stream>>>(offsets, num_tensors,
                                                                          actual_total, noop_ptr);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          const IType *const input_ptr = reinterpret_cast<const IType *>(input->data.dptr);
          OType *const output_ptr = reinterpret_cast<OType *>(output->data.dptr);
          float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
          float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);
          float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);

          if (use_current_scaling) {
            init_current_amax_kernel<<<metadata_grid, metadata_block, 0, stream>>>(amax_ptr,
                                                                                  num_tensors,
                                                                                  noop_ptr);
            NVTE_CHECK_CUDA(cudaGetLastError());

            grouped_amax_kernel<IType><<<data_grid, data_block, 0, stream>>>(
                input_ptr, amax_ptr, offsets, num_tensors, same_shape_tensor_elements,
                actual_total, noop_ptr);
            NVTE_CHECK_CUDA(cudaGetLastError());

            compute_current_scale_kernel<OType><<<metadata_grid, metadata_block, 0, stream>>>(
                amax_ptr, scale_ptr, scale_inv_ptr, num_tensors, force_pow_2_scales, amax_epsilon,
                noop_ptr);
            NVTE_CHECK_CUDA(cudaGetLastError());

            grouped_cast_kernel<IType, OType, /*UPDATE_AMAX=*/false>
                <<<data_grid, data_block, 0, stream>>>(input_ptr, output_ptr, amax_ptr, scale_ptr,
                                                       offsets, num_tensors,
                                                       same_shape_tensor_elements, actual_total,
                                                       noop_ptr);
            NVTE_CHECK_CUDA(cudaGetLastError());
          } else {
            init_delayed_metadata_kernel<<<metadata_grid, metadata_block, 0, stream>>>(
                amax_ptr, scale_inv_ptr, scale_ptr, num_tensors, noop_ptr);
            NVTE_CHECK_CUDA(cudaGetLastError());

            grouped_cast_kernel<IType, OType, /*UPDATE_AMAX=*/true>
                <<<data_grid, data_block, 0, stream>>>(input_ptr, output_ptr, amax_ptr, scale_ptr,
                                                       offsets, num_tensors,
                                                       same_shape_tensor_elements, actual_total,
                                                       noop_ptr);
            NVTE_CHECK_CUDA(cudaGetLastError());
          });  // NOLINT(*)
  );           // NOLINT(*)
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
