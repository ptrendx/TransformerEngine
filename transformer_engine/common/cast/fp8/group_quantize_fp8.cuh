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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "../../common.h"
#include "../../recipe/recipe_common.cuh"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace group_quantize_kernel {

constexpr int THREADS_PER_BLOCK = 256;
constexpr int ROWWISE_THREADS_PER_BLOCK = 512;
constexpr int ROWWISE_TARGET_INPUT_BYTES_PER_THREAD = 32;

template <typename IType>
struct RowwiseElemsPerThread {
  static constexpr int value =
      (ROWWISE_TARGET_INPUT_BYTES_PER_THREAD / static_cast<int>(sizeof(IType)) > 0)
          ? ROWWISE_TARGET_INPUT_BYTES_PER_THREAD / static_cast<int>(sizeof(IType))
          : 1;
};

static __global__ void initialize_current_scaling_metadata_kernel(float *amax, float *scale,
                                                                  float *scale_inv,
                                                                  const size_t num_tensors,
                                                                  const float *noop) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const size_t tensor_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (tensor_id >= num_tensors) {
    return;
  }

  amax[tensor_id] = 0.0f;
  scale[tensor_id] = 1.0f;
  scale_inv[tensor_id] = 1.0f;
}

template <bool VARYING_FIRST_DIM>
__device__ __forceinline__ bool row_to_tensor(
    const size_t row_id, const size_t num_tensors, const size_t first_logical_dim,
    const size_t last_dim, const int64_t *const __restrict__ offsets,
    size_t *const __restrict__ tensor_id, size_t *const __restrict__ tensor_start,
    size_t *const __restrict__ row_in_tensor) {
  if constexpr (VARYING_FIRST_DIM) {
    const size_t actual_rows = static_cast<size_t>(offsets[num_tensors]) / last_dim;
    if (row_id >= actual_rows) {
      return false;
    }

    size_t low = 1;
    size_t high = num_tensors;
    while (low < high) {
      const size_t mid = low + (high - low) / 2;
      const size_t mid_row = static_cast<size_t>(offsets[mid]) / last_dim;
      if (mid_row <= row_id) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }

    *tensor_id = low - 1;
    *tensor_start = static_cast<size_t>(offsets[*tensor_id]);
    const size_t tensor_start_row = *tensor_start / last_dim;
    *row_in_tensor = row_id - tensor_start_row;
    return true;
  } else {
    const size_t rows_per_tensor = first_logical_dim / num_tensors;
    if (row_id >= rows_per_tensor * num_tensors) {
      return false;
    }
    *tensor_id = row_id / rows_per_tensor;
    *row_in_tensor = row_id - *tensor_id * rows_per_tensor;
    *tensor_start = *tensor_id * rows_per_tensor * last_dim;
    return true;
  }
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          int ELEMS_PER_THREAD>
__device__ __forceinline__ float compute_vector_amax(
    const Vec<IType, ELEMS_PER_THREAD> &input_vec, const size_t count) {
  float thread_amax = 0.0f;

#pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    if (static_cast<size_t>(i) < count) {
      float elt = static_cast<float>(input_vec.data.elt[i]);
      if constexpr (IS_ACT) {
        elt = OP(elt, {});
      }
      __builtin_assume(thread_amax >= 0.0f);
      thread_amax = fmaxf(thread_amax, fabsf(elt));
    }
  }

  return thread_amax;
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          typename OType, int ELEMS_PER_THREAD>
__device__ __forceinline__ void cast_scaled_vector(
    const Vec<IType, ELEMS_PER_THREAD> &input_vec, Vec<OType, ELEMS_PER_THREAD> *output_vec,
    const size_t count, const float scale) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if constexpr (!IS_ACT && ELEMS_PER_THREAD % 4 == 0 &&
                (std::is_same_v<IType, bf16> || std::is_same_v<IType, fp16> ||
                 std::is_same_v<IType, float>) &&
                (std::is_same_v<OType, fp8e4m3> || std::is_same_v<OType, fp8e5m2>)) {
    const ptx::floatx2 scale_2x{scale, scale};
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i += 4) {
      if (static_cast<size_t>(i + 4) <= count) {
        const auto in_4x =
            *reinterpret_cast<const ptx::FPx4<IType> *>(&input_vec.data.elt[i]);
        ptx::FPx4<OType> out_4x;
        ptx::mul_cvt_4x(out_4x, in_4x, scale_2x);
        *reinterpret_cast<ptx::FPx4<OType> *>(&output_vec->data.elt[i]) = out_4x;
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int elem = i + j;
          if (static_cast<size_t>(elem) < count) {
            output_vec->data.elt[elem] =
                static_cast<OType>(static_cast<float>(input_vec.data.elt[elem]) * scale);
          }
        }
      }
    }
    return;
  }
#endif
#pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    if (static_cast<size_t>(i) < count) {
      float elt = static_cast<float>(input_vec.data.elt[i]);
      if constexpr (IS_ACT) {
        elt = OP(elt, {});
      }
      output_vec->data.elt[i] = static_cast<OType>(elt * scale);
    }
  }
}

template <bool VARYING_FIRST_DIM, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &), typename IType, int ELEMS_PER_THREAD>
static __global__ void grouped_rowwise_amax_kernel(
    const IType *const __restrict__ input, float *const __restrict__ amax,
    const size_t num_tensors, const size_t first_logical_dim, const size_t last_dim,
    const size_t column_tiles, const int64_t *const __restrict__ offsets, const float *noop) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const size_t tile_id = blockIdx.x;
  const size_t row_id = tile_id / column_tiles;
  const size_t column_tile = tile_id - row_id * column_tiles;

  __shared__ size_t shared_tensor_id;
  __shared__ size_t shared_tensor_start;
  __shared__ size_t shared_row_in_tensor;
  __shared__ int shared_is_valid_row;
  if (threadIdx.x == 0) {
    size_t tensor_id = 0;
    size_t tensor_start = 0;
    size_t row_in_tensor = 0;
    shared_is_valid_row = row_to_tensor<VARYING_FIRST_DIM>(
        row_id, num_tensors, first_logical_dim, last_dim, offsets, &tensor_id, &tensor_start,
        &row_in_tensor);
    shared_tensor_id = tensor_id;
    shared_tensor_start = tensor_start;
    shared_row_in_tensor = row_in_tensor;
  }
  __syncthreads();

  if (!shared_is_valid_row) {
    return;
  }

  const size_t column =
      (column_tile * blockDim.x + threadIdx.x) * static_cast<size_t>(ELEMS_PER_THREAD);
  float thread_amax = 0.0f;
  if (column < last_dim) {
    const size_t remaining_columns = last_dim - column;
    const size_t count = remaining_columns < static_cast<size_t>(ELEMS_PER_THREAD)
                             ? remaining_columns
                             : static_cast<size_t>(ELEMS_PER_THREAD);
    Vec<IType, ELEMS_PER_THREAD> input_vec;
    input_vec.load_from_elts(input + shared_tensor_start + shared_row_in_tensor * last_dim, column,
                             count);
    thread_amax =
        compute_vector_amax<IS_ACT, ParamOP, OP, IType, ELEMS_PER_THREAD>(input_vec, count);
  }

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  thread_amax = reduce_max<ROWWISE_THREADS_PER_BLOCK / THREADS_PER_WARP>(thread_amax, warp_id);
  if (threadIdx.x == 0) {
    atomicMaxFloat(amax + shared_tensor_id, thread_amax);
  }
}

template <bool VARYING_FIRST_DIM, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &), typename IType, typename OType,
          int ELEMS_PER_THREAD>
static __global__ void grouped_rowwise_cast_fp8_kernel(
    const IType *const __restrict__ input, OType *const __restrict__ output,
    const float *const __restrict__ scale, const size_t num_tensors,
    const size_t first_logical_dim, const size_t last_dim, const size_t column_tiles,
    const int64_t *const __restrict__ offsets, const float *noop) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const size_t tile_id = blockIdx.x;
  const size_t row_id = tile_id / column_tiles;
  const size_t column_tile = tile_id - row_id * column_tiles;

  __shared__ size_t shared_tensor_id;
  __shared__ size_t shared_tensor_start;
  __shared__ size_t shared_row_in_tensor;
  __shared__ int shared_is_valid_row;
  if (threadIdx.x == 0) {
    size_t tensor_id = 0;
    size_t tensor_start = 0;
    size_t row_in_tensor = 0;
    shared_is_valid_row = row_to_tensor<VARYING_FIRST_DIM>(
        row_id, num_tensors, first_logical_dim, last_dim, offsets, &tensor_id, &tensor_start,
        &row_in_tensor);
    shared_tensor_id = tensor_id;
    shared_tensor_start = tensor_start;
    shared_row_in_tensor = row_in_tensor;
  }
  __syncthreads();

  if (!shared_is_valid_row) {
    return;
  }

  const size_t column =
      (column_tile * blockDim.x + threadIdx.x) * static_cast<size_t>(ELEMS_PER_THREAD);
  if (column >= last_dim) {
    return;
  }

  const size_t remaining_columns = last_dim - column;
  const size_t count = remaining_columns < static_cast<size_t>(ELEMS_PER_THREAD)
                           ? remaining_columns
                           : static_cast<size_t>(ELEMS_PER_THREAD);
  const size_t row_offset = shared_tensor_start + shared_row_in_tensor * last_dim;
  Vec<IType, ELEMS_PER_THREAD> input_vec;
  Vec<OType, ELEMS_PER_THREAD> output_vec;
  input_vec.load_from_elts(input + row_offset, column, count);
  cast_scaled_vector<IS_ACT, ParamOP, OP, IType, OType, ELEMS_PER_THREAD>(
      input_vec, &output_vec, count, scale[shared_tensor_id]);
  output_vec.store_to_elts(output + row_offset, column, count);
}

template <typename OType>
__global__ void compute_grouped_scale_kernel(const float *const __restrict__ amax,
                                             float *const __restrict__ scale,
                                             float *const __restrict__ scale_inv,
                                             const size_t num_tensors,
                                             const bool force_pow_2_scales,
                                             const float epsilon, const float *noop) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const size_t tensor_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (tensor_id >= num_tensors) {
    return;
  }

  const float scale_val = compute_scale_from_amax(
      amax[tensor_id], Quantized_Limits<OType>::max_norm, force_pow_2_scales, epsilon,
      std::numeric_limits<float>::max());
  scale[tensor_id] = scale_val;
  reciprocal<float>(scale_inv + tensor_id, scale_val);
}

}  // namespace group_quantize_kernel

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void group_quantize(const GroupedTensor *input, const Tensor *noop, GroupedTensor *output,
                    const QuantizationConfig *quant_config, cudaStream_t stream) {
  using namespace group_quantize_kernel;

  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");
  CheckInputGroupedTensor(*input, "group_fp8_quantize_input");
  CheckOutputGroupedTensor(*output, "group_fp8_quantize_output", false);

  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->has_data(), "Cannot quantize grouped tensor without rowwise data.");
  NVTE_CHECK(output->has_data(), "Grouped FP8 current scaling requires rowwise output data.");
  NVTE_CHECK(!output->has_columnwise_data(),
             "Grouped FP8 current scaling does not support columnwise output yet.");
  NVTE_CHECK(output->all_same_last_dim(),
             "Grouped FP8 current scaling requires a common last dimension.");
  NVTE_CHECK(input->logical_shape.ndim == 2 && output->logical_shape.ndim == 2,
             "Grouped FP8 current scaling requires 2D logical shapes.");
  NVTE_CHECK(input->logical_shape.data[0] == output->logical_shape.data[0] &&
                 input->logical_shape.data[1] == output->logical_shape.data[1],
             "Input and output grouped logical shapes must match.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  NVTE_CHECK(!is_fp8_dtype(input->dtype()), "Input must be in higher precision.");

  const size_t num_tensors = output->num_tensors;
  const size_t first_logical_dim = output->logical_shape.data[0];
  const size_t last_logical_dim = output->logical_shape.data[1];
  const size_t total_elements = first_logical_dim * last_logical_dim;

  NVTE_CHECK(input->data.numel() == total_elements,
             "Input grouped data size must match the grouped logical shape.");
  NVTE_CHECK(output->data.numel() == total_elements,
             "Output grouped data size must match the grouped logical shape.");
  NVTE_CHECK(output->amax.dptr != nullptr && output->amax.dtype == DType::kFloat32 &&
                 output->amax.numel() == num_tensors,
             "Grouped FP8 current scaling requires one FP32 amax per tensor.");
  NVTE_CHECK(output->scale.dptr != nullptr && output->scale.dtype == DType::kFloat32 &&
                 output->scale.numel() == num_tensors,
             "Grouped FP8 current scaling requires one FP32 scale per tensor.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr && output->scale_inv.dtype == DType::kFloat32 &&
                 output->scale_inv.numel() == num_tensors,
             "Grouped FP8 current scaling requires one FP32 scale_inv per tensor.");

  const bool varying_first_dim = output->first_dims.has_data();
  NVTE_CHECK(!varying_first_dim || output->tensor_offsets.has_data(),
             "Grouped FP8 current scaling requires tensor_offsets when first_dims is set.");
  NVTE_CHECK(varying_first_dim || first_logical_dim % num_tensors == 0,
             "Uniform grouped FP8 current scaling requires first dimension divisible by "
             "num_tensors.");

  const int64_t *const offsets_ptr =
      varying_first_dim ? reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr) : nullptr;
  const float *const noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);

  const bool force_pow_2_scales = quant_config != nullptr && quant_config->force_pow_2_scales;
  const float amax_epsilon = quant_config != nullptr ? quant_config->amax_epsilon : 0.0f;

  const dim3 metadata_grid(DIVUP(num_tensors, static_cast<size_t>(THREADS_PER_BLOCK)));
  initialize_current_scaling_metadata_kernel<<<metadata_grid, THREADS_PER_BLOCK, 0, stream>>>(
      amax_ptr, scale_ptr, scale_inv_ptr, num_tensors, noop_ptr);
  NVTE_CHECK_CUDA(cudaGetLastError());

  if (total_elements == 0) {
    return;
  }

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          constexpr int elems_per_thread = RowwiseElemsPerThread<IType>::value;
          const size_t column_tiles =
              DIVUP(last_logical_dim,
                    static_cast<size_t>(ROWWISE_THREADS_PER_BLOCK * elems_per_thread));
          const size_t rowwise_blocks = first_logical_dim * column_tiles;
          NVTE_CHECK(rowwise_blocks <= static_cast<size_t>(std::numeric_limits<int>::max()),
                     "Grouped FP8 current scaling rowwise grid is too large.");
          const dim3 rowwise_grid(static_cast<unsigned int>(rowwise_blocks));

          if (varying_first_dim) {
            grouped_rowwise_amax_kernel<true, IS_ACT, ParamOP, OP, IType, elems_per_thread>
            <<<rowwise_grid, ROWWISE_THREADS_PER_BLOCK, 0, stream>>>(
                reinterpret_cast<const IType *>(input->data.dptr), amax_ptr, num_tensors,
                first_logical_dim, last_logical_dim, column_tiles, offsets_ptr, noop_ptr);
          } else {
            grouped_rowwise_amax_kernel<false, IS_ACT, ParamOP, OP, IType, elems_per_thread>
            <<<rowwise_grid, ROWWISE_THREADS_PER_BLOCK, 0, stream>>>(
                reinterpret_cast<const IType *>(input->data.dptr), amax_ptr, num_tensors,
                first_logical_dim, last_logical_dim, column_tiles, offsets_ptr, noop_ptr);
          }
          NVTE_CHECK_CUDA(cudaGetLastError());

          compute_grouped_scale_kernel<OType>
          <<<metadata_grid, THREADS_PER_BLOCK, 0, stream>>>(
              amax_ptr, scale_ptr, scale_inv_ptr, num_tensors, force_pow_2_scales, amax_epsilon,
              noop_ptr);
          NVTE_CHECK_CUDA(cudaGetLastError());

          if (varying_first_dim) {
            grouped_rowwise_cast_fp8_kernel<true, IS_ACT, ParamOP, OP, IType, OType,
                                            elems_per_thread>
            <<<rowwise_grid, ROWWISE_THREADS_PER_BLOCK, 0, stream>>>(
                reinterpret_cast<const IType *>(input->data.dptr),
                reinterpret_cast<OType *>(output->data.dptr), scale_ptr, num_tensors,
                first_logical_dim, last_logical_dim, column_tiles, offsets_ptr, noop_ptr);
          } else {
            grouped_rowwise_cast_fp8_kernel<false, IS_ACT, ParamOP, OP, IType, OType,
                                            elems_per_thread>
            <<<rowwise_grid, ROWWISE_THREADS_PER_BLOCK, 0, stream>>>(
                reinterpret_cast<const IType *>(input->data.dptr),
                reinterpret_cast<OType *>(output->data.dptr), scale_ptr, num_tensors,
                first_logical_dim, last_logical_dim, column_tiles, offsets_ptr, noop_ptr);
          }
          NVTE_CHECK_CUDA(cudaGetLastError()););  // NOLINT(*)
  );                                             // NOLINT(*)
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
