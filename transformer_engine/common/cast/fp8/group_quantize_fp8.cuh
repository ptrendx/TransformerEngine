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

#include <cuda.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <cstddef>
#include <cstdint>

#include "../../common.h"
#include "../../utils.cuh"
#include "../core/common.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace group_quantize_kernel {

constexpr size_t THREADS_PER_BLOCK = 256;

struct LogicalElement {
  bool valid = false;
  size_t tensor_id = 0;
  size_t row = 0;
  size_t col = 0;
  size_t rows = 0;
  size_t cols = 0;
  size_t local_offset = 0;
  size_t dense_tensor_base = 0;
};

__device__ __forceinline__ size_t scale_index(const size_t numel, const size_t tensor_id) {
  return numel == 1 ? 0 : tensor_id;
}

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ LogicalElement decode_logical_element(
    const size_t logical_idx, const size_t num_tensors, const size_t first_logical_dim,
    const size_t last_logical_dim, const int64_t *const __restrict__ first_dims_ptr,
    const int64_t *const __restrict__ last_dims_ptr) {
  LogicalElement desc;

  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
    const size_t rows = first_logical_dim / num_tensors;
    const size_t cols = last_logical_dim;
    const size_t tensor_numel = rows * cols;
    if (tensor_numel == 0) {
      return desc;
    }
    const size_t tensor_id = logical_idx / tensor_numel;
    if (tensor_id >= num_tensors) {
      return desc;
    }
    const size_t local_offset = logical_idx - tensor_id * tensor_numel;
    desc.valid = true;
    desc.tensor_id = tensor_id;
    desc.row = local_offset / cols;
    desc.col = local_offset % cols;
    desc.rows = rows;
    desc.cols = cols;
    desc.local_offset = local_offset;
    desc.dense_tensor_base = tensor_id * tensor_numel;
  } else if constexpr (SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM) {
    const size_t cols = last_logical_dim;
    if (cols == 0) {
      return desc;
    }
    const size_t logical_row = logical_idx / cols;
    const size_t col = logical_idx % cols;
    size_t row_base = 0;
    size_t dense_tensor_base = 0;
    for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
      const size_t rows = static_cast<size_t>(first_dims_ptr[tensor_id]);
      if (logical_row < row_base + rows) {
        const size_t row = logical_row - row_base;
        desc.valid = true;
        desc.tensor_id = tensor_id;
        desc.row = row;
        desc.col = col;
        desc.rows = rows;
        desc.cols = cols;
        desc.local_offset = row * cols + col;
        desc.dense_tensor_base = dense_tensor_base;
        return desc;
      }
      row_base += rows;
      dense_tensor_base += rows * cols;
    }
  } else if constexpr (SHAPE_REP == ShapeRepresentation::VARYING_LAST_DIM) {
    const size_t rows = first_logical_dim;
    const size_t logical_cols = last_logical_dim;
    if (logical_cols == 0) {
      return desc;
    }
    const size_t row = logical_idx / logical_cols;
    const size_t logical_col = logical_idx % logical_cols;
    size_t col_base = 0;
    size_t dense_tensor_base = 0;
    for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
      const size_t cols = static_cast<size_t>(last_dims_ptr[tensor_id]);
      if (logical_col < col_base + cols) {
        const size_t col = logical_col - col_base;
        desc.valid = row < rows;
        desc.tensor_id = tensor_id;
        desc.row = row;
        desc.col = col;
        desc.rows = rows;
        desc.cols = cols;
        desc.local_offset = row * cols + col;
        desc.dense_tensor_base = dense_tensor_base;
        return desc;
      }
      col_base += cols;
      dense_tensor_base += rows * cols;
    }
  } else if constexpr (SHAPE_REP == ShapeRepresentation::VARYING_BOTH_DIMS) {
    size_t remaining = logical_idx;
    size_t dense_tensor_base = 0;
    for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
      const size_t rows = static_cast<size_t>(first_dims_ptr[tensor_id]);
      const size_t cols = static_cast<size_t>(last_dims_ptr[tensor_id]);
      const size_t tensor_numel = rows * cols;
      if (remaining < tensor_numel) {
        desc.valid = cols != 0;
        desc.tensor_id = tensor_id;
        desc.row = remaining / cols;
        desc.col = remaining % cols;
        desc.rows = rows;
        desc.cols = cols;
        desc.local_offset = remaining;
        desc.dense_tensor_base = dense_tensor_base;
        return desc;
      }
      remaining -= tensor_numel;
      dense_tensor_base += tensor_numel;
    }
  }

  return desc;
}

__device__ __forceinline__ size_t physical_tensor_base(
    const LogicalElement &desc, const int64_t *const __restrict__ offsets_ptr) {
  if (offsets_ptr != nullptr) {
    return static_cast<size_t>(offsets_ptr[desc.tensor_id]);
  }
  return desc.dense_tensor_base;
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          typename OType, ShapeRepresentation SHAPE_REP>
__global__ void group_quantize_fp8_kernel(
    const IType *const __restrict__ input_ptr, const float *const __restrict__ noop_ptr,
    OType *const __restrict__ output_rowwise_ptr,
    OType *const __restrict__ output_colwise_ptr, float *const __restrict__ amax_ptr,
    const size_t amax_numel, float *const __restrict__ scale_inv_ptr,
    const size_t scale_inv_numel, float *const __restrict__ columnwise_scale_inv_ptr,
    const size_t columnwise_scale_inv_numel, const float *const __restrict__ scale_ptr,
    const size_t scale_numel, const size_t num_tensors, const size_t first_logical_dim,
    const size_t last_logical_dim, const size_t logical_numel,
    const int64_t *const __restrict__ input_offsets_ptr,
    const int64_t *const __restrict__ output_offsets_ptr,
    const int64_t *const __restrict__ first_dims_ptr,
    const int64_t *const __restrict__ last_dims_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  for (size_t logical_idx = blockIdx.x * blockDim.x + threadIdx.x; logical_idx < logical_numel;
       logical_idx += gridDim.x * blockDim.x) {
    const LogicalElement desc = decode_logical_element<SHAPE_REP>(
        logical_idx, num_tensors, first_logical_dim, last_logical_dim, first_dims_ptr,
        last_dims_ptr);
    if (!desc.valid) {
      continue;
    }

    const size_t input_base = physical_tensor_base(desc, input_offsets_ptr);
    const size_t output_base = physical_tensor_base(desc, output_offsets_ptr);
    const size_t input_idx = input_base + desc.local_offset;
    const size_t output_rowwise_idx = output_base + desc.local_offset;
    const size_t output_colwise_idx = output_base + desc.col * desc.rows + desc.row;
    const size_t tensor_scale_idx = scale_index(scale_numel, desc.tensor_id);
    const float scale = scale_ptr != nullptr ? scale_ptr[tensor_scale_idx] : 1.0f;

    float elt = static_cast<float>(input_ptr[input_idx]);
    if constexpr (IS_ACT) {
      elt = OP(elt, {});
    }

    if (amax_ptr != nullptr) {
      atomicMaxFloat(&amax_ptr[scale_index(amax_numel, desc.tensor_id)], fabsf(elt));
    }

    const OType quantized = static_cast<OType>(elt * scale);
    if (output_rowwise_ptr != nullptr) {
      output_rowwise_ptr[output_rowwise_idx] = quantized;
    }
    if (output_colwise_ptr != nullptr) {
      output_colwise_ptr[output_colwise_idx] = quantized;
    }

    if (desc.local_offset == 0) {
      if (scale_inv_ptr != nullptr) {
        reciprocal<float>(&scale_inv_ptr[scale_index(scale_inv_numel, desc.tensor_id)], scale);
      }
      if (columnwise_scale_inv_ptr != nullptr) {
        reciprocal<float>(
            &columnwise_scale_inv_ptr[scale_index(columnwise_scale_inv_numel, desc.tensor_id)],
            scale);
      }
    }
  }
}

}  // namespace group_quantize_kernel

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void group_quantize(const GroupedTensor *input, const GroupedTensor *activations,
                    const Tensor *noop, GroupedTensor *output, GroupedTensor *dbias,
                    Tensor *workspace, const QuantizationConfig *quant_config,
                    cudaStream_t stream) {
  using namespace group_quantize_kernel;

  NVTE_CHECK(activations == nullptr, "FP8 grouped quantize does not support activation input.");
  NVTE_CHECK(dbias == nullptr, "FP8 grouped quantize does not support dbias output.");
  NVTE_CHECK(workspace == nullptr, "FP8 grouped quantize does not use workspace.");

  CheckNoopTensor(*noop, "cast_noop");
  CheckInputGroupedTensor(*input, "cast_input");
  CheckOutputGroupedTensor(*output, "cast_output");

  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->logical_shape.ndim == output->logical_shape.ndim,
             "Input and output grouped tensors must have matching logical rank.");
  NVTE_CHECK(input->logical_shape.ndim == 2, "Grouped FP8 quantize expects a 2D logical shape.");
  NVTE_CHECK(input->logical_shape.data[0] == output->logical_shape.data[0] &&
                 input->logical_shape.data[1] == output->logical_shape.data[1],
             "Input and output grouped tensors must have matching logical shapes.");
  NVTE_CHECK(input->has_data(), "Cannot quantize grouped tensor without rowwise input data.");
  NVTE_CHECK(!input->has_columnwise_data(),
             "FP8 grouped quantize expects input data in rowwise layout only.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  NVTE_CHECK(!is_fp8_dtype(input->dtype()), "Input must be in higher precision.");
  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");

  if (quant_config != nullptr && quant_config->stochastic_rounding) {
    NVTE_ERROR("Stochastic rounding is only supported for NVFP4 quantization.");
  }

  const size_t num_tensors = input->num_tensors;
  const size_t first_logical_dim = input->logical_shape.data[0];
  const size_t last_logical_dim = input->logical_shape.data[1];
  const size_t logical_numel = first_logical_dim * last_logical_dim;

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

  auto check_grouped_fp8_param = [num_tensors](const SimpleTensor &tensor, const char *name) {
    if (tensor.has_data()) {
      NVTE_CHECK(tensor.dtype == DType::kFloat32, name, " must have dtype Float32.");
      NVTE_CHECK(tensor.numel() == 1 || tensor.numel() == num_tensors, name,
                 " must have either one entry or one entry per grouped tensor; got shape ",
                 tensor.shape, ".");
    }
  };

  check_grouped_fp8_param(output->scale, "Grouped FP8 scale");
  check_grouped_fp8_param(output->amax, "Grouped FP8 amax");
  if (output->has_data()) {
    check_grouped_fp8_param(output->scale_inv, "Grouped FP8 rowwise scale_inv");
  }
  if (output->has_columnwise_data()) {
    check_grouped_fp8_param(output->columnwise_scale_inv,
                            "Grouped FP8 columnwise scale_inv");
  }

  const size_t scale_numel = output->scale.has_data() ? output->scale.numel() : 1;
  if (output->scale.has_data() && output->scale.numel() != 1) {
    if (output->has_data()) {
      NVTE_CHECK(output->scale_inv.numel() == num_tensors,
                 "Grouped FP8 rowwise scale_inv must have one entry per tensor when scale has one "
                 "entry per tensor.");
    }
    if (output->has_columnwise_data()) {
      NVTE_CHECK(output->columnwise_scale_inv.numel() == num_tensors,
                 "Grouped FP8 columnwise scale_inv must have one entry per tensor when scale has "
                 "one entry per tensor.");
    }
  }

  const int64_t *const input_offsets_ptr =
      reinterpret_cast<const int64_t *>(input->tensor_offsets.dptr);
  const int64_t *const output_offsets_ptr =
      reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  const int64_t *const first_dims_ptr =
      reinterpret_cast<const int64_t *>(output->first_dims.dptr);
  const int64_t *const last_dims_ptr = reinterpret_cast<const int64_t *>(output->last_dims.dptr);

  const float *const noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  float *const columnwise_scale_inv_ptr =
      reinterpret_cast<float *>(output->columnwise_scale_inv.dptr);
  const float *const scale_ptr = reinterpret_cast<const float *>(output->scale.dptr);

  const size_t amax_numel = output->amax.has_data() ? output->amax.numel() : 1;
  const size_t scale_inv_numel = output->scale_inv.has_data() ? output->scale_inv.numel() : 1;
  const size_t columnwise_scale_inv_numel =
      output->columnwise_scale_inv.has_data() ? output->columnwise_scale_inv.numel() : 1;

  if (logical_numel == 0) {
    return;
  }

  const size_t blocks = DIVUP(logical_numel, THREADS_PER_BLOCK);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          TRANSFORMER_ENGINE_GROUP_TENSOR_SHAPE_REPRESENTATION_SWITCH(
              shape_rep, SHAPE_REP,
              {
                const IType *const input_ptr = reinterpret_cast<const IType *>(input->data.dptr);
                OType *const output_rowwise_ptr =
                    output->has_data() ? reinterpret_cast<OType *>(output->data.dptr) : nullptr;
                OType *const output_colwise_ptr =
                    output->has_columnwise_data()
                        ? reinterpret_cast<OType *>(output->columnwise_data.dptr)
                        : nullptr;

                group_quantize_fp8_kernel<IS_ACT, ParamOP, OP, IType, OType, SHAPE_REP>
                    <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
                        input_ptr, noop_ptr, output_rowwise_ptr, output_colwise_ptr, amax_ptr,
                        amax_numel, scale_inv_ptr, scale_inv_numel, columnwise_scale_inv_ptr,
                        columnwise_scale_inv_numel, scale_ptr, scale_numel, num_tensors,
                        first_logical_dim, last_logical_dim, logical_numel, input_offsets_ptr,
                        output_offsets_ptr, first_dims_ptr, last_dims_ptr);
              });  // NOLINT(*)
      );           // NOLINT(*)
  );               // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
