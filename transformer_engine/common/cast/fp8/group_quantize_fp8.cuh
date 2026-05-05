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

#include "../../common.h"

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

inline void group_quantize(const GroupedTensor *input, const Tensor *noop, GroupedTensor *output,
                           const QuantizationConfig *quant_config, cudaStream_t stream) {
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

  const bool compute_scale_from_amax =
      quant_config != nullptr && quant_config->compute_scale_from_amax;
  (void)compute_scale_from_amax;
  (void)stream;

  NVTE_ERROR("Grouped FP8 tensor-scaling rowwise quantize kernel is not implemented yet.");
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
