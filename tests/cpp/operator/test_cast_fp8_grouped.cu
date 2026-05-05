/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

enum class ScalingKind { Delayed, Current };
enum class ShapeKind { SameShape, VaryingFirstDim };

constexpr const char *kColumnwiseUnsupported =
    "Grouped FP8 tensor-scaling quantize currently supports rowwise output only; columnwise output "
    "is not implemented.";

struct GroupedHandleDeleter {
  void operator()(NVTEGroupedTensor tensor) const {
    if (tensor != nullptr) {
      nvte_destroy_grouped_tensor(tensor);
    }
  }
};

using GroupedHandle = std::unique_ptr<std::remove_pointer_t<NVTEGroupedTensor>,
                                      GroupedHandleDeleter>;

template <typename T>
uint8_t raw_byte(const T &value) {
  static_assert(sizeof(T) == 1, "FP8 output byte comparisons expect one-byte output elements.");
  uint8_t byte = 0;
  std::memcpy(&byte, &value, sizeof(byte));
  return byte;
}

template <typename InputType>
float to_float(const InputType &value) {
  return static_cast<float>(value);
}

template <typename OutputType>
float max_fp8_value() {
  return Quantized_Limits<OutputType>::max();
}

float compute_current_scale_ref(float amax, float max_fp8, bool force_pow_2_scales,
                                float epsilon) {
  if (amax < epsilon) {
    amax = epsilon;
  }

  float scale = 1.0f;
  if (std::isinf(amax) || amax == 0.0f || std::isnan(amax)) {
    return scale;
  }

  scale = max_fp8 / amax;
  if (std::isinf(scale)) {
    scale = std::numeric_limits<float>::max();
  }
  if (force_pow_2_scales) {
    uint32_t scale_bits = 0;
    std::memcpy(&scale_bits, &scale, sizeof(scale_bits));
    scale_bits &= 0xFF800000;
    std::memcpy(&scale, &scale_bits, sizeof(scale));
  }
  return scale;
}

template <typename InputType>
std::vector<InputType> make_input(const std::vector<int64_t> &first_dims, const size_t last_dim,
                                  const std::vector<int64_t> &offsets,
                                  const size_t allocation_total) {
  std::vector<InputType> input(allocation_total, static_cast<InputType>(200.0f));
  for (size_t tensor_id = 0; tensor_id < first_dims.size(); ++tensor_id) {
    const size_t tensor_start = static_cast<size_t>(offsets[tensor_id]);
    const size_t tensor_elems = static_cast<size_t>(first_dims[tensor_id]) * last_dim;
    const float tensor_amax = (tensor_id % 3 == 0) ? 1.75f : ((tensor_id % 3 == 1) ? 3.5f : 0.875f);
    for (size_t i = 0; i < tensor_elems; ++i) {
      float value = 0.0f;
      if (i == 0) {
        value = tensor_amax;
      } else if (i == 1) {
        value = -0.5f * tensor_amax;
      } else {
        const int bucket = static_cast<int>(i % 7) - 3;
        value = tensor_amax * static_cast<float>(bucket) * 0.125f;
      }
      input[tensor_start + i] = static_cast<InputType>(value);
    }
  }
  return input;
}

std::vector<int64_t> make_offsets(const std::vector<int64_t> &first_dims, const size_t last_dim) {
  std::vector<int64_t> offsets(first_dims.size() + 1, 0);
  for (size_t i = 0; i < first_dims.size(); ++i) {
    offsets[i + 1] = offsets[i] + first_dims[i] * static_cast<int64_t>(last_dim);
  }
  return offsets;
}

std::vector<int64_t> first_dims_for_shape(const ShapeKind shape_kind) {
  switch (shape_kind) {
    case ShapeKind::SameShape:
      return {2, 2, 2};
    case ShapeKind::VaryingFirstDim:
      return {2, 0, 4};
  }
  NVTE_ERROR("Invalid grouped FP8 shape kind.");
  return {};
}

std::string to_string(const ScalingKind scaling_kind) {
  switch (scaling_kind) {
    case ScalingKind::Delayed:
      return "Delayed";
    case ScalingKind::Current:
      return "Current";
  }
  return "";
}

std::string to_string(const ShapeKind shape_kind) {
  switch (shape_kind) {
    case ShapeKind::SameShape:
      return "SameShape";
    case ShapeKind::VaryingFirstDim:
      return "VaryingFirstDim";
  }
  return "";
}

template <typename T>
void copy_to_device(void *dst, const std::vector<T> &src) {
  if (!src.empty()) {
    NVTE_CHECK_CUDA(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
}

template <typename T>
std::vector<T> copy_to_host(const void *src, const size_t count) {
  std::vector<T> dst(count);
  if (count > 0) {
    NVTE_CHECK_CUDA(cudaMemcpy(dst.data(), src, count * sizeof(T), cudaMemcpyDeviceToHost));
  }
  return dst;
}

template <typename InputType, typename OutputType>
void compute_reference(const ScalingKind scaling_kind, const std::vector<InputType> &input,
                       const std::vector<int64_t> &first_dims, const size_t last_dim,
                       const std::vector<int64_t> &offsets, const bool force_pow_2_scales,
                       const float amax_epsilon, std::vector<OutputType> *output,
                       std::vector<float> *scale, std::vector<float> *amax,
                       std::vector<float> *scale_inv) {
  for (size_t tensor_id = 0; tensor_id < first_dims.size(); ++tensor_id) {
    const size_t tensor_start = static_cast<size_t>(offsets[tensor_id]);
    const size_t tensor_elems = static_cast<size_t>(first_dims[tensor_id]) * last_dim;

    float ref_amax = 0.0f;
    for (size_t i = 0; i < tensor_elems; ++i) {
      const float value = std::abs(to_float(input[tensor_start + i]));
      ref_amax = std::max(ref_amax, std::isnan(value) ? 0.0f : value);
    }
    (*amax)[tensor_id] = ref_amax;

    if (scaling_kind == ScalingKind::Current) {
      (*scale)[tensor_id] = compute_current_scale_ref(ref_amax, max_fp8_value<OutputType>(),
                                                      force_pow_2_scales, amax_epsilon);
    }
    (*scale_inv)[tensor_id] = 1.0f / (*scale)[tensor_id];

    for (size_t i = 0; i < tensor_elems; ++i) {
      const float value = to_float(input[tensor_start + i]) * (*scale)[tensor_id];
      (*output)[tensor_start + i] = static_cast<OutputType>(value);
    }
  }
}

template <typename InputType, typename OutputType>
void run_grouped_fp8_rowwise_test(const ScalingKind scaling_kind, const ShapeKind shape_kind) {
  const DType input_dtype = TypeInfo<InputType>::dtype;
  const DType output_dtype = TypeInfo<OutputType>::dtype;
  const std::vector<int64_t> first_dims = first_dims_for_shape(shape_kind);
  const size_t num_tensors = first_dims.size();
  const size_t last_dim = 7;
  const std::vector<int64_t> offsets = make_offsets(first_dims, last_dim);
  const size_t actual_total = static_cast<size_t>(offsets.back());
  const size_t allocation_total =
      actual_total + (shape_kind == ShapeKind::VaryingFirstDim ? 17 : 0);
  const size_t logical_rows = shape_kind == ShapeKind::SameShape
                                  ? static_cast<size_t>(first_dims[0]) * num_tensors
                                  : static_cast<size_t>(offsets.back()) / last_dim;
  const std::vector<size_t> logical_shape_vec{logical_rows, last_dim};
  NVTEShape logical_shape = nvte_make_shape(logical_shape_vec.data(), logical_shape_vec.size());

  std::vector<InputType> input_h =
      make_input<InputType>(first_dims, last_dim, offsets, allocation_total);
  const OutputType sentinel = static_cast<OutputType>(-13.0f);
  std::vector<OutputType> output_h(allocation_total, sentinel);
  std::vector<OutputType> output_ref(allocation_total, sentinel);

  std::vector<float> scale_h(num_tensors);
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    scale_h[tensor_id] = 0.25f * static_cast<float>(tensor_id + 2);
  }
  std::vector<float> scale_ref = scale_h;
  std::vector<float> amax_h(num_tensors, 1234.0f);
  std::vector<float> amax_ref(num_tensors, 0.0f);
  std::vector<float> scale_inv_h(num_tensors, -1.0f);
  std::vector<float> scale_inv_ref(num_tensors, 0.0f);

  constexpr bool force_pow_2_scales = false;
  constexpr float amax_epsilon = 0.0f;
  compute_reference<InputType, OutputType>(scaling_kind, input_h, first_dims, last_dim, offsets,
                                           force_pow_2_scales, amax_epsilon, &output_ref,
                                           &scale_ref, &amax_ref, &scale_inv_ref);

  auto input_d = cuda_alloc(allocation_total * sizeof(InputType));
  auto output_d = cuda_alloc(allocation_total * sizeof(OutputType));
  auto scale_d = cuda_alloc(num_tensors * sizeof(float));
  auto amax_d = cuda_alloc(num_tensors * sizeof(float));
  auto scale_inv_d = cuda_alloc(num_tensors * sizeof(float));
  copy_to_device(input_d.get(), input_h);
  copy_to_device(output_d.get(), output_h);
  copy_to_device(scale_d.get(), scale_h);
  copy_to_device(amax_d.get(), amax_h);
  copy_to_device(scale_inv_d.get(), scale_inv_h);

  GroupedHandle input_group(
      nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors, logical_shape));
  GroupedHandle output_group(
      nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors, logical_shape));

  NVTEShape data_shape = nvte_make_shape(&allocation_total, 1);
  NVTEBasicTensor input_data{input_d.get(), static_cast<NVTEDType>(input_dtype), data_shape};
  NVTEBasicTensor output_data{output_d.get(), static_cast<NVTEDType>(output_dtype), data_shape};
  nvte_set_grouped_tensor_param(input_group.get(), kNVTEGroupedRowwiseData, &input_data,
                                sizeof(input_data));
  nvte_set_grouped_tensor_param(output_group.get(), kNVTEGroupedRowwiseData, &output_data,
                                sizeof(output_data));

  size_t metadata_numel = num_tensors;
  NVTEShape metadata_shape = nvte_make_shape(&metadata_numel, 1);
  NVTEBasicTensor scale_tensor{scale_d.get(), kNVTEFloat32, metadata_shape};
  NVTEBasicTensor amax_tensor{amax_d.get(), kNVTEFloat32, metadata_shape};
  NVTEBasicTensor scale_inv_tensor{scale_inv_d.get(), kNVTEFloat32, metadata_shape};
  nvte_set_grouped_tensor_param(output_group.get(), kNVTEGroupedScale, &scale_tensor,
                                sizeof(scale_tensor));
  nvte_set_grouped_tensor_param(output_group.get(), kNVTEGroupedAmax, &amax_tensor,
                                sizeof(amax_tensor));
  nvte_set_grouped_tensor_param(output_group.get(), kNVTEGroupedRowwiseScaleInv,
                                &scale_inv_tensor, sizeof(scale_inv_tensor));

  CudaPtr<int64_t> first_dims_d;
  CudaPtr<int64_t> offsets_d;
  if (shape_kind == ShapeKind::VaryingFirstDim) {
    first_dims_d = cuda_alloc<int64_t>(num_tensors * sizeof(int64_t));
    offsets_d = cuda_alloc<int64_t>((num_tensors + 1) * sizeof(int64_t));
    copy_to_device(first_dims_d.get(), first_dims);
    copy_to_device(offsets_d.get(), offsets);

    NVTEShape first_dims_shape = nvte_make_shape(&metadata_numel, 1);
    size_t offsets_numel = num_tensors + 1;
    NVTEShape offsets_shape = nvte_make_shape(&offsets_numel, 1);
    NVTEBasicTensor first_dims_tensor{first_dims_d.get(), kNVTEInt64, first_dims_shape};
    NVTEBasicTensor offsets_tensor{offsets_d.get(), kNVTEInt64, offsets_shape};
    nvte_set_grouped_tensor_param(input_group.get(), kNVTEGroupedFirstDims, &first_dims_tensor,
                                  sizeof(first_dims_tensor));
    nvte_set_grouped_tensor_param(output_group.get(), kNVTEGroupedFirstDims, &first_dims_tensor,
                                  sizeof(first_dims_tensor));
    nvte_set_grouped_tensor_param(input_group.get(), kNVTEGroupedTensorOffsets, &offsets_tensor,
                                  sizeof(offsets_tensor));
    nvte_set_grouped_tensor_param(output_group.get(), kNVTEGroupedTensorOffsets, &offsets_tensor,
                                  sizeof(offsets_tensor));
  }

  QuantizationConfigWrapper quant_config;
  quant_config.set_compute_scale_from_amax(scaling_kind == ScalingKind::Current);
  quant_config.set_force_pow_2_scales(force_pow_2_scales);
  quant_config.set_amax_epsilon(amax_epsilon);
  nvte_group_quantize(input_group.get(), output_group.get(), quant_config, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  NVTE_CHECK_CUDA(cudaGetLastError());

  output_h = copy_to_host<OutputType>(output_d.get(), allocation_total);
  scale_h = copy_to_host<float>(scale_d.get(), num_tensors);
  amax_h = copy_to_host<float>(amax_d.get(), num_tensors);
  scale_inv_h = copy_to_host<float>(scale_inv_d.get(), num_tensors);

  for (size_t idx = 0; idx < actual_total; ++idx) {
    EXPECT_EQ(raw_byte(output_h[idx]), raw_byte(output_ref[idx]))
        << "Mismatch at logical output byte " << idx;
  }
  for (size_t idx = actual_total; idx < allocation_total; ++idx) {
    EXPECT_EQ(raw_byte(output_h[idx]), raw_byte(sentinel))
        << "Grouped FP8 quantize wrote output allocation slack at " << idx;
  }

  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    EXPECT_FLOAT_EQ(amax_h[tensor_id], amax_ref[tensor_id]) << "amax tensor " << tensor_id;
    EXPECT_FLOAT_EQ(scale_h[tensor_id], scale_ref[tensor_id]) << "scale tensor " << tensor_id;
    EXPECT_FLOAT_EQ(scale_inv_h[tensor_id], scale_inv_ref[tensor_id])
        << "scale_inv tensor " << tensor_id;
  }
}

template <typename InputType, typename OutputType>
void run_columnwise_fail_fast_test() {
  const DType input_dtype = TypeInfo<InputType>::dtype;
  const DType output_dtype = TypeInfo<OutputType>::dtype;
  const size_t num_tensors = 2;
  const std::vector<size_t> logical_shape_vec{4, 8};
  const size_t actual_total = logical_shape_vec[0] * logical_shape_vec[1];
  NVTEShape logical_shape = nvte_make_shape(logical_shape_vec.data(), logical_shape_vec.size());

  std::vector<InputType> input_h(actual_total, static_cast<InputType>(0.25f));
  const OutputType sentinel = static_cast<OutputType>(-7.0f);
  std::vector<OutputType> columnwise_h(actual_total, sentinel);

  auto input_d = cuda_alloc(actual_total * sizeof(InputType));
  auto columnwise_d = cuda_alloc(actual_total * sizeof(OutputType));
  copy_to_device(input_d.get(), input_h);
  copy_to_device(columnwise_d.get(), columnwise_h);

  GroupedHandle input_group(
      nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors, logical_shape));
  GroupedHandle output_group(
      nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors, logical_shape));

  NVTEShape data_shape = nvte_make_shape(&actual_total, 1);
  NVTEBasicTensor input_data{input_d.get(), static_cast<NVTEDType>(input_dtype), data_shape};
  NVTEBasicTensor columnwise_data{columnwise_d.get(), static_cast<NVTEDType>(output_dtype),
                                  data_shape};
  nvte_set_grouped_tensor_param(input_group.get(), kNVTEGroupedRowwiseData, &input_data,
                                sizeof(input_data));
  nvte_set_grouped_tensor_param(output_group.get(), kNVTEGroupedColumnwiseData, &columnwise_data,
                                sizeof(columnwise_data));

  QuantizationConfigWrapper quant_config;
  bool saw_expected_error = false;
  try {
    NVTE_CHECK_CUDA(cudaGetLastError());
    nvte_group_quantize(input_group.get(), output_group.get(), quant_config, 0);
  } catch (const std::runtime_error &err) {
    saw_expected_error = std::string(err.what()).find(kColumnwiseUnsupported) != std::string::npos;
  }
  EXPECT_TRUE(saw_expected_error);
  NVTE_CHECK_CUDA(cudaGetLastError());

  columnwise_h = copy_to_host<OutputType>(columnwise_d.get(), actual_total);
  for (size_t idx = 0; idx < actual_total; ++idx) {
    EXPECT_EQ(raw_byte(columnwise_h[idx]), raw_byte(sentinel))
        << "Columnwise fail-fast path wrote output data at " << idx;
  }
}

void run_mxfp8_preservation_smoke() {
  if (getDeviceComputeCapability() < blackwellComputeCapability) {
    GTEST_SKIP();
  }

  Tensor input_0("input_0", std::vector<size_t>{128, 128}, DType::kBFloat16);
  Tensor input_1("input_1", std::vector<size_t>{128, 128}, DType::kBFloat16);
  Tensor output_0("output_0", std::vector<size_t>{128, 128}, DType::kFloat8E4M3, true, false,
                  NVTE_MXFP8_1D_SCALING);
  Tensor output_1("output_1", std::vector<size_t>{128, 128}, DType::kFloat8E4M3, true, false,
                  NVTE_MXFP8_1D_SCALING);
  fillUniform(&input_0);
  fillUniform(&input_1);

  std::vector<Tensor *> inputs{&input_0, &input_1};
  std::vector<Tensor *> outputs{&output_0, &output_1};
  GroupedBuffers grouped_input = build_grouped_tensor(inputs, NVTE_DELAYED_TENSOR_SCALING);
  GroupedBuffers grouped_output = build_grouped_tensor(outputs, NVTE_MXFP8_1D_SCALING);

  QuantizationConfigWrapper quant_config;
  nvte_group_quantize(grouped_input.get_handle(), grouped_output.get_handle(), quant_config, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  NVTE_CHECK_CUDA(cudaGetLastError());
}

class GroupedFP8TensorScalingTestSuite
    : public ::testing::TestWithParam<
          std::tuple<ScalingKind, ShapeKind, DType, DType>> {};

TEST_P(GroupedFP8TensorScalingTestSuite, RowwiseQuantizeMatchesCPUReference) {
  const ScalingKind scaling_kind = std::get<0>(GetParam());
  const ShapeKind shape_kind = std::get<1>(GetParam());
  const DType input_type = std::get<2>(GetParam());
  const DType output_type = std::get<3>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(
          output_type, OutputType,
          run_grouped_fp8_rowwise_test<InputType, OutputType>(scaling_kind, shape_kind);
      );
  );
}

TEST(GroupedFP8TensorScalingTest, ColumnwiseOutputFailsBeforeLaunch) {
  run_columnwise_fail_fast_test<bf16, fp8e4m3>();
}

TEST(GroupedFP8TensorScalingTest, MXFP8GroupedQuantizeStillDispatches) {
  run_mxfp8_preservation_smoke();
}

std::string make_test_name(
    const testing::TestParamInfo<GroupedFP8TensorScalingTestSuite::ParamType> &info) {
  std::string name = to_string(std::get<0>(info.param));
  name += "_" + to_string(std::get<1>(info.param));
  name += "_" + test::typeName(std::get<2>(info.param));
  name += "_" + test::typeName(std::get<3>(info.param));
  return name;
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest_GroupedFP8TensorScaling, GroupedFP8TensorScalingTestSuite,
    ::testing::Combine(::testing::Values(ScalingKind::Delayed, ScalingKind::Current),
                       ::testing::Values(ShapeKind::SameShape, ShapeKind::VaryingFirstDim),
                       ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
                       ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2)),
    make_test_name);

}  // namespace
