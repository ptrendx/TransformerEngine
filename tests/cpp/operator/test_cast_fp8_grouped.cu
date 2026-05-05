/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <transformer_engine/cast.h>
#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

struct GroupedTensorDeleterLocal {
  void operator()(NVTEGroupedTensor tensor) const {
    if (tensor != nullptr) {
      nvte_destroy_grouped_tensor(tensor);
    }
  }
};

using GroupedTensorPtr =
    std::unique_ptr<std::remove_pointer_t<NVTEGroupedTensor>, GroupedTensorDeleterLocal>;

constexpr size_t kNumTensors = 2;
constexpr size_t kCols = 4;
constexpr size_t kAllocationElements = 28;
constexpr uint8_t kSentinel = 0xA5;

struct PaddedGroupedCase {
  std::vector<size_t> logical_shape{5, kCols};
  std::vector<int64_t> first_dims{2, 3};
  std::vector<int64_t> offsets{0, 12, 24};
  std::vector<float> input;
  std::vector<size_t> valid_indices;

  PaddedGroupedCase() : input(kAllocationElements, -999.0f) {
    float value = -3.5f;
    for (size_t tensor_id = 0; tensor_id < kNumTensors; ++tensor_id) {
      const size_t rows = static_cast<size_t>(first_dims[tensor_id]);
      const size_t base = static_cast<size_t>(offsets[tensor_id]);
      for (size_t local = 0; local < rows * kCols; ++local) {
        input[base + local] = value;
        valid_indices.push_back(base + local);
        value += 0.5f;
      }
    }
  }

  size_t logical_total() const { return logical_shape[0] * logical_shape[1]; }
};

template <typename T>
void CopyToDevice(T *dst, const std::vector<T> &src) {
  NVTE_CHECK_CUDA(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

void SetGroupedTensorParam(NVTEGroupedTensor tensor, NVTEGroupedTensorParam param,
                           NVTEBasicTensor *basic_tensor) {
  nvte_set_grouped_tensor_param(tensor, param, basic_tensor, sizeof(NVTEBasicTensor));
}

GroupedTensorPtr MakeGroupedInput(const PaddedGroupedCase &test_case, float *input_d,
                                  int64_t *first_dims_d, int64_t *offsets_d) {
  NVTEShape logical_shape = nvte_make_shape(test_case.logical_shape.data(), 2);
  GroupedTensorPtr input(nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, kNumTensors,
                                                    logical_shape));

  size_t allocation_elements = kAllocationElements;
  NVTEShape allocation_shape = nvte_make_shape(&allocation_elements, 1);
  NVTEBasicTensor data_tensor{input_d, kNVTEFloat32, allocation_shape};
  SetGroupedTensorParam(input.get(), kNVTEGroupedRowwiseData, &data_tensor);

  size_t num_tensors = kNumTensors;
  NVTEShape first_dims_shape = nvte_make_shape(&num_tensors, 1);
  NVTEBasicTensor first_dims_tensor{first_dims_d, kNVTEInt64, first_dims_shape};
  SetGroupedTensorParam(input.get(), kNVTEGroupedFirstDims, &first_dims_tensor);

  size_t offsets_elements = kNumTensors + 1;
  NVTEShape offsets_shape = nvte_make_shape(&offsets_elements, 1);
  NVTEBasicTensor offsets_tensor{offsets_d, kNVTEInt64, offsets_shape};
  SetGroupedTensorParam(input.get(), kNVTEGroupedTensorOffsets, &offsets_tensor);

  return input;
}

GroupedTensorPtr MakeGroupedOutput(const PaddedGroupedCase &test_case, uint8_t *output_d,
                                   float *scale_inv_d, float *amax_d, float *scale_d,
                                   int64_t *first_dims_d, int64_t *offsets_d,
                                   size_t scale_elements) {
  NVTEShape logical_shape = nvte_make_shape(test_case.logical_shape.data(), 2);
  GroupedTensorPtr output(nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, kNumTensors,
                                                     logical_shape));

  size_t allocation_elements = kAllocationElements;
  NVTEShape allocation_shape = nvte_make_shape(&allocation_elements, 1);
  NVTEBasicTensor data_tensor{output_d, kNVTEFloat8E4M3, allocation_shape};
  SetGroupedTensorParam(output.get(), kNVTEGroupedRowwiseData, &data_tensor);

  size_t num_tensors = kNumTensors;
  NVTEShape per_tensor_shape = nvte_make_shape(&num_tensors, 1);
  NVTEBasicTensor scale_inv_tensor{scale_inv_d, kNVTEFloat32, per_tensor_shape};
  SetGroupedTensorParam(output.get(), kNVTEGroupedRowwiseScaleInv, &scale_inv_tensor);
  NVTEBasicTensor amax_tensor{amax_d, kNVTEFloat32, per_tensor_shape};
  SetGroupedTensorParam(output.get(), kNVTEGroupedAmax, &amax_tensor);

  NVTEShape scale_shape = nvte_make_shape(&scale_elements, 1);
  NVTEBasicTensor scale_tensor{scale_d, kNVTEFloat32, scale_shape};
  SetGroupedTensorParam(output.get(), kNVTEGroupedScale, &scale_tensor);

  NVTEBasicTensor first_dims_tensor{first_dims_d, kNVTEInt64, per_tensor_shape};
  SetGroupedTensorParam(output.get(), kNVTEGroupedFirstDims, &first_dims_tensor);

  size_t offsets_elements = kNumTensors + 1;
  NVTEShape offsets_shape = nvte_make_shape(&offsets_elements, 1);
  NVTEBasicTensor offsets_tensor{offsets_d, kNVTEInt64, offsets_shape};
  SetGroupedTensorParam(output.get(), kNVTEGroupedTensorOffsets, &offsets_tensor);

  return output;
}

std::vector<float> PerTensorAmax(const PaddedGroupedCase &test_case) {
  std::vector<float> amax(kNumTensors, 0.0f);
  for (size_t tensor_id = 0; tensor_id < kNumTensors; ++tensor_id) {
    const size_t rows = static_cast<size_t>(test_case.first_dims[tensor_id]);
    const size_t base = static_cast<size_t>(test_case.offsets[tensor_id]);
    for (size_t local = 0; local < rows * kCols; ++local) {
      amax[tensor_id] = std::max(amax[tensor_id], std::fabs(test_case.input[base + local]));
    }
  }
  return amax;
}

std::vector<fp8e4m3> BuildReferenceOutput(const PaddedGroupedCase &test_case,
                                          const std::vector<float> &scale) {
  std::vector<fp8e4m3> ref(kAllocationElements);
  std::fill(reinterpret_cast<uint8_t *>(ref.data()),
            reinterpret_cast<uint8_t *>(ref.data()) + ref.size() * sizeof(fp8e4m3), kSentinel);

  for (size_t tensor_id = 0; tensor_id < kNumTensors; ++tensor_id) {
    const size_t rows = static_cast<size_t>(test_case.first_dims[tensor_id]);
    const size_t base = static_cast<size_t>(test_case.offsets[tensor_id]);
    for (size_t local = 0; local < rows * kCols; ++local) {
      ref[base + local] = static_cast<fp8e4m3>(test_case.input[base + local] * scale[tensor_id]);
    }
  }
  return ref;
}

void ExpectValidOutputAndSentinels(const PaddedGroupedCase &test_case,
                                   const std::vector<fp8e4m3> &output,
                                   const std::vector<fp8e4m3> &reference) {
  std::vector<uint8_t> is_valid(kAllocationElements, 0);
  for (size_t idx : test_case.valid_indices) {
    is_valid[idx] = 1;
    EXPECT_EQ(reinterpret_cast<const uint8_t *>(&output[idx])[0],
              reinterpret_cast<const uint8_t *>(&reference[idx])[0])
        << "FP8 output mismatch at physical element " << idx;
  }
  for (size_t idx = 0; idx < kAllocationElements; ++idx) {
    if (!is_valid[idx]) {
      EXPECT_EQ(reinterpret_cast<const uint8_t *>(&output[idx])[0], kSentinel)
          << "Padding/trailing element was modified at physical element " << idx;
    }
  }
}

TEST(OperatorTest, GroupedFP8DelayedTensorScalingStopsAtLogicalTotal) {
  PaddedGroupedCase test_case;
  auto input_d = cuda_alloc<float>(kAllocationElements * sizeof(float));
  auto output_d = cuda_alloc<uint8_t>(kAllocationElements * sizeof(uint8_t));
  auto scale_inv_d = cuda_alloc<float>(kNumTensors * sizeof(float));
  auto amax_d = cuda_alloc<float>(kNumTensors * sizeof(float));
  auto scale_d = cuda_alloc<float>(sizeof(float));
  auto first_dims_d = cuda_alloc<int64_t>(kNumTensors * sizeof(int64_t));
  auto offsets_d = cuda_alloc<int64_t>((kNumTensors + 1) * sizeof(int64_t));

  CopyToDevice(input_d.get(), test_case.input);
  CopyToDevice(first_dims_d.get(), test_case.first_dims);
  CopyToDevice(offsets_d.get(), test_case.offsets);
  NVTE_CHECK_CUDA(cudaMemset(output_d.get(), kSentinel, kAllocationElements * sizeof(uint8_t)));
  NVTE_CHECK_CUDA(cudaMemset(amax_d.get(), 0, kNumTensors * sizeof(float)));
  const std::vector<float> scale_h{2.0f};
  CopyToDevice(scale_d.get(), scale_h);

  auto input = MakeGroupedInput(test_case, input_d.get(), first_dims_d.get(), offsets_d.get());
  auto output = MakeGroupedOutput(test_case, output_d.get(), scale_inv_d.get(), amax_d.get(),
                                  scale_d.get(), first_dims_d.get(), offsets_d.get(), 1);

  QuantizationConfigWrapper config;
  nvte_group_quantize(input.get(), output.get(), config, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<fp8e4m3> output_h(kAllocationElements);
  NVTE_CHECK_CUDA(cudaMemcpy(output_h.data(), output_d.get(),
                             kAllocationElements * sizeof(fp8e4m3), cudaMemcpyDeviceToHost));
  const std::vector<fp8e4m3> ref =
      BuildReferenceOutput(test_case, std::vector<float>{scale_h[0], scale_h[0]});
  ExpectValidOutputAndSentinels(test_case, output_h, ref);

  std::vector<float> amax_h(kNumTensors);
  std::vector<float> scale_inv_h(kNumTensors);
  NVTE_CHECK_CUDA(cudaMemcpy(amax_h.data(), amax_d.get(), kNumTensors * sizeof(float),
                             cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(scale_inv_h.data(), scale_inv_d.get(), kNumTensors * sizeof(float),
                             cudaMemcpyDeviceToHost));
  const auto ref_amax = PerTensorAmax(test_case);
  for (size_t i = 0; i < kNumTensors; ++i) {
    EXPECT_FLOAT_EQ(amax_h[i], ref_amax[i]);
    EXPECT_FLOAT_EQ(scale_inv_h[i], 0.5f);
  }
}

TEST(OperatorTest, GroupedFP8CurrentTensorScalingComputesPerTensorScale) {
  PaddedGroupedCase test_case;
  auto input_d = cuda_alloc<float>(kAllocationElements * sizeof(float));
  auto output_d = cuda_alloc<uint8_t>(kAllocationElements * sizeof(uint8_t));
  auto scale_inv_d = cuda_alloc<float>(kNumTensors * sizeof(float));
  auto amax_d = cuda_alloc<float>(kNumTensors * sizeof(float));
  auto scale_d = cuda_alloc<float>(kNumTensors * sizeof(float));
  auto first_dims_d = cuda_alloc<int64_t>(kNumTensors * sizeof(int64_t));
  auto offsets_d = cuda_alloc<int64_t>((kNumTensors + 1) * sizeof(int64_t));

  CopyToDevice(input_d.get(), test_case.input);
  CopyToDevice(first_dims_d.get(), test_case.first_dims);
  CopyToDevice(offsets_d.get(), test_case.offsets);
  NVTE_CHECK_CUDA(cudaMemset(output_d.get(), kSentinel, kAllocationElements * sizeof(uint8_t)));

  auto input = MakeGroupedInput(test_case, input_d.get(), first_dims_d.get(), offsets_d.get());
  auto output = MakeGroupedOutput(test_case, output_d.get(), scale_inv_d.get(), amax_d.get(),
                                  scale_d.get(), first_dims_d.get(), offsets_d.get(),
                                  kNumTensors);

  QuantizationConfigWrapper config;
  nvte_group_quantize(input.get(), output.get(), config, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  const auto ref_amax = PerTensorAmax(test_case);
  std::vector<float> scale_h(kNumTensors);
  std::vector<float> scale_inv_h(kNumTensors);
  std::vector<float> amax_h(kNumTensors);
  NVTE_CHECK_CUDA(cudaMemcpy(scale_h.data(), scale_d.get(), kNumTensors * sizeof(float),
                             cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(scale_inv_h.data(), scale_inv_d.get(), kNumTensors * sizeof(float),
                             cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(amax_h.data(), amax_d.get(), kNumTensors * sizeof(float),
                             cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < kNumTensors; ++i) {
    const float ref_scale = test::Quantized_Limits<fp8e4m3>::max() / ref_amax[i];
    EXPECT_FLOAT_EQ(amax_h[i], ref_amax[i]);
    EXPECT_NEAR(scale_h[i], ref_scale, ref_scale * 1e-6f);
    EXPECT_NEAR(scale_inv_h[i], 1.0f / ref_scale, (1.0f / ref_scale) * 1e-6f);
  }

  std::vector<fp8e4m3> output_h(kAllocationElements);
  NVTE_CHECK_CUDA(cudaMemcpy(output_h.data(), output_d.get(),
                             kAllocationElements * sizeof(fp8e4m3), cudaMemcpyDeviceToHost));
  const std::vector<fp8e4m3> ref = BuildReferenceOutput(test_case, scale_h);
  ExpectValidOutputAndSentinels(test_case, output_h, ref);
}

}  // namespace
