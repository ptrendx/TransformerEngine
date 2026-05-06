/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include <transformer_engine/cast.h>
#include <transformer_engine/transformer_engine.h>

#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

template <typename T>
void cuda_copy_to_device(T *dst, const std::vector<T> &src) {
  NVTE_CHECK_CUDA(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void cuda_copy_to_host(std::vector<T> *dst, const T *src) {
  NVTE_CHECK_CUDA(cudaMemcpy(dst->data(), src, dst->size() * sizeof(T), cudaMemcpyDeviceToHost));
}

void set_grouped_tensor_param(NVTEGroupedTensor tensor, NVTEGroupedTensorParam param, void *dptr,
                              NVTEDType dtype, const std::vector<size_t> &shape) {
  NVTEShape nvte_shape = nvte_make_shape(shape.data(), shape.size());
  NVTEBasicTensor basic_tensor{dptr, dtype, nvte_shape};
  nvte_set_grouped_tensor_param(tensor, param, &basic_tensor, sizeof(basic_tensor));
}

uint8_t fp8_byte(fp8e4m3 value) {
  uint8_t byte = 0;
  std::memcpy(&byte, &value, sizeof(byte));
  return byte;
}

std::vector<float> per_tensor_amax(const std::vector<float> &input,
                                   const std::vector<int64_t> &offsets,
                                   const std::vector<int64_t> &first_dims, int64_t cols) {
  std::vector<float> expected(first_dims.size(), 0.0f);
  for (size_t tensor_id = 0; tensor_id < first_dims.size(); ++tensor_id) {
    const int64_t start = offsets[tensor_id];
    const int64_t elems = first_dims[tensor_id] * cols;
    for (int64_t i = 0; i < elems; ++i) {
      expected[tensor_id] = std::max(expected[tensor_id], std::abs(input[start + i]));
    }
  }
  return expected;
}

TEST(GroupedFP8QuantizeTest, DelayedScalingClearsStaleAmaxAndStopsAtLogicalTotal) {
  constexpr size_t num_tensors = 3;
  constexpr int64_t cols = 17;
  const std::vector<int64_t> first_dims = {2, 3, 1};
  const int64_t actual_rows = std::accumulate(first_dims.begin(), first_dims.end(), int64_t{0});
  const size_t actual_elements = static_cast<size_t>(actual_rows * cols);
  const size_t allocation_elements = actual_elements + 31;
  const std::vector<int64_t> offsets = {0, first_dims[0] * cols,
                                        (first_dims[0] + first_dims[1]) * cols,
                                        static_cast<int64_t>(actual_elements)};
  const std::vector<size_t> logical_shape = {static_cast<size_t>(actual_rows),
                                             static_cast<size_t>(cols)};

  std::vector<float> input_h(allocation_elements);
  for (size_t i = 0; i < actual_elements; ++i) {
    input_h[i] = (static_cast<int>(i % 19) - 9) * 0.125f;
  }
  for (size_t i = actual_elements; i < allocation_elements; ++i) {
    input_h[i] = 4096.0f;
  }

  const std::vector<float> scale_h = {1.0f, 0.5f, 2.0f};
  std::vector<float> amax_h(num_tensors, 8192.0f);
  std::vector<float> scale_inv_h(num_tensors, 0.0f);
  const std::vector<uint8_t> sentinel_h(allocation_elements, 0x5a);

  float *input_d = nullptr;
  uint8_t *output_d = nullptr;
  float *scale_d = nullptr;
  float *amax_d = nullptr;
  float *scale_inv_d = nullptr;
  int64_t *first_dims_d = nullptr;
  int64_t *offsets_d = nullptr;
  NVTE_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&input_d),
                             allocation_elements * sizeof(float)));
  NVTE_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&output_d), allocation_elements));
  NVTE_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&scale_d), num_tensors * sizeof(float)));
  NVTE_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&amax_d), num_tensors * sizeof(float)));
  NVTE_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&scale_inv_d),
                             num_tensors * sizeof(float)));
  NVTE_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&first_dims_d),
                             num_tensors * sizeof(int64_t)));
  NVTE_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&offsets_d),
                             (num_tensors + 1) * sizeof(int64_t)));

  cuda_copy_to_device(input_d, input_h);
  cuda_copy_to_device(scale_d, scale_h);
  cuda_copy_to_device(amax_d, amax_h);
  cuda_copy_to_device(scale_inv_d, scale_inv_h);
  cuda_copy_to_device(first_dims_d, first_dims);
  cuda_copy_to_device(offsets_d, offsets);
  cuda_copy_to_device(output_d, sentinel_h);

  NVTEGroupedTensor input =
      nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors,
                                 nvte_make_shape(logical_shape.data(), logical_shape.size()));
  NVTEGroupedTensor output =
      nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors,
                                 nvte_make_shape(logical_shape.data(), logical_shape.size()));

  set_grouped_tensor_param(input, kNVTEGroupedRowwiseData, input_d, kNVTEFloat32,
                           {allocation_elements});
  set_grouped_tensor_param(input, kNVTEGroupedFirstDims, first_dims_d, kNVTEInt64, {num_tensors});
  set_grouped_tensor_param(input, kNVTEGroupedTensorOffsets, offsets_d, kNVTEInt64,
                           {num_tensors + 1});

  set_grouped_tensor_param(output, kNVTEGroupedRowwiseData, output_d, kNVTEFloat8E4M3,
                           {allocation_elements});
  set_grouped_tensor_param(output, kNVTEGroupedScale, scale_d, kNVTEFloat32, {num_tensors});
  set_grouped_tensor_param(output, kNVTEGroupedAmax, amax_d, kNVTEFloat32, {num_tensors});
  set_grouped_tensor_param(output, kNVTEGroupedRowwiseScaleInv, scale_inv_d, kNVTEFloat32,
                           {num_tensors});
  set_grouped_tensor_param(output, kNVTEGroupedFirstDims, first_dims_d, kNVTEInt64, {num_tensors});
  set_grouped_tensor_param(output, kNVTEGroupedTensorOffsets, offsets_d, kNVTEInt64,
                           {num_tensors + 1});

  QuantizationConfigWrapper quant_config;
  nvte_group_quantize(input, output, quant_config, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<uint8_t> output_h(allocation_elements);
  cuda_copy_to_host(&output_h, output_d);
  cuda_copy_to_host(&amax_h, amax_d);
  cuda_copy_to_host(&scale_inv_h, scale_inv_d);

  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    const size_t start = static_cast<size_t>(offsets[tensor_id]);
    const size_t elems = static_cast<size_t>(first_dims[tensor_id] * cols);
    for (size_t i = 0; i < elems; ++i) {
      const fp8e4m3 expected = static_cast<fp8e4m3>(input_h[start + i] * scale_h[tensor_id]);
      EXPECT_EQ(output_h[start + i], fp8_byte(expected));
    }
  }
  for (size_t i = actual_elements; i < allocation_elements; ++i) {
    EXPECT_EQ(output_h[i], 0x5a);
  }

  const std::vector<float> expected_amax = per_tensor_amax(input_h, offsets, first_dims, cols);
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    EXPECT_FLOAT_EQ(amax_h[tensor_id], expected_amax[tensor_id]);
    EXPECT_FLOAT_EQ(scale_inv_h[tensor_id], 1.0f / scale_h[tensor_id]);
  }

  nvte_destroy_grouped_tensor(input);
  nvte_destroy_grouped_tensor(output);
  cudaFree(input_d);
  cudaFree(output_d);
  cudaFree(scale_d);
  cudaFree(amax_d);
  cudaFree(scale_inv_d);
  cudaFree(first_dims_d);
  cudaFree(offsets_d);
}

TEST(GroupedFP8QuantizeTest, CurrentScalingComputesPerTensorScale) {
  constexpr size_t num_tensors = 2;
  constexpr int64_t cols = 13;
  const std::vector<int64_t> first_dims = {2, 1};
  const size_t actual_elements = static_cast<size_t>((first_dims[0] + first_dims[1]) * cols);
  const std::vector<int64_t> offsets = {0, first_dims[0] * cols,
                                        static_cast<int64_t>(actual_elements)};
  const std::vector<size_t> logical_shape = {static_cast<size_t>(first_dims[0] + first_dims[1]),
                                             static_cast<size_t>(cols)};

  std::vector<float> input_h(actual_elements);
  for (size_t i = 0; i < actual_elements; ++i) {
    input_h[i] = (static_cast<int>(i % 11) - 5) * 0.25f;
  }

  std::vector<float> scale_h(num_tensors, -1.0f);
  std::vector<float> amax_h(num_tensors, -1.0f);
  std::vector<float> scale_inv_h(num_tensors, -1.0f);

  float *input_d = nullptr;
  uint8_t *output_d = nullptr;
  float *scale_d = nullptr;
  float *amax_d = nullptr;
  float *scale_inv_d = nullptr;
  int64_t *first_dims_d = nullptr;
  int64_t *offsets_d = nullptr;
  NVTE_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&input_d), actual_elements * sizeof(float)));
  NVTE_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&output_d), actual_elements));
  NVTE_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&scale_d), num_tensors * sizeof(float)));
  NVTE_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&amax_d), num_tensors * sizeof(float)));
  NVTE_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&scale_inv_d),
                             num_tensors * sizeof(float)));
  NVTE_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&first_dims_d),
                             num_tensors * sizeof(int64_t)));
  NVTE_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&offsets_d),
                             (num_tensors + 1) * sizeof(int64_t)));

  cuda_copy_to_device(input_d, input_h);
  cuda_copy_to_device(scale_d, scale_h);
  cuda_copy_to_device(amax_d, amax_h);
  cuda_copy_to_device(scale_inv_d, scale_inv_h);
  cuda_copy_to_device(first_dims_d, first_dims);
  cuda_copy_to_device(offsets_d, offsets);

  NVTEGroupedTensor input =
      nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors,
                                 nvte_make_shape(logical_shape.data(), logical_shape.size()));
  NVTEGroupedTensor output =
      nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors,
                                 nvte_make_shape(logical_shape.data(), logical_shape.size()));

  set_grouped_tensor_param(input, kNVTEGroupedRowwiseData, input_d, kNVTEFloat32,
                           {actual_elements});
  set_grouped_tensor_param(input, kNVTEGroupedFirstDims, first_dims_d, kNVTEInt64, {num_tensors});
  set_grouped_tensor_param(input, kNVTEGroupedTensorOffsets, offsets_d, kNVTEInt64,
                           {num_tensors + 1});

  set_grouped_tensor_param(output, kNVTEGroupedRowwiseData, output_d, kNVTEFloat8E4M3,
                           {actual_elements});
  set_grouped_tensor_param(output, kNVTEGroupedScale, scale_d, kNVTEFloat32, {num_tensors});
  set_grouped_tensor_param(output, kNVTEGroupedAmax, amax_d, kNVTEFloat32, {num_tensors});
  set_grouped_tensor_param(output, kNVTEGroupedRowwiseScaleInv, scale_inv_d, kNVTEFloat32,
                           {num_tensors});
  set_grouped_tensor_param(output, kNVTEGroupedFirstDims, first_dims_d, kNVTEInt64, {num_tensors});
  set_grouped_tensor_param(output, kNVTEGroupedTensorOffsets, offsets_d, kNVTEInt64,
                           {num_tensors + 1});

  QuantizationConfigWrapper quant_config;
  quant_config.set_compute_scale_from_amax(true);
  nvte_group_quantize(input, output, quant_config, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<uint8_t> output_h(actual_elements);
  cuda_copy_to_host(&output_h, output_d);
  cuda_copy_to_host(&amax_h, amax_d);
  cuda_copy_to_host(&scale_h, scale_d);
  cuda_copy_to_host(&scale_inv_h, scale_inv_d);

  const std::vector<float> expected_amax = per_tensor_amax(input_h, offsets, first_dims, cols);
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    const float expected_scale =
        expected_amax[tensor_id] == 0.0f ? 1.0f : test::Quantized_Limits<fp8e4m3>::max() /
                                                        expected_amax[tensor_id];
    EXPECT_FLOAT_EQ(amax_h[tensor_id], expected_amax[tensor_id]);
    EXPECT_FLOAT_EQ(scale_h[tensor_id], expected_scale);
    EXPECT_FLOAT_EQ(scale_inv_h[tensor_id], 1.0f / expected_scale);

    const size_t start = static_cast<size_t>(offsets[tensor_id]);
    const size_t elems = static_cast<size_t>(first_dims[tensor_id] * cols);
    for (size_t i = 0; i < elems; ++i) {
      const fp8e4m3 expected = static_cast<fp8e4m3>(input_h[start + i] * expected_scale);
      EXPECT_EQ(output_h[start + i], fp8_byte(expected));
    }
  }

  nvte_destroy_grouped_tensor(input);
  nvte_destroy_grouped_tensor(output);
  cudaFree(input_d);
  cudaFree(output_d);
  cudaFree(scale_d);
  cudaFree(amax_d);
  cudaFree(scale_inv_d);
  cudaFree(first_dims_d);
  cudaFree(offsets_d);
}

}  // namespace
