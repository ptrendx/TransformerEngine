/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include <transformer_engine/cast.h>
#include <transformer_engine/transformer_engine.h>

#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

uint8_t fp8_byte(float value) {
  const fp8e4m3 fp8_value = static_cast<fp8e4m3>(value);
  return *reinterpret_cast<const uint8_t *>(&fp8_value);
}

}  // namespace

TEST(GroupedCastFP8, VaryingFirstDimStopsAtLogicalCapacity) {
  constexpr size_t num_tensors = 3;
  constexpr size_t cols = 5;
  constexpr uint8_t sentinel = 0x7b;

  const std::vector<int64_t> first_dims = {3, 2, 4};
  const std::vector<int64_t> offsets = {0, 19, 32, 60};
  const std::vector<float> scales = {0.5f, 1.0f, 2.0f};

  const size_t logical_rows = static_cast<size_t>(
      std::accumulate(first_dims.begin(), first_dims.end(), int64_t{0}));
  const size_t logical_elements = logical_rows * cols;
  const size_t allocation_elements = static_cast<size_t>(offsets.back());
  ASSERT_GT(allocation_elements, logical_elements);

  std::vector<bf16> input_h(allocation_elements, static_cast<bf16>(0.0f));
  std::vector<float> expected_amax(num_tensors, 0.0f);
  std::vector<bool> touched(allocation_elements, false);

  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    const size_t rows = static_cast<size_t>(first_dims[tensor_id]);
    const size_t base = static_cast<size_t>(offsets[tensor_id]);
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        const size_t local_idx = row * cols + col;
        const size_t physical_idx = base + local_idx;
        const float value = static_cast<float>(tensor_id + 1) * 0.25f +
                            static_cast<float>(row) * 0.125f -
                            static_cast<float>(col) * 0.03125f;
        input_h[physical_idx] = static_cast<bf16>(value);
        expected_amax[tensor_id] =
            std::max(expected_amax[tensor_id], std::abs(static_cast<float>(input_h[physical_idx])));
        touched[physical_idx] = true;
      }
    }
  }

  auto input_d = cuda_alloc<bf16>(allocation_elements * sizeof(bf16));
  auto rowwise_output_d = cuda_alloc<fp8e4m3>(allocation_elements * sizeof(fp8e4m3));
  auto columnwise_output_d = cuda_alloc<fp8e4m3>(allocation_elements * sizeof(fp8e4m3));
  auto first_dims_d = cuda_alloc<int64_t>(num_tensors * sizeof(int64_t));
  auto offsets_d = cuda_alloc<int64_t>((num_tensors + 1) * sizeof(int64_t));
  auto scales_d = cuda_alloc<float>(num_tensors * sizeof(float));
  auto amax_d = cuda_alloc<float>(num_tensors * sizeof(float));
  auto rowwise_scale_inv_d = cuda_alloc<float>(num_tensors * sizeof(float));
  auto columnwise_scale_inv_d = cuda_alloc<float>(num_tensors * sizeof(float));

  NVTE_CHECK_CUDA(cudaMemcpy(input_d.get(), input_h.data(), input_h.size() * sizeof(bf16),
                             cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemset(rowwise_output_d.get(), sentinel, allocation_elements));
  NVTE_CHECK_CUDA(cudaMemset(columnwise_output_d.get(), sentinel, allocation_elements));
  NVTE_CHECK_CUDA(cudaMemcpy(first_dims_d.get(), first_dims.data(),
                             first_dims.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemcpy(offsets_d.get(), offsets.data(), offsets.size() * sizeof(int64_t),
                             cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemcpy(scales_d.get(), scales.data(), scales.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemset(amax_d.get(), 0, num_tensors * sizeof(float)));
  NVTE_CHECK_CUDA(cudaMemset(rowwise_scale_inv_d.get(), 0, num_tensors * sizeof(float)));
  NVTE_CHECK_CUDA(cudaMemset(columnwise_scale_inv_d.get(), 0, num_tensors * sizeof(float)));

  const std::vector<size_t> logical_shape = {logical_rows, cols};
  const std::vector<size_t> flat_shape = {allocation_elements};
  const std::vector<size_t> group_shape = {num_tensors};
  const std::vector<size_t> offsets_shape = {num_tensors + 1};

  GroupedTensorWrapper input(num_tensors, logical_shape, NVTE_DELAYED_TENSOR_SCALING);
  input.set_rowwise_data(input_d.get(), DType::kBFloat16, flat_shape)
      .set_first_dims(first_dims_d.get(), DType::kInt64, group_shape)
      .set_tensor_offsets(offsets_d.get(), DType::kInt64, offsets_shape);

  GroupedTensorWrapper output(num_tensors, logical_shape, NVTE_DELAYED_TENSOR_SCALING);
  output.set_rowwise_data(rowwise_output_d.get(), DType::kFloat8E4M3, flat_shape)
      .set_columnwise_data(columnwise_output_d.get(), DType::kFloat8E4M3, flat_shape)
      .set_scale(scales_d.get(), DType::kFloat32, group_shape)
      .set_amax(amax_d.get(), DType::kFloat32, group_shape)
      .set_rowwise_scale_inv(rowwise_scale_inv_d.get(), DType::kFloat32, group_shape)
      .set_columnwise_scale_inv(columnwise_scale_inv_d.get(), DType::kFloat32, group_shape)
      .set_first_dims(first_dims_d.get(), DType::kInt64, group_shape)
      .set_tensor_offsets(offsets_d.get(), DType::kInt64, offsets_shape);

  nvte_group_quantize(input.data(), output.data(), nullptr, 0);
  NVTE_CHECK_CUDA(cudaGetLastError());
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<uint8_t> rowwise_output_h(allocation_elements);
  std::vector<uint8_t> columnwise_output_h(allocation_elements);
  std::vector<float> amax_h(num_tensors);
  std::vector<float> rowwise_scale_inv_h(num_tensors);
  std::vector<float> columnwise_scale_inv_h(num_tensors);

  NVTE_CHECK_CUDA(cudaMemcpy(rowwise_output_h.data(), rowwise_output_d.get(), allocation_elements,
                             cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(columnwise_output_h.data(), columnwise_output_d.get(),
                             allocation_elements, cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(amax_h.data(), amax_d.get(), num_tensors * sizeof(float),
                             cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(rowwise_scale_inv_h.data(), rowwise_scale_inv_d.get(),
                             num_tensors * sizeof(float), cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(columnwise_scale_inv_h.data(), columnwise_scale_inv_d.get(),
                             num_tensors * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    const size_t rows = static_cast<size_t>(first_dims[tensor_id]);
    const size_t base = static_cast<size_t>(offsets[tensor_id]);
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        const size_t local_idx = row * cols + col;
        const size_t rowwise_idx = base + local_idx;
        const size_t columnwise_idx = base + col * rows + row;
        const float input_value = static_cast<float>(input_h[rowwise_idx]);
        const uint8_t expected = fp8_byte(input_value * scales[tensor_id]);
        EXPECT_EQ(rowwise_output_h[rowwise_idx], expected);
        EXPECT_EQ(columnwise_output_h[columnwise_idx], expected);
      }
    }
    EXPECT_NEAR(amax_h[tensor_id], expected_amax[tensor_id], 0.0f);
    EXPECT_NEAR(rowwise_scale_inv_h[tensor_id], 1.0f / scales[tensor_id], 1e-6f);
    EXPECT_NEAR(columnwise_scale_inv_h[tensor_id], 1.0f / scales[tensor_id], 1e-6f);
  }

  for (size_t idx = 0; idx < allocation_elements; ++idx) {
    if (!touched[idx]) {
      EXPECT_EQ(rowwise_output_h[idx], sentinel) << "rowwise index " << idx;
      EXPECT_EQ(columnwise_output_h[idx], sentinel) << "columnwise index " << idx;
    }
  }
}
