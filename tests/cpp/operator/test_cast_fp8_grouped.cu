/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include <transformer_engine/cast.h>
#include <transformer_engine/transformer_engine.h>

#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

template <typename T>
uint8_t to_byte(const T &value) {
  uint8_t byte = 0;
  std::memcpy(&byte, &value, sizeof(byte));
  return byte;
}

void set_basic_tensor(NVTEGroupedTensor tensor, NVTEGroupedTensorParam param, void *ptr,
                      NVTEDType dtype, const std::vector<size_t> &shape) {
  NVTEShape nvte_shape = nvte_make_shape(shape.data(), shape.size());
  NVTEBasicTensor basic_tensor{ptr, dtype, nvte_shape};
  nvte_set_grouped_tensor_param(tensor, param, &basic_tensor, sizeof(basic_tensor));
}

template <typename InputType, typename OutputType>
void run_grouped_fp8_delayed_capacity_test() {
  const DType itype = TypeInfo<InputType>::dtype;
  const DType otype = TypeInfo<OutputType>::dtype;

  const size_t num_tensors = 3;
  const std::vector<int64_t> first_dims = {2, 1, 4};
  const std::vector<int64_t> last_dims = {3, 5, 2};
  const std::vector<int64_t> offsets = {0, 6, 11, 19};
  const size_t logical_elements = static_cast<size_t>(offsets.back());
  const size_t allocation_elements = logical_elements + 7;
  const std::vector<size_t> logical_shape = {1, logical_elements};
  const uint8_t sentinel = 0xA5;

  std::vector<InputType> input_host(allocation_elements);
  for (size_t i = 0; i < logical_elements; ++i) {
    const float value = (static_cast<int>(i % 11) - 5) * 0.25f;
    input_host[i] = static_cast<InputType>(value);
  }
  for (size_t i = logical_elements; i < allocation_elements; ++i) {
    input_host[i] = static_cast<InputType>(1024.0f + static_cast<float>(i));
  }

  const std::vector<float> scales = {0.5f, 1.25f, 2.0f};

  auto input_dev = cuda_alloc<InputType>(allocation_elements * sizeof(InputType));
  auto output_rowwise_dev = cuda_alloc<uint8_t>(allocation_elements * sizeof(uint8_t));
  auto output_columnwise_dev = cuda_alloc<uint8_t>(allocation_elements * sizeof(uint8_t));
  auto first_dims_dev = cuda_alloc<int64_t>(first_dims.size() * sizeof(int64_t));
  auto last_dims_dev = cuda_alloc<int64_t>(last_dims.size() * sizeof(int64_t));
  auto offsets_dev = cuda_alloc<int64_t>(offsets.size() * sizeof(int64_t));
  auto scale_dev = cuda_alloc<float>(scales.size() * sizeof(float));
  auto amax_dev = cuda_alloc<float>(num_tensors * sizeof(float));
  auto scale_inv_dev = cuda_alloc<float>(num_tensors * sizeof(float));
  auto columnwise_scale_inv_dev = cuda_alloc<float>(num_tensors * sizeof(float));

  NVTE_CHECK_CUDA(cudaMemcpy(input_dev.get(), input_host.data(),
                             allocation_elements * sizeof(InputType), cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemset(output_rowwise_dev.get(), sentinel, allocation_elements));
  NVTE_CHECK_CUDA(cudaMemset(output_columnwise_dev.get(), sentinel, allocation_elements));
  NVTE_CHECK_CUDA(cudaMemcpy(first_dims_dev.get(), first_dims.data(),
                             first_dims.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemcpy(last_dims_dev.get(), last_dims.data(),
                             last_dims.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemcpy(offsets_dev.get(), offsets.data(), offsets.size() * sizeof(int64_t),
                             cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemcpy(scale_dev.get(), scales.data(), scales.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemset(amax_dev.get(), 0, num_tensors * sizeof(float)));
  NVTE_CHECK_CUDA(cudaMemset(scale_inv_dev.get(), 0, num_tensors * sizeof(float)));
  NVTE_CHECK_CUDA(cudaMemset(columnwise_scale_inv_dev.get(), 0, num_tensors * sizeof(float)));

  GroupedTensorHandle input(nvte_create_grouped_tensor(
      NVTE_DELAYED_TENSOR_SCALING, num_tensors,
      nvte_make_shape(logical_shape.data(), logical_shape.size())));
  GroupedTensorHandle output(nvte_create_grouped_tensor(
      NVTE_DELAYED_TENSOR_SCALING, num_tensors,
      nvte_make_shape(logical_shape.data(), logical_shape.size())));

  set_basic_tensor(input.get(), kNVTEGroupedRowwiseData, input_dev.get(),
                   static_cast<NVTEDType>(itype), {allocation_elements});

  set_basic_tensor(output.get(), kNVTEGroupedRowwiseData, output_rowwise_dev.get(),
                   static_cast<NVTEDType>(otype), {allocation_elements});
  set_basic_tensor(output.get(), kNVTEGroupedColumnwiseData, output_columnwise_dev.get(),
                   static_cast<NVTEDType>(otype), {allocation_elements});
  set_basic_tensor(output.get(), kNVTEGroupedScale, scale_dev.get(), kNVTEFloat32, {num_tensors});
  set_basic_tensor(output.get(), kNVTEGroupedAmax, amax_dev.get(), kNVTEFloat32, {num_tensors});
  set_basic_tensor(output.get(), kNVTEGroupedRowwiseScaleInv, scale_inv_dev.get(), kNVTEFloat32,
                   {num_tensors});
  set_basic_tensor(output.get(), kNVTEGroupedColumnwiseScaleInv, columnwise_scale_inv_dev.get(),
                   kNVTEFloat32, {num_tensors});
  set_basic_tensor(output.get(), kNVTEGroupedFirstDims, first_dims_dev.get(), kNVTEInt64,
                   {num_tensors});
  set_basic_tensor(output.get(), kNVTEGroupedLastDims, last_dims_dev.get(), kNVTEInt64,
                   {num_tensors});
  set_basic_tensor(output.get(), kNVTEGroupedTensorOffsets, offsets_dev.get(), kNVTEInt64,
                   {num_tensors + 1});

  nvte_group_quantize(input.get(), output.get(), nullptr, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<uint8_t> output_rowwise(allocation_elements);
  std::vector<uint8_t> output_columnwise(allocation_elements);
  std::vector<float> amax(num_tensors);
  std::vector<float> scale_inv(num_tensors);
  std::vector<float> columnwise_scale_inv(num_tensors);
  NVTE_CHECK_CUDA(cudaMemcpy(output_rowwise.data(), output_rowwise_dev.get(), allocation_elements,
                             cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(output_columnwise.data(), output_columnwise_dev.get(),
                             allocation_elements, cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(amax.data(), amax_dev.get(), num_tensors * sizeof(float),
                             cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(scale_inv.data(), scale_inv_dev.get(), num_tensors * sizeof(float),
                             cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(columnwise_scale_inv.data(), columnwise_scale_inv_dev.get(),
                             num_tensors * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<uint8_t> rowwise_ref(allocation_elements, sentinel);
  std::vector<uint8_t> columnwise_ref(allocation_elements, sentinel);
  std::vector<float> amax_ref(num_tensors, 0.0f);

  for (size_t t = 0; t < num_tensors; ++t) {
    const size_t rows = static_cast<size_t>(first_dims[t]);
    const size_t cols = static_cast<size_t>(last_dims[t]);
    const size_t base = static_cast<size_t>(offsets[t]);
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        const size_t rowwise_idx = base + row * cols + col;
        const size_t columnwise_idx = base + col * rows + row;
        const float value = static_cast<float>(input_host[rowwise_idx]);
        amax_ref[t] = std::max(amax_ref[t], std::fabs(value));
        const OutputType quantized = static_cast<OutputType>(value * scales[t]);
        rowwise_ref[rowwise_idx] = to_byte(quantized);
        columnwise_ref[columnwise_idx] = to_byte(quantized);
      }
    }
  }

  for (size_t i = 0; i < allocation_elements; ++i) {
    EXPECT_EQ(output_rowwise[i], rowwise_ref[i]) << "rowwise byte mismatch at " << i;
    EXPECT_EQ(output_columnwise[i], columnwise_ref[i]) << "columnwise byte mismatch at " << i;
  }
  for (size_t t = 0; t < num_tensors; ++t) {
    EXPECT_FLOAT_EQ(amax[t], amax_ref[t]) << "amax mismatch for tensor " << t;
    EXPECT_FLOAT_EQ(scale_inv[t], 1.0f / scales[t]) << "rowwise scale_inv mismatch for tensor "
                                                    << t;
    EXPECT_FLOAT_EQ(columnwise_scale_inv[t], 1.0f / scales[t])
        << "columnwise scale_inv mismatch for tensor " << t;
  }
}

template <typename InputType, typename OutputType>
void run_grouped_fp8_precomputed_scale_no_amax_test() {
  const DType itype = TypeInfo<InputType>::dtype;
  const DType otype = TypeInfo<OutputType>::dtype;

  const size_t num_tensors = 2;
  const size_t rows_per_tensor = 3;
  const size_t cols = 4;
  const size_t logical_elements = num_tensors * rows_per_tensor * cols;
  const std::vector<size_t> logical_shape = {num_tensors * rows_per_tensor, cols};

  std::vector<InputType> input_host(logical_elements);
  for (size_t i = 0; i < logical_elements; ++i) {
    input_host[i] = static_cast<InputType>((static_cast<int>(i) - 7) * 0.125f);
  }
  const std::vector<float> scales = {0.75f, 1.5f};

  auto input_dev = cuda_alloc<InputType>(logical_elements * sizeof(InputType));
  auto output_dev = cuda_alloc<uint8_t>(logical_elements * sizeof(uint8_t));
  auto scale_dev = cuda_alloc<float>(num_tensors * sizeof(float));
  auto scale_inv_dev = cuda_alloc<float>(num_tensors * sizeof(float));

  NVTE_CHECK_CUDA(cudaMemcpy(input_dev.get(), input_host.data(),
                             logical_elements * sizeof(InputType), cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemset(output_dev.get(), 0, logical_elements));
  NVTE_CHECK_CUDA(cudaMemcpy(scale_dev.get(), scales.data(), num_tensors * sizeof(float),
                             cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemset(scale_inv_dev.get(), 0, num_tensors * sizeof(float)));

  GroupedTensorHandle input(nvte_create_grouped_tensor(
      NVTE_DELAYED_TENSOR_SCALING, num_tensors,
      nvte_make_shape(logical_shape.data(), logical_shape.size())));
  GroupedTensorHandle output(nvte_create_grouped_tensor(
      NVTE_DELAYED_TENSOR_SCALING, num_tensors,
      nvte_make_shape(logical_shape.data(), logical_shape.size())));

  set_basic_tensor(input.get(), kNVTEGroupedRowwiseData, input_dev.get(),
                   static_cast<NVTEDType>(itype), {logical_elements});
  set_basic_tensor(output.get(), kNVTEGroupedRowwiseData, output_dev.get(),
                   static_cast<NVTEDType>(otype), {logical_elements});
  set_basic_tensor(output.get(), kNVTEGroupedScale, scale_dev.get(), kNVTEFloat32, {num_tensors});
  set_basic_tensor(output.get(), kNVTEGroupedRowwiseScaleInv, scale_inv_dev.get(), kNVTEFloat32,
                   {num_tensors});

  nvte_group_quantize(input.get(), output.get(), nullptr, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<uint8_t> output_host(logical_elements);
  std::vector<float> scale_inv(num_tensors);
  NVTE_CHECK_CUDA(
      cudaMemcpy(output_host.data(), output_dev.get(), logical_elements, cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(scale_inv.data(), scale_inv_dev.get(), num_tensors * sizeof(float),
                             cudaMemcpyDeviceToHost));

  for (size_t t = 0; t < num_tensors; ++t) {
    const size_t base = t * rows_per_tensor * cols;
    for (size_t i = 0; i < rows_per_tensor * cols; ++i) {
      const OutputType quantized = static_cast<OutputType>(
          static_cast<float>(input_host[base + i]) * scales[t]);
      EXPECT_EQ(output_host[base + i], to_byte(quantized)) << "output mismatch at tensor " << t
                                                           << " element " << i;
    }
    EXPECT_FLOAT_EQ(scale_inv[t], 1.0f / scales[t]) << "scale_inv mismatch for tensor " << t;
  }
}

TEST(GroupedCastFP8Test, DelayedScalingHandlesCapacityPadding) {
  run_grouped_fp8_delayed_capacity_test<fp32, fp8e4m3>();
}

TEST(GroupedCastFP8Test, PrecomputedScaleCanOmitAmax) {
  run_grouped_fp8_precomputed_scale_no_amax_test<bf16, fp8e5m2>();
}

}  // namespace
