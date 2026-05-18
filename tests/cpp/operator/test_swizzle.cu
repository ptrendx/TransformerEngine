/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/swizzle.h>

#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;

constexpr int MAT_TILE_DIM_M = 128;
constexpr int MAT_TILE_DIM_K = 128;
constexpr int MXFP8_BLOCK_SIZE = 32;

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K, bool row_scaling>
void compute_ref_swizzle(const uint8_t *h_input, uint8_t *h_output,
                         const size_t M, const size_t K) {

  constexpr int NEW_SF_TILE_DIM_M = SF_TILE_DIM_M / 4;
  constexpr int NEW_SF_TILE_DIM_K = SF_TILE_DIM_K * 4;
  constexpr int SF_TILE_SIZE = SF_TILE_DIM_M * SF_TILE_DIM_K;

  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {

      int tile_id_m = m / SF_TILE_DIM_M;
      int tile_id_k = k / SF_TILE_DIM_K;
      int m_in_tile = m % SF_TILE_DIM_M;
      int k_in_tile = k % SF_TILE_DIM_K;

      int row_in_new_tile = m_in_tile % NEW_SF_TILE_DIM_M;
      int col_in_new_tile = m_in_tile / NEW_SF_TILE_DIM_M * SF_TILE_DIM_K + k_in_tile;

      int tile_output_ptr = tile_id_m * SF_TILE_DIM_M * K + tile_id_k * SF_TILE_SIZE;
      int out_index = tile_output_ptr + row_in_new_tile * NEW_SF_TILE_DIM_K + col_in_new_tile;
      if constexpr(row_scaling)
        h_output[out_index] = h_input[k + m * K];
      else
        h_output[out_index] = h_input[k * M + m];
    }
  }
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K, bool row_scaling>
void compute_ref_unswizzle(const uint8_t *h_input, uint8_t *h_output,
                           const size_t M, const size_t K) {

  constexpr int NEW_SF_TILE_DIM_M = SF_TILE_DIM_M / 4;
  constexpr int NEW_SF_TILE_DIM_K = SF_TILE_DIM_K * 4;
  constexpr int SF_TILE_SIZE = SF_TILE_DIM_M * SF_TILE_DIM_K;

  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {

      int tile_id_m = m / SF_TILE_DIM_M;
      int tile_id_k = k / SF_TILE_DIM_K;
      int m_in_tile = m % SF_TILE_DIM_M;
      int k_in_tile = k % SF_TILE_DIM_K;

      int row_in_new_tile = m_in_tile % NEW_SF_TILE_DIM_M;
      int col_in_new_tile = m_in_tile / NEW_SF_TILE_DIM_M * SF_TILE_DIM_K + k_in_tile;

      int tile_input_ptr = tile_id_m * SF_TILE_DIM_M * K + tile_id_k * SF_TILE_SIZE;
      int in_index = tile_input_ptr + row_in_new_tile * NEW_SF_TILE_DIM_K + col_in_new_tile;
      if constexpr(row_scaling)
        h_output[k + m * K] = h_input[in_index];
      else
        h_output[k * M + m] = h_input[in_index];
    }
  }
}

void performTestSwizzle1D(const int num_tiles_M, const int num_tiles_K, bool rowwise, bool columnwise, const bool transa) {
  using namespace test;

  int SF_MODE_X, SF_MODE_Y;
  if (rowwise) {
    SF_MODE_X = 1;
    SF_MODE_Y = 32;
  }
  if (columnwise) {
    SF_MODE_X = 32;
    SF_MODE_Y = 1;
  }

  if ((rowwise && columnwise) || !(rowwise || columnwise)){
    GTEST_SKIP() << "TEST SKIPPED, The scaling mode " + std::to_string(SF_MODE_X) + "x" +
      std::to_string(SF_MODE_Y) + "is not implemented.";
  }

  DType dtype = DType::kFloat8E4M3;

  const size_t M = num_tiles_M * MAT_TILE_DIM_M;
  const size_t K = num_tiles_K * MAT_TILE_DIM_K;
  const auto data_shape = transa ? std::vector<size_t>{M, K} : std::vector<size_t>{K, M};

  const auto scale_shape = std::vector<size_t>{data_shape[0] / SF_MODE_X, data_shape[1] /SF_MODE_Y};

  std::vector<int> scaling_mode = {SF_MODE_X, SF_MODE_Y, 0};
  Tensor input("input", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  Tensor output("output", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  output.set_with_gemm_swizzled_scales(true);

  fillUniform(&input);

  std::unique_ptr<uint8_t[]> ref_output = std::make_unique<uint8_t[]>(scale_shape[0] * scale_shape[1]);

  nvte_swizzle_scaling_factors(input.data(), output.data(), 0);

  if (rowwise)
    compute_ref_swizzle<128, 4, true>(input.rowwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), scale_shape[0], scale_shape[1]);
  else
    compute_ref_swizzle<128, 4, false>(input.columnwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), scale_shape[1], scale_shape[0]);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  output.to_cpu();
  if (rowwise) {
    compareResults("output_swizzle", output.rowwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), scale_shape[0] * scale_shape[1]);
  } else {
    compareResults("output_swizzle", output.columnwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), scale_shape[0] * scale_shape[1]);
  }
}

void performTestUnswizzle1D(const size_t M, const size_t K, bool rowwise, bool columnwise, const bool transa) {
  using namespace test;

  int SF_MODE_X, SF_MODE_Y;
  if (rowwise) {
    SF_MODE_X = 1;
    SF_MODE_Y = 32;
  }
  if (columnwise) {
    SF_MODE_X = 32;
    SF_MODE_Y = 1;
  }

  if (!rowwise && !columnwise) {
    GTEST_SKIP() << "TEST SKIPPED, Either rowwise or columnwise scaling mode must be true.";
  }
  if (rowwise && columnwise) {
    GTEST_SKIP() << "TEST SKIPPED, The scaling mode " + std::to_string(SF_MODE_X) + "x" +
      std::to_string(SF_MODE_Y) + " is not implemented.";
  }

  DType dtype = DType::kFloat8E4M3;

  const auto data_shape = transa ? std::vector<size_t>{M, K} : std::vector<size_t>{K, M};

  Tensor input("input", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  input.set_with_gemm_swizzled_scales(true);
  Tensor output("output", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);

  fillUniform(&input);

  // Use the actual padded compact scale shape from the tensor for both the reference
  // and the comparison. This correctly covers padded cases where M is not a multiple
  // of 128 or K/32 is not a multiple of 4.
  const auto padded_scale_shape = rowwise
    ? input.rowwise_scale_inv_shape()
    : input.columnwise_scale_inv_shape();
  const size_t padded_dim0 = padded_scale_shape.data[0];
  const size_t padded_dim1 = padded_scale_shape.data[1];
  std::unique_ptr<uint8_t[]> ref_output = std::make_unique<uint8_t[]>(padded_dim0 * padded_dim1);

  nvte_unswizzle_scaling_factors(input.data(), output.data(), 0);

  if (rowwise)
    compute_ref_unswizzle<128, 4, true>(input.rowwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), padded_dim0, padded_dim1);
  else
    compute_ref_unswizzle<128, 4, false>(input.columnwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), padded_dim1, padded_dim0);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  output.to_cpu();
  if (rowwise) {
    compareResults("output_unswizzle", output.rowwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), padded_dim0 * padded_dim1);
  } else {
    compareResults("output_unswizzle", output.columnwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), padded_dim0 * padded_dim1);
  }
}

// Zero out padding in a scale_inv CPU buffer so that the CPU reference
// matches the kernel, which zeroes elements outside the original dims.
// The buffer is stored in leading-dim-major order (row-major for rowwise,
// column-major for colwise).  `padded_rows x padded_cols` is the full
// (padded) shape; `orig_rows` / `orig_cols` are the unpadded extents.
static void zero_scale_inv_padding(uint8_t *buf,
                                   size_t padded_rows, size_t padded_cols,
                                   size_t orig_rows, size_t orig_cols) {
  for (size_t r = 0; r < padded_rows; ++r) {
    for (size_t c = 0; c < padded_cols; ++c) {
      if (r >= orig_rows || c >= orig_cols) {
        buf[r * padded_cols + c] = 0;
      }
    }
  }
}

void performTestGroupedSwizzleMXFP8(const int num_tensors, const size_t M, const size_t K) {
  using namespace transformer_engine;
  using namespace test;

  std::vector<std::unique_ptr<Tensor>> input_tensors;
  std::vector<std::unique_ptr<Tensor>> output_tensors;
  std::vector<Tensor*> input_ptrs;
  std::vector<Tensor*> output_ptrs;
  input_tensors.reserve(num_tensors);
  output_tensors.reserve(num_tensors);
  input_ptrs.reserve(num_tensors);
  output_ptrs.reserve(num_tensors);

  constexpr size_t BLOCK_SIZE = 32;
  const std::vector<size_t> shape{M, K};
  for (int i = 0; i < num_tensors; ++i) {
    auto input = std::make_unique<Tensor>("input_" + std::to_string(i), shape,
                                          DType::kFloat8E4M3, true, true,
                                          NVTE_MXFP8_1D_SCALING);
    auto output = std::make_unique<Tensor>("output_" + std::to_string(i), shape,
                                           DType::kFloat8E4M3, true, true,
                                           NVTE_MXFP8_1D_SCALING);
    fillUniform(input.get());
    fillUniform(output.get());

    // The grouped swizzle kernel zeroes scale_inv elements that fall
    // outside the original (unpadded) dimensions.  Mirror that in the
    // per-tensor CPU buffers so the CPU reference produces identical output.
    input->to_cpu();
    const NVTEShape rs = input->rowwise_scale_inv_shape();
    zero_scale_inv_padding(input->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                           rs.data[0], rs.data[1],
                           M, divide_round_up(K, BLOCK_SIZE));
    const NVTEShape cs = input->columnwise_scale_inv_shape();
    zero_scale_inv_padding(input->columnwise_cpu_scale_inv_ptr<uint8_t>(),
                           cs.data[0], cs.data[1],
                           divide_round_up(M, BLOCK_SIZE), K);
    input->from_cpu();

    input_ptrs.push_back(input.get());
    output_ptrs.push_back(output.get());
    input_tensors.emplace_back(std::move(input));
    output_tensors.emplace_back(std::move(output));
  }

  GroupedBuffers grouped_input = build_grouped_tensor(input_ptrs, NVTE_MXFP8_1D_SCALING);
  GroupedBuffers grouped_output = build_grouped_tensor(output_ptrs, NVTE_MXFP8_1D_SCALING);
  const uint8_t input_swizzled = 0;
  nvte_set_grouped_tensor_param(grouped_input.get_handle(),
                                kNVTEGroupedWithGEMMSwizzledScales,
                                &input_swizzled, sizeof(input_swizzled));
  const uint8_t output_swizzled = 1;
  nvte_set_grouped_tensor_param(grouped_output.get_handle(),
                                kNVTEGroupedWithGEMMSwizzledScales,
                                &output_swizzled, sizeof(output_swizzled));

  const NVTEShape row_shape = input_tensors[0]->rowwise_scale_inv_shape();
  const NVTEShape col_shape = input_tensors[0]->columnwise_scale_inv_shape();
  const size_t row_numel = row_shape.data[0] * row_shape.data[1];
  const size_t col_numel = col_shape.data[0] * col_shape.data[1];

  NVTE_CHECK_CUDA(cudaMemset(grouped_output.scale_inv.get(), 0, num_tensors * row_numel));
  NVTE_CHECK_CUDA(cudaMemset(grouped_output.columnwise_scale_inv.get(), 0, num_tensors * col_numel));

  nvte_swizzle_grouped_scaling_factors(grouped_input.get_handle(),
                                       grouped_output.get_handle(), 0);

  std::vector<uint8_t> output_row(num_tensors * row_numel);
  std::vector<uint8_t> output_col(num_tensors * col_numel);
  NVTE_CHECK_CUDA(cudaMemcpy(output_row.data(), grouped_output.scale_inv.get(),
                             output_row.size(), cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(output_col.data(), grouped_output.columnwise_scale_inv.get(),
                             output_col.size(), cudaMemcpyDeviceToHost));

  std::vector<uint8_t> ref_row(num_tensors * row_numel);
  std::vector<uint8_t> ref_col(num_tensors * col_numel);
  for (int i = 0; i < num_tensors; ++i) {
    compute_ref_swizzle<128, 4, true>(input_tensors[i]->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                                      ref_row.data() + i * row_numel,
                                      row_shape.data[0], row_shape.data[1]);
    compute_ref_swizzle<128, 4, false>(
        input_tensors[i]->columnwise_cpu_scale_inv_ptr<uint8_t>(),
        ref_col.data() + i * col_numel,
        col_shape.data[1], col_shape.data[0]);
  }

  compareResults("grouped_swizzle_rowwise", output_row.data(), ref_row.data(),
                 num_tensors * row_numel);
  compareResults("grouped_swizzle_colwise", output_col.data(), ref_col.data(),
                 num_tensors * col_numel);
}

class SwizzleTestSuite : public ::testing::TestWithParam<std::tuple<std::pair<int, int>, std::pair<bool, bool>, bool>> {};


TEST_P(SwizzleTestSuite, TestSwizzle) {
    using namespace transformer_engine;
    using namespace test;

  const auto num_tiles = std::get<0>(GetParam());
  const auto scaling_mode = std::get<1>(GetParam());
  const auto transa = std::get<2>(GetParam());

  performTestSwizzle1D(num_tiles.first, num_tiles.second,
                       scaling_mode.first, scaling_mode.second,
                       transa);
}

TEST(SwizzleFullTileFastPathTest, TestSwizzleMXFP8RegularFullTileFastPaths) {
  if (test::getDeviceComputeCapability() < test::blackwellComputeCapability) {
    GTEST_SKIP() << "Blackwell full-tile swizzle fast paths require Blackwell or newer.";
  }

  // Rowwise K=1024 gives 32 scale columns and selects the M_TILES_PER_BLOCK=4
  // full-M path when M has 4 swizzle tiles.
  performTestSwizzle1D(/*num_tiles_M=*/4, /*num_tiles_K=*/8,
                       /*rowwise=*/true, /*columnwise=*/false, /*transa=*/true);

  // Rowwise K=2048 gives 64 scale columns and selects the M_TILES_PER_BLOCK=5
  // full-M path when M has 5 swizzle tiles.
  performTestSwizzle1D(/*num_tiles_M=*/5, /*num_tiles_K=*/16,
                       /*rowwise=*/true, /*columnwise=*/false, /*transa=*/true);

  // Columnwise M=4096, K=128 has no M/K padding and 32 coalesced K tile blocks.
  performTestSwizzle1D(/*num_tiles_M=*/32, /*num_tiles_K=*/1,
                       /*rowwise=*/false, /*columnwise=*/true, /*transa=*/true);
}

class UnswizzleTestSuite : public ::testing::TestWithParam<std::tuple<std::pair<size_t, size_t>, std::pair<bool, bool>, bool>> {};

TEST_P(UnswizzleTestSuite, TestUnswizzle) {
    using namespace transformer_engine;
    using namespace test;

  const auto data_shape = std::get<0>(GetParam());
  const auto scaling_mode = std::get<1>(GetParam());
  const auto transa = std::get<2>(GetParam());

  performTestUnswizzle1D(data_shape.first, data_shape.second,
                         scaling_mode.first, scaling_mode.second,
                         transa);
}

void performTestGroupedUnswizzleMXFP8(const int num_tensors, const size_t M, const size_t K) {
  using namespace transformer_engine;
  using namespace test;

  std::vector<std::unique_ptr<Tensor>> input_tensors;
  std::vector<std::unique_ptr<Tensor>> output_tensors;
  std::vector<Tensor*> input_ptrs;
  std::vector<Tensor*> output_ptrs;
  input_tensors.reserve(num_tensors);
  output_tensors.reserve(num_tensors);
  input_ptrs.reserve(num_tensors);
  output_ptrs.reserve(num_tensors);

  const std::vector<size_t> shape{M, K};
  for (int i = 0; i < num_tensors; ++i) {
    auto input = std::make_unique<Tensor>("input_" + std::to_string(i), shape,
                                          DType::kFloat8E4M3, true, true,
                                          NVTE_MXFP8_1D_SCALING);
    auto output = std::make_unique<Tensor>("output_" + std::to_string(i), shape,
                                           DType::kFloat8E4M3, true, true,
                                           NVTE_MXFP8_1D_SCALING);
    fillUniform(input.get());
    fillUniform(output.get());

    input_ptrs.push_back(input.get());
    output_ptrs.push_back(output.get());
    input_tensors.emplace_back(std::move(input));
    output_tensors.emplace_back(std::move(output));
  }

  GroupedBuffers grouped_input = build_grouped_tensor(input_ptrs, NVTE_MXFP8_1D_SCALING);
  GroupedBuffers grouped_output = build_grouped_tensor(output_ptrs, NVTE_MXFP8_1D_SCALING);
  const uint8_t input_swizzled = 1;
  nvte_set_grouped_tensor_param(grouped_input.get_handle(),
                                kNVTEGroupedWithGEMMSwizzledScales,
                                &input_swizzled, sizeof(input_swizzled));
  const uint8_t output_swizzled = 0;
  nvte_set_grouped_tensor_param(grouped_output.get_handle(),
                                kNVTEGroupedWithGEMMSwizzledScales,
                                &output_swizzled, sizeof(output_swizzled));

  const NVTEShape row_shape = input_tensors[0]->rowwise_scale_inv_shape();
  const NVTEShape col_shape = input_tensors[0]->columnwise_scale_inv_shape();
  const size_t row_numel = row_shape.data[0] * row_shape.data[1];
  const size_t col_numel = col_shape.data[0] * col_shape.data[1];

  NVTE_CHECK_CUDA(cudaMemset(grouped_output.scale_inv.get(), 0, num_tensors * row_numel));
  NVTE_CHECK_CUDA(cudaMemset(grouped_output.columnwise_scale_inv.get(), 0, num_tensors * col_numel));

  nvte_unswizzle_grouped_scaling_factors(grouped_input.get_handle(),
                                         grouped_output.get_handle(), 0);

  std::vector<uint8_t> output_row(num_tensors * row_numel);
  std::vector<uint8_t> output_col(num_tensors * col_numel);
  NVTE_CHECK_CUDA(cudaMemcpy(output_row.data(), grouped_output.scale_inv.get(),
                             output_row.size(), cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(output_col.data(), grouped_output.columnwise_scale_inv.get(),
                             output_col.size(), cudaMemcpyDeviceToHost));

  std::vector<uint8_t> ref_row(num_tensors * row_numel);
  std::vector<uint8_t> ref_col(num_tensors * col_numel);
  for (int i = 0; i < num_tensors; ++i) {
    compute_ref_unswizzle<128, 4, true>(input_tensors[i]->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                                        ref_row.data() + i * row_numel,
                                        row_shape.data[0], row_shape.data[1]);
    compute_ref_unswizzle<128, 4, false>(
        input_tensors[i]->columnwise_cpu_scale_inv_ptr<uint8_t>(),
        ref_col.data() + i * col_numel,
        col_shape.data[1], col_shape.data[0]);
  }

  compareResults("grouped_unswizzle_rowwise", output_row.data(), ref_row.data(),
                 num_tensors * row_numel);
  compareResults("grouped_unswizzle_colwise", output_col.data(), ref_col.data(),
                 num_tensors * col_numel);
}

void performTestGroupedSwizzleUnswizzleRoundtrip(const int num_tensors, const size_t M,
                                                  const size_t K) {
  using namespace transformer_engine;
  using namespace test;

  constexpr size_t BLOCK_SIZE = 32;
  const std::vector<size_t> shape{M, K};

  std::vector<std::unique_ptr<Tensor>> orig_tensors, mid_tensors, final_tensors;
  std::vector<Tensor*> orig_ptrs, mid_ptrs, final_ptrs;
  orig_tensors.reserve(num_tensors);
  mid_tensors.reserve(num_tensors);
  final_tensors.reserve(num_tensors);

  for (int i = 0; i < num_tensors; ++i) {
    auto orig = std::make_unique<Tensor>("orig_" + std::to_string(i), shape,
                                         DType::kFloat8E4M3, true, true, NVTE_MXFP8_1D_SCALING);
    auto mid = std::make_unique<Tensor>("mid_" + std::to_string(i), shape,
                                        DType::kFloat8E4M3, true, true, NVTE_MXFP8_1D_SCALING);
    auto fin = std::make_unique<Tensor>("fin_" + std::to_string(i), shape,
                                        DType::kFloat8E4M3, true, true, NVTE_MXFP8_1D_SCALING);
    fillUniform(orig.get());

    // Zero padding so the round-trip comparison is exact.
    orig->to_cpu();
    const NVTEShape rs = orig->rowwise_scale_inv_shape();
    zero_scale_inv_padding(orig->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                           rs.data[0], rs.data[1],
                           M, divide_round_up(K, BLOCK_SIZE));
    const NVTEShape cs = orig->columnwise_scale_inv_shape();
    zero_scale_inv_padding(orig->columnwise_cpu_scale_inv_ptr<uint8_t>(),
                           cs.data[0], cs.data[1],
                           divide_round_up(M, BLOCK_SIZE), K);
    orig->from_cpu();

    orig_ptrs.push_back(orig.get());
    mid_ptrs.push_back(mid.get());
    final_ptrs.push_back(fin.get());
    orig_tensors.emplace_back(std::move(orig));
    mid_tensors.emplace_back(std::move(mid));
    final_tensors.emplace_back(std::move(fin));
  }

  GroupedBuffers grouped_orig = build_grouped_tensor(orig_ptrs, NVTE_MXFP8_1D_SCALING);
  GroupedBuffers grouped_mid = build_grouped_tensor(mid_ptrs, NVTE_MXFP8_1D_SCALING);
  GroupedBuffers grouped_fin = build_grouped_tensor(final_ptrs, NVTE_MXFP8_1D_SCALING);

  const NVTEShape row_shape = orig_tensors[0]->rowwise_scale_inv_shape();
  const NVTEShape col_shape = orig_tensors[0]->columnwise_scale_inv_shape();
  const size_t row_numel = row_shape.data[0] * row_shape.data[1];
  const size_t col_numel = col_shape.data[0] * col_shape.data[1];

  const uint8_t no_swizzle = 0, has_swizzle = 1;
  nvte_set_grouped_tensor_param(grouped_orig.get_handle(), kNVTEGroupedWithGEMMSwizzledScales,
                                &no_swizzle, sizeof(no_swizzle));
  nvte_set_grouped_tensor_param(grouped_mid.get_handle(), kNVTEGroupedWithGEMMSwizzledScales,
                                &has_swizzle, sizeof(has_swizzle));
  nvte_set_grouped_tensor_param(grouped_fin.get_handle(), kNVTEGroupedWithGEMMSwizzledScales,
                                &no_swizzle, sizeof(no_swizzle));

  NVTE_CHECK_CUDA(cudaMemset(grouped_mid.scale_inv.get(), 0, num_tensors * row_numel));
  NVTE_CHECK_CUDA(cudaMemset(grouped_mid.columnwise_scale_inv.get(), 0, num_tensors * col_numel));
  NVTE_CHECK_CUDA(cudaMemset(grouped_fin.scale_inv.get(), 0, num_tensors * row_numel));
  NVTE_CHECK_CUDA(cudaMemset(grouped_fin.columnwise_scale_inv.get(), 0, num_tensors * col_numel));

  nvte_swizzle_grouped_scaling_factors(grouped_orig.get_handle(), grouped_mid.get_handle(), 0);
  nvte_unswizzle_grouped_scaling_factors(grouped_mid.get_handle(), grouped_fin.get_handle(), 0);

  std::vector<uint8_t> result_row(num_tensors * row_numel);
  std::vector<uint8_t> result_col(num_tensors * col_numel);
  NVTE_CHECK_CUDA(cudaMemcpy(result_row.data(), grouped_fin.scale_inv.get(),
                             result_row.size(), cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(result_col.data(), grouped_fin.columnwise_scale_inv.get(),
                             result_col.size(), cudaMemcpyDeviceToHost));

  std::vector<uint8_t> ref_row(num_tensors * row_numel);
  std::vector<uint8_t> ref_col(num_tensors * col_numel);
  for (int i = 0; i < num_tensors; ++i) {
    memcpy(ref_row.data() + i * row_numel,
           orig_tensors[i]->rowwise_cpu_scale_inv_ptr<uint8_t>(), row_numel);
    memcpy(ref_col.data() + i * col_numel,
           orig_tensors[i]->columnwise_cpu_scale_inv_ptr<uint8_t>(), col_numel);
  }

  compareResults("grouped_roundtrip_rowwise", result_row.data(), ref_row.data(),
                 num_tensors * row_numel);
  compareResults("grouped_roundtrip_colwise", result_col.data(), ref_col.data(),
                 num_tensors * col_numel);
}

void performTestGroupedSwizzleMXFP8Variable(const std::vector<std::pair<size_t, size_t>>& shapes) {
  using namespace transformer_engine;
  using namespace test;

  int num_tensors = shapes.size();
  std::vector<std::unique_ptr<Tensor>> input_tensors;
  std::vector<std::unique_ptr<Tensor>> output_tensors;
  std::vector<Tensor*> input_ptrs;
  std::vector<Tensor*> output_ptrs;
  input_tensors.reserve(num_tensors);
  output_tensors.reserve(num_tensors);
  input_ptrs.reserve(num_tensors);
  output_ptrs.reserve(num_tensors);

  constexpr size_t BLOCK_SIZE = 32;
  for (int i = 0; i < num_tensors; ++i) {
    const std::vector<size_t> shape{shapes[i].first, shapes[i].second};
    auto input = std::make_unique<Tensor>("input_" + std::to_string(i), shape,
                                          DType::kFloat8E4M3, true, true,
                                          NVTE_MXFP8_1D_SCALING);
    auto output = std::make_unique<Tensor>("output_" + std::to_string(i), shape,
                                           DType::kFloat8E4M3, true, true,
                                           NVTE_MXFP8_1D_SCALING);
    fillUniform(input.get());
    fillUniform(output.get());

    // Zero padding
    input->to_cpu();
    const NVTEShape rs = input->rowwise_scale_inv_shape();
    zero_scale_inv_padding(input->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                           rs.data[0], rs.data[1],
                           shapes[i].first, (shapes[i].second + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const NVTEShape cs = input->columnwise_scale_inv_shape();
    zero_scale_inv_padding(input->columnwise_cpu_scale_inv_ptr<uint8_t>(),
                           cs.data[0], cs.data[1],
                           (shapes[i].first + BLOCK_SIZE - 1) / BLOCK_SIZE, shapes[i].second);
    input->from_cpu();

    input_ptrs.push_back(input.get());
    output_ptrs.push_back(output.get());
    input_tensors.emplace_back(std::move(input));
    output_tensors.emplace_back(std::move(output));
  }

  GroupedBuffers grouped_input = build_grouped_tensor(input_ptrs, NVTE_MXFP8_1D_SCALING);
  GroupedBuffers grouped_output = build_grouped_tensor(output_ptrs, NVTE_MXFP8_1D_SCALING);

  const uint8_t input_swizzled = 0;
  nvte_set_grouped_tensor_param(grouped_input.get_handle(),
                                kNVTEGroupedWithGEMMSwizzledScales,
                                &input_swizzled, sizeof(input_swizzled));
  const uint8_t output_swizzled = 1;
  nvte_set_grouped_tensor_param(grouped_output.get_handle(),
                                kNVTEGroupedWithGEMMSwizzledScales,
                                &output_swizzled, sizeof(output_swizzled));

  nvte_swizzle_grouped_scaling_factors(grouped_input.get_handle(),
                                       grouped_output.get_handle(),
                                       0);

  cudaDeviceSynchronize();
  NVTE_CHECK_CUDA(cudaGetLastError());

  // Verification
  size_t row_offset = 0;
  size_t col_offset = 0;
  for (int i = 0; i < num_tensors; ++i) {
    const NVTEShape row_shape = input_tensors[i]->rowwise_scale_inv_shape();
    const NVTEShape col_shape = input_tensors[i]->columnwise_scale_inv_shape();
    const size_t row_numel = row_shape.data[0] * row_shape.data[1];
    const size_t col_numel = col_shape.data[0] * col_shape.data[1];

    std::vector<uint8_t> output_row_host(row_numel);
    std::vector<uint8_t> output_col_host(col_numel);
    NVTE_CHECK_CUDA(cudaMemcpy(output_row_host.data(),
                               static_cast<uint8_t*>(grouped_output.scale_inv.get()) + row_offset,
                               row_numel, cudaMemcpyDeviceToHost));
    NVTE_CHECK_CUDA(cudaMemcpy(output_col_host.data(),
                               static_cast<uint8_t*>(grouped_output.columnwise_scale_inv.get()) + col_offset,
                               col_numel, cudaMemcpyDeviceToHost));

    std::vector<uint8_t> ref_row(row_numel);
    std::vector<uint8_t> ref_col(col_numel);
    compute_ref_swizzle<128, 4, true>(input_tensors[i]->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                                      ref_row.data(),
                                      row_shape.data[0], row_shape.data[1]);
    compute_ref_swizzle<128, 4, false>(
        input_tensors[i]->columnwise_cpu_scale_inv_ptr<uint8_t>(),
        ref_col.data(),
        col_shape.data[1], col_shape.data[0]);

    compareResults("grouped_swizzle_variable_rowwise_" + std::to_string(i),
                   output_row_host.data(), ref_row.data(), row_numel);
    compareResults("grouped_swizzle_variable_colwise_" + std::to_string(i),
                   output_col_host.data(), ref_col.data(), col_numel);

    row_offset += row_numel;
    col_offset += col_numel;
  }
}

class SwizzleGroupedVariableTestSuite
    : public ::testing::TestWithParam<std::vector<std::pair<size_t, size_t>>> {};

TEST_P(SwizzleGroupedVariableTestSuite, TestGroupedSwizzleMXFP8Variable) {
  const auto shapes = GetParam();
  performTestGroupedSwizzleMXFP8Variable(shapes);
}

TEST(SwizzleGroupedVariableTest, TestGroupedSwizzleMXFP8VariableRowwise32And64ScaleColumns) {
  // Explicitly cover grouped-variable rowwise cases with 32 and 64 scale columns
  // (K=1024 and K=2048 for MXFP8) because the Blackwell row-coalesced path has
  // specialized shared-memory pitch handling for those common widths.
  const std::vector<std::vector<std::pair<size_t, size_t>>> shape_sets{
      {{128, 1024}, {512, 1024}, {64, 1024}},
      {{128, 2048}, {512, 2048}, {64, 2048}},
      {{200, 1024}, {33, 1024}, {1025, 1024}},
  };

  for (const auto& shapes : shape_sets) {
    performTestGroupedSwizzleMXFP8Variable(shapes);
  }
}

TEST(SwizzleGroupedVariablePersistentTest, TestGroupedSwizzleMXFP8VariableDirectGridLargeWork) {
  int device = 0;
  NVTE_CHECK_CUDA(cudaGetDevice(&device));
  int sm_count = 0;
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  ASSERT_GT(sm_count, 0);

  constexpr size_t rows_per_swizzle_tile = 128;
  constexpr size_t k_dim = 512;
  const size_t row_coalesced_grid_tiles = static_cast<size_t>(sm_count) * 4;
  const size_t first_tensor_tiles =
      row_coalesced_grid_tiles > 1 ? row_coalesced_grid_tiles - 1 : 1;
  const std::vector<std::pair<size_t, size_t>> shapes{
      {first_tensor_tiles * rows_per_swizzle_tile, k_dim},
      {9 * rows_per_swizzle_tile, k_dim},
  };

  performTestGroupedSwizzleMXFP8Variable(shapes);
}

TEST(SwizzleGroupedVariablePersistentTest, TestGroupedSwizzleMXFP8VariableAlignedStackedFastPath) {
  // All M dimensions are already swizzle-tile aligned, so the grouped-variable
  // scale buffers can be swizzled as one stacked scale matrix.
  const std::vector<std::pair<size_t, size_t>> shapes{
      {2 * MAT_TILE_DIM_M, 1024},
      {5 * MAT_TILE_DIM_M, 1024},
      {7 * MAT_TILE_DIM_M, 1024},
      {11 * MAT_TILE_DIM_M, 1024},
  };

  performTestGroupedSwizzleMXFP8Variable(shapes);
}

TEST(SwizzleGroupedVariablePersistentTest,
     TestGroupedSwizzleMXFP8VariableAlignedStackedFullM32And64ScaleColumns) {
  if (test::getDeviceComputeCapability() < test::blackwellComputeCapability) {
    GTEST_SKIP() << "Blackwell aligned-stacked full-M swizzle requires Blackwell or newer.";
  }

  const std::vector<std::vector<std::pair<size_t, size_t>>> shape_sets{
      // Total stacked M tiles = 4, so K=1024/32=32 scale columns reaches the
      // M_TILES_PER_BLOCK=4 full-M branch.
      {{1 * MAT_TILE_DIM_M, 1024}, {3 * MAT_TILE_DIM_M, 1024}},
      // Total stacked M tiles = 5, so K=2048/32=64 scale columns reaches the
      // M_TILES_PER_BLOCK=5 full-M branch.
      {{2 * MAT_TILE_DIM_M, 2048}, {3 * MAT_TILE_DIM_M, 2048}},
  };

  for (const auto& shapes : shape_sets) {
    performTestGroupedSwizzleMXFP8Variable(shapes);
  }
}

TEST(SwizzleGroupedVariableSharedMemoryTest,
     TestGroupedSwizzleMXFP8VariableRowCoalescedSharedMemoryBoundary) {
  if (test::getDeviceComputeCapability() < test::blackwellComputeCapability) {
    GTEST_SKIP() << "Row-coalesced grouped variable swizzle requires Blackwell or newer.";
  }

  int device = 0;
  NVTE_CHECK_CUDA(cudaGetDevice(&device));
  int max_smem = 0;
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&max_smem,
                                         cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                         device));

  constexpr size_t scale_tile_m = 128;
  constexpr size_t row_coalesced_min_k = 32;
  constexpr size_t row_coalesced_k_alignment = sizeof(int4);
  const size_t max_scale_elems = static_cast<size_t>(max_smem) / scale_tile_m;
  if (max_scale_elems <= sizeof(int)) {
    GTEST_SKIP() << "Opt-in shared memory is too small for row-coalesced boundary coverage.";
  }

  size_t padded_scale_k =
      ((max_scale_elems - sizeof(int)) / row_coalesced_k_alignment) *
      row_coalesced_k_alignment;
  size_t boundary_padded_scale_k = 0;
  while (padded_scale_k >= row_coalesced_min_k) {
    const size_t slm_size = scale_tile_m * (padded_scale_k + sizeof(int));
    if (slm_size <= static_cast<size_t>(max_smem) &&
        slm_size + scale_tile_m * row_coalesced_k_alignment > static_cast<size_t>(max_smem)) {
      boundary_padded_scale_k = padded_scale_k;
      break;
    }
    if (padded_scale_k < row_coalesced_min_k + row_coalesced_k_alignment) {
      break;
    }
    padded_scale_k -= row_coalesced_k_alignment;
  }

  if (boundary_padded_scale_k == 0) {
    GTEST_SKIP() << "No aligned common K reaches the row-coalesced metadata boundary.";
  }

  const size_t k_dim = boundary_padded_scale_k * MXFP8_BLOCK_SIZE;
  const std::vector<std::pair<size_t, size_t>> shapes{
      {MAT_TILE_DIM_M, k_dim},
      {2 * MAT_TILE_DIM_M, k_dim},
  };

  performTestGroupedSwizzleMXFP8Variable(shapes);
}

TEST(SwizzleGroupedVariableSharedMemoryTest,
     TestGroupedSwizzleMXFP8VariableRowBatchedFallbackMetadataBoundary) {
  int device = 0;
  NVTE_CHECK_CUDA(cudaGetDevice(&device));
  int max_smem = 0;
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&max_smem,
                                         cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                         device));
  ASSERT_GT(max_smem, 0);

  constexpr size_t scale_tile_m = 128;
  constexpr size_t scale_tile_k = 4;
  constexpr size_t threadblock_dim = 32;
  constexpr size_t per_k_tile_slm_size = scale_tile_m * scale_tile_k * sizeof(uint8_t);
  constexpr size_t row_coalesced_min_k = 32;
  constexpr size_t row_coalesced_k_alignment = sizeof(int4);
  constexpr size_t variable_scheduler_metadata_bytes = sizeof(int);

  size_t boundary_num_tiles_k = 0;
  const size_t max_num_tiles_k = static_cast<size_t>(max_smem) / per_k_tile_slm_size;
  for (size_t num_tiles_k = max_num_tiles_k; num_tiles_k > 0; --num_tiles_k) {
    const size_t per_m_tile_slm_size = num_tiles_k * per_k_tile_slm_size;
    const size_t m_tiles_per_block =
        std::min(threadblock_dim, static_cast<size_t>(max_smem) / per_m_tile_slm_size);
    if (m_tiles_per_block > 1 &&
        per_m_tile_slm_size * m_tiles_per_block <= static_cast<size_t>(max_smem) &&
        per_m_tile_slm_size * m_tiles_per_block + variable_scheduler_metadata_bytes >
            static_cast<size_t>(max_smem)) {
      boundary_num_tiles_k = num_tiles_k;
      break;
    }
  }

  if (boundary_num_tiles_k == 0) {
    GTEST_SKIP() << "No grouped variable rowwise batched fallback shared-memory boundary.";
  }

  const size_t padded_scale_k = boundary_num_tiles_k * scale_tile_k;
  size_t original_scale_k = padded_scale_k;
  if (test::getDeviceComputeCapability() >= test::blackwellComputeCapability &&
      padded_scale_k >= row_coalesced_min_k &&
      padded_scale_k % row_coalesced_k_alignment == 0) {
    original_scale_k = padded_scale_k - 1;
  }
  ASSERT_GT(original_scale_k, 0);

  const size_t k_dim = original_scale_k * MXFP8_BLOCK_SIZE;
  const std::vector<std::pair<size_t, size_t>> shapes{
      {MAT_TILE_DIM_M, k_dim},
      {2 * MAT_TILE_DIM_M, k_dim},
  };

  performTestGroupedSwizzleMXFP8Variable(shapes);
}

TEST(SwizzleGroupedVariableSharedMemoryTest,
     TestGroupedSwizzleMXFP8VariableColumnNarrowMMetadataBoundary) {
  if (test::getDeviceComputeCapability() >= test::blackwellComputeCapability) {
    GTEST_SKIP() << "Blackwell byte-scale columnwise swizzle uses the coalesced path.";
  }

  int device = 0;
  NVTE_CHECK_CUDA(cudaGetDevice(&device));
  int max_smem = 0;
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&max_smem,
                                         cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                         device));
  ASSERT_GT(max_smem, 0);

  constexpr size_t scale_tile_m = 128;
  constexpr size_t scale_tile_k = 4;
  constexpr size_t threadblock_dim = 32;
  constexpr size_t narrow_m_per_m_tile_slm_size =
      threadblock_dim * scale_tile_m * scale_tile_k * sizeof(uint8_t);
  constexpr size_t variable_scheduler_metadata_bytes = sizeof(int);

  size_t boundary_num_tiles_m = 0;
  for (size_t num_tiles_m = 1; num_tiles_m < threadblock_dim; ++num_tiles_m) {
    const size_t narrow_m_slm_size = num_tiles_m * narrow_m_per_m_tile_slm_size;
    if (narrow_m_slm_size <= static_cast<size_t>(max_smem) &&
        narrow_m_slm_size + variable_scheduler_metadata_bytes > static_cast<size_t>(max_smem)) {
      boundary_num_tiles_m = num_tiles_m;
      break;
    }
  }

  if (boundary_num_tiles_m == 0) {
    GTEST_SKIP() << "No grouped variable columnwise narrow-M shared-memory boundary.";
  }

  const size_t common_last_dim = boundary_num_tiles_m * MAT_TILE_DIM_M;
  const std::vector<std::pair<size_t, size_t>> shapes{
      {MAT_TILE_DIM_M, common_last_dim},
      {2 * MAT_TILE_DIM_M, common_last_dim},
  };

  performTestGroupedSwizzleMXFP8Variable(shapes);
}

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  SwizzleGroupedVariableTestSuite,
  ::testing::Values(
    // Case 1: num_tensors = 1 (n+3 = 4, even). Check simple alignment.
    std::vector<std::pair<size_t, size_t>>{{1024, 1024}},

    // Case 2: num_tensors = 2 (n+3 = 5, odd). Forces padding logic to trigger.
    std::vector<std::pair<size_t, size_t>>{{128, 128}, {256, 256}},

    // Case 3: Mixed small/irregular shapes.
    std::vector<std::pair<size_t, size_t>>{{200, 160}, {33, 64}, {1, 32}},

    // Case 4: Large workload to verify multi-wave grouped-variable launch mapping.
    std::vector<std::pair<size_t, size_t>>(10, {4096, 4096}),

    // Case 5: Variable M, Uniform K (Semi-variable)
    std::vector<std::pair<size_t, size_t>>{{128, 256}, {512, 256}, {64, 256}},

    // Case 6: Uniform M, Variable K (Semi-variable)
    std::vector<std::pair<size_t, size_t>>{{512, 128}, {512, 1024}, {512, 32}}
  ),
  [](const testing::TestParamInfo<SwizzleGroupedVariableTestSuite::ParamType>& info) {
    return "VariableShapes_" + std::to_string(info.index) + "_N" + std::to_string(info.param.size());
  }
);

class SwizzleGroupedTestSuite
    : public ::testing::TestWithParam<std::tuple<int, size_t, size_t>> {};

TEST_P(SwizzleGroupedTestSuite, TestGroupedSwizzleMXFP8) {
  const auto num_tensors = std::get<0>(GetParam());
  const auto M = std::get<1>(GetParam());
  const auto K = std::get<2>(GetParam());
  performTestGroupedSwizzleMXFP8(num_tensors, M, K);
}

TEST(SwizzleGroupedFullTileFastPathTest, TestGroupedSwizzleMXFP8UniformFullTileFastPaths) {
  if (test::getDeviceComputeCapability() < test::blackwellComputeCapability) {
    GTEST_SKIP() << "Blackwell grouped full-tile swizzle fast paths require Blackwell or newer.";
  }

  // Grouped-uniform rowwise K=1024/2048 selects the 32/64-scale-column full-M
  // row-coalesced branches when M has 4 or 5 swizzle tiles respectively.
  performTestGroupedSwizzleMXFP8(/*num_tensors=*/3, /*M=*/4 * MAT_TILE_DIM_M,
                                 /*K=*/1024);
  performTestGroupedSwizzleMXFP8(/*num_tensors=*/3, /*M=*/5 * MAT_TILE_DIM_M,
                                 /*K=*/2048);

  // Grouped-uniform columnwise full-tile branch: no padding and 32 coalesced K
  // tile blocks in the scale matrix.
  performTestGroupedSwizzleMXFP8(/*num_tensors=*/3, /*M=*/32 * MAT_TILE_DIM_M,
                                 /*K=*/MAT_TILE_DIM_M);
}

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  SwizzleGroupedTestSuite,
  ::testing::Values(
    // M and K both divisible by 128
    std::make_tuple(3, 256, 256),
    std::make_tuple(4, 128, 128),
    // M not divisible by 128
    std::make_tuple(3, 200, 256),
    std::make_tuple(2, 65, 256),
    // K not divisible by 128
    std::make_tuple(3, 256, 160),
    std::make_tuple(2, 256, 96),
    // Neither M nor K divisible by 128
    std::make_tuple(3, 200, 160),
    std::make_tuple(4, 33, 64),
    std::make_tuple(2, 1, 32)
  ),
  [](const testing::TestParamInfo<SwizzleGroupedTestSuite::ParamType>& info) {
    return "n" + std::to_string(std::get<0>(info.param)) +
           "_M" + std::to_string(std::get<1>(info.param)) +
           "_K" + std::to_string(std::get<2>(info.param));
  }
);

// Build a "compact" grouped MXFP8 scale_inv buffer for swizzle input. This is
// the layout produced by the grouped MXFP8 quantize kernel: the per-tensor
// stride is `M_per_tensor * padded_K` (rowwise) or `DIVUP(M,32) * padded_K_for_cols`
// (columnwise) -- i.e. NO per-tensor padding rows are inserted. The total buffer
// is rounded up at its very end to a multiple of 128 (rowwise) or 4 (columnwise)
// in the grouped first dim, matching what the C++ allocator hands out.
//
// Each tensor's compact scales are gathered from the unpadded-prefix rows of
// that tensor's per-tensor padded CPU scale buffer.
namespace {

struct CompactScaleBuffer {
  test::CudaPtr<> ptr;
  size_t numel{0};
};

CompactScaleBuffer gather_compact_grouped_scale(
    const std::vector<std::unique_ptr<test::Tensor>>& tensors,
    size_t M_per_tensor, size_t K_per_tensor, bool rowwise) {
  using namespace test;
  constexpr size_t BLOCK = 32;
  const size_t num_tensors = tensors.size();

  size_t per_tensor_first_unpadded;
  size_t per_tensor_last_padded;
  size_t group_first_align;
  if (rowwise) {
    per_tensor_first_unpadded = M_per_tensor;
    per_tensor_last_padded =
        round_up_to_nearest_multiple(divide_round_up(K_per_tensor, BLOCK), 4);
    group_first_align = 128;
  } else {
    per_tensor_first_unpadded = divide_round_up(M_per_tensor, BLOCK);
    per_tensor_last_padded = round_up_to_nearest_multiple(K_per_tensor, 128);
    group_first_align = 4;
  }

  const size_t per_tensor_compact_numel =
      per_tensor_first_unpadded * per_tensor_last_padded;
  const size_t total_first = round_up_to_nearest_multiple(
      num_tensors * per_tensor_first_unpadded, group_first_align);
  const size_t total_numel = total_first * per_tensor_last_padded;

  std::vector<uint8_t> host_buf(total_numel, 0);
  for (size_t i = 0; i < num_tensors; ++i) {
    tensors[i]->to_cpu();
    const NVTEShape padded_shape = rowwise ? tensors[i]->rowwise_scale_inv_shape()
                                           : tensors[i]->columnwise_scale_inv_shape();
    NVTE_CHECK(padded_shape.data[1] == per_tensor_last_padded,
               "Unexpected per-tensor padded last dim in compact gather.");
    const uint8_t* src = rowwise
        ? tensors[i]->rowwise_cpu_scale_inv_ptr<uint8_t>()
        : tensors[i]->columnwise_cpu_scale_inv_ptr<uint8_t>();
    uint8_t* dst = host_buf.data() + i * per_tensor_compact_numel;
    // Per-tensor padded buffer is row-major (padded_first, padded_last); copy
    // only the first `per_tensor_first_unpadded` rows.
    std::memcpy(dst, src, per_tensor_compact_numel);
  }

  CompactScaleBuffer out;
  out.ptr = cuda_alloc(total_numel);
  NVTE_CHECK_CUDA(cudaMemcpy(out.ptr.get(), host_buf.data(),
                             total_numel, cudaMemcpyHostToDevice));
  out.numel = total_numel;
  return out;
}

}  // namespace

// Tests that grouped_swizzle_for_gemm correctly handles a COMPACT input
// scale_inv buffer (no per-tensor padding rows), producing an output in the
// per-tensor padded layout with padded regions zeroed out. This is the layout
// produced by the grouped MXFP8 quantize kernel; previously the swizzle kernel
// asserted the input matched the per-tensor padded packed size, which broke
// grouped MLP weights with M not a multiple of 128.
void performTestGroupedSwizzleMXFP8CompactInput(const int num_tensors, const size_t M,
                                                const size_t K) {
  using namespace transformer_engine;
  using namespace test;

  std::vector<std::unique_ptr<Tensor>> input_tensors;
  std::vector<std::unique_ptr<Tensor>> output_tensors;
  std::vector<Tensor*> input_ptrs, output_ptrs;
  input_tensors.reserve(num_tensors);
  output_tensors.reserve(num_tensors);
  input_ptrs.reserve(num_tensors);
  output_ptrs.reserve(num_tensors);

  constexpr size_t BLOCK_SIZE = 32;
  const std::vector<size_t> shape{M, K};
  for (int i = 0; i < num_tensors; ++i) {
    auto input = std::make_unique<Tensor>("input_" + std::to_string(i), shape,
                                          DType::kFloat8E4M3, true, true,
                                          NVTE_MXFP8_1D_SCALING);
    auto output = std::make_unique<Tensor>("output_" + std::to_string(i), shape,
                                           DType::kFloat8E4M3, true, true,
                                           NVTE_MXFP8_1D_SCALING);
    fillUniform(input.get());
    fillUniform(output.get());

    // Zero the per-tensor padded regions so the reference (which sees the
    // padded layout) and the kernel (which sees the compact layout but writes
    // zeros into output padding) agree byte-for-byte.
    input->to_cpu();
    const NVTEShape rs = input->rowwise_scale_inv_shape();
    zero_scale_inv_padding(input->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                           rs.data[0], rs.data[1],
                           M, divide_round_up(K, BLOCK_SIZE));
    const NVTEShape cs = input->columnwise_scale_inv_shape();
    zero_scale_inv_padding(input->columnwise_cpu_scale_inv_ptr<uint8_t>(),
                           cs.data[0], cs.data[1],
                           divide_round_up(M, BLOCK_SIZE), K);
    input->from_cpu();

    input_ptrs.push_back(input.get());
    output_ptrs.push_back(output.get());
    input_tensors.emplace_back(std::move(input));
    output_tensors.emplace_back(std::move(output));
  }

  // Build a per-tensor padded grouped output via the standard helper, and a
  // compact-scale grouped input by overriding the scale_inv buffers of a
  // padded grouped input with newly allocated compact buffers.
  GroupedBuffers grouped_input = build_grouped_tensor(input_ptrs, NVTE_MXFP8_1D_SCALING);
  GroupedBuffers grouped_output = build_grouped_tensor(output_ptrs, NVTE_MXFP8_1D_SCALING);

  CompactScaleBuffer compact_row =
      gather_compact_grouped_scale(input_tensors, M, K, /*rowwise=*/true);
  CompactScaleBuffer compact_col =
      gather_compact_grouped_scale(input_tensors, M, K, /*rowwise=*/false);

  grouped_input.scale_inv = std::move(compact_row.ptr);
  grouped_input.columnwise_scale_inv = std::move(compact_col.ptr);
  {
    NVTEShape s = nvte_make_shape(&compact_row.numel, 1);
    NVTEBasicTensor t{grouped_input.scale_inv.get(), kNVTEFloat8E8M0, s};
    nvte_set_grouped_tensor_param(grouped_input.get_handle(),
                                  kNVTEGroupedRowwiseScaleInv, &t, sizeof(t));
  }
  {
    NVTEShape s = nvte_make_shape(&compact_col.numel, 1);
    NVTEBasicTensor t{grouped_input.columnwise_scale_inv.get(), kNVTEFloat8E8M0, s};
    nvte_set_grouped_tensor_param(grouped_input.get_handle(),
                                  kNVTEGroupedColumnwiseScaleInv, &t, sizeof(t));
  }

  const uint8_t input_swizzled = 0;
  nvte_set_grouped_tensor_param(grouped_input.get_handle(),
                                kNVTEGroupedWithGEMMSwizzledScales,
                                &input_swizzled, sizeof(input_swizzled));
  const uint8_t output_swizzled = 1;
  nvte_set_grouped_tensor_param(grouped_output.get_handle(),
                                kNVTEGroupedWithGEMMSwizzledScales,
                                &output_swizzled, sizeof(output_swizzled));

  const NVTEShape row_shape = input_tensors[0]->rowwise_scale_inv_shape();
  const NVTEShape col_shape = input_tensors[0]->columnwise_scale_inv_shape();
  const size_t row_numel = row_shape.data[0] * row_shape.data[1];
  const size_t col_numel = col_shape.data[0] * col_shape.data[1];

  // Memset to a non-zero sentinel so we can detect kernel failures to write
  // padded regions (those must be overwritten with zero by the kernel).
  NVTE_CHECK_CUDA(cudaMemset(grouped_output.scale_inv.get(), 0xCD,
                             num_tensors * row_numel));
  NVTE_CHECK_CUDA(cudaMemset(grouped_output.columnwise_scale_inv.get(), 0xCD,
                             num_tensors * col_numel));

  nvte_swizzle_grouped_scaling_factors(grouped_input.get_handle(),
                                       grouped_output.get_handle(), 0);
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  std::vector<uint8_t> output_row(num_tensors * row_numel);
  std::vector<uint8_t> output_col(num_tensors * col_numel);
  NVTE_CHECK_CUDA(cudaMemcpy(output_row.data(), grouped_output.scale_inv.get(),
                             output_row.size(), cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(output_col.data(),
                             grouped_output.columnwise_scale_inv.get(),
                             output_col.size(), cudaMemcpyDeviceToHost));

  std::vector<uint8_t> ref_row(num_tensors * row_numel);
  std::vector<uint8_t> ref_col(num_tensors * col_numel);
  for (int i = 0; i < num_tensors; ++i) {
    compute_ref_swizzle<128, 4, true>(
        input_tensors[i]->rowwise_cpu_scale_inv_ptr<uint8_t>(),
        ref_row.data() + i * row_numel,
        row_shape.data[0], row_shape.data[1]);
    compute_ref_swizzle<128, 4, false>(
        input_tensors[i]->columnwise_cpu_scale_inv_ptr<uint8_t>(),
        ref_col.data() + i * col_numel,
        col_shape.data[1], col_shape.data[0]);
  }

  compareResults("grouped_swizzle_compact_rowwise", output_row.data(),
                 ref_row.data(), num_tensors * row_numel);
  compareResults("grouped_swizzle_compact_colwise", output_col.data(),
                 ref_col.data(), num_tensors * col_numel);
}

class SwizzleGroupedCompactInputTestSuite
    : public ::testing::TestWithParam<std::tuple<int, size_t, size_t>> {};

TEST_P(SwizzleGroupedCompactInputTestSuite, TestGroupedSwizzleMXFP8CompactInput) {
  const auto num_tensors = std::get<0>(GetParam());
  const auto M = std::get<1>(GetParam());
  const auto K = std::get<2>(GetParam());
  performTestGroupedSwizzleMXFP8CompactInput(num_tensors, M, K);
}

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  SwizzleGroupedCompactInputTestSuite,
  ::testing::Values(
    // Aligned M and K. Per-tensor compact stride == per-tensor padded stride,
    // so the kernel may use either layout; serves as a sanity check that the
    // compact-input plumbing doesn't regress aligned shapes.
    std::make_tuple(3, 256, 256),
    std::make_tuple(4, 128, 128),
    // M NOT divisible by 128 (the original-bug case): per-tensor compact stride
    // shrinks vs padded. We pick (num_tensors, M) so that BOTH
    //   round_up(N * M, 128) != N * round_up(M, 128)               (rowwise)
    //   round_up(N * DIVUP(M,32), 4) != N * round_up(DIVUP(M,32),4) (colwise)
    // i.e. compact_total != padded_total on either axis, so the kernel
    // unambiguously detects the compact layout.
    std::make_tuple(4, 200, 256),
    std::make_tuple(4, 65, 256),
    std::make_tuple(2, 2880, 2880),  // shape from the originally failing workload
    // K not divisible by 128 (DIVUP(K,32) padded up to a multiple of 4).
    std::make_tuple(3, 256, 160),
    std::make_tuple(2, 256, 96),
    // Neither M nor K aligned.
    std::make_tuple(4, 200, 160),
    std::make_tuple(4, 33, 64),
    std::make_tuple(2, 1, 32),
    // num_tensors * M not aligned to 128 -> exercises trailing alignment slack
    // at the end of the compact rowwise buffer.
    std::make_tuple(3, 64, 128),
    std::make_tuple(5, 33, 96)
  ),
  [](const testing::TestParamInfo<SwizzleGroupedCompactInputTestSuite::ParamType>& info) {
    return "n" + std::to_string(std::get<0>(info.param)) +
           "_M" + std::to_string(std::get<1>(info.param)) +
           "_K" + std::to_string(std::get<2>(info.param));
  }
);

class UnswizzleGroupedTestSuite
    : public ::testing::TestWithParam<std::tuple<int, size_t, size_t>> {};

TEST_P(UnswizzleGroupedTestSuite, TestGroupedUnswizzleMXFP8) {
  const auto num_tensors = std::get<0>(GetParam());
  const auto M = std::get<1>(GetParam());
  const auto K = std::get<2>(GetParam());
  performTestGroupedUnswizzleMXFP8(num_tensors, M, K);
}

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  UnswizzleGroupedTestSuite,
  ::testing::Values(
    std::make_tuple(3, 256, 256),
    std::make_tuple(4, 128, 128),
    std::make_tuple(3, 200, 256),
    std::make_tuple(2, 65, 256),
    std::make_tuple(3, 256, 160),
    std::make_tuple(2, 256, 96),
    std::make_tuple(3, 200, 160),
    std::make_tuple(4, 33, 64),
    std::make_tuple(2, 1, 32)
  ),
  [](const testing::TestParamInfo<UnswizzleGroupedTestSuite::ParamType>& info) {
    return "n" + std::to_string(std::get<0>(info.param)) +
           "_M" + std::to_string(std::get<1>(info.param)) +
           "_K" + std::to_string(std::get<2>(info.param));
  }
);

class SwizzleUnswizzleGroupedRoundtripTestSuite
    : public ::testing::TestWithParam<std::tuple<int, size_t, size_t>> {};

TEST_P(SwizzleUnswizzleGroupedRoundtripTestSuite, TestGroupedSwizzleUnswizzleRoundtrip) {
  const auto num_tensors = std::get<0>(GetParam());
  const auto M = std::get<1>(GetParam());
  const auto K = std::get<2>(GetParam());
  performTestGroupedSwizzleUnswizzleRoundtrip(num_tensors, M, K);
}

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  SwizzleUnswizzleGroupedRoundtripTestSuite,
  ::testing::Values(
    std::make_tuple(3, 256, 256),
    std::make_tuple(4, 128, 128),
    std::make_tuple(3, 200, 256),
    std::make_tuple(2, 65, 256),
    std::make_tuple(3, 256, 160),
    std::make_tuple(2, 256, 96),
    std::make_tuple(3, 200, 160),
    std::make_tuple(4, 33, 64),
    std::make_tuple(2, 1, 32)
  ),
  [](const testing::TestParamInfo<SwizzleUnswizzleGroupedRoundtripTestSuite::ParamType>& info) {
    return "n" + std::to_string(std::get<0>(info.param)) +
           "_M" + std::to_string(std::get<1>(info.param)) +
           "_K" + std::to_string(std::get<2>(info.param));
  }
);

namespace {

std::vector<std::pair<int, int>> num_tiles = {
  {1, 1},
  {1, 132},
  {132, 1},
  {65, 256},
  {65, 257},
  {65, 258},
  {65, 259},
  // Additional narrow-path coverage: narrow_k (row) when num_tiles_K < 32,
  // narrow_m (col) when num_tiles_M < 32.
  {1, 4},     // narrow_k with 4 K-tiles
  {1, 8},     // narrow_k with 8 K-tiles
  {4, 1},     // narrow_m with 4 M-tiles
  {8, 1},     // narrow_m with 8 M-tiles
  {31, 1},    // narrow_m at boundary (31 < TB_DIM=32)
  {1, 31},    // narrow_k at boundary (31 < TB_DIM=32)
};

// Raw {M, K} data shapes for unswizzle tests. Includes aligned cases (scale dims
// already multiples of 128 and 4) and padded cases where M or K/32 are not yet
// aligned, forcing the compact scale_inv to carry a padded tail.
// All K values must be multiples of 32 (MXFP8 block size).
std::vector<std::pair<size_t, size_t>> unswizzle_data_shapes = {
  // Aligned: scale dims are already multiples of 128 and 4
  {128, 128},
  {128, 16896},   // K = 132 * 128, large K
  {16896, 128},   // M = 132 * 128, large M
  // M-padding only: M not a multiple of 128 (scale-M needs padding to 256)
  {160, 128},
  // scale-K padding only: K/32 = 3, padded to 4
  {128, 96},
  // Both M and scale-K need padding
  {160, 96},
  {16896, 16896},
};

std::vector<std::pair<bool, bool>> scaling_mode = {
  {true, false},
  {false, true}
};

std::vector<bool> transa = {true, false};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  SwizzleTestSuite,
  ::testing::Combine(
    ::testing::ValuesIn(num_tiles),
    ::testing::ValuesIn(scaling_mode),
    ::testing::ValuesIn(transa)
  ),
  [](const testing::TestParamInfo<SwizzleTestSuite::ParamType>& info) {
    std::string name = "ntiles" +
      std::to_string(std::get<0>(info.param).first) + "X" +
      std::to_string(std::get<0>(info.param).second) + "smode" +
      std::to_string(std::get<1>(info.param).first) + "X"+
      std::to_string(std::get<1>(info.param).second) + "trans" +
      std::to_string(std::get<2>(info.param));
    return name;
    });

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  UnswizzleTestSuite,
  ::testing::Combine(
    ::testing::ValuesIn(unswizzle_data_shapes),
    ::testing::ValuesIn(scaling_mode),
    ::testing::ValuesIn(transa)
  ),
  [](const testing::TestParamInfo<UnswizzleTestSuite::ParamType>& info) {
    std::string name = "MK" +
      std::to_string(std::get<0>(info.param).first) + "X" +
      std::to_string(std::get<0>(info.param).second) + "smode" +
      std::to_string(std::get<1>(info.param).first) + "X"+
      std::to_string(std::get<1>(info.param).second) + "trans" +
      std::to_string(std::get<2>(info.param));
    return name;
    });

void performTestSwizzleUnswizzleRoundtrip(const size_t M, const size_t K, bool rowwise, bool columnwise, const bool transa) {
  using namespace test;

  int SF_MODE_X, SF_MODE_Y;
  if (rowwise) {
    SF_MODE_X = 1;
    SF_MODE_Y = 32;
  }
  if (columnwise) {
    SF_MODE_X = 32;
    SF_MODE_Y = 1;
  }

  if (!rowwise && !columnwise) {
    GTEST_SKIP() << "TEST SKIPPED, Either rowwise or columnwise scaling mode must be true.";
  }
  if (rowwise && columnwise){
    GTEST_SKIP() << "TEST SKIPPED, The scaling mode " + std::to_string(SF_MODE_X) + "x" +
      std::to_string(SF_MODE_Y) + " is not implemented.";
  }

  DType dtype = DType::kFloat8E4M3;

  const auto data_shape = transa ? std::vector<size_t>{M, K} : std::vector<size_t>{K, M};
  const size_t logical_dim0 = data_shape[0] / SF_MODE_X;
  const size_t logical_dim1 = data_shape[1] / SF_MODE_Y;

  Tensor input("input", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  Tensor swizzled("swizzled", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  swizzled.set_with_gemm_swizzled_scales(true);
  Tensor output("output", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);

  fillUniform(&input);

  // fillUniform fills all scale_inv entries including the padded region with random bytes.
  // After swizzle, the swizzle kernel zeroes padded positions in the swizzled output, so
  // after unswizzle those positions come back as zero in the compact output. Zero them in
  // the input now so the full-buffer comparison is valid.
  const auto padded_scale_shape = rowwise
    ? input.rowwise_scale_inv_shape()
    : input.columnwise_scale_inv_shape();
  const size_t padded_dim0 = padded_scale_shape.data[0];
  const size_t padded_dim1 = padded_scale_shape.data[1];

  if (padded_dim0 != logical_dim0 || padded_dim1 != logical_dim1) {
    auto* scale_ptr = rowwise
      ? input.rowwise_cpu_scale_inv_ptr<uint8_t>()
      : input.columnwise_cpu_scale_inv_ptr<uint8_t>();
    for (size_t r = 0; r < padded_dim0; r++) {
      for (size_t c = 0; c < padded_dim1; c++) {
        if (r >= logical_dim0 || c >= logical_dim1) {
          scale_ptr[r * padded_dim1 + c] = 0;
        }
      }
    }
    input.from_cpu();
  }

  nvte_swizzle_scaling_factors(input.data(), swizzled.data(), 0);
  nvte_unswizzle_scaling_factors(swizzled.data(), output.data(), 0);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  input.to_cpu();
  output.to_cpu();
  if (rowwise) {
    compareResults("roundtrip_rowwise", output.rowwise_cpu_scale_inv_ptr<uint8_t>(),
                   input.rowwise_cpu_scale_inv_ptr<uint8_t>(), padded_dim0 * padded_dim1);
  } else {
    compareResults("roundtrip_columnwise", output.columnwise_cpu_scale_inv_ptr<uint8_t>(),
                   input.columnwise_cpu_scale_inv_ptr<uint8_t>(), padded_dim0 * padded_dim1);
  }
}

class SwizzleUnswizzleRoundtripTestSuite : public ::testing::TestWithParam<std::tuple<std::pair<size_t, size_t>, std::pair<bool, bool>, bool>> {};

TEST_P(SwizzleUnswizzleRoundtripTestSuite, TestSwizzleUnswizzleRoundtrip) {
  using namespace transformer_engine;
  using namespace test;

  const auto data_shape = std::get<0>(GetParam());
  const auto scaling_mode = std::get<1>(GetParam());
  const auto transa = std::get<2>(GetParam());

  performTestSwizzleUnswizzleRoundtrip(data_shape.first, data_shape.second,
                                       scaling_mode.first, scaling_mode.second,
                                       transa);
}

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  SwizzleUnswizzleRoundtripTestSuite,
  ::testing::Combine(
    ::testing::ValuesIn(unswizzle_data_shapes),
    ::testing::ValuesIn(scaling_mode),
    ::testing::ValuesIn(transa)
  ),
  [](const testing::TestParamInfo<SwizzleUnswizzleRoundtripTestSuite::ParamType>& info) {
    std::string name = "roundtrip_MK" +
      std::to_string(std::get<0>(info.param).first) + "X" +
      std::to_string(std::get<0>(info.param).second) + "smode" +
      std::to_string(std::get<1>(info.param).first) + "X"+
      std::to_string(std::get<1>(info.param).second) + "trans" +
      std::to_string(std::get<2>(info.param));
    return name;
    });
