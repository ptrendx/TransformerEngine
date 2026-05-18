/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/swizzle.h>

#include <algorithm>
#include <cassert>
#include <mutex>
#include <numeric>
#include <type_traits>
#include <vector>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/logging.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
namespace {

constexpr int MXFP8_BLOCK_SIZE = 32;
constexpr int NVFP4_BLOCK_SIZE = 16;

int get_max_dynamic_smem() {
  static int max_smem = -1;
  if (max_smem < 0) {
    int device;
    NVTE_CHECK_CUDA(cudaGetDevice(&device));
    NVTE_CHECK_CUDA(
        cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  }
  return max_smem;
}

constexpr __device__ __host__ int TB_DIM = 32;
constexpr __device__ __host__ int NEW_SF_TILE_DIM_K = 16;
constexpr __device__ __host__ int N_SF_PER_TD_PER_TILE = 4;
constexpr int ROW_COALESCED_THREADS = 256;
constexpr int ROW_COALESCED_MIN_K = 32;
// Keep row-coalesced CTAs small enough to expose more independent memory work
// on Blackwell. Large per-CTA M batching raised shared-memory residency and
// serialized independent swizzle tiles in the benchmarked large cases.
constexpr int ROW_COALESCED_MAX_M_TILES_PER_BLOCK = 4;
constexpr int ROW_COALESCED_TARGET_SMEM_BYTES = 12 * 1024;
constexpr int COL_COALESCED_THREADS = 256;
constexpr int COL_COALESCED_K_TILES_PER_BLOCK = 32;
constexpr int COL_WARP_TILE_WARPS = 8;
constexpr int COL_WARP_TILE_THREADS = COL_WARP_TILE_WARPS * 32;

// output is in ~K-major interleaved blocks
constexpr __device__ __host__ int NEW_SF_TILE_DIM_K_I32 = NEW_SF_TILE_DIM_K / 4;
constexpr __device__ __host__ int NEW_SF_TILE_DIM_M_I32 = 32;

template <typename Kernel>
void set_dynamic_smem_if_needed(Kernel kernel_fn, int slm_size, int& cached_smem_size) {
  if (cached_smem_size < slm_size) {
    NVTE_CHECK_CUDA(
        cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
    cached_smem_size = slm_size;
  }
}

__device__ __forceinline__ void store_global_cs(int4* ptr, const int4 value) {
  asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x),
               "r"(value.y), "r"(value.z), "r"(value.w)
               : "memory");
}

__device__ __forceinline__ void store_global_cs(uint4* ptr, const uint4 value) {
  asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x),
               "r"(value.y), "r"(value.z), "r"(value.w)
               : "memory");
}

__device__ __forceinline__ void divmod_tile_col_v4(const int tile_idx, const int row_v4,
                                                   const bool row_v4_is_pow2,
                                                   const int row_v4_log2, int* row,
                                                   int* col_v4) {
  if (row_v4_is_pow2) {
    *row = tile_idx >> row_v4_log2;
    *col_v4 = tile_idx & (row_v4 - 1);
  } else {
    *row = tile_idx / row_v4;
    *col_v4 = tile_idx - *row * row_v4;
  }
}

__host__ __device__ constexpr int row_coalesced_smem_pad_i32(const int k_i32) {
  // K_i32==8/16 correspond to 32/64 scale columns. A +1 pitch causes repeated
  // shared-store bank hits for the row-major loading warp; +3 keeps shared reads
  // conflict-free while spreading the stores across more banks.
  return (k_i32 == 8 || k_i32 == 16) ? 3 : 1;
}

__host__ __device__ constexpr int row_coalesced_smem_stride_i32(const int k_i32) {
  return k_i32 + row_coalesced_smem_pad_i32(k_i32);
}

__device__ __forceinline__ void transpose_4x4_bytes(const int32_t a, const int32_t b,
                                                    const int32_t c, const int32_t d,
                                                    int32_t* out0, int32_t* out1,
                                                    int32_t* out2, int32_t* out3) {
  const int32_t ab01 = __byte_perm(a, b, 0x5140);
  const int32_t cd01 = __byte_perm(c, d, 0x5140);
  const int32_t ab23 = __byte_perm(a, b, 0x7362);
  const int32_t cd23 = __byte_perm(c, d, 0x7362);
  *out0 = __byte_perm(ab01, cd01, 0x5410);
  *out1 = __byte_perm(ab01, cd01, 0x7632);
  *out2 = __byte_perm(ab23, cd23, 0x5410);
  *out3 = __byte_perm(ab23, cd23, 0x7632);
}

template <typename LType>
__device__ inline void regs_shuffle_with_bit_shifts(LType* regs_vec) {
  // inp, 4-byte chunks [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]
  // out, swapping byte to form new 4-byte chunks [0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15]

  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int kVectorSize = N_SF_PER_TD_PER_TILE * N_TILE_PER_TD;
  int32_t new_regs[kVectorSize];
  int32_t* regs = reinterpret_cast<int32_t*>(regs_vec);

  if constexpr (N_TILE_PER_TD == 4) {
#pragma unroll
    for (int i = 0; i < N_TILE_PER_TD; i++) {
      transpose_4x4_bytes(regs[i], regs[i + N_TILE_PER_TD], regs[i + 2 * N_TILE_PER_TD],
                          regs[i + 3 * N_TILE_PER_TD],
                          &new_regs[i * N_SF_PER_TD_PER_TILE + 0],
                          &new_regs[i * N_SF_PER_TD_PER_TILE + 1],
                          &new_regs[i * N_SF_PER_TD_PER_TILE + 2],
                          &new_regs[i * N_SF_PER_TD_PER_TILE + 3]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < N_TILE_PER_TD; i++) {
#pragma unroll
      for (int j = 0; j < N_SF_PER_TD_PER_TILE; j++) {
        new_regs[i * N_SF_PER_TD_PER_TILE + j] =
            (((regs[i + 0 * N_TILE_PER_TD] >> 8 * j) & 0xFF)) |
            (((regs[i + 1 * N_TILE_PER_TD] >> 8 * j) & 0xFF) << 8) |
            (((regs[i + 2 * N_TILE_PER_TD] >> 8 * j) & 0xFF) << 16) |
            (((regs[i + 3 * N_TILE_PER_TD] >> 8 * j) & 0xFF) << 24);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < kVectorSize; i++) regs[i] = new_regs[i];
}

template <typename LType>
__device__ inline void regs_unshuffle_with_bit_shifts(LType* regs_vec) {
  // Inverse of regs_shuffle_with_bit_shifts
  // inp, 4-byte chunks [0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15]
  // out, swapping byte to form new 4-byte chunks [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]

  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int kVectorSize = N_SF_PER_TD_PER_TILE * N_TILE_PER_TD;
  int32_t new_regs[kVectorSize];
  int32_t* regs = reinterpret_cast<int32_t*>(regs_vec);

  if constexpr (N_TILE_PER_TD == 4) {
#pragma unroll
    for (int i = 0; i < N_TILE_PER_TD; i++) {
      transpose_4x4_bytes(regs[i * N_SF_PER_TD_PER_TILE + 0],
                          regs[i * N_SF_PER_TD_PER_TILE + 1],
                          regs[i * N_SF_PER_TD_PER_TILE + 2],
                          regs[i * N_SF_PER_TD_PER_TILE + 3], &new_regs[i],
                          &new_regs[i + N_TILE_PER_TD], &new_regs[i + 2 * N_TILE_PER_TD],
                          &new_regs[i + 3 * N_TILE_PER_TD]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < N_TILE_PER_TD; i++) {
#pragma unroll
      for (int j = 0; j < N_SF_PER_TD_PER_TILE; j++) {
        new_regs[i + j * N_TILE_PER_TD] =
            ((regs[i * N_SF_PER_TD_PER_TILE + 0] >> 8 * j) & 0xFF) |
            (((regs[i * N_SF_PER_TD_PER_TILE + 1] >> 8 * j) & 0xFF) << 8) |
            (((regs[i * N_SF_PER_TD_PER_TILE + 2] >> 8 * j) & 0xFF) << 16) |
            (((regs[i * N_SF_PER_TD_PER_TILE + 3] >> 8 * j) & 0xFF) << 24);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < kVectorSize; i++) regs[i] = new_regs[i];
}

// IS_PADDED_K / IS_PADDED_M select the boundary-block specialization at compile
// time so the inner load loop avoids the per-iteration runtime checks. The
// caller computes the runtime predicates from blockIdx/gridDim once per block
// (uniform across the block) and dispatches to the right specialization.
template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K, bool IS_PADDED_K, bool IS_PADDED_M>
__device__ void swizzle_col_scaling_kernel_impl(const void* input, void* output, const int M,
                                                const int K, const int original_M,
                                                const int original_K, const int bid_x,
                                                const int bid_y, const int grid_dim_x,
                                                const int grid_dim_y) {
  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int N_SF_PER_TD = N_TILE_PER_TD * N_SF_PER_TD_PER_TILE;
  constexpr int SF_TILE_SIZE_I32 = SF_TILE_DIM_M * SF_TILE_DIM_K / 4;

  // input is in M-major
  constexpr int SF_TILE_DIM_M_I32 = SF_TILE_DIM_M / 4;
  constexpr int SF_TILE_DIM_K_I32 = SF_TILE_DIM_K;

  const int M_i32 = M / 4;
  const int K_i32 = K;

  int m_tiles_in_tb = N_TILE_PER_TD;
  int k_tiles_in_tb = TB_DIM;
  if (bid_x == grid_dim_x - 1) {
    k_tiles_in_tb = (K_i32 / SF_TILE_DIM_K_I32 - 1) % k_tiles_in_tb + 1;
  }
  if (bid_y == grid_dim_y - 1) {
    m_tiles_in_tb = (M_i32 / SF_TILE_DIM_M_I32 - 1) % m_tiles_in_tb + 1;
  }

  const int input_offset =
      bid_x * TB_DIM * SF_TILE_DIM_K_I32 * M_i32 + bid_y * N_TILE_PER_TD * SF_TILE_DIM_M_I32;
  const int32_t* input_i32 = reinterpret_cast<const int32_t*>(input) + input_offset;
  int32_t* output_i32[N_TILE_PER_TD];
#pragma unroll
  for (int i = 0; i < m_tiles_in_tb; i++) {
    output_i32[i] = reinterpret_cast<int32_t*>(output) + bid_x * TB_DIM * SF_TILE_SIZE_I32 +
                    (bid_y * N_TILE_PER_TD + i) * SF_TILE_DIM_M_I32 * K_i32;
  }
  extern __shared__ int slm[];

  // load, global -> regs
  // Each register read for a given i is along the M direction at K-coord
  // (bid_x * TB_DIM * SF_TILE_DIM_K + threadIdx.y * SF_TILE_DIM_K + i). When that
  // K-coord is past original_K, the entire register is out of the per-tensor data
  // region (which may be the unpadded compact extent), so we must NOT issue the
  // __ldg there -- it could read past the per-tensor buffer (and, for the last
  // tensor in a grouped allocation, past the end of the allocation entirely).
  LType regs_vec[N_SF_PER_TD_PER_TILE];
  if (threadIdx.x * N_TILE_PER_TD < m_tiles_in_tb * SF_TILE_DIM_M_I32 &&
      threadIdx.y < k_tiles_in_tb) {
    const int k_base = bid_x * TB_DIM * SF_TILE_DIM_K + threadIdx.y * SF_TILE_DIM_K;
#pragma unroll
    for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
      const int thread_offset =
          (threadIdx.y * SF_TILE_DIM_K_I32 + i) * M_i32 + threadIdx.x * N_TILE_PER_TD;
      const int k_coord = k_base + i;
      if constexpr (IS_PADDED_K) {
        if (k_coord >= original_K) {
          // Entire register is past original_K: zero directly without loading.
          uint8_t* zero_bytes = reinterpret_cast<uint8_t*>(regs_vec + i);
#pragma unroll
          for (int j = 0; j < static_cast<int>(sizeof(LType)); j++) zero_bytes[j] = 0;
          continue;
        }
      }
      regs_vec[i] = __ldg(reinterpret_cast<const LType*>(input_i32 + thread_offset));
      // Per-byte M masking is still needed when only part of the register is past
      // original_M (i.e. K-coord is in range but the M position spans the boundary).
      if constexpr (IS_PADDED_M) {
        for (int j = 0; j < N_TILE_PER_TD * sizeof(int); j++) {
          const int index = (input_offset + thread_offset) * sizeof(int) + j;
          if (index % M >= original_M) {
            reinterpret_cast<uint8_t*>(regs_vec + i)[j] = 0;
          }
        }
      }
    }

    // local shuffle
    regs_shuffle_with_bit_shifts(regs_vec);

    // store, regs -> shared
    int tM = threadIdx.x * N_SF_PER_TD;
    int* slm_tile = slm + (threadIdx.y * SF_TILE_SIZE_I32 +
                           tM / SF_TILE_DIM_M * k_tiles_in_tb * SF_TILE_SIZE_I32);
#pragma unroll
    for (int i = 0; i < N_SF_PER_TD; i++) {
      /* TODO rotate_i */
      slm_tile[(tM % SF_TILE_DIM_M) / NEW_SF_TILE_DIM_M_I32 +
               ((tM + i) % NEW_SF_TILE_DIM_M_I32) * NEW_SF_TILE_DIM_K_I32] =
          reinterpret_cast<int*>(regs_vec)[i];
    }
  }
  __syncthreads();

  // store, shared -> global
  int linear_id = threadIdx.y * blockDim.x + threadIdx.x;
#pragma unroll
  for (int i = 0; i < m_tiles_in_tb; i++) {
    __align__(16) int4* output_v4i = reinterpret_cast<int4*>(output_i32[i]);
    __align__(16) int4* slm_v4i =
        reinterpret_cast<int4*>(slm + i * k_tiles_in_tb * SF_TILE_SIZE_I32);
#pragma unroll
    for (int j = linear_id; j < SF_TILE_SIZE_I32 * k_tiles_in_tb / 4;
         j += blockDim.x * blockDim.y) {
      output_v4i[j] = slm_v4i[j];
    }
  }
}

// Dispatch helper: pick the right (IS_PADDED_K, IS_PADDED_M) col-scaling impl
// specialization at runtime based on the per-block padding predicates. The
// branching here is uniform across all threads in the block, so the indirect
// path each block takes still inlines cleanly.
template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ __forceinline__ void dispatch_swizzle_col_scaling_kernel_impl(
    const void* input, void* output, const int M, const int K, const int original_M,
    const int original_K, const int bid_x, const int bid_y, const int grid_dim_x,
    const int grid_dim_y, const bool padding_k, const bool padding_m) {
  if (padding_k && padding_m) {
    swizzle_col_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K, /*IS_PADDED_K=*/true,
                                    /*IS_PADDED_M=*/true>(
        input, output, M, K, original_M, original_K, bid_x, bid_y, grid_dim_x, grid_dim_y);
  } else if (padding_k) {
    swizzle_col_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K, /*IS_PADDED_K=*/true,
                                    /*IS_PADDED_M=*/false>(
        input, output, M, K, original_M, original_K, bid_x, bid_y, grid_dim_x, grid_dim_y);
  } else if (padding_m) {
    swizzle_col_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K, /*IS_PADDED_K=*/false,
                                    /*IS_PADDED_M=*/true>(
        input, output, M, K, original_M, original_K, bid_x, bid_y, grid_dim_x, grid_dim_y);
  } else {
    swizzle_col_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K, /*IS_PADDED_K=*/false,
                                    /*IS_PADDED_M=*/false>(
        input, output, M, K, original_M, original_K, bid_x, bid_y, grid_dim_x, grid_dim_y);
  }
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    swizzle_col_scaling_kernel(const void* input, void* output, const int M, const int K,
                               const int original_M, const int original_K) {
  const bool padding_m = (blockIdx.y == gridDim.y - 1) && (original_M < M);
  const bool padding_k = (blockIdx.x == gridDim.x - 1) && (original_K < K);
  dispatch_swizzle_col_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, blockIdx.x, blockIdx.y, gridDim.x, gridDim.y,
      padding_k, padding_m);
}

template <typename LType>
__device__ inline void regs_shuffle(LType* regs_vec) {
  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  if constexpr (N_TILE_PER_TD == 1) return;

  constexpr int kVectorSize = N_SF_PER_TD_PER_TILE * N_TILE_PER_TD;
  int32_t tmp[kVectorSize];
  int32_t* ptr = reinterpret_cast<int32_t*>(regs_vec);
#pragma unroll
  for (int i = 0; i < kVectorSize; i++)
    tmp[i % N_TILE_PER_TD * N_SF_PER_TD_PER_TILE + i / N_TILE_PER_TD] = ptr[i];

#pragma unroll
  for (int i = 0; i < kVectorSize; i++) ptr[i] = tmp[i];
}

// Inverse of regs_shuffle.
template <typename LType>
__device__ inline void regs_unshuffle(LType* regs_vec) {
  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  if constexpr (N_TILE_PER_TD == 1) return;

  constexpr int kVectorSize = N_SF_PER_TD_PER_TILE * N_TILE_PER_TD;
  int32_t tmp[kVectorSize];
  int32_t* ptr = reinterpret_cast<int32_t*>(regs_vec);
#pragma unroll
  for (int i = 0; i < kVectorSize; i++)
    tmp[i % N_SF_PER_TD_PER_TILE * N_TILE_PER_TD + i / N_SF_PER_TD_PER_TILE] = ptr[i];

#pragma unroll
  for (int i = 0; i < kVectorSize; i++) ptr[i] = tmp[i];
}

// IS_PADDED_K / IS_PADDED_M select the boundary-block specialization at compile
// time so the inner load loop avoids the per-iteration runtime checks. The
// caller computes the runtime predicates from blockIdx/gridDim once per block
// (uniform across the block) and dispatches to the right specialization.
template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K, bool IS_PADDED_K, bool IS_PADDED_M>
__device__ void swizzle_row_scaling_kernel_impl(const void* input, void* output, const int M,
                                                const int K, const int original_M,
                                                const int original_K, const int bid_x,
                                                const int bid_y, const int grid_dim_x,
                                                const int grid_dim_y) {
  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int N_TILES_IN_TB = TB_DIM * N_TILE_PER_TD;

  // input is in K-major
  constexpr int SF_TILE_SIZE_I32 = SF_TILE_DIM_M * SF_TILE_DIM_K / 4;
  constexpr int SF_TILE_DIM_M_I32 = SF_TILE_DIM_M;

  int n_tiles_in_tb = N_TILES_IN_TB;
  const int K_i32 = K / 4;
  if (bid_x == grid_dim_x - 1) {
    n_tiles_in_tb = (K_i32 - 1) % N_TILES_IN_TB + 1;
  }

  const int input_offset = bid_y * SF_TILE_DIM_M_I32 * K_i32 + bid_x * N_TILES_IN_TB;
  const int* input_i32 = reinterpret_cast<const int*>(input) + input_offset;
  int* output_i32 = reinterpret_cast<int*>(output) + bid_y * SF_TILE_DIM_M_I32 * K_i32 +
                    bid_x * N_TILES_IN_TB * SF_TILE_SIZE_I32;

  extern __shared__ int4 slm_v4i[];

  // load, global -> regs
  // Each register read for a given i is along the K direction at row
  // (bid_y * SF_TILE_DIM_M + i * TB_DIM + threadIdx.y). When that row is past
  // original_M, the entire register is out of the per-tensor data region (which
  // may be the unpadded compact extent), so we must NOT issue the __ldg there --
  // it could read past the per-tensor buffer (and, for the last tensor in a
  // grouped allocation, past the end of the allocation entirely).
  LType regs_vec[N_SF_PER_TD_PER_TILE];
  if (threadIdx.x * N_TILE_PER_TD < n_tiles_in_tb) {
#pragma unroll
    for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
      const int row = bid_y * SF_TILE_DIM_M + i * TB_DIM + threadIdx.y;
      const int thread_offset = (i * TB_DIM + threadIdx.y) * K_i32 + threadIdx.x * N_TILE_PER_TD;
      if constexpr (IS_PADDED_M) {
        if (row >= original_M) {
          // Entire register is past original_M: zero directly without loading.
          uint8_t* zero_bytes = reinterpret_cast<uint8_t*>(regs_vec + i);
#pragma unroll
          for (int j = 0; j < static_cast<int>(sizeof(LType)); j++) zero_bytes[j] = 0;
          continue;
        }
      }
      regs_vec[i] = __ldg(reinterpret_cast<const LType*>(input_i32 + thread_offset));
      // Per-byte K masking is still needed when only part of the register is past
      // original_K (i.e. row is in range but the K position spans the boundary).
      if constexpr (IS_PADDED_K) {
#pragma unroll
        for (int j = 0; j < N_TILE_PER_TD * sizeof(int); j++) {
          const int index = (input_offset + thread_offset) * sizeof(int) + j;
          if (index % K >= original_K) {
            reinterpret_cast<uint8_t*>(regs_vec + i)[j] = 0;
          }
        }
      }
    }

    // shuffle regs
    regs_shuffle<LType>(regs_vec);

// store, regs -> shared
#pragma unroll
    for (int i = 0; i < N_TILE_PER_TD; i++) {
      /* TODO rotate i */
      slm_v4i[(threadIdx.x * N_TILE_PER_TD + i) * SF_TILE_SIZE_I32 / 4 + threadIdx.y] =
          reinterpret_cast<int4*>(regs_vec)[i];
    }
  }
  __syncthreads();

  // store, shared -> global
  int linear_id = threadIdx.y * blockDim.x + threadIdx.x;
  __align__(16) int4* output_v4i = reinterpret_cast<int4*>(output_i32);
#pragma unroll
  for (int i = linear_id; i < SF_TILE_SIZE_I32 * n_tiles_in_tb / 4; i += blockDim.x * blockDim.y) {
    output_v4i[i] = slm_v4i[i];
  }
}

// Dispatch helper: pick the right (IS_PADDED_K, IS_PADDED_M) row-scaling impl
// specialization at runtime based on the per-block padding predicates. The
// branching here is uniform across all threads in the block, so the indirect
// path each block takes still inlines cleanly.
template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ __forceinline__ void dispatch_swizzle_row_scaling_kernel_impl(
    const void* input, void* output, const int M, const int K, const int original_M,
    const int original_K, const int bid_x, const int bid_y, const int grid_dim_x,
    const int grid_dim_y, const bool padding_k, const bool padding_m) {
  if (padding_k && padding_m) {
    swizzle_row_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K, /*IS_PADDED_K=*/true,
                                    /*IS_PADDED_M=*/true>(
        input, output, M, K, original_M, original_K, bid_x, bid_y, grid_dim_x, grid_dim_y);
  } else if (padding_k) {
    swizzle_row_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K, /*IS_PADDED_K=*/true,
                                    /*IS_PADDED_M=*/false>(
        input, output, M, K, original_M, original_K, bid_x, bid_y, grid_dim_x, grid_dim_y);
  } else if (padding_m) {
    swizzle_row_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K, /*IS_PADDED_K=*/false,
                                    /*IS_PADDED_M=*/true>(
        input, output, M, K, original_M, original_K, bid_x, bid_y, grid_dim_x, grid_dim_y);
  } else {
    swizzle_row_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K, /*IS_PADDED_K=*/false,
                                    /*IS_PADDED_M=*/false>(
        input, output, M, K, original_M, original_K, bid_x, bid_y, grid_dim_x, grid_dim_y);
  }
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    swizzle_row_scaling_kernel(const void* input, void* output, const int M, const int K,
                               const int original_M, const int original_K) {
  const bool padding_m = (blockIdx.y == gridDim.y - 1) && (original_M < M);
  const bool padding_k = (blockIdx.x == gridDim.x - 1) && (original_K < K);
  dispatch_swizzle_row_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, blockIdx.x, blockIdx.y, gridDim.x, gridDim.y,
      padding_k, padding_m);
}

// Narrow-K specialization for row scaling swizzle.
// When K is small (num_tiles_k < TB_DIM), the standard kernel wastes threadIdx.x
// because there aren't enough K-tiles to distribute across threads.
// This kernel repurposes the thread dimensions: threadIdx.x iterates rows within
// an M-tile, threadIdx.y indexes M-tiles within the block, processing TB_DIM
// M-tiles per block with full thread utilization.
template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ void swizzle_row_scaling_narrow_k_kernel_impl(const void* input, void* output,
                                                         const int M, const int K,
                                                         const int original_M, const int original_K,
                                                         const int bid, const int grid_dim) {
  constexpr int SF_TILE_SIZE_I32 = SF_TILE_DIM_M * SF_TILE_DIM_K / 4;
  const int K_i32 = K / 4;
  const int num_tiles_m = M / SF_TILE_DIM_M;

  const int m_tile = bid * blockDim.y + threadIdx.y;
  const bool active = (m_tile < num_tiles_m);

  extern __shared__ int4 slm_v4i[];
  const int slm_tile_v4i = K_i32 * (SF_TILE_SIZE_I32 / 4);

  if (active) {
    const bool padding_m = (m_tile == num_tiles_m - 1) && (original_M < M);
    const bool padding_k = (original_K < K);

    int4* my_slm = slm_v4i + threadIdx.y * slm_tile_v4i;

    for (int k = 0; k < K_i32; k++) {
      const int input_base = m_tile * SF_TILE_DIM_M * K_i32 + k;
      const int* input_i32 = reinterpret_cast<const int*>(input) + input_base;

      int regs[N_SF_PER_TD_PER_TILE];
#pragma unroll
      for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
        const int row = i * TB_DIM + threadIdx.x;
        const bool row_is_padding = padding_m && (m_tile * SF_TILE_DIM_M + row >= original_M);
        if (row_is_padding) {
          regs[i] = 0;
          continue;
        }
        regs[i] = __ldg(input_i32 + row * K_i32);
        if (padding_m || padding_k) {
          for (int j = 0; j < 4; j++) {
            const int byte_row = m_tile * SF_TILE_DIM_M + row;
            const int byte_col = k * 4 + j;
            if (byte_row >= original_M || byte_col >= original_K) {
              reinterpret_cast<uint8_t*>(&regs[i])[j] = 0;
            }
          }
        }
      }

      my_slm[k * (SF_TILE_SIZE_I32 / 4) + threadIdx.x] = *reinterpret_cast<int4*>(regs);
    }
  }

  __syncthreads();

  if (active) {
    int4* my_slm = slm_v4i + threadIdx.y * slm_tile_v4i;
    int4* out_v4i =
        reinterpret_cast<int4*>(reinterpret_cast<int*>(output) + m_tile * SF_TILE_DIM_M * K_i32);

    for (int i = threadIdx.x; i < slm_tile_v4i; i += blockDim.x) {
      out_v4i[i] = my_slm[i];
    }
  }
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    swizzle_row_scaling_narrow_k_kernel(const void* input, void* output, const int M, const int K,
                                        const int original_M, const int original_K) {
  swizzle_row_scaling_narrow_k_kernel_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, blockIdx.x, gridDim.x);
}

// Narrow-M variant of the column scaling swizzle kernel, for when num_tiles_m < TB_DIM.
// Analogous to the narrow-K row kernel: when the M dimension is small, the normal
// col kernel underutilizes threads in the load phase because threadIdx.x covers M
// positions with vectorized loads, leaving many threads idle. This kernel repurposes
// thread dimensions: threadIdx.y indexes K-tiles within the block, threadIdx.x covers
// one int32 column of an M-tile, and M-tiles are iterated serially.
template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ void swizzle_col_scaling_narrow_m_kernel_impl(const void* input, void* output,
                                                         const int M, const int K,
                                                         const int original_M, const int original_K,
                                                         const int bid, const int grid_dim) {
  constexpr int SF_TILE_SIZE_I32 = SF_TILE_DIM_M * SF_TILE_DIM_K / 4;
  constexpr int SF_TILE_DIM_M_I32 = SF_TILE_DIM_M / 4;
  constexpr int SF_TILE_DIM_K_I32 = SF_TILE_DIM_K;

  const int M_i32 = M / 4;
  const int K_i32 = K;
  const int num_tiles_m = M / SF_TILE_DIM_M;
  const int num_tiles_k = K / SF_TILE_DIM_K;

  const int k_tile = bid * blockDim.y + threadIdx.y;
  const bool active = (k_tile < num_tiles_k);
  const int remaining = num_tiles_k - bid * static_cast<int>(blockDim.y);
  const int k_tiles_in_block = remaining <= 0 ? 0 : (remaining < TB_DIM ? remaining : TB_DIM);

  extern __shared__ int slm_narrow_m[];

  if (active) {
    const bool padding_k = (k_tile == num_tiles_k - 1) && (original_K < K);
    const int32_t* input_i32 = reinterpret_cast<const int32_t*>(input);

    for (int m_tile = 0; m_tile < num_tiles_m; m_tile++) {
      const bool padding_m = (m_tile == num_tiles_m - 1) && (original_M < M);

      int regs[N_SF_PER_TD_PER_TILE];
#pragma unroll
      for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
        const int k_row = k_tile * SF_TILE_DIM_K_I32 + i;
        const int m_col = m_tile * SF_TILE_DIM_M_I32 + threadIdx.x;
        if (padding_k && k_row >= original_K) {
          regs[i] = 0;
          continue;
        }
        regs[i] = __ldg(input_i32 + k_row * M_i32 + m_col);
        if (padding_m || padding_k) {
          for (int j = 0; j < 4; j++) {
            if (m_col * 4 + j >= original_M || k_row >= original_K) {
              reinterpret_cast<uint8_t*>(&regs[i])[j] = 0;
            }
          }
        }
      }

      regs_shuffle_with_bit_shifts<int>(regs);

      int tM = threadIdx.x * N_SF_PER_TD_PER_TILE;
      int* slm_tile =
          slm_narrow_m + m_tile * TB_DIM * SF_TILE_SIZE_I32 + threadIdx.y * SF_TILE_SIZE_I32;
#pragma unroll
      for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
        slm_tile[(tM % SF_TILE_DIM_M) / NEW_SF_TILE_DIM_M_I32 +
                 ((tM + i) % NEW_SF_TILE_DIM_M_I32) * NEW_SF_TILE_DIM_K_I32] = regs[i];
      }
    }
  }

  __syncthreads();

  const int linear_id = threadIdx.y * blockDim.x + threadIdx.x;
  for (int m_tile = 0; m_tile < num_tiles_m; m_tile++) {
    int4* out_v4i = reinterpret_cast<int4*>(reinterpret_cast<int*>(output) +
                                            m_tile * SF_TILE_DIM_M_I32 * K_i32 +
                                            bid * TB_DIM * SF_TILE_SIZE_I32);
    int4* slm_v4i = reinterpret_cast<int4*>(slm_narrow_m + m_tile * TB_DIM * SF_TILE_SIZE_I32);
    const int n_v4i = k_tiles_in_block * SF_TILE_SIZE_I32 / 4;
    for (int j = linear_id; j < n_v4i; j += blockDim.x * blockDim.y) {
      out_v4i[j] = slm_v4i[j];
    }
  }
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    swizzle_col_scaling_narrow_m_kernel(const void* input, void* output, const int M, const int K,
                                        const int original_M, const int original_K) {
  swizzle_col_scaling_narrow_m_kernel_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, blockIdx.x, gridDim.x);
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ void swizzle_row_scaling_coalesced_tile_impl(const void* input, void* output,
                                                        const int M, const int K,
                                                        const int original_M,
                                                        const int m_tile, int4* slm_v4i) {
  (void)M;
  constexpr int SF_TILE_SIZE = SF_TILE_DIM_M * SF_TILE_DIM_K;
  constexpr int TILE_OUTPUT_V4 = SF_TILE_SIZE / static_cast<int>(sizeof(int4));
  const int row_v4 = K / static_cast<int>(sizeof(int4));
  const int tile_v4 = SF_TILE_DIM_M * row_v4;
  const int K_i32 = K / static_cast<int>(sizeof(int));
  const int smem_stride_i32 = row_coalesced_smem_stride_i32(K_i32);
  const int global_m = m_tile * SF_TILE_DIM_M;
  const bool row_v4_is_pow2 = (row_v4 & (row_v4 - 1)) == 0;
  const int row_v4_log2 = __ffs(row_v4) - 1;

  const int4* input_v4 = reinterpret_cast<const int4*>(input);
  const size_t input_tile_v4 = static_cast<size_t>(global_m) * row_v4;
  int* slm_i32 = reinterpret_cast<int*>(slm_v4i);

  for (int idx = threadIdx.x; idx < tile_v4; idx += blockDim.x) {
    int row, col_v4;
    divmod_tile_col_v4(idx, row_v4, row_v4_is_pow2, row_v4_log2, &row, &col_v4);
    const int col_i32 = col_v4 * static_cast<int>(sizeof(int4) / sizeof(int));
    int4 value;
    if (global_m + row < original_M) {
      value = __ldg(input_v4 + input_tile_v4 + idx);
    } else {
      value = make_int4(0, 0, 0, 0);
    }
    const int smem_base = row * smem_stride_i32 + col_i32;
    slm_i32[smem_base + 0] = value.x;
    slm_i32[smem_base + 1] = value.y;
    slm_i32[smem_base + 2] = value.z;
    slm_i32[smem_base + 3] = value.w;
  }
  __syncthreads();

  const int* slm_read_i32 = reinterpret_cast<const int*>(slm_v4i);
  int4* output_v4 = reinterpret_cast<int4*>(
      reinterpret_cast<uint8_t*>(output) + static_cast<size_t>(global_m) * K);

  for (int idx = threadIdx.x; idx < tile_v4; idx += blockDim.x) {
    const int tile_k = idx / TILE_OUTPUT_V4;
    const int row_in_new_tile = idx - tile_k * TILE_OUTPUT_V4;
    const int smem_base = row_in_new_tile * smem_stride_i32 + tile_k;
    const int4 value =
        make_int4(slm_read_i32[smem_base], slm_read_i32[smem_base + 32 * smem_stride_i32],
                  slm_read_i32[smem_base + 64 * smem_stride_i32],
                  slm_read_i32[smem_base + 96 * smem_stride_i32]);
    store_global_cs(output_v4 + idx, value);
  }
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(ROW_COALESCED_THREADS)
    swizzle_row_scaling_coalesced_kernel(const void* input, void* output, const int M,
                                         const int K, const int original_M) {
  extern __shared__ int4 slm_v4i[];
  swizzle_row_scaling_coalesced_tile_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, blockIdx.x, slm_v4i);
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ void swizzle_row_scaling_coalesced_batched_m_impl(
    const void* input, void* output, const int M, const int K, const int original_M,
    const int first_m_tile, const int m_tiles_per_block, int4* slm_v4i) {
  (void)M;
  constexpr int SF_TILE_SIZE = SF_TILE_DIM_M * SF_TILE_DIM_K;
  constexpr int TILE_OUTPUT_V4 = SF_TILE_SIZE / static_cast<int>(sizeof(int4));
  const int row_v4 = K / static_cast<int>(sizeof(int4));
  const int tile_v4 = SF_TILE_DIM_M * row_v4;
  const int K_i32 = K / static_cast<int>(sizeof(int));
  const int smem_stride_i32 = row_coalesced_smem_stride_i32(K_i32);
  const int smem_tile_i32 = SF_TILE_DIM_M * smem_stride_i32;
  const int num_m_tiles = M / SF_TILE_DIM_M;
  const int remaining_m_tiles = num_m_tiles - first_m_tile;
  const int active_m_tiles =
      remaining_m_tiles <= 0 ? 0
                             : (remaining_m_tiles < m_tiles_per_block ? remaining_m_tiles
                                                                       : m_tiles_per_block);

  const int4* input_v4 = reinterpret_cast<const int4*>(input);
  int* slm_i32 = reinterpret_cast<int*>(slm_v4i);
  const bool row_v4_is_pow2 = (row_v4 & (row_v4 - 1)) == 0;
  const int row_v4_log2 = __ffs(row_v4) - 1;

  for (int local_m_tile = 0; local_m_tile < active_m_tiles; ++local_m_tile) {
    const int global_m = (first_m_tile + local_m_tile) * SF_TILE_DIM_M;
    const size_t input_tile_v4 = static_cast<size_t>(global_m) * row_v4;
    int* smem_tile = slm_i32 + local_m_tile * smem_tile_i32;

    for (int tile_idx = threadIdx.x; tile_idx < tile_v4; tile_idx += blockDim.x) {
      int row, col_v4;
      divmod_tile_col_v4(tile_idx, row_v4, row_v4_is_pow2, row_v4_log2, &row, &col_v4);
      const int col_i32 = col_v4 * static_cast<int>(sizeof(int4) / sizeof(int));

      int4 value;
      if (global_m + row < original_M) {
        value = __ldg(input_v4 + input_tile_v4 + tile_idx);
      } else {
        value = make_int4(0, 0, 0, 0);
      }

      const int smem_base = row * smem_stride_i32 + col_i32;
      smem_tile[smem_base + 0] = value.x;
      smem_tile[smem_base + 1] = value.y;
      smem_tile[smem_base + 2] = value.z;
      smem_tile[smem_base + 3] = value.w;
    }
  }
  __syncthreads();

  const int* slm_read_i32 = reinterpret_cast<const int*>(slm_v4i);
  for (int local_m_tile = 0; local_m_tile < active_m_tiles; ++local_m_tile) {
    const int global_m = (first_m_tile + local_m_tile) * SF_TILE_DIM_M;
    int4* output_v4 = reinterpret_cast<int4*>(
        reinterpret_cast<uint8_t*>(output) + static_cast<size_t>(global_m) * K);

    for (int tile_idx = threadIdx.x; tile_idx < tile_v4; tile_idx += blockDim.x) {
      const int tile_k = tile_idx / TILE_OUTPUT_V4;
      const int row_in_new_tile = tile_idx - tile_k * TILE_OUTPUT_V4;
      const int smem_base = local_m_tile * smem_tile_i32 + row_in_new_tile * smem_stride_i32 +
                            tile_k;
      const int4 value =
          make_int4(slm_read_i32[smem_base], slm_read_i32[smem_base + 32 * smem_stride_i32],
                    slm_read_i32[smem_base + 64 * smem_stride_i32],
                    slm_read_i32[smem_base + 96 * smem_stride_i32]);
      store_global_cs(output_v4 + tile_idx, value);
    }
  }
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(ROW_COALESCED_THREADS)
    swizzle_row_scaling_coalesced_batched_m_kernel(const void* input, void* output, const int M,
                                                   const int K, const int original_M,
                                                   const int m_tiles_per_block) {
  extern __shared__ int4 slm_v4i[];
  const int first_m_tile = blockIdx.x * m_tiles_per_block;
  swizzle_row_scaling_coalesced_batched_m_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, first_m_tile, m_tiles_per_block, slm_v4i);
}

__device__ __forceinline__ uint4 load_col_swizzle_segment(const uint8_t* input, const int M,
                                                          const int original_M,
                                                          const int original_K, const int m_base,
                                                          const int k, const int segment) {
  const int m = m_base + segment * static_cast<int>(sizeof(uint4));
  if (k >= original_K || m >= original_M) {
    return make_uint4(0, 0, 0, 0);
  }
  if (m + static_cast<int>(sizeof(uint4)) <= original_M) {
    return __ldg(reinterpret_cast<const uint4*>(input + static_cast<size_t>(k) * M + m));
  }

  uint4 value = make_uint4(0, 0, 0, 0);
  uint8_t* value_bytes = reinterpret_cast<uint8_t*>(&value);
#pragma unroll
  for (int i = 0; i < static_cast<int>(sizeof(uint4)); ++i) {
    if (m + i < original_M) {
      value_bytes[i] = __ldg(input + static_cast<size_t>(k) * M + m + i);
    }
  }
  return value;
}

__device__ __forceinline__ uint32_t shuffle_uint4_byte(const uint4 value, const int src_lane,
                                                       const int byte_idx) {
  const int component = byte_idx / static_cast<int>(sizeof(uint32_t));
  const uint32_t src_x = __shfl_sync(0xffffffff, value.x, src_lane);
  const uint32_t src_y = __shfl_sync(0xffffffff, value.y, src_lane);
  const uint32_t src_z = __shfl_sync(0xffffffff, value.z, src_lane);
  const uint32_t src_w = __shfl_sync(0xffffffff, value.w, src_lane);
  const uint32_t src_component =
      component == 0 ? src_x : component == 1 ? src_y : component == 2 ? src_z : src_w;
  return (src_component >> (8 * (byte_idx % static_cast<int>(sizeof(uint32_t))))) & 0xff;
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ void swizzle_col_scaling_coalesced_tile_impl(
    const void* input, void* output, const int M, const int K, const int original_M,
    const int original_K, const int m_tile, const int k_tile_block,
    const int k_tiles_per_block, int* slm_i32) {
  constexpr int SF_TILE_SIZE = SF_TILE_DIM_M * SF_TILE_DIM_K;
  constexpr int M_TILE_V4 = SF_TILE_DIM_M / static_cast<int>(sizeof(uint4));
  constexpr int SMEM_STRIDE_I32 = SF_TILE_DIM_M / static_cast<int>(sizeof(int)) + 1;
  constexpr int SMEM_STRIDE_BYTES = SMEM_STRIDE_I32 * static_cast<int>(sizeof(int));

  const int num_tiles_k = K / SF_TILE_DIM_K;
  const int first_k_tile = k_tile_block * k_tiles_per_block;
  const int remaining_k_tiles = num_tiles_k - first_k_tile;
  const int active_k_tiles =
      remaining_k_tiles <= 0 ? 0
                             : (remaining_k_tiles < k_tiles_per_block ? remaining_k_tiles
                                                                       : k_tiles_per_block);
  const int active_k = active_k_tiles * SF_TILE_DIM_K;
  const int k_base = first_k_tile * SF_TILE_DIM_K;
  const int m_base = m_tile * SF_TILE_DIM_M;
  const uint8_t* input_u8 = reinterpret_cast<const uint8_t*>(input);
  uint32_t* slm_u32 = reinterpret_cast<uint32_t*>(slm_i32);

  for (int idx = threadIdx.x; idx < active_k * M_TILE_V4; idx += blockDim.x) {
    const int k_rel = idx / M_TILE_V4;
    const int segment = idx - k_rel * M_TILE_V4;
    const uint4 value =
        load_col_swizzle_segment(input_u8, M, original_M, original_K, m_base, k_base + k_rel,
                                 segment);
    uint32_t* smem_row = slm_u32 + k_rel * SMEM_STRIDE_I32 + segment * 4;
    smem_row[0] = value.x;
    smem_row[1] = value.y;
    smem_row[2] = value.z;
    smem_row[3] = value.w;
  }
  __syncthreads();

  uint4* output_v4 = reinterpret_cast<uint4*>(
      reinterpret_cast<uint8_t*>(output) + static_cast<size_t>(m_tile) * SF_TILE_DIM_M * K +
      static_cast<size_t>(first_k_tile) * SF_TILE_SIZE);
  const uint8_t* slm_u8 = reinterpret_cast<const uint8_t*>(slm_i32);

  const int active_output_v4 = active_k_tiles * (SF_TILE_SIZE / static_cast<int>(sizeof(uint4)));
  for (int idx = threadIdx.x; idx < active_output_v4;
       idx += blockDim.x) {
    const int tile_rel = idx / (SF_TILE_SIZE / static_cast<int>(sizeof(uint4)));
    const int row_in_new_tile = idx % (SF_TILE_SIZE / static_cast<int>(sizeof(uint4)));
    const int k_rel_base = tile_rel * SF_TILE_DIM_K;
    uint4 value;
    uint32_t* value_words = reinterpret_cast<uint32_t*>(&value);
#pragma unroll
    for (int m_group = 0; m_group < 4; ++m_group) {
      uint32_t word = 0;
#pragma unroll
      for (int k_group = 0; k_group < SF_TILE_DIM_K; ++k_group) {
        const int byte_offset =
            (k_rel_base + k_group) * SMEM_STRIDE_BYTES + row_in_new_tile + m_group * 32;
        word |= static_cast<uint32_t>(slm_u8[byte_offset]) << (8 * k_group);
      }
      value_words[m_group] = word;
    }
    store_global_cs(output_v4 + idx, value);
  }
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(COL_COALESCED_THREADS)
    swizzle_col_scaling_coalesced_kernel(const void* input, void* output, const int M,
                                         const int K, const int original_M,
                                         const int original_K,
                                         const int k_tiles_per_block) {
  extern __shared__ int slm_i32[];
  swizzle_col_scaling_coalesced_tile_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, blockIdx.y, blockIdx.x, k_tiles_per_block,
      slm_i32);
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ void swizzle_col_scaling_warp_tile_impl(const void* input, void* output, const int M,
                                                   const int K, const int original_M,
                                                   const int original_K, const int tile_id) {
  constexpr int SF_TILE_SIZE = SF_TILE_DIM_M * SF_TILE_DIM_K;
  const int lane = threadIdx.x & 31;
  const int num_tiles_k = K / SF_TILE_DIM_K;
  const int m_tile = tile_id / num_tiles_k;
  const int k_tile = tile_id - m_tile * num_tiles_k;
  const int m_base = m_tile * SF_TILE_DIM_M;
  const int k_base = k_tile * SF_TILE_DIM_K;
  const uint8_t* input_u8 = reinterpret_cast<const uint8_t*>(input);
  uint4* output_v4 = reinterpret_cast<uint4*>(
      reinterpret_cast<uint8_t*>(output) + static_cast<size_t>(m_tile) * SF_TILE_DIM_M * K +
      k_tile * SF_TILE_SIZE);

  const int input_row = lane / 8;
  const int input_segment = lane % 8;
  const uint4 input_segment_data = load_col_swizzle_segment(
      input_u8, M, original_M, original_K, m_base, k_base + input_row, input_segment);

  uint32_t output_words[4];
  const int lane_byte = lane % static_cast<int>(sizeof(uint4));
  const int lane_segment_base = lane / static_cast<int>(sizeof(uint4));
#pragma unroll
  for (int m_group = 0; m_group < 4; ++m_group) {
    uint32_t word = 0;
    const int src_segment = lane_segment_base + m_group * 2;
#pragma unroll
    for (int k_group = 0; k_group < SF_TILE_DIM_K; ++k_group) {
      const int src_lane = k_group * 8 + src_segment;
      word |= shuffle_uint4_byte(input_segment_data, src_lane, lane_byte) << (8 * k_group);
    }
    output_words[m_group] = word;
  }

  store_global_cs(output_v4 + lane,
                  make_uint4(output_words[0], output_words[1], output_words[2], output_words[3]));
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(COL_WARP_TILE_THREADS)
    swizzle_col_scaling_warp_tile_kernel(const void* input, void* output, const int M,
                                         const int K, const int original_M,
                                         const int original_K, const int total_tiles) {
  const int warp_id = threadIdx.x / 32;
  const int tile_id = blockIdx.x * COL_WARP_TILE_WARPS + warp_id;
  if (tile_id < total_tiles) {
    swizzle_col_scaling_warp_tile_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
        input, output, M, K, original_M, original_K, tile_id);
  }
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ void unswizzle_row_scaling_kernel_impl(const void* input, void* output, const int M,
                                                  const int K, const int bid_x, const int bid_y,
                                                  const int grid_dim_x, const int grid_dim_y) {
  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int N_TILES_IN_TB = TB_DIM * N_TILE_PER_TD;

  constexpr int SF_TILE_SIZE_I32 = SF_TILE_DIM_M * SF_TILE_DIM_K / 4;
  constexpr int SF_TILE_DIM_M_I32 = SF_TILE_DIM_M;

  int n_tiles_in_tb = N_TILES_IN_TB;
  const int K_i32 = K / 4;
  if (bid_x == grid_dim_x - 1) {
    n_tiles_in_tb = (K_i32 - 1) % N_TILES_IN_TB + 1;
  }

  const int input_offset =
      bid_y * SF_TILE_DIM_M_I32 * K_i32 + bid_x * N_TILES_IN_TB * SF_TILE_SIZE_I32;
  const int* input_i32 = reinterpret_cast<const int*>(input) + input_offset;
  const int output_offset = bid_y * SF_TILE_DIM_M_I32 * K_i32 + bid_x * N_TILES_IN_TB;
  int* output_i32 = reinterpret_cast<int*>(output) + output_offset;

  extern __shared__ int4 slm_v4i[];

  int linear_id = threadIdx.y * blockDim.x + threadIdx.x;
  const int4* input_v4i = reinterpret_cast<const int4*>(input_i32);
#pragma unroll
  for (int i = linear_id; i < SF_TILE_SIZE_I32 * n_tiles_in_tb / 4; i += blockDim.x * blockDim.y) {
    slm_v4i[i] = input_v4i[i];
  }
  __syncthreads();

  LType regs_vec[N_SF_PER_TD_PER_TILE];
  if (threadIdx.x * N_TILE_PER_TD < n_tiles_in_tb) {
#pragma unroll
    for (int i = 0; i < N_TILE_PER_TD; i++) {
      reinterpret_cast<int4*>(regs_vec)[i] =
          slm_v4i[(threadIdx.x * N_TILE_PER_TD + i) * SF_TILE_SIZE_I32 / 4 + threadIdx.y];
    }

    regs_unshuffle<LType>(regs_vec);

#pragma unroll
    for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
      const int thread_offset = (i * TB_DIM + threadIdx.y) * K_i32 + threadIdx.x * N_TILE_PER_TD;
      reinterpret_cast<LType*>(output_i32 + thread_offset)[0] = regs_vec[i];
    }
  }
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ void unswizzle_col_scaling_kernel_impl(const void* input, void* output, const int M,
                                                  const int K, const int bid_x, const int bid_y,
                                                  const int grid_dim_x, const int grid_dim_y) {
  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int N_SF_PER_TD = N_TILE_PER_TD * N_SF_PER_TD_PER_TILE;
  constexpr int SF_TILE_SIZE_I32 = SF_TILE_DIM_M * SF_TILE_DIM_K / 4;

  constexpr int SF_TILE_DIM_M_I32 = SF_TILE_DIM_M / 4;
  constexpr int SF_TILE_DIM_K_I32 = SF_TILE_DIM_K;

  const int M_i32 = M / 4;
  const int K_i32 = K;

  int m_tiles_in_tb = N_TILE_PER_TD;
  int k_tiles_in_tb = TB_DIM;
  if (bid_x == grid_dim_x - 1) {
    k_tiles_in_tb = (K_i32 / SF_TILE_DIM_K_I32 - 1) % k_tiles_in_tb + 1;
  }
  if (bid_y == grid_dim_y - 1) {
    m_tiles_in_tb = (M_i32 / SF_TILE_DIM_M_I32 - 1) % m_tiles_in_tb + 1;
  }

  const int32_t* input_i32[N_TILE_PER_TD];
#pragma unroll
  for (int i = 0; i < m_tiles_in_tb; i++) {
    input_i32[i] = reinterpret_cast<const int32_t*>(input) + bid_x * TB_DIM * SF_TILE_SIZE_I32 +
                   (bid_y * N_TILE_PER_TD + i) * SF_TILE_DIM_M_I32 * K_i32;
  }
  const int output_offset =
      bid_x * TB_DIM * SF_TILE_DIM_K_I32 * M_i32 + bid_y * N_TILE_PER_TD * SF_TILE_DIM_M_I32;
  int* output_i32 = reinterpret_cast<int*>(output) + output_offset;

  extern __shared__ int slm[];

  int linear_id = threadIdx.y * blockDim.x + threadIdx.x;
#pragma unroll
  for (int i = 0; i < m_tiles_in_tb; i++) {
    __align__(16) const int4* input_v4i = reinterpret_cast<const int4*>(input_i32[i]);
    __align__(16) int4* slm_v4i =
        reinterpret_cast<int4*>(slm + i * k_tiles_in_tb * SF_TILE_SIZE_I32);
#pragma unroll
    for (int j = linear_id; j < SF_TILE_SIZE_I32 * k_tiles_in_tb / 4;
         j += blockDim.x * blockDim.y) {
      slm_v4i[j] = input_v4i[j];
    }
  }
  __syncthreads();

  LType regs_vec[N_SF_PER_TD_PER_TILE];
  if (threadIdx.x * N_TILE_PER_TD < m_tiles_in_tb * SF_TILE_DIM_M_I32 &&
      threadIdx.y < k_tiles_in_tb) {
    int tM = threadIdx.x * N_SF_PER_TD;
    int* slm_tile = slm + (threadIdx.y * SF_TILE_SIZE_I32 +
                           tM / SF_TILE_DIM_M * k_tiles_in_tb * SF_TILE_SIZE_I32);
#pragma unroll
    for (int i = 0; i < N_SF_PER_TD; i++) {
      reinterpret_cast<int*>(regs_vec)[i] =
          slm_tile[(tM % SF_TILE_DIM_M) / NEW_SF_TILE_DIM_M_I32 +
                   ((tM + i) % NEW_SF_TILE_DIM_M_I32) * NEW_SF_TILE_DIM_K_I32];
    }

    regs_unshuffle_with_bit_shifts(regs_vec);

#pragma unroll
    for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
      const int thread_offset =
          (threadIdx.y * SF_TILE_DIM_K_I32 + i) * M_i32 + threadIdx.x * N_TILE_PER_TD;
      reinterpret_cast<LType*>(output_i32 + thread_offset)[0] = regs_vec[i];
    }
  }
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    unswizzle_scaling_kernel(const void* input, void* output, const int M, const int K,
                             const bool row_scaling) {
  const int bid_x = blockIdx.x;
  const int bid_y = blockIdx.y;
  const int grid_dim_x = gridDim.x;
  const int grid_dim_y = gridDim.y;
  if (row_scaling) {
    unswizzle_row_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
        input, output, M, K, bid_x, bid_y, grid_dim_x, grid_dim_y);
  } else {
    unswizzle_col_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
        input, output, M, K, bid_x, bid_y, grid_dim_x, grid_dim_y);
  }
}

constexpr int kMaxTensorsPerKernel = 64;  // Args must be <4 KB
struct MultiSwizzleArgs {
  // (input) Data buffers for input scaling factors
  void* input_list[kMaxTensorsPerKernel];
  // (output) Data buffers for swizzled scaling factors
  void* output_list[kMaxTensorsPerKernel];
  // Input scaling factor m
  int m_list[kMaxTensorsPerKernel];
  // Input scaling factor k
  int k_list[kMaxTensorsPerKernel];
  // Input scaling factor m before padding
  int original_m_list[kMaxTensorsPerKernel];
  // Input scaling factor k before padding
  int original_k_list[kMaxTensorsPerKernel];
  // Prefix sum (with leading zero) of CUDA blocks needed for each
  // tensor
  int block_range[kMaxTensorsPerKernel + 1];
  // Number of tensors being processed by kernel
  int num_tensors;
};

constexpr size_t round_up_to_multiple(size_t value, size_t multiple) {
  return DIVUP(value, multiple) * multiple;
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
int row_swizzle_m_tiles_per_block(const int num_tiles_k, const int reserved_smem_bytes = 0) {
  if (num_tiles_k <= 0) return 0;
  const int per_m_tile_slm_size =
      num_tiles_k * SF_TILE_DIM_M * SF_TILE_DIM_K * static_cast<int>(sizeof(int8_t));
  const int available_smem = get_max_dynamic_smem() - reserved_smem_bytes;
  if (available_smem <= 0 || per_m_tile_slm_size > available_smem) return 0;
  return std::min(TB_DIM, available_smem / per_m_tile_slm_size);
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
int row_coalesced_slm_size_bytes(const int padded_k, const size_t scale_elem_size) {
  if (scale_elem_size != sizeof(uint8_t)) {
    return SF_TILE_DIM_M * (padded_k + static_cast<int>(sizeof(int))) *
           static_cast<int>(scale_elem_size);
  }
  const int k_i32 = padded_k / static_cast<int>(sizeof(int));
  return SF_TILE_DIM_M * row_coalesced_smem_stride_i32(k_i32) *
         static_cast<int>(sizeof(int));
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
int row_coalesced_m_tiles_per_block(const int padded_k, const size_t scale_elem_size,
                                    const int reserved_smem_bytes = 0) {
  const int slm_size =
      row_coalesced_slm_size_bytes<SF_TILE_DIM_M, SF_TILE_DIM_K>(padded_k, scale_elem_size);
  const int available_smem = get_max_dynamic_smem() - reserved_smem_bytes;
  if (slm_size > available_smem) return 0;
  const int smem_limited_tiles = available_smem / slm_size;
  const int target_smem_tiles = std::max(1, ROW_COALESCED_TARGET_SMEM_BYTES / slm_size);
  return std::min(ROW_COALESCED_MAX_M_TILES_PER_BLOCK,
                  std::min(smem_limited_tiles, target_smem_tiles));
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
bool use_blackwell_row_coalesced_swizzle(const int padded_k, const int original_k,
                                         const size_t scale_elem_size,
                                         const int reserved_smem_bytes = 0) {
  if (!(cuda::sm_arch() >= 100 && scale_elem_size == sizeof(uint8_t) &&
        original_k == padded_k && padded_k >= ROW_COALESCED_MIN_K &&
        padded_k % static_cast<int>(sizeof(int4)) == 0)) {
    return false;
  }
  const int slm_size =
      row_coalesced_slm_size_bytes<SF_TILE_DIM_M, SF_TILE_DIM_K>(padded_k, scale_elem_size);
  const int available_smem = get_max_dynamic_smem() - reserved_smem_bytes;
  return slm_size <= available_smem;
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
int col_coalesced_slm_size_bytes(const int k_tiles_per_block) {
  constexpr int SMEM_STRIDE_I32 = SF_TILE_DIM_M / static_cast<int>(sizeof(int)) + 1;
  return k_tiles_per_block * SF_TILE_DIM_K * SMEM_STRIDE_I32 * static_cast<int>(sizeof(int));
}

bool use_blackwell_col_coalesced_swizzle(const size_t scale_elem_size) {
  return cuda::sm_arch() >= 100 && scale_elem_size == sizeof(uint8_t);
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    grouped_swizzle_row_scaling_uniform_shape_kernel(const void* input, void* output, const int M,
                                                     const int K, const int original_M,
                                                     const int original_K,
                                                     const size_t input_stride_bytes,
                                                     const size_t output_stride_bytes) {
  const int tensor_id = blockIdx.z;
  // Input and output strides may differ: input is in the kernel-produced "compact"
  // layout (per-tensor stride = original_M * padded_k * elem_size) when callers
  // pass the unswizzled grouped scale buffer as-is, while the output is always in
  // the per-tensor padded ("swizzle-ready") layout (padded_m * padded_k * elem_size).
  const uint8_t* input_base =
      reinterpret_cast<const uint8_t*>(input) + tensor_id * input_stride_bytes;
  uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + tensor_id * output_stride_bytes;
  const bool padding_m = (blockIdx.y == gridDim.y - 1) && (original_M < M);
  const bool padding_k = (blockIdx.x == gridDim.x - 1) && (original_K < K);
  dispatch_swizzle_row_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input_base, output_base, M, K, original_M, original_K, blockIdx.x, blockIdx.y, gridDim.x,
      gridDim.y, padding_k, padding_m);
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    grouped_swizzle_col_scaling_uniform_shape_kernel(const void* input, void* output, const int M,
                                                     const int K, const int original_M,
                                                     const int original_K,
                                                     const size_t input_stride_bytes,
                                                     const size_t output_stride_bytes) {
  const int tensor_id = blockIdx.z;
  // See the rowwise kernel for stride semantics. For columnwise the per-tensor
  // compact stride is DIVUP(original_K, 1) * padded_m * elem_size (i.e. the
  // unpadded scale-row count in the K direction times the padded M extent).
  const uint8_t* input_base =
      reinterpret_cast<const uint8_t*>(input) + tensor_id * input_stride_bytes;
  uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + tensor_id * output_stride_bytes;
  const bool padding_m = (blockIdx.y == gridDim.y - 1) && (original_M < M);
  const bool padding_k = (blockIdx.x == gridDim.x - 1) && (original_K < K);
  dispatch_swizzle_col_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input_base, output_base, M, K, original_M, original_K, blockIdx.x, blockIdx.y, gridDim.x,
      gridDim.y, padding_k, padding_m);
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    grouped_swizzle_row_scaling_uniform_shape_narrow_k_kernel(
        const void* input, void* output, const int M, const int K, const int original_M,
        const int original_K, const size_t input_stride_bytes, const size_t output_stride_bytes) {
  const int tensor_id = blockIdx.z;
  const uint8_t* input_base =
      reinterpret_cast<const uint8_t*>(input) + tensor_id * input_stride_bytes;
  uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + tensor_id * output_stride_bytes;
  swizzle_row_scaling_narrow_k_kernel_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input_base, output_base, M, K, original_M, original_K, blockIdx.x, gridDim.x);
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    grouped_swizzle_col_scaling_uniform_shape_narrow_m_kernel(
        const void* input, void* output, const int M, const int K, const int original_M,
        const int original_K, const size_t input_stride_bytes, const size_t output_stride_bytes) {
  const int tensor_id = blockIdx.z;
  const uint8_t* input_base =
      reinterpret_cast<const uint8_t*>(input) + tensor_id * input_stride_bytes;
  uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + tensor_id * output_stride_bytes;
  swizzle_col_scaling_narrow_m_kernel_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input_base, output_base, M, K, original_M, original_K, blockIdx.x, gridDim.x);
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(ROW_COALESCED_THREADS)
    grouped_swizzle_row_scaling_uniform_shape_coalesced_kernel(
        const void* input, void* output, const int M, const int K, const int original_M,
        const size_t input_stride_bytes, const size_t output_stride_bytes) {
  const int tensor_id = blockIdx.y;
  const uint8_t* input_base =
      reinterpret_cast<const uint8_t*>(input) + tensor_id * input_stride_bytes;
  uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + tensor_id * output_stride_bytes;
  extern __shared__ int4 slm_v4i[];
  swizzle_row_scaling_coalesced_tile_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input_base, output_base, M, K, original_M, blockIdx.x, slm_v4i);
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(ROW_COALESCED_THREADS)
    grouped_swizzle_row_scaling_uniform_shape_coalesced_batched_m_kernel(
        const void* input, void* output, const int M, const int K, const int original_M,
        const size_t input_stride_bytes, const size_t output_stride_bytes,
        const int m_tiles_per_block) {
  const int tensor_id = blockIdx.y;
  const uint8_t* input_base =
      reinterpret_cast<const uint8_t*>(input) + tensor_id * input_stride_bytes;
  uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + tensor_id * output_stride_bytes;
  extern __shared__ int4 slm_v4i[];
  const int first_m_tile = blockIdx.x * m_tiles_per_block;
  swizzle_row_scaling_coalesced_batched_m_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input_base, output_base, M, K, original_M, first_m_tile, m_tiles_per_block, slm_v4i);
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(COL_WARP_TILE_THREADS)
    grouped_swizzle_col_scaling_uniform_shape_warp_tile_kernel(
        const void* input, void* output, const int M, const int K, const int original_M,
        const int original_K, const size_t input_stride_bytes, const size_t output_stride_bytes,
        const int total_tiles) {
  const int tensor_id = blockIdx.y;
  const uint8_t* input_base =
      reinterpret_cast<const uint8_t*>(input) + tensor_id * input_stride_bytes;
  uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + tensor_id * output_stride_bytes;
  const int warp_id = threadIdx.x / 32;
  const int tile_id = blockIdx.x * COL_WARP_TILE_WARPS + warp_id;
  if (tile_id < total_tiles) {
    swizzle_col_scaling_warp_tile_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
        input_base, output_base, M, K, original_M, original_K, tile_id);
  }
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(COL_COALESCED_THREADS)
    grouped_swizzle_col_scaling_uniform_shape_coalesced_kernel(
        const void* input, void* output, const int M, const int K, const int original_M,
        const int original_K, const size_t input_stride_bytes, const size_t output_stride_bytes,
        const int k_tiles_per_block) {
  const int tensor_id = blockIdx.z;
  const uint8_t* input_base =
      reinterpret_cast<const uint8_t*>(input) + tensor_id * input_stride_bytes;
  uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + tensor_id * output_stride_bytes;
  extern __shared__ int slm_i32[];
  swizzle_col_scaling_coalesced_tile_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input_base, output_base, M, K, original_M, original_K, blockIdx.y, blockIdx.x,
      k_tiles_per_block, slm_i32);
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    grouped_unswizzle_scaling_uniform_shape_kernel(const void* input, void* output, const int M,
                                                   const int K, const size_t scale_stride_bytes,
                                                   const bool row_scaling) {
  const int tensor_id = blockIdx.z;
  const uint8_t* input_base =
      reinterpret_cast<const uint8_t*>(input) + tensor_id * scale_stride_bytes;
  uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + tensor_id * scale_stride_bytes;
  if (row_scaling) {
    unswizzle_row_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
        input_base, output_base, M, K, blockIdx.x, blockIdx.y, gridDim.x, gridDim.y);
  } else {
    unswizzle_col_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
        input_base, output_base, M, K, blockIdx.x, blockIdx.y, gridDim.x, gridDim.y);
  }
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void multi_tensor_unswizzle_row_scaling_kernel(MultiSwizzleArgs kernel_args) {
  const int bid = blockIdx.x;
  int tensor_id = 0;
  while (kernel_args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  const void* input = kernel_args.input_list[tensor_id];
  void* output = kernel_args.output_list[tensor_id];
  const int M = kernel_args.m_list[tensor_id];
  const int K = kernel_args.k_list[tensor_id];

  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int N_TILES_IN_TB = TB_DIM * N_TILE_PER_TD;

  const int num_tiles_k = K / SF_TILE_DIM_K;
  const int num_tiles_m = M / SF_TILE_DIM_M;
  const int flat_offset = bid - kernel_args.block_range[tensor_id];
  const int grid_dim_x = DIVUP(num_tiles_k, N_TILES_IN_TB);
  const int grid_dim_y = num_tiles_m;
  const int bid_x = flat_offset / grid_dim_y;
  const int bid_y = flat_offset % grid_dim_y;

  unswizzle_row_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, bid_x, bid_y, grid_dim_x, grid_dim_y);
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void multi_tensor_unswizzle_col_scaling_kernel(MultiSwizzleArgs kernel_args) {
  const int bid = blockIdx.x;
  int tensor_id = 0;
  while (kernel_args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  const void* input = kernel_args.input_list[tensor_id];
  void* output = kernel_args.output_list[tensor_id];
  const int M = kernel_args.m_list[tensor_id];
  const int K = kernel_args.k_list[tensor_id];

  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);

  const int num_tiles_k = K / SF_TILE_DIM_K;
  const int num_tiles_m = M / SF_TILE_DIM_M;
  const int flat_offset = bid - kernel_args.block_range[tensor_id];
  const int grid_dim_x = DIVUP(num_tiles_k, TB_DIM);
  const int grid_dim_y = DIVUP(num_tiles_m, N_TILE_PER_TD);
  const int bid_x = flat_offset / grid_dim_y;
  const int bid_y = flat_offset % grid_dim_y;

  unswizzle_col_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, bid_x, bid_y, grid_dim_x, grid_dim_y);
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void multi_tensor_swizzle_row_scaling_kernel(MultiSwizzleArgs kernel_args) {
  // Find tensor corresponding to block
  const int bid = blockIdx.x;
  int tensor_id = 0;
  while (kernel_args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  // Get args corresponding to block
  const void* input = kernel_args.input_list[tensor_id];
  void* output = kernel_args.output_list[tensor_id];
  const int M = kernel_args.m_list[tensor_id];
  const int K = kernel_args.k_list[tensor_id];
  const int original_M = kernel_args.original_m_list[tensor_id];
  const int original_K = kernel_args.original_k_list[tensor_id];

  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int N_TILES_IN_TB = TB_DIM * N_TILE_PER_TD;

  // Get block index in grid. Emulate 2D grid.
  const int num_tiles_k = K / SF_TILE_DIM_K;
  const int num_tiles_m = M / SF_TILE_DIM_M;
  const int grid_dim_x = DIVUP(num_tiles_k, N_TILES_IN_TB);
  const int grid_dim_y = num_tiles_m;
  const int bid_x = (bid - kernel_args.block_range[tensor_id]) / grid_dim_y;
  const int bid_y = (bid - kernel_args.block_range[tensor_id]) % grid_dim_y;

  const bool padding_m = (bid_y == grid_dim_y - 1) && (original_M < M);
  const bool padding_k = (bid_x == grid_dim_x - 1) && (original_K < K);
  dispatch_swizzle_row_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, bid_x, bid_y, grid_dim_x, grid_dim_y, padding_k,
      padding_m);
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void multi_tensor_swizzle_col_scaling_kernel(MultiSwizzleArgs kernel_args) {
  // Find tensor corresponding to block
  const int bid = blockIdx.x;
  int tensor_id = 0;
  while (kernel_args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  // Get args corresponding to block
  const void* input = kernel_args.input_list[tensor_id];
  void* output = kernel_args.output_list[tensor_id];
  const int M = kernel_args.m_list[tensor_id];
  const int K = kernel_args.k_list[tensor_id];
  const int original_M = kernel_args.original_m_list[tensor_id];
  const int original_K = kernel_args.original_k_list[tensor_id];

  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);

  // Get block index in grid. Emulate 2D grid.
  const int num_tiles_k = K / SF_TILE_DIM_K;
  const int num_tiles_m = M / SF_TILE_DIM_M;
  const int grid_dim_x = DIVUP(num_tiles_k, TB_DIM);
  const int grid_dim_y = DIVUP(num_tiles_m, N_TILE_PER_TD);
  const int bid_x = (bid - kernel_args.block_range[tensor_id]) / grid_dim_y;
  const int bid_y = (bid - kernel_args.block_range[tensor_id]) % grid_dim_y;

  const bool padding_m = (bid_y == grid_dim_y - 1) && (original_M < M);
  const bool padding_k = (bid_x == grid_dim_x - 1) && (original_K < K);
  dispatch_swizzle_col_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, bid_x, bid_y, grid_dim_x, grid_dim_y, padding_k,
      padding_m);
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    multi_tensor_swizzle_row_scaling_narrow_k_kernel(MultiSwizzleArgs kernel_args) {
  const int bid = blockIdx.x;
  int tensor_id = 0;
  while (kernel_args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  const void* input = kernel_args.input_list[tensor_id];
  void* output = kernel_args.output_list[tensor_id];
  const int M = kernel_args.m_list[tensor_id];
  const int K = kernel_args.k_list[tensor_id];
  const int original_M = kernel_args.original_m_list[tensor_id];
  const int original_K = kernel_args.original_k_list[tensor_id];
  const int flat_bid = bid - kernel_args.block_range[tensor_id];
  const int num_tiles_m = M / SF_TILE_DIM_M;
  const int grid_dim = DIVUP(num_tiles_m, TB_DIM);

  swizzle_row_scaling_narrow_k_kernel_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, flat_bid, grid_dim);
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    multi_tensor_swizzle_col_scaling_narrow_m_kernel(MultiSwizzleArgs kernel_args) {
  const int bid = blockIdx.x;
  int tensor_id = 0;
  while (kernel_args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  const void* input = kernel_args.input_list[tensor_id];
  void* output = kernel_args.output_list[tensor_id];
  const int M = kernel_args.m_list[tensor_id];
  const int K = kernel_args.k_list[tensor_id];
  const int original_M = kernel_args.original_m_list[tensor_id];
  const int original_K = kernel_args.original_k_list[tensor_id];
  const int flat_bid = bid - kernel_args.block_range[tensor_id];
  const int num_tiles_k = K / SF_TILE_DIM_K;
  const int grid_dim = DIVUP(num_tiles_k, TB_DIM);

  swizzle_col_scaling_narrow_m_kernel_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, flat_bid, grid_dim);
}

}  // namespace

void swizzle_scaling_factors(const Tensor* input, Tensor* output, cudaStream_t stream) {
  // Check scaling mode
  const auto& scaling_mode = input->scaling_mode;
  NVTE_CHECK(scaling_mode == NVTE_MXFP8_1D_SCALING || scaling_mode == NVTE_NVFP4_1D_SCALING,
             "Input tensor has invalid scaling mode (", to_string(input->scaling_mode), ").");

  // Check tensors
  CheckInputTensor(*input, "scaling_factor_input");
  CheckInputTensor(*output, "scaling_factor_output");
  NVTE_CHECK(!input->with_gemm_swizzled_scales,
             "Expected input tensor with scales in compact format.");
  NVTE_CHECK(output->with_gemm_swizzled_scales,
             "Expected output tensor with scales in GEMM swizzled format.");
  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING:
      NVTE_CHECK(is_fp8_dtype(input->dtype()), "Input tensor has invalid dtype (expected FP8, got ",
                 to_string(input->dtype()), ").");
      break;
    case NVTE_NVFP4_1D_SCALING:
      NVTE_CHECK(is_fp4_dtype(input->dtype()), "Input tensor has invalid dtype (expected FP4, got ",
                 to_string(input->dtype()), ").");
      break;
    default:
      NVTE_ERROR("Invalid scaling mode");
  }

  // Check if scaling factors are non-trivial
  const bool has_rowwise_scale_inv = input->scale_inv.has_data();
  const bool has_columnwise_scale_inv = input->columnwise_scale_inv.has_data();
  NVTE_CHECK(!has_rowwise_scale_inv || !has_columnwise_scale_inv,
             "Input tensor has both row-wise and column-wise scaling factors");
  if (!has_rowwise_scale_inv && !has_columnwise_scale_inv) {
    return;
  }

  // Deduce tensor dims
  int m{0}, k{0};
  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING: {
      if (has_rowwise_scale_inv) {
        NVTE_CHECK(input->scale_inv.shape.size() == 2,
                   "Expected 2D scaling factors, got shape=", input->scale_inv.shape, ".");
        m = input->scale_inv.shape[0];
        k = input->scale_inv.shape[1];
      } else if (has_columnwise_scale_inv) {
        NVTE_CHECK(input->columnwise_scale_inv.shape.size() == 2,
                   "Expected 2D scaling factors, got shape=", input->columnwise_scale_inv.shape,
                   ".");
        m = input->columnwise_scale_inv.shape[1];
        k = input->columnwise_scale_inv.shape[0];
      }
      break;
    }
    case NVTE_NVFP4_1D_SCALING: {
      if (has_rowwise_scale_inv) {
        NVTE_CHECK(input->scale_inv.shape.size() == 2,
                   "Expected 2D scaling factors, got shape=", input->scale_inv.shape, ".");
        m = input->scale_inv.shape[0];
        k = input->scale_inv.shape[1];
      } else if (has_columnwise_scale_inv) {
        NVTE_CHECK(input->columnwise_scale_inv.shape.size() == 2,
                   "Expected 2D scaling factors, got shape=", input->columnwise_scale_inv.shape,
                   ".");
        m = input->columnwise_scale_inv.shape[0];
        k = input->columnwise_scale_inv.shape[1];
      }
      break;
    }
    default:
      NVTE_ERROR("Invalid scaling mode");
  }

  // Check dims
  constexpr int SF_TILE_DIM_M = 128;
  constexpr int SF_TILE_DIM_K = 4;
  NVTE_CHECK(m % SF_TILE_DIM_M == 0, "Input should be padded in M/N dimension!");
  NVTE_CHECK(k % SF_TILE_DIM_K == 0, "Input should be padded in K dimension!");

  // Check that output tensor matches input tensor
  if (has_rowwise_scale_inv) {
    NVTE_CHECK(output->scale_inv.has_data(),
               "Output tensor does not have row-wise scaling factors.");
    NVTE_CHECK(m * k == output->scale_inv.numel(), "Expected output tensor to have ", m * k,
               " row-wise scaling factors, but got shape=", output->scale_inv.shape, ".");
  }
  if (has_columnwise_scale_inv) {
    NVTE_CHECK(output->columnwise_scale_inv.has_data(),
               "Output tensor does not have column-wise scaling factors.");
    NVTE_CHECK(
        m * k == output->columnwise_scale_inv.numel(), "Expected output tensor to have ", m * k,
        " column-wise scaling factors, but got shape=", output->columnwise_scale_inv.shape, ".");
  }

  // Choose swizzle implementation
  bool rowwise_swizzle{false}, columnwise_swizzle{false};
  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING: {
      rowwise_swizzle = has_rowwise_scale_inv;
      columnwise_swizzle = has_columnwise_scale_inv;
      break;
    }
    case NVTE_NVFP4_1D_SCALING: {
      // NVFP4 column-wise data is transposed, so row-wise and
      // column-wise scales have same swizzling format
      rowwise_swizzle = true;
      columnwise_swizzle = false;
      break;
    }
    default:
      NVTE_ERROR("Invalid scaling mode");
  }

  const dim3 block_size(TB_DIM, TB_DIM);
  const int num_tiles_m = m / SF_TILE_DIM_M;
  const int num_tiles_k = k / SF_TILE_DIM_K;

  // Perform row-wise swizzle
  if (rowwise_swizzle) {
    int original_M{0}, original_K{0};
    void *input_scale_inv_ptr{nullptr}, *output_scale_inv_ptr{nullptr};
    switch (scaling_mode) {
      case NVTE_MXFP8_1D_SCALING: {
        original_M = input->flat_first_dim();
        original_K = input->flat_last_dim() / MXFP8_BLOCK_SIZE;
        input_scale_inv_ptr = input->scale_inv.dptr;
        output_scale_inv_ptr = output->scale_inv.dptr;
        break;
      }
      case NVTE_NVFP4_1D_SCALING: {
        if (has_rowwise_scale_inv) {
          original_M = input->flat_first_dim();
          original_K = input->flat_last_dim() / NVFP4_BLOCK_SIZE;
          input_scale_inv_ptr = input->scale_inv.dptr;
          output_scale_inv_ptr = output->scale_inv.dptr;
        } else if (has_columnwise_scale_inv) {
          original_M = input->flat_last_dim();
          original_K = input->flat_first_dim() / NVFP4_BLOCK_SIZE;
          input_scale_inv_ptr = input->columnwise_scale_inv.dptr;
          output_scale_inv_ptr = output->columnwise_scale_inv.dptr;
        }
        break;
      }
      default:
        NVTE_ERROR("Invalid scaling mode");
    }

    const int row_coalesced_slm_size =
        row_coalesced_slm_size_bytes<SF_TILE_DIM_M, SF_TILE_DIM_K>(k, sizeof(uint8_t));
    const bool use_row_coalesced =
        use_blackwell_row_coalesced_swizzle<SF_TILE_DIM_M, SF_TILE_DIM_K>(
            k, original_K, sizeof(uint8_t));
    if (use_row_coalesced) {
      const int m_tiles_per_block =
          row_coalesced_m_tiles_per_block<SF_TILE_DIM_M, SF_TILE_DIM_K>(k, sizeof(uint8_t));
      const int slm_size = row_coalesced_slm_size * m_tiles_per_block;
      static int cached_regular_row_coalesced_batched = -1;
      set_dynamic_smem_if_needed(
          swizzle_row_scaling_coalesced_batched_m_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
          cached_regular_row_coalesced_batched);
      swizzle_row_scaling_coalesced_batched_m_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
          <<<DIVUP(num_tiles_m, m_tiles_per_block), ROW_COALESCED_THREADS, slm_size, stream>>>(
              input_scale_inv_ptr, output_scale_inv_ptr, m, k, original_M, m_tiles_per_block);
    } else {
      const int m_tiles_per_block =
          row_swizzle_m_tiles_per_block<SF_TILE_DIM_M, SF_TILE_DIM_K>(num_tiles_k);
      if (m_tiles_per_block > 1) {
        // Batch as many M-tiles per block as dynamic shared memory allows. This
        // keeps the rowwise path efficient for K-scale counts at and below the
        // 32-tile boundary where the generic kernel leaves most x lanes idle.
        const int slm_size = m_tiles_per_block * num_tiles_k * SF_TILE_DIM_M * SF_TILE_DIM_K *
                             static_cast<int>(sizeof(int8_t));
        dim3 block_size_batched(TB_DIM, m_tiles_per_block);
        dim3 num_blocks_narrow(DIVUP(num_tiles_m, m_tiles_per_block));
        static int cached_regular_row_batched = -1;
        set_dynamic_smem_if_needed(
            swizzle_row_scaling_narrow_k_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
            cached_regular_row_batched);
        swizzle_row_scaling_narrow_k_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<num_blocks_narrow, block_size_batched, slm_size, stream>>>(
                input_scale_inv_ptr, output_scale_inv_ptr, m, k, original_M, original_K);
      } else {
        int vec_load_size = (num_tiles_k - 1) % 4 + 1;
        /* there is no int3 and misaligned if using int4/int2 */
        if (vec_load_size == 3) vec_load_size = 1;
        int n_tiles_in_tb = TB_DIM * vec_load_size;
        dim3 num_blocks(DIVUP(num_tiles_k, n_tiles_in_tb), num_tiles_m);
        int slm_size = n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);

        switch (vec_load_size) {
          case 4: {
            static int cached_regular_row_int4 = -1;
            set_dynamic_smem_if_needed(
                swizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
                cached_regular_row_int4);
            swizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
                <<<num_blocks, block_size, slm_size, stream>>>(
                    input_scale_inv_ptr, output_scale_inv_ptr, m, k, original_M, original_K);
            break;
          }
          case 2: {
            static int cached_regular_row_int2 = -1;
            set_dynamic_smem_if_needed(
                swizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
                cached_regular_row_int2);
            swizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
                <<<num_blocks, block_size, slm_size, stream>>>(
                    input_scale_inv_ptr, output_scale_inv_ptr, m, k, original_M, original_K);
            break;
          }
          case 1: {
            static int cached_regular_row_int1 = -1;
            set_dynamic_smem_if_needed(
                swizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
                cached_regular_row_int1);
            swizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
                <<<num_blocks, block_size, slm_size, stream>>>(
                    input_scale_inv_ptr, output_scale_inv_ptr, m, k, original_M, original_K);
            break;
          }
          default:
            NVTE_ERROR("Not valid vec_load_size.");
            break;
        }
      }
    }
    NVTE_CHECK_CUDA(cudaGetLastError());
  }

  // Perform column-wise swizzle
  if (columnwise_swizzle) {
    const int original_M = input->flat_last_dim();
    const int original_K = input->flat_first_dim() / MXFP8_BLOCK_SIZE;

    const int narrow_m_slm_size =
        TB_DIM * num_tiles_m * SF_TILE_DIM_M * SF_TILE_DIM_K * static_cast<int>(sizeof(int8_t));
    if (use_blackwell_col_coalesced_swizzle(sizeof(uint8_t))) {
      const int k_tiles_per_block = COL_COALESCED_K_TILES_PER_BLOCK;
      const int slm_size =
          col_coalesced_slm_size_bytes<SF_TILE_DIM_M, SF_TILE_DIM_K>(k_tiles_per_block);
      const dim3 num_blocks(DIVUP(num_tiles_k, k_tiles_per_block), num_tiles_m);
      static int cached_regular_col_coalesced = -1;
      set_dynamic_smem_if_needed(
          swizzle_col_scaling_coalesced_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
          cached_regular_col_coalesced);
      swizzle_col_scaling_coalesced_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
          <<<num_blocks, COL_COALESCED_THREADS, slm_size, stream>>>(
              input->columnwise_scale_inv.dptr, output->columnwise_scale_inv.dptr, m, k,
              original_M, original_K, k_tiles_per_block);
    } else if (num_tiles_m < TB_DIM && narrow_m_slm_size <= get_max_dynamic_smem()) {
      // Narrow-M: batch TB_DIM K-tiles per block, fully utilizing all threads.
      dim3 num_blocks_narrow(DIVUP(num_tiles_k, TB_DIM));
      static int cached_regular_col_narrow_m = -1;
      set_dynamic_smem_if_needed(swizzle_col_scaling_narrow_m_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                 narrow_m_slm_size, cached_regular_col_narrow_m);
      swizzle_col_scaling_narrow_m_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
          <<<num_blocks_narrow, block_size, narrow_m_slm_size, stream>>>(
              input->columnwise_scale_inv.dptr, output->columnwise_scale_inv.dptr, m, k, original_M,
              original_K);
    } else {
      int vec_load_size = (num_tiles_m - 1) % 4 + 1;
      if (vec_load_size == 3) vec_load_size = 1; /* no int3 and misaligned if using int4/int2 */
      int n_tiles_in_tb = TB_DIM * vec_load_size;
      dim3 num_blocks(DIVUP(num_tiles_k, TB_DIM), DIVUP(num_tiles_m, vec_load_size));
      int slm_size = n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);

      switch (vec_load_size) {
        case 4: {
          static int cached_regular_col_int4 = -1;
          set_dynamic_smem_if_needed(swizzle_col_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                     slm_size, cached_regular_col_int4);
          swizzle_col_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(input->columnwise_scale_inv.dptr,
                                                             output->columnwise_scale_inv.dptr, m,
                                                             k, original_M, original_K);
          break;
        }
        case 2: {
          static int cached_regular_col_int2 = -1;
          set_dynamic_smem_if_needed(swizzle_col_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                     slm_size, cached_regular_col_int2);
          swizzle_col_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(input->columnwise_scale_inv.dptr,
                                                             output->columnwise_scale_inv.dptr, m,
                                                             k, original_M, original_K);
          break;
        }
        case 1: {
          static int cached_regular_col_int1 = -1;
          set_dynamic_smem_if_needed(swizzle_col_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                     slm_size, cached_regular_col_int1);
          swizzle_col_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(input->columnwise_scale_inv.dptr,
                                                             output->columnwise_scale_inv.dptr, m,
                                                             k, original_M, original_K);
          break;
        }
        default:
          NVTE_ERROR("Not valid vec_load_size.");
          break;
      }
    }
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
void launch_multi_tensor_swizzle_scaling_factors(MultiSwizzleArgs& kernel_args,
                                                 const int vec_load_size, const bool is_rowwise,
                                                 const bool use_narrow_k, const bool use_narrow_m,
                                                 cudaStream_t stream) {
  dim3 block_size(TB_DIM, TB_DIM);
  bool launched = false;

  if (is_rowwise && use_narrow_k) {
    // Batched rowwise path: each block handles as many M-tiles as fit in dynamic smem.
    // slm_size depends on num_tiles_k, which can vary per tensor; use the max.
    int max_num_tiles_k = 0;
    for (size_t j = 0; j < kernel_args.num_tensors; j++) {
      const int num_tiles_k = kernel_args.k_list[j] / SF_TILE_DIM_K;
      max_num_tiles_k = std::max(max_num_tiles_k, num_tiles_k);
    }
    const int m_tiles_per_block =
        row_swizzle_m_tiles_per_block<SF_TILE_DIM_M, SF_TILE_DIM_K>(max_num_tiles_k);
    if (m_tiles_per_block > 1) {
      for (size_t j = 0; j < kernel_args.num_tensors; j++) {
        const int num_tiles_m = kernel_args.m_list[j] / SF_TILE_DIM_M;
        kernel_args.block_range[j + 1] =
            kernel_args.block_range[j] + DIVUP(num_tiles_m, m_tiles_per_block);
      }
      const int slm_size = m_tiles_per_block * max_num_tiles_k * SF_TILE_DIM_M * SF_TILE_DIM_K *
                           static_cast<int>(sizeof(int8_t));
      const int num_blocks = kernel_args.block_range[kernel_args.num_tensors];
      dim3 block_size_batched(TB_DIM, m_tiles_per_block);

      static int cached_narrow_k = -1;
      set_dynamic_smem_if_needed(
          multi_tensor_swizzle_row_scaling_narrow_k_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
          cached_narrow_k);
      multi_tensor_swizzle_row_scaling_narrow_k_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
          <<<num_blocks, block_size_batched, slm_size, stream>>>(kernel_args);
      launched = true;
    }
  }

  if (!launched && !is_rowwise && use_narrow_m) {
    // Narrow-M path: each block handles TB_DIM K-tiles with full thread utilization.
    // slm_size depends on num_tiles_m, which can vary per tensor; use the max.
    int max_num_tiles_m = 0;
    for (size_t j = 0; j < kernel_args.num_tensors; j++) {
      const int num_tiles_m = kernel_args.m_list[j] / SF_TILE_DIM_M;
      const int num_tiles_k = kernel_args.k_list[j] / SF_TILE_DIM_K;
      max_num_tiles_m = std::max(max_num_tiles_m, num_tiles_m);
      kernel_args.block_range[j + 1] = kernel_args.block_range[j] + DIVUP(num_tiles_k, TB_DIM);
    }
    int slm_size = TB_DIM * max_num_tiles_m * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);
    const int num_blocks = kernel_args.block_range[kernel_args.num_tensors];

    static int cached_narrow_m = -1;
    set_dynamic_smem_if_needed(
        multi_tensor_swizzle_col_scaling_narrow_m_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
        cached_narrow_m);
    multi_tensor_swizzle_col_scaling_narrow_m_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
        <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
    launched = true;
  }

  if (!launched) {
    int n_tiles_in_tb = TB_DIM * vec_load_size;
    int slm_size = n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);
    /* Calculate number of CUDA blocks needed for each tensor.
    * We have to do it here because we have to iterate over all tensors in this batch to
    * get the minimum vec_load_size.
    */
    for (size_t j = 0; j < kernel_args.num_tensors; j++) {
      const int m = kernel_args.m_list[j];
      const int k = kernel_args.k_list[j];
      int num_tiles_m = m / SF_TILE_DIM_M;
      int num_tiles_k = k / SF_TILE_DIM_K;
      if (is_rowwise) {
        kernel_args.block_range[j + 1] =
            kernel_args.block_range[j] + DIVUP(num_tiles_k, n_tiles_in_tb) * num_tiles_m;
      } else {
        kernel_args.block_range[j + 1] =
            kernel_args.block_range[j] +
            DIVUP(num_tiles_k, TB_DIM) * DIVUP(num_tiles_m, vec_load_size);
      }
    }
    const int num_blocks = kernel_args.block_range[kernel_args.num_tensors];

    static int cached_row_int4 = -1, cached_row_int2 = -1, cached_row_int1 = -1;
    static int cached_col_int4 = -1, cached_col_int2 = -1, cached_col_int1 = -1;

    if (is_rowwise) {
      switch (vec_load_size) {
        case 4:
          set_dynamic_smem_if_needed(
              multi_tensor_swizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
              cached_row_int4);
          multi_tensor_swizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        case 2:
          set_dynamic_smem_if_needed(
              multi_tensor_swizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
              cached_row_int2);
          multi_tensor_swizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        case 1:
          set_dynamic_smem_if_needed(
              multi_tensor_swizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
              cached_row_int1);
          multi_tensor_swizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        default:
          NVTE_ERROR("Not valid vec_load_size.");
          break;
      }
    } else {
      switch (vec_load_size) {
        case 4:
          set_dynamic_smem_if_needed(
              multi_tensor_swizzle_col_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
              cached_col_int4);
          multi_tensor_swizzle_col_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        case 2:
          set_dynamic_smem_if_needed(
              multi_tensor_swizzle_col_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
              cached_col_int2);
          multi_tensor_swizzle_col_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        case 1:
          set_dynamic_smem_if_needed(
              multi_tensor_swizzle_col_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
              cached_col_int1);
          multi_tensor_swizzle_col_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        default:
          NVTE_ERROR("Not valid vec_load_size.");
          break;
      }
    }
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
void launch_multi_tensor_unswizzle_scaling_factors(MultiSwizzleArgs& kernel_args,
                                                   const int vec_load_size, const bool is_rowwise,
                                                   cudaStream_t stream) {
  int n_tiles_in_tb = TB_DIM * vec_load_size;
  int slm_size = n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);
  for (size_t j = 0; j < kernel_args.num_tensors; j++) {
    const int m = kernel_args.m_list[j];
    const int k = kernel_args.k_list[j];
    int num_tiles_m = m / SF_TILE_DIM_M;
    int num_tiles_k = k / SF_TILE_DIM_K;
    if (is_rowwise) {
      kernel_args.block_range[j + 1] =
          kernel_args.block_range[j] + DIVUP(num_tiles_k, n_tiles_in_tb) * num_tiles_m;
    } else {
      kernel_args.block_range[j + 1] =
          kernel_args.block_range[j] +
          DIVUP(num_tiles_k, TB_DIM) * DIVUP(num_tiles_m, vec_load_size);
    }
  }

  int num_blocks = kernel_args.block_range[kernel_args.num_tensors];
  if (num_blocks > 0) {
    dim3 block_size(TB_DIM, TB_DIM);
    if (is_rowwise) {
      switch (vec_load_size) {
        case 4:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              multi_tensor_unswizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          multi_tensor_unswizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        case 2:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              multi_tensor_unswizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          multi_tensor_unswizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        case 1:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              multi_tensor_unswizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          multi_tensor_unswizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        default:
          NVTE_ERROR("Not valid vec_load_size.");
          break;
      }
    } else {
      switch (vec_load_size) {
        case 4:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              multi_tensor_unswizzle_col_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          multi_tensor_unswizzle_col_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        case 2:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              multi_tensor_unswizzle_col_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          multi_tensor_unswizzle_col_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        case 1:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              multi_tensor_unswizzle_col_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          multi_tensor_unswizzle_col_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        default:
          NVTE_ERROR("Not valid vec_load_size.");
          break;
      }
    }
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

void multi_tensor_swizzle_scaling_factors(const std::vector<Tensor*>& input,
                                          std::vector<Tensor*>& output, cudaStream_t stream,
                                          bool check_scale_inv_shapes) {
  auto num_tensors = input.size();
  bool all_has_data = true;
  bool all_has_columnwise_data = true;
  bool all_nvfp4 = true;
  for (size_t i = 0; i < num_tensors; i++) {
    auto scaling_mode = input[i]->scaling_mode;
    auto is_fp8 = is_fp8_dtype(input[i]->dtype());
    auto is_fp4 = is_fp4_dtype(input[i]->dtype());
    NVTE_CHECK(
        (is_fp8 && is_mxfp8_scaling(scaling_mode)) || (is_fp4 && is_nvfp4_scaling(scaling_mode)),
        "Not implemented scaling mode " + to_string(scaling_mode) + ".");
    NVTE_CHECK(!input[i]->with_gemm_swizzled_scales,
               "Expected input tensors with scales in compact format.");
    NVTE_CHECK(output[i]->with_gemm_swizzled_scales,
               "Expected output tensors with scales in GEMM swizzled format.");

    // We don't allow empty tensors. They should be filtered out before calling this function.
    NVTE_CHECK(input[i]->numel() != 0, "Tensor input[", i, "] is empty.");
    CheckInputTensor(*input[i], "scaling_factor_input[" + std::to_string(i) + "]",
                     check_scale_inv_shapes);
    CheckInputTensor(*output[i], "scaling_factor_output[" + std::to_string(i) + "]",
                     check_scale_inv_shapes);
    all_has_data = all_has_data && input[i]->scale_inv.has_data();
    all_has_columnwise_data =
        (all_has_columnwise_data && input[i]->columnwise_scale_inv.has_data());
    all_nvfp4 = all_nvfp4 && is_nvfp4_scaling(scaling_mode);
  }
  NVTE_CHECK(all_has_data || all_has_columnwise_data,
             "All tensors should have data or columnwise data.");
  NVTE_CHECK(!all_has_data || !all_has_columnwise_data,
             "All tensors have both data and columnwise data.");

  const bool rowwise_swizzle = all_has_data || all_nvfp4;
  const bool columnwise_swizzle = all_has_columnwise_data && !all_nvfp4;

  constexpr int SF_TILE_DIM_M = 128;
  constexpr int SF_TILE_DIM_K = 4;
  if (rowwise_swizzle) {
    MultiSwizzleArgs kernel_args;
    kernel_args.num_tensors = 0;
    kernel_args.block_range[0] = 0;
    int vec_load_size = 4;
    bool all_narrow_k = true;
    for (size_t i = 0; i < num_tensors; i++) {
      //Launch kernel if argument struct is full
      if (kernel_args.num_tensors == kMaxTensorsPerKernel) {
        // There is no int3 and misaligned if using int4/int2.
        if (vec_load_size == 3) vec_load_size = 1;
        launch_multi_tensor_swizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
            kernel_args, vec_load_size, true, all_narrow_k, false, stream);
        // Reset the argument struct and vec_load_size
        kernel_args.num_tensors = 0;
        vec_load_size = 4;
        all_narrow_k = true;
      }

      int m, k;

      if (all_has_data) {
        m = input[i]->scale_inv.shape[0];
        k = input[i]->scale_inv.shape[1];
      } else {
        NVTE_CHECK(all_nvfp4, "When doing rowwise swizzle with rowwise data, it has to be NVFP4");
        m = input[i]->columnwise_scale_inv.shape[0];
        k = input[i]->columnwise_scale_inv.shape[1];
      }

      NVTE_CHECK(m % SF_TILE_DIM_M == 0, "Input should be padded in M/N dimension!");
      NVTE_CHECK(k % SF_TILE_DIM_K == 0, "Input should be padded in K dimension!");
      NVTE_CHECK(k > 0, "Input scale inverse should be 2D!");

      if (all_has_data) {
        NVTE_CHECK(output[i]->scale_inv.has_data(), "Output tensor ", i,
                   " does not have row-wise scaling factors.");
        NVTE_CHECK(m * k == output[i]->scale_inv.numel(), "Expected output tensor ", i, " to have ",
                   m * k, " row-wise scaling factors, but got shape=", output[i]->scale_inv.shape,
                   ".");
      }
      if (all_has_columnwise_data) {
        NVTE_CHECK(output[i]->columnwise_scale_inv.has_data(), "Output tensor ", i,
                   " does not have column-wise scaling factors.");
        NVTE_CHECK(m * k == output[i]->columnwise_scale_inv.numel(), "Expected output tensor ", i,
                   " to have ", m * k, " column-wise scaling factors, but got shape=",
                   output[i]->columnwise_scale_inv.shape, ".");
      }

      int num_tiles_k = k / SF_TILE_DIM_K;
      all_narrow_k =
          all_narrow_k &&
          (row_swizzle_m_tiles_per_block<SF_TILE_DIM_M, SF_TILE_DIM_K>(num_tiles_k) > 1);
      int vec_load_size_i = (num_tiles_k - 1) % 4 + 1;
      // We use the minimum vec_load_size across all tensors.
      // TODO(zhongbo): fix vec_load_size for NVFP4
      // Current unit test won't capture this issue, but in E2E
      // using vec_load_size = 1 other than 1 will lead to mis-aligned
      // address error in MOE training
      vec_load_size = all_nvfp4 ? 1 : std::min(vec_load_size, vec_load_size_i);

      const int pos = kernel_args.num_tensors;
      kernel_args.m_list[pos] = m;
      kernel_args.k_list[pos] = k;
      if (!all_nvfp4 || all_has_data) {
        int block_scale_size = all_nvfp4 ? NVFP4_BLOCK_SIZE : MXFP8_BLOCK_SIZE;
        kernel_args.input_list[pos] = const_cast<void*>(input[i]->scale_inv.dptr);
        kernel_args.output_list[pos] = output[i]->scale_inv.dptr;
        kernel_args.original_m_list[pos] = input[i]->flat_first_dim();
        kernel_args.original_k_list[pos] = input[i]->flat_last_dim() / block_scale_size;
      } else {
        kernel_args.input_list[pos] = const_cast<void*>(input[i]->columnwise_scale_inv.dptr);
        kernel_args.output_list[pos] = output[i]->columnwise_scale_inv.dptr;
        kernel_args.original_m_list[pos] = input[i]->flat_last_dim();
        kernel_args.original_k_list[pos] = input[i]->flat_first_dim() / NVFP4_BLOCK_SIZE;
      }
      kernel_args.num_tensors++;
    }
    // Launch the remaining tensors
    // There is no int3 and misaligned if using int4/int2.
    if (vec_load_size == 3) vec_load_size = 1;
    launch_multi_tensor_swizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
        kernel_args, vec_load_size, true, all_narrow_k, false, stream);
  }

  if (columnwise_swizzle) {
    // NVFP4 shouldn't end up here because it only needs rowwise swizzle
    NVTE_CHECK(!all_nvfp4, "NVFP4 shouldn't end up here because it only needs rowwise swizzle");

    MultiSwizzleArgs kernel_args;
    kernel_args.num_tensors = 0;
    kernel_args.block_range[0] = 0;
    int vec_load_size = 4;
    bool all_narrow_m = true;
    for (size_t i = 0; i < num_tensors; i++) {
      //Launch kernel if argument struct is full
      if (kernel_args.num_tensors == kMaxTensorsPerKernel) {
        // There is no int3 and misaligned if using int4/int2.
        if (vec_load_size == 3) vec_load_size = 1;
        launch_multi_tensor_swizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
            kernel_args, vec_load_size, false, false, all_narrow_m, stream);
        // Reset the argument struct and vec_load_size
        kernel_args.num_tensors = 0;
        vec_load_size = 4;
        all_narrow_m = true;
      }
      const int m = input[i]->columnwise_scale_inv.shape[1];
      const int k = input[i]->columnwise_scale_inv.shape[0];

      NVTE_CHECK(m % SF_TILE_DIM_M == 0, "Input should be padded in M/N dimension!");
      NVTE_CHECK(k % SF_TILE_DIM_K == 0, "Input should be padded in K dimension!");
      NVTE_CHECK(k > 0, "Input scale inverse should be 2D!");
      NVTE_CHECK(m * k == std::accumulate(output[i]->columnwise_scale_inv.shape.begin(),
                                          output[i]->columnwise_scale_inv.shape.end(), 1,
                                          std::multiplies<int>()),
                 "Input.columnwise_scale_inv size is not equal to "
                 "Output.columnwise_scale_inv size!");

      int num_tiles_m = m / SF_TILE_DIM_M;
      int num_tiles_k = k / SF_TILE_DIM_K;
      const int narrow_m_slm =
          TB_DIM * num_tiles_m * SF_TILE_DIM_M * SF_TILE_DIM_K * static_cast<int>(sizeof(int8_t));
      all_narrow_m =
          all_narrow_m && (num_tiles_m < TB_DIM) && (narrow_m_slm <= get_max_dynamic_smem());
      int vec_load_size_i = (num_tiles_k - 1) % 4 + 1;
      // We use the minimum vec_load_size across all tensors.
      vec_load_size = std::min(vec_load_size, vec_load_size_i);

      const int pos = kernel_args.num_tensors;
      kernel_args.input_list[pos] = const_cast<void*>(input[i]->columnwise_scale_inv.dptr);
      kernel_args.output_list[pos] = output[i]->columnwise_scale_inv.dptr;
      kernel_args.m_list[pos] = m;
      kernel_args.k_list[pos] = k;
      kernel_args.original_m_list[pos] = input[i]->flat_last_dim();
      kernel_args.original_k_list[pos] = input[i]->flat_first_dim() / MXFP8_BLOCK_SIZE;
      kernel_args.num_tensors++;
    }
    // Launch the remaining tensors
    // There is no int3 and misaligned if using int4/int2.
    if (vec_load_size == 3) vec_load_size = 1;
    launch_multi_tensor_swizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
        kernel_args, vec_load_size, false, false, all_narrow_m, stream);
  }
}

void unswizzle_scaling_factors(const Tensor* input, Tensor* output, cudaStream_t stream) {
  const auto& scaling_mode = output->scaling_mode;
  NVTE_CHECK(scaling_mode == NVTE_MXFP8_1D_SCALING || scaling_mode == NVTE_NVFP4_1D_SCALING,
             "Output tensor has invalid scaling mode (", to_string(output->scaling_mode), ").");

  CheckInputTensor(*input, "scaling_factor_input");
  CheckInputTensor(*output, "scaling_factor_output");
  NVTE_CHECK(input->with_gemm_swizzled_scales, "Expected input tensor with swizzled scales.");
  NVTE_CHECK(!output->with_gemm_swizzled_scales,
             "Expected output tensor in row-major compact format.");
  NVTE_CHECK(input->scaling_mode == scaling_mode,
             "Input and output tensors must have matching scaling modes, but got ",
             to_string(input->scaling_mode), " and ", to_string(output->scaling_mode), ".");

  const bool has_rowwise_scale_inv = output->scale_inv.has_data();
  const bool has_columnwise_scale_inv = output->columnwise_scale_inv.has_data();
  NVTE_CHECK(!has_rowwise_scale_inv || !has_columnwise_scale_inv,
             "Output tensor has both row-wise and column-wise scaling factors");
  if (!has_rowwise_scale_inv && !has_columnwise_scale_inv) {
    return;
  }
  if (has_rowwise_scale_inv) {
    NVTE_CHECK(input->scale_inv.has_data(),
               "Output tensor requests row-wise scaling factors, but input tensor does not "
               "provide them.");
  } else if (has_columnwise_scale_inv) {
    NVTE_CHECK(input->columnwise_scale_inv.has_data(),
               "Output tensor requests column-wise scaling factors, but input tensor does not "
               "provide them.");
  }

  constexpr int SF_TILE_DIM_M = 128;
  constexpr int SF_TILE_DIM_K = 4;
  const dim3 block_size(TB_DIM, TB_DIM);

  int m{0}, k{0};
  void* input_ptr{nullptr};
  void* output_ptr{nullptr};
  bool rowwise{false};

  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING: {
      NVTE_CHECK(is_fp8_dtype(input->dtype()), "Input tensor has invalid dtype (expected FP8, got ",
                 to_string(input->dtype()), ").");
      if (has_rowwise_scale_inv) {
        NVTE_CHECK(output->scale_inv.shape.size() == 2,
                   "Expected 2D scaling factors, got shape=", output->scale_inv.shape, ".");
        m = output->scale_inv.shape[0];
        k = output->scale_inv.shape[1];
        NVTE_CHECK(static_cast<size_t>(m) * k == input->scale_inv.numel(),
                   "Expected input tensor to have ", static_cast<size_t>(m) * k,
                   " row-wise scaling factors, but got shape=", input->scale_inv.shape, ".");
        NVTE_CHECK(static_cast<size_t>(m) * k == output->scale_inv.numel(),
                   "Expected output tensor to have ", static_cast<size_t>(m) * k,
                   " row-wise scaling factors, but got shape=", output->scale_inv.shape, ".");
        input_ptr = input->scale_inv.dptr;
        output_ptr = output->scale_inv.dptr;
        rowwise = true;
      } else if (has_columnwise_scale_inv) {
        NVTE_CHECK(output->columnwise_scale_inv.shape.size() == 2,
                   "Expected 2D scaling factors, got shape=", output->columnwise_scale_inv.shape,
                   ".");
        m = output->columnwise_scale_inv.shape[1];
        k = output->columnwise_scale_inv.shape[0];
        NVTE_CHECK(
            static_cast<size_t>(m) * k == input->columnwise_scale_inv.numel(),
            "Expected input tensor to have ", static_cast<size_t>(m) * k,
            " column-wise scaling factors, but got shape=", input->columnwise_scale_inv.shape, ".");
        NVTE_CHECK(static_cast<size_t>(m) * k == output->columnwise_scale_inv.numel(),
                   "Expected output tensor to have ", static_cast<size_t>(m) * k,
                   " column-wise scaling factors, but got shape=",
                   output->columnwise_scale_inv.shape, ".");
        input_ptr = input->columnwise_scale_inv.dptr;
        output_ptr = output->columnwise_scale_inv.dptr;
        rowwise = false;
      }
      break;
    }
    case NVTE_NVFP4_1D_SCALING: {
      NVTE_CHECK(is_fp4_dtype(input->dtype()), "Input tensor has invalid dtype (expected FP4, got ",
                 to_string(input->dtype()), ").");
      // NVFP4: always unswizzle rowwise regardless of which scale buffer holds the data
      if (has_rowwise_scale_inv) {
        NVTE_CHECK(output->scale_inv.shape.size() == 2,
                   "Expected 2D scaling factors, got shape=", output->scale_inv.shape, ".");
        m = output->scale_inv.shape[0];
        k = output->scale_inv.shape[1];
        // Example for NVFP4 rowwise path:
        NVTE_CHECK(static_cast<size_t>(m) * k == input->scale_inv.numel(),
                   "Expected input tensor to have ", static_cast<size_t>(m) * k,
                   " row-wise scaling factors, but got shape=", input->scale_inv.shape, ".");
        NVTE_CHECK(static_cast<size_t>(m) * k == output->scale_inv.numel(),
                   "Expected output tensor to have ", static_cast<size_t>(m) * k,
                   " row-wise scaling factors, but got shape=", output->scale_inv.shape, ".");
        input_ptr = input->scale_inv.dptr;
        output_ptr = output->scale_inv.dptr;
      } else if (has_columnwise_scale_inv) {
        NVTE_CHECK(output->columnwise_scale_inv.shape.size() == 2,
                   "Expected 2D scaling factors, got shape=", output->columnwise_scale_inv.shape,
                   ".");
        m = output->columnwise_scale_inv.shape[0];
        k = output->columnwise_scale_inv.shape[1];
        NVTE_CHECK(
            static_cast<size_t>(m) * k == input->columnwise_scale_inv.numel(),
            "Expected input tensor to have ", static_cast<size_t>(m) * k,
            " column-wise scaling factors, but got shape=", input->columnwise_scale_inv.shape, ".");
        NVTE_CHECK(static_cast<size_t>(m) * k == output->columnwise_scale_inv.numel(),
                   "Expected output tensor to have ", static_cast<size_t>(m) * k,
                   " column-wise scaling factors, but got shape=",
                   output->columnwise_scale_inv.shape, ".");
        input_ptr = input->columnwise_scale_inv.dptr;
        output_ptr = output->columnwise_scale_inv.dptr;
      }
      rowwise = true;
      break;
    }
    default:
      NVTE_ERROR("Invalid scaling mode");
  }

  NVTE_CHECK(m % SF_TILE_DIM_M == 0, "Output should be padded in M/N dimension!");
  NVTE_CHECK(k % SF_TILE_DIM_K == 0, "Output should be padded in K dimension!");

  const int num_tiles_m = m / SF_TILE_DIM_M;
  const int num_tiles_k = k / SF_TILE_DIM_K;

  auto launch_unswizzle = [&](int vec_load_size, const dim3& num_blocks, int slm_size) {
    switch (vec_load_size) {
      case 4:
        NVTE_CHECK_CUDA(
            cudaFuncSetAttribute(unswizzle_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
        unswizzle_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<num_blocks, block_size, slm_size, stream>>>(input_ptr, output_ptr, m, k, rowwise);
        break;
      case 2:
        NVTE_CHECK_CUDA(
            cudaFuncSetAttribute(unswizzle_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
        unswizzle_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<num_blocks, block_size, slm_size, stream>>>(input_ptr, output_ptr, m, k, rowwise);
        break;
      case 1:
        NVTE_CHECK_CUDA(
            cudaFuncSetAttribute(unswizzle_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
        unswizzle_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<num_blocks, block_size, slm_size, stream>>>(input_ptr, output_ptr, m, k, rowwise);
        break;
      default:
        NVTE_ERROR("Not valid vec_load_size.");
    }
    NVTE_CHECK_CUDA(cudaGetLastError());
  };

  int vec_load_size = rowwise ? (num_tiles_k - 1) % 4 + 1 : (num_tiles_m - 1) % 4 + 1;
  if (vec_load_size == 3) vec_load_size = 1;
  int n_tiles_in_tb = TB_DIM * vec_load_size;
  dim3 num_blocks = rowwise ? dim3(DIVUP(num_tiles_k, n_tiles_in_tb), num_tiles_m)
                            : dim3(DIVUP(num_tiles_k, TB_DIM), DIVUP(num_tiles_m, vec_load_size));
  int slm_size = n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);
  launch_unswizzle(vec_load_size, num_blocks, slm_size);
}

void multi_tensor_unswizzle_scaling_factors(const std::vector<Tensor*>& input,
                                            std::vector<Tensor*>& output, cudaStream_t stream) {
  size_t num_tensors = output.size();
  const auto& first_scaling_mode = output[0]->scaling_mode;

  constexpr int SF_TILE_DIM_M = 128;
  constexpr int SF_TILE_DIM_K = 4;

  bool all_has_data = true;
  bool all_has_columnwise_data = true;
  bool all_nvfp4 = true;
  for (size_t i = 0; i < num_tensors; i++) {
    const auto scaling_mode = output[i]->scaling_mode;
    const auto is_fp8 = is_fp8_dtype(input[i]->dtype());
    const auto is_fp4 = is_fp4_dtype(input[i]->dtype());

    NVTE_CHECK(scaling_mode == first_scaling_mode,
               "All tensors should have the same scaling mode in multi-tensor unswizzle.");
    NVTE_CHECK(
        (is_fp8 && is_mxfp8_scaling(scaling_mode)) || (is_fp4 && is_nvfp4_scaling(scaling_mode)),
        "Not implemented scaling mode " + to_string(scaling_mode) + ".");
    NVTE_CHECK(input[i]->with_gemm_swizzled_scales,
               "Expected input tensors with scales in GEMM swizzled format.");
    NVTE_CHECK(!output[i]->with_gemm_swizzled_scales,
               "Expected output tensors with scales in compact format.");
    NVTE_CHECK(input[i]->numel() != 0, "Tensor input[", i, "] is empty.");
    CheckInputTensor(*input[i], "scaling_factor_input[" + std::to_string(i) + "]");
    CheckInputTensor(*output[i], "scaling_factor_output[" + std::to_string(i) + "]");

    all_has_data = all_has_data && output[i]->scale_inv.has_data();
    all_has_columnwise_data =
        (all_has_columnwise_data && output[i]->columnwise_scale_inv.has_data());
    all_nvfp4 = all_nvfp4 && is_nvfp4_scaling(scaling_mode);
  }
  NVTE_CHECK(all_has_data || all_has_columnwise_data,
             "All tensors should have data or columnwise data.");
  NVTE_CHECK(!all_has_data || !all_has_columnwise_data,
             "All tensors have both data and columnwise data.");

  const bool rowwise_unswizzle = all_has_data || all_nvfp4;
  const bool columnwise_unswizzle = all_has_columnwise_data && !all_nvfp4;

  if (rowwise_unswizzle) {
    MultiSwizzleArgs kernel_args;
    kernel_args.num_tensors = 0;
    kernel_args.block_range[0] = 0;
    int vec_load_size = 4;
    for (size_t i = 0; i < num_tensors; i++) {
      if (kernel_args.num_tensors == kMaxTensorsPerKernel) {
        if (vec_load_size == 3) vec_load_size = 1;
        launch_multi_tensor_unswizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
            kernel_args, vec_load_size, true, stream);
        kernel_args.num_tensors = 0;
        vec_load_size = 4;
      }
      int m, k;
      if (all_has_data) {
        NVTE_CHECK(input[i]->scale_inv.has_data(), "Input tensor ", i,
                   " does not have row-wise scaling factors.");
        NVTE_CHECK(output[i]->scale_inv.shape.size() == 2, "Expected output tensor ", i,
                   " to have ", "2D scaling factors, got shape=", output[i]->scale_inv.shape, ".");
        m = output[i]->scale_inv.shape[0];
        k = output[i]->scale_inv.shape[1];
        NVTE_CHECK(m * k == input[i]->scale_inv.numel(), "Expected input tensor ", i, " to have ",
                   m * k, " row-wise scaling factors, but got shape=", input[i]->scale_inv.shape,
                   ".");
      }

      if (all_has_columnwise_data) {
        NVTE_CHECK(all_nvfp4,
                   "When doing rowwise unswizzle with columnwise data, it has to be NVFP4");
        NVTE_CHECK(input[i]->columnwise_scale_inv.has_data(), "Input tensor ", i,
                   " does not have column-wise scaling factors.");
        NVTE_CHECK(output[i]->columnwise_scale_inv.shape.size() == 2, "Expected output tensor ", i,
                   " to have ",
                   "2D scaling factors, got shape=", output[i]->columnwise_scale_inv.shape, ".");
        m = output[i]->columnwise_scale_inv.shape[0];
        k = output[i]->columnwise_scale_inv.shape[1];
        NVTE_CHECK(m * k == input[i]->columnwise_scale_inv.numel(), "Expected input tensor ", i,
                   " to have ", m * k, " column-wise scaling factors, but got shape=",
                   input[i]->columnwise_scale_inv.shape, ".");
      }

      NVTE_CHECK(m % SF_TILE_DIM_M == 0, "Output should be padded in M/N dimension!");
      NVTE_CHECK(k % SF_TILE_DIM_K == 0, "Output should be padded in K dimension!");
      NVTE_CHECK(k > 0, "Output scale inverse should be 2D!");

      int num_tiles_k = k / SF_TILE_DIM_K;
      int vec_load_size_i = (num_tiles_k - 1) % 4 + 1;
      vec_load_size = all_nvfp4 ? 1 : std::min(vec_load_size, vec_load_size_i);

      const int pos = kernel_args.num_tensors;
      kernel_args.m_list[pos] = m;
      kernel_args.k_list[pos] = k;
      if (!all_nvfp4 || all_has_data) {
        kernel_args.input_list[pos] = const_cast<void*>(input[i]->scale_inv.dptr);
        kernel_args.output_list[pos] = output[i]->scale_inv.dptr;
      } else {
        kernel_args.input_list[pos] = const_cast<void*>(input[i]->columnwise_scale_inv.dptr);
        kernel_args.output_list[pos] = output[i]->columnwise_scale_inv.dptr;
      }
      kernel_args.num_tensors++;
    }
    if (vec_load_size == 3) vec_load_size = 1;
    launch_multi_tensor_unswizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
        kernel_args, vec_load_size, true, stream);
  }

  if (columnwise_unswizzle) {
    NVTE_CHECK(!all_nvfp4, "NVFP4 shouldn't end up here because it only needs rowwise unswizzle");

    MultiSwizzleArgs kernel_args;
    kernel_args.num_tensors = 0;
    kernel_args.block_range[0] = 0;
    int vec_load_size = 4;
    for (size_t i = 0; i < num_tensors; i++) {
      if (kernel_args.num_tensors == kMaxTensorsPerKernel) {
        if (vec_load_size == 3) vec_load_size = 1;
        launch_multi_tensor_unswizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
            kernel_args, vec_load_size, false, stream);
        kernel_args.num_tensors = 0;
        vec_load_size = 4;
      }
      NVTE_CHECK(output[i]->columnwise_scale_inv.shape.size() == 2, "Expected output tensor ", i,
                 " to have ",
                 "2D scaling factors, got shape=", output[i]->columnwise_scale_inv.shape, ".");
      const int m = output[i]->columnwise_scale_inv.shape[1];
      const int k = output[i]->columnwise_scale_inv.shape[0];

      NVTE_CHECK(m % SF_TILE_DIM_M == 0, "Output should be padded in M/N dimension!");
      NVTE_CHECK(k % SF_TILE_DIM_K == 0, "Output should be padded in K dimension!");
      NVTE_CHECK(k > 0, "Output scale inverse should be 2D!");
      NVTE_CHECK(m * k == std::accumulate(input[i]->columnwise_scale_inv.shape.begin(),
                                          input[i]->columnwise_scale_inv.shape.end(), 1,
                                          std::multiplies<int>()),
                 "Input.columnwise_scale_inv size is not equal to "
                 "Output.columnwise_scale_inv size!");

      int num_tiles_k = k / SF_TILE_DIM_K;
      int vec_load_size_i = (num_tiles_k - 1) % 4 + 1;
      vec_load_size = std::min(vec_load_size, vec_load_size_i);

      const int pos = kernel_args.num_tensors;
      kernel_args.input_list[pos] = const_cast<void*>(input[i]->columnwise_scale_inv.dptr);
      kernel_args.output_list[pos] = output[i]->columnwise_scale_inv.dptr;
      kernel_args.m_list[pos] = m;
      kernel_args.k_list[pos] = k;
      kernel_args.num_tensors++;
    }
    if (vec_load_size == 3) vec_load_size = 1;
    launch_multi_tensor_unswizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
        kernel_args, vec_load_size, false, stream);
  }
}
}  // namespace transformer_engine

/*
* WIP (Phuong):
*   - Opt for bank conflicts
*   - Adding swizzle for 2d-block scaling.
*/
void nvte_swizzle_scaling_factors(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_swizzle_scaling_factors);
  using namespace transformer_engine;
  swizzle_scaling_factors(convertNVTETensorCheck(input), convertNVTETensorCheck(output), stream);
}

void nvte_multi_tensor_swizzle_scaling_factors(const NVTETensor* inputs, NVTETensor* outputs,
                                               const size_t num_tensors, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_swizzle_scaling_factors);
  using namespace transformer_engine;
  NVTE_CHECK(num_tensors > 0, "Number of tensors should be greater than 0.");
  std::vector<Tensor*> input_list, output_list;
  for (size_t i = 0; i < num_tensors; i++) {
    input_list.push_back(convertNVTETensorCheck(inputs[i]));
    output_list.push_back(convertNVTETensorCheck(outputs[i]));
  }
  multi_tensor_swizzle_scaling_factors(input_list, output_list, stream,
                                       /*check_scale_inv_shapes=*/true);
}

void nvte_multi_tensor_swizzle_scaling_factors_unchecked(const NVTETensor* inputs,
                                                         NVTETensor* outputs,
                                                         const size_t num_tensors,
                                                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_swizzle_scaling_factors_unchecked);
  using namespace transformer_engine;
  NVTE_CHECK(num_tensors > 0, "Number of tensors should be greater than 0.");
  std::vector<Tensor*> input_list, output_list;
  for (size_t i = 0; i < num_tensors; i++) {
    input_list.push_back(convertNVTETensorCheck(inputs[i]));
    output_list.push_back(convertNVTETensorCheck(outputs[i]));
  }
  multi_tensor_swizzle_scaling_factors(input_list, output_list, stream,
                                       /*check_scale_inv_shapes=*/false);
}

void nvte_unswizzle_scaling_factors(const NVTETensor input, NVTETensor output,
                                    cudaStream_t stream) {
  NVTE_API_CALL(nvte_unswizzle_scaling_factors);
  using namespace transformer_engine;
  unswizzle_scaling_factors(convertNVTETensorCheck(input), convertNVTETensorCheck(output), stream);
}

void nvte_multi_tensor_unswizzle_scaling_factors(const NVTETensor* inputs, NVTETensor* outputs,
                                                 const size_t num_tensors, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_unswizzle_scaling_factors);
  using namespace transformer_engine;
  NVTE_CHECK(num_tensors > 0, "Number of tensors should be greater than 0.");
  std::vector<Tensor*> input_list, output_list;
  for (size_t i = 0; i < num_tensors; i++) {
    input_list.push_back(convertNVTETensorCheck(inputs[i]));
    output_list.push_back(convertNVTETensorCheck(outputs[i]));
  }
  multi_tensor_unswizzle_scaling_factors(input_list, output_list, stream);
}

namespace transformer_engine {

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    grouped_swizzle_scaling_variable_shape_kernel(const void* input, void* output,
                                                  const int64_t* m_array, const int64_t* k_array,
                                                  int num_tensors, bool rowwise,
                                                  size_t scale_elem_size, size_t common_m,
                                                  size_t common_k) {
  extern __shared__ int s_metadata[];
  int* s_total_blocks = &s_metadata[0];

  // Warp reduction to compute total workload
  if (threadIdx.x < 32 && threadIdx.y == 0) {
    int local_blocks = 0;
    for (int i = threadIdx.x; i < num_tensors; i += 32) {
      size_t m = rowwise ? (m_array ? m_array[i] : common_m) : (k_array ? k_array[i] : common_k);
      size_t k = rowwise ? (k_array ? k_array[i] : common_k) : (m_array ? m_array[i] : common_m);

      size_t padded_m = round_up_to_multiple(m, 128);
      size_t padded_k = round_up_to_multiple(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)), 4);

      int num_tiles_m = padded_m / SF_TILE_DIM_M;
      int num_tiles_k = padded_k / SF_TILE_DIM_K;

      int vec_load_size = (rowwise ? ((num_tiles_k - 1) % 4 + 1) : ((num_tiles_m - 1) % 4 + 1));
      if (vec_load_size == 3) vec_load_size = 1;
      int n_tiles_in_tb = TB_DIM * vec_load_size;

      int grid_dim_x = rowwise ? DIVUP(num_tiles_k, n_tiles_in_tb) : DIVUP(num_tiles_k, TB_DIM);
      int grid_dim_y = rowwise ? num_tiles_m : DIVUP(num_tiles_m, vec_load_size);
      local_blocks += grid_dim_x * grid_dim_y;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
      local_blocks += __shfl_down_sync(0xffffffff, local_blocks, offset);
    }
    if (threadIdx.x == 0) *s_total_blocks = local_blocks;
  }
  __syncthreads();

  const int total_blocks = *s_total_blocks;

  // Persistent-grid loop
  for (int linear_block_id = blockIdx.x; linear_block_id < total_blocks;
       linear_block_id += gridDim.x) {
    // Discover tensor_id and local_block_id via linear scan
    int tensor_id = 0;
    int current_block_base = 0;
    size_t current_scale_base = 0;
    int grid_dim_x = 0;
    int grid_dim_y = 0;
    size_t M = 0, K = 0;
    int vec_load_size = 0;

    for (int i = 0; i < num_tensors; ++i) {
      M = rowwise ? (m_array ? m_array[i] : common_m) : (k_array ? k_array[i] : common_k);
      K = rowwise ? (k_array ? k_array[i] : common_k) : (m_array ? m_array[i] : common_m);

      size_t padded_m = round_up_to_multiple(M, 128);
      size_t padded_k = round_up_to_multiple(DIVUP(K, static_cast<size_t>(MXFP8_BLOCK_SIZE)), 4);

      int num_tiles_m = padded_m / SF_TILE_DIM_M;
      int num_tiles_k = padded_k / SF_TILE_DIM_K;

      vec_load_size = (rowwise ? ((num_tiles_k - 1) % 4 + 1) : ((num_tiles_m - 1) % 4 + 1));
      if (vec_load_size == 3) vec_load_size = 1;
      int n_tiles_in_tb = TB_DIM * vec_load_size;

      grid_dim_x = rowwise ? DIVUP(num_tiles_k, n_tiles_in_tb) : DIVUP(num_tiles_k, TB_DIM);
      grid_dim_y = rowwise ? num_tiles_m : DIVUP(num_tiles_m, vec_load_size);
      int blocks_i = grid_dim_x * grid_dim_y;

      if (linear_block_id < current_block_base + blocks_i) {
        tensor_id = i;
        break;
      }
      current_block_base += blocks_i;
      current_scale_base += padded_m * padded_k * scale_elem_size;
    }

    int local_block_id = linear_block_id - current_block_base;
    int block_x = local_block_id % grid_dim_x;
    int block_y = local_block_id / grid_dim_x;

    const uint8_t* input_base = reinterpret_cast<const uint8_t*>(input) + current_scale_base;
    uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + current_scale_base;

    const int padded_m = static_cast<int>(round_up_to_multiple(M, 128));
    const int padded_k =
        static_cast<int>(round_up_to_multiple(DIVUP(K, static_cast<size_t>(MXFP8_BLOCK_SIZE)), 4));
    const int original_M = static_cast<int>(M);
    const int original_K = static_cast<int>(DIVUP(K, static_cast<size_t>(MXFP8_BLOCK_SIZE)));
    const bool padding_m = (block_y == grid_dim_y - 1) && (original_M < padded_m);
    const bool padding_k = (block_x == grid_dim_x - 1) && (original_K < padded_k);

    if (rowwise) {
      if (vec_load_size == 4) {
        dispatch_swizzle_row_scaling_kernel_impl<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>(
            input_base, output_base, padded_m, padded_k, original_M, original_K, block_x, block_y,
            grid_dim_x, grid_dim_y, padding_k, padding_m);
      } else if (vec_load_size == 2) {
        dispatch_swizzle_row_scaling_kernel_impl<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>(
            input_base, output_base, padded_m, padded_k, original_M, original_K, block_x, block_y,
            grid_dim_x, grid_dim_y, padding_k, padding_m);
      } else {
        dispatch_swizzle_row_scaling_kernel_impl<int, SF_TILE_DIM_M, SF_TILE_DIM_K>(
            input_base, output_base, padded_m, padded_k, original_M, original_K, block_x, block_y,
            grid_dim_x, grid_dim_y, padding_k, padding_m);
      }
    } else {
      if (vec_load_size == 4) {
        dispatch_swizzle_col_scaling_kernel_impl<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>(
            input_base, output_base, padded_m, padded_k, original_M, original_K, block_x, block_y,
            grid_dim_x, grid_dim_y, padding_k, padding_m);
      } else if (vec_load_size == 2) {
        dispatch_swizzle_col_scaling_kernel_impl<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>(
            input_base, output_base, padded_m, padded_k, original_M, original_K, block_x, block_y,
            grid_dim_x, grid_dim_y, padding_k, padding_m);
      } else {
        dispatch_swizzle_col_scaling_kernel_impl<int, SF_TILE_DIM_M, SF_TILE_DIM_K>(
            input_base, output_base, padded_m, padded_k, original_M, original_K, block_x, block_y,
            grid_dim_x, grid_dim_y, padding_k, padding_m);
      }
    }

    // Persistent CTAs reuse the same dynamic shared memory on the next loop
    // iteration. Wait for all shared-memory reads from this tile to finish.
    __syncthreads();
  }
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    grouped_swizzle_row_scaling_variable_shape_batched_m_kernel(
        const void* input, void* output, const int64_t* m_array, int num_tensors,
        size_t scale_elem_size, size_t common_k) {
  extern __shared__ int s_metadata[];
  int* s_total_blocks = &s_metadata[0];

  const int padded_k =
      static_cast<int>(round_up_to_multiple(DIVUP(common_k, static_cast<size_t>(MXFP8_BLOCK_SIZE)),
                                            static_cast<size_t>(SF_TILE_DIM_K)));
  const int num_tiles_k = padded_k / SF_TILE_DIM_K;

  if (threadIdx.x < 32 && threadIdx.y == 0) {
    int local_blocks = 0;
    for (int i = threadIdx.x; i < num_tensors; i += 32) {
      const size_t m = static_cast<size_t>(m_array[i]);
      const size_t padded_m = round_up_to_multiple(m, SF_TILE_DIM_M);
      const int num_tiles_m = static_cast<int>(padded_m / SF_TILE_DIM_M);
      local_blocks += DIVUP(num_tiles_m, static_cast<int>(blockDim.y));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      local_blocks += __shfl_down_sync(0xffffffff, local_blocks, offset);
    }
    if (threadIdx.x == 0) *s_total_blocks = local_blocks;
  }
  __syncthreads();

  const int total_blocks = *s_total_blocks;
  for (int linear_block_id = blockIdx.x; linear_block_id < total_blocks;
       linear_block_id += gridDim.x) {
    int current_block_base = 0;
    size_t current_scale_base = 0;
    size_t m = 0;
    int tensor_blocks = 0;

    for (int i = 0; i < num_tensors; ++i) {
      m = static_cast<size_t>(m_array[i]);
      const size_t padded_m = round_up_to_multiple(m, SF_TILE_DIM_M);
      const int num_tiles_m = static_cast<int>(padded_m / SF_TILE_DIM_M);
      tensor_blocks = DIVUP(num_tiles_m, static_cast<int>(blockDim.y));
      if (linear_block_id < current_block_base + tensor_blocks) {
        break;
      }
      current_block_base += tensor_blocks;
      current_scale_base += padded_m * padded_k * scale_elem_size;
    }

    const int local_block_id = linear_block_id - current_block_base;
    const int padded_m = static_cast<int>(round_up_to_multiple(m, SF_TILE_DIM_M));
    const int original_M = static_cast<int>(m);
    const int original_K = static_cast<int>(DIVUP(common_k, static_cast<size_t>(MXFP8_BLOCK_SIZE)));
    const uint8_t* input_base = reinterpret_cast<const uint8_t*>(input) + current_scale_base;
    uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + current_scale_base;

    swizzle_row_scaling_narrow_k_kernel_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
        input_base, output_base, padded_m, padded_k, original_M, original_K, local_block_id,
        tensor_blocks);

    // Keep the next persistent iteration from overwriting shared staging data
    // while slower threads are still reading it to write this tile out.
    __syncthreads();
  }
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    grouped_swizzle_col_scaling_variable_shape_narrow_m_kernel(
        const void* input, void* output, const int64_t* k_array, int num_tensors,
        size_t scale_elem_size, size_t common_m) {
  extern __shared__ int s_metadata[];
  int* s_total_blocks = &s_metadata[0];

  const int padded_m = static_cast<int>(round_up_to_multiple(common_m, SF_TILE_DIM_M));
  const int num_tiles_m = padded_m / SF_TILE_DIM_M;

  if (threadIdx.x < 32 && threadIdx.y == 0) {
    int local_blocks = 0;
    for (int i = threadIdx.x; i < num_tensors; i += 32) {
      const size_t k = static_cast<size_t>(k_array[i]);
      const size_t padded_k =
          round_up_to_multiple(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)),
                               static_cast<size_t>(SF_TILE_DIM_K));
      const int num_tiles_k = static_cast<int>(padded_k / SF_TILE_DIM_K);
      local_blocks += DIVUP(num_tiles_k, static_cast<int>(blockDim.y));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      local_blocks += __shfl_down_sync(0xffffffff, local_blocks, offset);
    }
    if (threadIdx.x == 0) *s_total_blocks = local_blocks;
  }
  __syncthreads();

  const int total_blocks = *s_total_blocks;
  for (int linear_block_id = blockIdx.x; linear_block_id < total_blocks;
       linear_block_id += gridDim.x) {
    int current_block_base = 0;
    size_t current_scale_base = 0;
    size_t k = 0;
    int tensor_blocks = 0;

    for (int i = 0; i < num_tensors; ++i) {
      k = static_cast<size_t>(k_array[i]);
      const size_t padded_k =
          round_up_to_multiple(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)),
                               static_cast<size_t>(SF_TILE_DIM_K));
      const int num_tiles_k = static_cast<int>(padded_k / SF_TILE_DIM_K);
      tensor_blocks = DIVUP(num_tiles_k, static_cast<int>(blockDim.y));
      if (linear_block_id < current_block_base + tensor_blocks) {
        break;
      }
      current_block_base += tensor_blocks;
      current_scale_base += static_cast<size_t>(padded_m) * padded_k * scale_elem_size;
    }

    const int local_block_id = linear_block_id - current_block_base;
    const int padded_k =
        static_cast<int>(round_up_to_multiple(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)),
                                              static_cast<size_t>(SF_TILE_DIM_K)));
    const int original_M = static_cast<int>(common_m);
    const int original_K = static_cast<int>(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)));
    const uint8_t* input_base = reinterpret_cast<const uint8_t*>(input) + current_scale_base;
    uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + current_scale_base;

    swizzle_col_scaling_narrow_m_kernel_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
        input_base, output_base, padded_m, padded_k, original_M, original_K, local_block_id,
        tensor_blocks);

    // Keep the next persistent iteration from overwriting shared staging data
    // while slower threads are still reading it to write this tile out.
    __syncthreads();
  }
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(ROW_COALESCED_THREADS)
    grouped_swizzle_row_scaling_variable_shape_coalesced_kernel(
        const void* input, void* output, const int64_t* m_array, int num_tensors,
        size_t scale_elem_size, size_t common_k, const int m_tiles_per_block) {
  extern __shared__ int4 slm_v4i[];

  const int padded_k =
      static_cast<int>(round_up_to_multiple(DIVUP(common_k, static_cast<size_t>(MXFP8_BLOCK_SIZE)),
                                            static_cast<size_t>(SF_TILE_DIM_K)));

  const int linear_block_id = static_cast<int>(blockIdx.x);
  int current_block_base = 0;
  size_t current_scale_base = 0;
  size_t m = 0;
  int tensor_blocks = 0;
  bool found_tensor = false;

  for (int i = 0; i < num_tensors; ++i) {
    m = static_cast<size_t>(m_array[i]);
    const size_t padded_m = round_up_to_multiple(m, SF_TILE_DIM_M);
    tensor_blocks = DIVUP(static_cast<int>(padded_m / SF_TILE_DIM_M), m_tiles_per_block);
    if (linear_block_id < current_block_base + tensor_blocks) {
      found_tensor = true;
      break;
    }
    current_block_base += tensor_blocks;
    current_scale_base += padded_m * padded_k * scale_elem_size;
  }
  if (!found_tensor) return;

  const int local_m_tile = (linear_block_id - current_block_base) * m_tiles_per_block;
  const int padded_m = static_cast<int>(round_up_to_multiple(m, SF_TILE_DIM_M));
  const int original_M = static_cast<int>(m);
  const uint8_t* input_base = reinterpret_cast<const uint8_t*>(input) + current_scale_base;
  uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + current_scale_base;
  swizzle_row_scaling_coalesced_batched_m_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input_base, output_base, padded_m, padded_k, original_M, local_m_tile, m_tiles_per_block,
      slm_v4i);
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(COL_WARP_TILE_THREADS)
    grouped_swizzle_col_scaling_variable_shape_warp_tile_kernel(
        const void* input, void* output, const int64_t* k_array, int num_tensors,
        size_t scale_elem_size, size_t common_m) {
  extern __shared__ int s_metadata[];
  int* s_total_tiles = &s_metadata[0];

  const int padded_m = static_cast<int>(round_up_to_multiple(common_m, SF_TILE_DIM_M));
  const int num_tiles_m = padded_m / SF_TILE_DIM_M;

  if (threadIdx.x < 32) {
    int local_tiles = 0;
    for (int i = threadIdx.x; i < num_tensors; i += 32) {
      const size_t k = static_cast<size_t>(k_array[i]);
      const size_t padded_k =
          round_up_to_multiple(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)),
                               static_cast<size_t>(SF_TILE_DIM_K));
      local_tiles += num_tiles_m * static_cast<int>(padded_k / SF_TILE_DIM_K);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      local_tiles += __shfl_down_sync(0xffffffff, local_tiles, offset);
    }
    if (threadIdx.x == 0) *s_total_tiles = local_tiles;
  }
  __syncthreads();

  const int total_tiles = *s_total_tiles;
  const int warp_id = threadIdx.x / 32;
  for (int linear_tile_id = blockIdx.x * COL_WARP_TILE_WARPS + warp_id;
       linear_tile_id < total_tiles; linear_tile_id += gridDim.x * COL_WARP_TILE_WARPS) {
    int current_tile_base = 0;
    size_t current_scale_base = 0;
    size_t k = 0;
    int tensor_tiles = 0;
    int num_tiles_k = 0;

    for (int i = 0; i < num_tensors; ++i) {
      k = static_cast<size_t>(k_array[i]);
      const size_t padded_k =
          round_up_to_multiple(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)),
                               static_cast<size_t>(SF_TILE_DIM_K));
      num_tiles_k = static_cast<int>(padded_k / SF_TILE_DIM_K);
      tensor_tiles = num_tiles_m * num_tiles_k;
      if (linear_tile_id < current_tile_base + tensor_tiles) {
        break;
      }
      current_tile_base += tensor_tiles;
      current_scale_base += static_cast<size_t>(padded_m) * padded_k * scale_elem_size;
    }

    const int local_tile_id = linear_tile_id - current_tile_base;
    const int padded_k =
        static_cast<int>(round_up_to_multiple(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)),
                                              static_cast<size_t>(SF_TILE_DIM_K)));
    const int original_M = static_cast<int>(common_m);
    const int original_K = static_cast<int>(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)));
    const uint8_t* input_base = reinterpret_cast<const uint8_t*>(input) + current_scale_base;
    uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + current_scale_base;

    swizzle_col_scaling_warp_tile_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
        input_base, output_base, padded_m, padded_k, original_M, original_K, local_tile_id);
  }
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(COL_COALESCED_THREADS)
    grouped_swizzle_col_scaling_variable_shape_coalesced_kernel(
        const void* input, void* output, const int64_t* k_array, int num_tensors,
        size_t scale_elem_size, size_t common_m, const int k_tiles_per_block) {
  extern __shared__ int slm_i32[];

  const int padded_m = static_cast<int>(round_up_to_multiple(common_m, SF_TILE_DIM_M));
  const int num_tiles_m = padded_m / SF_TILE_DIM_M;

  const int linear_block_id = static_cast<int>(blockIdx.x);
  int current_block_base = 0;
  size_t current_scale_base = 0;
  size_t k = 0;
  int tensor_blocks = 0;
  int num_k_blocks = 0;
  bool found_tensor = false;

  for (int i = 0; i < num_tensors; ++i) {
    k = static_cast<size_t>(k_array[i]);
    const size_t padded_k =
        round_up_to_multiple(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)),
                             static_cast<size_t>(SF_TILE_DIM_K));
    const int num_tiles_k = static_cast<int>(padded_k / SF_TILE_DIM_K);
    num_k_blocks = DIVUP(num_tiles_k, k_tiles_per_block);
    tensor_blocks = num_tiles_m * num_k_blocks;
    if (linear_block_id < current_block_base + tensor_blocks) {
      found_tensor = true;
      break;
    }
    current_block_base += tensor_blocks;
    current_scale_base += static_cast<size_t>(padded_m) * padded_k * scale_elem_size;
  }
  if (!found_tensor) return;

  const int local_block_id = linear_block_id - current_block_base;
  const int m_tile = local_block_id / num_k_blocks;
  const int k_tile_block = local_block_id - m_tile * num_k_blocks;
  const int padded_k =
      static_cast<int>(round_up_to_multiple(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)),
                                            static_cast<size_t>(SF_TILE_DIM_K)));
  const int original_M = static_cast<int>(common_m);
  const int original_K = static_cast<int>(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)));
  const uint8_t* input_base = reinterpret_cast<const uint8_t*>(input) + current_scale_base;
  uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + current_scale_base;

  swizzle_col_scaling_coalesced_tile_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input_base, output_base, padded_m, padded_k, original_M, original_K, m_tile, k_tile_block,
      k_tiles_per_block, slm_i32);
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
int grouped_swizzle_variable_max_active_blocks_per_sm(int device_id) {
  static std::vector<int> cache(cuda::num_devices(), -1);
  static std::vector<std::once_flag> flags(cuda::num_devices());
  NVTE_CHECK(0 <= device_id && device_id < cuda::num_devices(), "invalid CUDA device ID");

  auto init = [&]() {
    constexpr int metadata_shmem = sizeof(int);  // s_total_blocks
    constexpr int dynamic_smem_size =
        TB_DIM * 4 * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t) + metadata_shmem;
    int max_active_blocks_per_sm;
    NVTE_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_per_sm,
        grouped_swizzle_scaling_variable_shape_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>,
        TB_DIM * TB_DIM, dynamic_smem_size));
    NVTE_CHECK(max_active_blocks_per_sm > 0, "Occupancy query returned 0 blocks per SM.");
    cache[device_id] = max_active_blocks_per_sm;
  };
  std::call_once(flags[device_id], init);
  return cache[device_id];
}

void swizzle_grouped_scaling_factors(const GroupedTensor* input, GroupedTensor* output,
                                     cudaStream_t stream) {
  // Check scaling mode
  NVTE_CHECK(input->scaling_mode == NVTE_MXFP8_1D_SCALING,
             "Grouped swizzle supports only MXFP8 scaling.");

  // Check tensors
  CheckInputGroupedTensor(*input, "input");
  CheckOutputGroupedTensor(*output, "output", false);
  NVTE_CHECK(!input->with_gemm_swizzled_scales,
             "Expected input grouped tensor with scales in compact format.");
  NVTE_CHECK(output->with_gemm_swizzled_scales,
             "Expected output grouped tensor with scales in GEMM swizzled format.");

  // Check scaling factors availability
  const bool has_rowwise_scale_inv = input->scale_inv.has_data();
  const bool has_columnwise_scale_inv = input->columnwise_scale_inv.has_data();
  if (!has_rowwise_scale_inv && !has_columnwise_scale_inv) {
    return;
  }

  const int64_t* m_array = reinterpret_cast<const int64_t*>(input->first_dims.dptr);
  const int64_t* k_array = reinterpret_cast<const int64_t*>(input->last_dims.dptr);
  const bool is_variable_shape = !input->all_same_shape();

  if (!is_variable_shape) {
    // Fallback to uniform shape implementation
    // Assumption is that all the tensors share the same shapes and are contgiuous.
    // And so we dont need to pass array of input/output pointers(due to conttiguity)
    // as well as array of shapes(due to uniform shapes).
    const size_t first_dim = input->get_common_first_dim();
    const size_t last_dim = input->get_common_last_dim();

    constexpr int SF_TILE_DIM_M = 128;
    constexpr int SF_TILE_DIM_K = 4;
    const dim3 block_size(TB_DIM, TB_DIM);

    auto launch_grouped_swizzle = [&](bool rowwise) {
      const size_t m = rowwise ? first_dim : last_dim;
      const size_t k = rowwise ? last_dim : first_dim;
      const size_t padded_m = round_up_to_multiple(m, 128);
      const size_t padded_k =
          round_up_to_multiple(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)), 4);
      // Per-tensor scale-element counts:
      //  - "padded" layout: each tensor occupies padded_m * padded_k elements
      //    (total buffer = num_tensors * padded_m * padded_k).
      //  - "compact" layout (what the grouped MXFP8 quantize kernel actually writes):
      //      per-tensor stride is m * padded_k (rowwise) or DIVUP(k,32) * padded_m
      //      (columnwise) and the total buffer the C++ allocator hands out has its
      //      grouped first dim padded up to a multiple of 128 (rowwise) or 4
      //      (columnwise) -- so the buffer may be slightly larger than
      //      num_tensors * compact_scale_elems, with trailing alignment slack at
      //      the very end (never read because of the per-tensor row/k guard in the
      //      kernel impl).
      // The output is always written in the padded layout. The input may be in
      // either layout; the kernel handles the compact case safely by using
      // different per-tensor strides for input vs output and skipping loads past
      // the per-tensor extent.
      const size_t padded_scale_elems = padded_m * padded_k;
      const size_t compact_scale_elems =
          rowwise ? m * padded_k : DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)) * padded_m;
      const size_t compact_total_scale_elems =
          rowwise ? round_up_to_multiple(input->num_tensors * m, 128) * padded_k
                  : round_up_to_multiple(
                        input->num_tensors * DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)), 4) *
                        padded_m;

      const size_t scale_elem_size = rowwise ? typeToSize(input->scale_inv.dtype)
                                             : typeToSize(input->columnwise_scale_inv.dtype);

      const size_t input_scale_numel =
          rowwise ? input->scale_inv.numel() : input->columnwise_scale_inv.numel();
      const size_t output_scale_numel =
          rowwise ? output->scale_inv.numel() : output->columnwise_scale_inv.numel();

      bool input_is_compact;
      if (input_scale_numel == input->num_tensors * padded_scale_elems) {
        input_is_compact = false;
      } else if (input_scale_numel == compact_total_scale_elems) {
        input_is_compact = true;
      } else {
        NVTE_ERROR("Grouped input ", (rowwise ? "scale_inv" : "columnwise_scale_inv"),
                   " size does not match expected packed size (got ", input_scale_numel,
                   ", expected either ", input->num_tensors * padded_scale_elems,
                   " (per-tensor padded) or ", compact_total_scale_elems, " (compact)).");
      }
      NVTE_CHECK(output_scale_numel == input->num_tensors * padded_scale_elems, "Grouped output ",
                 (rowwise ? "scale_inv" : "columnwise_scale_inv"),
                 " size does not match expected per-tensor padded size.");

      const size_t input_stride_bytes =
          (input_is_compact ? compact_scale_elems : padded_scale_elems) * scale_elem_size;
      const size_t output_stride_bytes = padded_scale_elems * scale_elem_size;

      const int original_M = static_cast<int>(rowwise ? first_dim : last_dim);
      const int original_K = static_cast<int>(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)));
      const int num_tiles_m = padded_m / SF_TILE_DIM_M;
      const int num_tiles_k = padded_k / SF_TILE_DIM_K;
      int vec_load_size = (rowwise ? ((num_tiles_k - 1) % 4 + 1) : ((num_tiles_m - 1) % 4 + 1));
      if (vec_load_size == 3) vec_load_size = 1;
      const int n_tiles_in_tb = TB_DIM * vec_load_size;
      const int m_tiles_per_block =
          row_swizzle_m_tiles_per_block<SF_TILE_DIM_M, SF_TILE_DIM_K>(num_tiles_k);
      const int narrow_m_slm_size =
          TB_DIM * num_tiles_m * SF_TILE_DIM_M * SF_TILE_DIM_K * static_cast<int>(sizeof(int8_t));
      const int row_coalesced_slm_size =
          row_coalesced_slm_size_bytes<SF_TILE_DIM_M, SF_TILE_DIM_K>(
              static_cast<int>(padded_k), scale_elem_size);
      const bool use_row_coalesced =
          rowwise && use_blackwell_row_coalesced_swizzle<SF_TILE_DIM_M, SF_TILE_DIM_K>(
                         static_cast<int>(padded_k), original_K, scale_elem_size);
      const bool use_col_coalesced =
          !rowwise && use_blackwell_col_coalesced_swizzle(scale_elem_size);
      const bool use_narrow_k = rowwise && !use_row_coalesced && m_tiles_per_block > 1;
      const bool use_narrow_m =
          !rowwise && !use_col_coalesced && num_tiles_m < TB_DIM &&
          narrow_m_slm_size <= get_max_dynamic_smem();
      const int row_coalesced_m_tiles =
          use_row_coalesced
              ? row_coalesced_m_tiles_per_block<SF_TILE_DIM_M, SF_TILE_DIM_K>(
                    static_cast<int>(padded_k), scale_elem_size)
              : 1;
      const int col_coalesced_k_tiles_per_block = COL_COALESCED_K_TILES_PER_BLOCK;

      dim3 num_blocks;
      if (use_row_coalesced) {
        num_blocks = dim3(DIVUP(num_tiles_m, row_coalesced_m_tiles), input->num_tensors);
      } else if (use_col_coalesced) {
        num_blocks = dim3(DIVUP(num_tiles_k, col_coalesced_k_tiles_per_block), num_tiles_m,
                          input->num_tensors);
      } else if (use_narrow_k) {
        num_blocks = dim3(DIVUP(num_tiles_m, m_tiles_per_block), 1, input->num_tensors);
      } else if (use_narrow_m) {
        num_blocks = dim3(DIVUP(num_tiles_k, TB_DIM), 1, input->num_tensors);
      } else if (rowwise) {
        num_blocks = dim3(DIVUP(num_tiles_k, n_tiles_in_tb), num_tiles_m, input->num_tensors);
      } else {
        num_blocks =
            dim3(DIVUP(num_tiles_k, TB_DIM), DIVUP(num_tiles_m, vec_load_size), input->num_tensors);
      }
      const int slm_size = use_row_coalesced
                               ? row_coalesced_slm_size * row_coalesced_m_tiles
                           : use_col_coalesced
                               ? col_coalesced_slm_size_bytes<SF_TILE_DIM_M, SF_TILE_DIM_K>(
                                     col_coalesced_k_tiles_per_block)
                           : use_narrow_k   ? m_tiles_per_block * num_tiles_k * SF_TILE_DIM_M *
                                                  SF_TILE_DIM_K *
                                                  static_cast<int>(sizeof(int8_t))
                           : use_narrow_m ? narrow_m_slm_size
                                          : n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K *
                                                static_cast<int>(sizeof(int8_t));

      const void* input_ptr = rowwise ? input->scale_inv.dptr : input->columnwise_scale_inv.dptr;
      void* output_ptr = rowwise ? output->scale_inv.dptr : output->columnwise_scale_inv.dptr;

      if (use_row_coalesced) {
        static int cached_grouped_uniform_row_coalesced_batched = -1;
        set_dynamic_smem_if_needed(
            grouped_swizzle_row_scaling_uniform_shape_coalesced_batched_m_kernel<
                SF_TILE_DIM_M, SF_TILE_DIM_K>,
            slm_size, cached_grouped_uniform_row_coalesced_batched);
        grouped_swizzle_row_scaling_uniform_shape_coalesced_batched_m_kernel<SF_TILE_DIM_M,
                                                                             SF_TILE_DIM_K>
            <<<num_blocks, ROW_COALESCED_THREADS, slm_size, stream>>>(
                input_ptr, output_ptr, padded_m, padded_k, original_M, input_stride_bytes,
                output_stride_bytes, row_coalesced_m_tiles);
      } else if (use_col_coalesced) {
        static int cached_grouped_uniform_col_coalesced = -1;
        set_dynamic_smem_if_needed(
            grouped_swizzle_col_scaling_uniform_shape_coalesced_kernel<SF_TILE_DIM_M,
                                                                       SF_TILE_DIM_K>,
            slm_size, cached_grouped_uniform_col_coalesced);
        grouped_swizzle_col_scaling_uniform_shape_coalesced_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<num_blocks, COL_COALESCED_THREADS, slm_size, stream>>>(
                input_ptr, output_ptr, padded_m, padded_k, original_M, original_K,
                input_stride_bytes, output_stride_bytes, col_coalesced_k_tiles_per_block);
      } else if (use_narrow_k) {
        dim3 block_size_batched(TB_DIM, m_tiles_per_block);
        static int cached_grouped_uniform_row_batched = -1;
        set_dynamic_smem_if_needed(
            grouped_swizzle_row_scaling_uniform_shape_narrow_k_kernel<SF_TILE_DIM_M,
                                                                      SF_TILE_DIM_K>,
            slm_size, cached_grouped_uniform_row_batched);
        grouped_swizzle_row_scaling_uniform_shape_narrow_k_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<num_blocks, block_size_batched, slm_size, stream>>>(
                input_ptr, output_ptr, padded_m, padded_k, original_M, original_K,
                input_stride_bytes, output_stride_bytes);
      } else if (use_narrow_m) {
        static int cached_grouped_uniform_col_narrow_m = -1;
        set_dynamic_smem_if_needed(
            grouped_swizzle_col_scaling_uniform_shape_narrow_m_kernel<SF_TILE_DIM_M,
                                                                      SF_TILE_DIM_K>,
            slm_size, cached_grouped_uniform_col_narrow_m);
        grouped_swizzle_col_scaling_uniform_shape_narrow_m_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<num_blocks, block_size, slm_size, stream>>>(
                input_ptr, output_ptr, padded_m, padded_k, original_M, original_K,
                input_stride_bytes, output_stride_bytes);
      } else if (rowwise) {
        TRANSFORMER_ENGINE_VECTORIZED_LOAD_INTEGER_TYPE_SWITCH(vec_load_size, LType, {
          static int cached_grouped_uniform_row = -1;
          set_dynamic_smem_if_needed(
              grouped_swizzle_row_scaling_uniform_shape_kernel<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              slm_size, cached_grouped_uniform_row);
          grouped_swizzle_row_scaling_uniform_shape_kernel<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(
                  input_ptr, output_ptr, padded_m, padded_k, original_M, original_K,
                  input_stride_bytes, output_stride_bytes);
        });
      } else {
        TRANSFORMER_ENGINE_VECTORIZED_LOAD_INTEGER_TYPE_SWITCH(vec_load_size, LType, {
          static int cached_grouped_uniform_col = -1;
          set_dynamic_smem_if_needed(
              grouped_swizzle_col_scaling_uniform_shape_kernel<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              slm_size, cached_grouped_uniform_col);
          grouped_swizzle_col_scaling_uniform_shape_kernel<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(
                  input_ptr, output_ptr, padded_m, padded_k, original_M, original_K,
                  input_stride_bytes, output_stride_bytes);
        });
      }
      NVTE_CHECK_CUDA(cudaGetLastError());
    };

    if (has_rowwise_scale_inv) {
      launch_grouped_swizzle(true);
    }
    if (has_columnwise_scale_inv) {
      launch_grouped_swizzle(false);
    }
  } else {
    // Variable shape implementation using Device-Side Block Scheduler
    size_t num_tensors = input->num_tensors;

    constexpr int SF_TILE_DIM_M = 128;
    constexpr int SF_TILE_DIM_K = 4;
    const int device_id = cuda::current_device();
    const int num_SMs = cuda::sm_count(device_id);
    const int metadata_shmem = sizeof(int);  // s_total_blocks
    size_t common_m = input->all_same_first_dim() ? input->get_common_first_dim() : 0;
    size_t common_k = input->all_same_last_dim() ? input->get_common_last_dim() : 0;
    bool rowwise_done = false;
    bool columnwise_done = false;

    if (has_rowwise_scale_inv && input->all_same_last_dim() && input->first_dims.has_data()) {
      const size_t k = input->get_common_last_dim();
      const size_t padded_k =
          round_up_to_multiple(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)), SF_TILE_DIM_K);
      const int num_tiles_k = static_cast<int>(padded_k / SF_TILE_DIM_K);
      const size_t scale_elem_size = typeToSize(input->scale_inv.dtype);
      const int original_k = static_cast<int>(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)));
      if (use_blackwell_row_coalesced_swizzle<SF_TILE_DIM_M, SF_TILE_DIM_K>(
              static_cast<int>(padded_k), original_k, scale_elem_size)) {
        const int slm_size = row_coalesced_slm_size_bytes<SF_TILE_DIM_M, SF_TILE_DIM_K>(
            static_cast<int>(padded_k), scale_elem_size);
        const int m_tiles_per_block =
            row_coalesced_m_tiles_per_block<SF_TILE_DIM_M, SF_TILE_DIM_K>(
                static_cast<int>(padded_k), scale_elem_size);
        const int dynamic_smem_size = slm_size * m_tiles_per_block;
        const size_t total_m_tiles_upper =
            DIVUP(input->logical_shape.data[0], static_cast<size_t>(SF_TILE_DIM_M)) + num_tensors;
        const size_t num_blocks =
            DIVUP(total_m_tiles_upper, static_cast<size_t>(m_tiles_per_block)) + num_tensors;
        static int cached_variable_row_coalesced = -1;
        set_dynamic_smem_if_needed(
            grouped_swizzle_row_scaling_variable_shape_coalesced_kernel<SF_TILE_DIM_M,
                                                                        SF_TILE_DIM_K>,
            dynamic_smem_size, cached_variable_row_coalesced);
        grouped_swizzle_row_scaling_variable_shape_coalesced_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<static_cast<unsigned int>(num_blocks), ROW_COALESCED_THREADS, dynamic_smem_size,
               stream>>>(
                input->scale_inv.dptr, output->scale_inv.dptr, m_array,
                static_cast<int>(num_tensors), scale_elem_size, k, m_tiles_per_block);
        NVTE_CHECK_CUDA(cudaGetLastError());
        rowwise_done = true;
      } else {
        const int m_tiles_per_block =
            row_swizzle_m_tiles_per_block<SF_TILE_DIM_M, SF_TILE_DIM_K>(num_tiles_k,
                                                                        metadata_shmem);
        if (m_tiles_per_block > 1) {
          const int slm_size = m_tiles_per_block * num_tiles_k * SF_TILE_DIM_M * SF_TILE_DIM_K *
                               static_cast<int>(sizeof(int8_t));
          const int dynamic_smem_size = slm_size + metadata_shmem;
          const dim3 block_size_batched(TB_DIM, m_tiles_per_block);
          static int cached_variable_row_batched = -1;
          set_dynamic_smem_if_needed(
              grouped_swizzle_row_scaling_variable_shape_batched_m_kernel<SF_TILE_DIM_M,
                                                                          SF_TILE_DIM_K>,
              dynamic_smem_size, cached_variable_row_batched);
          grouped_swizzle_row_scaling_variable_shape_batched_m_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_SMs, block_size_batched, dynamic_smem_size, stream>>>(
                  input->scale_inv.dptr, output->scale_inv.dptr, m_array,
                  static_cast<int>(num_tensors), scale_elem_size, k);
          NVTE_CHECK_CUDA(cudaGetLastError());
          rowwise_done = true;
        }
      }
    }

    if (has_columnwise_scale_inv && input->all_same_last_dim() && input->first_dims.has_data()) {
      const size_t m = input->get_common_last_dim();
      const size_t padded_m = round_up_to_multiple(m, SF_TILE_DIM_M);
      const int num_tiles_m = static_cast<int>(padded_m / SF_TILE_DIM_M);
      const size_t scale_elem_size = typeToSize(input->columnwise_scale_inv.dtype);
      const int narrow_m_slm_size =
          TB_DIM * num_tiles_m * SF_TILE_DIM_M * SF_TILE_DIM_K * static_cast<int>(sizeof(int8_t));
      if (use_blackwell_col_coalesced_swizzle(scale_elem_size)) {
        const int k_tiles_per_block = COL_COALESCED_K_TILES_PER_BLOCK;
        const int slm_size =
            col_coalesced_slm_size_bytes<SF_TILE_DIM_M, SF_TILE_DIM_K>(k_tiles_per_block);
        const int dynamic_smem_size = slm_size;
        const size_t total_k_scales_upper =
            DIVUP(input->logical_shape.data[0], static_cast<size_t>(MXFP8_BLOCK_SIZE)) +
            num_tensors;
        const size_t total_k_tiles_upper =
            DIVUP(total_k_scales_upper, static_cast<size_t>(SF_TILE_DIM_K)) + num_tensors;
        const size_t k_block_upper =
            DIVUP(total_k_tiles_upper, static_cast<size_t>(k_tiles_per_block)) + num_tensors;
        const size_t num_blocks = static_cast<size_t>(num_tiles_m) * k_block_upper;
        static int cached_variable_col_coalesced = -1;
        set_dynamic_smem_if_needed(
            grouped_swizzle_col_scaling_variable_shape_coalesced_kernel<SF_TILE_DIM_M,
                                                                        SF_TILE_DIM_K>,
            dynamic_smem_size, cached_variable_col_coalesced);
        grouped_swizzle_col_scaling_variable_shape_coalesced_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<static_cast<unsigned int>(num_blocks), COL_COALESCED_THREADS, dynamic_smem_size,
               stream>>>(
                input->columnwise_scale_inv.dptr, output->columnwise_scale_inv.dptr, m_array,
                static_cast<int>(num_tensors), scale_elem_size, m, k_tiles_per_block);
        NVTE_CHECK_CUDA(cudaGetLastError());
        columnwise_done = true;
      } else if (num_tiles_m < TB_DIM &&
                 narrow_m_slm_size + metadata_shmem <= get_max_dynamic_smem()) {
        const int dynamic_smem_size = narrow_m_slm_size + metadata_shmem;
        const dim3 block_size(TB_DIM, TB_DIM);
        static int cached_variable_col_narrow_m = -1;
        set_dynamic_smem_if_needed(
            grouped_swizzle_col_scaling_variable_shape_narrow_m_kernel<SF_TILE_DIM_M,
                                                                       SF_TILE_DIM_K>,
            dynamic_smem_size, cached_variable_col_narrow_m);
        grouped_swizzle_col_scaling_variable_shape_narrow_m_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<num_SMs, block_size, dynamic_smem_size, stream>>>(
                input->columnwise_scale_inv.dptr, output->columnwise_scale_inv.dptr, m_array,
                static_cast<int>(num_tensors), scale_elem_size, m);
        NVTE_CHECK_CUDA(cudaGetLastError());
        columnwise_done = true;
      }
    }

    const bool needs_generic_variable =
        (has_rowwise_scale_inv && !rowwise_done) || (has_columnwise_scale_inv && !columnwise_done);

    if (needs_generic_variable) {
      const dim3 block_size(TB_DIM, TB_DIM);
      const int max_slm_size = TB_DIM * 4 * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);
      const int dynamic_smem_size = max_slm_size + metadata_shmem;

      static int cached_variable_generic = -1;
      set_dynamic_smem_if_needed(
          grouped_swizzle_scaling_variable_shape_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>,
          dynamic_smem_size, cached_variable_generic);

      const int max_active_blocks_per_sm =
          grouped_swizzle_variable_max_active_blocks_per_sm<SF_TILE_DIM_M, SF_TILE_DIM_K>(
              device_id);
      const int persistent_blocks = num_SMs * max_active_blocks_per_sm;
      const dim3 num_blocks(persistent_blocks);

      auto launch_grouped_swizzle_variable = [&](bool rowwise) {
        const size_t scale_elem_size = rowwise ? typeToSize(input->scale_inv.dtype)
                                               : typeToSize(input->columnwise_scale_inv.dtype);

        const void* input_ptr = rowwise ? input->scale_inv.dptr : input->columnwise_scale_inv.dptr;
        void* output_ptr = rowwise ? output->scale_inv.dptr : output->columnwise_scale_inv.dptr;

        grouped_swizzle_scaling_variable_shape_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<num_blocks, block_size, dynamic_smem_size, stream>>>(
                input_ptr, output_ptr, m_array, k_array, num_tensors, rowwise, scale_elem_size,
                common_m, common_k);

        NVTE_CHECK_CUDA(cudaGetLastError());
      };

      if (has_rowwise_scale_inv && !rowwise_done) {
        launch_grouped_swizzle_variable(true);
      }
      if (has_columnwise_scale_inv && !columnwise_done) {
        launch_grouped_swizzle_variable(false);
      }
    }
  }
}

void unswizzle_grouped_scaling_factors(const GroupedTensor* input, GroupedTensor* output,
                                       cudaStream_t stream) {
  NVTE_CHECK(output->scaling_mode == NVTE_MXFP8_1D_SCALING,
             "Grouped unswizzle supports only MXFP8 scaling.");

  CheckInputGroupedTensor(*input, "input");
  CheckOutputGroupedTensor(*output, "output", false);
  NVTE_CHECK(input->with_gemm_swizzled_scales,
             "Expected input grouped tensor with scales in GEMM swizzled format.");
  NVTE_CHECK(!output->with_gemm_swizzled_scales,
             "Expected output grouped tensor with scales in compact format.");
  NVTE_CHECK(input->scaling_mode == output->scaling_mode,
             "Input and output grouped tensors must have matching scaling modes.");
  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Input and output grouped tensors must have the same number of tensors.");

  const bool has_rowwise_scale_inv = output->scale_inv.has_data();
  const bool has_columnwise_scale_inv = output->columnwise_scale_inv.has_data();
  if (!has_rowwise_scale_inv && !has_columnwise_scale_inv) {
    return;
  }

  NVTE_CHECK(input->all_same_shape() && output->all_same_shape(),
             "Grouped unswizzle requires uniform tensor shapes.");

  const size_t first_dim = output->get_common_first_dim();
  const size_t last_dim = output->get_common_last_dim();

  constexpr int SF_TILE_DIM_M = 128;
  constexpr int SF_TILE_DIM_K = 4;
  const dim3 block_size(TB_DIM, TB_DIM);

  auto launch_grouped_unswizzle = [&](bool rowwise) {
    const size_t m = rowwise ? first_dim : last_dim;
    const size_t k = rowwise ? last_dim : first_dim;
    const size_t padded_m = round_up_to_multiple(m, 128);
    const size_t padded_k =
        round_up_to_multiple(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)), 4);
    const size_t scale_elems = padded_m * padded_k;

    const size_t scale_elem_size = rowwise ? typeToSize(output->scale_inv.dtype)
                                           : typeToSize(output->columnwise_scale_inv.dtype);
    const size_t scale_stride_bytes = scale_elems * scale_elem_size;

    if (rowwise) {
      NVTE_CHECK(input->scale_inv.numel() == input->num_tensors * scale_elems,
                 "Grouped input scale_inv size does not match expected packed size.");
      NVTE_CHECK(output->scale_inv.numel() == output->num_tensors * scale_elems,
                 "Grouped output scale_inv size does not match expected packed size.");
    } else {
      NVTE_CHECK(input->columnwise_scale_inv.numel() == input->num_tensors * scale_elems,
                 "Grouped input columnwise_scale_inv size does not match expected packed size.");
      NVTE_CHECK(output->columnwise_scale_inv.numel() == output->num_tensors * scale_elems,
                 "Grouped output columnwise_scale_inv size does not match expected packed size.");
    }

    const int num_tiles_m = padded_m / SF_TILE_DIM_M;
    const int num_tiles_k = padded_k / SF_TILE_DIM_K;
    int vec_load_size = (rowwise ? ((num_tiles_k - 1) % 4 + 1) : ((num_tiles_m - 1) % 4 + 1));
    if (vec_load_size == 3) vec_load_size = 1;
    const int n_tiles_in_tb = TB_DIM * vec_load_size;

    dim3 num_blocks;
    if (rowwise) {
      num_blocks = dim3(DIVUP(num_tiles_k, n_tiles_in_tb), num_tiles_m, output->num_tensors);
    } else {
      num_blocks =
          dim3(DIVUP(num_tiles_k, TB_DIM), DIVUP(num_tiles_m, vec_load_size), output->num_tensors);
    }
    const int slm_size = n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);

    const void* input_ptr = rowwise ? input->scale_inv.dptr : input->columnwise_scale_inv.dptr;
    void* output_ptr = rowwise ? output->scale_inv.dptr : output->columnwise_scale_inv.dptr;

    using kernel_t = void (*)(const void*, void*, const int, const int, const size_t, const bool);
    kernel_t kernel_fn = nullptr;
    switch (vec_load_size) {
      case 4:
        kernel_fn =
            grouped_unswizzle_scaling_uniform_shape_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>;
        break;
      case 2:
        kernel_fn =
            grouped_unswizzle_scaling_uniform_shape_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>;
        break;
      case 1:
        kernel_fn =
            grouped_unswizzle_scaling_uniform_shape_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>;
        break;
      default:
        NVTE_ERROR("Not valid vec_load_size.");
    }
    NVTE_CHECK_CUDA(
        cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
    kernel_fn<<<num_blocks, block_size, slm_size, stream>>>(input_ptr, output_ptr, padded_m,
                                                            padded_k, scale_stride_bytes, rowwise);
    NVTE_CHECK_CUDA(cudaGetLastError());
  };

  if (has_rowwise_scale_inv) {
    launch_grouped_unswizzle(true);
  }
  if (has_columnwise_scale_inv) {
    launch_grouped_unswizzle(false);
  }
}

}  // namespace transformer_engine

void nvte_swizzle_grouped_scaling_factors(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                                          cudaStream_t stream) {
  NVTE_API_CALL(nvte_swizzle_grouped_scaling_factors);
  using namespace transformer_engine;
  swizzle_grouped_scaling_factors(convertNVTEGroupedTensorCheck(input),
                                  convertNVTEGroupedTensorCheck(output), stream);
}

void nvte_unswizzle_grouped_scaling_factors(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                                            cudaStream_t stream) {
  NVTE_API_CALL(nvte_unswizzle_grouped_scaling_factors);
  using namespace transformer_engine;
  unswizzle_grouped_scaling_factors(convertNVTEGroupedTensorCheck(input),
                                    convertNVTEGroupedTensorCheck(output), stream);
}
