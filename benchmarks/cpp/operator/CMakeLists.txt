# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

add_executable(bench_operator
               bench_activation.cu
               bench_cast.cu
               bench_fused_attn.cu
               bench_fused_rope.cu
               bench_fused_router.cu
               bench_gemm.cu
               bench_hadamard.cu
               bench_multi_tensor.cu
               bench_normalization.cu
               bench_softmax.cu
               bench_swizzle.cu
               bench_transpose.cu
               ../../../tests/cpp/test_common.cu)

list(APPEND bench_operator_LINKER_LIBS
     CUDA::cudart
     benchmark::benchmark
     benchmark::benchmark_main
     ${TE_LIB}
     CUDA::nvrtc
     CUDNN::cudnn)

target_link_libraries(bench_operator PUBLIC ${bench_operator_LINKER_LIBS})
target_compile_options(bench_operator PRIVATE -O2)
