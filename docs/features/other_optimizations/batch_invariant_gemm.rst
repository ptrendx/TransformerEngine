..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Batch-invariant GEMM
====================

``batch_invariant_gemm`` computes ``A @ B.T`` for contiguous BF16 tensors
``A[M, K]`` and ``B[N, K]``. A row of the result has the same bits whether it
is computed alone or as part of a larger batch. The operation is forward-only
and does not support bias or autograd.

.. code-block:: python

   from transformer_engine.pytorch.cpp_extensions.batch_invariant_gemm import (
       batch_invariant_gemm,
   )

   y = batch_invariant_gemm(a, b)  # [M, N]

The same operation is available through ``general_gemm``. Its ``TN`` convention
takes the weight first and the activation second:

.. code-block:: python

   from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm

   y, _, _, _ = general_gemm(b, a, batch_invariant=True)

This option rejects other layouts, output dtypes, quantization, bias, activations,
gradients, accumulation, non-unit alpha, nonzero beta, and communication overlap.
It preserves automatic CuTeDSL/Triton selection and never substitutes cuBLASLt.

The default ``backend="auto"`` uses a persistent CuTeDSL tensor-core kernel on
Blackwell when N and K are at least 512 and multiples of eight and the CuTeDSL
dependencies are available. It uses Triton for other configurations. This
choice depends on N and K, never on M. Use ``backend="triton"`` or
``backend="cutedsl"`` to select a backend explicitly. Explicit CuTeDSL use
requires Blackwell, CuTeDSL, TVM FFI, cuda-python, and 16-byte aligned BF16
input pointers and rows. Unsupported configurations raise ``ValueError``.

On compute capability 10.x, the Triton backend uses smaller output tiles for
small batches and larger tiles for large batches. The K tile and sequential
reduction order remain fixed for a given N and K, so rows stay bitwise stable
when batch composition changes. Other GPU architectures use the fixed Triton
tile.

The CuTeDSL backend uses separate warps for TMA loads, tensor-core computation,
and output stores. Resident blocks process multiple output tiles, with two
accumulator buffers to overlap computation and stores. The output tile is
128x128 for up to 256 rows, 128x256 for up to 1024 rows, and 256x256 for larger
batches. These choices run inside the cached compiled callable. All use the
same 64-element K tile and sequential reduction order, preserving bitwise
batch invariance across output tile sizes.

When TVM FFI headers are available during the PyTorch extension build, the
CuTeDSL path uses ``general_gemm``'s C++ binding for output allocation and native
TVM FFI dispatch. Python compiles and registers each configuration once; C++
caches the native function and passes stack-allocated DLTensor descriptors and
the current PyTorch CUDA stream directly to it. Steady-state calls bypass the
Python kernel launcher and cuBLASLt setup. Builds without those headers retain
the Python CuTeDSL launcher, including when dependencies are installed later.

Both backends accept an optional contiguous BF16 ``out`` tensor of shape
``[M, N]``. To compare their steady-state GPU and eager-call times with
``general_gemm`` on the same device, run:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 python benchmarks/gemm/benchmark_batch_invariant_gemm.py

Add ``--include-triton-baseline`` to include the original 64x64x64 Triton
kernel in the CUDA graph timing comparison.

To measure complete asynchronous PyTorch CPU submission, including warmed
output allocation, compare the native path with the previous Python launcher:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 taskset -c 0 python \
       benchmarks/gemm/benchmark_batch_invariant_dispatch.py \
       --output /tmp/batch-invariant-cpu.json

The benchmark reuses input and weight tensors, rotates backend order, and keeps
synchronization outside the CPU timer. It reports median and 10th/90th percentiles
for preallocated and internally allocated outputs. Compilation is excluded by
warmup. Add ``--batch 1`` to check that submission queues do not dominate the
result. For individual kernel durations, use its warmed NVTX ranges with Nsight:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda,nvtx,cublas \
       --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
       --capture-range-end=stop --cuda-graph-trace=node \
       --output=/tmp/batch-invariant-native \
       python benchmarks/gemm/benchmark_batch_invariant_dispatch.py --profile
