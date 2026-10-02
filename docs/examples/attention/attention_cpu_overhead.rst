..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

PyTorch attention host overhead
===============================

``DotProductAttention`` detects the storage layout of ordinary eager Q/K/V
inputs in the installed PyTorch extension. Detection reads tensor metadata;
it does not synchronize CUDA or allocate device memory on a supported layout.
When fitting a layout requires contiguous copies, ATen owns those copies and
their autograd behavior. Detection runs for each call because equal shapes can
have different storage relationships.

The Python reference handles compilation, export, tensor subclasses (including
Parameters), active dispatch modes and tensor transformations. Exceptional
native inputs also return to that reference. This preserves the existing
layouts and error behavior; it does not add supported layouts. The common
library and JAX are unaffected.

For packed inputs, pass ``qkv_layer`` or ``kv_layer`` and
``qkv_interleave_dim`` explicitly. That declaration bypasses detection and
preserves the packed buffer through compilation. Pointer-based packed-layout
detection retains its deprecation warning, with the existing CPU-offload
exception.

Measuring host work
-------------------

``benchmarks/attention/benchmark_attention_cpu.py`` compares the retained
Python detector, a shape-numel control and the installed native route in
rotated paired batches. The numel control replaces NumPy's shape product with
``torch.Size.numel`` in isolated function globals; native detection reads
``Tensor.numel`` directly. This control is not a separate production route.

For example, select an otherwise idle GPU and a CPU core local to it:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      taskset -c 16 python benchmarks/attention/benchmark_attention_cpu.py \
      --backend fused --case short --mode inference --batches 40 \
      --batch-size 10 --warmup 200 --repeats 3 --out /tmp/attention-cpu.json

Repeat for ``--backend flash``, ``--mode train``, ``--case prefill`` and
``--case gqa``. Controls include ``--layout declared``, ``packed_kv``,
``sbhd``, ``copy`` and ``parameter``. Use ``thd`` or ``ragged`` with
``--maxima both/q/kv/neither`` for optional THD maxima.

The report includes process CPU (all threads), submitting-thread CPU, host
submission time, completion time, paired deltas, repeat ranges, independent
FP32 output/gradient checks, backend selection, extension hash/path, software
versions, shapes, strides, seed and topology. Batches synchronize before
submission and afterwards; the final wait is excluded from CPU and host
submission measurements. Warmup and numerical checks are outside timing.
CUDA-event timing of graph replay measures a separate captured device
workload, with supplied maxima for THD. Capture and input-copy costs are
excluded. Reduced process CPU does not imply GPU acceleration.

Use ``--baseline /path/to/prior.json --max-regression-percent 5`` as a
configurable regression gate on native process CPU. The harness rejects
mismatched hardware, software and workload configurations. Run repeats on an
idle machine and inspect their spread before accepting a performance change;
the measured microsecond savings are not portable pass/fail thresholds.
