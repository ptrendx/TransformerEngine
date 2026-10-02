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
``sbhd`` and ``copy``. ``parameter`` is a detector-only fallback control because
the existing DPA backend selector does not accept Parameter Q/K/V.
``--detector-only`` also isolates classification for other layouts. Use ``thd`` or ``ragged`` with
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
The default uses deterministic algorithms and bitwise checks between routes.
``--allow-nondeterministic`` enables the corresponding TE option and uses the
same dtype-appropriate comparison tolerances as the independent FP32 reference;
small run-to-run backward differences are possible with that option.

THD synchronization contract
----------------------------

With both maximum lengths supplied, ordinary THD attention performs no
readback to infer lengths. With either or both missing, it uses one readback.
The single-missing case keeps its direct scalar path; the two-missing case
batches independent Q/KV reductions. ``DotProductAttention`` retains
64-element rounding and padded physical offsets. A direct Flash backend call
retains its unrounded length semantics.

Mixed-policy IDs also require data-dependent validation. Prepare them before
attention to include mixed policies in the zero/one-wait contract:

.. code-block:: python

    prepared = DotProductAttention.prepare_thd_attention_policies(
        policies, batch_size, cu_seqlens_q=cu_q, cu_seqlens_kv=cu_kv)
    output = attention(q, k, v, thd_attention_policies=prepared,
                       cu_seqlens_q=cu_q, cu_seqlens_kv=cu_kv,
                       max_seqlen_q=max_q, max_seqlen_kv=max_kv)

Preparation validates order and exact coverage, performs host readbacks and
owns independent snapshots of the ID tensors. Those snapshots track mutation
even when the original IDs were created under ``inference_mode``. Replaced or
mutated prepared IDs raise an error requiring preparation again; changing
mask/window metadata still goes through normal forward validation. Preparation
must happen outside CUDA graph capture, and its validation proof survives
ordinary validation-cache eviction. Providing cumulative lengths also prepares
indices for grouped token compaction, which avoids the ``nonzero`` readbacks
used by FlashAttention 2. Pass padded cumulative lengths to preparation when
using physical padding. Supplied cumulative-length tensors must have version
counters (create them outside ``inference_mode``) and be reused unchanged in
forward; replacement or mutation requires preparation again. As with other
version-based PyTorch caches, untracked writes through ``.data`` or external
storage pointers are outside this contract.

For compatibility, ordinary policy lists retain eager validation. Fresh raw
IDs add a validation readback, even when both maxima are supplied. Unversioned
inference tensors are revalidated rather than trusting an identity cache after
mutation. Use prepared policies for the zero/one-wait contract; an eager warmup
alone cannot safely establish that contract for mutable, unversioned IDs.

The synchronization audit checks host-wait APIs inside named warmed attention
calls, with harness synchronization and preparation outside those ranges:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python benchmarks/attention/audit_attention_sync.py --backend fused \
      --mode train --padded --policies prepared --out /tmp/attention-sync.json

Repeat with ``--backend flash``, ``--mode inference``, without ``--padded``,
and with ``--same-cu``. Padded THD is unsupported by FlashAttention 2;
use ``--flash-version 4`` on supported devices to audit that path. ``--policies fresh`` explicitly audits the additional
legacy validation wait and, on FlashAttention 2, grouped compaction waits. The script fails if the observed count differs from
its declared contract, and checks independent FP32 outputs and gradients.
A wait-count reduction alone is not a CPU speedup claim.
