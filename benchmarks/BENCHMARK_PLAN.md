# Transformer Engine Benchmark Suite

## Context

Transformer Engine (TE) has ~4 ad-hoc benchmark scripts in `benchmarks/` but no systematic, production-quality benchmark suite. The goal is to create a comprehensive 3-tier benchmark system (C++/CUDA, PyTorch, JAX) that covers all TE kernels and modules, supports both GPU-bound and CPU-bound scenarios, integrates with profilers (nsys, perf), runs on Slurm/enroot clusters, and optionally reports to a remote database.

The existing benchmarks (`benchmarks/linear/benchmark_linear.py`, `benchmarks/attention/benchmark_attention.py`, `benchmarks/benchmark_rht_cast.py`, `benchmarks/linear/benchmark_grouped_linear.py`) will be preserved as-is. The new suite lives alongside them in the `benchmarks/` directory.

---

## Directory Structure

```
benchmarks/
  (existing files preserved)
  
  config/
    __init__.py
    sizes.py                          # CPU-bound and GPU-bound shape definitions per operation
    benchmark_config.py               # BenchmarkConfig dataclass + CLI arg parser
  
  common/
    __init__.py
    result_types.py                   # BenchmarkResult dataclass + metadata collection
    timing.py                         # BenchmarkTimer (PyTorch), JAXBenchmarkTimer wrappers
    profiler_hooks.py                 # nsys (cudaProfilerStart/Stop, NVTX) and perf support
    reporter.py                       # Stdout table printer + CSV/JSON file output
    database_backend.py               # Abstract DB backend + HTTP/PostgreSQL/InfluxDB implementations
    distributed_utils.py              # Multi-GPU barrier/warmup/timing protocol
  
  cpp/
    CMakeLists.txt                    # Top-level: fetches Google Benchmark, finds TE lib
    bench_common.h                    # CudaEventTimer, NVTX helpers, custom counter macros
    bench_common.cu                   # Implementation
    operator/
      CMakeLists.txt                  # Builds bench_operator executable
      bench_activation.cu
      bench_cast.cu
      bench_normalization.cu
      bench_transpose.cu
      bench_gemm.cu
      bench_softmax.cu
      bench_fused_attn.cu
      bench_fused_rope.cu
      bench_hadamard.cu
      bench_swizzle.cu
      bench_fused_router.cu
      bench_multi_tensor.cu
  
  pytorch/
    __init__.py
    conftest.py                       # pytest fixtures: config, timer, reporter, skip logic
    bench_linear.py
    bench_grouped_linear.py
    bench_layernorm.py
    bench_rmsnorm.py
    bench_layernorm_linear.py
    bench_layernorm_mlp.py
    bench_attention.py
    bench_multi_head_attention.py
    bench_transformer_layer.py
    bench_optimizers.py
    bench_quantization.py
    distributed/
      __init__.py
      run_distributed.py              # Worker script for torchrun (runs all distributed benches)
      bench_tensor_parallel.py
      bench_sequence_parallel.py
      bench_context_parallel.py
      bench_comm_gemm_overlap.py
  
  jax/
    __init__.py
    conftest.py
    bench_dense.py
    bench_layernorm.py
    bench_layernorm_mlp.py
    bench_attention.py
    bench_transformer_layer.py
    bench_quantization.py
  
  scripts/
    run_cpp.sh                        # Build + run C++ benchmarks
    run_pytorch.sh                    # Run PyTorch benchmarks
    run_jax.sh                        # Run JAX benchmarks
    run_all.sh                        # Run everything
    upload_results.py                 # Parse JSON/CSV results and upload to DB
    slurm/
      submit_single_gpu.sbatch
      submit_multi_gpu.sbatch
      interactive.sh
```

---

## Phase 1: Common Infrastructure

### 1.1 `benchmarks/config/sizes.py`

Defines shape dictionaries per operation category. Two regimes:

- **CPU-bound** (small): Expose host overhead, kernel launch latency, recipe management. E.g., `(1, 128)`, `(4, 256)`, `(16, 64)`.
- **GPU-bound** (large): Saturate compute/bandwidth. E.g., `(8192, 4096)`, `(16384, 12288)`.

Operations and their shape formats:
- Activation: `(rows, cols)` -- small: `[(16, 64), (32, 128)]`, large: `[(8192, 4096), (16384, 12288)]`
- Normalization: `(rows, hidden)` -- small: `[(16, 64), (32, 256)]`, large: `[(8192, 4096), (16384, 12288)]`
- Cast/Quantize: `(rows, cols)` -- same as activation
- Transpose: `(rows, cols)` -- same as activation
- GEMM: `(M, K, N)` -- small: `[(16, 64, 64), (32, 128, 128)]`, large: `[(4096, 4096, 4096), (8192, 4096, 16384)]`
- Attention: `(batch, num_heads, num_gqa_groups, head_dim, seq_q, seq_kv)` -- small: `[(1, 4, 4, 64, 32, 32)]`, large: `[(2, 32, 4, 128, 8192, 8192)]`
- Linear (PyTorch): `(batch_seq, in_features, out_features)` -- reuses GEMM shapes
- TransformerLayer: model configs `(batch, seq_len, hidden, num_heads, ffn_hidden)`

Each entry tagged with `"cpu_bound"` or `"gpu_bound"` label for filtering.

### 1.2 `benchmarks/config/benchmark_config.py`

`BenchmarkConfig` dataclass:
- `warmup_iterations: int = 5`
- `min_run_time: float = 5.0` (seconds, for `blocked_autorange`)
- `num_iterations: int = 20` (for manual timing loops)
- `profile_mode: bool = False` (fewer iters, enable NVTX/profiler markers)
- `size_filter: str = "all"` (`"cpu_bound"`, `"gpu_bound"`, `"all"`)
- `database_url: Optional[str] = None`
- `output_format: str = "stdout"` (`"stdout"`, `"csv"`, `"json"`)
- `output_file: Optional[str] = None`

`add_benchmark_args(parser)` function to add these as argparse options.
`from_pytest(config)` classmethod to construct from pytest config.

### 1.3 `benchmarks/common/result_types.py`

```python
@dataclass
class BenchmarkResult:
    name: str                    # e.g., "nvte_quantize" or "te.Linear"
    category: str                # "activation", "normalization", "cast", "gemm", "attention", etc.
    framework: str               # "cpp", "pytorch", "jax"
    shape: tuple
    dtype: str                   # "bf16", "fp16", "fp32"
    recipe: str                  # "bf16", "fp8_block", "mxfp8", "nvfp4"
    direction: str               # "fwd", "bwd", "fwd_bwd"
    regime: str                  # "cpu_bound", "gpu_bound"
    median_time_us: float
    mean_time_us: float
    min_time_us: float
    max_time_us: float
    std_time_us: float
    throughput_gbps: Optional[float]
    tflops: Optional[float]
    num_iterations: int
    num_gpus: int = 1
    metadata: dict               # auto-collected: hostname, GPU model, CUDA version, TE version, git commit
```

`collect_metadata()` function: reads `torch.cuda.get_device_properties()`, `transformer_engine.__version__`, `subprocess.check_output(["git", "rev-parse", "HEAD"])`, `socket.gethostname()`, etc.

### 1.4 `benchmarks/common/timing.py`

**PyTorch wrapper** around `torch.utils.benchmark.Timer`:
```python
class BenchmarkTimer:
    def measure(self, stmt, globals_dict, label, sub_label, min_run_time) -> BenchmarkResult
```
Uses `blocked_autorange(min_run_time=...)`. Returns median/mean/min/max from the `Measurement` object.

**JAX wrapper** with manual timing:
```python
class JAXBenchmarkTimer:
    def measure(self, fn, args, label, sub_label, warmup, iterations) -> BenchmarkResult
```
Uses `jax.block_until_ready()` + `time.perf_counter_ns()`.

### 1.5 `benchmarks/common/profiler_hooks.py`

- `cuda_profiler_region()` context manager: calls `torch.cuda.cudart().cudaProfilerStart()` / `cudaProfilerStop()`
- `nvtx_range(name)` context manager: calls `torch.cuda.nvtx.range_push()` / `range_pop()`
- `setup_profile_mode(config)`: if `config.profile_mode`, reduces iterations, enables NVTX

For nsys: user wraps benchmark invocation externally:
```bash
nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx,cudnn,cublas python -m pytest benchmarks/pytorch/bench_linear.py --profile
```

For perf (CPU profiling): no code changes needed, user wraps:
```bash
perf stat python -m pytest benchmarks/pytorch/bench_linear.py -k cpu_bound
perf record -g python -m pytest benchmarks/pytorch/bench_linear.py -k cpu_bound
```

### 1.6 `benchmarks/common/reporter.py`

`BenchmarkReporter`:
- `report(result)`: prints a formatted line to stdout (always)
- `report_table(results)`: prints a full table at the end with columns: Name, Shape, Recipe, Direction, Median(us), Mean(us), Throughput(GB/s)
- If `config.output_format == "csv"`: appends to CSV file
- If `config.output_format == "json"`: writes JSON array
- If `config.database_url`: calls database backend

### 1.7 `benchmarks/common/database_backend.py`

Abstract `DatabaseBackend` with `report(results: List[BenchmarkResult])` method.

Implementations:
- `HTTPBackend(url)`: POST JSON to a REST endpoint
- `PostgreSQLBackend(url)`: INSERT via `psycopg2` (optional import, skip if not installed)
- `InfluxDBBackend(url)`: InfluxDB line protocol over HTTP

Selected via URL scheme: `http://...` -> HTTP, `postgresql://...` -> PostgreSQL, `influxdb://...` -> InfluxDB.

DB URL passed via `--database-url` CLI arg or `TE_BENCH_DB_URL` env var.

---

## Phase 2: C++/CUDA Benchmarks

### 2.1 Build System

**`benchmarks/cpp/CMakeLists.txt`**:
- Fetches Google Benchmark via `FetchContent` (v1.8.3+)
- Reuses the same TE library discovery pattern from `tests/cpp/CMakeLists.txt` (find `libtransformer_engine.so` via Python import)
- Same CUDA architecture and include path setup
- `add_subdirectory(operator)`

**`benchmarks/cpp/operator/CMakeLists.txt`**:
- Creates `bench_operator` executable from all `bench_*.cu` files
- Links: `benchmark::benchmark`, `benchmark::benchmark_main`, `${TE_LIB}`, `CUDA::cudart`, `CUDA::nvrtc`, `CUDNN::cudnn`

### 2.2 `bench_common.h`

- Includes `tests/cpp/test_common.h` (reuses the `test::Tensor` class, `fillUniform`, `setRandomScale`, etc.)
- `CudaEventTimer`: RAII class using `cudaEventCreate/Record/Synchronize/ElapsedTime/Destroy` for precise GPU timing with Google Benchmark's `UseManualTime()`
- `SetThroughputCounters(state, bytes_read, bytes_written)`: sets `state.SetBytesProcessed()` and custom counters
- NVTX marker helpers for nsys integration

### 2.3 Benchmark Files (each follows the Google Benchmark pattern)

Each benchmark:
1. Sets up input/output tensors using `test::Tensor`
2. Runs warmup outside the benchmark loop
3. Uses `CudaEventTimer` for GPU measurement inside `for (auto _ : state)`
4. Reports throughput via custom counters
5. Registers both CPU-bound (small) and GPU-bound (large) size variants

**`bench_activation.cu`**: Benchmarks `nvte_gelu`, `nvte_dgelu`, `nvte_silu`, `nvte_dsilu`, `nvte_relu`, `nvte_drelu`, `nvte_swiglu`, `nvte_dswiglu`, `nvte_geglu`, `nvte_dgeglu`. Input types: bf16. Output: bf16 and fp8.

**`bench_cast.cu`**: Benchmarks `nvte_quantize` with scaling modes: delayed tensor, block 1D, block 2D, MXFP8, NVFP4. Both rowwise and columnwise. Also `nvte_dequantize`, `nvte_quantize_dbias`, `nvte_quantize_dbias_dgelu`.

**`bench_normalization.cu`**: Benchmarks `nvte_layernorm_fwd`, `nvte_layernorm_bwd`, `nvte_rmsnorm_fwd`, `nvte_rmsnorm_bwd`. Hidden sizes from 64 to 12288.

**`bench_transpose.cu`**: Benchmarks `nvte_transpose`, `nvte_cast_transpose`, `nvte_cast_transpose_dbias`, `nvte_multi_cast_transpose`.

**`bench_gemm.cu`**: Benchmarks `nvte_cublas_gemm_v2` with bf16, fp8, mxfp8 inputs. Various MKN shapes. Also `nvte_grouped_gemm` with varying group counts.

**`bench_softmax.cu`**: Benchmarks `nvte_scaled_softmax_forward/backward`, `nvte_scaled_masked_softmax_forward/backward`, `nvte_scaled_upper_triang_masked_softmax_forward/backward`.

**`bench_fused_attn.cu`**: Benchmarks `nvte_fused_attn_fwd`, `nvte_fused_attn_bwd`. Parametrized by batch, heads, head_dim, seq_len, mask type.

**`bench_fused_rope.cu`**: Benchmarks `nvte_fused_rope_forward`, `nvte_fused_rope_backward`.

**`bench_hadamard.cu`**: Benchmarks `nvte_hadamard_transform` and fusion variants.

**`bench_swizzle.cu`**: Benchmarks `nvte_swizzle_scaling_factors`, `nvte_unswizzle_scaling_factors`.

**`bench_fused_router.cu`**: Benchmarks `nvte_fused_topk_with_score_function`, `nvte_fused_moe_aux_loss`.

**`bench_multi_tensor.cu`**: Benchmarks `nvte_multi_tensor_adam`, `nvte_multi_tensor_sgd`, `nvte_multi_tensor_l2norm`.

### 2.4 Running C++ Benchmarks

```bash
# Build
cd benchmarks/cpp && cmake -GNinja -B build && cmake --build build

# Run all
./build/operator/bench_operator

# Filter by name
./build/operator/bench_operator --benchmark_filter="BM_nvte_quantize.*"

# Only small sizes
./build/operator/bench_operator --benchmark_filter=".*cpu_bound.*"

# JSON output for DB upload
./build/operator/bench_operator --benchmark_format=json --benchmark_out=results.json

# With nsys
nsys profile ./build/operator/bench_operator --benchmark_filter="BM_nvte_gemm.*"

# With perf
perf stat ./build/operator/bench_operator --benchmark_filter=".*cpu_bound.*"
```

---

## Phase 3: PyTorch Benchmarks

### 3.1 `conftest.py`

- `pytest_addoption`: adds `--profile`, `--size-filter`, `--database-url`, `--output-format`, `--output-file`, `--min-run-time`, `--recipe`
- `benchmark_config` fixture: builds `BenchmarkConfig` from pytest options
- `benchmark_timer` fixture: creates `BenchmarkTimer(config)`
- `benchmark_reporter` fixture: creates `BenchmarkReporter(config)`
- `skip_if_unavailable` helpers: checks FP8/MXFP8/NVFP4 hardware availability, skips with `pytest.skip()`

### 3.2 Benchmark Pattern (exemplified by `bench_linear.py`)

Each file is a standalone pytest module. Benchmarks are parametrized test functions:

```python
# Parametrize shapes (from sizes.py), recipes, directions
@pytest.mark.parametrize("shape", LINEAR_SIZES, ids=shape_id)
@pytest.mark.parametrize("recipe_name", ["bf16", "fp8_block", "mxfp8", "nvfp4"])
@pytest.mark.parametrize("direction", ["fwd_only", "fwd_bwd"])
def test_bench_linear(shape, recipe_name, direction, benchmark_config, benchmark_timer, benchmark_reporter):
    # Skip if recipe not available on hardware
    # Create te.Linear module
    # Create input tensors
    # Define run_step() closure
    # Time with benchmark_timer.measure(...)
    # Report with benchmark_reporter.report(result)
```

### 3.3 PyTorch Benchmark Files

**`bench_linear.py`**: `te.Linear` with all recipes. Shapes from `sizes.LINEAR_*`. Measures fwd and fwd+bwd.

**`bench_grouped_linear.py`**: `te.GroupedLinear`. Additional param: `num_gemms` (4, 8, 16).

**`bench_layernorm.py`**: `te.LayerNorm`. Hidden sizes 64-12288. With/without `zero_centered_gamma`.

**`bench_rmsnorm.py`**: `te.RMSNorm`. Same shape range as LayerNorm.

**`bench_layernorm_linear.py`**: `te.LayerNormLinear`. Combined norm+linear.

**`bench_layernorm_mlp.py`**: `te.LayerNormMLP`. Activation variants: gelu, silu, swiglu.

**`bench_attention.py`**: `te.DotProductAttention`. Backends: FusedAttention, FlashAttention. QKV layouts, mask types, GQA configs.

**`bench_multi_head_attention.py`**: `te.MultiheadAttention` (includes QKV projection).

**`bench_transformer_layer.py`**: `te.TransformerLayer`. Model configs matching common LLM sizes.

**`bench_optimizers.py`**: `te.FusedAdam`, `te.FusedSGD`. Realistic parameter group sizes.

**`bench_quantization.py`**: Standalone recipe overhead. Measures `autocast` context manager cost, amax history management.

### 3.4 Running PyTorch Benchmarks

```bash
# All PyTorch benchmarks
python -m pytest benchmarks/pytorch/ -v

# Single file
python -m pytest benchmarks/pytorch/bench_linear.py -v

# Single recipe
python -m pytest benchmarks/pytorch/bench_linear.py -v -k "fp8_block"

# CPU-bound only
python -m pytest benchmarks/pytorch/bench_linear.py -v -k "cpu_bound"

# GPU-bound only
python -m pytest benchmarks/pytorch/bench_linear.py -v -k "gpu_bound"

# With profiling (nsys)
nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx,cudnn,cublas \
  python -m pytest benchmarks/pytorch/bench_linear.py --profile -k "fp8_block and gpu_bound"

# With CPU profiling (perf)
perf stat python -m pytest benchmarks/pytorch/bench_linear.py -k "cpu_bound"

# With DB reporting
python -m pytest benchmarks/pytorch/ -v --database-url="postgresql://user:pass@host/benchmarks"
```

---

## Phase 4: Multi-GPU Benchmarks

### 4.1 `benchmarks/common/distributed_utils.py`

Key functions:
- `distributed_warmup(fn, args, warmup_iters, group)`: runs warmup iterations with barriers between ranks
- `distributed_measure(fn, args, num_iters, group)`: barrier -> warmup -> barrier -> timed loop -> barrier -> report (rank 0 only)
- `init_nccl_warmup(group)`: trivial all-reduce to prime NCCL

### 4.2 `benchmarks/pytorch/distributed/run_distributed.py`

Single worker script launched via `torchrun`. Structure:
1. `dist.init_process_group("nccl")`
2. Parse `--benchmark` arg to select which distributed bench to run (or `--all`)
3. Run selected benchmarks sequentially within the same process group
4. Each benchmark uses `distributed_measure()` for proper barrier/warmup protocol
5. Only rank 0 reports results

### 4.3 Distributed Benchmark Files

Importable modules with `run(rank, world_size, config)` entry points (called from `run_distributed.py`):

**`bench_tensor_parallel.py`**: `te.Linear` with `tp_group` set. Both column-parallel (`parallel_mode="column"`) and row-parallel (`parallel_mode="row"`).

**`bench_sequence_parallel.py`**: Modules with `sequence_parallel=True`.

**`bench_context_parallel.py`**: `DotProductAttention` with context parallelism enabled.

**`bench_comm_gemm_overlap.py`**: Communication-GEMM overlap (all-gather overlapped with GEMM, reduce-scatter overlapped with GEMM).

### 4.4 Running Multi-GPU Benchmarks

```bash
# All distributed benchmarks, 4 GPUs
torchrun --nproc_per_node=4 benchmarks/pytorch/distributed/run_distributed.py --all

# Specific benchmark
torchrun --nproc_per_node=8 benchmarks/pytorch/distributed/run_distributed.py --benchmark tensor_parallel

# With profiling
nsys profile --capture-range=cudaProfilerApi \
  torchrun --nproc_per_node=4 benchmarks/pytorch/distributed/run_distributed.py --benchmark tensor_parallel --profile
```

---

## Phase 5: JAX Benchmarks

### 5.1 `conftest.py`

Same pattern as PyTorch conftest but uses `JAXBenchmarkTimer` and handles JAX-specific setup (device placement, JIT compilation warmup).

### 5.2 JAX Benchmark Files

Each uses `jax.block_until_ready()` for synchronization. JIT warmup is part of the warmup phase.

**`bench_dense.py`**: `te.jax.flax.DenseGeneral`. FP8 quantization variants.

**`bench_layernorm.py`**: `te.jax.flax.LayerNorm`.

**`bench_layernorm_mlp.py`**: `te.jax.flax.LayerNormMLP`.

**`bench_attention.py`**: `te.jax.flax.DotProductAttention`.

**`bench_transformer_layer.py`**: `te.jax.flax.TransformerLayer`.

**`bench_quantization.py`**: Quantization primitives from `te.jax.quantize`.

### 5.3 Running JAX Benchmarks

```bash
python -m pytest benchmarks/jax/ -v
python -m pytest benchmarks/jax/bench_dense.py -v -k "gpu_bound"
```

---

## Phase 6: Slurm/Enroot Scripts

### 6.1 `scripts/slurm/submit_single_gpu.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=te-bench-1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=bench_%j.out
#SBATCH --container-image=${TE_BENCH_CONTAINER_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}
#SBATCH --container-mounts=${TE_BENCH_TE_PATH:-/opt/transformerengine}:/workspace/te

cd /workspace/te

# C++ benchmarks
bash benchmarks/scripts/run_cpp.sh

# PyTorch benchmarks
bash benchmarks/scripts/run_pytorch.sh

# JAX benchmarks (if JAX installed)
bash benchmarks/scripts/run_jax.sh
```

### 6.2 `scripts/slurm/submit_multi_gpu.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=te-bench-multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=bench_multi_%j.out
#SBATCH --container-image=${TE_BENCH_CONTAINER_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}
#SBATCH --container-mounts=${TE_BENCH_TE_PATH:-/opt/transformerengine}:/workspace/te

cd /workspace/te
srun --ntasks=1 torchrun --nproc_per_node=${SLURM_GPUS_PER_NODE} \
    benchmarks/pytorch/distributed/run_distributed.py --all
```

### 6.3 `scripts/slurm/interactive.sh`

```bash
#!/bin/bash
NUM_GPUS=${1:-1}
srun --nodes=1 --ntasks=1 --gpus-per-node=$NUM_GPUS --time=01:00:00 --pty \
    --container-image=${TE_BENCH_CONTAINER_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3} \
    --container-mounts=$(pwd):/workspace/te \
    bash
```

### 6.4 Helper Scripts

**`scripts/run_cpp.sh`**: cmake build + run `bench_operator` with JSON output
**`scripts/run_pytorch.sh`**: `python -m pytest benchmarks/pytorch/ -v` with optional args forwarding
**`scripts/run_jax.sh`**: `python -m pytest benchmarks/jax/ -v` with optional args forwarding
**`scripts/run_all.sh`**: calls all three + optional DB upload
**`scripts/upload_results.py`**: parses C++ JSON (Google Benchmark format) + Python results and uploads to DB

---

## Key Files to Reuse

- **`tests/cpp/test_common.h`** (`test::Tensor`, `fillUniform`, `setRandomScale`, type helpers) -- reuse directly in C++ benchmarks via include path
- **`tests/cpp/CMakeLists.txt`** -- template for TE library discovery and CUDA arch setup
- **`benchmarks/linear/benchmark_linear.py`** -- pattern for `torch.utils.benchmark.Timer`, NVTX, recipe handling
- **`transformer_engine/pytorch/quantization.py`** -- `FP8GlobalStateManager.is_*_available()` for hardware capability checks
- **`transformer_engine/common/include/transformer_engine/*.h`** -- C API function signatures for all kernels

---

## Verification Plan

1. **C++ benchmarks**: Build with `cmake -GNinja -B build && cmake --build build`. Run `./build/operator/bench_operator --benchmark_filter="BM_nvte_quantize"` and verify output shows timing + throughput.

2. **PyTorch benchmarks**: Run `python -m pytest benchmarks/pytorch/bench_linear.py -v -k "bf16 and gpu_bound and fwd_only"` and verify stdout table with timing results.

3. **JAX benchmarks**: Run `python -m pytest benchmarks/jax/bench_dense.py -v -k "gpu_bound"` and verify timing output.

4. **Multi-GPU**: Run `torchrun --nproc_per_node=2 benchmarks/pytorch/distributed/run_distributed.py --benchmark tensor_parallel` and verify only rank 0 reports.

5. **Profiler integration**: Run `nsys profile --capture-range=cudaProfilerApi python -m pytest benchmarks/pytorch/bench_linear.py --profile -k "bf16 and gpu_bound"` and verify `.nsys-rep` file is generated with NVTX markers visible.

6. **Selective execution**: Verify `-k "cpu_bound"`, `-k "gpu_bound"`, `-k "mxfp8"`, and `--benchmark_filter=` all correctly filter to the expected subset.

7. **DB reporting**: Run with `--database-url=http://localhost:8080/results` (or mock endpoint) and verify JSON payload is sent.

---

## Implementation Order

1. `benchmarks/config/` -- sizes.py, benchmark_config.py
2. `benchmarks/common/` -- result_types.py, timing.py, profiler_hooks.py, reporter.py, database_backend.py
3. `benchmarks/cpp/` -- CMakeLists, bench_common, then operator benchmarks one by one (cast first as simplest, then activation, normalization, gemm, etc.)
4. `benchmarks/pytorch/conftest.py` + `bench_linear.py` (proves pattern)
5. Remaining PyTorch single-GPU benchmarks
6. `benchmarks/common/distributed_utils.py` + distributed benchmarks
7. `benchmarks/jax/` -- conftest + all JAX benchmarks
8. `benchmarks/scripts/` -- shell scripts, Slurm sbatch files
9. `benchmarks/common/database_backend.py` (database upload)
