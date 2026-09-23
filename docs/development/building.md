## Build and Environment

Use a compatible NGC PyTorch or JAX development container when possible. For a
host build, follow `docs/installation.rst`. The selected framework must be
installed before Transformer Engine because the root build imports it while
configuring the framework extension. Development builds therefore use
`--no-build-isolation`.

Initialize the pinned source dependencies before the first build:

```bash
git submodule update --init --recursive
```

The root editable build is the normal development build. Run it from the
repository root. The frontend is detected automatically but can also be selected explicitly
via NVTE_FRAMEWORK environment variable during build:

```bash
NVTE_FRAMEWORK=pytorch python -m pip install -e . -v --no-build-isolation
NVTE_FRAMEWORK=jax python -m pip install -e . -v --no-build-isolation
NVTE_FRAMEWORK=pytorch,jax python -m pip install -e . -v --no-build-isolation
NVTE_FRAMEWORK=none python -m pip install -e . -v --no-build-isolation
```

Common build controls are:

- `NVTE_CUDA_ARCHS`: semicolon-separated target compute capabilities. For
  development, set this to the exact architecture being tested to avoid
  compiling unused variants. On Blackwell, use the architecture-specific
  target such as `100a` rather than the `100` family selector, which expands
  to multiple variants. Use a family selector or explicit list when building
  for multiple GPU architectures or producing portable artifacts.
- `MAX_JOBS` or `NVTE_BUILD_MAX_JOBS`: concurrent compilation jobs.
- `NVTE_BUILD_THREADS_PER_JOB`: threads within a build job.
- `NVTE_USE_CCACHE=1` and `NVTE_CCACHE_BIN`: compiler caching.
- `NVTE_CMAKE_BUILD_DIR`: alternate common-library build directory.
- `NVTE_BUILD_DEBUG=1`: debug build.
- `CUDA_HOME`, `CUDA_PATH`, `CUDNN_PATH`, and `CXX`: toolchain selection.

Before limiting compilation to a small fixed number of jobs or threads, check
the number of CPU cores available on the system and choose concurrency that
fits the current machine.

Optional native components add discovery and linking requirements:

- `NVTE_WITH_NCCL_EP`: NCCL expert parallelism; enabled by default for
  applicable Hopper-or-newer targets and requires compatible NCCL headers and
  libraries.
- `NVTE_UB_WITH_MPI=1`: MPI userbuffers bootstrap; requires `MPI_HOME`.
- `NVTE_ENABLE_NVSHMEM=1`: NVSHMEM; requires `NVSHMEM_HOME`.
- `NVTE_WITH_CUBLASMP=1`: cuBLASMp; may require `CUBLASMP_HOME`.
- `NVTE_WITH_CUSOLVERMP=1`: cuSolverMp; may require `CUSOLVERMP_HOME`.

See `docs/envvars.rst`, `setup.py`, `build_tools/`, and
`transformer_engine/common/CMakeLists.txt` for less common settings.

### Rebuild scope

- Python-only change: no rebuild after an editable install; restart processes
  that already imported the module.
- C++ change, either in common code or a framework binding: rerun the same build
  command and configuration used to build Transformer Engine before the change.
  This preserves the selected frontends, architecture targets, and optional
  features.
- Build logic, compiler flags, optional features, or framework selection:
  rerun the full editable build; use a clean build tree if its generated state
  is incompatible.
- Submodule source or revision: rebuild native targets and validate the
  dependent component.

CMake normally reuses `build/cmake`. Before removing a build tree, check
`NVTE_CMAKE_BUILD_DIR` and the verbose build output. A clean tree is commonly
needed after changing toolchains, CMake generators, architecture targets, or
optional native dependencies.

### Verify the imported build

NGC images may already contain Transformer Engine. After building, confirm that
the framework sees a GPU and that TE resolves to the current worktree:

Sandboxing may block access to GPU devices. If a GPU visibility check fails or
reports no devices, do not conclude that the system has no GPUs until the check
has also been attempted outside the sandbox with the required permission.

```bash
python -c 'import torch, transformer_engine, transformer_engine.pytorch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(transformer_engine.__file__)'
```

```bash
python -c 'import jax, transformer_engine, transformer_engine.jax; print(jax.__version__, jax.devices()); print(transformer_engine.__file__)'
```
