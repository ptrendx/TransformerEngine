## Testing

PyTorch and JAX tests require `pytest`. The `qa/**/test.sh` launchers install
the pytest version and plugins used by their suites and contain the
authoritative environment variables, configuration files, and command-line
options for specialized tests. Before running a test directly, find its
invocation or the closest analogous test in the relevant launcher and preserve
those settings. Use a focused `python -m pytest` command during iteration, then
use the launcher for the corresponding full suite.

### Focused framework tests

```bash
# PyTorch
python -m pytest -v tests/pytorch/path/to/test_file.py -k 'specific_case'
python -m pytest -v tests/pytorch/test_sanity.py

# JAX: use the repository pytest configuration
python -m pytest -c tests/jax/pytest.ini -v \
    tests/jax/path/to/test_file.py -k 'specific_case'
python -m pytest -c tests/jax/pytest.ini -v tests/jax/test_layer.py
```

### Common C++ and CUDA tests

Separately built after the main build.

```bash
cmake -GNinja -S tests/cpp -B tests/cpp/build \
    -DCMAKE_CUDA_ARCHITECTURES="$NVTE_CUDA_ARCHS"
cmake --build tests/cpp/build
ctest --test-dir tests/cpp/build --output-on-failure
```

For a focused GoogleTest run:

```bash
tests/cpp/build/operator/test_operator --gtest_list_tests
tests/cpp/build/operator/test_operator --gtest_filter='*RelevantPattern*'
```

### QA launchers

`CONTRIBUTING.rst` requires the `L0_*` checks. Their scripts default to
`TE_PATH=/opt/transformerengine` and `XML_LOG_DIR=/logs`; override both for a
local checkout:

```bash
TE_PATH="$PWD" XML_LOG_DIR=/tmp/te-test-logs \
    bash qa/L0_pytorch_unittest/test.sh
TE_PATH="$PWD" XML_LOG_DIR=/tmp/te-test-logs \
    bash qa/L0_jax_unittest/test.sh
TE_PATH="$PWD" XML_LOG_DIR=/tmp/te-test-logs \
    bash qa/L0_cppunittest/test.sh
```

Inspect a QA script before running it: some install dependencies or require a
particular container, GPU architecture, GPU count, or external repository.
Use the higher-level launcher for distributed, Megatron Core, FSDP, ONNX, and
attention-backend coverage rather than reconstructing its environment.

### Validation by changed area

- Shared Python: focused behavior through each affected frontend.
- PyTorch or JAX API: focused frontend test plus related integration coverage.
- Private framework binding: loaded-extension check and a focused numerical or
  integration test.
- Common C/CUDA: focused C++ operator test plus each affected frontend.
- Build or packaging: build in an appropriate clean environment, verify
  artifact paths, then use `qa/L0_pytorch_wheel/` or `qa/L0_jax_wheel/`.
- Distributed component: smallest supported multi-process case, then its QA
  launcher; record the GPU count and topology.
- Public API: tests, documentation, and compatibility across supported callers.

## Numerical and Backend Validation

For numerical or low-precision changes:

- Compare outputs and applicable gradients with an independent native-framework
  or higher-precision reference, not only shapes or successful execution.
- Cover the affected dtypes, recipes, layouts, boundary shapes, forward and
  backward paths, and GPU architectures.
- Base tolerances on the operation, dtype, accumulation, and reference. Do not
  loosen them solely to make a failing case pass.
- Record flags such as `NVTE_ALLOW_NONDETERMINISTIC_ALGO` that change expected
  numerical behavior.
- Exercise the intended backend and fallback behavior when the operation can
  use cuDNN, FlashAttention, unfused attention, Triton, NVRTC, or an
  architecture-specific kernel.
- Give skips and expected failures a capability-based or tracked-defect reason;
  do not remove a failing configuration before identifying its cause.

Choose tolerances for the quantity being compared, not merely for the lowest
precision used internally. For example, an MXFP8 GEMM that returns BF16 should
use BF16-appropriate output tolerances rather than FP8-appropriate tolerances
when its operands are chosen to be exactly representable in MXFP8. One way to
construct such operands is to generate random values, quantize and dequantize
them, and use the resulting values in both the reference and operation under
test. Apply this construction to every quantized operand, including parameters
held by a module such as weights, not only to the module's explicit inputs.

When reporting a GPU or distributed failure, include the exact command, input
shape/dtype/layout, relevant `NVTE_*` variables, GPU model and count, compute
capability, topology, and framework/CUDA/cuDNN/NCCL/driver versions. Also verify
which backend was selected and that the imported extension came from the
current worktree. Reduce failures across ranks, shapes, recipes, and backends
while preserving the observed behavior.
