/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <Python.h>
#include <pybind11/pybind11.h>

#include <optional>
#include <string>

#include "../common.h"
#include "ATen/ops/split_with_sizes.h"
#include "common.h"
#include "common/util/cuda_runtime.h"
#include "common/util/system.h"
#include "extensions.h"
#include "pybind.h"
#include "transformer_engine/transformer_engine.h"
#include "util.h"
#include <nvtx3/nvToolsExt.h>

namespace {

void* get_data_ptr(MaybeTensor tensor) {
  if (tensor.has_value()) return tensor->data_ptr();
  return nullptr;
}

size_t get_size(MaybeTensor tensor, int dim) {
  if (tensor.has_value()) return static_cast<size_t>(tensor->size(dim));
  return 0;
}

}  // namespace

namespace transformer_engine::pytorch {

namespace detail {

bool is_low_precision(const DType type) {
  return type == DType::kFloat8E4M3 || type == DType::kFloat8E5M2;
}

std::vector<size_t> getGemmOutputShape(const NVTEShape& A_shape, const bool transa,
                                       const NVTEShape& B_shape, const bool transb) {
  // Flatten outer dims to get 2D matrices
  const size_t A0 = product(A_shape, 0, A_shape.ndim - 1);
  const size_t A1 = A_shape.data[A_shape.ndim - 1];
  const size_t B0 = product(B_shape, 0, B_shape.ndim - 1);
  const size_t B1 = B_shape.data[B_shape.ndim - 1];

  // Check matrix dims
  NVTE_CHECK((transa ? A1 : A0) == (transb ? B0 : B1), "Invalid matrix dimensions for GEMM (A=(",
             A0, ",", A1, "), transa=", transa, ", B=(", B0, ",", B1, "), transb=", transb, ")");

  // Construct output dims
  std::vector<size_t> ret;
  if (transb) {
    ret.emplace_back(B1);
  } else {
    // Unflatten B0
    for (size_t i = 0; i < B_shape.ndim - 1; ++i) {
      ret.emplace_back(B_shape.data[i]);
    }
  }
  if (transa) {
    ret.emplace_back(A0);
  } else {
    ret.emplace_back(A1);
  }
  return ret;
}

bool checkGemmShape(const std::vector<size_t>& expected, const NVTEShape& actual) {
  if (expected.size() != actual.ndim) return false;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual.data[i]) return false;
  }
  return true;
}

std::pair<std::vector<size_t>, std::vector<size_t>> createSplits(
    const NVTEShape& full_shape,
    const std::optional<std::vector<size_t>>& splits_first_dim,
    const std::optional<std::vector<size_t>>& splits_last_dim,
    const size_t num_splits, const bool transpose) {
  std::pair<std::vector<size_t>, std::vector<size_t>> ret;
  int first_dim_index = transpose ? 1 : 0;
  int last_dim_index = 1 - first_dim_index;
  NVTE_CHECK(full_shape.ndim == 2, "Expected 2-dimensional input. Got ",
             full_shape.ndim, " dimensions.");
  if (splits_first_dim != std::nullopt) {
    NVTE_CHECK(splits_first_dim->size() == num_splits,
               "Provided splits need to be either a list of size num_gemms or None, got ",
               splits_first_dim);
    ret.first = *splits_first_dim;
  } else {
    size_t first_dim = full_shape.data[first_dim_index];
    if (splits_last_dim == std::nullopt && first_dim_index == 0) {
      // Both splits are None, divide [Nm, k] into N [m, k] matrices
      NVTE_CHECK(first_dim % num_splits,
                 "The first dimension ", first_dim, " cannot be split into ",
                 num_splits, " parts.");
      ret.first = std::vector<size_t>(num_splits, first_dim / num_splits);
    } else {
      // Second split exists or the tensor is transposed,
      // so the first dimension stays the same
      ret.first = std::vector<size_t>(num_splits, first_dim);
    }
  }
  if (splits_last_dim != std::nullopt) {
    NVTE_CHECK(splits_last_dim->size() == num_splits,
               "Provided splits need to be either a list of size num_gemms or None, got ",
               splits_last_dim);
    ret.second = *splits_last_dim;
  } else {
    size_t last_dim = full_shape.data[last_dim_index];
    if (splits_first_dim == std::nullopt && last_dim_index == 0) {
      // Both splits are None, divide [Nm, k] into N [m, k] matrices
      NVTE_CHECK(last_dim % num_splits,
                 "The last dimension in transposed matrix ", last_dim, " cannot be split into ",
                 num_splits, " parts.");
      ret.second = std::vector<size_t>(num_splits, last_dim / num_splits);
    } else {
      // If possible, we are splitting the first dimension
      ret.second = std::vector<size_t>(num_splits, last_dim);
    }
  }

  return ret;
}

std::vector<NVTETensor> makeTransformerEngineTensorSplit(
    const TensorWrapper& t,
    const std::vector<size_t>& splits_first_dim,
    const std::vector<size_t>& splits_last_dim,
    const size_t num_splits) {
  std::vector<NVTETensor> ret(num_splits, nullptr);
  nvte_tensor_split(t.data(), splits_first_dim.data(), splits_last_dim.data(),
                    num_splits, ret.data());
  return ret;
}

}  // namespace detail

std::pair<TensorWrapper, py::object> createOutputTensor(const std::vector<size_t>& shape,
                                                        DType dtype, py::handle quantizer) {
  std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
  return my_quantizer->create_tensor(shape, dtype);
}

std::vector<py::object> gemm(py::handle A, bool transa, py::handle B, bool transb, py::object D,
                             py::handle quantizer, std::optional<DType> out_dtype, MaybeTensor bias,
                             DType bias_type, bool gelu, MaybeTensor gelu_in, bool grad,
                             at::Tensor workspace, size_t workspaceSize, bool accumulate,
                             bool use_split_accumulator, CommOverlapCore* comm_overlap,
                             std::optional<CommOverlapType> comm_type, MaybeTensor extra_output,
                             bool bulk_overlap) {
  // Input tensors
  NVTE_CHECK(!A.is_none(), "Tensor A has not been provided");
  NVTE_CHECK(!B.is_none(), "Tensor B has not been provided");
  auto none = py::none();
  TensorWrapper A_tensor = makeTransformerEngineTensor(A, none);
  TensorWrapper B_tensor = makeTransformerEngineTensor(B, none);

  const bool low_precision =
      detail::is_low_precision(A_tensor.dtype()) || detail::is_low_precision(B_tensor.dtype());

  // Check tensor dimensions
  const auto& A_shape = A_tensor.shape();
  const auto& B_shape = B_tensor.shape();
  const auto& D_shape = detail::getGemmOutputShape(A_shape, transa, B_shape, transb);
  NVTE_CHECK(A_shape.ndim >= 1, "Tensor A needs to have at least 1 dimension");
  NVTE_CHECK(B_shape.ndim >= 1, "Tensor B needs to have at least 1 dimension");

  // Output tensor
  TensorWrapper D_tensor;
  if (D.is_none()) {
    DType output_dtype = out_dtype ? *out_dtype : A_tensor.dtype();
    std::tie(D_tensor, D) = createOutputTensor(D_shape, output_dtype, quantizer);
  } else {
    D_tensor = makeTransformerEngineTensor(D, quantizer);
    NVTE_CHECK(detail::checkGemmShape(D_shape, D_tensor.shape()),
               "GEMM output has invalid dims (expected ", std::to_string(D_shape), ", got ",
               std::to_string(D_tensor.shape()), ")");
    if (out_dtype) {
      NVTE_CHECK(*out_dtype == D_tensor.dtype(), "GEMM output has invalid dtype (expected ",
                 static_cast<int>(*out_dtype), ", found ", static_cast<int>(D_tensor.dtype()), ")");
    }
  }

  // Bias tensor
  TensorWrapper bias_tensor;
  MaybeTensor bias_grad = std::nullopt;
  if (bias.has_value()) {
    if (grad) {
      auto opts = torch::TensorOptions().dtype(GetATenDType(D_tensor.dtype())).device(torch::kCUDA);
      bias_grad = at::empty({static_cast<int64_t>(B_shape.data[B_shape.ndim - 1])}, opts);
      bias_tensor = makeTransformerEngineTensor(*bias_grad);
    } else {
      if (!bias->is_contiguous()) {
        bias = bias->contiguous();
      }
      bias_tensor = makeTransformerEngineTensor(*bias);
    }
  }

  // Activation input tensor
  MaybeTensor pre_gelu_out = std::nullopt;
  DType gelu_type = low_precision ? bias_type : D_tensor.dtype();
  if (gelu) {
    if (!grad) {
      auto dtype = GetATenDType(gelu_type);
      auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
      std::vector<int64_t> torch_shape;
      for (auto v : D_shape) {
        torch_shape.push_back(v);
      }
      pre_gelu_out = at::empty(torch_shape, opts);
    } else {
      if (gelu_in.has_value()) {
        pre_gelu_out = *gelu_in;
      }
    }
  }
  const auto gelu_shape = gelu ? D_shape : std::vector<size_t>{0};

  auto te_pre_gelu_out =
      makeTransformerEngineTensor(get_data_ptr(pre_gelu_out), gelu_shape, gelu_type);

  // Workspace
  auto te_workspace = makeTransformerEngineTensor(workspace.data_ptr(),
                                                  std::vector<size_t>{workspaceSize}, DType::kByte);

  // Set an external SM Margin to all the GEMMs.
  // This comes in handy when DP is overlapped with GEMMs
  const int device_id = at::cuda::current_device();
  const int sm_count = transformer_engine::cuda::sm_count(device_id);
  int num_math_sms = sm_count - transformer_engine::getenv<int>("NVTE_EXT_MARGIN_SM", sm_count);

  // Keep the swizzled scaling factor tensors alive during the GEMM.
  std::vector<std::optional<at::Tensor>> swizzled_scale_inverses_list;
  auto main_stream = at::cuda::getCurrentCUDAStream();
  if (A_tensor.numel() != 0 && B_tensor.numel() != 0) {
    // Optionally swizzle the scaling factors
    swizzled_scale_inverses_list.emplace_back(std::move(swizzle_scaling_factors(A_tensor, transa)));
    swizzled_scale_inverses_list.emplace_back(
        std::move(swizzle_scaling_factors(B_tensor, !transb)));

    if (comm_overlap) {
      // Prepare extra output tensor
      TensorWrapper extra_output_tensor;
      if (extra_output.has_value()) {
        extra_output_tensor = makeTransformerEngineTensor(*extra_output);
      } else {
        extra_output_tensor =
            makeTransformerEngineTensor(nullptr, std::vector<size_t>{0}, DType::kByte);
      }

      // Direct GEMM call to the correct overlap
      if (bulk_overlap) {
        comm_overlap->bulk_overlap(A_tensor, transa, B_tensor, transb, D_tensor, bias_tensor,
                                   te_pre_gelu_out, te_workspace, grad, accumulate,
                                   use_split_accumulator, comm_type.value(), extra_output_tensor,
                                   main_stream);
      } else if (comm_type.value() == CommOverlapType::AG) {
        if (comm_overlap->is_atomic_gemm()) {
          comm_overlap->atomic_gemm_overlap_ag(A_tensor, transa, B_tensor, transb, D_tensor,
                                               bias_tensor, te_pre_gelu_out, te_workspace, grad,
                                               accumulate, use_split_accumulator,
                                               extra_output_tensor, main_stream);
        } else {
          comm_overlap->split_overlap_ag(A_tensor, transa, B_tensor, transb, D_tensor, bias_tensor,
                                         te_pre_gelu_out, te_workspace, grad, accumulate,
                                         use_split_accumulator, extra_output_tensor, main_stream);
        }
      } else {
        if (comm_overlap->is_atomic_gemm()) {
          comm_overlap->atomic_gemm_overlap_rs(A_tensor, transa, B_tensor, transb, D_tensor,
                                               bias_tensor, te_pre_gelu_out, te_workspace, grad,
                                               accumulate, use_split_accumulator,
                                               extra_output_tensor, main_stream);
        } else {
          comm_overlap->split_overlap_rs(A_tensor, transa, B_tensor, transb, D_tensor, bias_tensor,
                                         te_pre_gelu_out, te_workspace, grad, accumulate,
                                         use_split_accumulator, extra_output_tensor, main_stream);
        }
      }
    } else {
      // Launch GEMM
      nvte_cublas_gemm(A_tensor.data(), B_tensor.data(), D_tensor.data(), bias_tensor.data(),
                       te_pre_gelu_out.data(), transa, transb, grad, te_workspace.data(),
                       accumulate, use_split_accumulator, num_math_sms, main_stream);
    }
  } else {
    if (D_tensor.numel() != 0 && !accumulate) {
      D_tensor.zero_(main_stream);
    }
    if (bias.has_value()) {
      if (bias->numel() != 0 && grad) {
        bias_grad->zero_();
      }
    }
  }

  // Pack outputs
  std::vector<py::object> out;
  out.emplace_back(std::move(D));
  out.emplace_back(py::cast(bias_grad));
  if (gelu && !grad) {
    out.emplace_back(py::cast(*pre_gelu_out));
  } else {
    out.emplace_back(py::none());
  }
  if (extra_output.has_value()) {
    out.emplace_back(py::cast(extra_output));
  } else {
    out.emplace_back(py::none());
  }
  return out;
}

}  // namespace transformer_engine::pytorch

void te_atomic_gemm(at::Tensor A, at::Tensor A_scale_inverse, transformer_engine::DType A_type,
                    std::vector<int64_t> A_scaling_mode, bool transa, at::Tensor B,
                    at::Tensor B_scale_inverse, transformer_engine::DType B_type,
                    std::vector<int64_t> B_scaling_mode, bool transb, at::Tensor D,
                    at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
                    at::Tensor bias, transformer_engine::DType bias_type, at::Tensor pre_gelu_out,
                    bool grad, at::Tensor workspace, size_t workspaceSize, bool accumulate,
                    bool use_split_accumulator, int math_sm_count, int m_split, int n_split,
                    bool gemm_producer, at::Tensor counter) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;

  // TODO: Handle scaling modes
  NVTEScalingMode nvte_scaling_modeA = NVTE_DELAYED_TENSOR_SCALING;
  NVTEScalingMode nvte_scaling_modeB = NVTE_DELAYED_TENSOR_SCALING;

  auto te_A = makeTransformerEngineTensor(
      A.data_ptr(), {static_cast<size_t>(A.size(0)), static_cast<size_t>(A.size(1))}, A_type,
      nullptr, nullptr, A_scale_inverse.data_ptr(), getTensorShape(A_scale_inverse),
      nvte_scaling_modeA);
  auto te_B = makeTransformerEngineTensor(
      B.data_ptr(), {static_cast<size_t>(B.size(0)), static_cast<size_t>(B.size(1))}, B_type,
      nullptr, nullptr, B_scale_inverse.data_ptr(), getTensorShape(B_scale_inverse),
      nvte_scaling_modeB);
  // TODO: D_scale_inv cannot be nullptr when D_type is FP8.
  auto te_D = makeTransformerEngineTensor(
      D.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(D.size(0)), static_cast<size_t>(D.size(1))}, D_type,
      D_amax.data_ptr(), D_scale.data_ptr(), nullptr);
  auto te_bias = makeTransformerEngineTensor(
      bias.data_ptr(), std::vector<size_t>{static_cast<size_t>(bias.size(0))}, bias_type);
  auto te_counter = makeTransformerEngineTensor(
      counter.data_ptr(), std::vector<size_t>{static_cast<size_t>(counter.size(0))}, DType::kInt32);

  const auto gelu_shape = pre_gelu_out.data_ptr() == nullptr
                              ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                              : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                    static_cast<size_t>(pre_gelu_out.size(1))};
  auto te_pre_gelu_out = makeTransformerEngineTensor(
      pre_gelu_out.data_ptr(), gelu_shape, GetTransformerEngineDType(pre_gelu_out.scalar_type()));
  auto te_workspace = makeTransformerEngineTensor(workspace.data_ptr(),
                                                  std::vector<size_t>{workspaceSize}, DType::kByte);

  nvte_cublas_atomic_gemm(te_A.data(), te_B.data(), te_D.data(), te_bias.data(),
                          te_pre_gelu_out.data(), transa, transb, grad, te_workspace.data(),
                          accumulate, use_split_accumulator, math_sm_count, m_split, n_split,
                          gemm_producer, te_counter.data(), at::cuda::getCurrentCUDAStream());
}

std::vector<std::vector<at::Tensor>> te_general_grouped_gemm(
    std::vector<py::handle> A, bool transa, std::vector<py::handle> B, bool transb,
    std::optional<std::vector<at::Tensor>> D, transformer_engine::DType D_type,
    std::optional<std::vector<int64_t>> m_splits,
    std::optional<std::vector<int64_t>> n_splits,
    std::optional<std::vector<int64_t>> k_splits,
    std::vector<at::Tensor> bias,
    transformer_engine::DType bias_type, std::vector<at::Tensor> pre_gelu_out,
    bool grad, std::vector<at::Tensor> workspace, size_t workspaceSize, bool accumulate,
    bool use_split_accumulator, int math_sm_count) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;
  
  nvtxRangePush("te_general_grouped_gemm");
  std::vector<NVTETensor> te_A_vector, te_B_vector, te_D_vector, te_bias_vector,
      te_pre_gelu_out_vector, te_workspace_vector;
  auto none = py::none();
  std::vector<TensorWrapper> A_wrappers, B_wrappers;
  size_t num_splits = 1;
  if (m_splits != std::nullopt) {
    NVTE_CHECK(n_splits == std::nullopt && k_splits == std::nullopt,
               "Out of m_splits, n_splits and k_splits only 1 can be provided!");
  } else if (n_splits != std::nullopt) {
    NVTE_CHECK(m_splits == std::nullopt && k_splits == std::nullopt,
               "Out of m_splits, n_splits and k_splits only 1 can be provided!");
  } else if (k_splits != std::nullopt) {
    NVTE_CHECK(m_splits == std::nullopt && n_splits == std::nullopt,
               "Out of m_splits, n_splits and k_splits only 1 can be provided!");
  }

  A_wrappers.reserve(num_splits);
  B_wrappers.reserve(num_splits);
  if (A.size() == num_splits) {
    for (const auto& a : A) {
      A_wrappers.emplace_back(makeTransformerEngineTensor(a, none));
    }
  } else {
    NVTE_CHECK(A.size() == 1,
                 "Grouped GEMM expects the A input to be a list of either "
                 "1 or num_gemms tensors. Got ",
                 A.size(), " elements, while num_gemms is ", num_splits, ".");
  }
  if (B.size() == num_splits) {
    for (const auto& b : B) {
      B_wrappers.emplace_back(makeTransformerEngineTensor(b, none));
    }
  } else {
    NVTE_CHECK(B.size() == 1,
                 "Grouped GEMM expects the B input to be a list of either "
                 "1 or num_gemms tensors. Got ",
                 B.size(), " elements, while num_gemms is ", num_splits, ".");
  }
  std::vector<at::Tensor> D_tensors;
  bool single_output = false;

  if (D != std::nullopt) {
    if (m_splits.size() == D->size()) {
      // each output is separate
    } else {
      NVTE_CHECK(D->size() == 1,
                 "Grouped GEMM expects the provided output to be a list of either "
                 "1 or num_gemms tensors. Got ",
                 D->size(), " elements, while num_gemms is ", num_splits, ".");
      single_output = true;
    }
    D_tensors = *D;
  } else {
    D_tensors.reserve(m_splits.size());
    for (size_t i = 0; i < m_splits.size(); ++i) {
      const auto& te_A = A_wrappers[i];
      const auto& te_B = B_wrappers[i];

      // if there is single output
      at::Tensor out_tensor;
      auto size_t_shape =
          pytorch::detail::getGemmOutputShape(te_A.shape(), transa, te_B.shape(), transb);
      }
  }
  // Keep the swizzled scaling factor tensors alive during the GEMMs.
  std::vector<std::optional<at::Tensor>> swizzled_scale_inverses_list;


  void* output_data_ptr = nullptr;
  if (single_output) {
    output_data_ptr = (*D)[0].data_ptr();
  }

  for (size_t i = 0; i < A.size(); i++) {
    const auto& te_A = A_wrappers[i];
    const auto& te_B = B_wrappers[i];

    // if there is single output
    at::Tensor out_tensor;
    auto size_t_shape =
        pytorch::detail::getGemmOutputShape(te_A.shape(), transa, te_B.shape(), transb);
    bool D_numel_is_zero = false;
    std::vector<int64_t> D_shape;
    for (size_t t : size_t_shape) {
      D_shape.push_back(t);
      if (t == 0) {
        D_numel_is_zero = true;
      }
    }
    auto dtype = GetATenDType(D_type);
    auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
    if (single_output) {
      if (output_data_ptr == nullptr) {
        out_tensor = at::empty(D_shape, opts);
      } else {
        // We need to check !D_numel_is_zero because if the final input portion has zero elements,
        // output_data_ptr would point beyond the allocated memory of D. This would cause
        // at::from_blob to fail as it would reference memory not allocated by CUDA.
        if (!D_numel_is_zero) {
          out_tensor = at::from_blob(output_data_ptr, D_shape, opts);
        }
      }
      char* char_ptr = reinterpret_cast<char*>(output_data_ptr);
      char_ptr += D_shape[0] * D_shape[1] * (*D)[0].element_size();
      output_data_ptr = reinterpret_cast<void*>(char_ptr);
      D_vectors.emplace_back(out_tensor);
    } else {
      if (D == std::nullopt) {
        auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
        out_tensor = at::empty(D_shape, opts);
        D_vectors.emplace_back(out_tensor);
      } else {
        out_tensor = (*D)[i];
      }
    }

    if (te_A.numel() == 0 || te_B.numel() == 0) {
      if (out_tensor.numel() != 0 && !accumulate) out_tensor.zero_();
      if (bias[i].numel() != 0 && grad) {
        bias[i].zero_();
      }
      if (pre_gelu_out[i].numel() != 0) pre_gelu_out[i].zero_();
      continue;
    }

    // Optionally swizzle the scaling factors
    swizzled_scale_inverses_list.emplace_back(std::move(swizzle_scaling_factors(te_A, transa)));
    swizzled_scale_inverses_list.emplace_back(std::move(swizzle_scaling_factors(te_B, !transb)));

    auto te_D = makeTransformerEngineTensor(out_tensor);
    auto te_bias = makeTransformerEngineTensor(bias[i]);
    auto te_pre_gelu_out = makeTransformerEngineTensor(pre_gelu_out[i]);

    const auto gelu_shape = pre_gelu_out[i].data_ptr() == nullptr
                                ? std::vector<size_t>{static_cast<size_t>(te_pre_gelu_out.size(0))}
                                : std::vector<size_t>{static_cast<size_t>(te_pre_gelu_out.size(0)),
                                                      static_cast<size_t>(te_pre_gelu_out.size(1))};

    DType gelu_type = bias_type;
    te_pre_gelu_out =
        makeTransformerEngineTensor(get_data_ptr(pre_gelu_out[i]), gelu_shape, gelu_type);

    te_A_vector.emplace_back(te_A.data());
    te_B_vector.emplace_back(te_B.data());
    te_D_vector.emplace_back(te_D.data());
    te_bias_vector.emplace_back(te_bias.data());
    te_pre_gelu_out_vector.emplace_back(te_pre_gelu_out.data());

    wrappers.emplace_back(std::move(te_A));
    wrappers.emplace_back(std::move(te_B));
    wrappers.emplace_back(std::move(te_D));
    wrappers.emplace_back(std::move(te_bias));
    wrappers.emplace_back(std::move(te_pre_gelu_out));
  }
  for (size_t i = 0; i < workspace.size(); i++) {
    auto wsp = makeTransformerEngineTensor(workspace[i].data_ptr(),
                                           std::vector<size_t>{workspaceSize}, DType::kByte);
    te_workspace_vector.emplace_back(wsp.data());
    wrappers.emplace_back(std::move(wsp));
  }
  // For now, we only have multi-stream cublas backend.
  nvte_multi_stream_cublas_gemm(te_A_vector.data(), te_B_vector.data(), te_D_vector.data(),
                                te_bias_vector.data(), te_pre_gelu_out_vector.data(),
                                te_A_vector.size(), transa, transb, grad,
                                te_workspace_vector.data(), accumulate, use_split_accumulator,
                                math_sm_count, at::cuda::getCurrentCUDAStream());
  nvtxRangePop();
  return {std::move(D_tensors), bias};
}

std::vector<std::vector<at::Tensor>> te_general_grouped_gemm2(
    std::vector<py::handle> A, bool transa, std::vector<py::handle> B, bool transb,
    std::optional<std::vector<at::Tensor>> D, transformer_engine::DType D_type,
    const size_t num_gemms,
    std::optional<std::vector<size_t>> m_splits,
    std::optional<std::vector<size_t>> n_splits,
    std::optional<std::vector<size_t>> k_splits,
    bool single_output,
    std::vector<at::Tensor> bias,
    transformer_engine::DType bias_type, std::vector<at::Tensor> pre_gelu_out,
    bool grad, std::vector<at::Tensor> workspace, size_t workspaceSize, bool accumulate,
    bool use_split_accumulator, int math_sm_count) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;
  using transformer_engine::pytorch::detail::createSplits;
  using transformer_engine::pytorch::detail::getGemmOutputShape;
  using transformer_engine::pytorch::detail::makeTransformerEngineTensorSplit;

  nvtxRangePush("te_general_grouped_gemm");
  std::vector<NVTETensor> te_A, te_B, te_D, te_bias,
      te_pre_gelu_out, te_workspace;
  // Keep the swizzled scaling factor tensors alive during the GEMMs.
  std::vector<std::optional<at::Tensor>> swizzled_scale_inverses_list;

  auto none = py::none();
  if (m_splits != std::nullopt) {
    NVTE_CHECK(n_splits == std::nullopt && k_splits == std::nullopt,
               "Out of m_splits, n_splits and k_splits only 1 can be provided!");
  } else if (n_splits != std::nullopt) {
    NVTE_CHECK(m_splits == std::nullopt && k_splits == std::nullopt,
               "Out of m_splits, n_splits and k_splits only 1 can be provided!");
  } else if (k_splits != std::nullopt) {
    NVTE_CHECK(m_splits == std::nullopt && n_splits == std::nullopt,
               "Out of m_splits, n_splits and k_splits only 1 can be provided!");
  }

  if (A.size() == num_gemms) {
    te_A.reserve(num_gemms);

    if (m_splits == std::nullopt) {
      m_splits = std::vector<size_t>();
      m_splits->reserve(num_gemms);
    }
    if (k_splits == std::nullopt) {
      k_splits = std::vector<size_t>();
      k_splits->reserve(num_gemms);
    }
    for (size_t i = 0; i < A.size(); ++i) {
      auto wrapper = makeTransformerEngineTensor(A[i], none);
      swizzled_scale_inverses_list.emplace_back(swizzle_scaling_factors(wrapper, transa));
      NVTETensor t = wrapper.owned_data();
      te_A.emplace_back(t);
      const NVTEShape& s = nvte_tensor_shape(t);
      int m_dim = transa ? 0 : 1;
      int k_dim = 1 - m_dim;
      if (m_splits->size() <= i) {
        m_splits->emplace_back(s.data[m_dim]);
      } else {
        NVTE_CHECK((*m_splits)[i] == s.data[m_dim]);
      }
      if (k_splits->size() <= i) {
        k_splits->emplace_back(s.data[k_dim]);
      } else {
        NVTE_CHECK((*k_splits)[i] == s.data[k_dim]);
      }
    }
  } else {
    NVTE_CHECK(A.size() == 1,
                 "Grouped GEMM expects the A input to be a list of either "
                 "1 or num_gemms tensors. Got ",
                 A.size(), " elements, while num_gemms is ", num_gemms, ".");
    const auto& temp_wrapper = makeTransformerEngineTensor(A[0], none);
    std::tie(k_splits, m_splits) = createSplits(temp_wrapper.shape(), k_splits,
                                                m_splits, num_gemms, transa);
    te_A = makeTransformerEngineTensorSplit(temp_wrapper, 
                                            transa ? *m_splits : *k_splits,
                                            transa ? *k_splits : *m_splits,
                                            num_gemms);
    for (auto& t : te_A) {
      TensorWrapper wrapper(t);
      swizzled_scale_inverses_list.emplace_back(swizzle_scaling_factors(wrapper, transa));
      t = wrapper.owned_data();
    }
  }
  if (B.size() == num_gemms) {
    te_B.reserve(num_gemms);
    if (n_splits == std::nullopt) {
      n_splits = std::vector<size_t>();
      n_splits->reserve(num_gemms);
    }
    if (k_splits == std::nullopt) {
      k_splits = std::vector<size_t>();
      k_splits->reserve(num_gemms);
    }
    for (size_t i = 0; i < B.size(); ++i) {
      NVTETensor t = makeTransformerEngineTensor(B[i], none).owned_data();
      te_B.emplace_back(t);
      const NVTEShape& s = nvte_tensor_shape(t);
      int n_dim = transb ? 1 : 0;
      int k_dim = 1 - n_dim;
      if (n_splits->size() <= i) {
        n_splits->emplace_back(s.data[n_dim]);
      } else {
        NVTE_CHECK((*n_splits)[i] == s.data[n_dim]);
      }
      if (k_splits->size() <= i) {
        k_splits->emplace_back(s.data[k_dim]);
      } else {
        NVTE_CHECK((*k_splits)[i] == s.data[k_dim]);
      }
    }
  } else {
    NVTE_CHECK(B.size() == 1,
                 "Grouped GEMM expects the B input to be a list of either "
                 "1 or num_gemms tensors. Got ",
                 B.size(), " elements, while num_gemms is ", num_gemms, ".");
    const auto& temp_wrapper = makeTransformerEngineTensor(B[0], none);
    std::tie(n_splits, k_splits) = createSplits(temp_wrapper.shape(), n_splits,
                                                k_splits, num_gemms, transb);
    te_B = makeTransformerEngineTensorSplit(temp_wrapper,
                                            transb ? *k_splits : *n_splits,
                                            transb ? *n_splits : *k_splits,
                                            num_gemms);
    for (auto& t : te_B) {
      TensorWrapper wrapper(t);
      swizzled_scale_inverses_list.emplace_back(swizzle_scaling_factors(wrapper, !transb));
      t = wrapper.owned_data();
    }
  }

  // m_splits, n_splits, k_splits ready
  // te_A, te_B ready

  for (size_t i = 0; i < num_gemms; i++) {
    te_bias.emplace_back(makeTransformerEngineTensor(bias[i]).owned_data());
    auto t = makeTransformerEngineTensor(pre_gelu_out[i]);

    const auto gelu_shape = pre_gelu_out[i].data_ptr() == nullptr
                                ? std::vector<size_t>{static_cast<size_t>(t.size(0))}
                                : std::vector<size_t>{static_cast<size_t>(t.size(0)),
                                                      static_cast<size_t>(t.size(1))};

    DType gelu_type = bias_type;
    te_pre_gelu_out.emplace_back(
        makeTransformerEngineTensor(get_data_ptr(pre_gelu_out[i]), gelu_shape, gelu_type).owned_data());
  }

  // te_bias, te_pre_gelu_out ready

  for (size_t i = 0; i < workspace.size(); i++) {
    auto wsp = makeTransformerEngineTensor(workspace[i].data_ptr(),
                                           std::vector<size_t>{workspaceSize}, DType::kByte);
    te_workspace.emplace_back(wsp.owned_data());
  }

  // te_workspace ready

  std::vector<at::Tensor> D_tensors;

  if (D != std::nullopt) {
    if (num_gemms == D->size()) {
      // each output is separate
      for (const auto& d: *D) {
        te_D.emplace_back(makeTransformerEngineTensor(d).owned_data());
      }
      NVTE_CHECK(num_gemms == 1 || single_output == false);
    } else {
      NVTE_CHECK(D->size() == 1,
                 "Grouped GEMM expects the provided output to be a list of either "
                 "1 or num_gemms tensors. Got ",
                 D->size(), " elements, while num_gemms is ", num_gemms, ".");
      NVTE_CHECK(single_output == true);
      const auto& temp_wrapper = makeTransformerEngineTensor((*D)[0]);

      te_D = makeTransformerEngineTensorSplit(temp_wrapper, *n_splits, *m_splits, num_gemms);
    }
    D_tensors = *D;
  } else {
    int64_t total_size = 0;
    std::vector<int64_t> splits;
    splits.reserve(num_gemms);
    te_D.reserve(num_gemms);
    for (size_t i = 0; i < num_gemms; ++i) {
      total_size += (*n_splits)[i] * (*m_splits)[i];
      splits.push_back((*n_splits)[i] * (*m_splits)[i]);
    }
    auto dtype = GetATenDType(D_type);
    auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
    at::Tensor memory = at::empty(total_size, opts);
    D_tensors = at::split_with_sizes(memory, splits);
    for (size_t i = 0; i < num_gemms; ++i) {
      D_tensors[i] = D_tensors[i].view(std::vector<int64_t>{
          static_cast<int64_t>((*n_splits)[i]),
          static_cast<int64_t>((*m_splits)[i])});
      te_D.emplace_back(makeTransformerEngineTensor(D_tensors[i]));
    }
  }

  // For now, we only have multi-stream cublas backend.
  nvte_multi_stream_cublas_gemm(te_A.data(), te_B.data(), te_D.data(),
                                te_bias.data(), te_pre_gelu_out.data(),
                                num_gemms, transa, transb, grad,
                                te_workspace.data(), accumulate, use_split_accumulator,
                                math_sm_count, at::cuda::getCurrentCUDAStream());
  for (auto& t : te_A) {
    nvte_destroy_tensor(t);
  }
  for (auto& t : te_B) {
    nvte_destroy_tensor(t);
  }
  for (auto& t : te_D) {
    nvte_destroy_tensor(t);
  }
  for (auto& t : te_bias) {
    nvte_destroy_tensor(t);
  }
  for (auto& t : te_workspace) {
    nvte_destroy_tensor(t);
  }
  for (auto& t : te_pre_gelu_out) {
    nvte_destroy_tensor(t);
  }
  nvtxRangePop();

  return {std::move(D_tensors), bias};
}
