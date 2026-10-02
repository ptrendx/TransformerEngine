/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

#include <limits>
#include <stdexcept>
#include <string>
namespace py = pybind11;
namespace transformer_engine::pytorch {
namespace {
struct Metadata {
  const void* storage;
  at::IntArrayRef shape;
  at::IntArrayRef stride;
  int64_t offset;
  int64_t elements;
  explicit Metadata(const at::Tensor& t)
      : storage(t.storage().data_ptr().get()),
        shape(t.sizes()),
        stride(t.strides()),
        offset(t.storage_offset()),
        elements(t.numel()) {}
};
std::string classify(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                     const std::string& fmt, const std::string& qfmt, const std::string& kvfmt) {
  const Metadata a(q), b(k), c(v);
  const bool shared_qk = a.storage == b.storage;
  const bool shared_kv = b.storage == c.storage;
  const bool shared_qkv = shared_qk && shared_kv;
  const bool same_shapes = a.shape == b.shape && a.shape == c.shape;
  const bool same_kv_shapes =
      b.shape.slice(0, b.shape.size() - 1) == c.shape.slice(0, c.shape.size() - 1);
  const bool same_strides = a.stride == b.stride && a.stride == c.stride;
  bool same_kv_strides = b.stride.size() == c.stride.size();
  // Match Python true division, including unequal K/V head dimensions.
  if (same_kv_strides) {
    for (size_t i = 0; i + 1 < b.stride.size(); ++i) {
      if (static_cast<double>(b.stride[i]) / b.shape.back() !=
          static_cast<double>(c.stride[i]) / c.shape.back()) {
        same_kv_strides = false;
        break;
      }
    }
  }
  const int64_t qhd = a.shape[a.shape.size() - 2] * a.shape.back();
  const int64_t khd = b.shape[b.shape.size() - 2] * b.shape.back();
  const int64_t base = shared_qkv ? a.elements : 0;
  const bool offsets_3hd = a.offset == 0 && b.offset == qhd && c.offset == 2 * qhd;
  const bool offsets_h3d =
      a.offset == 0 && b.offset == a.shape.back() && c.offset == 2 * a.shape.back();
  const bool offsets_2hd = b.offset == base && c.offset == base + khd;
  const bool offsets_h2d = b.offset == base && c.offset == base + b.shape.back();
  const bool separate_qkv =
      shared_qkv ? a.offset == 0 && b.offset == a.elements && c.offset == a.elements + b.elements
                 : a.offset == 0 && b.offset == 0 && c.offset == 0;
  const bool separate_qk = !shared_qkv && shared_qk ? a.offset == 0 && b.offset == a.elements
                                                    : a.offset == 0 && b.offset == 0;
  const bool separate_kv = !shared_qkv && shared_kv ? b.offset == 0 && c.offset == b.elements
                                                    : b.offset == 0 && c.offset == 0;
  if (shared_qkv && same_strides && same_shapes && offsets_3hd)
    return fmt.substr(0, fmt.size() - 2) + "3" + fmt.substr(fmt.size() - 2);
  if (shared_qkv && same_strides && same_shapes && offsets_h3d)
    return fmt.substr(0, fmt.size() - 1) + "3" + fmt.substr(fmt.size() - 1);
  if (shared_kv && same_kv_strides && same_kv_shapes && offsets_2hd)
    return fmt + "_" + fmt.substr(0, fmt.size() - 2) + "2" + fmt.substr(fmt.size() - 2);
  if (shared_kv && same_kv_strides && same_kv_shapes && offsets_h2d)
    return fmt + "_" + fmt.substr(0, fmt.size() - 1) + "2" + fmt.substr(fmt.size() - 1);
  if (same_kv_strides && same_kv_shapes && (separate_qkv || separate_qk || separate_kv))
    return qfmt + "_" + kvfmt + "_" + kvfmt;
  return "not_supported";
}
}  // namespace

py::object get_qkv_layout(at::Tensor q, at::Tensor k, at::Tensor v, const std::string& fmt) {
  // Functional/Batched/GradTracking wrappers can have exact Python Tensor type.
  // Reject them before reading storage, including manually wrapped tensors.
  for (const auto* tensor : {&q, &k, &v}) {
    if (at::isTensorSubclassLike(*tensor)) return py::none();
  }
  // Validate formats/ranks before any indexing. Exceptional inputs retain Python
  // error behavior; returning None does not imply a newly supported layout.
  const auto split = fmt.find("_2");
  const std::string qfmt = split == std::string::npos ? fmt : fmt.substr(0, split);
  const std::string kvfmt = split == std::string::npos ? fmt : fmt.substr(split + 2);
  const auto valid_format = [](const std::string& format) {
    return format == "bshd" || format == "sbhd" || format == "thd";
  };
  if (!valid_format(qfmt) || !valid_format(kvfmt) || q.dim() != static_cast<int64_t>(qfmt.size()) ||
      k.dim() != static_cast<int64_t>(kvfmt.size()) || k.dim() != v.dim()) {
    return py::none();
  }
  // PyTorch normally bounds numel, but zero-size tensors can have very large
  // dimensions. Delegate arithmetic that would overflow signed metadata.
  const auto limit = std::numeric_limits<int64_t>::max();
  for (const auto* t : {&q, &k, &v}) {
    if (!t->device().is_cpu() && !t->device().is_cuda()) return py::none();
    if (t->size(-1) > 0 && t->size(-2) > limit / t->size(-1) / 2) return py::none();
    if (t->numel() > limit / 2) return py::none();
  }
  // Delegate exceptional/non-dense inputs to the untouched Python implementation.
  for (const auto* t : {&q, &k, &v}) {
    if (t->is_meta() || t->is_conj() || t->is_neg() || t->layout() != at::kStrided ||
        t->dim() < 2 || t->size(-1) == 0)
      return py::none();
    // Python integer true division rounds the exact ratio. Converting large
    // integers to double first can change normalized K/V stride equality.
    // Empty tensors can legitimately carry such strides without large storage.
    constexpr int64_t kMaxExactDoubleInteger = int64_t{1} << 53;
    if (t->size(-1) > kMaxExactDoubleInteger) return py::none();
    for (const auto stride : t->strides()) {
      if (stride > kMaxExactDoubleInteger || stride < -kMaxExactDoubleInteger) return py::none();
    }
  }
  for (const auto* t : {&q, &k, &v}) {
    if (t->stride(-1) != 1) {
      PyErr_SetString(PyExc_AssertionError,
                      "q, k and v must have stride 1 in their last dimension!");
      throw py::error_already_set();
    }
  }

  auto layout = classify(q, k, v, fmt, qfmt, kvfmt);
  if (layout == "not_supported") {
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    layout = classify(q, k, v, fmt, qfmt, kvfmt);
  }
  if (layout == "not_supported")
    throw std::runtime_error("The provided qkv memory layout is not supported!");
  return py::make_tuple(layout, q, k, v, qfmt, kvfmt);
}
}  // namespace transformer_engine::pytorch
