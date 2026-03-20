/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace test {

// Returns the random seed for test shape generation.
// Uses NVTE_TEST_SEED env var if set, otherwise generates a random seed.
inline uint64_t getTestSeed() {
    static uint64_t seed = []() {
        uint64_t s;
        const char* env = std::getenv("NVTE_TEST_SEED");
        if (env && env[0] != '\0') {
            s = std::stoull(env);
        } else {
            s = std::random_device{}();
        }
        return s;
    }();
    return seed;
}

// Converts a shape vector to a human-readable string, e.g. "{128, 256}".
inline std::string shapeToString(const std::vector<size_t>& shape) {
    std::string s = "{";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) s += ", ";
        s += std::to_string(shape[i]);
    }
    s += "}";
    return s;
}

namespace detail {

// A "nice" dimension: multiple of 128, in [128, 65536].
inline size_t niceDim(std::mt19937_64& rng) {
    std::uniform_int_distribution<size_t> d(1, 512);
    return d(rng) * 128;
}

// A fully random dimension drawn from a mix of magnitude buckets.
inline size_t randomDim(std::mt19937_64& rng) {
    std::uniform_int_distribution<int> cat(0, 3);
    switch (cat(rng)) {
        case 0: return std::uniform_int_distribution<size_t>(2, 16)(rng);
        case 1: return std::uniform_int_distribution<size_t>(17, 256)(rng);
        case 2: return std::uniform_int_distribution<size_t>(257, 4096)(rng);
        case 3: return std::uniform_int_distribution<size_t>(4097, 65536)(rng);
        default: return 128;
    }
}

// An edge-case dimension: 0 or 1.
inline size_t edgeDim(std::mt19937_64& rng) {
    return std::uniform_int_distribution<size_t>(0, 1)(rng);
}

// Caps a dimension so that total * dim <= max_total.
// If total is already 0 (from a prior 0-dimension), no cap is needed.
inline size_t capDim(size_t dim, size_t total, size_t max_total) {
    if (total == 0) return dim;
    if (dim > max_total / total) {
        return std::max<size_t>(1, max_total / total);
    }
    return dim;
}

// Shape generation strategies.
//   kAllNice    — all dimensions are multiples of 128
//   kMixedNice  — one dimension is a nice multiple, rest are random
//   kRandom     — all dimensions are unconstrained random values
//   kEdgeCase   — at least one dimension is 0 or 1, rest random
enum Strategy { kAllNice = 0, kMixedNice = 1, kRandom = 2, kEdgeCase = 3, kNumStrategies = 4 };

inline std::vector<size_t> generateShape(std::mt19937_64& rng,
                                          Strategy strategy,
                                          size_t ndim,
                                          size_t max_total) {
    std::vector<size_t> shape(ndim);
    size_t total = 1;

    // For kMixedNice: pick which dimension is nice
    size_t nice_pos = std::uniform_int_distribution<size_t>(0, ndim - 1)(rng);
    // For kEdgeCase: pick which dimension is edge
    size_t edge_pos = std::uniform_int_distribution<size_t>(0, ndim - 1)(rng);

    for (size_t d = 0; d < ndim; ++d) {
        size_t dim;
        switch (strategy) {
            case kAllNice:
                dim = niceDim(rng);
                // Keep nice alignment when capping
                dim = capDim(dim, total, max_total);
                if (dim > 0 && dim < 128) dim = 128;
                if (dim >= 128) dim = (dim / 128) * 128;
                break;
            case kMixedNice:
                dim = (d == nice_pos) ? niceDim(rng) : randomDim(rng);
                dim = capDim(dim, total, max_total);
                break;
            case kRandom:
                dim = randomDim(rng);
                dim = capDim(dim, total, max_total);
                break;
            case kEdgeCase:
                dim = (d == edge_pos) ? edgeDim(rng) : randomDim(rng);
                dim = capDim(dim, total, max_total);
                break;
            default:
                dim = randomDim(rng);
                dim = capDim(dim, total, max_total);
                break;
        }
        shape[d] = dim;
        total *= dim;
    }
    return shape;
}

}  // namespace detail

// Generates random shapes at runtime.
//
// count              — number of shapes to generate
// min_dims           — minimum number of dimensions per shape
// max_dims           — maximum number of dimensions per shape
// max_total_elements — cap on total elements per shape to avoid OOM
//
// The random shapes follow a mix of strategies (~25% each):
//   "all nice"  — every dim is a multiple of 128
//   "mixed"     — one dim is a nice multiple of 128, rest random
//   "random"    — all dims are arbitrary
//   "edge"      — at least one dim is 0 or 1, rest random
inline std::vector<std::vector<size_t>> generateRandomShapes(
    size_t count,
    size_t min_dims = 1,
    size_t max_dims = 4,
    size_t max_total_elements = 16 * 1024 * 1024) {
    std::mt19937_64 rng(getTestSeed());

    std::uniform_int_distribution<int> strategy_dist(0, detail::kNumStrategies - 1);
    std::uniform_int_distribution<size_t> ndim_dist(min_dims, max_dims);

    std::vector<std::vector<size_t>> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto strategy = static_cast<detail::Strategy>(strategy_dist(rng));
        size_t ndim = ndim_dist(rng);
        result.push_back(detail::generateShape(rng, strategy, ndim, max_total_elements));
    }
    return result;
}

}  // namespace test

// Wraps a test body so that NVTE_CHECK assertion failures (i.e. the library
// rejecting an unsupported configuration) are treated as a graceful skip.
//
// Exceptions from CUDA/cuBLAS/cuDNN/NVRTC/NCCL runtime errors are re-thrown
// so they still fail the test — those indicate real bugs, not unsupported
// shapes.  Crashes (segfaults) and wrong numerical results also still fail.
//
// NOTE: Use this only outside loops. Inside a loop over shapes, use
// NVTE_TEST_ALLOW_EXCEPTION_IN_LOOP which logs and continues instead of
// skipping the entire test.
//
// Usage:
//   NVTE_TEST_ALLOW_EXCEPTION(performTest<T>(shape));
#define NVTE_TEST_ALLOW_EXCEPTION(...)                                          \
    do {                                                                        \
        try {                                                                   \
            __VA_ARGS__;                                                        \
        } catch (const std::exception& e) {                                     \
            const std::string msg_ = e.what();                                  \
            if (msg_.find("Assertion failed:") != std::string::npos) {          \
                GTEST_SKIP() << "Unsupported configuration: " << msg_;          \
            }                                                                   \
            throw;                                                              \
        }                                                                       \
    } while (0)

// Loop-friendly variant: logs unsupported configurations and continues
// to the next iteration instead of skipping the entire test.
//
// Usage (inside a for loop):
//   for (const auto& shape : shapes) {
//     NVTE_TEST_ALLOW_EXCEPTION_IN_LOOP(performTest<T>(shape));
//   }
#define NVTE_TEST_ALLOW_EXCEPTION_IN_LOOP(...)                                  \
    try {                                                                       \
        __VA_ARGS__;                                                            \
    } catch (const std::exception& e) {                                         \
        const std::string msg_ = e.what();                                      \
        if (msg_.find("Assertion failed:") != std::string::npos) {              \
            std::cerr << "  [SKIP] Unsupported configuration: " << msg_         \
                      << std::endl;                                             \
            continue;                                                           \
        }                                                                       \
        throw;                                                                  \
    }

// Adds a SCOPED_TRACE with the seed and shape for reproducibility.
// On failure, GTest output will show e.g.:
//   Google Test trace:
//   NVTE_TEST_SEED=8374291 shape={11, 0}
#define NVTE_TRACE_RANDOM_SHAPE(shape)                                         \
    SCOPED_TRACE("NVTE_TEST_SEED=" + std::to_string(test::getTestSeed()) +     \
                 " shape=" + test::shapeToString(shape))
