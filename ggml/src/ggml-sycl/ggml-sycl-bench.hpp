//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstdint>
#include <vector>

#include <sycl/sycl.hpp>

#include "ggml.h"

namespace ggml_sycl {

// Arguments for benchmarking MMQ kernels directly (without ggml_tensor overhead)
struct mmq_bench_args {
    // Required: SYCL queue for kernel submission
    sycl::queue * stream = nullptr;

    // Quantization type of weights (e.g., GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    ggml_type weight_type = GGML_TYPE_Q4_0;

    // Layout mode for weights
    ggml_layout_mode layout = GGML_LAYOUT_AOS;

    // Pointers to data buffers (device memory)
    const void * weights     = nullptr;  // Quantized weight tensor
    const void * layout_base = nullptr;  // Base pointer for SoA/coalesced layouts (nullptr for AOS)
    const void * activations = nullptr;  // Activation tensor (f32 or quantized)
    float *      output      = nullptr;  // Output buffer

    // Matrix dimensions
    int64_t ncols = 0;  // Number of columns (K dimension)
    int64_t nrows = 0;  // Number of rows in weight matrix (output features)
    int64_t batch = 0;  // Batch size (number of input vectors)

    // Row range for partial computation
    int64_t row_low  = 0;  // Starting row (inclusive)
    int64_t row_high = 0;  // Ending row (exclusive)

    // Strides
    int64_t src1_padded_row_size = 0;  // Padded row size for activations
    int64_t dst_row_stride       = 0;  // Output row stride

    // Device ID (-1 for current device)
    int device_id = -1;
};

// Launch MMQ kernel for benchmarking
// Returns true on success, false on invalid arguments
// If events is non-null, kernel events are appended for timing
bool ggml_sycl_mmq_bench_launch(const mmq_bench_args & args, std::vector<sycl::event> * events);

}  // namespace ggml_sycl
