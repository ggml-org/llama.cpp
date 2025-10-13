//******************************************************************************
// Bare Metal CONT F16 Kernel
// Converts non-contiguous F16 tensors to contiguous memory layout
//
// Algorithm:
// 1. Iterate over destination tensor in contiguous order (sequential writes)
// 2. For each destination element, calculate corresponding source index
// 3. Read from source at scattered locations using stride information
// 4. Write to destination sequentially for optimal memory throughput
//
// The operation transforms a tensor with arbitrary strides (non-contiguous)
// into a tensor with standard C-order strides (contiguous).
//
// Memory Access Pattern:
// - Sequential writes to destination (optimal for memory controllers)
// - Scattered reads from source (uses stride-based addressing)
// - Prioritizes write performance over read performance
//
// Multi-threading:
// - Work is split across threads with cacheline-aligned boundaries
// - Each thread processes whole cachelines (64 bytes = 32 F16 elements)
// - No write contention between threads (separate cachelines)
// - Read contention is acceptable (memory throughput bounded)
//
// Operation: dst[linear_idx] = src[multi_dim_idx] where indices are
// converted using stride information from tensor metadata.
//
// Note: F16 is represented as uint16_t (IEEE 754 binary16 format)
//******************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include "ggml_tensor.h"
#include "platform.h"

struct ggml_et_cont_params {
    struct ggml_tensor src0;     // F16 input tensor (non-contiguous)
    struct ggml_tensor dst;      // F16 output tensor (contiguous)
};

KERNEL_TRAMPOLINE();

static void linear_to_indices(int64_t linear_idx,
                             const int64_t* ne,
                             int64_t* indices) {
    // Convert linear index to 4D indices [i0, i1, i2, i3]
    // Assumes C-order (row-major) layout in destination
    indices[3] = linear_idx / (ne[2] * ne[1] * ne[0]);
    int64_t remainder = linear_idx % (ne[2] * ne[1] * ne[0]);

    indices[2] = remainder / (ne[1] * ne[0]);
    remainder = remainder % (ne[1] * ne[0]);

    indices[1] = remainder / ne[0];
    indices[0] = remainder % ne[0];
}

static int64_t calculate_src_offset(const int64_t* indices,
                                   const int64_t* nb) {
    return indices[0] * nb[0] +
           indices[1] * nb[1] +
           indices[2] * nb[2] +
           indices[3] * nb[3];
}

int entry_point(struct ggml_et_cont_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return 0;
    }

    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1; // Invalid pointer
    }

    struct ggml_tensor* src0 = &params->src0;  // Non-contiguous input
    struct ggml_tensor* dst = &params->dst;    // Contiguous output

    if (src0->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F16) {
        return -1; // Unsupported type combination
    }

    uint16_t* src0_data = (uint16_t*)src0->data;
    uint16_t* dst_data = (uint16_t*)dst->data;

    if (!src0_data || !dst_data) {
        return -1; // Null data pointer
    }

    const int64_t src_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    const int64_t dst_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    if (src_elements != dst_elements) {
        return -1; // Element count mismatch
    }

    const int64_t total_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    const int64_t src_ne[4] = {src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]};
    const int64_t src_nb[4] = {src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]};

    const int64_t CACHELINE_SIZE = 64;
    const int64_t ELEMENTS_PER_CACHELINE = CACHELINE_SIZE / sizeof(uint16_t);  // 32 elements

    const int64_t total_cachelines = (total_elements + ELEMENTS_PER_CACHELINE - 1) / ELEMENTS_PER_CACHELINE;

    const int64_t cachelines_per_thread = (total_cachelines + num_threads - 1) / num_threads;

    const int64_t start_element = thread_id * cachelines_per_thread * ELEMENTS_PER_CACHELINE;
    const int64_t end_element = (start_element + cachelines_per_thread * ELEMENTS_PER_CACHELINE) < total_elements
                                ? (start_element + cachelines_per_thread * ELEMENTS_PER_CACHELINE)
                                : total_elements;

    if (start_element >= total_elements) {
        return 0;
    }

    for (int64_t dst_linear_idx = start_element; dst_linear_idx < end_element; dst_linear_idx++) {
        int64_t indices[4];
        linear_to_indices(dst_linear_idx, src_ne, indices);

        const int64_t src_offset_bytes = calculate_src_offset(indices, src_nb);

        const uint16_t* src_ptr = (const uint16_t*)((char*)src0_data + src_offset_bytes);

        dst_data[dst_linear_idx] = *src_ptr;
    }

    return 0;
}
