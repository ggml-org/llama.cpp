//******************************************************************************
// GLU F32 Kernel (SwiGLU specifically)
// Gated Linear Unit: y[i] = silu(x[i]) * g[i] where silu(x) = x * sigmoid(x)
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"

// GLU kernel parameters structure (from ET backend ops)
struct ggml_et_glu_params {
    struct ggml_tensor src0;     // F32 input tensor A (or combined tensor if src1 is null)
    struct ggml_tensor src1;     // F32 input tensor B (null for single tensor mode)
    struct ggml_tensor dst;      // F32 output tensor (n/2 columns)
    int32_t glu_op_type;         // GLU operation type (REGLU=0, GEGLU=1, SWIGLU=2, etc.)
    int32_t swapped;             // Whether gate and value are swapped
};

// GLU operation types (from ggml.h)
enum ggml_glu_op {
    GGML_GLU_OP_REGLU = 0,
    GGML_GLU_OP_GEGLU = 1,
    GGML_GLU_OP_SWIGLU = 2,
    GGML_GLU_OP_GEGLU_ERF = 3,
    GGML_GLU_OP_GEGLU_QUICK = 4
};

// SiLU activation function: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
static inline float silu_f32(float x) {
    // For numerical stability, use the mathematically equivalent form:
    // silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
    // For large negative x, exp(-x) -> inf, so silu(x) -> 0
    // For large positive x, exp(-x) -> 0, so silu(x) -> x

    if (x > 20.0f) {
        // For x > 20, exp(-x) is negligible, silu(x) ~ x
        return x;
    } else if (x < -20.0f) {
        // For x < -20, silu(x) ~ 0
        return 0.0f;
    } else {
        // Use standard formula: silu(x) = x / (1 + exp(-x))
        // Optimized using ET hardware division
        float exp_neg_x = et_expf(-x);
        float denominator = 1.0f + exp_neg_x;
        return et_fdiv(x, denominator);
    }
}

// Vectorized SwiGLU block processing (16 elements = 1 cache line)
static inline void block_swiglu(float* dst_block, const float* x_block, const float* g_block, int elements) {
    // Process 8 elements at a time using vector instructions
    int32_t vec_end = (elements / 8) * 8;

    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Constants for broadcasting
    float zero_const = 0.0f;
    float one_const = 1.0f;
    float log2e_const = 1.4426950408889634f;  // log2(e)

    for (int32_t i = 0; i < vec_end; i += 8) {
        // Vectorized SwiGLU: dst = silu(x) * g = (x / (1 + exp(-x))) * g
        // Using ET hardware: exp, reciprocal, multiply operations
        __asm__ volatile(
            // Load input vectors
            "flw.ps f10, %[x_vec]\n"            // f10 = x[0..7]
            "flw.ps f11, %[g_vec]\n"            // f11 = g[0..7]

            // Broadcast constants to vector registers
            "fbc.ps f20, %[zero_ptr]\n"         // f20 = broadcast(0.0f) to all 8 elements
            "fbc.ps f21, %[one_ptr]\n"          // f21 = broadcast(1.0f) to all 8 elements

            // Compute -x (negate x by subtracting from zero)
            "fsub.ps f12, f20, f10\n"           // f12 = 0 - x = -x

            // Convert to base-2 exponent: -x * log2(e) = -x * 1.44269504
            // Load log2(e) constant
            "fbc.ps f22, %[log2e_ptr]\n"        // f22 = broadcast(1.44269504f)
            "fmul.ps f13, f12, f22\n"           // f13 = -x * log2(e)

            // Compute 2^(-x * log2(e)) = exp(-x)
            "fexp.ps f14, f13\n"                // f14 = 2^(-x * log2(e)) = exp(-x)

            // Compute 1 + exp(-x)
            "fadd.ps f15, f14, f21\n"           // f15 = exp(-x) + 1

            // Compute 1 / (1 + exp(-x)) using reciprocal
            "frcp.ps f16, f15\n"                // f16 = 1 / (1 + exp(-x))

            // Compute silu(x) = x * (1 / (1 + exp(-x)))
            "fmul.ps f17, f10, f16\n"           // f17 = x * (1 / (1 + exp(-x))) = silu(x)

            // Compute final result: silu(x) * g
            "fmul.ps f18, f17, f11\n"           // f18 = silu(x) * g

            // Store result
            "fsw.ps f18, %[dst_out]\n"          // Store 8 results to destination

            : [dst_out] "=m"(*(float(*)[8])&dst_block[i])
            : [x_vec] "m"(*(const float(*)[8])&x_block[i]),
              [g_vec] "m"(*(const float(*)[8])&g_block[i]),
              [zero_ptr] "m"(zero_const), // Memory reference to 0.0f for broadcasting
              [one_ptr] "m"(one_const),   // Memory reference to 1.0f for broadcasting
              [log2e_ptr] "m"(log2e_const) // Memory reference to log2(e) for broadcasting
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f20", "f21", "f22"
        );
    }

    // Restore original mask
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

    // Handle remaining elements (< 8) with scalar operations
    for (int32_t i = vec_end; i < elements; i++) {
        dst_block[i] = silu_f32(x_block[i]) * g_block[i];
    }
}

// Main entry point for GLU kernel
int entry_point(struct ggml_et_glu_params* params, void* env) {
    // Cast env to proper type
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    // Validate environment pointer
    if (!kernel_env) {
        return -1;
    }

    // Get thread info using shire mask from environment
    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    // Return early if this hart is not active
    if (thread_id < 0) {
        return 0;
    }

    // TEMPORARY: Make kernel single-threaded to avoid race conditions
    // Only thread 0 should do the work, all others return immediately
    if (thread_id != 0) {
        return 0;
    }

    // Basic safety check on params
    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1; // Invalid pointer
    }

    // Only support SwiGLU for now
    if (params->glu_op_type != GGML_GLU_OP_SWIGLU) {
        return -1; // Unsupported GLU operation
    }

    // Extract tensor references
    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* src1 = &params->src1;  // May be null for single tensor mode
    struct ggml_tensor* dst = &params->dst;
    int32_t swapped = params->swapped;

    // Validate tensor types (F32 only)
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1; // Unsupported type combination
    }

    if (src1 && src1->type != GGML_TYPE_F32) {
        return -1; // Unsupported src1 type
    }

    // Get data pointers
    float* src0_data = (float*)src0->data;
    float* src1_data = src1 ? (float*)src1->data : src0_data;  // Same as src0 for single tensor mode
    float* dst_data = (float*)dst->data;

    // Validate data pointers
    if (!src0_data || !dst_data) {
        return -1; // Null data pointer
    }

    // Get tensor dimensions
    const int64_t nc = dst->ne[0];  // Output columns (input columns / 2)
    const int64_t nr = dst->ne[1] * dst->ne[2] * dst->ne[3];  // Total rows

    // Get strides
    const size_t src0_stride = src0->nb[1];  // Stride between rows in src0
    const size_t src1_stride = src1 ? src1->nb[1] : src0->nb[1];  // Stride between rows in src1
    const size_t dst_stride = dst->nb[1];    // Stride between rows in dst

    // Validate dimensions for split SwiGLU
    if (src1) {
        // Split tensor mode: src0 and src1 should have same shape as dst
        if (src0->ne[0] != nc || src1->ne[0] != nc) {
            return -1; // Dimension mismatch in split mode
        }
    } else {
        // Single tensor mode: src0 should have 2*nc columns
        if (src0->ne[0] != 2 * nc) {
            return -1; // Dimension mismatch in single tensor mode
        }
    }

    // Calculate total elements for cache line distribution
    const int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
    const int64_t total_elements = nr * nc;
    const int64_t total_cachelines = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    // TEMPORARY: Comment out multi-threaded work distribution
    // // Distribute cache lines across threads
    // int64_t cachelines_per_thread = total_cachelines / num_threads;
    // if (cachelines_per_thread == 0) cachelines_per_thread = 1;
    //
    // int64_t start_cacheline = thread_id * cachelines_per_thread;
    // int64_t end_cacheline = start_cacheline + cachelines_per_thread;
    //
    // // Clamp end_cacheline to actual number of cache lines
    // if (end_cacheline > total_cachelines) {
    //     end_cacheline = total_cachelines;
    // }
    //
    // // Thread should return if no work to do
    // if (start_cacheline >= total_cachelines) {
    //     return 0;
    // }

    // TEMPORARY: Single-threaded - thread 0 does all the work
    int64_t start_cacheline = 0;
    int64_t end_cacheline = total_cachelines;

    // Process cache lines assigned to this thread
    for (int64_t cl = start_cacheline; cl < end_cacheline; cl++) {
        // Map cache line back to element coordinates
        int64_t global_element_start = cl * elements_per_cacheline;
        int64_t row = global_element_start / nc;
        int64_t col = global_element_start % nc;

        // Skip if we're past the end of data
        if (global_element_start >= total_elements) break;

        // Calculate how many elements to process in this cache line
        int64_t elements_remaining = total_elements - global_element_start;
        int elements_this_block = (int)((elements_remaining < elements_per_cacheline) ?
                                       elements_remaining : elements_per_cacheline);

        // Process elements that span across rows
        int64_t elements_processed = 0;
        while (elements_processed < elements_this_block && row < nr) {
            // Calculate elements to process in current row
            int64_t elements_in_row = nc - col;
            int64_t elements_to_process = elements_this_block - elements_processed;
            if (elements_to_process > elements_in_row) {
                elements_to_process = elements_in_row;
            }

            // Get pointers for current row and column range
            float* dst_ptr = (float*)((char*)dst_data + row * dst_stride) + col;

            float* x_ptr;
            float* g_ptr;

            if (src1) {
                // Split tensor mode
                x_ptr = (float*)((char*)src0_data + row * src0_stride) + col;
                g_ptr = (float*)((char*)src1_data + row * src1_stride) + col;
            } else {
                // Single tensor mode - src0 contains both x and g
                float* src0_row = (float*)((char*)src0_data + row * src0_stride);
                if (swapped) {
                    g_ptr = src0_row + col;                // First half is gate
                    x_ptr = src0_row + nc + col;           // Second half is value
                } else {
                    x_ptr = src0_row + col;                // First half is value
                    g_ptr = src0_row + nc + col;           // Second half is gate
                }
            }

            // Process this segment
            block_swiglu(dst_ptr, x_ptr, g_ptr, (int)elements_to_process);

            // Update counters
            elements_processed += elements_to_process;
            col += elements_to_process;

            // Move to next row if current row is complete
            if (col >= nc) {
                row++;
                col = 0;
            }
        }
    }

    return 0;
}
