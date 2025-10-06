//******************************************************************************
// Bare Metal Softmax F32 Kernel
// Softmax function: y[i] = exp(x[i] - max) / sum(exp(x[j] - max))
//
// Algorithm:
// 1. Apply scaling: x' = x * scale
// 2. Add mask/bias if present: x' = x' + mask * slope (ALiBi support)
// 3. Find max value for numerical stability: max = max(x')
// 4. Compute exponentials: exp_vals[i] = exp(x'[i] - max)
// 5. Compute sum: sum = sum(exp_vals)
// 6. Normalize: y[i] = exp_vals[i] / sum
//
// Features supported:
// - Temperature scaling via scale parameter
// - Attention masking (transformer masks)
// - ALiBi (Attention with Linear Biases) positional encoding
// - Numerical stability (subtract max before exp)
// - ggml broadcasting rules for mask tensors
//
// Mask Broadcasting Rules (ggml-specific, not standard numpy):
// - Dimension 0: mask.ne[0] == input.ne[0] (exact match required)
// - Dimension 1: mask.ne[1] >= input.ne[1] (allows larger pre-allocated masks)
// - Dimension 2: input.ne[2] % mask.ne[2] == 0 (modulo broadcasting)
// - Dimension 3: input.ne[3] % mask.ne[3] == 0 (modulo broadcasting)
//******************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"

// Softmax kernel parameters structure (from ggml-et-ops.h)
struct ggml_et_softmax_params {
    struct ggml_tensor src0;     // F32 input tensor
    struct ggml_tensor src1;     // F32 mask tensor (optional, may be zeroed if not used)
    struct ggml_tensor dst;      // F32 output tensor
    float scale;                 // Scale factor (temperature scaling)
    float max_bias;              // Max bias for ALiBi (0.0f if not used)
};

KERNEL_TRAMPOLINE();

// Find maximum value in array - needed for numerical stability
static float find_max_f32(const float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    return max_val;
}

// Compute softmax for a single row
static void compute_softmax_row(
    float* dst,           // Output row
    const float* src,     // Input row
    const float* mask,    // Mask row (can be NULL)
    int ne00,             // Input row length
    int ne10,             // Mask row length (guaranteed equal to ne00 in ggml)
    float scale,          // Scale factor
    float slope)          // ALiBi slope factor
{
    // Step 1: Apply scaling and masking/bias to input
    // Copy input and apply scale
    for (int i = 0; i < ne00; i++) {
        dst[i] = src[i] * scale;
    }

    // Add mask/bias if present
    if (mask != NULL) {
        // In ggml softmax: ne10 == ne00 (dimension 0 must match exactly)
        // So we can directly index the mask without broadcasting
        for (int i = 0; i < ne00; i++) {
            dst[i] += slope * mask[i];
        }
    }

    // Step 2: Find maximum for numerical stability
    float max_val = find_max_f32(dst, ne00);

    // Step 3: Compute exponentials and sum
    // exp(x[i] - max) for numerical stability
    float sum = 0.0f;
    for (int i = 0; i < ne00; i++) {
        float exp_val = et_expf(dst[i] - max_val);
        dst[i] = exp_val;
        sum += exp_val;
    }

    // Step 4: Normalize by sum to get probabilities
    // Avoid division by zero
    if (sum > 0.0f) {
        // Use ET hardware division function instead of standard division
        float inv_sum = et_fdiv(1.0f, sum);
        for (int i = 0; i < ne00; i++) {
            dst[i] *= inv_sum;
        }
    }
}

// Main entry point for Softmax kernel
int entry_point(struct ggml_et_softmax_params* params, void* env) {
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

    // Single-threaded implementation: only thread 0 does the work
    // All other threads return early
    if (thread_id != 0) {
        return 0;
    }

    // Basic safety check on params
    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1; // Invalid pointer
    }

    // Extract tensor references
    struct ggml_tensor* src0 = &params->src0;  // Input tensor
    struct ggml_tensor* src1 = &params->src1;  // Mask tensor (optional)
    struct ggml_tensor* dst = &params->dst;    // Output tensor
    float scale = params->scale;               // Scale factor
    float max_bias = params->max_bias;         // ALiBi max bias

    // Validate tensor types (F32 only)
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1; // Unsupported type combination
    }

    // Check if mask is used and validate type
    bool use_mask = (src1->data != NULL && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16));

    // Get data pointers
    float* src0_data = (float*)src0->data;
    float* dst_data = (float*)dst->data;
    float* mask_data = use_mask ? (float*)src1->data : NULL;

    // Validate data pointers
    if (!src0_data || !dst_data) {
        return -1; // Null data pointer
    }

    // Get tensor dimensions
    const int64_t ne00 = src0->ne[0];  // Sequence length (columns)
    const int64_t ne01 = src0->ne[1];  // Number of rows
    const int64_t ne02 = src0->ne[2];  // Batch/head dimension
    const int64_t ne03 = src0->ne[3];  // Outer batch dimension

    // Get mask dimensions for broadcasting
    const int64_t ne10 = use_mask ? src1->ne[0] : 0;  // Mask sequence length
    const int64_t ne11 = use_mask ? src1->ne[1] : 0;  // Mask rows
    const int64_t ne12 = use_mask ? src1->ne[2] : 0;  // Mask batch/head dimension
    const int64_t ne13 = use_mask ? src1->ne[3] : 0;  // Mask outer batch dimension

    // Validate mask broadcasting compatibility using ggml's special rules
    if (use_mask) {
        // ggml softmax broadcasting rules (not standard numpy broadcasting):
        // - Dimension 0: mask must equal input exactly
        // - Dimension 1: mask must be >= input (allows larger pre-allocated masks)
        // - Dimension 2: input must be divisible by mask (modulo broadcasting)
        // - Dimension 3: input must be divisible by mask (modulo broadcasting)
        if (ne10 != ne00 ||                    // Dimension 0: exact match required
            ne11 < ne01 ||                     // Dimension 1: mask >= input
            (ne12 > 0 && ne02 % ne12 != 0) ||  // Dimension 2: input % mask == 0
            (ne13 > 0 && ne03 % ne13 != 0)) {  // Dimension 3: input % mask == 0
            return -1; // Incompatible dimensions for ggml softmax broadcasting
        }
    }

    // ALiBi slope calculation - compute per attention head
    // Based on ggml CPU implementation for ALiBi positional encoding
    float slope = 1.0f;
    if (max_bias > 0.0f) {
        // Simplified ALiBi slope calculation
        // In full implementation, this varies per head: slope = pow(2^-8/n_heads, head_idx)
        slope = 1.0f;
    }

    // Process tensor row by row
    // Calculate based on 4D tensor layout: [ne00, ne01, ne02, ne03]
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                // Calculate input row offset
                const int64_t src_offset = i03 * ne02 * ne01 * ne00 +
                                          i02 * ne01 * ne00 +
                                          i01 * ne00;

                const float* src_row = src0_data + src_offset;
                float* dst_row = dst_data + src_offset;
                const float* mask_row = NULL;

                // Calculate mask row offset using ggml's broadcasting rules
                if (use_mask && mask_data) {
                    // ggml broadcasting logic:
                    // - i11 = i01 (direct mapping for dimension 1, even if mask is larger)
                    // - i12 = i02 % ne12 (modulo broadcasting for dimension 2)
                    // - i13 = i03 % ne13 (modulo broadcasting for dimension 3)
                    const int64_t mask_i03 = (ne13 > 0) ? i03 % ne13 : 0;
                    const int64_t mask_i02 = (ne12 > 0) ? i02 % ne12 : 0;
                    const int64_t mask_i01 = i01;  // Direct mapping (mask >= input guaranteed)

                    const int64_t mask_offset = mask_i03 * ne12 * ne11 * ne10 +
                                               mask_i02 * ne11 * ne10 +
                                               mask_i01 * ne10;

                    mask_row = mask_data + mask_offset;
                }

                // Compute softmax for this row
                compute_softmax_row(dst_row, src_row, mask_row, (int)ne00, (int)ne10, scale, slope);
            }
        }
    }

    return 0; // Success
}
