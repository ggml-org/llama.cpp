#include "ggml-et-ops.h"
#include "ggml-et-kernels.h"
#include "ggml-et-cpu-compare.h"
#include "ggml-impl.h"

// CPU comparison configuration - can be enabled for debugging
static ggml_et_cpu_compare_config rope_cpu_compare_config = {
    /* .enabled = */ false,
    /* .use_cpu_result = */ false,    // Replace ET result with CPU result
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

bool ggml_et_op_mul(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node) {
    if (!dev_ctx || !node) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL operation\n");
        return false;
    }

    if (!node->src[0] || !node->src[1]) {
        GGML_LOG_ERROR("ET: MUL operation missing required inputs\n");
        return false;
    }

    const char* kernel_name;

    if (node->type == GGML_TYPE_F32 &&
        node->src[0]->type == GGML_TYPE_F32 &&
        node->src[1]->type == GGML_TYPE_F32) {

        kernel_name = "mul_f32";

    } else {
        GGML_LOG_ERROR("ET: MUL operation with unsupported types\n");
        return false;
    }

    // Pack parameters - copy full tensor structures
    ggml_et_binary_params params;
    params.src0 = *node->src[0];
    params.src1 = *node->src[1];
    params.dst = *node;

    GGML_LOG_DEBUG("ET: Launching MUL kernel %s\n", kernel_name);

    return ggml_et_launch_kernel(dev_ctx, kernel_name, &params, sizeof(params), 0xFFFFFFFF);
}

bool ggml_et_op_mul_mat(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node) {
    if (!dev_ctx || !node) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL_MAT operation\n");
        return false;
    }

    if (!node->src[0] || !node->src[1]) {
        GGML_LOG_ERROR("ET: MUL_MAT operation missing required inputs\n");
        return false;
    }

    const char* kernel_name;

    if (node->type == GGML_TYPE_F32 &&
        node->src[0]->type == GGML_TYPE_Q8_0 &&
        node->src[1]->type == GGML_TYPE_F32) {

        kernel_name = "mul_mat_q8_0_f32";

        GGML_LOG_DEBUG("ET: MUL_MAT Q8_0xF32->F32 kernel selected for shapes src0=[%lld,%lld,%lld,%lld] src1=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld]\n",
                       (long long)node->src[0]->ne[0], (long long)node->src[0]->ne[1], (long long)node->src[0]->ne[2], (long long)node->src[0]->ne[3],
                       (long long)node->src[1]->ne[0], (long long)node->src[1]->ne[1], (long long)node->src[1]->ne[2], (long long)node->src[1]->ne[3],
                       (long long)node->ne[0], (long long)node->ne[1], (long long)node->ne[2], (long long)node->ne[3]);

    } else {
        GGML_LOG_ERROR("ET: MUL_MAT operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(node->type),
                       ggml_type_name(node->src[0]->type),
                       ggml_type_name(node->src[1]->type));
        return false;
    }

    // Pack parameters - copy full tensor structures
    ggml_et_binary_params params;
    params.src0 = *node->src[0];  // weight matrix
    params.src1 = *node->src[1];  // activation matrix
    params.dst = *node;           // output matrix

    GGML_LOG_DEBUG("ET: Launching MUL_MAT kernel %s (Q8_0[%lld,%lld] x F32[%lld,%lld] -> F32[%lld,%lld])\n",
                   kernel_name,
                   (long long)node->src[0]->ne[0], (long long)node->src[0]->ne[1],
                   (long long)node->src[1]->ne[0], (long long)node->src[1]->ne[1],
                   (long long)node->ne[0], (long long)node->ne[1]);

    return ggml_et_launch_kernel(dev_ctx, kernel_name, &params, sizeof(params), 0xFFFFFFFF);
}

bool ggml_et_op_rope(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node) {
    if (!dev_ctx || !node) {
        GGML_LOG_ERROR("ET: Invalid parameters for ROPE operation\n");
        return false;
    }

    if (!node->src[0] || !node->src[1]) {
        GGML_LOG_ERROR("ET: ROPE operation missing required inputs\n");
        return false;
    }

    const char* kernel_name;

    if (node->type == GGML_TYPE_F32 &&
        node->src[0]->type == GGML_TYPE_F32 &&
        node->src[1]->type == GGML_TYPE_I32) {
        kernel_name = "rope_f32";
    } else {
        GGML_LOG_ERROR("ET: ROPE operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(node->type),
                       ggml_type_name(node->src[0]->type),
                       ggml_type_name(node->src[1]->type));
        return false;
    }

    GGML_LOG_DEBUG("ET: ROPE F32xI32->F32 kernel selected for shapes src0=[%lld,%lld,%lld,%lld] src1=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld] inplace=%s\n",
                   (long long)node->src[0]->ne[0], (long long)node->src[0]->ne[1], (long long)node->src[0]->ne[2], (long long)node->src[0]->ne[3],
                   (long long)node->src[1]->ne[0], (long long)node->src[1]->ne[1], (long long)node->src[1]->ne[2], (long long)node->src[1]->ne[3],
                   (long long)node->ne[0], (long long)node->ne[1], (long long)node->ne[2], (long long)node->ne[3],
                   (node->data == node->src[0]->data) ? "yes" : "no");

    GGML_LOG_DEBUG("ET: ROPE tensor strides - src0.nb=[%zu,%zu,%zu,%zu] src1.nb=[%zu,%zu,%zu,%zu] dst.nb=[%zu,%zu,%zu,%zu]\n",
                   node->src[0]->nb[0], node->src[0]->nb[1], node->src[0]->nb[2], node->src[0]->nb[3],
                   node->src[1]->nb[0], node->src[1]->nb[1], node->src[1]->nb[2], node->src[1]->nb[3],
                   node->nb[0], node->nb[1], node->nb[2], node->nb[3]);

    // Pack parameters - copy full tensor structures and op_params
    ggml_et_rope_params params;
    params.src0 = *node->src[0];                    // F32 input tensor
    params.src1 = *node->src[1];                    // I32 position tensor
    if (node->src[2]) {
        params.src2 = *node->src[2];                // F32 frequency factors (optional)
    } else {
        memset(&params.src2, 0, sizeof(params.src2)); // Zero if not provided
    }
    params.dst = *node;                             // F32 output tensor

    params.rope_params.n_past     = ((const int32_t *) node->op_params)[0];
    params.rope_params.n_dims     = ((const int32_t *) node->op_params)[1];
    params.rope_params.mode       = ((const int32_t *) node->op_params)[2];
    params.rope_params.n_ctx      = ((const int32_t *) node->op_params)[3];
    params.rope_params.n_ctx_orig = ((const int32_t *) node->op_params)[4];
    memcpy(&params.rope_params.freq_base,   (const int32_t *) node->op_params +  5, sizeof(float));
    memcpy(&params.rope_params.freq_scale,  (const int32_t *) node->op_params +  6, sizeof(float));
    memcpy(&params.rope_params.ext_factor,  (const int32_t *) node->op_params +  7, sizeof(float));
    memcpy(&params.rope_params.attn_factor, (const int32_t *) node->op_params +  8, sizeof(float));
    memcpy(&params.rope_params.beta_fast,   (const int32_t *) node->op_params +  9, sizeof(float));
    memcpy(&params.rope_params.beta_slow,   (const int32_t *) node->op_params + 10, sizeof(float));
    if (params.rope_params.mode & GGML_ROPE_TYPE_MROPE) {
        memcpy(params.rope_params.sections, (const int32_t *) node->op_params + 11, sizeof(int32_t)*4);
    } else {
        memset(params.rope_params.sections, 0, sizeof(params.rope_params.sections));
    }

    GGML_LOG_DEBUG("ET: ROPE params - n_past=%d n_dims=%d mode=0x%x n_ctx=%d n_ctx_orig=%d freq_base=%.6f freq_scale=%.6f ext_factor=%.6f attn_factor=%.6f beta_fast=%.6f beta_slow=%.6f\n",
                  params.rope_params.n_past, params.rope_params.n_dims, params.rope_params.mode,
                  params.rope_params.n_ctx, params.rope_params.n_ctx_orig,
                  params.rope_params.freq_base, params.rope_params.freq_scale, params.rope_params.ext_factor,
                  params.rope_params.attn_factor, params.rope_params.beta_fast, params.rope_params.beta_slow);

    if (params.rope_params.mode & GGML_ROPE_TYPE_MROPE) {
        GGML_LOG_DEBUG("ET: ROPE MROPE sections=[%d,%d,%d,%d]\n",
                      params.rope_params.sections[0], params.rope_params.sections[1],
                      params.rope_params.sections[2], params.rope_params.sections[3]);
    }

    GGML_LOG_DEBUG("ET: ROPE mode flags - NEOX=%s MROPE=%s VISION=%s\n",
                  (params.rope_params.mode & GGML_ROPE_TYPE_NEOX) ? "yes" : "no",
                  (params.rope_params.mode & GGML_ROPE_TYPE_MROPE) ? "yes" : "no",
                  (params.rope_params.mode & GGML_ROPE_TYPE_VISION) ? "yes" : "no");

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    bool cpu_comparison_active = false;
    if (rope_cpu_compare_config.enabled) {
        GGML_LOG_DEBUG("ET: Initializing CPU comparison for ROPE operation\n");
        if (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, node, GGML_OP_ROPE)) {
            cpu_comparison_active = true;
        } else {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for ROPE operation\n");
        }
    }

    GGML_LOG_DEBUG("ET: Launching ROPE kernel %s\n", kernel_name);

    bool kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, sizeof(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    if (cpu_comparison_active) {
        GGML_LOG_DEBUG("ET: Performing CPU computation and comparison for ROPE operation\n");
        if (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, node, &rope_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for ROPE operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    return kernel_result;
}
