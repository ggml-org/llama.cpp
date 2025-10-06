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

static ggml_et_cpu_compare_config rms_norm_cpu_compare_config = {
    /* .enabled = */ false,
    /* .use_cpu_result = */ false,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

static ggml_et_cpu_compare_config elmap_cpu_compare_config = {
    /* .enabled = */ false,
    /* .use_cpu_result = */ false,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

static ggml_et_cpu_compare_config glu_cpu_compare_config = {
    /* .enabled = */ false,
    /* .use_cpu_result = */ false,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

static ggml_et_cpu_compare_config mul_mat_cpu_compare_config = {
    /* .enabled = */ false,
    /* .use_cpu_result = */ false,
    /* .log_differences = */ true,
    /* .tolerance = */ 0.01,
    /* .max_log_elements = */ 4096
 };

static ggml_et_cpu_compare_config softmax_cpu_compare_config = {
    /* .enabled = */ false,
    /* .use_cpu_result = */ false,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 1024
};

static ggml_et_cpu_compare_config get_rows_cpu_compare_config = {
    /* .enabled = */ false,
    /* .use_cpu_result = */ false,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 2048
};

static ggml_et_cpu_compare_config cont_cpu_compare_config = {
    /* .enabled = */ false,
    /* .use_cpu_result = */ false,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

static ggml_et_cpu_compare_config set_rows_cpu_compare_config = {
    /* .enabled = */ false,
    /* .use_cpu_result = */ false,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 2048
};

bool ggml_et_op_mul(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node) {
    // Delegate to generic element map operation
    return ggml_et_op_elmap(dev_ctx, node);
}

bool ggml_et_op_add(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node) {
    // Delegate to generic element map operation
    return ggml_et_op_elmap(dev_ctx, node);
}

bool ggml_et_op_elmap(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node) {
    ET_PERF_START();

    if (!dev_ctx || !node) {
        GGML_LOG_ERROR("ET: Invalid parameters for element map operation\n");
        return false;
    }

    if (!node->src[0] || !node->src[1]) {
        GGML_LOG_ERROR("ET: Element map operation missing required inputs\n");
        return false;
    }

    if (node->type != GGML_TYPE_F32 ||
        node->src[0]->type != GGML_TYPE_F32 ||
        node->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: Element map operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(node->type),
                       ggml_type_name(node->src[0]->type),
                       ggml_type_name(node->src[1]->type));
        return false;
    }

    const char* op_name = ggml_op_name(node->op);

    ggml_et_elmap_params params;
    params.src0 = *node->src[0];
    params.src1 = *node->src[1];
    params.dst = *node;           // F32 output tensor (op type stored in dst.op)

    GGML_LOG_DEBUG("ET: Launching el_map_f32 kernel for %s (F32[%lld,%lld,%lld,%lld] %s F32[%lld,%lld,%lld,%lld] -> F32[%lld,%lld,%lld,%lld])\n",
                   op_name,
                   (long long)node->src[0]->ne[0], (long long)node->src[0]->ne[1],
                   (long long)node->src[0]->ne[2], (long long)node->src[0]->ne[3],
                   op_name,
                   (long long)node->src[1]->ne[0], (long long)node->src[1]->ne[1],
                   (long long)node->src[1]->ne[2], (long long)node->src[1]->ne[3],
                   (long long)node->ne[0], (long long)node->ne[1],
                   (long long)node->ne[2], (long long)node->ne[3]);

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    bool cpu_comparison_active = false;
    if (elmap_cpu_compare_config.enabled) {
        GGML_LOG_DEBUG("ET: Initializing CPU comparison for %s operation\n", op_name);
        if (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, node, node->op)) {
            cpu_comparison_active = true;
        } else {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for %s operation\n", op_name);
        }
    }

    bool kernel_result = ggml_et_launch_kernel(dev_ctx, "el_map_f32", &params, sizeof(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    if (cpu_comparison_active) {
        GGML_LOG_DEBUG("ET: Performing CPU computation and comparison for %s operation\n", op_name);
        if (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, node, &elmap_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for %s operation\n", op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END(op_name, "el_map_f32", node);
    return kernel_result;
}

bool ggml_et_op_glu(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node) {
    ET_PERF_START();

    // Validate inputs
    if (!dev_ctx || !node) {
        GGML_LOG_ERROR("ET: Invalid parameters for GLU operation\n");
        return false;
    }

    // Validate split tensor mode (only mode we support)
    if (!node->src[0] || !node->src[1]) {
        GGML_LOG_ERROR("ET: GLU operation missing required inputs (split mode only)\n");
        return false;
    }

    // Only support F32 (as validated by supports_op)
    if (node->type != GGML_TYPE_F32 ||
        node->src[0]->type != GGML_TYPE_F32 ||
        node->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: GLU operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(node->type),
                       ggml_type_name(node->src[0]->type),
                       ggml_type_name(node->src[1]->type));
        return false;
    }

    // Extract GLU operation parameters from op_params
    int32_t glu_op_type = ggml_get_op_params_i32(node, 0);  // GLU variant (REGLU, GEGLU, SWIGLU, etc.)
    int32_t swapped = ggml_get_op_params_i32(node, 1);      // Whether gate/value are swapped

    // Only support SWIGLU for now
    if (glu_op_type != GGML_GLU_OP_SWIGLU) {
        GGML_LOG_ERROR("ET: GLU operation with unsupported variant: %s (only SWIGLU supported)\n",
                       ggml_glu_op_name((ggml_glu_op)glu_op_type));
        return false;
    }

    // Get GLU operation name for logging
    const char* glu_op_name = ggml_glu_op_name((ggml_glu_op)glu_op_type);

    // Pack parameters - copy full tensor structures and GLU parameters (split mode)
    ggml_et_glu_params params;
    params.src0 = *node->src[0];              // F32 input tensor A
    params.src1 = *node->src[1];              // F32 input tensor B (split mode)
    params.dst = *node;                       // F32 output tensor
    params.glu_op_type = glu_op_type;         // GLU variant type
    params.swapped = swapped;                 // Swapped flag (unused in split mode)

    GGML_LOG_DEBUG("ET: Launching glu_f32 kernel for %s (split mode) "
                   "(F32[%lld,%lld,%lld,%lld] x F32[%lld,%lld,%lld,%lld] -> F32[%lld,%lld,%lld,%lld])\n",
                   glu_op_name,
                   (long long)node->src[0]->ne[0], (long long)node->src[0]->ne[1],
                   (long long)node->src[0]->ne[2], (long long)node->src[0]->ne[3],
                   (long long)node->src[1]->ne[0], (long long)node->src[1]->ne[1],
                   (long long)node->src[1]->ne[2], (long long)node->src[1]->ne[3],
                   (long long)node->ne[0], (long long)node->ne[1],
                   (long long)node->ne[2], (long long)node->ne[3]);

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    bool cpu_comparison_active = false;
    if (glu_cpu_compare_config.enabled) {
        GGML_LOG_DEBUG("ET: Initializing CPU comparison for %s operation\n", glu_op_name);
        if (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, node, GGML_OP_GLU)) {
            cpu_comparison_active = true;
        } else {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for %s operation\n", glu_op_name);
        }
    }

    // Launch ET kernel
    bool kernel_result = ggml_et_launch_kernel(dev_ctx, "glu_f32", &params, sizeof(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    if (cpu_comparison_active) {
        GGML_LOG_DEBUG("ET: Performing CPU computation and comparison for %s operation\n", glu_op_name);
        if (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, node, &glu_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for %s operation\n", glu_op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("GLU", "glu_f32", node);
    return kernel_result;
}

bool ggml_et_op_mul_mat(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node) {
    ET_PERF_START();

    if (!dev_ctx || !node) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL_MAT operation\n");
        return false;
    }

    if (!node->src[0] || !node->src[1]) {
        GGML_LOG_ERROR("ET: MUL_MAT operation missing required inputs\n");
        return false;
    }

    const char* kernel_name;
    const char* src0_type_name;

    if (node->type == GGML_TYPE_F32 &&
        node->src[0]->type == GGML_TYPE_Q8_0 &&
        node->src[1]->type == GGML_TYPE_F32) {

        kernel_name = "mul_mat_f32";
        src0_type_name = "Q8_0";

        GGML_LOG_DEBUG("ET: MUL_MAT Q8_0xF32->F32 kernel selected for shapes src0=[%lld,%lld,%lld,%lld] src1=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld]\n",
                       (long long)node->src[0]->ne[0], (long long)node->src[0]->ne[1], (long long)node->src[0]->ne[2], (long long)node->src[0]->ne[3],
                       (long long)node->src[1]->ne[0], (long long)node->src[1]->ne[1], (long long)node->src[1]->ne[2], (long long)node->src[1]->ne[3],
                       (long long)node->ne[0], (long long)node->ne[1], (long long)node->ne[2], (long long)node->ne[3]);

    } else if (node->type == GGML_TYPE_F32 &&
               node->src[0]->type == GGML_TYPE_F16 &&
               node->src[1]->type == GGML_TYPE_F32) {

        kernel_name = "mul_mat_f32";
        src0_type_name = "F16";

        GGML_LOG_DEBUG("ET: MUL_MAT F16xF32->F32 kernel selected for shapes src0=[%lld,%lld,%lld,%lld] src1=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld]\n",
                       (long long)node->src[0]->ne[0], (long long)node->src[0]->ne[1], (long long)node->src[0]->ne[2], (long long)node->src[0]->ne[3],
                       (long long)node->src[1]->ne[0], (long long)node->src[1]->ne[1], (long long)node->src[1]->ne[2], (long long)node->src[1]->ne[3],
                       (long long)node->ne[0], (long long)node->ne[1], (long long)node->ne[2], (long long)node->ne[3]);

    } else if (node->type == GGML_TYPE_F32 &&
               node->src[0]->type == GGML_TYPE_F32 &&
               node->src[1]->type == GGML_TYPE_F32) {

        kernel_name = "mul_mat_f32";
        src0_type_name = "F32";

        GGML_LOG_DEBUG("ET: MUL_MAT F32xF32->F32 kernel selected for shapes src0=[%lld,%lld,%lld,%lld] src1=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld]\n",
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

    GGML_LOG_DEBUG("ET: MUL_MAT tensor strides - src0.nb=[%zu,%zu,%zu,%zu] src1.nb=[%zu,%zu,%zu,%zu] dst.nb=[%zu,%zu,%zu,%zu]\n",
                   node->src[0]->nb[0], node->src[0]->nb[1], node->src[0]->nb[2], node->src[0]->nb[3],
                   node->src[1]->nb[0], node->src[1]->nb[1], node->src[1]->nb[2], node->src[1]->nb[3],
                   node->nb[0], node->nb[1], node->nb[2], node->nb[3]);

    ggml_et_binary_params params;
    params.src0 = *node->src[0];  // weight matrix
    params.src1 = *node->src[1];  // activation matrix
    params.dst = *node;           // output matrix

    GGML_LOG_DEBUG("ET: Launching MUL_MAT kernel %s (%s[%lld,%lld] x F32[%lld,%lld] -> F32[%lld,%lld])\n",
                   kernel_name, src0_type_name,
                   (long long)node->src[0]->ne[0], (long long)node->src[0]->ne[1],
                   (long long)node->src[1]->ne[0], (long long)node->src[1]->ne[1],
                   (long long)node->ne[0], (long long)node->ne[1]);

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    bool cpu_comparison_active = false;
    if (mul_mat_cpu_compare_config.enabled) {
        GGML_LOG_DEBUG("ET: Initializing CPU comparison for MUL_MAT operation\n");
        if (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, node, GGML_OP_MUL_MAT)) {
            cpu_comparison_active = true;
        } else {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MUL_MAT operation\n");
        }
    }

    bool kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, sizeof(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    if (cpu_comparison_active) {
        GGML_LOG_DEBUG("ET: Performing CPU computation and comparison for MUL_MAT operation\n");
        if (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, node, &mul_mat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MUL_MAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    {
        // Calculate actual FLOPs including batch/sequence dimensions
        // dst shape: [M, N, ne2, ne3] where M=ne[1], N=ne[0]
        int64_t m = node->ne[1];
        int64_t n = node->ne[0];
        int64_t k = node->src[0]->ne[0];
        int64_t ne2 = node->ne[2];
        int64_t ne3 = node->ne[3];

        // Total FLOPs = (batch_size) * M * N * (2*K - 1)
        // Each MxN matrix-matrix multiply does M*N*(2*K-1) FLOPs
        // Broadcasting is handled by repeating computation, so count actual operations
        int64_t batch_size = ne2 * ne3;
        int64_t total_flops = batch_size * m * n * (2 * k - 1);

        char kernel_variant[64];
        snprintf(kernel_variant, sizeof(kernel_variant), "%s_%sx%s", kernel_name, src0_type_name, ggml_type_name(node->src[1]->type));
        ET_PERF_END_EXT("MUL_MAT", kernel_variant, node, "flops=%" PRId64,
                        total_flops);
    }
    return kernel_result;
}

bool ggml_et_op_rope(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node) {
    ET_PERF_START();

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

    ET_PERF_END_EXT("ROPE", kernel_name, node, "mode=0x%x|n_dims=%d|freq_base=%.2f|freq_scale=%.2f",
                    params.rope_params.mode, params.rope_params.n_dims,
                    (double)params.rope_params.freq_base, (double)params.rope_params.freq_scale);
    return kernel_result;
}

bool ggml_et_op_rms_norm(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node) {
    ET_PERF_START();

    if (!dev_ctx || !node) {
        GGML_LOG_ERROR("ET: Invalid parameters for RMS_NORM operation\n");
        return false;
    }

    if (!node->src[0]) {
        GGML_LOG_ERROR("ET: RMS_NORM operation missing required input\n");
        return false;
    }

    const char* kernel_name;

    if (node->type == GGML_TYPE_F32 &&
        node->src[0]->type == GGML_TYPE_F32) {

        kernel_name = "rms_norm_f32";

    } else {
        GGML_LOG_ERROR("ET: RMS_NORM operation with unsupported types: dst=%s src0=%s\n",
                       ggml_type_name(node->type),
                       ggml_type_name(node->src[0]->type));
        return false;
    }

    float eps;
    memcpy(&eps, node->op_params, sizeof(float));

    ggml_et_rms_norm_params params;
    params.src0 = *node->src[0];  // F32 input tensor
    params.dst = *node;           // F32 output tensor
    params.eps = eps;             // Epsilon parameter for numerical stability

    GGML_LOG_DEBUG("ET: Launching RMS_NORM kernel %s (F32[%lld,%lld,%lld,%lld] -> F32[%lld,%lld,%lld,%lld], eps=%.6f)\n",
                   kernel_name,
                   (long long)node->src[0]->ne[0], (long long)node->src[0]->ne[1],
                   (long long)node->src[0]->ne[2], (long long)node->src[0]->ne[3],
                   (long long)node->ne[0], (long long)node->ne[1],
                   (long long)node->ne[2], (long long)node->ne[3],
                   eps);

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    bool cpu_comparison_active = false;
    if (rms_norm_cpu_compare_config.enabled) {
        GGML_LOG_DEBUG("ET: Initializing CPU comparison for RMS_NORM operation\n");
        if (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, node, GGML_OP_RMS_NORM)) {
            cpu_comparison_active = true;
        } else {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RMS_NORM operation\n");
        }
    }

    bool kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, sizeof(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    if (cpu_comparison_active) {
        GGML_LOG_DEBUG("ET: Performing CPU computation and comparison for RMS_NORM operation\n");
        if (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, node, &rms_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RMS_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("RMS_NORM", kernel_name, node, "eps=%.6f", (double)eps);
    return kernel_result;
}

bool ggml_et_op_softmax(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node) {
    ET_PERF_START();

    if (!dev_ctx || !node) {
        GGML_LOG_ERROR("ET: Invalid parameters for SOFTMAX operation\n");
        return false;
    }

    if (!node->src[0]) {
        GGML_LOG_ERROR("ET: SOFTMAX operation missing required input\n");
        return false;
    }

    const char* kernel_name;

    if (node->type == GGML_TYPE_F32 &&
        node->src[0]->type == GGML_TYPE_F32) {

        kernel_name = "softmax_f32";

    } else {
        GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported types: dst=%s src0=%s\n",
                       ggml_type_name(node->type),
                       ggml_type_name(node->src[0]->type));
        return false;
    }

    // Validate contiguity requirements
    if (!ggml_is_contiguous(node)) {
        GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous destination tensor\n");
        return false;
    }

    if (!ggml_is_contiguous(node->src[0])) {
        GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous source tensor\n");
        return false;
    }

    // Check optional mask tensor
    if (node->src[1]) {
        if (node->src[1]->type != GGML_TYPE_F32) {
            GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported mask type: %s (F32 required)\n",
                           ggml_type_name(node->src[1]->type));
            return false;
        }
        if (!ggml_is_contiguous(node->src[1])) {
            GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous mask tensor\n");
            return false;
        }
    }

    // Extract scale and max_bias from op_params
    float scale = 1.0f;
    float max_bias = 0.0f;
    if (node->op_params) {
        memcpy(&scale, (const float*)node->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float*)node->op_params + 1, sizeof(float));
    }

    ggml_et_softmax_params params;
    params.src0 = *node->src[0];  // F32 input tensor
    if (node->src[1]) {
        params.src1 = *node->src[1];  // F32 mask tensor
    } else {
        memset(&params.src1, 0, sizeof(params.src1));  // Zero if no mask
    }
    params.dst = *node;           // F32 output tensor
    params.scale = scale;         // Scale factor
    params.max_bias = max_bias;   // ALiBi bias

    if (node->src[1]) {
        GGML_LOG_DEBUG("ET: Launching SOFTMAX kernel %s with mask (F32[%lld,%lld,%lld,%lld] + F32[%lld,%lld,%lld,%lld] -> F32[%lld,%lld,%lld,%lld], scale=%.6f, max_bias=%.6f)\n",
                       kernel_name,
                       (long long)node->src[0]->ne[0], (long long)node->src[0]->ne[1],
                       (long long)node->src[0]->ne[2], (long long)node->src[0]->ne[3],
                       (long long)node->src[1]->ne[0], (long long)node->src[1]->ne[1],
                       (long long)node->src[1]->ne[2], (long long)node->src[1]->ne[3],
                       (long long)node->ne[0], (long long)node->ne[1],
                       (long long)node->ne[2], (long long)node->ne[3],
                       scale, max_bias);
    } else {
        GGML_LOG_DEBUG("ET: Launching SOFTMAX kernel %s (F32[%lld,%lld,%lld,%lld] -> F32[%lld,%lld,%lld,%lld], scale=%.6f, max_bias=%.6f)\n",
                       kernel_name,
                       (long long)node->src[0]->ne[0], (long long)node->src[0]->ne[1],
                       (long long)node->src[0]->ne[2], (long long)node->src[0]->ne[3],
                       (long long)node->ne[0], (long long)node->ne[1],
                       (long long)node->ne[2], (long long)node->ne[3],
                       scale, max_bias);
    }

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    bool cpu_comparison_active = false;
    if (softmax_cpu_compare_config.enabled) {
        GGML_LOG_DEBUG("ET: Initializing CPU comparison for SOFTMAX operation\n");
        if (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, node, GGML_OP_SOFT_MAX)) {
            cpu_comparison_active = true;
        } else {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SOFTMAX operation\n");
        }
    }

    bool kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, sizeof(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    if (cpu_comparison_active) {
        GGML_LOG_DEBUG("ET: Performing CPU computation and comparison for SOFTMAX operation\n");
        if (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, node, &softmax_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SOFTMAX operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("SOFTMAX", kernel_name, node, "scale=%.6f|max_bias=%.6f|has_mask=%s",
                    (double)scale, (double)max_bias, node->src[1] ? "yes" : "no");
    return kernel_result;
}

bool ggml_et_op_get_rows(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node) {
    ET_PERF_START();

    if (!dev_ctx || !node) {
        GGML_LOG_ERROR("ET: Invalid parameters for GET_ROWS operation\n");
        return false;
    }

    if (!node->src[0] || !node->src[1]) {
        GGML_LOG_ERROR("ET: GET_ROWS operation missing required inputs\n");
        return false;
    }

    const char* kernel_name;

    if (node->type == GGML_TYPE_F32 &&
        node->src[1]->type == GGML_TYPE_I32 &&
        (node->src[0]->type == GGML_TYPE_F32 || node->src[0]->type == GGML_TYPE_Q8_0)) {

        kernel_name = "get_rows_f32";

    } else {
        GGML_LOG_ERROR("ET: GET_ROWS operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(node->type),
                       ggml_type_name(node->src[0]->type),
                       ggml_type_name(node->src[1]->type));
        return false;
    }

    // Validate contiguity requirements
    if (!ggml_is_contiguous(node)) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous destination tensor\n");
        return false;
    }

    if (!ggml_is_contiguous(node->src[0])) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous data tensor\n");
        return false;
    }

    if (!ggml_is_contiguous(node->src[1])) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous indices tensor\n");
        return false;
    }

    // Validate dimension constraints from ggml implementation
    if (node->src[0]->ne[2] != node->src[1]->ne[1] || node->src[1]->ne[3] != 1) {
        GGML_LOG_ERROR("ET: GET_ROWS operation dimension constraint failed: src0.ne[2]=%lld != src1.ne[1]=%lld or src1.ne[3]=%lld != 1\n",
                       (long long)node->src[0]->ne[2], (long long)node->src[1]->ne[1], (long long)node->src[1]->ne[3]);
        return false;
    }

    ggml_et_get_rows_params params;
    params.src0 = *node->src[0];  // Data tensor (F32 or Q8_0)
    params.src1 = *node->src[1];  // Indices tensor (I32)
    params.dst = *node;           // Output tensor (F32)

    GGML_LOG_DEBUG("ET: Launching GET_ROWS kernel %s (%s[%lld,%lld,%lld,%lld] x I32[%lld,%lld,%lld,%lld] -> F32[%lld,%lld,%lld,%lld])\n",
                   kernel_name,
                   ggml_type_name(node->src[0]->type),
                   (long long)node->src[0]->ne[0], (long long)node->src[0]->ne[1],
                   (long long)node->src[0]->ne[2], (long long)node->src[0]->ne[3],
                   (long long)node->src[1]->ne[0], (long long)node->src[1]->ne[1],
                   (long long)node->src[1]->ne[2], (long long)node->src[1]->ne[3],
                   (long long)node->ne[0], (long long)node->ne[1],
                   (long long)node->ne[2], (long long)node->ne[3]);

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    bool cpu_comparison_active = false;
    if (get_rows_cpu_compare_config.enabled) {
        GGML_LOG_DEBUG("ET: Initializing CPU comparison for GET_ROWS operation\n");
        if (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, node, GGML_OP_GET_ROWS)) {
            cpu_comparison_active = true;
        } else {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GET_ROWS operation\n");
        }
    }

    bool kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, sizeof(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    if (cpu_comparison_active) {
        GGML_LOG_DEBUG("ET: Performing CPU computation and comparison for GET_ROWS operation\n");
        if (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, node, &get_rows_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GET_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("GET_ROWS", kernel_name, node);
    return kernel_result;
}

bool ggml_et_op_cont(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node) {
    ET_PERF_START();

    GGML_LOG_DEBUG("ET: CONT operation called\n");

    // Validate tensor types
    if (node->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: CONT operation only supports F32 output, got %s\n", ggml_type_name(node->type));
        return false;
    }

    if (!node->src[0] || node->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: CONT operation requires F32 input tensor, got %s\n",
                       node->src[0] ? ggml_type_name(node->src[0]->type) : "null");
        return false;
    }

    // Validate contiguity - output must be contiguous, input can be non-contiguous
    if (!ggml_is_contiguous(node)) {
        GGML_LOG_ERROR("ET: CONT operation requires contiguous output tensor\n");
        return false;
    }

    ggml_et_cont_params params;
    params.src0 = *node->src[0];  // F32 input tensor (potentially non-contiguous)
    params.dst = *node;           // F32 output tensor (contiguous)

    GGML_LOG_DEBUG("ET: Launching CONT kernel (F32[%lld,%lld,%lld,%lld] -> F32[%lld,%lld,%lld,%lld])\n",
                   (long long)node->src[0]->ne[0], (long long)node->src[0]->ne[1],
                   (long long)node->src[0]->ne[2], (long long)node->src[0]->ne[3],
                   (long long)node->ne[0], (long long)node->ne[1],
                   (long long)node->ne[2], (long long)node->ne[3]);

    GGML_LOG_DEBUG("ET: CONT tensor strides:\n");
    GGML_LOG_DEBUG("ET:   src0 nb=[%zu,%zu,%zu,%zu] (contiguous=%s)\n",
                   node->src[0]->nb[0], node->src[0]->nb[1], node->src[0]->nb[2], node->src[0]->nb[3],
                   ggml_is_contiguous(node->src[0]) ? "yes" : "no");
    GGML_LOG_DEBUG("ET:   dst  nb=[%zu,%zu,%zu,%zu] (contiguous=%s)\n",
                   node->nb[0], node->nb[1], node->nb[2], node->nb[3],
                   ggml_is_contiguous(node) ? "yes" : "no");

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    bool cpu_comparison_active = false;
    if (cont_cpu_compare_config.enabled) {
        GGML_LOG_DEBUG("ET: Initializing CPU comparison for CONT operation\n");
        if (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, node, GGML_OP_CONT)) {
            cpu_comparison_active = true;
        } else {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CONT operation\n");
        }
    }

    bool kernel_result = ggml_et_launch_kernel(dev_ctx, "cont_f32", &params, sizeof(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    if (cpu_comparison_active) {
        GGML_LOG_DEBUG("ET: Performing CPU computation and comparison for CONT operation\n");
        if (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, node, &cont_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CONT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("CONT", "cont_f32", node);
    return kernel_result;
}

bool ggml_et_op_set_rows(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node) {
    ET_PERF_START();

    if (!dev_ctx || !node) {
        GGML_LOG_ERROR("ET: Invalid parameters for SET_ROWS operation\n");
        return false;
    }

    if (!node->src[0] || !node->src[1]) {
        GGML_LOG_ERROR("ET: SET_ROWS operation missing required inputs\n");
        return false;
    }

    const char* kernel_name;

    // Support F32 data with I64 indices -> F32/F16 output (scatter operation)
    if (node->src[0]->type == GGML_TYPE_F32 &&
        node->src[1]->type == GGML_TYPE_I64 &&
        (node->type == GGML_TYPE_F32 || node->type == GGML_TYPE_F16)) {

        if (node->type == GGML_TYPE_F32 || node->type == GGML_TYPE_F16) {
            kernel_name = "set_rows_f32";
        } else {
            GGML_LOG_ERROR("ET: SET_ROWS unsupported output type: %s\n", ggml_type_name(node->type));
            return false;
        }

    } else {
        GGML_LOG_ERROR("ET: SET_ROWS operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(node->type),
                       ggml_type_name(node->src[0]->type),
                       ggml_type_name(node->src[1]->type));
        return false;
    }

    // Validate contiguity requirements
    if (!ggml_is_contiguous_rows(node)) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous-rows destination tensor\n");
        return false;
    }

    if (!ggml_is_contiguous_rows(node->src[0])) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous-rows source tensor\n");
        return false;
    }

    if (!ggml_is_contiguous(node->src[1])) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous indices tensor\n");
        return false;
    }

    // Validate dimension constraints from ggml implementation
    if (!(node->ne[0] == node->src[0]->ne[0] &&                    // same number of columns
          node->ne[2] == node->src[0]->ne[2] &&                    // same batch size
          node->ne[3] == node->src[0]->ne[3] &&                    // same outer dimension
          node->src[0]->ne[1] == node->src[1]->ne[0] &&            // src rows = index count
          node->src[0]->ne[2] % node->src[1]->ne[1] == 0 &&        // batch constraint
          node->src[0]->ne[3] % node->src[1]->ne[2] == 0 &&        // outer constraint
          node->src[1]->ne[3] == 1)) {                             // indices constraint
        GGML_LOG_ERROR("ET: SET_ROWS operation dimension constraint failed\n");
        return false;
    }

    ggml_et_set_rows_params params;
    params.src0 = *node->src[0];  // F32 source data tensor
    params.src1 = *node->src[1];  // I64 indices tensor
    params.dst = *node;           // F32/F16 destination tensor

    GGML_LOG_DEBUG("ET: Launching SET_ROWS kernel %s (F32[%lld,%lld,%lld,%lld] x I64[%lld,%lld,%lld,%lld] -> %s[%lld,%lld,%lld,%lld])\n",
                   kernel_name,
                   (long long)node->src[0]->ne[0], (long long)node->src[0]->ne[1],
                   (long long)node->src[0]->ne[2], (long long)node->src[0]->ne[3],
                   (long long)node->src[1]->ne[0], (long long)node->src[1]->ne[1],
                   (long long)node->src[1]->ne[2], (long long)node->src[1]->ne[3],
                   ggml_type_name(node->type),
                   (long long)node->ne[0], (long long)node->ne[1],
                   (long long)node->ne[2], (long long)node->ne[3]);

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    bool cpu_comparison_active = false;
    if (set_rows_cpu_compare_config.enabled) {
        GGML_LOG_DEBUG("ET: Initializing CPU comparison for SET_ROWS operation\n");
        if (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, node, GGML_OP_SET_ROWS)) {
            cpu_comparison_active = true;
        } else {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SET_ROWS operation\n");
        }
    }

    bool kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, sizeof(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    if (cpu_comparison_active) {
        GGML_LOG_DEBUG("ET: Performing CPU computation and comparison for SET_ROWS operation\n");
        if (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, node, &set_rows_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SET_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("SET_ROWS", kernel_name, node);
    return kernel_result;
}
