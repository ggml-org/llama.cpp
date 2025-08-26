#include "ggml-et-ops.h"
#include "ggml-et-kernels.h"
#include "ggml-impl.h"

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
