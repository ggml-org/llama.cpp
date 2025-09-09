#include "ggml-et-ops.h"
#include "ggml-et-kernels.h"
#include "ggml-impl.h"

bool ggml_et_op_mul(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node) {
    // Validate inputs
    if (!dev_ctx || !node) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL operation\n");
        return false;
    }

    if (!node->src[0] || !node->src[1]) {
        GGML_LOG_ERROR("ET: MUL operation missing required inputs\n");
        return false;
    }

    // Intelligent kernel selection based on tensor properties
    const char* kernel_name;

    // For now, only support F32 (as validated by supports_op)
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

    return ggml_et_launch_kernel(dev_ctx, kernel_name, &params, sizeof(params));
}
