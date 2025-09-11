#include "ggml-et-cpu-compare.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu/ops.h"
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>

bool ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx* ctx, const ggml_tensor* node, ggml_op op) {
    if (!ctx || !node) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        return false;
    }

    // Clear context
    memset(ctx, 0, sizeof(*ctx));

    // Calculate tensor sizes
    ctx->src0_size = node->src[0] ? ggml_nbytes(node->src[0]) : 0;
    ctx->src1_size = node->src[1] ? ggml_nbytes(node->src[1]) : 0;
    ctx->src2_size = node->src[2] ? ggml_nbytes(node->src[2]) : 0;
    ctx->dst_size = ggml_nbytes(node);

    GGML_LOG_DEBUG("ET: CPU compare init for operation %s\n", ggml_op_name(op));
    GGML_LOG_DEBUG("ET: Tensor sizes - src0:%zu src1:%zu src2:%zu dst:%zu bytes\n",
                   ctx->src0_size, ctx->src1_size, ctx->src2_size, ctx->dst_size);

    // Allocate CPU buffers for all tensors
    if (ctx->src0_size > 0) {
        ctx->cpu_src0_data = malloc(ctx->src0_size);
        if (!ctx->cpu_src0_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            goto cleanup;
        }
    }

    if (ctx->src1_size > 0) {
        ctx->cpu_src1_data = malloc(ctx->src1_size);
        if (!ctx->cpu_src1_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            goto cleanup;
        }
    }

    if (ctx->src2_size > 0) {
        ctx->cpu_src2_data = malloc(ctx->src2_size);
        if (!ctx->cpu_src2_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            goto cleanup;
        }
    }

    ctx->cpu_dst_data = malloc(ctx->dst_size);
    if (!ctx->cpu_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        goto cleanup;
    }

    ctx->et_dst_data = malloc(ctx->dst_size);
    if (!ctx->et_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        goto cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    GGML_LOG_DEBUG("ET: Copying data from ET device buffers to CPU host buffers\n");
    
    if (ctx->src0_size > 0) {
        ggml_backend_tensor_get(node->src[0], ctx->cpu_src0_data, 0, ctx->src0_size);
    }
    if (ctx->src1_size > 0) {
        ggml_backend_tensor_get(node->src[1], ctx->cpu_src1_data, 0, ctx->src1_size);
    }
    if (ctx->src2_size > 0) {
        ggml_backend_tensor_get(node->src[2], ctx->cpu_src2_data, 0, ctx->src2_size);
    }
    
    // Zero destination buffer
    memset(ctx->cpu_dst_data, 0, ctx->dst_size);

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    if (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        goto cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_size = ggml_tensor_overhead() * 4 + ggml_graph_overhead(); // up to 4 tensors + graph
    ctx_params.mem_buffer = nullptr;
    ctx_params.no_alloc = true; // We'll manage data ourselves
    ctx->ggml_ctx = ggml_init(ctx_params);
    if (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        goto cleanup;
    }

    // Create CPU tensors with proper context
    GGML_LOG_DEBUG("ET: Creating CPU tensors\n");
    
    if (node->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, node->src[0]->type, GGML_MAX_DIMS, node->src[0]->ne);
        if (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            goto cleanup;
        }
        ctx->cpu_src0->data = ctx->cpu_src0_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src0->nb, node->src[0]->nb, sizeof(node->src[0]->nb));
        // Copy op_params if present
        if (node->src[0]->op_params) {
            memcpy(ctx->cpu_src0->op_params, node->src[0]->op_params, sizeof(node->src[0]->op_params));
        }
    }

    if (node->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, node->src[1]->type, GGML_MAX_DIMS, node->src[1]->ne);
        if (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            goto cleanup;
        }
        ctx->cpu_src1->data = ctx->cpu_src1_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src1->nb, node->src[1]->nb, sizeof(node->src[1]->nb));
        // Copy op_params if present
        if (node->src[1]->op_params) {
            memcpy(ctx->cpu_src1->op_params, node->src[1]->op_params, sizeof(node->src[1]->op_params));
        }
    }

    if (node->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, node->src[2]->type, GGML_MAX_DIMS, node->src[2]->ne);
        if (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            goto cleanup;
        }
        ctx->cpu_src2->data = ctx->cpu_src2_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src2->nb, node->src[2]->nb, sizeof(node->src[2]->nb));
        // Copy op_params if present
        if (node->src[2]->op_params) {
            memcpy(ctx->cpu_src2->op_params, node->src[2]->op_params, sizeof(node->src[2]->op_params));
        }
    }

    return true;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    return false;
}

bool ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx* ctx, const ggml_tensor* node,
                                           const ggml_et_cpu_compare_config* config) {
    if (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !node || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        return false;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = node->op;
    switch (op) {
        case GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            break;
        case GGML_OP_ADD:
            ctx->cpu_dst = ggml_add(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            break;
        case GGML_OP_MUL_MAT:
            ctx->cpu_dst = ggml_mul_mat(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            break;
        case GGML_OP_ROPE:
            // Copy op_params to destination tensor for ROPE operation
            ctx->cpu_dst = ggml_rope_ext(
                ctx->ggml_ctx, 
                ctx->cpu_src0, 
                ctx->cpu_src1, 
                ctx->cpu_src2,  // freq_factors (may be null)
                ((const int32_t*)node->op_params)[1], // n_dims
                ((const int32_t*)node->op_params)[2], // mode
                ((const int32_t*)node->op_params)[4], // n_ctx_orig
                *((const float*)((const int32_t*)node->op_params + 5)), // freq_base
                *((const float*)((const int32_t*)node->op_params + 6)), // freq_scale
                *((const float*)((const int32_t*)node->op_params + 7)), // ext_factor
                *((const float*)((const int32_t*)node->op_params + 8)), // attn_factor
                *((const float*)((const int32_t*)node->op_params + 9)), // beta_fast
                *((const float*)((const int32_t*)node->op_params + 10)) // beta_slow
            );
            break;
        case GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                float eps;
                memcpy(&eps, node->op_params, sizeof(float));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            break;
        case GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t glu_op_type = ggml_get_op_params_i32(node, 0);  // GLU variant
                ggml_glu_op glu_op = (ggml_glu_op)glu_op_type;
                
                // Only support split tensor mode
                if (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    return false;
                }
                ctx->cpu_dst = ggml_glu_split(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
            }
            break;
        default:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            return false;
    }

    if (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        return false;
    }

    ctx->cpu_dst->data = ctx->cpu_dst_data;
    // Copy stride array (nb) for correct memory layout
    memcpy(ctx->cpu_dst->nb, node->nb, sizeof(node->nb));

    // Create minimal computation graph
    GGML_LOG_DEBUG("ET: Creating CPU computation graph\n");
    ctx->cpu_graph = ggml_new_graph_custom(ctx->ggml_ctx, 1, false);
    if (!ctx->cpu_graph) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        return false;
    }
    ctx->cpu_graph->nodes[0] = ctx->cpu_dst;
    ctx->cpu_graph->n_nodes = 1;

    // Log input data for debugging if enabled
    if (config && config->log_differences) {
        if (ctx->cpu_src0_data && ctx->src0_size >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n",
                          ((uint8_t*)ctx->cpu_src0_data)[0], ((uint8_t*)ctx->cpu_src0_data)[1],
                          ((uint8_t*)ctx->cpu_src0_data)[2], ((uint8_t*)ctx->cpu_src0_data)[3]);
        }
        if (ctx->cpu_src1_data && ctx->src1_size >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n",
                          ((float*)ctx->cpu_src1_data)[0], ((float*)ctx->cpu_src1_data)[1],
                          ((float*)ctx->cpu_src1_data)[2], ((float*)ctx->cpu_src1_data)[3]);
        }
    }

    // Compute using CPU backend
    GGML_LOG_DEBUG("ET: Computing reference result with CPU backend\n");
    ggml_status cpu_result = ggml_backend_graph_compute(ctx->cpu_backend, ctx->cpu_graph);

    if (cpu_result != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_result);
        return false;
    }

    // Log output data for debugging if enabled
    if (config && config->log_differences && ctx->dst_size >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                      ((float*)ctx->cpu_dst_data)[0], ((float*)ctx->cpu_dst_data)[1],
                      ((float*)ctx->cpu_dst_data)[2], ((float*)ctx->cpu_dst_data)[3]);
    }

    GGML_LOG_DEBUG("ET: CPU reference computation completed successfully\n");

    // Now copy ET device destination to host for comparison
    GGML_LOG_DEBUG("ET: Copying ET device destination buffer for comparison\n");
    ggml_backend_tensor_get(node, ctx->et_dst_data, 0, ctx->dst_size);

    if (config->log_differences) {
        GGML_LOG_DEBUG("ET: Comparing ET vs CPU results\n");

        float* cpu_float = (float*)ctx->cpu_dst_data;
        float* et_float = (float*)ctx->et_dst_data;
        size_t num_elements = ggml_nelements(node);
        size_t max_log = std::min(num_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        bool is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_ADD || op == GGML_OP_GLU);
        float* cpu_src0_float = is_elementwise ? (float*)ctx->cpu_src0_data : nullptr;
        float* cpu_src1_float = is_elementwise ? (float*)ctx->cpu_src1_data : nullptr;

        // Compare all elements but log only the first max_log_elements
        GGML_LOG_DEBUG("ET: First %zu elements comparison (checking all %zu elements):\n", max_log, num_elements);
        bool matches = true;
        size_t total_mismatches = 0;
        
        // First pass: check all elements for mismatches
        for (size_t i = 0; i < num_elements; i++) {
            float diff = fabsf(cpu_float[i] - et_float[i]);
            float rel_diff = diff / (fabsf(cpu_float[i]) + 1e-8f);

            if (rel_diff > config->tolerance) {
                matches = false;
                total_mismatches++;
            }
        }
        
        // Second pass: log detailed info for first max_log elements only
        for (size_t i = 0; i < max_log; i++) {
            float diff = fabsf(cpu_float[i] - et_float[i]);
            float rel_diff = diff / (fabsf(cpu_float[i]) + 1e-8f);

            if (is_elementwise && cpu_src0_float && cpu_src1_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f, src1=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n",
                              i, cpu_src0_float[i], cpu_src1_float[i], cpu_float[i], et_float[i], diff);
            } else if (is_elementwise && cpu_src0_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n",
                              i, cpu_src0_float[i], cpu_float[i], et_float[i], diff);
            } else {
                GGML_LOG_DEBUG("ET: [%zu] CPU=%.6f, ET=%.6f, diff=%.6f\n",
                              i, cpu_float[i], et_float[i], diff);
            }
        }

        // Check some elements from the middle and end for full coverage
        if (num_elements > max_log) {
            size_t mid = num_elements / 2;
            size_t end = num_elements - 1;
            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n",
                          mid, cpu_float[mid], et_float[mid]);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n",
                          end, cpu_float[end], et_float[end]);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n",
                      matches ? "MATCH" : "DIFFER", num_elements - total_mismatches, num_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    if (config->use_cpu_result) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        ggml_backend_tensor_set(const_cast<ggml_tensor*>(node), ctx->cpu_dst_data, 0, ctx->dst_size);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    return true;
}


void ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx* ctx) {
    if (!ctx) return;

    if (ctx->cpu_src0_data) { free(ctx->cpu_src0_data); ctx->cpu_src0_data = nullptr; }
    if (ctx->cpu_src1_data) { free(ctx->cpu_src1_data); ctx->cpu_src1_data = nullptr; }
    if (ctx->cpu_src2_data) { free(ctx->cpu_src2_data); ctx->cpu_src2_data = nullptr; }
    if (ctx->cpu_dst_data) { free(ctx->cpu_dst_data); ctx->cpu_dst_data = nullptr; }
    if (ctx->et_dst_data) { free(ctx->et_dst_data); ctx->et_dst_data = nullptr; }

    if (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = nullptr;
    }

    if (ctx->cpu_backend) {
        ggml_backend_free(ctx->cpu_backend);
        ctx->cpu_backend = nullptr;
    }

    // Clear pointers
    ctx->cpu_src0 = nullptr;
    ctx->cpu_src1 = nullptr;
    ctx->cpu_src2 = nullptr;
    ctx->cpu_dst = nullptr;
    ctx->cpu_graph = nullptr;
}

