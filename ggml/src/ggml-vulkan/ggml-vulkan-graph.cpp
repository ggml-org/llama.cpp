#include "ggml-vulkan-common.h"

void ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx) {
#if defined(GGML_VULKAN_RUN_TESTS)
    const std::vector<size_t> vals {
        512, 512, 128,
        128, 512, 512,
        4096, 512, 4096,
        11008, 512, 4096,
        4096, 512, 11008,
        32000, 512, 4096,
        8, 8, 8,
        100, 46, 576,
        623, 111, 128,
        100, 46, 558,
        512, 1, 256,
        128, 110, 622,
        511, 511, 127,
        511, 511, 7,
        511, 511, 17,
        49, 49, 128,
        128, 49, 49,
        4096, 49, 4096,
    };
    const size_t num_it = 100;

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 0, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 1, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 2, GGML_TYPE_Q4_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 0, GGML_TYPE_Q4_0, true);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 1, GGML_TYPE_Q4_0, true);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 2, GGML_TYPE_Q4_0, true);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 0, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 1, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 2, GGML_TYPE_Q8_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 0, GGML_TYPE_Q8_0, true);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 1, GGML_TYPE_Q8_0, true);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 2, GGML_TYPE_Q8_0, true);

    abort();

    for (size_t i = 0; i < vals.size(); i += 3) {
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 0);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 1);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 0);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 1);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 0);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 1);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 2);
        std::cerr << '\n' << std::endl;

        if (vals[i + 2] % 32 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n' << std::endl;
        }

        if (vals[i + 2] % 256 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n' << std::endl;
        }
    }

    GGML_ABORT("fatal error");
#endif

    if (subctx) {
        // Submit and wait for any pending work before reallocating the buffers
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, {});
        ctx->submit_pending = true;
        ggml_vk_synchronize(ctx);
        GGML_ASSERT(ctx->compute_ctx.expired());
        ggml_vk_ctx_begin(ctx->device, subctx);
        ctx->compute_ctx = subctx;
    }

    if (ctx->prealloc_x == nullptr || (ctx->prealloc_size_x > 0 && ctx->prealloc_x->size < ctx->prealloc_size_x)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(x_size: " << ctx->prealloc_size_x << ")");
        // Resize buffer
        if (ctx->prealloc_x != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_x);
        }
        ctx->prealloc_x = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_x);
    }
    if (ctx->prealloc_y == nullptr || (ctx->prealloc_size_y > 0 && ctx->prealloc_y->size < ctx->prealloc_size_y)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(y_size: " << ctx->prealloc_size_y << ")");
        // Resize buffer
        if (ctx->prealloc_y != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_y);
        }
        ctx->prealloc_y = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_y);
        ctx->prealloc_y_last_pipeline_used = nullptr;
        ctx->prealloc_y_last_tensor_used = nullptr;
        ctx->prealloc_y_last_decode_vector_staging = false;
    }
    if (ctx->prealloc_split_k == nullptr || (ctx->prealloc_size_split_k > 0 && ctx->prealloc_split_k->size < ctx->prealloc_size_split_k)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(split_k_size: " << ctx->prealloc_size_split_k << ")");
        // Resize buffer
        if (ctx->prealloc_split_k != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_split_k);
        }
        ctx->prealloc_split_k = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_split_k);
    }
    if (ctx->prealloc_add_rms_partials == nullptr || (ctx->prealloc_size_add_rms_partials > 0 && ctx->prealloc_add_rms_partials->size < ctx->prealloc_size_add_rms_partials)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(add_partials_size: " << ctx->prealloc_add_rms_partials << ")");
        // Resize buffer
        if (ctx->prealloc_add_rms_partials != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_add_rms_partials);
        }
        ctx->prealloc_add_rms_partials = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_add_rms_partials);
    }
}

bool ggml_vk_build_graph(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, int node_idx, ggml_tensor *node_begin, int node_idx_begin, bool last_node, bool almost_ready, bool submit){
    ggml_tensor * node = cgraph->nodes[node_idx];
    if (ggml_is_empty(node) || ggml_op_is_empty(node->op) || !node->buffer) {
        return false;
    }
    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return false;
    }

    VK_LOG_DEBUG("ggml_vk_build_graph(" << node << ", " << ggml_op_name(node->op) << ")");
    ctx->semaphore_idx = 0;

    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];
    ggml_tensor * src2 = node->src[2];
    ggml_tensor * src3 = node->src[3];

    if (node->op == GGML_OP_ADD) {
        int next_node_idx = node_idx + 1 + ctx->num_additional_fused_ops;
        if (next_node_idx < cgraph->n_nodes &&
            cgraph->nodes[next_node_idx]->op == GGML_OP_RMS_NORM &&
            cgraph->nodes[next_node_idx]->src[0] == cgraph->nodes[next_node_idx - 1] &&
            ggml_nrows(cgraph->nodes[next_node_idx]) == 1 &&
            ctx->device->add_rms_fusion) {
            uint32_t size = ggml_vk_rms_partials_size(ctx, cgraph->nodes[node_idx]);
            ctx->do_add_rms_partials_offset_calculation = true;
            if (ctx->prealloc_size_add_rms_partials_offset + size <= ctx->prealloc_size_add_rms_partials) {
                ctx->do_add_rms_partials = true;
            }
        }
    }

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    {
        // This logic detects dependencies between modes in the graph and calls ggml_vk_sync_buffers
        // to synchronize them. This handles most "normal" synchronization when computing the graph, and when
        // there is no auxiliary memory use, it shouldn't be necessary to call ggml_vk_sync_buffers
        // outside of this logic. When a node uses one of the prealloc buffers for something like
        // dequantization or split_k, additional synchronization is needed between those passes.
        bool need_sync = false;

        // Check whether "node" requires synchronization. The node requires synchronization if it
        // overlaps in memory with another unsynchronized node and at least one of them is a write.
        // Destination nodes are checked against both the written/read lists. Source nodes are only
        // checked against the written list. Two nodes overlap in memory if they come from the same
        // buffer and the tensor or view ranges overlap.
        auto const &overlaps_unsynced = [&](const ggml_tensor *node, const std::vector<const ggml_tensor *> &unsynced_nodes) -> bool {
            if (unsynced_nodes.size() == 0) {
                return false;
            }
            auto n_base = vk_tensor_offset(node) + node->view_offs;
            auto n_size = ggml_nbytes(node);
            ggml_backend_vk_buffer_context * a_buf_ctx = (ggml_backend_vk_buffer_context *)node->buffer->context;
            vk_buffer a_buf = a_buf_ctx->dev_buffer;
            for (auto &other : unsynced_nodes) {
                ggml_backend_vk_buffer_context * o_buf_ctx = (ggml_backend_vk_buffer_context *)other->buffer->context;
                vk_buffer o_buf = o_buf_ctx->dev_buffer;
                if (a_buf == o_buf) {
                    auto o_base = vk_tensor_offset(other) + other->view_offs;
                    auto o_size = ggml_nbytes(other);

                    if ((o_base <= n_base && n_base < o_base + o_size) ||
                        (n_base <= o_base && o_base < n_base + n_size)) {
                        return true;
                    }
                }
            }
            return false;
        };

        // For all fused ops, check if the destination node or any of the source
        // nodes require synchronization.
        for (int32_t i = 0; i < ctx->num_additional_fused_ops + 1 && !need_sync; ++i) {
            const ggml_tensor *cur_node = cgraph->nodes[node_idx + i];
            // If the node actually writes to memory, then check if it needs to sync
            if (ctx->fused_ops_write_mask & (1 << i)) {
                if (overlaps_unsynced(cur_node, ctx->unsynced_nodes_read) || overlaps_unsynced(cur_node, ctx->unsynced_nodes_written)) {
                    need_sync = true;
                    break;
                }
            }
            for (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                if (!cur_node->src[j]) {
                    continue;
                }
                if (overlaps_unsynced(cur_node->src[j], ctx->unsynced_nodes_written)) {
                    need_sync = true;
                    break;
                }
            }
        }

        if (need_sync) {
            if (vk_enable_sync_logger) {
                std::cerr <<  "sync" << std::endl;
            }
            ctx->unsynced_nodes_written.clear();
            ctx->unsynced_nodes_read.clear();
            ggml_vk_sync_buffers(ctx, compute_ctx);

            if (vk_perf_logger_enabled && vk_perf_logger_concurrent) {
                ctx->query_node_idx[ctx->query_idx] = node_idx;
                compute_ctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            }
        }
        // Add all fused nodes to the unsynchronized lists.
        for (int32_t i = 0; i < ctx->num_additional_fused_ops + 1; ++i) {
            const ggml_tensor *cur_node = cgraph->nodes[node_idx + i];
            // Multiple outputs could be written, e.g. in topk_moe. Add them all to the list.
            if (ctx->fused_ops_write_mask & (1 << i)) {
                ctx->unsynced_nodes_written.push_back(cur_node);
            }
            for (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                if (!cur_node->src[j]) {
                    continue;
                }
                ctx->unsynced_nodes_read.push_back(cur_node->src[j]);
            }
        }
    }
    if (vk_enable_sync_logger) {
        for (int i = 0; i < ctx->num_additional_fused_ops + 1; ++i) {
            auto *n = cgraph->nodes[node_idx + i];
            std::cerr << node_idx + i << " " << ggml_op_name(n->op) << " " <<  n->name;
            if (n->op == GGML_OP_GLU) {
                std::cerr << " " << ggml_glu_op_name(ggml_get_glu_op(n)) << " " << (n->src[1] ? "split" : "single") << " ";
            }
            if (n->op == GGML_OP_ROPE) {
                const int mode = ((const int32_t *) n->op_params)[2];
                std::cerr << " rope mode: " << mode;
            }
            std::cerr << std::endl;
        }
    }

    switch (node->op) {
    case GGML_OP_REPEAT:
        ggml_vk_repeat(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_REPEAT_BACK:
        ggml_vk_repeat_back(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_ACC:
    case GGML_OP_SET:
        ggml_vk_acc(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_GET_ROWS:
        ggml_vk_get_rows(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_ADD:
        if (ctx->num_additional_fused_ops) {
            ggml_vk_multi_add(ctx, compute_ctx, cgraph, node_idx);
        } else {
            ggml_vk_add(ctx, compute_ctx, src0, src1, node);
        }
        break;
    case GGML_OP_SUB:
        ggml_vk_sub(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_MUL:
        if (ctx->num_additional_fused_ops) {
            ggml_vk_snake_dispatch_fused(ctx, compute_ctx, cgraph, node_idx);
        } else {
            ggml_vk_mul(ctx, compute_ctx, src0, src1, node);
        }

        break;
    case GGML_OP_DIV:
        ggml_vk_div(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_ADD_ID:
        ggml_vk_add_id(ctx, compute_ctx, src0, src1, src2, node);

        break;
    case GGML_OP_CONCAT:
        ggml_vk_concat(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_UPSCALE:
        ggml_vk_upscale(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_ADD1:
        ggml_vk_add1(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_ARANGE:
        ggml_vk_arange(ctx, compute_ctx, node);

        break;
    case GGML_OP_FILL:
        ggml_vk_fill(ctx, compute_ctx, node);

        break;
    case GGML_OP_SCALE:
        ggml_vk_scale(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SQR:
        ggml_vk_sqr(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SQRT:
        ggml_vk_sqrt(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SIN:
        ggml_vk_sin(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_COS:
        ggml_vk_cos(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_LOG:
        ggml_vk_log(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_TRI:
        ggml_vk_tri(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_DIAG:
        ggml_vk_diag(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CLAMP:
        ggml_vk_clamp(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_PAD:
        ggml_vk_pad(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_ROLL:
        ggml_vk_roll(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
        ggml_vk_cpy(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SET_ROWS:
        ggml_vk_set_rows(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_SILU_BACK:
        ggml_vk_silu_back(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_NORM:
        ggml_vk_norm(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_GROUP_NORM:
        ggml_vk_group_norm(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_RMS_NORM:
        ggml_vk_rms_norm(ctx, compute_ctx, cgraph, node_idx, (float *)node->op_params);
        break;
    case GGML_OP_RMS_NORM_BACK:
        ggml_vk_rms_norm_back(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_L2_NORM:
        ggml_vk_l2_norm(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_UNARY:
        if (ctx->fused_topk_moe_mode != TOPK_MOE_COUNT) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, node_idx);
            break;
        }

        switch (ggml_get_unary_op(node)) {
        case GGML_UNARY_OP_ELU:
        case GGML_UNARY_OP_EXP:
        case GGML_UNARY_OP_EXPM1:
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_ERF:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_NEG:
        case GGML_UNARY_OP_TANH:
        case GGML_UNARY_OP_SIGMOID:
        case GGML_UNARY_OP_HARDSIGMOID:
        case GGML_UNARY_OP_HARDSWISH:
        case GGML_UNARY_OP_ABS:
        case GGML_UNARY_OP_SOFTPLUS:
        case GGML_UNARY_OP_STEP:
        case GGML_UNARY_OP_ROUND:
        case GGML_UNARY_OP_CEIL:
        case GGML_UNARY_OP_FLOOR:
        case GGML_UNARY_OP_TRUNC:
        case GGML_UNARY_OP_SGN:
            ggml_vk_unary(ctx, compute_ctx, src0, node);
            break;
        case GGML_UNARY_OP_XIELU:
            ggml_vk_xielu(ctx, compute_ctx, src0, node);
            break;
        default:
            return false;
        }
        break;
    case GGML_OP_GLU:
        switch (ggml_get_glu_op(node)) {
        case GGML_GLU_OP_GEGLU:
        case GGML_GLU_OP_REGLU:
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_SWIGLU_OAI:
        case GGML_GLU_OP_GEGLU_ERF:
        case GGML_GLU_OP_GEGLU_QUICK:
            ggml_vk_glu(ctx, compute_ctx, src0, src1, node);
            break;
        default:
            return false;
        }
        break;
    case GGML_OP_DIAG_MASK_INF:
        ggml_vk_diag_mask_inf(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SOFT_MAX:
        if (ctx->fused_topk_moe_mode != TOPK_MOE_COUNT) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, node_idx);
        } else {
            ggml_vk_soft_max(ctx, compute_ctx, src0, src1, src2, node);
        }

        break;
    case GGML_OP_SOFT_MAX_BACK:
        ggml_vk_soft_max_back(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_ROPE:
        ggml_vk_rope(ctx, compute_ctx, cgraph, node_idx, false);

        break;
    case GGML_OP_ROPE_BACK:
        ggml_vk_rope(ctx, compute_ctx, cgraph, node_idx, true);

        break;
    case GGML_OP_ARGSORT:
        if (ctx->fused_topk_moe_mode != TOPK_MOE_COUNT) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, node_idx);
        } else {
            ggml_vk_argsort(ctx, compute_ctx, src0, node);
        }

        break;
    case GGML_OP_TOP_K:
        ggml_vk_topk(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SUM:
        ggml_vk_sum(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SUM_ROWS:
        ggml_vk_sum_rows(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CUMSUM:
        ggml_vk_cumsum(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_MEAN:
        ggml_vk_mean(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_ARGMAX:
        ggml_vk_argmax(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_COUNT_EQUAL:
        ggml_vk_count_equal(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_SOLVE_TRI:
        ggml_vk_solve_tri(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_IM2COL:
        ggml_vk_im2col(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_IM2COL_3D:
        ggml_vk_im2col_3d(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_TIMESTEP_EMBEDDING:
        ggml_vk_timestep_embedding(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_COL2IM_1D:
        ggml_vk_col2im_1d(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CONV_TRANSPOSE_1D:
        ggml_vk_conv_transpose_1d(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_POOL_2D:
        ggml_vk_pool_2d(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CONV_2D:
    case GGML_OP_CONV_TRANSPOSE_2D:
        ggml_vk_conv_2d(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_CONV_2D_DW:
        ggml_vk_conv_2d_dw(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_LEAKY_RELU:
        ggml_vk_leaky_relu(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_MUL_MAT:
        ggml_vk_mul_mat(ctx, compute_ctx, cgraph, node_idx);

        break;
    case GGML_OP_MUL_MAT_ID:
        ggml_vk_mul_mat_id(ctx, compute_ctx, cgraph, node_idx);

        break;

    case GGML_OP_FLASH_ATTN_EXT:
        ggml_vk_flash_attn(ctx, compute_ctx, src0, src1, src2, src3, node->src[4], node);

        break;

    case GGML_OP_RWKV_WKV6:
        ggml_vk_rwkv_wkv6(ctx, compute_ctx, node);

        break;

    case GGML_OP_RWKV_WKV7:
        ggml_vk_rwkv_wkv7(ctx, compute_ctx, node);

        break;

    case GGML_OP_GATED_DELTA_NET:
        ggml_vk_gated_delta_net(ctx, compute_ctx, node);

        break;

    case GGML_OP_SSM_SCAN:
        ggml_vk_ssm_scan(ctx, compute_ctx, node);

        break;

    case GGML_OP_SSM_CONV:
        ggml_vk_ssm_conv(ctx, compute_ctx, cgraph, node_idx);

        break;

    case GGML_OP_OPT_STEP_ADAMW:
        ggml_vk_opt_step_adamw(ctx, compute_ctx, node);

        break;

    case GGML_OP_OPT_STEP_SGD:
        ggml_vk_opt_step_sgd(ctx, compute_ctx, src0, src1, src2, node);

        break;
    default:
        return false;
    }

    ctx->tensor_ctxs[node_idx] = compute_ctx;

#if defined(GGML_VULKAN_CHECK_RESULTS)
    // Force context reset on each node so that each tensor ends up in its own context
    // and can be run and compared to its CPU equivalent separately
    last_node = true;
#endif

    if (submit || last_node) {
        ggml_vk_ctx_end(compute_ctx);

        // TODO probably it'd be better to pass a exit_node flag to ggml_vk_compute_forward
        if (last_node) {
            compute_ctx->exit_tensor_idx = node_idx_begin;
        }
        else {
            compute_ctx->exit_tensor_idx = -1;
        }

        ctx->compute_ctx.reset();

        ggml_vk_compute_forward(ctx, cgraph, node_begin, node_idx_begin, almost_ready);
    }
    return true;
}

void ggml_vk_compute_forward(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ggml_tensor * tensor, int tensor_idx, bool almost_ready) {
    GGML_UNUSED(cgraph);
    GGML_UNUSED(tensor);

    VK_LOG_DEBUG("ggml_vk_compute_forward(" << tensor << ", name=" << tensor->name << ", op=" << ggml_op_name(tensor->op) << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << ", view_src=" << tensor->view_src << ", view_offs=" << tensor->view_offs << ")");

    vk_context subctx = ctx->tensor_ctxs[tensor_idx].lock();

    // Only run if ctx hasn't been submitted yet
    if (!subctx->seqs.empty()) {
#ifdef GGML_VULKAN_CHECK_RESULTS
        ggml_vk_check_results_0(ctx, cgraph, tensor_idx);
#endif

        // Do staging buffer copies
        for (auto& cpy : subctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        for (auto& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        if (almost_ready && !ctx->almost_ready_fence_pending) {
            ggml_vk_submit(subctx, ctx->almost_ready_fence);
            ctx->almost_ready_fence_pending = true;
        } else {
            ggml_vk_submit(subctx, {});
        }
        ctx->submit_pending = true;

#ifdef GGML_VULKAN_CHECK_RESULTS
        ggml_vk_synchronize(ctx);
        ggml_vk_check_results_1(ctx, cgraph, tensor_idx);
#endif
    }

    if (tensor_idx == subctx->exit_tensor_idx) {
        // Do staging buffer copies
        for (auto& cpy : subctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
        subctx->in_memcpys.clear();
        subctx->out_memcpys.clear();
        subctx->memsets.clear();
    }
}

void ggml_vk_graph_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_graph_cleanup()");
    ctx->prealloc_y_last_pipeline_used = {};
    ctx->prealloc_y_last_tensor_used = nullptr;
    ctx->prealloc_y_last_decode_vector_staging = false;

    ctx->unsynced_nodes_written.clear();
    ctx->unsynced_nodes_read.clear();
    ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_split_k_need_sync = false;

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
    if (ctx->device->async_use_transfer_queue) {
        ggml_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);
    }

    for (size_t i = 0; i < ctx->gc.semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.semaphores[i].s });
    }
    ctx->gc.semaphores.clear();

    for (size_t i = 0; i < ctx->gc.tl_semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.tl_semaphores[i].s });
    }
    ctx->gc.tl_semaphores.clear();
    ctx->semaphore_idx = 0;

    ctx->event_idx = 0;

    for (auto& event : ctx->gc.events) {
        ctx->device->device.resetEvent(event);
    }

    ctx->tensor_ctxs.clear();
    ctx->gc.contexts.clear();
    ctx->pipeline_descriptor_set_requirements = 0;
    ctx->descriptor_set_idx = 0;
}

void ggml_vk_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_cleanup(" << ctx->name << ")");
    // discard any unsubmitted command buffers
    ctx->compute_ctx.reset();
    // wait for any pending command buffers to finish
    ggml_vk_synchronize(ctx);

    ggml_vk_graph_cleanup(ctx);

    ggml_vk_destroy_buffer(ctx->prealloc_x);
    ggml_vk_destroy_buffer(ctx->prealloc_y);
    ggml_vk_destroy_buffer(ctx->prealloc_split_k);
    ggml_vk_destroy_buffer(ctx->prealloc_add_rms_partials);
    ggml_vk_destroy_buffer(ctx->sync_staging);

    ctx->prealloc_y_last_pipeline_used = nullptr;
    ctx->prealloc_y_last_tensor_used = nullptr;
    ctx->prealloc_y_last_decode_vector_staging = false;

    ctx->prealloc_size_x = 0;
    ctx->prealloc_size_y = 0;
    ctx->prealloc_size_split_k = 0;

    for (auto& event : ctx->gc.events) {
        ctx->device->device.destroyEvent(event);
    }
    ctx->gc.events.clear();

    ctx->device->device.destroyFence(ctx->fence);
    ctx->device->device.destroyFence(ctx->almost_ready_fence);

    for (auto& pool : ctx->descriptor_pools) {
        ctx->device->device.destroyDescriptorPool(pool);
    }
    ctx->descriptor_pools.clear();
    ctx->descriptor_sets.clear();

    ctx->compute_cmd_pool.destroy(ctx->device->device);
    if (ctx->device->async_use_transfer_queue) {
        ctx->device->device.destroySemaphore(ctx->transfer_semaphore.s);

        ctx->transfer_cmd_pool.destroy(ctx->device->device);
    }
    if (vk_perf_logger_enabled) {
        ctx->perf_logger->print_timings(true);
    }
}

void ggml_vk_synchronize(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_synchronize()");

    bool do_transfer = !ctx->compute_ctx.expired();

    if (ggml_vk_submit_transfer_ctx(ctx)) {
        ctx->submit_pending = true;
    }

    vk_context compute_ctx;
    vk_command_buffer* cmd_buf = nullptr;
    if (do_transfer) {
        compute_ctx = ctx->compute_ctx.lock();
        if (compute_ctx->s) {
            cmd_buf = compute_ctx->s->buffer;
        }

        ggml_vk_ctx_end(compute_ctx);

        for (auto& cpy : compute_ctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        ggml_vk_submit(compute_ctx, {});
        ctx->submit_pending = true;
    }

    if (ctx->submit_pending) {
        if (ctx->device->async_use_transfer_queue && ctx->transfer_semaphore_last_submitted < ctx->transfer_semaphore.value) {
            vk::TimelineSemaphoreSubmitInfo tl_info{
                1, &ctx->transfer_semaphore.value,
                0, nullptr,
            };
            vk::PipelineStageFlags stage = ctx->device->transfer_queue.stage_flags;
            vk::SubmitInfo si{
                1, &ctx->transfer_semaphore.s, &stage,
                0, nullptr,
                0, nullptr,
            };
            si.setPNext(&tl_info);
            std::lock_guard<std::mutex> guard(queue_mutex);
            ctx->device->compute_queue.queue.submit({ si }, ctx->fence);
            ctx->transfer_semaphore_last_submitted = ctx->transfer_semaphore.value;
        } else {
            std::lock_guard<std::mutex> guard(queue_mutex);
            ctx->device->compute_queue.queue.submit({}, ctx->fence);
        }
        ggml_vk_wait_for_fence(ctx);
        ctx->submit_pending = false;
        if (cmd_buf) {
            cmd_buf->in_use = false;
            cmd_buf->buf.reset();
        }
    }

    if (do_transfer) {
        for (auto& cpy : compute_ctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
        ctx->compute_ctx.reset();
    }
}

bool ggml_vk_is_empty(ggml_tensor * node) {
    return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
}

bool ggml_vk_can_fuse(const ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {
    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul = cgraph->nodes[node_idx + 1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        // rms_norm only supports f32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }
        // if rms_norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] &&
            !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }
        // rms_norm shader assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }
    }
    auto const &mm_add_ok = [&](const ggml_tensor *mul, const ggml_tensor *add) {
        const ggml_tensor *bias = add->src[0] == mul ? add->src[1] : add->src[0];

        // mat-vec only
        if (ggml_nrows(mul) != 1) {
            return false;
        }
        // shaders assume the types match
        if (mul->type != bias->type) {
            return false;
        }
        // shaders reuse the D shape for bias
        if (!ggml_are_same_shape(mul, bias) ||
            !ggml_are_same_stride(mul, bias)) {
            return false;
        }
        // unaligned bias isn't handled
        if (get_misalign_bytes(ctx, bias) != 0) {
            return false;
        }
        return true;
    };

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_ADD) {
        // additional constraints specific to this fusion
        const ggml_tensor *mul = cgraph->nodes[node_idx];
        const ggml_tensor *add = cgraph->nodes[node_idx + 1];

        if (!mm_add_ok(mul, add)) {
            return false;
        }
        if (ops.size() == 3) {
            if (ops.begin()[2] != GGML_OP_ADD) {
                return false;
            }
            if (!mm_add_ok(add, cgraph->nodes[node_idx + 2])) {
                return false;
            }
        }
    }

    auto const &mmid_mul_ok = [&](const ggml_tensor *mmid, const ggml_tensor *mul) {
        const ggml_tensor *scale = mul->src[1];

        if (mmid != mul->src[0]) {
            return false;
        }
        // mat-vec only
        if (!ggml_vk_use_mul_mat_vec_id(cgraph, node_idx)) {
            return false;
        }
        // shaders assume the types match
        if (mmid->type != scale->type) {
            return false;
        }
        // shaders assume the bias is contiguous
        if (!ggml_is_contiguous(scale)) {
            return false;
        }
        // unaligned bias isn't handled
        if (get_misalign_bytes(ctx, scale) != 0) {
            return false;
        }
        // shader only indexes by expert index
        if (scale->ne[0] != 1 ||
            scale->ne[1] != mul->ne[1] ||
            scale->ne[2] != 1 ||
            scale->ne[3] != 1) {
            return false;
        }
        return true;
    };

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_ADD_ID) {
        // additional constraints specific to this fusion
        const ggml_tensor *mul = cgraph->nodes[node_idx];
        const ggml_tensor *add = cgraph->nodes[node_idx + 1];
        const ggml_tensor *bias = add->src[1];

        if (mul != add->src[0]) {
            return false;
        }
        // mat-vec only
        if (!ggml_vk_use_mul_mat_vec_id(cgraph, node_idx)) {
            return false;
        }
        // shaders assume the types match
        if (mul->type != bias->type) {
            return false;
        }
        // shaders assume the bias is contiguous
        if (!ggml_is_contiguous(bias)) {
            return false;
        }
        // the ID tensor must be the same for mul_mat_id and add_id
        if (mul->src[2] != add->src[2]) {
            return false;
        }
        // unaligned bias isn't handled
        if (get_misalign_bytes(ctx, bias) != 0) {
            return false;
        }

        if (ops.size() == 3) {
            if (ops.begin()[2] != GGML_OP_MUL) {
                return false;
            }
            const ggml_tensor *mul = cgraph->nodes[node_idx + 2];
            return mmid_mul_ok(add, mul);
        }
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        const ggml_tensor *mmid = cgraph->nodes[node_idx];
        const ggml_tensor *mul = cgraph->nodes[node_idx + 1];

        if (!mmid_mul_ok(mmid, mul)) {
            return false;
        }
    }

    return true;
}

bool ggml_vk_can_fuse_ssm_conv(const ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph,
                                      int node_idx, int num_extra) {
    const ggml_tensor * conv = cgraph->nodes[node_idx];
    if (conv->op != GGML_OP_SSM_CONV) {
        return false;
    }

    const ggml_tensor * silu = nullptr;
    const ggml_tensor * bias = nullptr;

    if (num_extra == 1) {
        if (!ggml_can_fuse(cgraph, node_idx, { GGML_OP_SSM_CONV, GGML_OP_UNARY })) {
            return false;
        }
        silu = cgraph->nodes[node_idx + 1];
    } else if (num_extra == 2) {
        if (!ggml_can_fuse(cgraph, node_idx, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY })) {
            return false;
        }
        const ggml_tensor * add = cgraph->nodes[node_idx + 1];
        silu = cgraph->nodes[node_idx + 2];
        bias = (add->src[0] == conv) ? add->src[1] : add->src[0];

        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        // bias must be channel-wise (one element per channel of the conv output)
        if (ggml_nelements(bias) != conv->ne[0] || bias->ne[0] != conv->ne[0]) {
            return false;
        }
        if (add->type != GGML_TYPE_F32) {
            return false;
        }
        // The shader doesn't apply per-tensor offsets, so reject misaligned bias.
        if (get_misalign_bytes(ctx, bias) != 0) {
            return false;
        }
    } else {
        return false;
    }

    if (ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) {
        return false;
    }
    if (conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
        return false;
    }
    // The shader writes to the fused dst using its own strides, but the push constants don't
    // carry a per-tensor offset, so the binding must be naturally aligned.
    if (get_misalign_bytes(ctx, silu) != 0) {
        return false;
    }
    return true;
}

bool ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph,
                                      int node_idx, topk_moe_mode mode) {

    const ggml_tensor * softmax;
    const ggml_tensor * weights;
    const ggml_tensor * get_rows;
    const ggml_tensor * argsort;

    switch (mode) {
    case TOPK_MOE_EARLY_SOFTMAX_NORM:
        softmax = cgraph->nodes[node_idx + 0];
        weights = cgraph->nodes[node_idx + 9];
        get_rows = cgraph->nodes[node_idx + 4];
        argsort = cgraph->nodes[node_idx + 2];
        break;
    case TOPK_MOE_SIGMOID_NORM_BIAS:
        softmax = cgraph->nodes[node_idx + 0]; // really sigmoid
        weights = cgraph->nodes[node_idx + 10];
        get_rows = cgraph->nodes[node_idx + 5];
        argsort = cgraph->nodes[node_idx + 3];
        if (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SIGMOID) {
            return false;
        }
        // bias is expected to be 1D
        if (ggml_nrows(cgraph->nodes[node_idx + 2]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[node_idx + 2]->src[1])) {
            return false;
        }
        // sigmoid fusion seems to generate infinities on moltenvk
        if (ctx->device->driver_id == vk::DriverId::eMoltenvk) {
            return false;
        }
        break;
    case TOPK_MOE_EARLY_SOFTMAX:
        softmax = cgraph->nodes[node_idx + 0];
        weights = cgraph->nodes[node_idx + 4];
        get_rows = cgraph->nodes[node_idx + 4];
        argsort = cgraph->nodes[node_idx + 2];
        break;
    case TOPK_MOE_LATE_SOFTMAX:
        softmax = cgraph->nodes[node_idx + 4];
        weights = cgraph->nodes[node_idx + 5];
        get_rows = cgraph->nodes[node_idx + 2];
        argsort = cgraph->nodes[node_idx + 0];
        break;
    default:
        return false;
    }

    ggml_tensor * probs = get_rows->src[0];
    if (probs->op != GGML_OP_RESHAPE) {
        return false;
    }
    probs = probs->src[0];
    ggml_tensor * selection_probs = argsort->src[0];

    if (probs != selection_probs && mode != TOPK_MOE_SIGMOID_NORM_BIAS) {
        return false;
    }

    if (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(weights)) {
        return false;
    }

    if (softmax->op == GGML_OP_SOFT_MAX) {
        const float * op_params = (const float *)softmax->op_params;

        float scale = op_params[0];
        float max_bias = op_params[1];

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    }

    const int n_expert = softmax->ne[0];
    if (n_expert > (1 << (num_topk_moe_pipelines-1))) {
        return false;
    }

    if (!ctx->device->subgroup_arithmetic ||
        !ctx->device->subgroup_shuffle ||
        !ctx->device->subgroup_require_full_support ||
        ctx->device->disable_fusion) {
        return false;
    }

    return true;
}

bool ggml_vk_can_fuse_rope_set_rows(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph,
                                           int node_idx) {
    GGML_UNUSED(ctx);
    const ggml_tensor *rope = cgraph->nodes[node_idx + 0];
    const ggml_tensor *view = cgraph->nodes[node_idx + 1];
    const ggml_tensor *set_rows = cgraph->nodes[node_idx + 2];

    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) ||
        view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox/mrope shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_MROPE) {
        return false;
    }

    return true;
}

bool ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx) {
    GGML_UNUSED(ctx);
    if (!ggml_can_fuse(cgraph, node_idx, snake_pattern)) {
        return false;
    }

    const ggml_tensor * mul0     = cgraph->nodes[node_idx + 0];
    const ggml_tensor * sin_node = cgraph->nodes[node_idx + 1];
    const ggml_tensor * sqr      = cgraph->nodes[node_idx + 2];
    const ggml_tensor * mul1     = cgraph->nodes[node_idx + 3];
    const ggml_tensor * add      = cgraph->nodes[node_idx + 4];

    const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    const ggml_tensor * inv_b    = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];
    const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    if (x_in_add != x) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16 && x->type != GGML_TYPE_BF16) {
        return false;
    }
    // Shader bindings: data_a is A_TYPE so it follows x's precision, while
    // data_b and data_c are hardcoded float, so the broadcast operands must
    // be F32 regardless of x's type.
    if (a->type     != GGML_TYPE_F32) return false;
    if (inv_b->type != GGML_TYPE_F32) return false;
    // Chain intermediates and output share x's precision (single A_TYPE / D_TYPE pipeline).
    if (mul0->type     != x->type) return false;
    if (sin_node->type != x->type) return false;
    if (sqr->type      != x->type) return false;
    if (mul1->type     != x->type) return false;
    if (add->type      != x->type) return false;
    if (!ggml_are_same_shape(a, inv_b)) {
        return false;
    }
    if (a->ne[0] != 1 || a->ne[1] != x->ne[1]) {
        return false;
    }
    // Dispatch is 2D over (ne0, ne1), so x and add must be 2D and a / inv_b
    // must collapse to [1, C, 1, 1]. Higher dims are not handled by the shader.
    if (x->ne[2]     != 1 || x->ne[3]     != 1) return false;
    if (add->ne[2]   != 1 || add->ne[3]   != 1) return false;
    if (a->ne[2]     != 1 || a->ne[3]     != 1) return false;
    if (inv_b->ne[2] != 1 || inv_b->ne[3] != 1) return false;
    // Shader uses idx = i0 + i1 * ne0 and reads data_b[i1] / data_c[i1],
    // so every operand must be contiguous.
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(add) ||
        !ggml_is_contiguous(a) || !ggml_is_contiguous(inv_b)) {
        return false;
    }
    return true;
}

bool ggml_vk_tensors_overlap(const ggml_tensor * a, const ggml_tensor * b, bool elementwise) {
    ggml_backend_vk_buffer_context * a_buf_ctx = (ggml_backend_vk_buffer_context *)a->buffer->context;
    vk_buffer a_buf = a_buf_ctx->dev_buffer;
    ggml_backend_vk_buffer_context * b_buf_ctx = (ggml_backend_vk_buffer_context *)b->buffer->context;
    vk_buffer b_buf = b_buf_ctx->dev_buffer;
    if (a_buf == b_buf) {
        auto a_base = vk_tensor_offset(a) + a->view_offs;
        auto a_size = ggml_nbytes(a);
        auto b_base = vk_tensor_offset(b) + b->view_offs;
        auto b_size = ggml_nbytes(b);

        if (elementwise && a_base == b_base && a_size == b_size) {
            return false;
        }

        if ((b_base <= a_base && a_base < b_base + b_size) ||
            (a_base <= b_base && b_base < a_base + a_size)) {
            return true;
        }
    }
    return false;
}

bool ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph,
                                               int node_idx) {
    GGML_UNUSED(ctx);
    const ggml_tensor *rms = cgraph->nodes[node_idx + 0];
    const ggml_tensor *mul = cgraph->nodes[node_idx + 1];
    const ggml_tensor *rope = cgraph->nodes[node_idx + 2];

    const int mode = ((const int32_t *) rope->op_params)[2];

    // noncontig tensors aren't tested, and don't seem common in practice
    if (!ggml_is_contiguous(rms) ||
        !ggml_is_contiguous(mul) ||
        !ggml_is_contiguous(rope)) {
        return false;
    }

    // only norm/neox are handled in the shader
    if (mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_NORMAL) {
        return false;
    }

    // shared memory size for passing data from mul->rope
    if (mul->ne[0] > 1024) {
        return false;
    }

    // conditions for pipeline creation
    if (sizeof(vk_op_rms_norm_mul_rope_push_constants) > ctx->device->properties.limits.maxPushConstantsSize) {
        return false;
    }

    return true;
}

uint32_t ggml_vk_fuse_multi_add(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx) {

    const ggml_tensor *first_node = cgraph->nodes[node_idx];
    if (first_node->op != GGML_OP_ADD) {
        return 0;
    }

    if (!ctx->device->multi_add) {
        return 0;
    }

    int32_t num_adds = 1;
    while (node_idx + num_adds < cgraph->n_nodes &&
           cgraph->nodes[node_idx + num_adds]->op == GGML_OP_ADD &&
           num_adds < MAX_FUSED_ADDS) {
        num_adds++;
    }

    // The shader currently requires same shapes (but different strides are allowed),
    // everything f32, and no misalignment
    for (int32_t i = 0; i < num_adds; ++i) {
        const ggml_tensor *next_node = cgraph->nodes[node_idx + i];
        if (!ggml_are_same_shape(first_node, next_node->src[0]) ||
            !ggml_are_same_shape(first_node, next_node->src[1]) ||
            next_node->type != GGML_TYPE_F32 ||
            next_node->src[0]->type != GGML_TYPE_F32 ||
            next_node->src[1]->type != GGML_TYPE_F32 ||
            get_misalign_bytes(ctx, next_node) ||
            get_misalign_bytes(ctx, next_node->src[0]) ||
            get_misalign_bytes(ctx, next_node->src[1])) {
            num_adds = i;
        }
    }

    // Verify we can fuse these
    ggml_op adds[MAX_FUSED_ADDS];
    for (int32_t i = 0; i < num_adds; ++i) {
        adds[i] = GGML_OP_ADD;
    }

    // decrease num_adds if they can't all be fused
    while (num_adds > 1 && !ggml_can_fuse(cgraph, node_idx, adds, num_adds)) {
        num_adds--;
    }

    // a single add is not "fused", so just return zero
    if (num_adds == 1) {
        return 0;
    }
    return num_adds;
}

void ggml_vk_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * graph)
{
    VK_LOG_DEBUG("ggml_vk_graph_optimize(" << graph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    if (ctx->device->disable_graph_optimize) {
        return;
    }

    auto const &is_empty = [](ggml_tensor * node) -> bool {
        return node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    auto const &is_src_of = [](const ggml_tensor *dst, const ggml_tensor *src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor *dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor *src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    std::vector<ggml_tensor *> new_order;
    std::vector<bool> used(graph->n_nodes, false);
    std::set<ggml_tensor *> used_node_set;

    int first_unused = 0;
    while (first_unused < graph->n_nodes) {
        std::vector<int> current_set;

        // Check for fusion patterns and avoid reordering them
        auto const &match_pattern = [&](const std::initializer_list<ggml_op> &pattern, int start) -> bool {
            if (start + (int)pattern.size() <= graph->n_nodes) {
                bool is_pattern = true;
                for (size_t j = 0; j < pattern.size(); ++j) {
                    if (graph->nodes[start + j]->op != pattern.begin()[j] || used[start + j]) {
                        is_pattern = false;
                    }
                }
                return is_pattern;
            }
            return false;
        };

        auto const &keep_pattern = [&](const std::initializer_list<ggml_op> &pattern) -> bool {
            if (match_pattern(pattern, first_unused)) {
                for (size_t j = 0; j < pattern.size(); ++j) {
                    new_order.push_back(graph->nodes[first_unused + j]);
                    used_node_set.insert(graph->nodes[first_unused + j]);
                    used[first_unused + j] = true;
                }
                while (first_unused < graph->n_nodes && used[first_unused]) {
                    first_unused++;
                }
                return true;
            }
            return false;
        };

        if (keep_pattern(topk_moe_early_softmax_norm)) {
            continue;
        }
        if (keep_pattern(topk_moe_sigmoid_norm_bias)) {
            continue;
        }
        if (keep_pattern(topk_moe_early_softmax)) {
            continue;
        }
        if (keep_pattern(topk_moe_late_softmax)) {
            continue;
        }
        if (keep_pattern(snake_pattern)) {
            continue;
        }

        // First, grab the next unused node.
        current_set.push_back(first_unused);

        // Loop through the next N nodes. Grab any that don't depend on other nodes that
        // haven't already been run. Nodes that have already been run have used[i] set
        // to true. Allow nodes that depend on the previous node if it's a fusion pattern
        // that we support (e.g. RMS_NORM + MUL).
        // This first pass only grabs "real" (non-view nodes). Second pass grabs view nodes.
        // The goal is to not interleave real and view nodes in a way that breaks fusion.
        const int NUM_TO_CHECK = 20;
        for (int j = first_unused+1; j < std::min(first_unused + NUM_TO_CHECK, graph->n_nodes); ++j) {
            if (used[j]) {
                continue;
            }
            if (is_empty(graph->nodes[j])) {
                continue;
            }
            // Don't pull forward nodes from fusion patterns
            if (match_pattern(topk_moe_early_softmax_norm, j) ||
                match_pattern(topk_moe_sigmoid_norm_bias, j) ||
                match_pattern(topk_moe_early_softmax, j) ||
                match_pattern(topk_moe_late_softmax, j) ||
                match_pattern(snake_pattern, j)) {
                continue;
            }
            bool ok = true;
            for (int c = first_unused; c < j; ++c) {
                if (!used[c] &&
                    is_src_of(graph->nodes[j], graph->nodes[c]) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_RMS_NORM && graph->nodes[j]->op == GGML_OP_MUL) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT && graph->nodes[j]->op == GGML_OP_ADD) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_ADD_ID) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_MUL) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_ADD && graph->nodes[j]->op == GGML_OP_ADD) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_ADD) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_UNARY)) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                current_set.push_back(j);

                int rope_idx = j;

                // When we've found RMS_NORM + MUL, try to find a ROPE that uses it
                if (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_MUL &&
                    graph->nodes[j-1]->op == GGML_OP_RMS_NORM) {
                    for (int k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        if (graph->nodes[k]->op == GGML_OP_ROPE &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // Check that other srcs are already valid
                            graph->nodes[k]->src[1]->op == GGML_OP_NONE &&
                            (graph->nodes[k]->src[2] == nullptr || graph->nodes[k]->src[2]->op == GGML_OP_NONE)) {
                            rope_idx = k;
                            current_set.push_back(rope_idx);
                            used[rope_idx] = true;
                            break;
                        }
                    }
                }
                // Look for ROPE + VIEW + SET_ROWS and make them consecutive
                if (graph->nodes[rope_idx]->op == GGML_OP_ROPE) {
                    int view_idx = -1;
                    int set_rows_idx = -1;
                    for (int k = rope_idx+1; k < std::min(rope_idx + 10, graph->n_nodes); ++k) {
                        if (view_idx == -1 &&
                            graph->nodes[k]->op == GGML_OP_VIEW &&
                            graph->nodes[k]->src[0] == graph->nodes[rope_idx]) {
                            view_idx = k;
                            continue;
                        }
                        if (view_idx != -1 &&
                            set_rows_idx == -1 &&
                            graph->nodes[k]->op == GGML_OP_SET_ROWS &&
                            graph->nodes[k]->src[0] == graph->nodes[view_idx]) {
                            set_rows_idx = k;
                            break;
                        }
                    }
                    if (set_rows_idx != -1) {
                        current_set.push_back(view_idx);
                        current_set.push_back(set_rows_idx);
                        used[view_idx] = true;
                        used[set_rows_idx] = true;
                    }
                }
                // Look for MUL_MAT_ID + ADD_ID + MUL
                if (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ADD_ID &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT_ID) {
                    for (int k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        if (graph->nodes[k]->op == GGML_OP_MUL &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || used_node_set.find(graph->nodes[k]->src[1]) != used_node_set.end())) {
                            current_set.push_back(k);
                            used[k] = true;
                            break;
                        }
                    }
                }
                // Look for MUL_MAT + ADD + ADD
                if (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ADD &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT) {
                    for (int k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        if (graph->nodes[k]->op == GGML_OP_ADD &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || used_node_set.find(graph->nodes[k]->src[1]) != used_node_set.end())) {
                            current_set.push_back(k);
                            used[k] = true;
                            break;
                        }
                    }
                }
                // SSM_CONV + ADD + UNARY: pull the consuming UNARY forward
                if (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ADD &&
                    graph->nodes[j-1]->op == GGML_OP_SSM_CONV) {
                    for (int k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        if (graph->nodes[k]->op == GGML_OP_UNARY &&
                            graph->nodes[k]->src[0] == graph->nodes[j]) {
                            current_set.push_back(k);
                            used[k] = true;
                            break;
                        }
                    }
                }
            }
        }
        // Second pass grabs view nodes.
        // Skip this if it would break a fusion optimization (don't split up add->rms_norm or add->add).
        if (graph->nodes[current_set.back()]->op != GGML_OP_ADD) {
            for (int j = first_unused+1; j < std::min(first_unused + NUM_TO_CHECK, graph->n_nodes); ++j) {
                if (used[j]) {
                    continue;
                }
                if (!is_empty(graph->nodes[j])) {
                    continue;
                }
                bool ok = true;
                for (int c = first_unused; c < j; ++c) {
                    bool c_in_current_set = std::find(current_set.begin(), current_set.end(), c) != current_set.end();
                    // skip views whose srcs haven't been processed.
                    if (!used[c] &&
                        is_src_of(graph->nodes[j], graph->nodes[c]) &&
                        !c_in_current_set) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    current_set.push_back(j);
                }
            }
        }

        // Push the current set into new_order
        for (auto c : current_set) {
            new_order.push_back(graph->nodes[c]);
            used_node_set.insert(graph->nodes[c]);
            used[c] = true;
        }
        while (first_unused < graph->n_nodes && used[first_unused]) {
            first_unused++;
        }
    }
    // Replace the graph with the new order.
    for (int i = 0; i < graph->n_nodes; ++i) {
        graph->nodes[i] = new_order[i];
    }
}

