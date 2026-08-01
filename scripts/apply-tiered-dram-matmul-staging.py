#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

path = Path("ggml/src/ggml-cuda/tiered.cu")
text = path.read_text(encoding="utf-8")

marker = "tiered-memory: stage DRAM MUL_MAT weights through temporary VRAM"
if marker in text:
    print(f"already patched {path}")
    raise SystemExit(0)

pattern = re.compile(
    r"static ggml_status tiered_backend_graph_compute\(ggml_backend_t backend, ggml_cgraph \* graph\) \{.*?\n\}\n\nstatic void tiered_backend_event_record",
    re.DOTALL,
)

replacement = r'''static ggml_status tiered_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * ctx = static_cast<tiered_backend_context *>(backend->context);
    int segment_begin = 0;

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        tensor_state * ssd_state = nullptr;
        tiered_buffer_context * weight_ctx = nullptr;

        tensor_state * dram_state = nullptr;
        tiered_buffer_context * dram_ctx = nullptr;
        ggml_tensor * dram_base = nullptr;

        for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
            ggml_tensor * src = node->src[src_index];
            tensor_state * state = state_for(src);
            if (!state) {
                continue;
            }

            if (state->tier == GGML_CUDA_TIERED_MEMORY_SSD) {
                if (node->op != GGML_OP_MUL_MAT_ID || src_index != 0) {
                    GGML_LOG_ERROR("tiered-memory: SSD tensor %s is used by unsupported op %s\n",
                            src->name, ggml_op_name(node->op));
                    return GGML_STATUS_FAILED;
                }
                ssd_state = state;
                weight_ctx = buffer_context(src);
                continue;
            }

            // tiered-memory: stage DRAM MUL_MAT weights through temporary VRAM.
            // CUDA kernels can directly dereference mapped host memory, but the
            // cuBLAS path used by dense/F32 MUL_MAT is not reliable with those
            // aliases on Turing-class devices. Stage only the active base weight
            // for this graph node, compute, then immediately release it.
            if (state->tier == GGML_CUDA_TIERED_MEMORY_DRAM &&
                    node->op == GGML_OP_MUL_MAT && src_index == 0) {
                size_t view_offset = 0;
                const ggml_tensor * base = tiered_view_base(src, &view_offset);
                GGML_UNUSED(view_offset);
                dram_state = state;
                dram_ctx = buffer_context(src);
                dram_base = const_cast<ggml_tensor *>(base);
            }
        }

        if (!ssd_state && !dram_state) {
            continue;
        }

        ggml_status status = compute_view(ctx, graph, segment_begin, i);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
        ggml_backend_synchronize(ctx->inner);

        if (dram_state) {
            void * staged = nullptr;
            void * original_data = dram_base ? dram_base->data : nullptr;

            try {
                set_device(ctx->device);
                TIERED_CUDA_CHECK(cudaMalloc(&staged, dram_state->alloc_size));
                TIERED_CUDA_CHECK(cudaMemcpy(
                        staged,
                        dram_state->host_ptr,
                        dram_state->size,
                        cudaMemcpyHostToDevice));
                if (dram_state->alloc_size > dram_state->size) {
                    TIERED_CUDA_CHECK(cudaMemset(
                            static_cast<char *>(staged) + dram_state->size,
                            0,
                            dram_state->alloc_size - dram_state->size));
                }

                dram_base->data = staged;
                tiered_refresh_views(dram_ctx);

                status = compute_view(ctx, graph, i, i + 1);
                ggml_backend_synchronize(ctx->inner);

                dram_base->data = original_data;
                tiered_refresh_views(dram_ctx);
                TIERED_CUDA_CHECK(cudaFree(staged));
                staged = nullptr;
            } catch (const std::exception & error) {
                if (dram_base) {
                    dram_base->data = original_data;
                    tiered_refresh_views(dram_ctx);
                }
                if (staged) {
                    (void) cudaFree(staged);
                }
                GGML_LOG_ERROR("tiered-memory: failed to stage DRAM weight %s: %s\n",
                        node->src[0] ? node->src[0]->name : "unknown", error.what());
                return GGML_STATUS_FAILED;
            }
        } else {
            ggml_tensor * weight = node->src[0];
            try {
                stage_sparse_experts(weight_ctx, weight, ssd_state, node->src[2]);
                status = compute_view(ctx, graph, i, i + 1);
                ggml_backend_synchronize(ctx->inner);
                unstage_sparse_experts(ssd_state);
            } catch (const std::exception & error) {
                GGML_LOG_ERROR("tiered-memory: failed to stream %s: %s\n",
                        weight->name, error.what());
                try {
                    unstage_sparse_experts(ssd_state);
                } catch (...) {
                }
                return GGML_STATUS_FAILED;
            }
        }

        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
        segment_begin = i + 1;
    }

    return compute_view(ctx, graph, segment_begin, graph->n_nodes);
}

static void tiered_backend_event_record'''

new_text, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise SystemExit("tiered backend graph-compute function did not match expected source")

path.write_text(new_text, encoding="utf-8")
print(f"patched {path}")
