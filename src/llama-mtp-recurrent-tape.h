#pragma once

// MTP single-context replay tape.
//
// Instead of widening the recurrent-state cache to (1 + n_rs_seq) FULL state planes (one per draft
// token), MTP keeps a single base state row and records the RAW
// inputs of the gated-delta-net recurrence for the verify window. On partial acceptance the base is
// restored and the accepted prefix is replayed through the same GDN op, reconstructing the state.
//
// The tape holds, per recurrent layer, the exact tensors the GDN op consumes in the forward pass:
//   k    - post-conv key            [S_k, H_k, max_tokens]
//   v    - post-conv value          [S_v, H_v, max_tokens]
//   gate - pre-exp decay gate       [1,   H_v, max_tokens]
//   beta - POST-sigmoid beta        [1,   H_v, max_tokens]   (qwen35 applies sigmoid before the op)
//   qkv  - pre-transpose conv input [conv_channels, max_tokens] (for the conv-state rebuild)
//
// Recording is a ggml_cpy per tensor, emitted by the graph builder in delta-net-base.cpp; the
// buffers are allocated by llama_context. Replay re-runs ggml_gated_delta_net over the recorded
// prefix, so the state update uses the SAME kernel as the forward pass (bit-identical by
// construction.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

struct llama_mtp_recurrent_tape_layer {
    ggml_tensor * k    = nullptr;
    ggml_tensor * v    = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * beta = nullptr;
    ggml_tensor * qkv  = nullptr;

    int64_t S_k = 0;
    int64_t H_k = 0;
    int64_t S_v = 0;
    int64_t H_v = 0;
    int64_t conv_channels = 0;
};

struct llama_mtp_recurrent_tape {
    std::vector<llama_mtp_recurrent_tape_layer> layers;  // one per recurrent layer
    std::vector<int32_t> layer_ids;                  // model layer index for each tape slot

    ggml_context * ctx = nullptr;                    // owns the tensor descriptors
    ggml_backend_buffer_t buf = nullptr;             // single buffer holding all tape tensors

    int max_tokens = 0;                              // allocated capacity (>= n_max + 1)
    int n_tokens = 0;                                // tokens actually recorded by the last verify
    bool recording = false;                          // graph writes into the tape this pass

    // Per-accepted-length cached replay graphs. The graph structure depends only on n_accepted, so
    // rebuilding the context + n_rec*10 tensors + the allocation on every cycle was ~2.5 ms/step of
    // pure CPU (the measured "gdn" phase). Index = n_accepted.
    struct replay_graph {
        ggml_context *        ctx   = nullptr;
        ggml_cgraph *         graph = nullptr;
        ggml_backend_buffer_t buf   = nullptr;
        std::vector<ggml_tensor *> q, k, v, g, b;    // one per tape layer
    };
    std::vector<replay_graph> replay_cache;

    bool empty() const { return layers.empty(); }

    ~llama_mtp_recurrent_tape() {
        for (auto & rc : replay_cache) {
            if (rc.buf) {
                ggml_backend_buffer_free(rc.buf);
            }
            if (rc.ctx) {
                ggml_free(rc.ctx);
            }
        }
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }

    llama_mtp_recurrent_tape_layer * layer_for(int32_t il) {
        for (size_t i = 0; i < layer_ids.size(); ++i) {
            if (layer_ids[i] == il) {
                return &layers[i];
            }
        }
        return nullptr;
    }

    const llama_mtp_recurrent_tape_layer * layer_for(int32_t il) const {
        for (size_t i = 0; i < layer_ids.size(); ++i) {
            if (layer_ids[i] == il) {
                return &layers[i];
            }
        }
        return nullptr;
    }
};
