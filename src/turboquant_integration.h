#pragma once
#include "turboquant.h"

#ifdef LLAMA_TURBOQUANT

#include <cstdint>

struct llama_tq_context;

namespace turboquant_integration {

struct tq_layer_state {
    int32_t il = -1;
    bool kv_ready = false;
};

struct llama_tq_integration {
    struct llama_tq_context * tq_ctx = nullptr;
    bool enabled = false;
    bool initialized = false;
    int32_t n_layers = 0;
    int32_t n_heads_kv = 0;
    int32_t head_dim = 0;
    int32_t max_seq_len = 0;
    int32_t key_bits = 3;
    int32_t val_bits = 2;
    tq_layer_state * layer_states = nullptr;

    bool init(int n_layers, int n_heads_kv, int head_dim, int max_seq_len, int key_bits = 3, int val_bits = 2);
    void shutdown();

    bool is_enabled() const { return enabled && initialized && tq_ctx != nullptr; }

    void set_layer_kv_ready(int32_t il) {
        if (layer_states && il >= 0 && il < n_layers) {
            layer_states[il].kv_ready = true;
        }
    }

    void reset_layer_kv_ready(int32_t il) {
        if (layer_states && il >= 0 && il < n_layers) {
            layer_states[il].kv_ready = false;
        }
    }

    bool is_layer_kv_ready(int32_t il) const {
        if (layer_states && il >= 0 && il < n_layers) {
            return layer_states[il].kv_ready;
        }
        return false;
    }
};

bool init_tq_integration(
    llama_tq_integration * tq,
    int n_layers,
    int n_heads_kv,
    int head_dim,
    int max_seq_len,
    int key_bits,
    int val_bits
);

void shutdown_tq_integration(llama_tq_integration * tq);

} // namespace turboquant_integration

#endif // LLAMA_TURBOQUANT