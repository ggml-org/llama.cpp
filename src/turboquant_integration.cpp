#include "turboquant_integration.h"
#include "turboquant.h"

#ifdef LLAMA_TURBOQUANT

#include <cstdio>
#include <cstdlib>

namespace turboquant_integration {

bool llama_tq_integration::init(
    int n_layers,
    int n_heads_kv,
    int head_dim,
    int max_seq_len,
    int key_bits,
    int val_bits
) {
    if (initialized) {
        return true;
    }

    this->n_layers = n_layers;
    this->n_heads_kv = n_heads_kv;
    this->head_dim = head_dim;
    this->max_seq_len = max_seq_len;
    this->key_bits = key_bits;
    this->val_bits = val_bits;

    layer_states = new tq_layer_state[n_layers];
    for (int i = 0; i < n_layers; i++) {
        layer_states[i].il = i;
        layer_states[i].kv_ready = false;
    }

    tq_ctx = llama_tq_create(n_layers, n_heads_kv, head_dim, max_seq_len, key_bits, val_bits);
    if (!tq_ctx) {
        fprintf(stderr, "[TurboQuant] Failed to create TurboQuant context\n");
        delete[] layer_states;
        layer_states = nullptr;
        return false;
    }

    llama_tq_set_global(tq_ctx);

    initialized = true;
    enabled = true;

    fprintf(stderr, "[TurboQuant] Integration initialized: %d layers, %d KV heads, dim=%d\n",
            n_layers, n_heads_kv, head_dim);

    return true;
}

void llama_tq_integration::shutdown() {
    if (tq_ctx) {
        llama_tq_free(tq_ctx);
        tq_ctx = nullptr;
    }

    if (layer_states) {
        delete[] layer_states;
        layer_states = nullptr;
    }

    initialized = false;
    enabled = false;
    n_layers = 0;
}

bool init_tq_integration(
    llama_tq_integration * tq,
    int n_layers,
    int n_heads_kv,
    int head_dim,
    int max_seq_len,
    int key_bits,
    int val_bits
) {
    if (!tq) return false;
    return tq->init(n_layers, n_heads_kv, head_dim, max_seq_len, key_bits, val_bits);
}

void shutdown_tq_integration(llama_tq_integration * tq) {
    if (tq) {
        tq->shutdown();
    }
}

} // namespace turboquant_integration

#endif // LLAMA_TURBOQUANT