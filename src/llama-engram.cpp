#include "llama-engram.h"

#include <cassert>
#include <cstring>
#include <limits>
#include <numeric>

// ---- Initialization ----

void engram_hash_mapping::init(const engram_hash_params & p) {
    params = p;
    layer_info.clear();

    assert(params.max_ngram_size >= 2);
    assert(params.n_head_per_ngram >= 1);
    assert(params.engram_vocab_size.size() == (size_t) (params.max_ngram_size - 1));

    uint32_t n_layer_ids = params.layer_ids.size();
    for (uint32_t i = 0; i < n_layer_ids; i++) {
        int32_t                layer_id = params.layer_ids[i];
        engram_hash_layer_info info;

        info.multipliers.resize(params.max_ngram_size);
        info.multipliers.assign(params.layer_multipliers.begin() + i * params.max_ngram_size,
                                params.layer_multipliers.begin() + (i + 1) * params.max_ngram_size);

        const uint32_t n_ngram_orders = params.max_ngram_size - 1;
        info.vocab_sizes.resize(n_ngram_orders);
        for (uint32_t ng = 0; ng < n_ngram_orders; ng++) {
            info.vocab_sizes[ng].resize(params.n_head_per_ngram);

            info.vocab_sizes[ng].assign(
                params.layer_vocab_sizes.begin() + (i * n_ngram_orders + ng) * params.n_head_per_ngram,
                params.layer_vocab_sizes.begin() + (i * n_ngram_orders + (ng + 1)) * params.n_head_per_ngram);
        }

        // ---- Calculate MHE offsets ----
        const uint32_t total_heads = n_ngram_orders * params.n_head_per_ngram;
        info.mhe_offsets.resize(total_heads);
        int64_t  running_offset = 0;
        uint32_t head_idx       = 0;
        for (uint32_t ng = 0; ng < n_ngram_orders; ng++) {
            for (uint32_t h = 0; h < params.n_head_per_ngram; h++) {
                info.mhe_offsets[head_idx] = running_offset;
                running_offset += info.vocab_sizes[ng][h];
                head_idx++;
            }
        }

        layer_info[layer_id] = std::move(info);
    }
}

// ---- Hash computation ----

void engram_hash_mapping::compute_hash(const int32_t * input_ids,
                                       int32_t         n_tokens,
                                       int32_t         layer_id,
                                       int32_t *       out_hash) const {
    auto it = layer_info.find(layer_id);
    assert(it != layer_info.end());
    const auto & info = it->second;

    // Compress token ids first
    std::vector<uint32_t> compressed_ids(n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        int32_t tid = input_ids[t];
        if (tid >= 0) {
            compressed_ids[t] = params.compressed_lookup[tid];
        } else {
            compressed_ids[t] = params.pad_id;
        }
    }

    const uint32_t n_ngram_orders = params.max_ngram_size - 1;
    const int64_t  pad            = (int64_t) params.pad_id;

    std::vector<std::vector<int64_t>> shifts(params.max_ngram_size);
    for (uint32_t k = 0; k < params.max_ngram_size; k++) {
        shifts[k].resize(n_tokens);
        for (int32_t t = 0; t < n_tokens; t++) {
            if (t >= (int32_t) k) {
                shifts[k][t] = (int64_t) compressed_ids[t - (int32_t) k];
            } else {
                shifts[k][t] = pad;
            }
        }
    }

    // For each n-gram order and each head, compute hash and apply MHE offset
    uint32_t head_idx = 0;
    for (uint32_t ng = 0; ng < n_ngram_orders; ng++) {
        uint32_t n = ng + 2;

        std::vector<int64_t> mix(n_tokens);
        for (int32_t t = 0; t < n_tokens; t++) {
            int64_t m = shifts[0][t] * info.multipliers[0];
            for (uint32_t k = 1; k < n; k++) {
                m ^= shifts[k][t] * info.multipliers[k];
            }
            mix[t] = m;
        }

        for (uint32_t h = 0; h < params.n_head_per_ngram; h++) {
            int64_t mod_val = info.vocab_sizes[ng][h];
            int64_t offset  = info.mhe_offsets[head_idx];

            for (int32_t t = 0; t < n_tokens; t++) {
                int64_t hash_val = mix[t] % mod_val;
                if (hash_val < 0) {
                    hash_val += mod_val;
                }
                out_hash[t * n_hash_heads() + (int32_t) head_idx] = (int32_t) (hash_val + offset);
            }
            head_idx++;
        }
    }
}

int32_t engram_hash_mapping::n_hash_heads() const {
    return (int32_t) ((params.max_ngram_size - 1) * params.n_head_per_ngram);
}

int32_t engram_hash_mapping::d_per_head() const {
    return (int32_t) (params.n_embed_per_ngram / params.n_head_per_ngram);
}

int32_t engram_hash_mapping::engram_hidden_size() const {
    return n_hash_heads() * d_per_head();
}

bool engram_hash_mapping::has_layer(int32_t layer_id) const {
    return layer_info.find(layer_id) != layer_info.end();
}
