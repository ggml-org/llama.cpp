#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <vector>

struct engram_hash_params {
    uint32_t max_ngram_size       = 0;
    uint32_t n_head_per_ngram     = 0;
    uint32_t n_embed_per_ngram    = 0;
    uint32_t seed                 = 0;
    int32_t  pad_id               = 0;
    uint32_t tokenizer_vocab_size = 0;

    std::vector<uint32_t> compressed_lookup;

    std::vector<uint32_t> engram_vocab_size;

    std::vector<uint32_t> layer_ids;

    // all layer multipliers (flattened)
    std::vector<uint64_t> layer_multipliers;

    // all layer vocab sizes (flattened)
    std::vector<uint64_t> layer_vocab_sizes;
};

struct engram_hash_layer_info {
    std::vector<uint64_t> multipliers;

    std::vector<std::vector<uint64_t>> vocab_sizes;

    std::vector<int64_t> mhe_offsets;
};

class engram_hash_mapping {
  public:
    engram_hash_mapping() = default;

    void init(const engram_hash_params & params);

    void compute_hash(const int32_t * input_ids, int32_t n_tokens, int32_t layer_id, int32_t * out_hash) const;

    // Total number of hash heads per token
    int32_t n_hash_heads() const;

    // D per head
    int32_t d_per_head() const;

    // Total engram hidden size
    int32_t engram_hidden_size() const;

    bool has_layer(int32_t layer_id) const;

    const engram_hash_params & get_params() const { return params; }

  private:
    engram_hash_params                        params;
    std::map<int32_t, engram_hash_layer_info> layer_info;
};
