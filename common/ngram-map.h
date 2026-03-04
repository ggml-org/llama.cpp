#pragma once
//
// common/ngram-map.h: structures used to manage a map from n-grams to a list of m-grams
//
// These structures are used to do a lookup of n-grams followed by m-grams in token history.
//
// There are two algorithms implemented:
// 1. ngram_simple: lookup of n-grams followed by m-grams in token history.
// 2. ngram_map: lookup of n-grams followed by m-grams in token history using a map.
//    The map is a vector of key n-grams, and for each key n-gram there is a list of value m-grams.
//
// ref: https://github.com/ggml-org/llama.cpp/pull/18471
//

#include "llama.h"
#include "common.h"

#include <vector>

// n-gram simple
//

// config of n-gram simple.
struct common_ngram_simple_config {
    uint16_t   size_ngram;      // size of n-grams to lookup in self-mode
    uint16_t   size_mgram;      // size of m-grams to draft in self-mode
};

// Searches for a n-gram in the history and checks whether a draft sequence should be generated.
llama_tokens common_ngram_simple_draft(
        const common_ngram_simple_config & config,
        const llama_tokens & tokens, llama_token sampled);


// n-gram map
//

// maximum number of m-gram values stored for each key n-gram.
#define COMMON_NGRAM_MAX_VALUES 4

// number of entries in the (optional, size 0 to disable) map from ngram-hash to ngram-index.
#define COMMON_NGRAM_HASH_MAP_SIZE 262144

// statistics of a m-gram after a known n-gram
struct common_ngram_map_value {
    size_t   value_idx =  0;  // index of value m-gram in token-history (0 if unused)
    uint16_t value_num =  0;  // number of occurrences of this value m-gram after the key n-gram (0 in an unused values-slot)
    int16_t n_accepted = -1;  // number of accepted tokens at last draft (-1 if unused)
};

// statistics of a n-gram
struct common_ngram_map_key {
    size_t   key_idx;   // index of key n-gram in token-history
    size_t   stat_idx;  // index of last token of stastistics computation (key_num, values)

    uint16_t key_num;   // number of occurrences of this key n-gram in token-history
    common_ngram_map_value values[COMMON_NGRAM_MAX_VALUES]; // some known values after the key
};

// map from n-grams to following m-grams in token-history
struct common_ngram_map {
    uint16_t size_key;   // size of key n-grams
    uint16_t size_value; // size of value m-grams

    bool key_only;       // true if only key n-grams are used, no values.

    std::vector<common_ngram_map_key> keys; // key n-grams which occur several times in token-history
    uint16_t min_hits;   // minimum number of key hits to consider a draft

    bool     show_key_map_stats = false; // true, if statistics of the key_map should be printed.

    common_ngram_map(uint16_t sz_key, uint16_t sz_value, bool only_keys,
                     uint16_t min_hits)
        : size_key(sz_key), size_value(sz_value), key_only(only_keys),
          min_hits(min_hits) {
        key_map.resize(COMMON_NGRAM_HASH_MAP_SIZE); // 2^18 hash entries, 0 entries if key_map shouldn't be used
    }

    // In reasoning chats the previous reasoning block will be removed from context history.
    // A rebuild of the ngram map is needed after that.

    size_t   size_last_begin      = 0; // number of tokens at previous start of generation

    bool     last_draft_created   = false; // true if a draft was created at last call.
    size_t   last_draft_key_idx   = 0; // index of last key used for draft generation (0 = no draft)
    uint16_t last_draft_value_idx = 0; // index of last value used for draft generation.

    size_t   idx_last_check       = 0; // index of last check in context history

    // optional map "hash to ngram-index" for faster lookup of n-grams. map is empty if unused.
    //
    // uint32_t instead of size_t (size of current histories is << UINT32_MAX)
    std::vector<uint32_t> key_map;              // key_map[hash] = index of ngram in context window
    uint32_t              key_map_last_idx = 0; // index of the last ngram added to key_map
};

// Initialize the n-gram map with the given token history.
// map:                the ngram map to initialize.
// tokens:             the token history to base the map on.
void common_ngram_map_begin(
    common_ngram_map & map,
    const llama_tokens & tokens);

// Searches for the n-gram in the history and checks whether a draft sequence should be generated.
// map:                the ngram map to search in.
// inp:                the tokens generated so far.
// sampled:            the token that was just sampled.
// draft:              vector to store the draft tokens, initially empty.
void common_ngram_map_draft(
    common_ngram_map & map,
    const llama_tokens & inp, llama_token sampled,
    llama_tokens & draft);

// Update the statistics of a value after a draft was processed.
void common_ngram_map_accept(common_ngram_map & map, uint16_t n_accepted);
