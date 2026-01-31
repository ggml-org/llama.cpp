#include "common.h"
#include "log.h"
#include "ngram-map.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <sstream>

// Print the values of a sublist of `llama_tokens & inp` to a string in the form [v0, v1, v2, ...].
static std::string common_tokens_to_str(const llama_tokens & inp, size_t start, size_t length) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < length; ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << inp[start + i];
    }
    oss << ']';
    return oss.str();
}


// n-gram simple
//

/**
 * Perform speculative generation using the model's own token history.
 * Searches for a matching pattern in the token history and returns draft tokens.
 *
 * @param state     Current state of this implementation
 * @param tokens    Token history to search in
 * @param sampled   Last sampled token
 * @return Vector of draft tokens, empty if no matching pattern is found
 */
llama_tokens common_ngram_simple_draft(
        common_ngram_simple_state & state,
        const llama_tokens & tokens, llama_token sampled) {

    // Simple implementation of self-speculative decoding without a draft model.
    //
    const size_t cur_len = tokens.size();
    // Only check every check_rate tokens to save compute
    // i.e., perform check if (cur_len - idx_last_check) >= check_rate
    if (state.idx_last_check + state.config.check_rate > cur_len) {
        llama_tokens draft_tokens;
        return draft_tokens;
    }

    size_t n_draft_min = state.config.size_ngram; // size of n-gram to lookup in token history
    size_t n_draft_max = state.config.size_mgram; // the m-gram following the found n-gram is used for draft

    // vector for tokens we want to verify.
    // return empty vector if there is no match.
    llama_tokens draft_tokens;

    // We need at least n_draft_min + n_draft_max + 1 tokens.
    if (cur_len <= static_cast<size_t>(n_draft_min + n_draft_max + 1)) {
        return draft_tokens;
    }

    // pattern search
    llama_tokens pattern;
    pattern.reserve(n_draft_min);
    for (size_t j = cur_len - n_draft_min + 1; j < cur_len; ++j) {
        pattern.push_back(tokens[j]);
    }
    pattern.push_back(sampled); // add the last token to the pattern

    // We do a search in the token history.
    state.idx_last_check = cur_len;

    size_t match_pos = 0; // we ignore position 0, position 0 == no match
                          // search backwards, but skip the current match (we are currently there)
    for (size_t j = cur_len - n_draft_min - 1; j > 0; --j) {
        bool match = true;
        for (size_t k = 0; k < pattern.size(); ++k) {
            if (tokens[j + k] != pattern[k]) {
                match = false;
                break;
            }
        }
        if (match) {
            match_pos = j;
            break;
        }
    }
    if (match_pos == 0) {
        return draft_tokens;
    }

    const size_t copy_max = std::min(
            n_draft_max,
            cur_len - (match_pos + n_draft_min)
            );
    if (copy_max < n_draft_min) {
        return draft_tokens;
    }
    LOG_DBG("%s: #tokens = %zu: found matching pattern at pos %zu, length %zu, draft length %zu\n",
            __func__, cur_len,
            match_pos, pattern.size(), copy_max);

    draft_tokens.reserve(copy_max);
    for (size_t j = 0; j < copy_max; ++j) {
        draft_tokens.push_back(tokens[match_pos + n_draft_min + j]);
    }
    return draft_tokens;
}


// n-gram map
//

// maximum number of counted values of a ngram map value.
#define COMMON_NGRAM_MAX_VALUE_COUNT 16380

void common_ngram_map_draft(common_ngram_map & map,
        const llama_tokens & inp, llama_token sampled,
        llama_tokens & draft) {
    // reset last key and value.
    map.last_draft_created   = false;
    map.last_draft_key_idx   = 0;
    map.last_draft_value_idx = 0;

    const size_t cur_len = inp.size();
    const uint16_t n = map.size_key;
    const uint16_t m = map.size_value;
    if (cur_len < static_cast<size_t>(2 * n + m)) {
        return;
    }

    // Only check every check_rate tokens to save compute
    // i.e., perform check if (cur_len - idx_last_check) >= check_rate
    if (map.idx_last_check + map.check_rate > cur_len) {
        return;
    }
    map.idx_last_check = cur_len;

    // search pattern, the key n-gram
    std::vector<llama_token> key_tokens;
    key_tokens.reserve(n);
    for (size_t j = cur_len - n + 1; j < cur_len; ++j) {
        key_tokens.push_back(inp[j]);
    }
    key_tokens.push_back(sampled);

    // search for the key in the map
    size_t match_pos = 0;
    for (size_t j = cur_len - n - m - 1; j > 0; --j) {
        bool match = true;
        for (size_t k = 0; k < n; ++k) {
            if (inp[j + k] != key_tokens[k]) {
                match = false;
                break;
            }
        }
        if (match) {
           match_pos = j;
           break;
        }
    }
    if (match_pos > 0) {
        LOG_INF("%s: cur_len = %zu, n = %d, m = %d, sz_tkns = %zu, sampled = %d, match_pos = %zu\n", __func__,
            cur_len, n, m, key_tokens.size(), sampled, match_pos);
    }

    if (match_pos == 0) {
        return;
    }

    // We have a match, now we look for the statistics of the key.
    size_t key_offset = map.keys.size(); // offset in the map
    // We iterate through the std::vector<common_ngram_map_key> map->keys.
    for (size_t i = 0; i < map.keys.size(); ++i) {
        bool match = true;
        for (size_t j = 0; j < n; ++j) {
            if (inp[map.keys[i].key_idx + j] != key_tokens[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            key_offset = i;
            break;
        }
    }
    if (key_offset == map.keys.size()) {
        // We create a new key-entry, it will get offset key_offset.
        common_ngram_map_key new_key;
        new_key.key_idx = match_pos;
        new_key.stat_idx = 0;
        new_key.key_num = 0;
        for (int i = 0; i < COMMON_NGRAM_MAX_VALUES; ++i) {
            new_key.values[i].value_num = 0;
            new_key.values[i].n_accepted = m;
        }
        map.keys.push_back(new_key);
    }

    // our key n-gram:
    common_ngram_map_key & curr_key = map.keys[key_offset];

    // update number of key hits
    curr_key.key_num = (uint16_t) std::min((int) map.keys[key_offset].key_num + 1,
            (int) COMMON_NGRAM_MAX_VALUE_COUNT);

    if (map.key_only) {
        // simple mode:
        // Fill in the draft with the m tokens following the key.
        // We work with value values[0] only.
        int n_draft_tokens = std::min((int) m, (int) curr_key.values[0].n_accepted);

        for (int i = 0; i < n_draft_tokens; ++i) {
            draft.push_back(inp[match_pos + n + i]);
        }

        LOG_INF("%s: key_offset = %zu, key_num = %d, draft.size = %zu\n", __func__,
                key_offset, curr_key.key_num, draft.size());

        map.last_draft_created   = false;
        map.last_draft_key_idx   = key_offset;
        map.last_draft_value_idx = 0; // value 0 is used for simple mode
        return;
    }

    if (curr_key.key_num < map.min_hits) {
        // not enough hits to consider this a good draft
        LOG_DBG("%s: key_offset = %zu, key_num = %d, min_hits = %d, no draft\n", __func__,
                key_offset, curr_key.key_num, map.min_hits);
        return;
    }

    // complex mode: examine the different m-grams after this key n-gram.
    //

    // determine all (max COMMON_NGRAM_MAX_VALUES) m-grams after the key n-gram.
    for (size_t i = curr_key.stat_idx; i <= match_pos; ++i) {
        // begins the key n-gram at index i?
        bool match_key = true;
        for (size_t k = 0; k < n; ++k) {
            if (inp[i + k] != key_tokens[k]) {
                match_key = false;
                break;
            }
        }
        if (!match_key) {
            continue;
        }

        // Do we haven a existing value m-gram or a new one after the key at index i?
        size_t idx_begin_value_key = i + n;
        int idx_value = -1;
        for (int v = 0; v < COMMON_NGRAM_MAX_VALUES; ++v) {
            size_t idx_begin_value_v = curr_key.values[v].value_idx;
            if (idx_begin_value_v == 0) {
                // We found an empty value slot => we found a new value m-gram after the key n-gram.
                curr_key.values[v].value_idx = idx_begin_value_key;
                curr_key.values[v].value_num = 0;
                curr_key.values[v].n_accepted = m;
                idx_value = v;
                break;
            }
            bool match = true;
            for (size_t j = 0; j < m; ++j) {
                if (inp[idx_begin_value_key + j] != inp[idx_begin_value_v + j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                // We found an existing value m-gram after the key n-gram.
                idx_value = v;
                break;
            }
        }
        if (idx_value >= 0) {
            // We found a value m-gram of the key n-gram.
            curr_key.values[idx_value].value_num = (uint16_t) std::min((int) curr_key.values[idx_value].value_num + 1,
                    (int) COMMON_NGRAM_MAX_VALUE_COUNT);
        }
    }
    // the statistics are updated up to match_pos.
    curr_key.stat_idx = match_pos;

    // Do we have a value we could use for the draft?
    uint16_t max_occur = 0;
    int slot_max = 0;
    for (int v = 0; v < COMMON_NGRAM_MAX_VALUES; ++v) {
        uint16_t curr_occur = curr_key.values[v].value_num;
        if (curr_occur > max_occur) {
            max_occur = curr_occur;
            slot_max = v;
        }
    }
    // What is sum of the other occurences?
    uint32_t sum_occur = 0;
    for (int v = 0; v < COMMON_NGRAM_MAX_VALUES; ++v) {
        if (v == slot_max) {
            continue;
        }
        uint16_t curr_occur = curr_key.values[v].value_num;
        sum_occur += curr_occur;
    }

    LOG_INF("%s: key_offset = %zu, max_occur = %d, sum_occur = %d, slot_max = %d [%zu/%d, %zu/%d, %zu/%d, %zu/%d]\n", __func__,
            key_offset,
            max_occur, sum_occur, slot_max,
            curr_key.values[0].value_idx, curr_key.values[0].value_num,
            curr_key.values[1].value_idx, curr_key.values[1].value_num,
            curr_key.values[2].value_idx, curr_key.values[2].value_num,
            curr_key.values[3].value_idx, curr_key.values[3].value_num
        );
    // Print the tokens of the four values (if idx != 0), use LOG_INF
    for (int v = 0; v < COMMON_NGRAM_MAX_VALUES; ++v) {
        if (curr_key.values[v].value_idx != 0) {
            LOG_INF("%s: value[%d] = %s\n", __func__, v, common_tokens_to_str(inp, curr_key.values[v].value_idx, m).c_str());
        }
    }

    if (sum_occur > 0 && max_occur < 3 * sum_occur) {
        // The most frequent value is not much more frequent than the other values.
        // We do not use the draft.
        return;
    }

    // We use the most frequent value values[slot_max] for the draft.
    // Fill in the draft with the m tokens following the key.
    int n_draft_tokens = std::min((int) m, (int) curr_key.values[slot_max].n_accepted);

    for (int i = 0; i < n_draft_tokens; ++i) {
        draft.push_back(inp[match_pos + n + i]);
    }

    LOG_INF("%s: key_offset = %zu, slot_max = %d, key_num = %d, draft.size = %zu\n", __func__,
            key_offset, slot_max,
            curr_key.key_num, draft.size());

    map.last_draft_created   = true;
    map.last_draft_key_idx   = key_offset;
    map.last_draft_value_idx = slot_max; // value used for draft generation.
}

void common_ngram_map_accept(common_ngram_map & map, uint16_t n_accepted) {
    if (!map.last_draft_created) {
        return;
    }

    // find the key and its chosen value.
    const size_t key_idx = map.last_draft_key_idx;
    const size_t val_idx = map.last_draft_value_idx;

    // find key corresponding to key_idx.
    common_ngram_map_key & curr_key = map.keys[key_idx];
    // find value corresponding to val_idx.
    struct common_ngram_map_value & curr_value = curr_key.values[val_idx]; // value used for draft generation.

    // update the value statistics
    LOG_INF("common_ngram_map_send_accepted: n_accepted = %d, prev value_num = %d\n",
            n_accepted, curr_value.n_accepted);
    curr_value.n_accepted = n_accepted;
}
