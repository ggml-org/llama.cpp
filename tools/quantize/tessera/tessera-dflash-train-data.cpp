#include "tessera-dflash-train-data.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Parse and validate a line as a usable DFlash training example. On success
// moves the parsed record into out and returns true. Any schema mismatch,
// missing field, wrong n_dft, or parse error returns false (skip, not error).
static bool parse_usable(const char * line, int block_size, json & out) {
    if (line == nullptr || block_size < 0) return false;
    try {
        json rec = json::parse(line);
        if (rec.value("schema", "") != "llama.tessera.dflash-block.v1") return false;
        const int n_dft = rec.value("n_dft", -1);
        if (n_dft != block_size)                                          return false;
        if (n_dft <= 0)                                                   return false;
        if (!rec.contains("target_tokens")    || !rec["target_tokens"].is_array())    return false;
        if (!rec.contains("acceptance_probs") || !rec["acceptance_probs"].is_array()) return false;
        if (!rec.contains("dpace_weights")    || !rec["dpace_weights"].is_array())    return false;
        if (!rec.contains("decay_weights")    || !rec["decay_weights"].is_array())    return false;
        if ((int) rec["target_tokens"].size()    < n_dft) return false;
        if ((int) rec["acceptance_probs"].size() < n_dft) return false;
        if ((int) rec["dpace_weights"].size()    < n_dft) return false;
        if ((int) rec["decay_weights"].size()    < n_dft) return false;
        out = std::move(rec);
        return true;
    } catch (...) {
        return false;
    }
}

int ts_dflash_train_line_usable(const char * line, int block_size) {
    json rec;
    return parse_usable(line, block_size, rec) ? 1 : 0;
}

int ts_dflash_train_example_from_line(const char * line,
                                      int block_size,
                                      int weight_scheme,
                                      int32_t * out_tokens,
                                      int32_t * out_labels_sparse,
                                      float   * out_weights) {
    if (out_tokens == nullptr || out_labels_sparse == nullptr ||
        out_weights == nullptr || block_size <= 0) {
        return -1;
    }

    json rec;
    if (!parse_usable(line, block_size, rec)) return 0;

    try {
        // Anchor at pos 0: no dataset field, sentinel any in-vocab token.
        // The cross-entropy at pos 0 is gated by weight 0, so the model
        // backprops nothing from it (it is a context token, not a target).
        out_tokens[0]         = 0;
        out_labels_sparse[0]  = 0;
        out_weights[0]        = 0.0f;

        const auto & tgt  = rec["target_tokens"];
        const auto & dpace = rec["dpace_weights"];
        const auto & decay = rec["decay_weights"];

        for (int j = 0; j < block_size; ++j) {
            out_tokens[j + 1]        = tgt[j].get<int32_t>();
            out_labels_sparse[j + 1] = tgt[j].get<int32_t>();
            // weight_scheme 0 = dpace (default), 1 = decay baseline. Both
            // arrays are smoothed+normalized in the dataset prep, so a swap
            // is a data-side change with no graph edit.
            if (weight_scheme == 1) {
                out_weights[j + 1] = static_cast<float>(decay[j].get<double>());
            } else {
                out_weights[j + 1] = static_cast<float>(dpace[j].get<double>());
            }
        }
    } catch (...) {
        return -1;
    }

    return 1;
}

int ts_dflash_train_detect_block_size(const char * dflash_jsonl_path) {
    if (dflash_jsonl_path == nullptr) return -1;
    std::ifstream fin(dflash_jsonl_path);
    if (!fin) return -1;

    std::unordered_map<int, int> hist;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        try {
            json rec = json::parse(line);
            if (rec.value("schema", "") != "llama.tessera.dflash-block.v1") continue;
            const int n_dft = rec.value("n_dft", -1);
            if (n_dft > 0) hist[n_dft]++;
        } catch (...) {
            continue;
        }
    }
    if (hist.empty()) return -1;

    // Modal n_dft; tie-break on the smaller block size for determinism.
    int best = -1, best_count = -1;
    for (const auto & kv : hist) {
        if (kv.second > best_count || (kv.second == best_count && kv.first < best)) {
            best       = kv.first;
            best_count = kv.second;
        }
    }
    return best;
}
