#include "tessera-lk-train-data.h"
#include "tessera-lk-loss.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Parse and validate a line as a usable LK training example. On success moves
// the parsed record into out and returns true. Any schema mismatch, missing
// field, wrong drafted count, or parse error returns false (skip, not error).
static bool parse_usable(const char * line, int block_size, json & out) {
    if (line == nullptr || block_size < 0) return false;
    try {
        json rec = json::parse(line);
        if (rec.value("schema", "") != "llama.tessera.spec.v1") return false;
        if (rec.value("drafted", -1) != block_size)              return false;
        if (!rec.contains("prime_token"))                        return false;
        if (!rec.contains("drafted_tokens") || !rec["drafted_tokens"].is_array()) return false;
        if ((int) rec["drafted_tokens"].size() < block_size)     return false;
        if (!rec.contains("verifier_topk_tokens") || !rec["verifier_topk_tokens"].is_array()) return false;
        if (!rec.contains("verifier_topk_probs")  || !rec["verifier_topk_probs"].is_array())  return false;
        if ((int) rec["verifier_topk_tokens"].size() < block_size + 1) return false;
        if ((int) rec["verifier_topk_probs"].size()  < block_size + 1) return false;
        out = std::move(rec);
        return true;
    } catch (...) {
        return false;
    }
}

int ts_lk_train_line_usable(const char * line, int block_size) {
    json rec;
    return parse_usable(line, block_size, rec) ? 1 : 0;
}

int ts_lk_train_example_from_line(const char * line, int block_size, int n_vocab,
                                  int32_t * out_tokens, float * out_labels) {
    if (out_tokens == nullptr || out_labels == nullptr || n_vocab <= 0) return -1;

    json rec;
    if (!parse_usable(line, block_size, rec)) return 0;

    try {
        // Input tokens: [prime, draft[0..block_size-1]] (length block_size + 1).
        out_tokens[0] = rec["prime_token"].get<int32_t>();
        const auto & dt = rec["drafted_tokens"];
        for (int j = 0; j < block_size; ++j) {
            out_tokens[j + 1] = dt[j].get<int32_t>();
        }

        // Labels: densify verifier_topk[p] into column p, position-major.
        // Position p aligns with model position p: logits[p] predicts the
        // distribution after [prime, draft[0..p-1]], which is exactly the
        // conditioning verifier_topk[p] was collected under.
        const auto & vt = rec["verifier_topk_tokens"];
        const auto & vp = rec["verifier_topk_probs"];
        for (int p = 0; p <= block_size; ++p) {
            const auto & vt_p = vt[p];
            const auto & vp_p = vp[p];
            const int k = (int) std::min(vt_p.size(), vp_p.size());
            std::vector<int32_t> toks(k);
            std::vector<float>   probs(k);
            for (int t = 0; t < k; ++t) {
                toks[t]  = vt_p[t].get<int32_t>();
                probs[t] = vp_p[t].get<float>();
            }
            if (ts_lk_dense_from_topk(toks.data(), probs.data(), k, n_vocab,
                                      out_labels + (size_t) p * n_vocab) != 0) {
                return -1;
            }
        }
    } catch (...) {
        return -1;
    }

    return 1;
}

int ts_lk_train_detect_block_size(const char * traces_path) {
    if (traces_path == nullptr) return -1;
    std::ifstream fin(traces_path);
    if (!fin) return -1;

    std::unordered_map<int, int> hist;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        try {
            json rec = json::parse(line);
            if (rec.value("schema", "") != "llama.tessera.spec.v1") continue;
            const int n_dft = rec.value("drafted", -1);
            if (n_dft > 0) hist[n_dft]++;
        } catch (...) {
            continue;
        }
    }
    if (hist.empty()) return -1;

    // Modal drafted count; tie-break on the smaller block size for determinism.
    int best = -1, best_count = -1;
    for (const auto & kv : hist) {
        if (kv.second > best_count || (kv.second == best_count && kv.first < best)) {
            best       = kv.first;
            best_count = kv.second;
        }
    }
    return best;
}
