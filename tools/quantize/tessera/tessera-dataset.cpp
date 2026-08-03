#include "tessera-dataset.h"
#include "tessera-dpace.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

void ts_dataset_default_params(ts_dataset_params * p) {
    std::memset(p, 0, sizeof(*p));
    p->mode         = TS_DATASET_MODE_TEXT;
    p->min_accepted = 1;
    p->dpace_alpha  = TS_DPACE_DEFAULT_ALPHA;
    p->dflash_gamma = TS_DFLASH_DEFAULT_GAMMA;
}

int ts_dataset_mode_from_string(const char * s, ts_dataset_mode * out) {
    if (std::strcmp(s, "text")   == 0) { *out = TS_DATASET_MODE_TEXT;   return 0; }
    if (std::strcmp(s, "pairs")  == 0) { *out = TS_DATASET_MODE_PAIRS;  return 0; }
    if (std::strcmp(s, "lk")     == 0) { *out = TS_DATASET_MODE_LK;     return 0; }
    if (std::strcmp(s, "dflash") == 0) { *out = TS_DATASET_MODE_DFLASH; return 0; }
    return -1;
}

// -------------------------------------------------------------------------
// per-line writers

static void write_text_line(std::ofstream & out, const json & rec) {
    if (!rec.contains("accepted_tokens")) return;
    const auto & toks = rec["accepted_tokens"];
    bool first = true;
    for (const auto & t : toks) {
        if (!first) out << ' ';
        out << t.get<int64_t>();
        first = false;
    }
    out << '\n';
}

static void write_pairs_line(std::ofstream & out, const json & rec) {
    json o;
    o["context"]  = rec.value("prime_token", 0);
    o["drafted"]  = rec.value("drafted_tokens", json::array());
    o["accepted"] = rec.value("accepted_tokens", json::array());
    o["n_acc"]    = rec.value("accepted", 0);
    o["n_dft"]    = rec.value("drafted", 0);
    out << o.dump() << '\n';
}

static void write_lk_lines(std::ofstream & out, const json & rec) {
    // One record per position in the spec step. Position i uses
    // verifier_topk_*[i] and drafter_topk_*[i].
    if (!rec.contains("verifier_topk_tokens") || !rec.contains("drafter_topk_tokens")) return;

    const auto & vt = rec["verifier_topk_tokens"];
    const auto & vp = rec["verifier_topk_probs"];
    const auto & dt = rec["drafter_topk_tokens"];
    const auto & dp = rec["drafter_topk_probs"];
    const auto & accepted_tokens = rec.value("accepted_tokens", json::array());
    const auto & drafted_tokens  = rec.value("drafted_tokens", json::array());

    const int n_pos = (int)std::min(vt.size(), dt.size());
    for (int i = 0; i < n_pos; ++i) {
        json o;
        o["position"]  = i;
        o["p_tokens"]  = vt[i];
        o["p_probs"]   = (i < (int)vp.size()) ? vp[i] : json::array();
        o["q_tokens"]  = dt[i];
        o["q_probs"]   = (i < (int)dp.size()) ? dp[i] : json::array();
        // Position i is "accepted" if the drafter's pick at i matched
        // the verifier's (i.e. drafted_tokens[i] == accepted_tokens[i]).
        bool acc = (i < (int)drafted_tokens.size() &&
                    i < (int)accepted_tokens.size() &&
                    drafted_tokens[i] == accepted_tokens[i]);
        o["accepted"] = acc;
        out << o.dump() << '\n';
    }
}

// One block-structured record per spec step for DFlash/D-PACE training.
// The block is the n_dft drafted positions. Target token at position j is
// verifier_argmax[j] (ground truth the block drafter should emit); the
// acceptance proxy is confidence[j]. D-PACE weights are smoothed+normalized
// and baked in so the training driver can pre-weight a standard CE label.
// Returns true if a record was written.
static bool write_dflash_line(std::ofstream & out, const json & rec,
                              float alpha, float gamma) {
    if (!rec.contains("confidence") || !rec["confidence"].is_array()) return false;
    if (!rec.contains("verifier_argmax") || !rec["verifier_argmax"].is_array()) return false;

    const int block_size = rec.value("drafted", 0);
    if (block_size <= 0) return false;

    const auto & conf  = rec["confidence"];
    const auto & varg  = rec["verifier_argmax"];
    if ((int)conf.size() < block_size || (int)varg.size() < block_size) return false;

    std::vector<float> acc(block_size);
    for (int j = 0; j < block_size; ++j) {
        acc[j] = conf[j].get<float>();
    }

    std::vector<double> dpace_w(block_size);
    ts_dpace_weights_smoothed(acc.data(), block_size, alpha, dpace_w.data());
    ts_dpace_normalize_weights(dpace_w.data(), block_size);

    std::vector<double> decay_w(block_size);
    ts_dflash_decay_weights(block_size, gamma, decay_w.data());
    ts_dpace_normalize_weights(decay_w.data(), block_size);

    const double surrogate = ts_dpace_accepted_length_surrogate(acc.data(), block_size);

    json o;
    o["schema"]       = "llama.tessera.dflash-block.v1";
    o["block_size"]   = block_size;
    o["target_tokens"]    = json::array();
    o["acceptance_probs"] = json::array();
    o["dpace_weights"]    = json::array();
    o["decay_weights"]    = json::array();
    for (int j = 0; j < block_size; ++j) {
        o["target_tokens"].push_back(varg[j].get<int64_t>());
        o["acceptance_probs"].push_back(acc[j]);
        o["dpace_weights"].push_back(dpace_w[j]);
        o["decay_weights"].push_back(decay_w[j]);
    }
    o["n_acc"]     = rec.value("accepted", 0);
    o["n_dft"]     = block_size;
    o["surrogate"] = surrogate;
    out << o.dump() << '\n';
    return true;
}

// -------------------------------------------------------------------------
// main entry

int ts_dataset_run(const ts_dataset_params * params,
                   int * n_records_out,
                   int * n_skipped_out,
                   std::string * err_msg) {
    std::ifstream fin(params->input_path);
    if (!fin) {
        if (err_msg) *err_msg = std::string("cannot open input: ") + params->input_path;
        return -1;
    }
    std::ofstream fout(params->output_path, std::ios::binary);
    if (!fout) {
        if (err_msg) *err_msg = std::string("cannot open output: ") + params->output_path;
        return -1;
    }

    int n_written = 0;
    int n_skipped = 0;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        json rec;
        try {
            rec = json::parse(line);
        } catch (...) {
            n_skipped++;
            continue;
        }

        // Schema check: only process llama.tessera.spec.v1 records.
        const std::string schema = rec.value("schema", "");
        if (schema != "llama.tessera.spec.v1") {
            n_skipped++;
            continue;
        }

        // Minimum accepted filter.
        const int n_acc = rec.value("accepted", 0);
        if (n_acc < params->min_accepted) {
            n_skipped++;
            continue;
        }

        switch (params->mode) {
            case TS_DATASET_MODE_TEXT:  write_text_line(fout, rec);  n_written++; break;
            case TS_DATASET_MODE_PAIRS: write_pairs_line(fout, rec); n_written++; break;
            case TS_DATASET_MODE_LK:    write_lk_lines(fout, rec);   n_written++; break;
            case TS_DATASET_MODE_DFLASH:
                if (write_dflash_line(fout, rec, params->dpace_alpha, params->dflash_gamma)) {
                    n_written++;
                } else {
                    n_skipped++;
                }
                break;
        }
    }

    if (!fout) {
        if (err_msg) *err_msg = std::string("write failed: ") + params->output_path;
        return -1;
    }
    if (n_records_out)  *n_records_out  = n_written;
    if (n_skipped_out)  *n_skipped_out  = n_skipped;
    return 0;
}
