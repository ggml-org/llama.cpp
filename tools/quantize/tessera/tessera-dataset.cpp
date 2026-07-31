#include "tessera-dataset.h"

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
}

int ts_dataset_mode_from_string(const char * s, ts_dataset_mode * out) {
    if (std::strcmp(s, "text")  == 0) { *out = TS_DATASET_MODE_TEXT;  return 0; }
    if (std::strcmp(s, "pairs") == 0) { *out = TS_DATASET_MODE_PAIRS; return 0; }
    if (std::strcmp(s, "lk")    == 0) { *out = TS_DATASET_MODE_LK;    return 0; }
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

// -------------------------------------------------------------------------
// main entry

int ts_dataset_run(const ts_dataset_params * params,
                   int * n_records_out,
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

        // Schema check: only process spec_calib.v2 records.
        const std::string schema = rec.value("schema", "");
        if (schema != "llama.spec_calib.v2") {
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
        }
    }

    if (!fout) {
        if (err_msg) *err_msg = std::string("write failed: ") + params->output_path;
        return -1;
    }
    if (n_records_out) *n_records_out = n_written;
    return 0;
}
