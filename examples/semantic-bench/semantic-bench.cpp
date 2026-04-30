// llama-semantic-bench: Semantic Fidelity Benchmark
//
// Measures how well a model preserves semantic meaning in its embedding space,
// using Sentence Textual Similarity (STS) evaluation. Complements perplexity
// by directly measuring distributional overlap between embedding vectors via
// the Bhattacharyya Coefficient — an information-geometric similarity measure.
//
// Methodology reference: Sathyavageeswaran (2026), US Patent 19/287,703;
//                         IJITCE Vol. 13 (2025).
//
// TSV input format: score<TAB>sentence1<TAB>sentence2
//   Compatible with STS-B, STS12-16, SICK-R datasets.
//
// Usage:
//   ./llama-semantic-bench -m model.gguf -f pairs.tsv [--output-file results.csv]
//                          [--score-min 0] [--score-max 5]

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Statistical metrics
// ---------------------------------------------------------------------------

/// Bhattacharyya Coefficient between two embedding vectors treated as
/// unnormalised distributions. Returns 1.0 for identical, 0.0 for orthogonal.
/// Measures distributional overlap — the core fidelity metric.
static double bhattacharyya_coefficient(const float * p, const float * q, int n) {
    double sum_p = 0.0, sum_q = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_p += (double)p[i] * p[i];
        sum_q += (double)q[i] * q[i];
    }
    if (sum_p < 1e-10 || sum_q < 1e-10) return 0.0;
    double bc = 0.0;
    for (int i = 0; i < n; ++i) {
        bc += std::sqrt((p[i] * p[i] / (sum_p + 1e-10)) * (q[i] * q[i] / (sum_q + 1e-10)));
    }
    return bc;
}

static double mean(const std::vector<double> & v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
}

/// Pearson product-moment correlation coefficient.
static double pearson_r(const std::vector<double> & x, const std::vector<double> & y) {
    const int n = (int)x.size();
    if (n < 2) return 0.0;
    double mx = mean(x), my = mean(y);
    double num = 0.0, dx2 = 0.0, dy2 = 0.0;
    for (int i = 0; i < n; ++i) {
        double xi = x[i] - mx, yi = y[i] - my;
        num += xi * yi;
        dx2 += xi * xi;
        dy2 += yi * yi;
    }
    double denom = std::sqrt(dx2 * dy2);
    return (denom < 1e-10) ? 0.0 : num / denom;
}

/// Spearman rank correlation coefficient (uses Pearson on ranks).
static double spearman_rho(std::vector<double> x, std::vector<double> y) {
    const int n = (int)x.size();
    if (n < 2) return 0.0;
    auto rank_of = [&](std::vector<double> & v) {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return v[a] < v[b]; });
        std::vector<double> r(n);
        for (int i = 0; i < n; ++i) r[idx[i]] = (double)(i + 1);
        return r;
    };
    auto rx = rank_of(x);
    auto ry = rank_of(y);
    return pearson_r(rx, ry);
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

struct sts_pair {
    double      label;   // similarity label, normalised to [0, 1]
    std::string sent1;
    std::string sent2;
};

static bool load_tsv(const std::string & path,
                     std::vector<sts_pair> & pairs,
                     double score_min, double score_max) {
    std::ifstream f(path);
    if (!f.is_open()) {
        LOG_ERR("%s: cannot open '%s'\n", __func__, path.c_str());
        return false;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string tok, s1, s2;
        if (!std::getline(ss, tok, '\t')) continue;
        if (!std::getline(ss, s1,  '\t')) continue;
        if (!std::getline(ss, s2)) continue;
        if (s2.empty()) continue;
        try {
            double score = std::stod(tok);
            double range = score_max - score_min;
            double norm  = (range > 1e-10) ? (score - score_min) / range : 0.5;
            pairs.push_back({norm, s1, s2});
        } catch (...) {
            // Skip malformed lines
        }
    }
    return !pairs.empty();
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// Encode a single sentence to a normalised embedding vector.
static std::vector<float> encode(llama_context * ctx,
                                  llama_model   * model,
                                  const std::string & text) {
    const int n_embd = llama_model_n_embd(model);
    std::vector<float> out(n_embd, 0.0f);

    auto tokens = common_tokenize(ctx, text, /*add_special=*/true, /*parse_special=*/false);
    if (tokens.empty()) return out;

    const int seq_id = 0;
    llama_batch batch = llama_batch_init((int32_t)tokens.size(), 0, 1);
    for (int32_t i = 0; i < (int32_t)tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], i, {seq_id}, i == (int32_t)tokens.size() - 1);
    }
    llama_memory_clear(llama_get_memory(ctx), false);
    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("%s: llama_decode failed\n", __func__);
        llama_batch_free(batch);
        return out;
    }

    const float * embd = llama_get_embeddings_seq(ctx, seq_id);
    if (!embd) {
        // Fallback: use last token embedding
        embd = llama_get_embeddings_ith(ctx, batch.n_tokens - 1);
    }
    if (embd) {
        std::copy(embd, embd + n_embd, out.begin());
        common_embd_normalize(out.data(), out.data(), n_embd, /*norm=*/2);
    }
    llama_batch_free(batch);
    return out;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    common_params params;
    params.embedding = true;

    // Parse custom flags before common_params_parse
    std::string tsv_path, out_path;
    double score_min = 0.0, score_max = 5.0;
    std::vector<char *> fargv;
    fargv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if ((a == "-f" || a == "--tsv-file") && i + 1 < argc) {
            tsv_path = argv[++i];
        } else if (a == "--output-file" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "--score-min" && i + 1 < argc) {
            score_min = std::stod(argv[++i]);
        } else if (a == "--score-max" && i + 1 < argc) {
            score_max = std::stod(argv[++i]);
        } else {
            fargv.push_back(argv[i]);
        }
    }
    int fargc = (int)fargv.size();
    if (!common_params_parse(fargc, fargv.data(), params, LLAMA_EXAMPLE_EMBEDDING)) {
        return 1;
    }
    common_init();

    if (tsv_path.empty()) {
        LOG_ERR("Usage: %s -m model.gguf -f pairs.tsv [--output-file results.csv] "
                "[--score-min 0] [--score-max 5]\n", argv[0]);
        return 1;
    }

    // Load model
    auto init_result = common_init_from_params(params);
    llama_model   * model = init_result->model();
    llama_context * ctx   = init_result->context();
    if (!model || !ctx) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }

    // Load dataset
    std::vector<sts_pair> pairs;
    if (!load_tsv(tsv_path, pairs, score_min, score_max)) {
        LOG_ERR("%s: no valid pairs loaded from '%s'\n", __func__, tsv_path.c_str());
        return 1;
    }
    LOG_INF("Loaded %zu sentence pairs from %s\n", pairs.size(), tsv_path.c_str());

    // Open CSV output
    std::FILE * csv_out = nullptr;
    if (!out_path.empty()) {
        csv_out = std::fopen(out_path.c_str(), "w");
        if (!csv_out) {
            LOG_ERR("%s: cannot open output file '%s'\n", __func__, out_path.c_str());
        } else {
            std::fprintf(csv_out, "label,cosine_sim,bhattacharyya_coeff\n");
        }
    }

    // Evaluate pairs
    std::vector<double> labels, cos_sims, bc_scores;
    labels.reserve(pairs.size());
    cos_sims.reserve(pairs.size());
    bc_scores.reserve(pairs.size());

    for (size_t i = 0; i < pairs.size(); ++i) {
        const auto & p = pairs[i];
        auto e1 = encode(ctx, model, p.sent1);
        auto e2 = encode(ctx, model, p.sent2);
        const int n = (int)e1.size();

        double cs = (double)common_embd_similarity_cos(e1.data(), e2.data(), n);
        double bc = bhattacharyya_coefficient(e1.data(), e2.data(), n);

        labels.push_back(p.label);
        cos_sims.push_back(cs);
        bc_scores.push_back(bc);

        if (csv_out) {
            std::fprintf(csv_out, "%.4f,%.4f,%.4f\n", p.label, cs, bc);
        }
        if ((i + 1) % 10 == 0) {
            LOG_INF("  Processed %zu / %zu pairs\n", i + 1, pairs.size());
        }
    }
    if (csv_out) std::fclose(csv_out);

    // Compute aggregate metrics
    double pr     = pearson_r(labels, cos_sims);
    double sr     = spearman_rho(labels, cos_sims);
    double bc_avg = mean(bc_scores);

    LOG_INF("\n=== Semantic Fidelity Results ===\n");
    LOG_INF("Pairs evaluated   : %zu\n", pairs.size());
    LOG_INF("Pearson  r        : %+.4f  (cosine sim vs label)\n", pr);
    LOG_INF("Spearman rho      : %+.4f  (cosine sim vs label)\n", sr);
    LOG_INF("Mean BC score     : %.4f   (1.0=identical, 0.0=orthogonal)\n", bc_avg);
    if (!out_path.empty()) {
        LOG_INF("Per-pair CSV      : %s\n", out_path.c_str());
    }

    return 0;
}
