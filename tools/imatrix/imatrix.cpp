#include "arg.h"
#include "common.h"
#include "gguf.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <regex>
#include <thread>
#include <unordered_map>
#include <valarray>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s \\\n"
            "       -m model.gguf -f some-text.txt [-o imatrix.gguf] [--output-format {gguf,dat}] [--no-ppl] \\\n"
            "       [--process-output] [--chunk 123] [--save-frequency 0] [--output-frequency 10] \\\n"
            "       [--in-file imatrix-prev-0.gguf --in-file imatrix-prev-1.gguf ...] [--parse-special] \\\n"
            "       [--output-format gguf|dat] [--show-statistics] [...]\n" , argv[0]);
    LOG("\n");
}

static const char * const LLM_KV_IMATRIX_DATASETS    = "imatrix.datasets";
static const char * const LLM_KV_IMATRIX_CHUNK_COUNT = "imatrix.chunk_count";
static const char * const LLM_KV_IMATRIX_CHUNK_SIZE  = "imatrix.chunk_size";

struct Stats {
    std::vector<float>   activations;
    std::vector<float>   values;
    std::vector<int64_t> counts;
};

struct tensor_statistics {
    std::string tensor;
    Stats stats;
    float sum_val = 0.0f;
    float mean_val = 0.0f;
    int64_t elements = 0;
    float std_deviation = 0.0f;
    float skewness = 0.0f;
    float kurtosis = 0.0f;
    float gain = std::numeric_limits<float>::quiet_NaN();
    float entropy = 0.0f;
    float l2_dist = std::numeric_limits<float>::quiet_NaN();
    float cossim = std::numeric_limits<float>::quiet_NaN();
    float pearson = std::numeric_limits<float>::quiet_NaN();
    float covariance = std::numeric_limits<float>::quiet_NaN();
    double cov_sum = 0.0;
    double var_c_sum = 0.0;
    double var_p_sum = 0.0;
    double dot_prod = 0.0;
    double norm1_sq = 0.0;
    double norm2_sq = 0.0;
    double l2_dist_sq = 0.0;
};

class IMatrixCollector {
public:
    IMatrixCollector() = default;
    void set_params(common_params params) { m_params = std::move(params); }
    bool collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data);
    void save_imatrix_legacy(int32_t ncall = -1) const;
    void save_imatrix(int32_t n_chunk = -1) const;
    bool load_imatrix_legacy(const char * fname);
    bool load_imatrix(const char * file_name);
    const std::unordered_map<std::string, Stats> & get_mstats() const { return m_stats; }
private:
    std::unordered_map<std::string, Stats> m_stats;
    common_params                          m_params;
    std::mutex                             m_mutex;
    std::vector<std::string>               m_datasets;
    int32_t                                m_last_chunk = 0;
    std::vector<char>                      m_src1_data;
    std::vector<char>                      m_ids; // the expert ids from ggml_mul_mat_id
};

// remove any prefix and suffixes from the name
// CUDA0#blk.0.attn_k.weight#0 => blk.0.attn_k.weight
static std::string filter_tensor_name(const char * name) {
    std::string wname;
    const char * p = strchr(name, '#');
    if (p != NULL) {
        p = p + 1;
        const char * q = strchr(p, '#');
        if (q != NULL) {
            wname = std::string(p, q - p);
        } else {
            wname = p;
        }
    } else {
        wname = name;
    }
    return wname;
}

static void process_tensor_name(const std::string & input, std::string & layer, std::string & tensor) {
    layer.clear();
    tensor.clear();

    std::vector<std::string> name;
    std::istringstream stream(input);
    std::string item;

    while (std::getline(stream, item, '.')) { name.push_back(item); }
    for (size_t i = 0; i < name.size(); ++i) {
        if (name[i] == "blk" && i + 1 < name.size()) {
            layer = name[i + 1];
            break;
        }
    }
    for (size_t i = 0; i < name.size(); ++i) {
        if (name[i] == "weight" && i > 0) {
            for (size_t j = 0; j < name.size(); ++j) {
                if (name[j] == "blk") {
                    j+=2;
                    continue; }
                if (j == i) { break; }
                if (!tensor.empty()) { tensor += "."; }
                tensor += name[j];
            }
            break;
        }
    }

    if (tensor.empty()) { tensor = input; }
    if (layer.empty()) { layer = "-"; }
}

static std::vector<float> compute_tensor_averages(const Stats & tstats) {
    if (tstats.counts.empty()) { return {}; }

    const size_t n_mat = tstats.counts.size();
    const size_t len = !tstats.activations.empty() ? tstats.activations.size() : tstats.values.size();
    if (len == 0 || n_mat == 0 || len % n_mat != 0) { return {}; }

    const size_t row = len / n_mat;
    std::vector<float> vec;
    vec.resize(len);

    bool has_valid = false;
    const bool use_activations = !tstats.activations.empty();

    for (size_t m = 0; m < n_mat; ++m) {
        const auto c = (float) tstats.counts[m];
        const size_t off = m * row;

        if (c <= 0.0f) { continue; }

        has_valid = true;
        const float scale = 1.0f / c;
        const float * src = use_activations ? &tstats.activations[off] : &tstats.values[off];
        float * dst = & vec[off];

        for (size_t j = 0; j < row; ++j) { dst[j] = src[j] * scale; }
    }

    if (!has_valid) { return {}; }
    return vec;
}

static bool compute_vector_statistics(std::vector<tensor_statistics> & tstats, const std::string & name, const Stats & e, bool & legacy) {
    constexpr auto fnan = std::numeric_limits<float>::quiet_NaN();
    legacy = e.activations.empty();
    const size_t n_mat = e.counts.size();
    const size_t len = legacy ? e.values.size() : e.activations.size();

    if (n_mat == 0 || len == 0 || len % n_mat != 0) {
        LOG_ERR("%s: data size mismatch or empty for tensor %s\n", __func__, name.c_str());
        return false;
    }
    if (!legacy && e.values.size() != len) {
        LOG_ERR("%s: activations/values size mismatch for %s\n", __func__, name.c_str());
        return false;
    }

    const size_t row_size = len / n_mat;
    double sum = 0.0;
    double mean = 0.0;
    double sum_sq_diff = 0.0;
    double sum_cu_diff = 0.0;
    double sum_qd_diff = 0.0;
    double sum_energy = 0.0;
    size_t valid_n = 0;

    // Mean
    for (size_t i = 0; i < n_mat; ++i) {
        const auto c = (float)e.counts[i];
        if (c <= 0.0f) { continue; }

        const double inv_c = 1.0 / (double)c;
        const size_t off = i * row_size;

        for (size_t j = 0; j < row_size; ++j) {
            const double v_act = legacy ? 0.0 : (double)e.activations[off + j] * inv_c;
            const double v_val = (double)e.values[off + j] * inv_c;
            const double v = legacy ? v_val : v_act; // Use activation average for non-legacy

            sum += v_val;
            valid_n++;
            const double delta = v - mean;
            mean += delta / (double)valid_n;

            if (v_val > 0.0) { sum_energy += v_val; }
        }
    }

    if (valid_n == 0) { return false; }

    float std_deviation = 0.0f;
    float entropy = 0.0f;

    // Std Dev, Skew, Kurtosis, Entropy
    const double inv_sum_energy = sum_energy > 0.0 ? 1.0 / sum_energy : 0.0;
    const double log2_inv = 1.0 / std::log(2.0);

    for (size_t i = 0; i < n_mat; ++i) {
        const auto c = (float)e.counts[i];
        if (c <= 0.0f) { continue; }
        const double inv_c = 1.0 / (double)c;
        const size_t off = i * row_size;

        for (size_t j = 0; j < row_size; ++j) {
            const double v_act = legacy ? 0.0 : (double)e.activations[off + j] * inv_c;
            const double v_val = (double)e.values[off + j] * inv_c;
            const double v = legacy ? v_val : v_act;
            const double diff = v - mean;

            sum_sq_diff += diff * diff;
            sum_cu_diff += diff * diff * diff;
            sum_qd_diff += diff * diff * diff * diff;

            // Entropy (Distribution of Energy)
            if (inv_sum_energy > 0.0) {
                const double v_energy = (double)e.values[off + j] * inv_c;
                const double p = std::max(0.0, v_energy) * inv_sum_energy;
                if (p > 1e-10) { entropy -= (float)(p * std::log(p) * log2_inv); }
            }
        }
    }

    const double variance = valid_n > 1 ? sum_sq_diff / (double)valid_n : 0.0;
    std_deviation = std::sqrt((float)std::max(variance, 0.0));
    float skewness = 0.0f;
    float kurtosis = 0.0f;
    if (std_deviation > 1e-10f) {
        const double m2 = sum_sq_diff / (double)valid_n;
        skewness = (float)(sum_cu_diff / (double)valid_n / (m2 * std::sqrt(m2)));
        kurtosis = (float)(sum_qd_diff / (double)valid_n / (m2 * m2) - 3.0);
    }

    auto & ts = tstats.emplace_back();
    ts.tensor = name;
    ts.stats = e;
    ts.sum_val = (float)sum;
    ts.mean_val = (float)mean;
    ts.elements = (int64_t)valid_n;
    ts.std_deviation = std_deviation;
    ts.skewness = skewness;
    ts.kurtosis = kurtosis;
    ts.gain = fnan;
    ts.entropy = std::abs(entropy);
    ts.l2_dist = fnan;
    ts.cossim = fnan;
    ts.pearson = fnan;
    ts.covariance = fnan;

    return true;
}

static void compute_tensor_statistics(std::vector<tensor_statistics> & tstats) {
    constexpr auto fnan = std::numeric_limits<float>::quiet_NaN();
    std::unordered_map<std::string, size_t> tensor_map;
    tensor_map.reserve(tstats.size());
    for (size_t i = 0; i < tstats.size(); ++i) { tensor_map[tstats[i].tensor] = i; }

    for (auto & ts : tstats) {
        std::string layer_str;
        std::string dummy_tensor;
        process_tensor_name(ts.tensor, layer_str, dummy_tensor);

        int blk = -1;
        try { blk = std::stoi(layer_str); } catch (...) { continue; }
        if (blk <= 0) { continue; }

        const size_t blk_start_pos = ts.tensor.find("blk." + layer_str);
        if (blk_start_pos == std::string::npos) { continue; }

        std::string tname = ts.tensor;
        tname.replace(blk_start_pos, layer_str.length() + 4, "blk." + std::to_string(blk - 1));

        auto it = tensor_map.find(tname);
        if (it == tensor_map.end()) {
            LOG_WRN("%s: missing previous-layer tensor '%s'\n", __func__, tname.c_str());
            continue;
        }

        const auto & prev_ts = tstats[it->second];
        const auto curr_avg = compute_tensor_averages(ts.stats);
        const auto prev_avg = compute_tensor_averages(prev_ts.stats);

        if (curr_avg.empty() || curr_avg.size() != prev_avg.size()) { continue; }

        double dot_prod = 0.0;
        double norm1_sq = 0.0;
        double norm2_sq = 0.0;
        double l2_dist_sq = 0.0;
        double sum_c = 0.0;
        double sum_p = 0.0;
        const size_t n = curr_avg.size();

        // Sums for Means
        for (size_t i = 0; i < n; ++i) {
            sum_c += curr_avg[i];
            sum_p += prev_avg[i];
        }
        const double mean_c = sum_c / n;
        const double mean_p = sum_p / n;

        double cov_sum = 0.0;
        double var_c_sum = 0.0;
        double var_p_sum = 0.0;

        // Metrics
        for (size_t i = 0; i < n; ++i) {
            const double c_val = curr_avg[i];
            const double p_val = prev_avg[i];

            // Cosine Similarity & L2 Distance
            dot_prod += c_val * p_val;
            norm1_sq += c_val * c_val;
            norm2_sq += p_val * p_val;
            const double diff = c_val - p_val;
            l2_dist_sq += diff * diff;

            // Pearson (Centered stats)
            const double dc = c_val - mean_c;
            const double dp = p_val - mean_p;
            cov_sum += dc * dp;
            var_c_sum += dc * dc;
            var_p_sum += dp * dp;
        }

        ts.dot_prod = dot_prod;
        ts.norm1_sq = norm1_sq;
        ts.norm2_sq = norm2_sq;
        ts.cov_sum = cov_sum;
        ts.var_c_sum = var_c_sum;
        ts.var_p_sum = var_p_sum;
        ts.l2_dist_sq = l2_dist_sq;
        ts.l2_dist = (float)std::sqrt(l2_dist_sq);

        if (n > 1) {
            ts.covariance = (float)(cov_sum / (double)(n - 1));
        }

        if (norm1_sq > 1e-12 && norm2_sq > 1e-12) {
            ts.cossim = (float)(dot_prod / (std::sqrt(norm1_sq) * std::sqrt(norm2_sq)));
            ts.cossim = std::clamp(ts.cossim, -1.0f, 1.0f);
        } else {
            ts.cossim = (norm1_sq == 0.0 && norm2_sq == 0.0) ? fnan : 0.0f;
        }

        if (var_c_sum > 1e-12 && var_p_sum > 1e-12) {
            ts.pearson = (float)(cov_sum / (std::sqrt(var_c_sum) * std::sqrt(var_p_sum)));
            ts.pearson = std::clamp(ts.pearson, -1.0f, 1.0f);
        } else {
            ts.pearson = (var_c_sum == 0.0 && var_p_sum == 0.0) ? fnan : 0.0f;
        }

        if (prev_ts.sum_val > 1e-10f) {
            ts.gain = std::sqrt(ts.sum_val) / std::sqrt(prev_ts.sum_val);
        } else {
            ts.gain = ts.sum_val <= 1e-10f ? 1.0f : fnan;
        }
    }
}

static void compute_layer_statistics(const std::vector<tensor_statistics> & tstats,
                                              std::map<int, float> & layer_cossim,
                                              std::map<int, float> & layer_l2_dist,
                                              std::map<int, float> & layer_pearson,
                                              std::map<int, float> & layer_covariance,
                                              std::map<int, float> & layer_gain) {
    struct layer_aggregation {
        double sum_dot_prod = 0.0;
        double sum_norm1_sq = 0.0;
        double sum_norm2_sq = 0.0;
        double sum_l2_dist_sq = 0.0;
        double sum_cov = 0.0;
        double sum_var_c = 0.0;
        double sum_var_p = 0.0;
        double sum_covariance_n = 0.0;
        double sum_total_energy_curr = 0.0;
        double sum_total_energy_prev = 0.0;
        int n_tensors = 0;
    };

    constexpr auto fnan = std::numeric_limits<float>::quiet_NaN();
    std::map<int, layer_aggregation> laggr;

    for (const auto & ts : tstats) {
        std::string layer_str;
        std::string dummy;
        process_tensor_name(ts.tensor, layer_str, dummy);
        int blk = -1;
        try {
            blk = std::stoi(layer_str);
        } catch(...) {
            if (layer_str == "-") { blk = -1; }
        }

        if (blk <= 0) { continue; }

        if (ts.norm1_sq == 0.0 && ts.norm2_sq == 0.0 && ts.l2_dist_sq == 0.0) { continue; }
        auto & entry = laggr[blk];
        entry.sum_dot_prod += ts.dot_prod;
        entry.sum_norm1_sq += ts.norm1_sq;
        entry.sum_norm2_sq += ts.norm2_sq;
        entry.sum_l2_dist_sq += ts.l2_dist_sq;
        entry.sum_cov += ts.cov_sum;
        entry.sum_var_c += ts.var_c_sum;
        entry.sum_var_p += ts.var_p_sum;
        entry.sum_covariance_n += ts.cov_sum;
        entry.n_tensors++;

        // Accumulate Energy for correct Layer Gain calculation
        entry.sum_total_energy_curr += ts.sum_val;
        if (std::isfinite(ts.gain) && ts.gain > 0.0f) {
            entry.sum_total_energy_prev += ts.sum_val / (ts.gain * ts.gain);
        }
    }

    for (const auto & [layer, agg] : laggr) {
        if (agg.n_tensors == 0) { continue; }

        float cossim = 0.0f;
        if (agg.sum_norm1_sq > 0.0 && agg.sum_norm2_sq > 0.0) {
            cossim = (float)(agg.sum_dot_prod / (std::sqrt(agg.sum_norm1_sq) * std::sqrt(agg.sum_norm2_sq)));
            cossim = std::clamp(cossim, -1.0f, 1.0f);
        } else if (agg.sum_norm1_sq == 0.0 && agg.sum_norm2_sq == 0.0) {
            cossim = fnan;
        }

        float gain = 0.0f;
        if (agg.sum_total_energy_prev > 0.0) {
            gain = (float)(std::sqrt(agg.sum_total_energy_curr) / std::sqrt(agg.sum_total_energy_prev));
        } else {
            gain = fnan;
        }

        if (agg.sum_var_c > 0.0 && agg.sum_var_p > 0.0) {
            auto pearson = (float)(agg.sum_cov / (std::sqrt(agg.sum_var_c) * std::sqrt(agg.sum_var_p)));
            layer_pearson[layer] = std::clamp(pearson, -1.0f, 1.0f);
        } else if (agg.sum_var_c == 0.0 && agg.sum_var_p == 0.0) {
            layer_pearson[layer] = fnan;
        } else {
            layer_pearson[layer] = 0.0f;
        }

        layer_cossim[layer] = cossim;
        layer_l2_dist[layer] = (float)std::sqrt(agg.sum_l2_dist_sq);
        layer_covariance[layer] = agg.n_tensors > 0 ? (float)(agg.sum_covariance_n / agg.n_tensors) : fnan;
        layer_gain[layer] = gain;
    }
}

bool IMatrixCollector::collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];
    std::string wname = filter_tensor_name(src0->name);

    const int32_t chunk_size = m_params.n_ctx / m_params.n_parallel;

    // when ask is true, the scheduler wants to know if we are interested in data from this tensor
    // if we return true, a follow-up call will be made with ask=false in which we can do the actual collection
    if (ask) {
        if (t->op == GGML_OP_MUL_MAT_ID) return true; // collect all indirect matrix multiplications
        if (t->op != GGML_OP_MUL_MAT) return false;
        // why are small batches ignored (<16 tokens)?
        if (src1->ne[1] < 16 || src1->type != GGML_TYPE_F32) return false;
        if (!(wname.substr(0, 4) == "blk." || (m_params.process_output && wname == "output.weight"))) return false;
        return true;
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(src1->buffer);

    if (!is_host) {
        const size_t src1_nbytes = ggml_nbytes(src1);
        m_src1_data.resize(src1_nbytes);
        ggml_backend_tensor_get(src1, m_src1_data.data(), 0, src1_nbytes);
    }

    const char * data = is_host ? (const char *) src1->data : m_src1_data.data();
    GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));

    // this has been adapted to the new format of storing merged experts in a single 3d tensor
    // ref: https://github.com/ggml-org/llama.cpp/pull/6387
    if (t->op == GGML_OP_MUL_MAT_ID) {
        //   ids  -> [n_experts_used, n_tokens]
        //   src1 -> [cols, n_expert_used, n_tokens]
        const ggml_tensor * ids = t->src[2];
        const int64_t n_as = src0->ne[2];
        const int64_t n_ids = ids->ne[0];

        // the top-k selected expert ids are stored in the ids tensor
        // for simplicity, always copy ids to host, because it is small
        // take into account that ids is not contiguous!

        GGML_ASSERT(ids->ne[1] == src1->ne[2]);

        // the extra dimension would need to be stored somewhere to be reflected in the imatrix file
        if (ggml_nrows(src1) != src1->ne[1] * src1->ne[2]) {
            LOG_ERR("%s: tensor has more than 3 dimensions: %s", __func__, wname.c_str());
            GGML_ASSERT(false);
        }

        m_ids.resize(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, m_ids.data(), 0, ggml_nbytes(ids));

        auto & e = m_stats[wname];

        if (e.counts.size() == 1 && n_as > 1) {
            // broadcast, when loading an old imatrix
            e.counts.resize(n_as, e.counts[0]);
        }
        if (e.values.empty()) {
            e.activations.resize(src1->ne[0]*n_as, 0);
            e.values.resize(src1->ne[0]*n_as, 0);
            e.counts.resize(n_as, 0);
        }
        else if (e.values.size() != (size_t)src1->ne[0]*n_as) {
            LOG_ERR("%s: inconsistent size for %s (%d vs %d)\n", __func__, wname.c_str(), (int)e.values.size(), (int)(src1->ne[0]*n_as));
            exit(1); //GGML_ABORT("fatal error");
        }
        else if (e.counts.size() != (size_t)n_as) {
            LOG_ERR("%s: inconsistent expert count for %s (%d vs %d)\n", __func__, wname.c_str(), (int)e.counts.size(), (int)n_as);
            exit(1); //GGML_ABORT("fatal error");
        }
        LOG_DBGV(2, "%s[%d]: %32s, %s, %5d x %5d, %d\n", __func__, m_last_chunk, wname.c_str(), ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[2], (int)src1->type);
        // loop over all possible experts, regardless if they are used or not in the batch
        for (int64_t ex = 0; ex < n_as; ++ex) {
            size_t e_start = ex*src1->ne[0];

            for (int64_t idx = 0; idx < n_ids; ++idx) {
                for (int64_t row = 0; row < src1->ne[2]; ++row) {
                    const int excur = *(const int32_t *) (m_ids.data() + row*ids->nb[1] + idx*ids->nb[0]);

                    GGML_ASSERT(excur >= 0 && excur < n_as); // sanity check

                    if (excur != ex) continue;

                    const int64_t i11 = idx % src1->ne[1];
                    const int64_t i12 = row;
                    const float * x = (const float *)(data + i11*src1->nb[1] + i12*src1->nb[2]);

                    e.counts[ex]++;

                    for (int64_t j = 0; j < src1->ne[0]; ++j) {
                        e.activations[e_start + j] += x[j];
                        e.values[e_start + j] += x[j] * x[j];
                        if (!std::isfinite((float)e.values[e_start + j])) {
                            LOG_ERR("%f detected in %s\n", (float)e.values[e_start + j], wname.c_str());
                            exit(1);
                        }
                    }
                }
            }
            const int32_t n_chunk = e.counts[ex] / chunk_size;
            if (n_chunk > m_last_chunk) {
                const int32_t chunk_step = n_chunk - m_last_chunk;
                m_last_chunk = n_chunk;
                if ((m_last_chunk % m_params.n_out_freq) / chunk_step == 0) {
                    save_imatrix();
                }
                if (m_params.n_save_freq > 0 && (m_last_chunk % m_params.n_save_freq) / chunk_step == 0) {
                    save_imatrix(m_last_chunk);
                }
            }
        }
    } else {
        auto & e = m_stats[wname];
        const int64_t n_mat = src0->ne[2] * src0->ne[3];

        // use a single count per dense tensor
        // (necessary when merging older GGUF-imatrix files with 3d tensors)
        if (e.counts.size() > 1) {
            bool all_equal = true;
            for (size_t i = 1; i < e.counts.size(); ++i) {
                if (e.counts[0] != e.counts[i]) {
                    all_equal = false;
                    break;
                }
            }
            if (all_equal) {
                e.counts.resize(1);
            }
        }
        if (e.values.empty()) {
            e.activations.resize(src1->ne[0] * n_mat, 0);
            e.values.resize(src1->ne[0] * n_mat, 0);
            e.counts.resize(1, 0);
        }
        else if (e.values.size() != (size_t)(src1->ne[0] * n_mat)) {
            LOG_ERR("%s: inconsistent size for %s (%d vs %d)\n", __func__, wname.c_str(), (int)e.values.size(), (int)(src1->ne[0] * n_mat));
            exit(1); //GGML_ABORT("fatal error");
        }
        LOG_DBGV(2, "%s[%d]: %32s, %s, %5d x %5d x %5d, %d\n", __func__, m_last_chunk, wname.c_str(), ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[1], (int)src1->ne[2], (int)src1->type);

        for (int64_t i3 = 0; i3 < src1->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < src1->ne[2]; ++i2) {
                // handle 3D+ tensors, but flatten 3D+ activations when model tensor is 2D
                const int64_t mat_id = (i3 % src0->ne[3]) * src0->ne[2] + (i2 % src0->ne[2]);
                const int64_t mat_start = mat_id * src1->ne[0];

                for (int64_t row = 0; row < src1->ne[1]; ++row) {
                    const float * x = (const float *) (data + row * src1->nb[1] + i2 * src1->nb[2] + i3 * src1->nb[3]);
                    for (int64_t j = 0; j < src1->ne[0]; ++j) {
                        e.activations[mat_start + j] += x[j];
                        e.values[mat_start + j] += x[j] * x[j];
                        if (!std::isfinite((float)e.values[j])) {
                            LOG_ERR("%f detected in %s\n", (float)e.values[j], wname.c_str());
                            exit(1);
                        }
                    }
                }
            }
        }
        // only 1 count in practice, except when a tensor is used for both MUL_MAT_ID and MUL_MAT
        for (size_t i = 0; i < e.counts.size(); ++i) {
            e.counts[i] += ggml_nrows(src1) / n_mat;
            const int32_t n_chunk = e.counts[i] / chunk_size;
            if (n_chunk > m_last_chunk) {
                const int32_t chunk_step = n_chunk - m_last_chunk;
                m_last_chunk = n_chunk;
                if ((m_last_chunk % m_params.n_out_freq) / chunk_step == 0) {
                    save_imatrix();
                }
                if (m_params.n_save_freq > 0 && (m_last_chunk % m_params.n_save_freq) / chunk_step == 0) {
                    save_imatrix(m_last_chunk);
                }
            }
        }
    }

    return true;
}

void IMatrixCollector::save_imatrix_legacy(int32_t ncall) const {
    auto fname = m_params.out_file;

    if (ncall > 0) {
        fname += ".at_";
        fname += std::to_string(ncall);
    }

    // warn when writing imatrix entries that do not have full data
    // this can happen with MoE models where some of the experts end up not being exercised by the provided training data

    int n_entries = 0;
    std::vector<std::string> to_store;

    bool is_first = true; // for printing
    for (const auto & kv : m_stats) {
        const int n_all = kv.second.counts.size();

        if (n_all == 0) {
            continue;
        }

        int n_zeros = 0;
        for (const int c : kv.second.counts) {
            if (c == 0) {
                n_zeros++;
            }
        }

        if (n_zeros != 0 && is_first) {
            LOG_INF("\n");
            is_first = false;
        }

        if (n_zeros == n_all) {
            LOG_WRN("%s: entry '%40s' has no data - skipping\n", __func__, kv.first.c_str());
            continue;
        }

        if (n_zeros > 0) {
            LOG_WRN("%s: entry '%40s' has partial data (%.2f%%)\n", __func__, kv.first.c_str(), 100.0f * (n_all - n_zeros) / n_all);
        }

        n_entries++;
        to_store.push_back(kv.first);
    }

    if (to_store.size() < m_stats.size()) {
        LOG_WRN("%s: storing only %zu out of %zu entries\n", __func__, to_store.size(), m_stats.size());
    }

    // deterministic tensor name order
    std::sort(to_store.begin(), to_store.end());

    const int32_t chunk_size = m_params.n_ctx / m_params.n_parallel;

    std::ofstream out(fname, std::ios::binary);
    out.write((const char *) &n_entries, sizeof(n_entries));
    for (const auto & name : to_store) {
        const auto & stat = m_stats.at(name);
        const int32_t len = name.size();
        out.write((const char *) &len, sizeof(len));
        out.write(name.c_str(), len);
        // ceiling division to avoid accidental zeros
        const int32_t ncall = (*std::max_element(stat.counts.begin(), stat.counts.end()) + (chunk_size - 1)) / chunk_size;
        out.write((const char *) &ncall, sizeof(ncall));
        const int32_t nval = stat.values.size();
        const int32_t nmat = stat.counts.size();
        out.write((const char *) &nval, sizeof(nval));
        if (nval > 0 && nmat > 0) {
            std::vector<float> tmp(nval);
            for (int32_t i = 0; i < nval; i++) {
                float count = static_cast<float>(stat.counts[i / (nval / nmat)]);
                float value = stat.values[i];
                if (count == 0.0f) {
                    // store 1 for partial data
                    value = 1.0f;
                    count = 1.0f;
                }
                tmp[i] = (value / count) * static_cast<float>(ncall);
            }
            out.write((const char *) tmp.data(), nval * sizeof(float));
        }
    }

    // Write the number of call the matrix was computed with
    out.write((const char *) &m_last_chunk, sizeof(m_last_chunk));

    // Write the input filename at the end of the file to later on specify it in quantize
    {
        const char * dataset_file = m_params.prompt_file.c_str();
        int32_t len = m_params.prompt_file.size();
        // When there is no prompt but there were other imatrix files loaded, use the last dataset
        if (m_params.prompt_file.empty() && !m_datasets.empty()) {
            const std::string & dataset_str = m_datasets[m_datasets.size() - 1];
            dataset_file = dataset_str.c_str();
            len = dataset_str.size();
        }
        out.write((const char *) &len, sizeof(len));
        out.write(dataset_file, len);
    }

    LOGV(1, "\n");
    LOG_DBGV(1, "%s: stored collected data after %d chunks in %s\n", __func__, m_last_chunk, fname.c_str());
}

void IMatrixCollector::save_imatrix(int32_t n_chunk) const {
    auto fname = m_params.out_file;
    int8_t use_legacy_format = m_params.imat_dat;

    if (use_legacy_format > 0) {
        this->save_imatrix_legacy(n_chunk);
        return;
    }
    // only warn when `--output-format gguf` is not specified
    if (use_legacy_format == 0 && !string_ends_with(fname, ".gguf")) {
        LOG_WRN("\n%s: saving imatrix using GGUF format with a different suffix than .gguf\n", __func__);
        LOG_WRN("%s: if you want the previous imatrix format, use --output-format dat\n", __func__);
    }

    if (n_chunk > 0) {
        fname += ".at_";
        fname += std::to_string(n_chunk);
    }

    // write imatrix entries even if they don't have full data. (can be corrected when reading)
    // this can happen with MoE models where some of the experts end up not being exercised by the provided training data

    std::vector<std::string> to_store;
    size_t data_size = 0;

    bool is_first = true; // for printing
    for (const auto & kv : m_stats) {
        const int n_all = kv.second.counts.size();

        int n_zeros = 0;
        for (const auto c : kv.second.counts) {
            if (c == 0) {
                n_zeros++;
            }
        }

        if (n_zeros != 0 && is_first) {
            LOG_INF("\n");
            is_first = false;
        }

        if (n_zeros > 0) {
            LOG_WRN("%s: entry '%40s' has partial data (%.2f%%)\n", __func__, kv.first.c_str(), 100.0f * (n_all - n_zeros) / n_all);
        }

        to_store.push_back(kv.first);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * kv.second.activations.size(), GGML_MEM_ALIGN);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * kv.second.values.size(), GGML_MEM_ALIGN);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * kv.second.counts.size(), GGML_MEM_ALIGN);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * 10, GGML_MEM_ALIGN);
    }

    // deterministic tensor name order
    std::sort(to_store.begin(), to_store.end());

    // Compute per-tensor statistics
    std::vector<tensor_statistics> tstats;
    tstats.reserve(m_stats.size());
    bool legacy;
    for (const auto & kv : m_stats) {
        compute_vector_statistics(tstats, kv.first, kv.second, legacy);
    }
    if (!tstats.empty()) { compute_tensor_statistics(tstats); }

    // index by tensor name
    std::unordered_map<std::string, const tensor_statistics *> tstat_index;
    tstat_index.reserve(tstats.size());
    for (const auto & ts : tstats) { tstat_index[ts.tensor] = &ts; }

    struct ggml_init_params params = {
        /* .mem_size   = */ data_size,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct gguf_context * ctx_gguf = gguf_init_empty();

    {
        std::vector<const char *> datasets;
        datasets.reserve(m_datasets.size() + 1);
        for (size_t i = 0; i < m_datasets.size(); ++i) {
            datasets.push_back(m_datasets[i].c_str());
        }
        if (!m_params.prompt_file.empty()) {
            datasets.push_back(m_params.prompt_file.c_str());
        }

        gguf_set_val_str(ctx_gguf, "general.type", "imatrix");
        // Write the dataset paths
        gguf_set_arr_str(ctx_gguf, LLM_KV_IMATRIX_DATASETS, datasets.data(), datasets.size());
        // Write the number of chunks the matrix was computed with
        gguf_set_val_u32(ctx_gguf, LLM_KV_IMATRIX_CHUNK_COUNT, m_last_chunk);
        gguf_set_val_u32(ctx_gguf, LLM_KV_IMATRIX_CHUNK_SIZE, m_params.n_ctx / m_params.n_parallel);
    }

    for (const auto & name : to_store) {
        const auto & stat = m_stats.at(name);
        const int32_t nval = (int32_t) stat.values.size();
        const int32_t nmat = (int32_t) stat.counts.size();
        if (nval > 0 && nmat > 0) {
            struct ggml_tensor * in_sum2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nval / nmat, nmat);
            struct ggml_tensor * counts  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, nmat);
            ggml_format_name(in_sum2, "%s.in_sum2", name.c_str());
            ggml_format_name(counts, "%s.counts", name.c_str());

            for (int32_t j = 0; j < nval; ++j) {
                ((float *) in_sum2->data)[j] = (float) stat.values[j];
            }
            for (int32_t j = 0; j < nmat; ++j) {
                ((float *) counts->data)[j] = (float) stat.counts[j];
            }

            gguf_add_tensor(ctx_gguf, in_sum2);
            gguf_add_tensor(ctx_gguf, counts);

            if (!stat.activations.empty()) {
                const int32_t nact = (int32_t) stat.activations.size();
                struct ggml_tensor * in_sum  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nact / nmat, nmat);
                ggml_format_name(in_sum, "%s.in_sum", name.c_str());
                for (int32_t j = 0; j < nact; ++j) {
                    ((float *) in_sum->data)[j] = (float) stat.activations[j];
                }
                gguf_add_tensor(ctx_gguf, in_sum);
            }
        }

        // Store per-tensor statistics as a small 1D tensor
        {
            float nan = std::numeric_limits<float>::quiet_NaN();
            float sum_sq = 0.0f;
            float mean = 0.0f;
            float std_deviation = 0.0f;
            float skewness = 0.0f;
            float kurtosis = 0.0f;
            float gain = 0.0f;
            float h_norm = 0.0f;
            float l2_dist = 0.0f;
            float cossim = 0.0f;
            float pearson = 0.0f;
            float covariance = 0.0f;
            auto ts = tstat_index.find(name);
            if (ts != tstat_index.end() && ts->second != nullptr) {
                sum_sq = ts->second->sum_val;
                mean = ts->second->mean_val;
                std_deviation = ts->second->std_deviation;
                skewness = ts->second->skewness;
                kurtosis = ts->second->kurtosis;
                gain = ts->second->gain;
                h_norm = ts->second->elements > 0 ? 100.0f * (ts->second->entropy / std::log2f((float)ts->second->elements)) : nan;
                l2_dist = ts->second->l2_dist;
                cossim = ts->second->cossim;
                pearson = ts->second->pearson;
                covariance = ts->second->covariance;
            }

            struct ggml_tensor * stats = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 11);
            ggml_format_name(stats, "%s.stats", name.c_str());
            ((float *)stats->data)[0] = sum_sq;
            ((float *) stats->data)[1] = mean;
            ((float *) stats->data)[2] = std_deviation;
            ((float *) stats->data)[3] = skewness;
            ((float *) stats->data)[4] = kurtosis;
            ((float *) stats->data)[5] = gain;
            ((float *)stats->data)[6] = h_norm;
            ((float *) stats->data)[7] = l2_dist;
            ((float *) stats->data)[8] = cossim;
            ((float *) stats->data)[9] = pearson;
            ((float *) stats->data)[10] = covariance;
            gguf_add_tensor(ctx_gguf, stats);
        }
    }

    gguf_write_to_file(ctx_gguf, fname.c_str(), false);

    LOGV(1, "\n");
    LOG_DBGV(1, "%s: stored collected data after %d chunks in %s\n", __func__, m_last_chunk, fname.c_str());

    gguf_free(ctx_gguf);
    ggml_free(ctx);
}

bool IMatrixCollector::load_imatrix_legacy(const char * fname) {
    std::ifstream in(fname, std::ios::binary);
    if (!in) {
        LOG_ERR("%s: failed to open %s\n", __func__, fname);
        return false;
    }
    int n_entries;
    in.read((char *) &n_entries, sizeof(n_entries));
    if (in.fail() || n_entries < 1) {
        LOG_ERR("%s: no data in file %s\n", __func__, fname);
        return false;
    }
    // Guess the chunk size because it's not stored in the file
    const int32_t chunk_size = m_params.n_ctx / m_params.n_parallel;

    for (int i = 0; i < n_entries; ++i) {
        int32_t len = 0;
        in.read((char *) &len, sizeof(len));
        std::vector<char> name_as_vec(len + 1);
        in.read((char *) name_as_vec.data(), len);
        if (in.fail()) {
            LOG_ERR("%s: failed reading name for entry %d from %s\n", __func__, i + 1, fname);
            return false;
        }
        name_as_vec[len] = 0;
        std::string name{ name_as_vec.data() };
        auto & e = m_stats[std::move(name)];
        int32_t ncall = 0;
        in.read((char *) &ncall, sizeof(ncall));
        int32_t nval = 0;
        in.read((char *) &nval, sizeof(nval));
        if (in.fail() || nval < 1) {
            LOG_ERR("%s: failed reading number of values for entry %d\n", __func__, i);
            m_stats = {};
            return false;
        }

        if (e.values.empty()) {
            e.values.resize(nval, 0.0f);
            e.counts.resize(1, 0);
        }

        std::vector<float> tmp(nval);
        in.read((char *) tmp.data(), nval * sizeof(float));
        if (in.fail()) {
            LOG_ERR("%s: failed reading data for entry %d\n", __func__, i);
            m_stats = {};
            return false;
        }

        // Recreate the state as expected by save_imatrix(), and correct for weighted sum.
        for (int i = 0; i < nval; i++) {
            e.values[i] += tmp[i] * chunk_size;
        }
        // The legacy format doesn't distinguish the counts for different experts
        for (size_t j = 0; j < e.counts.size(); ++j) {
            e.counts[j] += ncall * chunk_size;
        }
    }

    {
        // TODO: extract into its own method; this is also used by the GGUF-based format
        // Calculate the last chunk count
        int64_t max_count = 0;
        for (const auto & stats : m_stats) {
            for (int64_t count : stats.second.counts) {
                if (count > max_count) {
                    max_count = count;
                }
            }
        }
        m_last_chunk = max_count / (chunk_size);
    }

    {
        // Read the number of calls the matrix was computed with
        int32_t n_calls;
        in.read((char *) &n_calls, sizeof(n_calls));
        // ignore it because it's not important
    }

    // Read the dataset path to include it when writing to GGUF
    if (!in.fail()){
        int32_t len = 0;
        in.read((char *) &len, sizeof(len));
        if (!in.fail()) {
            std::vector<char> dataset;
            dataset.resize(len + 1, 0);
            in.read(dataset.data(), len);
            if (!in.fail()) {
                m_datasets.push_back(dataset.data());
            }
        }
    }

    return true;
}

// Using GGUF as the file format, for greater extensibility
bool IMatrixCollector::load_imatrix(const char * file_name) {
    struct ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false, // the data is needed
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(file_name, meta_gguf_params);
    if (!ctx_gguf) {
        return this->load_imatrix_legacy(file_name);
    }
    const int32_t n_entries = gguf_get_n_tensors(ctx_gguf);
    if (n_entries < 1) {
        LOG_ERR("%s: no data in file %s\n", __func__, file_name);
        gguf_free(ctx_gguf);
        ggml_free(ctx);
        return false;
    }

    const int64_t datasets_key = gguf_find_key(ctx_gguf, LLM_KV_IMATRIX_DATASETS);
    if (datasets_key != -1 && gguf_get_arr_type(ctx_gguf, datasets_key) == GGUF_TYPE_STRING) {
        const int64_t n = gguf_get_arr_n(ctx_gguf, datasets_key);
        m_datasets.reserve(m_datasets.size() + n);
        for (int64_t i = 0; i < n; ++i) {
            m_datasets.push_back(gguf_get_arr_str(ctx_gguf, datasets_key, i));
        }
    }

    const std::string in_sum_suffix{ ".in_sum" };
    const std::string in_sum2_suffix{ ".in_sum2" };
    const std::string counts_suffix{ ".counts" };

    // Could re-use m_stats instead, but this allows
    // checking for completeness of *each* loaded imatrix file
    // and also makes it easier to re-use a similar implementation in quantize.cpp
    // Using an ordered map to get a deterministic iteration order.
    std::map<std::string, std::tuple<struct ggml_tensor *, struct ggml_tensor *, struct ggml_tensor *>> sums_counts_for;

    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string name = cur->name;

        if (name.empty()) { continue; }

        if (string_remove_suffix(name, in_sum2_suffix)) {
            // in_sum2
            std::get<0>(sums_counts_for[std::move(name)]) = cur;
        } else if (string_remove_suffix(name, counts_suffix)) {
            // counts
            std::get<1>(sums_counts_for[std::move(name)]) = cur;
        }  else if (string_remove_suffix(name, in_sum_suffix)) {
            // in_sum
            std::get<2>(sums_counts_for[std::move(name)]) = cur;
        }
        else {
            // ignore other tensors
        }
    }

    for (const auto & sc : sums_counts_for) {
        const std::string &        name    = sc.first;
        const struct ggml_tensor * in_sum  = std::get<2>(sc.second);
        const struct ggml_tensor * in_sum2 = std::get<0>(sc.second);
        const struct ggml_tensor * counts  = std::get<1>(sc.second);

        if (!in_sum2 || !counts || (in_sum != nullptr && ggml_nelements(in_sum) != ggml_nelements(in_sum2))) {
            LOG_ERR("%s: mismatched sums and counts for %s\n", __func__, name.c_str());
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            return false;
        }

        auto & e = m_stats[name];

        int64_t nval = ggml_nelements(in_sum2);
        if (e.values.empty()) {
            e.values.resize(nval, 0.0f);
        } else if ((size_t) nval != e.values.size()) {
            LOG_ERR("%s: mismatched sums size for %s: %zu != %zu\n", __func__, name.c_str(), (size_t) nval, e.values.size());
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            return false;
        }
        if (in_sum != nullptr) {
            if (e.activations.empty()) {
                e.activations.resize(nval, 0.0f);
            } else if ((size_t) nval != e.activations.size()) {
                LOG_ERR("%s: mismatched activations size for %s: %zu != %zu\n", __func__, name.c_str(), (size_t) nval, e.activations.size());
                gguf_free(ctx_gguf);
                ggml_free(ctx);
                return false;
            }
        }

        int64_t ncounts = ggml_nelements(counts);
        if (e.counts.empty()) {
            e.counts.resize(ncounts, 0);
        } else if (e.counts.size() == 1 && ncounts > 1) {
            // broadcast, when loading an old imatrix
            e.counts.resize(ncounts, e.counts[0]);
        } else if ((size_t) ncounts != e.counts.size()) {
            LOG_ERR("%s: mismatched counts size for %s: %zu != %zu\n", __func__, name.c_str(), (size_t) ncounts, e.counts.size());
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            return false;
        }

        // Recreate the state as expected by save_imatrix()
        for (int64_t j = 0; j < nval; j++) {
            if (in_sum != nullptr) { e.activations[j] += ((const float *) in_sum->data)[j]; }
            e.values[j] += ((const float *) in_sum2->data)[j];
        }
        for (int64_t j = 0; j < ncounts; j++) {
            e.counts[j] += std::lround(((const float *) counts->data)[j]);
        }
    }

    // TODO: extract into its own method; this is also used by the legacy format
    // Calculate the last chunk count
    int64_t max_count = 0;
    for (const auto & stats : m_stats) {
        for (int64_t count : stats.second.counts) {
            if (count > max_count) {
                max_count = count;
            }
        }
    }
    m_last_chunk = max_count / (m_params.n_ctx / m_params.n_parallel);

    gguf_free(ctx_gguf);
    ggml_free(ctx);
    return true;
}

static IMatrixCollector g_collector;

static bool ik_collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data) {
    return g_collector.collect_imatrix(t, ask, user_data);
}

struct results_log_softmax {
    double log_softmax;
    float  logit;
    float  prob;
};

static std::vector<float> softmax(const std::vector<float> & logits) {
    std::vector<float> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) {
        max_logit = std::max(max_logit, v);
    }
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        const float logit = logits[i] - max_logit;
        const float exp_logit = expf(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sum_exp;
    }
    return probs;
}

static results_log_softmax log_softmax(int n_vocab, const float * logits, int tok) {
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    return {logits[tok] - max_logit - log(sum_exp), logits[tok], expf(logits[tok] - max_logit) / (float) sum_exp};
}

static void process_logits(
    int n_vocab, const float * logits, const int * tokens, int n_token, std::vector<std::thread> & workers,
    double & nll, double & nll2, float * logit_history, float * prob_history) {
    std::mutex mutex;
    int counter = 0;
    auto compute = [&mutex, &counter, &nll, &nll2, logit_history, prob_history, n_vocab, logits, tokens, n_token] () {
        double local_nll  = 0;
        double local_nll2 = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int i = counter++;
            if (i >= n_token) {
                nll += local_nll; nll2 += local_nll2;
                break;
            }
            lock.unlock();
            const results_log_softmax results = log_softmax(n_vocab, logits + i*n_vocab, tokens[i+1]);
            const double v = -results.log_softmax;
            local_nll += v;
            local_nll2 += v*v;

            logit_history[i] = results.logit;
            prob_history[i]  = results.prob;
        }
    };
    for (auto & w : workers) {
        w = std::thread(compute);
    }
    compute();
    for (auto & w : workers) {
        w.join();
    }
}

static bool compute_imatrix(llama_context * ctx, const common_params & params, const int32_t n_ctx) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    GGML_ASSERT(!llama_vocab_get_add_eos(vocab));

    auto tim1 = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: tokenizing the input ..\n", __func__);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, params.parse_special);

    auto tim2 = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: tokenization took %g ms\n",__func__,1e-3*std::chrono::duration_cast<std::chrono::microseconds>(tim2-tim1).count());

    if (params.i_chunk > 0) {
        if (size_t((params.i_chunk + 2)*n_ctx) >= tokens.size()) {
            LOG_ERR("%s: there will be not enough tokens left after removing %d chunks\n", __func__, params.i_chunk);
            return false;
        }
        LOG_INF("%s: removing initial %d chunks (%d tokens)\n", __func__, params.i_chunk, params.i_chunk*n_ctx);
        tokens.erase(tokens.begin(), tokens.begin() + params.i_chunk*n_ctx);
    }

    if (int(tokens.size()) < 2*n_ctx) {
        LOG_ERR("%s: you need at least %d tokens for a context of %d tokens\n", __func__, 2*n_ctx, n_ctx);
        LOG_ERR("%s: the data file you provided tokenizes to only %zu tokens\n", __func__, tokens.size());
        return false;
    }

    std::vector<float> logit_history;
    std::vector<float> prob_history;

    if (params.compute_ppl) {
        logit_history.resize(tokens.size());
        prob_history.resize(tokens.size());
    }

    const int n_chunk_max = tokens.size() / n_ctx;

    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_batch = params.n_batch;

    int count = 0;
    double nll = 0.0;
    double nll2 = 0.0;

    const int num_batches = (n_ctx + n_batch - 1) / n_batch;
    const int n_seq = std::max(1, n_batch / n_ctx);

    GGML_ASSERT(n_batch < n_ctx || n_batch % n_ctx == 0);
    GGML_ASSERT(params.n_ctx == n_seq * n_ctx);

    llama_batch batch = llama_batch_init(std::min(n_batch, n_ctx*n_seq), 0, 1);

    std::vector<float> logits;
    if (params.compute_ppl && num_batches > 1) {
        logits.reserve((size_t)n_ctx * n_vocab);
    }

    LOG_INF("%s: computing over %d chunks, n_ctx=%d, batch_size=%d, n_seq=%d\n", __func__, n_chunk, n_ctx, n_batch, n_seq);

    std::vector<std::thread> workers(std::thread::hardware_concurrency() - 1);

    for (int i = 0; i < n_chunk; i += n_seq) {
        const int start =     i * n_ctx;
        const int end   = start + n_ctx;

        const int n_seq_batch = std::min(n_seq, n_chunk - i);

        const auto t_start = std::chrono::high_resolution_clock::now();

        // clear the KV cache
        llama_memory_clear(llama_get_memory(ctx), true);

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            // clear the batch
            common_batch_clear(batch);

            for (int seq = 0; seq < n_seq_batch; seq++) {
                int seq_start = batch_start + seq*n_ctx;

                // save original token and restore it after eval
                const auto token_org = tokens[seq_start];

                // add BOS token for the first batch of each chunk
                if (add_bos && j == 0) {
                    tokens[seq_start] = llama_vocab_bos(vocab);
                }
                for (int k = 0; k < batch_size; ++k) {
                    // NOTE: specifying all logits to get activations for the output.weight tensor
                    //       and also for the perplexity calculation.
                    // TODO: only get outputs when (params.process_output || params.compute_ppl)
                    //       (not possible when this skips FFN computation of the last layer)
                    common_batch_add(batch, tokens[seq_start + k], j*n_batch + k, { seq }, true);
                }

                // restore the original token in case it was set to BOS
                tokens[seq_start] = token_org;
            }

            if (llama_decode(ctx, batch)) {
                LOG_ERR("%s : failed to eval\n", __func__);
                llama_batch_free(batch);
                return false;
            }

            if (params.compute_ppl && num_batches > 1) {
                const auto * batch_logits = llama_get_logits(ctx);
                logits.insert(logits.end(), batch_logits, batch_logits + batch_size * n_vocab);
            }
        }


        if (i == 0) {
            llama_synchronize(ctx);
            const auto t_end = std::chrono::high_resolution_clock::now();
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            LOG_INF("%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * n_chunk / n_seq);
            if (total_seconds >= 60*60) {
                LOG("%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            LOG("%.2f minutes\n", total_seconds / 60.0);
        }

        if (params.compute_ppl) {
            const int first = n_ctx/2;
            for (int seq = 0; seq < n_seq_batch; seq++) {
                const float * all_logits = num_batches > 1 ? logits.data() : llama_get_logits_ith(ctx, seq*n_ctx);

                llama_token * tokens_data = tokens.data() + start + seq*n_ctx + first;

                process_logits(n_vocab, all_logits + first*n_vocab,
                        tokens_data, n_ctx - 1 - first,
                        workers, nll, nll2,
                        logit_history.data() + start + seq*n_ctx + first,
                        prob_history.data()  + start + seq*n_ctx + first);

                count += n_ctx - first - 1;

                LOG("[%d]%.4lf,", i + seq + 1, std::exp(nll / count));
            }
            fflush(stdout);

            logits.clear();
        }
    }

    LOG("\n");

    if (params.compute_ppl) {
        nll2 /= count;
        nll /= count;
        const double ppl = exp(nll);
        nll2 -= nll * nll;
        if (nll2 > 0) {
            nll2 = sqrt(nll2/(count-1));
            LOG("Final estimate: PPL = %.4lf +/- %.5lf\n", ppl, nll2*ppl);
        } else {
            LOG("Unexpected negative standard deviation of log(prob)\n");
        }
    }

    llama_batch_free(batch);

    return true;
}

static bool show_statistics(const common_params & params) {
    constexpr auto fnan = std::numeric_limits<float>::quiet_NaN();
    g_collector.set_params(params);
    std::vector<tensor_statistics> ts;

    if (params.in_files.empty()) { return false; }

    // Load and process data
    if (g_collector.load_imatrix(params.in_files[0].c_str())) {
        ts.reserve(g_collector.get_mstats().size());
        for (const auto & [name, stats] : g_collector.get_mstats()) {
            bool legacy_imatrix = true;
            if (!compute_vector_statistics(ts, name, stats, legacy_imatrix)) { continue; }
        }
    } else {
        return false;
    }

    if (ts.empty()) { return false; }

    bool legacy = ts.empty() ? true : ts[0].stats.activations.empty();
    compute_tensor_statistics(ts);

    // Sorting logic (Layer index -> Tensor Name)
    struct tensor_comparer {
        bool operator()(const tensor_statistics & a, const tensor_statistics & b) const {
            std::string lay_a;
            std::string lay_b;
            std::string name_a;
            std::string name_b;
            process_tensor_name(a.tensor, lay_a, name_a);
            process_tensor_name(b.tensor, lay_b, name_b);

            // Handle non-numeric layers (e.g., "output")
            int blk_a = 9999;
            int blk_b = 9999;
            try {
                blk_a = std::stoi(lay_a);
            } catch(...) {
                if (a.tensor.find("output") != std::string::npos) { blk_a = 10000; }
            }
            try {
                blk_b = std::stoi(lay_b);
            } catch(...) {
                if (b.tensor.find("output") != std::string::npos) { blk_b = 10000; }
            }

            if (blk_a != blk_b) { return blk_a < blk_b; }
            return name_a < name_b;
        }
    };
    std::sort(ts.begin(), ts.end(), tensor_comparer());

    struct layer_stats {
        float layer_sum = 0.0f;
        int n = 0;
    };
    std::map<int, layer_stats> ls;

    // Shorten names for table formatting
    auto label_fmt = [](std::string s, size_t w) -> std::string {
        if (s.length() <= w) { return s; }
        return ".." + s.substr(s.length() - (w - 2));
    };

    constexpr int w_lay = 6;
    constexpr int w_nam = 40; // Should be wide enough for most tensor names
    const auto * sep = " | ";

    LOG_INF("\nComputing tensor statistics for %s (%d tensors)\n", params.in_files[0].c_str(), static_cast<int>(ts.size()));

    if (legacy) {
        LOG_INF("\n%*s%s%-*s%s%10s%10s%12s%12s%9s%s%17s%8s%s%10s%10s\n",
            w_lay, "Layer", sep,
            w_nam, "Tensor", sep,
            "Mean", "StdDev", "Skew", "Kurt", "H Norm", sep,
            "∑ E[A²]", "Gain", sep,
            "PCC", "Cov");
        LOG_INF("%s\n", std::string(153, '-').c_str());
    } else {
        LOG_INF("\n%*s%s%-*s%s%10s%10s%12s%12s%9s%s%17s%8s%s%12s%10s%10s\n",
            w_lay, "Layer", sep,
            w_nam, "Tensor", sep,
            "Mean", "StdDev", "Skew", "Kurt", "H Norm", sep,
            "∑ E[A²]", "Gain", sep,
            "L2 Dist", "PCC", "Cov");
        LOG_INF("%s\n", std::string(165, '-').c_str());
    }

    // Tensor Statistics
    for (const auto & tstat : ts) {
        std::string layer;
        std::string name;

        process_tensor_name(tstat.tensor, layer, name);
        const float h_norm = tstat.elements > 1 ? 100.0f * (tstat.entropy / std::log2f((float)tstat.elements)) : fnan;

        int blk;
        try {
            blk = std::stoi(layer);
        } catch (...) {
            if (tstat.tensor.find("output") != std::string::npos) { blk = 10000; }
            else { blk = -1; }
        }

        if (legacy) {
            LOG_INF("%*s%s%-*s%s%10.4f%10.4f%12.4f%12.4f%8.2f%%%s%14.4f%8.2f%s%10.4f%10.4f\n",
                w_lay, layer.c_str(), sep,
                w_nam, label_fmt(tstat.tensor, w_nam).c_str(), sep,
                tstat.mean_val, tstat.std_deviation, tstat.skewness, tstat.kurtosis, h_norm, sep,
                tstat.sum_val, tstat.gain, sep,
                tstat.pearson, tstat.covariance
            );
        } else {
            LOG_INF("%*s%s%-*s%s%10.4f%10.4f%12.4f%12.4f%8.2f%%%s%14.4f%8.2f%s%12.4f%10.4f%10.4f\n",
                w_lay, layer.c_str(), sep,
                w_nam, label_fmt(tstat.tensor, w_nam).c_str(), sep,
                tstat.mean_val, tstat.std_deviation, tstat.skewness, tstat.kurtosis, h_norm, sep,
                tstat.sum_val, tstat.gain, sep,
                tstat.l2_dist, tstat.pearson, tstat.covariance
            );
        }

        // Aggregate Layer Stats
        auto & l = ls[blk];
        l.layer_sum += tstat.sum_val;
        l.n += tstat.elements;
    }

    // Layer Statistics
    std::map<int, float> layer_cossim;
    std::map<int, float> layer_l2_dist;
    std::map<int, float> layer_pearson;
    std::map<int, float> layer_covariance;
    std::map<int, float> layer_gain;
    compute_layer_statistics(ts, layer_cossim, layer_l2_dist, layer_pearson, layer_covariance, layer_gain);

    size_t layers = 0;
    int min = std::numeric_limits<int>::max();
    int max = -1;

    for (const auto & [layer, stats] : ls) {
        if (layer >= 0 && layer < 9999 && stats.n > 0) {
            layers++;
            min = std::min(layer, min);
            max = std::max(layer, max);
        }
    }

    if (layers > 0) {
        const auto expected = (size_t)(max - min + 1);
        if (layers != expected) {
            LOG_WRN("\n%s: layer sequence gap detected (found %zu layers in range %d-%d, expected %zu); layer statistics will not be shown\n",
                __func__, layers, min, max, expected);
            return false;
        }
    }

    LOG_INF("\n\nComputing layer statistics for %s (%zu layers)\n\n", params.in_files[0].c_str(), layers);

    if (legacy) {
        LOG_INF("%*s%s%17s%8s%s%9s%9s%12s\n",
            w_lay, "Layer", sep,
            "∑ E[A²]", "Gain", sep,
            "CosSim", "PCC", "Cov");
        LOG_INF("%s\n", std::string(61, '-').c_str());
    } else {
        LOG_INF("%*s%s%17s%8s%s%12s%9s%9s%12s\n",
            w_lay, "Layer", sep,
            "∑ E[A²]", "Gain", sep,
            "L2 Dist", "CosSim", "PCC", "Cov");
        LOG_INF("%s\n", std::string(73, '-').c_str());
    }

    for (const auto & [layer, stats] : ls) {
        if (layer < 0 || stats.n == 0) { continue; }

        float lgn = layer == 0 || layer == 10000 ? fnan : layer_gain[layer];
        float ll2 = layer == 0 || layer == 10000 ? fnan : layer_l2_dist[layer];
        float lcs = layer == 0 || layer == 10000 ? fnan : layer_cossim[layer];
        float lpc = layer == 0 || layer == 10000 ? fnan : layer_pearson[layer];
        float lcv = layer == 0 || layer == 10000 ? fnan : layer_covariance[layer];
        auto str = std::to_string(layer);
        const auto *lyr = layer == 10000 ? "-" : str.c_str();

        if (legacy) {
            LOG_INF("%*s%s%14.4f%8.2f%s%9.4f%9.4f%12.4f\n",
                w_lay, lyr, sep,
                stats.layer_sum, lgn, sep,
                lcs, lpc, lcv);
        } else {
            LOG_INF("%*s%s%14.4f%8.2f%s%12.4f%9.4f%9.4f%12.4f\n",
                w_lay, lyr, sep,
                stats.layer_sum, lgn, sep,
                ll2, lcs, lpc, lcv);
        }
    }

    LOG_INF("\n");
    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    params.out_file = "imatrix.gguf";

    params.n_ctx = 512;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    if (params.show_statistics) {
        if (!show_statistics(params)) {
            return 1;
        }
        return 0;
    }

    common_init();

    const int32_t n_ctx = params.n_ctx;

    if (n_ctx <= 0) {
        LOG_ERR("%s: imatrix tool requires '--ctx-size' > 0\n", __func__);
        return 1;
    }

    {
        const int32_t n_seq = std::max(1, params.n_batch / n_ctx);
        const int32_t n_kv = n_seq * n_ctx;

        params.n_parallel = n_seq;
        params.n_ctx      = n_kv;

        params.n_batch = std::min(params.n_batch, n_kv);
    }

    g_collector.set_params(params);

    for (const auto & in_file : params.in_files) {
        LOG_INF("%s : loading imatrix from '%s'\n", __func__, in_file.c_str());
        if (!g_collector.load_imatrix(in_file.c_str())) {
            LOG_ERR("%s : failed to load %s\n", __func__, in_file.c_str());
            return 1;
        }
    }

    if (params.prompt.empty()) {
        LOG_INF("No prompt provided; combining precomputed matrices only.\n");

        if (params.in_files.empty()) {
            LOG_ERR("Error: No prompt provided and no precomputed matrices (--in-file) to combine.\n");
            return 1;
        }

        if (params.in_files.size() == 1) {
            LOG_INF("%s : saving imatrix to '%s'\n", __func__, params.out_file.c_str());
        } else if (params.in_files.size() > 1) {
            LOG_INF("%s : saving combined imatrix to '%s'\n", __func__, params.out_file.c_str());
        }

        g_collector.save_imatrix();

        return 0;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ik_collect_imatrix;
    params.cb_eval_user_data = NULL;
    params.warmup = false;

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_model_n_ctx_train(model);
    if (params.n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, params.n_ctx);
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    if (!compute_imatrix(ctx, params, n_ctx)) {
        return 1;
    }

    g_collector.save_imatrix();

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
