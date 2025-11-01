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
    float sum_values    = 0.0f;
    float mean_values   = 0.0f;
    float max_values    = 0.0f;
    float min_values    = 0.0f;
    int   elements      = 0;
    float std_deviation = 0.0f;
    float entropy       = 0.0f;
    float zd_score      = 0.0f;
    float cossim        = 0.0f;
    float l2_dist       = 0.0f;
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
    std::vector<std::string> name;
    std::istringstream stream(input);
    std::string item;

    while (std::getline(stream, item, '.')) {
        name.push_back(item);
    }
    for (size_t i = 0; i < name.size(); ++i) {
        if (name[i] == "blk" && i + 1 < name.size()) {
            layer = name[i + 1];
            break;
        }
    }
    for (size_t i = 0; i < name.size(); ++i) {
        if (name[i] == "weight" && i > 0) {
            tensor = name[i - 1];
            break;
        }
    }

    if (tensor.empty()) {
        tensor = input;
    }
    if (layer.empty()) {
        layer = "-";
    }
}

static std::vector<float> compute_tensor_averages(const Stats & tstats) {
    if (tstats.counts.empty()) { return {}; }
    const size_t n_mat = tstats.counts.size();
    const size_t len = !tstats.activations.empty() ? tstats.activations.size() : tstats.values.size();
    if (len == 0 || n_mat == 0 || len % n_mat != 0) { return {}; }
    const size_t row = len / n_mat;
    std::vector<float> vec;
    vec.reserve(len);

    if (tstats.activations.empty()) {
        // Mean of squares
        for (size_t m = 0; m < n_mat; ++m) {
            const auto c = (float)tstats.counts[m];
            const size_t off = m * row;
            if (c <= 0.0f) {
                vec.insert(vec.end(), row, 0.0f); // zero-fill rows for experts with zero count to preserve shape
                continue;
            }
            for (size_t j = 0; j < row; ++j) { vec.push_back(tstats.values[off + j] / c); }
        }
    } else {
        // Mean
        for (size_t m = 0; m < n_mat; ++m) {
            const auto c = (float)tstats.counts[m];
            const size_t off = m * row;
            if (c <= 0.0f) {
                vec.insert(vec.end(), row, 0.0f); // zero-fill rows for experts with zero count to preserve shape
                continue;
            }
            for (size_t j = 0; j < row; ++j) { vec.push_back(tstats.activations[off + j] / c); }
        }
    }

    return vec;
}

static bool compute_vector_statistics(std::vector<tensor_statistics> & tstats, const std::string & name, const Stats & e) {
    const size_t n_mat = e.counts.size();
    const size_t len = e.activations.empty() ? e.values.size() : e.activations.size();
    const bool legacy = e.activations.empty();
    if (n_mat == 0) {
        LOG_ERR("%s: there are no activations for tensor %s. The imatrix may be suboptimal\n", __func__, name.c_str());
        return false;
    }
    if (len == 0 || (len % n_mat) != 0) {
        LOG_ERR("%s: activation size mismatch for tensor %s (len=%zu, counts=%zu)\n", __func__, name.c_str(), len, n_mat);
        return false;
    }
    if (!legacy && e.values.size() != len) {
        LOG_ERR("%s: activations/values size mismatch for tensor %s (act=%zu, val=%zu)\n", __func__, name.c_str(), len, e.values.size());
        return false;
    }

    const size_t row_size = len / n_mat;
    double mean = 0.0;
    double M2 = 0.0;
    double sum = 0.0;
    float vmin = std::numeric_limits<float>::infinity();
    float vmax = -std::numeric_limits<float>::infinity();
    double energy_sum = 0.0;
    size_t valid_n = 0;
    for (size_t i = 0; i < n_mat; ++i) {
        const auto c = (float)e.counts[i];
        if (c <= 0.0f) { continue; } // skip experts with zero count
        const size_t off = i * row_size;

        for (size_t j = 0; j < row_size; ++j) {
            const double v_avg = legacy ? 0.0 : (double)e.activations[off + j] / (double)c; // E[x]
            const double v_energy = (double)e.values[off + j] / (double)c; // E[x^2]
            const double v = legacy ? v_energy : v_avg;

            ++valid_n;
            sum += v;
            vmin = std::min(vmin, (float)v);
            vmax = std::max(vmax, (float)v);

            const double delta = v - mean;
            mean += delta / (double)valid_n;
            M2 += delta * (v - mean);
            energy_sum += std::max(0.0, v_energy);
        }
    }

    if (valid_n == 0) {
        LOG_ERR("%s: there are no activations for tensor %s. The imatrix may be suboptimal\n", __func__, name.c_str());
        return false;
    }

    float std_deviation = 0.0f;
    float entropy = 0.0f;
    double zd_count = 0.0;
    double variance = valid_n > 1 ? M2 / ((double)valid_n - 1) : 0.0;
    variance = std::max(variance, 0.0);
    std_deviation = std::sqrt((float)variance);
    if (energy_sum > 0.0) {
        for (size_t i = 0; i < n_mat; ++i) {
            const auto c = (float)e.counts[i];
            if (c <= 0.0f) { continue; }
            const size_t off = i * row_size;
            for (size_t j = 0; j < row_size; ++j) {
                const double v_energy = (double)e.values[off + j] / (double)c; // E[x^2]
                const double w = std::max(0.0, v_energy);
                const double p = w / energy_sum;
                if (p > 0.0) { entropy -= (float)(p * std::log2(p)); }
            }
        }
    }
    if (std_deviation > 0.0f) {
        for (size_t i = 0; i < n_mat; ++i) {
            const float c = (float)e.counts[i];
            if (c <= 0.0f) { continue; }
            const size_t off = i * row_size;
            for (size_t j = 0; j < row_size; ++j) {
                const double v_avg = legacy ? 0.0 : (double)e.activations[off + j] / (double)c; // E[x]
                const double v_energy = (double)e.values[off + j] / (double)c; // E[x^2]
                const float v = (float)(legacy ? v_energy : v_avg);
                const float z = (v - (float)mean) / std_deviation;
                if (std::fabs(z) > 1.0f) { zd_count += 1.0; }
            }
        }
    }

    auto & ts = tstats.emplace_back();
    ts.tensor = name;
    ts.stats = e;
    ts.sum_values = (float)sum;
    ts.mean_values = (float)mean;
    ts.max_values = vmax;
    ts.min_values = vmin;
    ts.elements = valid_n;
    ts.std_deviation = std_deviation;
    ts.entropy = entropy;
    ts.zd_score = (float)(zd_count / (double)valid_n);

    return e.activations.empty();
}

static void compute_tensor_statistics(std::vector<tensor_statistics> & tstats) {
    static const std::regex pattern(R"(blk\.(\d+)\.)");
    for (auto & ts : tstats) {
        ts.cossim = 0.0f;
        ts.l2_dist = 0.0f;

        if (std::smatch match; std::regex_search(ts.tensor, match, pattern)) {
            const int blk = std::stoi(match[1]);
            if (blk <= 0) { continue; }
            std::string tname(ts.tensor);
            tname.replace(match.position(1), match.length(1), std::to_string(blk - 1));
            auto prev_it = std::find_if(tstats.begin(), tstats.end(),
                [tname](const tensor_statistics & t) { return t.tensor == tname; });
            if (prev_it == tstats.end()) { continue; }

            const auto curr_avg = compute_tensor_averages(ts.stats);
            const auto prev_avg = compute_tensor_averages(prev_it->stats);
            if (curr_avg.empty() || curr_avg.size() != prev_avg.size()) { continue; }

            float dot_prod = 0.0f;
            float norm1_sq = 0.0f;
            float norm2_sq = 0.0f;
            float l2_dist_sq = 0.0f;

            for (size_t i = 0; i < curr_avg.size(); ++i) {
                const float c_val = curr_avg[i];
                const float p_val = prev_avg[i];
                dot_prod += c_val * p_val;
                norm1_sq += c_val * c_val;
                norm2_sq += p_val * p_val;
                const float diff = c_val - p_val;
                l2_dist_sq += diff * diff;
            }

            // Compute Cosine Similarity
            float cs = 0.0f;
            if (norm1_sq > 0.0f && norm2_sq > 0.0f) {
                cs = dot_prod / (std::sqrt(norm1_sq) * std::sqrt(norm2_sq));
                cs = std::min(cs, 1.0f);
                cs = std::max(cs, -1.0f);
            } else if (norm1_sq == 0.0f && norm2_sq == 0.0f) {
                cs = 1.0f;
            }
            ts.cossim = cs;

            // Compute L2 Norm (Euclidean Distance)
            ts.l2_dist = std::sqrt(l2_dist_sq);
        }
    }
}

static void compute_layer_statistics(const std::vector<tensor_statistics> & tstats,
                                              std::map<int, float> & layer_cossim,
                                              std::map<int, float> & layer_l2_dist,
                                              const std::unordered_map<std::string, Stats> & stats_map) {
    struct layer_aggregation {
        double sum_dot_prod = 0.0;
        double sum_norm1_sq = 0.0;
        double sum_norm2_sq = 0.0;
        double sum_l2_dist_sq = 0.0;
        int n_tensors = 0;
    };

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    std::unordered_map<std::string, const tensor_statistics*> tidx;
    tidx.reserve(tstats.size());
    for (const auto & ts : tstats) { tidx[ts.tensor] = &ts; }

    std::map<int, layer_aggregation> agr;
    for (const auto & ts : tstats) {
        std::smatch match;
        if (!std::regex_search(ts.tensor, match, pattern)) { continue; }
        const int blk = std::stoi(match[1]);
        if (blk <= 0) { continue; }

        std::string prev_lyr(ts.tensor);
        prev_lyr.replace(match.position(1), match.length(1), std::to_string(blk - 1));
        if (tidx.find(prev_lyr) == tidx.end()) { continue; }

        auto it_curr = stats_map.find(ts.tensor);
        auto it_prev = stats_map.find(prev_lyr);
        if (it_curr == stats_map.end() || it_prev == stats_map.end()) { continue; }

        const auto curr_avg = compute_tensor_averages(it_curr->second);
        const auto prev_avg = compute_tensor_averages(it_prev->second);
        if (curr_avg.empty() || prev_avg.empty() || curr_avg.size() != prev_avg.size()) { continue; }

        // Compute statistics for each tensor pair individually
        double dot_prod = 0.0;
        double norm1_sq = 0.0;
        double norm2_sq = 0.0;
        double l2_dist_sq = 0.0;

        for (size_t i = 0; i < curr_avg.size(); ++i) {
            const double c_val = curr_avg[i];
            const double p_val = prev_avg[i];
            dot_prod += c_val * p_val;
            norm1_sq += c_val * c_val;
            norm2_sq += p_val * p_val;
            const double diff = c_val - p_val;
            l2_dist_sq += diff * diff;
        }

        if (norm1_sq == 0.0 && norm2_sq == 0.0) { continue; }

        // Accumulate statistics for the layer
        auto & entry = agr[blk];
        entry.sum_dot_prod += dot_prod;
        entry.sum_norm1_sq += norm1_sq;
        entry.sum_norm2_sq += norm2_sq;
        entry.sum_l2_dist_sq += l2_dist_sq;
        entry.n_tensors++;
    }

    // Compute aggregated layer statistics
    for (auto & kv : agr) {
        const auto & agg = kv.second;
        if (agg.n_tensors == 0) { continue; }

        // Compute aggregated Cosine Similarity
        float cossim = 0.0f;
        if (agg.sum_norm1_sq > 0.0 && agg.sum_norm2_sq > 0.0) {
            cossim = agg.sum_dot_prod / (std::sqrt(agg.sum_norm1_sq) * std::sqrt(agg.sum_norm2_sq));
            cossim = std::min(cossim, 1.0f);
            cossim = std::max(cossim, -1.0f);
        } else if (agg.sum_norm1_sq == 0.0 && agg.sum_norm2_sq == 0.0) {
            cossim = 1.0f;
        }
        layer_cossim[kv.first] = cossim;

        // Compute aggregated L2 Distance (Euclidean Distance)
        layer_l2_dist[kv.first] = (float)std::sqrt(agg.sum_l2_dist_sq);
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
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * 4, GGML_MEM_ALIGN);
    }

    // deterministic tensor name order
    std::sort(to_store.begin(), to_store.end());

    // Compute per-tensor statistics (CosSim, L2 Dist, ECS) to store alongside sums
    std::vector<tensor_statistics> tstats;
    tstats.reserve(m_stats.size());
    bool legacy_mode = true;
    for (const auto & kv : m_stats) {
        const bool is_legacy = compute_vector_statistics(tstats, kv.first, kv.second);
        legacy_mode = legacy_mode && is_legacy;
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

        // Store per-tensor statistics as a small 1D tensor: [ECS, L2 Dist, CosSim, ZD Score]
        {
            float l2 = 0.0f;
            float cs = 0.0f;
            float zd = 0.0f;
            float ecs = 0.0f;
            auto it_ts = tstat_index.find(name);
            if (it_ts != tstat_index.end() && it_ts->second != nullptr) {
                l2 = it_ts->second->l2_dist;
                cs = it_ts->second->cossim;
                zd = it_ts->second->zd_score;
                ecs = 100.0f * (1.0f - std::exp(-0.01f * l2) * std::pow(std::fabs(cs), 10.0f));
            }

            struct ggml_tensor * stats_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
            ggml_format_name(stats_t, "%s.stats", name.c_str());
            ((float *)stats_t->data)[0] = ecs;
            ((float *)stats_t->data)[1] = l2;
            ((float *)stats_t->data)[2] = cs;
            ((float *)stats_t->data)[3] = zd;
            gguf_add_tensor(ctx_gguf, stats_t);
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
            if (in_sum != nullptr) {
                e.activations.resize(nval, 0.0f);
            }
        } else if ((size_t) nval != e.values.size()) {
            LOG_ERR("%s: mismatched sums size for %s: %zu != %zu\n", __func__, name.c_str(), (size_t) nval, e.values.size());
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            return false;
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
    std::vector<tensor_statistics> ts;
    bool legacy_mode = true;

    if (params.in_files.empty() || params.in_files.size() > 1) {
        LOG_ERR("\nError: a single imatrix file is required to compute tensor statistics\n\n");
        return false;
    }
    if (g_collector.load_imatrix(params.in_files[0].c_str())) {
        for (const auto & [name, stats] : g_collector.get_mstats()) {
            const bool is_legacy = compute_vector_statistics(ts, name, stats);
            legacy_mode = legacy_mode && is_legacy;
        }
    } else {
        LOG_ERR("\nError: %s is not a valid imatrix file\n\n", params.in_files[0].c_str());
        return false;
    }
    if (!ts.empty()) {
        compute_tensor_statistics(ts);
    } else {
        LOG_ERR("Error: cannot compute statistics for %s\n\n", params.in_files[0].c_str());
        return false;
    }

    struct tensor_comparer {
        bool legacy_mode;
        explicit tensor_comparer(const bool legacy) : legacy_mode(legacy) {}

        bool operator()(const tensor_statistics & a, const tensor_statistics & b) const {
            std::string layer;
            std::string name_a;
            std::string name_b;
            process_tensor_name(a.tensor, layer, name_a);
            process_tensor_name(b.tensor, layer, name_b);
            return legacy_mode ? name_a < name_b || (name_a == name_b && a.sum_values > b.sum_values)
                               : name_a < name_b || (name_a == name_b && a.cossim > b.cossim);
        }
    };
    std::sort(ts.begin(), ts.end(), tensor_comparer(legacy_mode));

    struct layer_stats {
        float layer_sum = 0.0f;
        float layer_zd = 0.0f;
        int n = 0;
    };

    std::map<int, layer_stats> ls;
    LOG_INF("\nComputing tensor statistics for %s (%d tensors)\n", params.in_files[0].c_str(), static_cast<int>(ts.size()));
    LOG_INF("\n%6s\t%18s\t%13s\t%8s\t%8s\t%7s\t%15s\t%13s\t%11s\t%8s\t%5s\t%10s\n",
        "Layer",
        "Tensor",
        legacy_mode ? "Σ E[Act²]" : "L₂ Dist",
        "Min",
        "Max",
        "μ",
        "σ",
        "N",
        "H Norm",
        legacy_mode ? "H" : "ECS",
        "ZD",
        "CosSim");
    LOG_INF(
        "=============================================================================================================="
        "=============================================================\n");

    for (const auto & tstat : ts) {
        std::string layer;
        std::string name;
        process_tensor_name(tstat.tensor, layer, name);

        int blk;
        try {
            blk = std::stoi(layer);
        } catch (const std::exception &) {
            blk = -1; // not a block layer
        }

        const float h_norm = tstat.elements > 1 ? 100.0f * (tstat.entropy / std::log2((float) tstat.elements)) : 0.0f;
        const float ecs = 100.0f * (1.0f - std::exp(-0.01f * tstat.l2_dist) * std::pow(std::fabs(tstat.cossim), 10.0f)); // Euclidean-Cosine score

        LOG_INF("%5s\t%-20s\t%11.4f\t%10.4f\t%10.4f\t%8.4f\t%8.4f\t%7d\t%10.2f%%\t%10.4f\t%6.2f%%\t%10.4f\n",
            layer.c_str(),
            name.c_str(),
            legacy_mode ? tstat.sum_values : tstat.l2_dist,
            tstat.min_values,
            tstat.max_values,
            tstat.mean_values,
            tstat.std_deviation,
            tstat.elements,
            h_norm,
            legacy_mode ? tstat.entropy : ecs,
            100.0f * tstat.zd_score,
            tstat.cossim);

        const float zd = tstat.elements * tstat.zd_score;
        if (ls.find(blk) != ls.end()) {
            ls[blk].layer_sum += tstat.sum_values;
            ls[blk].layer_zd += zd;
            ls[blk].n += tstat.elements;
        } else {
            layer_stats temp_ls;
            temp_ls.layer_sum = tstat.sum_values;
            temp_ls.layer_zd = zd;
            temp_ls.n = tstat.elements;
            ls[blk] = temp_ls;
        }
    }

    std::map<int, float> layer_cossim;
    std::map<int, float> layer_l2_dist;
    compute_layer_statistics(ts, layer_cossim, layer_l2_dist, g_collector.get_mstats());

    const size_t layers = std::count_if(ls.begin(), ls.end(), [](const auto & kv) { return kv.first >= 0; });
    LOG_INF("\nComputing layer statistics (%zu layers)\n", layers);
    LOG_INF("\n%6s\t%13s\t%6s\t%11s\t%6s\n",
        "Layer",
        legacy_mode ? "Σ E[Act²]" : "L₂ Dist",
        "ZD",
        "CosSim",
        legacy_mode ? "" : "ECS");
    if (legacy_mode) {
        LOG_INF("============================================\n");
    } else {
        LOG_INF("=========================================================\n");
    }
    for (const auto & [layer, stats] : ls) {
        if (layer < 0 || stats.n == 0) { continue; }
        const auto lcs = layer_cossim.find(layer);
        const float layer_cs = lcs != layer_cossim.end() ? lcs->second : 0.0f;
        const auto ll2n = layer_l2_dist.find(layer);
        const float layer_l2n = ll2n != layer_l2_dist.end() ? ll2n->second : 0.0f;
        if (legacy_mode) {
            LOG_INF("%5d\t%11.4f\t%6.2f%%\t%11.4f\n",
                layer,
                stats.layer_sum,
                100.0f * stats.layer_zd / stats.n,
                layer_cs);
        } else {
            LOG_INF("%5d\t%11.4f\t%6.2f%%\t%11.4f\t%8.4f\n",
                layer,
                layer_l2n,
                100.0f * stats.layer_zd / stats.n,
                layer_cs,
                100.0f * (1.0f - std::exp(-0.01f * layer_l2n) * std::pow(std::fabs(layer_cs), 10.0f))); // Euclidean-Cosine score
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
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

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
