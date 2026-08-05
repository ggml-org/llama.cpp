#include "arg.h"
#include "common.h"
#include "imatrix-loader.h"
#include "log.h"
#include "llama.h"
#include "gguf.h"
#include "nlohmann/json.hpp"
#include "sampling.h"
#include "speculative-calibration.h"
#include "speculative.h"

// staging API: llama_set_embeddings_layer_inp / llama_get_embeddings_layer_inp
// (per-layer hidden-state capture used by the offline feature pass)
#include "../../src/llama-ext.h"

// offline DFlash feature-capture file format (shared with the training driver)
#include "tessera-features.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <thread>
#include <mutex>
#include <vector>
#include <fstream>
#include <future>
#include <unordered_map>
#include <map>
#include <regex>
#include <numeric>
#include <set>
#include <csignal>
#include <sys/stat.h>

#include "imatrix-safeguards.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  include <io.h>
#  define getpid _getpid
#else
#  include <unistd.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s \\\n"
            "       -m model.gguf -f some-text.txt [-o imatrix.gguf] [--output-format {gguf,dat}] [--no-ppl] \\\n"
            "       [--process-output] [--chunk 123] [--save-frequency 0] [--output-frequency 10] \\\n"
            "       [--in-file imatrix-prev-0.gguf --in-file imatrix-prev-1.gguf ...] [--parse-special] \\\n"
            "       [--show-statistics] \\\n"
            "       [--model-draft drafter.gguf --spec-steps 64]   # spec-decoding calibration\n"
            "       [--features-out feats --feature-layers 0,15,31 --features-warmup 256]   # offline DFlash feature capture\n"
            "       [...]\n" , argv[0]);
    LOG("\n");
}

struct Stats {
    std::vector<float>   values;
    std::vector<float>   abs_values;
    std::vector<float>   fourth_values;
    std::vector<float>   max_values;
    std::vector<int64_t> counts;
};

struct tensor_statistics {
    std::string tensor;
    Stats stats;
    float total_sqract = 0.0f;
    float mean_sqract  = 0.0f;
    float max_sqract   = 0.0f;
    float min_sqract   = 0.0f;
    int elements       = 0;
    float stddev       = 0.0f;
    float active       = 0.0f;
    float entropy      = 0.0f;
    float zd           = 0.0f;
    float cossim       = 0.0f;
};

struct observer_transfer_state {
    std::vector<double> previous_moments;
    std::vector<int64_t> previous_counts;
    std::vector<float> signature;
    int32_t stable_windows = 0;
    int32_t frozen_at = 0;
    int32_t next_probe = 0;
    bool frozen = false;
    bool probe_active = false;
};

class IMatrixCollector {
public:
    IMatrixCollector() = default;
    void set_params(common_params params) { m_params = std::move(params); }
    // Set the scope this collector attributes incoming observers to. Pair with
    // llama_set_imatrix_observer_scope on the same context so the filter and
    // the bucketing agree.
    void set_observer_scope(int scope) { m_observer_scope = scope; }

    // Stats are bucketed by (scope, name) so a verifier context and a drafter
    // context sharing one collector land in separate ledgers without any name
    // prefix on the observer tensors. The scope mirrors the scope the owning
    // llama_context selected via llama_set_imatrix_observer_scope.
    struct stats_key {
        int         scope;
        std::string name;
        bool operator==(const stats_key & o) const {
            return scope == o.scope && name == o.name;
        }
    };
    struct stats_key_hash {
        size_t operator()(const stats_key & k) const {
            return std::hash<int>()(k.scope) ^
                   (std::hash<std::string>()(k.name) << 1);
        }
    };
    // Serialize a (scope, name) key to the tensor name written to disk. The
    // verifier scope keeps the bare name (unchanged on-disk contract for
    // verifier-only runs); the drafter scope is tagged so the two ledgers do
    // not collide. This tag lives only in the collector, never in the graph
    // build, so it carries no forward-compat risk if new drafter archs appear.
    static std::string stats_key_name(const stats_key & key) {
        if (key.scope == LLAMA_OBSERVER_SCOPE_VERIFIER) {
            return key.name;
        }
        return std::string("dft.") + key.name;
    }
    // Inverse of stats_key_name: recover the (scope, name) bucket from a
    // tensor name read off disk. Names carrying the drafter tag land in the
    // drafter scope; everything else is verifier.
    static stats_key parse_stats_key_name(const std::string & full) {
        static const std::string k_drafter_tag = "dft.";
        if (full.compare(0, k_drafter_tag.size(), k_drafter_tag) == 0) {
            return { LLAMA_OBSERVER_SCOPE_DRAFTER, full.substr(k_drafter_tag.size()) };
        }
        return { LLAMA_OBSERVER_SCOPE_VERIFIER, full };
    }
    bool collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data);
    void begin_graph_observers();
    bool collect_graph_observers();
    bool flush_graph_observers();
    void log_observer_performance();
    bool update_progressive_transfer(
            int32_t chunks_processed,
            bool allow_freeze,
            float * delta_out);
    void set_observer_chunk(int32_t chunk);
    bool take_observer_topology_changed();
    bool observer_enabled(const char * tensor_name);
    static bool observer_filter(const char * tensor_name, void * user_data);
    void save_imatrix_legacy(int32_t ncall = -1) const;
    void save_imatrix(int32_t n_chunk = -1) const;
    bool load_imatrix(const char * file_name);
    const std::unordered_map<stats_key, Stats, stats_key_hash> & get_mstats() const { return m_stats; }
private:
    static constexpr size_t k_observer_slots = 2;
    struct graph_observer_snapshot {
        std::string names;
        std::string backend_name;
        int64_t channels;
        int64_t experts;
        size_t offset;
        int scope;  // enum llama_observer_scope the snapshot was collected under
    };

    bool reduce_graph_observers(
            std::vector<graph_observer_snapshot> snapshots,
            const std::vector<float> & staging);
    bool flush_graph_observer_slot(size_t slot);
    void save_transfer_ledger(
            const std::string & imatrix_path,
            int32_t checkpoint_chunk) const;
    bool load_transfer_ledger(
            const std::string & imatrix_path,
            int32_t checkpoint_chunk);
    std::unordered_map<stats_key, Stats, stats_key_hash> m_stats;
    // Scope the collector attributes incoming observers to. The imatrix tool
    // sets this together with llama_set_imatrix_observer_scope so the two
    // stay in sync without the collector having to inspect the graph.
    int m_observer_scope = LLAMA_OBSERVER_SCOPE_VERIFIER;
    common_params                          m_params;
    std::mutex                             m_mutex;
    std::vector<std::string>               m_datasets;
    int32_t                                m_last_chunk = 0;
    std::vector<char>                      m_src1_data;
    std::vector<char>                      m_ids; // the expert ids from ggml_mul_mat_id
    std::vector<ggml_tensor *>             m_graph_observers;
    std::set<std::string>                  m_observer_backends;
    // Two slots let CPU reduction overlap the following Metal graph while
    // keeping staging and completion ownership one-to-one.
    std::array<std::vector<float>, k_observer_slots> m_observer_staging;
    std::vector<size_t>                    m_observer_offsets;
    std::array<std::future<bool>, k_observer_slots> m_observer_reduction;
    size_t                                  m_observer_staging_slot = 0;
    double                                  m_observer_copy_seconds = 0.0;
    double                                  m_observer_reduce_seconds = 0.0;
    double                                  m_observer_slot_wait_seconds = 0.0;
    uint64_t                                m_observer_copy_bytes = 0;
    uint64_t                                m_observer_batches = 0;
    std::unordered_map<std::string, observer_transfer_state> m_transfer;
    int32_t                                m_observer_chunk = 0;
    bool                                   m_observer_topology_changed = false;
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

static int32_t imatrix_op_param_i32(
        const ggml_tensor * tensor, size_t index) {
    int32_t value;
    memcpy(&value, tensor->op_params + index * sizeof(value), sizeof(value));
    return value;
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

static void compute_statistics(std::vector<tensor_statistics> & tstats, const std::string & name, const Stats & e) {
    if (e.values.size() % e.counts.size() != 0) {
        LOG_ERR("%s: activation size mismatch for tensor %s (%zu vs %zu)\n", __func__, name.c_str(), e.counts.size(), e.values.size());
        return;
    }
    if (e.counts.empty()) {
        LOG_ERR("%s: there are no activations for tensor %s. The imatrix may be suboptimal\n", __func__, name.c_str());
        return;
    }

    const int n_mat = e.counts.size();
    const int row_size = e.values.size() / n_mat;

    std::vector<float> activations;
    activations.reserve(e.values.size());

    for (int i = 0; i < n_mat; ++i) {
        if (e.counts[i] == 0) {
            LOG_DBG("%s: skipping tensor %s due to zero count at index %d\n", __func__, name.c_str(), i);
            continue;
        }
        for (int j = 0; j < row_size; ++j) {
            activations.push_back(e.values[i*row_size + j] / e.counts[i]);
        }
    }

    if (activations.empty()) {
        LOG_ERR("%s: all counts are zero for tensor %s, skipping statistics computation\n", __func__, name.c_str());
        return;
    }

    const float act_total     = std::accumulate(activations.begin(), activations.end(), 0.0f);
    const float act_max       = *std::max_element(activations.begin(), activations.end());
    const float act_min       = *std::min_element(activations.begin(), activations.end());
    const float act_mean      = act_total / activations.size();
    const float act_sqr_total = std::inner_product(activations.begin(), activations.end(), activations.begin(), 0.0f);
    const float act_var       = (act_sqr_total / activations.size()) - (act_mean * act_mean);
    const float act_dev       = std::sqrt(std::max(0.0f, act_var));
    float threshold           = 1e-5f;
    const int inactive_count  = std::count_if(activations.begin(), activations.end(),
                                               [threshold](const float v) { return fabsf(v) <= threshold; });
    const float active_ratio  = 1 - static_cast<float>(inactive_count) / activations.size();

    float entropy = 0;
    if (act_total > 0) {
        for (const auto act : activations) {
            if (const float p = act / act_total; p > 0) {
                entropy -= p * std::log2(p);
            }
        }
    }

    int z_score = 0;
    if (act_dev > 0.0f) {
        for (const auto act : activations) {
            if (const float p = (act - act_mean) / act_dev; p > 1) {
                z_score++;
            }
        }
    }

    auto & ts = tstats.emplace_back();
    ts.tensor     = name;
    ts.stats      = e;
    ts.total_sqract = act_total;
    ts.mean_sqract  = act_mean;
    ts.max_sqract   = act_max;
    ts.min_sqract   = act_min;
    ts.elements   = static_cast<int>(activations.size());
    ts.stddev     = act_dev;
    ts.active     = active_ratio;
    ts.entropy    = entropy;
    ts.zd         = static_cast<float>(z_score) / ts.elements;
}

static void compute_cossim(std::vector<tensor_statistics> & tstats) {
    static const std::regex pattern(R"(blk\.(\d+)\.)");
    for (auto & ts : tstats) {
        if (std::smatch match; std::regex_search(ts.tensor, match, pattern)) {
            const int blk = std::stoi(match[1]);
            std::string tname(ts.tensor);
            tname.replace(match.position(1), match.length(1), std::to_string(blk-1));
            auto prev = std::find_if(tstats.begin(), tstats.end(),
                [tname](const tensor_statistics & t) { return t.tensor == tname; });
            if (prev != tstats.end()) {
                const float dp = std::inner_product(ts.stats.values.begin(), ts.stats.values.end(),
                    prev->stats.values.begin(), 0.0f);
                const float curr_mag = std::sqrt(std::inner_product(ts.stats.values.begin(), ts.stats.values.end(),
                    ts.stats.values.begin(), 0.0f));
                const float prev_mag = std::sqrt(std::inner_product(prev->stats.values.begin(), prev->stats.values.end(),
                    prev->stats.values.begin(), 0.0f));
                const float cs = dp / (curr_mag * prev_mag);
                ts.cossim = cs;
            }
        } else {
            ts.cossim = 0;
        }
    }
}

bool IMatrixCollector::collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data) {
    if (m_params.imatrix_observers) {
        const bool fused_stats_view =
            ggml_imatrix_observer_is_stats(t);
        const bool regular_observer =
            t->op == GGML_OP_IMATRIX_OBSERVER &&
            t->type == GGML_TYPE_F32 &&
            imatrix_op_param_i32(t, 1) == 0;
        if (ask && (regular_observer ||
                    fused_stats_view)) {
            m_graph_observers.push_back(t);
            return false;
        }
        if (ask) {
            // Synchronize once at the ordinary graph output, after all compact
            // observer nodes for this internal ubatch have executed.
            return (t->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                   !m_graph_observers.empty();
        }
        const bool ok = collect_graph_observers();
        m_graph_observers.clear();
        return ok;
    }
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
        // The classification is intentionally broad: any tensor whose name
        // starts with "blk." is in scope. This covers all per-layer
        // attention (attn_q/attn_k/attn_v/attn_output) and FFN tensors
        // (ffn_gate/ffn_up/ffn_down) for every Llama-family block
        // (blk.0.ffn_down.weight, blk.7.ffn_gate.weight, etc.), as well
        // as the per-expert / per-ternary variants (ffn_*_exps,
        // ffn_*_shexp). ffn_down is NOT special-cased; it is in scope
        // because it shares the "blk." prefix with every other per-layer
        // weight. The downstream tensor_category::FFN_DOWN branch in
        // src/llama-quant.cpp and the
        // blk\.\d*\.ffn_down(_exps)?.weight pattern in src/llama-model.cpp
        // are the only places that name-check ffn_down explicitly; the
        // collector itself is name-agnostic above the blk./output split.
        // tests/test-imatrix-ffn-down-coverage.cpp pins this.
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

        auto & e = m_stats[stats_key{ m_observer_scope, wname }];

        if (e.counts.size() == 1 && n_as > 1) {
            // broadcast, when loading an old imatrix
            e.counts.resize(n_as, e.counts[0]);
        }
        if (e.values.empty()) {
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
        auto & e = m_stats[stats_key{ m_observer_scope, wname }];
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

void IMatrixCollector::begin_graph_observers() {
    m_graph_observers.clear();
}

bool IMatrixCollector::collect_graph_observers() {
    // Reuse one of two persistent arenas. The other arena may still be
    // reducing the preceding graph while Metal executes this graph.
    // Keep producer and completion ownership in the same bounded ring.  This
    // must be derived from the completion array: a stale binary once had
    // three staging vectors and two futures, which allowed slot 2 to escape
    // the reduction ring after two graphs.
    static_assert(k_observer_slots == std::tuple_size_v<decltype(m_observer_reduction)>);
    const size_t slot = m_observer_staging_slot++ % m_observer_reduction.size();
    if (!flush_graph_observer_slot(slot)) {
        return false;
    }
    // Pack all compact observer outputs into one reusable host arena. Metal
    // uses shared buffers on Apple silicon, so the tensor gets are ordinary
    // bounded copies after the graph-level synchronization rather than
    // individual GPU readback transactions.
    const auto copy_start = std::chrono::steady_clock::now();
    m_observer_offsets.resize(m_graph_observers.size() + 1);
    m_observer_offsets[0] = 0;
    for (size_t i = 0; i < m_graph_observers.size(); ++i) {
        ggml_tensor * observer = m_graph_observers[i];
        const bool fused_cast =
            ggml_imatrix_observer_is_stats(observer);
        GGML_UNUSED(fused_cast);
        const size_t compact_elements = ggml_nelements(observer);
        m_observer_offsets[i + 1] =
            m_observer_offsets[i] + compact_elements;
    }
    std::vector<float> & staging = m_observer_staging[slot];
    staging.resize(m_observer_offsets.back());
    for (size_t i = 0; i < m_graph_observers.size(); ++i) {
        ggml_tensor * observer = m_graph_observers[i];
        const bool fused_cast =
            ggml_imatrix_observer_is_stats(observer);
        GGML_UNUSED(fused_cast);
        const size_t read_bytes =
            (m_observer_offsets[i + 1] - m_observer_offsets[i]) *
            sizeof(float);
        if (read_bytes > ggml_nbytes(observer)) {
            LOG_ERR(
                "%s: compact observer read exceeds logical tensor '%s': "
                "requested=%zu, available=%zu, op=%s, type=%s, view=%d\n",
                __func__, observer->name, read_bytes,
                ggml_nbytes(observer), ggml_op_name(observer->op),
                ggml_type_name(observer->type),
                observer->view_src != nullptr);
            return false;
        }
        // On Apple silicon a shared Metal observer buffer is directly mapped
        // into this process.  Bypass the backend virtual dispatch in that
        // common case; non-host backends retain the portable tensor-get path.
        ggml_backend_buffer_t observer_buffer = observer->buffer;
        if (!observer_buffer && observer->view_src) {
            observer_buffer = observer->view_src->buffer;
        }
        if (observer_buffer && ggml_backend_buffer_is_host(observer_buffer)) {
            memcpy(
                staging.data() + m_observer_offsets[i],
                observer->data,
                read_bytes);
        } else {
            ggml_backend_tensor_get(
                observer,
                staging.data() + m_observer_offsets[i],
                0,
                read_bytes);
        }
    }
    const auto copy_end = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_observer_copy_seconds +=
            std::chrono::duration<double>(copy_end - copy_start).count();
        m_observer_copy_bytes += m_observer_offsets.back() * sizeof(float);
        ++m_observer_batches;
    }

    std::vector<graph_observer_snapshot> snapshots;
    snapshots.reserve(m_graph_observers.size());
    size_t observer_index = 0;
    for (ggml_tensor * observer : m_graph_observers) {
        const bool fused_cast =
            ggml_imatrix_observer_is_stats(observer);
        GGML_ASSERT(observer->op == GGML_OP_IMATRIX_OBSERVER ||
                    fused_cast);
        ggml_backend_buffer_t observer_buffer = observer->buffer;
        if (!observer_buffer && observer->view_src) {
            observer_buffer = observer->view_src->buffer;
        }
        const int64_t channels = fused_cast
            ? imatrix_op_param_i32(observer->view_src, 2)
            : (observer->ne[0] - 1) / 4;
        const int64_t experts = fused_cast ? 1 : observer->ne[1];
        GGML_ASSERT(fused_cast || (observer->ne[0] - 1) % 4 == 0);
        snapshots.push_back({
            filter_tensor_name(observer->name),
            ggml_backend_buffer_name(observer_buffer),
            channels,
            experts,
            m_observer_offsets[observer_index++],
            m_observer_scope,
        });
    }

    // The alternate arena remains available to the next callback. CPU
    // accumulation therefore overlaps the following Metal decode without
    // per-ubatch allocation churn.
    m_observer_reduction[slot] = std::async(
        std::launch::async,
        [this, slot, snapshots = std::move(snapshots)]() mutable {
            return reduce_graph_observers(
                std::move(snapshots), m_observer_staging[slot]);
        });
    return true;
}

bool IMatrixCollector::flush_graph_observer_slot(size_t slot) {
    GGML_ASSERT(slot < m_observer_reduction.size());
    if (!m_observer_reduction[slot].valid()) {
        return true;
    }
    const auto wait_start = std::chrono::steady_clock::now();
    const bool result = m_observer_reduction[slot].get();
    const double wait_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wait_start).count();
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_observer_slot_wait_seconds += wait_seconds;
    }
    return result;
}

bool IMatrixCollector::flush_graph_observers() {
    for (size_t slot = 0; slot < m_observer_reduction.size(); ++slot) {
        if (!flush_graph_observer_slot(slot)) {
            return false;
        }
    }
    return true;
}

bool IMatrixCollector::reduce_graph_observers(
        std::vector<graph_observer_snapshot> snapshots,
        const std::vector<float> & staging) {
    const auto reduce_start = std::chrono::steady_clock::now();
    const int32_t chunk_size = m_params.n_ctx / m_params.n_parallel;
    // Build a private delta while Metal starts the following graph.  Holding
    // m_mutex across every channel used to make the decode thread contend
    // with both async reduction slots merely to check an observer filter.
    // A short, deterministic merge below preserves the on-disk imatrix
    // contract while keeping that critical section proportional to the
    // compact statistics, not their parsing and allocation cost.
    std::unordered_map<stats_key, Stats, stats_key_hash> delta;
    std::set<std::string> backends;
    for (const graph_observer_snapshot & snapshot : snapshots) {
        backends.insert(snapshot.backend_name);
        const int64_t channels = snapshot.channels;
        const int64_t experts = snapshot.experts;
        const float * compact = staging.data() + snapshot.offset;
        const std::string & observer_names = snapshot.names;
        size_t name_begin = 0;
        while (name_begin <= observer_names.size()) {
            const size_t name_end = observer_names.find('|', name_begin);
            const std::string wname = observer_names.substr(
                name_begin,
                name_end == std::string::npos
                    ? std::string::npos
                    : name_end - name_begin);
            auto & entry = delta[stats_key{ snapshot.scope, wname }];
            if (entry.values.empty()) {
                entry.values.resize(channels * experts, 0.0f);
            }
            if (entry.abs_values.empty()) {
                entry.abs_values.resize(channels * experts, 0.0f);
            }
            if (entry.fourth_values.empty()) {
                entry.fourth_values.resize(channels * experts, 0.0f);
            }
            if (entry.max_values.empty()) {
                entry.max_values.resize(channels * experts, 0.0f);
            }
            if (entry.counts.empty()) {
                entry.counts.resize(experts, 0);
            }
            if (entry.values.size() != (size_t) (channels * experts) ||
                entry.abs_values.size() != (size_t) (channels * experts) ||
                entry.fourth_values.size() != (size_t) (channels * experts) ||
                entry.max_values.size() != (size_t) (channels * experts) ||
                entry.counts.size() != (size_t) experts) {
                LOG_ERR(
                    "%s: observer shape changed within graph for %s: "
                    "existing=[%zu,%zu] incoming=[%lld,%lld]\n",
                    __func__, wname.c_str(),
                    entry.values.size(), entry.counts.size(),
                    (long long) (channels * experts),
                    (long long) experts);
                return false;
            }

            for (int64_t expert = 0; expert < experts; ++expert) {
                const float * row =
                    compact + expert * (4 * channels + 1);
                for (int64_t channel = 0; channel < channels; ++channel) {
                    const int64_t index = expert * channels + channel;
                    entry.values[index] += row[channel];
                    entry.abs_values[index] += row[channels + channel];
                    entry.fourth_values[index] +=
                        row[2 * channels + channel];
                    entry.max_values[index] = std::max(
                        entry.max_values[index],
                        row[3 * channels + channel]);
                }
                entry.counts[expert] +=
                    (int64_t) llroundf(row[4 * channels]);
            }
            if (name_end == std::string::npos) {
                break;
            }
            name_begin = name_end + 1;
        }
    }

    const double reduce_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - reduce_start).count();
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const std::string & backend : backends) {
        if (m_observer_backends.insert(backend).second) {
            LOG_INF("%s: graph observers active on backend buffer '%s'\n",
                    __func__, backend.c_str());
        }
    }
    for (auto & [key, source] : delta) {
        auto & destination = m_stats[key];
        if (destination.values.empty()) {
            destination = Stats{
                std::vector<float>(source.values.size(), 0.0f),
                std::vector<float>(source.abs_values.size(), 0.0f),
                std::vector<float>(source.fourth_values.size(), 0.0f),
                std::vector<float>(source.max_values.size(), 0.0f),
                std::vector<int64_t>(source.counts.size(), 0),
            };
        }
        // Older checkpoints predate the rich observer moments.  Their
        // squared-activation sums and counts remain valid; introduce zeroed
        // accumulators for the newly observed moments so resume can continue
        // rather than misclassifying an absent optional field as a tensor
        // shape transition.
        if (destination.abs_values.empty() &&
            destination.fourth_values.empty() &&
            destination.max_values.empty() &&
            destination.values.size() == source.values.size()) {
            destination.abs_values.resize(source.abs_values.size(), 0.0f);
            destination.fourth_values.resize(source.fourth_values.size(), 0.0f);
            destination.max_values.resize(source.max_values.size(), 0.0f);
        }
        if (destination.values.size() != source.values.size() ||
            destination.abs_values.size() != source.abs_values.size() ||
            destination.fourth_values.size() != source.fourth_values.size() ||
            destination.max_values.size() != source.max_values.size() ||
            destination.counts.size() != source.counts.size()) {
            LOG_ERR(
                "%s: observer shape changed against accumulated state for %s: "
                "stored=[%zu,%zu] incoming=[%zu,%zu]\n",
                __func__, key.name.c_str(),
                destination.values.size(), destination.counts.size(),
                source.values.size(), source.counts.size());
            return false;
        }
        for (size_t i = 0; i < source.values.size(); ++i) {
            destination.values[i] += source.values[i];
            destination.abs_values[i] += source.abs_values[i];
            destination.fourth_values[i] += source.fourth_values[i];
            destination.max_values[i] = std::max(
                destination.max_values[i], source.max_values[i]);
        }
        for (size_t i = 0; i < source.counts.size(); ++i) {
            destination.counts[i] += source.counts[i];
            m_last_chunk = std::max(
                m_last_chunk,
                (int32_t) (destination.counts[i] / chunk_size));
        }
    }
    m_observer_reduce_seconds += reduce_seconds;
    return true;
}

void IMatrixCollector::log_observer_performance() {
    if (!flush_graph_observers()) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_observer_batches == 0) {
        return;
    }
    LOG_INF(
        "%s: observer batches=%llu payload=%.2f MiB copy=%.3fs reduce=%.3fs "
        "slot_wait=%.3fs\n",
        __func__, (unsigned long long) m_observer_batches,
        (double) m_observer_copy_bytes / (1024.0 * 1024.0),
        m_observer_copy_seconds, m_observer_reduce_seconds,
        m_observer_slot_wait_seconds);
}

bool IMatrixCollector::update_progressive_transfer(
        int32_t chunks_processed,
        bool allow_freeze,
        float * delta_out) {
    if (!flush_graph_observers()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    // Use the serialized (scope-tagged) name for both the transfer ledger
    // and the sort order; recover the in-memory bucket key for the lookup.
    std::vector<std::string> names;
    names.reserve(m_stats.size());
    for (const auto & item : m_stats) {
        names.push_back(stats_key_name(item.first));
    }
    std::sort(names.begin(), names.end());

    float worst_delta = 0.0f;
    size_t active = 0;
    for (const std::string & name : names) {
        const Stats & stats = m_stats.at(parse_stats_key_name(name));
        if (stats.values.empty() || stats.counts.empty() ||
            stats.abs_values.size() != stats.values.size() ||
            stats.fourth_values.size() != stats.values.size()) {
            continue;
        }
        const size_t channels = stats.values.size() / stats.counts.size();
        if (channels == 0) {
            continue;
        }
        std::vector<double> moments;
        std::vector<float> signature;
        moments.reserve(stats.counts.size() * 96);
        signature.reserve(stats.counts.size() * 96);
        const size_t stride = std::max<size_t>(1, channels / 32);
        for (size_t expert = 0; expert < stats.counts.size(); ++expert) {
            const size_t begin = expert * channels;
            for (size_t channel = 0; channel < channels; channel += stride) {
                const size_t index = begin + channel;
                moments.push_back(stats.values[index]);
                if (index < stats.abs_values.size()) {
                    moments.push_back(stats.abs_values[index]);
                }
                if (index < stats.fourth_values.size()) {
                    moments.push_back(stats.fourth_values[index]);
                }
            }
        }

        auto & state = m_transfer[name];
        if (moments.empty() ||
            moments.size() != state.previous_moments.size() ||
            stats.counts.size() != state.previous_counts.size()) {
            state.previous_moments = std::move(moments);
            state.previous_counts = stats.counts;
            state.signature.clear();
            state.stable_windows = 0;
            ++active;
            continue;
        }

        bool complete_window = true;
        size_t moment = 0;
        for (size_t expert = 0; expert < stats.counts.size(); ++expert) {
            const int64_t window_count =
                stats.counts[expert] - state.previous_counts[expert];
            if (window_count <= 0) {
                complete_window = false;
            }
            for (size_t channel = 0; channel < channels; channel += stride) {
                GGML_UNUSED(channel);
                for (int statistic = 0; statistic < 3; ++statistic) {
                    const double window_sum =
                        moments[moment] - state.previous_moments[moment];
                    signature.push_back(log1pf(fabsf((float) (
                        window_sum / std::max<int64_t>(1, window_count)))));
                    ++moment;
                }
            }
        }
        state.previous_moments = std::move(moments);
        state.previous_counts = stats.counts;
        if (!complete_window) {
            ++active;
            continue;
        }
        const auto [min_count_it, max_count_it] = std::minmax_element(
            stats.counts.begin(), stats.counts.end());
        const float expert_coverage = stats.counts.size() <= 1
            ? 1.0f
            : (float) *min_count_it / std::max<int64_t>(1, *max_count_it);
        const bool coverage_ready = expert_coverage >=
            m_params.imatrix_min_expert_coverage;
        if (signature.size() != state.signature.size()) {
            state.signature = std::move(signature);
            state.stable_windows = 0;
            ++active;
            continue;
        }

        double squared_delta = 0.0;
        for (size_t i = 0; i < signature.size(); ++i) {
            const double delta =
                (double) signature[i] - state.signature[i];
            squared_delta += delta * delta;
        }
        const float rms_delta = (float) sqrt(
            squared_delta / std::max<size_t>(1, signature.size()));
        worst_delta = std::max(worst_delta, rms_delta);
        state.signature = std::move(signature);

        if (state.frozen) {
            if (chunks_processed >= state.next_probe) {
                if (rms_delta >
                    m_params.imatrix_convergence_tolerance * 2.0f) {
                    state.frozen = false;
                    state.probe_active = false;
                    state.stable_windows = 0;
                    LOG_INF(
                        "%s: reopened observer %s at chunk %d "
                        "(probe rms_delta=%.7g)\n",
                        __func__, name.c_str(), chunks_processed, rms_delta);
                } else {
                    state.next_probe = chunks_processed +
                        4 * m_params.imatrix_convergence_interval;
                    state.probe_active = false;
                    m_observer_topology_changed = true;
                }
            }
        } else if (rms_delta <= m_params.imatrix_convergence_tolerance) {
            ++state.stable_windows;
            if (allow_freeze &&
                coverage_ready &&
                state.stable_windows >=
                    m_params.imatrix_convergence_patience) {
                state.frozen = true;
                state.frozen_at = chunks_processed;
                state.next_probe = chunks_processed +
                    4 * m_params.imatrix_convergence_interval;
                state.probe_active = false;
                m_observer_topology_changed = true;
                LOG_INF(
                    "%s: froze transferable observer %s at chunk %d "
                    "(rms_delta=%.7g, expert_coverage=%.3f)\n",
                    __func__, name.c_str(), chunks_processed, rms_delta,
                    expert_coverage);
            }
        } else {
            state.stable_windows = 0;
        }
        if (allow_freeze && !coverage_ready && stats.counts.size() > 1) {
            LOG_DBGV(2,
                "%s: retaining routed observer %s at chunk %d "
                "(expert_coverage=%.3f < %.3f)\n",
                __func__, name.c_str(), chunks_processed, expert_coverage,
                m_params.imatrix_min_expert_coverage);
        }
        if (!state.frozen) {
            ++active;
        }
    }

    if (delta_out) {
        *delta_out = names.empty() ? INFINITY : worst_delta;
    }
    const size_t frozen = m_transfer.size() - active;
    LOG_INF(
        "%s: progressive observer transfer at chunk %d: "
        "frozen=%zu active=%zu\n",
        __func__, chunks_processed, frozen, active);
    return allow_freeze && !m_transfer.empty() && active == 0;
}

void IMatrixCollector::set_observer_chunk(int32_t chunk) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_observer_chunk = chunk;
    for (auto & [name, state] : m_transfer) {
        GGML_UNUSED(name);
        if (state.frozen && !state.probe_active &&
            chunk >= state.next_probe) {
            state.probe_active = true;
            m_observer_topology_changed = true;
        }
    }
}

bool IMatrixCollector::take_observer_topology_changed() {
    std::lock_guard<std::mutex> lock(m_mutex);
    const bool changed = m_observer_topology_changed;
    m_observer_topology_changed = false;
    return changed;
}

bool IMatrixCollector::observer_enabled(const char * tensor_name) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const std::string name = filter_tensor_name(tensor_name);
    auto found = m_transfer.find(name);
    if (found == m_transfer.end() || !found->second.frozen) {
        return true;
    }
    return m_observer_chunk >= found->second.next_probe;
}

bool IMatrixCollector::observer_filter(
        const char * tensor_name,
        void * user_data) {
    return static_cast<IMatrixCollector *>(user_data)->
        observer_enabled(tensor_name);
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
    // Pairs of (serialized on-disk name, in-memory bucket key) so the sort
    // orders by the written name while the stats lookup uses the (scope, name)
    // key the collector buckets on.
    std::vector<std::pair<std::string, stats_key>> to_store;

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
            LOG_WRN("%s: entry '%40s' has no data - skipping\n", __func__, kv.first.name.c_str());
            continue;
        }

        if (n_zeros > 0) {
            LOG_WRN("%s: entry '%40s' has partial data (%.2f%%)\n", __func__, kv.first.name.c_str(), 100.0f * (n_all - n_zeros) / n_all);
        }

        n_entries++;
        to_store.push_back({ stats_key_name(kv.first), kv.first });
    }

    if (to_store.size() < m_stats.size()) {
        LOG_WRN("%s: storing only %zu out of %zu entries\n", __func__, to_store.size(), m_stats.size());
    }

    // deterministic tensor name order
    std::sort(to_store.begin(), to_store.end(),
        [](const std::pair<std::string, stats_key> & a,
           const std::pair<std::string, stats_key> & b) {
            return a.first < b.first;
        });

    const int32_t chunk_size = m_params.n_ctx / m_params.n_parallel;

    std::ofstream out(fname, std::ios::binary);
    out.write((const char *) &n_entries, sizeof(n_entries));
    for (const auto & [name, key] : to_store) {
        const auto & stat = m_stats.at(key);
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

    std::vector<std::pair<std::string, stats_key>> to_store;
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
            LOG_WRN("%s: entry '%40s' has partial data (%.2f%%)\n", __func__, kv.first.name.c_str(), 100.0f * (n_all - n_zeros) / n_all);
        }

        to_store.push_back({ stats_key_name(kv.first), kv.first });
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * kv.second.values.size(), GGML_MEM_ALIGN);
        if (!kv.second.abs_values.empty()) {
            data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * kv.second.abs_values.size(), GGML_MEM_ALIGN);
            data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * kv.second.fourth_values.size(), GGML_MEM_ALIGN);
            data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * kv.second.max_values.size(), GGML_MEM_ALIGN);
        }
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * kv.second.counts.size(), GGML_MEM_ALIGN);
    }

    // deterministic tensor name order
    std::sort(to_store.begin(), to_store.end(),
        [](const std::pair<std::string, stats_key> & a,
           const std::pair<std::string, stats_key> & b) {
            return a.first < b.first;
        });

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

    for (const auto & [name, key] : to_store) {
        const auto & stat = m_stats.at(key);
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

            if (stat.abs_values.size() == stat.values.size() &&
                stat.fourth_values.size() == stat.values.size() &&
                stat.max_values.size() == stat.values.size()) {
                struct ggml_tensor * in_sumabs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nval / nmat, nmat);
                struct ggml_tensor * in_sum4   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nval / nmat, nmat);
                struct ggml_tensor * in_maxabs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nval / nmat, nmat);
                ggml_format_name(in_sumabs, "%s.in_sumabs", name.c_str());
                ggml_format_name(in_sum4,   "%s.in_sum4",   name.c_str());
                ggml_format_name(in_maxabs, "%s.in_maxabs", name.c_str());
                for (int32_t j = 0; j < nval; ++j) {
                    ((float *) in_sumabs->data)[j] = stat.abs_values[j];
                    ((float *) in_sum4->data)[j]   = stat.fourth_values[j];
                    ((float *) in_maxabs->data)[j] = stat.max_values[j];
                }
                gguf_add_tensor(ctx_gguf, in_sumabs);
                gguf_add_tensor(ctx_gguf, in_sum4);
                gguf_add_tensor(ctx_gguf, in_maxabs);
            }
        }
    }

    // Crash-safe publish: write to .tmp, then atomic rename. A SIGKILL or
    // jetsam mid-write leaves the previous good copy at `fname` intact
    // (POSIX rename is atomic on the same filesystem; mirrors the
    // transfer-ledger pattern at the bottom of this file).
    const std::string fname_tmp = fname + ".tmp";
    const bool write_ok = gguf_write_to_file(ctx_gguf, fname_tmp.c_str(), false);
    bool publish_ok = false;
    if (write_ok) {
        if (std::rename(fname_tmp.c_str(), fname.c_str()) != 0) {
            LOG_ERR("%s: failed to publish imatrix %s (rename from %s)\n",
                    __func__, fname.c_str(), fname_tmp.c_str());
            std::remove(fname_tmp.c_str());
        } else {
            publish_ok = true;
        }
    } else {
        LOG_ERR("%s: failed to write imatrix to %s\n", __func__, fname_tmp.c_str());
        std::remove(fname_tmp.c_str());
    }

    if (publish_ok) {
        LOGV(1, "\n");
        LOG_DBGV(1, "%s: stored collected data after %d chunks in %s\n",
                 __func__, m_last_chunk, fname.c_str());
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    save_transfer_ledger(
        fname,
        n_chunk > 0 ? n_chunk : m_last_chunk);
}

void IMatrixCollector::save_transfer_ledger(
        const std::string & imatrix_path,
        int32_t checkpoint_chunk) const {
    nlohmann::ordered_json root = {
        { "schema", "llama.tessera.progressive-observer-ledger.v1" },
        { "checkpoint_chunk", checkpoint_chunk },
        { "convergence_interval", m_params.imatrix_convergence_interval },
        { "convergence_patience", m_params.imatrix_convergence_patience },
        { "convergence_tolerance", m_params.imatrix_convergence_tolerance },
        { "tensors", nlohmann::ordered_json::object() },
    };
    size_t frozen = 0;
    for (const auto & [name, state] : m_transfer) {
        root["tensors"][name] = {
            { "previous_moments", state.previous_moments },
            { "previous_counts", state.previous_counts },
            { "signature", state.signature },
            { "stable_windows", state.stable_windows },
            { "frozen_at", state.frozen_at },
            { "next_probe", state.next_probe },
            { "frozen", state.frozen },
            { "probe_active", state.probe_active },
        };
        frozen += state.frozen ? 1 : 0;
    }
    root["frozen_tensors"] = frozen;
    root["total_tensors"] = m_transfer.size();
    root["frozen_fraction"] = m_transfer.empty()
        ? 0.0
        : (double) frozen / m_transfer.size();

    const std::string path = imatrix_path + ".transfer.json";
    const std::string temporary = path + ".tmp";
    {
        std::ofstream output(temporary, std::ios::trunc);
        if (!output) {
            LOG_ERR("%s: failed to create %s\n", __func__, temporary.c_str());
            return;
        }
        output << root.dump(2) << '\n';
        if (!output) {
            LOG_ERR("%s: failed to write %s\n", __func__, temporary.c_str());
            return;
        }
    }
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        LOG_ERR("%s: failed to publish %s\n", __func__, path.c_str());
        return;
    }
    LOG_INF(
        "%s: saved progressive ledger %s (frozen=%zu/%zu)\n",
        __func__, path.c_str(), frozen, m_transfer.size());
}

bool IMatrixCollector::load_transfer_ledger(
        const std::string & imatrix_path,
        int32_t checkpoint_chunk) {
    const std::string path = imatrix_path + ".transfer.json";
    std::ifstream input(path);
    if (!input) {
        LOG_INF(
            "%s: no progressive ledger beside %s; seeding fresh windows\n",
            __func__, imatrix_path.c_str());
        return true;
    }
    try {
        const nlohmann::ordered_json root =
            nlohmann::ordered_json::parse(input);
        if (root.value("schema", std::string()) !=
                "llama.tessera.progressive-observer-ledger.v1" ||
            root.value("checkpoint_chunk", -1) != checkpoint_chunk) {
            LOG_ERR("%s: incompatible progressive ledger %s\n",
                    __func__, path.c_str());
            return false;
        }
        std::unordered_map<std::string, observer_transfer_state> loaded;
        for (auto item : root.at("tensors").items()) {
            const auto & value = item.value();
            observer_transfer_state state;
            state.previous_moments =
                value.at("previous_moments").get<std::vector<double>>();
            state.previous_counts =
                value.at("previous_counts").get<std::vector<int64_t>>();
            state.signature =
                value.at("signature").get<std::vector<float>>();
            state.stable_windows = value.value("stable_windows", 0);
            state.frozen_at = value.value("frozen_at", 0);
            state.next_probe = value.value("next_probe", 0);
            state.frozen = value.value("frozen", false);
            state.probe_active = value.value("probe_active", false);
            loaded.emplace(item.key(), std::move(state));
        }
        m_transfer = std::move(loaded);
        m_observer_topology_changed = !m_transfer.empty();
        LOG_INF(
            "%s: restored progressive ledger %s (%zu tensors)\n",
            __func__, path.c_str(), m_transfer.size());
        return true;
    } catch (const std::exception & error) {
        LOG_ERR("%s: failed to parse %s: %s\n",
                __func__, path.c_str(), error.what());
        return false;
    }
}

bool IMatrixCollector::load_imatrix(const char * file_name) {
    common_imatrix loaded;
    if (!common_imatrix_load(file_name, loaded)) {
        return false;
    }

    const int32_t chunk_size = m_params.n_ctx / m_params.n_parallel;
    const bool is_legacy = loaded.is_legacy;

    for (auto & [name, entry] : loaded.entries) {
        auto & e = m_stats[parse_stats_key_name(name)];

        if (is_legacy) {
            // Legacy format: sums contain (raw_sum/raw_count)*ncall, counts contain {ncall}
            // Reconstruct raw form by multiplying by chunk_size
            if (e.values.empty()) {
                e.values.resize(entry.sums.size(), 0.0f);
                e.counts.resize(1, 0);
            }
            for (size_t j = 0; j < entry.sums.size(); ++j) {
                e.values[j] += entry.sums[j] * chunk_size;
            }
            for (size_t j = 0; j < e.counts.size(); ++j) {
                e.counts[j] += entry.counts[0] * chunk_size;
            }
        } else {
            // GGUF format: raw sums and counts, accumulate directly
            const int64_t nval    = entry.sums.size();
            const int64_t ncounts = entry.counts.size();

            if (e.values.empty()) {
                e.values.resize(nval, 0.0f);
            } else if ((size_t) nval != e.values.size()) {
                LOG_ERR("%s: mismatched sums size for %s: %zu != %zu\n", __func__, name.c_str(), (size_t) nval, e.values.size());
                return false;
            }

            if (e.counts.empty()) {
                e.counts.resize(ncounts, 0);
            } else if (e.counts.size() == 1 && ncounts > 1) {
                e.counts.resize(ncounts, e.counts[0]);
            } else if ((size_t) ncounts != e.counts.size()) {
                LOG_ERR("%s: mismatched counts size for %s: %zu != %zu\n", __func__, name.c_str(), (size_t) ncounts, e.counts.size());
                return false;
            }

            for (int64_t j = 0; j < nval; ++j) {
                e.values[j] += entry.sums[j];
            }
            if (!entry.abs_sums.empty()) {
                if (e.abs_values.empty()) {
                    e.abs_values.resize(nval, 0.0f);
                    e.fourth_values.resize(nval, 0.0f);
                    e.max_values.resize(nval, 0.0f);
                }
                if (entry.abs_sums.size() != (size_t) nval ||
                    entry.fourth_sums.size() != (size_t) nval ||
                    entry.max_abs.size() != (size_t) nval) {
                    LOG_ERR(
                        "%s: mismatched rich moment size for %s\n",
                        __func__, name.c_str());
                    return false;
                }
                for (int64_t j = 0; j < nval; ++j) {
                    e.abs_values[j] += entry.abs_sums[j];
                    e.fourth_values[j] += entry.fourth_sums[j];
                    e.max_values[j] = std::max(
                        e.max_values[j], entry.max_abs[j]);
                }
            }
            for (int64_t j = 0; j < ncounts; ++j) {
                e.counts[j] += entry.counts[j];
            }
        }
    }

    m_datasets.insert(m_datasets.end(), loaded.datasets.begin(), loaded.datasets.end());

    // Calculate the last chunk count
    int64_t max_count = 0;
    for (const auto & stats : m_stats) {
        for (int64_t count : stats.second.counts) {
            if (count > max_count) {
                max_count = count;
            }
        }
    }
    m_last_chunk = max_count / chunk_size;

    return load_transfer_ledger(file_name, loaded.chunk_count);
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

    if (llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_LAST) {
        GGML_ASSERT(!llama_vocab_get_add_eos(vocab));
    }

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

    // Wall-time anchor for the dynamic save-frequency ladder. The first
    // chunk's per-iteration t_start would include model load + first
    // decode; we anchor t=0 at the start of the chunk loop so the ladder
    // reflects runtime stability, not startup.
    const auto t_loop_start = std::chrono::steady_clock::now();

    for (int i = 0; i < n_chunk; i += n_seq) {
        const int start =     i * n_ctx;
        const int end   = start + n_ctx;

        const int n_seq_batch = std::min(n_seq, n_chunk - i);
        const int32_t chunks_processed =
            params.i_chunk + i + n_seq_batch;
        g_collector.set_observer_chunk(chunks_processed);
        const double elapsed_sec =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_loop_start).count();
        if (g_collector.take_observer_topology_changed()) {
            llama_bump_imatrix_observer_epoch(ctx);
            LOG_INF("%s: observer topology transition at chunk %d; "
                    "rebuilding once for the new mask\n",
                    __func__, chunks_processed);
        }

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

            g_collector.begin_graph_observers();
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

        if (params.imatrix_observers &&
            params.imatrix_convergence_min_chunks > 0 &&
            chunks_processed % params.imatrix_convergence_interval == 0) {
            float convergence_delta = INFINITY;
            const bool allow_freeze =
                chunks_processed >=
                params.imatrix_convergence_min_chunks;
            const bool converged =
                g_collector.update_progressive_transfer(
                    chunks_processed, allow_freeze, &convergence_delta);
            if (g_collector.take_observer_topology_changed()) {
                llama_bump_imatrix_observer_epoch(ctx);
                LOG_INF("%s: observer topology transition after chunk %d; "
                        "rebuilding once for the new mask\n",
                        __func__, chunks_processed);
            }
            g_collector.log_observer_performance();
            LOG_INF(
                "%s: observer convergence at chunk %d: rms_delta=%.7g, "
                "tolerance=%.7g%s\n",
                __func__, chunks_processed, convergence_delta,
                params.imatrix_convergence_tolerance,
                converged ? " (stable; stopping)" : "");
            if (converged) {
                g_collector.save_imatrix(chunks_processed);
                break;
            }
        }
        if (params.n_out_freq > 0 &&
            chunks_processed % params.n_out_freq == 0) {
            if (!g_collector.flush_graph_observers()) {
                return false;
            }
            g_collector.save_imatrix();
        }
        // Phase 16.9: dynamic save-frequency ladder. When the user did
        // not pass --save-frequency (n_save_freq == 0) AND
        // dynamic_save_freq is on (the default), consult the ladder
        // instead. The ladder starts paranoid (every 8 chunks) and
        // relaxes as runtime stability is proven over wall time. A
        // SIGKILL or jetsam during the run therefore loses at most one
        // ladder-step's worth of work; the prior checkpoint is intact
        // thanks to the atomic .tmp + rename in save_imatrix.
        int32_t effective_save_freq = params.n_save_freq;
        if (effective_save_freq == 0 && params.dynamic_save_freq) {
            effective_save_freq = tessera_imatrix_safeguards::dynamic_save_freq_ladder(elapsed_sec);
        }
        if (effective_save_freq > 0 &&
            chunks_processed % effective_save_freq == 0) {
            if (!g_collector.flush_graph_observers()) {
                return false;
            }
            g_collector.save_imatrix(chunks_processed);
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

    if (!g_collector.flush_graph_observers()) {
        return false;
    }

    return true;
}

// Offline trunk-feature capture for DFlash drafter training (Path 1 in
// docs/tessera-dflash-training-design.md).
//
// This is a dedicated trunk-only forward pass, deliberately separate from the
// speculative telemetry loop above: that loop interleaves drafter forwards,
// KV trimming, and output_reorder in ways that make hidden-state readback
// fragile. Here we run the plain chunked forward (same structure as
// compute_imatrix) with per-layer input-embedding capture enabled, and stream
// the target-layer hidden states to disk token-major.
//
// Output: <features_out>.bin (raw [n_tokens, n_layers*n_embd], f32) plus
// <features_out>.json (header). The drafter encoder's FC input is a flat read
// of one row per token, layers concatenated in --feature-layers order.
//
// Note on context: like compute_imatrix, the KV cache is cleared per window, so
// a window's first tokens are processed without a full left window. We run
// those "warmup" tokens to build context but do NOT emit their features
// (--features-warmup, default 256); only the well-contextualized tail of each
// window is written. Windows advance by stride = n_ctx - warmup (NOT n_ctx), so
// consecutive windows overlap by exactly `warmup` tokens: each window re-decodes
// the previous window's tail to prime its KV, and the emitted rows form ONE
// contiguous corpus sequence (row r == corpus token warmup + r) instead of
// dropping a warmup prefix per window. Every emitted token still sees >= warmup
// genuine left-context tokens, now spanning the window boundary. This matches
// the steady-state inference regime (the trunk also conditions on a ~n_ctx
// window) and how EAGLE-style feature capture is done, at a decode overhead of
// n_ctx/(n_ctx - warmup) (~1.07x at warmup=256, n_ctx=4096). The warmup, window
// size, and stride are recorded in the header so the training driver can map
// feature rows back to corpus tokens. Requesting logits for every token keeps
// output_reorder on the exact proven path compute_imatrix uses, so the layer
// buffers come back in batch order.
static bool compute_features(llama_context * ctx, const common_params & params, const int32_t n_ctx) {
    const llama_model * model = llama_get_model(ctx);

    const int32_t n_embd   = llama_model_n_embd(model);
    const int32_t n_layer  = llama_model_n_layer(model);
    const int32_t n_batch  = params.n_batch;

    const std::vector<int32_t> & layers = params.feature_layers;
    if (layers.empty()) {
        LOG_ERR("%s: --features-out requires --feature-layers\n", __func__);
        return false;
    }
    for (int32_t lid : layers) {
        if (lid < 0 || lid >= n_layer) {
            LOG_ERR("%s: feature layer %d out of range [0, %d)\n", __func__, lid, n_layer);
            return false;
        }
    }

    // clamp the per-chunk warmup so at least one token per chunk is emitted.
    int32_t warmup = params.features_warmup;
    if (warmup < 0) {
        warmup = 0;
    }
    if (warmup > n_ctx - 1) {
        LOG_WRN("%s: --features-warmup %d >= n_ctx %d; clamping to %d\n",
                __func__, warmup, n_ctx, n_ctx - 1);
        warmup = n_ctx - 1;
    }

    // windows advance by stride, overlapping by `warmup` tokens so the emitted
    // rows are contiguous (see the note above). warmup <= n_ctx-1 => stride >= 1.
    const int32_t stride = n_ctx - warmup;

    LOG_INF("%s: tokenizing the input ..\n", __func__);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, params.parse_special);
    LOG_INF("%s: %zu tokens\n", __func__, tokens.size());

    if (int(tokens.size()) < n_ctx) {
        LOG_ERR("%s: need at least %d tokens for a context of %d\n", __func__, n_ctx, n_ctx);
        return false;
    }

    // enable per-layer input-embedding capture for the target layers. this
    // flags a scheduler reserve, so the first decode grows the output buffer
    // to hold the layer buffers (see llama_context::init_buffers).
    for (int32_t lid : layers) {
        llama_set_embeddings_layer_inp(ctx, (uint32_t) lid, true);
    }

    ts_features_writer writer;
    if (!writer.open(params.features_out, n_embd, layers, TS_FEATURES_F32)) {
        LOG_ERR("%s: failed to open feature output '%s'\n", __func__, params.features_out.c_str());
        return false;
    }
    // record the window layout so the training driver can map rows -> tokens.
    writer.header.chunk_tokens = n_ctx;
    writer.header.warmup       = warmup;
    writer.header.stride       = stride;

    // window i decodes [i*stride, i*stride + n_ctx); the last window must stay
    // within the corpus, so i*stride + n_ctx <= tokens.size().
    const int n_chunk_max = (int) ((tokens.size() - n_ctx) / stride) + 1;
    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const int num_batches = (n_ctx + n_batch - 1) / n_batch;

    llama_batch batch = llama_batch_init(std::min(n_batch, n_ctx), 0, 1);

    const int32_t row_floats = n_embd * (int32_t) layers.size();
    LOG_INF("%s: capturing %d windows, n_ctx=%d, batch=%d, layers=%zu, n_embd=%d (%.1f KB/token f32)\n",
            __func__, n_chunk, n_ctx, n_batch, layers.size(), n_embd,
            row_floats * sizeof(float) / 1024.0);
    LOG_INF("%s: warmup=%d, stride=%d -> %d contiguous tokens/window, ~%lld total (%.2fx decode overhead)\n",
            __func__, warmup, stride, stride, (long long) n_chunk * stride, (double) n_ctx / stride);

    std::vector<const float *> layer_rows(layers.size());

    for (int i = 0; i < n_chunk; ++i) {
        const int start = i * stride;
        const int end   = start + n_ctx;

        const auto t_start = std::chrono::high_resolution_clock::now();

        llama_memory_clear(llama_get_memory(ctx), true);

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            common_batch_clear(batch);
            for (int k = 0; k < batch_size; ++k) {
                // logits=true for all tokens: keeps output_reorder on the same
                // proven path as compute_imatrix so layer buffers are batch-order.
                common_batch_add(batch, tokens[batch_start + k], j*n_batch + k, { 0 }, true);
            }

            if (llama_decode(ctx, batch)) {
                LOG_ERR("%s: failed to eval chunk %d batch %d\n", __func__, i, j);
                llama_batch_free(batch);
                return false;
            }

            // fetch each target layer's batch buffer once, then stream one
            // fused row per token. buffer is token-major: row t at +t*n_embd.
            std::vector<const float *> layer_bases(layers.size());
            for (size_t li = 0; li < layers.size(); ++li) {
                const float * base = llama_get_embeddings_layer_inp(ctx, (uint32_t) layers[li]);
                if (base == nullptr) {
                    LOG_ERR("%s: null layer-input buffer for layer %d\n", __func__, layers[li]);
                    llama_batch_free(batch);
                    return false;
                }
                layer_bases[li] = base;
            }

            for (int t = 0; t < batch_size; ++t) {
                // window-local position; the warmup prefix is the previous
                // window's tail, re-decoded for context but not re-emitted.
                const int32_t chunk_pos = j*n_batch + t;
                if (chunk_pos < warmup) {
                    continue;
                }
                for (size_t li = 0; li < layers.size(); ++li) {
                    layer_rows[li] = layer_bases[li] + (size_t) t * n_embd;
                }
                if (!writer.append_token_layers(layer_rows.data())) {
                    LOG_ERR("%s: failed to write feature row\n", __func__);
                    llama_batch_free(batch);
                    return false;
                }
            }
        }

        if (i == 0) {
            llama_synchronize(ctx);
            const auto t_end = std::chrono::high_resolution_clock::now();
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            LOG_INF("%s: %.2f seconds per chunk - ETA %.2f minutes\n",
                    __func__, t_total, t_total * n_chunk / 60.0);
        }
    }

    llama_batch_free(batch);

    if (!writer.close()) {
        LOG_ERR("%s: failed to finalize feature output\n", __func__);
        return false;
    }

    LOG_INF("%s: wrote %lld tokens x %d floats to '%s.bin' (+ '%s.json')\n",
            __func__, (long long) writer.n_tokens_written(), row_floats,
            params.features_out.c_str(), params.features_out.c_str());

    return true;
}

// Spec-decoding variant of compute_imatrix. Routes through
// common/speculative-calibration.{h,cpp} (the new module) and wires
// the imatrix graph-observer lifecycle into its observer_hooks. The
// actual drafter-forward + per-prefix verifier-forward + KV rollback
// + per-step telemetry lives in common/speculative-calibration.cpp.
// The new module calls begin/flush hooks around the per-step main
// forward and at the tail of the run.

static bool show_statistics(const common_params & params) {
    std::vector<tensor_statistics> ts;
    if (params.in_files.empty() || params.in_files.size() > 1) {
        LOG_ERR("\nError: a single imatrix file is required to compute tensor statistics\n\n");
        return false;
    }
    if (g_collector.load_imatrix(params.in_files[0].c_str())) {
        for (const auto & [key, stats] : g_collector.get_mstats()) {
            compute_statistics(ts, IMatrixCollector::stats_key_name(key), stats);
        }
    } else {
        LOG_ERR("\nError: %s is not a valid imatrix file\n\n", params.in_files[0].c_str());
        return false;
    }
    if (!ts.empty()) {
        compute_cossim(ts);
    } else {
        LOG_ERR("Error: cannot compute statistics for %s\n\n", params.in_files[0].c_str());
        return false;
    }

    struct tensor_comparer {
        bool operator()(const tensor_statistics & a, const tensor_statistics & b) const {
            std::string layer, name_a, name_b;
            ;
            process_tensor_name(a.tensor, layer, name_a);
            process_tensor_name(b.tensor, layer, name_b);
            return name_a < name_b || (name_a == name_b && a.total_sqract > b.total_sqract);
        }
    };
    std::sort(ts.begin(), ts.end(), tensor_comparer());

    struct weighted_stats {
        float weighted_bias   = 0.0f;
        float weighted_zd     = 0.0f;
        float weighted_cossim = 0.0f;
        int   total_elements  = 0;
    };
    std::map<int, weighted_stats> ws;

    LOG_INF("\nComputing statistics for %s (%d tensors)\n", params.in_files[0].c_str(), static_cast<int>(ts.size()));
    LOG_INF("\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", " Layer", "       Tensor", "          Σ(Act²)",
            "  Min", "            Max", "           μ", "   σ", " % Active", "N", "   Entropy", "E (norm)", "ZD",
            "  CosSim");
    LOG_INF(
        "=============================================================================================================="
        "===========================================================\n");
    for (const auto & tstat : ts) {
        std::string layer, name;
        process_tensor_name(tstat.tensor, layer, name);

        int blk;
        try {
            blk = std::stoi(layer);
        } catch (const std::exception & e) {
            blk = -1;  // not a block layer
        }

        const float entropy_norm = (tstat.elements > 0) ? 100.0f * (tstat.entropy / std::log2(tstat.elements)) : 0.0f;

        LOG_INF("%5s\t%-20s\t%10.2f\t%8.4f\t%11.4f\t%6.2f\t%6.2f\t%8.2f%%\t%6d\t%10.4f\t%6.2f%%\t%10.2f%%\t%8.4f\n",
                layer.c_str(), name.c_str(), tstat.total_sqract, tstat.min_sqract, tstat.max_sqract, tstat.mean_sqract,
                tstat.stddev, tstat.active * 100.0f, tstat.elements, tstat.entropy,
                entropy_norm, 100.0f * tstat.zd, tstat.cossim);

        const float weighted_bias   = tstat.elements * tstat.total_sqract;
        const float weighted_zd     = tstat.elements * tstat.zd;
        const float weighted_cossim = tstat.elements * tstat.cossim;

        if (ws.find(blk) != ws.end()) {
            ws[blk].weighted_bias += weighted_bias;
            ws[blk].weighted_zd += weighted_zd;
            ws[blk].weighted_cossim += weighted_cossim;
            ws[blk].total_elements += tstat.elements;
        } else {
            weighted_stats temp_ws;
            temp_ws.weighted_bias   = weighted_bias;
            temp_ws.weighted_zd     = weighted_zd;
            temp_ws.weighted_cossim = weighted_cossim;
            temp_ws.total_elements  = tstat.elements;
            ws[blk]                 = temp_ws;
        }
    }

    const int layers = std::count_if(ws.begin(), ws.end(), [](const auto & kv) { return kv.first >= 0; });
    LOG_INF("\nComputing weighted average statistics per layer (%d layers)\n", layers);
    LOG_INF("\n%s\t%s\t%s\t%s\n", "  Layer", "     μΣ(Act²)", "      μZD", "μCosSim");
    LOG_INF("================================================\n");
    for (const auto & [first, second] : ws) {
        const auto & layer = first;
        const auto & stats = second;

        if (stats.total_elements == 0) {
            continue;
        }

        if (layer >= 0) {
            const float bias   = stats.weighted_bias / stats.total_elements;
            const float zd     = stats.weighted_zd / stats.total_elements;
            const float cossim = stats.weighted_cossim / stats.total_elements;

            LOG_INF("%5d\t%14.2f\t%10.4f%%\t%6.4f\n", layer, bias, 100.0f * zd, cossim);
        }
    }
    LOG_INF("\n");

    return true;
}

// Crash-safety / long-run safeguards (Phase 16.9, 2026-08-04)
//
// These four guards replace the previous Python smoke_imatrix.py wrapper.
// They live in the binary so the safeguards are universal (not opt-in via a
// wrapper script), so a 30-60 min run on a memory-constrained M1 does not
// crash silently, and so an operator does not have to remember which
// binary to invoke.
//
// 1. Memory precheck: refuse to start if the model is bigger than the
//    configured fraction of physmem. Mac M1 + 23 GB BF16 model + 9.8 GB
//    already compressed is exactly the failure mode this catches.
// 2. PID file: write <output>.pid at startup; an operator can SIGTERM the
//    wrapper PID and the binary exits at the next chunk boundary.
// 3. Wall-time cap: SIGALRM at --max-minutes; the binary exits cleanly.
// 4. Dynamic save-frequency ladder: start paranoid (every 8 chunks), relax
//    as runtime stability is proven over wall time (16 -> 32 -> 64 -> 128
//    chunks between intermediate saves). Final save is still atomic
//    (the .tmp + rename in save_imatrix above) regardless of ladder.
//
// The implementation of physmem_bytes / file_size_or_zero /
// dynamic_save_freq_ladder lives in imatrix-safeguards.h (header-only)
// so a test binary can unit-test them without dragging in the full
// llama_init / ggml dependency chain.
// ---------------------------------------------------------------------------

// SIGALRM handler for the --max-minutes wall-time cap. The atomic save
// in save_imatrix means the prior checkpoint is intact; we just exit.
// (_exit avoids running static destructors; the atomic .tmp + rename in
// save_imatrix already published or the file is unchanged.)
[[noreturn]] static void on_sigalrm_max_minutes(int) {
    _exit(0);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    params.out_file = "imatrix.gguf";

    params.n_ctx = 512;
    params.escape = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    // set_params before show_statistics so load_imatrix has valid n_ctx/n_parallel
    g_collector.set_params(params);

    if (params.show_statistics) {
        if (!show_statistics(params)) {
            return 1;
        }
        return 0;
    }

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

    // Keep the collector and graph builder on the same graph-resident
    // observer path. Setting this only after set_params() leaves the
    // collector in the legacy callback mode and silently emits an empty
    // imatrix for Tile640 graphs.
    params.imatrix_observers = true;
    params.warmup = false;
    g_collector.set_params(params);

    // If the spec-decoding path will be used (--model-draft set), enable
    // ctx_shift on the verifier so the priming decode can be followed by
    // the spec loop without the KV cache running out of slots. This must
    // be set before common_init_from_params below.
    if (!params.speculative.draft.mparams.path.empty()) {
        params.ctx_shift = true;
    }

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

    // ---- Crash-safety / long-run safeguards (Phase 16.9) -------------

    // 1. Memory precheck: refuse to start if model is too big for physmem.
    //    The previous Python smoke_imatrix.py did this; the safeguard now
    //    lives in the binary so it cannot be skipped by accident. On
    //    unknown platforms (Windows, BSD) physmem_bytes() returns 0 and
    //    the precheck is a no-op (we cannot make a safe decision without
    //    a probe).
    if (!params.no_memory_check && params.memory_budget_fraction > 0.0f) {
        const int64_t model_size = tessera_imatrix_safeguards::file_size_or_zero(params.model.path);
        const int64_t physmem    = tessera_imatrix_safeguards::physmem_bytes();
        if (model_size > 0 && physmem > 0) {
            const double ratio = (double) model_size / (double) physmem;
            if (ratio > params.memory_budget_fraction) {
                fprintf(stderr,
                    "imatrix: refusing to start: model is %lld bytes on "
                    "%lld bytes physmem (ratio=%.2f > "
                    "memory_budget_fraction=%.2f).\n"
                    "imatrix: pass --force or --no-memory-check to override "
                    "(NOT recommended; jetsam SIGKILL is likely on macOS / "
                    "iOS when this is exceeded).\n",
                    (long long) model_size, (long long) physmem,
                    ratio, params.memory_budget_fraction);
                return 1;
            }
            LOG_INF("%s: model=%lld bytes, physmem=%lld bytes, "
                    "ratio=%.2f (budget=%.2f)\n",
                    __func__, (long long) model_size, (long long) physmem,
                    ratio, params.memory_budget_fraction);
        }
    }

    // 2. PID file: write <output>.pid at startup. An operator can
    //    `kill -TERM <pid>` and the binary will exit at the next chunk
    //    boundary (the atomic save in save_imatrix keeps the prior
    //    checkpoint intact).
    if (!params.no_pid_file && !params.out_file.empty()) {
        const std::string pid_path = params.out_file + ".pid";
        std::ofstream f(pid_path, std::ios::trunc);
        if (f.good()) {
#ifdef _WIN32
            f << (long long) GetCurrentProcessId() << "\n";
#else
            f << (long long) getpid() << "\n";
#endif
            LOG_INF("%s: pidfile=%s\n", __func__, pid_path.c_str());
        } else {
            LOG_WRN("%s: could not write pidfile %s\n",
                    __func__, pid_path.c_str());
        }
    }

    // 3. Wall-time cap (SIGALRM). The handler _exit(0)s; the atomic save
    //    in save_imatrix means the prior checkpoint is intact either way.
    if (params.max_minutes > 0) {
        struct sigaction sa = {};
        sa.sa_handler = on_sigalrm_max_minutes;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGALRM, &sa, nullptr);
        struct itimerval it = {};
        it.it_value.tv_sec = (time_t) params.max_minutes * 60;
        setitimer(ITIMER_REAL, &it, nullptr);
        LOG_INF("%s: wall-time cap set to %d minutes\n",
                __func__, params.max_minutes);
    }

    // ---- end safeguards ------------------------------------------------

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ik_collect_imatrix;
    params.cb_eval_user_data = NULL;
    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }
    // The verifier context collects into the VERIFIER scope. Both the
    // context's active scope (drives the graph-build filter dispatch) and the
    // collector's scope (drives the m_stats bucketing) are set together so
    // they cannot drift. A drafter context, if any, would bind to the DRAFTER
    // scope on its own context and a DRAFTER-scoped collector.
    llama_set_imatrix_observer_scope(ctx, LLAMA_OBSERVER_SCOPE_VERIFIER);
    g_collector.set_observer_scope(LLAMA_OBSERVER_SCOPE_VERIFIER);
    llama_set_imatrix_observer_filter(
        ctx, IMatrixCollector::observer_filter, &g_collector);

    const int n_ctx_train = llama_model_n_ctx_train(model);
    if (params.n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, params.n_ctx);
    }

    // ─── Spec-decoding calibration path ─────────────────────────────────────
    // If --model-draft is set, load the drafter and run a real spec-decoding
    // loop instead of the plain text loop. The verifier (this ctx) keeps its
    // graph observers; the drafter is observer-free.
    common_speculative_init_result_ptr spec_init;
    common_speculative_ptr spec;
    const bool use_spec     = !params.speculative.draft.mparams.path.empty();
    const bool use_features = !params.features_out.empty();
    if (use_features && use_spec) {
        LOG_ERR("%s: --features-out is a dedicated trunk-only pass and cannot be combined with --model-draft\n", __func__);
        return 1;
    }
    if (use_spec) {
        LOG_INF("%s: spec-decoding calibration requested; loading drafter '%s'\n",
                __func__, params.speculative.draft.mparams.path.c_str());

        common_params params_dft = common_base_params_to_speculative(params);
        params_dft.speculative.draft.target_model_path = params.model.path;
        // Make sure the drafter picks the right speculative type. We default
        // to DRAFT_SIMPLE; the user can override via --spec-type. We have
        // to set BOTH params_dft.speculative.types (used by
        // common_speculative_init_from_params) AND params.speculative.types
        // (used by common_speculative_init below).
        if (params.speculative.types.empty() ||
            params.speculative.types[0] == COMMON_SPECULATIVE_TYPE_NONE) {
            params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };
        }
        params_dft.speculative.types = params.speculative.types;

        spec_init = common_speculative_init_from_params(params_dft, model, ctx);
        if (!spec_init || spec_init->model() == nullptr || spec_init->context() == nullptr) {
            LOG_ERR("%s : failed to load drafter model '%s'\n",
                    __func__, params_dft.model.path.c_str());
            return 1;
        }
        params.speculative.draft.ctx_tgt = ctx;
        params.speculative.draft.ctx_dft = spec_init->context();

        spec.reset(common_speculative_init(params.speculative, /*n_seq=*/1));
        if (!spec) {
            LOG_ERR("%s : failed to create spec context\n", __func__);
            return 1;
        }
    }
    // ────────────────────────────────────────────────────────────────────────

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    if (use_features) {
        if (!compute_features(ctx, params, n_ctx)) {
            return 1;
        }
    } else if (use_spec) {
        common_speculative_calibration_options opts;
        opts.telemetry_out        = params.telemetry_out;
        opts.telemetry_topk       = params.n_telemetry_topk;
        opts.observer_hooks.begin = [](void *) { g_collector.begin_graph_observers(); };
        opts.observer_hooks.flush = [](void *) { return g_collector.flush_graph_observers(); };
        if (!common_speculative_calibration_run(ctx, model, spec.get(), params, n_ctx, opts)) {
            return 1;
        }
    } else {
        if (!compute_imatrix(ctx, params, n_ctx)) {
            return 1;
        }
    }

    if (!use_features) {
        // the feature pass is trunk-only and writes its own sidecar; it does
        // not produce (or want) an imatrix, so skip the default save.
        g_collector.save_imatrix();
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
