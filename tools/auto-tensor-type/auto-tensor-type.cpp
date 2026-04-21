// auto-tensor-type.cpp
// Automatically determine optimal per-tensor-role quantization types to meet a target BPW,
// by measuring the KLD impact of each quant type on MUL_MAT outputs using real activations.
//
// Usage:
//   llama-auto-tensor-type -m MODEL -i IMATRIX --quants IQ1_BN,IQ2_TQ,IQ3_TQ,Q4_KPT,Q6_K --target-bpw 3.2 -o output.txt
//
// The tool:
//   1. Loads the model and runs forward passes to capture MUL_MAT activations
//   2. For each (tensor_role, quant_type) pair, measures average KLD of MUL_MAT output
//   3. Optimizes the assignment to minimize total KLD while meeting the BPW target
//   4. Outputs a tensor-type-file compatible with llama-quantize --tensor-type-file

#include "arg.h"
#include "common.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <chrono>
#include <mutex>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================================================================
// Section 1: Extern C declarations for special quant types
// ============================================================================

extern "C" {
    // IQ2_TQ
    size_t quantize_iq2_tq(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
    void   iq2tq_train_grid(const float * data, int64_t nrow, int64_t n_per_row, const float * imatrix, int8_t grid_out[64]);
    void   iq2tq_set_grid(const int8_t grid[64]);
    // IQ3_TQ
    size_t quantize_iq3_tq(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
    void   iq3tq_train_grid(const float * data, int64_t nrow, int64_t n_per_row, const float * imatrix, int8_t grid_out[128]);
    void   iq3tq_set_grid(const int8_t grid[128]);
    // IQ1_BN
    size_t quantize_iq1_bn(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
    void   iq1bn_train_codebook(const float * data, int64_t nrow, int64_t n_per_row, const float * imatrix, int8_t aux_out[32768], int nthread);
    void   iq1bn_set_aux(const int8_t aux[32768]);
    // Q3_PT
    void q3pt_train_levels(const float * data, int64_t nrow, int64_t n_per_row, const float * imatrix, float levels_out[8]);
    void q3pt_set_levels(const float levels[8]);
    // Q3_KPT
    void q3kpt_train_levels(const float * data, int64_t nrow, int64_t n_per_row, const float * imatrix, float levels_out[8]);
    void q3kpt_set_levels(const float levels[8]);
    // Q4_DPT
    void q4dpt_train_levels(const float * data, int64_t nrow, int64_t n_per_row, const float * imatrix, int8_t levels_out[16]);
    void q4dpt_set_levels(const int8_t levels[16]);
    // Q2_KPT
    const float * q2kpt_get_levels(void);
    void          q2kpt_prepare_levels(int64_t nrows, int64_t n_per_row);
    void          q2kpt_free_levels(void);
}

// ============================================================================
// Section 2: Data structures
// ============================================================================

struct config {
    std::string model_path;
    std::string imatrix_path;
    std::vector<ggml_type> quant_types;
    float target_bpw           = 0.0f;
    float bpw_tol_high         = 0.0f;   // can be up to this much above target
    float bpw_tol_low          = 0.2f;   // can be up to this much below target
    std::string test_data_path;           // optional test text file
    int n_test_samples           = 1;     // number of samples per test size
    std::vector<int> test_sizes = {32, 128, 512};
    int64_t min_elements        = 40000;
    ggml_type output_tensor_type = GGML_TYPE_COUNT; // default: highest from list
    int max_iterations           = 100;
    std::string output_path;
    int n_threads                = 1;
    // Number of layer buckets per equivalence class. Captures/assignments are
    // made per (role, bucket), so early/middle/late layers of the same role
    // can get different quant types (matches hand-tuned recipes in llama-quant.cpp).
    // Set to 1 to reproduce the older per-role-only behavior.
    int n_layer_buckets          = 3;
};

struct tensor_info {
    std::string name;       // e.g., "blk.0.attn_q.weight"
    std::string role;       // e.g., "attn_q"
    int layer;              // e.g., 0; -1 for global tensors
    ggml_type orig_type;
    int64_t ne[4]           = {};
    size_t n_elements       = 0;
};

// One captured MUL_MAT operation
struct mul_mat_capture {
    std::string weight_name;   // full name, e.g., "blk.0.attn_q.weight"
    std::string role;          // e.g., "attn_q"
    int layer;

    // Weight tensor metadata (needed to read data from file later)
    ggml_type weight_type;
    int64_t weight_ne0, weight_ne1;  // ne0=input features, ne1=output features

    // Captured MUL_MAT input (src[1]) — always F32 after the kernel
    std::vector<float> input_data;
    int64_t input_ne0, input_ne1;  // [ne0=weight_ne0, ne1=n_tokens]

    // Captured reference MUL_MAT output — always F32
    std::vector<float> ref_output_data;
    int64_t ref_ne0, ref_ne1;      // [ne0=weight_ne1, ne1=n_tokens]
};

// Cost of assigning a specific quant type to a (role, bucket)
struct cost_entry {
    double kld;   // average relative-L2 error across captures for this role-bucket
    double bpw;   // bits per weight for this quant type
};

// Key for the cost matrix: role + layer-bucket index.
// Bucket -1 is reserved for globals (token_embd, output).
// Encoded as "role\x01<bucket>" — \x01 is not a character that appears in tensor names.
// Kept as std::string so existing std::map<std::string, ...> types are unaffected.
static constexpr char RB_SEP = '\x01';
static std::string make_rb_key(const std::string & role, int bucket) {
    return role + RB_SEP + std::to_string(bucket);
}
static std::string rb_role(const std::string & key) {
    auto p = key.find(RB_SEP);
    return (p == std::string::npos) ? key : key.substr(0, p);
}
static int rb_bucket(const std::string & key) {
    auto p = key.find(RB_SEP);
    return (p == std::string::npos) ? 0 : std::stoi(key.substr(p + 1));
}
// Human-readable form for logs: "ffn_down[0]" or "output[G]"
static std::string rb_display(const std::string & key) {
    std::string r = rb_role(key);
    int b = rb_bucket(key);
    if (b < 0) return r + "[G]";
    return r + "[" + std::to_string(b) + "]";
}
// Bucket index of a layer given its position within its equivalence class.
// Returns a bucket in [0, n_buckets).
static int compute_bucket(int pos_in_class, int n_in_class, int n_buckets) {
    if (n_in_class <= 1 || n_buckets <= 1) return 0;
    int b = (int)(((long long)n_buckets * pos_in_class) / n_in_class);
    if (b >= n_buckets) b = n_buckets - 1;
    if (b < 0) b = 0;
    return b;
}

// ============================================================================
// Section 3: Utility functions
// ============================================================================

static std::string extract_role(const std::string & name) {
    // "blk.0.attn_q.weight" -> "attn_q"
    // "blk.0.ffn_gate.weight" -> "ffn_gate"
    // "output.weight" -> "output"
    // "token_embd.weight" -> "token_embd"
    static const std::regex layer_re("blk\\.\\d+\\.([^.]+)\\.weight");
    static const std::regex global_re("([^.]+)\\.weight");

    std::smatch m;
    if (std::regex_match(name, m, layer_re) && m.size() > 1) {
        return m[1].str();
    }
    if (std::regex_match(name, m, global_re) && m.size() > 1) {
        return m[1].str();
    }
    return name;
}

// 2D .weight matrix, regardless of size. Used for BPW accounting — llama-quantize
// will quantize these to the fallback ftype even if we don't measure them.
static bool is_matrix_weight(const tensor_info & ti) {
    if (ti.name.size() < 8 || ti.name.substr(ti.name.size() - 7) != ".weight") return false;
    if (ti.ne[2] != 1 || ti.ne[3] != 1) return false;
    if (ti.ne[0] <= 1 || ti.ne[1] <= 1) return false;
    return true;
}

static bool is_quantizable_weight(const tensor_info & ti, int64_t min_elements) {
    // Must be a 2D matrix weight…
    if (!is_matrix_weight(ti)) return false;
    // …and large enough to justify per-role KLD measurement.
    if ((int64_t)ti.n_elements < min_elements) return false;
    return true;
}

static double compute_bpw(ggml_type type) {
    return ggml_get_bpw(type);
}

// Get the quant types sorted by BPW (lowest first)
static std::vector<ggml_type> sorted_by_bpw(const std::vector<ggml_type> & types) {
    std::vector<ggml_type> result = types;
    std::sort(result.begin(), result.end(), [](ggml_type a, ggml_type b) {
        return compute_bpw(a) < compute_bpw(b);
    });
    return result;
}

// Get the highest-BPW quant type from a list
static ggml_type highest_bpw(const std::vector<ggml_type> & types) {
    if (types.empty()) return GGML_TYPE_COUNT;
    ggml_type best = types[0];
    for (auto t : types) {
        if (compute_bpw(t) > compute_bpw(best)) best = t;
    }
    return best;
}

// Check if a quant type supports get_rows (needed for token_embd)
// Per-tensor-trained types (Q3_PT, Q3_KPT, Q4_DPT, Q2_KPT, Q2_DPT) do NOT.
static bool supports_get_rows(ggml_type type) {
    return type != GGML_TYPE_Q3_PT  && type != GGML_TYPE_Q3_KPT &&
           type != GGML_TYPE_Q4_DPT && type != GGML_TYPE_Q2_KPT &&
           type != GGML_TYPE_Q2_DPT;
}

// Get the quant type with BPW closest to a target BPW
static ggml_type closest_bpw(const std::vector<ggml_type> & types, float target) {
    if (types.empty()) return GGML_TYPE_COUNT;
    ggml_type best = types[0];
    double best_diff = fabs(compute_bpw(best) - target);
    for (auto t : types) {
        double diff = fabs(compute_bpw(t) - target);
        if (diff < best_diff) {
            best_diff = diff;
            best = t;
        }
    }
    return best;
}

// Get the quant type with BPW closest to target, filtered by a predicate
template<typename Pred>
static ggml_type closest_bpw_if(const std::vector<ggml_type> & types, float target, Pred pred) {
    if (types.empty()) return GGML_TYPE_COUNT;
    ggml_type best = GGML_TYPE_COUNT;
    double best_diff = 1e30;
    for (auto t : types) {
        if (!pred(t)) continue;
        double diff = fabs(compute_bpw(t) - target);
        if (diff < best_diff) {
            best_diff = diff;
            best = t;
        }
    }
    return best;
}

// Compute relative squared L2 error between two rows of size n:
//   err = ||ref - quant||^2 / (||ref||^2 + eps)
// Why this metric instead of softmax-KLD?
// Softmax over an intermediate MUL_MAT output is not a real distribution, and
// its peakedness depends on the row's absolute magnitude. Rows with large
// magnitudes (attn_q/k/v read from the post-norm residual) produce near one-hot
// softmaxes and dramatic KLDs; rows with small magnitudes (ffn_down/attn_output
// write into the residual) produce flat softmaxes and tiny KLDs. The signal is
// thus inversely correlated with the downstream cost of quantizing the tensor,
// and the `importance_alpha` hack is a scalar fix for a metric-level problem.
// Relative L2 is scale-free per-row: it measures fractional output perturbation
// directly, which is a much better proxy for how the error propagates through
// the residual stream.
static double compute_kld_row(const float * ref, const float * quant, int64_t n) {
    double num = 0;
    double den = 0;
    for (int64_t i = 0; i < n; i++) {
        double r = (double)ref[i];
        double d = r - (double)quant[i];
        num += d * d;
        den += r * r;
    }
    // Guard rows whose reference is all-zero (degenerate, no signal to match).
    if (den <= 1e-20) {
        return num > 1e-20 ? 1.0 : 0.0;
    }
    return num / den;
}

// Compute average relative-L2 across all rows of two F32 matrices of shape [ne0, ne1]
static double compute_avg_kld(const float * ref, const float * quant, int64_t ne0, int64_t ne1) {
    double total = 0;
    int64_t valid_rows = 0;
    for (int64_t row = 0; row < ne1; row++) {
        const float * ref_row   = ref   + row * ne0;
        const float * quant_row = quant + row * ne0;
        double e = compute_kld_row(ref_row, quant_row, ne0);
        if (std::isfinite(e)) {
            total += e;
            valid_rows++;
        }
    }
    return valid_rows > 0 ? total / valid_rows : 1e30;
}

// Read a tensor's raw data from the GGUF file and dequantize to F32
static bool read_tensor_f32(const std::string & fname, const struct gguf_context * gguf_ctx,
                            const std::string & tensor_name, std::vector<float> & out) {
    int64_t tid = gguf_find_tensor(gguf_ctx, tensor_name.c_str());
    if (tid < 0) {
        LOG_ERR("Tensor '%s' not found in GGUF\n", tensor_name.c_str());
        return false;
    }

    ggml_type type = gguf_get_tensor_type(gguf_ctx, tid);
    size_t offset   = gguf_get_tensor_offset(gguf_ctx, tid);

    // Compute total elements from tensor dimensions stored in gguf
    // We need to read the ggml context to get the dimensions
    // Alternative: just read the raw bytes and dequantize
    size_t raw_size = gguf_get_tensor_size(gguf_ctx, tid);

    FILE * f = fopen(fname.c_str(), "rb");
    if (!f) {
        LOG_ERR("Failed to open model file: %s\n", fname.c_str());
        return false;
    }

    size_t data_offset = gguf_get_data_offset(gguf_ctx);
    fseek(f, data_offset + offset, SEEK_SET);

    std::vector<uint8_t> raw(raw_size);
    if (fread(raw.data(), 1, raw_size, f) != raw_size) {
        LOG_ERR("Failed to read tensor data for '%s'\n", tensor_name.c_str());
        fclose(f);
        return false;
    }
    fclose(f);

    // Compute n_elements
    size_t type_size = ggml_type_size(type);
    int64_t blck_size = ggml_blck_size(type);
    size_t n_elements = (raw_size / type_size) * blck_size;

    out.resize(n_elements);
    if (type == GGML_TYPE_F32) {
        memcpy(out.data(), raw.data(), n_elements * sizeof(float));
    } else if (type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *)raw.data();
        for (size_t i = 0; i < n_elements; i++) {
            out[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else if (type == GGML_TYPE_BF16) {
        const ggml_bf16_t * src = (const ggml_bf16_t *)raw.data();
        for (size_t i = 0; i < n_elements; i++) {
            out[i] = ggml_bf16_to_fp32(src[i]);
        }
    } else {
        // Use ggml's type traits for dequantization
        const auto * traits = ggml_get_type_traits(type);
        if (traits && traits->to_float) {
            traits->to_float(raw.data(), out.data(), n_elements, nullptr);
        } else {
            LOG_ERR("Cannot dequantize type %s for tensor '%s'\n", ggml_type_name(type), tensor_name.c_str());
            return false;
        }
    }

    return true;
}

// Get target layers: first, middle, next-to-last
// Layer signature: sorted list of (role, ne0, ne1) for all quantizable weight tensors
// in a layer. Two layers are equivalent iff their signatures match.
using layer_signature = std::vector<std::tuple<std::string, int64_t, int64_t>>;

static layer_signature compute_layer_signature(int layer, const std::vector<tensor_info> & all_tensors, int64_t min_elements) {
    layer_signature sig;
    for (const auto & ti : all_tensors) {
        if (ti.layer != layer) continue;
        if (!is_quantizable_weight(ti, min_elements)) continue;
        sig.emplace_back(ti.role, ti.ne[0], ti.ne[1]);
    }
    std::sort(sig.begin(), sig.end());
    return sig;
}

// Identify layer equivalence classes and pick representative layers from each.
// Returns: vector of (class_index, representative_layer_indices, all_layer_indices)
struct layer_class_info {
    size_t class_index;
    std::vector<int> reps;        // representative layers (first, middle, last)
    std::vector<int> all_layers;  // all layers in this class
    layer_signature signature;    // the signature for diagnostics
};

static std::vector<layer_class_info> get_layer_equivalence_classes(
        int n_layer, const std::vector<tensor_info> & all_tensors, int64_t min_elements) {
    // Compute signatures and group by equivalence class
    std::map<layer_signature, std::vector<int>> class_groups;
    for (int l = 0; l < n_layer; l++) {
        auto sig = compute_layer_signature(l, all_tensors, min_elements);
        if (sig.empty()) continue;  // skip layers with no quantizable tensors
        class_groups[sig].push_back(l);
    }

    // Build result with representatives (first, middle, last of each class)
    std::vector<layer_class_info> result;
    size_t class_idx = 0;
    for (const auto & [sig, layers] : class_groups) {
        layer_class_info info;
        info.class_index = class_idx;
        info.all_layers = layers;
        info.signature = sig;

        info.reps.push_back(layers.front());
        if (layers.size() > 1) {
            info.reps.push_back(layers[layers.size() / 2]);
        }
        if (layers.size() > 2) {
            info.reps.push_back(layers.back());
        }
        // Deduplicate
        std::sort(info.reps.begin(), info.reps.end());
        info.reps.erase(std::unique(info.reps.begin(), info.reps.end()), info.reps.end());

        result.push_back(std::move(info));
        class_idx++;
    }
    return result;
}

// Check if a quant type requires per-tensor training
static bool requires_training(ggml_type type) {
    return type == GGML_TYPE_IQ2_TQ || type == GGML_TYPE_IQ3_TQ ||
           type == GGML_TYPE_IQ1_BN || type == GGML_TYPE_Q3_PT ||
           type == GGML_TYPE_Q3_KPT || type == GGML_TYPE_Q4_DPT ||
           type == GGML_TYPE_Q2_KPT;
}

// ============================================================================
// Section 4: Eval callback for activation capture
// ============================================================================

struct capture_state {
    // Set of weight tensor names we want to capture MUL_MAT for
    std::unordered_set<std::string> target_weight_names;

    // Map from weight tensor name -> role
    std::unordered_map<std::string, std::string> weight_to_role;
    // Map from weight tensor name -> layer index
    std::unordered_map<std::string, int> weight_to_layer;
    // Map from weight tensor name -> layer bucket (0..n_buckets-1, or -1 for globals)
    std::unordered_map<std::string, int> weight_to_bucket;

    // Captures, organized by rb_key = make_rb_key(role, bucket)
    std::unordered_map<std::string, std::vector<mul_mat_capture>> captures_by_role;

    // Temporary buffer for non-host tensor data
    std::vector<uint8_t> tmp_data;

    // Count of captured MUL_MATs
    int captured = 0;
};

static bool capture_callback(ggml_tensor * t, bool ask, void * user_data) {
    auto * state = (capture_state *) user_data;

    if (t->op != GGML_OP_MUL_MAT) return false;
    if (!t->src[0] || !t->src[1]) return false;

    const char * weight_name = t->src[0]->name;

    // Check if this MUL_MAT uses one of our target weight tensors
    if (state->target_weight_names.find(weight_name) == state->target_weight_names.end()) {
        return false;  // Not interested
    }

    if (ask) {
        return true;  // Yes, we want the output data
    }

    // ask=false: data is available, capture it
    mul_mat_capture cap;
    cap.weight_name = weight_name;
    cap.role        = state->weight_to_role[weight_name];
    cap.layer       = state->weight_to_layer[weight_name];
    cap.weight_type = t->src[0]->type;
    cap.weight_ne0  = t->src[0]->ne[0];
    cap.weight_ne1  = t->src[0]->ne[1];
    const int cap_bucket = state->weight_to_bucket.count(weight_name)
        ? state->weight_to_bucket[weight_name] : (cap.layer < 0 ? -1 : 0);

    // Capture input (src[1])
    {
        size_t nbytes = ggml_nbytes(t->src[1]);
        const float * src_ptr = nullptr;
        std::vector<float> host_copy;

        if (ggml_backend_buffer_is_host(t->src[1]->buffer)) {
            src_ptr = (const float *) t->src[1]->data;
        } else {
            host_copy.resize(nbytes / sizeof(float));
            ggml_backend_tensor_get(t->src[1], host_copy.data(), 0, nbytes);
            src_ptr = host_copy.data();
        }

        cap.input_ne0 = t->src[1]->ne[0];
        cap.input_ne1 = t->src[1]->ne[1];
        cap.input_data.assign(src_ptr, src_ptr + (nbytes / sizeof(float)));
    }

    // Capture reference output
    {
        size_t nbytes = ggml_nbytes(t);
        const float * src_ptr = nullptr;
        std::vector<float> host_copy;

        if (ggml_backend_buffer_is_host(t->buffer)) {
            src_ptr = (const float *) t->data;
        } else {
            host_copy.resize(nbytes / sizeof(float));
            ggml_backend_tensor_get(t, host_copy.data(), 0, nbytes);
            src_ptr = host_copy.data();
        }

        cap.ref_ne0 = t->ne[0];
        cap.ref_ne1 = t->ne[1];
        cap.ref_output_data.assign(src_ptr, src_ptr + (nbytes / sizeof(float)));
    }

    state->captures_by_role[make_rb_key(cap.role, cap_bucket)].push_back(std::move(cap));
    state->captured++;

    return true;
}

// ============================================================================
// Section 5: Imatrix loading (simplified from quantize.cpp)
// ============================================================================

static bool load_imatrix_data(const std::string & imatrix_file,
                              std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    struct ggml_context * ctx = nullptr;
    struct gguf_init_params meta_params = {false, &ctx};
    struct gguf_context * ctx_gguf = gguf_init_from_file(imatrix_file.c_str(), meta_params);
    if (!ctx_gguf) {
        LOG_ERR("Failed to open imatrix file: %s\n", imatrix_file.c_str());
        return false;
    }

    const std::string sums_suffix{".in_sum2"};
    const std::string counts_suffix{".counts"};

    std::map<std::string, std::pair<struct ggml_tensor *, struct ggml_tensor *>> sums_counts_for;
    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string name = cur->name;
        if (name.empty()) continue;
        if (name.size() > sums_suffix.size() && name.substr(name.size() - sums_suffix.size()) == sums_suffix) {
            sums_counts_for[name.substr(0, name.size() - sums_suffix.size())].first = cur;
        } else if (name.size() > counts_suffix.size() && name.substr(name.size() - counts_suffix.size()) == counts_suffix) {
            sums_counts_for[name.substr(0, name.size() - counts_suffix.size())].second = cur;
        }
    }

    for (const auto & sc : sums_counts_for) {
        const auto & name   = sc.first;
        const auto * sums   = sc.second.first;
        const auto * counts = sc.second.second;
        if (!sums || !counts) continue;

        int64_t ne0 = sums->ne[0];
        int64_t ne1 = sums->ne[1];
        auto & e = imatrix_data[name];
        e.resize(ggml_nelements(sums));
        for (int64_t j = 0; j < ne1; ++j) {
            float count = ((const float *) counts->data)[j];
            if (count > 0) {
                for (int64_t i = 0; i < ne0; ++i) {
                    e[j * ne0 + i] = ((const float *) sums->data)[j * ne0 + i] / count;
                }
            } else {
                for (int64_t i = 0; i < ne0; ++i) {
                    e[j * ne0 + i] = 1.0f;
                }
            }
        }
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    LOG("Loaded %d imatrix entries from %s\n", (int)imatrix_data.size(), imatrix_file.c_str());
    return true;
}

// Look up imatrix data for a tensor. Returns a uniform vector of 1s if not found.
static std::vector<float> get_imatrix_for_tensor(
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data,
        const std::string & tensor_name, int64_t n_per_row) {
    auto it = imatrix_data.find(tensor_name);
    if (it != imatrix_data.end() && (int64_t)it->second.size() >= n_per_row) {
        return it->second;
    }
    // Not found or wrong size — return uniform
    return std::vector<float>(n_per_row, 1.0f);
}

// ============================================================================
// Section 6: MUL_MAT evaluation and cost matrix
// ============================================================================

// Result of quantizing a weight tensor — includes quantized data and optional per-tensor levels
struct quant_result {
    std::vector<uint8_t> data;        // quantized weight data
    std::vector<uint8_t> levels;      // per-tensor levels/grid (empty if not needed)
};

// Train per-tensor params and quantize a weight tensor to the target type.
static quant_result quantize_weight_to_type(
        ggml_type target_type,
        const float * f32_data,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix) {
    quant_result result;
    size_t quant_size = nrows * ggml_row_size(target_type, n_per_row);
    result.data.resize(quant_size);

    // Train per-tensor params if needed
    if (target_type == GGML_TYPE_IQ2_TQ) {
        int8_t grid[64];
        iq2tq_train_grid(f32_data, nrows, n_per_row, imatrix, grid);
        iq2tq_set_grid(grid);
        // Store a copy of the grid for MUL_MAT dequantization
        result.levels.assign((const uint8_t *)grid, (const uint8_t *)grid + sizeof(grid));
        quantize_iq2_tq(f32_data, result.data.data(), nrows, n_per_row, imatrix);
    } else if (target_type == GGML_TYPE_IQ3_TQ) {
        int8_t grid[128];
        iq3tq_train_grid(f32_data, nrows, n_per_row, imatrix, grid);
        iq3tq_set_grid(grid);
        result.levels.assign((const uint8_t *)grid, (const uint8_t *)grid + sizeof(grid));
        quantize_iq3_tq(f32_data, result.data.data(), nrows, n_per_row, imatrix);
    } else if (target_type == GGML_TYPE_IQ1_BN) {
        int8_t aux[32768];
        iq1bn_train_codebook(f32_data, nrows, n_per_row, imatrix, aux, 1);
        iq1bn_set_aux(aux);
        result.levels.assign((const uint8_t *)aux, (const uint8_t *)aux + sizeof(aux));
        quantize_iq1_bn(f32_data, result.data.data(), nrows, n_per_row, imatrix);
    } else if (target_type == GGML_TYPE_Q3_PT) {
        float levels[8];
        q3pt_train_levels(f32_data, nrows, n_per_row, imatrix, levels);
        q3pt_set_levels(levels);
        // Store a copy of the levels for MUL_MAT dequantization
        result.levels.assign((const uint8_t *)levels, (const uint8_t *)levels + sizeof(levels));
        ggml_quantize_chunk(target_type, f32_data, result.data.data(), 0, nrows, n_per_row, imatrix);
    } else if (target_type == GGML_TYPE_Q3_KPT) {
        float levels[8];
        q3kpt_train_levels(f32_data, nrows, n_per_row, imatrix, levels);
        q3kpt_set_levels(levels);
        result.levels.assign((const uint8_t *)levels, (const uint8_t *)levels + sizeof(levels));
        ggml_quantize_chunk(target_type, f32_data, result.data.data(), 0, nrows, n_per_row, imatrix);
    } else if (target_type == GGML_TYPE_Q4_DPT) {
        int8_t levels[16];
        q4dpt_train_levels(f32_data, nrows, n_per_row, imatrix, levels);
        q4dpt_set_levels(levels);
        result.levels.assign((const uint8_t *)levels, (const uint8_t *)levels + sizeof(levels));
        ggml_quantize_chunk(target_type, f32_data, result.data.data(), 0, nrows, n_per_row, imatrix);
    } else if (target_type == GGML_TYPE_Q2_KPT) {
        q2kpt_prepare_levels(nrows, n_per_row);
        // Store the Q2_KPT levels (they're float[8])
        const float * lv = q2kpt_get_levels();
        if (lv) result.levels.assign((const uint8_t *)lv, (const uint8_t *)lv + 8 * sizeof(float));
        ggml_quantize_chunk(target_type, f32_data, result.data.data(), 0, nrows, n_per_row, imatrix);
    } else {
        // Standard quant type
        ggml_quantize_init(target_type);
        ggml_quantize_chunk(target_type, f32_data, result.data.data(), 0, nrows, n_per_row, imatrix);
    }

    return result;
}

// Run MUL_MAT: result = weight^T * input
// weight: [ne0, ne1] in some quantized type
// input:  [ne0, n_tokens] in F32
// result: [ne1, n_tokens] in F32
// quant_levels: optional per-tensor levels data (for Q3_KPT, Q4_DPT, etc.)
// backend: pre-created backend (CPU or CUDA) — reused across calls
static bool eval_mul_mat(ggml_type weight_type, const void * weight_data,
                         int64_t weight_ne0, int64_t weight_ne1,
                         const float * input_data, int64_t input_ne0, int64_t input_ne1,
                         const std::vector<uint8_t> & quant_levels,
                         std::vector<float> & result_data,
                         ggml_backend_t backend) {
    // Size calculations
    size_t weight_bytes = ggml_row_size(weight_type, weight_ne0) * weight_ne1;
    size_t input_bytes  = (size_t)input_ne0 * input_ne1 * sizeof(float);

    // Create ggml context with no_alloc=true; backend allocator will own tensor data
    size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false) + 4096;
    struct ggml_init_params params = {(size_t) ctx_size, NULL, /*.no_alloc =*/ true};
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        LOG_ERR("Failed to create ggml context for MUL_MAT eval\n");
        return false;
    }

    // Create weight tensor
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, weight_type, weight_ne0, weight_ne1);

    // Set per-tensor levels if provided (needed for Q3_KPT, Q4_DPT, etc.)
    if (!quant_levels.empty()) {
        w->quant_levels = const_cast<void *>(static_cast<const void *>(quant_levels.data()));
    }

    // Create input tensor
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_ne0, input_ne1);

    // Create MUL_MAT
    struct ggml_tensor * result = ggml_mul_mat(ctx, w, x);

    // Allocate all tensors on the provided backend
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        LOG_ERR("Failed to allocate tensors for MUL_MAT eval (weight %s [%lld,%lld])\n",
                ggml_type_name(weight_type), (long long)weight_ne0, (long long)weight_ne1);
        ggml_free(ctx);
        return false;
    }

    // Verify buffers were assigned
    if (!w->buffer || !x->buffer || !result->buffer) {
        LOG_ERR("Tensor buffers not set after allocation (w=%p, x=%p, res=%p)\n",
                (void *)w->buffer, (void *)x->buffer, (void *)result->buffer);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    // Upload data to tensors
    ggml_backend_tensor_set(w, weight_data, 0, weight_bytes);
    ggml_backend_tensor_set(x, input_data, 0, input_bytes);

    // Build graph
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(graph, result);

    // Compute
    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("ggml_backend_graph_compute failed: %d\n", status);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    // Download result
    result_data.resize((size_t)weight_ne1 * input_ne1);
    ggml_backend_tensor_get(result, result_data.data(), 0, ggml_nbytes(result));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// Build the cost matrix: for each (role, quant_type), compute average KLD
// Quant types are evaluated in parallel per capture (each type has its own
// global state so they don't interfere with each other).
static std::map<std::string, std::map<ggml_type, cost_entry>> build_cost_matrix(
        const config & cfg,
        const std::vector<tensor_info> & /*tensors*/,
        std::unordered_map<std::string, std::vector<mul_mat_capture>> & captures_by_role,
        const std::string & model_path,
        const struct gguf_context * gguf_ctx,
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data) {

    std::map<std::string, std::map<ggml_type, cost_entry>> cost_matrix;

    // Accumulate KLD per (role, quant_type)
    std::mutex accum_mutex;
    std::map<std::string, std::map<ggml_type, double>> kld_sums;
    std::map<std::string, std::map<ggml_type, int>> kld_counts;

    const int n_parallel = std::max(1, cfg.n_threads);

    // Create a pool of backends — try CUDA first, fall back to CPU
    // Each thread gets its own backend to avoid contention (VMM pool requires LIFO alloc/free).
    struct backend_pool {
        std::vector<ggml_backend_t> backends;
        std::atomic<int> next_idx{0};
        std::string backend_name;

        backend_pool(int count) {
            // Initialize first backend to determine type
            ggml_backend_t be = ggml_backend_init_best();
            if (!be) {
                LOG_ERR("Failed to init any backend\n");
                return;
            }
            backend_name = ggml_backend_name(be);
            backends.push_back(be);

            // Create separate backends for remaining threads
            // This is required because the CUDA VMM pool is a stack allocator
            // that requires strict LIFO allocation/deallocation order per pool.
            for (int i = 1; i < count; i++) {
                ggml_backend_t be_i = ggml_backend_init_best();
                if (be_i) {
                    backends.push_back(be_i);
                } else {
                    LOG_WRN("Failed to init backend %d, reusing backend 0\n", i);
                    backends.push_back(be);  // Fallback to sharing
                }
            }
        }
        ~backend_pool() {
            // Free all unique backends
            std::unordered_set<ggml_backend_t> freed;
            for (auto be : backends) {
                if (freed.insert(be).second) {
                    ggml_backend_free(be);
                }
            }
        }
        ggml_backend_t get() {
            return backends[next_idx.fetch_add(1) % (int)backends.size()];
        }
    };

    backend_pool pool(n_parallel);
    LOG("Phase 3 backend: %s (%d parallel)\n",
        pool.backend_name.c_str(), n_parallel);

    // For each role that has captures
    for (auto & [role, captures] : captures_by_role) {
        // Group captures by weight_name. quantize_weight_to_type() depends only on
        // (weight_name, qtype) — with multiple test_sizes the same weight is captured
        // N times, so we quantize once per unique weight and reuse across captures.
        std::unordered_map<std::string, std::vector<const mul_mat_capture *>> caps_by_weight;
        for (const auto & cap : captures) {
            caps_by_weight[cap.weight_name].push_back(&cap);
        }

        LOG("Building cost matrix for %s (%zu captures across %zu unique weights, %d parallel threads)...\n",
            rb_display(role).c_str(), captures.size(), caps_by_weight.size(), n_parallel);

        const size_t n_types = cfg.quant_types.size();

        for (const auto & [weight_name, caps_for_weight] : caps_by_weight) {
            // Read weight data as F32 — scoped to this weight, freed at end of iteration
            std::vector<float> weight_f32;
            if (!read_tensor_f32(model_path, gguf_ctx, weight_name, weight_f32)) {
                LOG_WRN("  Failed to read weight data for '%s', skipping\n", weight_name.c_str());
                continue;
            }

            const mul_mat_capture & first_cap = *caps_for_weight.front();
            auto imat = get_imatrix_for_tensor(imatrix_data, weight_name, first_cap.weight_ne0);

            // Quantize each qtype once for this weight (parallel across qtypes).
            // Different qtypes use disjoint per-type globals, so parallel training is safe.
            std::unordered_map<ggml_type, quant_result> quant_cache;
            std::mutex quant_cache_mutex;
            {
                std::vector<std::future<void>> qfutures;
                qfutures.reserve(n_parallel);
                for (size_t ti = 0; ti < n_types; /* advanced inside */) {
                    qfutures.clear();
                    for (int p = 0; p < n_parallel && ti < n_types; p++, ti++) {
                        ggml_type qtype = cfg.quant_types[ti];
                        qfutures.push_back(std::async(std::launch::async,
                            [&first_cap, &weight_f32, &imat, qtype, &quant_cache, &quant_cache_mutex]() {
                                auto qres = quantize_weight_to_type(
                                    qtype, weight_f32.data(),
                                    first_cap.weight_ne1, first_cap.weight_ne0, imat.data());
                                std::lock_guard<std::mutex> lock(quant_cache_mutex);
                                quant_cache.emplace(qtype, std::move(qres));
                            }));
                    }
                    for (auto & f : qfutures) f.get();
                }
            }

            // For each capture × qtype, run MUL_MAT + KLD using the cached quant_result.
            // eval_mul_mat reads the trained grid/levels from tensor->quant_levels
            // (set from quant_result::levels), not from per-type globals, so a stale
            // global state from later quantize calls is irrelevant here.
            for (const auto * cap_ptr : caps_for_weight) {
                const auto & cap = *cap_ptr;

                std::vector<std::future<std::pair<ggml_type, double>>> futures;
                futures.reserve(n_parallel);

                for (size_t ti = 0; ti < n_types; /* advanced inside */) {
                    futures.clear();
                    for (int p = 0; p < n_parallel && ti < n_types; p++, ti++) {
                        ggml_type qtype = cfg.quant_types[ti];
                        auto qit = quant_cache.find(qtype);
                        if (qit == quant_cache.end()) continue;  // quantization failed earlier
                        const quant_result * qres = &qit->second;
                        ggml_backend_t backend = pool.get();
                        futures.push_back(std::async(std::launch::async,
                            [&cap, qtype, qres, backend]() -> std::pair<ggml_type, double> {
                                std::vector<float> quant_output;
                                if (!eval_mul_mat(qtype, qres->data.data(),
                                                  cap.weight_ne0, cap.weight_ne1,
                                                  cap.input_data.data(), cap.input_ne0, cap.input_ne1,
                                                  qres->levels,
                                                  quant_output,
                                                  backend)) {
                                    return {qtype, std::numeric_limits<double>::quiet_NaN()};
                                }
                                return {qtype, compute_avg_kld(cap.ref_output_data.data(), quant_output.data(),
                                                               cap.ref_ne0, cap.ref_ne1)};
                            }));
                    }

                    for (auto & f : futures) {
                        auto [qtype, kld] = f.get();
                        if (std::isfinite(kld)) {
                            std::lock_guard<std::mutex> lock(accum_mutex);
                            kld_sums[role][qtype] += kld;
                            kld_counts[role][qtype]++;
                        }
                    }
                }
            }
        }

        // Build cost entries from accumulated KLD
        for (ggml_type qtype : cfg.quant_types) {
            int n_valid = kld_counts[role][qtype];
            cost_entry entry;
            entry.kld = n_valid > 0 ? kld_sums[role][qtype] / n_valid : 1e30;
            entry.bpw = compute_bpw(qtype);
            cost_matrix[role][qtype] = entry;
        }

        // Release this role's captured input/output buffers — largest single chunk of
        // Phase-2 memory, no longer needed once the cost matrix row is populated.
        std::vector<mul_mat_capture>().swap(captures);
    }

    return cost_matrix;
}

// ============================================================================
// Section 6b: Fusion map — split fused MUL_MAT scores to component tensors
// ============================================================================

// Maps a fused role name to its component roles.
// When the model does a fused MUL_MAT (e.g., QKV projection as one operation),
// the captured KLD is split proportionally among the individual tensor roles
// so the optimizer can assign types to each component independently.
static const std::map<std::string, std::vector<std::string>> fusion_map = {
    {"attn_qkv", {"attn_q", "attn_k", "attn_v"}},
    // Add more fusion patterns here for other models:
    // {"ffn_gate_up", {"ffn_gate", "ffn_up"}},
};

// ============================================================================
// Section 7: Optimization
// ============================================================================

struct role_info {
    std::string role;
    size_t n_elements;   // total elements across all layers
    size_t n_bytes_orig; // original size in bytes
};

// Post-process the cost matrix: for fused roles, split the per-(role, bucket)
// error proportionally among component roles (by element count) at the same
// bucket. Component role-buckets that already have their own captures keep
// their directly-measured value.
static void split_fused_roles(
        std::map<std::string, std::map<ggml_type, cost_entry>> & cost_matrix,
        const std::map<std::string, role_info> & roles) {

    // Collect rb_keys to split first — we'll mutate cost_matrix while iterating.
    std::vector<std::string> fused_keys;
    for (const auto & [k, _] : cost_matrix) {
        std::string role = rb_role(k);
        if (fusion_map.count(role)) fused_keys.push_back(k);
    }

    for (const auto & fused_key : fused_keys) {
        std::string fused_role = rb_role(fused_key);
        int bucket             = rb_bucket(fused_key);
        const auto & components = fusion_map.at(fused_role);

        // Element counts for each component role at THIS bucket.
        size_t total_comp_elements = 0;
        std::vector<size_t> comp_elems;
        for (const auto & comp : components) {
            auto rit = roles.find(make_rb_key(comp, bucket));
            size_t ne = (rit != roles.end()) ? rit->second.n_elements : 0;
            comp_elems.push_back(ne);
            total_comp_elements += ne;
        }
        if (total_comp_elements == 0) continue;

        const auto fused_entries = cost_matrix[fused_key]; // copy — we'll be mutating

        for (const auto & [qtype, fused_entry] : fused_entries) {
            for (size_t i = 0; i < components.size(); i++) {
                double fraction = (double)comp_elems[i] / (double)total_comp_elements;
                std::string comp_key = make_rb_key(components[i], bucket);
                auto & comp_entries = cost_matrix[comp_key];
                if (comp_entries.count(qtype)) continue; // direct measurement wins

                cost_entry comp_entry;
                comp_entry.kld = fused_entry.kld * fraction;
                comp_entry.bpw = fused_entry.bpw;
                comp_entries[qtype] = comp_entry;
            }
        }

        LOG("Split fused %s error into components:\n", rb_display(fused_key).c_str());
        for (size_t i = 0; i < components.size(); i++) {
            double fraction = (double)comp_elems[i] / (double)total_comp_elements;
            LOG("  %s: %.1f%% (%zu elements)\n",
                rb_display(make_rb_key(components[i], bucket)).c_str(),
                fraction * 100, comp_elems[i]);
        }
    }
}

struct assignment {
    std::map<std::string, ggml_type> role_to_type;
    double total_kld;
    double total_bpw;
};

// Compute the total BPW for an assignment (quantizable roles + non-quantizable overhead)
static double compute_total_bpw(
        const std::map<std::string, ggml_type> & role_to_type,
        const std::map<std::string, role_info> & roles,
        size_t total_all_elements,
        double non_quantizable_bits) {
    double weighted_sum = 0;
    for (const auto & [role, type] : role_to_type) {
        auto it = roles.find(role);
        if (it == roles.end()) continue;
        weighted_sum += compute_bpw(type) * it->second.n_elements;
    }
    weighted_sum += non_quantizable_bits;
    return total_all_elements > 0 ? weighted_sum / total_all_elements : 0;
}

// Compute the total KLD for an assignment
static double compute_total_kld(
        const std::map<std::string, ggml_type> & role_to_type,
        const std::map<std::string, std::map<ggml_type, cost_entry>> & cost_matrix) {
    double total = 0;
    for (const auto & [role, type] : role_to_type) {
        auto it = cost_matrix.find(role);
        if (it == cost_matrix.end()) continue;
        auto it2 = it->second.find(type);
        if (it2 != it->second.end()) {
            total += it2->second.kld;
        }
    }
    return total;
}

// Exact multi-choice knapsack DP over per-(role, bucket) items.
// Each item has a set of (qtype, bpw, error) choices; pick one per item,
// minimize total error, constrained by total BPW in [target - tol_low, target + tol_high].
//
// Budget is discretized: 1 DP unit ≈ total_all_elements / N_UNITS bits. At
// N_UNITS = 16384 on an 800M-param model, 1 unit ≈ 50K bits ≈ 0.00006 BPW,
// well below any realistic tolerance. Table size is O(n_items × budget_units).
static assignment optimize_assignment(
        const config & cfg,
        const std::map<std::string, std::map<ggml_type, cost_entry>> & cost_matrix,
        const std::map<std::string, role_info> & roles,
        size_t total_all_elements,
        double non_quantizable_bits) {

    struct dp_choice { ggml_type qt; int units; double kld; };
    struct dp_item   { std::string key; size_t n_elements; std::vector<dp_choice> choices; };

    constexpr int N_UNITS = 16384;
    const double bits_per_unit = (double)total_all_elements / (double)N_UNITS;

    const double hi_bpw = cfg.target_bpw + cfg.bpw_tol_high;
    const double lo_bpw = std::max(0.0, (double)cfg.target_bpw - (double)cfg.bpw_tol_low);
    const int budget_hi = (int)std::ceil(hi_bpw * N_UNITS);
    const int budget_lo = (int)std::floor(lo_bpw * N_UNITS);
    const int non_quant_units =
        (int)std::round(non_quantizable_bits / bits_per_unit);

    // DP state axis is sized to allow up to 2 BPW of overshoot beyond the upper
    // tolerance. This matters when the forced minimum (e.g. token_embd at Q6_K
    // with tied embeddings) already overshoots the target — we still want to
    // report the best achievable assignment rather than fail.
    const int overshoot_cap = (int)std::ceil(2.0 * N_UNITS);
    const int B = budget_hi + overshoot_cap + 1;

    // Build items from the cost matrix (skip items with zero elements — they
    // correspond to roles with no actual tensors, e.g. fused-role entries
    // materialized only through split_fused_roles).
    //
    // Per-choice cost is element-weighted: n_elements * relative_L2. The raw
    // per-row relative-L2 is dimensionless and treats all roles equally, which
    // makes the DP happy to spend Q6_K on a tiny tensor (4M-param attn_output)
    // while starving a 37M-param attn_qkv to IQ2_TQ — same bit cost per role,
    // very different aggregate quality impact. Element-weighting aligns the
    // objective with "total number of parameters perturbed, weighted by
    // fractional output error" and keeps big tensors from being sacrificed.
    std::vector<dp_item> items;
    items.reserve(cost_matrix.size());
    for (const auto & [key, row] : cost_matrix) {
        auto rit = roles.find(key);
        size_t ne = (rit != roles.end()) ? rit->second.n_elements : 0;
        if (ne == 0) continue;
        dp_item it;
        it.key = key;
        it.n_elements = ne;
        for (auto qt : cfg.quant_types) {
            auto cit = row.find(qt);
            if (cit == row.end()) continue;
            if (cit->second.kld >= 1e29) continue; // sentinel: forbidden choice
            dp_choice c;
            c.qt = qt;
            c.units = (int)std::round((double)ne * compute_bpw(qt) / bits_per_unit);
            if (c.units < 1) c.units = 1;
            c.kld = cit->second.kld * (double)ne;
            it.choices.push_back(c);
        }
        if (it.choices.empty()) continue;
        items.push_back(std::move(it));
    }
    if (items.empty()) {
        LOG_ERR("No optimization items — empty cost matrix?\n");
        return {};
    }

    const int n = (int)items.size();
    constexpr double INF = 1e300;

    std::vector<double> dp(B, INF);
    std::vector<double> next_dp(B, INF);
    // choice_taken[i][u] = which choice index was used at item i to reach state u
    // prev_u[i][u]       = the u state this came from
    std::vector<std::vector<int>> choice_taken(n, std::vector<int>(B, -1));
    std::vector<std::vector<int>> prev_u(n, std::vector<int>(B, -1));

    if (non_quant_units >= 0 && non_quant_units < B) dp[non_quant_units] = 0.0;

    for (int i = 0; i < n; i++) {
        std::fill(next_dp.begin(), next_dp.end(), INF);
        for (int u = 0; u < B; u++) {
            if (dp[u] >= INF) continue;
            for (int ci = 0; ci < (int)items[i].choices.size(); ci++) {
                const auto & c = items[i].choices[ci];
                int nu = u + c.units;
                if (nu >= B) continue;
                double nk = dp[u] + c.kld;
                if (nk < next_dp[nu]) {
                    next_dp[nu] = nk;
                    choice_taken[i][nu] = ci;
                    prev_u[i][nu] = u;
                }
            }
        }
        dp.swap(next_dp);
    }

    // Preferred: best terminal state in [budget_lo, budget_hi].
    int best_u = -1;
    double best_kld = INF;
    for (int u = std::max(0, budget_lo); u <= budget_hi && u < B; u++) {
        if (dp[u] < best_kld) { best_kld = dp[u]; best_u = u; }
    }
    // Fallback 1: lower bound unreachable (e.g. almost everything overshoots
    // the floor). Take any u in [0, budget_hi].
    if (best_u < 0) {
        for (int u = 0; u <= budget_hi && u < B; u++) {
            if (dp[u] < best_kld) { best_kld = dp[u]; best_u = u; }
        }
    }
    // Fallback 2: target itself is infeasible (common with small/tied-embedding
    // models where token_embd at Q6_K alone exceeds the target). Report best
    // overshoot — smallest u with finite dp, tie-break on lower KLD.
    if (best_u < 0) {
        int min_u = -1;
        for (int u = 0; u < B; u++) {
            if (dp[u] < INF) { min_u = u; break; }
        }
        if (min_u >= 0) {
            best_u = min_u;
            best_kld = dp[min_u];
            LOG("  [warning] target BPW %.4f is infeasible — falling back to lowest achievable at %.4f BPW\n",
                cfg.target_bpw, (double)min_u / (double)N_UNITS);
        }
    }
    if (best_u < 0) {
        LOG_ERR("No feasible DP assignment at all — something is wrong with cost matrix\n");
        return {};
    }

    // Reconstruct the assignment by walking back through the choice tables.
    assignment out;
    out.total_kld = best_kld;
    int cur_u = best_u;
    for (int i = n - 1; i >= 0; i--) {
        int ci = choice_taken[i][cur_u];
        if (ci < 0) {
            LOG_ERR("DP reconstruction failed at item %d (u=%d)\n", i, cur_u);
            return {};
        }
        out.role_to_type[items[i].key] = items[i].choices[ci].qt;
        cur_u = prev_u[i][cur_u];
    }
    out.total_bpw = compute_total_bpw(out.role_to_type, roles, total_all_elements, non_quantizable_bits);

    LOG("DP optimizer: %d items, budget units=[%d, %d] (step=%.6f bpw), "
        "optimum at u=%d (bpw=%.4f, element-weighted error=%.3e)\n",
        n, budget_lo, budget_hi, 1.0 / (double)N_UNITS, best_u,
        (double)best_u / (double)N_UNITS, best_kld);

    return out;
}

// ============================================================================
// Section 8: Output
// ============================================================================

// Build a regex chunk like "(0|3|27)" from a sorted layer list.
static std::string layer_alternation(const std::vector<int> & layers) {
    std::string s = "(";
    for (size_t i = 0; i < layers.size(); i++) {
        if (i) s += "|";
        s += std::to_string(layers[i]);
    }
    s += ")";
    return s;
}

static bool write_tensor_type_file(const std::string & path,
                                   const std::map<std::string, ggml_type> & role_to_type,
                                   ggml_type output_tensor_type,
                                   ggml_type token_embd_tensor_type,
                                   const std::map<std::string, std::vector<int>> & bucket_layers) {
    std::ofstream file(path);
    if (!file) {
        LOG_ERR("Failed to open output file: %s\n", path.c_str());
        return false;
    }

    // Globals first. Anchored with ^ so they don't collide with layer tensors.
    if (token_embd_tensor_type != GGML_TYPE_COUNT) {
        file << "^token_embd=" << ggml_type_name(token_embd_tensor_type) << "\n";
    }
    if (output_tensor_type != GGML_TYPE_COUNT) {
        file << "^output=" << ggml_type_name(output_tensor_type) << "\n";
    }

    // Per-(role, bucket) entries. For each entry we list the concrete layer
    // indices from bucket_layers so the regex matches exactly the layers in
    // this bucket:
    //   blk\.(0|1|2)\.ffn_down\.=Q5_K
    //   blk\.(3|4|...)\.ffn_down\.=Q4_K
    //   blk\.(28|29|30|31)\.ffn_down\.=Q6_K
    // If bucket_layers has no entry for a key (e.g. single-bucket fallback),
    // we emit the role-only pattern as before.
    for (const auto & [key, type] : role_to_type) {
        std::string role = rb_role(key);
        if (role == "token_embd" || role == "output") continue;

        auto bit = bucket_layers.find(key);
        if (bit == bucket_layers.end() || bit->second.empty()) {
            file << "\\." << role << "\\.=" << ggml_type_name(type) << "\n";
        } else {
            file << "blk\\." << layer_alternation(bit->second)
                 << "\\." << role << "\\.=" << ggml_type_name(type) << "\n";
        }
    }

    file.close();
    LOG("Wrote tensor-type-file to %s\n", path.c_str());
    return true;
}

// ============================================================================
// Section 9: Argument parsing
// ============================================================================

static ggml_type parse_ggml_type_str(const char * arg) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        auto type = (ggml_type)i;
        const auto * name = ggml_type_name(type);
        if (name && strcasecmp(name, arg) == 0) {
            return type;
        }
    }
    return GGML_TYPE_COUNT;
}

static std::vector<ggml_type> parse_quant_list(const std::string & s) {
    std::vector<ggml_type> result;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        ggml_type t = parse_ggml_type_str(token.c_str());
        if (t == GGML_TYPE_COUNT) {
            LOG_ERR("Unknown quant type: '%s'\n", token.c_str());
            exit(1);
        }
        result.push_back(t);
    }
    return result;
}

static void print_usage(const char * prog) {
    LOG("Usage: %s [options]\n", prog);
    LOG("\n");
    LOG("Options:\n");
    LOG("  -m, --model PATH         Path to input GGUF model (required)\n");
    LOG("  -i, --imatrix PATH       Path to importance matrix file (required)\n");
    LOG("  -q, --quants LIST        Comma-separated list of candidate quant types (required)\n");
    LOG("                           e.g., IQ1_BN,IQ2_TQ,IQ3_TQ,Q4_KPT,Q6_K\n");
    LOG("  -b, --target-bpw N       Target bits per weight (required)\n");
    LOG("  -o, --output PATH        Output tensor-type-file path (required)\n");
    LOG("  --bpw-tolerance HIGH,LOW BPW tolerance: +HIGH, -LOW from target (default: +0,-0.2)\n");
    LOG("  --test-data PATH         Text file for test inputs (optional, synthetic if not given)\n");
    LOG("  --test-samples N         Number of samples per test size (default: 1)\n");
    LOG("  --test-sizes S1,S2,S3    Token counts for test inputs (default: 32,128,512)\n");
    LOG("  --min-elements N         Skip tensors with fewer elements (default: 40000)\n");
    LOG("  --output-tensor-type T   Quant type for output.weight (default: highest from list)\n");
    LOG("  --max-iterations N       Max optimization iterations (default: 100)\n");
    LOG("  --threads N              Number of threads (default: 1)\n");
    LOG("  --layer-buckets N        Per-class layer buckets for independent quant assignment\n");
    LOG("                           (default: 3 — first/middle/last third).\n");
    LOG("                           1 reproduces the older per-role-only behavior.\n");
}

static config parse_args(int argc, char ** argv) {
    config cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if ((arg == "-i" || arg == "--imatrix") && i + 1 < argc) {
            cfg.imatrix_path = argv[++i];
        } else if ((arg == "-q" || arg == "--quants") && i + 1 < argc) {
            cfg.quant_types = parse_quant_list(argv[++i]);
        } else if ((arg == "-b" || arg == "--target-bpw") && i + 1 < argc) {
            cfg.target_bpw = std::stof(argv[++i]);
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            cfg.output_path = argv[++i];
        } else if (arg == "--bpw-tolerance" && i + 1 < argc) {
            std::string tol = argv[++i];
            // Parse "+HIGH,-LOW" format
            size_t comma = tol.find(',');
            if (comma != std::string::npos) {
                cfg.bpw_tol_high = std::stof(tol.substr(0, comma));
                cfg.bpw_tol_low = std::stof(tol.substr(comma + 1));
            } else {
                cfg.bpw_tol_high = std::stof(tol);
            }
        } else if (arg == "--test-data" && i + 1 < argc) {
            cfg.test_data_path = argv[++i];
        } else if (arg == "--test-samples" && i + 1 < argc) {
            cfg.n_test_samples = std::stoi(argv[++i]);
            if (cfg.n_test_samples < 1) {
                LOG_ERR("--test-samples must be >= 1\n");
                exit(1);
            }
        } else if (arg == "--test-sizes" && i + 1 < argc) {
            std::string sizes = argv[++i];
            std::istringstream ss(sizes);
            std::string token;
            cfg.test_sizes.clear();
            while (std::getline(ss, token, ',')) {
                cfg.test_sizes.push_back(std::stoi(token));
            }
        } else if (arg == "--min-elements" && i + 1 < argc) {
            cfg.min_elements = std::stoll(argv[++i]);
        } else if (arg == "--output-tensor-type" && i + 1 < argc) {
            cfg.output_tensor_type = parse_ggml_type_str(argv[++i]);
        } else if (arg == "--max-iterations" && i + 1 < argc) {
            cfg.max_iterations = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            cfg.n_threads = std::stoi(argv[++i]);
        } else if (arg == "--layer-buckets" && i + 1 < argc) {
            cfg.n_layer_buckets = std::stoi(argv[++i]);
            if (cfg.n_layer_buckets < 1) {
                LOG_ERR("--layer-buckets must be >= 1\n");
                exit(1);
            }
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else {
            LOG_ERR("Unknown argument: %s\n\n", arg.c_str());
            print_usage(argv[0]);
            exit(1);
        }
    }

    if (cfg.model_path.empty() || cfg.imatrix_path.empty() ||
        cfg.quant_types.empty() || cfg.target_bpw <= 0 || cfg.output_path.empty()) {
        LOG_ERR("Missing required arguments\n\n");
        print_usage(argv[0]);
        exit(1);
    }

    if (cfg.output_tensor_type == GGML_TYPE_COUNT) {
        cfg.output_tensor_type = highest_bpw(cfg.quant_types);
    }

    return cfg;
}

// ============================================================================
// Section 10: Main
// ============================================================================

int main(int argc, char ** argv) {
    config cfg = parse_args(argc, argv);

    LOG("=== llama-auto-tensor-type ===\n");
    LOG("Model:       %s\n", cfg.model_path.c_str());
    LOG("Imatrix:     %s\n", cfg.imatrix_path.c_str());
    LOG("Target BPW:  %.2f (tolerance: +%.2f, -%.2f)\n",
        cfg.target_bpw, cfg.bpw_tol_high, cfg.bpw_tol_low);
    LOG("Quant types: ");
    for (auto t : cfg.quant_types) LOG("%s ", ggml_type_name(t));
    LOG("\n");
    LOG("Output:      %s\n", cfg.output_path.c_str());
    LOG("\n");

    // ---- Phase 1: Load model metadata from GGUF ----
    LOG("--- Phase 1: Loading model metadata ---\n");

    // no_alloc=true: we only need tensor metadata (dims, type, offset); raw weight data
    // is read from disk in Phase 3 via read_tensor_f32. Loading it here would duplicate
    // the entire model in host RAM alongside the llama-backend copy.
    struct ggml_context * ggml_ctx = nullptr;
    struct gguf_init_params gguf_params = {true, &ggml_ctx};
    struct gguf_context * gguf_ctx = gguf_init_from_file(cfg.model_path.c_str(), gguf_params);
    if (!gguf_ctx) {
        LOG_ERR("Failed to open model: %s\n", cfg.model_path.c_str());
        return 1;
    }

    int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
    LOG("Model has %lld tensors\n", (long long)n_tensors);

    // Determine n_layer from tensor names
    int n_layer = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        std::string sname(name);
        static const std::regex blk_re("blk\\.(\\d+)\\.");
        std::smatch m;
        if (std::regex_search(sname, m, blk_re) && m.size() > 1) {
            int layer = std::stoi(m[1].str());
            if (layer + 1 > n_layer) n_layer = layer + 1;
        }
    }
    LOG("Model has %d layers\n", n_layer);

    // Enumerate tensor info
    std::vector<tensor_info> all_tensors;
    for (int64_t i = 0; i < n_tensors; i++) {
        tensor_info ti;
        ti.name = gguf_get_tensor_name(gguf_ctx, i);
        ti.orig_type = gguf_get_tensor_type(gguf_ctx, i);
        ti.role = extract_role(ti.name);
        ti.n_elements = gguf_get_tensor_size(gguf_ctx, i) / ggml_type_size(ti.orig_type) * ggml_blck_size(ti.orig_type);

        // Get dimensions from ggml context
        struct ggml_tensor * gt = ggml_get_tensor(ggml_ctx, ti.name.c_str());
        if (gt) {
            for (int d = 0; d < 4; d++) ti.ne[d] = gt->ne[d];
        }

        // Determine layer from name
        static const std::regex blk_re2("blk\\.(\\d+)\\.");
        std::smatch m;
        if (std::regex_search(ti.name, m, blk_re2) && m.size() > 1) {
            ti.layer = std::stoi(m[1].str());
        } else {
            ti.layer = -1;  // global tensor
        }

        all_tensors.push_back(ti);
    }

    // Identify quantizable weight tensors
    std::vector<tensor_info> quantizable;
    for (const auto & ti : all_tensors) {
        if (is_quantizable_weight(ti, cfg.min_elements)) {
            quantizable.push_back(ti);
        }
    }
    LOG("Found %zu quantizable weight tensors (>= %lld elements, 2D)\n",
        quantizable.size(), (long long)cfg.min_elements);

    // Get unique roles
    std::set<std::string> unique_roles;
    for (const auto & ti : quantizable) {
        unique_roles.insert(ti.role);
    }
    LOG("Found %zu unique tensor roles: ", unique_roles.size());
    for (const auto & r : unique_roles) LOG("%s ", r.c_str());
    LOG("\n");

    // Determine target layers using layer equivalence classes.
    // Hybrid models (e.g., Qwen3.5) have different layer types (dense attention vs SSM/Mamba)
    // with different tensor roles and shapes. We group layers by their tensor signature
    // and sample representatives from each class.
    auto layer_classes = get_layer_equivalence_classes(n_layer, all_tensors, cfg.min_elements);

    // Collect all target layer indices
    std::vector<int> target_layers;
    for (const auto & lc : layer_classes) {
        for (int l : lc.reps) target_layers.push_back(l);
    }
    std::sort(target_layers.begin(), target_layers.end());
    target_layers.erase(std::unique(target_layers.begin(), target_layers.end()), target_layers.end());

    LOG("Layer equivalence classes: %zu classes, %zu total target layers\n",
        layer_classes.size(), target_layers.size());
    for (const auto & lc : layer_classes) {
        std::string roles_str;
        for (const auto & [role, ne0, ne1] : lc.signature) {
            if (!roles_str.empty()) roles_str += ", ";
            roles_str += role;
        }
        LOG("  Class %zu: %zu layers, reps=[", lc.class_index, lc.all_layers.size());
        for (size_t i = 0; i < lc.reps.size(); i++) {
            if (i > 0) LOG(", ");
            LOG("%d", lc.reps[i]);
        }
        LOG("], roles=[%s]\n", roles_str.c_str());
    }

    // Compute per-layer bucket (0..n_layer_buckets-1) based on position within
    // the layer's equivalence class. Layers outside any class (i.e. no quantizable
    // weights) get bucket 0 — they won't be sampled anyway. Globals use -1.
    std::map<int, int> layer_to_bucket;
    for (const auto & lc : layer_classes) {
        const int n = (int)lc.all_layers.size();
        for (int i = 0; i < n; i++) {
            layer_to_bucket[lc.all_layers[i]] = compute_bucket(i, n, cfg.n_layer_buckets);
        }
    }
    // For each (role, bucket), list the concrete layer indices that belong to it
    // (used at emission to write layer-alternation regexes).
    std::map<std::string, std::vector<int>> bucket_layers;
    for (const auto & ti : quantizable) {
        if (ti.layer < 0) continue; // globals handled separately
        auto bit = layer_to_bucket.find(ti.layer);
        if (bit == layer_to_bucket.end()) continue;
        std::string key = make_rb_key(ti.role, bit->second);
        auto & v = bucket_layers[key];
        if (std::find(v.begin(), v.end(), ti.layer) == v.end()) v.push_back(ti.layer);
    }
    for (auto & [k, v] : bucket_layers) std::sort(v.begin(), v.end());

    if (cfg.n_layer_buckets > 1) {
        LOG("Layer bucketing (n_buckets=%d):\n", cfg.n_layer_buckets);
        for (const auto & lc : layer_classes) {
            LOG("  Class %zu buckets:", lc.class_index);
            for (int b = 0; b < cfg.n_layer_buckets; b++) {
                std::string layers_str;
                for (int L : lc.all_layers) {
                    if (layer_to_bucket[L] != b) continue;
                    if (!layers_str.empty()) layers_str += ",";
                    layers_str += std::to_string(L);
                }
                if (!layers_str.empty()) LOG(" [%d]={%s}", b, layers_str.c_str());
            }
            LOG("\n");
        }
    }

    // Build set of target weight tensor names (for the eval callback)
    // NOTE: token_embd uses ggml_get_rows (embedding lookup), NOT ggml_mul_mat,
    // so it cannot be measured via MUL_MAT KLD. It gets the highest-BPW type as preset.
    std::unordered_set<std::string> target_weight_names;
    for (const auto & ti : quantizable) {
        if (ti.role == "token_embd") continue;  // skip — not MUL_MAT
        if (ti.layer >= 0) {
            // Include if this tensor's layer is one of our target layers
            if (std::binary_search(target_layers.begin(), target_layers.end(), ti.layer)) {
                target_weight_names.insert(ti.name);
            }
        } else {
            // Global tensors (output.weight, etc.) — capture those too
            target_weight_names.insert(ti.name);
        }
    }

    // ---- Phase 2: Capture reference activations ----
    LOG("\n--- Phase 2: Capturing reference activations ---\n");

    // Load imatrix data
    std::unordered_map<std::string, std::vector<float>> imatrix_data;
    if (!cfg.imatrix_path.empty()) {
        load_imatrix_data(cfg.imatrix_path, imatrix_data);
    }

    // Load model with llama API
    common_params params;
    params.model.path = cfg.model_path;
    params.n_gpu_layers = 99;  // Offload to GPU if available, falls back to CPU
    // Size batch/context to the largest requested test size (+BOS); n_batch must be
    // >= n_tokens_all or llama_decode asserts.
    int max_test_size = 0;
    for (int s : cfg.test_sizes) max_test_size = std::max(max_test_size, s);
    const int min_batch = std::max(512, max_test_size + 1);
    params.n_batch  = min_batch;
    params.n_ubatch = min_batch;
    params.n_ctx    = std::max(1024, max_test_size + 1);

    capture_state cap_state;
    for (const auto & ti : quantizable) {
        if (target_weight_names.count(ti.name)) {
            cap_state.target_weight_names.insert(ti.name);
            cap_state.weight_to_role[ti.name] = ti.role;
            cap_state.weight_to_layer[ti.name] = ti.layer;
            if (ti.layer < 0) {
                cap_state.weight_to_bucket[ti.name] = -1;
            } else {
                auto bit = layer_to_bucket.find(ti.layer);
                cap_state.weight_to_bucket[ti.name] =
                    (bit != layer_to_bucket.end()) ? bit->second : 0;
            }
        }
    }

    params.cb_eval = capture_callback;
    params.cb_eval_user_data = &cap_state;

    common_init();
    ggml_backend_load_all();
    llama_backend_init();

    auto llama_init = common_init_from_params(params);
    if (!llama_init) {
        LOG_ERR("Failed to load model\n");
        return 1;
    }

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("Failed to initialize context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Prepare test tokens for each (size, sample) combination.
    // Tokens are taken as consecutive non-overlapping chunks from the file:
    //   [size0_sample0][size0_sample1]...[size1_sample0][size1_sample1]...
    // With --test-sizes 64,128 --test-samples 3, the layout is:
    //   [64][64][64][128][128][128]  (6 consecutive chunks)
    struct test_set_info {
        int size;
        int sample;
    };
    std::vector<std::vector<llama_token>> test_token_sets;
    std::vector<test_set_info> test_set_infos;

    if (!cfg.test_data_path.empty()) {
        std::ifstream tf(cfg.test_data_path);
        if (!tf) {
            LOG_ERR("Failed to open test data file: %s\n", cfg.test_data_path.c_str());
            return 1;
        }
        std::string test_text((std::istreambuf_iterator<char>(tf)),
                               std::istreambuf_iterator<char>());
        auto all_tokens = common_tokenize(ctx, test_text, add_bos, false);
        LOG("Tokenized test data: %zu tokens\n", all_tokens.size());

        size_t total_needed = 0;
        for (int size : cfg.test_sizes) {
            total_needed += (size_t)size * cfg.n_test_samples;
        }
        if (all_tokens.size() < total_needed) {
            LOG_ERR("Test data file too small: need %zu tokens (%d samples × %d sizes: ",
                    total_needed, cfg.n_test_samples, (int)cfg.test_sizes.size());
            for (size_t i = 0; i < cfg.test_sizes.size(); i++) {
                if (i > 0) LOG_ERR(" + ");
                LOG_ERR("%d×%d", cfg.n_test_samples, cfg.test_sizes[i]);
            }
            LOG_ERR("), but only %zu tokens available in '%s'\n",
                    all_tokens.size(), cfg.test_data_path.c_str());
            return 1;
        }

        size_t offset = 0;
        for (int size : cfg.test_sizes) {
            for (int sample = 0; sample < cfg.n_test_samples; sample++) {
                test_token_sets.emplace_back(all_tokens.begin() + offset,
                                             all_tokens.begin() + offset + size);
                test_set_infos.push_back({size, sample});
                offset += size;
            }
        }
        LOG("Prepared %zu test sets from file (%zu tokens consumed):\n",
            test_token_sets.size(), offset);
        for (size_t i = 0; i < test_set_infos.size(); i++) {
            LOG("  [%zu] size=%d, sample=%d, tokens=%zu\n",
                i, test_set_infos[i].size, test_set_infos[i].sample + 1,
                test_token_sets[i].size());
        }
    } else {
        LOG("Using synthetic test inputs\n");
        srand(42);
        for (int size : cfg.test_sizes) {
            for (int sample = 0; sample < cfg.n_test_samples; sample++) {
                std::vector<llama_token> tokens;
                if (add_bos) tokens.push_back(llama_vocab_bos(vocab));
                while ((int)tokens.size() < size) {
                    tokens.push_back(rand() % n_vocab);
                }
                test_token_sets.push_back(tokens);
                test_set_infos.push_back({size, sample});
            }
        }
    }

    // Run forward passes with the eval callback
    for (size_t si = 0; si < test_token_sets.size(); si++) {
        const auto & tokens = test_token_sets[si];
        const auto & info   = test_set_infos[si];
        LOG("Running forward pass %zu/%zu: size=%d, sample=%d, %zu tokens...\n",
            si + 1, test_token_sets.size(), info.size, info.sample + 1, tokens.size());

        // Clear KV cache
        llama_memory_clear(llama_get_memory(ctx), true);

        // Decode in a single batch
        int ret = llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(tokens.data()), tokens.size()));
        if (ret != 0) {
            LOG_ERR("llama_decode failed with code %d\n", ret);
            // Try smaller batches
            int batch_size = 64;
            for (int b = 0; b < (int)tokens.size(); b += batch_size) {
                int n = std::min(batch_size, (int)tokens.size() - b);
                ret = llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(tokens.data() + b), n));
                if (ret != 0) {
                    LOG_ERR("llama_decode failed at batch %d with code %d\n", b, ret);
                    break;
                }
            }
        }
    }

    LOG("Captured %d MUL_MAT operations\n", cap_state.captured);
    for (const auto & [rb, captures] : cap_state.captures_by_role) {
        LOG("  %s: %zu captures\n", rb_display(rb).c_str(), captures.size());
    }

    // Free the llama model — we don't need it anymore
    llama_init.reset();
    llama_backend_free();

    // Free Phase 2 scaffolding: only the captured input/output tensors are needed from here.
    // swap-with-empty forces the underlying bucket arrays to actually return memory.
    decltype(cap_state.target_weight_names)().swap(cap_state.target_weight_names);
    decltype(cap_state.weight_to_role)().swap(cap_state.weight_to_role);
    decltype(cap_state.weight_to_layer)().swap(cap_state.weight_to_layer);
    decltype(cap_state.weight_to_bucket)().swap(cap_state.weight_to_bucket);
    std::vector<uint8_t>().swap(cap_state.tmp_data);
    std::vector<std::vector<llama_token>>().swap(test_token_sets);
    std::vector<test_set_info>().swap(test_set_infos);

    // ---- Phase 3: Build cost matrix ----
    LOG("\n--- Phase 3: Building cost matrix ---\n");

    auto cost_matrix = build_cost_matrix(cfg, quantizable, cap_state.captures_by_role,
                                         cfg.model_path, gguf_ctx, imatrix_data);

    // Build preliminary role_info map (per-bucket) so split_fused_roles can
    // apportion fused error by element count at the same bucket.
    std::map<std::string, role_info> roles;
    size_t total_quant_elements = 0;
    for (const auto & ti : quantizable) {
        int bucket = (ti.layer < 0) ? -1 :
                     (layer_to_bucket.count(ti.layer) ? layer_to_bucket[ti.layer] : 0);
        std::string key = make_rb_key(ti.role, bucket);
        auto & ri = roles[key];
        ri.role = key;
        ri.n_elements += ti.n_elements;
        ri.n_bytes_orig += ggml_nbytes(ggml_get_tensor(ggml_ctx, ti.name.c_str()));
        total_quant_elements += ti.n_elements;
    }

    // Split fused-MUL_MAT captures (e.g. attn_qkv) into their component roles.
    split_fused_roles(cost_matrix, roles);

    // Print cost matrix
    LOG("\nCost matrix (avg relative-L2 error by role-bucket × quant type):\n");
    LOG("%-20s", "Role[bucket]");
    for (auto qt : cfg.quant_types) LOG(" %10s", ggml_type_name(qt));
    LOG("\n");
    for (const auto & [key, costs] : cost_matrix) {
        LOG("%-20s", rb_display(key).c_str());
        for (auto qt : cfg.quant_types) {
            auto it = costs.find(qt);
            if (it != costs.end()) {
                LOG(" %10.6f", it->second.kld);
            } else {
                LOG(" %10s", "N/A");
            }
        }
        LOG("\n");
    }

    // ---- Phase 4: Optimize assignment ----
    LOG("\n--- Phase 4: Optimizing assignment ---\n");

    // Determine fixed types for global tensors (before building roles map)
    ggml_type output_type = cfg.output_tensor_type != GGML_TYPE_COUNT
                            ? cfg.output_tensor_type
                            : highest_bpw(cfg.quant_types);

    // Detect tied embeddings: llama-quantize (src/llama-quant.cpp:~534) silently
    // promotes token_embd to output_tensor_type when the arch lacks a distinct
    // output.weight. If we pick a different (lower-bpw) token_embd type in that
    // case, our BPW estimate is off by a lot (e.g. Q4_K vs Q6_K on a 150k-vocab
    // embedding table is ~60MB). Match llama-quantize's behavior up front.
    bool has_output_weight = false;
    for (const auto & ti : all_tensors) {
        if (ti.name == "output.weight") { has_output_weight = true; break; }
    }
    const bool has_tied_embeddings = !has_output_weight;

    ggml_type token_embd_type;
    if (has_tied_embeddings) {
        token_embd_type = output_type;
        LOG("Detected tied embeddings: token_embd will be quantized with output_type (%s)\n",
            ggml_type_name(token_embd_type));
    } else {
        token_embd_type = closest_bpw_if(cfg.quant_types, cfg.target_bpw, supports_get_rows);
        if (token_embd_type == GGML_TYPE_COUNT) {
            token_embd_type = highest_bpw(cfg.quant_types); // fallback
        }
    }

    // Add global tensors (bucket -1) to the cost matrix with their fixed types.
    // KLD=0 since they're not optimized — we just need their BPW in the budget.
    const std::string key_tok_embd = make_rb_key("token_embd", -1);
    const std::string key_output   = make_rb_key("output",     -1);
    for (ggml_type qtype : cfg.quant_types) {
        cost_entry e;
        e.kld = (qtype == token_embd_type) ? 0.0 : 1e30;
        e.bpw = compute_bpw(qtype);
        cost_matrix[key_tok_embd][qtype] = e;

        e.kld = (qtype == output_type) ? 0.0 : 1e30;
        cost_matrix[key_output][qtype] = e;
    }
    LOG("Global tensors: token_embd=%s (%.4f bpw), output=%s (%.4f bpw)\n",
        ggml_type_name(token_embd_type), compute_bpw(token_embd_type),
        ggml_type_name(output_type), compute_bpw(output_type));

    // Small 2D .weight tensors below min_elements aren't measured for KLD, but
    // llama-quantize will still quantize them (to the fallback ftype passed on
    // the CLI). We don't know the user's chosen fallback, so we approximate it
    // as the listed type closest to the target BPW.
    ggml_type small_matrix_type = closest_bpw(cfg.quant_types, cfg.target_bpw);
    if (small_matrix_type == GGML_TYPE_COUNT) small_matrix_type = token_embd_type;

    size_t total_all_elements = 0;
    double non_quantizable_bits = 0;
    size_t small_matrix_elements = 0;
    for (const auto & ti : all_tensors) {
        total_all_elements += ti.n_elements;
        if (is_quantizable_weight(ti, cfg.min_elements)) {
            continue;  // counted via role_to_type in compute_total_bpw
        }
        if (is_matrix_weight(ti)) {
            // Small 2D weight: llama-quantize will quantize it with the fallback ftype.
            non_quantizable_bits += compute_bpw(small_matrix_type) * ti.n_elements;
            small_matrix_elements += ti.n_elements;
        } else {
            // 1D, biases, norms: stay at original precision.
            non_quantizable_bits += compute_bpw(ti.orig_type) * ti.n_elements;
        }
    }

    LOG("Total elements in quantizable tensors: %zu\n", total_quant_elements);
    LOG("Total elements in all tensors:         %zu\n", total_all_elements);
    LOG("Non-quantizable overhead:              %.4f BPW\n",
        total_all_elements > 0 ? non_quantizable_bits / total_all_elements : 0);
    if (small_matrix_elements > 0) {
        LOG("Small matrix weights below --min-elements (%zu elements) assumed %s for BPW estimate\n",
            small_matrix_elements, ggml_type_name(small_matrix_type));
    }
    for (const auto & [key, ri] : roles) {
        LOG("  %s: %zu elements%s\n", rb_display(key).c_str(), ri.n_elements,
            cost_matrix.count(key) ? "" : " [NO CAPTURES]");
    }

    // Warn about any quantizable (role, bucket) pairs that still have no captures.
    // With proper layer equivalence class sampling this should be rare, but can
    // happen for tensors that don't appear as MUL_MAT in any layer type.
    {
        for (const auto & [key, ri] : roles) {
            if (cost_matrix.count(key)) continue;
            LOG("  WARNING: %s has no captures — assigning error=0 for all types\n",
                rb_display(key).c_str());
            for (ggml_type qtype : cfg.quant_types) {
                cost_entry e;
                e.kld = 0.0;
                e.bpw = compute_bpw(qtype);
                cost_matrix[key][qtype] = e;
            }
        }
    }
    auto result = optimize_assignment(cfg, cost_matrix, roles, total_all_elements, non_quantizable_bits);

    LOG("\nOptimal assignment:\n");
    for (const auto & [key, type] : result.role_to_type) {
        auto it = cost_matrix.find(key);
        double kld = (it != cost_matrix.end() && it->second.count(type)) ?
                      it->second.at(type).kld : -1;
        LOG("  %-20s = %-10s  (error=%.6f, BPW=%.4f)\n",
            rb_display(key).c_str(), ggml_type_name(type), kld, compute_bpw(type));
    }
    LOG("Total BPW: %.4f (target: %.2f +%.2f/-%.2f)\n",
        result.total_bpw, cfg.target_bpw, cfg.bpw_tol_high, cfg.bpw_tol_low);
    LOG("Total error (unweighted sum): %.6f\n",
        compute_total_kld(result.role_to_type, cost_matrix));

    // ---- Phase 5: Output tensor-type-file ----
    LOG("\n--- Phase 5: Writing output ---\n");
    write_tensor_type_file(cfg.output_path, result.role_to_type,
                           output_type, token_embd_type, bucket_layers);

    // Cleanup
    gguf_free(gguf_ctx);
    ggml_free(ggml_ctx);

    LOG("\nDone!\n");
    return 0;
}
