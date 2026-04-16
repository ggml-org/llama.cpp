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
    std::vector<int> test_sizes = {32, 128, 512};
    int64_t min_elements        = 40000;
    ggml_type output_tensor_type = GGML_TYPE_COUNT; // default: highest from list
    int max_iterations           = 100;
    std::string output_path;
    int n_threads                = 1;
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

// Cost of assigning a specific quant type to a role
struct cost_entry {
    double kld;   // average KLD across all captures for this role
    double bpw;   // bits per weight for this quant type
};

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

static bool is_quantizable_weight(const tensor_info & ti, int64_t min_elements) {
    // Must end with .weight
    if (ti.name.size() < 8 || ti.name.substr(ti.name.size() - 7) != ".weight") return false;
    // Must be 2D (matrix)
    if (ti.ne[2] != 1 || ti.ne[3] != 1) return false;
    // Must have enough elements
    if ((int64_t)ti.n_elements < min_elements) return false;
    // Skip norms (1D-like: small ne[0] or ne[1] == 1)
    if (ti.ne[0] <= 1 || ti.ne[1] <= 1) return false;
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

// Compute KLD between two rows treated as probability distributions (after softmax)
// p = softmax(ref), q = softmax(quant)
// KLD = sum(p * log(p/q))
static double compute_kld_row(const float * ref, const float * quant, int64_t n) {
    // Find max for numerical stability
    float max_ref = -1e30f, max_qt = -1e30f;
    for (int64_t i = 0; i < n; i++) {
        if (ref[i] > max_ref) max_ref = ref[i];
        if (quant[i] > max_qt) max_qt = quant[i];
    }

    // Compute softmax
    std::vector<float> p(n), q(n);
    double sum_p = 0, sum_q = 0;
    for (int64_t i = 0; i < n; i++) {
        p[i] = expf(ref[i] - max_ref);
        q[i] = expf(quant[i] - max_qt);
        sum_p += p[i];
        sum_q += q[i];
    }
    float inv_sum_p = 1.0f / (float)sum_p;
    float inv_sum_q = 1.0f / (float)sum_q;
    for (int64_t i = 0; i < n; i++) {
        p[i] *= inv_sum_p;
        q[i] *= inv_sum_q;
    }

    // KLD = sum(p * log(p/q))
    const float eps = 1e-10f;
    double kld = 0;
    for (int64_t i = 0; i < n; i++) {
        if (p[i] > eps) {
            float q_clipped = std::max(q[i], eps);
            kld += (double)p[i] * log((double)p[i] / (double)q_clipped);
        }
    }
    return kld;
}

// Compute average KLD across all rows of two F32 matrices of the same shape [ne0, ne1]
static double compute_avg_kld(const float * ref, const float * quant, int64_t ne0, int64_t ne1) {
    double total_kld = 0;
    int64_t valid_rows = 0;
    for (int64_t row = 0; row < ne1; row++) {
        const float * ref_row   = ref   + row * ne0;
        const float * quant_row = quant + row * ne0;
        double kld = compute_kld_row(ref_row, quant_row, ne0);
        if (std::isfinite(kld)) {
            total_kld += kld;
            valid_rows++;
        }
    }
    return valid_rows > 0 ? total_kld / valid_rows : 1e30;
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

    // Captures, organized by role
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

    state->captures_by_role[cap.role].push_back(std::move(cap));
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
        w->quant_levels = (void *)quant_levels.data();
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
        const std::unordered_map<std::string, std::vector<mul_mat_capture>> & captures_by_role,
        const std::string & model_path,
        const struct gguf_context * gguf_ctx,
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data) {

    std::map<std::string, std::map<ggml_type, cost_entry>> cost_matrix;

    // Accumulate KLD per (role, quant_type)
    std::mutex accum_mutex;
    std::map<std::string, std::map<ggml_type, double>> kld_sums;
    std::map<std::string, std::map<ggml_type, int>> kld_counts;

    // Cache for F32 weight data: tensor_name → float data
    std::unordered_map<std::string, std::vector<float>> weight_cache;

    const int n_parallel = std::max(1, cfg.n_threads);

    // Create a pool of backends — try CUDA first, fall back to CPU
    // Each thread gets its own backend to avoid contention.
    struct backend_pool {
        std::vector<ggml_backend_t> backends;
        std::atomic<int> next_idx{0};
        std::string backend_name;

        backend_pool(int count) {
            // Use ggml_backend_init_best() — picks GPU if available, falls back to CPU
            ggml_backend_t be = ggml_backend_init_best();
            if (!be) {
                LOG_ERR("Failed to init any backend\n");
                return;
            }
            backend_name = ggml_backend_name(be);
            // Single shared backend — ggml handles concurrency internally
            for (int i = 0; i < count; i++) {
                backends.push_back(be);
            }
        }
        ~backend_pool() {
            if (!backends.empty()) {
                ggml_backend_free(backends[0]);
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
    for (const auto & [role, captures] : captures_by_role) {
        LOG("Building cost matrix for role '%s' (%zu captures, %d parallel threads)...\n",
            role.c_str(), captures.size(), n_parallel);

        for (const auto & cap : captures) {
            // Read weight data as F32 (cached)
            auto cache_it = weight_cache.find(cap.weight_name);
            if (cache_it == weight_cache.end()) {
                std::vector<float> weight_f32;
                if (!read_tensor_f32(model_path, gguf_ctx, cap.weight_name, weight_f32)) {
                    LOG_WRN("  Failed to read weight data for '%s', skipping\n", cap.weight_name.c_str());
                    continue;
                }
                weight_cache[cap.weight_name] = std::move(weight_f32);
                cache_it = weight_cache.find(cap.weight_name);
            }
            const auto & weight_f32 = cache_it->second;

            // Get imatrix for this tensor
            auto imat = get_imatrix_for_tensor(imatrix_data, cap.weight_name, cap.weight_ne0);

            // Evaluate all quant types for this capture in parallel batches
            const size_t n_types = cfg.quant_types.size();
            std::vector<std::future<std::pair<ggml_type, double>>> futures;
            futures.reserve(n_parallel);

            for (size_t ti = 0; ti < n_types; /* advanced inside */) {
                // Launch up to n_parallel evaluations
                futures.clear();
                for (int p = 0; p < n_parallel && ti < n_types; p++, ti++) {
                    ggml_type qtype = cfg.quant_types[ti];
                    ggml_backend_t backend = pool.get();
                    futures.push_back(std::async(std::launch::async,
                        [&cap, &weight_f32, &imat, qtype, backend]() -> std::pair<ggml_type, double> {
                            // Quantize (sets per-type global state — safe since each type is different)
                            auto qres = quantize_weight_to_type(
                                qtype, weight_f32.data(), cap.weight_ne1, cap.weight_ne0, imat.data());

                            // Run MUL_MAT with quantized weight
                            std::vector<float> quant_output;
                            if (!eval_mul_mat(qtype, qres.data.data(),
                                              cap.weight_ne0, cap.weight_ne1,
                                              cap.input_data.data(), cap.input_ne0, cap.input_ne1,
                                              qres.levels,
                                              quant_output,
                                              backend)) {
                                return {qtype, std::numeric_limits<double>::quiet_NaN()};
                            }

                            // Compute KLD
                            return {qtype, compute_avg_kld(cap.ref_output_data.data(), quant_output.data(),
                                                           cap.ref_ne0, cap.ref_ne1)};
                        }));
                }

                // Collect results
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

        // Free cache entries for this role's captures (no longer needed)
        for (const auto & cap : captures) {
            weight_cache.erase(cap.weight_name);
        }

        // Build cost entries from accumulated KLD
        for (ggml_type qtype : cfg.quant_types) {
            int n_valid = kld_counts[role][qtype];
            cost_entry entry;
            entry.kld = n_valid > 0 ? kld_sums[role][qtype] / n_valid : 1e30;
            entry.bpw = compute_bpw(qtype);
            cost_matrix[role][qtype] = entry;
        }
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

// Post-process the cost matrix: for fused roles, split KLD proportionally
// among component roles (by element count). Component roles that already
// have their own captures keep their directly-measured KLD.
static void split_fused_roles(
        std::map<std::string, std::map<ggml_type, cost_entry>> & cost_matrix,
        const std::map<std::string, role_info> & roles) {

    for (const auto & [fused_role, components] : fusion_map) {
        auto fused_it = cost_matrix.find(fused_role);
        if (fused_it == cost_matrix.end()) continue;

        // Compute element counts for each component role
        size_t total_comp_elements = 0;
        std::vector<size_t> comp_elems;
        for (const auto & comp : components) {
            auto rit = roles.find(comp);
            size_t ne = (rit != roles.end()) ? rit->second.n_elements : 0;
            comp_elems.push_back(ne);
            total_comp_elements += ne;
        }
        if (total_comp_elements == 0) continue;

        // For each quant type, create proportional cost entries for component roles
        for (const auto & [qtype, fused_entry] : fused_it->second) {
            for (size_t i = 0; i < components.size(); i++) {
                double fraction = (double)comp_elems[i] / (double)total_comp_elements;

                auto & comp_entries = cost_matrix[components[i]];
                if (comp_entries.count(qtype)) {
                    // Component already has its own measurement — keep it, don't overwrite
                    continue;
                }

                cost_entry comp_entry;
                comp_entry.kld = fused_entry.kld * fraction;
                comp_entry.bpw = fused_entry.bpw;
                comp_entries[qtype] = comp_entry;
            }
        }

        LOG("Split fused role '%s' KLD into components:", fused_role.c_str());
        for (size_t i = 0; i < components.size(); i++) {
            double fraction = (double)comp_elems[i] / (double)total_comp_elements;
            LOG("  %s: %.1f%% (%zu elements)\n",
                components[i].c_str(), fraction * 100, comp_elems[i]);
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

static assignment optimize_assignment(
        const config & cfg,
        const std::map<std::string, std::map<ggml_type, cost_entry>> & cost_matrix,
        const std::map<std::string, role_info> & roles,
        size_t total_all_elements,
        double non_quantizable_bits) {

    auto types_by_bpw = sorted_by_bpw(cfg.quant_types);
    if (types_by_bpw.empty()) {
        LOG_ERR("No quant types specified\n");
        return {};
    }

    // Initialize: assign each role the lowest-BPW quant type (greedy fill)
    // We start from the bottom and work up until we hit the BPW budget
    std::map<std::string, ggml_type> current;
    for (const auto & [role, info] : roles) {
        // Start with the lowest-BPW type that has a valid KLD
        auto it = cost_matrix.find(role);
        if (it == cost_matrix.end()) continue;
        ggml_type best = GGML_TYPE_COUNT;
        for (auto qt : types_by_bpw) {
            auto it2 = it->second.find(qt);
            if (it2 != it->second.end() && it2->second.kld < 1e29) {
                best = qt;
                break;
            }
        }
        if (best == GGML_TYPE_COUNT) continue;
        current[role] = best;
    }

    // Greedily upgrade roles to higher-quality types until we hit the BPW budget
    // For each role, find the "best value" upgrade (most KLD reduction per BPW increase)
    bool improved = true;
    while (improved) {
        improved = false;
        (void)0; // cur_bpw used only for debugging

        double best_ratio = 0;
        std::string best_role;
        ggml_type best_type = GGML_TYPE_COUNT;

        for (const auto & [role, cur_type] : current) {
            // Find next higher-BPW type
            auto it = cost_matrix.find(role);
            if (it == cost_matrix.end()) continue;

            double cur_kld = it->second.count(cur_type) ? it->second.at(cur_type).kld : 1e30;

            for (auto qt : types_by_bpw) {
                if (compute_bpw(qt) <= compute_bpw(cur_type)) continue; // skip lower/same
                auto it2 = it->second.find(qt);
                if (it2 == it->second.end() || it2->second.kld >= 1e29) continue;

                // Check if this upgrade would exceed BPW budget
                auto test = current;
                test[role] = qt;
                double test_bpw = compute_total_bpw(test, roles, total_all_elements, non_quantizable_bits);
                if (test_bpw > cfg.target_bpw + cfg.bpw_tol_high) continue;

                // Compute improvement ratio (KLD reduction per BPW increase)
                double kld_reduction = cur_kld - it2->second.kld;
                double bpw_increase = compute_bpw(qt) - compute_bpw(cur_type);
                if (bpw_increase <= 0) continue;
                double ratio = kld_reduction / bpw_increase;

                if (ratio > best_ratio) {
                    best_ratio = ratio;
                    best_role = role;
                    best_type = qt;
                }
            }
        }

        if (!best_role.empty() && best_type != GGML_TYPE_COUNT) {
            current[best_role] = best_type;
            improved = true;
        }
    }

    // Iterative improvement: try swapping roles up/down
    std::set<std::vector<std::pair<std::string, ggml_type>>> visited;
    auto make_key = [&](const std::map<std::string, ggml_type> & a) {
        std::vector<std::pair<std::string, ggml_type>> v(a.begin(), a.end());
        std::sort(v.begin(), v.end());
        return v;
    };

    assignment best_assign;
    best_assign.role_to_type = current;
    best_assign.total_bpw = compute_total_bpw(current, roles, total_all_elements, non_quantizable_bits);
    best_assign.total_kld = compute_total_kld(current, cost_matrix);
    visited.insert(make_key(current));

    for (int iter = 0; iter < cfg.max_iterations; iter++) {
        bool found_improvement = false;

        // Try upgrading each role and downgrading another to compensate
        for (const auto & [role_up, cur_type_up] : best_assign.role_to_type) {
            auto it_up = cost_matrix.find(role_up);
            if (it_up == cost_matrix.end()) continue;

            for (auto qt_up : types_by_bpw) {
                if (compute_bpw(qt_up) <= compute_bpw(cur_type_up)) continue;
                auto it2_up = it_up->second.find(qt_up);
                if (it2_up == it_up->second.end() || it2_up->second.kld >= 1e29) continue;

                (void)0; // bpw_increase no longer used directly
                double kld_decrease_up = (it_up->second.count(cur_type_up) ?
                    it_up->second.at(cur_type_up).kld : 1e30) - it2_up->second.kld;

                // Try downgrading each other role to compensate
                for (const auto & [role_dn, cur_type_dn] : best_assign.role_to_type) {
                    if (role_dn == role_up) continue;
                    auto it_dn = cost_matrix.find(role_dn);
                    if (it_dn == cost_matrix.end()) continue;

                    for (auto qt_dn : types_by_bpw) {
                        if (compute_bpw(qt_dn) >= compute_bpw(cur_type_dn)) continue;
                        auto it2_dn = it_dn->second.find(qt_dn);
                        if (it2_dn == it_dn->second.end() || it2_dn->second.kld >= 1e29) continue;

                        // Check BPW constraint
                        auto test = best_assign.role_to_type;
                        test[role_up] = qt_up;
                        test[role_dn] = qt_dn;
                        double test_bpw = compute_total_bpw(test, roles, total_all_elements, non_quantizable_bits);
                        if (test_bpw > cfg.target_bpw + cfg.bpw_tol_high) continue;
                        if (test_bpw < cfg.target_bpw - cfg.bpw_tol_low) continue;

                        // Check if already visited
                        auto key = make_key(test);
                        if (visited.count(key)) continue;

                        // Compute KLD change
                        double kld_increase_dn = it2_dn->second.kld -
                            (it_dn->second.count(cur_type_dn) ? it_dn->second.at(cur_type_dn).kld : 1e30);
                        double net_kld_change = -kld_decrease_up + kld_increase_dn;

                        if (net_kld_change < -1e-10) {
                            // Improvement found!
                            double new_kld = best_assign.total_kld + net_kld_change;
                            visited.insert(key);

                            best_assign.role_to_type = test;
                            best_assign.total_bpw = test_bpw;
                            best_assign.total_kld = new_kld;
                            found_improvement = true;
                            break;
                        }
                        visited.insert(key);
                    }
                    if (found_improvement) break;
                }
                if (found_improvement) break;
            }
            if (found_improvement) break;
        }

        if (!found_improvement) {
            // Also try single-role upgrades (if BPW budget allows)
            for (const auto & [role, cur_type] : best_assign.role_to_type) {
                auto it = cost_matrix.find(role);
                if (it == cost_matrix.end()) continue;

                for (auto qt : types_by_bpw) {
                    if (compute_bpw(qt) <= compute_bpw(cur_type)) continue;
                    auto it2 = it->second.find(qt);
                    if (it2 == it->second.end() || it2->second.kld >= 1e29) continue;

                    auto test = best_assign.role_to_type;
                    test[role] = qt;
                    double test_bpw = compute_total_bpw(test, roles, total_all_elements, non_quantizable_bits);
                    if (test_bpw > cfg.target_bpw + cfg.bpw_tol_high) continue;

                    auto key = make_key(test);
                    if (visited.count(key)) continue;

                    double kld_change = it2->second.kld -
                        (it->second.count(cur_type) ? it->second.at(cur_type).kld : 1e30);
                    if (kld_change < -1e-10) {
                        best_assign.role_to_type = test;
                        best_assign.total_bpw = test_bpw;
                        best_assign.total_kld += kld_change;
                        visited.insert(key);
                        found_improvement = true;
                        break;
                    }
                    visited.insert(key);
                }
                if (found_improvement) break;
            }
        }

        if (!found_improvement) break;

        LOG("  Iteration %d: BPW=%.4f, total_KLD=%.6f\n",
            iter, best_assign.total_bpw, best_assign.total_kld);
    }

    return best_assign;
}

// ============================================================================
// Section 8: Output
// ============================================================================

static bool write_tensor_type_file(const std::string & path,
                                   const std::map<std::string, ggml_type> & role_to_type,
                                   ggml_type output_tensor_type,
                                   ggml_type token_embd_tensor_type) {
    std::ofstream file(path);
    if (!file) {
        LOG_ERR("Failed to open output file: %s\n", path.c_str());
        return false;
    }

    // Write token_embd type — anchor with ^ so it only matches the global tensor,
    // not e.g. some hypothetical "blk.X.something_token_embd.weight"
    if (token_embd_tensor_type != GGML_TYPE_COUNT) {
        file << "^token_embd=" << ggml_type_name(token_embd_tensor_type) << "\n";
    }

    // Write output tensor type — anchor with ^ so "output" matches "output.weight"
    // but NOT "blk.X.attn_output.weight"
    if (output_tensor_type != GGML_TYPE_COUNT) {
        file << "^output=" << ggml_type_name(output_tensor_type) << "\n";
    }

    // Write per-role types (skip global tensors already written above).
    // Use \.ROLE\. patterns so they match between dots, e.g.:
    //   "ffn_down"  →  "\.ffn_down\."   matches blk.X.ffn_down.weight
    //   "attn_q"    →  "\.attn_q\."     matches blk.X.attn_q.weight but NOT blk.X.attn_qkv.weight
    //   "attn_qkv"  →  "\.attn_qkv\."   matches blk.X.attn_qkv.weight
    for (const auto & [role, type] : role_to_type) {
        if (role == "token_embd" || role == "output") continue;
        file << "\\." << role << "\\.=" << ggml_type_name(type) << "\n";
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
    LOG("  --test-sizes S1,S2,S3    Token counts for test inputs (default: 32,128,512)\n");
    LOG("  --min-elements N         Skip tensors with fewer elements (default: 40000)\n");
    LOG("  --output-tensor-type T   Quant type for output.weight (default: highest from list)\n");
    LOG("  --max-iterations N       Max optimization iterations (default: 100)\n");
    LOG("  --threads N              Number of threads (default: 1)\n");
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

    struct ggml_context * ggml_ctx = nullptr;
    struct gguf_init_params gguf_params = {false, &ggml_ctx};
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
    params.n_batch = 512;
    params.n_ubatch = 512;
    params.n_ctx = 1024;  // enough for test sizes

    capture_state cap_state;
    for (const auto & ti : quantizable) {
        if (target_weight_names.count(ti.name)) {
            cap_state.target_weight_names.insert(ti.name);
            cap_state.weight_to_role[ti.name] = ti.role;
            cap_state.weight_to_layer[ti.name] = ti.layer;
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

    // Prepare test tokens for each size
    std::vector<std::vector<llama_token>> test_token_sets;

    if (!cfg.test_data_path.empty()) {
        // Read test data file
        std::ifstream tf(cfg.test_data_path);
        if (!tf) {
            LOG_ERR("Failed to open test data file: %s\n", cfg.test_data_path.c_str());
            return 1;
        }
        std::string test_text((std::istreambuf_iterator<char>(tf)),
                               std::istreambuf_iterator<char>());
        auto all_tokens = common_tokenize(ctx, test_text, add_bos, false);
        LOG("Tokenized test data: %zu tokens\n", all_tokens.size());

        for (int size : cfg.test_sizes) {
            int n = std::min(size, (int)all_tokens.size());
            test_token_sets.push_back(std::vector<llama_token>(all_tokens.begin(), all_tokens.begin() + n));
        }
    } else {
        // Synthetic: random tokens
        LOG("Using synthetic test inputs\n");
        srand(42);
        for (int size : cfg.test_sizes) {
            std::vector<llama_token> tokens;
            if (add_bos) tokens.push_back(llama_vocab_bos(vocab));
            while ((int)tokens.size() < size) {
                tokens.push_back(rand() % n_vocab);
            }
            test_token_sets.push_back(tokens);
        }
    }

    // Run forward passes with the eval callback
    for (size_t si = 0; si < test_token_sets.size(); si++) {
        const auto & tokens = test_token_sets[si];
        LOG("Running forward pass with %zu tokens (size %d)...\n",
            tokens.size(), cfg.test_sizes[si]);

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
    for (const auto & [role, captures] : cap_state.captures_by_role) {
        LOG("  Role '%s': %zu captures\n", role.c_str(), captures.size());
    }

    // Free the llama model — we don't need it anymore
    llama_init.reset();
    llama_backend_free();

    // ---- Phase 3: Build cost matrix ----
    LOG("\n--- Phase 3: Building cost matrix ---\n");

    auto cost_matrix = build_cost_matrix(cfg, quantizable, cap_state.captures_by_role,
                                         cfg.model_path, gguf_ctx, imatrix_data);

    // Print cost matrix
    LOG("\nCost matrix (avg KLD by role × quant type):\n");
    LOG("%-16s", "Role");
    for (auto qt : cfg.quant_types) LOG(" %10s", ggml_type_name(qt));
    LOG("\n");
    for (const auto & [role, costs] : cost_matrix) {
        LOG("%-16s", role.c_str());
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
    ggml_type token_embd_type = closest_bpw_if(cfg.quant_types, cfg.target_bpw, supports_get_rows);
    if (token_embd_type == GGML_TYPE_COUNT) {
        token_embd_type = highest_bpw(cfg.quant_types); // fallback
    }
    ggml_type output_type = cfg.output_tensor_type != GGML_TYPE_COUNT
                            ? cfg.output_tensor_type
                            : highest_bpw(cfg.quant_types);

    // Add global tensors to the cost matrix with their fixed types.
    // KLD=0 since they're not optimized — we just need their BPW in the budget.
    for (ggml_type qtype : cfg.quant_types) {
        cost_entry e;
        e.kld = (qtype == token_embd_type) ? 0.0 : 1e30;
        e.bpw = compute_bpw(qtype);
        cost_matrix["token_embd"][qtype] = e;

        e.kld = (qtype == output_type) ? 0.0 : 1e30;
        cost_matrix["output"][qtype] = e;
    }
    LOG("Global tensors: token_embd=%s (%.4f bpw), output=%s (%.4f bpw)\n",
        ggml_type_name(token_embd_type), compute_bpw(token_embd_type),
        ggml_type_name(output_type), compute_bpw(output_type));

    // Build role info (n_elements, n_bytes per role) — include ALL quantizable tensors
    std::map<std::string, role_info> roles;
    size_t total_quant_elements = 0;
    for (const auto & ti : quantizable) {
        auto & ri = roles[ti.role];
        ri.role = ti.role;
        ri.n_elements += ti.n_elements;
        ri.n_bytes_orig += ggml_nbytes(ggml_get_tensor(ggml_ctx, ti.name.c_str()));
        total_quant_elements += ti.n_elements;
    }

    // Compute non-quantizable tensor overhead (small tensors kept at original precision)
    // These contribute fixed BPW that must be accounted for in the budget.
    size_t total_all_elements = 0;
    double non_quantizable_bits = 0;
    for (const auto & ti : all_tensors) {
        total_all_elements += ti.n_elements;
        if (!is_quantizable_weight(ti, cfg.min_elements)) {
            non_quantizable_bits += compute_bpw(ti.orig_type) * ti.n_elements;
        }
    }

    LOG("Total elements in quantizable tensors: %zu\n", total_quant_elements);
    LOG("Total elements in all tensors:         %zu\n", total_all_elements);
    LOG("Non-quantizable overhead:              %.4f BPW\n",
        total_all_elements > 0 ? non_quantizable_bits / total_all_elements : 0);
    for (const auto & [role, ri] : roles) {
        LOG("  %s: %zu elements%s\n", role.c_str(), ri.n_elements,
            cost_matrix.count(role) ? "" : " [NO CAPTURES]");
    }

    // Warn about any quantizable roles that still have no captures.
    // With proper layer equivalence class sampling, this should be rare,
    // but can happen for tensors that don't appear as MUL_MAT in any layer type.
    {
        for (const auto & [role, ri] : roles) {
            if (cost_matrix.count(role)) continue;
            LOG("  WARNING: Role '%s' has no captures — assigning KLD=0 for all types\n",
                role.c_str());
            for (ggml_type qtype : cfg.quant_types) {
                cost_entry e;
                e.kld = 0.0;
                e.bpw = compute_bpw(qtype);
                cost_matrix[role][qtype] = e;
            }
        }
    }
    auto result = optimize_assignment(cfg, cost_matrix, roles, total_all_elements, non_quantizable_bits);

    LOG("\nOptimal assignment:\n");
    for (const auto & [role, type] : result.role_to_type) {
        auto it = cost_matrix.find(role);
        double kld = (it != cost_matrix.end() && it->second.count(type)) ?
                      it->second.at(type).kld : -1;
        LOG("  %-16s = %-10s  (KLD=%.6f, BPW=%.4f)\n",
            role.c_str(), ggml_type_name(type), kld, compute_bpw(type));
    }
    LOG("Total BPW: %.4f (target: %.2f +%.2f/-%.2f)\n",
        result.total_bpw, cfg.target_bpw, cfg.bpw_tol_high, cfg.bpw_tol_low);
    LOG("Total KLD: %.6f\n", result.total_kld);

    // ---- Phase 5: Output tensor-type-file ----
    LOG("\n--- Phase 5: Writing output ---\n");
    write_tensor_type_file(cfg.output_path, result.role_to_type,
                           output_type, token_embd_type);

    // Cleanup
    gguf_free(gguf_ctx);
    ggml_free(ggml_ctx);

    LOG("\nDone!\n");
    return 0;
}
