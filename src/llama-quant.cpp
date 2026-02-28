#include "llama-quant.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include <cmath>
#include <mutex>
#include <regex>
#include <atomic>
#include <thread>
#include <cstring>
#include <fstream>
#include <cinttypes>
#include <unordered_map>
#include <condition_variable>

// Quantization types. Changes to this struct must be replicated in quantize.cpp
struct tensor_quantization {
    std::string name;
    ggml_type quant = GGML_TYPE_COUNT;
};

// tensor categorization (used to avoid repeated string matching)
enum class tensor_category {
    TOKEN_EMBD,
    ATTENTION_Q,
    ATTENTION_V,
    ATTENTION_K,
    ATTENTION_QKV,
    ATTENTION_KV_B,
    ATTENTION_OUTPUT,
    FFN_UP,
    FFN_GATE,
    FFN_DOWN,
    OUTPUT,
    OTHER
};

// cached metadata per tensor
struct tensor_metadata {
    tensor_category category;
    ggml_type target_type;
    std::string remapped_imatrix_name;
    bool allows_quantization;
    bool requires_imatrix;
};

// threads are spawned once and reused for all work, avoiding overhead
struct quantize_thread_pool {
    std::vector<std::thread> workers;
    int32_t n_workers = 0;

    size_t batch_idx = 0; // incremented each distribute call

    // synchronization
    std::mutex mutex;
    std::condition_variable cv_start; // wakes workers when a new batch is ready
    std::condition_variable cv_done;  // wakes main when all workers are idle

    // current batch parameters
    int64_t n_items = 0;
    std::atomic<int64_t> next_item{0};
    std::function<void(int64_t)> task_fn;

    // barrier state (protected by mutex)
    int n_ready  = 0; // number of workers currently idle and waiting
    bool stopping = false;

    void start(int nthread) {
        n_workers = std::max(0, nthread - 1);
        workers.reserve(n_workers);
        for (int i = 0; i < n_workers; i++) {
            workers.emplace_back([this] { worker(); });
        }
        if (n_workers > 0) {
            // wait for all workers to enter their idle state
            std::unique_lock<std::mutex> lock(mutex);
            cv_done.wait(lock, [this] { return n_ready == n_workers; });
        }
    }

    void worker() {
        uint64_t my_batch = 0;
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            // signal that this worker is idle
            n_ready++;
            if (n_ready == n_workers) {
                cv_done.notify_one();
            }

            // wait for new work or shutdown
            cv_start.wait(lock, [&] { return stopping || batch_idx > my_batch; });
            if (stopping) {
                return;
            }

            // leave idle state and grab batch parameters
            n_ready--;
            my_batch = batch_idx;
            lock.unlock();

            // grab work items until exhausted
            while (true) {
                const int64_t idx = next_item.fetch_add(1, std::memory_order_relaxed);
                if (idx >= n_items) {
                    break;
                }
                task_fn(idx);
            }

            lock.lock();
        }
    }

    // distribute fn(0..n-1) across all pool threads + the calling thread.
    // blocks until every item has been processed.
    void distribute(int64_t n, std::function<void(int64_t)> fn) {
        if (n <= 0) {
            return;
        }

        if (n_workers == 0) {
            // single-threaded fast path: no synchronization overhead
            for (int64_t i = 0; i < n; i++) {
                fn(i);
            }
            return;
        }

        {
            std::unique_lock<std::mutex> lock(mutex);
            // ensure all workers are idle from previous batch
            cv_done.wait(lock, [this] { return n_ready == n_workers; });

            // set up the new batch
            task_fn = std::move(fn);
            n_items = n;
            next_item.store(0, std::memory_order_relaxed);
            batch_idx++;
        }
        cv_start.notify_all();

        // main thread participates in the work
        while (true) {
            const int64_t idx = next_item.fetch_add(1, std::memory_order_relaxed);
            if (idx >= n_items) {
                break;
            }
            task_fn(idx);
        }

        // wait for all workers to finish and return to idle
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv_done.wait(lock, [this] { return n_ready == n_workers; });
        }
    }

    void stop() {
        if (workers.empty()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        cv_start.notify_all();
        for (auto & w : workers) {
            w.join();
        }
        workers.clear();
        n_workers = 0;
    }

    ~quantize_thread_pool() {
        stop();
    }

    // non-copyable, non-movable
    quantize_thread_pool() = default;
    quantize_thread_pool(const quantize_thread_pool &) = delete;
    quantize_thread_pool & operator=(const quantize_thread_pool &) = delete;
};

struct quantize_state_impl {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    // per-model
    int32_t n_attention_wv = 0;
    int32_t n_ffn_down     = 0;
    int32_t n_ffn_gate     = 0;
    int32_t n_ffn_up       = 0;

    // per-layer
    int32_t i_attention_wv = 0;
    int32_t i_ffn_down     = 0;
    int32_t i_ffn_gate     = 0;
    int32_t i_ffn_up       = 0;

    // per-quantization
    int32_t n_fallback = 0;

    // flags
    bool has_imatrix = false; // do we have imatrix data?
    bool has_output  = false; // used to figure out if a model shares tok_embd with the output weight

    // tensor type override patterns
    std::vector<std::pair<std::regex, ggml_type>> tensor_type_patterns;

    // worker thread pool
    quantize_thread_pool pool;

    quantize_state_impl(const llama_model & model, const llama_model_quantize_params * params)
        : model(model)
        , params(params)
    {
        // compile regex patterns once - they are expensive, and used twice
        if (params->tensor_types) {
            const auto & tensor_types = *static_cast<const std::vector<tensor_quantization> *>(params->tensor_types);
            for (const auto & [tname, qtype] : tensor_types) {
                tensor_type_patterns.emplace_back(std::regex(tname), qtype);
            }
        }
    }
};

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

static std::string remap_layer(const std::string & orig_name, const std::vector<int> & prune, std::map<int, std::string> & mapped, int & next_id) {
    if (prune.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const int blk = std::stoi(match[1]);
        std::string new_name = orig_name;

        if (mapped.count(blk)) {
            // Already mapped, do nothing
        } else if (std::find(prune.begin(), prune.end(), blk) != prune.end()) {
            mapped[blk] = "";
        } else if (blk < prune.front()) {
            mapped[blk] = std::to_string(blk);
            next_id = blk + 1;
        } else {
            mapped[blk] = std::to_string(next_id);
            ++next_id;
        }

        return mapped[blk].empty() ? mapped[blk] : new_name.replace(match.position(1), match.length(1), mapped[blk]);
    }

    return orig_name;
}

static std::string remap_imatrix(const std::string & orig_name, const std::map<int, std::string> & mapped) {
    if (mapped.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const std::string blk(match[1]);
        std::string new_name = orig_name;

        for (const auto & p : mapped) {
            if (p.second == blk) {
                LLAMA_LOG_DEBUG("(blk.%d imatrix) ", p.first);
                return new_name.replace(match.position(1), match.length(1), std::to_string(p.first));
            }
        }
        GGML_ABORT("\n%s: imatrix mapping error for %s\n", __func__, orig_name.c_str());
    }

    return orig_name;
}

static bool tensor_name_match_token_embd(const char * tensor_name) {
    return std::strcmp(tensor_name, "token_embd.weight") == 0 ||
           std::strcmp(tensor_name, "per_layer_token_embd.weight") == 0;
}

static bool tensor_name_match_output_weight(const char * tensor_name) {
    return std::strcmp(tensor_name, "output.weight") == 0;
}

// do we allow this tensor to be quantized?
static bool tensor_allows_quantization(const llama_model_quantize_params * params, llm_arch arch, const ggml_tensor * tensor) {
    // trivial checks first — no string ops needed
    if (params->only_copy)       return false;

    // quantize only 2D and 3D tensors
    if (ggml_n_dims(tensor) < 2) return false;

    const std::string_view name_v(tensor->name);

    // must end with "weight"
    if (name_v.size() < 6 || name_v.compare(name_v.size() - 6, 6, "weight") != 0) {
        return false;
    }

    if (!params->quantize_output_tensor && tensor_name_match_output_weight(tensor->name)) {
        return false;
    }

    // do not quantize norm tensors
    if (name_v.find("_norm.weight")          != std::string_view::npos) return false;

    // do not quantize expert gating tensors
    if (name_v.find("ffn_gate_inp.weight")   != std::string_view::npos) return false;

    // these are very small (e.g. 4x4)
    if (name_v.find("altup")                 != std::string_view::npos) return false;
    if (name_v.find("laurel")                != std::string_view::npos) return false;

    // these are not too big so keep them as it is
    if (name_v.find("per_layer_model_proj")  != std::string_view::npos) return false;

    // do not quantize Mamba/Kimi's small conv1d weights
    if (name_v.find("ssm_conv1d")            != std::string_view::npos) return false;
    if (name_v.find("shortconv.conv.weight") != std::string_view::npos) return false;

    // do not quantize relative position bias (T5)
    if (name_v.find("attn_rel_b.weight")     != std::string_view::npos) return false;

    // do not quantize specific multimodal tensors
    if (name_v.find(".position_embd.")       != std::string_view::npos) return false;

    // do not quantize RWKV's small yet 2D weights
    if (const auto pos = name_v.find("time_mix_"); pos != std::string_view::npos) {
        const std::string_view rest = name_v.substr(pos + 9); // skip "time_mix_"
        static constexpr std::string_view blocked_suffixes[] = {
            "first.weight",
            "w0.weight",         "w1.weight",       "w2.weight",
            "v0.weight",         "v1.weight",       "v2.weight",
            "a0.weight",         "a1.weight",       "a2.weight",
            "g1.weight",         "g2.weight",
            "decay_w1.weight",   "decay_w2.weight",
            "lerp_fused.weight",
        };
        for (const auto & suffix : blocked_suffixes) {
            if (rest.size() >= suffix.size() &&
                rest.compare(0, suffix.size(), suffix) == 0) {
                return false;
            }
        }
    }

    const auto name = std::string(name_v); // LLM_TN doesn't play nice with string_view

    // do not quantize positional embeddings and token types (BERT)
    if (name == LLM_TN(arch)(LLM_TENSOR_POS_EMBD,    "weight")) return false;
    if (name == LLM_TN(arch)(LLM_TENSOR_TOKEN_TYPES, "weight")) return false;

    return true;
}

// incompatible tensor shapes are handled here - fallback to a compatible type
static ggml_type tensor_type_fallback(quantize_state_impl * qs, const ggml_tensor * t, const ggml_type target_type) {
    ggml_type return_type = target_type;

    const int64_t ncols = t->ne[0];
    const int64_t qk_k = ggml_blck_size(target_type);

    if (ncols % qk_k != 0) { // this tensor's shape is incompatible with this quant
        LLAMA_LOG_WARN("warning: %-36s: ncols %6" PRId64 " not divisible by %3" PRId64 " (required for type %7s) ",
                        t->name, ncols, qk_k, ggml_type_name(target_type));
        ++qs->n_fallback;

        switch (target_type) {
            // types on the left: block size 256
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:   // types on the right: block size 32
            case GGML_TYPE_IQ4_XS:  return_type = GGML_TYPE_IQ4_NL; break;
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_TQ1_0:
            case GGML_TYPE_TQ2_0:   return_type = GGML_TYPE_Q4_0;   break;
            case GGML_TYPE_Q4_K:    return_type = GGML_TYPE_Q5_0;   break;
            case GGML_TYPE_Q5_K:    return_type = GGML_TYPE_Q5_1;   break;
            case GGML_TYPE_Q6_K:    return_type = GGML_TYPE_Q8_0;   break;
            default:
                throw std::runtime_error(format("no tensor type fallback is defined for type %s",
                                                ggml_type_name(target_type)));
        }
        if (ncols % ggml_blck_size(return_type) != 0) {
            //
            // the fallback return type is still not compatible for this tensor!
            //
            // most likely, this tensor's first dimension is not divisible by 32.
            // this is very rare. we can either abort the quantization, or
            // fallback to F16 / F32.
            //
            LLAMA_LOG_WARN("(WARNING: must use F16 due to unusual shape) ");
            return_type = GGML_TYPE_F16;
        }
        LLAMA_LOG_WARN("-> falling back to %7s\n", ggml_type_name(return_type));
    }
    return return_type;
}

// categorize this tensor
static tensor_category tensor_get_category(const std::string & tensor_name) {
    if (tensor_name_match_output_weight(tensor_name.c_str())) {
        return tensor_category::OUTPUT;
    }
    if (tensor_name_match_token_embd(tensor_name.c_str())) {
        return tensor_category::TOKEN_EMBD;
    }
    if (tensor_name.find("attn_qkv.weight") != std::string::npos) {
        return tensor_category::ATTENTION_QKV;
    }
    if (tensor_name.find("attn_kv_b.weight") != std::string::npos) {
        return tensor_category::ATTENTION_KV_B;
    }
    if (tensor_name.find("attn_v.weight") != std::string::npos) {
        return tensor_category::ATTENTION_V;
    }
    if (tensor_name.find("attn_k.weight") != std::string::npos) {
        return tensor_category::ATTENTION_K;
    }
    if (tensor_name.find("attn_q.weight") != std::string::npos) {
        return tensor_category::ATTENTION_Q;
    }
    if (tensor_name.find("attn_output.weight") != std::string::npos) {
        return tensor_category::ATTENTION_OUTPUT;
    }
    if (tensor_name.find("ffn_up") != std::string::npos) {
        return tensor_category::FFN_UP;
    }
    if (tensor_name.find("ffn_gate") != std::string::npos) {
        return tensor_category::FFN_GATE;
    }
    if (tensor_name.find("ffn_down") != std::string::npos) {
        return tensor_category::FFN_DOWN;
    }
    return tensor_category::OTHER;
}

// check if category is for attention-v-like tensors (more sensitive to quantization)
static bool category_is_attn_v(tensor_category cat) {
    return cat == tensor_category::ATTENTION_V     ||
           cat == tensor_category::ATTENTION_QKV   ||
           cat == tensor_category::ATTENTION_KV_B;
}

// internal standard logic for selecting the target tensor type for a given tensor, ftype, and model arch
static ggml_type llama_tensor_get_type_impl(
    quantize_state_impl * qs,
              ggml_type   new_type,
      const ggml_tensor * tensor,
            llama_ftype   ftype,
        tensor_category   category)
{
    const std::string name = ggml_get_name(tensor);
    const llm_arch arch = qs->model.arch;

    auto use_more_bits = [](int i_layer, int n_layers) -> bool {
        return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
    };
    const int n_expert = std::max(1, (int)qs->model.hparams.n_expert);
    auto layer_info = [n_expert] (int i_layer, int n_layer, const char * name) {
        if (n_expert > 1) {
            // Believe it or not, "experts" in the FFN of Mixtral-8x7B are not consecutive, but occasionally randomly
            // sprinkled in the model. Hence, simply dividing i_ffn_down by n_expert does not work
            // for getting the current layer as I initially thought, and we need to resort to parsing the
            // tensor name.
            if (sscanf(name, "blk.%d.", &i_layer) != 1) {
                throw std::runtime_error(format("Failed to determine layer for tensor %s", name));
            }
            if (i_layer < 0 || i_layer >= n_layer) {
                throw std::runtime_error(format("Bad layer %d for tensor %s. Must be in [0, %d)", i_layer, name, n_layer));
            }
        }
        return std::make_pair(i_layer, n_layer);
    };

    // for arches that share the same tensor between the token embeddings and the output, we quantize the token embeddings
    // with the quantization of the output tensor
    if (category == tensor_category::OUTPUT || (!qs->has_output && category == tensor_category::TOKEN_EMBD)) {
        if (qs->params->output_tensor_type < GGML_TYPE_COUNT) {
            new_type = qs->params->output_tensor_type;
        } else {
            const int64_t ncols = tensor->ne[0];
            const int64_t qk_k = ggml_blck_size(new_type);

            if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (arch == LLM_ARCH_FALCON || ncols % qk_k != 0) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ2_S  || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M   ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q5_K;
            }
            else if (new_type != GGML_TYPE_Q8_0) {
                new_type = GGML_TYPE_Q6_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
        // MoE   tensors -> MXFP4
        // other tensors -> Q8_0
        if (tensor->ne[2] > 1) {
            new_type = GGML_TYPE_MXFP4;
        } else {
            new_type = GGML_TYPE_Q8_0;
        }
    } else if (category == tensor_category::TOKEN_EMBD) {
        if (qs->params->token_embedding_type < GGML_TYPE_COUNT) {
            new_type = qs->params->token_embedding_type;
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q2_K;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_TQ1_0 || ftype == LLAMA_FTYPE_MOSTLY_TQ2_0) {
                new_type = GGML_TYPE_Q4_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ1_S ||
               ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M    || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
        if (category_is_attn_v(category)) {
            if (qs->model.hparams.n_gqa() >= 4 || qs->model.hparams.n_expert >= 4) new_type = GGML_TYPE_Q4_K;
            else new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            ++qs->i_attention_wv;
        }
        else if (qs->model.hparams.n_expert == 8 && category == tensor_category::ATTENTION_K) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (category == tensor_category::FFN_DOWN) {
            if (qs->i_ffn_down < qs->n_ffn_down/8) {
                new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            }
            ++qs->i_ffn_down;
        }
        else if (category == tensor_category::ATTENTION_OUTPUT) {
            if (qs->model.hparams.n_expert == 8) {
                new_type = GGML_TYPE_Q5_K;
            } else {
                if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_S || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) new_type = GGML_TYPE_IQ2_XXS;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) new_type = GGML_TYPE_IQ3_S;
            }
        }
    } else if (category_is_attn_v(category)) {
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = qs->model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && qs->model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = qs->model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : !qs->has_imatrix ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S) && qs->model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = qs->i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && qs->model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                use_more_bits(qs->i_attention_wv, qs->n_attention_wv)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs->i_attention_wv < 4) new_type = GGML_TYPE_Q5_K;
        if (qs->model.type == LLM_TYPE_70B) {
            // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
            // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
            // nearly negligible increase in model size by quantizing this tensor with more bits:
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) new_type = GGML_TYPE_Q5_K;
        }
        if (qs->model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        ++qs->i_attention_wv;
    } else if (category == tensor_category::ATTENTION_K) {
        if (qs->model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (category == tensor_category::ATTENTION_Q) {
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (category == tensor_category::FFN_DOWN) {
        auto info = layer_info(qs->i_ffn_down, qs->n_ffn_down, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) {
            if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS && !qs->has_imatrix) {
            new_type = i_layer < n_layer/8 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = i_layer < n_layer/16 ? GGML_TYPE_Q5_K
                     : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer/8 ||
                    (qs->model.hparams.n_expert == 8 && use_more_bits(i_layer, n_layer)))) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = arch == LLM_ARCH_FALCON ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            if (arch == LLM_ARCH_FALCON) {
                new_type = i_layer < n_layer/16 ? GGML_TYPE_Q6_K :
                           use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
            } else {
                if (use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
            }
        }
        else if (i_layer < n_layer/8 && (ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && !qs->has_imatrix) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_0 || ftype == LLAMA_FTYPE_MOSTLY_Q5_0)
                && qs->has_imatrix && i_layer < n_layer/8) {
            // Guard against craziness in the first few ffn_down layers that can happen even with imatrix for Q4_0/Q5_0.
            // We only do it when an imatrix is provided because a) we want to make sure that one can always get the
            // same quantization as before imatrix stuff, and b) Q4_1/Q5_1 do go crazy on ffn_down without an imatrix.
            new_type = ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q5_1;
        }
        ++qs->i_ffn_down;
    } else if (category == tensor_category::ATTENTION_OUTPUT) {
        if (arch != LLM_ARCH_FALCON) {
            if (qs->model.hparams.n_expert == 8) {
                if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL  ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S  ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) {
                    new_type = GGML_TYPE_Q5_K;
                }
            } else {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   ) new_type = GGML_TYPE_Q3_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) new_type = GGML_TYPE_IQ3_S;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M ) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L ) new_type = GGML_TYPE_Q5_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  ) new_type = GGML_TYPE_Q4_K;
            }
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q4_K;
        }
    }
    else if (category == tensor_category::ATTENTION_QKV) {
        if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L || ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) new_type = GGML_TYPE_Q5_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) new_type = GGML_TYPE_Q6_K;
    }
    else if (category == tensor_category::FFN_GATE) {
        auto info = layer_info(qs->i_ffn_gate, qs->n_ffn_gate, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs->i_ffn_gate;
    }
    else if (category == tensor_category::FFN_UP) {
        auto info = layer_info(qs->i_ffn_up, qs->n_ffn_up, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs->i_ffn_up;
    }

    return new_type;
}

// determine the ggml_type that this tensor should be quantized to
static ggml_type llama_tensor_get_type(
                  quantize_state_impl * qs,
    const llama_model_quantize_params * params,
                    const ggml_tensor * tensor,
                      const ggml_type   default_type,
                const tensor_metadata & tm)
{
    if (!tm.allows_quantization) {
        return tensor->type;
    }

    if (params->token_embedding_type < GGML_TYPE_COUNT && tm.category == tensor_category::TOKEN_EMBD) {
        return params->token_embedding_type;
    }
    if (params->output_tensor_type < GGML_TYPE_COUNT && tm.category == tensor_category::OUTPUT) {
        return params->output_tensor_type;
    }

    ggml_type new_type = default_type;

    // get more optimal quantization type based on the tensor shape, layer, etc.
    if (!params->pure && ggml_is_quantized(default_type)) {

        // if the user provided tensor types - use those
        bool manual = false;
        if (!qs->tensor_type_patterns.empty()) {
            const std::string tensor_name = tensor->name;
            for (const auto & [pattern, qtype] : qs->tensor_type_patterns) {
                if (std::regex_search(tensor_name, pattern)) {
                    if (qtype != new_type) {
                        new_type = qtype;
                        manual = true;
                        break;
                    }
                }
            }
        }

        // if not manual - use the internal logic for choosing the quantization type
        if (!manual) {
            new_type = llama_tensor_get_type_impl(qs, new_type, tensor, params->ftype, tm.category);
        }

        // fallback to a compatible type if necessary
        new_type = tensor_type_fallback(qs, tensor, new_type);
    }
    return new_type;
}

//
// dequantization
//

static constexpr int64_t MIN_CHUNK_SIZE = 32 * 512;

// low-level: dequantize a contiguous slab of elements (src -> f32).
// parallelizes across chunks within the slab using qs->pool.
static void llama_tensor_dequantize_impl(
              ggml_type   src_type,
             const void * src_data,
                  float * f32_output,
                int64_t   nelements,
    quantize_state_impl * qs,
                   int    nthread)
{
    const ggml_type_traits * qtype = ggml_get_type_traits(src_type);
    if (ggml_is_quantized(src_type)) {
        if (qtype->to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(src_type)));
        }
    } else if (src_type != GGML_TYPE_F16 &&
               src_type != GGML_TYPE_BF16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(src_type)));
    }

    if (nthread < 2) {
        if (src_type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *)src_data, f32_output, nelements);
        } else if (src_type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *)src_data, f32_output, nelements);
        } else if (ggml_is_quantized(src_type)) {
            qtype->to_float(src_data, f32_output, nelements);
        } else {
            GGML_ABORT("fatal error"); // unreachable
        }
        return;
    }

    const size_t block_size       = (size_t)ggml_blck_size(src_type);
    const size_t block_size_bytes = ggml_type_size(src_type);

    GGML_ASSERT(nelements % block_size == 0);
    const size_t nblocks          = nelements / block_size;
    const size_t blocks_per_chunk = nblocks / nthread;
    const size_t spare_blocks     = nblocks - (blocks_per_chunk * nthread);

    const uint8_t * src_bytes = (const uint8_t *)src_data;

    qs->pool.distribute(nthread, [=](int64_t tnum) {
        const size_t thr_blocks = blocks_per_chunk + ((size_t)tnum == (size_t)(nthread - 1) ? spare_blocks : 0);
        const size_t thr_elems  = thr_blocks * block_size;

        const size_t block_offset = (size_t)tnum * blocks_per_chunk;
        const size_t in_offset    = block_offset * block_size_bytes;
        const size_t out_offset   = block_offset * block_size;

        if (src_type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *)(src_bytes + in_offset), f32_output + out_offset, thr_elems);
        } else if (src_type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *)(src_bytes + in_offset), f32_output + out_offset, thr_elems);
        } else {
            qtype->to_float(src_bytes + in_offset, f32_output + out_offset, thr_elems);
        }
    });
}

// dequantize an entire tensor to f32
// supports parallelism for 3D tensors (such as MoE experts)
static void llama_tensor_dequantize(
                const ggml_tensor * tensor,
    std::vector<no_init<float>>   & output,
              quantize_state_impl * qs,
                              int   nthread)
{
    const int64_t ne = ggml_nelements(tensor);
    const int64_t ne0 = tensor->ne[0]; // ncols
    const int64_t ne1 = tensor->ne[1]; // nrows
    const int64_t ne2 = tensor->ne[2]; // n_expert (or any 3rd tensor dimension)
    const int64_t ne0_x_1 = ne0 * ne1;

    if ((size_t)ne > output.size()) {
        output.resize(ne);
    }

    const ggml_type src_type      = tensor->type;
    const size_t    src_blk_size  = ggml_blck_size(src_type);
    const size_t    src_blk_bytes = ggml_type_size(src_type);

    const uint8_t * src_data  = (const uint8_t *)tensor->data;
    float         * f32_data  = (float *)output.data();

    // same chunk-sizing logic as llama_tensor_quantize
    const int64_t chunk_size = (ne0 >= MIN_CHUNK_SIZE
        ? ne0
        : ne1 * ((MIN_CHUNK_SIZE + ne1 - 1) / ne0));
    const int64_t nchunk = (ne0_x_1 + chunk_size - 1) / chunk_size;

    // enough chunks to feed all threads?
    const bool expert_parallel = ne2 >= nthread && nthread > 1 && nchunk < nthread;

    if (expert_parallel) {
        qs->pool.distribute(ne2, [&](int64_t expert) {
            const size_t src_expert_blocks = ne0_x_1 / src_blk_size;
            const void * esrc = src_data + expert * src_expert_blocks * src_blk_bytes;
            float      * ef32 = f32_data + expert * ne0_x_1;

            // dequant entire expert single-threaded
            llama_tensor_dequantize_impl(src_type, esrc, ef32, ne0_x_1, qs, 1);
        });
    } else {
        for (int64_t i03 = 0; i03 < ne2; ++i03) {
            const size_t src_expert_blocks = ne0_x_1 / src_blk_size;
            const void * esrc = src_data + i03 * src_expert_blocks * src_blk_bytes;
            float      * ef32 = f32_data + i03 * ne0_x_1;
            llama_tensor_dequantize_impl(src_type, esrc, ef32, ne0_x_1, qs, nthread);
        }
    }
}

//
// quantization
//

static size_t llama_tensor_quantize_impl(
         enum ggml_type   new_type,
            const float * f32_data,
                   void * new_data,
          const int64_t   chunk_size,
                int64_t   nrows,
                int64_t   ne0,
            const float * imatrix,
    quantize_state_impl * qs,
              const int   nthread)
{
    if (nthread < 2) {
        // single-thread
        size_t new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, ne0, imatrix);
        if (!ggml_validate_row_data(new_type, new_data, new_size)) {
            throw std::runtime_error("quantized data validation failed");
        }
        return new_size;
    }

    const int64_t nrows_per_chunk = chunk_size / ne0;
    const int64_t nchunks = (nrows + nrows_per_chunk - 1) / nrows_per_chunk;

    std::atomic<size_t> new_size{0};
    std::atomic<bool> valid{true};

    qs->pool.distribute(nchunks, [&](int64_t chunk_idx) {
        if (!valid.load(std::memory_order_relaxed)) {
            return;
        }

        const int64_t first_row = chunk_idx * nrows_per_chunk;
        const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);

        size_t this_size = ggml_quantize_chunk(new_type, f32_data, new_data,
            first_row * ne0, this_nrow, ne0, imatrix);

        // validate the quantized data
        const size_t row_size  = ggml_row_size(new_type, ne0);
        void * this_data = (char *) new_data + first_row * row_size;
        if (!ggml_validate_row_data(new_type, this_data, this_size)) {
            valid.store(false, std::memory_order_relaxed);
            return;
        }

        new_size.fetch_add(this_size, std::memory_order_relaxed);
    });

    if (!valid.load()) {
        throw std::runtime_error("quantized data validation failed");
    }
    return new_size.load();
}

// quantize a tensor from f32
// supports parallelism for 3D tensors (such as MoE experts)
static size_t llama_tensor_quantize(
                const ggml_tensor * tensor,
                        ggml_type   new_type,
                      const float * f32_data,
                      const float * imatrix,
    std::vector<no_init<uint8_t>> & work,
              quantize_state_impl * qs,
                              int   nthread)
{
    const int64_t ne = ggml_nelements(tensor);
    const int64_t ne0 = tensor->ne[0]; // n_cols
    const int64_t ne1 = tensor->ne[1]; // n_rows
    const int64_t ne2 = tensor->ne[2]; // n_experts (or any 3rd tensor dimension)

    const int64_t chunk_size = (ne0 >= MIN_CHUNK_SIZE
        ? ne0
        : ne0 * ((MIN_CHUNK_SIZE + ne0 - 1)/ne0));

    const int64_t ne0_x_1  = tensor->ne[0] * tensor->ne[1];
    const int64_t nchunk = (ne0_x_1 + chunk_size - 1) / chunk_size;

    const ggml_type type = tensor->type;

    if (work.size() < (size_t)ne * 4) {
        work.resize(ne * 4);
    }

    void * new_data = work.data();

    // should we use expert-parallel quantization?
    const bool expert_parallel = (
        // static types are fast enough to quantize that it's probably not worth it
        type != GGML_TYPE_Q8_0 &&
        type != GGML_TYPE_Q5_1 &&
        type != GGML_TYPE_Q5_0 &&
        type != GGML_TYPE_Q4_1 &&
        type != GGML_TYPE_Q4_0
        // enough chunks to feed all threads?
        ) && (ne2 >= nthread && nthread > 1 && nchunk < nthread);

    if (expert_parallel) {
        std::atomic<size_t> new_size{0};
        std::atomic<bool> valid{true};

        qs->pool.distribute(ne2, [&](int64_t expert) {
            if (!valid.load(std::memory_order_relaxed)) {
                return;
            }
            const float * f32_data_expert = f32_data + expert * ne0_x_1;
            void        * new_data_expert = (char *)new_data + ggml_row_size(new_type, ne0)
                                                             * expert * ne1;
            const float * imatrix_expert  = imatrix ? imatrix + expert * ne0 : nullptr;
            size_t expert_size = ggml_quantize_chunk(
                new_type, f32_data_expert, new_data_expert,
                0, ne1, ne0, imatrix_expert);

            if (!ggml_validate_row_data(new_type, new_data_expert, expert_size)) {
                valid.store(false, std::memory_order_relaxed);
                return;
            }

            new_size.fetch_add(expert_size, std::memory_order_relaxed);
        });

        if (!valid.load()) {
            throw std::runtime_error("quantized data validation failed");
        }

        return new_size.load();
    } else {
        const int64_t nthread_use = nthread > 1
            ? std::max((int64_t)1, std::min((int64_t)nthread, nchunk))
            : 1;

        size_t new_size = 0;
        for (int64_t i03 = 0; i03 < ne2; ++i03) {
            const float * imatrix_03  = imatrix ? imatrix + i03 * ne0 : nullptr;
            const float * f32_data_03 = f32_data + i03 * ne0_x_1;
            void        * new_data_03 = (char *)new_data + ggml_row_size(new_type, ne0)
                                                         * i03 * ne1;
            new_size += llama_tensor_quantize_impl(
                new_type, f32_data_03, new_data_03, chunk_size,
                ne1, ne0, imatrix_03, qs, nthread_use);
        }

        return new_size;
    }
}

// fused dequant + quantize
//
// dequantize and quantize a non-f32 tensor in a single distribute() call,
// so that f32 intermediate data stays cache-hot between the two operations.
// handles both expert-parallel and chunk-parallel strategies.
static size_t tensor_dequant_and_quantize_fused_impl(
                const ggml_tensor * tensor,
                        ggml_type   new_type,
                      const float * imatrix,
      std::vector<no_init<float>> & dequant_buf,
    std::vector<no_init<uint8_t>> & work_buf,
              quantize_state_impl * qs,
                              int   nthread)
{
    const ggml_type src_type  = tensor->type;
    const auto    * src_traits = ggml_get_type_traits(src_type);

    // validate source type
    if (ggml_is_quantized(src_type)) {
        if (src_traits->to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available",
                                            ggml_type_name(src_type)));
        }
    } else if (src_type != GGML_TYPE_F16 && src_type != GGML_TYPE_BF16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(src_type)));
    }

    const int64_t ne0 = tensor->ne[0]; // n_cols
    const int64_t ne1 = tensor->ne[1]; // n_rows
    const int64_t ne2 = tensor->ne[2]; // n_expert (or any 3rd tensor dimension)
    const int64_t ne0_x_1 = ne0 * ne1;

    const size_t src_blk_size  = ggml_blck_size(src_type);
    const size_t src_blk_bytes = ggml_type_size(src_type);
    const size_t dst_row_size  = ggml_row_size(new_type, ne0);

    const int64_t chunk_size = (ne0 >= MIN_CHUNK_SIZE
        ? ne0
        : ne0 * ((MIN_CHUNK_SIZE + ne0 - 1) / ne0));
    const int64_t nrows_per_chunk = chunk_size / ne0;
    const int64_t nchunk = (ne0_x_1 + chunk_size - 1) / chunk_size;

    uint8_t * src_data = (uint8_t *)tensor->data;
    float   * f32_data = (float *)dequant_buf.data();
    void    * dst_data = work_buf.data();

    // helper: dequantize `n` elements from src to f32
    auto dequant_slab = [src_type, src_traits](const void * src, float * dst, int64_t n) {
        if (src_type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *)src, dst, n);
        } else if (src_type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *)src, dst, n);
        } else {
            src_traits->to_float(src, dst, n);
        }
    };

    const bool expert_parallel = (
        // static types are fast enough to quantize that it's probably not worth it
        new_type != GGML_TYPE_Q8_0 &&
        new_type != GGML_TYPE_Q5_1 &&
        new_type != GGML_TYPE_Q5_0 &&
        new_type != GGML_TYPE_Q4_1 &&
        new_type != GGML_TYPE_Q4_0
        // enough chunks to feed all threads?
        ) && (ne2 >= nthread && nthread > 1 && nchunk < nthread);

    std::atomic<size_t> total_size{0};
    std::atomic<bool>   valid{true};

    if (expert_parallel) {
        // each thread gets one expert: dequant then quant, all cache-hot
        qs->pool.distribute(ne2, [&](int64_t expert) {
            if (!valid.load(std::memory_order_relaxed)) return;

            const size_t src_expert_blocks = ne0_x_1 / src_blk_size;

            uint8_t     * esrc  = src_data + expert * src_expert_blocks * src_blk_bytes;
            float       * ef32  = f32_data + expert * ne0_x_1;
            void        * edst  = (char *)dst_data + expert * ne1 * dst_row_size;
            const float * eimat = imatrix ? imatrix + expert * ne0 : nullptr;

            dequant_slab(esrc, ef32, ne0_x_1);

            size_t esize = ggml_quantize_chunk(new_type, ef32, edst, 0, ne1, ne0, eimat);

            if (!ggml_validate_row_data(new_type, edst, esize)) {
                valid.store(false, std::memory_order_relaxed);
                return;
            }

            total_size.fetch_add(esize, std::memory_order_relaxed);
        });
    } else {
        // chunk-parallel within each expert
        for (int64_t i03 = 0; i03 < ne2; ++i03) {
            const size_t src_expert_blocks = ne0_x_1 / src_blk_size;

            uint8_t     * esrc  = src_data + i03 * src_expert_blocks * src_blk_bytes;
            float       * ef32  = f32_data + i03 * ne0_x_1;
            char        * edst  = (char *)dst_data + i03 * ne1 * dst_row_size;
            const float * eimat = imatrix ? imatrix + i03 * ne0 : nullptr;

            const int64_t nchunks_expert = (ne1 + nrows_per_chunk - 1) / nrows_per_chunk;

            qs->pool.distribute(nchunks_expert, [&](int64_t chunk_idx) {
                if (!valid.load(std::memory_order_relaxed)) return;

                const int64_t first_row  = chunk_idx * nrows_per_chunk;
                const int64_t this_nrows = std::min(ne1 - first_row, nrows_per_chunk);
                const int64_t this_elems = this_nrows * ne0;

                // dequant this chunk
                const size_t src_offset = (first_row * ne0 / src_blk_size) * src_blk_bytes;
                const size_t f32_offset = first_row * ne0;

                dequant_slab(esrc + src_offset, ef32 + f32_offset, this_elems);

                // quant this chunk (f32 data is cache-hot)
                size_t csize = ggml_quantize_chunk(new_type, ef32, edst, first_row * ne0, this_nrows, ne0, eimat);

                void * cdata = edst + first_row * dst_row_size;
                if (!ggml_validate_row_data(new_type, cdata, csize)) {
                    valid.store(false, std::memory_order_relaxed);
                    return;
                }

                total_size.fetch_add(csize, std::memory_order_relaxed);
            });
        }
    }

    if (!valid.load()) {
        throw std::runtime_error("quantized data validation failed");
    }
    return total_size.load();
}

// process a single tensor: dequantize (if needed) and quantize, returning
// the output size and setting *out_data to point at the result.
// this is the single entry point called from the main quantization loop.
static size_t llama_tensor_process(
                      ggml_tensor *  tensor,
                        ggml_type    new_type,
                      const float *  imatrix,
                             bool    allow_requantize,
      std::vector<no_init<float>> &  dequant_buf,
    std::vector<no_init<uint8_t>> &  work_buf,
              quantize_state_impl *  qs,
                              int    nthread,
                             void ** out_data)
{
    // no type change
    if (new_type == tensor->type) {
        *out_data = tensor->data;
        return ggml_nbytes(tensor);
    }

    // source is already f32
    if (tensor->type == GGML_TYPE_F32) {
        size_t sz = llama_tensor_quantize(
            tensor, new_type, (float *)tensor->data, imatrix, work_buf, qs, nthread);
        *out_data = work_buf.data();
        return sz;
    }

    // source is quantized: check if requantization is allowed
    if (ggml_is_quantized(tensor->type) && !allow_requantize) {
        throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
    }

    // non-f32 source: fused dequant -> quant in a single pass
    size_t sz = tensor_dequant_and_quantize_fused_impl(
        tensor, new_type, imatrix, dequant_buf, work_buf, qs, nthread);
    *out_data = work_buf.data();
    return sz;
}

// does this tensor require importance matrix data?
static bool tensor_requires_imatrix(const char * tensor_name, const ggml_type dst_type, const llama_ftype ftype) {
    if (tensor_name_match_token_embd(tensor_name) || tensor_name_match_output_weight(tensor_name)) {
        return false;
    }
    switch (dst_type) {
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_S:
            return true;
        case GGML_TYPE_Q2_K:
            // as a general rule, the k-type quantizations don't require imatrix data.
            // the only exception is Q2_K tensors that are part of a Q2_K_S file.
            return ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S;
        default:
            return false;
    }
}

// given a file type, get the default tensor type
static ggml_type llama_ftype_get_default_type(llama_ftype ftype) {
    ggml_type return_type;
    switch (ftype) {
        // floating-point
        case LLAMA_FTYPE_ALL_F32:     return_type = GGML_TYPE_F32;  break;
        case LLAMA_FTYPE_MOSTLY_F16:  return_type = GGML_TYPE_F16;  break;
        case LLAMA_FTYPE_MOSTLY_BF16: return_type = GGML_TYPE_BF16; break;

        // static quants
        case LLAMA_FTYPE_MOSTLY_Q4_0: return_type = GGML_TYPE_Q4_0; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1: return_type = GGML_TYPE_Q4_1; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0: return_type = GGML_TYPE_Q5_0; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1: return_type = GGML_TYPE_Q5_1; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0: return_type = GGML_TYPE_Q8_0; break;

        // k-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:   return_type = GGML_TYPE_Q2_K; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L: return_type = GGML_TYPE_Q3_K; break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M: return_type = GGML_TYPE_Q4_K; break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M: return_type = GGML_TYPE_Q5_K; break;
        case LLAMA_FTYPE_MOSTLY_Q6_K:   return_type = GGML_TYPE_Q6_K; break;

        // i-quants
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   return_type = GGML_TYPE_IQ1_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:   return_type = GGML_TYPE_IQ1_M;   break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: return_type = GGML_TYPE_IQ2_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   return_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   return_type = GGML_TYPE_IQ2_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: return_type = GGML_TYPE_IQ3_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:
        case LLAMA_FTYPE_MOSTLY_IQ3_S:
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   return_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  return_type = GGML_TYPE_IQ4_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  return_type = GGML_TYPE_IQ4_NL;  break;

        // MXFP4
        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: return_type = GGML_TYPE_MXFP4; break;

        // ternary
        case LLAMA_FTYPE_MOSTLY_TQ1_0: return_type = GGML_TYPE_TQ1_0; break;
        case LLAMA_FTYPE_MOSTLY_TQ2_0: return_type = GGML_TYPE_TQ2_0; break;

        // otherwise, invalid
        default: throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }
    return return_type;
}

// main quantization driver
static void llama_model_quantize_impl(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    ggml_type default_type;
    llama_ftype ftype = params->ftype;

    int nthread = params->nthread <= 0 ? std::thread::hardware_concurrency() : params->nthread;

    default_type = llama_ftype_get_default_type(ftype);

    // mmap consistently increases speed on Linux, and also increases speed on Windows with
    // hot cache. It may cause a slowdown on macOS, possibly related to free memory.
#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;
#else
    constexpr bool use_mmap = false;
#endif

    llama_model_kv_override * kv_overrides = nullptr;
    if (params->kv_overrides) {
        auto * v = (std::vector<llama_model_kv_override>*)params->kv_overrides;
        kv_overrides = v->data();
    }

    std::vector<std::string> splits = {};
    llama_model_loader ml(fname_inp, splits, use_mmap, /*use_direct_io*/ false, /*check_tensors*/ true, /*no_alloc*/ false, kv_overrides, nullptr);
    ml.init_mappings(false); // no prefetching

    llama_model model(llama_model_default_params());

    model.load_arch   (ml);
    model.load_hparams(ml);
    model.load_stats  (ml);

    // quantization state
    auto qs = std::make_unique<quantize_state_impl>(model, params);

    // these need to be set to n_layer by default
    qs->n_ffn_down = qs->n_ffn_gate = qs->n_ffn_up = (int)model.hparams.n_layer;

    if (params->only_copy) {
        ftype = ml.ftype;
    }

    const std::unordered_map<std::string, std::vector<float>> * imatrix_data = nullptr;
    if (params->imatrix) {
        imatrix_data = static_cast<const std::unordered_map<std::string, std::vector<float>>*>(params->imatrix);
        if (imatrix_data) {
            LLAMA_LOG_INFO("\n%s: have importance matrix data with %d entries\n",
                           __func__, (int)imatrix_data->size());
            for (const auto & kv : *imatrix_data) {
                for (float f : kv.second) {
                    if (!std::isfinite(f)) {
                        throw std::runtime_error(format("imatrix contains non-finite value %f\n", f));
                    }
                }
            }
            qs->has_imatrix = true;
        }
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    gguf_context_ptr ctx_out { gguf_init_empty() };

    std::vector<int> prune_list = {};
    if (params->prune_layers) {
        prune_list = *static_cast<const std::vector<int> *>(params->prune_layers);
    }

    // copy the KV pairs from the input file
    gguf_set_kv     (ctx_out.get(), ml.meta.get());
    gguf_set_val_u32(ctx_out.get(), "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out.get(), "general.file_type", ftype);

    // Remove split metadata
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str());

    if (params->kv_overrides) {
        const std::vector<llama_model_kv_override> & overrides = *(const std::vector<llama_model_kv_override> *)params->kv_overrides;
        for (const auto & o : overrides) {
            if (o.key[0] == 0) break;
            if (o.tag == LLAMA_KV_OVERRIDE_TYPE_FLOAT) {
                gguf_set_val_f32(ctx_out.get(), o.key, o.val_f64);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_INT) {
                gguf_set_val_u32(ctx_out.get(), o.key, (uint32_t)std::abs(o.val_i64));
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_BOOL) {
                gguf_set_val_bool(ctx_out.get(), o.key, o.val_bool);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_STR) {
                gguf_set_val_str(ctx_out.get(), o.key, o.val_str);
            } else {
                LLAMA_LOG_WARN("%s: unknown KV override type for key '%s' (not applied)\n",
                               __func__, o.key);
            }
        }
    }

    std::map<int, std::string> mapped;
    int blk_id = 0;

    // make a list of weights
    std::vector<const llama_model_loader::llama_tensor_weight *> weights;
    weights.reserve(ml.weights_map.size());
    for (const auto & it : ml.weights_map) {
        const std::string remapped_name(remap_layer(it.first, prune_list, mapped, blk_id));
        if (remapped_name.empty()) {
            LLAMA_LOG_DEBUG("%s: pruning tensor %s\n", __func__, it.first.c_str());
            continue;
        }

        if (remapped_name != it.first) {
            ggml_set_name(it.second.tensor, remapped_name.c_str());
            LLAMA_LOG_DEBUG("%s: tensor %s remapped to %s\n",
                            __func__, it.first.c_str(), ggml_get_name(it.second.tensor));
        }
        weights.push_back(&it.second);
    }
    if (!prune_list.empty()) {
        gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_BLOCK_COUNT).c_str(), blk_id);
    }

    // keep_split requires that the weights are sorted by split index
    if (params->keep_split) {
        std::sort(weights.begin(), weights.end(), [](
            const llama_model_loader::llama_tensor_weight * a,
            const llama_model_loader::llama_tensor_weight * b
        ) {
            if (a->idx == b->idx) {
                return a->offs < b->offs;
            }
            return a->idx < b->idx;
        });
    }

    int idx = 0;
    uint16_t n_split = 1;
    if (params->keep_split) {
        for (const auto * it : weights) {
            n_split = std::max(uint16_t(it->idx + 1), n_split);
        }
    }
    std::vector<gguf_context_ptr> ctx_outs(n_split);
    ctx_outs[0] = std::move(ctx_out);

    // compute tensor metadata once and cache it
    std::vector<tensor_metadata> metadata(weights.size());

    // flag for `--dry-run`, to let the user know if imatrix will be required
    // for a real quantization, as a courtesy
    bool will_require_imatrix = false;

    // create the persistent thread pool
    qs->pool.start(nthread);

    // scratch buffers
    std::vector<no_init<float>>   scratch_dequant_buf;
    std::vector<no_init<uint8_t>> scratch_read_buf;
    std::vector<no_init<uint8_t>> scratch_buf;

    size_t max_nelements_dequant = 0;
    size_t max_tensor_bytes = 0;
    size_t max_nelements = 0;

    //
    // preliminary iteration over all weights
    //

    for (size_t i = 0; i < weights.size(); ++i) {
        const auto * it = weights[i];
        const ggml_tensor * tensor = it->tensor;
        const char * name = tensor->name;

        metadata[i].category = tensor_get_category(name);

        if (category_is_attn_v(metadata[i].category)) {
            ++qs->n_attention_wv;
        }

        if (tensor_name_match_output_weight(name)) {
            qs->has_output = true;
        }

        uint16_t i_split = params->keep_split ? it->idx : 0;
        if (!ctx_outs[i_split]) {
            ctx_outs[i_split].reset(gguf_init_empty());
        }
        gguf_add_tensor(ctx_outs[i_split].get(), tensor);

        metadata[i].allows_quantization = tensor_allows_quantization(params, model.arch, tensor);
        metadata[i].target_type = llama_tensor_get_type(qs.get(), params, tensor, default_type, metadata[i]);
        metadata[i].requires_imatrix = tensor_requires_imatrix(tensor->name, metadata[i].target_type, ftype);

        if (params->imatrix) {
            metadata[i].remapped_imatrix_name = remap_imatrix(tensor->name, mapped);
        } else if (metadata[i].allows_quantization && metadata[i].requires_imatrix) {
            if (params->dry_run) {
                will_require_imatrix = true;
            } else {
                LLAMA_LOG_ERROR("\n============================================================================\n"
                                  " ERROR: this quantization requires an importance matrix!\n"
                                  "        - offending tensor: %s\n"
                                  "        - target type: %s\n"
                                  "============================================================================\n\n",
                                  name, ggml_type_name(metadata[i].target_type));
                throw std::runtime_error("this quantization requires an imatrix!");
            }
        }
        max_tensor_bytes = std::max(max_tensor_bytes, ggml_nbytes(tensor));
        max_nelements    = std::max(max_nelements,    (size_t)ggml_nelements(tensor));
        if (tensor->type != GGML_TYPE_F32) {
            max_nelements_dequant = std::max(max_nelements_dequant, (size_t)ggml_nelements(tensor));
        }
    }

    // resize scratch buffers

    scratch_buf.resize(max_nelements * 4);
    if (max_nelements_dequant > 0) {
        scratch_dequant_buf.resize(max_nelements_dequant);
    }

    const size_t dequant_sz = scratch_dequant_buf.size() * sizeof(decltype(scratch_dequant_buf)::value_type);
    const size_t scratch_sz = scratch_buf.size()         * sizeof(decltype(scratch_buf)::value_type);

    LLAMA_LOG_INFO("%s: dequant buffer: %8.2f MiB\n", __func__, dequant_sz / (1024.0*1024.0));
    LLAMA_LOG_INFO("%s: scratch buffer: %8.2f MiB\n", __func__, scratch_sz / (1024.0*1024.0));
    if (!ml.use_mmap) {
        scratch_read_buf.resize(max_tensor_bytes);
        const size_t read_sz = scratch_read_buf.size() * sizeof(decltype(scratch_read_buf)::value_type);
        LLAMA_LOG_INFO("%s:    read buffer: %8.2f MiB\n", __func__, read_sz / (1024.0*1024.0));
    }

    // set split info if needed

    if (n_split > 1) {
        for (size_t i = 0; i < ctx_outs.size(); ++i) {
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str(), i);
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str(), n_split);
            gguf_set_val_i32(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str(), (int32_t)weights.size());
        }
    }

    int cur_split = -1;
    std::ofstream fout;
    auto close_ofstream = [&]() {
        // Write metadata and close file handler
        if (fout.is_open()) {
            fout.seekp(0);
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_outs[cur_split].get()));
            gguf_get_meta_data(ctx_outs[cur_split].get(), data.data());
            fout.write((const char *) data.data(), data.size());
            fout.close();
        }
    };
    auto new_ofstream = [&](int index) {
        cur_split = index;
        GGML_ASSERT(ctx_outs[cur_split] && "Find uninitialized gguf_context");
        std::string fname = fname_out;
        if (params->keep_split) {
            std::vector<char> split_path(llama_path_max(), 0);
            llama_split_path(split_path.data(), split_path.size(), fname_out.c_str(), cur_split, n_split);
            fname = std::string(split_path.data());
        }

        fout = std::ofstream(fname, std::ios::binary);
        fout.exceptions(std::ofstream::failbit);
        const size_t meta_size = gguf_get_meta_size(ctx_outs[cur_split].get());
        // placeholder for the meta data
        ::zeros(fout, meta_size);
    };

    // no output file for --dry-run
    if (!params->dry_run) {
        new_ofstream(0);
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    //
    // iterate over all weights (main loop)
    //

    for (size_t i = 0; i < weights.size(); ++i) {
        const auto * it = weights[i];
        const auto & weight = *it;
        ggml_tensor * tensor = weight.tensor;

        const auto & tm = metadata[i];

        if (!params->dry_run && (weight.idx != cur_split && params->keep_split)) {
            close_ofstream();
            new_ofstream(weight.idx);
        }

        const size_t tensor_size = ggml_nbytes(tensor);

        if (!params->dry_run) {
            // load tensor data
            if (!ml.use_mmap) {
                tensor->data = scratch_read_buf.data();
            }
            ml.load_data_for(tensor);
        }

        LLAMA_LOG_INFO("[%4d/%4d] %-36s - [%s], type: %7s, ",
                       ++idx, ml.n_tensors,
                       ggml_get_name(tensor),
                       llama_format_tensor_shape(tensor).c_str(),
                       ggml_type_name(tensor->type));

        const ggml_type new_type = tm.target_type;
        const bool do_quantize = (new_type != tensor->type);

        void * new_data;
        size_t new_size;

        //
        // perform quantization (or dry run)
        //

        if (params->dry_run) {
            // the --dry-run option calculates the final quantization size without quantizing
            if (do_quantize) {
                new_size = ggml_nrows(tensor) * ggml_row_size(new_type, tensor->ne[0]);
                LLAMA_LOG_INFO("size: %8.2f MiB -> type: %7s, size: %8.2f MiB\n",
                               tensor_size/1024.0/1024.0,
                               ggml_type_name(new_type),
                               new_size/1024.0/1024.0);
                if (!will_require_imatrix && tm.requires_imatrix) {
                    will_require_imatrix = true;
                }
            } else {
                new_size = tensor_size;
                LLAMA_LOG_INFO("size: %8.2f MiB\n", new_size/1024.0/1024.0);
            }
            total_size_org += tensor_size;
            total_size_new += new_size;
            continue;
        }

        const float * imatrix = nullptr;

        if (imatrix_data && do_quantize) {
            auto it_imatrix = imatrix_data->find(tm.remapped_imatrix_name);
            if (it_imatrix == imatrix_data->end()) {
                LLAMA_LOG_INFO("\n%s: did not find imatrix data for %s; ", __func__, tensor->name);
            } else {
                if (it_imatrix->second.size() == (size_t)tensor->ne[0]*tensor->ne[2]) {
                    imatrix = it_imatrix->second.data();
                } else {
                    LLAMA_LOG_INFO("\n%s: imatrix size %d is different from tensor size %d for %s\n",
                                   __func__, int(it_imatrix->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name);

                    if (tm.category != tensor_category::TOKEN_EMBD) {
                        throw std::runtime_error(format("imatrix size %d is different from tensor size %d for %s",
                                int(it_imatrix->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name));
                    }
                }
            }
        }

        if (do_quantize) {
            LLAMA_LOG_INFO("quantizing to %7s ... ", ggml_type_name(new_type));
            fflush(stdout);
        }

        new_size = llama_tensor_process(
            tensor, new_type, imatrix, params->allow_requantize,
            scratch_dequant_buf, scratch_buf, qs.get(), nthread, &new_data);

        if (do_quantize) {
            LLAMA_LOG_INFO("%8.2f MiB -> %8.2f MiB\n", tensor_size/1024.0/1024.0, new_size/1024.0/1024.0);
        } else {
            LLAMA_LOG_INFO("size: %8.2f MiB\n", tensor_size/1024.0/1024.0);
        }

        total_size_org += tensor_size;
        total_size_new += new_size;

        // update the gguf meta data as we go
        const char * name = tensor->name;
        gguf_set_tensor_type(ctx_outs[cur_split].get(), name, new_type);
        GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), name)) == new_size);
        gguf_set_tensor_data(ctx_outs[cur_split].get(), name, new_data);

        // write tensor data + padding
        fout.write((const char *) new_data, new_size);
        zeros(fout, GGML_PAD(new_size, align) - new_size);
    }

    if (!params->dry_run) {
        close_ofstream();
    }

    LLAMA_LOG_INFO("%s: model size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_org/1024.0/1024.0, total_size_org*8.0/ml.n_elements);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_new/1024.0/1024.0, total_size_new*8.0/ml.n_elements);

    if (!params->imatrix && params->dry_run && will_require_imatrix) {
        LLAMA_LOG_WARN("%s: WARNING: dry run completed successfully, but actually completing this quantization will require an imatrix!\n",
                       __func__
        );
    }

    if (qs->n_fallback > 0) {
        LLAMA_LOG_WARN("%s: WARNING: %d of %d tensor(s) required fallback quantization\n",
                       __func__, qs->n_fallback, ml.n_tensors);
    }
}

//
// interface implementation
//

llama_model_quantize_params llama_model_quantize_default_params() {
    llama_model_quantize_params result = {
        /*.nthread                     =*/ 0,
        /*.ftype                       =*/ LLAMA_FTYPE_MOSTLY_Q8_0,
        /*.output_tensor_type          =*/ GGML_TYPE_COUNT,
        /*.token_embedding_type        =*/ GGML_TYPE_COUNT,
        /*.allow_requantize            =*/ false,
        /*.quantize_output_tensor      =*/ true,
        /*.only_copy                   =*/ false,
        /*.pure                        =*/ false,
        /*.keep_split                  =*/ false,
        /*.dry_run                     =*/ false,
        /*.imatrix                     =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.tensor_type                 =*/ nullptr,
        /*.prune_layers                =*/ nullptr
    };

    return result;
}

uint32_t llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_model_quantize_params * params) {
    try {
        llama_model_quantize_impl(fname_inp, fname_out, params);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }

    return 0;
}
