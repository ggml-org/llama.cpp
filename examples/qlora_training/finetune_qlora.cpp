// QLoRA fine-tuning for quantized GGUF models.
//
// The base model weights stay frozen (quantized tensors are skipped by
// llama_set_param because they are not GGML_TYPE_F32).  Only the freshly
// allocated F32 LoRA A/B tensors are trained.  After training the adapter
// is saved as a GGUF file that is directly compatible with the existing
// llama_adapter_lora_init() loader and llama-export-lora merge tool.
//
// Usage example:
/*   llama-finetune-qlora \
         --model model-q4_k_m.gguf \
         --train-file train.jsonl \
         --lora-rank 16 --lora-alpha 16 \
         --lora-out adapter.gguf \
         --epochs 3 -c 4096 -b 4096 -ub 512
*/
// Default targets: attn_q, attn_output, ffn_gate, ffn_up, ffn_down
// Override with --lora-targets "comma,separated,substrings"
//
// NOTE: attn_k and attn_v are excluded from defaults.  The KV write path uses
// ggml_set_rows (scatter op) — backward cannot propagate gradients through it.
// LoRA K/V would receive zero gradient.
//
// NOTE: ssm_in and ssm_out (Mamba/NemotronH) are excluded from defaults.
// SSM_SCAN/SSM_CONV have no backward implementation — LoRA on these layers
// would receive zero gradient.  Adding them wastes memory with no benefit.
//
// NOTE: MoE expert tensors (*_exps) are excluded regardless of --lora-targets.
// The quantized expert weights are frozen (stop-gradient), but LoRA on dense
// FFN layers (ffn_gate, ffn_up, ffn_down) works via MUL_MAT_ID backward.
//
// Target substrings use llama.cpp internal GGUF names (NOT HuggingFace names):
//   attn_q      = q_proj       attn_k     = k_proj
//   attn_v      = v_proj       attn_output= o_proj
//   ffn_gate    = gate_proj    ffn_up     = up_proj    ffn_down = down_proj
//   ssm_in      = in_proj (Mamba/NemotronH)  — zero gradient, not in defaults
//   ssm_out     = out_proj (Mamba/NemotronH)  — zero gradient, not in defaults

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "gguf.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// Internal adapter struct — included directly to avoid the temp-GGUF roundtrip
// for wiring trainable LoRA tensors into the compute graph.
#include "../../src/llama-adapter.h"

#include <cerrno>
#include <csignal>
#include <iostream>

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Expand a leading ~/ to the HOME directory (the shell doesn't do this for us
// when a path is passed as a string argument to std::ofstream).
static std::string expand_tilde(const std::string & path) {
    if (path.size() >= 2 && path[0] == '~' && path[1] == '/') {
        const char * home = getenv("HOME");
        if (!home) home = getenv("USERPROFILE"); // Windows fallback
        if (home) return std::string(home) + path.substr(1);
    }
    return path;
}

static std::vector<std::string> split_csv(const std::string & s) {
    std::vector<std::string> out;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(tok);
    }
    return out;
}

// Tensors whose names contain these substrings use MUL_MAT_ID (sparse MoE expert dispatch)
// which has no backward implementation — exclude them from LoRA targets unconditionally.
static const std::vector<std::string> EXCLUDED_SUBSTRINGS = {
    "_exps",      // MoE expert weight stacks (ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_gate_up_exps)
};

static bool tensor_is_excluded(const char * name) {
    const std::string n(name);
    for (const auto & ex : EXCLUDED_SUBSTRINGS) {
        if (n.find(ex) != std::string::npos) return true;
    }
    return false;
}

// Extract the transformer block index from a tensor name of the form "blk.NN.<rest>".
// Returns -1 if the name does not follow this pattern.
static int tensor_layer_index(const char * name) {
    // All per-layer tensors in llama.cpp GGUF are named "blk.<N>.<suffix>"
    const char * p = strstr(name, "blk.");
    if (!p) return -1;
    p += 4; // skip "blk."
    char * end = nullptr;
    long idx = strtol(p, &end, 10);
    if (end == p || (*end != '.' && *end != '\0')) return -1;
    return (int) idx;
}

static bool tensor_matches_targets(const char * name, const std::vector<std::string> & targets,
                                   int freeze_layers = 0) {
    if (tensor_is_excluded(name)) return false;
    if (freeze_layers > 0) {
        const int layer = tensor_layer_index(name);
        if (layer >= 0 && layer < freeze_layers) return false;
    }
    for (const auto & t : targets) {
        if (std::string(name).find(t) != std::string::npos) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// JSONL dataset loading
// ---------------------------------------------------------------------------

struct training_sample {
    std::vector<llama_token> tokens;   // full token sequence
    std::vector<bool>        is_label; // true for tokens that contribute to loss
    float                    reward;   // reward/score weight (1.0 = neutral, 0.0 = ignore)
};

// Apply a very simple ChatML fallback template when the model has no template.
static std::string apply_chatml(const std::vector<common_chat_msg> & msgs) {
    std::string out;
    for (const auto & m : msgs) {
        out += "<|im_start|>" + m.role + "\n";
        // content_parts is a vector; build a plain text string
        std::string text;
        if (!m.content_parts.empty()) {
            for (const auto & p : m.content_parts) {
                text += p.text;
            }
        }
        out += text + "<|im_end|>\n";
    }
    return out;
}

static std::vector<training_sample> load_jsonl(
        const std::string & path,
        llama_context      * ctx,
        common_chat_templates * tmpls) {

    std::ifstream f(path);
    if (!f.is_open()) {
        LOG_ERR("%s: cannot open %s\n", __func__, path.c_str());
        return {};
    }

    std::vector<training_sample> samples;
    std::string line;
    int lineno = 0;

    while (std::getline(f, line)) {
        ++lineno;
        if (line.empty()) continue;

        nlohmann::json j;
        try { j = nlohmann::json::parse(line); }
        catch (...) {
            LOG_WRN("%s: skipping invalid JSON on line %d\n", __func__, lineno);
            continue;
        }

        float reward = 1.0f;
        if      (j.contains("reward")) reward = j["reward"].get<float>();
        else if (j.contains("score"))  reward = j["score"].get<float>();

        std::string prompt_text;
        std::string response_text;

        if (j.contains("messages")) {
            // chat format — apply template
            std::vector<common_chat_msg> msgs;
            for (const auto & m : j["messages"]) {
                common_chat_msg msg;
                msg.role = m.value("role", "user");
                common_chat_msg_content_part part;
                part.type = "text";
                part.text = m.value("content", "");
                msg.content_parts.push_back(part);
                msgs.push_back(msg);
            }

            // Skip samples where the last assistant turn contains an error marker.
            // These are malformed/failed generations that should not be trained on.
            {
                std::string last_assistant_content;
                for (int mi = (int)msgs.size() - 1; mi >= 0; --mi) {
                    if (msgs[mi].role == "assistant") {
                        last_assistant_content = msgs[mi].content_parts.empty()
                            ? "" : msgs[mi].content_parts[0].text;
                        break;
                    }
                }
                // // this should be done on the python side...
                // if (last_assistant_content.find("Error:") != std::string::npos ||
                //     last_assistant_content.find("error:") != std::string::npos) {
                //     LOG_DBG("%s: skipping line %d — assistant response contains error marker\n", __func__, lineno);
                //     continue;
                // }
            }

            // Split into prompt (no loss) + last assistant response (loss).
            // Render all messages except the last assistant turn as the prompt
            // (with add_generation_prompt=true so the template adds the assistant
            // prefix), then use the raw last assistant content as response_text.
            // This ensures only the assistant's response tokens get loss, not the
            // user turns or system prompt.
            if (msgs.empty()) continue;
            std::string last_assistant_content;
            std::vector<common_chat_msg> prompt_msgs;
            // Find the last assistant message
            int last_asst_idx = -1;
            for (int mi = (int)msgs.size() - 1; mi >= 0; --mi) {
                if (msgs[mi].role == "assistant") { last_asst_idx = mi; break; }
            }
            if (last_asst_idx < 0) {
                // No assistant turn — skip; nothing to train on
                LOG_DBG("%s: skipping line %d — no assistant turn\n", __func__, lineno);
                continue;
            }
            last_assistant_content = msgs[last_asst_idx].content_parts.empty()
                ? "" : msgs[last_asst_idx].content_parts[0].text;
            for (int mi = 0; mi < last_asst_idx; ++mi) prompt_msgs.push_back(msgs[mi]);

            if (tmpls) {
                common_chat_templates_inputs inp;
                inp.messages = prompt_msgs;
                inp.add_generation_prompt = true;
                prompt_text   = common_chat_templates_apply(tmpls, inp).prompt;
                response_text = last_assistant_content;
            } else {
                // Fallback: render everything as ChatML, use full text as response
                std::vector<common_chat_msg> all_msgs = prompt_msgs;
                all_msgs.push_back(msgs[last_asst_idx]);
                prompt_text   = "";
                response_text = apply_chatml(all_msgs);
            }
        } else if (j.contains("prompt") && j.contains("response")) {
            response_text = j["response"].get<std::string>();
            // // this should be done on the python side...
            // if (response_text.find("Error:") != std::string::npos ||
            //     response_text.find("error:") != std::string::npos) {
            //     LOG_DBG("%s: skipping line %d — response contains error marker\n", __func__, lineno);
            //     continue;
            // }
            prompt_text = j["prompt"].get<std::string>();
        } else if (j.contains("text")) {
            response_text = j["text"].get<std::string>();
        } else {
            LOG_WRN("%s: unknown format on line %d, skipping\n", __func__, lineno);
            continue;
        }

        // Tokenize: prompt (no loss) + response (loss)
        auto tok_prompt   = common_tokenize(ctx, prompt_text,   /*add_special=*/true);
        auto tok_response = common_tokenize(ctx, response_text, /*add_special=*/false);

        if (tok_prompt.empty() && tok_response.empty()) continue;

        training_sample s;
        s.reward = reward;
        s.tokens.insert(s.tokens.end(), tok_prompt.begin(),   tok_prompt.end());
        s.tokens.insert(s.tokens.end(), tok_response.begin(), tok_response.end());
        s.is_label.resize(s.tokens.size(), false);
        // Only response tokens contribute to the loss
        for (size_t i = tok_prompt.size(); i < s.tokens.size(); ++i) {
            s.is_label[i] = true;
        }
        samples.push_back(std::move(s));
    }

    LOG_INF("%s: loaded %zu samples from %s\n", __func__, samples.size(), path.c_str());
    return samples;
}

// Pack variable-length samples into fixed-context-length windows and create
// an ggml_opt_dataset. Labels for prompt tokens are set to -1 (ignored by
// the loss in the epoch loop).
// window_rewards is filled with one reward weight per window (averaged over
// the sample tokens that fall in that window). If all samples have reward=1.0
// the vector is all-ones and has no effect.
static ggml_opt_dataset_t build_dataset(
        const std::vector<training_sample> & samples,
        int32_t                              n_ctx,
        std::vector<float>                 & window_rewards,
        bool                                 train_on_prompt = false,
        llama_token                          bos_token = -1) {

    // Flatten samples into token/label/reward streams
    std::vector<llama_token> flat_tokens;
    std::vector<int32_t>     flat_labels;  // -1 = no loss, token_id = loss target
    std::vector<float>       flat_rewards; // per-token reward from the source sample

    for (size_t si = 0; si < samples.size(); ++si) {
        const auto & s = samples[si];

        // Insert BOS separator between samples to prevent cross-sample predictions.
        // The first sample already has BOS from tokenization (add_special=true).
        if (si > 0 && bos_token >= 0 && !s.tokens.empty()) {
            flat_tokens .push_back(bos_token);
            flat_labels .push_back(-1);  // no loss on separator
            flat_rewards.push_back(s.reward);
        }

        for (size_t i = 0; i + 1 < s.tokens.size(); ++i) {
            flat_tokens .push_back(s.tokens[i]);
            if (train_on_prompt) {
                // All positions get correct next-token label (prompt + response)
                flat_labels.push_back((int32_t)s.tokens[i + 1]);
            } else {
                // Only response positions get loss; prompt positions get -1 (sentinel).
                // The sentinel is passed through to labels_sparse; opt_epoch_iter skips
                // writing to the label tensor for those positions, leaving them zeroed →
                // zero cross-entropy contribution.  No gradient flows from prompt tokens.
                flat_labels.push_back(s.is_label[i + 1] ? (int32_t)s.tokens[i + 1] : -1);
            }
            flat_rewards.push_back(s.reward);
        }
    }

    if ((int64_t)flat_tokens.size() < n_ctx) {
        LOG_ERR("%s: dataset too small (%zu tokens) for context %d\n",
                __func__, flat_tokens.size(), n_ctx);
        return nullptr;
    }

    const int64_t stride = n_ctx / 2;
    int64_t ndata  = ((int64_t)flat_tokens.size() - n_ctx) / stride;
    if (ndata < 1) ndata = 1;  // at least one window when flat_tokens >= n_ctx

    window_rewards.resize(ndata);

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
            GGML_TYPE_I32, GGML_TYPE_I32, n_ctx, n_ctx, ndata, 1);

    int32_t * data   = (int32_t *) ggml_opt_dataset_data  (dataset)->data;
    int32_t * labels = (int32_t *) ggml_opt_dataset_labels(dataset)->data;

    for (int64_t i = 0; i < ndata; ++i) {
        const int64_t off = i * stride;
        float reward_sum = 0.0f;
        for (int32_t j = 0; j < n_ctx; ++j) {
            data  [i * n_ctx + j] = flat_tokens[off + j];
            // Pass -1 sentinel through unchanged for masked (prompt) positions.
            // opt_epoch_iter skips these positions (no label tensor write → zero
            // cross-entropy contribution).  Do NOT substitute the current token
            // here — that trains the model to predict itself (off-by-one) and
            // causes repetition degeneration.
            labels[i * n_ctx + j] = flat_labels[off + j];
            reward_sum += flat_rewards[off + j];
        }
        window_rewards[i] = reward_sum / n_ctx;
    }

    // Normalize window rewards to [0, 1].
    // Step 1: clip to [-1, 1] — outliers like 1.3/1.4 would otherwise compress the
    //         useful signal range after min-max scaling (a reward=1.0 would map to
    //         only 0.83 instead of 1.0 if the max is 1.4).
    // Step 2: min-max scale clipped values → [0, 1].
    //         min → 0.0 (window ignored), max → 1.0 (full weight).
    // If all rewards are identical (pure SFT dataset) keep at 1.0.
    for (float & r : window_rewards) {
        r = std::max(-1.0f, std::min(1.0f, r));
    }
    float rmin = *std::min_element(window_rewards.begin(), window_rewards.end());
    float rmax = *std::max_element(window_rewards.begin(), window_rewards.end());
    const float rrange = rmax - rmin;
    if (rrange > 1e-6f) {
        for (float & r : window_rewards) {
            r = (r - rmin) / rrange;
        }
        LOG_INF("%s: reward range [%.4f, %.4f] (after clip to [-1,1]) → normalized to [0, 1]\n", __func__, rmin, rmax);
    } else {
        std::fill(window_rewards.begin(), window_rewards.end(), 1.0f);
    }

    return dataset;
}

// ---------------------------------------------------------------------------
// LoRA tensor allocation
// ---------------------------------------------------------------------------

struct lora_tensors {
    struct ggml_context      * ctx  = nullptr;
    struct ggml_backend_buffer * buf = nullptr;
    // map: base tensor name → {lora_a, lora_b}
    std::unordered_map<std::string, std::pair<ggml_tensor*, ggml_tensor*>> ab;
};

static lora_tensors alloc_lora_tensors(
        const std::string        & model_path,
        const std::vector<std::string> & targets,
        int32_t                   rank,
        std::mt19937            & rng,
        int32_t                   freeze_layers = 0) {

    lora_tensors lt;

    // Open the model GGUF to discover tensor names and shapes
    // without needing access to private llama_model internals.
    struct ggml_context * ctx_meta = nullptr;
    struct gguf_init_params gguf_params = { /*.no_alloc=*/true, /*.ctx=*/&ctx_meta };
    struct gguf_context * ctx_gguf = gguf_init_from_file(model_path.c_str(), gguf_params);
    if (!ctx_gguf) {
        LOG_ERR("%s: failed to open model GGUF for tensor discovery: %s\n",
                __func__, model_path.c_str());
        return lt;
    }

    // Collect matching 2-D tensors
    struct tensor_info { std::string name; int64_t ne0, ne1; };
    std::vector<tensor_info> matched;

    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta);
         t; t = ggml_get_next_tensor(ctx_meta, t)) {
        if (ggml_n_dims(t) < 2) continue;
        if (!tensor_matches_targets(t->name, targets, freeze_layers)) continue;
        matched.push_back({t->name, t->ne[0], t->ne[1]});
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);

    if (matched.empty()) {
        LOG_ERR("%s: no model tensors matched --lora-targets; check spelling\n", __func__);
        return lt;
    }

    if (freeze_layers > 0) {
        LOG_INF("%s: freezing layers blk.0 .. blk.%d (no LoRA allocated; backward already pruned by grads_needed)\n",
                __func__, freeze_layers - 1);
    }
    LOG_INF("%s: allocating LoRA A/B tensors for %zu weight matrices, rank=%d\n",
            __func__, matched.size(), rank);

    // Allocate ggml context for A+B tensors (2 tensors per matched weight)
    const size_t mem = (2 * matched.size() + 16) * ggml_tensor_overhead();
    struct ggml_init_params ip = { mem, nullptr, /*no_alloc=*/true };
    lt.ctx = ggml_init(ip);

    for (const auto & ti : matched) {
        const int64_t in_dim  = ti.ne0; // columns (input features)
        const int64_t out_dim = ti.ne1; // rows    (output features)

        // lora_a: [in_dim, rank]   applied first: a @ x
        // lora_b: [rank,   out_dim] applied second: b @ (a @ x)
        // Convention matches llama-adapter.cpp:48-60:
        //   a->ne[0] == in_dim,  a->ne[1] == rank
        //   b->ne[0] == rank,    b->ne[1] == out_dim
        ggml_tensor * la = ggml_new_tensor_2d(lt.ctx, GGML_TYPE_F32, in_dim, rank);
        ggml_tensor * lb = ggml_new_tensor_2d(lt.ctx, GGML_TYPE_F32, rank,   out_dim);

        ggml_set_name(la, (ti.name + ".lora_a").c_str());
        ggml_set_name(lb, (ti.name + ".lora_b").c_str());

        lt.ab[ti.name] = {la, lb};
    }

    // Allocate backend buffer for all LoRA tensors at once
    lt.buf = ggml_backend_alloc_ctx_tensors_from_buft(lt.ctx, ggml_backend_cpu_buffer_type());

    // Initialize: A ~ N(0, 1/sqrt(rank)), B = 0
    const float std_a = 1.0f / std::sqrt((float)rank);
    std::normal_distribution<float> dist(0.0f, std_a);

    for (auto & kv : lt.ab) {
        ggml_tensor * la = kv.second.first;
        ggml_tensor * lb = kv.second.second;

        // Fill A
        float * data_a = (float *) la->data;
        for (int64_t i = 0; i < ggml_nelements(la); ++i) data_a[i] = dist(rng);
        // Zero B
        memset(lb->data, 0, ggml_nbytes(lb));
    }

    return lt;
}

// ---------------------------------------------------------------------------
// Param filter: only train lora_a / lora_b tensors
// ---------------------------------------------------------------------------

static bool lora_param_filter(const struct ggml_tensor * t, void * /*ud*/) {
    const char * n = t->name;
    const size_t len = strlen(n);
    if (len > 7 && strcmp(n + len - 7, ".lora_a") == 0) return true;
    if (len > 7 && strcmp(n + len - 7, ".lora_b") == 0) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Save adapter GGUF
// ---------------------------------------------------------------------------

static std::string basename_from_path(const std::string & p) {
    const size_t pos = p.find_last_of("/\\");
    if (pos == std::string::npos) return p;
    return p.substr(pos + 1);
}

static void save_adapter(
        const lora_tensors & lt,
        const std::string  & out_path,
        const std::string  & arch,
        float                alpha,
        const std::string  & base_model_path) {

    // Build output GGUF context
    struct gguf_context * gctx = gguf_init_empty();

    // Metadata required by llama_adapter_lora_init
    gguf_set_val_str(gctx, "general.type",         "adapter");
    gguf_set_val_str(gctx, "general.architecture",  arch.c_str());
    gguf_set_val_str(gctx, "adapter.type",          "lora");
    gguf_set_val_f32(gctx, "adapter.lora.alpha",    alpha);
    gguf_set_val_str(gctx, "adapter.base_model",    basename_from_path(base_model_path).c_str());

    // Register tensors
    for (const auto & kv : lt.ab) {
        gguf_add_tensor(gctx, kv.second.first);   // lora_a
        gguf_add_tensor(gctx, kv.second.second);  // lora_b
    }

    // Write: meta placeholder → tensor data → rewrite meta
    const std::string real_path = expand_tilde(out_path);
    std::ofstream fout(real_path, std::ios::binary);
    if (!fout.is_open()) {
        LOG_ERR("%s: cannot open %s for writing\n", __func__, real_path.c_str());
        gguf_free(gctx);
        return;
    }

    // Write meta placeholder
    const size_t meta_size = gguf_get_meta_size(gctx);
    std::vector<char> zeros_buf(meta_size, 0);
    fout.write(zeros_buf.data(), meta_size);

    // Write tensor data — copy to CPU first in case tensors live on GPU
    for (const auto & kv : lt.ab) {
        for (ggml_tensor * t : {kv.second.first, kv.second.second}) {
            const size_t nb = ggml_nbytes(t);
            std::vector<char> cpu_buf(nb);
            ggml_backend_tensor_get(t, cpu_buf.data(), 0, nb);
            fout.write(cpu_buf.data(), nb);
            // GGUF tensors are 32-byte aligned
            const size_t pad = GGML_PAD(nb, 32) - nb;
            if (pad > 0) {
                std::vector<char> pad_buf(pad, 0);
                fout.write(pad_buf.data(), pad);
            }
        }
    }

    // Re-write metadata at offset 0
    std::vector<uint8_t> meta(meta_size);
    gguf_get_meta_data(gctx, meta.data());
    fout.seekp(0);
    fout.write((const char *) meta.data(), meta_size);

    fout.close();
    gguf_free(gctx);

    LOG_INF("%s: adapter saved to %s\n", __func__, real_path.c_str());
}

// ---------------------------------------------------------------------------
// Periodic checkpoint callback
// ---------------------------------------------------------------------------

struct save_ctx {
    const lora_tensors * lt;
    const std::string  * lora_out;
    const std::string  * arch;
    const std::string  * base_model_path;
    float                lora_alpha;
    int32_t              save_every;     // 0 = disabled
    int32_t              ubatch_per_ctx;
    int64_t              last_saved;     // last window index at which we saved
};

// TLS pointer set before each epoch so the static callback can access it.
static thread_local save_ctx * g_save_ctx = nullptr;

static void save_every_callback(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    ggml_opt_epoch_callback_progress_bar(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us);

    // Log loss at every window boundary so we can see if/when it diverges.
    if (train && g_save_ctx) {
        const int64_t window = ibatch / g_save_ctx->ubatch_per_ctx;
        const int64_t ubatch_in_window = ibatch % g_save_ctx->ubatch_per_ctx;
        if (ubatch_in_window == g_save_ctx->ubatch_per_ctx - 1) {
            double loss = 0.0, loss_unc = 0.0;
            ggml_opt_result_loss(result, &loss, &loss_unc);
            fprintf(stderr, "\n[window %4ld] loss=%.4f ± %.4f\n", (long)window, loss, loss_unc);
        }
    }

    if (!train || !g_save_ctx || g_save_ctx->save_every <= 0) return;
    const int64_t window = ibatch / g_save_ctx->ubatch_per_ctx;
    if (window > 0 && window != g_save_ctx->last_saved && window % g_save_ctx->save_every == 0) {
        g_save_ctx->last_saved = window;
        const std::string ckpt = *g_save_ctx->lora_out + ".ckpt" + std::to_string(window) + ".gguf";
        save_adapter(*g_save_ctx->lt, ckpt, *g_save_ctx->arch, g_save_ctx->lora_alpha, *g_save_ctx->base_model_path);
        fprintf(stderr, "\n");
        LOG_INF("save_every_callback: checkpoint saved -> %s (window %ld)\n", ckpt.c_str(), (long)window);
    }
}

// ---------------------------------------------------------------------------
// IPC helpers  (stdout protocol, stdin commands)
// ---------------------------------------------------------------------------

// Escape newlines and backslashes for single-line IPC transmission.
// Mirrors _escape() in gguf_trainer.py.
static std::string ipc_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if      (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else                out += c;
    }
    return out;
}

static void ipc_emit(const char * msg) {
    fputs(msg, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

// Read one line from stdin, trimming the trailing newline.
// Returns false on EOF or error.
static bool ipc_read_line(std::string & out) {
    out.clear();
    if (!std::getline(std::cin, out)) return false;
    // Strip trailing \r if present (Windows line endings)
    if (!out.empty() && out.back() == '\r') out.pop_back();
    return true;
}

// Parse "REWARD r1 r2 ... rN" into a float vector.
static std::vector<float> ipc_parse_rewards(const std::string & line) {
    std::vector<float> rewards;
    if (line.size() < 8 || line.substr(0, 7) != "REWARD ") return rewards;
    std::istringstream ss(line.substr(7));
    float r;
    while (ss >> r) rewards.push_back(r);
    return rewards;
}

// ---------------------------------------------------------------------------
// Greedy / temperature sampling for GRPO rollout generation
// ---------------------------------------------------------------------------

static std::string generate_response(
        llama_context      * ctx,
        llama_model        * model,
        const std::string  & prompt,
        int32_t              max_tokens,
        float                temperature,
        std::mt19937       & rng) {

    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto tokens = common_tokenize(ctx, prompt, /*add_special=*/true);
    if (tokens.empty()) return "";

    // Clear KV cache before each generation (don't carry over previous prompt state)
    llama_memory_clear(llama_get_memory(ctx), true);
    {
        llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode failed on prompt\n", __func__);
            return "";
        }
    }

    std::string output;
    const llama_token eos = llama_vocab_eos(vocab);
    const llama_token nl  = llama_vocab_nl(vocab);

    // For ChatML models <|im_end|> is the turn-end marker but may not be the
    // vocab EOS token.  Look it up by tokenizing the string and taking the
    // first token if it tokenizes to exactly one piece.
    llama_token im_end = -1;
    {
        std::vector<llama_token> im_end_tokens(8);
        static const char im_end_str[] = "<|im_end|>";
        int n = llama_tokenize(vocab, im_end_str, (int32_t)strlen(im_end_str), im_end_tokens.data(), (int32_t)im_end_tokens.size(), /*add_special=*/false, /*parse_special=*/true);
        if (n == 1) im_end = im_end_tokens[0];
    }
    const llama_token eot = llama_vocab_eot(vocab);  // may equal eos on some models

    for (int32_t i = 0; i < max_tokens; ++i) {
        // Sample next token — use ith=-1 to always get the LAST output position's
        // logits.  llama_get_logits(ctx) returns position 0 which is wrong when the
        // prompt batch has multiple output tokens (training context).
        float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            LOG_ERR("%s: llama_get_logits_ith(-1) returned NULL\n", __func__);
            break;
        }
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);

        llama_token next_token;
        if (temperature <= 0.0f) {
            // Greedy
            next_token = (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
        } else {
            // Temperature sampling via softmax + categorical draw
            std::vector<float> probs(n_vocab);
            float max_logit = *std::max_element(logits, logits + n_vocab);
            float sum = 0.0f;
            for (int32_t k = 0; k < n_vocab; ++k) {
                probs[k] = std::exp((logits[k] - max_logit) / temperature);
                sum += probs[k];
            }
            for (float & p : probs) p /= sum;
            std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
            next_token = dist(rng);
        }

        if (next_token == eos)     break;
        if (next_token == eot)     break;
        if (im_end >= 0 && next_token == im_end && !output.empty()) break;

        // Decode token to text
        char buf[256] = {};
        llama_token_to_piece(vocab, next_token, buf, sizeof(buf) - 1, 0, true);
        output += buf;

        // Feed token back for next step
        llama_batch batch = llama_batch_get_one(&next_token, 1);
        if (llama_decode(ctx, batch) != 0) break;
    }

    return output;
}

// ---------------------------------------------------------------------------
// GRPO IPC training loop
// ---------------------------------------------------------------------------

// Volatile flag set by SIGINT so the loop can exit cleanly.
static volatile sig_atomic_t g_grpo_stop = 0;
static void grpo_sigint_handler(int) { g_grpo_stop = 1; }

static int run_grpo_mode(
        common_params    & params,
        llama_model      * model,
        llama_context    * ctx,
        lora_tensors     & lt,
        const std::string & arch,
    float              lora_alpha,
    const std::string & base_model_path) {

    const int32_t n_ctx    = llama_n_ctx(ctx);
    const int32_t n_gen    = params.grpo_n_gen;
    const int32_t n_steps  = params.grpo_n_steps;
    const float   temp     = params.grpo_temperature;
    const int32_t max_tok  = params.grpo_max_tokens;

    std::mt19937 rng(params.sampling.seed != LLAMA_DEFAULT_SEED
                     ? params.sampling.seed : 42);

    // Initialize optimizer
    struct llama_opt_params lopt_params {
        /*.n_ctx_train              =*/0,
        /*.param_filter             =*/lora_param_filter,
        /*.param_filter_ud          =*/nullptr,
        /*.get_opt_pars             =*/common_opt_lr_pars,
        /*.get_opt_pars_ud          =*/&params.lr,
        /*.optimizer_type           =*/params.optimizer,
        /*.grad_checkpoint_interval =*/params.grad_checkpoint_interval,
    };
    llama_opt_init(ctx, model, lopt_params);

    const llama_token bos = llama_vocab_bos(llama_model_get_vocab(model));

    signal(SIGINT, grpo_sigint_handler);

    // Signal Python that we are ready
    ipc_emit("[QLORA:READY]");

    float last_loss = 0.0f;
    int   step      = 0;

    while (step < n_steps && !g_grpo_stop) {

        // ── Request prompt ────────────────────────────────────────────────
        {
            char buf[64];
            snprintf(buf, sizeof(buf), "[QLORA:PROMPT_REQ:%d]", step + 1);
            ipc_emit(buf);
        }

        std::string prompt_line;
        if (!ipc_read_line(prompt_line)) break;
        if (prompt_line == "STOP") {
            LOG_INF("grpo: received STOP from Python\n");
            break;
        }
        if (prompt_line.size() < 8 || prompt_line.substr(0, 7) != "PROMPT ") {
            char buf[128];
            snprintf(buf, sizeof(buf), "[QLORA:ERROR] expected PROMPT, got: %.80s", prompt_line.c_str());
            ipc_emit(buf);
            return 1;
        }
        // Unescape the prompt (\\n → \n etc.)
        std::string prompt;
        {
            const std::string esc = prompt_line.substr(7);
            prompt.reserve(esc.size());
            for (size_t i = 0; i < esc.size(); ++i) {
                if (esc[i] == '\\' && i + 1 < esc.size()) {
                    char next = esc[i+1];
                    if      (next == 'n')  { prompt += '\n'; ++i; }
                    else if (next == 'r')  { prompt += '\r'; ++i; }
                    else if (next == '\\') { prompt += '\\'; ++i; }
                    else                   { prompt += esc[i]; }
                } else {
                    prompt += esc[i];
                }
            }
        }

        // ── Generate N responses ──────────────────────────────────────────
        std::vector<std::string> generations(n_gen);
        for (int k = 0; k < n_gen; ++k) {
            generations[k] = generate_response(ctx, model, prompt, max_tok, temp, rng);

            char hdr[64];
            snprintf(hdr, sizeof(hdr), "[QLORA:GEN:%d/%d] ", k + 1, n_gen);
            std::string msg = std::string(hdr) + ipc_escape(generations[k]);
            ipc_emit(msg.c_str());
        }

        // ── Request rewards ───────────────────────────────────────────────
        {
            char buf[64];
            snprintf(buf, sizeof(buf), "[QLORA:REWARD_REQ:%d]", n_gen);
            ipc_emit(buf);
        }

        std::string reward_line;
        if (!ipc_read_line(reward_line)) break;
        if (reward_line == "STOP") {
            LOG_INF("grpo: received STOP from Python\n");
            break;
        }
        std::vector<float> rewards = ipc_parse_rewards(reward_line);
        if ((int32_t)rewards.size() != n_gen) {
            char buf[128];
            snprintf(buf, sizeof(buf), "[QLORA:ERROR] expected %d rewards, got %zu", n_gen, rewards.size());
            ipc_emit(buf);
            return 1;
        }

        // ── Build single-step mini-dataset: prompt+generations with rewards ─
        // Each generation is a separate sample; prompt = no-loss, generation = loss.
        std::vector<training_sample> step_samples;
        step_samples.reserve(n_gen);
        for (int k = 0; k < n_gen; ++k) {
            training_sample s;
            s.reward = rewards[k];

            auto tok_prompt = common_tokenize(ctx, prompt,         /*add_special=*/true);
            auto tok_gen    = common_tokenize(ctx, generations[k], /*add_special=*/false);

            s.tokens.insert(s.tokens.end(), tok_prompt.begin(), tok_prompt.end());
            s.tokens.insert(s.tokens.end(), tok_gen.begin(),    tok_gen.end());
            s.is_label.resize(s.tokens.size(), false);
            for (size_t i = tok_prompt.size(); i < s.tokens.size(); ++i) {
                s.is_label[i] = true;
            }
            step_samples.push_back(std::move(s));
        }

        // Ensure minimum token count for one context window.
        // build_dataset drops the last token per sample during flattening,
        // so we need total raw tokens > n_ctx to guarantee ndata >= 1.
        while (true) {
            size_t total = 0;
            for (const auto & s : step_samples) total += s.tokens.size();
            if ((int64_t)total > n_ctx + (int64_t)step_samples.size()) break;
            step_samples.push_back(step_samples.back());
        }

        std::vector<float> window_rewards;
        ggml_opt_dataset_t step_dataset = build_dataset(
                step_samples, n_ctx, window_rewards, /*train_on_prompt=*/false, bos);
        if (!step_dataset) {
            ipc_emit("[QLORA:ERROR] build_dataset failed for step");
            return 1;
        }

        // Apply reward weights for this step
        const bool has_rewards = std::any_of(window_rewards.begin(), window_rewards.end(),
                                             [](float r){ return std::abs(r - 1.0f) > 1e-4f; });
        if (has_rewards) {
            llama_opt_set_reward_weights(window_rewards.data(), (int64_t)window_rewards.size());
        }

        // ── One optimizer step (full dataset = one mini-epoch) ────────────
        const int64_t idata_all = ggml_opt_dataset_ndata(step_dataset);
        ggml_opt_result_t step_result = ggml_opt_result_init();

        llama_opt_epoch(ctx, step_dataset, step_result, nullptr, idata_all,
                        nullptr,   // no progress bar callback — clean stdout
                        nullptr,
                        false);    // no shuffle for single-step

        double loss = 0.0, loss_unc = 0.0;
        ggml_opt_result_loss(step_result, &loss, &loss_unc);
        last_loss = (float)loss;

        ggml_opt_result_free(step_result);
        ggml_opt_dataset_free(step_dataset);
        llama_opt_set_reward_weights(nullptr, 0);

        ++step;

        // ── Emit progress ─────────────────────────────────────────────────
        {
            char buf[128];
            snprintf(buf, sizeof(buf),
                     "[QLORA:PROGRESS] step=%d/%d loss=%.4f epoch=1/1",
                     step, n_steps, last_loss);
            ipc_emit(buf);
        }

        // ── Optional checkpoint ───────────────────────────────────────────
        if (params.save_every > 0 && step % params.save_every == 0) {
            std::string ckpt = params.lora_out + ".ckpt" + std::to_string(step) + ".gguf";
            save_adapter(lt, ckpt, arch, lora_alpha, base_model_path);
            char buf[512];
            snprintf(buf, sizeof(buf), "[QLORA:CHECKPOINT] %s", ckpt.c_str());
            ipc_emit(buf);
        }
    }

    // Save final adapter
    save_adapter(lt, params.lora_out, arch, lora_alpha, base_model_path);

    {
        char buf[64];
        snprintf(buf, sizeof(buf), "[QLORA:DONE] final_loss=%.4f", last_loss);
        ipc_emit(buf);
    }

    return 0;
}

// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE_QLORA)) {
        return 1;
    }

    if (!params.grpo_mode && params.train_file.empty()) {
        LOG_ERR("%s: --train-file is required (or use --grpo-mode for IPC training)\n", __func__);
        return 1;
    }

    // Force settings required for training
    params.use_mmap     = false;
    params.cache_type_k = GGML_TYPE_F32;
    params.cache_type_v = GGML_TYPE_F32;
    // Warmup runs inference with PARAM-flagged tensors which causes a segfault;
    // training never benefits from warmup, so disable it unconditionally.
    params.warmup       = false;
    // Flash attention has no backward implementation; force standard attention for training.
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    const float lora_alpha = (params.lora_alpha > 0.0f)
        ? params.lora_alpha : (float) params.lora_rank;
    const auto targets = split_csv(params.lora_targets);

    // --- Step 1: Discover tensor shapes from model GGUF (no model load yet) ---
    std::string arch;
    {
        struct ggml_context * ctx_meta = nullptr;
        struct gguf_init_params gp = { true, &ctx_meta };
        struct gguf_context * ctx_gguf = gguf_init_from_file(params.model.path.c_str(), gp);
        if (!ctx_gguf) { LOG_ERR("failed to open model GGUF\n"); return 1; }
        int kid = gguf_find_key(ctx_gguf, "general.architecture");
        if (kid >= 0) arch = gguf_get_val_str(ctx_gguf, kid);
        gguf_free(ctx_gguf);
        ggml_free(ctx_meta);
    }

    // --- Step 2: Allocate LoRA tensors and save initial adapter GGUF ---
    // If the user already supplied a --lora adapter we reuse it (resume training).
    // Otherwise we allocate fresh tensors (B=0, A=random), write them to a temp
    // .init.gguf so common_init_from_params can load them before context creation
    // (this makes sched_reserve size the graph to include LoRA nodes).
    const bool resume_from_lora = !params.lora_adapters.empty();

    std::mt19937 rng(42);
    lora_tensors lt; // will be populated after context load (Step 4)
    std::string init_adapter_path;

    if (!resume_from_lora) {
        lt = alloc_lora_tensors(params.model.path, targets, params.lora_rank, rng, params.lora_freeze_layers);
        if (lt.ab.empty()) return 1;

        init_adapter_path = params.lora_out + ".init.gguf";
        save_adapter(lt, init_adapter_path, arch, lora_alpha, params.model.path);

        // Register adapter so common_init_from_params loads it before context creation
        common_adapter_lora_info adapter_info;
        adapter_info.path  = init_adapter_path;
        adapter_info.scale = 1.0f;
        params.lora_adapters.push_back(adapter_info);
    } else {
        LOG_INF("%s: resuming training from existing LoRA adapter: %s\n",
                __func__, params.lora_adapters.back().path.c_str());
    }

    // --- Step 3: Load model + context (graph sized with LoRA nodes) ---
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model) { LOG_ERR("failed to load model\n"); return 1; }

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    // Arch fallback if not in GGUF metadata
    if (arch.empty()) {
        char buf[256] = {};
        llama_model_desc(model, buf, sizeof(buf));
        arch = std::string(buf);
        arch = arch.substr(0, arch.find_first_of(" /"));
    }

    // --- Step 4: Mark the loaded adapter tensors as trainable ---
    // common_init_from_params loaded the adapter; params.lora_adapters[back].ptr
    // points to the live llama_adapter_lora with its own tensor copies in device
    // memory. Mark those tensors trainable so the optimizer graph includes them.
    {
        llama_adapter_lora * loaded = params.lora_adapters.back().ptr;
        if (!loaded) {
            LOG_ERR("%s: adapter was not loaded by common_init_from_params\n", __func__);
            return 1;
        }
        for (auto & kv : loaded->ab_map) {
            ggml_set_param(kv.second.a);  // lora_a → trainable
            ggml_set_param(kv.second.b);  // lora_b → trainable
        }
        // Point lt.ab at the live device tensors so save_adapter writes
        // the trained weights (not the original init tensors).
        lt.ab.clear();
        for (auto & kv : loaded->ab_map) {
            lt.ab[kv.first] = {kv.second.a, kv.second.b};
        }
    }

    // Remove temp init file when we created it (resume path has no init file)
    if (!resume_from_lora && !init_adapter_path.empty()) {
        std::remove(expand_tilde(init_adapter_path).c_str());
    }

    // --- Step 5: Load dataset ---
    // In GRPO mode the dataset comes from Python via stdin/stdout — skip file loading.
    auto tmpls = common_chat_templates_init(model, "");
    if (params.grpo_mode) {
        int rc = run_grpo_mode(params, model, ctx, lt, arch, lora_alpha, params.model.path);
        if (lt.buf) ggml_backend_buffer_free(lt.buf);
        if (lt.ctx) ggml_free(lt.ctx);
        llama_backend_free();
        return rc;
    }
    auto samples = load_jsonl(params.train_file, ctx, tmpls.get());
    if (samples.empty()) {
        LOG_ERR("%s: no training samples loaded\n", __func__);
        return 1;
    }

    const int32_t n_ctx = llama_n_ctx(ctx);
    std::vector<float> window_rewards;
    const llama_token bos = llama_vocab_bos(llama_model_get_vocab(model));
    auto dataset = build_dataset(samples, n_ctx, window_rewards, params.train_on_prompt, bos);
    if (!dataset) return 1;

    // Check if any reward deviates from 1.0 — if so, enable reward-weighted SFT
    const bool has_rewards = std::any_of(window_rewards.begin(), window_rewards.end(),
                                         [](float r){ return std::abs(r - 1.0f) > 1e-4f; });
    if (has_rewards) {
        LOG_INF("%s: reward-weighted SFT enabled (found non-uniform rewards in dataset)\n", __func__);
        llama_opt_set_reward_weights(window_rewards.data(), (int64_t)window_rewards.size());
    }

    // Initialize optimizer — our custom param filter restricts training to lora_a/b
    struct llama_opt_params lopt_params {
        /*.n_ctx_train              =*/0,
        /*.param_filter             =*/lora_param_filter,
        /*.param_filter_ud          =*/nullptr,
        /*.get_opt_pars             =*/common_opt_lr_pars,
        /*.get_opt_pars_ud          =*/&params.lr,
        /*.optimizer_type           =*/params.optimizer,
        /*.grad_checkpoint_interval =*/params.grad_checkpoint_interval,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    const int32_t n_ubatch       = llama_n_ubatch(ctx);
    const int32_t ubatch_per_ctx = (n_ubatch > 0) ? (n_ctx / n_ubatch) : 1;

    save_ctx sctx { &lt, &params.lora_out, &arch, &params.model.path, lora_alpha, params.save_every, ubatch_per_ctx, 0 };
    g_save_ctx = &sctx;

    const int64_t total_windows = ggml_opt_dataset_ndata(dataset);
    LOG_INF("%s: starting QLoRA training — rank=%d alpha=%.1f epochs=%d loss=%s\n",
            __func__, params.lora_rank, lora_alpha, params.lr.epochs,
            params.train_on_prompt ? "prompt+response" : "response-only");
    LOG_INF("%s: dataset: %ld windows × %d ubatches = %ld steps per epoch  (n_ctx=%d n_ubatch=%d stride=%d)\n",
            __func__, (long)total_windows, ubatch_per_ctx, (long)(idata_split * ubatch_per_ctx),
            n_ctx, n_ubatch, n_ctx / 2);
    if (params.save_every > 0) {
        LOG_INF("%s: will save checkpoint every %d windows → %s.ckptN.gguf\n",
                __func__, params.save_every, params.lora_out.c_str());
    }

    ggml_opt_epoch_callback cb_train = (params.save_every > 0)
        ? save_every_callback
        : ggml_opt_epoch_callback_progress_bar;

    for (params.lr.epoch = 0; params.lr.epoch < params.lr.epochs; ++params.lr.epoch) {
        sctx.last_saved = 0;  // reset per-epoch window counter
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                        cb_train,
                        ggml_opt_epoch_callback_progress_bar,
                        params.shuffle_dataset);
        fprintf(stderr, "\n");

        // Per-epoch loss summary
        {
            double train_loss = 0.0, train_unc = 0.0;
            ggml_opt_result_loss(result_train, &train_loss, &train_unc);
            if (idata_split < ggml_opt_dataset_ndata(dataset)) {
                double val_loss = 0.0, val_unc = 0.0;
                ggml_opt_result_loss(result_eval, &val_loss, &val_unc);
                LOG_INF("epoch %d/%d: train_loss=%.4f ± %.4f  val_loss=%.4f ± %.4f\n",
                        params.lr.epoch + 1, params.lr.epochs, train_loss, train_unc, val_loss, val_unc);
            } else {
                LOG_INF("epoch %d/%d: train_loss=%.4f ± %.4f\n",
                        params.lr.epoch + 1, params.lr.epochs, train_loss, train_unc);
            }
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);
    llama_opt_set_reward_weights(nullptr, 0);

    // Save final trained adapter
    save_adapter(lt, params.lora_out, arch, lora_alpha, params.model.path);

    // Free scratch buffers only when we allocated them (not in resume path)
    if (lt.buf) ggml_backend_buffer_free(lt.buf);
    if (lt.ctx) ggml_free(lt.ctx);
    ggml_opt_dataset_free(dataset);
    llama_backend_free();

    return 0;
}
