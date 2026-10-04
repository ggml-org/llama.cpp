#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"
#include "../src/llama-ext.h"
#include "../src/llama-model-saver.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

static constexpr int n_embd = 64;
static constexpr int n_hc = 2;
static constexpr int n_hidden = n_embd * n_hc;
static constexpr int n_vocab = 32;
static constexpr int n_rank = 4;
static ggml_backend_dev_t test_device = nullptr;
static bool q8_kv = false;

static void require(bool ok, const char * message) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

// Small canonical files exercise the real GGUF loader, including missing required tensors.
static void write_fixture(const std::string & path, bool mtp_only, int omit_head = -1, float trunk_scale = 1.0f) {
    gguf_context_ptr meta(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_QWEN4EXP, meta.get());
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE, "qwen4exp");
    ms.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(128));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH, uint32_t(n_embd));
    ms.add_kv(LLM_KV_BLOCK_COUNT, uint32_t(3));
    ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, uint32_t(64));
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, uint32_t(64));
    ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, uint32_t(64));
    ms.add_kv(LLM_KV_EXPERT_COUNT, uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT, uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-6f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({8, 8, 8, 8}));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL, uint32_t(4));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE, uint32_t(64));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE, uint32_t(64));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK, uint32_t(1));
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT, uint32_t(n_hc));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(n_rank));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K, uint32_t(128));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL, "test");
    std::vector<std::string> vocab;
    for (int i = 0; i < n_vocab; ++i) {
        vocab.push_back("tok_" + std::to_string(i));
    }
    ms.add_kv(LLM_KV_TOKENIZER_LIST, vocab);
    ms.add_kv(LLM_KV_TOKENIZER_SCORES, std::vector<float>(n_vocab, 0.0f));

    ggml_context_ptr tensors(ggml_init({8 * 1024 * 1024, nullptr, false}));
    auto add = [&](const std::string & name, std::initializer_list<int64_t> dims, float scale = 1.0f) {
        ggml_tensor * t = ggml_new_tensor(tensors.get(), GGML_TYPE_F32, dims.size(), dims.begin());
        ggml_set_name(t, name.c_str());
        // Name-based initialization leaves every other tensor unchanged when a mixer changes.
        uint32_t seed = 17;
        for (unsigned char c : name) {
            seed = seed * 1664525u + c;
        }
        float * data = static_cast<float *>(t->data);
        const bool norm = name.find("norm") != std::string::npos;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            seed = seed * 1664525u + 1013904223u;
            const float noise = (float(seed >> 8) / 16777216.0f - 0.5f);
            data[i] = name.find("ssm_a") != std::string::npos && name.find("ssm_alpha") == std::string::npos
                    ? -0.1f : scale * (norm ? 1.0f + noise * 0.2f : noise * 0.2f);
        }
        gguf_add_tensor(meta.get(), t);
    };
    add("token_embd.weight", {n_embd, n_vocab});
    add("output.weight", {n_embd, n_vocab});
    if (!mtp_only) {
        add("output_hc_norm.weight", {n_hidden}, trunk_scale);
        add("output_hc_down.weight", {n_hidden, n_rank}, trunk_scale);
        add("output_hc_up.weight", {n_rank, n_hidden}, trunk_scale);
    }
    for (int il = mtp_only ? 2 : 0; il < 3; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        for (const char * part : {"attn", "ffn"}) {
            const std::string h = p + "hc_" + part;
            add(h + "_norm.weight", {n_hidden});
            add(h + "_down.weight", {n_hidden, n_rank});
            add(h + "_up.weight", {n_rank, n_hidden});
            add(h + "_inject.weight", {n_hidden, n_hc});
        }
        if (il == 0) {
            add(p + "attn_qkv.weight", {n_embd, 3 * n_embd});
            add(p + "attn_gate.weight", {n_embd, n_embd});
            add(p + "ssm_conv1d.weight", {4, 3 * n_embd});
            add(p + "ssm_dt.bias", {1});
            add(p + "ssm_a", {1}, -1.0f);
            add(p + "ssm_beta.weight", {n_embd, 1});
            add(p + "ssm_alpha.weight", {n_embd, 1});
            add(p + "ssm_norm.weight", {n_embd});
            add(p + "ssm_out.weight", {n_embd, n_embd});
        } else {
            add(p + "attn_q.weight", {n_embd, n_embd * 2});
            add(p + "attn_k.weight", {n_embd, n_embd});
            add(p + "attn_v.weight", {n_embd, n_embd});
            add(p + "attn_output.weight", {n_embd, n_embd});
            add(p + "attn_q_norm.weight", {n_embd});
            add(p + "attn_k_norm.weight", {n_embd});
            add(p + "indexer.q_proj.weight", {n_embd, n_embd});
            add(p + "indexer.k_proj.weight", {n_embd, n_embd});
            add(p + "indexer.q_norm.weight", {n_embd});
            add(p + "indexer.k_norm.weight", {n_embd});
        }
        add(p + "ffn_gate_inp.weight", {n_embd, 2});
        add(p + "ffn_down_exps.weight", {64, n_embd, 2});
        add(p + "ffn_gate_exps.weight", {n_embd, 64, 2});
        add(p + "ffn_up_exps.weight", {n_embd, 64, 2});
        add(p + "ffn_gate_inp_shexp.weight", {n_embd});
        add(p + "ffn_gate_shexp.weight", {n_embd, 64});
        add(p + "ffn_up_shexp.weight", {n_embd, 64});
        add(p + "ffn_down_shexp.weight", {64, n_embd});
        if (il == 2) {
            add(p + "nextn.enorm.weight", {n_embd});
            add(p + "nextn.hnorm.weight", {n_hidden});
            add(p + "nextn.eh_proj.weight", {2 * n_embd, n_embd});
            if (omit_head != 0 && omit_head != 3) { add(p + "nextn.hc_head_norm.weight", {n_hidden}); }
            if (omit_head != 1 && omit_head != 3) { add(p + "nextn.hc_head_down.weight", {n_hidden, n_rank}); }
            if (omit_head != 2 && omit_head != 3) { add(p + "nextn.hc_head_up.weight", {n_rank, n_hidden}); }
        }
    }
    require(gguf_write_to_file(meta.get(), path.c_str(), false), "write fixture");
}

static llama_model_ptr load_model(const std::string & path, bool load_mtp = true) {
    auto params = llama_model_default_params();
    ggml_backend_dev_t devices[] = {test_device, nullptr};
    params.devices = devices;
    params.n_gpu_layers = test_device ? 99 : 0;
    params.load_mtp = load_mtp;
    return llama_model_ptr(llama_model_load_from_file(path.c_str(), params));
}

static llama_context_ptr make_context(llama_model * model, bool flash) {
    auto params = llama_context_default_params();
    params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    params.n_ctx = 128;
    params.n_batch = 32;
    params.n_ubatch = 32;
    params.n_threads = params.n_threads_batch = 2;
    params.type_k = q8_kv ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    params.type_v = q8_kv ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    params.flash_attn_type = flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context_ptr ctx(llama_init_from_model(model, params));
    require(bool(ctx), "create MTP context");
    llama_set_embeddings_nextn(ctx.get(), true, true);
    return ctx;
}

struct decoded {
    std::vector<float> logits;
    std::vector<float> hidden;
};

static decoded decode(llama_context * ctx, const std::vector<llama_token> & tokens,
                      const std::vector<float> & hidden, int pos, bool chain = false, int catchup = 0, bool masked = true) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], pos + i, {0}, int(i) >= catchup);
    }
    // MTP accepts token IDs and the target's wide hidden state in the same batch.
    batch.embd = const_cast<float *>(hidden.data());
    llama_set_embeddings_nextn(ctx, true, masked);
    llama_set_mtp_chain(ctx, chain);
    const int rc = llama_decode(ctx, batch);
    llama_set_mtp_chain(ctx, false);
    batch.embd = nullptr;
    llama_batch_free(batch);
    require(rc == 0, "MTP decode");
    const int n_out = tokens.size() - catchup;
    const float * logits = llama_get_logits(ctx);
    const float * h = llama_get_embeddings_nextn(ctx);
    require(logits && h, "MTP outputs");
    return {{logits, logits + n_out * (chain ? 2 : n_vocab)}, {h, h + (masked ? n_out : int(tokens.size())) * n_hidden}};
}

static std::vector<float> initial_hidden(int rows = 1) {
    std::vector<float> result(rows * n_hidden);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = std::sin(float(i) * 0.17f);
    }
    return result;
}

static bool close(const std::vector<float> & a, const std::vector<float> & b, float tolerance = 2e-5f) {
    if (a.size() != b.size()) { return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::abs(a[i] - b[i]) > tolerance * (1 + std::abs(a[i]))) {
            fprintf(stderr, "mismatch [%zu]: %g vs %g\n", i, double(a[i]), double(b[i]));
            return false;
        }
    }
    return true;
}

static void test_chain(llama_model * model, bool flash, int depth, int catchup, bool masked, bool at_limit = false) {
    auto seq = make_context(model, flash);
    auto chain = make_context(model, flash);
    std::vector<llama_token> tokens(depth + catchup, 0);
    const int pos0 = at_limit ? int(llama_n_ctx(seq.get())) - int(tokens.size()) : 0;
    tokens[catchup] = 5;
    auto inputs = initial_hidden(depth + catchup);
    std::vector<float> all_hidden;
    if (catchup) {
        std::vector<llama_token> past(catchup, 3);
        std::copy(past.begin(), past.end(), tokens.begin());
        auto past_out = decode(seq.get(), past, initial_hidden(catchup), pos0);
        if (!masked) { all_hidden = past_out.hidden; }
    }
    std::vector<float> h(inputs.begin() + catchup * n_hidden, inputs.begin() + (catchup + 1) * n_hidden);
    llama_token token = 5;
    std::vector<float> pairs;
    for (int j = 0; j < depth; ++j) {
        auto out = decode(seq.get(), {token}, h, pos0 + catchup + j);
        token = std::max_element(out.logits.begin(), out.logits.end()) - out.logits.begin();
        std::vector<float> sorted = out.logits;
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());
        double denominator = 0;
        for (int k = 0; k < 10; ++k) { denominator += std::exp(double(sorted[k] - sorted[0])); }
        pairs.push_back(float(token));
        pairs.push_back(float(1.0 / denominator));
        h = out.hidden;
        all_hidden.insert(all_hidden.end(), h.begin(), h.end());
    }
    auto out = decode(chain.get(), tokens, inputs, pos0, true, catchup, masked);
    require(close(pairs, out.logits), "chain greedy tokens and top-10 probabilities match sequential");
    require(close(all_hidden, out.hidden), "chain exports full-width hidden states matching sequential");
    // Reject every candidate after the first, then re-enter both caches with a different token.
    require(llama_memory_seq_rm(llama_get_memory(seq.get()), 0, pos0 + catchup + 1, -1), "sequential rollback");
    require(llama_memory_seq_rm(llama_get_memory(chain.get()), 0, pos0 + catchup + 1, -1), "chain rollback");
    const int first = masked ? 0 : catchup * n_hidden;
    h.assign(all_hidden.begin() + first, all_hidden.begin() + first + n_hidden);
    const auto a = decode(seq.get(), {7}, h, pos0 + catchup + 1);
    const auto b = decode(chain.get(), {7}, h, pos0 + catchup + 1);
    require(close(a.logits, b.logits) && close(a.hidden, b.hidden), "continuation matches after rejection");
    fprintf(stderr, "PASS chain flash=%d depth=%d catchup=%d masked=%d pos=%d\n", flash, depth, catchup, masked, pos0);
}

static void test_masked_catchup(llama_model * model, bool flash) {
    auto all = make_context(model, flash);
    auto masked = make_context(model, flash);
    const auto reference = decode(all.get(), {3, 4, 5}, initial_hidden(3), 0);
    const auto output = decode(masked.get(), {3, 4, 5}, initial_hidden(3), 0, false, 2);
    require(close(std::vector<float>(reference.logits.end() - n_vocab, reference.logits.end()), output.logits),
            "masked catchup logits match reference row");
    require(close(std::vector<float>(reference.hidden.end() - n_hidden, reference.hidden.end()), output.hidden),
            "masked catchup hidden state matches reference row");
}

#include "test-qwen4exp-mtp-driver.h"

struct fit_log_capture {
    ggml_log_callback previous;
    void * previous_data;
    int contexts = 0;
    bool context_error = false;

    fit_log_capture() {
        llama_log_get(&previous, &previous_data);
        llama_log_set(callback, this);
    }

    ~fit_log_capture() {
        llama_log_set(previous, previous_data);
    }

    static void callback(ggml_log_level, const char * text, void * data) {
        auto & capture = *static_cast<fit_log_capture *>(data);
        const std::string message(text);
        if (message.find("constructing llama_context") != std::string::npos) {
            ++capture.contexts;
        }
        capture.context_error |= message.find("failed to initialize the context") != std::string::npos;
    }
};

static void test_fit(const std::string & target_path, const std::string & head_path, bool separate_only = false) {
    for (auto mode : {COMMON_SPECULATIVE_TYPE_DRAFT_MTP, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE}) {
        for (bool separate : {false, true}) {
            if (separate_only && !separate) { continue; }
            fprintf(stderr, "TEST fit adaptive=%d separate=%d\n", mode == COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, separate);
            common_params params;
            params.model.path = target_path;
            params.n_ctx = 128;
            params.n_batch = params.n_ubatch = 8;
            params.n_gpu_layers = 0;
            params.devices = {nullptr};
            params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 1;
            params.fit_params = true;
            params.fit_params_min_ctx = 128;
            std::fill(params.fit_params_target.begin(), params.fit_params_target.end(), 0);
            params.speculative.types = {mode};
            if (separate) {
                params.speculative.draft.mparams.path = head_path;
                params.speculative.draft.n_gpu_layers = 0;
            }
            fit_log_capture logs;
            const auto result = common_init_from_params(params, true);
            require(result && result->model(), "fit initialization loads target");
            require(!logs.context_error, "fit constructs target and draft graphs without errors");
            // Model-only initialization leaves exactly the two fit measurement contexts.
            // This also detects shared adaptive drafts silently omitted from fitting.
            require(logs.contexts == 2, "fit measures both target and MTP draft contexts");
        }
    }
}

int main(int argc, char ** argv) {
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) { fputs(text, stderr); }
    }, nullptr);
    llama_backend_init();
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--q8-kv") {
            q8_kv = true;
        } else if (std::string(argv[i]) == "--backend" && i + 1 < argc) {
            ggml_backend_load_all();
            test_device = ggml_backend_dev_by_name(argv[++i]);
            require(test_device != nullptr, "requested backend is available");
        }
    }
    const auto dir = std::filesystem::temp_directory_path() / ("test-qwen4exp-mtp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    require(std::filesystem::create_directory(dir), "create temporary fixture directory");
    const std::string head_path = (dir / "head.gguf").string();
    const std::string target_path = (dir / "combined.gguf").string();
    const std::string changed_path = (dir / "changed.gguf").string();
    write_fixture(head_path, true);
    write_fixture(target_path, false);
    write_fixture(changed_path, false, -1, 9.0f);
    if (argc == 2 && (std::string(argv[1]) == "--fit-only" || std::string(argv[1]) == "--fit-separate-only")) {
        test_fit(target_path, head_path, std::string(argv[1]) == "--fit-separate-only");
        std::filesystem::remove_all(dir);
        llama_backend_free();
        fprintf(stderr, "PASS Qwen4Exp MTP fit regression suite\n");
        return 0;
    }
    auto head = load_model(head_path);
    auto target = load_model(target_path);
    auto changed = load_model(changed_path);
    require(head && target && changed, "canonical head and combined models load");
    if (argc == 2 && std::string(argv[1]) == "--head-only") {
        auto ctx = make_context(head.get(), false);
        decode(ctx.get(), {5}, initial_hidden(), 0);
        fprintf(stderr, "PASS canonical head decode\n");
        std::filesystem::remove_all(dir);
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--mixer-only") {
        auto a = make_context(target.get(), false);
        auto b = make_context(changed.get(), false);
        require(close(decode(a.get(), {5}, initial_hidden(), 0).logits,
                      decode(b.get(), {5}, initial_hidden(), 0).logits), "MTP logits independent of trunk mixer");
        fprintf(stderr, "PASS independent draft mixer\n");
        std::filesystem::remove_all(dir);
        return 0;
    }
    for (int omitted = 0; omitted < 4; ++omitted) {
        const std::string bad = (dir / "missing.gguf").string();
        write_fixture(bad, true, omitted);
        require(!load_model(bad), "missing or partial head mixer rejected");
    }
    const std::string output_only = (dir / "output-only.gguf").string();
    write_fixture(output_only, false, 3);
    require(!load_model(output_only), "trunk mixer does not substitute for missing draft mixer");
    require(bool(load_model(output_only, false)), "ordinary loading does not require unused draft mixers");
    require(llama_model_supports_mtp_chain(head.get()), "Qwen4Exp advertises implemented chain support");
    for (bool flash : {false, true}) {
        if (q8_kv && !flash) { continue; }
        auto a = make_context(head.get(), flash);
        auto b = make_context(target.get(), flash);
        auto c = make_context(changed.get(), flash);
        const auto ah = decode(a.get(), {5}, initial_hidden(), 0);
        const auto bh = decode(b.get(), {5}, initial_hidden(), 0);
        const auto ch = decode(c.get(), {5}, initial_hidden(), 0);
        require(close(ah.logits, bh.logits) && close(bh.logits, ch.logits), "MTP logits independent of trunk mixer");
        test_masked_catchup(head.get(), flash);
        for (int depth : {1, 3, 4}) {
            for (int catchup : {0, 2}) {
                for (bool masked : {true, false}) { test_chain(head.get(), flash, depth, catchup, masked); }
            }
        }
        test_chain(head.get(), flash, 4, 2, true, true);
    }
    test_driver(target.get(), head.get());
    test_fit(target_path, head_path);
    head.reset(); target.reset(); changed.reset();
    std::filesystem::remove_all(dir);
    llama_backend_free();
    fprintf(stderr, "PASS Qwen4Exp MTP regression suite\n");
}
