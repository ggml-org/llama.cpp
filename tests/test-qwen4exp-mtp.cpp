#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"
#include "speculative.h"
#include "../src/llama-ext.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "../src/llama-memory-recurrent.h"

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
static void write_fixture(const std::string & path, bool mtp_only, int omit_head = -1, float trunk_scale = 1.0f,
                          bool shared_embd = true, float output_scale = 1.0f, int vocab_size = n_vocab,
                          const char * omit_tensor = nullptr) {
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
    for (int i = 0; i < vocab_size; ++i) {
        vocab.push_back("tok_" + std::to_string(i));
    }
    ms.add_kv(LLM_KV_TOKENIZER_LIST, vocab);
    ms.add_kv(LLM_KV_TOKENIZER_SCORES, std::vector<float>(vocab_size, 0.0f));

    ggml_context_ptr tensors(ggml_init({8 * 1024 * 1024, nullptr, false}));
    auto add = [&](const std::string & name, std::initializer_list<int64_t> dims, float scale = 1.0f) {
        if (omit_tensor != nullptr && name == omit_tensor) {
            return;
        }
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
    if (shared_embd) {
        add("token_embd.weight", {n_embd, vocab_size});
        add("output.weight", {n_embd, vocab_size}, output_scale);
    }
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

static llama_context_ptr make_context(llama_model * model, bool flash, int n_seq = 1, int n_ubatch = 32,
                                      llama_context * other = nullptr,
                                      ggml_backend_sched_eval_callback cb_eval = nullptr, void * cb_data = nullptr) {
    auto params = llama_context_default_params();
    params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    params.ctx_other = other;
    params.n_ctx = 128;
    params.n_batch = 32;
    params.n_ubatch = n_ubatch;
    params.n_seq_max = n_seq;
    params.n_threads = params.n_threads_batch = 2;
    params.type_k = q8_kv ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    params.type_v = q8_kv ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    params.flash_attn_type = flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = cb_data;
    llama_context_ptr ctx(llama_init_from_model(model, params));
    require(bool(ctx), "create MTP context");
    llama_set_embeddings_nextn(ctx.get(), true, true);
    return ctx;
}

struct decoded {
    std::vector<float> logits;
    std::vector<float> hidden;
};

static size_t compute_bytes(const llama_context * ctx) {
    size_t total = 0;
    for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx)) {
        total += mb.compute;
    }
    return total;
}

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
    const size_t reserved = compute_bytes(chain.get());
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
    // --fit measures the ordinary reservation, which never builds the chain graph. The chain has to
    // run inside it, or at least inside what one-row decodes of the same rows grew it to.
    require(compute_bytes(chain.get()) <= std::max(reserved, compute_bytes(seq.get())),
            "chain decode stays within the compute buffers of the ordinary graph");
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

// The chain's hidden export is read back after the whole graph ran. At depth one it is the step's
// own hidden state, which the LM head consumes, so only the output flag keeps the allocator from
// handing its memory to a later node. llm_graph_result::set_outputs() sets it on t_h_nextn after
// every build; this pins that for the chain branch, which does not set it itself.
static void test_chain_export_is_output(llama_model * model, bool flash) {
    struct exports_seen {
        int exports = 0;
        int outputs = 0;
    } seen;
    const auto observe = [](ggml_tensor * t, bool ask, void * data) {
        if (ask && std::string(t->name) == "h_nextn") {
            auto * s = static_cast<exports_seen *>(data);
            ++s->exports;
            s->outputs += (t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0;
        }
        return !ask;
    };
    for (int depth : {1, 3}) {
        auto ctx = make_context(model, flash, 1, 32, nullptr, observe, &seen);
        std::vector<llama_token> tokens(depth, 0);
        tokens[0] = 5;
        decode(ctx.get(), tokens, initial_hidden(depth), 0, true);
    }
    require(seen.exports >= 2, "chain decodes export their hidden state");
    require(seen.outputs == seen.exports, "the chain's hidden export is a graph output");
    fprintf(stderr, "PASS chain export is a graph output flash=%d\n", flash);
}

// A sequence snapshot kept on the device restores the cache it was taken from for every K/V type.
// Block-quantized rows are sized in blocks, so both directions have to convert to elements.
static void test_on_device_state(llama_model * model, bool flash) {
    auto ctx = make_context(model, flash);
    decode(ctx.get(), {3, 4, 5}, initial_hidden(3), 0);
    const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
    std::vector<uint8_t> state(llama_state_seq_get_size_ext(ctx.get(), 0, flags));
    require(llama_state_seq_get_data_ext(ctx.get(), state.data(), state.size(), 0, flags) == state.size(),
            "on-device snapshot");
    const auto expected = decode(ctx.get(), {6}, initial_hidden(), 3);

    // put other rows into the same cells, then restore over them
    require(llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, 0, -1), "drop the snapshotted rows");
    decode(ctx.get(), {7, 8, 9}, initial_hidden(3), 0);
    require(llama_state_seq_set_data_ext(ctx.get(), state.data(), state.size(), 0, flags) == state.size(),
            "on-device restore");
    const auto actual = decode(ctx.get(), {6}, initial_hidden(), 3);
    require(close(expected.logits, actual.logits) && close(expected.hidden, actual.hidden),
            "on-device restore reproduces the snapshotted cache");
    fprintf(stderr, "PASS on-device sequence state flash=%d\n", flash);
}

// A speculative checkpoint stays on the device only while the device keeps its margin, and that is
// decided again whenever the checkpoint needs a larger device copy than the context holds.
static void test_checkpoint_placement(llama_model * model, llama_model * model_host, bool flash) {
    const llama_state_seq_flags on_host   = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const llama_state_seq_flags on_device = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
    const std::vector<size_t> no_margin = {0};
    const std::vector<size_t> no_room   = {(size_t) 1 << 50}; // a margin that no device can keep

    auto ctx = make_context(model, flash);
    decode(ctx.get(), {3, 4, 5}, initial_hidden(3), 0);

    common_speculative_checkpoint_place place;
    require(common_speculative_checkpoint_flags(place, ctx.get(), 0, no_margin, model) == on_device && place.flags == on_device,
            "a checkpoint with room stays on the device");
    const size_t size_first = place.size_copy;
    require(size_first > 0, "the checkpoint has tensor data to place");

    require(common_speculative_checkpoint_flags(place, ctx.get(), 0, no_room, model) == on_device && place.size_copy == size_first,
            "a checkpoint of the same size keeps its device copy: nothing new is allocated");

    // this context saves its whole cache, so the checkpoint grows with the sequence
    decode(ctx.get(), {6, 7, 8}, initial_hidden(3), 3);
    const llama_state_seq_flags grown = common_speculative_checkpoint_flags(place, ctx.get(), 0, no_room, model);
    if (test_device != nullptr) {
        require(grown == on_host && place.flags == on_host,
                "a checkpoint that outgrew its device copy is checked against the margin again");
        require(place.size_copy == size_first, "the context still holds the copy of the last device checkpoint");
    } else {
        // without a device the tensors live in host memory, which has no margin to keep
        require(grown == on_device && place.size_copy > size_first, "a host-only model has no device to check");
    }

    require(common_speculative_checkpoint_flags(place, ctx.get(), 0, no_margin, model) == on_device && place.size_copy > size_first,
            "a grown checkpoint with room goes to the device");

    // The margins follow the device order of the model --fit-target was given for. This model's device is
    // the first of its own list, and no device of a host-only model.
    const std::vector<size_t> first_only = {0, (size_t) 1 << 50};
    common_speculative_checkpoint_place own, other;
    require(common_speculative_checkpoint_flags(own, ctx.get(), 0, first_only, model) == on_device,
            "a device takes the margin of its position in the reference model");
    if (test_device != nullptr) {
        require(common_speculative_checkpoint_flags(other, ctx.get(), 0, first_only, model_host) == on_host,
                "a device the margins were not given for keeps the largest one");
    }
    fprintf(stderr, "PASS speculative checkpoint placement flash=%d\n", flash);
}

static std::vector<uint8_t> sequence_state(llama_context * ctx, llama_seq_id seq) {
    std::vector<uint8_t> bytes(llama_state_seq_get_size(ctx, seq));
    require(llama_state_seq_get_data(ctx, bytes.data(), bytes.size(), seq) == bytes.size(), "snapshot sequence cache");
    return bytes;
}

static void test_invalid_chain(llama_model * model, bool flash) {
    enum invalid_case { NON_PREFIX_MASK, NO_OUTPUT, OVER_UBATCH, MULTIPLE_SEQUENCES, NO_HIDDEN_STATE, NULL_TOKEN,
                        BACKEND_SAMPLER, EMBEDDINGS };
    for (auto kind : {NON_PREFIX_MASK, NO_OUTPUT, OVER_UBATCH, MULTIPLE_SEQUENCES, NO_HIDDEN_STATE, NULL_TOKEN,
                      BACKEND_SAMPLER, EMBEDDINGS}) {
        auto ctx = make_context(model, flash, 2, 2);
        auto reference = make_context(model, flash, 2, 2);
        const auto seed = decode(ctx.get(), {3}, initial_hidden(), 0);
        decode(reference.get(), {3}, initial_hidden(), 0);
        const auto before_0 = sequence_state(ctx.get(), 0);
        const auto before_1 = sequence_state(ctx.get(), 1);
        const llama_pos max_before = llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0);

        const int n_bad = kind == OVER_UBATCH ? 4 : 2;
        auto inputs = initial_hidden(n_bad);
        llama_batch bad = llama_batch_init(n_bad, 0, 1);
        for (int i = 0; i < n_bad; ++i) {
            const llama_seq_id seq = kind == MULTIPLE_SEQUENCES ? i : 0;
            const llama_pos pos = seq == 1 ? 0 : i + 1;
            // One output row keeps the sampler case inside the generic per-sequence output limit.
            const bool output = kind == NO_OUTPUT ? false : kind == NON_PREFIX_MASK ? i == 0 :
                    kind == OVER_UBATCH ? i >= 2 : kind == BACKEND_SAMPLER ? i == 1 : true;
            common_batch_add(bad, kind == NULL_TOKEN && i == 1 ? LLAMA_TOKEN_NULL : 5, pos, {seq}, output);
        }
        bad.embd = kind == NO_HIDDEN_STATE ? nullptr : inputs.data();
        // A chain packs [token, probability] rows where a backend sampler expects vocabulary logits.
        llama_sampler_ptr sampler;
        if (kind == BACKEND_SAMPLER) {
            sampler.reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
            require(llama_set_sampler(ctx.get(), 0, sampler.get()), "backend sampler attaches to the draft context");
        }
        // With embeddings on every row counts as an output, and a chain builds no embedding tensor.
        llama_set_embeddings(ctx.get(), kind == EMBEDDINGS);
        llama_set_mtp_chain(ctx.get(), true);
        const int rc = llama_decode(ctx.get(), bad);
        bad.embd = nullptr;
        llama_batch_free(bad);
        llama_set_embeddings(ctx.get(), false);
        if (kind == BACKEND_SAMPLER) {
            require(llama_set_sampler(ctx.get(), 0, nullptr), "backend sampler detaches");
        }
        require(rc == -1, "invalid public chain input returns -1");
        require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == max_before &&
                sequence_state(ctx.get(), 0) == before_0 && sequence_state(ctx.get(), 1) == before_1,
                "rejected chain leaves existing and empty sequence caches unchanged");

        // Keep chain mode enabled: a failed call must not poison the next valid decode.
        std::vector<float> retry_input(2 * n_hidden, 0.0f);
        std::copy(seed.hidden.begin(), seed.hidden.end(), retry_input.begin());
        llama_batch retry = llama_batch_init(2, 0, 1);
        common_batch_add(retry, 5, 1, {0}, true);
        common_batch_add(retry, 0, 2, {0}, true);
        retry.embd = retry_input.data();
        const int retry_rc = llama_decode(ctx.get(), retry);
        retry.embd = nullptr;
        llama_batch_free(retry);
        require(retry_rc == 0, "valid chain retry succeeds on the same context");
        const float * logits = llama_get_logits(ctx.get());
        const float * hidden = llama_get_embeddings_nextn(ctx.get());
        require(logits && hidden, "valid retry has chain outputs");
        const decoded actual{{logits, logits + 4}, {hidden, hidden + 2 * n_hidden}};
        const auto expected = decode(reference.get(), {5, 0}, retry_input, 1, true);
        require(close(actual.logits, expected.logits) && close(actual.hidden, expected.hidden),
                "retry preserves packed chain probabilities and per-step hidden states");
        llama_set_mtp_chain(ctx.get(), false);
        fprintf(stderr, "PASS invalid chain flash=%d case=%d cache and retry\n", flash, int(kind));
    }
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

static std::vector<float> tensor_data(const ggml_tensor * t) {
    require(t != nullptr && t->type == GGML_TYPE_F32, "fixture tensor is f32");
    std::vector<float> data(ggml_nelements(t));
    ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
    return data;
}

static size_t context_memory(const llama_context * ctx) {
    size_t total = 0;
    for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx)) {
        total += mb.context;
    }
    return total;
}

// An MTP-only head has no trunk blocks. Its ordinary context runs what the file
// carries, the shared embedding and LM head, and caches no layers.
static void test_ordinary_context(llama_model * head, llama_model * combined) {
    auto params = llama_context_default_params();
    params.n_ctx = 128;
    params.n_batch = params.n_ubatch = 4;
    params.n_threads = params.n_threads_batch = 1;
    llama_context_ptr ctx(llama_init_from_model(head, params));
    require(bool(ctx), "ordinary context accepts an MTP-only head");
    llama_context_ptr full(llama_init_from_model(combined, params));
    require(bool(full), "ordinary context still accepts a combined model");
    require(context_memory(ctx.get()) == 0 && context_memory(full.get()) > 0,
            "trunkless context holds no cache layers");

    const auto embd = tensor_data(head->tok_embd);
    const auto head_w = tensor_data(head->output);
    auto expected = [&](llama_token token) {
        std::vector<float> logits(n_vocab, 0.0f);
        for (int v = 0; v < n_vocab; ++v) {
            for (int i = 0; i < n_embd; ++i) {
                logits[v] += head_w[v * n_embd + i] * embd[token * n_embd + i];
            }
        }
        return logits;
    };
    auto run = [&](llama_context * c, const std::vector<llama_token> & tokens, int pos) {
        llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); ++i) {
            common_batch_add(batch, tokens[i], pos + i, {0}, i + 1 == tokens.size());
        }
        const int rc = llama_decode(c, batch);
        llama_batch_free(batch);
        require(rc == 0, "trunkless decode");
        const float * logits = llama_get_logits_ith(c, -1);
        require(logits != nullptr, "trunkless logits");
        return std::vector<float>(logits, logits + n_vocab);
    };
    // Backends may evaluate the head in f16, so compare against the host product loosely.
    const float tolerance = 2e-3f;
    require(close(run(ctx.get(), {3, 4, 5}, 0), expected(5), tolerance), "trunkless logits are head(embedding)");
    require(close(run(ctx.get(), {7}, 3), expected(7), tolerance), "trunkless continuation");

    llama_memory_t memory = llama_get_memory(ctx.get());
    require(llama_memory_seq_pos_max(memory, 0) == 3, "trunkless context tracks positions");
    const auto state = sequence_state(ctx.get(), 0);
    llama_context_ptr restored(llama_init_from_model(head, params));
    require(bool(restored), "second trunkless context");
    require(llama_state_seq_set_data(restored.get(), state.data(), state.size(), 0) == state.size(),
            "trunkless sequence state restores");
    require(llama_memory_seq_pos_max(llama_get_memory(restored.get()), 0) == 3, "restored positions");
    require(llama_memory_seq_rm(memory, 0, 2, -1) && llama_memory_seq_pos_max(memory, 0) == 1,
            "trunkless rollback");
    require(close(run(ctx.get(), {6}, 2), expected(6), tolerance), "trunkless decode after rollback");

    // Whole-context state and the cache type query both walk the cache layers.
    require(llama_get_kv_cache_type_k(ctx.get()) == GGML_TYPE_COUNT &&
            llama_get_kv_cache_type_v(ctx.get()) == GGML_TYPE_COUNT, "trunkless cache reports no K/V type");
    std::vector<uint8_t> full_state(llama_state_get_size(ctx.get()));
    require(llama_state_get_data(ctx.get(), full_state.data(), full_state.size()) == full_state.size(),
            "trunkless context state saves");
    require(llama_state_set_data(restored.get(), full_state.data(), full_state.size()) == full_state.size() &&
            llama_memory_seq_pos_max(llama_get_memory(restored.get()), 0) == 2, "trunkless context state restores");

    // The hidden export is switched on after the context exists. The buffers were planned without it,
    // so the context has to plan them again or a later node overwrites the exported tensor in place.
    {
        std::vector<float> computed;
        auto export_params = params;
        export_params.cb_eval = [](ggml_tensor * t, bool ask, void * data) {
            if (std::string(t->name) != "h_nextn") { return !ask; }
            if (!ask) {
                auto & values = *static_cast<std::vector<float> *>(data);
                values.resize(ggml_nelements(t));
                ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
            }
            return true;
        };
        export_params.cb_eval_user_data = &computed;
        llama_context_ptr export_ctx(llama_init_from_model(combined, export_params));
        require(bool(export_ctx), "hidden export context");
        llama_set_embeddings_nextn(export_ctx.get(), true, true);
        llama_batch batch = llama_batch_init(2, 0, 1);
        common_batch_add(batch, 4, 0, {0}, true);
        common_batch_add(batch, 8, 1, {0}, true);
        const int rc = llama_decode(export_ctx.get(), batch);
        llama_batch_free(batch);
        require(rc == 0 && computed.size() == size_t(2 * n_hidden), "hidden export decode");
        const float * exported = llama_get_embeddings_nextn(export_ctx.get());
        require(exported != nullptr && close(computed, std::vector<float>(exported, exported + 2 * n_hidden)),
                "the hidden export is the wide residual as it was computed");
        fprintf(stderr, "PASS hidden export survives the graph\n");
    }

    // Embeddings are read n_embd_out = hc * n_embd wide, which is the wide residual the hidden export
    // carries, for the trunkless head and for a full trunk alike.
    for (llama_model * model : {head, combined}) {
        auto embd_params = params;
        embd_params.embeddings = true;
        embd_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        llama_context_ptr embd_ctx(llama_init_from_model(model, embd_params));
        require(bool(embd_ctx), "embeddings context");
        llama_set_embeddings_nextn(embd_ctx.get(), true, true);
        llama_batch batch = llama_batch_init(2, 0, 1);
        common_batch_add(batch, 4, 0, {0}, true);
        common_batch_add(batch, 8, 1, {0}, true);
        const int rc = llama_decode(embd_ctx.get(), batch);
        llama_batch_free(batch);
        require(rc == 0, "embeddings decode");
        const float * embd_out = llama_get_embeddings(embd_ctx.get());
        const float * hidden_out = llama_get_embeddings_nextn(embd_ctx.get());
        require(embd_out != nullptr && hidden_out != nullptr, "embeddings and hidden export");
        require(close(std::vector<float>(embd_out, embd_out + 2 * n_hidden),
                      std::vector<float>(hidden_out, hidden_out + 2 * n_hidden)),
                "embeddings are the wide residual");
    }
    fprintf(stderr, "PASS embeddings width\n");

    // The wide hidden export is the zero-block residual: hc copies of the embedding.
    llama_set_embeddings_nextn(ctx.get(), true, true);
    run(ctx.get(), {4, 8}, 3);
    const float * hidden = llama_get_embeddings_nextn(ctx.get());
    require(hidden != nullptr, "trunkless masked hidden export");
    std::vector<float> wide;
    for (int c = 0; c < n_hc; ++c) {
        wide.insert(wide.end(), embd.begin() + 8 * n_embd, embd.begin() + 9 * n_embd);
    }
    require(close(std::vector<float>(hidden, hidden + n_hidden), wide, tolerance), "masked hidden row is the output row");
    llama_set_embeddings_nextn(ctx.get(), true, false);
    require(close(run(ctx.get(), {4, 8}, 5), expected(8), tolerance), "trunkless logits with unmasked hidden export");
    hidden = llama_get_embeddings_nextn(ctx.get());
    require(hidden != nullptr, "trunkless unmasked hidden export");
    require(close(std::vector<float>(hidden + n_hidden, hidden + 2 * n_hidden), wide, tolerance) &&
            close(std::vector<float>(hidden, hidden + n_embd),
                  std::vector<float>(embd.begin() + 4 * n_embd, embd.begin() + 5 * n_embd), tolerance),
            "unmasked hidden rows cover every input token");
    fprintf(stderr, "PASS ordinary context runs MTP-only head without a trunk\n");
}

static void test_chain_metadata(llama_model * model) {
    for (uint32_t ubatch : {32u, 512u}) {
        auto params = llama_context_default_params();
        params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        params.n_ctx = 1024;
        params.n_batch = params.n_ubatch = ubatch;
        params.n_threads = params.n_threads_batch = 1;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context_ptr ctx(llama_init_from_model(model, params));
        require(bool(ctx), "metadata test context");
        const uint32_t base_nodes = std::max<uint32_t>(40 * ubatch, 32 * model->n_tensors());
        auto * reserve = ctx->get_gf_res_reserve();
        require(reserve->get_max_nodes() == base_nodes, "sequential-only MTP retains original node budget");
        const size_t initial_bytes = reserve->buf_compute_meta.size();
        const uint32_t eager_nodes = std::max<uint32_t>(512 * ubatch, 32 * model->n_tensors());
        const size_t eager_bytes = ggml_tensor_overhead() * eager_nodes + ggml_graph_overhead_custom(eager_nodes, false);
        fprintf(stderr, "METADATA ubatch=%u initial=%zu prior_eager=%zu bytes_per_arena\n", ubatch, initial_bytes, eager_bytes);

        const int depth = ubatch == 32 ? 32 : 64;
        decode(ctx.get(), std::vector<llama_token>(depth, 5), initial_hidden(depth), 0, true);
        reserve = ctx->get_gf_res_reserve();
        const uint32_t grown_nodes = std::max<uint32_t>(base_nodes, 512 * depth);
        require(reserve->get_max_nodes() == grown_nodes, "metadata grows with actual chain rows");
        const size_t grown_bytes = reserve->buf_compute_meta.size();
        fprintf(stderr, "METADATA ubatch=%u depth=%d grown=%zu bytes_per_arena\n", ubatch, depth, grown_bytes);

        // Toggle through both graph types. A shorter chain must reuse the reservation.
        decode(ctx.get(), {5}, initial_hidden(), depth);
        decode(ctx.get(), {5}, initial_hidden(), depth + 1, true);
        require(ctx->get_gf_res_reserve() == reserve && reserve->buf_compute_meta.size() == grown_bytes,
                "chain toggles and shorter drafts retain the existing reservation");
    }
}

#include "test-qwen4exp-mtp-driver.h"

static void test_batch_validation(llama_model * head, const std::filesystem::path & dir) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = cp.n_ubatch = 32;
    cp.n_threads = cp.n_threads_batch = 1;
    llama_context_ptr draft(llama_init_from_model(head, cp));
    require(bool(draft), "batch validation context");
    common_batch batch(draft.get());
    require(batch.add(3, 0, 0, true) == 0, "valid batch row");
    require(batch.add(n_vocab, 1, 0, true) == -2 && batch.add(LLAMA_TOKEN_NULL, 1, 0, true) == -2,
            "invalid tokens are rejected before rendering");
    require(batch.add(3, 1, -1, true) == -3 && batch.add(3, 1, 1, true) == -3,
            "invalid sequences are rejected before rendering");
    require(batch.add(3, 1, std::vector<llama_seq_id>{0, 1}, true) == -3 && batch.size() == 1,
            "failed shared row leaves the batch unchanged");
    require(!batch.add_seq(0, 1) && batch.tokens[0].seq_ids_extra.empty(), "invalid extra sequence rejected");
    require(batch.remove_last() && batch.size() == 0 && !batch.remove_last(), "remove an unrendered row");
    for (uint32_t i = 0; i < llama_n_batch(draft.get()) + 1; ++i) {
        require(batch.add(3, i, 0, true) == int32_t(i), "buffer beyond decode capacity");
    }
    require(llama_process(draft.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get_sub_batch(0, 32)) == 0 &&
            llama_process(draft.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get_sub_batch(32, 1)) == 0,
            "buffered rows decode in chunks");
    llama_memory_clear(llama_get_memory(draft.get()), true);

    const std::string wider_path = (dir / "batch-wider.gguf").string();
    write_fixture(wider_path, false, -1, 1.0f, true, 1.0f, n_vocab + 16);
    auto wider = load_model(wider_path);
    require(bool(wider), "wider target model");
    llama_context_ptr target(llama_init_from_model(wider.get(), cp));
    require(bool(target), "wider target context");
    common_params_speculative params;
    params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE};
    params.draft.ctx_tgt = target.get();
    params.draft.ctx_dft = draft.get();
    params.draft.n_max = 2;
    params.draft.backend_sampling = false;
    common_speculative_ptr spec(common_speculative_init(params, 1));
    require(bool(spec), "narrower compatible draft initializes");
    common_batch input(target.get());
    require(input.add(3, 0, 0, true) == 0 && input.add(n_vocab, 1, 0, true) == 1 &&
            input.add(4, 2, 0, true) == 2, "target-only token in prompt");
    require(common_speculative_process(spec.get(), input), "target-only token skips drafting without aborting");
    // MROPE permits later position gaps; check the skipped suffix within this mixed batch.
    require(llama_memory_seq_pos_max(llama_get_memory(draft.get()), 0) == 0,
            "draft mirrors only the valid prefix");
    fprintf(stderr, "PASS batch validation and narrower draft vocabulary\n");
}

static void test_empty_recurrent_memory(llama_model * model) {
    llama_memory_recurrent memory(*model, GGML_TYPE_F32, GGML_TYPE_F32, false, 4, 1, 3,
                                  [](int32_t) { return false; });
    require(memory.n_rs_seq == 0, "empty filtered memory disables rollback snapshots on the member");
    fprintf(stderr, "PASS empty recurrent memory disables rollback snapshots\n");
}

// A head exported without token_embd.weight and output.weight drafts with the tables of the
// model it drafts for, reached through llama_context_params::ctx_other.
static void test_borrowed_tables(const std::string & bare_path, const std::string & doubled_path,
                                 llama_model * head, llama_model * target) {
    auto bare = load_model(bare_path);
    auto doubled = load_model(doubled_path);
    require(bare && doubled, "table-less head and target with a doubled LM head load");

    auto parent_params = llama_context_default_params();
    parent_params.n_ctx = 128;
    parent_params.n_threads = parent_params.n_threads_batch = 1;
    llama_context_ptr parent(llama_init_from_model(target, parent_params));
    llama_context_ptr parent_doubled(llama_init_from_model(doubled.get(), parent_params));
    require(parent && parent_doubled, "parent contexts for the table-less head");

    for (bool flash : {false, true}) {
        if (q8_kv && !flash) { continue; }
        // The fixtures seed tensors by name, so the target's tables equal the canonical head's.
        auto own = make_context(head, flash);
        auto borrowed = make_context(bare.get(), flash, 1, 32, parent.get());
        const auto a = decode(own.get(), {5}, initial_hidden(), 0);
        const auto b = decode(borrowed.get(), {5}, initial_hidden(), 0);
        require(close(a.logits, b.logits) && close(a.hidden, b.hidden),
                "borrowed tables reproduce the canonical head");

        // A chain looks up the embedding of every drafted token inside the graph.
        const int depth = 3, catchup = 2;
        std::vector<llama_token> tokens(depth + catchup, 3);
        tokens[catchup] = 5;
        auto own_chain = make_context(head, flash);
        auto borrowed_chain = make_context(bare.get(), flash, 1, 32, parent.get());
        const auto c = decode(own_chain.get(), tokens, initial_hidden(depth + catchup), 0, true, catchup);
        const auto d = decode(borrowed_chain.get(), tokens, initial_hidden(depth + catchup), 0, true, catchup);
        require(close(c.logits, d.logits) && close(c.hidden, d.hidden),
                "borrowed tables reproduce the canonical chain");

        // The LM head is the last, linear step, so doubling the target's output.weight doubles the logits.
        auto scaled = make_context(bare.get(), flash, 1, 32, parent_doubled.get());
        auto e = decode(scaled.get(), {5}, initial_hidden(), 0);
        for (float & logit : e.logits) { logit *= 0.5f; }
        require(close(a.logits, e.logits) && close(a.hidden, e.hidden), "the LM head is the target model's");
        fprintf(stderr, "PASS borrowed tables flash=%d\n", flash);
    }

    // A head on other devices than its target cannot use tables that sit in the target's device buffers.
    if (test_device != nullptr) {
        ggml_backend_dev_t device_saved = test_device;
        test_device = nullptr;
        auto bare_host = load_model(bare_path);
        test_device = device_saved;
        require(bool(bare_host), "host-only table-less head loads");
        auto params = llama_context_default_params();
        params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        params.ctx_other = parent.get();
        params.n_ctx = 128;
        require(!llama_context_ptr(llama_init_from_model(bare_host.get(), params)),
                "borrowed tables in a buffer the head's devices cannot use are rejected");
        fprintf(stderr, "PASS borrowed tables on a foreign device\n");
    }

    // The public driver hands the target context to the draft context the same way the server does.
    for (bool chained : {false, true}) {
        require(driver_drafts(target, bare.get(), chained, 1, 8, false) == driver_drafts(target, head, chained, 1, 8, false),
                "driver drafts match with borrowed tables");
    }
    fprintf(stderr, "PASS borrowed tables through the public driver\n");

    // A parent whose model has no tables either cannot lend any: creation fails instead of aborting.
    {
        auto lender = make_context(bare.get(), q8_kv, 1, 32, parent.get());
        auto params = llama_context_default_params();
        params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        params.n_ctx = 128;
        params.n_threads = params.n_threads_batch = 1;
        params.ctx_other = lender.get();
        require(!llama_context_ptr(llama_init_from_model(bare.get(), params)),
                "a target without tables is reported, not borrowed from");
    }
    fprintf(stderr, "PASS table-less target rejected\n");

    // A target with another vocabulary has tables of the wrong shape: creation fails instead of reading them.
    {
        const std::string wider_path = (std::filesystem::path(bare_path).parent_path() / "wider.gguf").string();
        write_fixture(wider_path, false, -1, 1.0f, true, 1.0f, n_vocab + 16);
        auto wider = load_model(wider_path);
        require(bool(wider), "target with a wider vocabulary loads");
        llama_context_ptr parent_wider(llama_init_from_model(wider.get(), parent_params));
        require(bool(parent_wider), "parent context for the wider vocabulary");
        auto params = llama_context_default_params();
        params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        params.n_ctx = 128;
        params.n_threads = params.n_threads_batch = 1;
        params.ctx_other = parent_wider.get();
        require(!llama_context_ptr(llama_init_from_model(bare.get(), params)),
                "a target with another vocabulary is reported, not borrowed from");
    }
    fprintf(stderr, "PASS mismatched target tables rejected\n");
}

struct fit_log_capture {
    ggml_log_callback previous;
    void * previous_data;
    int contexts = 0;
    int context_errors = 0;
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
        if (message.find("failed to initialize the context") != std::string::npos) {
            ++capture.context_errors;
            capture.context_error = true;
        }
    }
};

struct reserve_log_capture {
    ggml_log_callback previous;
    void * previous_data;
    int reserves = 0;
    int catchup_graphs = 0; // graphs without an output row in the latest reservation

    reserve_log_capture() {
        llama_log_get(&previous, &previous_data);
        llama_log_set(callback, this);
    }

    ~reserve_log_capture() {
        llama_log_set(previous, previous_data);
    }

    static void callback(ggml_log_level, const char * text, void * data) {
        auto * capture = static_cast<reserve_log_capture *>(data);
        const std::string line(text);
        if (line.find("sched_reserve: reserving ...") != std::string::npos) {
            ++capture->reserves;
            capture->catchup_graphs = 0;
        }
        if (line.find("reserving a graph for ubatch") != std::string::npos &&
            line.find("n_outputs =    0") != std::string::npos) {
            ++capture->catchup_graphs;
        }
    }
};

// Chain batches grow by a row at a time while catch-up rows accumulate. A new maximum may need a
// larger graph, but not a scheduler reservation for every single row.
static void test_chain_reserve_growth(llama_model * model) {
    auto ctx = make_context(model, true);
    reserve_log_capture logs;
    const int max_rows = 32;
    for (int rows = 1; rows <= max_rows; ++rows) {
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        std::vector<llama_token> tokens(rows, 0);
        tokens[0] = 5;
        decode(ctx.get(), tokens, initial_hidden(rows), 0, true);
    }
    fprintf(stderr, "chain rows 1..%d took %d scheduler reservations\n", max_rows, logs.reserves);
    // one for switching the hidden export on, then one per doubling: 4, 8, 16, 32 rows
    require(logs.reserves <= 6, "chain row growth reserves geometrically, not per row");
    fprintf(stderr, "PASS chain reserve growth\n");
}

// A catch-up decode marks no row as an output. The allocator does not plan that graph as a
// subset of the prompt graph: with the scheduler's copy of the output indices empty, two free
// blocks merge and best fit places the tensors above them differently. On a real head that plan
// was 11 MiB larger than the reserved one, so the context has to reserve that shape as well.
static void test_catchup_reservation(llama_model * model) {
    reserve_log_capture logs;
    auto ctx = make_context(model, true);
    decode(ctx.get(), {5}, initial_hidden(), 0);
    require(logs.reserves >= 1, "an MTP context reserves its graphs");
    require(logs.catchup_graphs >= 1, "the MTP reservation covers a graph without output rows");
    fprintf(stderr, "PASS catch-up reservation\n");
}

// The same on real weights: a head-only GGUF given on the command line. One row first, so that
// the full catch-up ubatch that follows is planned from scratch, as after any draft.
static void test_catchup_real_head(const std::string & path) {
    auto model = load_model(path);
    require(bool(model), "load the real head");
    auto params = llama_context_default_params();
    params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    params.n_ctx = 4096;
    params.n_batch = 2048;
    params.n_ubatch = 512;
    params.n_seq_max = 1;
    params.n_outputs_max = 1; // as the server's draft context
    params.n_threads = params.n_threads_batch = 12;
    params.type_k = params.type_v = GGML_TYPE_Q8_0;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context_ptr ctx(llama_init_from_model(model.get(), params));
    require(bool(ctx), "create an MTP context on the real head");
    const int rows = int(llama_n_ubatch(ctx.get()));
    const std::vector<float> hidden(size_t(rows) * llama_model_n_embd_out(model.get()), 0.01f);
    int pos = 0;
    const auto run = [&](int n_rows, int n_outputs) {
        llama_batch batch = llama_batch_init(n_rows, 0, 1);
        for (int i = 0; i < n_rows; ++i) {
            common_batch_add(batch, 5, pos + i, {0}, i >= n_rows - n_outputs);
        }
        batch.embd = const_cast<float *>(hidden.data());
        llama_set_embeddings_nextn(ctx.get(), true, true);
        const int rc = llama_decode(ctx.get(), batch);
        batch.embd = nullptr;
        llama_batch_free(batch);
        require(rc == 0, "decode on the real head");
        pos += n_rows;
    };
    run(1, 1);
    const size_t reserved = compute_bytes(ctx.get());
    run(rows, 0);
    const size_t used = compute_bytes(ctx.get());
    fprintf(stderr, "real head: %.4f MiB reserved, %.4f MiB after a %d-row catch-up\n",
            reserved / 1048576.0, used / 1048576.0, rows);
    // Not an equality. The reserved graph carries the attention mask of a full cache and a decode a
    // smaller one, and best fit is not monotonic in tensor sizes: on the Q8_0 Qwen3.8 head one
    // 8 KiB tensor then lands above a 20 MiB one instead of in a hole, and the decode plans 8 KiB
    // more than was reserved. The missing reservation this test is about was 11 MiB.
    const size_t plan_noise_bytes = 1024 * 1024;
    require(used <= reserved + plan_noise_bytes, "a full catch-up ubatch fits the reserved compute buffers");
    fprintf(stderr, "PASS catch-up on a real head\n");
}

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

// A head without its own tables only builds next to its target's context. The fit must
// measure it there instead of dropping it and overcommitting the device.
static void test_fit_borrowed_head(const std::string & target_path, const std::string & bare_path) {
    for (auto mode : {COMMON_SPECULATIVE_TYPE_DRAFT_MTP, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE}) {
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
        params.speculative.draft.mparams.path = bare_path;
        params.speculative.draft.n_gpu_layers = 0;
        fit_log_capture logs;
        const auto result = common_init_from_params(params, true);
        require(result && result->model(), "fit initialization loads target");
        // target, the head alone (refused), then the target again as parent and the head beside it
        require(logs.contexts == 4, "fit measures a table-less head next to its target");
        // A failed measurement beside the parent is logged and the fit goes on without the head, so the
        // count alone does not show that it succeeded: only the head alone may fail.
        require(logs.context_errors == 1, "the table-less head constructs next to its target");
        fprintf(stderr, "PASS fit table-less head adaptive=%d\n", mode == COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE);
    }
}

// With a separate draft head the target's own MTP block is never used, so the target skips it.
// A combined target whose block lacks the draft mixer then still serves next to a complete head.
static void test_separate_head_target(const std::string & mixerless_path, const std::string & head_path,
                                      llama_model * target, llama_model * head) {
    for (auto mode : {COMMON_SPECULATIVE_TYPE_DRAFT_MTP, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE}) {
        common_params params;
        params.model.path = mixerless_path;
        params.n_ctx = 128;
        params.n_batch = params.n_ubatch = 8;
        params.n_gpu_layers = 0;
        params.devices = {nullptr};
        params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 1;
        params.fit_params = false;
        params.speculative.types = {mode};
        require(common_model_params_to_llama(params).load_mtp, "a target that drafts for itself loads its MTP block");
        params.speculative.draft.mparams.path = head_path;
        params.speculative.draft.n_gpu_layers = 0;
        require(!common_model_params_to_llama(params).load_mtp, "a target with a separate draft head skips its own MTP block");
        common_params params_dft = common_base_params_to_speculative(params);
        require(common_model_params_to_llama(params_dft).load_mtp, "the separate draft head loads its MTP block");
        // The same file named as target and as draft still loads twice, and only the draft runs its block.
        params.speculative.draft.mparams.path = params.model.path;
        require(!common_model_params_to_llama(params).load_mtp, "a target with a same-file draft skips its own MTP block");
        common_params params_same = common_base_params_to_speculative(params);
        require(common_model_params_to_llama(params_same).load_mtp, "the same-file draft loads its MTP block");
        params.speculative.draft.mparams.path = head_path;
        const auto result = common_init_from_params(params, true);
        require(result && result->model(), "a target without the draft mixer loads next to a separate head");
    }
    // The trunk is the same in both files, so skipping the target's MTP block leaves the drafts unchanged.
    auto mixerless = load_model(mixerless_path, false);
    require(bool(mixerless), "target without the draft mixer loads with its MTP block skipped");
    for (bool chained : {false, true}) {
        require(driver_drafts(mixerless.get(), head, chained, 1, 8, false) == driver_drafts(target, head, chained, 1, 8, false),
                "driver drafts match when the target skips its own MTP block");
    }
    fprintf(stderr, "PASS separate head target\n");
}

// The catch-up rows of a failed draft decode have to reach the draft cache on the next attempt.
// Dropping them leaves a hole below the draft position and every later draft decode is refused.
static void test_driver_failed_draft(llama_model * target, llama_model * head) {
    // only a chain decode refuses the backend sampler the driver test uses to make it fail
    require(driver_drafts(target, head, true, 1, 8, false, 1, false, nullptr, true) ==
            driver_drafts(target, head, true, 1, 8, false),
            "drafting recovers after a failed chain decode");
    fprintf(stderr, "PASS driver recovers from a failed chain decode\n");
}

// A head reached without an MTP speculative type gets an ordinary context. The fit
// must measure it instead of crashing or dropping it.
static void test_fit_ordinary_head(const std::string & target_path, const std::string & head_path) {
    auto base = [](const std::string & model_path) {
        common_params params;
        params.model.path = model_path;
        params.n_ctx = 128;
        params.n_batch = params.n_ubatch = 8;
        params.n_gpu_layers = 0;
        params.devices = {nullptr};
        params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 1;
        params.fit_params = true;
        params.fit_params_min_ctx = 128;
        std::fill(params.fit_params_target.begin(), params.fit_params_target.end(), 0);
        return params;
    };
    for (auto mode : {COMMON_SPECULATIVE_TYPE_NONE, COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE}) {
        fprintf(stderr, "TEST fit ordinary head as draft type=%d\n", int(mode));
        common_params params = base(target_path);
        params.speculative.types = {mode};
        params.speculative.draft.mparams.path = head_path;
        params.speculative.draft.n_gpu_layers = 0;
        fit_log_capture logs;
        const auto result = common_init_from_params(params, true);
        require(result && result->model(), "non-MTP fit loads target");
        require(!logs.context_error, "non-MTP fit constructs the head context without errors");
        require(logs.contexts == 2, "non-MTP fit measures both target and head contexts");
    }
    fprintf(stderr, "TEST fit ordinary head as main model\n");
    common_params params = base(head_path);
    fit_log_capture logs;
    const auto result = common_init_from_params(params, false);
    require(result && result->model() && result->context(), "head fits and initializes as the main model");
    require(!logs.context_error, "head main-model fit constructs contexts without errors");
}

int main(int argc, char ** argv) {
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) { fputs(text, stderr); }
    }, nullptr);
    llama_backend_init();
    std::string real_head_path;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--q8-kv") {
            q8_kv = true;
        } else if (std::string(argv[i]) == "--backend" && i + 1 < argc) {
            ggml_backend_load_all();
            test_device = ggml_backend_dev_by_name(argv[++i]);
            require(test_device != nullptr, "requested backend is available");
        } else if (std::string(argv[i]) == "--catchup-real-head" && i + 1 < argc) {
            real_head_path = argv[++i];
        }
    }
    if (!real_head_path.empty()) {
        // opt-in: needs a real head-only GGUF, which the fixtures cannot stand in for
        test_catchup_real_head(real_head_path);
        llama_backend_free();
        return 0;
    }
    const auto dir = std::filesystem::temp_directory_path() / ("test-qwen4exp-mtp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    require(std::filesystem::create_directory(dir), "create temporary fixture directory");
    const std::string head_path = (dir / "head.gguf").string();
    const std::string target_path = (dir / "combined.gguf").string();
    const std::string changed_path = (dir / "changed.gguf").string();
    const std::string bare_path = (dir / "bare.gguf").string();
    const std::string doubled_path = (dir / "doubled.gguf").string();
    write_fixture(head_path, true);
    write_fixture(bare_path, true, -1, 1.0f, false);
    write_fixture(doubled_path, false, -1, 1.0f, true, 2.0f);
    write_fixture(target_path, false);
    write_fixture(changed_path, false, -1, 9.0f);
    if (argc == 2 && (std::string(argv[1]) == "--fit-only" || std::string(argv[1]) == "--fit-separate-only")) {
        test_fit(target_path, head_path, std::string(argv[1]) == "--fit-separate-only");
        test_fit_ordinary_head(target_path, head_path);
        test_fit_borrowed_head(target_path, bare_path);
        std::filesystem::remove_all(dir);
        llama_backend_free();
        fprintf(stderr, "PASS Qwen4Exp MTP fit regression suite\n");
        return 0;
    }
    auto head = load_model(head_path);
    // the same head without a device, as the reference of a model on other devices
    ggml_backend_dev_t device_saved = test_device;
    test_device = nullptr;
    auto head_host = load_model(head_path);
    test_device = device_saved;
    auto target = load_model(target_path);
    auto changed = load_model(changed_path);
    require(head && head_host && target && changed, "canonical head and combined models load");
    // Every model has to be gone before the fixtures are deleted and the backends freed: a model
    // keeps its file mapped, and Windows cannot delete a mapped file.
    const auto finish = [&]() {
        head.reset();
        head_host.reset();
        target.reset();
        changed.reset();
        std::filesystem::remove_all(dir);
        llama_backend_free();
    };
    if (argc == 2 && std::string(argv[1]) == "--ordinary-head-only") {
        test_batch_validation(head.get(), dir);
        test_empty_recurrent_memory(target.get());
        test_ordinary_context(head.get(), target.get());
        test_ordinary_draft_driver(target.get(), head.get());
        finish();
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--borrowed-tables-only") {
        test_borrowed_tables(bare_path, doubled_path, head.get(), target.get());
        finish();
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--invalid-chain-only") {
        test_invalid_chain(head.get(), true);
        finish();
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--head-only") {
        {
            auto ctx = make_context(head.get(), false);
            decode(ctx.get(), {5}, initial_hidden(), 0);
        }
        fprintf(stderr, "PASS canonical head decode\n");
        finish();
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--mixer-only") {
        {
            auto a = make_context(target.get(), false);
            auto b = make_context(changed.get(), false);
            require(close(decode(a.get(), {5}, initial_hidden(), 0).logits,
                          decode(b.get(), {5}, initial_hidden(), 0).logits), "MTP logits independent of trunk mixer");
        }
        fprintf(stderr, "PASS independent draft mixer\n");
        finish();
        return 0;
    }
    test_batch_validation(head.get(), dir);
    test_empty_recurrent_memory(target.get());
    for (int omitted = 0; omitted < 4; ++omitted) {
        const std::string bad = (dir / "missing.gguf").string();
        write_fixture(bad, true, omitted);
        require(!load_model(bad), "missing or partial head mixer rejected");
    }
    const std::string output_only = (dir / "output-only.gguf").string();
    write_fixture(output_only, false, 3);
    require(!load_model(output_only), "trunk mixer does not substitute for missing draft mixer");
    require(bool(load_model(output_only, false)), "ordinary loading does not require unused draft mixers");
    for (const char * missing : {"blk.0.hc_attn_norm.weight", "blk.1.attn_q.weight"}) {
        const std::string holed = (dir / "holed-trunk.gguf").string();
        write_fixture(holed, false, -1, 1.0f, true, 1.0f, n_vocab, missing);
        require(!load_model(holed) && !load_model(holed, false), "a trunk that lacks a tensor is rejected, not run as a head");
    }
    test_separate_head_target(output_only, head_path, target.get(), head.get());
    require(llama_model_supports_mtp_chain(head.get()), "Qwen4Exp advertises implemented chain support");
    test_ordinary_context(head.get(), target.get());
    auto head_without_mtp = load_model(head_path, false);
    require(bool(head_without_mtp), "MTP-only file loads with MTP weights disabled");
    test_ordinary_context(head_without_mtp.get(), target.get());
    head_without_mtp.reset();
    auto bare = load_model(bare_path, false);
    require(bool(bare), "head without the shared embedding loads");
    {
        // A parent context satisfies the context-level ctx_other requirement, so the
        // graph itself must report that nothing is left to run.
        auto bare_params = llama_context_default_params();
        bare_params.n_ctx = 128;
        bare_params.n_threads = bare_params.n_threads_batch = 1;
        llama_context_ptr parent(llama_init_from_model(target.get(), bare_params));
        require(bool(parent), "parent context for the embedding-less head");
        bare_params.ctx_other = parent.get();
        require(!llama_context_ptr(llama_init_from_model(bare.get(), bare_params)),
                "ordinary context reports a head with nothing to run");
    }
    bare.reset();
    test_borrowed_tables(bare_path, doubled_path, head.get(), target.get());
    for (bool flash : {false, true}) {
        if (q8_kv && !flash) { continue; }
        auto a = make_context(head.get(), flash);
        auto b = make_context(target.get(), flash);
        auto c = make_context(changed.get(), flash);
        const auto ah = decode(a.get(), {5}, initial_hidden(), 0);
        const auto bh = decode(b.get(), {5}, initial_hidden(), 0);
        const auto ch = decode(c.get(), {5}, initial_hidden(), 0);
        require(close(ah.logits, bh.logits) && close(bh.logits, ch.logits), "MTP logits independent of trunk mixer");
        test_invalid_chain(head.get(), flash);
        test_on_device_state(head.get(), flash);
        test_checkpoint_placement(head.get(), head_host.get(), flash);
        test_masked_catchup(head.get(), flash);
        for (int depth : {1, 3, 4}) {
            for (int catchup : {0, 2}) {
                for (bool masked : {true, false}) { test_chain(head.get(), flash, depth, catchup, masked); }
            }
        }
        test_chain(head.get(), flash, 4, 2, true, true);
        test_chain(head.get(), flash, 32, 0, true);
        test_chain_export_is_output(head.get(), flash);
    }
    test_driver(target.get(), head.get());
    test_driver_failed_draft(target.get(), head.get());
    test_ordinary_draft_driver(target.get(), head.get());
    test_chain_metadata(head.get());
    test_chain_reserve_growth(head.get());
    test_catchup_reservation(head.get());
    test_fit(target_path, head_path);
    test_fit_ordinary_head(target_path, head_path);
    test_fit_borrowed_head(target_path, bare_path);
    finish();
    fprintf(stderr, "PASS Qwen4Exp MTP regression suite\n");
    return 0;
}
