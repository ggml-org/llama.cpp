// GigaChat context cases
//
#include "test-case.h"

#include "test-util.h"

#include "common.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "../../src/llama-ext.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct case_nextn : ctx_test {
    const char * name() const override { return "nextn"; }

    const char * description() const override {
        return "capture GigaChat layer outputs and nextn embeddings, compare MTP prefill with token-by-token decode";
    }

    bool needs_mtp(const std::string & model_path) const override {
        gguf_context_ptr metadata(gguf_init_from_file(model_path.c_str(), { true, nullptr }));
        if (!metadata) {
            return false;
        }
        const int64_t key = gguf_find_key(metadata.get(), "general.architecture");
        return key >= 0 && gguf_get_kv_type(metadata.get(), key) == GGUF_TYPE_STRING &&
            strcmp(gguf_get_val_str(metadata.get(), key), "gigachat35") == 0;
    }

    std::string skip_reason(model_run & mr) const override {
        char arch[64] = {};
        llama_model_meta_val_str(mr.model, "general.architecture", arch, sizeof(arch));
        if (strcmp(arch, "gigachat35") != 0) {
            return "not GigaChat35";
        }
        return llama_model_n_layer_nextn(mr.model) == 0 ? "no MTP layers" : "";
    }

    case_result run(model_run & mr) const override {
        const int n_tokens = 8;
        const int n_embd   = llama_model_n_embd_out(mr.model);
        CTX_REQUIRE(mr.tokens.size() >= n_tokens, "need at least eight tokens");

        std::vector<std::vector<float>> captured(llama_model_n_layer(mr.model));

        auto cparams = common_context_params_to_llama(mr.params);
        cparams.n_batch = cparams.n_ubatch = n_tokens;
        cparams.n_seq_max = 1;
        cparams.cb_eval_user_data = &captured;
        cparams.cb_eval = [](ggml_tensor * t, bool ask, void * user_data) {
            auto & data = *static_cast<std::vector<std::vector<float>> *>(user_data);
            std::vector<float> * output = nullptr;
            int il = -1;
            char suffix;
            if (sscanf(t->name, "l_out-%d%c", &il, &suffix) == 1 && il >= 0 && (size_t) il < data.size()) {
                output = &data[il];
            }
            if (ask) {
                return output != nullptr;
            }
            if (output) {
                if (t->type != GGML_TYPE_F32) {
                    return false;
                }
                output->resize(ggml_nelements(t));
                ggml_backend_tensor_get(t, output->data(), 0, output->size() * sizeof(float));
            }
            return true;
        };
        llama_context_ptr ctx(llama_init_from_model(mr.model, cparams));
        CTX_REQUIRE(ctx, "failed to create target context");

        common_batch batch(ctx.get());
        for (int i = 0; i < n_tokens; ++i) {
            batch.add(mr.tokens[i], i, 0, i == n_tokens - 1);
        }

        std::vector<float> hidden;
        for (int mode = 0; mode < 3; ++mode) {
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            llama_set_embeddings_nextn(ctx.get(), mode != 0, mode == 1);
            for (auto & layer : captured) {
                layer.clear();
            }
            CTX_REQUIRE(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "target decode failed");
            for (size_t il = 0; il < captured.size(); ++il) {
                CTX_REQUIRE(!captured[il].empty(), "missing l_out-" + std::to_string(il));
                for (float value : captured[il]) {
                    CTX_REQUIRE(std::isfinite(value), "non-finite l_out-" + std::to_string(il));
                }
            }
            const size_t n_rows = mode == 2 ? n_tokens : 1;
            CTX_REQUIRE(captured.back().size() == n_rows * n_embd, "wrong final layer row count");
            if (mode != 0) {
                const float * h = llama_get_embeddings_nextn(ctx.get());
                CTX_REQUIRE(h, "missing nextn embeddings");
                const auto & expected = captured.back();
                CTX_REQUIRE(compare_logits(h, expected.data(), expected.size()).diff_max == 0, "nextn embeddings differ from the captured tensor");
                hidden.assign(h, h + expected.size());
            }
        }

        cparams.cb_eval = nullptr;
        cparams.cb_eval_user_data = nullptr;
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cparams.ctx_other = ctx.get();
        llama_context_ptr ctx_mtp(llama_init_from_model(mr.model, cparams));
        CTX_REQUIRE(ctx_mtp, "failed to create MTP context");

        const std::vector<float> zero(n_embd, 0.0f);
        common_batch draft(ctx_mtp.get());
        for (bool masked : { true, false }) {
            llama_set_embeddings_nextn(ctx_mtp.get(), true, masked);
            for (int il = 0; il < llama_model_n_layer_nextn(mr.model); ++il) {
                llama_set_nextn_layer_offset(ctx_mtp.get(), il);
                llama_memory_clear(llama_get_memory(ctx_mtp.get()), true);
                draft.clear();
                for (int i = 0; i < n_tokens; ++i) {
                    const int idx = draft.add(mr.tokens[i], i, 0, i == n_tokens - 1);
                    const float * h = i == 0 ? zero.data() : hidden.data() + (size_t) (i - 1) * n_embd;
                    CTX_REQUIRE(draft.set_embd_state(idx, { h, 1, (size_t) n_embd }), "failed to set MTP hidden state");
                }
                CTX_REQUIRE(llama_process(ctx_mtp.get(), LLAMA_PROCESS_TYPE_DECODE, draft.get()) == 0, "MTP prefill failed");
                std::vector<float> expected;
                CTX_REQUIRE(get_current_logits(ctx_mtp.get(), expected), "missing MTP prefill logits");
                const float * h = llama_get_embeddings_nextn(ctx_mtp.get());
                CTX_REQUIRE(h, "missing MTP output embeddings");
                const float * h_last = h + (masked ? 0 : (n_tokens - 1) * n_embd);
                const std::vector<float> expected_hidden(h_last, h_last + n_embd);

                llama_memory_clear(llama_get_memory(ctx_mtp.get()), true);
                for (int i = 0; i < n_tokens; ++i) {
                    draft.clear();
                    const int idx = draft.add(mr.tokens[i], i, 0, true);
                    const float * h_in = i == 0 ? zero.data() : hidden.data() + (size_t) (i - 1) * n_embd;
                    CTX_REQUIRE(draft.set_embd_state(idx, { h_in, 1, (size_t) n_embd }), "failed to set MTP hidden state");
                    CTX_REQUIRE(llama_process(ctx_mtp.get(), LLAMA_PROCESS_TYPE_DECODE, draft.get()) == 0, "MTP decode failed");
                }
                std::vector<float> actual;
                CTX_REQUIRE(get_current_logits(ctx_mtp.get(), actual), "missing MTP decode logits");
                CTX_REQUIRE(actual.size() == expected.size(), "MTP logit count changed");
                CTX_REQUIRE(compare_logits(expected.data(), actual.data(), expected.size()).nmse() < NMSE_THRESHOLD, "MTP prefill and decode logits differ");
                h = llama_get_embeddings_nextn(ctx_mtp.get());
                CTX_REQUIRE(h, "missing MTP decode embeddings");
                CTX_REQUIRE(compare_logits(expected_hidden.data(), h, n_embd).nmse() < NMSE_THRESHOLD, "MTP prefill and decode embeddings differ");

                llama_memory_clear(llama_get_memory(ctx_mtp.get()), true);
                draft.clear();
                for (int i = 0; i < n_tokens; ++i) {
                    const int idx = draft.add(mr.tokens[i], i, 0, i == n_tokens - 1);
                    CTX_REQUIRE(draft.set_embd_state(idx, { zero.data(), 1, (size_t) n_embd }), "failed to clear MTP input embeddings");
                }
                CTX_REQUIRE(llama_process(ctx_mtp.get(), LLAMA_PROCESS_TYPE_DECODE, draft.get()) == 0, "MTP decode with zero embeddings failed");
                CTX_REQUIRE(get_current_logits(ctx_mtp.get(), actual), "missing MTP logits with zero embeddings");
                const auto diff = compare_logits(expected.data(), actual.data(), expected.size());
                CTX_REQUIRE(diff.n_nan == 0 && diff.diff_max > 0, "MTP logits did not respond to input embeddings");
            }
        }
        return case_result::pass();
    }
};

std::vector<ctx_test_ptr> make_gigachat_cases() {
    std::vector<ctx_test_ptr> cases;

    cases.push_back(std::make_unique<case_nextn>());

    return cases;
}
