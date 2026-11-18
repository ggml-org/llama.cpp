// GigaChat4 context cases
//
#include "test-case.h"

#include "test-util.h"

#include "common.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct case_nextn : ctx_test {
    const char * name() const override { return "nextn"; }

    const char * description() const override {
        return "check that the MTP input is the normed state the LM heads read, compare MTP prefill with token-by-token decode";
    }

    std::string skip_reason(model_run & mr) const override {
        char arch[64] = {};
        llama_model_meta_val_str(mr.model, "general.architecture", arch, sizeof(arch));
        if (strcmp(arch, "gigachat4") != 0) {
            return "not GigaChat4";
        }
        return llama_model_n_layer_nextn(mr.model) == 0 ? "no MTP layers" : "";
    }

    case_result run(model_run & mr) const override {
        const int n_tokens = 8;
        CTX_REQUIRE(mr.tokens.size() >= n_tokens, "need at least eight tokens");

        // the runner loads the model with the options of the run, this case needs the MTP layers and loads its own copy
        llama_model_params mparams = common_model_params_to_llama(mr.params);
        mparams.load_mtp = true;
        llama_model_ptr model(llama_model_load_from_file(mr.params.model.path.c_str(), mparams));
        CTX_REQUIRE(model, "failed to load the model with its MTP layers");

        const int n_embd = llama_model_n_embd_out(model.get());

        struct target_capture {
            std::vector<std::vector<float>> l_out;
            std::vector<float>              result_norm;
        } captured;
        captured.l_out.resize(llama_model_n_layer(model.get()));

        auto cparams = common_context_params_to_llama(mr.params);
        cparams.n_batch = cparams.n_ubatch = n_tokens;
        cparams.n_seq_max = 1;
        cparams.cb_eval_user_data = &captured;
        cparams.cb_eval = [](ggml_tensor * t, bool ask, void * user_data) {
            auto & data = *static_cast<target_capture *>(user_data);
            std::vector<float> * output = nullptr;
            int il = -1;
            char suffix;
            if (sscanf(t->name, "l_out-%d%c", &il, &suffix) == 1 && il >= 0 && (size_t) il < data.l_out.size()) {
                output = &data.l_out[il];
            } else if (strcmp(t->name, "result_norm") == 0) {
                output = &data.result_norm;
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
        llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
        CTX_REQUIRE(ctx, "failed to create target context");

        // GigaChat4 feeds the MTP block the normed state, which the LM head reads for every token of this pass
        common_batch batch_all(ctx.get());
        for (int i = 0; i < n_tokens; ++i) {
            batch_all.add(mr.tokens[i], i, 0, true);
        }
        llama_set_embeddings_nextn(ctx.get(), false, false);
        CTX_REQUIRE(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch_all.get()) == 0, "target decode of all outputs failed");
        const std::vector<float> normed = captured.result_norm;
        CTX_REQUIRE(normed.size() == (size_t) n_tokens * n_embd, "wrong normed state row count");

        common_batch batch(ctx.get());
        for (int i = 0; i < n_tokens; ++i) {
            batch.add(mr.tokens[i], i, 0, i == n_tokens - 1);
        }

        std::vector<float> hidden;
        for (int mode = 0; mode < 3; ++mode) {
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            llama_set_embeddings_nextn(ctx.get(), mode != 0, mode == 1);
            for (auto & layer : captured.l_out) {
                layer.clear();
            }
            CTX_REQUIRE(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "target decode failed");
            for (size_t il = 0; il < captured.l_out.size(); ++il) {
                CTX_REQUIRE(!captured.l_out[il].empty(), "missing l_out-" + std::to_string(il));
                for (float value : captured.l_out[il]) {
                    CTX_REQUIRE(std::isfinite(value), "non-finite l_out-" + std::to_string(il));
                }
            }
            const size_t n_rows = mode == 2 ? n_tokens : 1;
            CTX_REQUIRE(captured.l_out.back().size() == n_rows * n_embd, "wrong final layer row count");
            if (mode != 0) {
                const float * h = llama_get_embeddings_nextn(ctx.get());
                CTX_REQUIRE(h, "missing nextn embeddings");
                const float * expected = normed.data() + (n_tokens - n_rows) * n_embd;
                CTX_REQUIRE(compare_logits(expected, h, n_rows * n_embd).nmse() < NMSE_THRESHOLD, "nextn embeddings are not the normed state");
                hidden.assign(h, h + n_rows * n_embd);
            }
        }

        // the state the MTP head reads, which the MTP block also passes on to the next draft step
        std::vector<float> head_input;
        cparams.cb_eval_user_data = &head_input;
        cparams.cb_eval = [](ggml_tensor * t, bool ask, void * user_data) {
            if (strcmp(t->name, "mtp_shared_head_norm") != 0) {
                return !ask;
            }
            if (ask) {
                return true;
            }
            if (t->type != GGML_TYPE_F32) {
                return false;
            }
            auto & output = *static_cast<std::vector<float> *>(user_data);
            output.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, output.data(), 0, output.size() * sizeof(float));
            return true;
        };
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cparams.ctx_other = ctx.get();
        llama_context_ptr ctx_mtp(llama_init_from_model(model.get(), cparams));
        CTX_REQUIRE(ctx_mtp, "failed to create MTP context");

        const std::vector<float> zero(n_embd, 0.0f);
        common_batch draft(ctx_mtp.get());
        for (bool masked : { true, false }) {
            llama_set_embeddings_nextn(ctx_mtp.get(), true, masked);
            for (int il = 0; il < llama_model_n_layer_nextn(model.get()); ++il) {
                llama_set_nextn_layer_offset(ctx_mtp.get(), il);
                llama_memory_clear(llama_get_memory(ctx_mtp.get()), true);
                draft.clear();
                for (int i = 0; i < n_tokens; ++i) {
                    const int idx = draft.add(mr.tokens[i], i, 0, i == n_tokens - 1);
                    const float * h = i == 0 ? zero.data() : hidden.data() + (size_t) (i - 1) * n_embd;
                    CTX_REQUIRE(draft.set_embd_state(idx, { h, 1, (size_t) n_embd }), "failed to set MTP hidden state");
                }
                head_input.clear();
                CTX_REQUIRE(llama_process(ctx_mtp.get(), LLAMA_PROCESS_TYPE_DECODE, draft.get()) == 0, "MTP prefill failed");
                std::vector<float> expected;
                CTX_REQUIRE(get_current_logits(ctx_mtp.get(), expected), "missing MTP prefill logits");
                const float * h = llama_get_embeddings_nextn(ctx_mtp.get());
                CTX_REQUIRE(h, "missing MTP output embeddings");
                const float * h_last = h + (masked ? 0 : (n_tokens - 1) * n_embd);
                CTX_REQUIRE(head_input.size() == (size_t) n_embd, "missing the MTP head input");
                CTX_REQUIRE(compare_logits(head_input.data(), h_last, n_embd).diff_max == 0, "MTP output embeddings are not the state its head reads");
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

std::vector<ctx_test_ptr> make_gigachat4_cases() {
    std::vector<ctx_test_ptr> cases;

    cases.push_back(std::make_unique<case_nextn>());

    return cases;
}
