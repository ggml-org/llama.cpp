// a failed state restore must not leave data behind that a later decode can read
// each case corrupts a saved state, checks that the restore fails, and compares a decode on another sequence against a clean run

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

static std::vector<llama_token> make_tokens(size_t n, int base) {
    std::vector<llama_token> tokens(n);
    for (size_t i = 0; i < n; ++i) {
        tokens[i] = base + (int) ((i*37) % 100);
    }
    return tokens;
}

static bool decode(llama_context * ctx, const std::vector<llama_token> & tokens, llama_seq_id seq_id, std::vector<float> * logits_out) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], (llama_pos) i, {seq_id}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (ret != 0) {
        fprintf(stderr, "%s : llama_decode failed (%d)\n", __func__, ret);
        return false;
    }

    if (logits_out) {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        const float * logits = llama_get_logits_ith(ctx, -1);
        logits_out->assign(logits, logits + n_vocab);
    }

    return true;
}

// overwrite the tensor data with 0xff bytes (NaN when read as f16/f32)
// the tail is kept so that the restore fails after the corrupted data has been read
static void corrupt_state(std::vector<uint8_t> & data) {
    GGML_ASSERT(data.size() >= 3*4096);
    std::fill(data.begin() + 4096, data.end() - data.size()/4, 0xff);
}

int main(int argc, char ** argv) {
    common_params params;

    params.kv_unified = true;
    params.n_parallel = 4;
    params.n_ctx = 256;

    // without flash attention, NaN values in masked cells propagate to the logits and make leftover data detectable
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_context * ctx = llama_init->context();
    if (ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx);

    // the registered tests share a working directory, so the state file is named after the model
    const std::string path = "test-state-restore-failure." + params.model.path.substr(params.model.path.find_last_of("/\\") + 1) + ".tmp.bin";

    const std::vector<llama_token> tokens_save   = make_tokens(24, 5);
    const std::vector<llama_token> tokens_verify = make_tokens( 8, 7);

    llama_memory_clear(mem, true);

    std::vector<float> baseline;
    if (!decode(ctx, tokens_verify, 1, &baseline)) {
        return 1;
    }

    // each case restores a corrupted copy of the state of seq 0 and returns true if the restore failed
    const std::vector<std::pair<const char *, std::function<bool()>>> cases = {
        { "buffer", [&]() {
            std::vector<uint8_t> state(llama_state_seq_get_size(ctx, 0));
            GGML_ASSERT(llama_state_seq_get_data(ctx, state.data(), state.size(), 0) == state.size());
            llama_memory_seq_rm(mem, 0, -1, -1);

            corrupt_state(state);
            return llama_state_seq_set_data(ctx, state.data(), state.size(), 0) == 0;
        }},
        { "file", [&]() {
            GGML_ASSERT(llama_state_seq_save_file(ctx, path.c_str(), 0, tokens_save.data(), tokens_save.size()) > 0);
            llama_memory_seq_rm(mem, 0, -1, -1);

            std::vector<uint8_t> data;
            {
                std::ifstream f(path, std::ios::binary);
                data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            }
            corrupt_state(data);
            {
                std::ofstream f(path, std::ios::binary);
                f.write((const char *) data.data(), data.size());
            }

            std::vector<llama_token> tokens_out(tokens_save.size());
            size_t n_token_count = 0;
            const size_t nread = llama_state_seq_load_file(ctx, path.c_str(), 0, tokens_out.data(), tokens_out.size(), &n_token_count);
            std::remove(path.c_str());
            return nread == 0;
        }},
        { "on device", [&]() {
            const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;

            std::vector<uint8_t> state(llama_state_seq_get_size_ext(ctx, 0, flags));
            GGML_ASSERT(llama_state_seq_get_data_ext(ctx, state.data(), state.size(), 0, flags) == state.size());
            llama_memory_seq_rm(mem, 0, -1, -1);

            // the tensor data of an on-device state stays in the memory buffers, so only the metadata can be corrupted
            std::fill(state.end() - state.size()/4, state.end(), 0xff);
            return llama_state_seq_set_data_ext(ctx, state.data(), state.size(), 0, flags) == 0;
        }},
        { "whole context", [&]() {
            std::vector<uint8_t> state(llama_state_get_size(ctx));
            GGML_ASSERT(llama_state_get_data(ctx, state.data(), state.size()) == state.size());
            llama_memory_clear(mem, true);

            corrupt_state(state);
            return llama_state_set_data(ctx, state.data(), state.size()) == 0;
        }},
    };

    for (const auto & [name, restore_failed] : cases) {
        llama_memory_clear(mem, true);

        if (!decode(ctx, tokens_save, 0, nullptr)) {
            return 1;
        }

        if (!restore_failed()) {
            fprintf(stderr, "%s : %s: restoring a corrupted state did not fail\n", __func__, name);
            return 1;
        }

        if (llama_memory_seq_pos_max(mem, 0) != -1) {
            fprintf(stderr, "%s : %s: sequence not empty after failed restore\n", __func__, name);
            return 1;
        }

        std::vector<float> logits;
        if (!decode(ctx, tokens_verify, 1, &logits)) {
            return 1;
        }

        float diff_max = 0.0f;
        size_t n_nan = 0;
        for (size_t i = 0; i < logits.size(); ++i) {
            if (std::isnan(logits[i]) || std::isnan(baseline[i])) {
                n_nan++;
            } else {
                diff_max = std::max(diff_max, std::fabs(logits[i] - baseline[i]));
            }
        }

        if (n_nan > 0 || diff_max > 1e-6f) {
            fprintf(stderr, "%s : %s: FAILED - logits changed after failed restore (max diff = %g, nan = %zu)\n", __func__, name, diff_max, n_nan);
            return 1;
        }

        fprintf(stderr, "%s : %s: logits match (max diff = %g)\n", __func__, name, diff_max);
    }

    return 0;
}
