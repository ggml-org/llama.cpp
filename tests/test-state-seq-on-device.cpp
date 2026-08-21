#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

static size_t set_on_device(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id dest_seq_id, bool * escaped) {
    *escaped = false;
    try {
        return llama_state_seq_set_data_ext(ctx, src, size, dest_seq_id, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    } catch (const std::exception & err) {
        fprintf(stderr, "%s : exception escaped C API: %s\n", __func__, err.what());
        *escaped = true;
        return 0;
    } catch (...) {
        fprintf(stderr, "%s : exception escaped C API\n", __func__);
        *escaped = true;
        return 0;
    }
}

static bool expect_fail(llama_context * ctx, const uint8_t * src, size_t size, const char * desc) {
    bool escaped = false;
    const size_t nset = set_on_device(ctx, src, size, 0, &escaped);
    if (escaped) {
        fprintf(stderr, "%s : %s\n", __func__, desc);
        return false;
    }
    if (nset != 0) {
        fprintf(stderr, "%s : expected 0 for %s, got %zu\n", __func__, desc, nset);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to load the model\n", __func__);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 256;
    cparams.n_batch   = 64;
    cparams.n_seq_max = 4;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "%s : failed to create the context\n", __func__);
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens;
    for (int i = 0; i < 8; ++i) {
        tokens.push_back(1 + i % (n_vocab - 1));
    }

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
        fprintf(stderr, "%s : failed to decode\n", __func__);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
    std::vector<uint8_t> state(llama_state_seq_get_size_ext(ctx, 0, flags));
    if (state.size() < 8) {
        fprintf(stderr, "%s : on-device state is too small: %zu\n", __func__, state.size());
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_state_seq_get_data_ext(ctx, state.data(), state.size(), 0, flags) != state.size()) {
        fprintf(stderr, "%s : failed to save the state\n", __func__);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    bool escaped = false;
    const size_t nset_ok = set_on_device(ctx, state.data(), state.size(), 0, &escaped);
    if (escaped || nset_ok != state.size()) {
        fprintf(stderr, "%s : valid on-device restore failed: returned %zu, expected %zu\n",
                __func__, nset_ok, state.size());
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    {
        std::vector<uint8_t> buf = state;
        const uint32_t magic_bad = 0xdeadbeef;
        memcpy(buf.data(), &magic_bad, sizeof(magic_bad));
        if (!expect_fail(ctx, buf.data(), buf.size(), "wrong magic")) {
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    }

    if (!expect_fail(ctx, state.data(), 2, "buffer shorter than magic")) {
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (!expect_fail(ctx, state.data(), 6, "truncated seq_id")) {
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    {
        std::vector<uint8_t> buf = state;
        const llama_seq_id seq_id_bad = 3;
        memcpy(buf.data() + sizeof(uint32_t), &seq_id_bad, sizeof(seq_id_bad));
        if (!expect_fail(ctx, buf.data(), buf.size(), "unknown source seq_id")) {
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
