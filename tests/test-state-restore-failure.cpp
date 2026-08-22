// a failed state restore rolls back the cell metadata, but the attention still reads the data of free cells
// each case corrupts a saved state and compares the decodes after the failed restore against a clean run

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static std::vector<llama_token> make_tokens(size_t n, int base) {
    std::vector<llama_token> tokens(n);
    for (size_t i = 0; i < n; ++i) {
        tokens[i] = base + (int) ((i*37) % 100);
    }
    return tokens;
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, llama_seq_id seq_id, llama_pos p0, std::vector<float> * logits_out) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], p0 + (llama_pos) i, {seq_id}, false);
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

static bool logits_match(const std::vector<float> & logits, const std::vector<float> & expected, const char * label) {
    float diff_max = 0.0f;
    size_t n_nan = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        // a NaN in expected would be hidden by std::max(), so both sides are counted here
        if (std::isnan(logits[i]) || std::isnan(expected[i])) {
            n_nan++;
            continue;
        }
        diff_max = std::max(diff_max, std::fabs(logits[i] - expected[i]));
    }

    if (n_nan > 0 || diff_max > 1e-6f) {
        fprintf(stderr, "%s : %s: FAILED - logits changed after failed restore (max diff = %g, nan = %zu)\n",
                __func__, label, diff_max, n_nan);
        fprintf(stderr, "%s : %s: data from the failed restore is influencing the output\n", __func__, label);
        return false;
    }

    fprintf(stderr, "%s : %s: logits match (max diff = %g)\n", __func__, label, diff_max);
    return true;
}

// decode the verification prompt on a cold sequence and compare against the clean baseline
static bool verify_decode(llama_context * ctx, const std::vector<llama_token> & tokens, const std::vector<float> & baseline, const char * label) {
    std::vector<float> logits;
    if (!decode_tokens(ctx, tokens, 1, 0, &logits)) {
        fprintf(stderr, "%s : %s: verification decode failed\n", __func__, label);
        return false;
    }

    return logits_match(logits, baseline, label);
}

static bool read_file(const std::string & path, std::vector<uint8_t> & data, const char * label) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "%s : %s: failed to open the state file\n", __func__, label);
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long fsize = ftell(f);
    if (fsize < 0) {
        fprintf(stderr, "%s : %s: failed to read the state file\n", __func__, label);
        fclose(f);
        return false;
    }
    data.resize(fsize);
    fseek(f, 0, SEEK_SET);
    if (fread(data.data(), 1, data.size(), f) != data.size()) {
        fprintf(stderr, "%s : %s: failed to read the state file\n", __func__, label);
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}

static bool write_file(const std::string & path, const std::vector<uint8_t> & data, const char * label) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f || fwrite(data.data(), 1, data.size(), f) != data.size()) {
        fprintf(stderr, "%s : %s: failed to write the state file\n", __func__, label);
        if (f) {
            fclose(f);
        }
        return false;
    }
    fclose(f);
    return true;
}

// overwrite the K/V payload with 0xff bytes (NaN when interpreted as f16/f32)
// the start and the tail of the state are kept intact so that the restore fails after the corrupted data has been written
static bool corrupt_state(std::vector<uint8_t> & data, const char * label) {
    if (data.size() < 3*4096) {
        fprintf(stderr, "%s : %s: state unexpectedly small (%zu bytes)\n", __func__, label, data.size());
        return false;
    }
    std::fill(data.begin() + 4096, data.end() - data.size()/4, 0xff);
    return true;
}

// removes the state file on every exit path, including the early returns of the cases below
struct scoped_state_file {
    std::string path;

    ~scoped_state_file() {
        std::remove(path.c_str());
    }
};

static int run_cases(llama_context * ctx, const std::string & state_path) {
    const scoped_state_file state_file { state_path };

    llama_memory_t mem = llama_get_memory(ctx);

    const std::vector<llama_token> tokens_save   = make_tokens(24, 5);
    const std::vector<llama_token> tokens_verify = make_tokens( 8, 7);

    // baseline: the verification prompt decoded on a clean context
    llama_memory_clear(mem, true);

    std::vector<float> baseline;
    if (!decode_tokens(ctx, tokens_verify, 1, 0, &baseline)) {
        fprintf(stderr, "%s : failed to decode the baseline\n", __func__);
        return 1;
    }
    fprintf(stderr, "%s : baseline logits computed (%zu tokens)\n", __func__, tokens_verify.size());

    // case 1: per-sequence restore via the buffer API
    {
        llama_memory_clear(mem, true);

        if (!decode_tokens(ctx, tokens_save, 0, 0, nullptr)) {
            return 1;
        }

        std::vector<uint8_t> state(llama_state_seq_get_size(ctx, 0));
        if (llama_state_seq_get_data(ctx, state.data(), state.size(), 0) != state.size()) {
            fprintf(stderr, "%s : buffer API: failed to save the state\n", __func__);
            return 1;
        }

        llama_memory_seq_rm(mem, 0, -1, -1);

        std::vector<uint8_t> corrupt = state;
        if (!corrupt_state(corrupt, "buffer API")) {
            return 1;
        }

        const size_t nset = llama_state_seq_set_data(ctx, corrupt.data(), corrupt.size(), 0);
        if (nset != 0) {
            fprintf(stderr, "%s : buffer API: restoring a corrupted state did not fail (nset = %zu)\n", __func__, nset);
            return 1;
        }
        fprintf(stderr, "%s : buffer API: corrupted restore failed as expected\n", __func__);

        if (llama_memory_seq_pos_max(mem, 0) != -1) {
            fprintf(stderr, "%s : buffer API: sequence not empty after failed restore\n", __func__);
            return 1;
        }

        if (!verify_decode(ctx, tokens_verify, baseline, "buffer API")) {
            return 1;
        }

        // the original state must still be restorable in the same process
        if (llama_state_seq_set_data(ctx, state.data(), state.size(), 0) != state.size()) {
            fprintf(stderr, "%s : buffer API: failed to restore a valid state after the failure\n", __func__);
            return 1;
        }

        if (!decode_tokens(ctx, {tokens_save.back()}, 0, tokens_save.size(), nullptr)) {
            fprintf(stderr, "%s : buffer API: failed to decode with the restored state\n", __func__);
            return 1;
        }
        fprintf(stderr, "%s : buffer API: valid restore and decode succeeded after the failure\n", __func__);
    }

    // case 2: per-sequence restore via the file API
    {
        llama_memory_clear(mem, true);

        if (!decode_tokens(ctx, tokens_save, 0, 0, nullptr)) {
            return 1;
        }

        const std::string & path = state_file.path;

        if (llama_state_seq_save_file(ctx, path.c_str(), 0, tokens_save.data(), tokens_save.size()) == 0) {
            fprintf(stderr, "%s : file API: failed to save the state file\n", __func__);
            return 1;
        }

        llama_memory_seq_rm(mem, 0, -1, -1);

        // corrupt the K/V payload of the state file
        std::vector<uint8_t> data;
        if (!read_file(path, data, "file API")) {
            return 1;
        }

        if (!corrupt_state(data, "file API")) {
            return 1;
        }

        if (!write_file(path, data, "file API")) {
            return 1;
        }

        std::vector<llama_token> tokens_out(tokens_save.size());
        size_t n_token_count = 0;

        const size_t nread = llama_state_seq_load_file(ctx, path.c_str(), 0, tokens_out.data(), tokens_out.size(), &n_token_count);
        std::remove(path.c_str());
        if (nread != 0) {
            fprintf(stderr, "%s : file API: loading a corrupted state file did not fail (nread = %zu)\n", __func__, nread);
            return 1;
        }
        fprintf(stderr, "%s : file API: corrupted restore failed as expected\n", __func__);

        if (llama_memory_seq_pos_max(mem, 0) != -1) {
            fprintf(stderr, "%s : file API: sequence not empty after failed restore\n", __func__);
            return 1;
        }

        if (!verify_decode(ctx, tokens_verify, baseline, "file API")) {
            return 1;
        }
    }

    // case 3: fragmented per-sequence restore next to a surviving sequence
    // the free cells form two blocks smaller than the saved state, which forces the restore into a non-contiguous slot
    {
        const std::vector<llama_token> tokens_a = make_tokens( 16,  5); // removed before the restore
        const std::vector<llama_token> tokens_b = make_tokens(  8,  7); // survives the restore
        const std::vector<llama_token> tokens_c = make_tokens(  8,  9); // removed before the restore
        const std::vector<llama_token> tokens_d = make_tokens(224, 11); // fills the rest of the cache

        const std::string & path = state_file.path;

        std::vector<uint8_t> state;

        std::vector<float> logits_control;
        std::vector<float> logits_test;

        for (int pass = 0; pass < 2; ++pass) {
            llama_memory_clear(mem, true);

            if (pass == 0) {
                // save a state that is larger than any contiguous block of free cells will be
                if (!decode_tokens(ctx, tokens_save, 0, 0, nullptr)) {
                    return 1;
                }
                if (llama_state_seq_save_file(ctx, path.c_str(), 0, tokens_save.data(), tokens_save.size()) == 0) {
                    fprintf(stderr, "%s : fragmented: failed to save the state file\n", __func__);
                    return 1;
                }
                const bool ok = read_file(path, state, "fragmented");
                std::remove(path.c_str());
                if (!ok) {
                    return 1;
                }
                llama_memory_clear(mem, true);
            }

            // fill the cache: [seq 0: 16] [seq 1: 8] [seq 2: 8] [seq 3: 224]
            if (!decode_tokens(ctx, tokens_a, 0, 0, nullptr) ||
                !decode_tokens(ctx, tokens_b, 1, 0, nullptr) ||
                !decode_tokens(ctx, tokens_c, 2, 0, nullptr) ||
                !decode_tokens(ctx, tokens_d, 3, 0, nullptr)) {
                return 1;
            }

            // free two separate blocks of 16 and 8 cells - the 24-cell restore cannot be contiguous
            llama_memory_seq_rm(mem, 0, -1, -1);
            llama_memory_seq_rm(mem, 2, -1, -1);

            if (pass == 1) {
                std::vector<uint8_t> corrupt = state;
                if (!corrupt_state(corrupt, "fragmented")) {
                    return 1;
                }

                if (!write_file(path, corrupt, "fragmented")) {
                    return 1;
                }

                std::vector<llama_token> tokens_out(tokens_save.size());
                size_t n_token_count = 0;

                const size_t nread = llama_state_seq_load_file(ctx, path.c_str(), 0, tokens_out.data(), tokens_out.size(), &n_token_count);
                std::remove(path.c_str());
                if (nread != 0) {
                    fprintf(stderr, "%s : fragmented: loading a corrupted state file did not fail (nread = %zu)\n", __func__, nread);
                    return 1;
                }
                fprintf(stderr, "%s : fragmented: corrupted restore failed as expected\n", __func__);

                if (llama_memory_seq_pos_max(mem, 0) != -1) {
                    fprintf(stderr, "%s : fragmented: sequence not empty after failed restore\n", __func__);
                    return 1;
                }

                if (llama_memory_seq_pos_max(mem, 1) != (llama_pos) tokens_b.size() - 1) {
                    fprintf(stderr, "%s : fragmented: surviving sequence was modified by the failed restore\n", __func__);
                    return 1;
                }
            }

            // continue the surviving sequence
            std::vector<float> & logits = pass == 0 ? logits_control : logits_test;
            if (!decode_tokens(ctx, {tokens_b.back()}, 1, tokens_b.size(), &logits)) {
                fprintf(stderr, "%s : fragmented: failed to decode the surviving sequence\n", __func__);
                return 1;
            }
        }

        if (!logits_match(logits_test, logits_control, "fragmented")) {
            return 1;
        }
    }

    // case 4: per-sequence restore via the buffer API with the state kept in device buffers
    {
        llama_memory_clear(mem, true);

        if (!decode_tokens(ctx, tokens_save, 0, 0, nullptr)) {
            return 1;
        }

        const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;

        std::vector<uint8_t> state(llama_state_seq_get_size_ext(ctx, 0, flags));
        if (llama_state_seq_get_data_ext(ctx, state.data(), state.size(), 0, flags) != state.size()) {
            fprintf(stderr, "%s : device API: failed to save the state\n", __func__);
            return 1;
        }

        llama_memory_seq_rm(mem, 0, -1, -1);

        // the tensor data of an on-device state stays in the memory buffers, so only the metadata can be corrupted
        std::vector<uint8_t> corrupt = state;
        std::fill(corrupt.end() - corrupt.size()/4, corrupt.end(), 0xff);

        const size_t nset = llama_state_seq_set_data_ext(ctx, corrupt.data(), corrupt.size(), 0, flags);
        if (nset != 0) {
            fprintf(stderr, "%s : device API: restoring a corrupted state did not fail (nset = %zu)\n", __func__, nset);
            return 1;
        }
        fprintf(stderr, "%s : device API: corrupted restore failed as expected\n", __func__);

        if (llama_memory_seq_pos_max(mem, 0) != -1) {
            fprintf(stderr, "%s : device API: sequence not empty after failed restore\n", __func__);
            return 1;
        }

        if (!verify_decode(ctx, tokens_verify, baseline, "device API")) {
            return 1;
        }
    }

    // case 5: whole context restore via the buffer API
    //
    // The rollback of a whole context restore clears the memory, but the reader still holds the tensor data it has read.
    {
        llama_memory_clear(mem, true);

        if (!decode_tokens(ctx, tokens_save, 0, 0, nullptr)) {
            return 1;
        }

        std::vector<uint8_t> state(llama_state_get_size(ctx));
        if (llama_state_get_data(ctx, state.data(), state.size()) != state.size()) {
            fprintf(stderr, "%s : whole context: failed to save the state\n", __func__);
            return 1;
        }

        llama_memory_clear(mem, true);

        std::vector<uint8_t> corrupt = state;
        if (!corrupt_state(corrupt, "whole context")) {
            return 1;
        }

        const size_t nset = llama_state_set_data(ctx, corrupt.data(), corrupt.size());
        if (nset != 0) {
            fprintf(stderr, "%s : whole context: restoring a corrupted state did not fail (nset = %zu)\n", __func__, nset);
            return 1;
        }
        fprintf(stderr, "%s : whole context: corrupted restore failed as expected\n", __func__);

        if (!verify_decode(ctx, tokens_verify, baseline, "whole context")) {
            return 1;
        }
    }

    return 0;
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

    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    // the registered tests share a working directory, so the state file is named after the model
    const size_t pos = params.model.path.find_last_of("/\\");
    const std::string state_path = "test-state-restore-failure." + params.model.path.substr(pos + 1) + ".tmp.bin";

    fprintf(stderr, "%s : === flash attention disabled ===\n", __func__);
    if (run_cases(ctx, state_path) != 0) {
        return 1;
    }

    // run the same cases with flash attention enabled to cover the non-transposed layout of the V cache
    // the surviving sequence check detects cleanup of cells that belong to another sequence
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context * ctx_fa = llama_init_from_model(model, common_context_params_to_llama(params));
    if (ctx_fa == nullptr) {
        fprintf(stderr, "%s : failed to init the flash attention context\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s : === flash attention enabled ===\n", __func__);
    const int ret = run_cases(ctx_fa, state_path);
    llama_free(ctx_fa);
    if (ret != 0) {
        return 1;
    }

    fprintf(stderr, "%s : SUCCESS - failed restores did not contaminate the KV cache\n", __func__);

    return 0;
}
