// integration test for mmap-backed KV cache
// validates the full code path: param plumbing, constructor mmap branch,
// tensor allocation, decode, token parity vs heap, and file persistence

#include "llama.h"
#include "get-model.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <sys/stat.h>
#include <unistd.h>
#define KV_MMAP_TEST_SUPPORTED
#endif // defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))

static llama_token greedy_sample(llama_context * ctx, int32_t idx) {
    const float * logits = llama_get_logits_ith(ctx, idx);
    if (!logits) {
        return -1;
    }

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    llama_token best = 0;

    for (int32_t i = 1; i < n_vocab; i++) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

// run a short generation and return the token sequence
static std::vector<llama_token> run_generation(
        llama_model * model,
        const char * kv_mmap_path,
        int32_t n_generate) {

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = 128;
    cparams.n_batch      = 64;
    cparams.offload_kqv  = false;
    cparams.kv_mmap_path = kv_mmap_path;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "FAIL: llama_init_from_model (mmap_path=%s)\n",
                kv_mmap_path ? kv_mmap_path : "NULL");
        return {};
    }

    // tokenize a short prompt
    const char * prompt = "Once upon a time";
    const int32_t max_tokens = 64;
    std::vector<llama_token> prompt_tokens(max_tokens);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t n_prompt = llama_tokenize(vocab, prompt, (int32_t) strlen(prompt),
                                       prompt_tokens.data(), max_tokens, true, false);
    if (n_prompt < 0) {
        fprintf(stderr, "FAIL: tokenize\n");
        llama_free(ctx);
        return {};
    }
    prompt_tokens.resize(n_prompt);

    // decode prompt
    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int32_t i = 0; i < n_prompt; i++) {
        batch.token[i]    = prompt_tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = (i == n_prompt - 1) ? 1 : 0;
    }
    batch.n_tokens = n_prompt;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "FAIL: decode prompt\n");
        llama_batch_free(batch);
        llama_free(ctx);
        return {};
    }
    llama_batch_free(batch);

    // generate tokens autoregressively
    std::vector<llama_token> result;
    llama_token tok = greedy_sample(ctx, n_prompt - 1);
    result.push_back(tok);

    int32_t cur_pos = n_prompt;
    for (int32_t g = 1; g < n_generate; g++) {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0]    = tok;
        b.pos[0]      = cur_pos;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0]   = 1;
        b.n_tokens    = 1;

        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "FAIL: decode step %d\n", g);
            llama_batch_free(b);
            break;
        }
        llama_batch_free(b);

        tok = greedy_sample(ctx, 0);
        result.push_back(tok);
        cur_pos++;
    }

    llama_free(ctx);
    return result;
}

int main(int argc, char ** argv) {
#ifndef KV_MMAP_TEST_SUPPORTED
    fprintf(stderr, "mmap KV cache not supported on this platform, skipping\n");
    GGML_UNUSED(argc);
    GGML_UNUSED(argv);
    return 0;
#else

    auto * model_path = get_model_or_exit(argc, argv);
    const std::string mmap_path = "/tmp/test-kv-cache-mmap-model.bin";
    const std::string meta_path = mmap_path + ".meta";
    const int32_t n_generate = 10;
    int failures = 0;

    unlink(mmap_path.c_str());
    unlink(meta_path.c_str());

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "FAIL: could not load model %s\n", model_path);
        return 1;
    }

    printf("model: %s\n", model_path);
    printf("mmap:  %s\n", mmap_path.c_str());
    printf("tokens: %d (greedy)\n\n", n_generate);

    // test 1: heap baseline
    printf("[1] heap baseline... ");
    std::vector<llama_token> heap_tokens = run_generation(model, nullptr, n_generate);
    if ((int32_t) heap_tokens.size() != n_generate) {
        printf("FAIL (got %zu tokens)\n", heap_tokens.size());
        failures++;
    } else {
        printf("ok (%zu tokens)\n", heap_tokens.size());
    }

    // test 2: mmap generation
    printf("[2] mmap generation... ");
    std::vector<llama_token> mmap_tokens = run_generation(model, mmap_path.c_str(), n_generate);
    if ((int32_t) mmap_tokens.size() != n_generate) {
        printf("FAIL (got %zu tokens)\n", mmap_tokens.size());
        failures++;
    } else {
        printf("ok (%zu tokens)\n", mmap_tokens.size());
    }

    // test 3: token-for-token parity
    printf("[3] parity check... ");
    if (heap_tokens.size() != mmap_tokens.size()) {
        printf("FAIL (different lengths: %zu vs %zu)\n", heap_tokens.size(), mmap_tokens.size());
        failures++;
    } else {
        bool match = true;
        for (size_t i = 0; i < heap_tokens.size(); i++) {
            if (heap_tokens[i] != mmap_tokens[i]) {
                printf("FAIL (diverge at token %zu: heap=%d mmap=%d)\n",
                        i, heap_tokens[i], mmap_tokens[i]);
                match = false;
                failures++;
                break;
            }
        }
        if (match) {
            printf("ok (%zu/%zu identical)\n", heap_tokens.size(), heap_tokens.size());
        }
    }

    // test 4: mmap file exists and has nonzero size
    printf("[4] mmap file check... ");
    {
        struct stat sb;
        if (stat(mmap_path.c_str(), &sb) != 0) {
            printf("FAIL (file does not exist)\n");
            failures++;
        } else if (sb.st_size == 0) {
            printf("FAIL (file is empty)\n");
            failures++;
        } else {
            printf("ok (%lld bytes)\n", (long long) sb.st_size);
        }
    }

    // test 5: metadata sidecar exists
    printf("[5] metadata sidecar... ");
    {
        struct stat sb;
        if (stat(meta_path.c_str(), &sb) != 0) {
            printf("FAIL (sidecar does not exist)\n");
            failures++;
        } else if (sb.st_size == 0) {
            printf("FAIL (sidecar is empty)\n");
            failures++;
        } else {
            printf("ok (%lld bytes)\n", (long long) sb.st_size);
        }
    }

    // test 6: state serialization
    printf("[6] state serialization... ");
    {
        // create a context with mmap, decode something, then try to save state
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx        = 128;
        cp.n_batch      = 64;
        cp.offload_kqv  = false;
        cp.kv_mmap_path = mmap_path.c_str();

        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            printf("FAIL (could not create context)\n");
            failures++;
        } else {
            // decode a short prompt to populate KV cache
            const char * prompt = "Hello world";
            const llama_vocab * vocab = llama_model_get_vocab(model);
            std::vector<llama_token> toks(32);
            int32_t n = llama_tokenize(vocab, prompt, (int32_t) strlen(prompt),
                                       toks.data(), (int32_t) toks.size(), true, false);
            toks.resize(n);

            llama_batch batch = llama_batch_init(n, 0, 1);
            for (int32_t i = 0; i < n; i++) {
                batch.token[i]    = toks[i];
                batch.pos[i]      = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]   = (i == n - 1) ? 1 : 0;
            }
            batch.n_tokens = n;
            llama_decode(ctx, batch);
            llama_batch_free(batch);

            // try to serialize state for seq 0
            size_t state_size = llama_state_seq_get_size(ctx, 0);
            if (state_size == 0) {
                printf("FAIL (state_seq_get_size returned 0)\n");
                failures++;
            } else {
                std::vector<uint8_t> state_buf(state_size);
                size_t written = llama_state_seq_get_data(ctx, state_buf.data(), state_size, 0);
                if (written == 0) {
                    printf("FAIL (state_seq_get_data returned 0)\n");
                    failures++;
                } else {
                    printf("ok (serialized %zu bytes for seq 0)\n", written);
                }
            }
            llama_free(ctx);
        }
    }

    // cleanup
    unlink(mmap_path.c_str());
    unlink(meta_path.c_str());
    llama_model_free(model);
    llama_backend_free();

    printf("\n%s (%d failures)\n", failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED", failures);
    return failures == 0 ? 0 : 1;

#endif // KV_MMAP_TEST_SUPPORTED
}
