// test-ane-prefill-chunked
//
// Smoke test for the ANE fast-start chunked path. Confirms that a prompt
// longer than the largest declared bucket can be processed end-to-end:
// the ANE slab runs over the first max_bucket tokens, the imported K/V
// drives layer 0 of the continuation, and the remaining tail tokens flow
// through ordinary llama_decode.  The test is deliberately light on
// numerical parity (we already have it at the slab level via the bucket
// test) because the all-Metal baseline on CPU is slow enough to mask the
// signal we actually care about — the chunked path returning a clean
// decode result.
//
// Usage: test-ane-prefill-chunked MODEL_GGUF [N_TOKENS]

#include "ane-mtp.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "llama.h"

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL_GGUF [N_TOKENS]\n", argv[0]);
        return 2;
    }
    const std::string gguf_path = argv[1];
    const int n_tokens = argc == 3 ? std::atoi(argv[2]) : 540;
    if (n_tokens < 1) {
        std::fprintf(stderr, "N_TOKENS must be >= 1\n");
        return 2;
    }

    common_ane_prefill_manifest manifest;
    if (!common_ane_prefill_manifest_load(gguf_path, &manifest)) {
        std::fprintf(stderr, "no Tessera ANE prefill manifest in %s\n", gguf_path.c_str());
        return 1;
    }
    const uint32_t max_bucket = manifest.sequence_buckets.empty() ? 0
        : manifest.sequence_buckets.back();
    if (max_bucket == 0) {
        std::fprintf(stderr, "manifest declares no sequence buckets\n");
        return 1;
    }
    if ((uint32_t) n_tokens <= max_bucket) {
        std::fprintf(stderr, "N_TOKENS=%d must exceed max_bucket=%u for chunked path\n",
                n_tokens, max_bucket);
        return 2;
    }

    auto program = common_ane_prefill_program_load(gguf_path, 0);
    if (!program || !common_ane_mtp_program_is_warm(program)) {
        std::fprintf(stderr, "failed to warm Tessera ANE prefill program\n");
        return 1;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 999;
    llama_model * model = llama_load_model_from_file(gguf_path.c_str(), model_params);
    if (!model) {
        std::fprintf(stderr, "failed to load model from %s\n", gguf_path.c_str());
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = (uint32_t) n_tokens + 32;
    ctx_params.n_batch = (uint32_t) n_tokens;
    ctx_params.n_ubatch = (uint32_t) n_tokens;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_free_model(model);
        return 1;
    }

    std::vector<llama_token> tokens((size_t) n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        tokens[(size_t) i] = (llama_token) (1 + (i * 37) % 2000);
    }

    llama_batch batch = llama_batch_init((uint32_t) n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; ++i) {
        common_batch_add(batch, tokens[(size_t) i], (llama_pos) i,
                std::vector<llama_seq_id>{0}, false);
    }

    int32_t ane_result = -1;
    const bool ane_used = common_ane_prefill_decode_chunked(
            program, manifest, ctx, batch, &ane_result);
    if (!ane_used) {
        std::fprintf(stderr, "ANE chunked path refused; the test setup is wrong\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_free_model(model);
        return 1;
    }
    if (ane_result != 0) {
        std::fprintf(stderr, "ANE chunked decode failed: %d\n", ane_result);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_free_model(model);
        return 1;
    }
    std::printf("ANE fast-start: n_tokens=%d max_bucket=%u ane_used=%d decode_result=%d\n",
            n_tokens, max_bucket, ane_used ? 1 : 0, ane_result);

    // Confirm the K/V cache actually grew to cover the full prompt by
    // requesting a single token decode — if the K/V import was dropped,
    // a normal decode would still succeed on this short sequence, so this
    // is a weak signal.  A stronger test compares logits against the
    // all-Metal path; that lives in the bucket parity harness instead.
    llama_batch next = llama_batch_init(1, 0, 1);
    common_batch_add(next, (llama_token) 7, (llama_pos) n_tokens,
            std::vector<llama_seq_id>{0}, true);
    const int32_t next_result = llama_decode(ctx, next);
    std::printf("post-ANE decode: result=%d logits_ptr=%p\n",
            next_result, (const void *) llama_get_logits(ctx));
    llama_batch_free(next);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    if (next_result != 0) {
        std::fprintf(stderr, "post-ANE decode failed: %d\n", next_result);
        return 1;
    }
    if (llama_get_logits(ctx) == nullptr) {
        std::fprintf(stderr, "no logits returned after post-ANE decode\n");
        return 1;
    }
    return 0;
}
