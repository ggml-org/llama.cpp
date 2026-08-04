// s3_evict_test.cpp - correctness probe for llama_memory_seq_evict_oldest.
// Fills KV with a known prompt+decode range, calls the eviction helper, and
// asserts: (1) the surviving range is dense [0, total-keep_recent],
// (2) pos_min/pos_max reflect the shift, (3) re-call is a true no-op.
// NOT part of the S3 diff.
#include "llama.h"
#include "common.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    std::string model_path;
    std::string prompt = "The quick brown fox jumps over the lazy dog. Explain why";
    int ngl = 99;
    int n_predict = 64;
    uint32_t keep_recent = 16;
    {
        int i = 1;
        for (; i < argc; i++) {
            if (!strcmp(argv[i], "-m") && i+1 < argc) model_path = argv[++i];
            else if (!strcmp(argv[i], "-n") && i+1 < argc) n_predict = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-ngl") && i+1 < argc) ngl = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-p") && i+1 < argc) prompt = argv[++i];
            else if (!strcmp(argv[i], "-keep") && i+1 < argc) keep_recent = (uint32_t) atoi(argv[++i]);
        }
        if (model_path.empty()) { fprintf(stderr, "no -m\n"); return 1; }
    }

    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 2; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> toks(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, true);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_prompt + n_predict + 16;
    cp.n_batch = cp.n_ctx; cp.n_ubatch = cp.n_ctx;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 4; }

    llama_memory_t mem = llama_get_memory(ctx);

    llama_batch batch = llama_batch_get_one(toks.data(), (int) toks.size());
    if (llama_decode(ctx, batch)) { fprintf(stderr, "prompt decode failed\n"); return 5; }
    llama_token last = toks.back();
    for (int i = 0; i < n_predict; i++) {
        float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        llama_token best = 0; float bv = logits[0];
        for (int v = 1; v < llama_vocab_n_tokens(vocab); v++) if (logits[v] > bv) { bv = logits[v]; best = (llama_token) v; }
        last = best;
        batch = llama_batch_get_one(&last, 1);
        if (llama_decode(ctx, batch)) { fprintf(stderr, "step %d failed\n", i); return 6; }
    }

    llama_seq_id sid = 0;
    llama_pos pmax_before = llama_memory_seq_pos_max(mem, sid);
    llama_pos pmin_before = llama_memory_seq_pos_min(mem, sid);
    uint32_t total_before = (uint32_t) pmax_before + 1;
    printf("before: pos_min=%d pos_max=%d total=%u\n", (int)pmin_before, (int)pmax_before, total_before);

    size_t dropped = llama_memory_seq_evict_oldest(mem, sid, keep_recent);
    llama_pos pmax_after = llama_memory_seq_pos_max(mem, sid);
    llama_pos pmin_after = llama_memory_seq_pos_min(mem, sid);
    uint32_t total_after = (uint32_t) pmax_after + 1;
    printf("after (keep_recent=%u): dropped=%zu pos_min=%d pos_max=%d total=%u\n",
           keep_recent, dropped, (int)pmin_after, (int)pmax_after, total_after);

    int rc = 0;
    size_t expect_dropped = (size_t)(total_before - keep_recent);
    if (dropped != expect_dropped) {
        printf("FAIL: dropped %zu != expected %zu\n", dropped, expect_dropped); rc |= 1;
    } else printf("OK: dropped count = %zu\n", dropped);
    // Density: pos_min must be 0, pos_max must equal keep_recent-1 after the shift.
    if (pmin_after != 0) { printf("FAIL: pos_min_after %d != 0\n", (int)pmin_after); rc |= 2; }
    else printf("OK: dense pos_min=0\n");
    if ((uint32_t) pmax_after != keep_recent - 1) {
        printf("FAIL: pos_max_after %d != keep_recent-1=%u\n", (int)pmax_after, keep_recent-1); rc |= 4;
    } else printf("OK: dense pos_max = keep_recent-1\n");

    // Idempotency: re-call must be a true no-op (already at floor).
    size_t d2 = llama_memory_seq_evict_oldest(mem, sid, keep_recent);
    if (d2 != 0) { printf("FAIL: re-call dropped %zu (should be 0)\n", d2); rc |= 8; }
    else printf("OK: re-call is true no-op\n");

    // After eviction, decode one more token to confirm the cache still works
    // (the shifted positions must not corrupt attention).
    float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    llama_token best = 0; float bv = logits[0];
    for (int v = 1; v < llama_vocab_n_tokens(vocab); v++) if (logits[v] > bv) { bv = logits[v]; best = (llama_token) v; }
    last = best;
    batch = llama_batch_get_one(&last, 1);
    int step_rc = llama_decode(ctx, batch);
    if (step_rc != 0) { printf("FAIL: post-eviction decode returned %d\n", step_rc); rc |= 16; }
    else printf("OK: post-eviction decode succeeds\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    printf("%s\n", rc == 0 ? "S3_EVICT_TEST_PASS" : "S3_EVICT_TEST_FAIL");
    return rc;
}
