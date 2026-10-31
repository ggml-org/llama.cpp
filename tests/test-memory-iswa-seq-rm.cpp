// check that llama_memory_seq_rm() reports failure when the SWA cache can no longer satisfy a removal
// usage: test-memory-iswa-seq-rm <model with SWA layers, e.g. gemma3-dense.gguf from test-llama-archs -o>

#include "llama.h"

#include <cstdio>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.devices = nullptr;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    CHECK(model != nullptr);

    const int32_t n_swa = llama_model_n_swa(model);
    CHECK(n_swa > 0);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = 1024;
    cparams.n_batch  = 16;
    cparams.n_ubatch = 16;
    cparams.swa_full = false;
    llama_context * ctx = llama_init_from_model(model, cparams);
    CHECK(ctx != nullptr);

    llama_memory_t mem = llama_get_memory(ctx);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // fill well past the size of the SWA cache so that early SWA cells get reused
    const int32_t n_tokens = 768;
    llama_batch batch = llama_batch_init(cparams.n_batch, 0, 1);
    for (int32_t pos = 0; pos < n_tokens; ) {
        batch.n_tokens = 0;
        for (uint32_t i = 0; i < cparams.n_batch && pos < n_tokens; i++, pos++) {
            batch.token   [batch.n_tokens]    = pos % n_vocab;
            batch.pos     [batch.n_tokens]    = pos;
            batch.n_seq_id[batch.n_tokens]    = 1;
            batch.seq_id  [batch.n_tokens][0] = 0;
            batch.logits  [batch.n_tokens]    = false;
            batch.n_tokens++;
        }
        CHECK(llama_decode(ctx, batch) == 0);
    }
    llama_batch_free(batch);

    const llama_pos pos_min = llama_memory_seq_pos_min(mem, 0);
    const llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
    fprintf(stderr, "n_swa = %d, pos_min = %d, pos_max = %d\n", n_swa, pos_min, pos_max);
    CHECK(pos_max == n_tokens - 1);
    CHECK(pos_min > n_swa);

    // cutting into overwritten SWA rows must fail and leave the sequence untouched
    CHECK(!llama_memory_seq_rm(mem, 0, pos_min - 1, -1));
    CHECK(llama_memory_seq_pos_max(mem, 0) == pos_max);

    // a cut that keeps the last n_swa - 1 positions available is fine
    const llama_pos p0 = pos_max - 4;
    CHECK(p0 - n_swa + 1 >= pos_min);
    CHECK(llama_memory_seq_rm(mem, 0, p0, -1));
    CHECK(llama_memory_seq_pos_max(mem, 0) == p0 - 1);

    // removing the whole sequence always works
    CHECK(llama_memory_seq_rm(mem, 0, -1, -1));
    CHECK(llama_memory_seq_pos_max(mem, 0) == -1);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "OK\n");
    return 0;
}
