// Minimal exactness harness for DiffusionGemma: feed golden token ids [prompt | canvas] through a single
// no-cache bidirectional forward and dump the canvas-position logits as raw float32 vs transformers goldens.
//
// Usage: llama-diffusion-gemma-eval <model.gguf> <prompt_ids.i32> <canvas_ids.i32> <out.bin> [prev_logits.bin]
// Id files are raw little-endian int32. The optional 5th arg (previous step's logits [C, n_vocab]) enables
// self-conditioning (temp_inv=1), else the zero-SC path.

#include "llama.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>

static std::vector<int32_t> read_i32(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<int32_t> v(sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    return v;
}

static std::vector<float> read_f32(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<float> v(sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt_ids.i32> <canvas_ids.i32> <out_logits.bin>\n", argv[0]);
        return 1;
    }
    const char * model_path  = argv[1];
    const char * prompt_path = argv[2];
    const char * canvas_path = argv[3];
    const char * out_path    = argv[4];
    const char * prev_path   = (argc >= 6) ? argv[5] : nullptr;

    std::vector<int32_t> prompt_ids = read_i32(prompt_path);
    std::vector<int32_t> canvas_ids = read_i32(canvas_path);
    const int P = (int) prompt_ids.size();
    const int C = (int) canvas_ids.size();
    const int N = P + C;
    fprintf(stderr, "prompt=%d canvas=%d total=%d\n", P, C, N);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = atoi(getenv("NGL") ? getenv("NGL") : "0");
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    fprintf(stderr, "n_vocab=%d  is_diffusion=%d\n", n_vocab, llama_model_is_diffusion(model));

    // the graph splits on the GGUF diffusion.canvas_length, not the canvas file; validate they agree
    char canvas_meta[32] = {};
    if (llama_model_meta_val_str(model, "diffusion.canvas_length", canvas_meta, sizeof(canvas_meta)) < 0) {
        fprintf(stderr, "model is missing diffusion.canvas_length metadata\n");
        return 1;
    }
    const long model_C = strtol(canvas_meta, nullptr, 10);
    if (C != (int) model_C) {
        fprintf(stderr, "canvas_ids length %d != model diffusion.canvas_length %ld\n", C, model_C);
        return 1;
    }
    if (P <= 0) {
        fprintf(stderr, "exactness eval requires a non-empty prompt (P=%d)\n", P);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = N;
    cparams.n_batch   = N;
    cparams.n_ubatch  = N;   // non-causal requires the whole sequence in one ubatch
    cparams.no_perf   = true;
    // match the eager fp32-softmax reference: disable flash attention unless asked
    cparams.flash_attn_type = getenv("FA") && atoi(getenv("FA"))
                                ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    // bidirectional for the whole pass (the arch's region mask makes the prompt block causal internally)
    llama_set_causal_attn(ctx, false);

    // optional self-conditioning: previous step's logits [C, n_vocab] (in cached mode, DECODE only)
    std::vector<float> prev_logits;
    if (prev_path) {
        prev_logits = read_f32(prev_path);
        const size_t expect = (size_t) C * n_vocab;
        if (prev_logits.size() != expect) {
            fprintf(stderr, "prev_logits size %zu != C*n_vocab %zu\n", prev_logits.size(), expect);
            return 1;
        }
    }

    // DG_CACHED=1 exercises prompt-KV caching: PREFILL the prompt (writing the store) then DECODE the
    // canvas reading it; canvas logits must match the unified forward to F32 round-off.
    const bool cached = getenv("DG_CACHED") && atoi(getenv("DG_CACHED"));

    FILE * out = fopen(out_path, "wb");
    if (!out) { fprintf(stderr, "cannot open %s for write\n", out_path); return 1; }

    if (!cached) {
        // build the batch: [prompt | canvas], positions 0..N-1, single sequence, logits for all
        llama_batch batch = llama_batch_init(N, 0, 1);
        batch.n_tokens = N;
        for (int i = 0; i < N; ++i) {
            batch.token[i]     = (i < P) ? prompt_ids[i] : canvas_ids[i - P];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }
        if (prev_path) {
            llama_diffusion_set_sc(model, prev_logits.data(), /*use_sc=*/1.0f, /*temp_inv=*/1.0f, /*enabled=*/true);
            fprintf(stderr, "self-conditioning ENABLED from %s\n", prev_path);
        }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "llama_decode failed\n"); return 1; }
        for (int i = P; i < N; ++i) {
            const float * row = llama_get_logits_ith(ctx, i);
            if (!row) { fprintf(stderr, "null logits at %d\n", i); return 1; }
            fwrite(row, sizeof(float), n_vocab, out);
        }
        llama_batch_free(batch);
    } else {
        fprintf(stderr, "CACHED mode: PREFILL(P=%d) then DECODE(C=%d)\n", P, C);

        // PREFILL: forward the prompt only, no SC, writing each layer's K,V to the store (logits unused,
        // request just the last row so n_outputs > 0).
        llama_diffusion_set_phase(model, /*PKV_PREFILL=*/1, P);
        llama_diffusion_set_sc(model, nullptr, /*use_sc=*/0.0f, /*temp_inv=*/1.0f, /*enabled=*/false);
        {
            llama_batch pre = llama_batch_init(P, 0, 1);
            pre.n_tokens = P;
            for (int i = 0; i < P; ++i) {
                pre.token[i]     = prompt_ids[i];
                pre.pos[i]       = i;
                pre.n_seq_id[i]  = 1;
                pre.seq_id[i][0] = 0;
                pre.logits[i]    = (i == P - 1) ? 1 : 0;
            }
            if (llama_decode(ctx, pre) != 0) { fprintf(stderr, "PREFILL decode failed\n"); return 1; }
            llama_batch_free(pre);
        }

        // DECODE: forward the canvas only (P..P+C-1), reading the cached prompt K,V (SC enabled if given)
        llama_diffusion_set_phase(model, /*PKV_DECODE=*/2, P);
        if (prev_path) {
            llama_diffusion_set_sc(model, prev_logits.data(), /*use_sc=*/1.0f, /*temp_inv=*/1.0f, /*enabled=*/true);
            fprintf(stderr, "self-conditioning ENABLED from %s\n", prev_path);
        }
        {
            llama_batch dec = llama_batch_init(C, 0, 1);
            dec.n_tokens = C;
            for (int i = 0; i < C; ++i) {
                dec.token[i]     = canvas_ids[i];
                dec.pos[i]       = P + i;
                dec.n_seq_id[i]  = 1;
                dec.seq_id[i][0] = 0;
                dec.logits[i]    = 1;
            }
            if (llama_decode(ctx, dec) != 0) { fprintf(stderr, "DECODE decode failed\n"); return 1; }
            for (int i = 0; i < C; ++i) {
                const float * row = llama_get_logits_ith(ctx, i);
                if (!row) { fprintf(stderr, "null logits at %d\n", i); return 1; }
                fwrite(row, sizeof(float), n_vocab, out);
            }
            llama_batch_free(dec);
        }
        llama_diffusion_set_phase(model, /*PKV_UNIFIED=*/0, 0);
    }
    fclose(out);
    fprintf(stderr, "wrote %d x %d float32 logits to %s\n", C, n_vocab, out_path);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
