// Persistent forward "logits server" for DiffusionGemma: load the GGUF once, then service many
// [prompt | canvas] forward requests over stdin/stdout so a Python driver can run the block-diffusion
// loop without reloading the model each step.
//
// Protocol (synchronous, one request per line on stdin):
//   stdin  : a line containing a request-file path R
//   file R : int32 P, int32 C, [int32 use_sc, float32 temp,] then (P+C) int32 token ids (canvas last C).
//            The optional use_sc/temp enables self-conditioning (use_sc=1 conditions on the previous
//            step's cached logits; temp scales this step's logits; a block's first step sends use_sc=0).
//   output : C * n_vocab float32 canvas-row logits to "R.resp", then "OK <C>\n". "QUIT"/EOF -> exit.
//
// Usage: llama-diffusion-gemma-server <model.gguf>   (env NGL for gpu layers, FA for flash-attn)

#include "llama.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<int32_t> read_i32_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<int32_t> v(sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fclose(f); return {}; }
    fclose(f);
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 1; }
    const int MAXTOK = atoi(getenv("MAXTOK") ? getenv("MAXTOK") : "2304");

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = atoi(getenv("NGL") ? getenv("NGL") : "0");
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = MAXTOK;
    cparams.n_batch  = MAXTOK;
    cparams.n_ubatch = MAXTOK;   // non-causal: whole sequence in one ubatch
    cparams.no_perf  = true;
    cparams.flash_attn_type = getenv("FA") && atoi(getenv("FA"))
                                ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }
    llama_set_causal_attn(ctx, false);

    llama_batch batch = llama_batch_init(MAXTOK, 0, 1);

    // self-conditioning state: previous step's raw canvas logits + the temperature that produced them
    std::vector<float> sc_cache;   // [C * n_vocab], lazily sized on first request
    float prev_temp = 1.0f;

    // Prompt KV caching (opt-in via DG_KVCACHE=1): on a new block (prompt ids change) PREFILL the prompt
    // once then DECODE only the canvas each step. Off -> the UNIFIED forward (safe default).
    const bool kvcache = getenv("DG_KVCACHE") && atoi(getenv("DG_KVCACHE"));
    std::vector<int32_t> cur_prompt;   // the prompt currently cached in the K,V store

    fprintf(stderr, "diffusion-gemma-server ready (n_vocab=%d, MAXTOK=%d, NGL=%d)\n",
            n_vocab, MAXTOK, mparams.n_gpu_layers);
    printf("READY %d\n", n_vocab); fflush(stdout);

    char line[4096];
    while (fgets(line, sizeof(line), stdin)) {
        size_t L = strlen(line);
        while (L && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = 0;
        if (L == 0) continue;
        if (strcmp(line, "QUIT") == 0) break;

        std::vector<int32_t> req = read_i32_file(line);
        if (req.size() < 2) { printf("ERR badreq\n"); fflush(stdout); continue; }
        const int P = req[0];
        const int C = req[1];
        const int N = P + C;
        // header is either [P,C] (zero-SC) or [P,C,use_sc,temp] (self-conditioning)
        int   use_sc = 0;
        float temp   = 1.0f;
        int   hdr    = 2;
        if ((int) req.size() == 4 + N) {
            hdr = 4;
            use_sc = req[2];
            memcpy(&temp, &req[3], sizeof(float));
        } else if ((int) req.size() != 2 + N) {
            printf("ERR badsize %d %d\n", N, (int) req.size()); fflush(stdout); continue;
        }
        if (N <= 0 || N > MAXTOK) {
            printf("ERR badN %d\n", N); fflush(stdout); continue;
        }

        if ((int) sc_cache.size() != C * n_vocab) {
            sc_cache.assign((size_t) C * n_vocab, 0.0f);
        }

        // row_base = batch index of the first canvas logit row: P in UNIFIED, 0 in cached DECODE
        int row_base = P;

        // caching is valid only for this single-threaded server; P==0 (pure-canvas) -> UNIFIED path
        const bool use_kv = kvcache && P > 0;

        if (!use_kv) {
            // UNIFIED forward over [prompt | canvas] (default, recomputes the prompt every step)
            batch.n_tokens = N;
            for (int i = 0; i < N; ++i) {
                batch.token[i]     = req[hdr + i];
                batch.pos[i]       = i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = (i >= P) ? 1 : 0;   // only need canvas-row logits
            }
            // SC stays enabled every step (constant graph shape); the use_sc gate zeroes it when 0
            llama_diffusion_set_sc(model, sc_cache.data(), use_sc ? 1.0f : 0.0f,
                                   use_sc ? 1.0f / prev_temp : 1.0f, true);
            if (llama_decode(ctx, batch) != 0) { printf("ERR decode\n"); fflush(stdout); continue; }
            row_base = P;
        } else {
            // PREFILL on a new block: forward the prompt only (pos 0..P-1), writing the K,V store.
            bool new_block = ((int) cur_prompt.size() != P);
            for (int i = 0; !new_block && i < P; ++i) {
                if (cur_prompt[i] != req[hdr + i]) new_block = true;
            }
            if (new_block) {
                llama_diffusion_set_phase(model, /*PKV_PREFILL=*/1, P);
                llama_diffusion_set_sc(model, sc_cache.data(), 0.0f, 1.0f, false); // prompt has no SC
                batch.n_tokens = P;
                for (int i = 0; i < P; ++i) {
                    batch.token[i]     = req[hdr + i];
                    batch.pos[i]       = i;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i]    = (i == P - 1) ? 1 : 0;  // logits unused; mark one so n_outputs>0
                }
                if (llama_decode(ctx, batch) != 0) {
                    // PREFILL failed: invalidate the cache so we re-prefill (don't DECODE a half-written store)
                    cur_prompt.clear();
                    printf("ERR prefill\n"); fflush(stdout); continue;
                }
                // commit the cached prompt only after the store was successfully written
                cur_prompt.assign(req.begin() + hdr, req.begin() + hdr + P);
            }
            // DECODE: forward the canvas only (pos P..P+C-1), reading the cached prompt K,V.
            llama_diffusion_set_phase(model, /*PKV_DECODE=*/2, P);
            llama_diffusion_set_sc(model, sc_cache.data(), use_sc ? 1.0f : 0.0f,
                                   use_sc ? 1.0f / prev_temp : 1.0f, true);
            batch.n_tokens = C;
            for (int i = 0; i < C; ++i) {
                batch.token[i]     = req[hdr + P + i];
                batch.pos[i]       = P + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }
            if (llama_decode(ctx, batch) != 0) { printf("ERR decode\n"); fflush(stdout); continue; }
            row_base = 0;
        }

        std::string resp = std::string(line) + ".resp";
        FILE * out = fopen(resp.c_str(), "wb");
        if (!out) { printf("ERR open\n"); fflush(stdout); continue; }
        for (int j = 0; j < C; ++j) {
            const float * row = llama_get_logits_ith(ctx, row_base + j);
            if (!row) { fclose(out); printf("ERR nullrow %d\n", j); fflush(stdout); out = nullptr; break; }
            // cache this step's raw logits for the NEXT step's self-conditioning, and write the response
            memcpy(&sc_cache[(size_t) j * n_vocab], row, n_vocab * sizeof(float));
            fwrite(row, sizeof(float), n_vocab, out);
        }
        if (out) { fclose(out); prev_temp = temp; printf("OK %d\n", C); fflush(stdout); }
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
