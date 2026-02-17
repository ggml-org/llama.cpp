/**
 * voxtral-stream-cli.cpp - Voxtral Realtime streaming transcription
 *
 * Implements the Voxtral Realtime dual-stream inference protocol:
 * - Audio embeddings from the encoder are combined with text token embeddings
 *   at each position: input[pos] = audio_embed[pos] + text_embed[token]
 * - Prefix positions use streaming pad tokens
 * - Autoregressive positions use the previously generated token
 *
 * Usage:
 *   llama-voxtral-stream -m <text-model.gguf> --mmproj <mmproj.gguf> --image <file.wav>
 *   llama-voxtral-stream -m <text-model.gguf> --mmproj <mmproj.gguf>  (mic capture)
 *
 * By default only transcription text and a brief perf summary are printed.
 * Pass --verbose-prompt to see full model-loading and encoder diagnostics.
 */

#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "voxtral-stream.h"

#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>
#include <atomic>
#include <algorithm>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#else
#include <signal.h>
#include <unistd.h>
#endif

// ============================================================================
// Globals
// ============================================================================

static std::atomic<bool> g_running{true};
static bool g_verbose = false;   // controlled by --verbose-prompt

#define LOG_V(...) do { if (g_verbose) fprintf(stderr, __VA_ARGS__); } while (0)

static void signal_handler(int signo) {
    if (signo == SIGINT) {
        g_running.store(false);
    }
}

static void show_usage(int /*argc*/, char ** argv) {
    fprintf(stderr,
        "Voxtral Realtime Streaming Transcription\n\n"
        "Usage: %s [options]\n\n"
        "Required:\n"
        "  -m <path>            Text model GGUF file\n"
        "  --mmproj <path>      Multimodal projector GGUF file\n\n"
        "Optional:\n"
        "  --image <path>       Process audio file (for testing)\n"
        "  -n <N>               Max tokens to generate (default: 500)\n"
        "  -t <N>               Number of threads (default: 4)\n"
        "  -c <N>               Context size (default: 2048, lower = faster)\n"
        "  --no-mmproj-offload  Don't offload mmproj to GPU\n"
        "  --verbose-prompt     Show full debug / model-loading output\n\n"
        "Without --image, captures from the default microphone.\n"
        "Press Ctrl+C to stop.\n",
        argv[0]);
}

// ============================================================================
// Token embedding table loader
// ============================================================================

struct token_embedding_table {
    std::vector<float> data;  // [vocab_size * n_embd] in F32
    int n_embd = 0;
    int vocab_size = 0;

    bool load_from_gguf(const char * model_path) {
        struct gguf_init_params gguf_params = {
            /*.no_alloc =*/ false,
            /*.ctx      =*/ nullptr,
        };

        struct ggml_init_params ggml_params = {
            /*.mem_size   =*/ 1024ull * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        struct ggml_context * ggml_ctx = ggml_init(ggml_params);
        gguf_params.ctx = &ggml_ctx;

        struct gguf_context * gguf_ctx = gguf_init_from_file(model_path, gguf_params);
        if (gguf_ctx == nullptr) {
            fprintf(stderr, "Failed to load GGUF file: %s\n", model_path);
            ggml_free(ggml_ctx);
            return false;
        }

        int64_t tensor_id = gguf_find_tensor(gguf_ctx, "token_embd.weight");
        if (tensor_id < 0) {
            fprintf(stderr, "token_embd.weight not found in GGUF file\n");
            gguf_free(gguf_ctx);
            ggml_free(ggml_ctx);
            return false;
        }

        struct ggml_tensor * tensor = ggml_get_tensor(ggml_ctx, "token_embd.weight");
        if (tensor == nullptr) {
            fprintf(stderr, "Failed to get token_embd.weight tensor\n");
            gguf_free(gguf_ctx);
            ggml_free(ggml_ctx);
            return false;
        }

        n_embd = (int)tensor->ne[0];
        vocab_size = (int)tensor->ne[1];
        LOG_V("Loaded token_embd.weight: [%d, %d], type=%d\n",
                n_embd, vocab_size, tensor->type);

        data.resize((size_t)n_embd * vocab_size);

        if (tensor->type == GGML_TYPE_F32) {
            memcpy(data.data(), tensor->data, data.size() * sizeof(float));
        } else if (tensor->type == GGML_TYPE_F16) {
            const ggml_fp16_t * src = (const ggml_fp16_t *)tensor->data;
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = ggml_fp16_to_fp32(src[i]);
            }
        } else {
            const auto * type_traits = ggml_get_type_traits(tensor->type);
            if (type_traits == nullptr || type_traits->to_float == nullptr) {
                fprintf(stderr, "Unsupported tensor type for token_embd.weight: %d\n", tensor->type);
                gguf_free(gguf_ctx);
                ggml_free(ggml_ctx);
                return false;
            }
            for (int row = 0; row < vocab_size; row++) {
                const void * src = (const char *)tensor->data + row * tensor->nb[1];
                type_traits->to_float(src, data.data() + row * n_embd, n_embd);
            }
        }

        gguf_free(gguf_ctx);
        ggml_free(ggml_ctx);
        return true;
    }

    const float * get_embedding(int token_id) const {
        if (token_id < 0 || token_id >= vocab_size) {
            return nullptr;
        }
        return data.data() + (size_t)token_id * n_embd;
    }
};

// ============================================================================
// Voxtral Realtime inference  (returns number of text tokens generated)
// ============================================================================

struct transcribe_result {
    int   n_tokens      = 0;
    int   n_audio_pos   = 0;
    float encode_ms     = 0.0f;
    float decode_ms     = 0.0f;
};

static transcribe_result voxtral_realtime_transcribe(
        mtmd_context * ctx_vision,
        llama_context * lctx,
        const llama_model * model,
        const llama_vocab * vocab,
        common_sampler * smpl,
        const float * pcm_samples,
        size_t n_samples,
        const token_embedding_table & tok_embd,
        int n_batch,
        int max_tokens) {

    transcribe_result result;

    const int n_embd = llama_model_n_embd(model);
    const int n_embd_mmproj = llama_model_n_embd_inp(model);

    // Voxtral streaming constants
    const int TOKEN_BOS = 1;
    const int TOKEN_STREAMING_PAD = 32;
    const int N_LEFT_PAD_TOKENS = 32;
    const int N_DELAY_TOKENS = 6;

    std::vector<llama_token> prompt_ids;
    prompt_ids.push_back(TOKEN_BOS);
    for (int i = 0; i < N_LEFT_PAD_TOKENS + N_DELAY_TOKENS; i++) {
        prompt_ids.push_back(TOKEN_STREAMING_PAD);
    }
    int n_prefix = (int)prompt_ids.size();

    LOG_V("Prefix: %d tokens (BOS + %d STREAMING_PAD)\n",
            n_prefix, N_LEFT_PAD_TOKENS + N_DELAY_TOKENS);

    // Create audio bitmap and encode
    mtmd_bitmap * bmp = mtmd_bitmap_init_from_audio(n_samples, pcm_samples);
    if (bmp == nullptr) {
        fprintf(stderr, "ERR: Failed to create audio bitmap\n");
        return result;
    }

    std::string prompt = std::string(mtmd_default_marker());
    mtmd_input_text text;
    text.text          = prompt.c_str();
    text.add_special   = true;
    text.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps[] = { bmp };
    int32_t res = mtmd_tokenize(ctx_vision, chunks, &text, bitmaps, 1);
    if (res != 0) {
        fprintf(stderr, "ERR: Failed to tokenize, res = %d\n", res);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        return result;
    }

    size_t n_chunks = mtmd_input_chunks_size(chunks);
    const mtmd_input_chunk * audio_chunk = nullptr;
    for (size_t i = 0; i < n_chunks; i++) {
        auto chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            audio_chunk = chunk;
            break;
        }
    }

    if (audio_chunk == nullptr) {
        fprintf(stderr, "ERR: No audio chunk found\n");
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        return result;
    }

    // ---- Encode audio ----
    auto t_enc_start = std::chrono::high_resolution_clock::now();
    LOG_V("Encoding audio...\n");
    res = mtmd_encode_chunk(ctx_vision, audio_chunk);
    if (res != 0) {
        fprintf(stderr, "ERR: Failed to encode audio, res = %d\n", res);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        return result;
    }
    auto t_enc_end = std::chrono::high_resolution_clock::now();
    result.encode_ms = std::chrono::duration<float, std::milli>(t_enc_end - t_enc_start).count();

    float * audio_embd = mtmd_get_output_embd(ctx_vision);
    int n_audio_tokens = mtmd_input_chunk_get_n_tokens(audio_chunk);
    result.n_audio_pos = n_audio_tokens;

    LOG_V("Audio tokens: %d, encode: %.0f ms\n", n_audio_tokens, result.encode_ms);

    if (n_embd_mmproj != n_embd) {
        LOG_V("WARNING: n_embd_mmproj (%d) != n_embd (%d)\n", n_embd_mmproj, n_embd);
    }

    std::vector<float> audio_embds((size_t)n_audio_tokens * n_embd_mmproj);
    memcpy(audio_embds.data(), audio_embd, audio_embds.size() * sizeof(float));

    LOG_V("Running dual-stream inference (%d audio tokens, %d prefix)...\n",
            n_audio_tokens, n_prefix);

    if (n_prefix > n_audio_tokens) {
        fprintf(stderr, "ERR: prefix (%d) > audio tokens (%d)\n", n_prefix, n_audio_tokens);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        return result;
    }

    // ---- Decode ----
    auto t_dec_start = std::chrono::high_resolution_clock::now();

    // Prepare combined embeddings for prefill
    std::vector<float> combined_embd((size_t)n_prefix * n_embd);
    for (int i = 0; i < n_prefix; i++) {
        const float * a_embd = audio_embds.data() + (size_t)i * n_embd_mmproj;
        const float * t_embd = tok_embd.get_embedding(prompt_ids[i]);
        float * dst = combined_embd.data() + (size_t)i * n_embd;
        for (int j = 0; j < n_embd; j++) {
            dst[j] = a_embd[j] + t_embd[j];
        }
    }

    // Prefill
    llama_pos n_past = 0;
    {
        for (int offset = 0; offset < n_prefix; offset += n_batch) {
            int this_batch = std::min(n_batch, n_prefix - offset);

            llama_batch batch = llama_batch_init(this_batch, n_embd, 1);
            batch.n_tokens = this_batch;
            for (int i = 0; i < this_batch; i++) {
                memcpy(batch.embd + (size_t)i * n_embd,
                       combined_embd.data() + (size_t)(offset + i) * n_embd,
                       n_embd * sizeof(float));
                batch.pos[i] = n_past + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (offset + i == n_prefix - 1) ? 1 : 0;
            }

            if (llama_decode(lctx, batch) != 0) {
                fprintf(stderr, "ERR: decode prefill failed at offset %d\n", offset);
                llama_batch_free(batch);
                mtmd_input_chunks_free(chunks);
                mtmd_bitmap_free(bmp);
                return result;
            }
            n_past += this_batch;
            llama_batch_free(batch);
        }
    }

    LOG_V("Prefill done (n_past=%d)\n", (int)n_past);

    // Sample the first token
    llama_token prev_token = common_sampler_sample(smpl, lctx, -1);
    common_sampler_accept(smpl, prev_token, true);

    std::string piece = common_token_to_piece(lctx, prev_token);
    if (piece.find("[STREAMING_PAD]") == std::string::npos &&
        piece.find("[STREAMING_WORD]") == std::string::npos) {
        fprintf(stdout, "%s", piece.c_str());
        fflush(stdout);
    }

    // Autoregressive generation (reuse a single batch to avoid alloc/free overhead)
    int n_generated = 1;
    llama_batch ar_batch = llama_batch_init(1, n_embd, 1);

    for (int pos = n_prefix; pos < n_audio_tokens && n_generated < max_tokens; pos++) {
        if (!g_running.load()) break;

        if (llama_vocab_is_eog(vocab, prev_token)) {
            break;
        }

        const float * a_embd = audio_embds.data() + (size_t)pos * n_embd_mmproj;
        const float * t_embd = tok_embd.get_embedding(prev_token);
        if (t_embd == nullptr) {
            LOG_V("WARNING: no embedding for token %d\n", prev_token);
            break;
        }

        // Combine audio + text embeddings directly into the batch buffer
        float * dst = ar_batch.embd;
        for (int j = 0; j < n_embd; j++) {
            dst[j] = a_embd[j] + t_embd[j];
        }

        ar_batch.n_tokens    = 1;
        ar_batch.pos[0]      = n_past;
        ar_batch.n_seq_id[0] = 1;
        ar_batch.seq_id[0][0] = 0;
        ar_batch.logits[0]   = 1;

        if (llama_decode(lctx, ar_batch) != 0) {
            fprintf(stderr, "ERR: decode failed at pos %d\n", pos);
            break;
        }
        n_past++;

        prev_token = common_sampler_sample(smpl, lctx, -1);
        common_sampler_accept(smpl, prev_token, true);
        n_generated++;

        piece = common_token_to_piece(lctx, prev_token);
        if (piece.find("[STREAMING_PAD]") == std::string::npos &&
            piece.find("[STREAMING_WORD]") == std::string::npos) {
            fprintf(stdout, "%s", piece.c_str());
            fflush(stdout);
        }
    }

    llama_batch_free(ar_batch);

    auto t_dec_end = std::chrono::high_resolution_clock::now();
    result.decode_ms = std::chrono::duration<float, std::milli>(t_dec_end - t_dec_start).count();
    result.n_tokens = n_generated;

    fprintf(stdout, "\n");
    fflush(stdout);

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);
    return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    ggml_time_init();

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_usage)) {
        return 1;
    }

    // Use --verbose-prompt as our verbose flag
    g_verbose = params.verbose_prompt;

    // Suppress llama.cpp's own logging when not verbose
    if (!g_verbose) {
        common_log_set_verbosity_thold(1);  // only warnings and errors
    }

    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (params.mmproj.path.empty()) {
        show_usage(argc, argv);
        fprintf(stderr, "\nERR: Missing --mmproj argument\n");
        return 1;
    }

    // Reduce default context if user didn't explicitly set one
    // Voxtral chunks rarely exceed ~400 audio tokens, so 2048 is plenty
    if (params.n_ctx == 0) {
        params.n_ctx = 2048;
    }

    // Setup signal handler
#if defined(_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#else
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
#endif

    // Load token embedding table
    LOG_V("Loading token embedding table...\n");
    token_embedding_table tok_embd;
    if (!tok_embd.load_from_gguf(params.model.path.c_str())) {
        fprintf(stderr, "Failed to load token embedding table\n");
        return 1;
    }
    LOG_V("Token embeddings: vocab=%d, n_embd=%d\n",
            tok_embd.vocab_size, tok_embd.n_embd);

    // Initialize llama model and context
    auto llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    llama_context * lctx = llama_init->context();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    common_sampler * smpl = common_sampler_init(model, params.sampling);
    int n_batch = params.n_batch;

    if (!model || !lctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Initialize audio context
    const char * clip_path = params.mmproj.path.c_str();
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu         = params.mmproj_use_gpu;
    mparams.print_timings   = g_verbose;
    mparams.n_threads       = params.cpuparams.n_threads;
    mparams.flash_attn_type = params.flash_attn_type;
    mparams.warmup          = params.warmup;

    mtmd_context * ctx_vision = mtmd_init_from_file(clip_path, model, mparams);
    if (ctx_vision == nullptr) {
        fprintf(stderr, "Failed to load mmproj model from %s\n", clip_path);
        return 1;
    }

    int audio_sample_rate = mtmd_get_audio_bitrate(ctx_vision);
    int n_predict = params.n_predict < 0 ? 500 : params.n_predict;

    // Determine mode
    bool use_mic = params.image.empty();

    if (!g_verbose) {
        fprintf(stderr, "Voxtral Realtime | threads: %d | ctx: %d | mode: %s\n",
                params.cpuparams.n_threads,
                params.n_ctx,
                use_mic ? "microphone" : "file");
    } else {
        fprintf(stderr, "\n=== Voxtral Realtime Streaming ===\n");
        fprintf(stderr, "Audio sample rate: %d Hz\n", audio_sample_rate);
    }

    if (use_mic) {
        if (!g_verbose) {
            fprintf(stderr, "Listening... (Ctrl+C to stop)\n\n");
        } else {
            fprintf(stderr, "Mode: Microphone capture (real-time chunked)\n");
            fprintf(stderr, "Press Ctrl+C to stop.\n\n");
        }

        voxtral_mic_capture mic;
        if (mic.start(audio_sample_rate, 60.0f) != 0) {
            fprintf(stderr, "Failed to start microphone capture\n");
            mtmd_free(ctx_vision);
            return 1;
        }

        const float CHUNK_SECONDS = 5.0f;
        const size_t chunk_samples = (size_t)(audio_sample_rate * CHUNK_SECONDS);

        std::vector<float> all_samples;
        all_samples.reserve(audio_sample_rate * 120);

        std::vector<float> read_buf(audio_sample_rate / 10);  // 100ms
        size_t last_processed = 0;
        int chunk_idx = 0;
        float total_encode_ms = 0, total_decode_ms = 0;
        int   total_tokens = 0;

        while (g_running.load()) {
            size_t nread = mic.read_samples(read_buf.data(), read_buf.size());
            if (nread > 0) {
                all_samples.insert(all_samples.end(), read_buf.begin(), read_buf.begin() + nread);
            }

            size_t new_samples = all_samples.size() - last_processed;
            if (new_samples >= chunk_samples) {
                chunk_idx++;

                LOG_V("\n--- Chunk %d (%.1fs - %.1fs) ---\n",
                        chunk_idx,
                        (float)last_processed / audio_sample_rate,
                        (float)all_samples.size() / audio_sample_rate);

                const float * chunk_start = all_samples.data() + last_processed;
                size_t chunk_len = all_samples.size() - last_processed;

                llama_memory_clear(llama_get_memory(lctx), true);
                common_sampler_reset(smpl);

                auto r = voxtral_realtime_transcribe(
                    ctx_vision, lctx, model, vocab, smpl,
                    chunk_start, chunk_len,
                    tok_embd, n_batch, n_predict);

                total_encode_ms += r.encode_ms;
                total_decode_ms += r.decode_ms;
                total_tokens    += r.n_tokens;

                // Brief per-chunk stats
                float tok_per_sec = (r.decode_ms > 0) ? (r.n_tokens * 1000.0f / r.decode_ms) : 0;
                fprintf(stderr, "  [chunk %d: %d tok, enc %.0fms, dec %.0fms, %.1f tok/s]\n",
                        chunk_idx, r.n_tokens, r.encode_ms, r.decode_ms, tok_per_sec);

                last_processed = all_samples.size();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        mic.stop();

        // Process remaining audio
        if (all_samples.size() > last_processed) {
            size_t remaining = all_samples.size() - last_processed;
            if (remaining > (size_t)(audio_sample_rate / 2)) {
                chunk_idx++;
                LOG_V("\n--- Final chunk %d ---\n", chunk_idx);

                llama_memory_clear(llama_get_memory(lctx), true);
                common_sampler_reset(smpl);

                auto r = voxtral_realtime_transcribe(
                    ctx_vision, lctx, model, vocab, smpl,
                    all_samples.data() + last_processed,
                    all_samples.size() - last_processed,
                    tok_embd, n_batch, n_predict);

                total_encode_ms += r.encode_ms;
                total_decode_ms += r.decode_ms;
                total_tokens    += r.n_tokens;

                float tok_per_sec = (r.decode_ms > 0) ? (r.n_tokens * 1000.0f / r.decode_ms) : 0;
                fprintf(stderr, "  [chunk %d: %d tok, enc %.0fms, dec %.0fms, %.1f tok/s]\n",
                        chunk_idx, r.n_tokens, r.encode_ms, r.decode_ms, tok_per_sec);
            }
        }

        // Summary
        float total_audio_s = (float)all_samples.size() / audio_sample_rate;
        float total_ms = total_encode_ms + total_decode_ms;
        float avg_tok_s = (total_decode_ms > 0) ? (total_tokens * 1000.0f / total_decode_ms) : 0;
        fprintf(stderr, "\n--- Summary ---\n");
        fprintf(stderr, "  audio: %.1fs | chunks: %d | tokens: %d\n",
                total_audio_s, chunk_idx, total_tokens);
        fprintf(stderr, "  encode: %.0fms | decode: %.0fms | total: %.0fms\n",
                total_encode_ms, total_decode_ms, total_ms);
        fprintf(stderr, "  avg decode: %.1f tok/s | RTF: %.2fx\n",
                avg_tok_s, total_ms / (total_audio_s * 1000.0f));

    } else {
        // File mode
        const std::string & audio_path = params.image[0];
        if (!g_verbose) {
            fprintf(stderr, "File: %s\n", audio_path.c_str());
        } else {
            fprintf(stderr, "Mode: File transcription\n");
            fprintf(stderr, "File: %s\n\n", audio_path.c_str());
        }

        std::vector<float> pcm_data;
        {
            mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_file(ctx_vision, audio_path.c_str());
            if (bmp == nullptr) {
                fprintf(stderr, "Failed to load audio file: %s\n", audio_path.c_str());
                mtmd_free(ctx_vision);
                return 1;
            }

            size_t n_bytes = mtmd_bitmap_get_n_bytes(bmp);
            size_t n_samp = n_bytes / sizeof(float);
            const unsigned char * data = mtmd_bitmap_get_data(bmp);

            pcm_data.resize(n_samp);
            memcpy(pcm_data.data(), data, n_bytes);
            mtmd_bitmap_free(bmp);
        }

        float audio_s = (float)pcm_data.size() / audio_sample_rate;
        LOG_V("Loaded %zu samples (%.1f seconds) at %d Hz\n",
                pcm_data.size(), audio_s, audio_sample_rate);

        auto r = voxtral_realtime_transcribe(
            ctx_vision, lctx, model, vocab, smpl,
            pcm_data.data(), pcm_data.size(),
            tok_embd, n_batch, n_predict);

        float tok_per_sec = (r.decode_ms > 0) ? (r.n_tokens * 1000.0f / r.decode_ms) : 0;
        float total_ms = r.encode_ms + r.decode_ms;
        fprintf(stderr, "\n--- Perf ---\n");
        fprintf(stderr, "  audio: %.1fs | tokens: %d\n", audio_s, r.n_tokens);
        fprintf(stderr, "  encode: %.0fms | decode: %.0fms | total: %.0fms\n",
                r.encode_ms, r.decode_ms, total_ms);
        fprintf(stderr, "  decode: %.1f tok/s | RTF: %.2fx\n",
                tok_per_sec, total_ms / (audio_s * 1000.0f));
    }

    if (g_verbose) {
        fprintf(stderr, "\n=== Done ===\n");
        llama_perf_context_print(lctx);
    }

    mtmd_free(ctx_vision);
    common_sampler_free(smpl);

    return 0;
}
