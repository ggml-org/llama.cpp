#include "arg.h"
#include "debug.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include "console.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "mtmd-mic.h"

#include <vector>
#include <limits.h>
#include <cinttypes>
#include <chrono>
#include <thread>
#include <atomic>
#include <string>
#include <cstring>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

static volatile bool g_is_generating = false;
static volatile bool g_is_interrupted = false;

/**
 * Please note that this is NOT a production-ready stuff.
 * It is a playground for trying multimodal support in llama.cpp.
 * For contributors: please keep this code simple and easy to understand.
 */

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG(
        "Experimental CLI for multimodal\n\n"
        "Usage: %s [options] -m <model> --mmproj <mmproj> --image <image> --audio <audio> -p <prompt>\n\n"
        "  -m and --mmproj are required\n"
        "  -hf user/repo can replace both -m and --mmproj in most cases\n"
        "  --image, --audio and -p are optional, if NOT provided, the CLI will run in chat mode\n"
        "  to disable using GPU for mmproj model, add --no-mmproj-offload\n"
        "\n"
        "  For Voxtral Realtime models (auto-detected from mmproj):\n"
        "    --image <audio.wav>           transcribe an audio file\n"
        "    (no --image)                  live microphone transcription\n",
        argv[0]
    );
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (g_is_generating) {
            g_is_generating = false;
        } else {
            console::cleanup();
            if (g_is_interrupted) {
                _exit(1);
            }
            g_is_interrupted = true;
        }
    }
}
#endif

// ============================================================================
// Peek at mmproj GGUF to detect projector type without loading the full model
// ============================================================================

static bool is_mmproj_voxtral_realtime(const char * mmproj_path) {
    struct gguf_init_params params = { true, nullptr };
    struct gguf_context * ctx = gguf_init_from_file(mmproj_path, params);
    if (ctx == nullptr) return false;

    bool result = false;

    // Check clip.projector_type or clip.audio.projector_type
    const char * keys[] = { "clip.projector_type", "clip.audio.projector_type" };
    for (const char * key : keys) {
        int idx = gguf_find_key(ctx, key);
        if (idx >= 0 && gguf_get_kv_type(ctx, idx) == GGUF_TYPE_STRING) {
            const char * val = gguf_get_val_str(ctx, idx);
            if (val && strcmp(val, "voxtral_realtime") == 0) {
                result = true;
                break;
            }
        }
    }

    gguf_free(ctx);
    return result;
}

// ============================================================================
// Token embedding table loader (for Voxtral Realtime dual-stream protocol)
// ============================================================================

struct token_embedding_table {
    std::vector<float> data;
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
            LOG_ERR("Failed to load GGUF file for token embeddings: %s\n", model_path);
            ggml_free(ggml_ctx);
            return false;
        }

        int64_t tensor_id = gguf_find_tensor(gguf_ctx, "token_embd.weight");
        if (tensor_id < 0) {
            LOG_ERR("token_embd.weight not found in GGUF file\n");
            gguf_free(gguf_ctx);
            ggml_free(ggml_ctx);
            return false;
        }

        struct ggml_tensor * tensor = ggml_get_tensor(ggml_ctx, "token_embd.weight");
        if (tensor == nullptr) {
            LOG_ERR("Failed to get token_embd.weight tensor\n");
            gguf_free(gguf_ctx);
            ggml_free(ggml_ctx);
            return false;
        }

        n_embd = (int)tensor->ne[0];
        vocab_size = (int)tensor->ne[1];

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
                LOG_ERR("Unsupported tensor type for token_embd.weight: %d\n", tensor->type);
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
        if (token_id < 0 || token_id >= vocab_size) return nullptr;
        return data.data() + (size_t)token_id * n_embd;
    }
};

// ============================================================================
// Standard mtmd CLI context (for non-Voxtral-Realtime models)
// ============================================================================

struct mtmd_cli_context {
    mtmd::context_ptr ctx_vision;
    common_init_result_ptr llama_init;

    llama_model       * model;
    llama_context     * lctx;
    const llama_vocab * vocab;
    common_sampler    * smpl;
    llama_batch         batch;
    int                 n_batch;

    mtmd::bitmaps bitmaps;

    common_chat_templates_ptr tmpls;
    std::vector<common_chat_msg> chat_history;
    bool use_jinja = false;

    llama_tokens antiprompt_tokens;

    int n_threads    = 1;
    llama_pos n_past = 0;

    base_callback_data cb_data;

    mtmd_cli_context(common_params & params) : llama_init(common_init_from_params(params)) {
        model = llama_init->model();
        lctx = llama_init->context();
        vocab = llama_model_get_vocab(model);
        smpl = common_sampler_init(model, params.sampling);
        n_threads = params.cpuparams.n_threads;
        batch = llama_batch_init(1, 0, 1);
        n_batch = params.n_batch;

        if (!model || !lctx) {
            exit(1);
        }

        if (!llama_model_chat_template(model, nullptr) && params.chat_template.empty()) {
            LOG_ERR("Model does not have chat template.\n");
            LOG_ERR("  For old llava models, you may need to use '--chat-template vicuna'\n");
            LOG_ERR("  For MobileVLM models, use '--chat-template deepseek'\n");
            LOG_ERR("  For Mistral Small 3.1, use '--chat-template mistral-v7'\n");
            exit(1);
        }

        tmpls = common_chat_templates_init(model, params.chat_template);
        use_jinja = params.use_jinja;
        chat_history.clear();
        LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(tmpls.get(), params.use_jinja, params.default_template_kwargs).c_str());

        init_vision_context(params);

        if (params.chat_template == "vicuna") {
            antiprompt_tokens = common_tokenize(lctx, "ASSISTANT:", false, true);
        } else if (params.chat_template == "deepseek") {
            antiprompt_tokens = common_tokenize(lctx, "###", false, true);
        }
    }

    ~mtmd_cli_context() {
        llama_batch_free(batch);
        common_sampler_free(smpl);
    }

    void init_vision_context(common_params & params) {
        const char * clip_path = params.mmproj.path.c_str();
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu          = params.mmproj_use_gpu;
        mparams.print_timings    = true;
        mparams.n_threads        = params.cpuparams.n_threads;
        mparams.flash_attn_type  = params.flash_attn_type;
        mparams.warmup           = params.warmup;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;
        if (std::getenv("MTMD_DEBUG_GRAPH") != nullptr) {
            mparams.cb_eval_user_data = &cb_data;
            mparams.cb_eval = common_debug_cb_eval<false>;
        }
        ctx_vision.reset(mtmd_init_from_file(clip_path, model, mparams));
        if (!ctx_vision.get()) {
            LOG_ERR("Failed to load vision model from %s\n", clip_path);
            exit(1);
        }
    }

    bool check_antiprompt(const llama_tokens & generated_tokens) {
        if (antiprompt_tokens.empty() || generated_tokens.size() < antiprompt_tokens.size()) {
            return false;
        }
        return std::equal(
            generated_tokens.end() - antiprompt_tokens.size(),
            generated_tokens.end(),
            antiprompt_tokens.begin()
        );
    }

    bool load_media(const std::string & fname) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), fname.c_str()));
        if (!bmp.ptr) {
            return false;
        }
        bitmaps.entries.push_back(std::move(bmp));
        return true;
    }
};

static int generate_response(mtmd_cli_context & ctx, int n_predict) {
    llama_tokens generated_tokens;
    for (int i = 0; i < n_predict; i++) {
        if (i > n_predict || !g_is_generating || g_is_interrupted) {
            LOG("\n");
            break;
        }

        llama_token token_id = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
        generated_tokens.push_back(token_id);
        common_sampler_accept(ctx.smpl, token_id, true);

        if (llama_vocab_is_eog(ctx.vocab, token_id) || ctx.check_antiprompt(generated_tokens)) {
            LOG("\n");
            break;
        }

        LOG("%s", common_token_to_piece(ctx.lctx, token_id).c_str());
        fflush(stdout);

        if (g_is_interrupted) {
            LOG("\n");
            break;
        }

        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("failed to decode token\n");
            return 1;
        }
    }

    std::string generated_text = common_detokenize(ctx.lctx, generated_tokens);
    common_chat_msg msg;
    msg.role    = "assistant";
    msg.content = generated_text;
    ctx.chat_history.push_back(std::move(msg));

    return 0;
}

static std::string chat_add_and_format(mtmd_cli_context & ctx, common_chat_msg & new_msg) {
    LOG_DBG("chat_add_and_format: new_msg.role='%s', new_msg.content='%s'\n",
        new_msg.role.c_str(), new_msg.content.c_str());
    auto formatted = common_chat_format_single(ctx.tmpls.get(), ctx.chat_history,
        new_msg, new_msg.role == "user",
        ctx.use_jinja);
    ctx.chat_history.push_back(new_msg);
    return formatted;
}

static int eval_message(mtmd_cli_context & ctx, common_chat_msg & msg) {
    bool add_bos = ctx.chat_history.empty();
    auto formatted_chat = chat_add_and_format(ctx, msg);
    LOG_DBG("formatted_chat.prompt: %s\n", formatted_chat.c_str());

    mtmd_input_text text;
    text.text          = formatted_chat.c_str();
    text.add_special   = add_bos;
    text.parse_special = true;

    if (g_is_interrupted) return 0;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = ctx.bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(ctx.ctx_vision.get(),
                        chunks.ptr.get(),
                        &text,
                        bitmaps_c_ptr.data(),
                        bitmaps_c_ptr.size());
    if (res != 0) {
        LOG_ERR("Unable to tokenize prompt, res = %d\n", res);
        return 1;
    }

    ctx.bitmaps.entries.clear();

    llama_pos new_n_past;
    if (mtmd_helper_eval_chunks(ctx.ctx_vision.get(),
                ctx.lctx,
                chunks.ptr.get(),
                ctx.n_past,
                0,
                ctx.n_batch,
                true,
                &new_n_past)) {
        LOG_ERR("Unable to eval prompt\n");
        return 1;
    }

    ctx.n_past = new_n_past;

    LOG("\n");

    return 0;
}

// ============================================================================
// Voxtral Realtime dual-stream transcription
// ============================================================================

struct voxtral_rt_result {
    int   n_tokens    = 0;
    int   n_audio_pos = 0;
    float encode_ms   = 0.0f;
    float decode_ms   = 0.0f;
};

static voxtral_rt_result voxtral_realtime_transcribe(
        mtmd_context * ctx_mtmd,
        llama_context * lctx,
        const llama_model * model,
        const llama_vocab * vocab,
        common_sampler * smpl,
        const float * pcm_samples,
        size_t n_samples,
        const token_embedding_table & tok_embd,
        int n_batch,
        int max_tokens) {

    voxtral_rt_result result;

    const int n_embd = llama_model_n_embd(model);
    const int n_embd_mmproj = llama_model_n_embd_inp(model);

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

    mtmd_bitmap * bmp = mtmd_bitmap_init_from_audio(n_samples, pcm_samples);
    if (bmp == nullptr) {
        LOG_ERR("Failed to create audio bitmap\n");
        return result;
    }

    std::string prompt = std::string(mtmd_default_marker());
    mtmd_input_text text;
    text.text          = prompt.c_str();
    text.add_special   = true;
    text.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps[] = { bmp };
    int32_t res = mtmd_tokenize(ctx_mtmd, chunks, &text, bitmaps, 1);
    if (res != 0) {
        LOG_ERR("Failed to tokenize audio, res = %d\n", res);
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
        LOG_ERR("No audio chunk found\n");
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        return result;
    }

    auto t_enc_start = std::chrono::high_resolution_clock::now();
    res = mtmd_encode_chunk(ctx_mtmd, audio_chunk);
    if (res != 0) {
        LOG_ERR("Failed to encode audio, res = %d\n", res);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        return result;
    }
    auto t_enc_end = std::chrono::high_resolution_clock::now();
    result.encode_ms = std::chrono::duration<float, std::milli>(t_enc_end - t_enc_start).count();

    float * audio_embd = mtmd_get_output_embd(ctx_mtmd);
    int n_audio_tokens = mtmd_input_chunk_get_n_tokens(audio_chunk);
    result.n_audio_pos = n_audio_tokens;

    std::vector<float> audio_embds((size_t)n_audio_tokens * n_embd_mmproj);
    memcpy(audio_embds.data(), audio_embd, audio_embds.size() * sizeof(float));

    if (n_prefix > n_audio_tokens) {
        LOG_ERR("prefix (%d) > audio tokens (%d)\n", n_prefix, n_audio_tokens);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        return result;
    }

    auto t_dec_start = std::chrono::high_resolution_clock::now();

    // Prefill: input[pos] = audio_embed[pos] + text_embed[prompt_token[pos]]
    std::vector<float> combined_embd((size_t)n_prefix * n_embd);
    for (int i = 0; i < n_prefix; i++) {
        const float * a_embd = audio_embds.data() + (size_t)i * n_embd_mmproj;
        const float * t_embd = tok_embd.get_embedding(prompt_ids[i]);
        float * dst = combined_embd.data() + (size_t)i * n_embd;
        for (int j = 0; j < n_embd; j++) {
            dst[j] = a_embd[j] + t_embd[j];
        }
    }

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
                LOG_ERR("decode prefill failed at offset %d\n", offset);
                llama_batch_free(batch);
                mtmd_input_chunks_free(chunks);
                mtmd_bitmap_free(bmp);
                return result;
            }
            n_past += this_batch;
            llama_batch_free(batch);
        }
    }

    llama_token prev_token = common_sampler_sample(smpl, lctx, -1);
    common_sampler_accept(smpl, prev_token, true);

    std::string piece = common_token_to_piece(lctx, prev_token);
    if (piece.find("[STREAMING_PAD]") == std::string::npos &&
        piece.find("[STREAMING_WORD]") == std::string::npos) {
        fprintf(stdout, "%s", piece.c_str());
        fflush(stdout);
    }

    // Autoregressive: input[pos] = audio_embed[pos] + text_embed[prev_token]
    int n_generated = 1;
    llama_batch ar_batch = llama_batch_init(1, n_embd, 1);

    for (int pos = n_prefix; pos < n_audio_tokens && n_generated < max_tokens; pos++) {
        if (g_is_interrupted) break;
        if (llama_vocab_is_eog(vocab, prev_token)) break;

        const float * a_embd = audio_embds.data() + (size_t)pos * n_embd_mmproj;
        const float * t_embd = tok_embd.get_embedding(prev_token);
        if (t_embd == nullptr) break;

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
            LOG_ERR("decode failed at pos %d\n", pos);
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
// Voxtral Realtime mode (file or mic) — auto-detected from mmproj
// ============================================================================

static int run_voxtral_realtime(common_params & params) {
    auto llama_init_ptr = common_init_from_params(params);
    llama_model * model = llama_init_ptr->model();
    llama_context * lctx = llama_init_ptr->context();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    common_sampler * smpl = common_sampler_init(model, params.sampling);
    int n_batch = params.n_batch;

    if (!model || !lctx) {
        LOG_ERR("Failed to load model\n");
        return 1;
    }

    token_embedding_table tok_embd;
    if (!tok_embd.load_from_gguf(params.model.path.c_str())) {
        LOG_ERR("Failed to load token embedding table\n");
        common_sampler_free(smpl);
        return 1;
    }
    LOG_INF("Token embeddings: vocab=%d, n_embd=%d\n", tok_embd.vocab_size, tok_embd.n_embd);

    const char * clip_path = params.mmproj.path.c_str();
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu         = params.mmproj_use_gpu;
    mparams.print_timings   = params.verbose_prompt;
    mparams.n_threads       = params.cpuparams.n_threads;
    mparams.flash_attn_type = params.flash_attn_type;
    mparams.warmup          = params.warmup;

    mtmd_context * ctx_mtmd = mtmd_init_from_file(clip_path, model, mparams);
    if (ctx_mtmd == nullptr) {
        LOG_ERR("Failed to load mmproj model from %s\n", clip_path);
        common_sampler_free(smpl);
        return 1;
    }

    int audio_sample_rate = mtmd_get_audio_bitrate(ctx_mtmd);
    int n_predict = params.n_predict < 0 ? 500 : params.n_predict;

    bool use_mic = params.image.empty();

    LOG_INF("Voxtral Realtime | threads: %d | ctx: %d | mode: %s\n",
            params.cpuparams.n_threads, params.n_ctx,
            use_mic ? "microphone" : "file");

    int ret = 0;

    if (use_mic) {
        LOG_INF("Listening... (Ctrl+C to stop)\n\n");

        mtmd_mic_capture mic;
        if (mic.start(audio_sample_rate, 60.0f) != 0) {
            LOG_ERR("Failed to start microphone capture\n");
            mtmd_free(ctx_mtmd);
            common_sampler_free(smpl);
            return 1;
        }

        const float CHUNK_SECONDS = 5.0f;
        const size_t chunk_samples = (size_t)(audio_sample_rate * CHUNK_SECONDS);

        std::vector<float> all_samples;
        all_samples.reserve(audio_sample_rate * 120);

        std::vector<float> read_buf(audio_sample_rate / 10);
        size_t last_processed = 0;
        int chunk_idx = 0;
        float total_encode_ms = 0, total_decode_ms = 0;
        int   total_tokens = 0;

        while (!g_is_interrupted) {
            size_t nread = mic.read_samples(read_buf.data(), read_buf.size());
            if (nread > 0) {
                all_samples.insert(all_samples.end(), read_buf.begin(), read_buf.begin() + nread);
            }

            size_t new_samples = all_samples.size() - last_processed;
            if (new_samples >= chunk_samples) {
                chunk_idx++;

                const float * chunk_start = all_samples.data() + last_processed;
                size_t chunk_len = all_samples.size() - last_processed;

                llama_memory_clear(llama_get_memory(lctx), true);
                common_sampler_reset(smpl);

                auto r = voxtral_realtime_transcribe(
                    ctx_mtmd, lctx, model, vocab, smpl,
                    chunk_start, chunk_len,
                    tok_embd, n_batch, n_predict);

                total_encode_ms += r.encode_ms;
                total_decode_ms += r.decode_ms;
                total_tokens    += r.n_tokens;

                float tok_per_sec = (r.decode_ms > 0) ? (r.n_tokens * 1000.0f / r.decode_ms) : 0;
                LOG_INF("  [chunk %d: %d tok, enc %.0fms, dec %.0fms, %.1f tok/s]\n",
                        chunk_idx, r.n_tokens, r.encode_ms, r.decode_ms, tok_per_sec);

                last_processed = all_samples.size();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        mic.stop();

        if (all_samples.size() > last_processed) {
            size_t remaining = all_samples.size() - last_processed;
            if (remaining > (size_t)(audio_sample_rate / 2)) {
                chunk_idx++;

                llama_memory_clear(llama_get_memory(lctx), true);
                common_sampler_reset(smpl);

                auto r = voxtral_realtime_transcribe(
                    ctx_mtmd, lctx, model, vocab, smpl,
                    all_samples.data() + last_processed,
                    all_samples.size() - last_processed,
                    tok_embd, n_batch, n_predict);

                total_encode_ms += r.encode_ms;
                total_decode_ms += r.decode_ms;
                total_tokens    += r.n_tokens;

                float tok_per_sec = (r.decode_ms > 0) ? (r.n_tokens * 1000.0f / r.decode_ms) : 0;
                LOG_INF("  [chunk %d: %d tok, enc %.0fms, dec %.0fms, %.1f tok/s]\n",
                        chunk_idx, r.n_tokens, r.encode_ms, r.decode_ms, tok_per_sec);
            }
        }

        float total_audio_s = (float)all_samples.size() / audio_sample_rate;
        float total_ms = total_encode_ms + total_decode_ms;
        float avg_tok_s = (total_decode_ms > 0) ? (total_tokens * 1000.0f / total_decode_ms) : 0;
        LOG_INF("\n--- Summary ---\n");
        LOG_INF("  audio: %.1fs | chunks: %d | tokens: %d\n",
                total_audio_s, chunk_idx, total_tokens);
        LOG_INF("  encode: %.0fms | decode: %.0fms | total: %.0fms\n",
                total_encode_ms, total_decode_ms, total_ms);
        LOG_INF("  avg decode: %.1f tok/s | RTF: %.2fx\n",
                avg_tok_s, total_ms / (total_audio_s * 1000.0f));

    } else {
        const std::string & audio_path = params.image[0];
        LOG_INF("File: %s\n", audio_path.c_str());

        std::vector<float> pcm_data;
        {
            mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_file(ctx_mtmd, audio_path.c_str());
            if (bmp == nullptr) {
                LOG_ERR("Failed to load audio file: %s\n", audio_path.c_str());
                mtmd_free(ctx_mtmd);
                common_sampler_free(smpl);
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
        LOG_INF("Loaded %zu samples (%.1f seconds) at %d Hz\n",
                pcm_data.size(), audio_s, audio_sample_rate);

        auto r = voxtral_realtime_transcribe(
            ctx_mtmd, lctx, model, vocab, smpl,
            pcm_data.data(), pcm_data.size(),
            tok_embd, n_batch, n_predict);

        float tok_per_sec = (r.decode_ms > 0) ? (r.n_tokens * 1000.0f / r.decode_ms) : 0;
        float total_ms = r.encode_ms + r.decode_ms;
        LOG_INF("\n--- Perf ---\n");
        LOG_INF("  audio: %.1fs | tokens: %d\n", audio_s, r.n_tokens);
        LOG_INF("  encode: %.0fms | decode: %.0fms | total: %.0fms\n",
                r.encode_ms, r.decode_ms, total_ms);
        LOG_INF("  decode: %.1f tok/s | RTF: %.2fx\n",
                tok_per_sec, total_ms / (audio_s * 1000.0f));
    }

    if (params.verbose_prompt) {
        llama_perf_context_print(lctx);
    }

    mtmd_free(ctx_mtmd);
    common_sampler_free(smpl);
    return g_is_interrupted ? 130 : ret;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    ggml_time_init();

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }

    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (params.mmproj.path.empty()) {
        show_additional_info(argc, argv);
        LOG_ERR("ERR: Missing --mmproj argument\n");
        return 1;
    }

    // Ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (g_is_interrupted) return 130;

    // Peek at the mmproj GGUF to detect Voxtral Realtime without loading the full model
    if (is_mmproj_voxtral_realtime(params.mmproj.path.c_str())) {
        LOG_INF("Detected Voxtral Realtime model\n");
        if (params.n_ctx == 0) {
            params.n_ctx = 2048;
        }
        return run_voxtral_realtime(params);
    }

    // Standard multimodal path
    mtmd_cli_context ctx(params);
    LOG_INF("%s: loading model: %s\n", __func__, params.model.path.c_str());

    bool is_single_turn = !params.prompt.empty() && !params.image.empty();

    int n_predict = params.n_predict < 0 ? INT_MAX : params.n_predict;

    if (g_is_interrupted) return 130;

    auto eval_system_prompt_if_present = [&] {
        if (params.system_prompt.empty()) {
            return 0;
        }

        common_chat_msg msg;
        msg.role = "system";
        msg.content = params.system_prompt;
        return eval_message(ctx, msg);
    };

    LOG_WRN("WARN: This is an experimental CLI for testing multimodal capability.\n");
    LOG_WRN("      For normal use cases, please use the standard llama-cli\n");

    if (eval_system_prompt_if_present()) {
        return 1;
    }

    if (is_single_turn) {
        g_is_generating = true;
        if (params.prompt.find(mtmd_default_marker()) == std::string::npos) {
            for (size_t i = 0; i < params.image.size(); i++) {
                params.prompt = mtmd_default_marker() + params.prompt;
            }
        }

        common_chat_msg msg;
        msg.role = "user";
        msg.content = params.prompt;
        for (const auto & image : params.image) {
            if (!ctx.load_media(image)) {
                return 1;
            }
        }
        if (eval_message(ctx, msg)) {
            return 1;
        }
        if (!g_is_interrupted && generate_response(ctx, n_predict)) {
            return 1;
        }

    } else {
        LOG("\n Running in chat mode, available commands:");
        if (mtmd_support_vision(ctx.ctx_vision.get())) {
            LOG("\n   /image <path>    load an image");
        }
        if (mtmd_support_audio(ctx.ctx_vision.get())) {
            LOG("\n   /audio <path>    load an audio");
        }
        LOG("\n   /clear           clear the chat history");
        LOG("\n   /quit or /exit   exit the program");
        LOG("\n");

        std::string content;

        while (!g_is_interrupted) {
            g_is_generating = false;
            LOG("\n> ");
            console::set_display(DISPLAY_TYPE_USER_INPUT);
            std::string line;
            console::readline(line, false);
            if (g_is_interrupted) break;
            console::set_display(DISPLAY_TYPE_RESET);
            line = string_strip(line);
            if (line.empty()) {
                continue;
            }
            if (line == "/quit" || line == "/exit") {
                break;
            }
            if (line == "/clear") {
                ctx.n_past = 0;
                ctx.chat_history.clear();
                llama_memory_clear(llama_get_memory(ctx.lctx), true);
                if (eval_system_prompt_if_present()) {
                    return 1;
                }
                LOG("Chat history cleared\n\n");
                continue;
            }
            g_is_generating = true;
            bool is_image = line == "/image" || line.find("/image ") == 0;
            bool is_audio = line == "/audio" || line.find("/audio ") == 0;
            if (is_image || is_audio) {
                if (line.size() < 8) {
                    LOG_ERR("ERR: Missing media filename\n");
                    continue;
                }
                std::string media_path = line.substr(7);
                if (ctx.load_media(media_path)) {
                    LOG("%s %s loaded\n", media_path.c_str(), is_image ? "image" : "audio");
                    content += mtmd_default_marker();
                }
                continue;
            } else {
                content += line;
            }
            common_chat_msg msg;
            msg.role = "user";
            msg.content = content;
            int ret = eval_message(ctx, msg);
            if (ret) {
                return 1;
            }
            if (g_is_interrupted) break;
            if (generate_response(ctx, n_predict)) {
                return 1;
            }
            content.clear();
        }
    }
    if (g_is_interrupted) LOG("\nInterrupted by user\n");
    LOG("\n\n");
    llama_perf_context_print(ctx.lctx);
    return g_is_interrupted ? 130 : 0;
}
