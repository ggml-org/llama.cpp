// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "mtmd.h"
#include "mtmd-helper.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <vector>

//#define MTMD_AUDIO_DEBUG

#define MINIAUDIO_IMPLEMENTATION
#ifndef MTMD_AUDIO_DEBUG
#   define MA_NO_ENCODING
#endif
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#ifdef MTMD_INTERNAL_HEADER
#error "mtmd-helper is a public library outside of mtmd. it must not include internal headers"
#endif

//
// internal logging functions
//

struct mtmd_helper_logger {
    ggml_log_callback default_callback = [](ggml_log_level level, const char * text, void * user_data) {
        (void) level;
        (void) user_data;
        fputs(text, stderr);
        fflush(stderr);
    };

    ggml_log_callback log_callback = default_callback;
    void * log_callback_user_data;

    void log_v(enum ggml_log_level level, const char * format, va_list args) {
        if (format == NULL) {
            return;
        }
        va_list args_copy;
        va_copy(args_copy, args);
        char buffer[128];
        int len = vsnprintf(buffer, 128, format, args);
        if (len < 128) {
            log_callback(level, buffer, log_callback_user_data);
        } else {
            char * buffer2 = (char *) calloc(len + 1, sizeof(char));
            vsnprintf(buffer2, len + 1, format, args_copy);
            buffer2[len] = 0;
            log_callback(level, buffer2, log_callback_user_data);
            free(buffer2);
        }
        va_end(args_copy);
    }

    void log(enum ggml_log_level level, const char * format, ...) {
        va_list args;
        va_start(args, format);
        log_v(level, format, args);
        va_end(args);
    }
} g_logger;

#define LOG_INF(...) g_logger.log(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_WRN(...) g_logger.log(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_ERR(...) g_logger.log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

void mtmd_helper_log_set(ggml_log_callback log_callback, void * user_data) {
    if (log_callback == nullptr) {
        log_callback = g_logger.default_callback;
    }
    g_logger.log_callback = log_callback;
    g_logger.log_callback_user_data = user_data;
    mtmd_log_set(log_callback, user_data);
}

//
// helper functions
//

size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks * chunks) {
    size_t n_tokens = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        auto chunk = mtmd_input_chunks_get(chunks, i);
        n_tokens += mtmd_input_chunk_get_n_tokens(chunk);
    }
    return n_tokens;
}

llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks * chunks) {
    llama_pos n_pos = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        auto chunk = mtmd_input_chunks_get(chunks, i);
        n_pos += mtmd_input_chunk_get_n_pos(chunk);
    }
    return n_pos;
}

// helper struct to make working with embd batch easier
// note: this will be removed after llama_batch_ext refactoring
struct decode_embd_batch {
    int n_pos_per_embd;
    int n_mmproj_embd;
    std::vector<llama_pos>      pos;
    std::vector<llama_pos>      pos_view; // used by mrope
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch;
    decode_embd_batch(float * embd, int32_t n_tokens, int n_pos_per_embd, int n_mmproj_embd) : n_pos_per_embd(n_pos_per_embd), n_mmproj_embd(n_mmproj_embd) {
        pos     .resize(n_tokens * n_pos_per_embd);
        n_seq_id.resize(n_tokens);
        seq_ids .resize(n_tokens + 1);
        logits  .resize(n_tokens);
        seq_id_0.resize(1);
        seq_ids [n_tokens] = nullptr;
        batch = {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ embd,
            /*pos            =*/ pos.data(),
            /*n_seq_id       =*/ n_seq_id.data(),
            /*seq_id         =*/ seq_ids.data(),
            /*logits         =*/ logits.data(),
        };
    }

    void set_position_normal(llama_pos pos_0, llama_seq_id seq_id) {
        seq_id_0[0] = seq_id;
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.pos     [i] = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    // M-RoPE for image
    void set_position_mrope_2d(llama_pos pos_0, int nx, int ny, llama_seq_id seq_id) {
        GGML_ASSERT(n_pos_per_embd == 4);
        seq_id_0[0] = seq_id;
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                int i = y * nx + x;
                pos[i                     ] = pos_0;
                pos[i + batch.n_tokens    ] = pos_0 + y;
                pos[i + batch.n_tokens * 2] = pos_0 + x;
                pos[i + batch.n_tokens * 3] = 0; // last pos dim is unused
            }
        }
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    // M-RoPE for audio
    void set_position_mrope_1d(llama_pos pos_0, llama_seq_id seq_id) {
        GGML_ASSERT(n_pos_per_embd == 4);
        seq_id_0[0] = seq_id;
        for (int i = 0; i < batch.n_tokens; i++) {
            pos[i                     ] = pos_0 + i;
            pos[i + batch.n_tokens    ] = pos_0 + i;
            pos[i + batch.n_tokens * 2] = pos_0 + i;
            pos[i + batch.n_tokens * 3] = 0; // last pos dim is unused
        }
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    llama_batch get_view(int offset, int n_tokens) {
        llama_pos * pos_ptr;
        pos_view.clear();
        pos_view.reserve(n_tokens * n_pos_per_embd);
        if (n_pos_per_embd > 1) {
            // mrope
            // for example, with layout of src: 1234...1234...1234...1234...
            //       offset 2 will give us dst: 34...34...34...34...
            for (int i = 0; i < n_pos_per_embd; i++) {
                // assume n_tokens is less than or equal to batch.n_tokens
                // batch.n_tokens is number of **total** tokens
                // n_tokens is number of viewed token
                size_t src_idx = i * batch.n_tokens + offset;
                pos_view.insert(pos_view.end(),
                    pos.data() + src_idx,
                    pos.data() + src_idx + n_tokens);
            }
            pos_ptr = pos_view.data();
        } else {
            // normal
            pos_ptr = pos.data() + offset;
        }
        return {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ batch.embd     + offset * n_mmproj_embd,
            /*pos            =*/ pos_ptr,
            /*n_seq_id       =*/ batch.n_seq_id + offset,
            /*seq_id         =*/ batch.seq_id   + offset,
            /*logits         =*/ batch.logits   + offset,
        };
    }
};

// Helper function for decoding an image whose embeddings have already been calculated
int32_t mtmd_helper_decode_image_chunk(
        mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * chunk,
        float * encoded_embd,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        llama_pos * new_n_past) {
    auto chunk_type = mtmd_input_chunk_get_type(chunk);
    const char * name = chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        LOG_ERR("failed to decode chunk: input chunk not of image/audio type\n");
        return -1;
    }

    const llama_model * model = llama_get_model(lctx);
    int n_mmproj_embd = llama_model_n_embd_inp(model);
    int n_pos_per_embd = mtmd_decode_use_mrope(ctx) ? 4 : 1;

    int32_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
    int32_t i_batch = 0;
    int32_t n_img_batches = (n_tokens + n_batch - 1) / n_batch;
    decode_embd_batch batch_embd(encoded_embd, n_tokens, n_pos_per_embd, n_mmproj_embd);

    if (mtmd_decode_use_mrope(ctx)) {
        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            const auto image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
            if (!image_tokens) {
                LOG_ERR("failed to decode chunk: image tokens are null\n");
                return -1;
            }
            const int nx = mtmd_image_tokens_get_nx(image_tokens);
            const int ny = mtmd_image_tokens_get_ny(image_tokens);
            batch_embd.set_position_mrope_2d(n_past, nx, ny, seq_id);
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            batch_embd.set_position_mrope_1d(n_past, seq_id);
        } else {
            GGML_ABORT("invalid chunk type for M-RoPE");
        }
    } else {
        batch_embd.set_position_normal(n_past, seq_id);
    }

    if (mtmd_decode_use_non_causal(ctx)) {
        llama_set_causal_attn(lctx, false);
        // TODO @ngxson : need to make sure only one image is processed at a time, and n_ubatch must be enough to hold the image
    }

    while (i_batch < n_img_batches) { // split into batches
        int pos_offset = i_batch*n_batch;
        int n_tokens_batch = std::min(n_batch, n_tokens - pos_offset);
        llama_batch batch_embd_view = batch_embd.get_view(pos_offset, n_tokens_batch);

        LOG_INF("decoding %s batch %d/%d, n_tokens_batch = %d\n", name, i_batch+1, n_img_batches, n_tokens_batch);

        int64_t t1 = ggml_time_ms();
        int32_t ret = llama_decode(lctx, batch_embd_view);
        if (ret != 0) {
            LOG_ERR("failed to decode %s\n", name);
            llama_set_causal_attn(lctx, true); // restore causal attn
            return ret;
        }

        LOG_INF("%s decoded (batch %d/%d) in %" PRId64 " ms\n", name, i_batch+1, n_img_batches, ggml_time_ms() - t1);

        i_batch++;
    }

    n_past += mtmd_input_chunk_get_n_pos(chunk);
    *new_n_past = n_past;

    if (mtmd_decode_use_non_causal(ctx)) {
        llama_set_causal_attn(lctx, true);
    }
    return 0;
}

int32_t mtmd_helper_eval_chunk_single(mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * chunk,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        bool logits_last,
        llama_pos * new_n_past) {
    int32_t ret;
    llama_batch text_batch = llama_batch_init(n_batch, 0, 1);
    auto chunk_type = mtmd_input_chunk_get_type(chunk);

    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        size_t n_tokens;
        const auto tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
        // LOG_INF("decoding text chunk, n_tokens = %zu\n", n_tokens);
        size_t i = 0;
        while (i < n_tokens) { // split into batches
            text_batch.n_tokens = 0; // clear the batch
            for (; i < n_tokens && text_batch.n_tokens < n_batch; i++) {
                int32_t j = text_batch.n_tokens;
                text_batch.token   [j]    = tokens[i];
                text_batch.pos     [j]    = n_past++;
                text_batch.n_seq_id[j]    = 1;
                text_batch.seq_id  [j][0] = seq_id;
                text_batch.logits  [j]    = false;

                text_batch.n_tokens++;
            }
            bool is_last_token = (i == n_tokens);
            if (logits_last && is_last_token) {
                text_batch.logits[text_batch.n_tokens - 1] = true;
            }
            ret = llama_decode(lctx, text_batch);
            if (ret != 0) {
                LOG_ERR("failed to decode text\n");
                llama_batch_free(text_batch);
                return ret;
            }
            *new_n_past += text_batch.n_tokens;
        }

    } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        const char * name = chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
        int64_t t0 = ggml_time_ms();

        LOG_INF("encoding %s slice...\n", name);

        ret = mtmd_encode_chunk(ctx, chunk);
        if (ret != 0) {
            LOG_ERR("failed to encode %s slice\n", name);
            llama_batch_free(text_batch);
            return ret;
        }

        LOG_INF("%s slice encoded in %" PRId64 " ms\n", name, ggml_time_ms() - t0);

        float * embd = mtmd_get_output_embd(ctx);
        ret = mtmd_helper_decode_image_chunk(ctx, lctx, chunk, embd, n_past, seq_id, n_batch, new_n_past);
        if (ret != 0) {
            LOG_ERR("failed to decode %s\n", name);
            llama_batch_free(text_batch);
            return ret;
        }
    } else {
        GGML_ABORT("chunk type not supported");
    }

    llama_batch_free(text_batch);
    return 0;
}

int32_t mtmd_helper_eval_chunks(mtmd_context * ctx,
                                struct llama_context * lctx,
                                const mtmd_input_chunks * chunks,
                                llama_pos n_past,
                                llama_seq_id seq_id,
                                int32_t n_batch,
                                bool logits_last,
                                llama_pos * new_n_past) {
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    if (n_chunks == 0) {
        LOG_WRN("no chunks to eval\n");
        return 0;
    }

    for (size_t i = 0; i < n_chunks; i++) {
        bool chunk_logits_last = (i == n_chunks - 1) && logits_last;
        auto chunk = mtmd_input_chunks_get(chunks, i);

        int32_t res = mtmd_helper_eval_chunk_single(ctx, lctx, chunk, n_past, seq_id, n_batch, chunk_logits_last, &n_past);
        if (res != 0) {
            LOG_ERR("failed to eval chunk %zu\n", i);
            return res;
        }
        *new_n_past = n_past;
    }

    return 0;
}


// ============================================================================
// Voxtral Realtime dual-stream evaluation
// ============================================================================

struct voxtral_rt_tok_embd_table {
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
        if (!gguf_ctx) {
            LOG_ERR("voxtral_rt: failed to load GGUF for token embeddings: %s\n", model_path);
            ggml_free(ggml_ctx);
            return false;
        }

        struct ggml_tensor * tensor = ggml_get_tensor(ggml_ctx, "token_embd.weight");
        if (!tensor) {
            LOG_ERR("voxtral_rt: token_embd.weight not found\n");
            gguf_free(gguf_ctx);
            ggml_free(ggml_ctx);
            return false;
        }

        n_embd = (int)tensor->ne[0];
        vocab_size = (int)tensor->ne[1];
        data.resize((size_t)n_embd * vocab_size);

        // dequantize to f32
        const auto * type_traits = ggml_get_type_traits(tensor->type);
        if (tensor->type == GGML_TYPE_F32) {
            memcpy(data.data(), tensor->data, data.size() * sizeof(float));
        } else if (tensor->type == GGML_TYPE_F16) {
            const ggml_fp16_t * src = (const ggml_fp16_t *)tensor->data;
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = ggml_fp16_to_fp32(src[i]);
            }
        } else if (type_traits && type_traits->to_float) {
            for (int row = 0; row < vocab_size; row++) {
                const void * src = (const char *)tensor->data + row * tensor->nb[1];
                type_traits->to_float(src, data.data() + row * n_embd, n_embd);
            }
        } else {
            LOG_ERR("voxtral_rt: unsupported tensor type for token_embd.weight\n");
            gguf_free(gguf_ctx);
            ggml_free(ggml_ctx);
            return false;
        }

        gguf_free(gguf_ctx);
        ggml_free(ggml_ctx);
        LOG_INF("voxtral_rt: loaded token embeddings [%d x %d]\n", vocab_size, n_embd);
        return true;
    }

    const float * get(int token_id) const {
        if (token_id < 0 || token_id >= vocab_size) return nullptr;
        return data.data() + (size_t)token_id * n_embd;
    }
};

int32_t mtmd_helper_eval_voxtral_realtime(
        mtmd_context * ctx,
        struct llama_context * lctx,
        const char * model_path,
        const float * pcm_samples,
        size_t n_samples,
        int32_t n_batch,
        int32_t max_tokens,
        llama_pos * new_n_past,
        std::vector<llama_token> * output_tokens) {

    const llama_model * model = llama_get_model(lctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);

    // Voxtral Realtime constants
    const llama_token TOKEN_BOS = 1;
    const llama_token TOKEN_STREAMING_PAD = 32;
    const int N_LEFT_PAD_TOKENS = 32;
    const int N_DELAY_TOKENS = 6;

    // 1. Load token embedding table
    voxtral_rt_tok_embd_table tok_embd;
    if (!tok_embd.load_from_gguf(model_path)) {
        return -1;
    }

    // 2. Build prefix tokens: [BOS] + PAD*32 + DELAY*6
    std::vector<llama_token> prompt_ids;
    prompt_ids.push_back(TOKEN_BOS);
    for (int i = 0; i < N_LEFT_PAD_TOKENS + N_DELAY_TOKENS; i++) {
        prompt_ids.push_back(TOKEN_STREAMING_PAD);
    }
    int n_prefix = (int)prompt_ids.size();

    // 3. Create audio bitmap and tokenize
    mtmd_bitmap * bmp = mtmd_bitmap_init_from_audio(n_samples, pcm_samples);
    if (!bmp) {
        LOG_ERR("voxtral_rt: failed to create audio bitmap\n");
        return -1;
    }

    std::string prompt = std::string(mtmd_default_marker());
    mtmd_input_text text;
    text.text          = prompt.c_str();
    text.add_special   = true;
    text.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps[] = { bmp };
    int32_t res = mtmd_tokenize(ctx, chunks, &text, bitmaps, 1);
    if (res != 0) {
        LOG_ERR("voxtral_rt: tokenize failed, res = %d\n", res);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        return -1;
    }

    // 4. Find and encode the audio chunk
    const mtmd_input_chunk * audio_chunk = nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        auto chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            audio_chunk = chunk;
            break;
        }
    }
    if (!audio_chunk) {
        LOG_ERR("voxtral_rt: no audio chunk found\n");
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        return -1;
    }

    int64_t t0 = ggml_time_ms();
    res = mtmd_encode_chunk(ctx, audio_chunk);
    if (res != 0) {
        LOG_ERR("voxtral_rt: encode failed\n");
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        return -1;
    }
    LOG_INF("voxtral_rt: audio encoded in %" PRId64 " ms\n", ggml_time_ms() - t0);

    float * audio_embd = mtmd_get_output_embd(ctx);
    int n_audio_tokens = mtmd_input_chunk_get_n_tokens(audio_chunk);

    // Copy audio embeddings (buffer may be reused)
    std::vector<float> audio_embds((size_t)n_audio_tokens * n_embd);
    memcpy(audio_embds.data(), audio_embd, audio_embds.size() * sizeof(float));

    if (n_prefix > n_audio_tokens) {
        LOG_ERR("voxtral_rt: prefix (%d) > audio tokens (%d)\n", n_prefix, n_audio_tokens);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        return -1;
    }

    // 5. Dual-stream: combined[i] = audio_embd[i] + tok_embd[prompt_ids[i]]
    std::vector<float> combined((size_t)n_prefix * n_embd);
    for (int i = 0; i < n_prefix; i++) {
        const float * a = audio_embds.data() + (size_t)i * n_embd;
        const float * t = tok_embd.get(prompt_ids[i]);
        float * dst = combined.data() + (size_t)i * n_embd;
        for (int j = 0; j < n_embd; j++) {
            dst[j] = a[j] + t[j];
        }
    }

    // 6. Prefill: send combined embeddings to decoder
    llama_pos n_past = 0;
    for (int offset = 0; offset < n_prefix; offset += n_batch) {
        int this_batch = std::min(n_batch, n_prefix - offset);
        llama_batch batch = llama_batch_init(this_batch, n_embd, 1);
        batch.n_tokens = this_batch;
        for (int i = 0; i < this_batch; i++) {
            memcpy(batch.embd + (size_t)i * n_embd,
                   combined.data() + (size_t)(offset + i) * n_embd,
                   n_embd * sizeof(float));
            batch.pos[i]      = n_past + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]   = (offset + i == n_prefix - 1) ? 1 : 0;
        }
        res = llama_decode(lctx, batch);
        llama_batch_free(batch);
        if (res != 0) {
            LOG_ERR("voxtral_rt: prefill decode failed\n");
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bmp);
            return res;
        }
    }
    n_past += n_prefix;

    LOG_INF("voxtral_rt: prefill done, n_past = %d, n_audio_tokens = %d\n", (int)n_past, n_audio_tokens);

    // 7. Autoregressive decoding with dual-stream continuation
    const llama_token eos_token = llama_vocab_eos(vocab);
    int n_decoded = 0;

    for (int i = 0; i < max_tokens && n_decoded < max_tokens; i++) {
        // sample from logits
        const float * logits = llama_get_logits_ith(lctx, -1);
        if (!logits) {
            LOG_ERR("voxtral_rt: failed to get logits\n");
            break;
        }

        // greedy sampling (take argmax)
        int n_vocab_size = llama_vocab_n_tokens(vocab);
        llama_token best_token = 0;
        float best_logit = logits[0];
        for (int v = 1; v < n_vocab_size; v++) {
            if (logits[v] > best_logit) {
                best_logit = logits[v];
                best_token = v;
            }
        }

        if (best_token == eos_token) break;

        if (output_tokens) {
            output_tokens->push_back(best_token);
        }
        n_decoded++;

        // next step: dual-stream if still within audio range, text-only otherwise
        std::vector<float> next_embd(n_embd);
        const float * t = tok_embd.get(best_token);
        if (!t) {
            LOG_ERR("voxtral_rt: invalid token %d\n", best_token);
            break;
        }

        if (n_past < n_audio_tokens) {
            const float * a = audio_embds.data() + (size_t)n_past * n_embd;
            for (int j = 0; j < n_embd; j++) {
                next_embd[j] = a[j] + t[j];
            }
        } else {
            memcpy(next_embd.data(), t, n_embd * sizeof(float));
        }

        llama_batch batch = llama_batch_init(1, n_embd, 1);
        batch.n_tokens     = 1;
        memcpy(batch.embd, next_embd.data(), n_embd * sizeof(float));
        batch.pos[0]       = n_past;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;

        res = llama_decode(lctx, batch);
        llama_batch_free(batch);
        if (res != 0) {
            LOG_ERR("voxtral_rt: decode step %d failed\n", n_decoded);
            break;
        }
        n_past++;
    }

    *new_n_past = n_past;
    LOG_INF("voxtral_rt: decoded %d tokens\n", n_decoded);

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);
    return 0;
}

namespace audio_helpers {

static bool is_audio_file(const char * buf, size_t len) {
    if (len < 12) {
        return false;
    }

    // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
    // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    bool is_wav = memcmp(buf, "RIFF", 4) == 0 && memcmp(buf + 8, "WAVE", 4) == 0;
    bool is_mp3 = len >= 3 && (
        memcmp(buf, "ID3", 3) == 0 ||
        // Check for MPEG sync word (simplified check)
        ((unsigned char)buf[0] == 0xFF && ((unsigned char)buf[1] & 0xE0) == 0xE0)
    );
    bool is_flac = memcmp(buf, "fLaC", 4) == 0;

    return is_wav || is_mp3 || is_flac;
}

// returns true if the buffer is a valid audio file
static bool decode_audio_from_buf(const unsigned char * buf_in, size_t len, int target_sampler_rate, std::vector<float> & pcmf32_mono) {
    ma_result result;
    const int channels = 1;
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, channels, target_sampler_rate);
    ma_decoder decoder;

    result = ma_decoder_init_memory(buf_in, len, &decoder_config, &decoder);
    if (result != MA_SUCCESS) {
        return false;
    }

    ma_uint64 frame_count;
    ma_uint64 frames_read;
    result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

    pcmf32_mono.resize(frame_count);
    result = ma_decoder_read_pcm_frames(&decoder, pcmf32_mono.data(), frame_count, &frames_read);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

#ifdef MTMD_AUDIO_DEBUG
    // save audio to wav file
    ma_encoder_config config = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, target_sampler_rate);
    ma_encoder encoder;
    ma_encoder_init_file("output.wav", &config, &encoder);
    ma_encoder_write_pcm_frames(&encoder, pcmf32_mono.data(), pcmf32_mono.size(), &frames_read);
    ma_encoder_uninit(&encoder);
#endif

    ma_decoder_uninit(&decoder);
    return true;
}

} // namespace audio_helpers

mtmd_bitmap * mtmd_helper_bitmap_init_from_buf(mtmd_context * ctx, const unsigned char * buf, size_t len) {
    if (audio_helpers::is_audio_file((const char *)buf, len)) {
        std::vector<float> pcmf32;
        const int sample_rate = mtmd_get_audio_sample_rate(ctx);
        if (sample_rate < 0) {
            LOG_ERR("This model does not support audio input\n");
            return nullptr;
        }
        if (!audio_helpers::decode_audio_from_buf(buf, len, sample_rate, pcmf32)) {
            LOG_ERR("Unable to read WAV audio file from buffer\n");
            return nullptr;
        }
        return mtmd_bitmap_init_from_audio(pcmf32.size(), pcmf32.data());
    }

    // otherwise, we assume it's an image
    mtmd_bitmap * result = nullptr;
    {
        int nx, ny, nc;
        auto * data = stbi_load_from_memory(buf, len, &nx, &ny, &nc, 3);
        if (!data) {
            LOG_ERR("%s: failed to decode image bytes\n", __func__);
            return nullptr;
        }
        result = mtmd_bitmap_init(nx, ny, data);
        stbi_image_free(data);
    }
    return result;
}

mtmd_bitmap * mtmd_helper_bitmap_init_from_file(mtmd_context * ctx, const char * fname) {
    std::vector<unsigned char> buf;
    FILE * f = fopen(fname, "rb");
    if (!f) {
        LOG_ERR("Unable to open file %s: %s\n", fname, strerror(errno));
        return nullptr;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf.resize(file_size);

    size_t n_read = fread(buf.data(), 1, file_size, f);
    fclose(f);
    if (n_read != (size_t)file_size) {
        LOG_ERR("Failed to read entire file %s", fname);
        return nullptr;
    }

    return mtmd_helper_bitmap_init_from_buf(ctx, buf.data(), buf.size());
}
