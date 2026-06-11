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

#ifdef MTMD_VIDEO
#include "sheredom/subprocess.h"
#include <thread>
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

#define LOG_DBG(...) g_logger.log(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
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

void mtmd_helper_image_get_decoder_pos(const mtmd_image_tokens * chunks, llama_pos pos_0, mtmd_decoder_pos * out_pos) {
    size_t n_tokens = mtmd_image_tokens_get_n_tokens(chunks);
    for (size_t i = 0; i < n_tokens; i++) {
        out_pos[i] = mtmd_image_tokens_get_decoder_pos(chunks, pos_0, i);
    }
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
        GGML_ASSERT(n_tokens > 0 && n_pos_per_embd > 0 && n_mmproj_embd > 0);
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
    void set_position_mrope_2d(const std::vector<mtmd_decoder_pos> & rel_pos, llama_seq_id seq_id) {
        GGML_ASSERT(n_pos_per_embd == 4);
        GGML_ASSERT(!rel_pos.empty() && (int32_t)rel_pos.size() == batch.n_tokens);
        seq_id_0[0] = seq_id;
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            pos[i                     ] = rel_pos[i].t;
            pos[i + batch.n_tokens    ] = rel_pos[i].y;
            pos[i + batch.n_tokens * 2] = rel_pos[i].x;
            pos[i + batch.n_tokens * 3] = rel_pos[i].z;
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
            pos[i + batch.n_tokens * 3] = pos_0 + i;
        }
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    llama_batch get_view(int offset, int n_tokens) {
        GGML_ASSERT(offset >= 0 && n_tokens > 0 && offset + n_tokens <= batch.n_tokens);
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
    GGML_ASSERT(n_batch > 0);
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
            const auto n_tokens = mtmd_image_tokens_get_n_tokens(image_tokens);
            std::vector<mtmd_decoder_pos> rel_pos(n_tokens);
            mtmd_helper_image_get_decoder_pos(image_tokens, n_past, rel_pos.data());
            batch_embd.set_position_mrope_2d(rel_pos, seq_id);
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            batch_embd.set_position_mrope_1d(n_past, seq_id);
        } else {
            GGML_ABORT("invalid chunk type for M-RoPE");
        }
    } else {
        batch_embd.set_position_normal(n_past, seq_id);
    }

    const bool use_non_causal = mtmd_decode_use_non_causal(ctx, chunk);
    if (use_non_causal) {
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

    if (use_non_causal) {
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
    GGML_ASSERT(n_batch > 0);
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

// Computes FNV-1a hash of the data
static std::string fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return std::to_string(hash);
}

mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf(mtmd_context * ctx, const unsigned char * buf, size_t len, bool placeholder) {
    // calculate the hash if needed
    std::string id;
    mtmd_bitmap * result = nullptr;

    if (!placeholder) {
        id = fnv_hash(buf, len);
    }

    if (audio_helpers::is_audio_file((const char *)buf, len)) {
        std::vector<float> pcmf32;
        const int sample_rate = mtmd_get_audio_sample_rate(ctx);
        if (sample_rate < 0) {
            LOG_ERR("This model does not support audio input\n");
            return {nullptr, nullptr};
        }
        if (!audio_helpers::decode_audio_from_buf(buf, len, sample_rate, pcmf32)) {
            LOG_ERR("Unable to read WAV audio file from buffer\n");
            return {nullptr, nullptr};
        }
        result = mtmd_bitmap_init_from_audio(pcmf32.size(), placeholder ? nullptr : pcmf32.data());
        mtmd_bitmap_set_id(result, id.empty() ? nullptr : id.c_str());
        return {result, nullptr};
    }

    // otherwise, we assume it's an image
    if (!result) {
        int nx, ny, nc;
        auto * data = stbi_load_from_memory(buf, len, &nx, &ny, &nc, 3);
        if (data) {
            result = mtmd_bitmap_init(nx, ny, placeholder ? nullptr : data);
            mtmd_bitmap_set_id(result, id.empty() ? nullptr : id.c_str());
            stbi_image_free(data);
            return {result, nullptr};
        }
        // otherwise, fallthrough to video decoding (if supported)
    }

    // last try: load as video
#ifdef MTMD_VIDEO
    if (!result) {
        auto params = mtmd_helper_video_init_params_default();
        auto video_ctx = mtmd_helper_video_init_from_buf(ctx, buf, len, params);
        if (!video_ctx) {
            LOG_ERR("%s: failed to decode buffer as either image/audio/video\n", __func__);
            return {nullptr, nullptr};
        }
        result = mtmd_bitmap_init_lazy(ctx,
            id.empty() ? nullptr : id.c_str(),
            video_ctx,
            [](size_t, void * user_data, mtmd_bitmap ** out_bitmap, char ** out_text) -> int {
                auto * vctx = static_cast<mtmd_helper_video *>(user_data);
                char * text = nullptr;
                int ret = mtmd_helper_video_read_next(vctx, out_bitmap, &text);
                *out_text = text; // heap-allocated by read_next; freed automatically by mtmd
                return ret;
            });
         return {result, video_ctx};
    }
#else
    if (!result) {
        LOG_ERR("%s: failed to decode buffer as either image or audio (video support not compiled in)\n", __func__);
        return {nullptr, nullptr};
    }
#endif

    // should not reach here
    return {nullptr, nullptr};
}

mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_file(mtmd_context * ctx, const char * fname, bool placeholder) {
    std::vector<unsigned char> buf;
    FILE * f = fopen(fname, "rb");
    if (!f) {
        LOG_ERR("Unable to open file %s: %s\n", fname, strerror(errno));
        return {nullptr, nullptr};
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (file_size < 0) {
        LOG_ERR("Failed to get file size of %s\n", fname);
        fclose(f);
        return {nullptr, nullptr};
    }
    buf.resize(file_size);

    size_t n_read = fread(buf.data(), 1, file_size, f);
    fclose(f);
    if (n_read != (size_t)file_size) {
        LOG_ERR("Failed to read entire file %s", fname);
        return {nullptr, nullptr};
    }

    return mtmd_helper_bitmap_init_from_buf(ctx, buf.data(), buf.size(), placeholder);
}

bool mtmd_helper_support_video(mtmd_context * ctx) {
#ifdef MTMD_VIDEO
    return mtmd_support_vision(ctx);
#else
    return false;
#endif
}

//
// Video input helpers
//

#ifdef MTMD_VIDEO

struct mtmd_helper_video {
    mtmd_context * mctx;
    std::string path;
    std::vector<uint8_t> input_buf; // non-empty when initialized from buffer
    std::string ffmpeg_bin;
    std::string ffprobe_bin;
    float fps_target = 0.0f;
    mtmd_helper_video_info info = {};

    // RAII wrapper for managing subprocess
    struct subprocess_handle {
        struct subprocess_s proc = {};
        bool alive = false;
        std::thread feeder;

        subprocess_handle() = default;
        subprocess_handle(const subprocess_handle &) = delete;
        subprocess_handle & operator=(const subprocess_handle &) = delete;
        ~subprocess_handle() { stop(); }

        void stop() {
            if (alive) {
                subprocess_terminate(&proc);
            }
            // join before destroy: feeder holds a FILE* from subprocess_stdin;
            // subprocess_destroy closes it, so the thread must finish first
            if (feeder.joinable()) {
                feeder.join();
            }
            if (alive) {
                subprocess_destroy(&proc);
                alive = false;
            }
        }

        FILE * stdout_pipe() {
            return subprocess_stdout(&proc);
        }

        // buf is tied to lifetime of mtmd_helper_video, so it's guaranteed to outlive the feeder thread
        void start_feeder(const std::vector<uint8_t> & buf) {
            feeder = std::thread([this, &buf]() {
                FILE * f = subprocess_stdin(&proc);
                if (!f) {
                    return;
                }
                fwrite(buf.data(), 1, buf.size(), f);
                fclose(f);
                proc.stdin_file = nullptr; // prevent double-close in subprocess_destroy
            });
        }
    };

    subprocess_handle sp;
    int32_t current_frame = 0;

    std::string prompt_start         = "Video:";
    int32_t     timestamp_interval_ms = 5000; // emit a timestamp text every N ms (0 = disabled)
    float       next_timestamp_ms     = 0.0f; // next elapsed-ms threshold at which to emit

    std::vector<uint8_t> frame_buf;
    std::string pending_text; // text queued to be returned before the next frame
    bool        start_emitted = false;

    bool is_buf_input() const {
        return !input_buf.empty();
    }

    bool probe(float fps_target_arg) {
        const char * input_arg = is_buf_input() ? "pipe:0" : path.c_str();
        const char * cmd[] = {
            ffprobe_bin.c_str(),
            "-v", "quiet",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
            "-select_streams", "v:0",
            "-of", "default=noprint_wrappers=1",
            input_arg,
            nullptr,
        };

        LOG_DBG("%s: launching:", __func__);
        for (size_t i = 0; cmd[i]; i++) { LOG_DBG(" %s", cmd[i]); }
        LOG_DBG("\n");

        subprocess_handle probe_sp;
        if (subprocess_create(cmd,
                subprocess_option_search_user_path | subprocess_option_inherit_environment,
                &probe_sp.proc) != 0) {
            LOG_ERR("%s: failed to launch ffprobe\n", __func__);
            return false;
        }
        probe_sp.alive = true;

        if (is_buf_input()) {
            probe_sp.start_feeder(input_buf);
        }

        uint32_t width  = 0;
        uint32_t height = 0;
        float orig_fps = 0.0f;
        float duration = -1.0f;
        int32_t n_frames_orig = -1;
        char line[256];
        FILE * fp = probe_sp.stdout_pipe();

        while (fgets(line, sizeof(line), fp)) {
            char * eq = strchr(line, '=');
            if (!eq) continue;
            *eq = '\0';
            const char * key = line;
            const char * val = eq + 1;
            char * nl = (char *)strchr(val, '\n');
            if (nl) *nl = '\0';

            if (strcmp(key, "width") == 0) {
                width = (uint32_t)atoi(val);
            } else if (strcmp(key, "height") == 0) {
                height = (uint32_t)atoi(val);
            } else if (strcmp(key, "r_frame_rate") == 0) {
                orig_fps = parse_rational(val);
            } else if (strcmp(key, "nb_frames") == 0 && strcmp(val, "N/A") != 0) {
                n_frames_orig = atoi(val);
            } else if (strcmp(key, "duration") == 0 && strcmp(val, "N/A") != 0) {
                duration = (float)atof(val);
            }
        }

        probe_sp.stop();

        if (width == 0 || height == 0 || orig_fps <= 0.0f) {
            return false;
        }

        if (duration < 0.0f && n_frames_orig > 0) {
            duration = (float)n_frames_orig / orig_fps;
        }

        fps_target = fps_target_arg > 0.0f ? fps_target_arg : orig_fps;
        info.width    = width;
        info.height   = height;
        info.fps      = fps_target;
        LOG_DBG("%s: %ux%u fps=%.2f duration=%.2fs n_frames=%d\n",
                __func__, width, height, fps_target, duration, info.n_frames);
        info.n_frames = duration > 0.0f ? (int32_t)(duration * fps_target + 0.5f) : -1;
        frame_buf.resize((size_t)width * height * 3);
        return true;
    }

    bool start_ffmpeg(float seek_seconds) {
        char seek_buf[64];
        char fps_buf[64];

        std::vector<const char *> cmd;
        cmd.push_back(ffmpeg_bin.c_str());

        if (!is_buf_input() && seek_seconds > 0.0f) {
            // input-side seek: fast, keyframe-accurate; only valid for seekable file inputs
            snprintf(seek_buf, sizeof(seek_buf), "%.6f", seek_seconds);
            cmd.push_back("-ss");
            cmd.push_back(seek_buf);
        }

        cmd.push_back("-nostdin");
        cmd.push_back("-i");
        // cache:pipe:0 wraps stdin with a seekable in-memory cache, letting ffmpeg seek
        // backwards for container headers (e.g. MP4 moov atom at end of file)
        cmd.push_back(is_buf_input() ? "cache:pipe:0" : path.c_str());

        if (seek_seconds > 0.0f && is_buf_input()) {
            // output-side seek: frame-accurate but decodes and discards frames up to seek point
            snprintf(seek_buf, sizeof(seek_buf), "%.6f", seek_seconds);
            cmd.push_back("-ss");
            cmd.push_back(seek_buf);
        }

        if (fps_target > 0.0f) {
            snprintf(fps_buf, sizeof(fps_buf), "fps=%.6f", fps_target);
            cmd.push_back("-vf");
            cmd.push_back(fps_buf);
        }

        cmd.push_back("-f");
        cmd.push_back("rawvideo");
        cmd.push_back("-pix_fmt");
        cmd.push_back("rgb24");
        cmd.push_back("pipe:1");
        cmd.push_back("-loglevel");
        cmd.push_back("error");
        cmd.push_back(nullptr);

        LOG_DBG("%s: launching:", __func__);
        for (size_t i = 0; cmd[i]; i++) {
            LOG_DBG(" %s", cmd[i]);
        }
        LOG_DBG("\n");

        int ret = subprocess_create(
            cmd.data(),
            subprocess_option_search_user_path | subprocess_option_inherit_environment,
            &sp.proc);

        sp.alive = (ret == 0);
        LOG_DBG("%s: subprocess_create ret=%d proc_alive=%d\n", __func__, ret, (int)sp.alive);

        if (sp.alive && is_buf_input()) {
            LOG_DBG("%s: starting feeder thread for %zu-byte buffer\n", __func__, input_buf.size());
            sp.start_feeder(input_buf);
        }

        return sp.alive;
    }

    void stop_ffmpeg() {
        sp.stop();
    }

    mtmd_bitmap * read_next_frame() {
        if (!sp.alive) return nullptr;

        FILE * fp = sp.stdout_pipe();
        const size_t frame_size = (size_t)info.width * info.height * 3;
        LOG_DBG("%s: reading frame %d, expecting %zu bytes (%ux%u)\n",
                __func__, current_frame, frame_size, info.width, info.height);

        size_t total_read = 0;
        while (total_read < frame_size) {
            size_t n = fread(frame_buf.data() + total_read, 1, frame_size - total_read, fp);
            if (n == 0) {
                // clean EOF only if no bytes read yet; partial frame is an error
                LOG_DBG("%s: fread returned 0 after %zu/%zu bytes (ferror=%d)\n",
                        __func__, total_read, frame_size, ferror(fp));
                sp.alive = false;
                return nullptr;
            }
            total_read += n;
        }

        LOG_DBG("%s: frame %d read OK\n", __func__, current_frame);
        current_frame++;
        return mtmd_bitmap_init(info.width, info.height, frame_buf.data());
    }

    int32_t read_next(mtmd_bitmap ** out_bitmap, char ** out_text) {
        *out_bitmap = nullptr;
        *out_text   = nullptr;

        if (!pending_text.empty()) {
            *out_text = strdup(pending_text.c_str());
            pending_text.clear();
            return *out_text ? 0 : -2;
        }

        LOG_DBG("%s: proc_alive=%d start_emitted=%d current_frame=%d\n",
                __func__, (int)sp.alive, (int)start_emitted, current_frame);

        if (!sp.alive) {
            return (current_frame == 0) ? -2 : -1;
        }

        if (!start_emitted) {
            start_emitted = true;
            if (!prompt_start.empty()) {
                *out_text = strdup(prompt_start.c_str());
                return *out_text ? 0 : -2;
            }
        }

        mtmd_bitmap * frame = read_next_frame();
        if (!frame) return -1;
        *out_bitmap = frame;

        if (timestamp_interval_ms > 0) {
            // current_frame was already incremented by read_next_frame(); undo for elapsed calc
            float elapsed_ms = (float)(current_frame - 1) / info.fps * 1000.0f;
            if (elapsed_ms >= next_timestamp_ms) {
                char ts_buf[32];
                float elapsed_s = elapsed_ms / 1000.0f;
                int   minutes   = (int)(elapsed_s / 60);
                float seconds   = elapsed_s - minutes * 60.0f;
                snprintf(ts_buf, sizeof(ts_buf), "[%dm%.2fs]", minutes, seconds);
                pending_text = ts_buf;
                next_timestamp_ms += (float)timestamp_interval_ms;
            }
        }

        return 0;
    }

    static float parse_rational(const char * s) {
        int num = 0, den = 1;
        if (sscanf(s, "%d/%d", &num, &den) == 2 && den > 0) {
            return (float)num / (float)den;
        }
        float val;
        if (sscanf(s, "%f", &val) == 1) {
            return val;
        }
        return 0.0f;
    }
};
#endif

mtmd_helper_video_init_params mtmd_helper_video_init_params_default() {
    return {
        /* fps_target             */ 4.0f,
        /* ffmpeg_bin_dir         */ nullptr,
        /* timestamp_interval_ms  */ 5000,
    };
}

static std::string video_resolve_bin(const char * bin_dir, const char * name) {
    if (!bin_dir || bin_dir[0] == '\0') {
        return name; // rely on PATH
    }
    std::string result = bin_dir;
    char last = result.back();
    if (last != '/' && last != '\\') {
#ifdef _WIN32
        result += '\\';
#else
        result += '/';
#endif
    }
    result += name;
#ifdef _WIN32
    result += ".exe";
#endif
    return result;
}

mtmd_helper_video * mtmd_helper_video_init(
        mtmd_context * mctx,
        const char * path,
        mtmd_helper_video_init_params params) {
#ifdef MTMD_VIDEO
    auto * ctx = new mtmd_helper_video();

    ctx->mctx                 = mctx;
    ctx->path                 = path;
    ctx->ffmpeg_bin           = video_resolve_bin(params.ffmpeg_bin_dir, "ffmpeg");
    ctx->ffprobe_bin          = video_resolve_bin(params.ffmpeg_bin_dir, "ffprobe");
    ctx->timestamp_interval_ms = params.timestamp_interval_ms;

    if (!ctx->probe(params.fps_target)) {
        LOG_ERR("%s: ffprobe failed for '%s' (is ffprobe in PATH?)\n", __func__, path);
        delete ctx;
        return nullptr;
    }

    if (!ctx->start_ffmpeg(0.0f)) {
        LOG_ERR("%s: failed to start ffmpeg for '%s' (is ffmpeg in PATH?)\n", __func__, path);
        delete ctx;
        return nullptr;
    }

    return ctx;
#else
    LOG_ERR("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", __func__);
    return nullptr;
#endif
}

mtmd_helper_video * mtmd_helper_video_init_from_buf(
        mtmd_context * mctx,
        const unsigned char * buf, size_t len,
        mtmd_helper_video_init_params params) {
#ifdef MTMD_VIDEO
    auto * ctx = new mtmd_helper_video();

    ctx->mctx                  = mctx;
    ctx->input_buf.assign(buf, buf + len);
    ctx->ffmpeg_bin            = video_resolve_bin(params.ffmpeg_bin_dir, "ffmpeg");
    ctx->ffprobe_bin           = video_resolve_bin(params.ffmpeg_bin_dir, "ffprobe");
    ctx->timestamp_interval_ms = params.timestamp_interval_ms;

    if (!ctx->probe(params.fps_target)) {
        LOG_ERR("%s: ffprobe failed on buffer (is ffprobe in PATH?)\n", __func__);
        delete ctx;
        return nullptr;
    }

    if (!ctx->start_ffmpeg(0.0f)) {
        LOG_ERR("%s: failed to start ffmpeg on buffer (is ffmpeg in PATH?)\n", __func__);
        delete ctx;
        return nullptr;
    }

    return ctx;
#else
    LOG_ERR("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", __func__);
    return nullptr;
#endif
}

void mtmd_helper_video_free(mtmd_helper_video * ctx) {
#ifdef MTMD_VIDEO
    if (!ctx) return;
    ctx->stop_ffmpeg();
    delete ctx;
#else
    LOG_ERR("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", __func__);
#endif
}

mtmd_helper_video_info mtmd_helper_video_get_info(const mtmd_helper_video * ctx) {
#ifdef MTMD_VIDEO
    return ctx->info;
#else
    GGML_ASSERT(false && "video is not supported in this build (MTMD_VIDEO is set to OFF)");
#endif
}

int32_t mtmd_helper_video_read_next(mtmd_helper_video * ctx,
        mtmd_bitmap ** out_bitmap, char ** out_text) {
#ifdef MTMD_VIDEO
    if (!ctx) return -2;
    return ctx->read_next(out_bitmap, out_text);
#else
    GGML_ASSERT(false && "video is not supported in this build (MTMD_VIDEO is set to OFF)");
#endif
}
