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
#include "mtmd-helper-common.h"
#include "llama.h"

#include "hash/hash.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
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
#ifndef _WIN32
#include <csignal>
#include <fcntl.h>
#include <pthread.h>
#endif
#endif

//
// internal logging functions
//

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

// Helper class to set non-causal attention via RAII
class scope_non_causal {
public:
    scope_non_causal(llama_context * context, bool enabled) : context_(context), enabled_(enabled) {
        if (enabled_) {
            // TODO @ngxson : need to make sure only one image is processed at a time, and n_ubatch must be enough to hold the image
            llama_set_causal_attn(context_, false);
        }
    }
    ~scope_non_causal() {
        if (enabled_) {
            llama_set_causal_attn(context_, true);
        }
    }

    scope_non_causal(const scope_non_causal &) = delete;
    scope_non_causal & operator=(const scope_non_causal &) = delete;

private:
    llama_context * context_;
    bool enabled_;
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
        llama_pos * new_n_past,
        mtmd_helper_post_decode_callback callback,
        void * user_data) {
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
    const scope_non_causal non_causal(lctx, use_non_causal);

    while (i_batch < n_img_batches) { // split into batches
        int pos_offset = i_batch*n_batch;
        int n_tokens_batch = std::min(n_batch, n_tokens - pos_offset);

        LOG_INF("decoding %s batch %d/%d, n_tokens_batch = %d\n", name, i_batch+1, n_img_batches, n_tokens_batch);

        int64_t t1 = ggml_time_ms();
        int32_t ret = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch_embd.render(lctx, pos_offset, n_tokens_batch));
        if (ret != 0) {
            LOG_ERR("failed to decode %s\n", name);
            return ret;
        }

        if (callback != nullptr) {
            const mtmd_helper_embd_batch view = batch_embd.get_view(pos_offset, n_tokens_batch);
            ret = callback(&view, user_data);
            if (ret != 0) {
                LOG_ERR("post-decode callback failed\n");
                return ret;
            }
        }

        LOG_INF("%s decoded (batch %d/%d) in %" PRId64 " ms\n", name, i_batch+1, n_img_batches, ggml_time_ms() - t1);

        i_batch++;
    }

    n_past += mtmd_input_chunk_get_n_pos(chunk);
    *new_n_past = n_past;

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
    auto chunk_type = mtmd_input_chunk_get_type(chunk);

    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        size_t n_tokens;
        const auto tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
        // LOG_INF("decoding text chunk, n_tokens = %zu\n", n_tokens);
        llama_batch_ext_ptr text_batch(llama_batch_ext_init(lctx));
        size_t i = 0;
        while (i < n_tokens) { // split into batches
            llama_batch_ext_clear(text_batch.get());
            int32_t n_added = 0;
            int32_t idx     = -1;
            for (; i < n_tokens && n_added < n_batch; i++) {
                idx = llama_batch_ext_add_token(text_batch.get(), seq_id, tokens[i]);
                GGML_ASSERT(idx >= 0);
                llama_pos pos = n_past++;
                llama_batch_ext_set_pos(text_batch.get(), idx, &pos);
                n_added++;
            }
            bool is_last_token = (i == n_tokens);
            if (logits_last && is_last_token) {
                llama_batch_ext_set_output_logits(text_batch.get(), idx, true);
            }
            ret = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, text_batch.get());
            if (ret != 0) {
                LOG_ERR("failed to decode text\n");
                return ret;
            }
            *new_n_past += n_added;
        }

    } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        const char * name = chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
        int64_t t0 = ggml_time_ms();

        LOG_INF("encoding %s slice...\n", name);

        ret = mtmd_encode_chunk(ctx, chunk);
        if (ret != 0) {
            LOG_ERR("failed to encode %s slice\n", name);
            return ret;
        }

        LOG_INF("%s slice encoded in %" PRId64 " ms\n", name, ggml_time_ms() - t0);

        float * embd = mtmd_get_output_embd(ctx);
        ret = mtmd_helper_decode_image_chunk(ctx, lctx, chunk, embd, n_past, seq_id, n_batch, new_n_past, nullptr, nullptr);
        if (ret != 0) {
            LOG_ERR("failed to decode %s\n", name);
            return ret;
        }
    } else {
        GGML_ABORT("chunk type not supported");
    }

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

static bool is_webp_file(const unsigned char * buf, size_t len) {
    // WEBP ref: https://developers.google.com/speed/webp/docs/riff_container
    return len >= 12 && memcmp(buf, "RIFF", 4) == 0 && memcmp(buf + 8, "WEBP", 4) == 0;
}

#ifdef MTMD_VIDEO
static mtmd_bitmap * decode_webp_with_ffmpeg(const mtmd_context * mctx, const unsigned char * buf, size_t len, bool placeholder,
                                             const mtmd_helper_video_init_params & params);
static void mtmd_helper_video_set_id(mtmd_helper_video * vctx, const std::string & id);
#endif

mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf(const mtmd_context * ctx, const unsigned char * buf, size_t len, bool placeholder,
                                                            mtmd_helper_init_opt opt) {
    // calculate the hash if needed
    std::string id;
    mtmd_bitmap * result = nullptr;

    GGML_UNUSED(opt); // only used by video code paths

    if (!placeholder) {
        // use sha256 to prevent cache poisoning
        id = hash_sha256_hex(buf, len);
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

#ifdef MTMD_VIDEO
    // stb_image does not support webp; decode it with ffmpeg as a single frame
    if (!result && is_webp_file(buf, len)) {
        result = decode_webp_with_ffmpeg(ctx, buf, len, placeholder, opt.video_params);
        if (!result) {
            LOG_ERR("%s: failed to decode webp buffer\n", __func__);
            return {nullptr, nullptr};
        }
        mtmd_bitmap_set_id(result, id.empty() ? nullptr : id.c_str());
        return {result, nullptr};
    }
#endif

    // last try: load as video
#ifdef MTMD_VIDEO
    if (!result) {
        auto video_ctx = mtmd_helper_video_init_from_buf(ctx, buf, len, opt.video_params);
        if (!video_ctx) {
            LOG_ERR("%s: failed to decode buffer as either image/audio/video\n", __func__);
            return {nullptr, nullptr};
        }
        mtmd_helper_video_set_id(video_ctx, id); // propagate the hash to the frames
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

mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_file(const mtmd_context * ctx, const char * fname, bool placeholder,
                                                             mtmd_helper_init_opt opt) {
#ifdef _WIN32
    int wlen = MultiByteToWideChar(CP_UTF8, 0, fname, -1, NULL, 0);
    if (!wlen) {
        LOG_ERR("Unable to convert filename to UTF-16: %s\n", fname);
        return {nullptr, nullptr};
    }
    std::vector<wchar_t> wfname(wlen);
    wlen = MultiByteToWideChar(CP_UTF8, 0, fname, -1, wfname.data(), wlen);
    if (!wlen) {
        LOG_ERR("Unable to convert filename to UTF-16: %s\n", fname);
        return {nullptr, nullptr};
    }
    FILE * f = _wfopen(wfname.data(), L"rb");
#else
    FILE * f = fopen(fname, "rb");
#endif
    if (!f) {
        LOG_ERR("Unable to open file %s: %s\n", fname, strerror(errno));
        return {nullptr, nullptr};
    }

    std::vector<unsigned char> buf;

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

    return mtmd_helper_bitmap_init_from_buf(ctx, buf.data(), buf.size(), placeholder, opt);
}

bool mtmd_helper_support_video(const mtmd_context * ctx) {
#ifdef MTMD_VIDEO
    return mtmd_support_vision(ctx);
#else
    GGML_UNUSED(ctx);
    return false;
#endif
}

//
// Video input helpers
//

#ifdef MTMD_VIDEO

// MSVC deprecates strdup(), use a std::string version directly
static char * mtmd_strdup(const std::string & s) {
    size_t n = s.size() + 1;
    char * p = (char *) malloc(n);
    if (p) {
        memcpy(p, s.c_str(), n);
    }
    return p;
}

struct mtmd_helper_video {
    const mtmd_context * mctx;
    std::string path;
    std::vector<uint8_t> input_buf; // non-empty when initialized from buffer
    std::string ffmpeg_bin;
    std::string ffprobe_bin;
    float fps_target = 0.0f;
    mtmd_helper_video_info info = {};
    std::string id; // hash of the input video

    // RAII wrapper for managing subprocess
    struct subprocess_handle {
        struct subprocess_s proc = {};
        bool created = false; // process exists and must be cleaned up
        bool alive   = false; // process can still give us data
        std::thread feeder;

        subprocess_handle() = default;
        subprocess_handle(const subprocess_handle &) = delete;
        subprocess_handle & operator=(const subprocess_handle &) = delete;
        ~subprocess_handle() { stop(); }

        void stop() {
            // note: alive becomes false on stdout EOF, but the process still needs cleanup
            if (!created) {
                return;
            }
            subprocess_terminate(&proc);
#ifdef _WIN32
            // no SIGPIPE on windows: a blocked feeder only gets a broken pipe once we close our read end of the child stdin
            if (proc.hStdInput) {
                CloseHandle(proc.hStdInput);
                proc.hStdInput = nullptr;
            }
#endif
            // join before destroy: feeder holds a FILE* from subprocess_stdin;
            // subprocess_destroy closes it, so the thread must finish first
            if (feeder.joinable()) {
                feeder.join();
            }
            subprocess_join(&proc, nullptr); // reap the child, or else it stays a zombie
            subprocess_destroy(&proc);
            created = false;
            alive   = false;
        }

        FILE * stdout_pipe() {
            return subprocess_stdout(&proc);
        }

        // buf is tied to lifetime of mtmd_helper_video, so it's guaranteed to outlive the feeder thread
        void start_feeder(const std::vector<uint8_t> & buf) {
            feeder = std::thread([this, &buf]() {
#ifndef _WIN32
                // ffmpeg can exit before it reads all the input, for example when ffprobe already got the metadata.
                // the write below must then fail with EPIPE, instead of killing the process with SIGPIPE
                sigset_t sigpipe_set;
                sigemptyset(&sigpipe_set);
                sigaddset(&sigpipe_set, SIGPIPE);
                pthread_sigmask(SIG_BLOCK, &sigpipe_set, nullptr); // linux sends the signal to the writing thread
#endif
                FILE * f = subprocess_stdin(&proc);
                if (!f) {
                    return;
                }
#ifdef F_SETNOSIGPIPE
                fcntl(fileno(f), F_SETNOSIGPIPE, 1); // macos/bsd send it to the process, so turn it off per fd
#endif
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

    // youtu-omni: frames and audio pieces in the layout of the reference video processor, see build_plan()
    struct plan_item {
        std::string   text;
        mtmd_bitmap * bitmap;
        std::string   id_suffix;
    };
    bool    interleave_audio = false;
    int     sample_rate      = 0;
    float   src_fps          = 0.0f;
    int64_t src_n_frames     = -1;
    double  duration_sec     = 0.0;
    double  select_step      = 0.0; // > 0: keep source frames floor(k * select_step), instead of the fps filter
    std::vector<double> frame_sec;
    std::vector<float> pcm;
    std::vector<mtmd_bitmap *> frames;
    std::vector<plan_item> plan;
    size_t plan_pos = 0;

    ~mtmd_helper_video() {
        for (auto * frame : frames) {
            if (frame) {
                mtmd_bitmap_free(frame);
            }
        }
        for (auto & item : plan) {
            if (item.bitmap) {
                mtmd_bitmap_free(item.bitmap);
            }
        }
    }

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
        probe_sp.created = true;
        probe_sp.alive   = true;

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
        src_fps      = orig_fps;
        src_n_frames = n_frames_orig > 0 ? n_frames_orig : duration > 0.0f ? (int64_t) std::llround(duration * orig_fps) : -1;

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

    void push_ffmpeg_input(std::vector<const char *> & cmd) const {
        cmd.push_back("-nostdin");
        if (is_buf_input()) {
            // remove the 64KB read-ahead limit of cache:, or else ffmpeg cannot reach a moov atom at end of file
            cmd.push_back("-read_ahead_limit");
            cmd.push_back("-1");
        }
        cmd.push_back("-i");
        // cache:pipe:0 wraps stdin with a seekable in-memory cache, letting ffmpeg seek
        // backwards for container headers (e.g. MP4 moov atom at end of file)
        cmd.push_back(is_buf_input() ? "cache:pipe:0" : path.c_str());
    }

    // cmd must be null-terminated
    bool spawn_ffmpeg(subprocess_handle & h, const std::vector<const char *> & cmd) {
        LOG_DBG("%s: launching:", __func__);
        for (size_t i = 0; cmd[i]; i++) {
            LOG_DBG(" %s", cmd[i]);
        }
        LOG_DBG("\n");

        int ret = subprocess_create(
            cmd.data(),
            subprocess_option_search_user_path | subprocess_option_inherit_environment,
            &h.proc);

        h.created = (ret == 0);
        h.alive   = (ret == 0);
        LOG_DBG("%s: subprocess_create ret=%d proc_alive=%d\n", __func__, ret, (int)h.alive);

        if (h.alive && is_buf_input()) {
            LOG_DBG("%s: starting feeder thread for %zu-byte buffer\n", __func__, input_buf.size());
            h.start_feeder(input_buf);
        }

        return h.alive;
    }

    bool start_ffmpeg(float seek_seconds) {
        char seek_buf[64];
        char fps_buf[64];
        char select_buf[128];

        std::vector<const char *> cmd;
        cmd.push_back(ffmpeg_bin.c_str());

        if (!is_buf_input() && seek_seconds > 0.0f) {
            // input-side seek: fast, keyframe-accurate; only valid for seekable file inputs
            snprintf(seek_buf, sizeof(seek_buf), "%.6f", seek_seconds);
            cmd.push_back("-ss");
            cmd.push_back(seek_buf);
        }

        push_ffmpeg_input(cmd);

        if (seek_seconds > 0.0f && is_buf_input()) {
            // output-side seek: frame-accurate but decodes and discards frames up to seek point
            snprintf(seek_buf, sizeof(seek_buf), "%.6f", seek_seconds);
            cmd.push_back("-ss");
            cmd.push_back(seek_buf);
        }

        if (select_step > 0.0) {
            // step is an integer or n/128, so %.7f is exact
            snprintf(select_buf, sizeof(select_buf), "select='eq(n,floor(ceil(n/%.7f)*%.7f))'", select_step, select_step);
            cmd.push_back("-vf");
            cmd.push_back(select_buf);
            cmd.push_back("-fps_mode");
            cmd.push_back("passthrough");
        } else if (fps_target > 0.0f) {
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

        return spawn_ffmpeg(sp, cmd);
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
        mtmd_bitmap * frame = mtmd_bitmap_init(info.width, info.height, frame_buf.data());
        mtmd_bitmap_set_mergeable(frame, true);
        if (!id.empty()) {
            // each frame gets a unique id in the form of {hash}+{frame}, so that it can be identified in cache
            std::string frame_id = id + "+" + std::to_string(current_frame);
            mtmd_bitmap_set_id(frame, frame_id.c_str());
        }
        current_frame++;
        return frame;
    }

    bool extract_pcm() {
        const std::string sr = std::to_string(sample_rate);
        std::vector<const char *> cmd;
        cmd.push_back(ffmpeg_bin.c_str());
        push_ffmpeg_input(cmd);
        cmd.push_back("-vn");
        cmd.push_back("-ac");
        cmd.push_back("1");
        cmd.push_back("-ar");
        cmd.push_back(sr.c_str());
        cmd.push_back("-f");
        cmd.push_back("f32le");
        cmd.push_back("-loglevel");
        cmd.push_back("error");
        cmd.push_back("pipe:1");
        cmd.push_back(nullptr);

        subprocess_handle audio;
        if (!spawn_ffmpeg(audio, cmd)) {
            return false;
        }
        FILE * fp = audio.stdout_pipe();
        std::vector<float> buf(4096);
        while (true) {
            size_t n = fread(buf.data(), sizeof(float), buf.size(), fp);
            if (n == 0) {
                break;
            }
            pcm.insert(pcm.end(), buf.begin(), buf.begin() + n);
        }
        audio.stop();
        return !pcm.empty();
    }

    // youtu-omni: at most 128 frames and at most 1 fps, a frame is at (source frame index) / round(fps) seconds
    void setup_sampling() {
        const int    max_frames = 128;
        const double fps_r = std::max(1.0, std::nearbyint((double) src_fps));
        const double step  = src_n_frames > 0 ? std::max(fps_r, (double) src_n_frames / max_frames) : fps_r;
        for (int i = 0; i < max_frames; i++) {
            const int64_t idx = (int64_t) (i * step);
            if (src_n_frames > 0 && idx >= src_n_frames) {
                break;
            }
            frame_sec.push_back(1.0 / fps_r * idx);
        }
        select_step   = step;
        fps_target    = (float) (fps_r / step);
        info.fps      = fps_target;
        info.n_frames = (int32_t) frame_sec.size();
        duration_sec  = src_n_frames > 0 ? src_n_frames / (double) src_fps : 0.0;
    }

    bool drain_frames() {
        const int cap = 128;
        while ((int) frames.size() < cap) {
            mtmd_bitmap * frame = read_next_frame();
            if (!frame) {
                break;
            }
            mtmd_bitmap_set_mergeable(frame, false);
            frames.push_back(frame);
        }
        stop_ffmpeg();
        return !frames.empty();
    }

    static std::string timestamp_hhmmss(int sec) {
        if (sec < 0) {
            sec = 0;
        }
        const int h = sec / 3600;
        const int m = (sec % 3600) / 60;
        const int s = sec % 60;
        char buf[16];
        snprintf(buf, sizeof(buf), "%02d:%02d:%02d", h, m, s);
        return buf;
    }

    // <|video_start|> {chunk} ... <|video_end|>, a chunk is "[ts] frame" for each of its frames, then "[ts] audio" pieces.
    // Each frame owns the audio up to the next frame. A chunk is closed once its audio reaches 1.5 s, and the audio of a single frame is split at 29.5 s.
    // Audio pieces are cut at mel frame borders, so mtmd can compute the mel of the whole track and slice it.
    void build_plan() {
        const size_t  n_fr      = std::min(frames.size(), frame_sec.size());
        const int64_t hop       = sample_rate / 100;
        const int64_t n_mel     = hop > 0 ? (int64_t) pcm.size() / hop : 0;
        const double  total_sec = n_mel > 0 ? (double) pcm.size() / sample_rate : 0.0;

        std::vector<int64_t> beg(n_fr, 0);
        std::vector<int64_t> end(n_fr, 0);
        int64_t n_used = 0;
        for (size_t i = 0; i < n_fr && n_mel > 0; i++) {
            const double ed = i + 1 < n_fr ? frame_sec[i + 1] : total_sec;
            beg[i] = std::min(n_mel, (int64_t) (frame_sec[i] / total_sec * n_mel));
            end[i] = std::max(beg[i], std::min(n_mel, (int64_t) (ed / total_sec * n_mel)));
            n_used += end[i] - beg[i];
        }
        const double sec_per_mel = n_used > 0 ? (duration_sec > 0.0 ? duration_sec : total_sec) / n_used : 0.0;

        int n_audio = 0;
        auto add_text = [&](const std::string & text) {
            plan.push_back({text, nullptr, {}});
        };
        auto add_ts = [&](double sec) {
            add_text(timestamp_hhmmss((int) std::nearbyint(sec)));
        };
        auto add_frame = [&](size_t i) {
            add_ts(frame_sec[i]);
            plan.push_back({{}, frames[i], "+" + std::to_string(i)});
            frames[i] = nullptr;
        };
        auto add_audio = [&](int64_t b, int64_t e, double sec) {
            if (e <= b) {
                return;
            }
            const size_t s0 = (size_t) (b * hop);
            const size_t s1 = e == n_mel ? pcm.size() : (size_t) (e * hop);
            add_ts(sec);
            plan.push_back({{}, mtmd_bitmap_init_from_audio(s1 - s0, pcm.data() + s0), "+a" + std::to_string(n_audio++)});
        };

        std::vector<size_t> pending;
        double pending_sec = 0.0;
        auto flush = [&]() {
            if (pending.empty()) {
                return;
            }
            for (size_t i : pending) {
                add_frame(i);
            }
            add_audio(beg[pending.front()], end[pending.back()], frame_sec[pending.front()]);
            pending.clear();
            pending_sec = 0.0;
        };

        const double min_sec = 1.5;
        const double max_sec = 29.5;
        add_text("<|video_start|>");
        for (size_t i = 0; i < n_fr; i++) {
            if (n_used == 0) {
                add_frame(i);
                continue;
            }
            const double sec = (end[i] - beg[i]) * sec_per_mel;
            if (sec + pending_sec < min_sec) {
                pending.push_back(i);
                pending_sec += sec;
            } else if (sec + pending_sec <= max_sec) {
                pending.push_back(i);
                flush();
            } else {
                flush(); // the reference expects nothing pending here
                add_frame(i);
                const int64_t piece = std::max<int64_t>(1, (int64_t) (max_sec / sec_per_mel));
                double piece_sec = frame_sec[i];
                for (int64_t b = beg[i]; b < end[i]; b += piece) {
                    const int64_t e = std::min(end[i], b + piece);
                    add_audio(b, e, piece_sec);
                    piece_sec += (e - b) * sec_per_mel;
                }
            }
        }
        flush();
        add_text("<|video_end|>");
    }

    int32_t read_next_interleaved(mtmd_bitmap ** out_bitmap, char ** out_text) {
        *out_bitmap = nullptr;
        *out_text   = nullptr;
        if (plan_pos >= plan.size()) {
            return -1;
        }
        plan_item & item = plan[plan_pos++];
        if (item.bitmap) {
            *out_bitmap = item.bitmap;
            item.bitmap = nullptr;
            return 0;
        }
        *out_text = mtmd_strdup(item.text);
        return *out_text ? 0 : -2;
    }

    int32_t read_next(mtmd_bitmap ** out_bitmap, char ** out_text) {
        if (interleave_audio) {
            return read_next_interleaved(out_bitmap, out_text);
        }
        *out_bitmap = nullptr;
        *out_text   = nullptr;

        if (!pending_text.empty()) {
            *out_text = mtmd_strdup(pending_text);
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
                *out_text = mtmd_strdup(prompt_start);
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

mtmd_helper_init_opt mtmd_helper_init_opt_default() {
    return {
        /* video_params */ mtmd_helper_video_init_params_default(),
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

#ifdef MTMD_VIDEO
static void mtmd_helper_video_set_id(mtmd_helper_video * vctx, const std::string & id) {
    vctx->id = id;
    // the plan is built during init, before the file id is known
    // placeholder mode has no hash, but mtmd still needs the suffix to find the video
    const std::string base = id.empty() ? "video" : id;
    for (auto & item : vctx->plan) {
        if (item.bitmap) {
            mtmd_bitmap_set_id(item.bitmap, (base + item.id_suffix).c_str());
        }
    }
}

static mtmd_bitmap * decode_webp_with_ffmpeg(const mtmd_context * mctx, const unsigned char * buf, size_t len, bool placeholder,
                                             const mtmd_helper_video_init_params & params) {
    mtmd_helper_video vctx;
    vctx.mctx        = mctx;
    vctx.input_buf.assign(buf, buf + len);
    vctx.ffmpeg_bin  = video_resolve_bin(params.ffmpeg_bin_dir, "ffmpeg");
    vctx.ffprobe_bin = video_resolve_bin(params.ffmpeg_bin_dir, "ffprobe");
    if (!vctx.probe(0.0f)) {
        return nullptr;
    }
    if (placeholder) {
        return mtmd_bitmap_init(vctx.info.width, vctx.info.height, nullptr);
    }
    // still image: the fps filter would output no frame, so disable it
    vctx.fps_target = 0.0f;
    if (!vctx.start_ffmpeg(0.0f)) {
        return nullptr;
    }
    mtmd_bitmap * frame = vctx.read_next_frame();
    if (frame) {
        mtmd_bitmap_set_mergeable(frame, false);
    }
    return frame;
}
#endif

mtmd_helper_video * mtmd_helper_video_init(
        const mtmd_context * mctx,
        const char * path,
        mtmd_helper_video_init_params params) {
#ifdef MTMD_VIDEO
    auto * ctx = new mtmd_helper_video();

    ctx->mctx                 = mctx;
    ctx->path                 = path;
    ctx->ffmpeg_bin           = video_resolve_bin(params.ffmpeg_bin_dir, "ffmpeg");
    ctx->ffprobe_bin          = video_resolve_bin(params.ffmpeg_bin_dir, "ffprobe");
    ctx->timestamp_interval_ms = params.timestamp_interval_ms;
    ctx->interleave_audio     = mctx && mtmd_video_interleave_audio(mctx);
    ctx->sample_rate          = ctx->interleave_audio ? mtmd_get_audio_sample_rate(mctx) : 0;
    const float fps = ctx->interleave_audio ? 0.0f : params.fps_target;

    if (!ctx->probe(fps)) {
        LOG_ERR("%s: ffprobe failed for '%s' (is ffprobe in PATH?)\n", __func__, path);
        delete ctx;
        return nullptr;
    }

    if (ctx->interleave_audio) {
        ctx->setup_sampling();
    }

    if (ctx->interleave_audio && !ctx->extract_pcm()) {
        LOG_WRN("%s: no audio track in '%s'\n", __func__, path);
    }

    if (!ctx->start_ffmpeg(0.0f)) {
        LOG_ERR("%s: failed to start ffmpeg for '%s' (is ffmpeg in PATH?)\n", __func__, path);
        delete ctx;
        return nullptr;
    }

    if (ctx->interleave_audio) {
        if (!ctx->drain_frames()) {
            LOG_ERR("%s: failed to read frames from '%s'\n", __func__, path);
            delete ctx;
            return nullptr;
        }
        ctx->build_plan();
    }

    return ctx;
#else
    GGML_UNUSED(mctx);
    GGML_UNUSED(path);
    GGML_UNUSED(params);
    LOG_ERR("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", __func__);
    return nullptr;
#endif
}

mtmd_helper_video * mtmd_helper_video_init_from_buf(
        const mtmd_context * mctx,
        const unsigned char * buf, size_t len,
        mtmd_helper_video_init_params params) {
#ifdef MTMD_VIDEO
    auto * ctx = new mtmd_helper_video();

    ctx->mctx                  = mctx;
    ctx->input_buf.assign(buf, buf + len);
    ctx->ffmpeg_bin            = video_resolve_bin(params.ffmpeg_bin_dir, "ffmpeg");
    ctx->ffprobe_bin           = video_resolve_bin(params.ffmpeg_bin_dir, "ffprobe");
    ctx->timestamp_interval_ms = params.timestamp_interval_ms;
    ctx->interleave_audio      = mctx && mtmd_video_interleave_audio(mctx);
    ctx->sample_rate           = ctx->interleave_audio ? mtmd_get_audio_sample_rate(mctx) : 0;
    const float fps = ctx->interleave_audio ? 0.0f : params.fps_target;

    if (!ctx->probe(fps)) {
        LOG_ERR("%s: ffprobe failed on buffer (is ffprobe in PATH?)\n", __func__);
        delete ctx;
        return nullptr;
    }

    if (ctx->interleave_audio) {
        ctx->setup_sampling();
    }

    if (ctx->interleave_audio && !ctx->extract_pcm()) {
        LOG_WRN("%s: no audio track in buffer\n", __func__);
    }

    if (!ctx->start_ffmpeg(0.0f)) {
        LOG_ERR("%s: failed to start ffmpeg on buffer (is ffmpeg in PATH?)\n", __func__);
        delete ctx;
        return nullptr;
    }

    if (ctx->interleave_audio) {
        if (!ctx->drain_frames()) {
            LOG_ERR("%s: failed to read frames from buffer\n", __func__);
            delete ctx;
            return nullptr;
        }
        ctx->build_plan();
    }

    return ctx;
#else
    GGML_UNUSED(mctx);
    GGML_UNUSED(buf);
    GGML_UNUSED(len);
    GGML_UNUSED(params);
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
    GGML_UNUSED(ctx);
    LOG_ERR("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", __func__);
#endif
}

mtmd_helper_video_info mtmd_helper_video_get_info(const mtmd_helper_video * ctx) {
#ifdef MTMD_VIDEO
    return ctx->info;
#else
    GGML_UNUSED(ctx);
    GGML_ASSERT(false && "video is not supported in this build (MTMD_VIDEO is set to OFF)");
#endif
}

int32_t mtmd_helper_video_read_next(mtmd_helper_video * ctx,
        mtmd_bitmap ** out_bitmap, char ** out_text) {
#ifdef MTMD_VIDEO
    if (!ctx) return -2;
    return ctx->read_next(out_bitmap, out_text);
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(out_bitmap);
    GGML_UNUSED(out_text);
    GGML_ASSERT(false && "video is not supported in this build (MTMD_VIDEO is set to OFF)");
#endif
}

bool mtmd_helper_model_can_chat(const llama_context * lctx, const mtmd_context * mctx) {
    if (!mctx) {
        return true;
    }

    auto * model = llama_get_model(lctx);
    auto * tmpl = llama_model_chat_template(model, nullptr);
    auto info = mtmd_gen_audio_get_info(mctx);

    // tts-only model cannot be used for chat (no chat template)
    bool is_tts_only = info.type != MTMD_GEN_AUDIO_TYPE_NONE && tmpl == nullptr;

    return !is_tts_only;
}
