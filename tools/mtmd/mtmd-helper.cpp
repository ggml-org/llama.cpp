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
#include "../src/llama-ext.h"

#include "hash/hash.h"

#include <algorithm>
#include <cmath>
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

struct mtmd_helper_resampler {
    ma_resampler       rs;
    bool               passthrough = false;
    std::vector<float> out;
};

mtmd_helper_resampler * mtmd_helper_resampler_init(int32_t rate_in, int32_t rate_out) {
    if (rate_in <= 0 || rate_out <= 0) {
        LOG_ERR("%s: invalid sample rate %d -> %d\n", __func__, rate_in, rate_out);
        return nullptr;
    }
    auto * ctx = new mtmd_helper_resampler;
    if (rate_in == rate_out) {
        ctx->passthrough = true;
        return ctx;
    }
    ma_resampler_config config = ma_resampler_config_init(ma_format_f32, 1, (ma_uint32) rate_in, (ma_uint32) rate_out,
                                                          ma_resample_algorithm_linear);
    config.linear.lpfOrder = MA_MAX_FILTER_ORDER;
    if (ma_resampler_init(&config, nullptr, &ctx->rs) != MA_SUCCESS) {
        LOG_ERR("%s: failed to initialize the resampler %d -> %d\n", __func__, rate_in, rate_out);
        delete ctx;
        return nullptr;
    }
    return ctx;
}

void mtmd_helper_resampler_free(mtmd_helper_resampler * ctx) {
    if (!ctx) {
        return;
    }
    if (!ctx->passthrough) {
        ma_resampler_uninit(&ctx->rs, nullptr);
    }
    delete ctx;
}

int64_t mtmd_helper_resampler_process(mtmd_helper_resampler * ctx, const float * in, size_t n_in, const float ** out) {
    if (ctx->passthrough) {
        *out = in;
        return (int64_t) n_in;
    }
    ma_uint64 n_expected = 0;
    if (ma_resampler_get_expected_output_frame_count(&ctx->rs, n_in, &n_expected) != MA_SUCCESS) {
        return -1;
    }
    ctx->out.resize(n_expected + 16);
    size_t n_done_in  = 0;
    size_t n_done_out = 0;
    while (n_done_in < n_in) {
        if (n_done_out == ctx->out.size()) {
            ctx->out.resize(ctx->out.size() * 2);
        }
        ma_uint64 cnt_in  = n_in - n_done_in;
        ma_uint64 cnt_out = ctx->out.size() - n_done_out;
        if (ma_resampler_process_pcm_frames(&ctx->rs, in + n_done_in, &cnt_in, ctx->out.data() + n_done_out, &cnt_out) != MA_SUCCESS) {
            return -1;
        }
        if (cnt_in == 0 && cnt_out == 0) {
            break;
        }
        n_done_in  += cnt_in;
        n_done_out += cnt_out;
    }
    *out = ctx->out.data();
    return (int64_t) n_done_out;
}

static bool is_webp_file(const unsigned char * buf, size_t len) {
    // WEBP ref: https://developers.google.com/speed/webp/docs/riff_container
    return len >= 12 && memcmp(buf, "RIFF", 4) == 0 && memcmp(buf + 8, "WEBP", 4) == 0;
}

static mtmd_bitmap * decode_still_image(const char * path, const unsigned char * buf, size_t len, bool placeholder,
                                        const mtmd_helper_video_init_params & params);
static mtmd_helper_video * video_init(const mtmd_context * mctx, const char * path, const unsigned char * buf, size_t len,
                                      const mtmd_helper_video_init_params & params);
static void mtmd_helper_video_set_id(mtmd_helper_video * vctx, const std::string & id);

static mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_impl(const mtmd_context * ctx, const char * fname,
                                                               const unsigned char * buf, size_t len, bool placeholder,
                                                               mtmd_helper_init_opt opt) {
    // calculate the hash if needed
    std::string id;
    mtmd_bitmap * result = nullptr;

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

    // stb_image does not support webp; decode it with the video decoder as a single frame
    if (!result && is_webp_file(buf, len)) {
        result = decode_still_image(fname, buf, len, placeholder, opt.video_params);
        if (!result) {
            LOG_ERR("%s: failed to decode webp buffer\n", __func__);
            return {nullptr, nullptr};
        }
        mtmd_bitmap_set_id(result, id.empty() ? nullptr : id.c_str());
        return {result, nullptr};
    }

    // last try: load as video
    if (!result) {
        auto video_ctx = video_init(ctx, fname, buf, len, opt.video_params);
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
        mtmd_bitmap_set_is_video(result, true);
        return {result, video_ctx};
    }

    // should not reach here
    return {nullptr, nullptr};
}

mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf(const mtmd_context * ctx, const unsigned char * buf, size_t len, bool placeholder,
                                                            mtmd_helper_init_opt opt) {
    return mtmd_helper_bitmap_init_impl(ctx, nullptr, buf, len, placeholder, opt);
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

    return mtmd_helper_bitmap_init_impl(ctx, fname, buf.data(), buf.size(), placeholder, opt);
}

//
// Video input helpers
//

// MSVC deprecates strdup(), use a std::string version directly
static char * mtmd_strdup(const std::string & s) {
    size_t n = s.size() + 1;
    char * p = (char *) malloc(n);
    if (p) {
        memcpy(p, s.c_str(), n);
    }
    return p;
}

mtmd_helper_video_seq::~mtmd_helper_video_seq() {
    mtmd_bitmap_free(pending_frame);
    mtmd_bitmap_free(pending_audio);
}

int32_t mtmd_helper_video_seq::ts_per_n_frames() const {
    return tmpl == MTMD_VIDEO_TEMPLATE_LFM3 || tmpl == MTMD_VIDEO_TEMPLATE_LFM3_OMNI || tmpl == MTMD_VIDEO_TEMPLATE_QWEN3VL ? 2 : 0;
}

void mtmd_helper_video_seq::configure(const mtmd_video_config & cfg, const mtmd_helper_video_stream_info & src, float fps_target_arg) {
    tmpl = cfg.tmpl;
    uint32_t width  = src.width;
    uint32_t height = src.height;
    float duration  = src.duration;
    if (duration < 0.0f && src.n_frames > 0) {
        duration = (float)src.n_frames / src.fps;
    }

    // the model's recommended sampling rate wins over the generic default
    fps_target = cfg.fps > 0.0f ? cfg.fps
               : fps_target_arg > 0.0f ? fps_target_arg : src.fps;

    if (cfg.n_frames > 0 && duration > 0.0f) {
        const int32_t n = cfg.n_frames;
        target_frames = n;
        fps_target    = (float) n / duration;

        // same frames as the reference processors: round(linspace(0, n_frames - 1, n))
        const double step = n > 1 ? (double) (src.n_frames - 1) / (n - 1) : 0.0;
        frame_indices.resize(n);
        for (int32_t i = 0; i < n; i++) {
            frame_indices[i] = (int32_t) std::nearbyint(i * step);
        }
        src_fps      = src.fps;
        n_src_frames = src.n_frames;

        if (cfg.frame_width > 0 && cfg.frame_height > 0 &&
                (cfg.frame_width != (int32_t) width || cfg.frame_height != (int32_t) height)) {
            out_width  = cfg.frame_width;
            out_height = cfg.frame_height;
        }
    }

    if (out_width > 0 && out_height > 0) {
        width  = (uint32_t) out_width;
        height = (uint32_t) out_height;
    }

    info.width    = width;
    info.height   = height;
    info.fps      = fps_target;
    LOG_DBG("%s: %ux%u fps=%.2f duration=%.2fs n_frames=%d target_frames=%d\n",
            __func__, width, height, fps_target, duration, src.n_frames, target_frames);
    info.n_frames = target_frames > 0 ? target_frames
                  : duration > 0.0f ? (int32_t)(duration * fps_target + 0.5f) : -1;
    frame_buf.resize((size_t)width * height * 3);
}

void mtmd_helper_video_seq::sample_by_index(const mtmd_helper_video_stream_info & src) {
    if (!frame_indices.empty() || src.fps <= 0.0f) {
        return;
    }
    const int32_t n_src = src.n_frames > 0 ? src.n_frames
                        : src.duration > 0.0f ? (int32_t) std::nearbyint(src.duration * src.fps) : 0;
    if (info.n_frames <= 0 || n_src <= 0) {
        // unknown length: keep every frame
        fps_target = src.fps;
        info.fps   = src.fps;
        return;
    }
    frame_indices.resize(info.n_frames);
    for (int32_t i = 0; i < info.n_frames; i++) {
        frame_indices[i] = std::min(n_src - 1, (int32_t) ((double) i * src.fps / fps_target));
    }
    src_fps      = src.fps;
    n_src_frames = n_src;
}

void mtmd_helper_video_seq::set_soundtrack(const float * pcm, size_t n, int rate) {
    if (!pcm || n == 0) {
        LOG_WRN("%s: the video has no soundtrack\n", __func__);
        return;
    }
    // keep only the span of the video stream: n_frames / fps seconds from its first frame
    const size_t len = (size_t) std::nearbyint((double) n_src_frames / src_fps * rate);
    audio_rate = rate;
    soundtrack.assign(pcm, pcm + std::min(n, len));
}

// the soundtrack of the group of frames that ends with frame idx, nullptr if the soundtrack has ended
mtmd_bitmap * mtmd_helper_video_seq::soundtrack_slice(int32_t idx) const {
    const int32_t ts_per_n_frames = this->ts_per_n_frames();
    const int32_t group    = idx / ts_per_n_frames;
    const int32_t n_groups = ((int32_t) frame_indices.size() + ts_per_n_frames - 1) / ts_per_n_frames;
    auto group_start = [&](int32_t g) {
        const double t = frame_indices[g * ts_per_n_frames] / src_fps;
        return std::min(soundtrack.size(), (size_t) std::nearbyint(t * audio_rate));
    };
    const size_t begin = group == 0 ? 0 : group_start(group);
    const size_t end   = group == n_groups - 1 ? soundtrack.size() : group_start(group + 1);
    if (begin >= end) {
        return nullptr;
    }
    mtmd_bitmap * slice = mtmd_bitmap_init_from_audio(end - begin, soundtrack.data() + begin);
    if (!id.empty()) {
        mtmd_bitmap_set_id(slice, (id + "+audio" + std::to_string(group)).c_str());
    }
    return slice;
}

mtmd_bitmap * mtmd_helper_video_seq::read_next_frame() {
    if (!src_alive) return nullptr;

    if (!frame_indices.empty()) {
        if (next_index >= frame_indices.size()) {
            return nullptr;
        }
        // skip to the wanted source frame; a repeated index reuses the frame in frame_buf
        const int32_t want = frame_indices[next_index];
        while (src_frame <= want) {
            if (!(src_frame < want ? skip_raw_frame() : read_raw_frame())) {
                src_alive = false;
                return nullptr;
            }
            src_frame++;
        }
        next_index++;
    } else if (!read_raw_frame()) {
        src_alive = false;
        return nullptr;
    }

    LOG_DBG("%s: frame %d read OK\n", __func__, current_frame);
    mtmd_bitmap * frame = mtmd_bitmap_init(info.width, info.height, frame_buf.data());
    mtmd_bitmap_set_mergeable(frame, true);
    mtmd_bitmap_set_is_video(frame, true);
    if (!id.empty()) {
        // each frame gets a unique id in the form of {hash}+{frame}, so that it can be identified in cache
        std::string frame_id = id + "+" + std::to_string(current_frame);
        mtmd_bitmap_set_id(frame, frame_id.c_str());
    }
    current_frame++;
    return frame;
}

int32_t mtmd_helper_video_seq::read_next(mtmd_bitmap ** out_bitmap, char ** out_text) {
    *out_bitmap = nullptr;
    *out_text   = nullptr;

    if (!pending_text.empty()) {
        *out_text = mtmd_strdup(pending_text);
        pending_text.clear();
        return *out_text ? 0 : -2;
    }

    if (pending_frame) {
        *out_bitmap  = pending_frame;
        pending_frame = nullptr;
        return 0;
    }

    if (pending_audio) {
        *out_bitmap   = pending_audio;
        pending_audio = nullptr;
        return 0;
    }

    LOG_DBG("%s: proc_alive=%d start_emitted=%d current_frame=%d\n",
            __func__, (int)src_alive, (int)start_emitted, current_frame);

    if (!src_alive) {
        return (current_frame == 0) ? -2 : -1;
    }

    if (!start_emitted) {
        start_emitted = true;
        // model templates have no prompt prefix, the video begin/end tokens are added by mtmd_tokenize
        if (tmpl == MTMD_VIDEO_TEMPLATE_DEFAULT && !prompt_start.empty()) {
            *out_text = mtmd_strdup(prompt_start);
            return *out_text ? 0 : -2;
        }
    }

    mtmd_bitmap * frame = read_next_frame();
    if (!frame) {
        return -1;
    }

    const int32_t ts_per_n_frames = this->ts_per_n_frames();
    if (ts_per_n_frames > 0) {
        // emit the group timestamp before the group's first frame; the value is the
        // midpoint time of the group, matching HF processors that stamp each temporal group
        const int32_t idx = current_frame - 1; // read_next_frame() already incremented
        // the soundtrack of a group follows its last frame
        const bool group_ends = idx % ts_per_n_frames == ts_per_n_frames - 1 || idx + 1 == (int32_t) frame_indices.size();
        if (!soundtrack.empty() && !frame_indices.empty() && group_ends) {
            pending_audio = soundtrack_slice(idx);
        }
        if (idx % ts_per_n_frames == 0) {
            char ts_buf[64];
            double ts_s = ((double)idx + 0.5*(double)(ts_per_n_frames - 1)) / info.fps;
            if (!frame_indices.empty()) {
                // stamp from the source frames; a short last group repeats its last frame
                const size_t first = idx;
                const size_t last  = std::min(first + ts_per_n_frames - 1, frame_indices.size() - 1);
                ts_s = (frame_indices[first] / src_fps + frame_indices[last] / src_fps) / 2.0;
            }
            snprintf(ts_buf, sizeof(ts_buf), "<%.1f seconds>", ts_s);
            pending_frame = frame;
            *out_text = mtmd_strdup(ts_buf);
            return *out_text ? 0 : -2;
        }
        *out_bitmap = frame;
        return 0;
    }

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

struct mtmd_helper_video : mtmd_helper_video_seq {
    mtmd_helper_video_decoder dec   = {};
    void *                    state = nullptr;

    ~mtmd_helper_video() override {
        if (state) {
            dec.close(state);
        }
    }

    bool read_raw_frame() override {
        return dec.read_frame(state, frame_buf.data(), info.width, info.height);
    }

    bool skip_raw_frame() override {
        return dec.read_frame(state, nullptr, info.width, info.height);
    }

    bool start(const mtmd_helper_video_stream_info & src, float fps) {
        bool fps_picked = false;
        if (dec.start && !dec.start(state, info.width, info.height, fps, &fps_picked)) {
            LOG_ERR("%s: the video decoder failed to start\n", __func__);
            return false;
        }
        if (fps > 0.0f && !fps_picked) {
            sample_by_index(src);
        }
        return true;
    }

    void load_soundtrack(const mtmd_context * mctx) {
        const int rate = mtmd_get_audio_sample_rate(mctx);
        if (tmpl != MTMD_VIDEO_TEMPLATE_LFM3_OMNI || frame_indices.empty() || rate <= 0 || !dec.read_audio) {
            return;
        }
        size_t n = 0;
        const float * pcm = dec.read_audio(state, rate, &n);
        set_soundtrack(pcm, n, rate);
    }
};

static const mtmd_helper_video_decoder * video_builtin_decoder() {
#ifdef MTMD_VIDEO
    return mtmd_helper_video_decoder_ffmpeg();
#else
    return nullptr;
#endif
}

static std::unique_ptr<mtmd_helper_video> video_open(const char * path, const unsigned char * buf, size_t len,
                                                     const mtmd_helper_video_init_params & params,
                                                     mtmd_helper_video_stream_info & src) {
    const mtmd_helper_video_decoder * dec = params.decoder;
    void * user_data = dec ? dec->user_data : nullptr;
    if (!dec) {
        dec = video_builtin_decoder();
        if (!dec) {
            LOG_ERR("%s: video is not supported in this build (MTMD_VIDEO is set to OFF) and no decoder is given\n", __func__);
            return nullptr;
        }
        user_data = const_cast<char *>(params.ffmpeg_bin_dir);
    }

    if (!dec->open || !dec->read_frame || !dec->close) {
        LOG_ERR("%s: the video decoder must have open, read_frame and close\n", __func__);
        return nullptr;
    }

    auto ctx = std::make_unique<mtmd_helper_video>();
    ctx->dec = *dec;

    src = {0, 0, 0.0f, -1, -1.0f};
    ctx->state = ctx->dec.open(user_data, path, buf, len, &src);
    if (!ctx->state || src.width == 0 || src.height == 0 || src.fps <= 0.0f) {
        LOG_ERR("%s: the video decoder failed to open the video\n", __func__);
        return nullptr;
    }
    return ctx;
}

static mtmd_helper_video * video_init(const mtmd_context * mctx, const char * path, const unsigned char * buf, size_t len,
                                      const mtmd_helper_video_init_params & params) {
    mtmd_helper_video_stream_info src;
    auto ctx = video_open(path, buf, len, params, src);
    if (!ctx) {
        return nullptr;
    }

    ctx->timestamp_interval_ms = params.timestamp_interval_ms;
    ctx->configure(mtmd_get_video_config(mctx, src.width, src.height, src.n_frames, src.fps), src, params.fps_target);
    if (!ctx->start(src, ctx->frame_indices.empty() ? ctx->fps_target : 0.0f)) {
        return nullptr;
    }
    if (!params.no_soundtrack) {
        ctx->load_soundtrack(mctx);
    }
    ctx->src_alive = true;
    return ctx.release();
}

static mtmd_bitmap * decode_still_image(const char * path, const unsigned char * buf, size_t len, bool placeholder,
                                        const mtmd_helper_video_init_params & params) {
    // an application's decoder may not read images, prefer the built-in one
    mtmd_helper_video_init_params still_params = params;
    if (video_builtin_decoder()) {
        still_params.decoder = nullptr;
    }
    mtmd_helper_video_stream_info src;
    auto vctx = video_open(path, buf, len, still_params, src);
    if (!vctx) {
        return nullptr;
    }
    if (placeholder) {
        return mtmd_bitmap_init(src.width, src.height, nullptr);
    }
    // still image: the fps filter would output no frame, so disable it
    vctx->configure({MTMD_VIDEO_TEMPLATE_DEFAULT, 0.0f, 0, 0, 0}, src, 0.0f);
    if (!vctx->start(src, 0.0f)) {
        return nullptr;
    }
    vctx->src_alive = true;
    mtmd_bitmap * frame = vctx->read_next_frame();
    if (frame) {
        mtmd_bitmap_set_mergeable(frame, false);
        mtmd_bitmap_set_is_video(frame, false);
    }
    return frame;
}

static void mtmd_helper_video_set_id(mtmd_helper_video * vctx, const std::string & id) {
    vctx->id = id;
}

mtmd_helper_video_init_params mtmd_helper_video_init_params_default() {
    return {
        /* fps_target             */ 4.0f,
        /* ffmpeg_bin_dir         */ nullptr,
        /* timestamp_interval_ms  */ 5000,
        /* no_soundtrack          */ false,
        /* decoder                */ nullptr,
    };
}

mtmd_helper_init_opt mtmd_helper_init_opt_default() {
    return {
        /* video_params */ mtmd_helper_video_init_params_default(),
    };
}

mtmd_helper_video * mtmd_helper_video_init(
        const mtmd_context * mctx,
        const char * path,
        mtmd_helper_video_init_params params) {
    return video_init(mctx, path, nullptr, 0, params);
}

mtmd_helper_video * mtmd_helper_video_init_from_buf(
        const mtmd_context * mctx,
        const unsigned char * buf, size_t len,
        mtmd_helper_video_init_params params) {
    return video_init(mctx, nullptr, buf, len, params);
}

void mtmd_helper_video_free(mtmd_helper_video * ctx) {
    delete ctx;
}

mtmd_helper_video_info mtmd_helper_video_get_info(const mtmd_helper_video * ctx) {
    return ctx->info;
}

int32_t mtmd_helper_video_read_next(mtmd_helper_video * ctx,
        mtmd_bitmap ** out_bitmap, char ** out_text) {
    if (!ctx) return -2;
    return ctx->read_next(out_bitmap, out_text);
}

mtmd_helper_cap mtmd_helper_get_capabilities(const llama_model * model, const mtmd_context * mctx) {
    mtmd_helper_cap cap = {};
    cap.chat     = true;
    cap.text_gen = true;
    if (!mctx) {
        return cap;
    }

    cap.vision = mtmd_support_vision(mctx);
    cap.audio  = mtmd_support_audio(mctx);
    cap.audio_stream = mtmd_support_audio_stream(mctx);
    cap.video  = cap.vision && video_builtin_decoder() != nullptr;
    const auto gen_audio_type = mtmd_gen_audio_get_info(mctx).type;
    cap.audio_gen = gen_audio_type != MTMD_GEN_AUDIO_TYPE_NONE;
    cap.audio_gen_spk_ref = gen_audio_type == MTMD_GEN_AUDIO_TYPE_QWEN3TTS
                         || gen_audio_type == MTMD_GEN_AUDIO_TYPE_POCKETTTS;

    // a tts model without a chat template only outputs audio, it cannot be used for chat
    // neither can a realtime transcription model (e.g. Voxtral Realtime): its text follows the audio by a token delay
    cap.text_gen = !cap.audio_gen || llama_model_chat_template(model, nullptr) != nullptr;
    cap.chat     = cap.text_gen && llama_model_n_tokens_delay(model) == 0;

    return cap;
}

bool mtmd_helper_model_can_chat(const llama_context * lctx, const mtmd_context * mctx) {
    return mtmd_helper_get_capabilities(llama_get_model(lctx), mctx).chat;
}

bool mtmd_helper_support_video(const mtmd_context * ctx) {
    return mtmd_support_vision(ctx) && video_builtin_decoder() != nullptr;
}
