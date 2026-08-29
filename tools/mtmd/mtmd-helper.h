#ifndef MTMD_HELPER_H
#define MTMD_HELPER_H

#include "ggml.h"
#include "llama.h"
#include "mtmd.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// libmtmd helper functions
//
// Please note that these helpers are not guaranteed to be stable.
// BREAKING CHANGES are expected.
//

struct mtmd_helper_video;
typedef struct mtmd_helper_video mtmd_helper_video;

struct mtmd_helper_video_init_params {
    float fps_target;            // desired output fps; <= 0 means use the video's native fps, defaulted to 4.0f
    const char * ffmpeg_bin_dir; // directory containing ffmpeg/ffprobe binaries; NULL means search PATH
    int64_t timestamp_interval_ms; // interval for adding timestamp as text chunk (example: "[10m50.5s]"); <= 0 means no timestamp, defaulted to 5000ms
    // TODO @ngxson : allow "placeholder" bitmap output for counting tokens
};

MTMD_API struct mtmd_helper_video_init_params mtmd_helper_video_init_params_default(void);

// opt for mtmd_helper_bitmap_init_from_*()
struct mtmd_helper_init_opt {
    struct mtmd_helper_video_init_params video_params;
};
typedef struct mtmd_helper_init_opt mtmd_helper_init_opt;

MTMD_API struct mtmd_helper_init_opt mtmd_helper_init_opt_default(void);

// Set callback for all future logging events.
// If this is not called, or NULL is supplied, everything is output on stderr.
// Note: this also call mtmd_log_set() internally
MTMD_API void mtmd_helper_log_set(ggml_log_callback log_callback, void * user_data);

// Returns true if this build includes video support (MTMD_VIDEO was ON at compile time).
MTMD_API bool mtmd_helper_support_video(mtmd_context * ctx);

struct mtmd_helper_bitmap_wrapper {
    mtmd_bitmap * bitmap;
    mtmd_helper_video * video_ctx;
};

// helper function to construct a mtmd_bitmap from a file
// it calls mtmd_helper_bitmap_init_from_buf() internally
// returns nullptr on failure
// this function is thread-safe
MTMD_API struct mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_file(
                    mtmd_context * ctx,
                    const char * fname,
                    bool placeholder,
                    struct mtmd_helper_init_opt opt);

// helper function to construct a mtmd_bitmap from a buffer containing a file
// supported formats:
//     image: formats supported by stb_image: jpg, png, bmp, gif, etc.
//            webp is decoded via ffmpeg, requires MTMD_VIDEO build with ffmpeg in PATH
//     audio: formats supported by miniaudio: wav, mp3, flac
// note:
//   - for now, video input is only supported via C++ helper functions
//   - audio files will be auto-detected based on magic bytes
//   - output bitmap will have SHA-256 hash (hex string) as the ID
// returns nullptr on failure
// this function is thread-safe
MTMD_API struct mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf(
                    mtmd_context * ctx,
                    const unsigned char * buf, size_t len,
                    bool placeholder,
                    struct mtmd_helper_init_opt opt);

// helper to count the total number of tokens from a list of chunks, useful to keep track of KV cache
MTMD_API size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks * chunks);

// helper to count the total position of tokens from a list of chunks, useful to keep track of n_past
// normally, n_pos is equal to n_tokens, but for M-RoPE it is different
MTMD_API llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks * chunks);

// helper to get the list of relative positions corresponding to the embedding tokens, to be used by M-RoPE
// out_pos must have length == mtmd_helper_get_n_tokens(image)
MTMD_API void mtmd_helper_image_get_decoder_pos(const mtmd_image_tokens * image, llama_pos pos_0, struct mtmd_decoder_pos * out_pos);

// helper function that automatically:
// 1. run llama_decode() on text chunks
// 2. run mtmd_encode_chunk() on image chunks, then mtmd_get_output_embd() and then llama_decode()
// if any of the mtmd_encode_chunk() or llama_decode() calls return non-zero, stop and forward the error
// otherwise, returns 0 on success
// this function is NOT thread-safe
MTMD_API int32_t mtmd_helper_eval_chunks(mtmd_context * ctx,
                                         struct llama_context * lctx,
                                         const mtmd_input_chunks * chunks,
                                         llama_pos n_past,
                                         llama_seq_id seq_id,
                                         int32_t n_batch,
                                         bool logits_last,
                                         llama_pos * new_n_past);

// works like mtmd_helper_eval_chunks(), but only for a single chunk
// this function is NOT thread-safe
MTMD_API int32_t mtmd_helper_eval_chunk_single(mtmd_context * ctx,
                                               struct llama_context * lctx,
                                               const mtmd_input_chunk * chunk,
                                               llama_pos n_past,
                                               llama_seq_id seq_id,
                                               int32_t n_batch,
                                               bool logits_last,
                                               llama_pos * new_n_past);

typedef int32_t (*mtmd_helper_post_decode_callback)(struct llama_batch batch, void * user_data);

// helper function to decode an image whose embeddings have already been calculated
// this helper will handle batching and pre/post decoding setup (for ex. gemma 3 requires non-causal attention)
// ret 0 on success, -1 on chunk not being a valid image chunk, 1 on decode failure
MTMD_API int32_t mtmd_helper_decode_image_chunk(mtmd_context * ctx,
                                                struct llama_context * lctx,
                                                const mtmd_input_chunk * chunk,
                                                float * encoded_embd,
                                                llama_pos n_past,
                                                llama_seq_id seq_id,
                                                int32_t n_batch,
                                                llama_pos * new_n_past,
                                                mtmd_helper_post_decode_callback callback,
                                                void * user_data);

//
// video input helpers (requires ffmpeg/ffprobe installed on the system)
// the notion of video only exists at the helper level, it is not visible to the core mtmd library
//
// NOTE: this implementation is model-agnostic, it can be used with any vision-capable model
//       however, it may not be accurate for some specific models
//       (this is expected for now, to keep the implementation simple)
//

struct mtmd_helper_video_info {
    uint32_t width;
    uint32_t height;
    float    fps;      // effective fps (fps_target if set, else original video fps)
    int32_t  n_frames; // estimated total frames at effective fps (-1 if unknown)
};

// note: mtmd_helper_video_init_params is defined at the top, as it is part of mtmd_helper_init_opt

// returns NULL on failure (ffprobe not found, file unreadable, etc.)
MTMD_API mtmd_helper_video * mtmd_helper_video_init(
                    struct mtmd_context * mctx,
                    const char * path,
                    struct mtmd_helper_video_init_params params);

// Same as mtmd_helper_video_init(), but reads from an in-memory buffer.
// The buffer is copied internally; the caller does not need to keep it alive.
// Note: pipe input is not seekable, so seeking will use output-side seeking
// (ffmpeg decodes and discards frames up to the target position).
MTMD_API mtmd_helper_video * mtmd_helper_video_init_from_buf(
                    struct mtmd_context * mctx,
                    const unsigned char * buf, size_t len,
                    struct mtmd_helper_video_init_params params);
MTMD_API void mtmd_helper_video_free(mtmd_helper_video * ctx);
MTMD_API struct mtmd_helper_video_info mtmd_helper_video_get_info(const mtmd_helper_video * ctx);

// Read the next item from the video stream; exactly one of out_bitmap or out_text is set per call.
// *out_bitmap - heap-allocated; caller must free with mtmd_bitmap_free()
// *out_text   - heap-allocated (always via strdup/malloc); caller must free with free()
// returns 0 on success, -1 on EOF, -2 on error
MTMD_API int32_t mtmd_helper_video_read_next(mtmd_helper_video * ctx,
            mtmd_bitmap ** out_bitmap,
            char ** out_text);

// return true if model can be used for chat
MTMD_API bool mtmd_helper_model_can_chat(struct llama_context * lctx, struct mtmd_context * mctx);

//
// Audio generation helpers
// (early-stage experimental, subjected to breaking changes)
//

// audio generation helper context
// contains accumulator for generated audio features and PCM audio
struct mtmd_helper_gen_audio;
typedef struct mtmd_helper_gen_audio mtmd_helper_gen_audio;

enum mtmd_helper_gen_audio_outtype {
    MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM, // raw PCM
    MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV, // WAV PCM 16-bit LE, mono
};
struct mtmd_helper_gen_audio_inp {
    llama_seq_id seq_id;

    const char * prompt;
    size_t       prompt_len;

    mtmd_bitmap * speaker_ref; // optional, can be NULL
    const char * lang; // optional, can be NULL

    int32_t  top_k;
    float    top_p;
    uint32_t seed; // UINT32_MAX for random (default: random)

    enum mtmd_helper_gen_audio_outtype out_type;
};

MTMD_API mtmd_helper_gen_audio * mtmd_helper_gen_audio_init(
                                    struct llama_context * lctx,
                                    struct mtmd_context * mctx);

MTMD_API void mtmd_helper_gen_audio_free(mtmd_helper_gen_audio * ctx);

MTMD_API void mtmd_helper_gen_audio_reset(mtmd_helper_gen_audio * ctx);

MTMD_API int32_t mtmd_helper_gen_audio_set_input(
                        mtmd_helper_gen_audio * ctx,
                        const struct mtmd_helper_gen_audio_inp * inp);

// processes at most n_batch prompt tokens per call
// returns: >0 = number of prompt tokens remaining, 0 = done, <0 = error
MTMD_API int32_t mtmd_helper_gen_audio_step_prompt(
                        mtmd_helper_gen_audio * ctx,
                        int32_t n_batch);

// generates one frame; must only be called after step_prompt() has returned 0
// sampled can be LLAMA_TOKEN_NULL for pipelines with no discrete backbone token
// out_stop (optional) is set on end-of-speech, the caller must then stop the loop
// h_state_out is valid until next step_gen() or reset() call, null if no frame is generated
MTMD_API int32_t mtmd_helper_gen_audio_step_gen(
                        mtmd_helper_gen_audio * ctx,
                        llama_token sampled,
                        const float *  h_state_in,
                        const float ** h_state_out,
                        bool * out_stop);

// out_data valid until next get_output() or reset() call
// out_n_samples (optional, can be NULL) receives the number of generated PCM samples
MTMD_API int32_t mtmd_helper_gen_audio_get_output(
                        mtmd_helper_gen_audio * ctx,
                        int32_t * out_sample_rate,
                        const char ** out_data,
                        size_t * out_data_len,
                        int64_t * out_n_samples);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus
#include <set>
#include <memory>

namespace mtmd_helper {

//
// C++ wrappers
//

// video-related C++ wrappers
struct mtmd_helper_video_deleter {
    void operator()(mtmd_helper_video * val) { mtmd_helper_video_free(val); }
};
using video_ptr = std::unique_ptr<mtmd_helper_video, mtmd_helper_video_deleter>;

// audio generation-related C++ wrappers
struct mtmd_helper_gen_audio_deleter {
    void operator()(mtmd_helper_gen_audio * val) { mtmd_helper_gen_audio_free(val); }
};
using gen_audio_ptr = std::unique_ptr<mtmd_helper_gen_audio, mtmd_helper_gen_audio_deleter>;
struct gen_audio {
    gen_audio_ptr ctx;
    gen_audio(struct llama_context * lctx, struct mtmd_context * mctx) : ctx(mtmd_helper_gen_audio_init(lctx, mctx)) {}
    void reset() {
        mtmd_helper_gen_audio_reset(ctx.get());
    }
    int32_t set_input(const struct mtmd_helper_gen_audio_inp * inp) {
        return mtmd_helper_gen_audio_set_input(ctx.get(), inp);
    }
    int32_t step_prompt(int32_t n_batch) {
        return mtmd_helper_gen_audio_step_prompt(ctx.get(), n_batch);
    }
    int32_t step_gen(llama_token sampled, const float * h_state, const float ** h_state_out, bool * out_stop = nullptr) {
        return mtmd_helper_gen_audio_step_gen(ctx.get(), sampled, h_state, h_state_out, out_stop);
    }
    int32_t get_output(int32_t * out_sample_rate, const char ** out_data, size_t * out_data_len, int64_t * out_n_samples = nullptr) {
        return mtmd_helper_gen_audio_get_output(ctx.get(), out_sample_rate, out_data, out_data_len, out_n_samples);
    }
};

} // namespace mtmd_helper
#endif

#endif
