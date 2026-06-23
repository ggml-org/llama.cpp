#ifndef MTMD_H
#define MTMD_H

#include "ggml.h"
#include "llama.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
#include <map>
#include <string>
#include <vector>
#include <cinttypes>
#include <memory>
#endif

/**
 * libmtmd: A library for multimodal support in llama.cpp.
 *
 * WARNING: This API is experimental and subject to many BREAKING CHANGES.
 *          Issues related to API usage may receive lower priority support.
 *
 * For the usage, see an example in mtmd-cli.cpp
 *
 * For contributors:
 * - Make sure the C API is aligned with the libllama C API (as in llama.h)
 * - Do not include model name (e.g., qwen, gemma) in the API, use generic terms instead
 * - Keep the API minimal, do not expose internal details unless necessary
 *
 * IMPORTANT: The mtmd module does NOT accept pull requests that are fully or predominantly AI-generated.
 * We encourage human contributors to ensure the quality and reliability of the codebase.
 */

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define MTMD_API __declspec(dllexport)
#        else
#            define MTMD_API __declspec(dllimport)
#        endif
#    else
#        define MTMD_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define MTMD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum mtmd_input_chunk_type {
    MTMD_INPUT_CHUNK_TYPE_TEXT,
    MTMD_INPUT_CHUNK_TYPE_IMAGE,
    MTMD_INPUT_CHUNK_TYPE_AUDIO,
};

// opaque types
struct mtmd_context;
struct mtmd_bitmap;
struct mtmd_image_tokens;
struct mtmd_input_chunk;
struct mtmd_input_chunks;
struct mtmd_batch;

struct mtmd_input_text {
    const char * text;
    bool add_special;
    bool parse_special;
};

//
// C API
//

typedef struct mtmd_context      mtmd_context;
typedef struct mtmd_bitmap       mtmd_bitmap;
typedef struct mtmd_image_tokens mtmd_image_tokens;
typedef struct mtmd_input_chunk  mtmd_input_chunk;
typedef struct mtmd_input_chunks mtmd_input_chunks;
typedef struct mtmd_input_text   mtmd_input_text;
typedef struct mtmd_batch        mtmd_batch;

typedef bool (*mtmd_progress_callback)(float progress, void * user_data);

struct mtmd_context_params {
    bool use_gpu;
    bool print_timings;
    int n_threads;
    const char * image_marker; // deprecated, use media_marker instead
    const char * media_marker;
    enum llama_flash_attn_type flash_attn_type;
    bool warmup; // whether to run a warmup encode pass after initialization

    // limit number of image tokens, only for vision models with dynamic resolution
    int image_min_tokens; // minimum number of tokens for image input (default: read from metadata)
    int image_max_tokens; // maximum number of tokens for image input (default: read from metadata)

    // callback function passed over to mtmd proper
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    // batching params
    int32_t batch_max_tokens; // maximum number of output tokens in a batch
                              // (note: this is not a hard-limit, the first image will always be added even if it exceeds this limit)
                              // (default: 1024)

    // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
    // If the provided progress_callback returns true, model loading continues.
    // If it returns false, model loading is immediately aborted.
    mtmd_progress_callback progress_callback;
    void * progress_callback_user_data;
};

MTMD_API const char * mtmd_default_marker(void);

MTMD_API struct mtmd_context_params mtmd_context_params_default(void);

// initialize the mtmd context
// return nullptr on failure
MTMD_API mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
                                            const struct llama_model * text_model,
                                            const struct mtmd_context_params ctx_params);

MTMD_API void mtmd_free(mtmd_context * ctx);

// whether we need to set non-causal mask before llama_decode
// if chunk is nullptr, we assume the default case where chunk is an image chunk
MTMD_API bool mtmd_decode_use_non_causal(const mtmd_context * ctx, const mtmd_input_chunk * chunk);

// whether the current model use M-RoPE for llama_decode
MTMD_API bool mtmd_decode_use_mrope(const mtmd_context * ctx);

// whether the current model supports vision input
MTMD_API bool mtmd_support_vision(const mtmd_context * ctx);

// whether the current model supports audio input
MTMD_API bool mtmd_support_audio(const mtmd_context * ctx);

// get audio sample rate in Hz, for example 16000 for Whisper
// return -1 if audio is not supported
MTMD_API int mtmd_get_audio_sample_rate(const mtmd_context * ctx);

// get the current marker string
MTMD_API const char * mtmd_get_marker(const mtmd_context * ctx);

// mtmd_bitmap
//
// if bitmap is image:
//     length of data must be nx * ny * 3
//     the data is in RGBRGBRGB... format
//     note: some video-capable models (i.e. qwen-vl) can merge consecutive bitmaps
//           into one chunk, mtmd_tokenize() will automatically handle this
// if bitmap is audio:
//     length of data must be n_samples * sizeof(float)
//     the data is in float format (PCM F32)
//
// if data == nullptr:
//     the bitmap is considered "empty", and will be treated as a placeholder for counting tokens
//     you can pass the bitmap via mtmd_tokenize(), then call mtmd_*_get_n_tokens() to count the tokens
//     note: passing a placeholder bitmap to mtmd_encode() will return an error
MTMD_API mtmd_bitmap *         mtmd_bitmap_init           (uint32_t nx, uint32_t ny, const unsigned char * data);
MTMD_API mtmd_bitmap *         mtmd_bitmap_init_from_audio(size_t n_samples,         const float         * data);
MTMD_API uint32_t              mtmd_bitmap_get_nx     (const mtmd_bitmap * bitmap);
MTMD_API uint32_t              mtmd_bitmap_get_ny     (const mtmd_bitmap * bitmap);
MTMD_API const unsigned char * mtmd_bitmap_get_data   (const mtmd_bitmap * bitmap);
MTMD_API size_t                mtmd_bitmap_get_n_bytes(const mtmd_bitmap * bitmap);
MTMD_API bool                  mtmd_bitmap_is_audio   (const mtmd_bitmap * bitmap);
MTMD_API void                  mtmd_bitmap_free       (mtmd_bitmap * bitmap);
// bitmap ID is optional, but useful for KV cache tracking
// these getters/setters are dedicated functions, so you can for example calculate the hash of the image based on mtmd_bitmap_get_data()
MTMD_API const char * mtmd_bitmap_get_id(const mtmd_bitmap * bitmap);
MTMD_API void         mtmd_bitmap_set_id(mtmd_bitmap * bitmap, const char * id);

// mtmd_bitmap lazy
//
// this is a special bitmap that:
// - does not hold the actual data
// - can be expanded into one or more chunks (either media to text chunks)
// user must provide a callback to fill in the data when mtmd_tokenize() is called
// this is useful for large video inputs:
// - allow reading video frame by frame, without loading the entire video into memory
// - allow tracking the whole video with a single ID (for example, the file hash)

// set (*out_bitmap) to non-nullptr to emit a bitmap chunk; it will be freed automatically
// set (*out_text) to non-nullptr to emit a text chunk; it must be heap-allocated, null-terminated and will be freed automatically
// either out_bitmap or out_text can be set, but not both
// out_bitmap cannot be another lazy bitmap (no nested lazy allowed)
// return value:
//    0 on success
//   -1 on EOF (signal to mtmd_tokenize to move on)
//   -2 on error (signal to mtmd_tokenize to abort)
typedef int(* mtmd_bitmap_lazy_callback)(
    size_t chunk_idx,
    void * user_data,
    mtmd_bitmap ** out_bitmap,
    char ** out_text);

MTMD_API mtmd_bitmap * mtmd_bitmap_init_lazy(mtmd_context * ctx,
                                             const char * id, // usually set to file hash
                                             void * user_data,
                                             mtmd_bitmap_lazy_callback callback);

// mtmd_input_chunks
//
// this is simply a list of mtmd_input_chunk
// the elements can only be populated via mtmd_tokenize()
MTMD_API mtmd_input_chunks *      mtmd_input_chunks_init(void);
MTMD_API size_t                   mtmd_input_chunks_size(const mtmd_input_chunks * chunks);
MTMD_API const mtmd_input_chunk * mtmd_input_chunks_get (const mtmd_input_chunks * chunks, size_t idx);
MTMD_API void                     mtmd_input_chunks_free(mtmd_input_chunks * chunks);

// mtmd_input_chunk
//
// the instance will be constructed via mtmd_tokenize()
// it will be freed along with mtmd_input_chunks
MTMD_API enum mtmd_input_chunk_type mtmd_input_chunk_get_type        (const mtmd_input_chunk * chunk);
MTMD_API const llama_token *        mtmd_input_chunk_get_tokens_text (const mtmd_input_chunk * chunk, size_t * n_tokens_output);
MTMD_API const mtmd_image_tokens *  mtmd_input_chunk_get_tokens_image(const mtmd_input_chunk * chunk);
MTMD_API size_t                     mtmd_input_chunk_get_n_tokens    (const mtmd_input_chunk * chunk);
// returns nullptr for ID on text chunk
MTMD_API const char *               mtmd_input_chunk_get_id          (const mtmd_input_chunk * chunk);
// number of temporal positions (equals to max(t,h,w) for M-RoPE; equals to n_tokens otherwise)
MTMD_API llama_pos                  mtmd_input_chunk_get_n_pos       (const mtmd_input_chunk * chunk);

// in case you want to use custom logic to handle the chunk (i.e. KV cache management)
// you can move the chunk ownership to your own code by copying it
// remember to free the chunk when you are done with it
MTMD_API mtmd_input_chunk * mtmd_input_chunk_copy(const mtmd_input_chunk * chunk);
MTMD_API void               mtmd_input_chunk_free(mtmd_input_chunk * chunk);


// mtmd_image_tokens
//
// the instance will be constructed via mtmd_tokenize()
// it will be freed along with mtmd_input_chunk
MTMD_API size_t       mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens * image_tokens); // TODO: deprecate
MTMD_API const char * mtmd_image_tokens_get_id      (const mtmd_image_tokens * image_tokens); // TODO: deprecate
// number of temporal positions (equals to max(t,h,w) for M-RoPE; equals to n_tokens otherwise)
MTMD_API llama_pos    mtmd_image_tokens_get_n_pos   (const mtmd_image_tokens * image_tokens); // TODO: deprecate

DEPRECATED(MTMD_API size_t mtmd_image_tokens_get_nx(const mtmd_image_tokens * image_tokens),
           "use mtmd_image_tokens_get_decoder_pos() instead");
DEPRECATED(MTMD_API size_t mtmd_image_tokens_get_ny(const mtmd_image_tokens * image_tokens),
           "use mtmd_image_tokens_get_decoder_pos() instead");

struct mtmd_decoder_pos {
    uint32_t t;
    uint32_t x;
    uint32_t y;
    uint32_t z; // unused for now, reserved for future use
};
// get position for decoder attention, to be used by M-RoPE models
// i is the index of the embedding token, ranging from 0 to mtmd_image_tokens_get_n_tokens() - 1
// pos_0 is the absolute position of the first token
// return relative position (for example, embedding 0 will have position (0, 0, 0); remember to adjust it to the current absolute position)
MTMD_API struct mtmd_decoder_pos mtmd_image_tokens_get_decoder_pos(const mtmd_image_tokens * image_tokens, llama_pos pos_0, size_t i);

// tokenize an input text prompt and a list of bitmaps (images/audio)
// the prompt must have the input image marker (default: "<__media__>") in it
// the default marker is defined by mtmd_default_marker()
// the marker will be replaced with the image/audio chunk
// for example:
//   "here is an image: <__media__>\ndescribe it in detail."
//   this will gives 3 chunks:
//   1. "here is an image: <start_of_image>"
//   2. (image/audio tokens)
//   3. "<end_of_image>\ndescribe it in detail."
// number of bitmaps must be equal to the number of markers in the prompt
// this function is thread-safe (shared ctx)
// return values:
//   0 on success
//   1 on number of bitmaps not matching the number of markers
//   2 on image preprocessing error
MTMD_API int32_t mtmd_tokenize(mtmd_context * ctx,
                               mtmd_input_chunks * output,
                               const mtmd_input_text * text,
                               const mtmd_bitmap ** bitmaps,
                               size_t n_bitmaps);

DEPRECATED(MTMD_API int32_t mtmd_encode(mtmd_context * ctx, const mtmd_image_tokens * image_tokens),
           "use mtmd_encode_chunk() instead");

// text chunk will be ignored silently, only media chunk will be encoded
// returns 0 on success
// returns 1 on generic error
MTMD_API int32_t mtmd_encode_chunk(mtmd_context * ctx,
                                   const mtmd_input_chunk * chunk);

// get output embeddings from the last encode pass
// the reading size (in bytes) is equal to:
// llama_model_n_embd_inp(model) * mtmd_input_chunk_get_n_tokens(chunk) * sizeof(float)
MTMD_API float * mtmd_get_output_embd(mtmd_context * ctx);


// batch encoding API
// chunks are not owned by the batch, they will not be freed by mtmd_batch_free()
// batch is valid for a given context, cannot be shared across contexts
MTMD_API mtmd_batch * mtmd_batch_init(mtmd_context * ctx);
MTMD_API void         mtmd_batch_free(mtmd_batch * batch);

// only media chunks are allowed, text chunks will be rejected
// returns 0 on success
// returns 1 on generic error
// returns 2 if the batch is too large (chunk won't be added)
// returns 3 if it cannot be batched with the existing chunks in the batch
MTMD_API int32_t mtmd_batch_add_chunk(mtmd_batch * batch, const mtmd_input_chunk * chunk);

// returns 0 on success
// returns 1 on generic error
MTMD_API int32_t mtmd_batch_encode(mtmd_batch * batch);
MTMD_API float * mtmd_batch_get_output_embd(mtmd_batch * batch, const mtmd_input_chunk * chunk);


// Set callback for all future logging events.
// If this is not called, or NULL is supplied, everything is output on stderr.
MTMD_API void mtmd_log_set(ggml_log_callback log_callback, void * user_data);

// EXPERIMENTAL API to get mmproj's capabilities without initializing the full context
// This is only intended to be used by llama-server, breaking changes is expected
struct mtmd_caps {
    bool inp_vision;
    bool inp_audio;
};
MTMD_API struct mtmd_caps mtmd_get_cap_from_file(const char * mmproj_fname);

/////////////////////////////////////////

// test function, to be used in test-mtmd-c-api.c
MTMD_API mtmd_input_chunks * mtmd_test_create_input_chunks(void);

#ifdef __cplusplus
} // extern "C"
#endif

// Get memory usage of the current model in bytes, per backend device
// Note: this is an unstable API, used internally by fit_params; it WILL be removed or changed without deprecation
#ifdef __cplusplus
MTMD_API std::map<ggml_backend_dev_t, size_t> mtmd_get_memory_usage(
    const char * mmproj_fname,
    struct mtmd_context_params ctx_params);
#endif

//
// C++ wrappers
//

#ifdef __cplusplus

namespace mtmd {

struct mtmd_context_deleter {
    void operator()(mtmd_context * val) { mtmd_free(val); }
};
using context_ptr = std::unique_ptr<mtmd_context, mtmd_context_deleter>;

struct mtmd_bitmap_deleter {
    void operator()(mtmd_bitmap * val) { mtmd_bitmap_free(val); }
};
using bitmap_ptr = std::unique_ptr<mtmd_bitmap, mtmd_bitmap_deleter>;

struct mtmd_input_chunks_deleter {
    void operator()(mtmd_input_chunks * val) { mtmd_input_chunks_free(val); }
};
using input_chunks_ptr = std::unique_ptr<mtmd_input_chunks, mtmd_input_chunks_deleter>;

struct mtmd_input_chunk_deleter {
    void operator()(mtmd_input_chunk * val) { mtmd_input_chunk_free(val); }
};
using input_chunk_ptr = std::unique_ptr<mtmd_input_chunk, mtmd_input_chunk_deleter>;

struct mtmd_batch_deleter {
    void operator()(mtmd_batch * val) { mtmd_batch_free(val); }
};
using batch_ptr = std::unique_ptr<mtmd_batch, mtmd_batch_deleter>;

struct bitmap {
    bitmap_ptr ptr;
    bitmap() : ptr(nullptr) {}
    bitmap(mtmd_bitmap * bitmap) : ptr(bitmap) {}
    bitmap(bitmap && other) noexcept : ptr(std::move(other.ptr)) {}
    bitmap(uint32_t nx, uint32_t ny, const unsigned char * data) {
        ptr.reset(mtmd_bitmap_init(nx, ny, data));
    }
    ~bitmap() = default;
    uint32_t nx() const { return mtmd_bitmap_get_nx(ptr.get()); }
    uint32_t ny() const { return mtmd_bitmap_get_ny(ptr.get()); }
    const unsigned char * data() const { return mtmd_bitmap_get_data(ptr.get()); }
    size_t n_bytes() const { return mtmd_bitmap_get_n_bytes(ptr.get()); }
    std::string id() const { return mtmd_bitmap_get_id(ptr.get()); }
    void set_id(const char * id) const { mtmd_bitmap_set_id(ptr.get(), id); }
};

struct bitmaps {
    std::vector<bitmap> entries;
    ~bitmaps() = default;
    // return list of pointers to mtmd_bitmap
    // example:
    //   auto bitmaps_c_ptr = bitmaps.c_ptr();
    //   int32_t res = mtmd_tokenize(... bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
    std::vector<const mtmd_bitmap *> c_ptr() {
        std::vector<const mtmd_bitmap *> res(entries.size());
        for (size_t i = 0; i < entries.size(); i++) {
            res[i] = entries[i].ptr.get();
        }
        return res;
    }
};

struct input_chunks {
    input_chunks_ptr ptr;
    input_chunks() = default;
    input_chunks(mtmd_input_chunks * chunks) : ptr(chunks) {}
    ~input_chunks() = default;
    size_t size() const { return mtmd_input_chunks_size(ptr.get()); }
    const mtmd_input_chunk * operator[](size_t idx) const {
        return mtmd_input_chunks_get(ptr.get(), idx);
    }
};

} // namespace mtmd

#endif

#endif
