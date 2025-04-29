#ifndef MTMD_H
#define MTMD_H

#include "ggml.h"
#include "llama.h"
#include "clip.h"

#include <vector>
#include <cinttypes>
#include <memory>

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

enum mtmd_input_chunk_type {
    MTMD_INPUT_CHUNK_TYPE_TEXT,
    MTMD_INPUT_CHUNK_TYPE_IMAGE,
};

struct mtmd_context;
struct mtmd_image_tokens;

// represents raw image data, layout is RGBRGBRGB...
// length of data must be nx * ny * 3
struct mtmd_bitmap {
    uint32_t nx;
    uint32_t ny;
    std::vector<unsigned char> data;
    std::string id; // optional user-defined id, for ex: can be set to image hash, useful for KV cache tracking
};

struct mtmd_image_tokens_deleter {
    void operator()(mtmd_image_tokens * val); // forward declaration
};
using mtmd_image_tokens_ptr = std::unique_ptr<mtmd_image_tokens, mtmd_image_tokens_deleter>;

struct mtmd_input_chunk {
    mtmd_input_chunk_type type;
    std::vector<llama_token> tokens_text;
    mtmd_image_tokens_ptr tokens_image;
};

using mtmd_input_chunks = std::vector<mtmd_input_chunk>;

struct mtmd_context_params {
    bool use_gpu = true;
    bool print_timings = true;
    int n_threads = 4;
    enum ggml_log_level verbosity = GGML_LOG_LEVEL_INFO;
    const char * image_marker = "<__image__>";
};

struct mtmd_input_text {
    std::string text;
    bool add_special;
    bool parse_special;
};

// initialize the mtmd context
// return nullptr on failure
MTMD_API mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
                                                const llama_model * text_model,
                                                const mtmd_context_params ctx_params);

MTMD_API void mtmd_free(mtmd_context * ctx);

// tokenize an input text prompt and an image
// the prompt must have the input image marker (default: "<__image__>") in it
// the marker will be replaced with the image tokens
// for example:
//   "here is an image: <__image__>\ndescribe it in detail."
//   this will gives 3 chunks:
//   1. "here is an image: <start_of_image>"
//   2. (image tokens)
//   3. "<end_of_image>\ndescribe it in detail."
// number of bitmaps must be equal to the number of image markers in the prompt
// this function is thread-safe (shared ctx)
// return values:
//   0 on success
//   1 on number of images not matching the number of markers
//   2 on image preprocessing error
MTMD_API int32_t mtmd_tokenize(mtmd_context * ctx,
                                std::vector<mtmd_input_chunk> & output,
                                const mtmd_input_text & text,
                                const std::vector<mtmd_bitmap> & bitmaps);

// access mtmd_image_tokens
MTMD_API size_t      mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens * image_tokens);
MTMD_API size_t      mtmd_image_tokens_get_nx(const mtmd_image_tokens * image_tokens);
MTMD_API size_t      mtmd_image_tokens_get_ny(const mtmd_image_tokens * image_tokens);
MTMD_API std::string mtmd_image_tokens_get_id(const mtmd_image_tokens * image_tokens);
MTMD_API llama_pos   mtmd_image_tokens_get_n_pos(const mtmd_image_tokens * image_tokens); // number of temporal positions (always 1 for M-RoPE, n_tokens otherwise)
MTMD_API void        mtmd_image_tokens_free(mtmd_image_tokens * image_tokens);

// returns 0 on success
MTMD_API int32_t mtmd_encode(mtmd_context * ctx,
                            const mtmd_image_tokens * image_tokens);

// get output embeddings from the last encode pass
MTMD_API float * mtmd_get_output_embd(mtmd_context * ctx);

// whether we need to set non-causal mask before llama_decode
MTMD_API bool mtmd_decode_use_non_causal(mtmd_context * ctx);

// whether the current model use M-RoPE for llama_decode
MTMD_API bool mtmd_decode_use_mrope(mtmd_context * ctx);



//
// helper functions (can be implemented based on other functions)
//

// helper to count the total number of tokens from a list of chunks, useful to keep track of KV cache
MTMD_API size_t mtmd_helper_get_n_tokens(mtmd_input_chunks & chunks);

// helper to count the total position of tokens from a list of chunks, useful to keep track of n_past
MTMD_API llama_pos mtmd_helper_get_n_pos(mtmd_input_chunks & chunks);

// helper function that automatically:
// 1. run llama_decode() on text chunks
// 2. run mtmd_encode() on image chunks, then mtmd_get_output_embd() and then llama_decode()
// if any of the mtmd_encode() or llama_decode() calls return non-zero, stop and forward the error
// otherwise, returns 0 on success
MTMD_API int32_t mtmd_helper_eval(mtmd_context * ctx,
                                llama_context * lctx,
                                mtmd_input_chunks & chunks,
                                llama_pos pos0,
                                llama_seq_id seq_id,
                                int32_t n_batch);

// helper function to construct a mtmd_bitmap from a file
// returns 0 on success
// this function is thread-safe
MTMD_API int32_t mtmd_helper_bitmap_init_from_file(const char * fname, mtmd_bitmap & output);

// helper function to construct a mtmd_bitmap from a buffer
// the buffer must be an image in format supported by stb_image (jpg, png, bmp, gif, etc.)
// returns 0 on success
// this function is thread-safe
MTMD_API int32_t mtmd_helper_bitmap_init_from_buf(const unsigned char * buf, size_t len, mtmd_bitmap & output);

// convenient unique_ptr wrappers
struct mtmd_context_deleter {
    void operator()(mtmd_context * val) { mtmd_free(val); }
};
using mtmd_context_ptr = std::unique_ptr<mtmd_context, mtmd_context_deleter>;

#else

static_assert(false && "C header is not yet supported by this library");

#endif

#endif
