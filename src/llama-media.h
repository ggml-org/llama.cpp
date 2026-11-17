#pragma once

#include "llama.h"
#include "llama-ext.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct llama_model;
struct llama_vocab;
struct llama_context;

// the clip engine (src/clip/clip-media.h)
struct clip_image_tokens;
struct clip_audio_tokens;
struct clip_layout_config;
struct clip_audio_preprocessor;
struct clip_cpu_compute;
struct clip_gen_ref;

// how the decoder positions the rows of a media span
enum llama_media_pos_type {
    LLAMA_MEDIA_POS_NORMAL, // one position per row
    LLAMA_MEDIA_POS_MROPE,  // M-RoPE: [t, y, x] per row, the span advances by max(grid_x, grid_y)
};

// preprocessed input of one media span, read by the encoder graph
struct llama_media_input {
    enum llama_media_type type = LLAMA_MEDIA_TYPE_IMAGE;

    int32_t nx = 0;       // input width in pixels
    int32_t ny = 0;       // input height in pixels
    int32_t n_frames = 1; // frames merged into this span (temporal merge)

    // frames back to back, each [ny][nx][3] normalized RGB; empty for placeholders and reservation
    std::vector<float> data;

    // a graph of the clip engine: its runtime and request, see llama_context::clip_exec()
    struct clip_subcontext           * clip_sub = nullptr;
    struct clip_subctx_encode_params * clip_req = nullptr;
};

struct llama_media {
    std::atomic<int32_t> n_ref { 1 };

    std::string id;

    llama_media_input inp;

    const llama_model * model_enc = nullptr;

    uint32_t  n_tokens = 0; // output rows = KV cells in the decoder
    llama_pos n_pos    = 0; // positions the span advances

    llama_media_pos_type pos_type = LLAMA_MEDIA_POS_NORMAL;

    uint32_t grid_x = 0; // output grid of 2D media (M-RoPE)
    uint32_t grid_y = 0;

    bool bidirectional = false; // attention inside the span is bidirectional in the decoder
    bool placeholder   = false;

    // the chunk of the clip engine this media is: its preprocessed input and its layout in the decoder
    // exactly one is set, except for media built from raw input by the native graphs
    std::shared_ptr<clip_image_tokens> img;
    std::shared_ptr<clip_audio_tokens> aud;

    // computes the log-mel of aud at encode time (when its source needs it), on the CPU scheduler of its proc
    std::shared_ptr<clip_audio_preprocessor> aud_preproc;
    std::shared_ptr<clip_cpu_compute>        aud_cpu;

    // gen-side reference of a speaker clip (set by its encode) or of a predefined voice, see llama_gen_set_ref()
    std::shared_ptr<const clip_gen_ref> gen_ref;

    // encoded rows, host memory in this version
    int64_t            n_embd = 0;
    std::vector<float> rows;
};

struct llama_media_proc {
    llama_context     * ctx_enc   = nullptr;
    const llama_model * model_enc = nullptr;
    const llama_model * model_dec = nullptr;
    const llama_vocab * vocab     = nullptr;

    llama_media_proc_params params;

    // per-projector layout and preprocessors of the clip engine
    std::unique_ptr<clip_layout_config> cfg;

    // runs the preprocessing graphs (log-mel), shared with the media that compute later
    std::shared_ptr<clip_cpu_compute> cpu;

    // serializes the streams, they all encode on ctx_enc
    std::mutex stream_mutex;

    ~llama_media_proc();
};

struct llama_media_track;

struct llama_media_layout {
    struct piece {
        std::vector<llama_token> tokens;
        llama_media * media = nullptr; // owns one reference
    };

    const llama_media_proc * proc = nullptr;

    llama_pos pos = 0; // position of the next piece

    std::vector<piece> pieces;

    size_t   n_tokens       = 0; // tokens of the pieces so far
    uint32_t n_images_added = 0; // images laid out so far

    std::unique_ptr<llama_media_track> track; // see llama_media_layout_begin_track()

    ~llama_media_layout();

    void add_tokens(const llama_token * tokens, size_t n);
    void add_media(llama_media * m); // takes the reference
};

// the decoder positions of one row of a media span starting at pos_0, n_pos_per_embd values
void llama_media_row_pos(const llama_media & m, llama_pos pos_0, uint32_t row, llama_pos * pos, uint32_t n_pos_per_embd);

// the input of the native encoder graphs, built from the preprocessed entries of an engine media
llama_media_input llama_media_native_input(const llama_media & m);
