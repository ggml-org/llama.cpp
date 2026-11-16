#pragma once

#include "llama.h"
#include "llama-ext.h"

#include <atomic>
#include <string>
#include <vector>

struct llama_model;
struct llama_vocab;

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

    // encoded rows, host memory in this version
    int64_t            n_embd = 0;
    std::vector<float> rows;
};

struct llama_media_proc {
    const llama_model * model_enc = nullptr;
    const llama_model * model_dec = nullptr;
    const llama_vocab * vocab     = nullptr;

    llama_media_proc_params params;

    uint32_t n_rows_max = 0; // n_ubatch of the encoder context
};

struct llama_media_layout {
    struct piece {
        std::vector<llama_token> tokens;
        llama_media * media = nullptr; // owns one reference
    };

    const llama_media_proc * proc = nullptr;

    llama_pos pos = 0; // position of the next piece

    std::vector<piece> pieces;

    ~llama_media_layout();

    void add_tokens(const llama_token * tokens, size_t n);
    void add_media(llama_media * m); // takes the reference
};

// the decoder positions of one row of a media span starting at pos_0, n_pos_per_embd values
void llama_media_row_pos(const llama_media & m, llama_pos pos_0, uint32_t row, llama_pos * pos, uint32_t n_pos_per_embd);
