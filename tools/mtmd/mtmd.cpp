#include "clip.h"
#include "clip-impl.h"
#include "mtmd.h"

#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

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

struct mtmd_input_chunks {
    std::vector<mtmd_input_chunk> entries;
};

// slice template, used by some llava-uhd models to correctly place the special tokens around image embeddings
// models not having it (llava-1.6) will process embeddings without any special tokens in-between
enum mtmd_slice_tmpl {
    MTMD_SLICE_TMPL_NONE,
    MTMD_SLICE_TMPL_MINICPMV_2_5,
    MTMD_SLICE_TMPL_MINICPMV_2_6,
    // TODO @ngxson : add support for idefics (SmolVLM)
};

mtmd_context_params mtmd_context_params_default() {
    mtmd_context_params params;
    params.use_gpu = true;
    params.print_timings = true;
    params.n_threads = 4;
    params.verbosity = GGML_LOG_LEVEL_INFO;
    params.image_marker = MTMD_DEFAULT_IMAGE_MARKER;
    return params;
}

struct mtmd_context {
    struct clip_ctx * ctx_clip;
    const struct llama_model * text_model;
    std::vector<float> image_embd_v; // image embedding vector

    bool print_timings;
    int n_threads;
    std::string image_marker;

    // for minicpmv, we need special tokens in-between slices
    mtmd_slice_tmpl slice_tmpl    = MTMD_SLICE_TMPL_NONE;
    llama_token tok_ov_img_start  = LLAMA_TOKEN_NULL; // overview image
    llama_token tok_ov_img_end    = LLAMA_TOKEN_NULL; // overview image
    llama_token tok_slices_start  = LLAMA_TOKEN_NULL; // start of all slices
    llama_token tok_slices_end    = LLAMA_TOKEN_NULL; // end of all slices
    llama_token tok_sli_img_start = LLAMA_TOKEN_NULL; // single slice
    llama_token tok_sli_img_end   = LLAMA_TOKEN_NULL; // single slice
    llama_token tok_row_end       = LLAMA_TOKEN_NULL; // end of row

    bool use_mrope = false; // for Qwen2VL, we need to use M-RoPE

    // TODO @ngxson : add timings

    mtmd_context(const char * mmproj_fname,
                   const llama_model * text_model,
                   const mtmd_context_params & ctx_params) :
        text_model   (text_model),
        print_timings(ctx_params.print_timings),
        n_threads    (ctx_params.n_threads),
        image_marker (ctx_params.image_marker)
    {
        clip_context_params ctx_clip_params;
        ctx_clip_params.use_gpu   = ctx_params.use_gpu;
        ctx_clip_params.verbosity = ctx_params.verbosity;
        ctx_clip = clip_init(mmproj_fname, ctx_clip_params);
        if (!ctx_clip) {
            throw std::runtime_error(string_format("Failed to load CLIP model from %s\n", mmproj_fname));
        }

        use_mrope = clip_is_qwen2vl(ctx_clip);

        int minicpmv_version = clip_is_minicpmv(ctx_clip);
        if (minicpmv_version == 2) {
            // minicpmv 2.5 format:
            // <image> (overview) </image><slice><image> (slice) </image><image> (slice) </image>\n ... </slice>
            slice_tmpl        = MTMD_SLICE_TMPL_MINICPMV_2_5;
            tok_ov_img_start  = lookup_token("<image>");
            tok_ov_img_end    = lookup_token("</image>");
            tok_slices_start  = lookup_token("<slice>");
            tok_slices_end    = lookup_token("</slice>");
            tok_sli_img_start = tok_ov_img_start;
            tok_sli_img_end   = tok_ov_img_end;
            tok_row_end       = lookup_token("\n");

        } else if (minicpmv_version == 3 || minicpmv_version == 4) {
            // minicpmv 2.6 format:
            // <image> (overview) </image><slice> (slice) </slice><slice> (slice) </slice>\n ...
            slice_tmpl        = MTMD_SLICE_TMPL_MINICPMV_2_6;
            tok_ov_img_start  = lookup_token("<image>");
            tok_ov_img_end    = lookup_token("</image>");
            tok_sli_img_start = lookup_token("<slice>");
            tok_sli_img_end   = lookup_token("</slice>");
            tok_row_end       = lookup_token("\n");

        } else if (minicpmv_version != 0) {
            GGML_ASSERT(false && "unsupported minicpmv version");
        }
    }

    ~mtmd_context() {
        clip_free(ctx_clip);
    }

private:
    llama_token lookup_token(const std::string & token_text) {
        const llama_vocab * vocab = llama_model_get_vocab(text_model);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < n_vocab; i++) {
            if (token_to_piece(vocab, i, true) == token_text) {
                return i;
            }
        }
        return LLAMA_TOKEN_NULL;
    }

    std::string token_to_piece(const llama_vocab * vocab, llama_token token, bool special) {
        std::string piece;
        piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
        const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        if (n_chars < 0) {
            piece.resize(-n_chars);
            int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
            GGML_ASSERT(check == -n_chars);
        } else {
            piece.resize(n_chars);
        }
        return piece;
    }
};

struct mtmd_image_tokens_data {
    clip_image_f32_batch batch_f32; // preprocessed image patches
};

struct mtmd_image_tokens {
    uint32_t nx; // number of tokens in x direction
    uint32_t ny; // number of tokens in y direction
    bool use_mrope_pos = false; // use M-RoPE position counting (the whole image is 1 temporal position)
    uint32_t n_tokens() const { return nx * ny; }
    clip_image_f32_batch batch_f32; // preprocessed image patches
    std::string id; // optional user-defined ID, useful for KV cache tracking

    mtmd_image_tokens clone() {
        return mtmd_image_tokens{
            nx,
            ny,
            use_mrope_pos,
            batch_f32.clone(),
            id
        };
    }
};

mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
        const struct llama_model * text_model,
        const struct mtmd_context_params ctx_params) {
    try {
        return new mtmd_context(mmproj_fname, text_model, ctx_params);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return nullptr;
    }
}

void mtmd_free(mtmd_context * ctx) {
    if (ctx) {
        delete ctx;
    }
}

// copied from common_tokenize
static std::vector<llama_token> mtmd_tokenize_text_internal(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

int32_t mtmd_tokenize(mtmd_context * ctx,
            mtmd_input_chunks * output,
            const mtmd_input_text * text,
            const mtmd_bitmap ** bitmaps,
            size_t n_bitmaps) {
    auto vocab = llama_model_get_vocab(ctx->text_model);

    std::string prompt_modified(text->text);
    std::string marker_modified(ctx->image_marker);
    projector_type proj_type = clip_get_projector_type(ctx->ctx_clip);

    // a bit hacky here, but works for now
    // for some models, we need to add prefix and suffix to the image embeddings
    if (clip_is_gemma3(ctx->ctx_clip)) {
        // gemma 3
        // <start_of_image> ... (image embeddings) ... <end_of_image>
        marker_modified = "<start_of_image>" + ctx->image_marker + "<end_of_image>";
        string_replace_all(prompt_modified, ctx->image_marker, marker_modified);

    } else if (proj_type == PROJECTOR_TYPE_IDEFICS3) {
        // https://github.com/huggingface/transformers/blob/a42ba80fa520c784c8f11a973ca9034e5f859b79/src/transformers/models/idefics3/processing_idefics3.py#L192-L215
        marker_modified = "<fake_token_around_image><global-img>" + ctx->image_marker + "<fake_token_around_image>";
        string_replace_all(prompt_modified, ctx->image_marker, marker_modified);

    } else if (proj_type == PROJECTOR_TYPE_PIXTRAL) {
        // https://github.com/huggingface/transformers/blob/1cd110c6cb6a6237614130c470e9a902dbc1a4bd/docs/source/en/model_doc/pixtral.md
        marker_modified = ctx->image_marker + "[IMG_END]";
        string_replace_all(prompt_modified, ctx->image_marker, marker_modified);
    }

    else if (proj_type == PROJECTOR_TYPE_QWEN2VL || proj_type == PROJECTOR_TYPE_QWEN25VL) {
        // <|vision_start|> ... (image embeddings) ... <|vision_end|>
        marker_modified = "<|vision_start|>" + ctx->image_marker + "<|vision_end|>";
        string_replace_all(prompt_modified, ctx->image_marker, marker_modified);

    }

    else if (proj_type == PROJECTOR_TYPE_INTERNVL) {
        // <img> ... (image embeddings) ... </img>
        marker_modified = "<img>" + ctx->image_marker + "</img>";
        string_replace_all(prompt_modified, ctx->image_marker, marker_modified);

    }

    // llava-1.5, llava-1.6, Yi-VL, Yi-34B, granite: don't need to add prefix and suffix
    // for glm-edge, BOI and EOI token's embeddings are not present in the text model

    std::vector<std::string> parts = string_split_str(prompt_modified, ctx->image_marker);
    output->entries.clear();
    output->entries.reserve(parts.size());

    size_t i_img = 0;

    // utility for adding raw tokens
    auto add_text_chunk = [&output](std::vector<llama_token> && tokens) {
        mtmd_input_chunk chunk{
            MTMD_INPUT_CHUNK_TYPE_TEXT,
            std::move(tokens),
            {},
        };
        output->entries.emplace_back(std::move(chunk));
    };

    // utility for splitting batch of multiple images into chunks of batch having single images
    auto split_batch_to_chunk = [&ctx](clip_image_f32_batch && batch_f32, const std::string & id) {
        std::vector<mtmd_input_chunk> chunks;

        for (auto & entry : batch_f32.entries) {
            mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
            image_tokens->nx = clip_n_output_tokens(ctx->ctx_clip, entry.get());
            image_tokens->ny = 1;
            image_tokens->batch_f32.entries.push_back(std::move(entry));
            image_tokens->id = id;

            mtmd_input_chunk chunk{
                MTMD_INPUT_CHUNK_TYPE_IMAGE,
                {},
                std::move(image_tokens),
            };
            chunks.emplace_back(std::move(chunk));
        }

        return chunks;
    };

    for (const auto & part : parts) {
        // printf("tokenizing part: %s\n", part.c_str());
        bool add_bos = &parts.front() == &part;
        auto tokens = mtmd_tokenize_text_internal(vocab, part, text->add_special && add_bos, text->parse_special);
        if (tokens.empty()) {
            continue;
        }
        mtmd_input_chunk chunk{
            MTMD_INPUT_CHUNK_TYPE_TEXT,
            std::move(tokens),
            {},
        };
        output->entries.emplace_back(std::move(chunk));

        if (&parts.back() != &part) {
            // add image token to middle of 2 parts

            if (i_img >= n_bitmaps) {
                LOG_ERR("%s: error: not enough images for %d parts\n", __func__, (int)parts.size());
                return 1;
            }

            // convert mtmd_bitmap to clip_image_u8
            clip_image_u8_ptr img_u8(clip_image_u8_init());
            img_u8->nx = bitmaps[i_img]->nx;
            img_u8->ny = bitmaps[i_img]->ny;
            img_u8->buf.resize(bitmaps[i_img]->data.size());
            std::memcpy(img_u8->buf.data(), bitmaps[i_img]->data.data(), img_u8->nx * img_u8->ny * 3);
            clip_image_size img_u8_size{img_u8->nx, img_u8->ny};

            // preprocess image
            clip_image_f32_batch batch_f32;
            bool ok = clip_image_preprocess(ctx->ctx_clip, img_u8.get(), &batch_f32);
            if (!ok) {
                LOG_ERR("Unable to preprocess image\n");
                return 2;
            }

            if (ctx->slice_tmpl == MTMD_SLICE_TMPL_MINICPMV_2_5 || ctx->slice_tmpl == MTMD_SLICE_TMPL_MINICPMV_2_6) {
                // split batch into chunks of single images
                auto chunks = split_batch_to_chunk(std::move(batch_f32), bitmaps[i_img]->id);
                GGML_ASSERT(chunks.size() > 0);

                // add overview image
                add_text_chunk({ctx->tok_ov_img_start});
                output->entries.emplace_back(std::move(chunks.front()));
                chunks.erase(chunks.begin());
                add_text_chunk({ctx->tok_ov_img_end});

                // add slices
                if (!chunks.empty()) {
                    clip_add_load_image_size(ctx->ctx_clip, &img_u8_size);
                    int n_col = clip_uhd_num_image_embeds_col(ctx->ctx_clip);
                    int n_row = (int)chunks.size() / n_col;
                    GGML_ASSERT(n_row * n_col == (int)chunks.size());
                    if (ctx->tok_slices_start != LLAMA_TOKEN_NULL) {
                        add_text_chunk({ctx->tok_slices_start});
                    }
                    for (int y = 0; y < n_row; y++) {
                        for (int x = 0; x < n_col; x++) {
                            if (ctx->tok_sli_img_start != LLAMA_TOKEN_NULL) {
                                add_text_chunk({ctx->tok_sli_img_start});
                            }
                            output->entries.emplace_back(std::move(chunks[y * n_col + x]));
                            if (ctx->tok_sli_img_end != LLAMA_TOKEN_NULL) {
                                add_text_chunk({ctx->tok_sli_img_end});
                            }
                        }
                        if (ctx->tok_row_end != LLAMA_TOKEN_NULL && y != n_row - 1) {
                            add_text_chunk({ctx->tok_row_end});
                        }
                    }
                    if (ctx->tok_slices_end != LLAMA_TOKEN_NULL) {
                        add_text_chunk({ctx->tok_slices_end});
                    }
                }

            } else {
                size_t n_tokens = 0;
                for (const auto & entry : batch_f32.entries) {
                    n_tokens += clip_n_output_tokens(ctx->ctx_clip, entry.get());
                }

                mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
                if (ctx->use_mrope) {
                    // for Qwen2VL, we need this information for M-RoPE decoding positions
                    image_tokens->nx = clip_n_output_tokens_x(ctx->ctx_clip, batch_f32.entries[0].get());
                    image_tokens->ny = clip_n_output_tokens_y(ctx->ctx_clip, batch_f32.entries[0].get());
                    image_tokens->use_mrope_pos = true;
                } else {
                    // other models, we only need the total number of tokens
                    image_tokens->nx = n_tokens;
                    image_tokens->ny = 1;
                }
                image_tokens->batch_f32 = std::move(batch_f32);
                image_tokens->id = bitmaps[i_img]->id; // optional

                LOG_DBG("image_tokens->nx = %d\n", image_tokens->nx);
                LOG_DBG("image_tokens->ny = %d\n", image_tokens->ny);
                LOG_DBG("batch_f32 size = %d\n", (int)image_tokens->batch_f32.entries.size());

                mtmd_input_chunk chunk{
                    MTMD_INPUT_CHUNK_TYPE_IMAGE,
                    {},
                    std::move(image_tokens),
                };
                output->entries.emplace_back(std::move(chunk));
            }

            i_img++; // move to next image
        }
    }

    return 0;
}

static void mtmd_image_tokens_free(mtmd_image_tokens * image_tokens) {
    if (image_tokens) {
        delete image_tokens;
    }
}

int32_t mtmd_encode(mtmd_context * ctx, const mtmd_image_tokens * image_tokens) {
    int n_mmproj_embd = clip_n_mmproj_embd(ctx->ctx_clip);
    ctx->image_embd_v.resize(image_tokens->n_tokens() * n_mmproj_embd);
    bool ok = false;

    // only effective for minicpmv and qwen2vl, other models will ignore load_image_size
    {
        clip_image_size slice_size{
            image_tokens->batch_f32.entries[0]->nx,
            image_tokens->batch_f32.entries[0]->ny};
        clip_add_load_image_size(ctx->ctx_clip, &slice_size);
    }

    if (clip_is_llava(ctx->ctx_clip) || clip_is_minicpmv(ctx->ctx_clip) || clip_is_glm(ctx->ctx_clip)) {
        // TODO @ngxson : llava does not support batched encoding ; this should be fixed inside clip_image_batch_encode()
        const auto & entries = image_tokens->batch_f32.entries;
        for (size_t i = 0; i < entries.size(); i++) {
            int n_tokens_per_image = clip_n_output_tokens(ctx->ctx_clip, entries[i].get());
            ok = clip_image_encode(
                ctx->ctx_clip,
                ctx->n_threads,
                entries[i].get(),
                ctx->image_embd_v.data() + i*n_mmproj_embd*n_tokens_per_image);
        }
    } else {
        ok = clip_image_batch_encode(
            ctx->ctx_clip,
            ctx->n_threads,
            &image_tokens->batch_f32,
            ctx->image_embd_v.data());
    }

    return ok ? 0 : 1;
}

float * mtmd_get_output_embd(mtmd_context * ctx) {
    return ctx->image_embd_v.data();
}

bool mtmd_decode_use_non_causal(mtmd_context * ctx) {
    projector_type proj_type = clip_get_projector_type(ctx->ctx_clip);
    if (proj_type == PROJECTOR_TYPE_GEMMA3) {
        return true;
    }
    return false;
}

bool mtmd_decode_use_mrope(mtmd_context * ctx) {
    return ctx->use_mrope;
}

void mtmd_image_tokens_deleter::operator()(mtmd_image_tokens * val) {
    mtmd_image_tokens_free(val);
}

// these 2 helpers below use internal clip_image_u8_ptr,
// so unfortunately they cannot moved to mtmd-helper.h
// however, in theory, user can decode image file to bitmap using
// whichever library they want, and then use mtmd_bitmap_init() to create bitmap

mtmd_bitmap * mtmd_helper_bitmap_init_from_buf(const unsigned char * buf, size_t len) {
    clip_image_u8_ptr img_u8(clip_image_u8_init());
    bool ok = clip_image_load_from_bytes(buf, len, img_u8.get());
    if (!ok) {
        LOG_ERR("Unable to load image from buffer\n");
        return nullptr;
    }
    uint32_t nx, ny;
    unsigned char * data = clip_image_u8_get_data(img_u8.get(), &nx, &ny);
    return mtmd_bitmap_init(nx, ny, data);
}

mtmd_bitmap * mtmd_helper_bitmap_init_from_file(const char * fname) {
    clip_image_u8_ptr img_u8(clip_image_u8_init());
    bool ok = clip_image_load_from_file(fname, img_u8.get());
    if (!ok) {
        LOG_ERR("Unable to load image %s\n", fname);
        return nullptr;
    }
    uint32_t nx, ny;
    unsigned char * data = clip_image_u8_get_data(img_u8.get(), &nx, &ny);
    return mtmd_bitmap_init(nx, ny, data);
}

//
// public API functions
//

// mtmd_bitmap

mtmd_bitmap * mtmd_bitmap_init(uint32_t nx,
                               uint32_t ny,
                               const unsigned char * data) {
    mtmd_bitmap * bitmap = new mtmd_bitmap;
    bitmap->nx = nx;
    bitmap->ny = ny;
    size_t data_size = (size_t)nx * ny * 3;
    bitmap->data.resize(data_size);
    std::memcpy(bitmap->data.data(), data, data_size);
    return bitmap;
}

uint32_t mtmd_bitmap_get_nx(const mtmd_bitmap * bitmap) {
    return bitmap->nx;
}

uint32_t mtmd_bitmap_get_ny(const mtmd_bitmap * bitmap) {
    return bitmap->ny;
}

const unsigned char * mtmd_bitmap_get_data(const mtmd_bitmap * bitmap) {
    return bitmap->data.data();
}

const char * mtmd_bitmap_get_id(const mtmd_bitmap * bitmap) {
    return bitmap->id.c_str();
}

void mtmd_bitmap_set_id(mtmd_bitmap * bitmap, const char * id) {
    if (id) {
        bitmap->id = std::string(id);
    } else {
        bitmap->id.clear();
    }
}

void mtmd_bitmap_free(mtmd_bitmap * bitmap) {
    if (bitmap) {
        delete bitmap;
    }
}

// mtmd_input_chunks

mtmd_input_chunks * mtmd_input_chunks_init() {
    return new mtmd_input_chunks;
}

size_t mtmd_input_chunks_size(const mtmd_input_chunks * chunks) {
    return chunks->entries.size();
}

const mtmd_input_chunk * mtmd_input_chunks_get(const mtmd_input_chunks * chunks, size_t idx) {
    if (idx >= chunks->entries.size()) {
        return nullptr;
    }
    return &chunks->entries[idx];
}

void mtmd_input_chunks_free(mtmd_input_chunks * chunks) {
    if (chunks) {
        delete chunks;
    }
}

// mtmd_input_chunk

enum mtmd_input_chunk_type mtmd_input_chunk_get_type(const mtmd_input_chunk * chunk) {
    return chunk->type;
}

const llama_token * mtmd_input_chunk_get_tokens_text(const mtmd_input_chunk * chunk, size_t * n_tokens_output) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        *n_tokens_output = chunk->tokens_text.size();
        return chunk->tokens_text.data();
    }
    *n_tokens_output = 0;
    return nullptr;
}

const mtmd_image_tokens * mtmd_input_chunk_get_tokens_image(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return chunk->tokens_image.get();
    }
    return nullptr;
}

mtmd_input_chunk * mtmd_input_chunk_copy(const mtmd_input_chunk * chunk) {
    mtmd_input_chunk * copy = new mtmd_input_chunk{
        chunk->type,
        chunk->tokens_text,
        mtmd_image_tokens_ptr(),
    };
    if (chunk->tokens_image) {
        // copy the image tokens
        copy->tokens_image = mtmd_image_tokens_ptr(new mtmd_image_tokens());
        *copy->tokens_image = chunk->tokens_image->clone();
    }
    return copy;
}

void mtmd_input_chunk_free(mtmd_input_chunk * chunk) {
    if (chunk) {
        delete chunk;
    }
}

// mtmd_image_tokens

size_t mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens * image_tokens) {
    return image_tokens->n_tokens();
}

size_t mtmd_image_tokens_get_nx(const mtmd_image_tokens * image_tokens) {
    return image_tokens->nx;
}

size_t mtmd_image_tokens_get_ny(const mtmd_image_tokens * image_tokens) {
    return image_tokens->ny;
}

const char * mtmd_image_tokens_get_id(const mtmd_image_tokens * image_tokens) {
    return image_tokens->id.c_str();
}

llama_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens * image_tokens) {
    if (image_tokens->use_mrope_pos) {
        return 1; // for M-RoPE, the whole image is 1 in temporal dimension
    }
    return image_tokens->n_tokens();
}

// test function

mtmd_input_chunks * mtmd_test_create_input_chunks() {
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (!chunks) {
        return nullptr;
    }

    // create a text chunk
    std::vector<llama_token> tokens_text = { 1, 2, 3, 4, 5 };
    mtmd_input_chunk chunk_text{
        MTMD_INPUT_CHUNK_TYPE_TEXT,
        std::move(tokens_text),
        {},
    };
    chunks->entries.emplace_back(std::move(chunk_text));

    // create an image chunk
    mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
    image_tokens->nx = 4;
    image_tokens->ny = 4;
    image_tokens->batch_f32.entries.resize(16);
    image_tokens->id = "image_1";
    mtmd_input_chunk chunk_image{
        MTMD_INPUT_CHUNK_TYPE_IMAGE,
        {},
        std::move(image_tokens),
    };
    chunks->entries.emplace_back(std::move(chunk_image));

    return chunks;
}
