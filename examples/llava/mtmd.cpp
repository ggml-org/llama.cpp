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

// slice template, used by some llava-uhd models to correctly place the special tokens around image embeddings
// models not having it (llava-1.6) will process embeddings without any special tokens in-between
enum mtmd_slice_tmpl {
    MTMD_SLICE_TMPL_NONE,
    MTMD_SLICE_TMPL_MINICPMV_2_5,
    MTMD_SLICE_TMPL_MINICPMV_2_6,
    // TODO @ngxson : add support for idefics (SmolVLM)
};

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
                        std::vector<mtmd_input_chunk> & output,
                        const mtmd_input_text & text,
                        const std::vector<mtmd_bitmap> & bitmaps) {
    auto vocab = llama_model_get_vocab(ctx->text_model);

    std::string prompt_modified(text.text);
    std::string marker_modified(ctx->image_marker);
    projector_type proj_type = clip_get_projector_type(ctx->ctx_clip);

    // a bit hacky here, but works for now
    // for some models, we need to add prefix and suffix to the image embeddings
    if (clip_is_gemma3(ctx->ctx_clip)) {
        // gemma 3
        // <start_of_image> ... (image embeddings) ... <end_of_image>
        marker_modified = "<start_of_image>" + ctx->image_marker + "<end_of_image>";
        string_replace_all(prompt_modified, ctx->image_marker, marker_modified);

    } else if (proj_type == PROJECTOR_TYPE_GLM_EDGE) {
        // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>
        marker_modified = "<|begin_of_image|>" + ctx->image_marker + "<|end_of_image|>";
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

    // llava-1.5, llava-1.6, Yi-VL, Yi-34B, granite: don't need to add prefix and suffix

    std::vector<std::string> parts = string_split_str(prompt_modified, ctx->image_marker);
    output.clear();
    output.reserve(parts.size());

    size_t i_img = 0;

    // utility for adding raw tokens
    auto add_text_chunk = [&output](std::vector<llama_token> && tokens) {
        mtmd_input_chunk chunk{
            MTMD_INPUT_CHUNK_TYPE_TEXT,
            std::move(tokens),
            {},
        };
        output.emplace_back(std::move(chunk));
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
        auto tokens = mtmd_tokenize_text_internal(vocab, part, text.add_special && add_bos, text.parse_special);
        if (tokens.empty()) {
            continue;
        }
        mtmd_input_chunk chunk{
            MTMD_INPUT_CHUNK_TYPE_TEXT,
            std::move(tokens),
            {},
        };
        output.emplace_back(std::move(chunk));

        if (&parts.back() != &part) {
            // add image token to middle of 2 parts

            if (i_img >= bitmaps.size()) {
                LOG_ERR("%s: error: not enough images for %d parts\n", __func__, (int)parts.size());
                return 1;
            }

            // convert mtmd_bitmap to clip_image_u8
            clip_image_u8_ptr img_u8(clip_image_u8_init());
            img_u8->nx = bitmaps[i_img].nx;
            img_u8->ny = bitmaps[i_img].ny;
            img_u8->buf.resize(bitmaps[i_img].data.size());
            std::memcpy(img_u8->buf.data(), bitmaps[i_img].data.data(), img_u8->nx * img_u8->ny * 3);
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
                auto chunks = split_batch_to_chunk(std::move(batch_f32), bitmaps[i_img].id);
                GGML_ASSERT(chunks.size() > 0);

                // add overview image
                add_text_chunk({ctx->tok_ov_img_start});
                output.emplace_back(std::move(chunks.front()));
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
                            output.emplace_back(std::move(chunks[y * n_col + x]));
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
                image_tokens->id = bitmaps[i_img].id; // optional

                LOG_DBG("image_tokens->nx = %d\n", image_tokens->nx);
                LOG_DBG("image_tokens->ny = %d\n", image_tokens->ny);
                LOG_DBG("batch_f32 size = %d\n", (int)image_tokens->batch_f32.entries.size());

                mtmd_input_chunk chunk{
                    MTMD_INPUT_CHUNK_TYPE_IMAGE,
                    {},
                    std::move(image_tokens),
                };
                output.emplace_back(std::move(chunk));
            }

            i_img++; // move to next image
        }
    }

    return 0;
}

void mtmd_image_tokens_free(mtmd_image_tokens * image_tokens) {
    if (image_tokens) {
        delete image_tokens;
    }
}

size_t mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens * image_tokens) {
    return image_tokens->n_tokens();
}

size_t mtmd_image_tokens_get_nx(const mtmd_image_tokens * image_tokens) {
    return image_tokens->nx;
}

size_t mtmd_image_tokens_get_ny(const mtmd_image_tokens * image_tokens) {
    return image_tokens->ny;
}

std::string mtmd_image_tokens_get_id(const mtmd_image_tokens * image_tokens) {
    return image_tokens->id;
}

llama_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens * image_tokens) {
    if (image_tokens->use_mrope_pos) {
        return 1; // for M-RoPE, the whole image is 1 in temporal dimension
    }
    return image_tokens->n_tokens();
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

size_t mtmd_helper_get_n_tokens(mtmd_input_chunks & chunks) {
    size_t n_tokens = 0;
    for (auto & chunk : chunks) {
        if (chunk.type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            n_tokens += chunk.tokens_text.size();
        } else if (chunk.type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            n_tokens += mtmd_image_tokens_get_n_tokens(chunk.tokens_image.get());
        } else {
            GGML_ASSERT(false && "chunk type not supported");
        }
    }
    return n_tokens;
}

llama_pos mtmd_helper_get_n_pos(mtmd_input_chunks & chunks) {
    llama_pos n_pos = 0;
    for (auto & chunk : chunks) {
        if (chunk.type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            n_pos += chunk.tokens_text.size();
        } else if (chunk.type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            n_pos += mtmd_image_tokens_get_n_pos(chunk.tokens_image.get());
        } else {
            GGML_ASSERT(false && "chunk type not supported");
        }
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

    void set_position_mrope(llama_pos pos_0, int nx, int ny, llama_seq_id seq_id) {
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

    llama_batch get_view(int offset, int n_tokens) {
        llama_pos * pos_ptr;
        pos_view.clear();
        pos_view.resize(n_tokens * n_pos_per_embd);
        if (n_pos_per_embd > 1) {
            // mrope
            // for example, with layout of src: 1234...1234...1234...1234...
            //       offset 2 will give us dst: 34...34...34...34...
            for (int i = 0; i < n_pos_per_embd; i++) {
                auto src = pos.begin() + i * batch.n_tokens + offset;
                pos_view.insert(pos_view.end(), src, src + n_tokens);
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

int32_t mtmd_helper_eval(mtmd_context * ctx,
        llama_context * lctx,
        mtmd_input_chunks & chunks,
        llama_pos pos0,
        llama_seq_id seq_id,
        int32_t n_batch) {
    int32_t ret;
    llama_pos n_past = pos0;
    llama_batch text_batch = llama_batch_init(n_batch, 0, 1);
    int n_mmproj_embd = clip_n_mmproj_embd(ctx->ctx_clip);
    int n_pos_per_embd = mtmd_decode_use_mrope(ctx) ? 4 : 1;

    for (auto & chunk : chunks) {
        bool is_last = &chunk == &chunks.back();
        if (chunk.type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            text_batch.n_tokens = chunk.tokens_text.size();
            size_t i = 0;
            while (i < chunk.tokens_text.size()) { // split into batches
                for (; i < chunk.tokens_text.size() && text_batch.n_tokens < n_batch; i++) {
                    text_batch.token   [i]    = chunk.tokens_text[i];
                    text_batch.pos     [i]    = n_past++;
                    text_batch.n_seq_id[i]    = 1;
                    text_batch.seq_id  [i][0] = seq_id;
                    text_batch.logits  [i]    = false;
                }
                if (is_last) {
                    // always get logits for last input chunk
                    text_batch.logits[text_batch.n_tokens - 1] = true;
                }
                ret = llama_decode(lctx, text_batch);
                if (ret != 0) {
                    LOG_ERR("failed to decode text\n");
                    llama_batch_free(text_batch);
                    return ret;
                }
            }

        } else if (chunk.type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            GGML_ASSERT(!is_last && "logits for last image chunk is not yet support");
            GGML_ASSERT(chunk.tokens_image != nullptr);
            int64_t t0 = ggml_time_ms();
            if (ctx->print_timings) {
                LOG_INF("encoding image or slice...\n");
            }
            ret = mtmd_encode(ctx, chunk.tokens_image.get());
            if (ret != 0) {
                LOG_ERR("failed to encode image\n");
                llama_batch_free(text_batch);
                return ret;
            }
            if (ctx->print_timings) {
                LOG_INF("image/slice encoded in %" PRId64 " ms\n", ggml_time_ms() - t0);
            }

            int32_t n_tokens = mtmd_image_tokens_get_n_tokens(chunk.tokens_image.get());
            int32_t i_batch = 0;
            int32_t n_img_batches = GGML_PAD(n_tokens, n_batch) / n_batch;
            float * embd = mtmd_get_output_embd(ctx);
            decode_embd_batch batch_embd(embd, n_tokens, n_pos_per_embd, n_mmproj_embd);

            const int nx = mtmd_image_tokens_get_nx(chunk.tokens_image.get());
            const int ny = mtmd_image_tokens_get_ny(chunk.tokens_image.get());

            if (mtmd_decode_use_mrope(ctx)) {
                batch_embd.set_position_mrope(n_past, nx, ny, seq_id);
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

                LOG_INF("decoding image batch %d/%d, n_tokens_batch = %d\n", i_batch+1, n_img_batches, n_tokens_batch);

                int64_t t1 = ggml_time_ms();
                ret = llama_decode(lctx, batch_embd_view);
                if (ret != 0) {
                    LOG_ERR("failed to decode image\n");
                    llama_set_causal_attn(lctx, true); // restore causal attn
                    llama_batch_free(text_batch);
                    return ret;
                }

                if (ctx->print_timings) {
                    LOG_INF("image decoded (batch %d/%d) in %" PRId64 " ms\n", i_batch+1, n_img_batches, ggml_time_ms() - t1);
                }

                i_batch++;
            }

            // for mrope, one image is one single **temporal** position
            n_past += mtmd_decode_use_mrope(ctx) ? 1 : n_tokens;

            if (mtmd_decode_use_non_causal(ctx)) {
                llama_set_causal_attn(lctx, true);
            }

        } else {
            GGML_ASSERT(false && "chunk type not supported");
        }
    }

    llama_batch_free(text_batch);
    return 0;
}

int32_t mtmd_helper_bitmap_init_from_buf(const unsigned char * buf, size_t len, mtmd_bitmap & output) {
    clip_image_u8_ptr img_u8(clip_image_u8_init());
    bool ok = clip_image_load_from_bytes(buf, len, img_u8.get());
    if (!ok) {
        LOG_ERR("Unable to load image from buffer\n");
        return 1;
    }
    unsigned char * data = clip_image_u8_get_data(img_u8.get(), &output.nx, &output.ny);
    output.data.resize(output.nx * output.ny * 3);
    std::memcpy(output.data.data(), data, output.nx * output.ny * 3);
    return 0;
}

int32_t mtmd_helper_bitmap_init_from_file(const char * fname, mtmd_bitmap & output) {
    clip_image_u8_ptr img_u8(clip_image_u8_init());
    bool ok = clip_image_load_from_file(fname, img_u8.get());
    if (!ok) {
        LOG_ERR("Unable to load image %s\n", fname);
        return 1;
    }
    unsigned char * data = clip_image_u8_get_data(img_u8.get(), &output.nx, &output.ny);
    output.data.resize(output.nx * output.ny * 3);
    std::memcpy(output.data.data(), data, output.nx * output.ny * 3);
    return 0;
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
