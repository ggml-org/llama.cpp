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

struct mtmd_context {
    struct clip_ctx * ctx_clip;
    const struct llama_model * text_model;
    std::vector<float> image_embd_v; // image embedding vector
    bool print_timings;
    int n_threads;
    std::string image_marker;

    // TODO @ngxson : add timings

    mtmd_context(const char * mmproj_fname,
                   const llama_model * text_model,
                   const mtmd_context_params & ctx_params) : print_timings(ctx_params.print_timings), n_threads(ctx_params.n_threads), image_marker(ctx_params.image_marker) {
        clip_context_params ctx_clip_params;
        ctx_clip_params.use_gpu   = ctx_params.use_gpu;
        ctx_clip_params.verbosity = ctx_params.verbosity;
        ctx_clip = clip_init(mmproj_fname, ctx_clip_params);
        if (!ctx_clip) {
            throw std::runtime_error(string_format("Failed to load CLIP model from %s\n", mmproj_fname));
        }
        this->text_model = text_model;
    }

    ~mtmd_context() {
        clip_free(ctx_clip);
    }
};

struct mtmd_image_tokens_data {
    clip_image_f32_batch batch_f32; // preprocessed image patches
};

struct mtmd_image_tokens {
    uint32_t nx; // number of tokens in x direction
    uint32_t ny; // number of tokens in y direction
    uint32_t n_tokens() const { return nx * ny; }
    clip_image_f32_batch batch_f32; // preprocessed image patches
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

mtmd_input_chunks * mtmd_tokenize(mtmd_context * ctx,
                                const mtmd_input_text & text,
                                const std::vector<mtmd_bitmap> & bitmaps) {
    mtmd_input_chunks * output = new mtmd_input_chunks;
    auto vocab = llama_model_get_vocab(ctx->text_model);

    std::string prompt_modified(text.text);
    std::string marker_modified(ctx->image_marker);
    projector_type proj_type = clip_get_projector_type(ctx->ctx_clip);
    // a bit hacky here, but works for now
    // for some models, we need to add prefix and suffix to the image embeddings
    if (proj_type == PROJECTOR_TYPE_GEMMA3) {
        // <start_of_image> ... (image embeddings) ... <end_of_image>
        marker_modified = "<start_of_image>" + ctx->image_marker + "<end_of_image>";
        string_replace_all(prompt_modified, ctx->image_marker, marker_modified);
    }

    std::vector<std::string> parts = string_split_str(text.text, ctx->image_marker);
    output->clear();
    output->reserve(parts.size());

    size_t i_img = 0;

    for (const auto & part : parts) {
        //printf("tokenizing part: %s\n", part.c_str());
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
        output->emplace_back(std::move(chunk));

        if (&parts.back() != &part) {
            // add image token to middle of 2 parts

            if (i_img >= bitmaps.size()) {
                LOG_ERR("%s: error: not enough images for %d parts\n", __func__, (int)parts.size());
                return nullptr;
            }

            // shim layer
            clip_image_u8_ptr img_u8(clip_image_u8_init());
            img_u8->nx = bitmaps[i_img].nx;
            img_u8->ny = bitmaps[i_img].ny;
            img_u8->buf.resize(bitmaps[i_img].data.size());
            std::memcpy(img_u8->buf.data(), bitmaps[i_img].data.data(), img_u8->nx * img_u8->ny * 3);

            // preprocess image
            clip_image_f32_batch batch_f32;
            bool ok = clip_image_preprocess(ctx->ctx_clip, img_u8.get(), &batch_f32);
            if (!ok) {
                LOG_ERR("Unable to preprocess image\n");
                return nullptr;
            }

            mtmd_image_tokens * image_tokens = new mtmd_image_tokens;
            image_tokens->nx = clip_n_patches(ctx->ctx_clip); // TODO @ngxson : use clip_n_patches_by_image
            image_tokens->ny = 1; // TODO
            image_tokens->batch_f32 = std::move(batch_f32);

            mtmd_input_chunk chunk{
                MTMD_INPUT_CHUNK_TYPE_IMAGE,
                {},
                image_tokens,
            };
            output->emplace_back(std::move(chunk));
            i_img++;
        }
    }

    return output;
}

void mtmd_input_chunks_free(mtmd_input_chunks * chunks) {
    for (auto & chunk : *chunks) {
        if (chunk.type == MTMD_INPUT_CHUNK_TYPE_IMAGE && chunk.tokens_image) {
            delete chunk.tokens_image;
        }
    }
    delete chunks;
}

int32_t mtmd_encode(mtmd_context * ctx, const mtmd_image_tokens * image_tokens) {
    int n_mmproj_embd = clip_n_mmproj_embd(ctx->ctx_clip);
    ctx->image_embd_v.resize(image_tokens->n_tokens() * n_mmproj_embd);
    bool ok = clip_image_batch_encode(
        ctx->ctx_clip,
        ctx->n_threads,
        &image_tokens->batch_f32,
        ctx->image_embd_v.data());
    return ok ? 0 : 1;
}

float * mtmd_get_output_embd(mtmd_context * ctx) {
    return ctx->image_embd_v.data();
}

size_t mtmd_helper_get_n_tokens(mtmd_input_chunks * chunks) {
    size_t n_tokens = 0;
    for (auto & chunk : *chunks) {
        if (chunk.type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            n_tokens += chunk.tokens_text.size();
        } else if (chunk.type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            n_tokens += chunk.tokens_image->n_tokens();
        } else {
            GGML_ASSERT(false && "chunk type not supported");
        }
    }
    return n_tokens;
}

// helper struct to make working with embd batch easier
// note: this will be removed after llama_batch_ext refactoring
struct decode_embd_batch {
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch;
    decode_embd_batch(float * embd, int32_t n_tokens, llama_pos pos_0, llama_seq_id seq_id) {
        pos     .resize(n_tokens);
        n_seq_id.resize(n_tokens);
        seq_ids .resize(n_tokens + 1);
        logits  .resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0] = seq_id;
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
        for (int i = 0; i < n_tokens; i++) {
            batch.pos     [i] = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }
};

int32_t mtmd_helper_eval(mtmd_context * ctx,
        llama_context * lctx,
        mtmd_input_chunks * chunks,
        llama_pos pos0,
        llama_seq_id seq_id,
        int32_t n_batch) {
    int32_t ret;
    llama_pos n_past = pos0;
    llama_batch text_batch = llama_batch_init(n_batch, 0, 1);

    for (auto & chunk : *chunks) {
        bool is_last = &chunk == &chunks->back();
        if (chunk.type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            // TODO @ngxson : may need to split into smaller batches
            text_batch.n_tokens = chunk.tokens_text.size();
            for (size_t i = 0; i < chunk.tokens_text.size(); i++) {
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

        } else if (chunk.type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            GGML_ASSERT(!is_last && "logits for last image chunk is not yet support");
            GGML_ASSERT(chunk.tokens_image != nullptr);
            int64_t t0 = ggml_time_ms();
            if (ctx->print_timings) {
                LOG_INF("encoding image...\n");
            }
            ret = mtmd_encode(ctx, chunk.tokens_image);
            if (ret != 0) {
                LOG_ERR("failed to encode image\n");
                llama_batch_free(text_batch);
                return ret;
            }
            if (ctx->print_timings) {
                LOG_INF("image encoded in %" PRId64 " ms\n", ggml_time_ms() - t0);
            }

            int32_t n_tokens = chunk.tokens_image->n_tokens();
            float * embd = mtmd_get_output_embd(ctx);
            decode_embd_batch batch_img(embd, n_tokens, n_past, 0);
            int64_t t1 = ggml_time_ms();
            ret = llama_decode(lctx, batch_img.batch);
            if (ret != 0) {
                LOG_ERR("failed to decode image\n");
                llama_batch_free(text_batch);
                return ret;
            }
            if (ctx->print_timings) {
                LOG_INF("image decoded in %" PRId64 " ms\n", ggml_time_ms() - t1);
            }

            n_past += n_tokens;

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
