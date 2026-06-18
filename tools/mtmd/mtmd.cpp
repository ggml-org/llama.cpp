#include "clip.h"
#include "clip-impl.h"
#include "mtmd.h"
#include "mtmd-audio.h"
#include "mtmd-image.h"
#include "debug/mtmd-debug.h"

#include "llama.h"

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>

// for still image data, layout is RGBRGBRGB...
// length of data must be nx * ny * 3 bytes
//
// for audio bitmap: nx = sample count, ny = 1, layout is F32 F32 F32 ...
// length of data must be nx * sizeof(float) bytes
struct mtmd_bitmap {
    uint32_t nx = 0;
    uint32_t ny = 0;
    std::string id; // optional user-defined id, for ex: can be set to image hash, useful for KV cache tracking
    bool is_audio = false; // true if the bitmap is audio

    // lazy-loaded bitmap
    mtmd_bitmap_lazy_callback lazy_callback = nullptr;
    void * lazy_user_data = nullptr;

    mtmd_bitmap(const unsigned char * data, uint32_t nx, uint32_t ny)
        : nx(nx), ny(ny), is_audio(false) {
        if (data) {
            size_t data_size = (size_t)nx * ny * 3;
            this->data.resize(data_size);
            std::memcpy(this->data.data(), data, data_size);
        }
    }

    mtmd_bitmap(const unsigned char * data, uint32_t n_samples)
        : nx(n_samples), ny(1), is_audio(true) {
        if (data) {
            size_t data_size = (size_t)nx * sizeof(float);
            this->data.resize(data_size);
            std::memcpy(this->data.data(), data, data_size);
        }
    }

    const std::vector<unsigned char> & get_ro_buf() const {
        return data;
    }

    bool is_placeholder() const {
        return data.empty();
    }

    size_t n_bytes() const {
        return data.size();
    }

    bool can_merge_with(const mtmd_bitmap & other) const {
        // [QWEN_VIDEO] can (temporal) merge if both are images with same size
        return !is_audio && !other.is_audio && nx == other.nx && ny == other.ny;
    }

  private:
    std::vector<unsigned char> data;
};

// position indexing for decoder model
enum mtmd_pos_type {
    MTMD_POS_TYPE_NORMAL,    // number of positions equals to number of tokens
    MTMD_POS_TYPE_MROPE,     // qwen-vl mrope style, each image takes max(t,h,w) position indexes
    MTMD_POS_TYPE_HUNYUANVL, // HunyuanVL mrope + BOI/EOI/newline layout with XD-RoPE dim-3
};

struct mtmd_image_tokens {
    uint32_t nx = 0; // number of tokens in x direction
    uint32_t ny = 0; // number of tokens in y direction
    mtmd_pos_type pos = MTMD_POS_TYPE_NORMAL;
    uint32_t image_idx = 0; // 0-based position of this image among image chunks in the prompt(used by pos == MTMD_POS_TYPE_HUNYUANVL)
    uint32_t n_temporal_merge = 1; // for qwen-vl style temporal merge
    uint32_t n_tokens() const {
        if (pos == MTMD_POS_TYPE_HUNYUANVL) {
            // [BOI] [row0 tokens + newline] ... [row(ny-1) tokens + newline] [EOI]
            return (nx + 1) * ny + 2;
        }
        uint32_t nz = batch_f32.entries.size();
        if (n_temporal_merge > 1) {
            // [QWEN_VIDEO] this logic is quite ugly, it's mostly to make qwen-vl temporal merge work, can be improved in the future
            // TODO: simplify this by repeating the last frame until it fits the temporal merge
            if (nz % n_temporal_merge != 0) {
                nz = nz / n_temporal_merge + 1;
            } else {
                nz = nz / n_temporal_merge;
            }
        }
        return nx * ny * nz;
    }
    clip_image_f32_batch batch_f32; // preprocessed image patches
    std::string id; // optional user-defined ID, useful for KV cache tracking

    // true if one of entries in batch_f32 is a placeholder
    bool is_placeholder() const {
        for (const auto & entry : batch_f32.entries) {
            if (entry.is_placeholder()) {
                return true;
            }
        }
        return false;
    }

    bool can_batch_with(const mtmd_image_tokens & other) {
        return nx == other.nx && ny == other.ny && pos == other.pos;
    }

    mtmd_image_tokens clone() {
        return mtmd_image_tokens{
            nx,
            ny,
            pos,
            image_idx,
            n_temporal_merge,
            batch_f32.clone(),
            id
        };
    }
};
using mtmd_image_tokens_ptr = std::unique_ptr<mtmd_image_tokens>;

struct mtmd_audio_tokens {
    uint32_t n_tokens = 0; // number of tokens
    clip_image_f32_batch batch_f32; // preprocessed image patches
    std::string id; // optional user-defined ID, useful for KV cache tracking

    // true if one of entries in batch_f32 is a placeholder
    bool is_placeholder() const {
        for (const auto & entry : batch_f32.entries) {
            if (entry.is_placeholder()) {
                return true;
            }
        }
        return false;
    }

    mtmd_audio_tokens clone() {
        return mtmd_audio_tokens{
            n_tokens,
            batch_f32.clone(),
            id
        };
    }
};
using mtmd_audio_tokens_ptr = std::unique_ptr<mtmd_audio_tokens>;

struct mtmd_input_chunk {
    mtmd_input_chunk_type type;
    std::vector<llama_token> tokens_text;
    mtmd_image_tokens_ptr tokens_image;
    mtmd_audio_tokens_ptr tokens_audio;

    bool can_batch_with(const mtmd_input_chunk & other) const {
        if (type != other.type) {
            return false;
        }

        if (tokens_image && other.tokens_image) {
            return tokens_image->can_batch_with(*other.tokens_image);
        }

        // TODO: allow batching audio chunks of the same size

        return false;
    }

    bool is_placeholder() const {
        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            return tokens_image && tokens_image->is_placeholder();
        } else if (type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            return tokens_audio && tokens_audio->is_placeholder();
        }
        return false;
    }
};

struct mtmd_input_chunks {
    std::vector<mtmd_input_chunk> entries;
};

struct mtmd_batch {
    mtmd_context * ctx;
    std::vector<const mtmd_input_chunk *> entries;
    std::vector<float> output_embd; // aggregated output embedding for the whole batch
    mtmd_batch(mtmd_context * ctx): ctx(ctx) {}
    int32_t n_tokens() const {
        int32_t n = 0;
        for (const auto * chunk : entries) {
            n += mtmd_input_chunk_get_n_tokens(chunk);
        }
        return n;
    }
};

// slice template, used by some llava-uhd models to correctly place the special tokens around image embeddings
// models not having it (llava-1.6) will process embeddings without any special tokens in-between
enum mtmd_slice_tmpl {
    MTMD_SLICE_TMPL_NONE,
    MTMD_SLICE_TMPL_MINICPMV_2_5,
    MTMD_SLICE_TMPL_MINICPMV_2_6,
    MTMD_SLICE_TMPL_LLAMA4,
    MTMD_SLICE_TMPL_IDEFICS3,
    MTMD_SLICE_TMPL_LFM2,
    MTMD_SLICE_TMPL_STEP3VL,
};

const char * mtmd_default_marker() {
    return "<__media__>";
}

static clip_flash_attn_type mtmd_get_clip_flash_attn_type(enum llama_flash_attn_type flash_attn_type) {
    switch (flash_attn_type) {
        case LLAMA_FLASH_ATTN_TYPE_AUTO:     return CLIP_FLASH_ATTN_TYPE_AUTO;
        case LLAMA_FLASH_ATTN_TYPE_DISABLED: return CLIP_FLASH_ATTN_TYPE_DISABLED;
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:  return CLIP_FLASH_ATTN_TYPE_ENABLED;
    }
    return CLIP_FLASH_ATTN_TYPE_AUTO;
}

mtmd_context_params mtmd_context_params_default() {
    mtmd_context_params params {
        /* use_gpu           */ true,
        /* print_timings     */ true,
        /* n_threads         */ 4,
        /* image_marker      */ nullptr,
        /* media_marker      */ mtmd_default_marker(),
        /* flash_attn_type   */ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /* warmup            */ true,
        /* image_min_tokens  */ -1,
        /* image_max_tokens  */ -1,
        /* cb_eval           */ nullptr,
        /* cb_eval_user_data */ nullptr,
        /* batch_max_tokens  */ 1024,
    };
    return params;
}

struct mtmd_context {
    struct clip_ctx * ctx_v; // vision
    struct clip_ctx * ctx_a; // audio
    std::vector<float> out_embd; // image embedding vector

    bool print_timings;
    int n_threads;
    std::string media_marker;
    const int n_embd_text = -1; // -1 means llm context not provided, skip checking this
    const llama_vocab * vocab = nullptr; // can be nullptr if text_model is not provided
    mtmd_pos_type pos_type;

    // these are not token, but strings used to mark the beginning and end of image/audio embeddings
    std::string img_beg;
    std::string img_end;
    std::string aud_beg;
    std::string aud_end;

    // for llava-uhd style models, we need special tokens in-between slices
    // minicpmv calls them "slices", llama 4 calls them "tiles"
    mtmd_slice_tmpl slice_tmpl = MTMD_SLICE_TMPL_NONE;
    std::vector<llama_token> tok_ov_img_start;  // overview image
    std::vector<llama_token> tok_ov_img_end;    // overview image
    std::vector<llama_token> tok_slices_start;  // start of all slices
    std::vector<llama_token> tok_slices_end;    // end of all slices
    std::vector<llama_token> tok_sli_img_start; // single slice start
    std::vector<llama_token> tok_sli_img_end;   // single slice end
    std::vector<llama_token> tok_sli_img_mid;   // between 2 slices
    std::vector<llama_token> tok_row_end;       // end of row
    bool tok_row_end_trail = false;
    bool ov_img_first      = false;

    // string template for slice image delimiters with row/col (idefics3)
    std::string sli_img_start_tmpl;

    std::unique_ptr<mtmd_audio_preprocessor> audio_preproc;
    std::unique_ptr<mtmd_image_preprocessor> image_preproc;

    // batching
    int32_t batch_max_tokens;

    // TODO @ngxson : add timings

    mtmd_context(const char * mmproj_fname,
                   const llama_model * text_model,
                   const mtmd_context_params & ctx_params,
                   bool no_alloc = false) :
        print_timings   (ctx_params.print_timings),
        n_threads       (ctx_params.n_threads),
        media_marker    (ctx_params.media_marker),
        n_embd_text     (text_model ? llama_model_n_embd_inp(text_model) : -1),
        vocab           (text_model ? llama_model_get_vocab(text_model) : nullptr),
        batch_max_tokens(ctx_params.batch_max_tokens)
    {
        if (ctx_params.image_marker != nullptr) {
            throw std::runtime_error("custom image_marker is not supported anymore, use media_marker instead");
        }

        if (media_marker.empty()) {
            throw std::runtime_error("media_marker must not be empty");
        }

        if (text_model) {
            auto decoder_rope_type = llama_model_rope_type(text_model);
            switch (decoder_rope_type) {
                case LLAMA_ROPE_TYPE_NONE:
                case LLAMA_ROPE_TYPE_NORM:
                case LLAMA_ROPE_TYPE_NEOX:
                    {
                        pos_type = MTMD_POS_TYPE_NORMAL;
                    } break;
                case LLAMA_ROPE_TYPE_MROPE:
                case LLAMA_ROPE_TYPE_IMROPE:
                    {
                        pos_type = MTMD_POS_TYPE_MROPE;
                    } break;
                default:
                    throw std::runtime_error(string_format("unsupported decoder rope type: %d\n", decoder_rope_type));
            }
        }

        clip_context_params ctx_clip_params {
            /* use_gpu           */ ctx_params.use_gpu,
            /* flash_attn_type   */ mtmd_get_clip_flash_attn_type(ctx_params.flash_attn_type),
            /* image_min_tokens  */ ctx_params.image_min_tokens,
            /* image_max_tokens  */ ctx_params.image_max_tokens,
            /* warmup            */ ctx_params.warmup,
            /* cb_eval           */ ctx_params.cb_eval,
            /* cb_eval_user_data */ ctx_params.cb_eval_user_data,
            /* no_alloc          */ no_alloc,
        };

        auto res = clip_init(mmproj_fname, ctx_clip_params);
        ctx_v = res.ctx_v;
        ctx_a = res.ctx_a;
        if (!ctx_v && !ctx_a) {
            throw std::runtime_error(string_format("Failed to load CLIP model from %s\n", mmproj_fname));
        }

        // if both vision and audio mmproj are present, we need to validate their n_embd
        if (ctx_v && ctx_a) {
            int n_embd_v = clip_n_mmproj_embd(ctx_v);
            int n_embd_a = clip_n_mmproj_embd(ctx_a);
            if (n_embd_v != n_embd_a) {
                throw std::runtime_error(string_format(
                    "mismatch between vision and audio mmproj (n_embd_v = %d, n_embd_a = %d)\n",
                    n_embd_v, n_embd_a));
            }
        }

        // since we already validate n_embd of vision and audio mmproj,
        // we can safely assume that they are the same
        int n_embd_clip = clip_n_mmproj_embd(ctx_v ? ctx_v : ctx_a);
        if (n_embd_text > 0 && n_embd_text != n_embd_clip) {
            throw std::runtime_error(string_format(
                "mismatch between text model (n_embd = %d) and mmproj (n_embd = %d)\n"
                "hint: you may be using wrong mmproj\n",
                n_embd_text, n_embd_clip));
        }
        if (ctx_v) {
            init_vision();
        }
        if (ctx_a) {
            init_audio();
        }
    }

    void init_vision() {
        GGML_ASSERT(ctx_v != nullptr);
        image_preproc.reset();

        projector_type proj = clip_get_projector_type(ctx_v);

        switch (proj) {
            case PROJECTOR_TYPE_MLP:
            case PROJECTOR_TYPE_MLP_NORM:
            case PROJECTOR_TYPE_LDP:
            case PROJECTOR_TYPE_LDPV2:
            case PROJECTOR_TYPE_COGVLM:
            case PROJECTOR_TYPE_JANUS_PRO:
            case PROJECTOR_TYPE_GLM_EDGE:
                {
                    bool has_pinpoints = !clip_get_hparams(ctx_v)->image_res_candidates.empty();
                    if (has_pinpoints) {
                        image_preproc = std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx_v);
                    } else {
                        image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx_v);
                    }
                } break;
            case PROJECTOR_TYPE_MINICPMV:
                {
                    int minicpmv_version = clip_get_hparams(ctx_v)->minicpmv_version;
                    if (minicpmv_version == 2) {
                        // minicpmv 2.5 format:
                        // <image> (overview) </image><slice><image> (slice) </image><image> (slice) </image>\n ... </slice>
                        slice_tmpl        = MTMD_SLICE_TMPL_MINICPMV_2_5;
                        tok_ov_img_start  = {lookup_token("<image>")};
                        tok_ov_img_end    = {lookup_token("</image>")};
                        tok_slices_start  = {lookup_token("<slice>")};
                        tok_slices_end    = {lookup_token("</slice>")};
                        tok_sli_img_start = tok_ov_img_start;
                        tok_sli_img_end   = tok_ov_img_end;
                        tok_row_end       = {lookup_token("\n")};
                        tok_row_end_trail = false; // no trailing end-of-row token
                        ov_img_first      = true;
                    } else if (minicpmv_version == 3 || minicpmv_version == 4 || minicpmv_version == 5 || minicpmv_version == 6 || minicpmv_version == 100045) {
                        // minicpmv 2.6 format:
                        // <image> (overview) </image><slice> (slice) </slice><slice> (slice) </slice>\n ...
                        slice_tmpl        = MTMD_SLICE_TMPL_MINICPMV_2_6;
                        tok_ov_img_start  = {lookup_token("<image>")};
                        tok_ov_img_end    = {lookup_token("</image>")};
                        tok_sli_img_start = {lookup_token("<slice>")};
                        tok_sli_img_end   = {lookup_token("</slice>")};
                        tok_row_end       = {lookup_token("\n")};
                        tok_row_end_trail = false; // no trailing end-of-row token
                        ov_img_first      = true;

                    } else if (minicpmv_version != 0) {
                        throw std::runtime_error(string_format("unsupported minicpmv version: %d\n", minicpmv_version));
                    }
                    image_preproc = std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx_v);
                } break;
            case PROJECTOR_TYPE_MINICPMV4_6:
                {
                    slice_tmpl        = MTMD_SLICE_TMPL_MINICPMV_2_6;
                    tok_ov_img_start  = {lookup_token("<image>")};
                    tok_ov_img_end    = {lookup_token("</image>")};
                    tok_sli_img_start = {lookup_token("<slice>")};
                    tok_sli_img_end   = {lookup_token("</slice>")};
                    tok_row_end       = {lookup_token("\n")};
                    tok_row_end_trail = false; // no trailing end-of-row token
                    ov_img_first      = true;
                    image_preproc     = std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx_v);
                } break;
            case PROJECTOR_TYPE_QWEN2VL:
            case PROJECTOR_TYPE_QWEN25VL:
            case PROJECTOR_TYPE_QWEN3VL:
            case PROJECTOR_TYPE_MIMOVL:
                {
                    // <|vision_start|> ... (image embeddings) ... <|vision_end|>
                    img_beg = "<|vision_start|>";
                    img_end = "<|vision_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_YOUTUVL:
                {
                    // <|vision_start|> ... (image embeddings) ... <|vision_end|>
                    img_beg = "<|vision_start|>";
                    img_end = "<|vision_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_youtuvl>(ctx_v);
                } break;
            case PROJECTOR_TYPE_YASA2:
                {
                    img_beg = "<image>";
                    img_end = "</image>";
                    // Currently only supprots single-tile preprocessing: any input is downscaled
                    // to one image_size x image_size tile (64 output tokens via 8x8 adaptive avg
                    // pool).
                    // However, the model itself supports llava-uhd multi-tile tiling for high-res
                    // images. This will be implemented in a future PR (dispatch on has_pinpoints
                    // - see LDP/COGVLM branch above) and emit image_grid_pinpoints in the conversion
                    // script.
                    image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_GEMMA3:
            case PROJECTOR_TYPE_GEMMA3NV:
                {
                    // <start_of_image> ... (image embeddings) ... <end_of_image>
                    img_beg = "<start_of_image>";
                    img_end = "<end_of_image>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_IDEFICS3:
                {
                    // https://github.com/huggingface/transformers/blob/a42ba80fa520c784c8f11a973ca9034e5f859b79/src/transformers/models/idefics3/processing_idefics3.py#L192-L215
                    slice_tmpl         = MTMD_SLICE_TMPL_IDEFICS3;
                    tok_ov_img_start   = {lookup_token("\n\n"), lookup_token("<fake_token_around_image>"), lookup_token("<global-img>")};
                    tok_ov_img_end     = {lookup_token("<fake_token_around_image>")};
                    tok_row_end        = {lookup_token("\n")};
                    sli_img_start_tmpl = "<fake_token_around_image><row_%d_col_%d>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_idefics3>(ctx_v);
                } break;
            case PROJECTOR_TYPE_PIXTRAL:
                {
                    // https://github.com/huggingface/transformers/blob/1cd110c6cb6a6237614130c470e9a902dbc1a4bd/docs/source/en/model_doc/pixtral.md
                    img_end = "[IMG_END]";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_PHI4:
                {
                    // Phi-4 uses media marker insertion only. Keep image boundary text empty.
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_LLAMA4:
                {
                    // (more details in mtmd_context constructor)
                    img_beg = "<|image_start|>";
                    img_end = "<|image_end|>";
                    LOG_WRN("%s: llama 4 vision is known to have degraded quality:\n"
                            "    https://github.com/ggml-org/llama.cpp/pull/13282\n", __func__);
                    image_preproc = std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx_v);
                    ov_img_first = false;
                } break;
            case PROJECTOR_TYPE_STEP3VL:
                {
                    // Step3 format:
                    //   <patch_start> (patch) <patch_end> [<patch_newline>]
                    //   ... (all patch rows)
                    //   <im_start> (overview) <im_end>
                    slice_tmpl        = MTMD_SLICE_TMPL_STEP3VL;
                    tok_ov_img_start  = {lookup_token("<im_start>")};
                    tok_ov_img_end    = {lookup_token("<im_end>")};
                    tok_sli_img_start = {lookup_token("<patch_start>")};
                    tok_sli_img_end   = {lookup_token("<patch_end>")};
                    tok_row_end       = {lookup_token("<patch_newline>")};
                    tok_row_end_trail = false;
                    ov_img_first      = false; // patches first, overview last
                    image_preproc = std::make_unique<mtmd_image_preprocessor_step3vl>(ctx_v);
                } break;
            case PROJECTOR_TYPE_INTERNVL:
                {
                    // <img> ... (image embeddings) ... </img>
                    img_beg = "<img>";
                    img_end = "</img>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_internvl>(ctx_v);
                    ov_img_first = false;
                } break;
            case PROJECTOR_TYPE_KIMIVL:
                {
                    // <|media_start|> ... (image embeddings) ... <|media_end|>
                    img_beg = "<|media_start|>";
                    img_end = "<|media_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_KIMIK25:
                {
                    // <|media_begin|> ... (image embeddings) ... <|media_end|>
                    img_beg = "<|media_begin|>";
                    img_end = "<|media_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_LIGHTONOCR:
                {
                    // <|im_start|> ... (image embeddings) ... <|im_end|>
                    img_beg = "<|im_start|>";
                    img_end = "<|im_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_longest_edge>(ctx_v);
                } break;
            case PROJECTOR_TYPE_DOTS_OCR:
                {
                    // <|img|> ... (image embeddings) ... <|endofimg|>
                    img_beg = "<|img|>";
                    img_end = "<|endofimg|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_NEMOTRON_V2_VL:
                {
                    image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_LFM2:
                {
                    // multi-tile:
                    //   <|image_start|>
                    //     <|img_row_1_col_1|> (tile) <|img_row_1_col_2|> (tile) ...
                    //     <|img_thumbnail|> (thumbnail)
                    //   <|image_end|>
                    // single-tile:
                    //   <|image_start|> (image) <|image_end|>
                    img_beg            = "<|image_start|>";
                    img_end            = "<|image_end|>";
                    slice_tmpl         = MTMD_SLICE_TMPL_LFM2;
                    sli_img_start_tmpl = "<|img_row_%d_col_%d|>";
                    tok_ov_img_start   = {lookup_token("<|img_thumbnail|>")};
                    ov_img_first       = false;
                    image_preproc = std::make_unique<mtmd_image_preprocessor_lfm2>(ctx_v);
                } break;
            case PROJECTOR_TYPE_GLM4V:
                {
                    // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>
                    img_beg = "<|begin_of_image|>";
                    img_end = "<|end_of_image|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_PADDLEOCR:
                {
                    // <|IMAGE_START|> ... (image embeddings) ... <|IMAGE_END|>
                    img_beg = "<|IMAGE_START|>";
                    img_end = "<|IMAGE_END|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_GEMMA4V:
            case PROJECTOR_TYPE_GEMMA4UV:
                {
                    // <|image> ... (image embeddings) ... <image|>
                    img_beg = "<|image>";
                    img_end = "<image|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_DEEPSEEKOCR:
                {
                    img_end = "\n"; // prevent empty batch on llama-server
                    image_preproc = std::make_unique<mtmd_image_preprocessor_deepseekocr>(ctx_v);
                    ov_img_first = false;
                } break;
            case PROJECTOR_TYPE_DEEPSEEKOCR2:
                {
                    img_end = "\n"; // prevent empty batch on llama-server
                    image_preproc = std::make_unique<mtmd_image_preprocessor_deepseekocr2>(ctx_v);
                    ov_img_first = false;
                } break;
            case PROJECTOR_TYPE_HUNYUANVL:
                {
                    // note: these use fullwidth ｜ (U+FF5C) and ▁ (U+2581) to match the tokenizer vocabulary
                    img_beg = "<｜hy_place▁holder▁no▁100｜>";
                    img_end = "<｜hy_place▁holder▁no▁101｜>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_EXAONE4_5:
                {
                    // <vision> ... (image embeddings) ... </vision>
                    img_beg = "<vision>";
                    img_end = "</vision>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_GRANITE4_VISION:
                {
                    img_beg = "<image>";
                    img_end = "";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_granite>(ctx_v);
                    ov_img_first = true;
                } break;
            default:
                throw std::runtime_error(string_format("%s: unexpected vision projector type %d\n", __func__, proj));
        }

        GGML_ASSERT(image_preproc != nullptr);
    }

    void init_audio() {
        GGML_ASSERT(ctx_a != nullptr);
        audio_preproc.reset();

        projector_type proj = clip_get_projector_type(ctx_a);

        LOG_WRN("%s: audio input is in experimental stage and may have reduced quality:\n"
                "    https://github.com/ggml-org/llama.cpp/discussions/13759\n", __func__);

        // set preprocessor
        switch (proj) {
            case PROJECTOR_TYPE_QWEN2A:
            case PROJECTOR_TYPE_QWEN25O:
                {
                    // <|audio_bos|> ... (embeddings) ... <|audio_eos|>
                    aud_beg = "<|audio_bos|>";
                    aud_end = "<|audio_eos|>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_whisper>(ctx_a);
                } break;
            case PROJECTOR_TYPE_QWEN3A:
                {
                    aud_beg = "<|audio_start|>";
                    aud_end = "<|audio_end|>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_qwen3a>(ctx_a);
                } break;
            case PROJECTOR_TYPE_VOXTRAL:
                {
                    // [BEGIN_AUDIO] ... (embeddings) ...
                    aud_beg = "[BEGIN_AUDIO]";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_whisper>(ctx_a);
                } break;
            case PROJECTOR_TYPE_MUSIC_FLAMINGO:
                {
                    // <sound> ... (embeddings) ...
                    aud_beg = "<sound>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_whisper>(ctx_a);
                } break;
            case PROJECTOR_TYPE_ULTRAVOX:
            case PROJECTOR_TYPE_GLMA:
            case PROJECTOR_TYPE_MERALION:
                {
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_whisper>(ctx_a);
                } break;
            case PROJECTOR_TYPE_LFM2A:
                {
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_conformer>(ctx_a);
                } break;
            case PROJECTOR_TYPE_GRANITE_SPEECH:
                {
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_granite_speech>(ctx_a);
                } break;
            case PROJECTOR_TYPE_GEMMA4A:
                {
                    aud_beg = "<|audio>";
                    aud_end = "<audio|>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_gemma4a>(ctx_a);
                } break;
            case PROJECTOR_TYPE_GEMMA4UA:
                {
                    aud_beg = "<|audio>";
                    aud_end = "<audio|>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_gemma4ua>(ctx_a);
                } break;
            default:
                throw std::runtime_error(string_format("%s: unexpected audio projector type %d\n", __func__, proj));
        }

        // initialize audio preprocessor
        GGML_ASSERT(audio_preproc != nullptr);
        audio_preproc->initialize();
    }

    // get clip ctx based on chunk type
    clip_ctx * get_clip_ctx(const mtmd_input_chunk * chunk) const {
        if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            return ctx_v;
        } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            return ctx_a;
        }
        GGML_ABORT("unknown chunk type");
    }

    projector_type proj_type_v() const {
        return ctx_v ? clip_get_projector_type(ctx_v) : PROJECTOR_TYPE_UNKNOWN;
    }

    projector_type proj_type_a() const {
        return ctx_a ? clip_get_projector_type(ctx_a) : PROJECTOR_TYPE_UNKNOWN;
    }

    int64_t n_embd_out() const {
        if (ctx_v) {
            return clip_n_mmproj_embd(ctx_v);
        } else if (ctx_a) {
            return clip_n_mmproj_embd(ctx_a);
        } else {
            throw std::runtime_error("no CLIP model loaded");
        }
    }

    ~mtmd_context() {
        clip_free(ctx_a);
        clip_free(ctx_v);
    }

private:
    llama_token lookup_token(const std::string & token_text) {
        if (vocab == nullptr) {
            // TODO @ngxson : this case is currently hit by mtmd_get_memory_usage
            // but we should reconsider this if this case is needed in other places in the future
            return LLAMA_TOKEN_NULL;
        }
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < n_vocab; i++) {
            if (token_to_piece(vocab, i, true) == token_text) {
                return i;
            }
        }
        return LLAMA_TOKEN_NULL;
    }

    std::string token_to_piece(const llama_vocab * vocab, llama_token token, bool special) {
        if (vocab == nullptr) {
            throw std::runtime_error("llama_vocab is not provided");
        }
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
    delete ctx;
}

struct mtmd_tokenizer {
    mtmd_context * ctx;

    std::string input_text;
    bool add_special;
    bool parse_special;
    const llama_vocab * vocab;

    struct part {
        std::string text;
        const mtmd_bitmap * bitmap;
    };
    std::vector<part> parts;
    // these will be freed when mtmd_tokenizer finishes
    std::vector<mtmd::bitmap> bm_from_lazy; // TODO @ngxson : refactor, free bm_from_lazy progressively
    std::vector<const char *> text_from_lazy;

    mtmd_input_chunks cur;
    uint32_t n_images_added = 0; // 0-based index assigned to the next image chunk

    ~mtmd_tokenizer() {
        // note: mtmd::bitmap is already RAII
        for (auto & str : text_from_lazy) {
            free((void *)str);
        }
    }

    mtmd_tokenizer(mtmd_context * ctx,
            const mtmd_input_text * text,
            const mtmd_bitmap ** bmps,
            size_t n_bitmaps) : ctx(ctx) {
        add_special   = text->add_special;
        parse_special = text->parse_special;
        input_text    = text->text;
        vocab         = ctx->vocab;

        std::vector<const mtmd_bitmap *> bitmaps(bmps, bmps + n_bitmaps);
        auto parts_str = split_text(input_text, ctx->media_marker);
        size_t i_bm = 0;
        for (const auto & part : parts_str) {
            if (part == ctx->media_marker) {
                if (i_bm >= bitmaps.size()) {
                    throw std::runtime_error(string_format("number of media markers in text (%zu) exceeds number of bitmaps (%zu)", i_bm + 1, bitmaps.size()));
                }
                parts.push_back({"", bitmaps[i_bm++]});
            } else {
                parts.push_back({std::move(part), nullptr});
            }
        }

        size_t n_markers = 0;
        for (const auto & part : parts) {
            if (part.bitmap != nullptr) {
                n_markers++;
            }
        }
        if (n_markers != bitmaps.size()) {
            throw std::runtime_error(string_format("number of media markers in text (%zu) does not match number of bitmaps (%zu)", n_markers, bitmaps.size()));
        }

        expand_lazy_bitmaps();
    }

    void expand_lazy_bitmaps() {
        std::vector<part> expanded;
        expanded.reserve(parts.size());
        for (auto & p : parts) {
            if (p.bitmap != nullptr && p.bitmap->lazy_callback) {
                LOG_DBG("%s: expanding lazy bitmap\n", __func__);
                for (size_t i = 0;; i++) {
                    char * out_str = nullptr;
                    mtmd_bitmap * out_bm = nullptr;
                    int res = p.bitmap->lazy_callback(i,
                                    p.bitmap->lazy_user_data,
                                    &out_bm,
                                    &out_str);
                    if (out_bm && out_str) {
                        throw std::runtime_error(string_format("lazy callback cannot return both bitmap and text"));
                    }
                    if (res == 0) {
                        // OK, append the returned chunk; lazy part is not yet added
                        if (out_bm) {
                            auto & ptr = bm_from_lazy.emplace_back(out_bm); // remember to free it later
                            expanded.push_back({"", ptr.ptr.get()});
                            LOG_DBG("%s: lazy callback returned bitmap with dimensions %d x %d\n", __func__, out_bm->nx, out_bm->ny);
                        } else if (out_str) {
                            auto & ptr = text_from_lazy.emplace_back(out_str); // remember to free it later
                            expanded.push_back({ptr, nullptr});
                            LOG_DBG("%s: lazy callback returned text: %s\n", __func__, out_str);
                        }
                    } else if (res == -1) {
                        // EOF: lazy part removes itself (not added to expanded)
                        break;
                    } else if (res == -2) {
                        // error
                        throw std::runtime_error(string_format("lazy callback returned error"));
                    }
                }
            } else {
                expanded.push_back(std::move(p));
            }
        }
        parts = std::move(expanded);
    }

    int32_t tokenize(mtmd_input_chunks * output) {
        cur.entries.clear();

        // [QWEN_VIDEO] handle frame merging for models that support it (i.e. qwen-vl)
        int n_merge_frames = 1;
        if (ctx->ctx_v) {
            n_merge_frames = clip_model_n_temporal_merge(ctx->ctx_v);
            GGML_ASSERT(n_merge_frames <= 2 && "we only support merging maximum 2 images for now; open an issue if this model supports merging more");
        }

        // Build merged_bitmaps: each entry is a group of 1 or 2 bitmaps.
        // For consecutive mergeable bitmap parts, merge them and collapse the second part out of this->parts.
        std::vector<std::vector<const mtmd_bitmap *>> merged_bitmaps;
        if (n_merge_frames > 1) {
            for (size_t i = 0; i < parts.size(); ++i) {
                if (parts[i].bitmap == nullptr) {
                    continue;
                }
                if (i + 1 < parts.size() && parts[i + 1].bitmap != nullptr) {
                    const mtmd_bitmap * bm_a = parts[i].bitmap;
                    const mtmd_bitmap * bm_b = parts[i + 1].bitmap;
                    if (bm_a->can_merge_with(*bm_b)) {
                        LOG_DBG("%s: merging 2 frames at part index %zu and %zu\n", __func__, i, i + 1);
                        merged_bitmaps.push_back({bm_a, bm_b});
                        parts.erase(parts.begin() + i + 1); // collapse the second bitmap part
                        continue;
                    }
                }
                LOG_DBG("%s: no merging for part index %zu\n", __func__, i);
                merged_bitmaps.push_back({parts[i].bitmap});
            }
        } else {
            for (const auto & p : parts) {
                if (p.bitmap != nullptr) {
                    merged_bitmaps.push_back({p.bitmap});
                }
            }
        }

        size_t i_bm = 0;
        for (const auto & p : parts) {
            if (p.bitmap != nullptr) {
                if (i_bm >= merged_bitmaps.size()) {
                    LOG_ERR("%s: error: number of bitmaps (%zu) does not match number of markers (%zu)\n",
                            __func__, merged_bitmaps.size(), parts.size() - 1);
                    return 1;
                }
                auto bmps = merged_bitmaps[i_bm++];
                int32_t res = add_media(bmps);
                if (res != 0) {
                    return res;
                }
            } else {
                add_text(p.text, parse_special);
            }
        }

        if (vocab != nullptr) {
            if (add_special && llama_vocab_get_add_bos(vocab)) {
                // if first chunk is text, we add BOS token to first text chunk
                // otherwise, create a new text chunk with BOS token
                if (!cur.entries.empty() && cur.entries[0].type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    // add BOS token to the beginning of first text chunk
                    cur.entries[0].tokens_text.insert(cur.entries[0].tokens_text.begin(), llama_vocab_bos(vocab));
                } else {
                    // create a new text chunk with BOS token at the beginning
                    mtmd_input_chunk bos_chunk{
                        MTMD_INPUT_CHUNK_TYPE_TEXT,
                        {llama_vocab_bos(vocab)},
                        nullptr, // image tokens
                        nullptr, // audio tokens
                    };
                    cur.entries.insert(cur.entries.begin(), std::move(bos_chunk));
                }
            }

            if (add_special && llama_vocab_get_add_eos(vocab)) {
                // if last chunk is text, we add EOS token to it
                add_text({llama_vocab_eos(vocab)});
            }
        }

        if (i_bm != merged_bitmaps.size()) {
            LOG_ERR("%s: error: number of bitmaps (%zu) does not match number of markers (%zu)\n",
                    __func__, merged_bitmaps.size(), parts.size() - 1);
            return 1;
        }

        *output = std::move(cur);

        return 0;
    }

    void add_text(const std::string & txt, bool parse_special) {
        if (vocab == nullptr) {
            throw std::runtime_error("llama_vocab is not provided");
        }
        LOG_DBG("%s: %s\n", __func__, txt.c_str());
        auto tokens = mtmd_tokenize_text_internal(vocab, txt, /* add_special */ false, parse_special);
        add_text(tokens);
    }

    void add_text(const std::vector<llama_token> & tokens) {
        if (tokens.empty()) {
            return;
        }
        // if last entry is also a text chunk, add tokens to it instead of creating new chunk
        if (!cur.entries.empty() && cur.entries.back().type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            cur.entries.back().tokens_text.insert(
                                            cur.entries.back().tokens_text.end(),
                                            tokens.begin(),
                                            tokens.end());
        } else {
            mtmd_input_chunk chunk{
                MTMD_INPUT_CHUNK_TYPE_TEXT,
                tokens,
                nullptr, // image tokens
                nullptr, // audio tokens
            };
            cur.entries.emplace_back(std::move(chunk));
        }
    }

    int32_t add_media(std::vector<const mtmd_bitmap *> & bitmaps) {
        GGML_ASSERT(!bitmaps.empty());

        // note: only one type of media is supported per call, caller should enforce this
        const bool is_vision = !bitmaps[0]->is_audio;

        if (is_vision) {
            // handle image

            if (!ctx->ctx_v) {
                LOG_ERR("%s: error: model does not support vision input\n", __func__);
                return 2;
            }

            if (!ctx->img_beg.empty()) {
                add_text(ctx->img_beg, true); // add image begin token
            }

            // TODO @ngxson : this is quite hacky because preprocessor only support batch with one single element, that need to be fixed in the future (e.g. by changing the preprocessor interface always take single input)

            mtmd_image_preproc_out preproc_out;

            for (const auto * bmp : bitmaps) {
                // sanity check
                GGML_ASSERT(!bmp->is_audio);
                GGML_ASSERT(ctx->image_preproc != nullptr);
                if (bmp->nx <= 0 || bmp->ny <= 0) {
                    LOG_ERR("%s: error: invalid bitmap dimensions: nx = %d, ny = %d\n",
                            __func__, bmp->nx, bmp->ny);
                    return 2;
                }

                // convert mtmd_bitmap to clip_image_u8
                clip_image_u8 img_u8;
                img_u8.set_size(
                    {(int)bmp->nx, (int)bmp->ny},
                    bmp->is_placeholder());
                img_u8.cpy_buf(bmp->get_ro_buf());

                // preprocess image
                mtmd_image_preproc_out tmp_preproc_out = ctx->image_preproc->preprocess(img_u8);

                // move entries and grid dimensions to the "global" preproc_out
                for (auto & entry : tmp_preproc_out.entries) {
                    preproc_out.entries.emplace_back(std::move(entry));
                }

                // for llava-uhd style, we need to handle grid too
                // we don't care about overwriting these values for now because the case where bitmaps.size() > 1 is only for frame merging (qwen-vl), not supported by llava-uhd
                if ((tmp_preproc_out.grid_x > 0 && tmp_preproc_out.grid_y > 0)
                        || tmp_preproc_out.has_overview()) {
                    GGML_ASSERT(bitmaps.size() == 1);
                    preproc_out.grid_x = tmp_preproc_out.grid_x;
                    preproc_out.grid_y = tmp_preproc_out.grid_y;
                    preproc_out.overview = std::move(tmp_preproc_out.overview);
                }
            }

            LOG_DBG("%s: preproc_out has %zu entries, grid_x = %d, grid_y = %d, has_overview = %d\n",
                    __func__, preproc_out.entries.size(), preproc_out.grid_x, preproc_out.grid_y,
                    preproc_out.has_overview() ? 1 : 0);

            // handle llava-uhd style preprocessing
            // (output either a grid, or overview-only)
            const bool has_tiling_grid = (preproc_out.grid_x > 0 && preproc_out.grid_y > 0)
                || preproc_out.has_overview();

            if (has_tiling_grid) {
                // [QWEN_VIDEO] we do not support "frame merging" for llama-uhd style, so no batching for now
                GGML_ASSERT(bitmaps.size() == 1);

                const int n_col = preproc_out.grid_x;
                const int n_row = preproc_out.grid_y;

                // split batch into chunks of single images
                auto chunks = split_batch_to_chunk(std::move(preproc_out), bitmaps[0]->id);
                GGML_ASSERT(chunks.size() > 0);

                // NOTE: preproc_out is invalidated after this point, do not use it anymore

                // split_batch_to_chunk must always put the overview image first
                auto ov_chunk = std::move(chunks.front());
                chunks.erase(chunks.begin());

                // add overview image (first)
                if (ctx->ov_img_first) {
                    add_text(ctx->tok_ov_img_start);
                    cur.entries.emplace_back(std::move(ov_chunk));
                    add_text(ctx->tok_ov_img_end);
                }

                // add slices (or tiles)
                if (!chunks.empty()) {
                    GGML_ASSERT((int)chunks.size() == n_row * n_col);
                    add_text(ctx->tok_slices_start);
                    for (int y = 0; y < n_row; y++) {
                        for (int x = 0; x < n_col; x++) {
                            const bool is_last_in_row = (x == n_col - 1);
                            if (!ctx->tok_sli_img_start.empty()) {
                                add_text(ctx->tok_sli_img_start);
                            } else if (!ctx->sli_img_start_tmpl.empty()) {
                                // If using a template to preceed a slice image
                                const size_t sz = std::snprintf(nullptr, 0, ctx->sli_img_start_tmpl.c_str(), y+1, x+1) + 1;
                                std::unique_ptr<char[]> buf(new char[sz]);
                                std::snprintf(buf.get(), sz, ctx->sli_img_start_tmpl.c_str(), y+1, x+1);
                                add_text(std::string(buf.get(), buf.get() + sz - 1), true);
                            }

                            auto & curr_chunk = chunks[y * n_col + x];
                            auto & curr_batch = curr_chunk.tokens_image->batch_f32;
                            if (curr_batch.entries.size() != 1) {
                                throw std::runtime_error(string_format("%s: expect 1 image in batch_f32", __func__));
                            }

                            LOG_DBG("%s: adding slice image at row %d col %d\n", __func__, y, x);
                            cur.entries.emplace_back(std::move(curr_chunk));

                            add_text(ctx->tok_sli_img_end);
                            if (!is_last_in_row) {
                                add_text(ctx->tok_sli_img_mid);
                            }
                        }
                        if ((y != n_row - 1 || ctx->tok_row_end_trail)) {
                            add_text(ctx->tok_row_end);
                        }
                    }
                    add_text(ctx->tok_slices_end);
                }

                // add overview image (last)
                if (!ctx->ov_img_first) {
                    add_text(ctx->tok_ov_img_start);
                    cur.entries.emplace_back(std::move(ov_chunk));
                    add_text(ctx->tok_ov_img_end);
                }

            } else {

                if (preproc_out.entries.size() == 0) {
                    LOG_ERR("%s: no image tokens produced by preprocessor (ref: https://github.com/ggml-org/llama.cpp/pull/24769)\n", __func__);
                    return 2;
                }

                size_t n_tokens = 0;
                for (auto & e : preproc_out.entries) {
                    n_tokens += clip_n_output_tokens(ctx->ctx_v, &e);
                    if (clip_model_n_temporal_merge(ctx->ctx_v) == 2) {
                        // [QWEN_VIDEO] pair input is merged to the same embd, so only count as one image
                        break;
                    }
                }

                mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);

                // [QWEN_VIDEO] improve this in the future
                image_tokens->n_temporal_merge = clip_model_n_temporal_merge(ctx->ctx_v);

                if (mtmd_decode_use_mrope(ctx)) {
                    // for Qwen2VL, we need this information for M-RoPE decoding positions
                    image_tokens->nx = clip_n_output_tokens_x(ctx->ctx_v, &preproc_out.entries[0]);
                    image_tokens->ny = clip_n_output_tokens_y(ctx->ctx_v, &preproc_out.entries[0]);
                } else {
                    // other models, we only need the total number of tokens
                    image_tokens->nx = n_tokens;
                    image_tokens->ny = 1;
                }
                image_tokens->pos = ctx->pos_type;
                // HunyuanVL wraps the image grid with BOI/EOI and adds one newline per row,
                // and uses XD-RoPE (dim-3 = image index). Override the position type so that
                // n_tokens() and mtmd_image_tokens_get_decoder_pos pick the HunyuanVL layout.
                if (ctx->proj_type_v() == PROJECTOR_TYPE_HUNYUANVL) {
                    image_tokens->pos       = MTMD_POS_TYPE_HUNYUANVL;
                    image_tokens->image_idx = n_images_added;
                    GGML_ASSERT(n_tokens == (size_t)image_tokens->n_tokens());
                }

                clip_image_f32_batch batch_f32;
                batch_f32.is_audio = false;
                batch_f32.entries = std::move(preproc_out.entries);
                // do NOT use preproc_out from this point on, it's moved

                image_tokens->batch_f32 = std::move(batch_f32);
                image_tokens->id = bitmaps[0]->id; // optional

                LOG_DBG("image_tokens->nx = %d\n", image_tokens->nx);
                LOG_DBG("image_tokens->ny = %d\n", image_tokens->ny);
                LOG_DBG("batch_f32 size = %d\n", (int)image_tokens->batch_f32.entries.size());

                mtmd_input_chunk chunk{
                    MTMD_INPUT_CHUNK_TYPE_IMAGE,
                    {}, // text tokens
                    std::move(image_tokens),
                    nullptr, // audio tokens
                };
                cur.entries.emplace_back(std::move(chunk));
            }

            if (!ctx->img_end.empty()) {
                add_text(ctx->img_end, true); // add image end token
            }

            // advance image-chunk counter so the next image gets the next XD-RoPE dim-3 slot
            n_images_added++;

        } else {
            // handle audio

            GGML_ASSERT(bitmaps.size() == 1); // no batching support for now
            auto & bitmap = bitmaps[0];

            if (!ctx->ctx_a) {
                LOG_ERR("%s: error: model does not support audio input\n", __func__);
                return 2;
            }

            if (bitmap->nx == 0) {
                LOG_ERR("%s: error: empty audio data\n", __func__);
                return 2;
            }

            if (!ctx->aud_beg.empty()) {
                add_text(ctx->aud_beg, true); // add audio begin token
            }

            // sanity check
            GGML_ASSERT(ctx->audio_preproc != nullptr);

            // preprocess audio
            std::vector<mtmd_audio_mel> mel_spec_chunks;
            {
                std::vector<float> dummy;
                const float * samples = nullptr;
                size_t n_samples = 0;
                if (bitmap->is_placeholder()) {
                    // TODO @ngxson : skip underlay processing if bitmap is placeholder
                    GGML_ASSERT(bitmap->ny == 1);

                    dummy.resize(bitmap->nx);
                    samples = dummy.data();
                    n_samples = dummy.size();
                } else {
                    const auto & buf = bitmap->get_ro_buf();
                    GGML_ASSERT(buf.size() > sizeof(float));
                    GGML_ASSERT(buf.size() % sizeof(float) == 0);

                    samples = (const float *)buf.data();
                    n_samples = buf.size() / sizeof(float);
                }
                bool ok = ctx->audio_preproc->preprocess(samples, n_samples, mel_spec_chunks);
                if (!ok) {
                    LOG_ERR("Unable to preprocess audio\n");
                    return 2;
                }
            }

            // consider each mel_spec as a separate audio chunk
            // TODO: maybe support batching, but this may come with memory cost
            for (auto & mel_spec : mel_spec_chunks) {
                const bool is_placeholder = mel_spec.data.empty();

                clip_image_f32 mel_f32;
                mel_f32.set_size(
                    {mel_spec.n_len, mel_spec.n_mel},
                    is_placeholder, /* is_audio */ true);
                mel_f32.cpy_buf(mel_spec.data);

                size_t n_tokens = clip_n_output_tokens(ctx->ctx_a, &mel_f32);

                clip_image_f32_batch batch_f32;
                batch_f32.is_audio = true;
                batch_f32.entries.push_back(std::move(mel_f32));

                mtmd_audio_tokens_ptr audio_tokens(new mtmd_audio_tokens);
                audio_tokens->n_tokens = n_tokens;
                audio_tokens->batch_f32 = std::move(batch_f32);
                audio_tokens->id = bitmap->id; // optional

                LOG_DBG("audio_tokens->n_tokens = %d\n", audio_tokens->n_tokens);

                mtmd_input_chunk chunk{
                    MTMD_INPUT_CHUNK_TYPE_AUDIO,
                    {}, // text tokens
                    nullptr, // image tokens
                    std::move(audio_tokens),
                };
                cur.entries.emplace_back(std::move(chunk));
            }

            if (!ctx->aud_end.empty()) {
                add_text(ctx->aud_end, true); // add audio end token
            }
        }

        return 0;
    }

    std::vector<mtmd_input_chunk> split_batch_to_chunk(mtmd_image_preproc_out && preproc_out, const std::string & id) {
        std::vector<mtmd_input_chunk> chunks;

        auto process_chunk = [&](clip_image_f32 && img) {
            mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
            image_tokens->nx = clip_n_output_tokens(ctx->ctx_v, &img);
            image_tokens->ny = 1;
            image_tokens->batch_f32.entries.push_back(std::move(img));
            image_tokens->id = id;

            GGML_ASSERT(image_tokens->nx > 0);

            mtmd_input_chunk chunk{
                MTMD_INPUT_CHUNK_TYPE_IMAGE,
                {}, // text tokens
                std::move(image_tokens),
                nullptr, // audio tokens
            };
            chunks.emplace_back(std::move(chunk));
        };

        // overview image first
        auto & overview = preproc_out.overview;
        if (overview.nx() == 0 || overview.ny() == 0) {
            throw std::runtime_error(string_format("%s: invalid overview image for llava-uhd style preprocessing\n", __func__));
        }
        process_chunk(std::move(preproc_out.overview));

        // then, process slices
        for (auto & entry : preproc_out.entries) {
            if (entry.nx() == 0 || entry.ny() == 0) {
                throw std::runtime_error(string_format("%s: invalid image slice for llava-uhd style preprocessing\n", __func__));
            }
            process_chunk(std::move(entry));
        }

        return chunks;
    }

    // for example: "a <__media__> b <__media__> c" --> "a", "<__media__>", "b", "<__media__>", "c"
    static std::vector<std::string> split_text(const std::string & input, const std::string & delimiter) {
        std::vector<std::string> result;
        if (input.empty()) {
            return result;
        }
        size_t start = 0;
        size_t pos = 0;
        while ((pos = input.find(delimiter, start)) != std::string::npos) {
            if (pos > start) {
                result.push_back(input.substr(start, pos - start));
            }
            result.push_back(delimiter);
            start = pos + delimiter.length();
        }
        if (start < input.length()) {
            result.push_back(input.substr(start));
        }
        return result;
    }

    // copied from common_tokenize
    static std::vector<llama_token> mtmd_tokenize_text_internal(
        const struct llama_vocab * vocab,
               const std::string & text,
                            bool   add_special,
                            bool   parse_special) {
        if (vocab == nullptr) {
            throw std::runtime_error("llama_vocab is not provided");
        }
        // upper limit for the number of tokens
        int n_tokens = text.length() + 2 * add_special;
        std::vector<llama_token> result(n_tokens);
        n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        if (n_tokens == std::numeric_limits<int32_t>::min()) {
            throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
        }
        if (n_tokens < 0) {
            result.resize(-n_tokens);
            int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
            GGML_ASSERT(check == -n_tokens);
        } else {
            result.resize(n_tokens);
        }
        return result;
    }
};

int32_t mtmd_tokenize(mtmd_context * ctx,
            mtmd_input_chunks * output,
            const mtmd_input_text * text,
            const mtmd_bitmap ** bitmaps,
            size_t n_bitmaps) {
    try {
        mtmd_tokenizer tokenizer(ctx, text, bitmaps, n_bitmaps);
        return tokenizer.tokenize(output);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 2;
    }
}

static int32_t mtmd_encode_impl(mtmd_context * ctx, const mtmd_image_tokens * image_tokens, std::vector<float> & out_embd) {
    clip_ctx * ctx_clip = ctx->ctx_v;
    if (!ctx_clip) {
        LOG_ERR("%s: this API does not support non-vision input, please use mtmd_encode_chunk instead\n", __func__);
        return 1;
    }

    int n_embd_out = ctx->n_embd_out();
    auto n_tokens_out = image_tokens->n_tokens();
    out_embd.resize((size_t)n_embd_out * n_tokens_out);

    if (image_tokens->is_placeholder()) {
        LOG_ERR("%s: image tokens batch is placeholder\n", __func__);
        return 1;
    }

    bool ok = clip_image_batch_encode(
        ctx_clip,
        ctx->n_threads,
        &image_tokens->batch_f32,
        out_embd);

    return ok ? 0 : 1;
}

static int32_t mtmd_encode_chunk_impl(mtmd_context * ctx, const mtmd_input_chunk * chunk, std::vector<float> & out_embd) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        LOG_WRN("mtmd_encode_chunk has no effect for text chunks\n");
        return 0;
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        if (!ctx->ctx_v) {
            LOG_ERR("%s: model does not support vision input\n", __func__);
            return 1;
        }
        if (chunk->tokens_image == nullptr) {
            LOG_ERR("%s: image tokens are null\n", __func__);
            return 1;
        }
        if (chunk->tokens_image->is_placeholder()) {
            LOG_ERR("%s: image tokens batch is placeholder\n", __func__);
            return 1;
        }
        return mtmd_encode_impl(ctx, chunk->tokens_image.get(), out_embd);
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        if (!ctx->ctx_a) {
            LOG_ERR("%s: model does not support audio input\n", __func__);
            return 1;
        }
        if (chunk->tokens_audio == nullptr) {
            LOG_ERR("%s: audio tokens are null\n", __func__);
            return 1;
        }
        if (chunk->tokens_audio->is_placeholder()) {
            LOG_ERR("%s: audio tokens batch is placeholder\n", __func__);
            return 1;
        }
        int n_mmproj_embd = ctx->n_embd_out();
        out_embd.resize((size_t)chunk->tokens_audio->n_tokens * n_mmproj_embd);
        bool ok = clip_image_batch_encode(
            ctx->ctx_a,
            ctx->n_threads,
            &chunk->tokens_audio->batch_f32,
            out_embd);
        return ok ? 0 : 1;
    }

    LOG_ERR("%s: unknown chunk type %d\n", __func__, (int)chunk->type);
    return 1;
}

int32_t mtmd_encode_chunk(mtmd_context * ctx, const mtmd_input_chunk * chunk) {
    // this is the non-batching version
    try {
        return mtmd_encode_chunk_impl(ctx, chunk, ctx->out_embd);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

int32_t mtmd_encode(mtmd_context * ctx, const mtmd_image_tokens * image_tokens) {
    try {
        return mtmd_encode_impl(ctx, image_tokens, ctx->out_embd);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

float * mtmd_get_output_embd(mtmd_context * ctx) {
    return ctx->out_embd.data();
}

mtmd_batch * mtmd_batch_init(mtmd_context * ctx) {
    return new mtmd_batch(ctx);
}

void mtmd_batch_free(mtmd_batch * batch) {
    if (batch) {
        delete batch;
    }
}

int32_t mtmd_batch_add_chunk(mtmd_batch * batch, const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        LOG_ERR("%s: text chunk is not supported in batch\n", __func__);
        return 1;
    }

    auto * ctx = batch->ctx->get_clip_ctx(chunk);
    if (!ctx) {
        LOG_ERR("%s: model does not support input chunk type %d\n", __func__, (int)chunk->type);
        return 1;
    }

    if (batch->entries.empty()) {
        // batch must have at least one chunk
        batch->entries.push_back(chunk);
        return 0;
    }

    if (!clip_support_batch(ctx)) {
        // if no batching support, batch can only have one single chunk
        return 2; // "batch too large" error code
    }

    int32_t new_n_tokens = batch->n_tokens() + (int32_t)mtmd_input_chunk_get_n_tokens(chunk);
    if (new_n_tokens > batch->ctx->batch_max_tokens) {
        return 2; // "batch too large" error code
    }

    auto & first_chunk = batch->entries[0];
    if (first_chunk->can_batch_with(*chunk)) {
        batch->entries.push_back(chunk);
        return 0;
    }

    return 3; // "cannot batch" error code
}

static int32_t mtmd_batch_encode_impl(mtmd_batch * batch) {
    if (batch->entries.empty()) {
        LOG_ERR("%s: batch is empty\n", __func__);
        return 1;
    }
    for (const auto * chunk : batch->entries) {
        if (chunk->is_placeholder()) {
            LOG_ERR("%s: chunk is placeholder\n", __func__);
            return 1;
        }
    }

    // represent the whole batch as one single chunk
    mtmd::input_chunk_ptr batch_chunk(mtmd_input_chunk_copy(batch->entries[0]));
    if (batch_chunk->tokens_image) {
        auto & b0_f32 = batch_chunk->tokens_image->batch_f32;
        // copy all entries from other chunks into the first chunk's batch_f32
        // note: skip first entry because it's already in batch_chunk
        for (size_t ic = 1; ic < batch->entries.size(); ic++) {
            auto & chunk = batch->entries[ic];
            GGML_ASSERT(chunk->tokens_image);
            auto b1_f32 = chunk->tokens_image->batch_f32.clone();
            for (size_t i = 0; i < b1_f32.entries.size(); i++) {
                b0_f32.entries.push_back(std::move(b1_f32.entries[i]));
            }
        }
    } else if (batch_chunk->tokens_audio) {
        auto & b0_f32 = batch_chunk->tokens_audio->batch_f32;
        // copy all entries from other chunks into the first chunk's batch_f32
        // note: skip first entry because it's already in batch_chunk
        for (size_t ic = 1; ic < batch->entries.size(); ic++) {
            auto & chunk = batch->entries[ic];
            GGML_ASSERT(chunk->tokens_audio);
            auto b1_f32 = chunk->tokens_audio->batch_f32.clone();
            for (size_t i = 0; i < b1_f32.entries.size(); i++) {
                b0_f32.entries.push_back(std::move(b1_f32.entries[i]));
            }
        }
    } else {
        LOG_ERR("%s: unsupported chunk type\n", __func__);
        return 1;
    }

    LOG_DBG("%s: encoding batch with %zu entries and total %zu tokens\n",
            __func__, batch->entries.size(), mtmd_input_chunk_get_n_tokens(batch_chunk.get()));
    int32_t res = mtmd_encode_chunk_impl(
        batch->ctx,
        batch_chunk.get(),
        batch->output_embd);
    return res;
}

int32_t mtmd_batch_encode(mtmd_batch * batch) {
    try {
        return mtmd_batch_encode_impl(batch);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

float * mtmd_batch_get_output_embd(mtmd_batch * batch, const mtmd_input_chunk * chunk) {
    if (batch->output_embd.empty()) {
        LOG_ERR("%s: batch has not been encoded yet\n", __func__);
        return nullptr;
    }
    size_t offset = 0;
    const size_t n_embd = batch->ctx->n_embd_out();
    for (const auto * c : batch->entries) {
        size_t offset_prev = offset;
        size_t n_tokens = mtmd_input_chunk_get_n_tokens(c);
        offset += n_tokens * n_embd;
        GGML_ASSERT(offset_prev <  batch->output_embd.size());
        GGML_ASSERT(offset      <= batch->output_embd.size());
        if (c == chunk) {
            return &batch->output_embd.data()[offset_prev];
        }
    }
    return nullptr; // not found
}

bool mtmd_decode_use_non_causal(const mtmd_context * ctx, const mtmd_input_chunk * chunk) {
    auto proj_type = ctx->proj_type_v();
    if (chunk && chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        proj_type = ctx->proj_type_a();
    }
    switch (proj_type) {
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_GEMMA4V:
        case PROJECTOR_TYPE_GEMMA4UV:
            return true;
        default:
            return false;
    }
}

bool mtmd_decode_use_mrope(const mtmd_context * ctx) {
    return ctx->pos_type == MTMD_POS_TYPE_MROPE;
}

bool mtmd_support_vision(const mtmd_context * ctx) {
    return ctx->ctx_v != nullptr;
}

bool mtmd_support_audio(const mtmd_context * ctx) {
    return ctx->ctx_a != nullptr;
}

int mtmd_get_audio_sample_rate(const mtmd_context * ctx) {
    if (!ctx->ctx_a) {
        return -1;
    }
    return clip_get_hparams(ctx->ctx_a)->audio_sample_rate;
}

const char * mtmd_get_marker(const mtmd_context * ctx) {
    return ctx->media_marker.c_str();
}

//
// public API functions
//

// mtmd_bitmap

mtmd_bitmap * mtmd_bitmap_init(uint32_t nx,
                               uint32_t ny,
                               const unsigned char * data) {
    mtmd_bitmap * bitmap = new mtmd_bitmap(data, nx, ny);
    return bitmap;
}

mtmd_bitmap * mtmd_bitmap_init_from_audio(size_t n_samples,
                                          const float * data) {
    mtmd_bitmap * bitmap = new mtmd_bitmap((const unsigned char *)data, n_samples);
    GGML_ASSERT(bitmap->is_audio);
    if (!bitmap->is_placeholder()) {
        GGML_ASSERT(bitmap->get_ro_buf().size() == n_samples * sizeof(float));
    }
    return bitmap;
}

uint32_t mtmd_bitmap_get_nx(const mtmd_bitmap * bitmap) {
    return bitmap->nx;
}

uint32_t mtmd_bitmap_get_ny(const mtmd_bitmap * bitmap) {
    return bitmap->ny;
}

const unsigned char * mtmd_bitmap_get_data(const mtmd_bitmap * bitmap) {
    if (bitmap->is_placeholder()) {
        return nullptr;
    }
    return bitmap->get_ro_buf().data();
}

size_t mtmd_bitmap_get_n_bytes(const mtmd_bitmap * bitmap) {
    if (bitmap->is_placeholder()) {
        return 0;
    }
    return bitmap->get_ro_buf().size();
}

bool mtmd_bitmap_is_audio(const mtmd_bitmap * bitmap) {
    return bitmap->is_audio;
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

mtmd_bitmap * mtmd_bitmap_init_lazy(mtmd_context * ctx,
                                    const char * id,
                                    void * user_data,
                                    mtmd_bitmap_lazy_callback callback) {
    GGML_UNUSED(ctx); // reserved for future use
    mtmd_bitmap * bitmap = new mtmd_bitmap(nullptr, 0, 0);
    bitmap->lazy_callback = callback;
    bitmap->lazy_user_data = user_data;
    mtmd_bitmap_set_id(bitmap, id);
    return bitmap;
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

size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        return chunk->tokens_text.size();
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return mtmd_image_tokens_get_n_tokens(chunk->tokens_image.get());
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        return chunk->tokens_audio->n_tokens;
    } else {
        GGML_ABORT("invalid chunk type");
    }
}

llama_pos mtmd_input_chunk_get_n_pos(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        return chunk->tokens_text.size();
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return mtmd_image_tokens_get_n_pos(chunk->tokens_image.get());
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        return chunk->tokens_audio->n_tokens;
    } else {
        GGML_ABORT("invalid chunk type");
    }
}

const char * mtmd_input_chunk_get_id(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return chunk->tokens_image->id.c_str();
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        return chunk->tokens_audio->id.c_str();
    }
    return nullptr;
}

mtmd_input_chunk * mtmd_input_chunk_copy(const mtmd_input_chunk * chunk) {
    mtmd_input_chunk * copy = new mtmd_input_chunk{
        chunk->type,
        chunk->tokens_text,
        nullptr,
        nullptr,
    };
    if (chunk->tokens_image) {
        // copy the image tokens
        copy->tokens_image = mtmd_image_tokens_ptr(new mtmd_image_tokens());
        *copy->tokens_image = chunk->tokens_image->clone();
    }
    if (chunk->tokens_audio) {
        // copy the audio tokens
        copy->tokens_audio = mtmd_audio_tokens_ptr(new mtmd_audio_tokens());
        *copy->tokens_audio = chunk->tokens_audio->clone();
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

mtmd_decoder_pos mtmd_image_tokens_get_decoder_pos(const mtmd_image_tokens * image_tokens, llama_pos pos_0, size_t i) {
    mtmd_decoder_pos pos;
    switch (image_tokens->pos) {
        case MTMD_POS_TYPE_MROPE:
            {
                pos.t = pos_0;
                pos.x = pos_0 + (i % image_tokens->nx);
                pos.y = pos_0 + (i / image_tokens->nx);
                pos.z = 0; // unused for now
            } break;
        case MTMD_POS_TYPE_NORMAL:
            {
                pos.t = pos_0 + i;
                pos.x = pos_0 + i;
                pos.y = pos_0 + i;
                pos.z = pos_0 + i;
            } break;
        case MTMD_POS_TYPE_HUNYUANVL:
            {
                // HunyuanVL layout: [BOI] [row0 tokens + newline] ... [row(ny-1) tokens + newline] [EOI]
                // Total = 1 + ny*(nx+1) + 1. BOI and EOI use sequential positions in every dim;
                // content and row-newline tokens use (row, col) with XD-RoPE dim-3 = image_idx.
                const uint32_t nx      = image_tokens->nx;
                const uint32_t n_total = image_tokens->n_tokens();
                if (i == 0) {
                    // BOI
                    pos.t = pos_0 + i;
                    pos.x = pos_0 + i;
                    pos.y = pos_0 + i;
                    pos.z = pos_0 + i;
                } else if (i == n_total - 1) {
                    // EOI
                    pos.t = pos_0 + i;
                    pos.x = pos_0 + i;
                    pos.y = pos_0 + i;
                    pos.z = pos_0 + i;
                } else {
                    // content token at (row, col), or the trailing newline of a row (col == nx)
                    //   section 0 = sequential, section 1 = w(col), section 2 = h(row), section 3 = image_count.
                    // set_position_mrope_2d writes .y -> section 1 and .x -> section 2
                    const uint32_t offset = (uint32_t)i - 1;
                    const uint32_t row    = offset / (nx + 1);
                    const uint32_t col    = offset % (nx + 1);
                    pos.t = pos_0 + i;
                    pos.x = row;
                    pos.y = col;
                    pos.z = image_tokens->image_idx;
                }
            } break;
        default:
            GGML_ABORT("invalid position type");
    }
    return pos;
}

const char * mtmd_image_tokens_get_id(const mtmd_image_tokens * image_tokens) {
    return image_tokens->id.c_str();
}

llama_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens * image_tokens) {
    switch (image_tokens->pos) {
        case MTMD_POS_TYPE_MROPE:
            return std::max(image_tokens->nx, image_tokens->ny);
        case MTMD_POS_TYPE_NORMAL:
            return image_tokens->n_tokens();
        case MTMD_POS_TYPE_HUNYUANVL:
            // HunyuanVL: the sequential (dim-0) position advances by the full token count
            // (includes BOI/EOI and row newline tokens), not by max(nx, ny)
            return image_tokens->n_tokens();
        default:
            GGML_ABORT("invalid position type");
    }
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
        nullptr, // image tokens
        nullptr, // audio tokens
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
        {}, // text tokens
        std::move(image_tokens),
        nullptr, // audio tokens
    };
    chunks->entries.emplace_back(std::move(chunk_image));

    return chunks;
}

void mtmd_log_set(ggml_log_callback log_callback, void * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : clip_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

struct mtmd_caps mtmd_get_cap_from_file(const char * fname) {
    try {
        auto tmp = clip_get_cap(fname);
        mtmd_caps cap;
        cap.inp_audio  = tmp.has_audio;
        cap.inp_vision = tmp.has_vision;
        return cap;
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to get capabilities from file '%s': %s\n", __func__, fname, e.what());
        return mtmd_caps{ false, false };
    }
}

//
// Debugging API (NOT intended for public use)
//

static void mtmd_debug_encode_impl(mtmd_context * ctx, clip_ctx * ctx_clip, clip_image_f32 & image) {
    clip_set_debug_output_embeddings(ctx_clip, true);
    int n_mmproj_embd = clip_n_mmproj_embd(ctx_clip);
    int n_tokens = clip_n_output_tokens(ctx_clip, &image);
    std::vector<float> embd_output(n_tokens * n_mmproj_embd, 0.0f);
    bool ok = clip_image_encode(
        ctx_clip,
        ctx->n_threads,
        &image,
        embd_output);
    if (!ok) {
        LOG_ERR("%s: failed to encode image\n", __func__);
    }
}

void mtmd_debug_encode_image(mtmd_context * ctx, const std::vector<std::vector<float>> & image) {
    if (!ctx->ctx_v) {
        LOG_ERR("%s: model does not support vision input\n", __func__);
        return;
    }
    const int img_sz = (int)image.size();
    std::vector<float> img_buf;
    img_buf.reserve(img_sz * img_sz);
    for (const auto & row : image) {
        img_buf.insert(img_buf.end(), row.begin(), row.end());
    }
    clip_image_f32 inp_image;
    inp_image.set_size({img_sz, img_sz}, false, false);
    inp_image.cpy_buf(img_buf);
    LOG_INF("%s: created input image with nx=%d, ny=%d\n", __func__, img_sz, img_sz);
    mtmd_debug_encode_impl(ctx, ctx->ctx_v, inp_image);
}

void mtmd_debug_encode_audio(mtmd_context * ctx, const std::vector<float> & input) {
    if (!ctx->ctx_a) {
        LOG_ERR("%s: model does not support audio input\n", __func__);
        return;
    }
    int n_mel = clip_get_hparams(ctx->ctx_a)->n_mel_bins;
    const int audio_nx = (int)input.size();
    std::vector<float> audio_buf(audio_nx * n_mel);
    for (int i = 0; i < audio_nx; i++) {
        for (int j = 0; j < n_mel; j++) {
            audio_buf[j * audio_nx + i] = input[i];
        }
    }
    clip_image_f32 inp_audio;
    inp_audio.set_size({audio_nx, n_mel}, false, true);
    inp_audio.cpy_buf(audio_buf);
    LOG_INF("%s: created input audio with nx=%d, ny=%d\n", __func__, audio_nx, n_mel);
    mtmd_debug_encode_impl(ctx, ctx->ctx_a, inp_audio);
}

void mtmd_debug_preprocess_image(mtmd_context * ctx, const std::vector<uint8_t> & rgb_values, int nx, int ny) {
    if (!ctx->ctx_v) {
        LOG_ERR("%s: model does not support vision input\n", __func__);
        return;
    }
    clip_image_u8 img_u8;
    img_u8.set_size({nx, ny}, false);
    img_u8.cpy_buf(rgb_values);
    GGML_ASSERT(ctx->image_preproc != nullptr);
    mtmd_image_preproc_out preproc_out = ctx->image_preproc->preprocess(img_u8);

    clip_image_f32_batch batch_f32;
    batch_f32.is_audio = false;
    for (auto & entry : preproc_out.entries) {
        batch_f32.entries.push_back(std::move(entry));
    }

    LOG_INF("%s: preprocessed image to batch_f32 with %d entries\n", __func__, (int)batch_f32.entries.size());
    for (size_t i = 0; i < batch_f32.entries.size(); i++) {
        LOG_INF("%s: entry %zu has nx=%d, ny=%d\n", __func__, i, batch_f32.entries[i].nx(), batch_f32.entries[i].ny());
        // TODO: better way to dump entry content?
    }
}

void mtmd_debug_preprocess_audio(mtmd_context * ctx, const std::vector<float> & samples) {
    if (!ctx->ctx_a) {
        LOG_ERR("%s: model does not support audio input\n", __func__);
        return;
    }
    std::vector<mtmd_audio_mel> mel_spec_chunks;
    bool ok = ctx->audio_preproc->preprocess(samples.data(), samples.size(), mel_spec_chunks);
    if (!ok) {
        LOG_ERR("%s: failed to preprocess audio\n", __func__);
        return;
    }
    LOG_INF("%s: preprocessed audio to %zu mel spec chunks\n", __func__, mel_spec_chunks.size());
    for (size_t i = 0; i < mel_spec_chunks.size(); i++) {
        LOG_INF("%s: mel spec chunk %zu has n_len=%d, n_mel=%d\n", __func__, i, mel_spec_chunks[i].n_len, mel_spec_chunks[i].n_mel);

        // dump mel entries: data is stored as [n_mel][n_len] (mel-major)
        const auto & mel = mel_spec_chunks[i];
        for (int m = 0; m < mel.n_mel; m++) {
            for (int t = 0; t < mel.n_len; t++) {
                LOG_INF("mel[%zu][m=%d][t=%d] = %f\n", i, m, t, mel.data[m * mel.n_len + t]);
            }
        }
    }
}

static void stub_log_callback(enum ggml_log_level, const char *, void *) {
    // do nothing
}

std::map<ggml_backend_dev_t, size_t> mtmd_get_memory_usage(const char * mmproj_fname,
                                                            struct mtmd_context_params ctx_params) {
    mtmd::context_ptr ctx;
    auto saved_log_callback = g_logger_state.log_callback;
    auto saved_log_user_data = g_logger_state.log_callback_user_data;
    try {
        mtmd_log_set(stub_log_callback, nullptr); // suppress logging
        ctx.reset(new mtmd_context(mmproj_fname, nullptr, ctx_params));
        mtmd_log_set(saved_log_callback, saved_log_user_data); // restore log callback
        std::map<ggml_backend_dev_t, size_t> total_mem;
        auto merge = [&](const struct clip_ctx * c) {
            for (auto & [dev, size] : clip_get_mem_usage(c)) {
                total_mem[dev] += size;
            }
        };
        if (ctx->ctx_v) {
            merge(ctx->ctx_v);
        }
        if (ctx->ctx_a) {
            merge(ctx->ctx_a);
        }
        return total_mem;
    } catch (const std::exception & e) {
        mtmd_log_set(saved_log_callback, saved_log_user_data); // restore log callback
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return {};
    }
}
