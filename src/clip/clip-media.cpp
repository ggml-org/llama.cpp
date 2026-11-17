#include "clip-media.h"
#include "models/models.h"

#include "llama.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <limits>
#include <stdexcept>

std::string clip_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

// only mtmd.cpp needs these, so they're implemented here rather than in mtmd-impl.h
void clip_image_f32::serialize(clip_serialization & ser) const {
    // remember to bump CLIP_SERIALIZATION_VERSION if this is changed
    // note: buf is intentionally NOT serialized; the loaded clip_image_f32 will always be a placeholder
    ser.write(add_viewsep);
    ser.write(add_newline);
    ser.write(lead_pad);
    ser.write((int32_t)suffix_type);
    ser.write((int32_t)anyres.grid_x);
    ser.write((int32_t)anyres.grid_y);
    ser.write((int32_t)anyres.orig_nx);
    ser.write((int32_t)anyres.orig_ny);
    ser.write((int32_t)nx_);
    ser.write((int32_t)ny_);
    ser.write(audio_n_ctx);
}
void clip_image_f32::deserialize(clip_serialization & ser) {
    add_viewsep = ser.read<bool>();
    add_newline = ser.read<bool>();
    lead_pad = ser.read<int32_t>();
    if (ser.version >= 3) {
        const int32_t suffix_raw = ser.read<int32_t>();
        if (suffix_raw < 0 || suffix_raw >= CLIP_SUFFIX_COUNT) {
            throw std::runtime_error("invalid suffix type");
        }
        suffix_type = (clip_suffix_type)suffix_raw;
        anyres.grid_x = ser.read<int32_t>();
        anyres.grid_y = ser.read<int32_t>();
        anyres.orig_nx = ser.read<int32_t>();
        anyres.orig_ny = ser.read<int32_t>();
    }
    nx_ = ser.read<int32_t>();
    ny_ = ser.read<int32_t>();
    if (ser.version >= 5) {
        audio_n_ctx = ser.read<int32_t>();
        if (audio_n_ctx < 0 || audio_n_ctx > nx_) {
            throw std::runtime_error("invalid audio_n_ctx");
        }
    }
    buf.clear(); // always a placeholder after loading
}

void clip_image_f32_batch::serialize(clip_serialization & ser) const {
    // remember to bump CLIP_SERIALIZATION_VERSION if this is changed
    ser.write(is_audio);
    ser.write<uint64_t>(entries.size());
    for (const auto & entry : entries) {
        entry.serialize(ser);
    }
}
void clip_image_f32_batch::deserialize(clip_serialization & ser) {
    is_audio = ser.read<bool>();
    uint64_t n = ser.read<uint64_t>();
    constexpr size_t min_entry_bytes = sizeof(uint8_t) * 2 + sizeof(int32_t) * 2;
    if (n > (ser.data.size() - ser.read_pos) / min_entry_bytes) {
        throw std::runtime_error("entries count exceeds buffer size");
    }
    entries.clear();
    entries.reserve(n);
    for (uint64_t i = 0; i < n; i++) {
        clip_image_f32 entry;
        entry.deserialize(ser);
        entries.push_back(std::move(entry));
    }
}

bool clip_cbx_is_mtl(const clip_model & model) {
    return model.cbx_read_tensor("a.cenc.perceiver.pre_attention_query", nullptr, 0) > 0;
}

clip_layout_config::clip_layout_config(const clip_model * v, const clip_model * a, const llama_model * model_dec,
                                       ggml_backend_sched_t sched_mel, int n_threads)
        : v(v), a(a), n_threads(n_threads), sched_mel(sched_mel) {
    vocab = model_dec ? llama_model_get_vocab(model_dec) : nullptr;
    if (model_dec) {
        switch (llama_model_rope_type(model_dec)) {
            case LLAMA_ROPE_TYPE_NONE:
            case LLAMA_ROPE_TYPE_NORM:
            case LLAMA_ROPE_TYPE_NEOX:
                pos_type = CLIP_POS_TYPE_NORMAL;
                break;
            case LLAMA_ROPE_TYPE_MROPE:
            case LLAMA_ROPE_TYPE_IMROPE:
                pos_type = CLIP_POS_TYPE_MROPE;
                break;
            default:
                throw std::runtime_error(clip_format("unsupported decoder rope type: %d\n", llama_model_rope_type(model_dec)));
        }
    }
    if (v) {
        init_vision();
    }
    if (a) {
        init_audio();
    }
}

bool clip_layout_config::decode_use_non_causal(bool is_audio) const {
    auto proj_type = is_audio ? proj_type_a() : proj_type_v();
    switch (proj_type) {
        case PROJECTOR_TYPE_GEMMA4V:
            {
                // E2B (n_embd = 1536) and E4B (n_embd = 2560) always use causal
                // note: use mmproj n_embd, because text model may not be provided (e.g. clip_get_memory_usage)
                const int n_embd = v->n_mmproj_embd();
                return n_embd != 1536 && n_embd != 2560;
            }
        case PROJECTOR_TYPE_GEMMA4UV:
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_SENSENOVA_U1:
        case PROJECTOR_TYPE_DEEPSEEK4V:
            return true;
        default:
            return false;
    }
}

void clip_layout_config::init_vision() {
        GGML_ASSERT(v != nullptr);
        image_preproc.reset();

        projector_type proj = v->proj_type;

        switch (proj) {
            case PROJECTOR_TYPE_MLP:
            case PROJECTOR_TYPE_MLP_NORM:
            case PROJECTOR_TYPE_LDP:
            case PROJECTOR_TYPE_LDPV2:
            case PROJECTOR_TYPE_COGVLM:
            case PROJECTOR_TYPE_JANUS_PRO:
            case PROJECTOR_TYPE_GLM_EDGE:
                {
                    bool has_pinpoints = !v->hparams.image_res_candidates.empty();
                    if (has_pinpoints) {
                        image_preproc = std::make_unique<clip_image_preprocessor_llava_uhd>(v);
                    } else {
                        image_preproc = std::make_unique<clip_image_preprocessor_fixed_size>(v);
                    }
                } break;
            case PROJECTOR_TYPE_MINICPMV:
                {
                    int minicpmv_version = v->hparams.minicpmv_version;
                    if (minicpmv_version == 2) {
                        // minicpmv 2.5 format:
                        // <image> (overview) </image><slice><image> (slice) </image><image> (slice) </image>\n ... </slice>
                        slice_tmpl        = CLIP_SLICE_TMPL_MINICPMV_2_5;
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
                        slice_tmpl        = CLIP_SLICE_TMPL_MINICPMV_2_6;
                        tok_ov_img_start  = {lookup_token("<image>")};
                        tok_ov_img_end    = {lookup_token("</image>")};
                        tok_sli_img_start = {lookup_token("<slice>")};
                        tok_sli_img_end   = {lookup_token("</slice>")};
                        tok_row_end       = {lookup_token("\n")};
                        tok_row_end_trail = false; // no trailing end-of-row token
                        ov_img_first      = true;

                    } else if (minicpmv_version != 0) {
                        throw std::runtime_error(clip_format("unsupported minicpmv version: %d\n", minicpmv_version));
                    }
                    image_preproc = std::make_unique<clip_image_preprocessor_llava_uhd>(v);
                } break;
            case PROJECTOR_TYPE_MINICPMV4_6:
            case PROJECTOR_TYPE_MINICPMV4_7:
            case PROJECTOR_TYPE_MINICPM_OCR_1_0:
                {
                    slice_tmpl        = CLIP_SLICE_TMPL_MINICPMV_2_6;
                    tok_ov_img_start  = {lookup_token("<image>")};
                    tok_ov_img_end    = {lookup_token("</image>")};
                    tok_sli_img_start = {lookup_token("<slice>")};
                    tok_sli_img_end   = {lookup_token("</slice>")};
                    tok_row_end       = {lookup_token("\n")};
                    tok_row_end_trail = false; // no trailing end-of-row token
                    ov_img_first      = true;
                    has_prompt_img_id = true;
                    if (v->proj_type == PROJECTOR_TYPE_MINICPM_OCR_1_0) {
                        image_preproc = std::make_unique<clip_image_preprocessor_minicpm_ocr_1_0>(v);
                    } else {
                        image_preproc = std::make_unique<clip_image_preprocessor_minicpmv>(v);
                    }
                } break;
            case PROJECTOR_TYPE_QWEN3VL:
                {
                    // <|vision_start|> ... (image embeddings) ... <|vision_end|>
                    img_beg = "<|vision_start|>";
                    img_end = "<|vision_end|>";
                    image_preproc = std::make_unique<clip_image_preprocessor_qwen3vl>(v);
                } break;
            case PROJECTOR_TYPE_QWEN2VL:
            case PROJECTOR_TYPE_QWEN25VL:
            case PROJECTOR_TYPE_LING3VL:
            case PROJECTOR_TYPE_MIMOVL:
            case PROJECTOR_TYPE_MAGEVL:
                {
                    // <|vision_start|> ... (image embeddings) ... <|vision_end|>
                    img_beg = "<|vision_start|>";
                    img_end = "<|vision_end|>";
                    image_preproc = std::make_unique<clip_image_preprocessor_dyn_size>(v);
                } break;
            case PROJECTOR_TYPE_MINIMAX_M3:
                {
                    // ]<]start of image[>[ ... (image embeddings) ... ]<]end of image[>[
                    img_beg = "]<]start of image[>[";
                    img_end = "]<]end of image[>[";
                    image_preproc = std::make_unique<clip_image_preprocessor_dyn_size>(v);
                } break;
            case PROJECTOR_TYPE_SENSENOVA_U1:
                {
                    img_beg = "<img>";
                    img_end = "</img>";
                    image_preproc = std::make_unique<clip_image_preprocessor_dyn_size>(v);
                } break;
            case PROJECTOR_TYPE_INKLING2V:
                {
                    // the chat template code gives the image its own message block
                    img_beg = "<|content_image|>";
                    image_preproc = std::make_unique<clip_image_preprocessor_inkling2v>(v);
                } break;
            case PROJECTOR_TYPE_MUSE_GLIMMER:
                {
                    img_beg = "<|image_start|>";
                    img_end = "<|image_end|>";
                    image_preproc = std::make_unique<clip_image_preprocessor_muse_glimmer>(v);
                } break;
            case PROJECTOR_TYPE_LFM3VL:
                {
                    // <|vision_start|> ... (image embeddings) ... <|vision_end|>
                    img_beg = "<|vision_start|>";
                    img_end = "<|vision_end|>";
                    vid_beg = img_beg;
                    vid_end = img_end;
                    image_preproc = std::make_unique<clip_image_preprocessor_lfm3vl>(v);
                } break;
            case PROJECTOR_TYPE_YOUTUVL:
                {
                    // <|vision_start|> ... (image embeddings) ... <|vision_end|>
                    img_beg = "<|vision_start|>";
                    img_end = "<|vision_end|>";
                    image_preproc = std::make_unique<clip_image_preprocessor_youtuvl>(v);
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
                    image_preproc = std::make_unique<clip_image_preprocessor_fixed_size>(v);
                } break;
            case PROJECTOR_TYPE_GEMMA3:
            case PROJECTOR_TYPE_GEMMA3NV:
                {
                    // <start_of_image> ... (image embeddings) ... <end_of_image>
                    img_beg = "<start_of_image>";
                    img_end = "<end_of_image>";
                    image_preproc = std::make_unique<clip_image_preprocessor_fixed_size>(v);
                } break;
            case PROJECTOR_TYPE_IDEFICS3:
                {
                    // https://github.com/huggingface/transformers/blob/a42ba80fa520c784c8f11a973ca9034e5f859b79/src/transformers/models/idefics3/processing_idefics3.py#L192-L215
                    slice_tmpl         = CLIP_SLICE_TMPL_IDEFICS3;
                    tok_ov_img_start   = {lookup_token("\n\n"), lookup_token("<fake_token_around_image>"), lookup_token("<global-img>")};
                    tok_ov_img_end     = {lookup_token("<fake_token_around_image>")};
                    tok_row_end        = {lookup_token("\n")};
                    sli_img_start_tmpl = "<fake_token_around_image><row_%d_col_%d>";
                    image_preproc = std::make_unique<clip_image_preprocessor_idefics3>(v);
                } break;
            case PROJECTOR_TYPE_PIXTRAL:
                {
                    // https://github.com/huggingface/transformers/blob/1cd110c6cb6a6237614130c470e9a902dbc1a4bd/docs/source/en/model_doc/pixtral.md
                    img_end = "[IMG_END]";
                    if (v->hparams.image_longest_edge > 0) {
                        image_preproc = std::make_unique<clip_image_preprocessor_longest_edge>(v);
                    } else {
                        image_preproc = std::make_unique<clip_image_preprocessor_dyn_size>(v);
                    }
                } break;
            case PROJECTOR_TYPE_PHI4:
                {
                    // Phi-4 uses media marker insertion only. Keep image boundary text empty.
                    image_preproc = std::make_unique<clip_image_preprocessor_dyn_size>(v);
                } break;
            case PROJECTOR_TYPE_LLAMA4:
                {
                    // (more details in clip_context constructor)
                    img_beg = "<|image_start|>";
                    img_end = "<|image_end|>";
                    LOG_WRN("%s: llama 4 vision is known to have degraded quality:\n"
                            "    https://github.com/ggml-org/llama.cpp/pull/13282\n", __func__);
                    image_preproc = std::make_unique<clip_image_preprocessor_llava_uhd>(v);
                    ov_img_first = false;
                } break;
            case PROJECTOR_TYPE_STEP3VL:
                {
                    // Step3 format:
                    //   <patch_start> (patch) <patch_end> [<patch_newline>]
                    //   ... (all patch rows)
                    //   <im_start> (overview) <im_end>
                    slice_tmpl        = CLIP_SLICE_TMPL_STEP3VL;
                    tok_ov_img_start  = {lookup_token("<im_start>")};
                    tok_ov_img_end    = {lookup_token("<im_end>")};
                    tok_sli_img_start = {lookup_token("<patch_start>")};
                    tok_sli_img_end   = {lookup_token("<patch_end>")};
                    tok_row_end       = {lookup_token("<patch_newline>")};
                    tok_row_end_trail = false;
                    ov_img_first      = false; // patches first, overview last
                    image_preproc = std::make_unique<clip_image_preprocessor_step3vl>(v);
                } break;
            case PROJECTOR_TYPE_INTERNVL:
                {
                    // <img> ... (image embeddings) ... </img>
                    img_beg = "<img>";
                    img_end = "</img>";
                    image_preproc = std::make_unique<clip_image_preprocessor_internvl>(v);
                    ov_img_first = false;
                } break;
            case PROJECTOR_TYPE_COHERE2V:
                {
                    // <|START_OF_IMG|> (tile embeddings) <|IMG_LINE_BREAK|> ... <|END_OF_IMG|>
                    img_beg = "<|START_OF_IMG|>";
                    img_end = "<|END_OF_IMG|>";
                    tok_sli_img_end = {lookup_token("<|IMG_LINE_BREAK|>")};
                    tok_ov_img_end  = tok_sli_img_end;
                    ov_img_first = false;
                    image_preproc = std::make_unique<clip_image_preprocessor_cohere2v>(v);
                } break;
            case PROJECTOR_TYPE_KIMIVL:
                {
                    // <|media_start|> ... (image embeddings) ... <|media_end|>
                    img_beg = "<|media_start|>";
                    img_end = "<|media_end|>";
                    image_preproc = std::make_unique<clip_image_preprocessor_dyn_size>(v);
                } break;
            case PROJECTOR_TYPE_KIMIK25:
                {
                    // GLM-5.2-V reuses the Kimi-K2.5 vision encoder and projector, but marks
                    // images with its own tokens, so decide based on the text model vocab
                    if (lookup_token("<|begin_of_image|>") != LLAMA_TOKEN_NULL) {
                        // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>
                        img_beg = "<|begin_of_image|>";
                        img_end = "<|end_of_image|>";
                        image_preproc = std::make_unique<clip_image_preprocessor_dyn_size>(v);
                    } else {
                        // <|media_begin|> ... (image embeddings) ... <|media_end|>
                        img_beg = "<|media_begin|>";
                        img_end = "<|media_end|>";
                        image_preproc = std::make_unique<clip_image_preprocessor_kimik25>(v);
                    }
                } break;
            case PROJECTOR_TYPE_KIMIK31:
                {
                    // <|media_begin|>image WxH<|media_content|> ... (image embeddings) ... <|media_end|>
                    img_beg = "<|media_begin|>";
                    img_end = "<|media_end|>";
                    has_prompt_img_size = true;
                    image_preproc = std::make_unique<clip_image_preprocessor_kimik25>(v);
                } break;
            case PROJECTOR_TYPE_LIGHTONOCR:
                {
                    // <|im_start|> ... (image embeddings) ... <|im_end|>
                    img_beg = "<|im_start|>";
                    img_end = "<|im_end|>";
                    image_preproc = std::make_unique<clip_image_preprocessor_longest_edge>(v);
                } break;
            case PROJECTOR_TYPE_DEEPSEEK4V:
                {
                    // no vocab tokens are added; the start/end/newline markers are learned embeddings emitted by the encoder
                    image_preproc = std::make_unique<clip_image_preprocessor_deepseek4v>(v);
                } break;
            case PROJECTOR_TYPE_DEEPSEEK41V:
                {
                    // as deepseek4v: markers come from the encoder, the block is decoded causally
                    image_preproc = std::make_unique<clip_image_preprocessor_deepseek41v>(v);
                } break;
            case PROJECTOR_TYPE_DOTS_OCR:
            case PROJECTOR_TYPE_DOTS3NOTE_V:
                {
                    // <|img|> ... (image embeddings) ... <|endofimg|>
                    img_beg = "<|img|>";
                    img_end = "<|endofimg|>";
                    image_preproc = std::make_unique<clip_image_preprocessor_dyn_size>(v);
                } break;
            case PROJECTOR_TYPE_NEMOTRON_V2_VL:
                {
                    image_preproc = std::make_unique<clip_image_preprocessor_fixed_size>(v);
                } break;
            case PROJECTOR_TYPE_NEMOTRON_OMNI:
                {
                    img_beg = "<img>";
                    img_end = "</img>";
                    image_preproc = std::make_unique<clip_image_preprocessor_dyn_size>(v);
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
                    slice_tmpl         = CLIP_SLICE_TMPL_LFM2;
                    sli_img_start_tmpl = "<|img_row_%d_col_%d|>";
                    tok_ov_img_start   = {lookup_token("<|img_thumbnail|>")};
                    ov_img_first       = false;
                    image_preproc = std::make_unique<clip_image_preprocessor_lfm2>(v);
                } break;
            case PROJECTOR_TYPE_D1OMNI_V:
                {
                    // same tiles and thumbnail as lfm2, without separator tokens
                    image_preproc = std::make_unique<clip_image_preprocessor_lfm2>(v);
                } break;
            case PROJECTOR_TYPE_GLM4V:
                {
                    // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>
                    img_beg = "<|begin_of_image|>";
                    img_end = "<|end_of_image|>";
                    image_preproc = std::make_unique<clip_image_preprocessor_dyn_size>(v);
                } break;
            case PROJECTOR_TYPE_GLM5V:
                {
                    // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>
                    img_beg = "<|begin_of_image|>";
                    img_end = "<|end_of_image|>";
                    image_preproc = std::make_unique<clip_image_preprocessor_glm5v>(v);
                } break;
            case PROJECTOR_TYPE_PADDLEOCR:
                {
                    // <|IMAGE_START|> ... (image embeddings) ... <|IMAGE_END|>
                    img_beg = "<|IMAGE_START|>";
                    img_end = "<|IMAGE_END|>";
                    image_preproc = std::make_unique<clip_image_preprocessor_dyn_size>(v);
                } break;
            case PROJECTOR_TYPE_GEMMA4V:
            case PROJECTOR_TYPE_GEMMA4UV:
                {
                    // <|image> ... (image embeddings) ... <image|>
                    img_beg = "<|image>";
                    img_end = "<image|>";
                    image_preproc = std::make_unique<clip_image_preprocessor_dyn_size>(v);
                } break;
            case PROJECTOR_TYPE_DEEPSEEKOCR:
            case PROJECTOR_TYPE_DEEPSEEKOCR2:
                {
                    img_end = "\n"; // prevent empty batch on llama-server
                    image_preproc = std::make_unique<clip_image_preprocessor_deepseekocr>(v);
                    ov_img_first = false;
                } break;
            case PROJECTOR_TYPE_HUNYUANVL:
                {
                    // note: these use fullwidth ｜ (U+FF5C) and ▁ (U+2581) to match the tokenizer vocabulary
                    img_beg = "<｜hy_place▁holder▁no▁100｜>";
                    img_end = "<｜hy_place▁holder▁no▁101｜>";
                    image_preproc = std::make_unique<clip_image_preprocessor_dyn_size>(v);
                } break;
            case PROJECTOR_TYPE_EXAONE4_5:
                {
                    // <vision> ... (image embeddings) ... </vision>
                    img_beg = "<vision>";
                    img_end = "</vision>";
                    image_preproc = std::make_unique<clip_image_preprocessor_dyn_size>(v);
                } break;
            case PROJECTOR_TYPE_GRANITE4_VISION:
                {
                    // ... (image embeddings) \n ...
                    img_beg = "";
                    img_end = "\n";
                    image_preproc = std::make_unique<clip_image_preprocessor_granite>(v);
                } break;
            default:
                throw std::runtime_error(clip_format("%s: unexpected vision projector type %d\n", __func__, proj));
        }

        GGML_ASSERT(image_preproc != nullptr);
    }

void clip_layout_config::init_audio() {
        GGML_ASSERT(a != nullptr);
        audio_preproc.reset();

        projector_type proj = a->proj_type;

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
                    audio_preproc = std::make_unique<clip_audio_preprocessor_whisper>(a);
                } break;
            case PROJECTOR_TYPE_HIGGS_AUDIO:
                {
                    // the 4s chunks are joined in one <|audio_bos|> ... <|audio_eos|> span
                    aud_beg = "<|audio_bos|>";
                    aud_end = "<|audio_eos|>";
                    audio_preproc = std::make_unique<clip_audio_preprocessor_higgs_audio>(a);
                } break;
            case PROJECTOR_TYPE_VIBEVOICE_ASR:
                {
                    aud_beg = "<|object_ref_start|>";
                    aud_end = "<|object_ref_end|>";
                    audio_preproc = std::make_unique<clip_audio_preprocessor_vibevoice_asr>(a);
                } break;
            case PROJECTOR_TYPE_MUSE_SPARK_AUDIO:
                {
                    aud_beg = "<|soft_audio_start|>";
                    aud_end = "<|soft_audio_end|>";
                    audio_preproc = std::make_unique<clip_audio_preprocessor_muse_spark_audio>(a);
                } break;
            case PROJECTOR_TYPE_QWEN3A:
                {
                    aud_beg = "<|audio_start|>";
                    aud_end = "<|audio_end|>";
                    audio_preproc = std::make_unique<clip_audio_preprocessor_qwen3a>(a);
                } break;
            case PROJECTOR_TYPE_VOXTRAL:
                {
                    // [BEGIN_AUDIO] ... (embeddings) ...
                    aud_beg = "[BEGIN_AUDIO]";
                    audio_preproc = std::make_unique<clip_audio_preprocessor_whisper>(a);
                } break;
            case PROJECTOR_TYPE_MUSIC_FLAMINGO:
                {
                    // <sound> ... (embeddings) ...
                    aud_beg = "<sound>";
                    audio_preproc = std::make_unique<clip_audio_preprocessor_whisper>(a);
                } break;
            case PROJECTOR_TYPE_VOXTRAL_RT:
                {
                    // audio and text share positions, only usable via clip_audio_stream
                } break;
            case PROJECTOR_TYPE_ULTRAVOX:
            case PROJECTOR_TYPE_GLMA:
            case PROJECTOR_TYPE_MERALION:
                {
                    audio_preproc = std::make_unique<clip_audio_preprocessor_whisper>(a);
                } break;
            case PROJECTOR_TYPE_LFM2A:
                {
                    audio_preproc = std::make_unique<clip_audio_preprocessor_conformer>(a);
                } break;
            case PROJECTOR_TYPE_D1OMNI_A:
                {
                    audio_preproc = std::make_unique<clip_audio_preprocessor_d1omni>(a);
                } break;
            case PROJECTOR_TYPE_LFM3A:
                {
                    // LFM3-Omni wraps the audio in the image span, LFM3-Audio (no vision tower) places it bare
                    if (v) {
                        aud_beg = img_beg;
                        aud_end = img_end;
                    }
                    audio_preproc = std::make_unique<clip_audio_preprocessor_lfm3a>(a);
                } break;
            case PROJECTOR_TYPE_GRANITE_SPEECH:
                {
                    audio_preproc = std::make_unique<clip_audio_preprocessor_granite_speech>(a);
                } break;
            case PROJECTOR_TYPE_GEMMA4A:
                {
                    aud_beg = "<|audio>";
                    aud_end = "<audio|>";
                    audio_preproc = std::make_unique<clip_audio_preprocessor_gemma4a>(a);
                } break;
            case PROJECTOR_TYPE_PARAKEET:
                {
                    audio_preproc = std::make_unique<clip_audio_preprocessor_parakeet>(a);
                } break;
            case PROJECTOR_TYPE_NEMOTRON_TRANSCRIBE:
                {
                    audio_preproc = std::make_unique<clip_audio_preprocessor_nemotron_transcribe>(a);
                } break;
            case PROJECTOR_TYPE_GEMMA4UA:
                {
                    aud_beg = "<|audio>";
                    aud_end = "<audio|>";
                    audio_preproc = std::make_unique<clip_audio_preprocessor_gemma4ua>(a);
                } break;
            case PROJECTOR_TYPE_DOTS3NOTE_A:
                {
                    // <|audio_comp_start|> ... (embeddings) ... <|audio_comp_end|>
                    aud_beg = "<|audio_comp_start|>";
                    aud_end = "<|audio_comp_end|>";
                    audio_preproc = std::make_unique<clip_audio_preprocessor_dots3note>(a);
                } break;
            case PROJECTOR_TYPE_MIMO_AUDIO:
                {
                    aud_beg = "<|mimo_audio_start|>";
                    aud_end = "<|mimo_audio_end|>";
                    audio_preproc = std::make_unique<clip_audio_preprocessor_mimo_audio>(a);
                } break;
            case PROJECTOR_TYPE_QWEN3TTS_SPKENC:
                {
                    audio_preproc = std::make_unique<clip_audio_preprocessor_qwen3tts_spk>(a);
                } break;
            case PROJECTOR_TYPE_ZONOS3_SPKENC:
                {
                    audio_preproc = std::make_unique<clip_audio_preprocessor_zonos3_spk>(a);
                } break;
            case PROJECTOR_TYPE_POCKETTTS_SPKENC:
                {
                    audio_preproc = std::make_unique<clip_audio_preprocessor_pockettts>(a);
                } break;
            case PROJECTOR_TYPE_FISH_S2_SPKENC:
                {
                    audio_preproc = std::make_unique<clip_audio_preprocessor_fish_s2>(a);
                } break;
            case PROJECTOR_TYPE_CHATTERBOX_SPKENC:
                {
                    audio_preproc = std::make_unique<clip_audio_preprocessor_chatterbox_ref>(a, clip_cbx_is_mtl(*a));
                } break;
            default:
                throw std::runtime_error(clip_format("%s: unexpected audio projector type %d\n", __func__, proj));
        }

        // initialize audio preprocessor
        if (audio_preproc) {
            audio_preproc->mel_config.n_threads = n_threads;
            audio_preproc->mel_config.sched     = sched_mel;
            audio_preproc->initialize();
        }
    }

static uint32_t clip_canvas_scale(uint32_t coord, uint32_t grid, uint32_t canvas) {
    if (grid <= 1 || canvas <= 1) {
        return 0;
    }
    const double v = (double) coord * (double) (canvas - 1) / (double) (grid - 1);
    return std::min((uint32_t) std::nearbyint(v), canvas - 1);
}

// CLIP_POS_TYPE_CANVAS: every tile shares the <image> token before the chunk as origin
// the overview is stretched over the whole canvas, each slice fills its own cell; the time component is the origin, in slot z
// a tile takes one position in slot t (the KV cache position), the separators after it take one position each
static clip_decoder_pos clip_canvas_decoder_pos(const clip_image_tokens * image_tokens, llama_pos pos_0, size_t i) {
    const auto & entries = image_tokens->batch_f32.entries;
    const auto & grid    = entries[0].anyres;
    const uint32_t nx = image_tokens->nx;
    const uint32_t ny = image_tokens->ny;
    const uint32_t canvas_w = grid.is_tiled() ? grid.grid_x * nx : nx;
    const uint32_t canvas_h = grid.is_tiled() ? grid.grid_y * ny : ny;
    const uint32_t base = pos_0 - 1;

    clip_decoder_pos pos;
    uint32_t t = pos_0;
    for (size_t k = 0; k < entries.size(); ++k) {
        const auto [gw, gh] = image_tokens->canvas_tile_grid(k);
        if (i < gw * gh) {
            const uint32_t row = i / gw;
            const uint32_t col = i % gw;
            uint32_t h;
            uint32_t w;
            if (k == 0) {
                h = clip_canvas_scale(row, gh, canvas_h);
                w = clip_canvas_scale(col, gw, canvas_w);
            } else {
                const uint32_t s = k - 1;
                h = (s / grid.grid_x) * ny + row;
                w = (s % grid.grid_x) * nx + col;
            }
            pos.t = t;
            pos.x = base + w;
            pos.y = base + h;
            pos.z = base;
            return pos;
        }
        i -= gw * gh;

        const size_t n_sep = clip_suffix_rows(entries[k].suffix_type).size();
        if (i < n_sep) {
            const uint32_t p = t + 1 + i;
            pos.t = p;
            pos.x = p;
            pos.y = p;
            pos.z = p;
            return pos;
        }
        i -= n_sep;
        t += 1 + n_sep;
    }
    GGML_ABORT("token index out of range");
}

clip_decoder_pos clip_image_tokens_get_decoder_pos(const clip_image_tokens * image_tokens, llama_pos pos_0, size_t i) {
    clip_decoder_pos pos;
    switch (image_tokens->pos) {
        case CLIP_POS_TYPE_MROPE:
            {
                pos.t = pos_0;
                pos.x = pos_0 + (i % image_tokens->nx);
                pos.y = pos_0 + (i / image_tokens->nx);
                pos.z = 0; // unused for now
            } break;
        case CLIP_POS_TYPE_NORMAL:
            {
                pos.t = pos_0 + i;
                pos.x = pos_0 + i;
                pos.y = pos_0 + i;
                pos.z = pos_0 + i;
            } break;
        case CLIP_POS_TYPE_SENSENOVA_U1:
            {
                pos.t = pos_0;
                pos.x = i % image_tokens->nx;
                pos.y = i / image_tokens->nx;
                pos.z = 0;
            } break;
        case CLIP_POS_TYPE_HUNYUANVL:
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
        case CLIP_POS_TYPE_CANVAS:
            {
                pos = clip_canvas_decoder_pos(image_tokens, pos_0, i);
            } break;
        default:
            GGML_ABORT("invalid position type");
    }
    return pos;
}

llama_pos clip_image_tokens_get_n_pos(const clip_image_tokens * image_tokens) {
    switch (image_tokens->pos) {
        case CLIP_POS_TYPE_MROPE:
            return std::max(image_tokens->nx, image_tokens->ny);
        case CLIP_POS_TYPE_NORMAL:
            return image_tokens->n_tokens();
        case CLIP_POS_TYPE_SENSENOVA_U1:
            return 1;
        case CLIP_POS_TYPE_HUNYUANVL:
            // HunyuanVL: the sequential (dim-0) position advances by the full token count
            // (includes BOI/EOI and row newline tokens), not by max(nx, ny)
            return image_tokens->n_tokens();
        case CLIP_POS_TYPE_CANVAS:
            {
                // one position per tile, plus one per separator
                llama_pos n_pos = 0;
                for (const auto & entry : image_tokens->batch_f32.entries) {
                    n_pos += 1 + (llama_pos) clip_suffix_rows(entry.suffix_type).size();
                }
                return n_pos;
            }
        default:
            GGML_ABORT("invalid position type");
    }
}
