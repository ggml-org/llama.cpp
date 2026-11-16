#include "models.h"

#include "llama-impl.h"
#include "llama-media.h"
#include "llama-media-image.h"
#include "llama-vocab.h"

#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

// mmproj GGUFs keep the "clip" name for compatibility with existing files
// keys of the vision tower are "clip.vision.<llama key>", see LLM_KV prefix

static const char * KEY_HAS_VISION    = "clip.has_vision_encoder";
static const char * KEY_PROJ_TYPE     = "clip.projector_type";
static const char * KEY_V_PROJ_TYPE   = "clip.vision.projector_type";
static const char * KEY_USE_GELU      = "clip.use_gelu";
static const char * KEY_USE_SILU      = "clip.use_silu";
static const char * KEY_IMAGE_SIZE    = "clip.vision.image_size";
static const char * KEY_PATCH_SIZE    = "clip.vision.patch_size";
static const char * KEY_IMAGE_MEAN    = "clip.vision.image_mean";
static const char * KEY_IMAGE_STD     = "clip.vision.image_std";
static const char * KEY_HEAD_DIM      = "clip.vision.attention.head_dim";
static const char * KEY_PROJ_DIM      = "clip.vision.projection_dim";
static const char * KEY_SCALE_FACTOR  = "clip.vision.projector.scale_factor";
static const char * KEY_SPATIAL_MERGE = "clip.vision.spatial_merge_size";
static const char * KEY_MIN_PIXELS    = "clip.vision.image_min_pixels";
static const char * KEY_MAX_PIXELS    = "clip.vision.image_max_pixels";
static const char * KEY_WA_PATTERN    = "clip.vision.n_wa_pattern";
static const char * KEY_WINDOW_SIZE   = "clip.vision.window_size";
static const char * KEY_DEEPSTACK     = "clip.vision.is_deepstack_layers";

static llama_clip_projector clip_projector_from_string(const std::string & name) {
    if (name == "gemma3")           { return LLAMA_CLIP_PROJ_GEMMA3; }
    if (name == "qwen2vl_merger")   { return LLAMA_CLIP_PROJ_QWEN2VL; }
    if (name == "qwen2.5vl_merger") { return LLAMA_CLIP_PROJ_QWEN25VL; }
    if (name == "qwen3vl_merger")   { return LLAMA_CLIP_PROJ_QWEN3VL; }
    return LLAMA_CLIP_PROJ_UNKNOWN;
}

// mmproj files store sizes as u32 (most) or i32 (some converters), accept both
static bool clip_get_i32(llama_model_loader & ml, const char * key, int32_t & out, bool required = true) {
    const int kid = gguf_find_key(ml.metadata, key);
    if (kid < 0) {
        if (required) {
            throw std::runtime_error(format("clip: key not found: %s", key));
        }
        return false;
    }
    switch (gguf_get_kv_type(ml.metadata, kid)) {
        case GGUF_TYPE_UINT32: out = (int32_t) gguf_get_val_u32(ml.metadata, kid); return true;
        case GGUF_TYPE_INT32:  out = gguf_get_val_i32(ml.metadata, kid);           return true;
        default: throw std::runtime_error(format("clip: key %s is not an integer", key));
    }
}

void llama_model_clip::load_vocab(llama_model_loader &) {
    // an mmproj has no vocabulary
}

void llama_model_clip::load_arch_hparams(llama_model_loader & ml) {
    hparams.causal_attn = false;
    hparams.n_ctx_train = 8192; // unused, encoders keep no memory

    ml.get_key(std::string(KEY_HAS_VISION), vhp.has_vision, false);

    std::string proj;
    if (!ml.get_key(std::string(KEY_V_PROJ_TYPE), proj, false)) {
        ml.get_key(std::string(KEY_PROJ_TYPE), proj, false);
    }
    vhp.proj_name = proj;
    vhp.proj      = clip_projector_from_string(proj);

    if (!vhp.has_vision) {
        // audio-only and generation mmproj files are not ported yet; llama-quantize still opens them
        return;
    }

    const LLM_KV kv(LLM_ARCH_CLIP, nullptr, "clip.vision");

    uint32_t n_ff      = 0;
    uint32_t n_head    = 0;
    uint32_t n_head_kv = 0;

    ml.get_key(kv(LLM_KV_EMBEDDING_LENGTH),       hparams.n_embd);
    ml.get_key(kv(LLM_KV_BLOCK_COUNT),            hparams.n_layer_all);
    ml.get_key(kv(LLM_KV_FEED_FORWARD_LENGTH),    n_ff);
    ml.get_key(kv(LLM_KV_ATTENTION_HEAD_COUNT),   n_head);
    ml.get_key(kv(LLM_KV_ATTENTION_LAYERNORM_EPS), hparams.f_norm_eps);

    n_head_kv = n_head;
    ml.get_key(kv(LLM_KV_ATTENTION_HEAD_COUNT_KV), n_head_kv, false);

    GGML_ASSERT(hparams.n_layer_all > 0 && hparams.n_layer_all <= LLAMA_MAX_LAYERS);
    GGML_ASSERT(n_head > 0);

    hparams.f_norm_rms_eps = hparams.f_norm_eps;

    std::fill(hparams.n_head_arr.begin(),    hparams.n_head_arr.end(),    0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(),      hparams.n_ff_arr.end(),      0);
    std::fill(hparams.n_head_arr.begin(),    hparams.n_head_arr.begin()    + hparams.n_layer_all, n_head);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.begin() + hparams.n_layer_all, n_head_kv);
    std::fill(hparams.n_ff_arr.begin(),      hparams.n_ff_arr.begin()      + hparams.n_layer_all, n_ff);

    uint32_t head_dim = hparams.n_embd / n_head;
    ml.get_key(std::string(KEY_HEAD_DIM), head_dim, false);
    hparams.n_embd_head_k_full = head_dim;
    hparams.n_embd_head_v_full = head_dim;
    hparams.n_rot_full         = 0;

    clip_get_i32(ml, KEY_IMAGE_SIZE, vhp.image_size);
    clip_get_i32(ml, KEY_PATCH_SIZE, vhp.patch_size);
    clip_get_i32(ml, KEY_PROJ_DIM, vhp.n_embd_out, false);

    {
        std::vector<float> mean;
        std::vector<float> std;
        ml.get_arr(std::string(KEY_IMAGE_MEAN), mean, false);
        ml.get_arr(std::string(KEY_IMAGE_STD),  std,  false);
        if (mean.size() < 3 || std.size() < 3) {
            throw std::runtime_error("clip: image_mean/image_std must have 3 elements");
        }
        for (int i = 0; i < 3; ++i) {
            vhp.image_mean[i] = mean[i];
            vhp.image_std[i]  = std[i];
        }
    }

    {
        bool use_gelu = false;
        bool use_silu = false;
        ml.get_key(std::string(KEY_USE_GELU), use_gelu, false);
        ml.get_key(std::string(KEY_USE_SILU), use_silu, false);
        vhp.ffn_op = use_gelu ? LLM_FFN_GELU : use_silu ? LLM_FFN_SILU : LLM_FFN_GELU_QUICK;
    }

    switch (vhp.proj) {
        case LLAMA_CLIP_PROJ_GEMMA3:
            {
                vhp.n_merge = 4;
                clip_get_i32(ml, KEY_SCALE_FACTOR, vhp.n_merge, false);
            } break;
        case LLAMA_CLIP_PROJ_QWEN2VL:
        case LLAMA_CLIP_PROJ_QWEN25VL:
        case LLAMA_CLIP_PROJ_QWEN3VL:
            {
                vhp.n_merge = 2;
                clip_get_i32(ml, KEY_SPATIAL_MERGE, vhp.n_merge, false);
                if (vhp.n_merge != 2) {
                    throw std::runtime_error("clip: Qwen-VL needs spatial_merge_size = 2");
                }
                vhp.n_temporal_merge = 2;
                // defaults of the HF processor: 8 to 4096 tokens, the file can override them
                const int32_t patch_area = vhp.patch_size * vhp.patch_size * vhp.n_merge * vhp.n_merge;
                vhp.image_min_pixels = 8    * patch_area;
                vhp.image_max_pixels = 4096 * patch_area;
                clip_get_i32(ml, KEY_MIN_PIXELS, vhp.file_min_pixels, false);
                clip_get_i32(ml, KEY_MAX_PIXELS, vhp.file_max_pixels, false);
                if (vhp.proj == LLAMA_CLIP_PROJ_QWEN25VL) {
                    clip_get_i32(ml, KEY_WA_PATTERN, vhp.n_wa_pattern);
                    clip_get_i32(ml, KEY_WINDOW_SIZE, vhp.attn_window, false);
                }
                if (vhp.proj == LLAMA_CLIP_PROJ_QWEN3VL) {
                    // bool array, not covered by the loader getters
                    const int kid = gguf_find_key(ml.metadata, KEY_DEEPSTACK);
                    if (kid >= 0 && gguf_get_kv_type(ml.metadata, kid) == GGUF_TYPE_ARRAY &&
                            gguf_get_arr_type(ml.metadata, kid) == GGUF_TYPE_BOOL) {
                        const auto * ds = (const int8_t *) gguf_get_arr_data(ml.metadata, kid);
                        for (size_t i = 0; i < gguf_get_arr_n(ml.metadata, kid); ++i) {
                            if (ds[i]) {
                                vhp.deepstack_layers.push_back((int32_t) i);
                            }
                        }
                    }
                }
            } break;
        case LLAMA_CLIP_PROJ_UNKNOWN:
            break;
    }
}

void llama_model_clip::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    if (!vhp.has_vision) {
        throw std::runtime_error("clip: this mmproj has no vision tower, other towers are not supported by libllama yet");
    }
    if (vhp.proj == LLAMA_CLIP_PROJ_UNKNOWN) {
        throw std::runtime_error(format("clip: projector type '%s' is not supported by libllama yet", vhp.proj_name.c_str()));
    }

    // shape from the file: legacy layouts and projector shapes that do not follow from hparams
    auto create_as_file = [&](const LLM_TN_IMPL & tname, int flags) -> ggml_tensor * {
        const std::string name = tname.str();
        const ggml_tensor * meta = ml.get_tensor_meta(name.c_str());
        if (meta == nullptr) {
            if (flags & TENSOR_NOT_REQUIRED) {
                return nullptr;
            }
            throw std::runtime_error(format("clip: missing tensor '%s'", name.c_str()));
        }
        switch (ggml_n_dims(meta)) {
            case 1:  return create_tensor(tname, {meta->ne[0]}, flags);
            case 2:  return create_tensor(tname, {meta->ne[0], meta->ne[1]}, flags);
            case 3:  return create_tensor(tname, {meta->ne[0], meta->ne[1], meta->ne[2]}, flags);
            default: return create_tensor(tname, {meta->ne[0], meta->ne[1], meta->ne[2], meta->ne[3]}, flags);
        }
    };

    const int64_t n_embd_q = n_embd_head_k * n_head;

    // stem
    v_patch_embd   = create_as_file(tn(LLM_TENSOR_V_ENC_EMBD_PATCH, "weight"),   0);
    v_patch_embd_1 = create_as_file(tn(LLM_TENSOR_V_ENC_EMBD_PATCH, "weight.1"), TENSOR_NOT_REQUIRED);
    v_patch_bias   = create_tensor (tn(LLM_TENSOR_V_ENC_EMBD_PATCH, "bias"), {n_embd}, TENSOR_NOT_REQUIRED);
    v_pos_embd     = create_as_file(tn(LLM_TENSOR_V_ENC_EMBD_POS,   "weight"),   TENSOR_NOT_REQUIRED);
    v_pre_ln_w     = create_tensor (tn(LLM_TENSOR_V_PRE_NORM,  "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
    v_pre_ln_b     = create_tensor (tn(LLM_TENSOR_V_PRE_NORM,  "bias"),   {n_embd}, TENSOR_NOT_REQUIRED);
    v_post_ln_w    = create_tensor (tn(LLM_TENSOR_V_POST_NORM, "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
    v_post_ln_b    = create_tensor (tn(LLM_TENSOR_V_POST_NORM, "bias"),   {n_embd}, TENSOR_NOT_REQUIRED);

    // ffn up/down were swapped by old conversion scripts for these projectors
    const bool ffn_maybe_swapped =
        vhp.proj == LLAMA_CLIP_PROJ_GEMMA3   ||
        vhp.proj == LLAMA_CLIP_PROJ_QWEN2VL  ||
        vhp.proj == LLAMA_CLIP_PROJ_QWEN25VL;

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_V_ENC_INPUT_NORM, "weight", il), {n_embd}, 0);
        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_V_ENC_INPUT_NORM, "bias",   il), {n_embd}, TENSOR_NOT_REQUIRED);

        layer.wqkv = create_tensor(tn(LLM_TENSOR_V_ENC_ATTN_QKV, "weight", il), {n_embd, 3*n_embd_q}, TENSOR_NOT_REQUIRED);
        if (layer.wqkv) {
            layer.wqkv_b = create_tensor(tn(LLM_TENSOR_V_ENC_ATTN_QKV, "bias", il), {3*n_embd_q}, TENSOR_NOT_REQUIRED);
        } else {
            layer.wq   = create_tensor(tn(LLM_TENSOR_V_ENC_ATTN_Q, "weight", il), {n_embd, n_embd_q}, 0);
            layer.wk   = create_tensor(tn(LLM_TENSOR_V_ENC_ATTN_K, "weight", il), {n_embd, n_embd_k_gqa}, 0);
            layer.wv   = create_tensor(tn(LLM_TENSOR_V_ENC_ATTN_V, "weight", il), {n_embd, n_embd_v_gqa}, 0);
            layer.wq_b = create_tensor(tn(LLM_TENSOR_V_ENC_ATTN_Q, "bias",   il), {n_embd_q},     TENSOR_NOT_REQUIRED);
            layer.wk_b = create_tensor(tn(LLM_TENSOR_V_ENC_ATTN_K, "bias",   il), {n_embd_k_gqa}, TENSOR_NOT_REQUIRED);
            layer.wv_b = create_tensor(tn(LLM_TENSOR_V_ENC_ATTN_V, "bias",   il), {n_embd_v_gqa}, TENSOR_NOT_REQUIRED);
        }
        layer.attn_q_norm = create_as_file(tn(LLM_TENSOR_V_ENC_ATTN_Q_NORM, "weight", il), TENSOR_NOT_REQUIRED);
        layer.attn_k_norm = create_as_file(tn(LLM_TENSOR_V_ENC_ATTN_K_NORM, "weight", il), TENSOR_NOT_REQUIRED);

        layer.wo   = create_tensor(tn(LLM_TENSOR_V_ENC_ATTN_O, "weight", il), {n_embd_q, n_embd}, 0);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_V_ENC_ATTN_O, "bias",   il), {n_embd}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_V_ENC_POST_ATTN_NORM, "weight", il), {n_embd}, 0);
        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_V_ENC_POST_ATTN_NORM, "bias",   il), {n_embd}, TENSOR_NOT_REQUIRED);

        layer.ffn_up     = create_as_file(tn(LLM_TENSOR_V_ENC_FFN_UP,   "weight", il), 0);
        layer.ffn_up_b   = create_as_file(tn(LLM_TENSOR_V_ENC_FFN_UP,   "bias",   il), TENSOR_NOT_REQUIRED);
        layer.ffn_gate   = create_as_file(tn(LLM_TENSOR_V_ENC_FFN_GATE, "weight", il), TENSOR_NOT_REQUIRED);
        layer.ffn_gate_b = create_as_file(tn(LLM_TENSOR_V_ENC_FFN_GATE, "bias",   il), TENSOR_NOT_REQUIRED);
        layer.ffn_down   = create_as_file(tn(LLM_TENSOR_V_ENC_FFN_DOWN, "weight", il), 0);
        layer.ffn_down_b = create_as_file(tn(LLM_TENSOR_V_ENC_FFN_DOWN, "bias",   il), TENSOR_NOT_REQUIRED);

        if (ffn_maybe_swapped && layer.ffn_down->ne[0] == n_embd) {
            std::swap(layer.ffn_up,   layer.ffn_down);
            std::swap(layer.ffn_up_b, layer.ffn_down_b);
            if (il == 0) {
                LLAMA_LOG_WARN("%s: ffn up/down are swapped in this file\n", __func__);
            }
        }
    }

    switch (vhp.proj) {
        case LLAMA_CLIP_PROJ_GEMMA3:
            {
                mm_inp_proj      = create_as_file(tn(LLM_TENSOR_V_MM_INP_PROJ, "weight"), 0);
                mm_soft_emb_norm = create_tensor (tn(LLM_TENSOR_V_MM_SOFT_EMB_NORM, "weight"), {n_embd}, 0);
                vhp.n_embd_out   = mm_inp_proj->ne[0];
            } break;
        case LLAMA_CLIP_PROJ_QWEN2VL:
        case LLAMA_CLIP_PROJ_QWEN25VL:
        case LLAMA_CLIP_PROJ_QWEN3VL:
            {
                mm_0_w = create_as_file(tn(LLM_TENSOR_V_MMPROJ, "weight", 0), 0);
                mm_0_b = create_as_file(tn(LLM_TENSOR_V_MMPROJ, "bias",   0), TENSOR_NOT_REQUIRED);
                mm_2_w = create_as_file(tn(LLM_TENSOR_V_MMPROJ, "weight", 2), 0);
                mm_2_b = create_as_file(tn(LLM_TENSOR_V_MMPROJ, "bias",   2), TENSOR_NOT_REQUIRED);
                vhp.n_embd_out = mm_2_w->ne[1];

                if (vhp.proj == LLAMA_CLIP_PROJ_QWEN3VL) {
                    deepstack.resize(n_layer);
                    for (int32_t il : vhp.deepstack_layers) {
                        auto & ds = deepstack[il];
                        ds.norm_w = create_as_file(tn(LLM_TENSOR_V_DS_NORM, "weight", il), 0);
                        ds.norm_b = create_as_file(tn(LLM_TENSOR_V_DS_NORM, "bias",   il), 0);
                        ds.fc1_w  = create_as_file(tn(LLM_TENSOR_V_DS_FC1,  "weight", il), 0);
                        ds.fc1_b  = create_as_file(tn(LLM_TENSOR_V_DS_FC1,  "bias",   il), 0);
                        ds.fc2_w  = create_as_file(tn(LLM_TENSOR_V_DS_FC2,  "weight", il), 0);
                        ds.fc2_b  = create_as_file(tn(LLM_TENSOR_V_DS_FC2,  "bias",   il), 0);
                    }
                    // deepstack rows are concatenated after the main row
                    vhp.n_embd_out *= 1 + (int32_t) vhp.deepstack_layers.size();
                }
            } break;
        case LLAMA_CLIP_PROJ_UNKNOWN:
            GGML_ABORT("unreachable");
    }

    hparams.n_embd_out_impl = vhp.n_embd_out;
}

//
// media input
//

// pixels of the media in the ubatch, uploaded as [nx, ny, 3, n_frames] planar
class llm_graph_input_media : public llm_graph_input_i {
public:
    llm_graph_input_media() = default;
    virtual ~llm_graph_input_media() = default;

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(ubatch->media != nullptr && "encoder ubatch without media");
        const llama_media_input & m = *ubatch->media;
        GGML_ASSERT(!m.data.empty() && "cannot encode a placeholder media");

        const int64_t n = (int64_t) m.nx * m.ny;
        GGML_ASSERT((int64_t) m.data.size() == 3 * n * m.n_frames);
        GGML_ASSERT(ggml_nelements(inp_raw) == 3 * n * m.n_frames);

        std::vector<float> planar(m.data.size());
        for (int f = 0; f < m.n_frames; ++f) {
            const float * src = m.data.data() + 3 * n * f;
            float       * dst = planar.data() + 3 * n * f;
            for (int64_t i = 0; i < n; ++i) {
                dst[        i] = src[3*i + 0];
                dst[1 * n + i] = src[3*i + 1];
                dst[2 * n + i] = src[3*i + 2];
            }
        }
        ggml_backend_tensor_set(inp_raw, planar.data(), 0, ggml_nbytes(inp_raw));

        for (auto & fn : extra) {
            fn(m);
        }
    }

    ggml_tensor * inp_raw = nullptr; // F32 [nx, ny, 3, n_frames]

    // family-specific inputs computed from the media geometry
    std::vector<std::function<void(const llama_media_input &)>> extra;
};

llama_media_input llama_model_clip::reserve_media(uint32_t n_rows_max) const {
    llama_media_input m;
    m.type = LLAMA_MEDIA_TYPE_IMAGE;

    switch (vhp.proj) {
        case LLAMA_CLIP_PROJ_GEMMA3:
            {
                m.nx = vhp.image_size;
                m.ny = vhp.image_size;
            } break;
        case LLAMA_CLIP_PROJ_QWEN2VL:
        case LLAMA_CLIP_PROJ_QWEN25VL:
        case LLAMA_CLIP_PROJ_QWEN3VL:
            {
                // a square image with n_rows_max output rows, capped by the model limit
                const int32_t align = vhp.patch_size * vhp.n_merge;
                const int64_t max_rows = std::min<int64_t>(n_rows_max, vhp.image_max_pixels / (align * align));
                const int32_t side = std::max<int32_t>(1, (int32_t) std::sqrt((double) max_rows));
                m.nx = side * align;
                m.ny = side * align;
                m.n_frames = vhp.n_temporal_merge;
            } break;
        case LLAMA_CLIP_PROJ_UNKNOWN:
            GGML_ABORT("unreachable");
    }

    return m;
}

//
// graphs
//

struct llama_model_clip::graph_vit : public llm_graph_context {
    const llama_model_clip & clip;

    llama_media_input media_reserve;

    const llama_media_input & media;

    const int64_t patch_size;
    const int64_t n_patches_x;
    const int64_t n_patches_y;
    const int64_t n_patches;

    llm_graph_input_media * inp_media = nullptr;

    graph_vit(const llama_model_clip & clip, const llm_graph_params & params) :
        llm_graph_context(params),
        clip(clip),
        media_reserve(params.ubatch.media ? llama_media_input() : clip.reserve_media(params.cparams.n_ubatch)),
        media(params.ubatch.media ? *params.ubatch.media : media_reserve),
        patch_size (clip.vhp.patch_size),
        n_patches_x(media.nx / patch_size),
        n_patches_y(media.ny / patch_size),
        n_patches  (n_patches_x * n_patches_y) {
        auto inp = std::make_unique<llm_graph_input_media>();
        inp->inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, media.nx, media.ny, 3, media.n_frames);
        ggml_set_input(inp->inp_raw);
        cb(inp->inp_raw, "inp_raw", -1);
        inp_media = static_cast<llm_graph_input_media *>(res->add_input(std::move(inp)));
    }

    // conv patch embedding: [n_embd, n_patches]
    ggml_tensor * build_inp_patches() {
        ggml_tensor * cur = ggml_conv_2d(ctx0, clip.v_patch_embd, inp_media->inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        cur = ggml_reshape_2d(ctx0, cur, n_patches, n_embd);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        if (clip.v_patch_bias) {
            cur = ggml_add(ctx0, cur, clip.v_patch_bias);
        }
        cb(cur, "inp_patches", -1);
        return cur;
    }

    // ViT on libllama's helpers; inp is [n_embd, n_pos]
    ggml_tensor * build_vit(
            ggml_tensor * inp,
            int64_t n_pos,
            llm_norm_type norm_t,
            ggml_tensor * pos_embd,
            const std::function<ggml_tensor * (ggml_tensor *, int)> & add_pos = nullptr,
            const std::function<ggml_tensor * (int)> & get_mask = nullptr,
            const std::function<void (ggml_tensor *, int)> & layer_out = nullptr) {
        const int64_t n_embd_head = hparams.n_embd_head_k();
        const float   kq_scale    = 1.0f / sqrtf((float) n_embd_head);

        ggml_tensor * inpL = inp;
        if (pos_embd) {
            inpL = ggml_add(ctx0, inpL, pos_embd);
            cb(inpL, "pos_embd", -1);
        }
        if (clip.v_pre_ln_w) {
            inpL = build_norm(inpL, clip.v_pre_ln_w, clip.v_pre_ln_b, norm_t, -1);
            cb(inpL, "pre_ln", -1);
        }

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            ggml_tensor * cur = build_norm(inpL, layer.attn_norm, layer.attn_norm_b, norm_t, il);
            cb(cur, "attn_norm", il);

            ggml_tensor * Qcur;
            ggml_tensor * Kcur;
            ggml_tensor * Vcur;
            if (layer.wqkv) {
                ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur);
                if (layer.wqkv_b) {
                    qkv = ggml_add(ctx0, qkv, layer.wqkv_b);
                }
                const size_t es = ggml_element_size(qkv);
                Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head,    n_pos, n_embd_head*es, qkv->nb[1], 0);
                Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_pos, n_embd_head*es, qkv->nb[1], n_embd_head*n_head*es);
                Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_pos, n_embd_head*es, qkv->nb[1], n_embd_head*(n_head + n_head_kv)*es);
            } else {
                Qcur = build_lora_mm(layer.wq, cur);
                Kcur = build_lora_mm(layer.wk, cur);
                Vcur = build_lora_mm(layer.wv, cur);
                if (layer.wq_b) { Qcur = ggml_add(ctx0, Qcur, layer.wq_b); }
                if (layer.wk_b) { Kcur = ggml_add(ctx0, Kcur, layer.wk_b); }
                if (layer.wv_b) { Vcur = ggml_add(ctx0, Vcur, layer.wv_b); }
                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_pos);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_pos);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_pos);
            }

            if (layer.attn_q_norm) {
                Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, norm_t, il);
            }
            if (layer.attn_k_norm) {
                Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, norm_t, il);
            }

            if (add_pos) {
                Qcur = add_pos(Qcur, il);
                Kcur = add_pos(Kcur, il);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            ggml_tensor * mask = get_mask ? get_mask(il) : nullptr;

            cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, mask, nullptr, nullptr, nullptr, 0, kq_scale, il);
            cur = build_lora_mm(layer.wo, cur);
            if (layer.wo_b) {
                cur = ggml_add(ctx0, cur, layer.wo_b);
            }
            cb(cur, "attn_out", il);

            cur = ggml_add(ctx0, cur, inpL);
            inpL = cur;

            cur = build_norm(cur, layer.ffn_norm, layer.ffn_norm_b, norm_t, il);
            cb(cur, "ffn_inp_normed", il);

            cur = build_ffn(cur,
                    layer.ffn_up,   layer.ffn_up_b,   nullptr,
                    layer.ffn_gate, layer.ffn_gate_b, nullptr,
                    layer.ffn_down, layer.ffn_down_b, nullptr,
                    nullptr,
                    clip.vhp.ffn_op, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, inpL);
            cb(cur, "layer_out", il);

            if (layer_out) {
                layer_out(cur, il);
            }

            inpL = cur;
        }

        if (clip.v_post_ln_w) {
            inpL = build_norm(inpL, clip.v_post_ln_w, clip.v_post_ln_b, norm_t, -1);
            cb(inpL, "post_ln", -1);
        }

        return inpL;
    }

    void finish(ggml_tensor * cur) {
        cb(cur, "result_embd", -1);
        res->t_embd = cur;
        ggml_build_forward_expand(gf, cur);
    }
};

// SigLIP tower, avg pooling and a soft-norm projection
struct llama_model_clip::graph_gemma3 : public llama_model_clip::graph_vit {
    graph_gemma3(const llama_model_clip & clip, const llm_graph_params & params) : graph_vit(clip, params) {
        ggml_tensor * cur = build_inp_patches();

        cur = build_vit(cur, n_patches, LLM_NORM, clip.v_pos_embd);

        GGML_ASSERT(n_patches_x == n_patches_y);
        const int k = clip.vhp.n_merge;

        cur = ggml_transpose(ctx0, cur);
        cur = ggml_cont_4d(ctx0, cur, n_patches_x, n_patches_y, n_embd, 1);
        cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, k, k, k, k, 0, 0);
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0] * cur->ne[0], n_embd, 1);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_eps);
        cur = ggml_mul(ctx0, cur, clip.mm_soft_emb_norm);

        cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, clip.mm_inp_proj)), cur);
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);

        finish(cur);
    }
};

// Qwen-VL: patch embedding of one frame or a merged frame pair, rows in 2x2 merge order, [n_embd, n_patches]
static ggml_tensor * clip_qwen_patches(ggml_context * ctx0, const llama_model_clip & clip, ggml_tensor * raw,
        const llama_media_input & media, int64_t n_embd, int64_t px, int64_t py) {
    const int p = clip.vhp.patch_size;
    ggml_tensor * f0 = raw;
    ggml_tensor * f1 = raw;
    if (media.n_frames == 2) {
        const size_t nb1 = ggml_row_size(raw->type, media.nx);
        const size_t nb2 = ggml_row_size(raw->type, (int64_t) media.nx * media.ny);
        f0 = ggml_view_3d(ctx0, raw, media.nx, media.ny, 3, nb1, nb2, 0);
        f1 = ggml_view_3d(ctx0, raw, media.nx, media.ny, 3, nb1, nb2, nb2 * 3);
    }
    ggml_tensor * inp = ggml_add(ctx0,
            ggml_conv_2d(ctx0, clip.v_patch_embd,   f0, p, p, 0, 0, 1, 1),
            ggml_conv_2d(ctx0, clip.v_patch_embd_1, f1, p, p, 0, 0, 1, 1));

    inp = ggml_permute(ctx0, inp, 1, 2, 0, 3); // [w, h, c, b] -> [c, w, h, b]
    inp = ggml_cont_4d(ctx0, inp, n_embd * 2, px / 2, py, 1);
    inp = ggml_reshape_4d(ctx0, inp, n_embd * 2, px / 2, 2, py / 2);
    inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
    inp = ggml_cont_2d(ctx0, inp, n_embd, px * py);
    return inp;
}

// vision M-RoPE positions [y, x, y, x] of each patch in 2x2 merge order, optionally remapped by window order
static void clip_qwen_positions(std::vector<int32_t> & out, int px, int py, const std::vector<int32_t> * win_idx) {
    const int n = px * py;
    out.assign((size_t) n * 4, 0);
    int ptr = 0;
    for (int ty = 0; ty < py / 2; ty++) {
        for (int tx = 0; tx < px / 2; tx++) {
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int dst = ptr;
                    if (win_idx) {
                        dst = (*win_idx)[ptr / 4] * 4 + ptr % 4;
                    }
                    const int y = ty * 2 + dy;
                    const int x = tx * 2 + dx;
                    out[          dst] = y;
                    out[    n   + dst] = x;
                    out[2 * n   + dst] = y;
                    out[3 * n   + dst] = x;
                    ptr++;
                }
            }
        }
    }
}

// Qwen2.5-VL windows: merged-token order inside windows, and the patch-level block mask
static void clip_qwen_windows(int px, int py, int patch_size, int attn_window,
        std::vector<int32_t> & idx, std::vector<int32_t> & inv_idx, std::vector<float> & mask) {
    const int pw = px / 2;
    const int ph = py / 2;
    const int n  = px * py;
    const int grid_window = attn_window / patch_size / 2;

    idx.assign((size_t) pw * ph, 0);
    inv_idx.assign((size_t) pw * ph, 0);
    mask.assign((size_t) n * n, -INFINITY);

    int dst = 0;
    int mask_row = 0;
    for (int y = 0; y < ph; y += grid_window) {
        for (int x = 0; x < pw; x += grid_window) {
            const int win_h = std::min(grid_window, ph - y);
            const int win_w = std::min(grid_window, pw - x);
            const int dst_0 = dst;
            for (int dy = 0; dy < win_h; dy++) {
                for (int dx = 0; dx < win_w; dx++) {
                    const int src = (y + dy) * pw + (x + dx);
                    idx[src]     = dst;
                    inv_idx[dst] = src;
                    dst++;
                }
            }
            for (int r = 0; r < win_h * win_w * 4; r++) {
                float * row = mask.data() + (size_t) mask_row * n;
                std::fill(row + dst_0 * 4, row + dst * 4, 0.0f);
                mask_row++;
            }
        }
    }
}

struct llama_model_clip::graph_qwen2vl : public llama_model_clip::graph_vit {
    graph_qwen2vl(const llama_model_clip & clip, const llm_graph_params & params) : graph_vit(clip, params) {
        const bool    is_25  = clip.vhp.proj == LLAMA_CLIP_PROJ_QWEN25VL;
        const bool    use_wa = is_25 && clip.vhp.n_wa_pattern > 0;
        const int64_t n_pos  = n_patches;
        const int64_t d_head = hparams.n_embd_head_k();

        ggml_tensor * cur = clip_qwen_patches(ctx0, clip, inp_media->inp_raw, media, n_embd, n_patches_x, n_patches_y);

        ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos * 4);
        ggml_set_input(positions);
        cb(positions, "positions", -1);

        ggml_tensor * win_idx  = nullptr;
        ggml_tensor * win_inv  = nullptr;
        ggml_tensor * win_mask = nullptr;
        if (use_wa) {
            win_idx  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / 4);
            win_inv  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / 4);
            win_mask = ggml_new_tensor_2d(ctx0, cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, n_pos, n_pos);
            ggml_set_input(win_idx);
            ggml_set_input(win_inv);
            ggml_set_input(win_mask);
            cb(win_mask, "window_mask", -1);

            cur = ggml_reshape_2d(ctx0, cur, n_embd * 4, n_pos / 4);
            cur = ggml_get_rows(ctx0, cur, win_inv);
            cur = ggml_reshape_2d(ctx0, cur, n_embd, n_pos);
        }

        const int     patch_size  = clip.vhp.patch_size;
        const int     attn_window = clip.vhp.attn_window;
        inp_media->extra.push_back([=](const llama_media_input & m) {
            const int px = m.nx / patch_size;
            const int py = m.ny / patch_size;
            std::vector<int32_t> idx;
            std::vector<int32_t> inv;
            std::vector<float>   mask;
            if (use_wa) {
                clip_qwen_windows(px, py, patch_size, attn_window, idx, inv, mask);
                ggml_backend_tensor_set(win_idx, idx.data(), 0, ggml_nbytes(win_idx));
                ggml_backend_tensor_set(win_inv, inv.data(), 0, ggml_nbytes(win_inv));
                if (win_mask->type == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> m16(mask.size());
                    ggml_fp32_to_fp16_row(mask.data(), m16.data(), (int64_t) mask.size());
                    ggml_backend_tensor_set(win_mask, m16.data(), 0, ggml_nbytes(win_mask));
                } else {
                    ggml_backend_tensor_set(win_mask, mask.data(), 0, ggml_nbytes(win_mask));
                }
            }
            std::vector<int32_t> pos;
            clip_qwen_positions(pos, px, py, use_wa ? &idx : nullptr);
            ggml_backend_tensor_set(positions, pos.data(), 0, ggml_nbytes(positions));
        });

        int sections[4] = { (int) d_head/4, (int) d_head/4, (int) d_head/4, (int) d_head/4 };
        auto add_pos = [&](ggml_tensor * x, int) {
            return ggml_rope_multi(ctx0, x, positions, nullptr, d_head/2, sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
        };
        const int n_wa_pattern = clip.vhp.n_wa_pattern;
        auto get_mask = [&](int il) -> ggml_tensor * {
            const bool full = !use_wa || (il + 1) % n_wa_pattern == 0;
            return full ? nullptr : win_mask;
        };

        cur = build_vit(cur, n_pos, is_25 ? LLM_NORM_RMS : LLM_NORM, nullptr, add_pos, get_mask);

        cur = ggml_reshape_2d(ctx0, cur, n_embd * 4, n_pos / 4);
        cur = build_ffn(cur,
                clip.mm_0_w, clip.mm_0_b, nullptr,
                nullptr, nullptr, nullptr,
                clip.mm_2_w, clip.mm_2_b, nullptr,
                nullptr, LLM_FFN_GELU, LLM_FFN_SEQ, -1);

        if (use_wa) {
            cur = ggml_get_rows(ctx0, cur, win_idx);
        }

        finish(cur);
    }
};

struct llama_model_clip::graph_qwen3vl : public llama_model_clip::graph_vit {
    graph_qwen3vl(const llama_model_clip & clip, const llm_graph_params & params) : graph_vit(clip, params) {
        const int64_t n_pos  = n_patches;
        const int64_t d_head = hparams.n_embd_head_k();

        ggml_tensor * cur = clip_qwen_patches(ctx0, clip, inp_media->inp_raw, media, n_embd, n_patches_x, n_patches_y);
        cur = ggml_add(ctx0, cur, clip.v_patch_bias);

        // learned position embedding, resized to the patch grid and put in merge order
        {
            ggml_tensor * pe = clip.v_pos_embd;
            const int n_side = (int) std::sqrt((double) pe->ne[1]);
            if (n_patches_x != n_side || n_patches_y != n_side) {
                pe = ggml_reshape_3d(ctx0, pe, n_embd, n_side, n_side);
                pe = ggml_permute(ctx0, pe, 2, 0, 1, 3);
                pe = ggml_interpolate(ctx0, pe, n_patches_x, n_patches_y, n_embd, 1, GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
                pe = ggml_permute(ctx0, pe, 1, 2, 0, 3);
                pe = ggml_cont_2d(ctx0, pe, n_embd, n_patches_x * n_patches_y);
            }
            pe = ggml_cont_4d(ctx0, pe, n_embd * 2, n_patches_x / 2, n_patches_y, 1);
            pe = ggml_reshape_4d(ctx0, pe, n_embd * 2, n_patches_x / 2, 2, n_patches_y / 2);
            pe = ggml_permute(ctx0, pe, 0, 2, 1, 3);
            pe = ggml_cont_2d(ctx0, pe, n_embd, n_patches_x * n_patches_y);
            cur = ggml_add(ctx0, cur, pe);
        }

        ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos * 4);
        ggml_set_input(positions);
        cb(positions, "positions", -1);

        const int patch_size = clip.vhp.patch_size;
        inp_media->extra.push_back([=](const llama_media_input & m) {
            std::vector<int32_t> pos;
            clip_qwen_positions(pos, m.nx / patch_size, m.ny / patch_size, nullptr);
            ggml_backend_tensor_set(positions, pos.data(), 0, ggml_nbytes(positions));
        });

        int sections[4] = { (int) d_head/4, (int) d_head/4, (int) d_head/4, (int) d_head/4 };
        auto add_pos = [&](ggml_tensor * x, int) {
            return ggml_rope_multi(ctx0, x, positions, nullptr, d_head/2, sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
        };

        ggml_tensor * deepstack = nullptr;
        auto layer_out = [&](ggml_tensor * x, int il) {
            const auto & ds = clip.deepstack.empty() ? deepstack_layer() : clip.deepstack[il];
            if (ds.fc1_w == nullptr) {
                return;
            }
            ggml_tensor * feat = ggml_reshape_2d(ctx0, x, n_embd * 4, n_pos / 4);
            feat = build_norm(feat, ds.norm_w, ds.norm_b, LLM_NORM, il);
            feat = build_ffn(feat,
                    ds.fc1_w, ds.fc1_b, nullptr,
                    nullptr, nullptr, nullptr,
                    ds.fc2_w, ds.fc2_b, nullptr,
                    nullptr, LLM_FFN_GELU, LLM_FFN_SEQ, il);
            deepstack = deepstack ? ggml_concat(ctx0, deepstack, feat, 0) : feat;
        };

        cur = build_vit(cur, n_pos, LLM_NORM, nullptr, add_pos, nullptr, layer_out);

        cur = ggml_reshape_2d(ctx0, cur, n_embd * 4, n_pos / 4);
        cur = build_ffn(cur,
                clip.mm_0_w, clip.mm_0_b, nullptr,
                nullptr, nullptr, nullptr,
                clip.mm_2_w, clip.mm_2_b, nullptr,
                nullptr, LLM_FFN_GELU, LLM_FFN_SEQ, -1);

        if (deepstack) {
            cur = ggml_concat(ctx0, cur, deepstack, 0);
        }

        finish(cur);
    }
};

std::unique_ptr<llm_graph_context> llama_model_clip::build_arch_graph(const llm_graph_params & params) const {
    switch (vhp.proj) {
        case LLAMA_CLIP_PROJ_GEMMA3:
            return std::make_unique<graph_gemma3>(*this, params);
        case LLAMA_CLIP_PROJ_QWEN2VL:
        case LLAMA_CLIP_PROJ_QWEN25VL:
            return std::make_unique<graph_qwen2vl>(*this, params);
        case LLAMA_CLIP_PROJ_QWEN3VL:
            return std::make_unique<graph_qwen3vl>(*this, params);
        default:
            GGML_ABORT("clip: no graph for projector type '%s'", vhp.proj_name.c_str());
    }
}

//
// media preprocessing and layout
//

bool llama_model_clip::media_preprocess_image(const llama_media_proc & proc, const llama_media_img::image_u8 * frames, int n_frames, llama_media & m) const {
    GGML_UNUSED(proc);

    using namespace llama_media_img;

    m.inp.type     = LLAMA_MEDIA_TYPE_IMAGE;
    m.inp.n_frames = n_frames;
    m.inp.data.clear();

    switch (vhp.proj) {
        case LLAMA_CLIP_PROJ_GEMMA3:
            {
                if (n_frames != 1) {
                    return false;
                }
                image_u8 resized;
                resize(frames[0], resized, { vhp.image_size, vhp.image_size }, RESIZE_ALGO_BILINEAR, PAD_CEIL, {0, 0, 0});
                m.inp.nx = vhp.image_size;
                m.inp.ny = vhp.image_size;
                if (!frames[0].is_placeholder()) {
                    to_f32_normalized(resized, vhp.image_mean, vhp.image_std, m.inp.data);
                }
                const int32_t side = vhp.image_size / vhp.patch_size / vhp.n_merge;
                m.grid_x   = side;
                m.grid_y   = side;
                m.n_tokens = side * side;
            } break;
        case LLAMA_CLIP_PROJ_QWEN2VL:
        case LLAMA_CLIP_PROJ_QWEN25VL:
        case LLAMA_CLIP_PROJ_QWEN3VL:
            {
                if (n_frames > vhp.n_temporal_merge) {
                    return false;
                }
                const int32_t align = vhp.patch_size * vhp.n_merge;
                const int64_t area  = (int64_t) align * align;

                // user limits, then file limits, then model defaults
                int64_t min_px = proc.params.image_min_tokens > 0 ? proc.params.image_min_tokens * area
                               : vhp.file_min_pixels > 0         ? vhp.file_min_pixels : vhp.image_min_pixels;
                int64_t max_px = proc.params.image_max_tokens > 0 ? proc.params.image_max_tokens * area
                               : vhp.file_max_pixels > 0         ? vhp.file_max_pixels : vhp.image_max_pixels;
                // the encoder context takes at most n_rows_max rows
                max_px = std::min<int64_t>(max_px, (int64_t) proc.n_rows_max * area);
                min_px = std::min(min_px, max_px);

                calc_size_opt opt;
                opt.align_size      = align;
                opt.min_pixels      = (int) min_px;
                opt.max_pixels      = (int) max_px;
                opt.round_half_even = true;
                const image_size target = calc_size_preserved_ratio(frames[0].get_size(), opt);

                m.inp.nx = target.width;
                m.inp.ny = target.height;
                for (int f = 0; f < n_frames; ++f) {
                    image_u8 resized;
                    resize(frames[f], resized, target, RESIZE_ALGO_BICUBIC, PAD_NONE, {0, 0, 0});
                    if (!frames[f].is_placeholder()) {
                        std::vector<float> one;
                        to_f32_normalized(resized, vhp.image_mean, vhp.image_std, one);
                        m.inp.data.insert(m.inp.data.end(), one.begin(), one.end());
                    }
                }
                m.grid_x   = target.width  / align;
                m.grid_y   = target.height / align;
                m.n_tokens = m.grid_x * m.grid_y;
            } break;
        default:
            return false;
    }

    return true;
}

void llama_model_clip::media_image_wrap(const llama_media_proc & proc, std::vector<llama_token> & beg, std::vector<llama_token> & end) const {
    beg.clear();
    end.clear();

    auto lookup = [&](const char * text) {
        std::vector<llama_token> toks(8);
        const int32_t n = llama_tokenize(proc.vocab, text, (int32_t) strlen(text), toks.data(), (int32_t) toks.size(), false, true);
        GGML_ASSERT(n > 0);
        toks.resize(n);
        return toks;
    };

    switch (vhp.proj) {
        case LLAMA_CLIP_PROJ_GEMMA3:
            {
                beg = lookup("<start_of_image>");
                end = lookup("<end_of_image>");
            } break;
        case LLAMA_CLIP_PROJ_QWEN2VL:
        case LLAMA_CLIP_PROJ_QWEN25VL:
        case LLAMA_CLIP_PROJ_QWEN3VL:
            {
                beg = lookup("<|vision_start|>");
                end = lookup("<|vision_end|>");
            } break;
        default:
            break;
    }
}
