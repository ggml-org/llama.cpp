#include "models.h"

#include "llama-impl.h"
#include "llama-media.h"
#include "llama-vocab.h"
#include "../clip/clip-gen-memory.h"

#include "../clip/clip-model.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

// the vision projectors with native graphs, from the projector the engine resolved (e.g. the vision tower of a
// legacy "qwen2.5o" file is Qwen2.5-VL)
static llama_clip_projector clip_projector_native(const clip_model * m) {
    switch (m ? m->proj_type : PROJECTOR_TYPE_UNKNOWN) {
        case PROJECTOR_TYPE_GEMMA3:   return LLAMA_CLIP_PROJ_GEMMA3;
        case PROJECTOR_TYPE_QWEN2VL:  return LLAMA_CLIP_PROJ_QWEN2VL;
        case PROJECTOR_TYPE_QWEN25VL: return LLAMA_CLIP_PROJ_QWEN25VL;
        case PROJECTOR_TYPE_QWEN3VL:  return LLAMA_CLIP_PROJ_QWEN3VL;
        default:                      return LLAMA_CLIP_PROJ_UNKNOWN;
    }
}

void llama_model_clip::load_vocab(llama_model_loader &) {
    // an mmproj has no vocabulary
}

// the gen tower (backbone hidden state of a gen step) and the outputs of a context of each type
uint32_t llama_model_clip::ctx_n_embd_state(enum llama_context_type type) const {
    return type == LLAMA_CONTEXT_TYPE_GEN_CODE && engine_gen() ? (uint32_t) engine_gen()->n_mmproj_embd() : 0;
}

uint32_t llama_model_clip::ctx_n_embd_out(enum llama_context_type type) const {
    if (type == LLAMA_CONTEXT_TYPE_GEN_CODE && engine_gen()) {
        return (uint32_t) engine_gen()->n_mmproj_embd(); // the feedback row of a gen step
    }
    if (type == LLAMA_CONTEXT_TYPE_ENCODER && !native_v) {
        if (const clip_model * m = engine_for_ctx(type)) {
            return (uint32_t) m->n_mmproj_embd();
        }
    }
    return hparams.n_embd_out();
}

// the gen pipelines keep their per-sequence state on the host side of the context
llama_memory_i * llama_model_clip::create_clip_memory(const llama_memory_params & params, const llama_cparams & cparams) const {
    GGML_UNUSED(params);
    if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_GEN_CODE && engine_gen() != nullptr) {
        return new llama_memory_clip_gen();
    }
    return nullptr;
}

void llama_model_clip::load_arch_hparams(llama_model_loader & ml) {
    hparams.causal_attn = false;
    hparams.n_ctx_train = 8192; // unused, encoders keep no memory

    // every modality of the file, for the engine graphs and for the layout of all media
    engine = std::make_unique<clip_models>(clip_model_load_hparams(ml));
    path   = ml.fname;

    // Gemma 3, Qwen2/2.5/3-VL and the gen steps of Qwen3-TTS run on their native graphs, the other towers on the
    // graphs of the clip engine; the engine loads the weights and the hparams of every tower, lays out and
    // preprocesses every media and runs the gen pipelines
    vhp.has_vision = engine_vision() != nullptr;
    vhp.proj       = clip_projector_native(engine_vision());
    native_gen     = engine_gen() != nullptr && engine_gen()->proj_type == PROJECTOR_TYPE_QWEN3TTS_GEN;

    if (!vhp.has_vision) {
        const clip_model * main = engine_audio() ? engine_audio() : engine_gen();
        if (main == nullptr) {
            throw std::runtime_error("clip: the file has no tower");
        }
        load_engine_hparams(*main);
        return;
    }

    native_v = vhp.proj != LLAMA_CLIP_PROJ_UNKNOWN;
    load_engine_hparams(*engine_vision());
}

void llama_model_clip::load_arch_tensors(llama_model_loader & ml) {
    load_engine_tensors(ml);
    if (native_v) {
        hparams.n_embd_out_impl = (uint32_t) engine_vision()->n_mmproj_embd(); // from the projector tensors
    }
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
    GGML_UNUSED(n_rows_max);

    // the warmup image of the clip engine, as mtmd reserved; larger media grow the compute buffers on demand
    const auto & hp = engine_vision()->hparams;

    llama_media_input m;
    m.type     = LLAMA_MEDIA_TYPE_IMAGE;
    m.nx       = hp.warmup_image_size;
    m.ny       = hp.warmup_image_size;
    m.n_frames = 1;
    return m;
}

//
// graphs
//

struct llama_model_clip::graph_vit : public llm_graph_context {
    const llama_model_clip & clip;
    const clip_model & cm; // the weights and hparams of the engine's vision tower

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
        cm(*clip.engine_vision()),
        media_reserve(params.ubatch.media ? llama_media_input() : clip.reserve_media(params.cparams.n_ubatch)),
        media(params.ubatch.media ? *params.ubatch.media : media_reserve),
        patch_size (cm.hparams.patch_size),
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
        ggml_tensor * cur = ggml_conv_2d(ctx0, cm.patch_embeddings_0, inp_media->inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        cur = ggml_reshape_2d(ctx0, cur, n_patches, n_embd);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        if (cm.patch_bias) {
            cur = ggml_add(ctx0, cur, cm.patch_bias);
        }
        cb(cur, "inp_patches", -1);
        return cur;
    }

    llm_ffn_op_type ffn_op() const {
        switch (cm.hparams.ffn_op) {
            case FFN_SILU:       return LLM_FFN_SILU;
            case FFN_GELU_QUICK: return LLM_FFN_GELU_QUICK;
            case FFN_GELU_ERF:   return LLM_FFN_GELU_ERF;
            default:             return LLM_FFN_GELU;
        }
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
        if (cm.pre_ln_w) {
            inpL = build_norm(inpL, cm.pre_ln_w, cm.pre_ln_b, norm_t, -1);
            cb(inpL, "pre_ln", -1);
        }

        for (int il = 0; il < n_layer; ++il) {
            const clip_layer & layer = cm.layers[il];

            ggml_tensor * cur = build_norm(inpL, layer.ln_1_w, layer.ln_1_b, norm_t, il);
            cb(cur, "attn_norm", il);

            ggml_tensor * Qcur;
            ggml_tensor * Kcur;
            ggml_tensor * Vcur;
            if (layer.qkv_w) {
                ggml_tensor * qkv = build_lora_mm(layer.qkv_w, cur);
                if (layer.qkv_b) {
                    qkv = ggml_add(ctx0, qkv, layer.qkv_b);
                }
                const size_t es = ggml_element_size(qkv);
                Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head,    n_pos, n_embd_head*es, qkv->nb[1], 0);
                Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_pos, n_embd_head*es, qkv->nb[1], n_embd_head*n_head*es);
                Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_pos, n_embd_head*es, qkv->nb[1], n_embd_head*(n_head + n_head_kv)*es);
            } else {
                Qcur = build_lora_mm(layer.q_w, cur);
                Kcur = build_lora_mm(layer.k_w, cur);
                Vcur = build_lora_mm(layer.v_w, cur);
                if (layer.q_b) { Qcur = ggml_add(ctx0, Qcur, layer.q_b); }
                if (layer.k_b) { Kcur = ggml_add(ctx0, Kcur, layer.k_b); }
                if (layer.v_b) { Vcur = ggml_add(ctx0, Vcur, layer.v_b); }
                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_pos);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_pos);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_pos);
            }

            if (layer.q_norm) {
                Qcur = build_norm(Qcur, layer.q_norm, nullptr, norm_t, il);
            }
            if (layer.k_norm) {
                Kcur = build_norm(Kcur, layer.k_norm, nullptr, norm_t, il);
            }

            if (add_pos) {
                Qcur = add_pos(Qcur, il);
                Kcur = add_pos(Kcur, il);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            ggml_tensor * mask = get_mask ? get_mask(il) : nullptr;

            // Q, K and V enter the graph before the attention, as in the clip engine: the K/V casts come after the
            // projections, which lowers the peak of the compute buffer (one activation less) and keeps the node order
            ggml_build_forward_order(gf, Qcur);
            ggml_build_forward_order(gf, Kcur);
            ggml_build_forward_order(gf, Vcur);

            cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, mask, nullptr, nullptr, nullptr, 0, kq_scale, il);
            cur = build_lora_mm(layer.o_w, cur);
            if (layer.o_b) {
                cur = ggml_add(ctx0, cur, layer.o_b);
            }
            cb(cur, "attn_out", il);

            cur = ggml_add(ctx0, cur, inpL);
            inpL = cur;

            cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, il);
            cb(cur, "ffn_inp_normed", il);

            cur = build_ffn(cur,
                    layer.ff_up_w,   layer.ff_up_b,   nullptr,
                    layer.ff_gate_w, layer.ff_gate_b, nullptr,
                    layer.ff_down_w, layer.ff_down_b, nullptr,
                    nullptr,
                    ffn_op(), LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, inpL);
            cb(cur, "layer_out", il);

            if (layer_out) {
                layer_out(cur, il);
            }

            inpL = cur;
        }

        if (cm.post_ln_w) {
            inpL = build_norm(inpL, cm.post_ln_w, cm.post_ln_b, norm_t, -1);
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

        cur = build_vit(cur, n_patches, LLM_NORM, cm.position_embeddings);

        GGML_ASSERT(n_patches_x == n_patches_y);
        const int k = cm.hparams.n_merge;

        cur = ggml_transpose(ctx0, cur);
        cur = ggml_cont_4d(ctx0, cur, n_patches_x, n_patches_y, n_embd, 1);
        cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, k, k, k, k, 0, 0);
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0] * cur->ne[0], n_embd, 1);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_eps);
        cur = ggml_mul(ctx0, cur, cm.mm_soft_emb_norm_w);

        cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cm.mm_input_proj_w)), cur);
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);

        finish(cur);
    }
};

// Qwen-VL: patch embedding of one frame or a merged frame pair, rows in 2x2 merge order, [n_embd, n_patches]
static ggml_tensor * clip_qwen_patches(ggml_context * ctx0, const clip_model & cm, ggml_tensor * raw,
        const llama_media_input & media, int64_t n_embd, int64_t px, int64_t py) {
    const int p = cm.hparams.patch_size;
    ggml_tensor * f0 = raw;
    ggml_tensor * f1 = raw;
    if (media.n_frames == 2) {
        const size_t nb1 = ggml_row_size(raw->type, media.nx);
        const size_t nb2 = ggml_row_size(raw->type, (int64_t) media.nx * media.ny);
        f0 = ggml_view_3d(ctx0, raw, media.nx, media.ny, 3, nb1, nb2, 0);
        f1 = ggml_view_3d(ctx0, raw, media.nx, media.ny, 3, nb1, nb2, nb2 * 3);
    }
    ggml_tensor * inp = ggml_add(ctx0,
            ggml_conv_2d(ctx0, cm.patch_embeddings_0,   f0, p, p, 0, 0, 1, 1),
            ggml_conv_2d(ctx0, cm.patch_embeddings_1, f1, p, p, 0, 0, 1, 1));

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
        const bool    use_wa = is_25 && cm.hparams.n_wa_pattern > 0;
        const int64_t n_pos  = n_patches;
        const int64_t d_head = hparams.n_embd_head_k();

        ggml_tensor * cur = clip_qwen_patches(ctx0, cm, inp_media->inp_raw, media, n_embd, n_patches_x, n_patches_y);

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

        const int     patch_size  = cm.hparams.patch_size;
        const int     attn_window = (cm.hparams.attn_window_size > 0 ? cm.hparams.attn_window_size : 112);
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
        const int n_wa_pattern = cm.hparams.n_wa_pattern;
        auto get_mask = [&](int il) -> ggml_tensor * {
            const bool full = !use_wa || (il + 1) % n_wa_pattern == 0;
            return full ? nullptr : win_mask;
        };

        cur = build_vit(cur, n_pos, is_25 ? LLM_NORM_RMS : LLM_NORM, nullptr, add_pos, get_mask);

        cur = ggml_reshape_2d(ctx0, cur, n_embd * 4, n_pos / 4);
        cur = build_ffn(cur,
                cm.mm_0_w, cm.mm_0_b, nullptr,
                nullptr, nullptr, nullptr,
                cm.mm_1_w, cm.mm_1_b, nullptr,
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

        ggml_tensor * cur = clip_qwen_patches(ctx0, cm, inp_media->inp_raw, media, n_embd, n_patches_x, n_patches_y);
        cur = ggml_add(ctx0, cur, cm.patch_bias);

        // learned position embedding, resized to the patch grid and put in merge order
        {
            ggml_tensor * pe = cm.position_embeddings;
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

        const int patch_size = cm.hparams.patch_size;
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
            const clip_layer & ds = cm.layers[il];
            if (ds.deepstack_fc1_w == nullptr) {
                return;
            }
            ggml_tensor * feat = ggml_reshape_2d(ctx0, x, n_embd * 4, n_pos / 4);
            feat = build_norm(feat, ds.deepstack_norm_w, ds.deepstack_norm_b, LLM_NORM, il);
            feat = build_ffn(feat,
                    ds.deepstack_fc1_w, ds.deepstack_fc1_b, nullptr,
                    nullptr, nullptr, nullptr,
                    ds.deepstack_fc2_w, ds.deepstack_fc2_b, nullptr,
                    nullptr, LLM_FFN_GELU, LLM_FFN_SEQ, il);
            deepstack = deepstack ? ggml_concat(ctx0, deepstack, feat, 0) : feat;
        };

        cur = build_vit(cur, n_pos, LLM_NORM, nullptr, add_pos, nullptr, layer_out);

        cur = ggml_reshape_2d(ctx0, cur, n_embd * 4, n_pos / 4);
        cur = build_ffn(cur,
                cm.mm_0_w, cm.mm_0_b, nullptr,
                nullptr, nullptr, nullptr,
                cm.mm_1_w, cm.mm_1_b, nullptr,
                nullptr, LLM_FFN_GELU, LLM_FFN_SEQ, -1);

        if (deepstack) {
            cur = ggml_concat(ctx0, cur, deepstack, 0);
        }

        finish(cur);
    }
};

std::unique_ptr<llm_graph_context> llama_model_clip::build_arch_graph(const llm_graph_params & params) const {
    const llama_media_input * mi = params.ubatch.media;
    const auto ctx_type = params.cparams.ctx_type;

    // a media of an engine tower (encode or reservation of a tower other than the main one) carries its runtime,
    // as a gen step (the native Qwen3-TTS graphs are built there, see graph_engine)
    if ((mi && mi->clip_sub) || ctx_type != LLAMA_CONTEXT_TYPE_ENCODER || !native_v) {
        return llama_model_clip_build_engine_graph(*this, params);
    }
    switch (vhp.proj) {
        case LLAMA_CLIP_PROJ_GEMMA3:
            return std::make_unique<graph_gemma3>(*this, params);
        case LLAMA_CLIP_PROJ_QWEN2VL:
        case LLAMA_CLIP_PROJ_QWEN25VL:
            return std::make_unique<graph_qwen2vl>(*this, params);
        case LLAMA_CLIP_PROJ_QWEN3VL:
            return std::make_unique<graph_qwen3vl>(*this, params);
        default:
            GGML_ABORT("clip: no native graph for projector type %d", (int) vhp.proj);
    }
}

