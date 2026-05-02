#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"
#include "clip-graph.h"
#include "models/models.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <cinttypes>
#include <limits>
#include <array>
#include <functional>
#include <float.h>

struct clip_logger_state g_logger_state = {clip_log_callback_default, NULL};

//#define CLIP_DEBUG_FUNCTIONS

#ifdef CLIP_DEBUG_FUNCTIONS
static void clip_image_write_image_to_ppm(const clip_image_u8& img, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }

    // PPM header: P6 format, width, height, and max color value
    file << "P6\n" << img.nx << " " << img.ny << "\n255\n";

    // Write pixel data
    for (size_t i = 0; i < img.buf.size(); i += 3) {
        // PPM expects binary data in RGB format, which matches our image buffer
        file.write(reinterpret_cast<const char*>(&img.buf[i]), 3);
    }

    file.close();
}

static void clip_image_save_to_bmp(const clip_image_u8& img, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }

    int fileSize = 54 + 3 * img.nx * img.ny; // File header + info header + pixel data
    int bytesPerPixel = 3;
    int widthInBytes = img.nx * bytesPerPixel;
    int paddingAmount = (4 - (widthInBytes % 4)) % 4;
    int stride = widthInBytes + paddingAmount;

    // Bitmap file header
    unsigned char fileHeader[14] = {
        'B','M',     // Signature
        0,0,0,0,    // Image file size in bytes
        0,0,0,0,    // Reserved
        54,0,0,0    // Start of pixel array
    };

    // Total file size
    fileSize = 54 + (stride * img.ny);
    fileHeader[2] = (unsigned char)(fileSize);
    fileHeader[3] = (unsigned char)(fileSize >> 8);
    fileHeader[4] = (unsigned char)(fileSize >> 16);
    fileHeader[5] = (unsigned char)(fileSize >> 24);

    // Bitmap information header (BITMAPINFOHEADER)
    unsigned char infoHeader[40] = {
        40,0,0,0,   // Size of this header (40 bytes)
        0,0,0,0,    // Image width
        0,0,0,0,    // Image height
        1,0,        // Number of color planes
        24,0,       // Bits per pixel
        0,0,0,0,    // No compression
        0,0,0,0,    // Image size (can be 0 for no compression)
        0,0,0,0,    // X pixels per meter (not specified)
        0,0,0,0,    // Y pixels per meter (not specified)
        0,0,0,0,    // Total colors (color table not used)
        0,0,0,0     // Important colors (all are important)
    };

    // Width and height in the information header
    infoHeader[4] = (unsigned char)(img.nx);
    infoHeader[5] = (unsigned char)(img.nx >> 8);
    infoHeader[6] = (unsigned char)(img.nx >> 16);
    infoHeader[7] = (unsigned char)(img.nx >> 24);
    infoHeader[8] = (unsigned char)(img.ny);
    infoHeader[9] = (unsigned char)(img.ny >> 8);
    infoHeader[10] = (unsigned char)(img.ny >> 16);
    infoHeader[11] = (unsigned char)(img.ny >> 24);

    // Write file headers
    file.write(reinterpret_cast<char*>(fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<char*>(infoHeader), sizeof(infoHeader));

    // Pixel data
    std::vector<unsigned char> padding(3, 0); // Max padding size to be added to each row
    for (int y = img.ny - 1; y >= 0; --y) { // BMP files are stored bottom-to-top
        for (int x = 0; x < img.nx; ++x) {
            // Each pixel
            size_t pixelIndex = (y * img.nx + x) * 3;
            unsigned char pixel[3] = {
                img.buf[pixelIndex + 2], // BMP stores pixels in BGR format
                img.buf[pixelIndex + 1],
                img.buf[pixelIndex]
            };
            file.write(reinterpret_cast<char*>(pixel), 3);
        }
        // Write padding for the row
        file.write(reinterpret_cast<char*>(padding.data()), paddingAmount);
    }

    file.close();
}

// debug function to convert f32 to u8
static void clip_image_convert_f32_to_u8(const clip_image_f32& src, clip_image_u8& dst) {
    dst.nx = src.nx;
    dst.ny = src.ny;
    dst.buf.resize(3 * src.nx * src.ny);
    for (size_t i = 0; i < src.buf.size(); ++i) {
        dst.buf[i] = static_cast<uint8_t>(std::min(std::max(int(src.buf[i] * 255.0f), 0), 255));
    }
}
#endif


struct clip_ctx {
    clip_model model;

    gguf_context_ptr ctx_gguf;
    ggml_context_ptr ctx_data;

    std::vector<uint8_t> buf_compute_meta;

    std::vector<ggml_backend_t> backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_ptr buf;


    int max_nodes = 8192;
    ggml_backend_sched_ptr sched;
    clip_flash_attn_type flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;
    bool is_allocated = false;

    bool debug_output_embeddings = false;

    clip_ctx(clip_context_params & ctx_params) {
        flash_attn_type = ctx_params.flash_attn_type;
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend_cpu) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        if (ctx_params.use_gpu) {
            auto backend_name = std::getenv("MTMD_BACKEND_DEVICE");
            if (backend_name != nullptr) {
                backend = ggml_backend_init_by_name(backend_name, nullptr);
                if (!backend) {
                    LOG_WRN("%s: Warning: Failed to initialize \"%s\" backend, falling back to default GPU backend\n", __func__, backend_name);
                }
            }
            if (!backend) {
                backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
                backend = backend ? backend : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
            }
        }

        if (backend) {
            LOG_INF("%s: CLIP using %s backend\n", __func__, ggml_backend_name(backend));
            backend_ptrs.push_back(backend);
            backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
        } else {
            backend = backend_cpu;
            LOG_INF("%s: CLIP using CPU backend\n", __func__);
        }

        if (ctx_params.image_min_tokens > 0) {
            model.hparams.custom_image_min_tokens = ctx_params.image_min_tokens;
        }
        if (ctx_params.image_max_tokens > 0) {
            model.hparams.custom_image_max_tokens = ctx_params.image_max_tokens;
        }

        backend_ptrs.push_back(backend_cpu);
        backend_buft.push_back(ggml_backend_get_default_buffer_type(backend_cpu));

        sched.reset(
            ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), 8192, false, true)
        );

        if (ctx_params.cb_eval != nullptr) {
            ggml_backend_sched_set_eval_callback(sched.get(), ctx_params.cb_eval, ctx_params.cb_eval_user_data);
        }

        debug_output_embeddings = std::getenv("MTMD_DEBUG_EMBEDDINGS") != nullptr;
    }

    ~clip_ctx() {
        ggml_backend_free(backend);
        if (backend != backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }

    // this function is added so that we don't change too much of the existing code
    projector_type proj_type() const {
        return model.proj_type;
    }
};

//
// clip_graph
//

clip_graph::clip_graph(clip_ctx * ctx, const clip_image_f32 & img) :
        model(ctx->model),
        hparams(model.hparams),
        proj_type(ctx->proj_type()),
        img(img),
        patch_size(hparams.patch_size),
        n_patches_x(img.nx / patch_size),
        n_patches_y(img.ny / patch_size),
        n_patches(n_patches_x * n_patches_y),
        n_embd(hparams.n_embd),
        n_head(hparams.n_head),
        d_head(n_embd / n_head),
        n_layer(hparams.n_layer),
        n_mmproj_embd(clip_n_mmproj_embd(ctx)),
        eps(hparams.eps),
        kq_scale(1.0f / sqrtf((float)d_head)),
        flash_attn_type(ctx->flash_attn_type) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    ctx0_ptr.reset(ggml_init(params));
    ctx0 = ctx0_ptr.get();
    gf = ggml_new_graph_custom(ctx0, ctx->max_nodes, false);
}

ggml_tensor * clip_graph::build_mm(ggml_tensor * w, ggml_tensor * x) const {
    return ggml_mul_mat(ctx0, w, x);
}

void clip_graph::cb(ggml_tensor * cur, const char * name, int il) const {
    if (il >= 0) {
        ggml_format_name(cur, "%s-%d", name, il);
    } else {
        ggml_set_name(cur, name);
    }
}

// siglip2 naflex
ggml_tensor * clip_graph::resize_position_embeddings(uint32_t interpolation_mode) {
    ggml_tensor * pos_embd = model.position_embeddings;
    const int height       = img.ny / patch_size;
    const int width        = img.nx / patch_size;
    const uint32_t mode    = interpolation_mode;
    const int n_per_side   = (int)std::sqrt(pos_embd->ne[1]);

    GGML_ASSERT(pos_embd);

    if (height == n_per_side && width == n_per_side) {
        return pos_embd;
    }

    pos_embd = ggml_reshape_3d(ctx0, pos_embd, n_embd, n_per_side, n_per_side);  // -> (n_embd, n_per_side, n_per_side)
    pos_embd = ggml_permute(ctx0, pos_embd, 2, 0, 1, 3);                         // -> (n_per_side, n_per_side, n_embd)
    pos_embd = ggml_interpolate(ctx0, pos_embd, width, height, n_embd, 1, mode); // -> (width, height, n_embd)
    pos_embd = ggml_permute(ctx0, pos_embd, 1, 2, 0, 3);                         // -> (n_embd, width, height)
    pos_embd = ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);             // -> (n_embd, width * height)

    return pos_embd;
}

// build vision transformer (ViT) cgraph
// this function should cover most of the models
// if your model has specific features, you should probably duplicate this function
ggml_tensor * clip_graph::build_vit(
            ggml_tensor * inp,
            int64_t n_pos,
            norm_type norm_t,
            ffn_op_type ffn_t,
            ggml_tensor * learned_pos_embd,
            std::function<ggml_tensor *(ggml_tensor *, const clip_layer &)> add_pos
        ) {
    if (learned_pos_embd) {
        inp = ggml_add(ctx0, inp, learned_pos_embd);
        cb(inp, "pos_embed", -1);
    }

    ggml_tensor * inpL = inp;

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
        cb(inpL, "pre_ln", -1);
    }

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        auto & layer = model.layers[il];
        ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

        // layernorm1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cb(cur, "layer_inp_normed", il);

        // self-attention
        {
            ggml_tensor * Qcur = nullptr;
            ggml_tensor * Kcur = nullptr;
            ggml_tensor * Vcur = nullptr;
            if (layer.qkv_w != nullptr) {
                // fused qkv
                cur = build_mm(layer.qkv_w, cur);
                if (layer.qkv_b != nullptr) {
                    cur = ggml_add(ctx0, cur, layer.qkv_b);
                }

                Qcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ 0);

                Kcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ ggml_row_size(cur->type, n_embd));

                Vcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ ggml_row_size(cur->type, 2 * n_embd));

                if (layer.q_norm) {
                    GGML_ASSERT(layer.q_norm->ne[0] == Qcur->ne[0]);
                    Qcur = build_norm(Qcur, layer.q_norm, NULL, norm_t, eps, il);
                    cb(Qcur, "Qcur_norm", il);
                }

                if (layer.k_norm) {
                    GGML_ASSERT(layer.k_norm->ne[0] == Kcur->ne[0]);
                    Kcur = build_norm(Kcur, layer.k_norm, NULL, norm_t, eps, il);
                    cb(Kcur, "Kcur_norm", il);
                }

            } else {
                // separate q, k, v
                Qcur = build_mm(layer.q_w, cur);
                if (layer.q_b) {
                    Qcur = ggml_add(ctx0, Qcur, layer.q_b);
                }

                Kcur = build_mm(layer.k_w, cur);
                if (layer.k_b) {
                    Kcur = ggml_add(ctx0, Kcur, layer.k_b);
                }

                Vcur = build_mm(layer.v_w, cur);
                if (layer.v_b) {
                    Vcur = ggml_add(ctx0, Vcur, layer.v_b);
                }

                // if true, norm must be applied after reshaping to (d_head, n_head, n_pos)
                bool norm_per_head = layer.q_norm && layer.q_norm->ne[0] == d_head;

                if (!norm_per_head) {
                    if (layer.q_norm) {
                        Qcur = build_norm(Qcur, layer.q_norm, NULL, norm_t, eps, il);
                        cb(Qcur, "Qcur_norm", il);
                    }
                    if (layer.k_norm) {
                        Kcur = build_norm(Kcur, layer.k_norm, NULL, norm_t, eps, il);
                        cb(Kcur, "Kcur_norm", il);
                    }
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
                Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
                Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

                if (norm_per_head) {
                    if (layer.q_norm) {
                        Qcur = build_norm(Qcur, layer.q_norm, NULL, norm_t, eps, il);
                        cb(Qcur, "Qcur_norm_per_head", il);
                    }
                    if (layer.k_norm) {
                        Kcur = build_norm(Kcur, layer.k_norm, NULL, norm_t, eps, il);
                        cb(Kcur, "Kcur_norm_per_head", il);
                    }
                }
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (add_pos) {
                Qcur = add_pos(Qcur, layer);
                Kcur = add_pos(Kcur, layer);
                cb(Qcur, "Qcur_pos", il);
                cb(Kcur, "Kcur_pos", il);
            }

            if (proj_type == PROJECTOR_TYPE_GEMMA4V) {
                Vcur = ggml_rms_norm(ctx0, Vcur, eps);
                cb(Vcur, "Vcur_normed", il);
            }

            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        if (layer.ls_1_w) {
            cur = ggml_mul(ctx0, cur, layer.ls_1_w);
            cb(cur, "attn_out_scaled", il);
        }

        if (layer.attn_post_norm_w) {
            cur = build_norm(cur, layer.attn_post_norm_w, nullptr, norm_t, eps, il);
            cb(cur, "attn_post_normed", il);
        }

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        cb(cur, "ffn_inp", il);

        // layernorm2 (pre-ffn norm)
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // ffn
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            ffn_t, il);

        cb(cur, "ffn_out", il);

        if (layer.ff_post_norm_w) {
            cur = build_norm(cur, layer.ff_post_norm_w, nullptr, norm_t, eps, il);
            cb(cur, "ffn_post_normed", il);
        }

        if (layer.ls_2_w) {
            cur = ggml_mul(ctx0, cur, layer.ls_2_w);
            cb(cur, "ffn_out_scaled", il);
        }

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        if (layer.ls_out_w) {
            cur = ggml_mul(ctx0, cur, layer.ls_out_w);
            cb(cur, "layer_out_scaled", il);
        }

        inpL = cur;
    }

    if (model.audio_has_avgpool()) {
        ggml_tensor * cur = inpL;
        cur = ggml_transpose(ctx0, cur);
        cur = ggml_cont(ctx0, cur);
        cur = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, 2, 2, 0);
        cur = ggml_transpose(ctx0, cur);
        cur = ggml_cont(ctx0, cur);
        inpL = cur;
    }

    // post-layernorm
    if (model.post_ln_w) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, -1);
    }
    return inpL;
}

// build the input after conv2d (inp_raw --> patches)
// returns tensor with shape [n_embd, n_patches]
ggml_tensor * clip_graph::build_inp() {
    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    inp = ggml_reshape_2d(ctx0, inp, n_patches, n_embd);
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    if (model.patch_bias) {
        inp = ggml_add(ctx0, inp, model.patch_bias);
        cb(inp, "patch_bias", -1);
    }
    return inp;
}

ggml_tensor * clip_graph::build_inp_raw(int channels) {
    ggml_tensor * inp_raw = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, img.nx, img.ny, channels);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);
    return inp_raw;
}

ggml_tensor * clip_graph::build_norm(
        ggml_tensor * cur,
        ggml_tensor * mw,
        ggml_tensor * mb,
        norm_type type,
        float norm_eps,
        int il) const {

    cur = type == NORM_TYPE_RMS
        ? ggml_rms_norm(ctx0, cur, norm_eps)
        : ggml_norm(ctx0, cur, norm_eps);

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        cb(cur, "norm_w", il);
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
        cb(cur, "norm_b", il);
    }

    return cur;
}

ggml_tensor * clip_graph::build_ffn(
        ggml_tensor * cur,
        ggml_tensor * up,
        ggml_tensor * up_b,
        ggml_tensor * gate,
        ggml_tensor * gate_b,
        ggml_tensor * down,
        ggml_tensor * down_b,
        ffn_op_type type_op,
        int il) const {

    ggml_tensor * tmp = up ? build_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (gate) {
        cur = build_mm(gate, cur);
        cb(cur, "ffn_gate", il);

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }
    } else {
        cur = tmp;
    }

    // we only support parallel ffn for now
    switch (type_op) {
        case FFN_SILU:
            if (gate) {
                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case FFN_GELU:
            if (gate) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
            } break;
        case FFN_GELU_ERF:
            if (gate) {
                cur = ggml_geglu_erf_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu_erf", il);
            } else {
                cur = ggml_gelu_erf(ctx0, cur);
                cb(cur, "ffn_gelu_erf", il);
            } break;
        case FFN_GELU_QUICK:
            if (gate) {
                cur = ggml_geglu_quick_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu_quick", il);
            } else {
                cur = ggml_gelu_quick(ctx0, cur);
                cb(cur, "ffn_gelu_quick", il);
            } break;
        case FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_relu_sqr", il);
            } break;
    }

    if (down) {
        cur = build_mm(down, cur);
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    return cur;
}

ggml_tensor * clip_graph::build_attn(
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_mask,
        float kq_scale,
        int il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    //cb(q, "q", il);

    ggml_tensor * k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
    //cb(k, "k", il);

    ggml_tensor * cur;

    if (flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
        ggml_tensor * v = ggml_permute(ctx0, v_cur, 0, 2, 1, 3);

        k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        v = ggml_cast(ctx0, v, GGML_TYPE_F16);

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

    } else {
        ggml_tensor * v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
        v = ggml_cont(ctx0, v);

        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        // F32 may not needed for vision encoders?
        // ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
    }

    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_mm(wo, cur);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

// implementation of the 2D RoPE without adding a new op in ggml
// this is not efficient (use double the memory), but works on all backends
// TODO: there was a more efficient which relies on ggml_view and ggml_rope_ext_inplace, but the rope inplace does not work well with non-contiguous tensors ; we should fix that and revert back to the original implementation in https://github.com/ggml-org/llama.cpp/pull/13065
ggml_tensor * clip_graph::build_rope_2d(
    ggml_context * ctx0,
    ggml_tensor * cur,
    ggml_tensor * pos_a, // first half
    ggml_tensor * pos_b, // second half
    const float freq_base,
    const bool interleave_freq
) {
    const int64_t n_dim  = cur->ne[0];
    const int64_t n_head = cur->ne[1];
    const int64_t n_pos  = cur->ne[2];

    // for example, if we have cur tensor of shape (n_dim=8, n_head, n_pos)
    // we will have a list of 4 inv_freq: 1e-0, 1e-1, 1e-2, 1e-3
    // first half of cur will use 1e-0, 1e-2 (even)
    // second half of cur will use 1e-1, 1e-3 (odd)
    // the trick here is to rotate just half of n_dim, so inv_freq will automatically be even
    //  ^ don't ask me why, it's math! -2(2i) / n_dim == -2i / (n_dim/2)
    // then for the second half, we use freq_scale to shift the inv_freq
    //  ^ why? replace (2i) with (2i+1) in the above equation
    const float freq_scale_odd = interleave_freq
                                ? std::pow(freq_base, (float)-2/n_dim)
                                : 1.0;

    // first half
    ggml_tensor * first;
    {
        first = ggml_view_3d(ctx0, cur,
            n_dim/2, n_head, n_pos,
            cur->nb[1],
            cur->nb[2],
            0);
        first = ggml_rope_ext(
            ctx0,
            first,
            pos_a,      // positions
            nullptr,    // freq factors
            n_dim/2,    // n_dims
            0, 0, freq_base,
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f
        );
    }

    // second half
    ggml_tensor * second;
    {
        second = ggml_view_3d(ctx0, cur,
            n_dim/2, n_head, n_pos,
            cur->nb[1],
            cur->nb[2],
            n_dim/2 * ggml_element_size(cur));
        second = ggml_rope_ext(
            ctx0,
            second,
            pos_b,      // positions
            nullptr,    // freq factors
            n_dim/2,    // n_dims
            0, 0, freq_base,
            freq_scale_odd,
            0.0f, 1.0f, 0.0f, 0.0f
        );
    }

    cur = ggml_concat(ctx0, first, second, 0);
    return cur;
}

// Generic function to stack frames for audio processing
// Abstracts out the StackAudioFrames logic used by ultravox
ggml_tensor * clip_graph::build_stack(ggml_tensor * cur, int32_t stack_factor, int32_t n_embed) {
    if (stack_factor <= 1) {
        return cur;
    }

    int64_t total_elements = ggml_nelements(cur);
    int64_t stride = n_embed * stack_factor;

    // Calculate padded length
    int64_t padded_len = GGML_PAD(total_elements, stride);
    int64_t pad = padded_len - total_elements;

    if (pad > 0) {
        // Pad the tensor to make it divisible by stride
        cur = ggml_view_1d(ctx0, cur, total_elements, 0);
        cur = ggml_pad(ctx0, cur, pad, 0, 0, 0);
    }

    // Reshape to [stride, padded_len / stride]
    cur = ggml_view_2d(ctx0, cur, stride, padded_len / stride,
                        ggml_row_size(cur->type, stride), 0);
    return cur;
}

// aka pixel_shuffle / pixel_unshuffle / patch_merger (Kimi-VL)
// support dynamic resolution
ggml_tensor * clip_graph::build_patch_merge_permute(ggml_tensor * cur, int scale_factor) {
    GGML_ASSERT(scale_factor > 1);

    const int n_embd = cur->ne[0];
    int width  = img.nx / patch_size;
    int height = img.ny / patch_size;

    // pad width and height to factor
    const int64_t pad_width  = CLIP_ALIGN(width,  scale_factor) - width;
    const int64_t pad_height = CLIP_ALIGN(height, scale_factor) - height;
    cur = ggml_reshape_3d(ctx0, cur, n_embd, width, height);
    if (pad_width || pad_height) {
        cur     = ggml_pad(ctx0, cur, 0, pad_width, pad_height, 0);
        width  += pad_width;
        height += pad_height;
    }

    // unshuffle h
    cur = ggml_reshape_3d(ctx0, cur, n_embd * scale_factor, width / scale_factor, height);
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);

    // unshuffle w
    cur = ggml_cont_3d(ctx0, cur, n_embd * scale_factor * scale_factor, height / scale_factor, width / scale_factor);
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);

    cur = ggml_cont_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
    cb(cur, "pixel_shuffle", -1);

    return cur;
}

static ggml_cgraph * clip_image_build_graph(clip_ctx * ctx, const clip_image_f32_batch & imgs) {
    GGML_ASSERT(imgs.entries.size() == 1 && "n_batch > 1 is not supported");

    const clip_image_f32 & img = *imgs.entries[0];
    std::unique_ptr<clip_graph> builder;

    switch (ctx->proj_type()) {
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_IDEFICS3:
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_JANUS_PRO:
        case PROJECTOR_TYPE_PHI4:
            {
                builder = std::make_unique<clip_graph_siglip>(ctx, img);
            } break;
        case PROJECTOR_TYPE_GEMMA3NV:
            {
                builder = std::make_unique<clip_graph_mobilenetv5>(ctx, img);
            } break;
        case PROJECTOR_TYPE_GEMMA4V:
            {
                builder = std::make_unique<clip_graph_gemma4v>(ctx, img);
            } break;
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_LIGHTONOCR:
            {
                builder = std::make_unique<clip_graph_pixtral>(ctx, img);
            } break;
        case PROJECTOR_TYPE_DOTS_OCR:
            {
                builder = std::make_unique<clip_graph_dotsocr>(ctx, img);
            } break;
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
            {
                builder = std::make_unique<clip_graph_qwen2vl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_QWEN3VL:
            {
                builder = std::make_unique<clip_graph_qwen3vl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_STEP3VL:
            {
                builder = std::make_unique<clip_graph_step3vl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_MINICPMV:
            {
                builder = std::make_unique<clip_graph_minicpmv>(ctx, img);
            } break;
        case PROJECTOR_TYPE_INTERNVL:
            {
                builder = std::make_unique<clip_graph_internvl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_NEMOTRON_V2_VL:
            {
                builder = std::make_unique<clip_graph_nemotron_v2_vl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_LLAMA4:
            {
                builder = std::make_unique<clip_graph_llama4>(ctx, img);
            } break;
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_QWEN2A:
        case PROJECTOR_TYPE_GLMA:
        case PROJECTOR_TYPE_MERALION:
        case PROJECTOR_TYPE_MUSIC_FLAMINGO:
            {
                builder = std::make_unique<clip_graph_whisper_enc>(ctx, img);
            } break;
        case PROJECTOR_TYPE_KIMIVL:
            {
                builder = std::make_unique<clip_graph_kimivl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_PADDLEOCR:
            {
                builder = std::make_unique<clip_graph_paddleocr>(ctx, img);
            } break;
        case PROJECTOR_TYPE_KIMIK25:
            {
                builder = std::make_unique<clip_graph_kimik25>(ctx, img);
            } break;
        case PROJECTOR_TYPE_COGVLM:
            {
                builder = std::make_unique<clip_graph_cogvlm>(ctx, img);
            } break;
        case PROJECTOR_TYPE_HUNYUANOCR:
        case PROJECTOR_TYPE_HUNYUANVL:
            {
                builder = std::make_unique<clip_graph_hunyuanocr>(ctx, img);
            } break;
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
        case PROJECTOR_TYPE_GLM_EDGE:
            {
                builder = std::make_unique<clip_graph_llava>(ctx, img);
            } break;
        case PROJECTOR_TYPE_DEEPSEEKOCR:
            {
                builder = std::make_unique<clip_graph_deepseekocr>(ctx, img);
            } break;
        case PROJECTOR_TYPE_LFM2A:
            {
                builder = std::make_unique<clip_graph_conformer>(ctx, img);
            } break;
        case PROJECTOR_TYPE_GEMMA4A:
            {
                builder = std::make_unique<clip_graph_gemma4a>(ctx, img);
            } break;
        case PROJECTOR_TYPE_GRANITE_SPEECH:
            {
                builder = std::make_unique<clip_graph_granite_speech>(ctx, img);
            } break;
        case PROJECTOR_TYPE_GLM4V:
            {
                builder = std::make_unique<clip_graph_glm4v>(ctx, img);
            } break;
        case PROJECTOR_TYPE_QWEN3A:
            {
                builder = std::make_unique<clip_graph_qwen3a>(ctx, img);
            } break;
        case PROJECTOR_TYPE_YOUTUVL:
            {
                builder = std::make_unique<clip_graph_youtuvl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_YASA2:
            {
                builder = std::make_unique<clip_graph_yasa2>(ctx, img);
            } break;
        default:
            GGML_ABORT("missing cgraph builder");
    }

    return builder->build();
}

//
// clip_model_loader
//

struct clip_model_loader {
    ggml_context_ptr ctx_meta;
    gguf_context_ptr ctx_gguf;

    std::string fname;

    size_t model_size = 0; // in bytes

    bool has_vision = false;
    bool has_audio  = false;

    // TODO @ngxson : we should not pass clip_ctx here, it should be clip_model
    clip_model_loader(const char * fname) : fname(fname) {
        struct ggml_context * meta = nullptr;

        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };

        ctx_gguf = gguf_context_ptr(gguf_init_from_file(fname, params));
        if (!ctx_gguf.get()) {
            throw std::runtime_error(string_format("%s: failed to load CLIP model from %s. Does this file exist?\n", __func__, fname));
        }

        ctx_meta.reset(meta);

        const int n_tensors = gguf_get_n_tensors(ctx_gguf.get());

        // print gguf info
        {
            std::string name;
            get_string(KEY_NAME, name, false);
            std::string description;
            get_string(KEY_DESCRIPTION, description, false);
            LOG_INF("%s: model name:   %s\n",  __func__, name.c_str());
            LOG_INF("%s: description:  %s\n",  __func__, description.c_str());
            LOG_INF("%s: GGUF version: %d\n",  __func__, gguf_get_version(ctx_gguf.get()));
            LOG_INF("%s: alignment:    %zu\n", __func__, gguf_get_alignment(ctx_gguf.get()));
            LOG_INF("%s: n_tensors:    %d\n",  __func__, n_tensors);
            LOG_INF("%s: n_kv:         %d\n",  __func__, (int)gguf_get_n_kv(ctx_gguf.get()));
            LOG_INF("\n");
        }

        // modalities
        {
            get_bool(KEY_HAS_VISION_ENC, has_vision, false);
            get_bool(KEY_HAS_AUDIO_ENC,  has_audio,  false);

            if (has_vision) {
                LOG_INF("%s: has vision encoder\n", __func__);
            }
            if (has_audio) {
                LOG_INF("%s: has audio encoder\n", __func__);
            }
        }

        // tensors
        {
            for (int i = 0; i < n_tensors; ++i) {
                const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
                const size_t offset = gguf_get_tensor_offset(ctx_gguf.get(), i);
                enum ggml_type type = gguf_get_tensor_type(ctx_gguf.get(), i);
                ggml_tensor * cur = ggml_get_tensor(meta, name);
                size_t tensor_size = ggml_nbytes(cur);
                model_size += tensor_size;
                LOG_DBG("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                    __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
            }
        }
    }

    void load_hparams(clip_model & model, clip_modality modality) {
        auto & hparams = model.hparams;
        std::string log_ffn_op; // for logging

        // sanity check
        if (modality == CLIP_MODALITY_VISION) {
            GGML_ASSERT(has_vision);
        } else if (modality == CLIP_MODALITY_AUDIO) {
            GGML_ASSERT(has_audio);
        }
        model.modality = modality;


        // projector type
        std::string proj_type;
        {
            // default key
            get_string(KEY_PROJ_TYPE, proj_type, false);

            // for models with mixed modalities
            if (proj_type.empty()) {
                if (modality == CLIP_MODALITY_VISION) {
                    get_string(KEY_VISION_PROJ_TYPE, proj_type, false);
                } else if (modality == CLIP_MODALITY_AUDIO) {
                    get_string(KEY_AUDIO_PROJ_TYPE, proj_type, false);
                } else {
                    GGML_ABORT("unknown modality");
                }
            }

            model.proj_type = clip_projector_type_from_string(proj_type);

            if (model.proj_type == PROJECTOR_TYPE_UNKNOWN) {
                throw std::runtime_error(string_format("%s: unknown projector type: %s\n", __func__, proj_type.c_str()));
            }

            // correct arch for multimodal models (legacy method)
            if (model.proj_type == PROJECTOR_TYPE_QWEN25O) {
                model.proj_type = modality == CLIP_MODALITY_VISION
                                    ? PROJECTOR_TYPE_QWEN25VL
                                    : PROJECTOR_TYPE_QWEN2A;
            }
        }

        const bool is_vision = model.modality == CLIP_MODALITY_VISION;
        const bool is_audio  = model.modality == CLIP_MODALITY_AUDIO;

        // other hparams
        {
            const char * prefix = is_vision ? "vision" : "audio";
            get_u32(string_format(KEY_N_EMBD,         prefix), hparams.n_embd);
            get_u32(string_format(KEY_N_HEAD,         prefix), hparams.n_head);
            get_u32(string_format(KEY_N_FF,           prefix), hparams.n_ff);
            get_u32(string_format(KEY_N_BLOCK,        prefix), hparams.n_layer);
            get_u32(string_format(KEY_PROJ_DIM,       prefix), hparams.projection_dim);
            get_f32(string_format(KEY_LAYER_NORM_EPS, prefix), hparams.eps, false);

            if (is_vision) {
                get_u32(KEY_IMAGE_SIZE, hparams.image_size);
                get_u32(KEY_PATCH_SIZE, hparams.patch_size);
                get_i32(KEY_MINICPMV_VERSION, hparams.minicpmv_version, false); // legacy
                get_u32(KEY_MINICPMV_QUERY_NUM, hparams.minicpmv_query_num, false);
                if (hparams.minicpmv_query_num == 0) {
                    // Fallback to hardcoded values for legacy models
                    if (hparams.minicpmv_version == 3) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 4) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 5) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 6) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 100045) {
                        hparams.minicpmv_query_num = 64;
                    } else {
                        hparams.minicpmv_query_num = 96;
                    }
                }
            } else if (is_audio) {
                get_u32(KEY_A_NUM_MEL_BINS, hparams.n_mel_bins);
                // some hparams are unused, but still need to set to avoid issues
                hparams.image_size = 0;
                hparams.patch_size = 1;

            } else {
                GGML_ASSERT(false && "unknown modality");
            }

            // for pinpoints, we need to convert it into a list of resolution candidates
            {
                std::vector<int> pinpoints;
                get_arr_int(KEY_IMAGE_GRID_PINPOINTS, pinpoints, false);
                if (!pinpoints.empty()) {
                    for (size_t i = 0; i < pinpoints.size(); i += 2) {
                        hparams.image_res_candidates.push_back({
                            pinpoints[i],
                            pinpoints[i+1],
                        });
                    }
                }
            }

            // default warmup value
            hparams.warmup_image_size = hparams.image_size;

            {
                bool use_gelu = false;
                bool use_silu = false;
                get_bool(KEY_USE_GELU, use_gelu, false);
                get_bool(KEY_USE_SILU, use_silu, false);
                if (use_gelu && use_silu) {
                    throw std::runtime_error(string_format("%s: both use_gelu and use_silu are set to true\n", __func__));
                }
                if (use_gelu) {
                    hparams.ffn_op = FFN_GELU;
                    log_ffn_op = "gelu";
                } else if (use_silu) {
                    hparams.ffn_op = FFN_SILU;
                    log_ffn_op = "silu";
                } else {
                    hparams.ffn_op = FFN_GELU_QUICK;
                    log_ffn_op = "gelu_quick";
                }
            }

            {
                std::string mm_patch_merge_type;
                get_string(KEY_MM_PATCH_MERGE_TYPE, mm_patch_merge_type, false);
                if (mm_patch_merge_type == "spatial_unpad") {
                    hparams.mm_patch_merge_type = PATCH_MERGE_SPATIAL_UNPAD;
                }
            }

            if (is_vision) {
                int idx_mean = gguf_find_key(ctx_gguf.get(), KEY_IMAGE_MEAN);
                int idx_std  = gguf_find_key(ctx_gguf.get(), KEY_IMAGE_STD);
                GGML_ASSERT(idx_mean >= 0 && "image_mean not found");
                GGML_ASSERT(idx_std >= 0  && "image_std not found");
                const float * mean_data = (const float *) gguf_get_arr_data(ctx_gguf.get(), idx_mean);
                const float * std_data  = (const float *) gguf_get_arr_data(ctx_gguf.get(), idx_std);
                for (int i = 0; i < 3; ++i) {
                    hparams.image_mean[i] = mean_data[i];
                    hparams.image_std[i]  = std_data[i];
                }
            }

            // Load the vision feature layer indices if they are explicitly provided;
            // if multiple vision feature layers are present, the values will be concatenated
            // to form the final visual features.
            // NOTE: gguf conversions should standardize the values of the vision feature layer to
            // be non-negative, since we use -1 to mark values as unset here.
            std::vector<int> vision_feature_layer;
            get_arr_int(KEY_FEATURE_LAYER, vision_feature_layer, false);
            // convert std::vector to std::unordered_set
            for (auto & layer : vision_feature_layer) {
                hparams.vision_feature_layer.insert(layer);
            }

            // model-specific params
            switch (model.proj_type) {
                case PROJECTOR_TYPE_MLP:
                case PROJECTOR_TYPE_MLP_NORM:
                case PROJECTOR_TYPE_LDP:
                case PROJECTOR_TYPE_LDPV2:
                case PROJECTOR_TYPE_COGVLM:
                    {
                        hparams.has_llava_projector = model.proj_type != PROJECTOR_TYPE_COGVLM;
                        hparams.image_pad_color     = {122, 116, 104};
                        if (!hparams.image_res_candidates.empty()) {
                            hparams.image_resize_pad  = true;
                            hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        } else {
                            // llava-1.6 default params
                            hparams.image_pad_ov         = false;
                            hparams.image_pad_rf         = true;
                            hparams.image_pad_color_rf   = {122, 116, 104};
                            hparams.image_resize_algo_rf = RESIZE_ALGO_BICUBIC;
                            hparams.image_resize_algo_ov = RESIZE_ALGO_BILINEAR;
                        }
                    } break;
                case PROJECTOR_TYPE_GLM_EDGE:
                    {
                        hparams.image_resize_pad  = true;
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                    } break;
                case PROJECTOR_TYPE_MINICPMV:
                    {
                        // use default llava-uhd preprocessing params
                        if (hparams.minicpmv_version == 0) {
                            hparams.minicpmv_version = 2; // default to 2 if not set
                        }
                    } break;
                case PROJECTOR_TYPE_INTERNVL:
                    {
                        // use default llava-uhd preprocessing params
                        // older version of internvl doesn't have min/max tiles, we need to provide default values for them to avoid issues
                        hparams.preproc_min_tiles = 1;
                        hparams.preproc_max_tiles = 12;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        get_u32(KEY_PREPROC_MIN_TILES, hparams.preproc_min_tiles, false);
                        get_u32(KEY_PREPROC_MAX_TILES, hparams.preproc_max_tiles, false);
                        GGML_ASSERT(hparams.preproc_min_tiles <= hparams.preproc_max_tiles && hparams.preproc_max_tiles < INT32_MAX);
                        set_internvl_dhr_res_candidates(model);
                    } break;
                case PROJECTOR_TYPE_NEMOTRON_V2_VL:
                    {
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                    } break;
                case PROJECTOR_TYPE_IDEFICS3:
                    {
                        // use default llava-uhd preprocessing params
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        get_u32(KEY_PREPROC_IMAGE_SIZE, hparams.image_longest_edge, false);
                    } break;
                case PROJECTOR_TYPE_LFM2:
                    {
                        hparams.image_resize_algo    = RESIZE_ALGO_BILINEAR;
                        hparams.image_resize_algo_rf = RESIZE_ALGO_BILINEAR;
                        hparams.image_resize_algo_ov = RESIZE_ALGO_BILINEAR;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        // ref: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B/blob/main/processor_config.json
                        hparams.set_limit_image_tokens(64, 256);
                    } break;
                case PROJECTOR_TYPE_PHI4:
                    {
                        hparams.n_merge = 1;
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);
                        hparams.set_warmup_n_tokens(16*16);
                    } break;
                case PROJECTOR_TYPE_PIXTRAL:
                    {
                        // ref: https://huggingface.co/mistral-community/pixtral-12b/blob/main/preprocessor_config.json
                        // TODO: verify the image_min_tokens
                        hparams.n_merge = 1; // the original pixtral does not use patch merging
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        hparams.set_limit_image_tokens(8, 1024);
                        hparams.set_warmup_n_tokens(256); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_LIGHTONOCR:
                    {
                        hparams.n_merge = 1;
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        hparams.image_longest_edge = hparams.image_size;
                        get_u32(KEY_PREPROC_IMAGE_SIZE, hparams.image_longest_edge, false);
                        hparams.set_warmup_n_tokens(256); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_DOTS_OCR:
                    {
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge);
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_KIMIVL:
                    {
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        // TODO: check kimivl preprocessor for exact values
                        hparams.set_limit_image_tokens(8, 1024);
                        hparams.set_warmup_n_tokens(256); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_KIMIK25:
                    {
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);

                        int min_pixels = 0, max_pixels = 0;
                        get_u32(KEY_IMAGE_MIN_PIXELS, min_pixels, false);
                        get_u32(KEY_IMAGE_MAX_PIXELS, max_pixels, false);
                        if (min_pixels > 0 && max_pixels > 0) {
                            hparams.image_min_pixels = min_pixels;
                            hparams.image_max_pixels = max_pixels;
                            hparams.warmup_image_size = static_cast<int>(std::sqrt(max_pixels));
                        } else {
                            hparams.set_limit_image_tokens(2, 4096);
                        }
                    } break;
                case PROJECTOR_TYPE_GEMMA3:
                    {
                        // default value (used by all model sizes in gemma 3 family)
                        // number of patches for each **side** is reduced by a factor of 4
                        hparams.n_merge = 4;
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        // test model (tinygemma3) has a different value, we optionally read it
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                    } break;

                case PROJECTOR_TYPE_GEMMA4V:
                    {
                        hparams.rope_theta = 100.0f;
                        hparams.n_merge = 3; // pooling_kernel_size
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        // @ngxson : the model performs quite poor with small images, we need to bump minimum image tokens to 40 to avoid that
                        hparams.set_limit_image_tokens(252, 280);
                        hparams.set_warmup_n_tokens(256); // avoid OOM on warmup
                    } break;

                case PROJECTOR_TYPE_GEMMA3NV:
                    {
                        // Gemma3n uses MobileNetV5 which produces 256 tokens (16x16)
                        // Similar configuration to Gemma3
                        hparams.n_merge = 1;  // MobileNetV5 handles resizing internally
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                    } break;
                case PROJECTOR_TYPE_QWEN2VL:
                case PROJECTOR_TYPE_QWEN25VL:
                case PROJECTOR_TYPE_QWEN3VL:
                    {
                        hparams.n_merge = 2; // default value for Qwen 2 and 2.5
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        get_u32(KEY_WIN_ATTN_PATTERN, hparams.n_wa_pattern, model.proj_type == PROJECTOR_TYPE_QWEN25VL); // only 2.5 requires it
                        // ref: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/preprocessor_config.json
                        hparams.set_limit_image_tokens(8, 4096);
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                        const int warn_min_pixels = 1024 * hparams.n_merge * hparams.n_merge * hparams.patch_size * hparams.patch_size;
                        if (hparams.image_min_pixels < warn_min_pixels) {
                            LOG_WRN("%s: Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks\n", __func__);
                            LOG_WRN("%s: if you encounter problems with accuracy, try adding --image-min-tokens 1024\n", __func__);
                            LOG_WRN("%s: more info: https://github.com/ggml-org/llama.cpp/issues/16842\n\n", __func__);
                        }
                    } break;
                case PROJECTOR_TYPE_STEP3VL:
                    {
                        hparams.n_merge = 4; // two stride-2 downsamplers after patching
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PREPROC_IMAGE_SIZE, hparams.image_longest_edge, false);
                        if (hparams.image_longest_edge == 0) {
                            hparams.image_longest_edge = 3024;
                        }
                        hparams.warmup_image_size = hparams.image_size;
                    } break;
                case PROJECTOR_TYPE_YOUTUVL:
                    {
                        hparams.n_merge = 2;
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        hparams.image_resize_pad  = false;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        get_u32(KEY_ATTN_WINDOW_SIZE, hparams.attn_window_size, true);
                        std::vector<int> wa_layer_indexes_vec;
                        get_arr_int(KEY_WIN_ATTN_LAYER_INDEXES, wa_layer_indexes_vec, true);
                        for (auto & layer : wa_layer_indexes_vec) {
                            hparams.wa_layer_indexes.insert(layer);
                        }
                        // support max_height * max_width = 8000 * 8000. 8000/16/2 = 250 image tokens
                        hparams.set_limit_image_tokens(1, 62500);
                        hparams.set_warmup_n_tokens(16*16); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_YASA2:
                    {
                        hparams.ffn_op = FFN_GELU_ERF;
                        log_ffn_op = "gelu_erf";
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;

                        // reka model performs better when using resize_bicubic, which stretches
                        // the image to fit fixed square size
                        hparams.image_resize_pad = false;
                    } break;
                case PROJECTOR_TYPE_GLM4V:
                    {
                        hparams.rope_theta = 10000.0f;
                        hparams.n_merge = 2; // default value for GLM4-V
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        hparams.set_limit_image_tokens(8, 4096);
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_LLAMA4:
                    {
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        set_llava_uhd_res_candidates(model, 3);
                    } break;
                case PROJECTOR_TYPE_ULTRAVOX:
                case PROJECTOR_TYPE_QWEN2A:
                case PROJECTOR_TYPE_QWEN3A:
                case PROJECTOR_TYPE_GLMA:
                case PROJECTOR_TYPE_VOXTRAL:
                case PROJECTOR_TYPE_MERALION:
                case PROJECTOR_TYPE_MUSIC_FLAMINGO:
                    {
                        bool require_stack = model.proj_type == PROJECTOR_TYPE_ULTRAVOX ||
                                             model.proj_type == PROJECTOR_TYPE_VOXTRAL ||
                                             model.proj_type == PROJECTOR_TYPE_MERALION ||
                                             model.proj_type == PROJECTOR_TYPE_GLMA;
                        get_u32(KEY_A_PROJ_STACK_FACTOR, hparams.proj_stack_factor, require_stack);
                        hparams.ffn_op = FFN_GELU_ERF;
                        log_ffn_op = "gelu_erf"; // temporary solution for logging

                        // audio preprocessing params
                        hparams.audio_chunk_len    = 30; // in seconds
                        hparams.audio_sample_rate  = 16000;
                        hparams.audio_n_fft        = 400;
                        hparams.audio_window_len   = 400;
                        hparams.audio_hop_len      = 160;
                    } break;
                case PROJECTOR_TYPE_PADDLEOCR:
                    {
                        hparams.n_merge = 2;
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);

                        hparams.set_warmup_n_tokens(28*28); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_DEEPSEEKOCR:
                    {
                        hparams.patch_size = 16;
                        hparams.image_size = 1024;
                        hparams.warmup_image_size = 1024;
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC_PILLOW;
                        hparams.image_pad_color[0] = hparams.image_mean[0];
                        hparams.image_pad_color[1] = hparams.image_mean[1];
                        hparams.image_pad_color[2] = hparams.image_mean[2];

                        get_u32(KEY_SAM_N_BLOCK, hparams.sam_n_layer, true);
                        get_u32(KEY_SAM_N_HEAD, hparams.sam_n_head, true);
                        get_u32(KEY_SAM_N_EMBD, hparams.sam_n_embd, true);
                        get_u32(KEY_ATTN_WINDOW_SIZE, hparams.attn_window_size, true);
                     } break;
                case PROJECTOR_TYPE_HUNYUANOCR:
                    {
                        hparams.n_merge = 2;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);
                        hparams.set_warmup_n_tokens(28*28);
                    } break;
                case PROJECTOR_TYPE_HUNYUANVL:
                    {
                        hparams.n_merge = 2;
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC_PILLOW;
                        hparams.image_resize_pad = false;
                        hparams.ffn_op = FFN_GELU;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        hparams.set_limit_image_tokens(256, 16384);
                        hparams.set_warmup_n_tokens(32*32);
                    } break;
                case PROJECTOR_TYPE_LFM2A:
                    {
                        // audio preprocessing params
                        hparams.audio_chunk_len        = 1; // in seconds
                        hparams.audio_sample_rate      = 16000;
                        hparams.audio_n_fft            = 512;
                        hparams.audio_window_len       = 400;
                        hparams.audio_hop_len          = 160;
                    } break;
                case PROJECTOR_TYPE_GEMMA4A:
                    {
                        // Gemma4 feature_extraction_gemma4.py:
                        // frame_length_ms=20 -> 320 samples, n_fft=512, hop=10ms -> 160
                        hparams.audio_chunk_len        = 0;  // no fixed-length padding
                        hparams.audio_sample_rate      = 16000;
                        hparams.audio_n_fft            = 512;
                        hparams.audio_window_len       = 320;  // 20ms frame (NOT 25ms/400)
                        hparams.audio_hop_len          = 160;
                    } break;
                case PROJECTOR_TYPE_GRANITE_SPEECH:
                    {
                        hparams.audio_chunk_len        = 0;
                        hparams.audio_sample_rate      = 16000;
                        hparams.audio_n_fft            = 512;
                        hparams.audio_window_len       = 400;
                        hparams.audio_hop_len          = 160;
                        get_u32(KEY_A_CHUNK_SIZE,           hparams.audio_chunk_size);
                        get_u32(KEY_A_CONV_KERNEL_SIZE,     hparams.audio_conv_kernel_size);
                        get_u32(KEY_A_MAX_POS_EMB,          hparams.audio_max_pos_emb);
                        get_u32(KEY_A_PROJ_WINDOW_SIZE,     hparams.audio_proj_window_size);
                        get_u32(KEY_A_PROJ_DOWNSAMPLE_RATE, hparams.audio_proj_downsample_rate);
                        get_u32(KEY_A_PROJ_HEAD_COUNT,      hparams.audio_proj_head_count);
                    } break;
                case PROJECTOR_TYPE_JANUS_PRO:
                    {
                        hparams.image_pad_color   = {127, 127, 127};
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                    } break;
                default:
                    throw std::runtime_error(string_format("%s: unknown vision projector type %s\n", __func__, proj_type.c_str()));
            }

            // sanity check
            {
                if (hparams.image_size < 0) {
                    // note: some models having hparams.image_size == 0, which means the image size is dynamic
                    throw std::runtime_error(string_format("%s: image_size (%d) cannot be negative\n", __func__, hparams.image_size));
                }
                if (hparams.patch_size <= 0) {
                    throw std::runtime_error(string_format("%s: patch_size (%d) must be greater than 0\n", __func__, hparams.patch_size));
                }
                if (hparams.n_embd <= 0) {
                    throw std::runtime_error(string_format("%s: n_embd (%d) must be greater than 0\n", __func__, hparams.n_embd));
                }
                if (hparams.image_max_pixels < hparams.image_min_pixels) {
                    throw std::runtime_error(string_format("%s: image_max_pixels (%d) is less than image_min_pixels (%d)\n", __func__, hparams.image_max_pixels, hparams.image_min_pixels));
                }
            }

            LOG_INF("%s: projector:          %s\n", __func__, proj_type.c_str());
            LOG_INF("%s: n_embd:             %d\n", __func__, hparams.n_embd);
            LOG_INF("%s: n_head:             %d\n", __func__, hparams.n_head);
            LOG_INF("%s: n_ff:               %d\n", __func__, hparams.n_ff);
            LOG_INF("%s: n_layer:            %d\n", __func__, hparams.n_layer);
            LOG_INF("%s: ffn_op:             %s\n", __func__, log_ffn_op.c_str());
            LOG_INF("%s: projection_dim:     %d\n", __func__, hparams.projection_dim);
            if (is_vision) {
                LOG_INF("\n--- vision hparams ---\n");
                LOG_INF("%s: image_size:         %d\n", __func__, hparams.image_size);
                LOG_INF("%s: patch_size:         %d\n", __func__, hparams.patch_size);
                LOG_INF("%s: has_llava_proj:     %d\n", __func__, hparams.has_llava_projector);
                LOG_INF("%s: minicpmv_version:   %d\n", __func__, hparams.minicpmv_version);
                LOG_INF("%s: n_merge:            %d\n", __func__, hparams.n_merge);
                LOG_INF("%s: n_wa_pattern: %d\n", __func__, hparams.n_wa_pattern);
                if (!hparams.wa_layer_indexes.empty()) {
                    LOG_INF("%s: wa_layer_indexes:  ", __func__);
                    for (auto & layer : hparams.wa_layer_indexes) {
                        LOG_INF("%d ", layer);
                    }
                    LOG_INF("\n");
                }
                if (hparams.image_min_pixels > 0) {
                    LOG_INF("%s: image_min_pixels:   %d%s\n", __func__, hparams.image_min_pixels, hparams.custom_image_min_tokens > 0 ? " (custom value)" : "");
                }
                if (hparams.image_max_pixels > 0) {
                    LOG_INF("%s: image_max_pixels:   %d%s\n", __func__, hparams.image_max_pixels, hparams.custom_image_max_tokens > 0 ? " (custom value)" : "");
                }
            } else if (is_audio) {
                LOG_INF("\n--- audio hparams ---\n");
                LOG_INF("%s: n_mel_bins:         %d\n", __func__, hparams.n_mel_bins);
                LOG_INF("%s: proj_stack_factor:  %d\n", __func__, hparams.proj_stack_factor);
                LOG_INF("%s: audio_chunk_len:    %d\n", __func__, hparams.audio_chunk_len);
                LOG_INF("%s: audio_sample_rate:  %d\n", __func__, hparams.audio_sample_rate);
                LOG_INF("%s: audio_n_fft:        %d\n", __func__, hparams.audio_n_fft);
                LOG_INF("%s: audio_window_len:   %d\n", __func__, hparams.audio_window_len);
                LOG_INF("%s: audio_hop_len:      %d\n", __func__, hparams.audio_hop_len);
            }
            LOG_INF("\n");
            LOG_INF("%s: model size:         %.2f MiB\n", __func__, model_size / 1024.0 / 1024.0);
            LOG_INF("%s: metadata size:      %.2f MiB\n", __func__, ggml_get_mem_size(ctx_meta.get()) / 1024.0 / 1024.0);
        }
    }

    void load_tensors(clip_ctx & ctx_clip) {
        auto & model = ctx_clip.model;
        auto & hparams = model.hparams;
        std::map<std::string, size_t> tensor_offset;
        std::vector<ggml_tensor *> tensors_to_load;

        auto fin = std::ifstream(fname, std::ios::binary);
        if (!fin) {
            throw std::runtime_error(string_format("%s: failed to open %s\n", __func__, fname.c_str()));
        }

        // TODO @ngxson : support both audio and video in the future
        const char * prefix = model.modality == CLIP_MODALITY_AUDIO ? "a" : "v";

        // get offsets
        for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
            tensor_offset[name] = gguf_get_data_offset(ctx_gguf.get()) + gguf_get_tensor_offset(ctx_gguf.get(), i);
        }

        // create data context
        struct ggml_init_params params = {
            /*.mem_size =*/ static_cast<size_t>(gguf_get_n_tensors(ctx_gguf.get()) + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ true,
        };
        ctx_clip.ctx_data.reset(ggml_init(params));
        if (!ctx_clip.ctx_data) {
            throw std::runtime_error(string_format("%s: failed to init ggml context\n", __func__));
        }

        // helper function
        std::unordered_set<std::string> loaded_tensor_names;
        auto get_tensor = [&](const std::string & name, bool required = true) {
            // Each tensor should only be loaded once; duplicates indicate a bug
            if (loaded_tensor_names.count(name)) {
                throw std::runtime_error(string_format("%s: tensor already loaded: %s\n", __func__, name.c_str()));
            }
            ggml_tensor * cur = ggml_get_tensor(ctx_meta.get(), name.c_str());
            if (!cur && required) {
                throw std::runtime_error(string_format("%s: unable to find tensor %s\n", __func__, name.c_str()));
            }
            if (cur) {
                tensors_to_load.push_back(cur);
                ggml_tensor * data_tensor = ggml_dup_tensor(ctx_clip.ctx_data.get(), cur);
                ggml_set_name(data_tensor, cur->name);
                loaded_tensor_names.insert(name);
                cur = data_tensor;
            }
            return cur;
        };

        auto get_scalar = [&](const std::string & name, float default_val) {
            auto it = tensor_offset.find(name);
            if (it == tensor_offset.end()) {
                return default_val;
            }
            size_t offset = it->second;
            fin.seekg(offset, std::ios::beg);
            float value;
            fin.read(reinterpret_cast<char*>(&value), sizeof(float));
            return value;
        };

        model.class_embedding = get_tensor(TN_CLASS_EMBD, false);

        model.pre_ln_w = get_tensor(string_format(TN_LN_PRE, prefix, "weight"), false);
        model.pre_ln_b = get_tensor(string_format(TN_LN_PRE, prefix, "bias"),   false);

        model.post_ln_w = get_tensor(string_format(TN_LN_POST, prefix, "weight"), false);
        model.post_ln_b = get_tensor(string_format(TN_LN_POST, prefix, "bias"),   false);

        model.patch_bias = get_tensor(TN_PATCH_BIAS, false);
        model.patch_embeddings_0 = get_tensor(TN_PATCH_EMBD,   false);
        model.patch_embeddings_1 = get_tensor(TN_PATCH_EMBD_1, false);

        model.norm_embd_w = get_tensor(string_format(TN_NORM_EMBD, "weight"), false);
        model.norm_embd_b = get_tensor(string_format(TN_NORM_EMBD, "bias"),   false);

        model.position_embeddings = get_tensor(string_format(TN_POS_EMBD, prefix), false);

        const bool has_standard_layers = (
            model.proj_type != PROJECTOR_TYPE_GEMMA3NV);

        // layers
        const int n_layers_to_load = has_standard_layers ? hparams.n_layer : 0;
        model.layers.resize(n_layers_to_load);
        for (int il = 0; il < n_layers_to_load; ++il) {
            auto & layer = model.layers[il];
            layer.k_w    = get_tensor(string_format(TN_ATTN_K,      prefix, il, "weight"), false);
            layer.q_w    = get_tensor(string_format(TN_ATTN_Q,      prefix, il, "weight"), false);
            layer.v_w    = get_tensor(string_format(TN_ATTN_V,      prefix, il, "weight"), false);
            layer.o_w    = get_tensor(string_format(TN_ATTN_OUTPUT, prefix, il, "weight"));
            layer.qkv_w  = get_tensor(string_format(TN_ATTN_QKV,    prefix, il, "weight"), false);
            layer.k_norm = get_tensor(string_format(TN_ATTN_K_NORM, prefix, il, "weight"), false);
            layer.q_norm = get_tensor(string_format(TN_ATTN_Q_NORM, prefix, il, "weight"), false);
            layer.ln_1_w = get_tensor(string_format(TN_LN_1,        prefix, il, "weight"), false);
            layer.ln_2_w = get_tensor(string_format(TN_LN_2,        prefix, il, "weight"), false);
            layer.ls_1_w        = get_tensor(string_format(TN_LS_1,         prefix, il, "weight"), false); // no bias
            layer.ls_2_w        = get_tensor(string_format(TN_LS_2,         prefix, il, "weight"), false); // no bias
            layer.ls_out_w      = get_tensor(string_format(TN_LS_OUT,        prefix, il, "weight"), false); // no bias
            layer.attn_post_norm_w = get_tensor(string_format(TN_ATTN_POST_NORM, prefix, il, "weight"), false); // no bias
            layer.ff_post_norm_w   = get_tensor(string_format(TN_FFN_POST_NORM,  prefix, il, "weight"), false); // no bias

            layer.k_b    = get_tensor(string_format(TN_ATTN_K,      prefix, il, "bias"), false);
            layer.q_b    = get_tensor(string_format(TN_ATTN_Q,      prefix, il, "bias"), false);
            layer.v_b    = get_tensor(string_format(TN_ATTN_V,      prefix, il, "bias"), false);
            layer.o_b    = get_tensor(string_format(TN_ATTN_OUTPUT, prefix, il, "bias"), false);
            layer.qkv_b  = get_tensor(string_format(TN_ATTN_QKV,    prefix, il, "bias"), false);
            layer.ln_1_b = get_tensor(string_format(TN_LN_1,        prefix, il, "bias"), false);
            layer.ln_2_b = get_tensor(string_format(TN_LN_2,        prefix, il, "bias"), false);

            // ffn
            layer.ff_up_w   = get_tensor(string_format(TN_FFN_UP,   prefix, il, "weight"));
            layer.ff_up_b   = get_tensor(string_format(TN_FFN_UP,   prefix, il, "bias"),   false);
            layer.ff_gate_w = get_tensor(string_format(TN_FFN_GATE, prefix, il, "weight"), false);
            layer.ff_gate_b = get_tensor(string_format(TN_FFN_GATE, prefix, il, "bias"),   false);
            layer.ff_down_w = get_tensor(string_format(TN_FFN_DOWN, prefix, il, "weight"));
            layer.ff_down_b = get_tensor(string_format(TN_FFN_DOWN, prefix, il, "bias"),   false);


            // qwen3vl deepstack layer
            layer.deepstack_norm_w = get_tensor(string_format(TN_DEEPSTACK_NORM, il, "weight"), false);
            layer.deepstack_norm_b = get_tensor(string_format(TN_DEEPSTACK_NORM, il, "bias"), false);
            layer.deepstack_fc1_w  = get_tensor(string_format(TN_DEEPSTACK_FC1,  il, "weight"), false);
            layer.deepstack_fc1_b  = get_tensor(string_format(TN_DEEPSTACK_FC1,  il, "bias"), false);
            layer.deepstack_fc2_w  = get_tensor(string_format(TN_DEEPSTACK_FC2,  il, "weight"), false);
            layer.deepstack_fc2_b  = get_tensor(string_format(TN_DEEPSTACK_FC2,  il, "bias"), false);
            if (layer.has_deepstack()) {
                model.n_deepstack_layers++;
            }

            // some models already exported with legacy (incorrect) naming which is quite messy, let's fix it here
            // note: Qwen model converted from the old surgery script has n_ff = 0, so we cannot use n_ff to check!
            bool is_ffn_swapped = (
                    // only old models need this fix
                    model.proj_type == PROJECTOR_TYPE_MLP
                    || model.proj_type == PROJECTOR_TYPE_MLP_NORM
                    || model.proj_type == PROJECTOR_TYPE_LDP
                    || model.proj_type == PROJECTOR_TYPE_LDPV2
                    || model.proj_type == PROJECTOR_TYPE_QWEN2VL
                    || model.proj_type == PROJECTOR_TYPE_QWEN25VL
                    || model.proj_type == PROJECTOR_TYPE_GLM_EDGE
                    || model.proj_type == PROJECTOR_TYPE_GEMMA3
                    || model.proj_type == PROJECTOR_TYPE_IDEFICS3
                    || model.proj_type == PROJECTOR_TYPE_MINICPMV
                ) && layer.ff_up_w && layer.ff_down_w && layer.ff_down_w->ne[0] == hparams.n_embd;
            if (is_ffn_swapped) {
                // swap up and down weights
                ggml_tensor * tmp = layer.ff_up_w;
                layer.ff_up_w = layer.ff_down_w;
                layer.ff_down_w = tmp;
                // swap up and down biases
                tmp = layer.ff_up_b;
                layer.ff_up_b = layer.ff_down_b;
                layer.ff_down_b = tmp;
                if (il == 0) {
                    LOG_WRN("%s: ffn up/down are swapped\n", __func__);
                }
            }
        }


        switch (model.proj_type) {
            case PROJECTOR_TYPE_MLP:
            case PROJECTOR_TYPE_MLP_NORM:
                {
                    // LLaVA projection
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"), false);
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"), false);
                    // Yi-type llava
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"), false);
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    // missing in Yi-type llava
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"), false);
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);
                    // Yi-type llava
                    model.mm_3_w = get_tensor(string_format(TN_LLAVA_PROJ, 3, "weight"), false);
                    model.mm_3_b = get_tensor(string_format(TN_LLAVA_PROJ, 3, "bias"), false);
                    model.mm_4_w = get_tensor(string_format(TN_LLAVA_PROJ, 4, "weight"), false);
                    model.mm_4_b = get_tensor(string_format(TN_LLAVA_PROJ, 4, "bias"), false);
                    if (model.mm_3_w) {
                        // TODO: this is a hack to support Yi-type llava
                        model.proj_type = PROJECTOR_TYPE_MLP_NORM;
                    }
                    model.image_newline = get_tensor(TN_IMAGE_NEWLINE, false);
                } break;
            case PROJECTOR_TYPE_LDP:
                {
                    // MobileVLM projection
                    model.mm_model_mlp_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_model_mlp_1_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "bias"));
                    model.mm_model_mlp_3_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "weight"));
                    model.mm_model_mlp_3_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "bias"));
                    model.mm_model_block_1_block_0_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "0.weight"));
                    model.mm_model_block_1_block_0_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.weight"));
                    model.mm_model_block_1_block_0_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.bias"));
                    model.mm_model_block_1_block_1_fc1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.weight"));
                    model.mm_model_block_1_block_1_fc1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.bias"));
                    model.mm_model_block_1_block_1_fc2_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.weight"));
                    model.mm_model_block_1_block_1_fc2_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.bias"));
                    model.mm_model_block_1_block_2_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "0.weight"));
                    model.mm_model_block_1_block_2_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.weight"));
                    model.mm_model_block_1_block_2_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.bias"));
                    model.mm_model_block_2_block_0_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "0.weight"));
                    model.mm_model_block_2_block_0_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.weight"));
                    model.mm_model_block_2_block_0_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.bias"));
                    model.mm_model_block_2_block_1_fc1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.weight"));
                    model.mm_model_block_2_block_1_fc1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.bias"));
                    model.mm_model_block_2_block_1_fc2_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.weight"));
                    model.mm_model_block_2_block_1_fc2_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.bias"));
                    model.mm_model_block_2_block_2_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "0.weight"));
                    model.mm_model_block_2_block_2_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.weight"));
                    model.mm_model_block_2_block_2_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.bias"));
                } break;
            case PROJECTOR_TYPE_LDPV2:
                {
                    // MobilVLM_V2 projection
                    model.mm_model_mlp_0_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "weight"));
                    model.mm_model_mlp_0_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "bias"));
                    model.mm_model_mlp_2_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 2, "weight"));
                    model.mm_model_mlp_2_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 2, "bias"));
                    model.mm_model_peg_0_w = get_tensor(string_format(TN_MVLM_PROJ_PEG, 0, "weight"));
                    model.mm_model_peg_0_b = get_tensor(string_format(TN_MVLM_PROJ_PEG, 0, "bias"));
                } break;
            case PROJECTOR_TYPE_MINICPMV:
                {
                    // model.mm_model_pos_embed = get_tensor(new_clip->ctx_data, TN_MINICPMV_POS_EMBD);
                    model.mm_model_pos_embed_k = get_tensor(TN_MINICPMV_POS_EMBD_K);
                    model.mm_model_query = get_tensor(TN_MINICPMV_QUERY);
                    model.mm_model_proj = get_tensor(TN_MINICPMV_PROJ);
                    model.mm_model_kv_proj = get_tensor(TN_MINICPMV_KV_PROJ);
                    model.mm_model_attn_q_w = get_tensor(string_format(TN_MINICPMV_ATTN, "q", "weight"));
                    model.mm_model_attn_k_w = get_tensor(string_format(TN_MINICPMV_ATTN, "k", "weight"));
                    model.mm_model_attn_v_w = get_tensor(string_format(TN_MINICPMV_ATTN, "v", "weight"));
                    model.mm_model_attn_q_b = get_tensor(string_format(TN_MINICPMV_ATTN, "q", "bias"));
                    model.mm_model_attn_k_b = get_tensor(string_format(TN_MINICPMV_ATTN, "k", "bias"));
                    model.mm_model_attn_v_b = get_tensor(string_format(TN_MINICPMV_ATTN, "v", "bias"));
                    model.mm_model_attn_o_w = get_tensor(string_format(TN_MINICPMV_ATTN, "out", "weight"));
                    model.mm_model_attn_o_b = get_tensor(string_format(TN_MINICPMV_ATTN, "out", "bias"));
                    model.mm_model_ln_q_w = get_tensor(string_format(TN_MINICPMV_LN, "q", "weight"));
                    model.mm_model_ln_q_b = get_tensor(string_format(TN_MINICPMV_LN, "q", "bias"));
                    model.mm_model_ln_kv_w = get_tensor(string_format(TN_MINICPMV_LN, "kv", "weight"));
                    model.mm_model_ln_kv_b = get_tensor(string_format(TN_MINICPMV_LN, "kv", "bias"));
                    model.mm_model_ln_post_w = get_tensor(string_format(TN_MINICPMV_LN, "post", "weight"));
                    model.mm_model_ln_post_b = get_tensor(string_format(TN_MINICPMV_LN, "post", "bias"));
                } break;
            case PROJECTOR_TYPE_GLM_EDGE:
                {
                    model.mm_model_adapter_conv_w = get_tensor(string_format(TN_GLM_ADAPER_CONV, "weight"));
                    model.mm_model_adapter_conv_b = get_tensor(string_format(TN_GLM_ADAPER_CONV, "bias"));
                    model.mm_model_mlp_0_w = get_tensor(string_format(TN_GLM_ADAPTER_LINEAR, "weight"));
                    model.mm_model_ln_q_w = get_tensor(string_format(TN_GLM_ADAPTER_NORM_1, "weight"));
                    model.mm_model_ln_q_b = get_tensor(string_format(TN_GLM_ADAPTER_NORM_1, "bias"));
                    model.mm_model_mlp_1_w = get_tensor(string_format(TN_GLM_ADAPTER_D_H_2_4H, "weight"));
                    model.mm_model_mlp_2_w = get_tensor(string_format(TN_GLM_ADAPTER_GATE, "weight"));
                    model.mm_model_mlp_3_w = get_tensor(string_format(TN_GLM_ADAPTER_D_4H_2_H, "weight"));
                    model.mm_boi = get_tensor(string_format(TN_TOK_GLM_BOI));
                    model.mm_eoi = get_tensor(string_format(TN_TOK_GLM_EOI));
                } break;
            case PROJECTOR_TYPE_QWEN2VL:
            case PROJECTOR_TYPE_QWEN25VL:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_QWEN3VL:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_STEP3VL:
                {
                    model.mm_0_w     = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b     = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"), false);
                    model.mm_1_w     = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b     = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    model.mm_model_proj = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                } break;
            case PROJECTOR_TYPE_YOUTUVL:
                {
                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM);        // merger.ln_q (RMS norm)
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));  // merger.mlp.0
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));  // merger.mlp.2
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_YASA2:
                {
                    // reuse tensors already loaded by the common section
                    // (TN_PATCH_EMBD and TN_PATCH_BIAS have the same tensor names)
                    GGML_ASSERT(model.patch_embeddings_0 && "yasa2 requires v.patch_embd.weight");
                    model.yasa_patch_w = model.patch_embeddings_0;
                    model.yasa_patch_b = model.patch_bias;
                    model.yasa_patch_ln_w = get_tensor(TN_YASA_PATCH_LN_W, false);
                    model.yasa_patch_ln_b = get_tensor(TN_YASA_PATCH_LN_B, false);
                    model.yasa_backbone_ln_w = get_tensor(TN_YASA_BACKBONE_LN_W, false);
                    model.yasa_backbone_ln_b = get_tensor(TN_YASA_BACKBONE_LN_B, false);
                    model.yasa_vision_pos_embed = get_tensor(TN_YASA_POS_EMBD, false);
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"), false);
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);

                    model.yasa_stages.clear();
                    for (int s = 0; ; ++s) {
                        yasa2_stage stage;
                        stage.down_ln_w   = get_tensor(string_format(TN_YASA_STAGE_DOWN_LN, s, "weight"), false);
                        stage.down_ln_b   = get_tensor(string_format(TN_YASA_STAGE_DOWN_LN, s, "bias"), false);
                        stage.down_conv_w = get_tensor(string_format(TN_YASA_STAGE_DOWN_CONV, s, "weight"), false);
                        stage.down_conv_b = get_tensor(string_format(TN_YASA_STAGE_DOWN_CONV, s, "bias"), false);

                        for (int bi = 0; ; ++bi) {
                            yasa2_block blk;
                            blk.dw_w = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "dw", "weight"), false);
                            if (!blk.dw_w) {
                                break;
                            }
                            blk.dw_b  = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "dw", "bias"), false);
                            blk.ln_w  = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "ln", "weight"), false);
                            blk.ln_b  = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "ln", "bias"), false);
                            blk.pw1_w = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "pw1", "weight"), false);
                            blk.pw1_b = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "pw1", "bias"), false);
                            blk.grn_w = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "grn", "weight"), false);
                            blk.grn_b = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "grn", "bias"), false);
                            blk.pw2_w = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "pw2", "weight"), false);
                            blk.pw2_b = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "pw2", "bias"), false);
                            stage.blocks.push_back(blk);
                        }

                        if (!stage.down_conv_w && stage.blocks.empty()) {
                            break;
                        }
                        model.yasa_stages.push_back(std::move(stage));
                    }
                } break;
            case PROJECTOR_TYPE_GLM4V:
                {
                    model.mm_fc_w        = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                    model.mm_ffn_up_w    = get_tensor(string_format(TN_MM_UP,        "weight"));
                    model.mm_ffn_up_b    = get_tensor(string_format(TN_MM_UP,        "bias"), false);
                    model.mm_ffn_gate_w  = get_tensor(string_format(TN_MM_GATE,      "weight"));
                    model.mm_ffn_gate_b  = get_tensor(string_format(TN_MM_GATE,      "bias"), false);
                    model.mm_ffn_down_w  = get_tensor(string_format(TN_MM_DOWN,      "weight"));
                    model.mm_ffn_down_b  = get_tensor(string_format(TN_MM_DOWN,      "bias"), false);
                    model.mm_post_norm_w = get_tensor(string_format(TN_MM_POST_NORM, "weight"));
                    model.mm_post_norm_b = get_tensor(string_format(TN_MM_POST_NORM, "bias"), false);
                    model.mm_patch_merger_w = get_tensor(string_format(TN_MM_PATCH_MERGER, "weight"));
                    model.mm_patch_merger_b = get_tensor(string_format(TN_MM_PATCH_MERGER, "bias"));
                } break;
            case PROJECTOR_TYPE_GEMMA3:
                {
                    model.mm_input_proj_w = get_tensor(TN_MM_INP_PROJ);
                    model.mm_soft_emb_norm_w = get_tensor(TN_MM_SOFT_EMB_N);
                } break;
            case PROJECTOR_TYPE_GEMMA4V:
                {
                    model.mm_input_proj_w = get_tensor(TN_MM_INP_PROJ);
                    model.std_bias  = get_tensor(TN_STD_BIAS,  false);
                    model.std_scale = get_tensor(TN_STD_SCALE, false);
                    // load scalar for Gemma4ClippableLinear
                    for (auto * tensor : tensors_to_load) {
                        std::string name = tensor->name;
                        if (string_ends_with(name, ".weight")) {
                            std::string name_inp_max = name;
                            std::string name_inp_min = name;
                            std::string name_out_max = name;
                            std::string name_out_min = name;
                            string_replace_all(name_inp_max, ".weight", ".input_max");
                            string_replace_all(name_inp_min, ".weight", ".input_min");
                            string_replace_all(name_out_max, ".weight", ".output_max");
                            string_replace_all(name_out_min, ".weight", ".output_min");
                            model.clamp_info_map[name] = {
                                get_scalar(name_inp_max, FLT_MAX),
                                get_scalar(name_inp_min, -FLT_MAX),
                                get_scalar(name_out_max, FLT_MAX),
                                get_scalar(name_out_min, -FLT_MAX)
                            };
                        }
                    }
                } break;
            case PROJECTOR_TYPE_GEMMA3NV:
                {
                    model.mobilenet_stem_conv_w = get_tensor(TN_MNV5_STEM_CONV, false);
                    model.mobilenet_stem_conv_b = get_tensor(TN_MNV5_STEM_BIAS, false);
                    model.mobilenet_stem_norm_w = get_tensor(TN_MNV5_STEM_BN, false);

                    model.msfa_ffn_expand_w  = get_tensor(TN_MNV5_MSFA_FFN_EXP_W, false);
                    model.msfa_ffn_expand_bn = get_tensor(TN_MNV5_MSFA_FFN_EXP_BN, false); // Consume BN if present but likely folded
                    model.msfa_ffn_project_w = get_tensor(TN_MNV5_MSFA_FFN_PROJ_W, false);
                    model.msfa_ffn_project_bn = get_tensor(TN_MNV5_MSFA_FFN_PROJ_BN, false);

                    model.msfa_concat_norm_w = get_tensor(TN_MNV5_MSFA_NORM, false);

                    // Dynamically load blocks stage by stage
                    for (int stage = 0; stage < 4; ++stage) {
                        int blocks_found_in_stage = 0;

                        for (int blk_idx = 0; ; ++blk_idx) {
                            bool found_block = false;
                            mobilenetv5_block block;

                            // 1. Check for Edge Residual (S0)
                            block.s0_conv_exp_w = get_tensor(string_format(TN_MNV5_BLK_S0_EXP_W, stage, blk_idx), false);
                            if (block.s0_conv_exp_w) {
                                found_block = true;
                                block.s0_bn1_w      = get_tensor(string_format(TN_MNV5_BLK_S0_BN1_W, stage, blk_idx), false);
                                block.s0_conv_pwl_w = get_tensor(string_format(TN_MNV5_BLK_S0_PWL_W, stage, blk_idx), false);
                                block.s0_bn2_w      = get_tensor(string_format(TN_MNV5_BLK_S0_BN2_W, stage, blk_idx), false);
                            }
                            // 2. Check for UIR (Universal Inverted Residual)
                            else {
                                // Check for dw_start OR pw_exp (some UIR blocks skip dw_start)
                                block.dw_start_w = get_tensor(string_format(TN_MNV5_BLK_DW_START_W, stage, blk_idx), false);
                                block.pw_exp_w   = get_tensor(string_format(TN_MNV5_BLK_PW_EXP_W, stage, blk_idx), false);

                                if (block.dw_start_w || block.pw_exp_w) {
                                    found_block = true;
                                    if (block.dw_start_w) {
                                        block.dw_start_bn_w = get_tensor(string_format(TN_MNV5_BLK_DW_START_BN, stage, blk_idx), false);
                                    }
                                    if (block.pw_exp_w) {
                                        block.pw_exp_bn_w   = get_tensor(string_format(TN_MNV5_BLK_PW_EXP_BN, stage, blk_idx), false);
                                    }
                                    block.dw_mid_w      = get_tensor(string_format(TN_MNV5_BLK_DW_MID_W, stage, blk_idx), false);
                                    if (block.dw_mid_w) {
                                        block.dw_mid_bn_w   = get_tensor(string_format(TN_MNV5_BLK_DW_MID_BN, stage, blk_idx), false);
                                    }
                                    block.pw_proj_w     = get_tensor(string_format(TN_MNV5_BLK_PW_PROJ_W, stage, blk_idx), false);
                                    if (block.pw_proj_w) {
                                        block.pw_proj_bn_w  = get_tensor(string_format(TN_MNV5_BLK_PW_PROJ_BN, stage, blk_idx), false);
                                    }
                                    block.layer_scale_w = get_tensor(string_format(TN_MNV5_BLK_LAYER_SCALE, stage, blk_idx), false);
                                }
                            }

                            // 3. Check for Attention (MQA)
                            // Even if UIR/Edge check failed, this might be a pure attention block
                            ggml_tensor* attn_q_check = get_tensor(string_format(TN_MNV5_ATTN_Q_W, stage, blk_idx), false);
                            if (attn_q_check) {
                                found_block = true;
                                block.attn_q_w = attn_q_check;
                                block.attn_k_w = get_tensor(string_format(TN_MNV5_ATTN_K_W, stage, blk_idx), false);
                                block.attn_v_w = get_tensor(string_format(TN_MNV5_ATTN_V_W, stage, blk_idx), false);
                                block.attn_o_w = get_tensor(string_format(TN_MNV5_ATTN_O_W, stage, blk_idx), false);
                                block.attn_k_dw_w   = get_tensor(string_format(TN_MNV5_ATTN_K_DW, stage, blk_idx), false);
                                block.attn_k_norm_w = get_tensor(string_format(TN_MNV5_ATTN_K_NORM, stage, blk_idx), false);
                                block.attn_v_dw_w   = get_tensor(string_format(TN_MNV5_ATTN_V_DW, stage, blk_idx), false);
                                block.attn_v_norm_w = get_tensor(string_format(TN_MNV5_ATTN_V_NORM, stage, blk_idx), false);
                                block.attn_norm_w   = get_tensor(string_format(TN_MNV5_ATTN_NORM, stage, blk_idx), false);
                                // Note: Attention blocks also have layer_scale, load it if not already loaded by UIR check
                                if (!block.layer_scale_w) {
                                    block.layer_scale_w = get_tensor(string_format(TN_MNV5_BLK_LAYER_SCALE, stage, blk_idx), false);
                                }
                            }

                            if (found_block) {
                                model.mobilenet_blocks.push_back(block);
                                blocks_found_in_stage++;
                            } else {
                                // End of blocks for this stage
                                break;
                            }
                        }

                        // Track where this stage ends in the flat vector
                        if (blocks_found_in_stage > 0) {
                            model.mobilenet_stage_ends.push_back(model.mobilenet_blocks.size() - 1);
                            LOG_INF("%s: Stage %d ended at global block index %zu\n", __func__, stage, model.mobilenet_blocks.size() - 1);
                        }
                    }
                    model.mm_input_proj_w = get_tensor(TN_MM_INP_PROJ);
                    model.mm_soft_emb_norm_w = get_tensor(TN_MM_SOFT_EMB_N);
                } break;
            case PROJECTOR_TYPE_IDEFICS3:
                {
                    model.mm_fc_w = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                } break;
            case PROJECTOR_TYPE_LFM2:
                {
                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM, false);
                    model.mm_input_norm_b = get_tensor(TN_MM_INP_NORM_B, false);
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_KIMIVL:
            case PROJECTOR_TYPE_PADDLEOCR:
            case PROJECTOR_TYPE_KIMIK25:
                {
                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM);
                    model.mm_input_norm_b = get_tensor(TN_MM_INP_NORM_B);
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_PIXTRAL:
                {
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);
                    // [IMG_BREAK] token embedding
                    model.token_embd_img_break = get_tensor(TN_TOK_IMG_BREAK);
                    // for mistral small 3.1
                    model.mm_input_norm_w   = get_tensor(TN_MM_INP_NORM, false);
                    model.mm_patch_merger_w = get_tensor(string_format(TN_MM_PATCH_MERGER, "weight"), false);
                } break;
            case PROJECTOR_TYPE_LIGHTONOCR:
                {
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);
                    model.mm_input_norm_w   = get_tensor(TN_MM_INP_NORM, false);
                    model.mm_patch_merger_w = get_tensor(string_format(TN_MM_PATCH_MERGER, "weight"), false);
                } break;
            case PROJECTOR_TYPE_DOTS_OCR:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM);
                    model.mm_input_norm_b = get_tensor(TN_MM_INP_NORM_B);
                    // post_trunk_norm: applied after all ViT blocks, before the merger
                    model.post_ln_w = get_tensor(string_format(TN_MM_POST_NORM, "weight"));
                } break;
            case PROJECTOR_TYPE_ULTRAVOX:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_norm_pre_w = get_tensor(string_format(TN_MM_NORM_PRE, "weight"));
                    model.mm_norm_mid_w = get_tensor(string_format(TN_MM_NORM_MID, "weight"));
                } break;
            case PROJECTOR_TYPE_MERALION:
                {
                    // Whisper encoder conv layers
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    // MERaLiON adaptor: 4 linear layers + ln_pre
                    // linear_0 = frame compression (19200->6400) + SiLU
                    // linear_1 = gate_proj (6400->6400) for GLU
                    // linear_2 = pool_proj (6400->6400) for GLU
                    // linear_3 = out_proj  (6400->3584)
                    model.mm_0_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "bias"));
                    model.mm_3_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 3, "weight"));
                    model.mm_3_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 3, "bias"));
                    // ln_speech (LayerNorm before adaptor)
                    model.mm_norm_pre_w = get_tensor(string_format(TN_MM_NORM_PRE, "weight"));
                    model.mm_norm_pre_b = get_tensor(string_format(TN_MM_NORM_PRE, "bias"));
                } break;
            case PROJECTOR_TYPE_QWEN2A:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_fc_w = get_tensor(string_format(TN_MM_AUDIO_FC, "weight"));
                    model.mm_fc_b = get_tensor(string_format(TN_MM_AUDIO_FC, "bias"));
                } break;
            case PROJECTOR_TYPE_QWEN3A:
                {
                    model.conv2d_1_w = get_tensor(string_format(TN_CONV2D, 1, "weight"));
                    model.conv2d_1_b = get_tensor(string_format(TN_CONV2D, 1, "bias"));
                    model.conv2d_2_w = get_tensor(string_format(TN_CONV2D, 2, "weight"));
                    model.conv2d_2_b = get_tensor(string_format(TN_CONV2D, 2, "bias"));
                    model.conv2d_3_w = get_tensor(string_format(TN_CONV2D, 3, "weight"));
                    model.conv2d_3_b = get_tensor(string_format(TN_CONV2D, 3, "bias"));
                    model.conv_out_w = get_tensor(string_format(TN_CONV_OUT, "weight")); // no bias
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_VOXTRAL:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                } break;
            case PROJECTOR_TYPE_MUSIC_FLAMINGO:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_INTERNVL:
                {
                    model.mm_0_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "bias"));
                    model.mm_3_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "weight"));
                    model.mm_3_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "bias"));
                } break;
            case PROJECTOR_TYPE_NEMOTRON_V2_VL:
                {
                    model.mm_0_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "weight"));
                    model.mm_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_3_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "weight"));
                } break;
            case PROJECTOR_TYPE_GLMA:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "bias"));
                    model.mm_norm_pre_w = get_tensor(string_format(TN_MM_NORM_PRE, "weight"));
                    model.mm_norm_pre_b = get_tensor(string_format(TN_MM_NORM_PRE, "bias"));
                    model.mm_boi = get_tensor(string_format(TN_TOK_BOI));
                    model.mm_eoi = get_tensor(string_format(TN_TOK_EOI));
                } break;
            case PROJECTOR_TYPE_LLAMA4:
                {
                    model.mm_model_proj    = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                    model.mm_model_mlp_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_model_mlp_2_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 2, "weight"));
                } break;
            case PROJECTOR_TYPE_COGVLM:
                {
                    model.mm_model_proj     = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                    model.mm_post_fc_norm_w = get_tensor(string_format(TN_MM_POST_FC_NORM, "weight"));
                    model.mm_post_fc_norm_b = get_tensor(string_format(TN_MM_POST_FC_NORM, "bias"));
                    model.mm_h_to_4h_w      = get_tensor(string_format(TN_MM_H_TO_4H,      "weight"));
                    model.mm_gate_w         = get_tensor(string_format(TN_MM_GATE,         "weight"));
                    model.mm_4h_to_h_w      = get_tensor(string_format(TN_MM_4H_TO_H,      "weight"));
                    model.mm_boi            = get_tensor(TN_TOK_BOI);
                    model.mm_eoi            = get_tensor(TN_TOK_EOI);
                } break;
            case PROJECTOR_TYPE_HUNYUANOCR:
            case PROJECTOR_TYPE_HUNYUANVL:
                {
                    // proj.0 -> mm.0 (conv1), proj.2 -> mm.2 (conv2), mlp -> mm.model.fc (linear)
                    model.mm_0_w            = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b            = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w            = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_1_b            = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                    model.mm_model_proj     = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                    model.mm_model_proj_b   = get_tensor(string_format(TN_MM_PROJECTOR, "bias"));
                    model.mm_pre_norm_w     = get_tensor(string_format(TN_MM_PRE_NORM, "weight"));
                    model.mm_post_norm_w    = get_tensor(string_format(TN_MM_POST_NORM, "weight"));
                    model.mm_img_begin      = get_tensor(TN_TOK_IMG_BEGIN);
                    model.mm_img_end        = get_tensor(TN_TOK_IMG_END);
                    model.image_newline     = get_tensor(TN_IMAGE_NEWLINE);
                    model.view_seperator    = get_tensor(TN_IMAGE_SEPERATOR, false);
                } break;
            case PROJECTOR_TYPE_JANUS_PRO:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"));
                } break;
            case PROJECTOR_TYPE_PHI4:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_DEEPSEEKOCR:
                {
                    model.pos_embed          = get_tensor(string_format(TN_SAM_POS_EMBD,   "weight"));
                    model.patch_embed_proj_w = get_tensor(string_format(TN_SAM_PATCH_EMBD, "weight"));
                    model.patch_embed_proj_b = get_tensor(string_format(TN_SAM_PATCH_EMBD, "bias"));
                    model.sam_layers.resize(model.n_sam_layers);
                    for (int il = 0; il < model.n_sam_layers; ++il) {
                        auto & layer    = model.sam_layers[il];
                        layer.qkv_w     = get_tensor(string_format(TN_SAM_ATTN_QKV, il, "weight"));
                        layer.qkv_b     = get_tensor(string_format(TN_SAM_ATTN_QKV, il, "bias"));
                        layer.o_w       = get_tensor(string_format(TN_SAM_ATTN_OUT, il, "weight"));
                        layer.o_b       = get_tensor(string_format(TN_SAM_ATTN_OUT, il, "bias"));
                        layer.ln_1_w    = get_tensor(string_format(TN_SAM_PRE_NORM, il, "weight"));
                        layer.ln_1_b    = get_tensor(string_format(TN_SAM_PRE_NORM, il, "bias"));
                        layer.ln_2_w    = get_tensor(string_format(TN_SAM_POST_NORM, il, "weight"));
                        layer.ln_2_b    = get_tensor(string_format(TN_SAM_POST_NORM, il, "bias"));
                        layer.rel_pos_h = get_tensor(string_format(TN_SAM_ATTN_POS_H, il, "weight"));
                        layer.rel_pos_w = get_tensor(string_format(TN_SAM_ATTN_POS_W, il, "weight"));
                        layer.ff_up_w   = get_tensor(string_format(TN_SAM_FFN_UP, il, "weight"));
                        layer.ff_up_b   = get_tensor(string_format(TN_SAM_FFN_UP, il, "bias"));
                        layer.ff_down_w = get_tensor(string_format(TN_SAM_FFN_DOWN, il, "weight"));
                        layer.ff_down_b = get_tensor(string_format(TN_SAM_FFN_DOWN, il, "bias"));
                    }
                    model.neck_0_w       = get_tensor(string_format(TN_SAM_NECK, 0, "weight"));
                    model.neck_1_b       = get_tensor(string_format(TN_SAM_NECK, 1, "bias"));
                    model.neck_1_w       = get_tensor(string_format(TN_SAM_NECK, 1, "weight"));
                    model.neck_2_w       = get_tensor(string_format(TN_SAM_NECK, 2, "weight"));
                    model.neck_3_b       = get_tensor(string_format(TN_SAM_NECK, 3, "bias"));
                    model.neck_3_w       = get_tensor(string_format(TN_SAM_NECK, 3, "weight"));
                    model.net_2          = get_tensor(string_format(TN_SAM_NET, 2, "weight"));
                    model.net_3          = get_tensor(string_format(TN_SAM_NET, 3, "weight"));
                    model.image_newline  = get_tensor(TN_IMAGE_NEWLINE);
                    model.view_seperator = get_tensor(TN_IMAGE_SEPERATOR);
                    model.mm_fc_w        = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                    model.mm_fc_b        = get_tensor(string_format(TN_MM_PROJECTOR, "bias"));
                 } break;
            case PROJECTOR_TYPE_GEMMA4A:
                {
                    for (int i = 0; i < 2; i++) {
                        model.sscp_conv_w[i] = get_tensor(string_format(TN_A_CONV1D, i, "weight"));
                        model.sscp_conv_b[i] = get_tensor(string_format(TN_A_CONV1D, i, "bias"), false);
                        model.sscp_norm_w[i] = get_tensor(string_format(TN_A_CONV1D_NORM, i, "weight"), false);
                    }
                    model.sscp_inp_proj_w = get_tensor(string_format(TN_A_INP_PROJ, "weight"));
                    model.sscp_inp_proj_b = get_tensor(string_format(TN_A_INP_PROJ, "bias"), false);
                    model.audio_out_proj_w = get_tensor(string_format(TN_A_OUT_PROJ, "weight"), false);
                    model.audio_out_proj_b = get_tensor(string_format(TN_A_OUT_PROJ, "bias"), false);
                    // audio multimodal embedder (mm.a.* namespace, not mm.*)
                    model.mm_soft_emb_norm_w = get_tensor(string_format(TN_A_MM_SOFT_EMB_N, "weight"), false);
                    model.mm_input_proj_w    = get_tensor(string_format(TN_A_MM_INP_PROJ, "weight"), false);

                    // Per-layer tensors NOT loaded by the generic loop above
                    for (int il = 0; il < hparams.n_layer; ++il) {
                        auto & layer = model.layers[il];

                        // Gemma4 audio conformer-specific tensors
                        layer.ff_norm_w        = get_tensor(string_format(TN_FFN_NORM, prefix, il, "weight"));
                        layer.attn_pre_norm_w  = get_tensor(string_format(TN_A_ATTN_PRE_NORM, prefix, il, "weight"), false);
                        layer.per_dim_scale_w  = get_tensor(string_format(TN_A_PER_DIM_SCALE, prefix, il, "weight"), false);
                        layer.per_dim_k_scale_w = get_tensor(string_format(TN_A_PER_DIM_K_SCALE, prefix, il, "weight"), false);
                        layer.attn_k_rel_w     = get_tensor(string_format(TN_A_ATTN_K_REL, prefix, il, "weight"), false);

                        // Convolution module
                        // Note: conv_norm / norm_conv are swapped in GGUF due to
                        // upstream tensor_mapping.py, so we load them in reverse order
                        layer.norm_conv_w  = get_tensor(string_format(TN_CONV_NORM, prefix, il, "weight"), false);
                        layer.norm_conv_b  = get_tensor(string_format(TN_CONV_NORM, prefix, il, "bias"), false);
                        layer.conv_pw1_w   = get_tensor(string_format(TN_CONV_PW1,  prefix, il, "weight"));
                        layer.conv_pw1_b   = get_tensor(string_format(TN_CONV_PW1,  prefix, il, "bias"), false);
                        layer.conv_dw_w    = get_tensor(string_format(TN_CONV_DW,   prefix, il, "weight"));
                        layer.conv_dw_b    = get_tensor(string_format(TN_CONV_DW,   prefix, il, "bias"), false);
                        layer.conv_norm_w  = get_tensor(string_format(TN_NORM_CONV, prefix, il, "weight"), false);
                        layer.conv_norm_b  = get_tensor(string_format(TN_NORM_CONV, prefix, il, "bias"), false);
                        layer.conv_pw2_w   = get_tensor(string_format(TN_CONV_PW2,  prefix, il, "weight"));
                        layer.conv_pw2_b   = get_tensor(string_format(TN_CONV_PW2,  prefix, il, "bias"), false);

                        // FFN2 (second half-step)
                        layer.ff_norm_1_w      = get_tensor(string_format(TN_FFN_NORM_1, prefix, il, "weight"));
                        layer.ff_up_1_w        = get_tensor(string_format(TN_FFN_UP_1, prefix, il, "weight"));
                        layer.ff_up_1_b        = get_tensor(string_format(TN_FFN_UP_1, prefix, il, "bias"), false);
                        layer.ff_down_1_w      = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "weight"));
                        layer.ff_down_1_b      = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "bias"), false);
                        layer.ff_post_norm_1_w = get_tensor(string_format(TN_A_FFN_POST_NORM_1, prefix, il, "weight"), false);
                    }

                    // Load clamp info for ClippableLinear AFTER all tensors are loaded
                    for (auto * tensor : tensors_to_load) {
                        std::string name = tensor->name;
                        if (string_ends_with(name, ".weight")) {
                            std::string name_inp_max = name;
                            std::string name_inp_min = name;
                            std::string name_out_max = name;
                            std::string name_out_min = name;
                            string_replace_all(name_inp_max, ".weight", ".input_max");
                            string_replace_all(name_inp_min, ".weight", ".input_min");
                            string_replace_all(name_out_max, ".weight", ".output_max");
                            string_replace_all(name_out_min, ".weight", ".output_min");
                            model.clamp_info_map[name] = {
                                get_scalar(name_inp_max, FLT_MAX),
                                get_scalar(name_inp_min, -FLT_MAX),
                                get_scalar(name_out_max, FLT_MAX),
                                get_scalar(name_out_min, -FLT_MAX)
                            };
                        }
                    }
                } break;
            case PROJECTOR_TYPE_LFM2A:
                {
                    for (int i : {0, 2, 3, 5, 6}) {
                        model.pre_encode_conv_X_w[i] = get_tensor(string_format(TN_CONV1D, i, "weight"));
                        model.pre_encode_conv_X_b[i] = get_tensor(string_format(TN_CONV1D, i, "bias"));
                    }
                    model.pre_encode_out_w    = get_tensor(string_format(TN_PRE_ENCODE_OUT, "weight"));
                    model.pre_encode_out_b    = get_tensor(string_format(TN_PRE_ENCODE_OUT, "bias"));

                    model.mm_0_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_3_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 3, "weight"));
                    model.mm_3_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 3, "bias"));

                    for (int il = 0; il < hparams.n_layer; ++il) {
                        auto & layer = model.layers[il];

                        layer.ff_norm_w   = get_tensor(string_format(TN_FFN_NORM,   prefix, il, "weight"));
                        layer.ff_norm_b   = get_tensor(string_format(TN_FFN_NORM,   prefix, il, "bias"));
                        layer.ff_norm_1_w = get_tensor(string_format(TN_FFN_NORM_1, prefix, il, "weight"));
                        layer.ff_norm_1_b = get_tensor(string_format(TN_FFN_NORM_1, prefix, il, "bias"));
                        layer.ff_up_1_w   = get_tensor(string_format(TN_FFN_UP_1,   prefix, il, "weight"));
                        layer.ff_up_1_b   = get_tensor(string_format(TN_FFN_UP_1,   prefix, il, "bias"));
                        layer.ff_down_1_w = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "weight"));
                        layer.ff_down_1_b = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "bias"));

                        layer.pos_bias_u = get_tensor(string_format(TN_POS_BIAS_U, prefix, il));
                        layer.pos_bias_v = get_tensor(string_format(TN_POS_BIAS_V, prefix, il));

                        layer.norm_conv_w = get_tensor(string_format(TN_NORM_CONV, prefix, il, "weight"));
                        layer.norm_conv_b = get_tensor(string_format(TN_NORM_CONV, prefix, il, "bias"));

                        layer.linear_pos_w = get_tensor(string_format(TN_LINEAR_POS, prefix, il, "weight"));

                        layer.conv_norm_w  = get_tensor(string_format(TN_CONV_NORM, prefix, il, "weight"));
                        layer.conv_norm_b  = get_tensor(string_format(TN_CONV_NORM, prefix, il, "bias"));
                        layer.conv_dw_w    = get_tensor(string_format(TN_CONV_DW,   prefix, il, "weight"));
                        layer.conv_dw_b    = get_tensor(string_format(TN_CONV_DW,   prefix, il, "bias"));
                        layer.conv_pw1_w   = get_tensor(string_format(TN_CONV_PW1,  prefix, il, "weight"));
                        layer.conv_pw1_b   = get_tensor(string_format(TN_CONV_PW1,  prefix, il, "bias"));
                        layer.conv_pw2_w   = get_tensor(string_format(TN_CONV_PW2,  prefix, il, "weight"));
                        layer.conv_pw2_b   = get_tensor(string_format(TN_CONV_PW2,  prefix, il, "bias"));
                    }
                } break;
            case PROJECTOR_TYPE_GRANITE_SPEECH:
                {
                    model.inp_proj_w     = get_tensor(string_format(TN_INP_PROJ,    "weight"));
                    model.inp_proj_b     = get_tensor(string_format(TN_INP_PROJ,    "bias"));
                    model.ctc_out_w      = get_tensor(string_format(TN_CTC_OUT,     "weight"));
                    model.ctc_out_b      = get_tensor(string_format(TN_CTC_OUT,     "bias"));
                    model.ctc_out_mid_w  = get_tensor(string_format(TN_CTC_OUT_MID, "weight"));
                    model.ctc_out_mid_b  = get_tensor(string_format(TN_CTC_OUT_MID, "bias"));

                    // per-layer tensors not loaded by the generic loop above
                    for (int il = 0; il < hparams.n_layer; ++il) {
                        auto & layer = model.layers[il];

                        layer.attn_rel_pos_emb = get_tensor(string_format(TN_ATTN_REL_POS_EMB, prefix, il));

                        layer.ff_norm_w   = get_tensor(string_format(TN_FFN_NORM,   prefix, il, "weight"));
                        layer.ff_norm_b   = get_tensor(string_format(TN_FFN_NORM,   prefix, il, "bias"));

                        layer.ff_norm_1_w = get_tensor(string_format(TN_FFN_NORM_1, prefix, il, "weight"));
                        layer.ff_norm_1_b = get_tensor(string_format(TN_FFN_NORM_1, prefix, il, "bias"));
                        layer.ff_up_1_w   = get_tensor(string_format(TN_FFN_UP_1,   prefix, il, "weight"));
                        layer.ff_up_1_b   = get_tensor(string_format(TN_FFN_UP_1,   prefix, il, "bias"));
                        layer.ff_down_1_w = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "weight"));
                        layer.ff_down_1_b = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "bias"));

                        layer.norm_conv_w = get_tensor(string_format(TN_NORM_CONV, prefix, il, "weight"));
                        layer.norm_conv_b = get_tensor(string_format(TN_NORM_CONV, prefix, il, "bias"));
                        layer.conv_norm_w = get_tensor(string_format(TN_CONV_NORM, prefix, il, "weight"));
                        layer.conv_norm_b = get_tensor(string_format(TN_CONV_NORM, prefix, il, "bias"));
                        layer.conv_dw_w   = get_tensor(string_format(TN_CONV_DW,   prefix, il, "weight"));
                        layer.conv_pw1_w  = get_tensor(string_format(TN_CONV_PW1,  prefix, il, "weight"));
                        layer.conv_pw1_b  = get_tensor(string_format(TN_CONV_PW1,  prefix, il, "bias"));
                        layer.conv_pw2_w  = get_tensor(string_format(TN_CONV_PW2,  prefix, il, "weight"));
                        layer.conv_pw2_b  = get_tensor(string_format(TN_CONV_PW2,  prefix, il, "bias"));
                    }

                    model.qf_proj_query    = get_tensor(TN_QF_PROJ_QUERY);
                    model.qf_proj_norm_w   = get_tensor(string_format(TN_QF_PROJ_NORM, "weight"));
                    model.qf_proj_norm_b   = get_tensor(string_format(TN_QF_PROJ_NORM, "bias"));
                    model.qf_proj_linear_w = get_tensor(string_format(TN_QF_PROJ_LINEAR, "weight"));
                    model.qf_proj_linear_b = get_tensor(string_format(TN_QF_PROJ_LINEAR, "bias"));

                    const int n_proj_layers = 2;
                    model.qf_proj_layers.resize(n_proj_layers);
                    for (int il = 0; il < n_proj_layers; ++il) {
                        auto & pl = model.qf_proj_layers[il];

                        pl.q_w    = get_tensor(string_format(TN_QF_SELF_ATTN_Q, il, "weight"));
                        pl.q_b    = get_tensor(string_format(TN_QF_SELF_ATTN_Q, il, "bias"));
                        pl.k_w    = get_tensor(string_format(TN_QF_SELF_ATTN_K, il, "weight"));
                        pl.k_b    = get_tensor(string_format(TN_QF_SELF_ATTN_K, il, "bias"));
                        pl.v_w    = get_tensor(string_format(TN_QF_SELF_ATTN_V, il, "weight"));
                        pl.v_b    = get_tensor(string_format(TN_QF_SELF_ATTN_V, il, "bias"));
                        pl.o_w    = get_tensor(string_format(TN_QF_SELF_ATTN_O, il, "weight"));
                        pl.o_b    = get_tensor(string_format(TN_QF_SELF_ATTN_O, il, "bias"));
                        pl.ln_1_w = get_tensor(string_format(TN_QF_SELF_ATTN_N, il, "weight"));
                        pl.ln_1_b = get_tensor(string_format(TN_QF_SELF_ATTN_N, il, "bias"));

                        pl.cross_attn_q_w    = get_tensor(string_format(TN_QF_CROSS_ATTN_Q, il, "weight"));
                        pl.cross_attn_q_b    = get_tensor(string_format(TN_QF_CROSS_ATTN_Q, il, "bias"));
                        pl.cross_attn_k_w    = get_tensor(string_format(TN_QF_CROSS_ATTN_K, il, "weight"));
                        pl.cross_attn_k_b    = get_tensor(string_format(TN_QF_CROSS_ATTN_K, il, "bias"));
                        pl.cross_attn_v_w    = get_tensor(string_format(TN_QF_CROSS_ATTN_V, il, "weight"));
                        pl.cross_attn_v_b    = get_tensor(string_format(TN_QF_CROSS_ATTN_V, il, "bias"));
                        pl.cross_attn_o_w    = get_tensor(string_format(TN_QF_CROSS_ATTN_O, il, "weight"));
                        pl.cross_attn_o_b    = get_tensor(string_format(TN_QF_CROSS_ATTN_O, il, "bias"));
                        pl.cross_attn_norm_w = get_tensor(string_format(TN_QF_CROSS_ATTN_N, il, "weight"));
                        pl.cross_attn_norm_b = get_tensor(string_format(TN_QF_CROSS_ATTN_N, il, "bias"));

                        pl.ff_up_w   = get_tensor(string_format(TN_QF_FFN_UP,   il, "weight"));
                        pl.ff_up_b   = get_tensor(string_format(TN_QF_FFN_UP,   il, "bias"));
                        pl.ff_down_w = get_tensor(string_format(TN_QF_FFN_DOWN, il, "weight"));
                        pl.ff_down_b = get_tensor(string_format(TN_QF_FFN_DOWN, il, "bias"));
                        pl.ln_2_w    = get_tensor(string_format(TN_QF_FFN_NORM, il, "weight"));
                        pl.ln_2_b    = get_tensor(string_format(TN_QF_FFN_NORM, il, "bias"));
                    }
                } break;
            default:
                GGML_ASSERT(false && "unknown projector type");
        }

        // load data
        {
            std::vector<uint8_t> read_buf;

            // alloc memory and offload data
            ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(ctx_clip.backend);
            ctx_clip.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_clip.ctx_data.get(), buft));
            ggml_backend_buffer_set_usage(ctx_clip.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            for (auto & t : tensors_to_load) {
                ggml_tensor * cur = ggml_get_tensor(ctx_clip.ctx_data.get(), t->name);
                GGML_ASSERT(cur && "tensor not found in ctx_data");
                auto it_off = tensor_offset.find(t->name);
                GGML_ASSERT(it_off != tensor_offset.end() && "no offset for tensor");
                const size_t offset = it_off->second;
                fin.seekg(offset, std::ios::beg);
                if (!fin) {
                    throw std::runtime_error(string_format("%s: failed to seek for tensor %s\n", __func__, t->name));
                }
                size_t num_bytes = ggml_nbytes(cur);
                if (ggml_backend_buft_is_host(buft)) {
                    // for the CPU and Metal backend, we can read directly into the tensor
                    fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
                } else {
                    // read into a temporary buffer first, then copy to device memory
                    read_buf.resize(num_bytes);
                    fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                    ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
                }
            }
            fin.close();

            LOG_DBG("%s: loaded %zu tensors from %s\n", __func__, tensors_to_load.size(), fname.c_str());
        }

    }

    struct support_info_op {
        ggml_tensor * op;

        // true if the op runs on the accelerated ctx_clip.backend
        bool is_accel = true;
    };

    struct support_info_graph {
        // whether the clip_ctx.backend supports flash attention
        bool fattn = true;
        ggml_tensor * fattn_op = nullptr; // for debugging

        std::vector<support_info_op> ops;
    };

    static void warmup(clip_ctx & ctx_clip) {
        // create a fake batch
        const auto & hparams = ctx_clip.model.hparams;
        clip_image_f32_batch batch;
        clip_image_f32_ptr img(clip_image_f32_init());
        if (ctx_clip.model.modality == CLIP_MODALITY_VISION) {
            img->nx = hparams.warmup_image_size;
            img->ny = hparams.warmup_image_size;
            LOG_INF("%s: warmup with image size = %d x %d\n", __func__, img->nx, img->ny);
        } else {
            img->nx = hparams.warmup_audio_size;
            img->ny = hparams.n_mel_bins;
            LOG_INF("%s: warmup with audio size = %d\n", __func__, img->nx);
        }
        batch.entries.push_back(std::move(img));
        warmup(ctx_clip, batch);
    }

    static void warmup(clip_ctx & ctx_clip, const clip_image_f32_batch & batch) {
        support_info_graph info;

        if (ctx_clip.flash_attn_type == CLIP_FLASH_ATTN_TYPE_AUTO) {
            // try to enable flash attention to see if it's supported
            ctx_clip.flash_attn_type = CLIP_FLASH_ATTN_TYPE_ENABLED;
            info = alloc_compute_meta(ctx_clip, batch);
            if (!info.fattn && info.fattn_op) {
                auto op = info.fattn_op;
                LOG_WRN("%s: *****************************************************************\n", __func__);
                LOG_WRN("%s: WARNING: flash attention not supported by %s, memory usage will increase\n", __func__, ggml_backend_name(ctx_clip.backend));
                LOG_WRN("%s: op params: \n", __func__);
                static auto print_shape = [](const char * fn, const char * name, ggml_tensor * t) {
                    LOG_WRN("%s:   %s: type = %s, ne = [%d %d %d %d], nb = [%d %d %d %d]\n", fn,
                            name, ggml_type_name(t->type),
                            t->ne[0], t->ne[1], t->ne[2], t->ne[3],
                            t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
                };
                print_shape(__func__, " dst", op);
                print_shape(__func__, "src0", op->src[0]);
                print_shape(__func__, "src1", op->src[1]);
                print_shape(__func__, "src2", op->src[2]);
                LOG_WRN("%s: please report this on github as an issue\n", __func__);
                LOG_WRN("%s: *****************************************************************\n", __func__);
                ctx_clip.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
                alloc_compute_meta(ctx_clip, batch);
            }
        } else {
            info = alloc_compute_meta(ctx_clip, batch);
            if (!info.fattn && ctx_clip.flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
                LOG_WRN("%s: flash attention is not supported by the current backend; falling back to CPU (performance will be degraded)\n", __func__);
            }
        }

        ctx_clip.is_allocated = true; // mark buffers as allocated

        LOG_INF("%s: flash attention is %s\n", __func__,
            (ctx_clip.flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) ? "enabled" : "disabled");

        // print ops that are not supported by the GPU backend (if there is one)
        if (ctx_clip.backend && ctx_clip.backend != ctx_clip.backend_cpu) {
            std::vector<support_info_op> unsupported_ops;
            for (const auto & op : info.ops) {
                if (!op.is_accel) {
                    unsupported_ops.push_back(op);
                }
            }
            if (!unsupported_ops.empty()) {
                LOG_WRN("%s: *****************************************************************\n", __func__);
                LOG_WRN("%s: WARNING: the CLIP graph uses unsupported operators by the backend\n", __func__);
                LOG_WRN("%s:          the performance will be suboptimal                      \n", __func__);
                LOG_WRN("%s:          list of unsupported ops (backend=%s):\n", __func__, ggml_backend_name(ctx_clip.backend));
                for (const auto & op : unsupported_ops) {
                    LOG_WRN("%s: %16s: type = %s, ne = [%d %d %d %d]\n", __func__,
                            ggml_op_name(op.op->op),
                            ggml_type_name(op.op->type),
                            op.op->ne[0], op.op->ne[1], op.op->ne[2], op.op->ne[3]);
                }
                LOG_WRN("%s: flash attention is %s\n", __func__,
                    (ctx_clip.flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) ? "enabled" : "disabled");
                LOG_WRN("%s: please report this on github as an issue\n", __func__);
                LOG_WRN("%s: ref: https://github.com/ggml-org/llama.cpp/pull/16837#issuecomment-3461676118\n", __func__);
                LOG_WRN("%s: *****************************************************************\n", __func__);
            }
        }
    }

    static support_info_graph alloc_compute_meta(clip_ctx & ctx_clip, const clip_image_f32_batch & batch) {
        ctx_clip.buf_compute_meta.resize(ctx_clip.max_nodes * ggml_tensor_overhead() + ggml_graph_overhead());

        ggml_cgraph * gf = clip_image_build_graph(&ctx_clip, batch);
        ggml_backend_sched_reserve(ctx_clip.sched.get(), gf);

        for (size_t i = 0; i < ctx_clip.backend_ptrs.size(); ++i) {
            ggml_backend_t backend = ctx_clip.backend_ptrs[i];
            ggml_backend_buffer_type_t buft = ctx_clip.backend_buft[i];
            size_t size = ggml_backend_sched_get_buffer_size(ctx_clip.sched.get(), backend);
            if (size > 1) {
                LOG_INF("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                        ggml_backend_buft_name(buft),
                        size / 1024.0 / 1024.0);
            }
        }

        const int n_splits = ggml_backend_sched_get_n_splits(ctx_clip.sched.get());
        const int n_nodes  = ggml_graph_n_nodes(gf);

        LOG_INF("%s: graph splits = %d, nodes = %d\n", __func__,  n_splits, n_nodes);

        support_info_graph res {
            /*.fattn    = */ true,
            /*.fattn_op = */ nullptr,
            /*.ops      = */ {},
        };

        // check op support
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            res.ops.push_back({node, true});
            if (!ggml_backend_supports_op(ctx_clip.backend, node)) {
                res.ops.back().is_accel = false;
                if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                    res.fattn    = false;
                    res.fattn_op = node;
                }
            }
        }

        return res;
    }

    void get_bool(const std::string & key, bool & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_bool(ctx_gguf.get(), i);
    }

    void get_i32(const std::string & key, int & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_i32(ctx_gguf.get(), i);
    }

    void get_u32(const std::string & key, int & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_u32(ctx_gguf.get(), i);
    }

    void get_f32(const std::string & key, float & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_f32(ctx_gguf.get(), i);
    }

    void get_string(const std::string & key, std::string & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = std::string(gguf_get_val_str(ctx_gguf.get(), i));
    }

    void get_arr_int(const std::string & key, std::vector<int> & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        int n = gguf_get_arr_n(ctx_gguf.get(), i);
        output.resize(n);
        const int32_t * values = (const int32_t *)gguf_get_arr_data(ctx_gguf.get(), i);
        for (int i = 0; i < n; ++i) {
            output[i] = values[i];
        }
    }

    static void set_llava_uhd_res_candidates(clip_model & model, const int max_patches_per_side) {
        auto & hparams = model.hparams;
        for (int x = 1; x <= max_patches_per_side; x++) {
            for (int y = 1; y <= max_patches_per_side; y++) {
                if (x == 1 && y == 1) {
                    continue; // skip the first point
                }
                hparams.image_res_candidates.push_back(clip_image_size{
                    x*hparams.image_size,
                    y*hparams.image_size,
                });
            }
        }
    }

    static void set_internvl_dhr_res_candidates(clip_model & model) {
        auto & hparams = model.hparams;
        int min_num = hparams.preproc_min_tiles;
        int max_num = hparams.preproc_max_tiles;
        if (min_num < 1) {
           return; // avoid  divide by 0
        }
        for (int a = min_num; a <= max_num; ++a) {
            int b_lo = (min_num + a - 1) / a;
            int b_hi = max_num / a;
            b_lo = std::max(b_lo, min_num);
            b_hi = std::min(b_hi, max_num);
            for (int b = b_lo; b <= b_hi; ++b) {
                hparams.image_res_candidates.push_back(clip_image_size {
                    a*hparams.image_size,
                    b*hparams.image_size,
                });
            }
        }
    }
};

struct clip_init_result clip_init(const char * fname, struct clip_context_params ctx_params) {
    clip_ctx * ctx_vision = nullptr;
    clip_ctx * ctx_audio = nullptr;

    try {
        clip_model_loader loader(fname);
        bool skip_audio = false;

        if (loader.has_vision) {
            ctx_vision = new clip_ctx(ctx_params);
            loader.load_hparams(ctx_vision->model, CLIP_MODALITY_VISION);
            loader.load_tensors(*ctx_vision);
            if (ctx_params.warmup) {
                loader.warmup(*ctx_vision);
            }

            // TODO: we don't support audio for Gemma 3N, but GGUF contains audio tensors
            // we can remove this check when we implement audio support for Gemma 3N
            skip_audio = ctx_vision->model.proj_type == PROJECTOR_TYPE_GEMMA3NV;
        }

        if (loader.has_audio && !skip_audio) {
            ctx_audio = new clip_ctx(ctx_params);
            loader.load_hparams(ctx_audio->model, CLIP_MODALITY_AUDIO);
            loader.load_tensors(*ctx_audio);
            if (ctx_params.warmup) {
                loader.warmup(*ctx_audio);
            }
        }

    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to load model '%s': %s\n", __func__, fname, e.what());

        delete ctx_vision;
        delete ctx_audio;

        return {nullptr, nullptr};
    }

    return {ctx_vision, ctx_audio};
}

struct clip_image_size * clip_image_size_init() {
    struct clip_image_size * load_image_size = new struct clip_image_size();
    load_image_size->width = 448;
    load_image_size->height = 448;
    return load_image_size;
}

struct clip_image_u8 * clip_image_u8_init() {
    return new clip_image_u8();
}

struct clip_image_f32 * clip_image_f32_init() {
    return new clip_image_f32();
}

struct clip_image_f32_batch * clip_image_f32_batch_init() {
    return new clip_image_f32_batch();
}

unsigned char * clip_image_u8_get_data(struct clip_image_u8 * img, uint32_t * nx, uint32_t * ny) {
    if (nx) *nx = img->nx;
    if (ny) *ny = img->ny;
    return img->buf.data();
}

void clip_image_size_free(struct clip_image_size * load_image_size) {
    if (load_image_size == nullptr) {
        return;
    }
    delete load_image_size;
}
void clip_image_u8_free(struct clip_image_u8  * img) { delete img; }
void clip_image_f32_free(struct clip_image_f32 * img) { delete img; }
void clip_image_u8_batch_free(struct clip_image_u8_batch * batch) { delete batch; }
void clip_image_f32_batch_free(struct clip_image_f32_batch * batch) { delete batch; }

size_t clip_image_f32_batch_n_images(const struct clip_image_f32_batch * batch) {
    return batch->entries.size();
}

size_t clip_image_f32_batch_nx(const struct clip_image_f32_batch * batch, int idx) {
    if (idx < 0 || idx >= (int)batch->entries.size()) {
        LOG_ERR("%s: invalid index %d\n", __func__, idx);
        return 0;
    }
    return batch->entries[idx]->nx;
}

size_t clip_image_f32_batch_ny(const struct clip_image_f32_batch * batch, int idx) {
    if (idx < 0 || idx >= (int)batch->entries.size()) {
        LOG_ERR("%s: invalid index %d\n", __func__, idx);
        return 0;
    }
    return batch->entries[idx]->ny;
}

clip_image_f32 * clip_image_f32_get_img(const struct clip_image_f32_batch * batch, int idx) {
    if (idx < 0 || idx >= (int)batch->entries.size()) {
        LOG_ERR("%s: invalid index %d\n", __func__, idx);
        return nullptr;
    }
    return batch->entries[idx].get();
}

void clip_build_img_from_pixels(const unsigned char * rgb_pixels, int nx, int ny, clip_image_u8 * img) {
    img->nx = nx;
    img->ny = ny;
    img->buf.resize(3 * nx * ny);
    memcpy(img->buf.data(), rgb_pixels, img->buf.size());
}

ggml_tensor * clip_get_newline_tensor(const struct clip_ctx * ctx) {
    return ctx->model.image_newline;
}

void clip_free(clip_ctx * ctx) {
    if (ctx == nullptr) {
        return;
    }
    delete ctx;
}

// deprecated
size_t clip_embd_nbytes(const struct clip_ctx * ctx) {
    const int32_t nx = ctx->model.hparams.image_size;
    const int32_t ny = ctx->model.hparams.image_size;
    return clip_embd_nbytes_by_img(ctx, nx, ny);
}

size_t clip_embd_nbytes_by_img(const struct clip_ctx * ctx, int img_w, int img_h) {
    clip_image_f32 img;
    img.nx = img_w;
    img.ny = img_h;
    return clip_n_output_tokens(ctx, &img) * clip_n_mmproj_embd(ctx) * sizeof(float);
}

int32_t clip_get_image_size(const struct clip_ctx * ctx) {
    return ctx->model.hparams.image_size;
}

int32_t clip_get_patch_size(const struct clip_ctx * ctx) {
    return ctx->model.hparams.patch_size;
}

int32_t clip_get_hidden_size(const struct clip_ctx * ctx) {
    return ctx->model.hparams.n_embd;
}

const char * clip_patch_merge_type(const struct clip_ctx * ctx) {
    return ctx->model.hparams.mm_patch_merge_type == PATCH_MERGE_SPATIAL_UNPAD ? "spatial_unpad" : "flat";
}

int clip_n_output_tokens_x(const struct clip_ctx * ctx, struct clip_image_f32 * img) {
    const auto & params = ctx->model.hparams;
    const int n_total = clip_n_output_tokens(ctx, img);
    const auto & proj = ctx->proj_type();
    switch (proj) {
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_GLM4V:
        case PROJECTOR_TYPE_PADDLEOCR:
        case PROJECTOR_TYPE_HUNYUANOCR:
        case PROJECTOR_TYPE_HUNYUANVL:
        case PROJECTOR_TYPE_YOUTUVL:
            return (img->nx / params.patch_size) / 2;
        case PROJECTOR_TYPE_STEP3VL:
            return img->nx / (params.patch_size * params.n_merge);
        default:
            break;
    }
    return n_total;
}

int clip_n_output_tokens_y(const struct clip_ctx * ctx, struct clip_image_f32 * img) {
    const auto & params = ctx->model.hparams;
    const auto & proj = ctx->proj_type();
    switch (proj) {
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_GLM4V:
        case PROJECTOR_TYPE_PADDLEOCR:
        case PROJECTOR_TYPE_HUNYUANVL:
        case PROJECTOR_TYPE_YOUTUVL:
            return (img->ny / params.patch_size) / 2;
        case PROJECTOR_TYPE_STEP3VL:
            return img->ny / (params.patch_size * params.n_merge);
        default:
            break;
    }
    return 1;
}

int clip_n_output_tokens(const struct clip_ctx * ctx, struct clip_image_f32 * img) {
    const auto & params = ctx->model.hparams;

    // for models with fixed size image, the input image is already pre-processed and resized to square
    int patch_size = params.patch_size;
    int n_patches = (img->nx / patch_size) * (img->ny / patch_size);

    projector_type proj = ctx->proj_type();

    switch (proj) {
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_JANUS_PRO:
        case PROJECTOR_TYPE_PHI4:
            {
                // do nothing
            } break;
        case PROJECTOR_TYPE_YASA2:
            {
                n_patches = 64; // adaptive average pooling to 8x8 tokens
            } break;
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
        case PROJECTOR_TYPE_GLM_EDGE:
            {
                n_patches /= 4;
                if (ctx->model.mm_boi) {
                    n_patches += 2; // for BOI and EOI token embeddings
                }
            } break;
        case PROJECTOR_TYPE_MINICPMV:
            {
                // Use actual config value if available, otherwise fall back to hardcoded values
                if (params.minicpmv_query_num > 0) {
                    n_patches = params.minicpmv_query_num;
                } else {
                    // Fallback to hardcoded values for legacy models
                    if (params.minicpmv_version == 2) {
                        n_patches = 96;
                    } else if (params.minicpmv_version == 3) {
                        n_patches = 64;
                    } else if (params.minicpmv_version == 4) {
                        n_patches = 64;
                    } else if (params.minicpmv_version == 5) {
                        // MiniCPM-V 4.0
                        n_patches = 64;
                    } else if (params.minicpmv_version == 6) {
                        // MiniCPM-V 4.5
                        n_patches = 64;
                    } else if (params.minicpmv_version == 100045) {
                        // MiniCPM-o 4.5
                        n_patches = 64;
                    } else {
                        GGML_ABORT("Unknown minicpmv version");
                    }
                }
            } break;
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_GLM4V:
        case PROJECTOR_TYPE_YOUTUVL:
            {
                // dynamic size (2 conv, so double patch size)
                int x_patch = img->nx / (params.patch_size * 2);
                int y_patch = img->ny / (params.patch_size * 2);
                n_patches = x_patch * y_patch;
            } break;
        case PROJECTOR_TYPE_STEP3VL:
            {
                int x_patch = img->nx / (params.patch_size * params.n_merge);
                int y_patch = img->ny / (params.patch_size * params.n_merge);
                n_patches = x_patch * y_patch;
            } break;
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_GEMMA4V:
        case PROJECTOR_TYPE_IDEFICS3:
        case PROJECTOR_TYPE_INTERNVL:
        case PROJECTOR_TYPE_NEMOTRON_V2_VL:
        case PROJECTOR_TYPE_LLAMA4:
            {
                // both X and Y are downscaled by the scale factor
                int scale_factor = ctx->model.hparams.n_merge;
                n_patches /= (scale_factor * scale_factor);
            } break;
        case PROJECTOR_TYPE_GEMMA3NV:
            {
                // MobileNetV5 MSFA adapter always outputs fixed 16x16 resolution
                // regardless of input size (see architecture description)
                n_patches = ctx->model.hparams.image_size / ctx->model.hparams.patch_size;
            } break;
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_KIMIVL:
        case PROJECTOR_TYPE_KIMIK25:
            {
                // dynamic size
                int out_patch_size = params.patch_size * ctx->model.hparams.n_merge;
                int x_patch = CLIP_ALIGN(img->nx, out_patch_size) / out_patch_size;
                int y_patch = CLIP_ALIGN(img->ny, out_patch_size) / out_patch_size;
                n_patches = x_patch * y_patch;
            } break;
        case PROJECTOR_TYPE_PADDLEOCR:
        case PROJECTOR_TYPE_DOTS_OCR:
            {
                // dynamic size
                int n_merge = ctx->model.hparams.n_merge;
                int stride = n_merge * n_merge;
                n_patches = CLIP_ALIGN(n_patches, stride) / stride;
            } break;
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_LIGHTONOCR:
            {
                // dynamic size
                int n_merge = ctx->model.hparams.n_merge;
                int n_patches_x = img->nx / patch_size / (n_merge > 0 ? n_merge : 1);
                int n_patches_y = img->ny / patch_size / (n_merge > 0 ? n_merge : 1);
                if (ctx->model.token_embd_img_break) {
                    n_patches = n_patches_y * n_patches_x + n_patches_y - 1; // + one [IMG_BREAK] per row, except the last row
                } else {
                    n_patches = n_patches_y * n_patches_x;
                }
            } break;
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_QWEN2A:
        case PROJECTOR_TYPE_MERALION:
        case PROJECTOR_TYPE_MUSIC_FLAMINGO:
            {
                n_patches = img->nx;

                const int proj_stack_factor = ctx->model.hparams.proj_stack_factor;
                if (ctx->model.audio_has_stack_frames()) {
                    GGML_ASSERT(proj_stack_factor > 0);
                    const int n_len = CLIP_ALIGN(n_patches, proj_stack_factor);
                    n_patches = n_len / proj_stack_factor;
                }

                // whisper downscales input token by half after conv1d
                n_patches /= 2;

                if (ctx->model.audio_has_avgpool()) {
                    // divide by 2 because of nn.AvgPool1d(2, stride=2)
                    n_patches /= 2;
                }
            } break;
        case PROJECTOR_TYPE_QWEN3A:
            {
                // 3x stride-2 conv2d: each step is floor((n-1)/2)+1
                int n = img->nx;
                n = (n - 1) / 2 + 1;
                n = (n - 1) / 2 + 1;
                n = (n - 1) / 2 + 1;
                n_patches = n;
            } break;
        case PROJECTOR_TYPE_GLMA:
            {
                n_patches = img->nx;
                // whisper downscales input token by half after conv1d
                n_patches /= 2;
                // reshape by merge_factor
                n_patches /= ctx->model.hparams.proj_stack_factor;
                // for BOI and EOI token embeddings
                n_patches += 2;
            } break;
        case PROJECTOR_TYPE_COGVLM:
            {
                n_patches += 2; // for BOI and EOI token embeddings
            } break;
        case PROJECTOR_TYPE_DEEPSEEKOCR:
        {
            // SAM encoder applies two stride-2 convolutions (net_2 and net_3)
            // which reduces spatial dimensions by 4x in each direction (16x total)
            // E.g., 64x64 -> 16x16 patches
            n_patches /= 16;

            // build_global_local_features adds image newlines and view separator
            // Formula: h*(w+1) + 1 where h = w = sqrt(n_patches)
            int h = static_cast<int>(std::sqrt(static_cast<float>(n_patches)));
            n_patches = h * (h + 1) + 1;
        } break;
        case PROJECTOR_TYPE_HUNYUANOCR:
        case PROJECTOR_TYPE_HUNYUANVL:
            {
                int merge = ctx->model.hparams.n_merge;
                int ow = (img->nx / patch_size) / merge;
                int oh = (img->ny / patch_size) / merge;
                n_patches = (ow + 1) * oh + 2;
            } break;
        case PROJECTOR_TYPE_LFM2A:
            {
                n_patches = ((((img->nx + 1) / 2) + 1) / 2 + 1) / 2;
            } break;
        case PROJECTOR_TYPE_GEMMA4A:
            {
                // Two Conv2D stride-2: O = floor((I + 2p - k) / s) + 1, p=1, k=3, s=2
                // O = floor((I - 1) / 2) + 1
                int n = img->nx;
                for (int i = 0; i < 2; i++) {
                    n = (n - 1) / 2 + 1;
                }
                n_patches = n;
            } break;
        case PROJECTOR_TYPE_GRANITE_SPEECH:
            {
                const int ws = ctx->model.hparams.audio_proj_window_size;
                const int ds = ctx->model.hparams.audio_proj_downsample_rate;
                n_patches = ((img->nx + ws - 1) / ws) * (ws / ds);
            } break;
        default:
            GGML_ABORT("unsupported projector type");
    }

    return n_patches;
}

bool clip_image_encode(struct clip_ctx * ctx, const int n_threads, clip_image_f32 * img, float * vec) {
    clip_image_f32_batch imgs;
    clip_image_f32_ptr img_copy(clip_image_f32_init());
    *img_copy = *img;
    imgs.entries.push_back(std::move(img_copy));

    return clip_image_batch_encode(ctx, n_threads, &imgs, vec);
}

bool clip_image_batch_encode(clip_ctx * ctx, const int n_threads, const clip_image_f32_batch * imgs_c_ptr, float * vec) {
    const clip_image_f32_batch & imgs = *imgs_c_ptr;
    int batch_size = imgs.entries.size();

    // TODO @ngxson : implement batch size > 1 as a loop
    //                we don't need true batching support because the cgraph will gonna be big anyway
    if (batch_size != 1) {
        return false; // only support batch size of 1
    }

    // if buffers are not allocated, we need to do a warmup run to allocate them
    if (!ctx->is_allocated) {
        clip_model_loader::warmup(*ctx, *imgs_c_ptr);
    }

    // build the inference graph
    ggml_backend_sched_reset(ctx->sched.get());
    ggml_cgraph * gf = clip_image_build_graph(ctx, imgs);
    ggml_backend_sched_alloc_graph(ctx->sched.get(), gf);

    // set inputs
    const auto & model   = ctx->model;
    const auto & hparams = model.hparams;

    const int image_size_width  = imgs.entries[0]->nx;
    const int image_size_height = imgs.entries[0]->ny;

    const int patch_size    = hparams.patch_size;
    const int num_patches   = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int n_pos = num_patches + (model.class_embedding ? 1 : 0);
    const int pos_w = image_size_width  / patch_size;
    const int pos_h = image_size_height / patch_size;


    auto get_inp_tensor = [&gf](const char * name) {
        ggml_tensor * inp = ggml_graph_get_tensor(gf, name);
        if (inp == nullptr) {
            GGML_ABORT("Failed to get tensor %s", name);
        }
        if (!(inp->flags & GGML_TENSOR_FLAG_INPUT)) {
            GGML_ABORT("Tensor %s is not an input tensor", name);
        }
        return inp;
    };

    auto set_input_f32 = [&get_inp_tensor](const char * name, std::vector<float> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    auto set_input_i32 = [&get_inp_tensor](const char * name, std::vector<int32_t> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    // set input pixel values
    if (!imgs.is_audio) {
        size_t nelem = 0;
        for (const auto & img : imgs.entries) {
            nelem += img->nx * img->ny * 3;
        }
        std::vector<float> inp_raw(nelem);

        // layout of data (note: the channel dim is unrolled to better visualize the layout):
        //
        // ┌──W──┐
        // │     H │  channel = R
        // ├─────┤ │
        // │     H │  channel = G
        // ├─────┤ │
        // │     H │  channel = B
        // └─────┘ │
        //   ──────┘ x B

        for (size_t i = 0; i < imgs.entries.size(); i++) {
            const int nx = imgs.entries[i]->nx;
            const int ny = imgs.entries[i]->ny;
            const int n = nx * ny;

            for (int b = 0; b < batch_size; b++) {
                float * batch_entry = inp_raw.data() + b * (3*n);
                for (int y = 0; y < ny; y++) {
                    for (int x = 0; x < nx; x++) {
                        size_t base_src = 3*(y * nx + x); // idx of the first channel
                        size_t base_dst =    y * nx + x;  // idx of the first channel
                        batch_entry[      base_dst] = imgs.entries[b]->buf[base_src    ];
                        batch_entry[1*n + base_dst] = imgs.entries[b]->buf[base_src + 1];
                        batch_entry[2*n + base_dst] = imgs.entries[b]->buf[base_src + 2];
                    }
                }
            }
        }
        set_input_f32("inp_raw", inp_raw);

    } else {
        // audio input
        GGML_ASSERT(imgs.entries.size() == 1);
        const auto & mel_inp = imgs.entries[0];
        const int n_step = mel_inp->nx;
        const int n_mel  = mel_inp->ny;
        std::vector<float> inp_raw(n_step * n_mel);
        std::memcpy(inp_raw.data(), mel_inp->buf.data(), n_step * n_mel * sizeof(float));
        set_input_f32("inp_raw", inp_raw);
    }

    // set input per projector
    switch (ctx->model.proj_type) {
        case PROJECTOR_TYPE_MINICPMV:
            {
                // inspired from siglip:
                //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit
                //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit/blob/d66538faeba44480d0bfaa42145eef26f9423199/modeling_siglip.py#L316
                std::vector<int32_t> positions(pos_h * pos_w);
                int bucket_coords_h[1024];
                int bucket_coords_w[1024];
                for (int i = 0; i < pos_h; i++){
                    bucket_coords_h[i] = std::floor(70.0*i/pos_h);
                }
                for (int i = 0; i < pos_w; i++){
                    bucket_coords_w[i] = std::floor(70.0*i/pos_w);
                }
                for (int i = 0, id = 0; i < pos_h; i++){
                    for (int j = 0; j < pos_w; j++){
                        positions[id++] = bucket_coords_h[i]*70 + bucket_coords_w[j];
                    }
                }
                set_input_i32("positions", positions);

                // inputs for resampler projector
                // set the 2D positions (using float for sinusoidal embedding)
                int n_patches_per_col = image_size_width / patch_size;
                std::vector<float> pos_data(n_pos);
                // dimension H
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = static_cast<float>(i / n_patches_per_col);
                }
                set_input_f32("pos_h", pos_data);
                // dimension W
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = static_cast<float>(i % n_patches_per_col);
                }
                set_input_f32("pos_w", pos_data);
                // base frequency omega
                const float base_freq   = 10000.0f;
                const int   n_embd_proj = clip_n_mmproj_embd(ctx);
                std::vector<float> omega(n_embd_proj / 4);
                for (int i = 0; i < n_embd_proj / 4; ++i) {
                    omega[i] = 1.0f / std::pow(base_freq, static_cast<float>(i) / (n_embd_proj / 4));
                }
                set_input_f32("omega", omega);
            } break;
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_GLM4V:
            {
                const int merge_ratio = hparams.n_merge;
                const int pw = image_size_width  / patch_size;
                const int ph = image_size_height / patch_size;
                std::vector<int> positions(n_pos * 4);
                int ptr = 0;
                for (int y = 0; y < ph; y += merge_ratio) {
                    for (int x = 0; x < pw; x += merge_ratio) {
                        for (int dy = 0; dy < 2; dy++) {
                            for (int dx = 0; dx < 2; dx++) {
                                positions[                  ptr] = y + dy;
                                positions[    num_patches + ptr] = x + dx;
                                positions[2 * num_patches + ptr] = y + dy;
                                positions[3 * num_patches + ptr] = x + dx;
                                ptr++;
                            }
                        }
                    }
                }

                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_STEP3VL:
            {
                std::vector<int32_t> pos_data(n_pos);
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i / pos_w;
                }
                set_input_i32("pos_h", pos_data);
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i % pos_w;
                }
                set_input_i32("pos_w", pos_data);
            } break;
        case PROJECTOR_TYPE_PADDLEOCR:
            {
                const int merge_ratio = hparams.n_merge;
                const int pw = image_size_width  / patch_size;
                const int ph = image_size_height / patch_size;
                std::vector<int> positions(n_pos * 4);
                int ptr = 0;
                // NOTE: same as Qwen-VL, but x and y are swapped
                for (int y = 0; y < ph; y += merge_ratio) {
                    for (int dy = 0; dy < 2; dy++) {
                        for (int x = 0; x < pw; x += merge_ratio) {
                            for (int dx = 0; dx < 2; dx++) {
                                positions[                  ptr] = y + dy;
                                positions[    num_patches + ptr] = x + dx;
                                positions[2 * num_patches + ptr] = y + dy;
                                positions[3 * num_patches + ptr] = x + dx;
                                ptr++;
                            }
                        }
                    }
                }

                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_DOTS_OCR:
            {
                const int pw = image_size_width / patch_size;
                const int ph = image_size_height / patch_size;
                const int n_pos = ph * pw;
                std::vector<int> positions(n_pos * 4);
                int ptr = 0;

                // flat layout: [h, w, h, w] for each patch
                // patches are in raster order (matching conv2d output)
                for (int y = 0; y < ph; y++) {
                    for (int x = 0; x < pw; x++) {
                        positions[          ptr] = y;
                        positions[  n_pos + ptr] = x;
                        positions[2*n_pos + ptr] = y;
                        positions[3*n_pos + ptr] = x;
                        ptr++;
                    }
                }

                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_YOUTUVL:
            {
                // pw * ph = number of tokens output by ViT after apply patch merger
                // ipw * ipw = number of vision token been processed inside ViT
                const bool use_window_attn = ctx->model.proj_type == PROJECTOR_TYPE_QWEN25VL ? hparams.n_wa_pattern > 0 : !hparams.wa_layer_indexes.empty();
                const int merge_ratio = 2;
                const int pw  = image_size_width  / patch_size / merge_ratio;
                const int ph  = image_size_height / patch_size / merge_ratio;
                const int ipw = image_size_width  / patch_size;
                const int iph = image_size_height / patch_size;

                std::vector<int> idx    (ph * pw);
                std::vector<int> inv_idx(ph * pw);

                if (use_window_attn) {
                    const int attn_window_size = hparams.attn_window_size > 0 ? hparams.attn_window_size : 112;
                    const int grid_window = attn_window_size / patch_size / merge_ratio;
                    int dst = 0;
                    // [num_vision_tokens, num_vision_tokens] attention mask tensor
                    std::vector<float> mask(pow(ipw * iph, 2), std::numeric_limits<float>::lowest());
                    int mask_row = 0;

                    for (int y = 0; y < ph; y += grid_window) {
                        for (int x = 0; x < pw; x += grid_window) {
                            const int win_h = std::min(grid_window, ph - y);
                            const int win_w = std::min(grid_window, pw - x);
                            const int dst_0 = dst;
                            // group all tokens belong to the same window togather (to a continue range)
                            for (int dy = 0; dy < win_h; dy++) {
                                for (int dx = 0; dx < win_w; dx++) {
                                    const int src = (y + dy) * pw + (x + dx);
                                    GGML_ASSERT(src < (int)idx.size());
                                    GGML_ASSERT(dst < (int)inv_idx.size());
                                    idx    [src] = dst;
                                    inv_idx[dst] = src;
                                    dst++;
                                }
                            }

                            for (int r=0; r < win_h * win_w * merge_ratio * merge_ratio; r++) {
                                int row_offset = mask_row * (ipw * iph);
                                std::fill(
                                    mask.begin() + row_offset + (dst_0 * merge_ratio * merge_ratio),
                                    mask.begin() + row_offset + (dst   * merge_ratio * merge_ratio),
                                    0.0);
                                mask_row++;
                            }
                        }
                    }

                    set_input_i32("window_idx",     idx);
                    set_input_i32("inv_window_idx", inv_idx);
                    set_input_f32("window_mask",    mask);
                } else {
                    for (int i = 0; i < ph * pw; i++) {
                        idx[i] = i;
                    }
                }

                const int mpow = merge_ratio * merge_ratio;
                std::vector<int> positions(n_pos * 4);

                int ptr = 0;
                for (int y = 0; y < iph; y += merge_ratio) {
                    for (int x = 0; x < ipw; x += merge_ratio) {
                        for (int dy = 0; dy < 2; dy++) {
                            for (int dx = 0; dx < 2; dx++) {
                                auto remap = idx[ptr / mpow];
                                remap = (remap * mpow) + (ptr % mpow);

                                positions[                  remap] = y + dy;
                                positions[    num_patches + remap] = x + dx;
                                positions[2 * num_patches + remap] = y + dy;
                                positions[3 * num_patches + remap] = x + dx;
                                ptr++;
                            }
                        }
                    }
                }

                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_KIMIVL:
        case PROJECTOR_TYPE_KIMIK25:
        case PROJECTOR_TYPE_LIGHTONOCR:
            {
                // set the 2D positions
                int n_patches_per_col = image_size_width / patch_size;
                std::vector<int> pos_data(n_pos);
                // dimension H
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i / n_patches_per_col;
                }
                set_input_i32("pos_h", pos_data);
                // dimension W
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i % n_patches_per_col;
                }
                set_input_i32("pos_w", pos_data);
            } break;
        case PROJECTOR_TYPE_GLM_EDGE:
        {
            // llava and other models
            std::vector<int32_t> positions(n_pos);
            for (int i = 0; i < n_pos; i++) {
                positions[i] = i;
            }
            set_input_i32("positions", positions);
        } break;
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
            {
                // llava and other models
                std::vector<int32_t> positions(n_pos);
                for (int i = 0; i < n_pos; i++) {
                    positions[i] = i;
                }
                set_input_i32("positions", positions);

                // The patches vector is used to get rows to index into the embeds with;
                // we should skip dim 0 only if we have CLS to avoid going out of bounds
                // when retrieving the rows.
                int patch_offset = model.class_embedding ? 1 : 0;
                std::vector<int32_t> patches(num_patches);
                for (int i = 0; i < num_patches; i++) {
                    patches[i] = i + patch_offset;
                }
                set_input_i32("patches", patches);
            } break;
        case PROJECTOR_TYPE_GEMMA4V:
            {
                // set (col, row) patch positions for learned positional embedding
                const int n_cols = image_size_width  / patch_size;
                std::vector<int> pos_x(num_patches), pos_y(num_patches);
                for (int i = 0; i < num_patches; i++) {
                    pos_x[i] = i % n_cols;
                    pos_y[i] = i / n_cols;
                }
                set_input_i32("pos_x", pos_x);
                set_input_i32("pos_y", pos_y);
            } break;
        case PROJECTOR_TYPE_DEEPSEEKOCR:
            {
                GGML_ASSERT(pos_w == pos_h);

                const int window = hparams.attn_window_size;
                const int pos = pos_w;
                std::vector<int32_t> rel_pos_indices_local(window * window);
                std::vector<int32_t> rel_pos_indices_global(pos * pos);

                for (int q = 0; q < window; q++) {
                    for (int k = 0; k < window; k++) {
                        rel_pos_indices_local[q * window + k] = q - k + window - 1;
                    }
                }

                for (int q = 0; q < pos; q++) {
                    for (int k = 0; k < pos; k++) {
                        rel_pos_indices_global[q * pos + k] = q - k + pos - 1;
                    }
                }

                set_input_i32("rel_pos_indices_local", rel_pos_indices_local);
                set_input_i32("rel_pos_indices_global", rel_pos_indices_global);
            } break;
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_GEMMA3NV:
        case PROJECTOR_TYPE_IDEFICS3:
        case PROJECTOR_TYPE_INTERNVL:
        case PROJECTOR_TYPE_NEMOTRON_V2_VL:
        case PROJECTOR_TYPE_QWEN2A:
        case PROJECTOR_TYPE_QWEN3A:
        case PROJECTOR_TYPE_GLMA:
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_MERALION:
        case PROJECTOR_TYPE_MUSIC_FLAMINGO:
        case PROJECTOR_TYPE_JANUS_PRO:
        case PROJECTOR_TYPE_PHI4:
        case PROJECTOR_TYPE_COGVLM:
        case PROJECTOR_TYPE_HUNYUANOCR:
        case PROJECTOR_TYPE_YASA2:
            {
                // do nothing
            } break;
        case PROJECTOR_TYPE_HUNYUANVL:
            {
                // Compute the HunyuanVL 2D position embedding on CPU (with the
                // custom sf=(target+0.1)/n_grid bilinear sampling that the
                // reference implementation uses) and upload it to the graph
                // input declared in clip_graph_hunyuanocr::build().
                GGML_ASSERT(model.position_embeddings != nullptr);
                ggml_tensor * src_t   = model.position_embeddings;
                const int64_t n_embd  = src_t->ne[0];
                const int64_t n_pos   = src_t->ne[1];            // = n_grid * n_grid
                const int     n_grid  = (int)std::lround(std::sqrt((double)n_pos));
                GGML_ASSERT((int64_t)n_grid * n_grid == n_pos);
                const int     out_w   = pos_w;                    // pw
                const int     out_h   = pos_h;                    // ph

                // Pull weight to host.
                std::vector<float> src(n_embd * n_pos);
                ggml_backend_tensor_get(src_t, src.data(), 0, ggml_nbytes(src_t));

                // Output layout matches ggml_new_tensor_2d(F32, n_embd, out_h*out_w):
                //   ne[0] = n_embd (fastest), ne[1] = out_h*out_w
                //   dst[(y*out_w + x) * n_embd + c]
                std::vector<float> dst((size_t)n_embd * out_h * out_w);

                const float sx = (float)(out_w + 0.1f) / (float)n_grid;
                const float sy = (float)(out_h + 0.1f) / (float)n_grid;

                for (int y = 0; y < out_h; ++y) {
                    // Match ggml_compute_forward_upscale_f32 pixel-center
                    // convention (align_corners=False): src_y = (y+0.5)/sy - 0.5.
                    const float fy = ((float)y + 0.5f) / sy - 0.5f;
                    int y0 = (int)std::floor(fy);
                    int y1 = y0 + 1;
                    y0 = std::clamp(y0, 0, n_grid - 1);
                    y1 = std::clamp(y1, 0, n_grid - 1);
                    float wy1 = std::clamp(fy - (float)y0, 0.0f, 1.0f);
                    const float wy0 = 1.0f - wy1;
                    for (int x = 0; x < out_w; ++x) {
                        const float fx = ((float)x + 0.5f) / sx - 0.5f;
                        int x0 = (int)std::floor(fx);
                        int x1 = x0 + 1;
                        x0 = std::clamp(x0, 0, n_grid - 1);
                        x1 = std::clamp(x1, 0, n_grid - 1);
                        float wx1 = std::clamp(fx - (float)x0, 0.0f, 1.0f);
                        const float wx0 = 1.0f - wx1;

                        const float w00 = wy0 * wx0;
                        const float w01 = wy0 * wx1;
                        const float w10 = wy1 * wx0;
                        const float w11 = wy1 * wx1;

                        const float * s00 = &src[((size_t)y0 * n_grid + x0) * n_embd];
                        const float * s01 = &src[((size_t)y0 * n_grid + x1) * n_embd];
                        const float * s10 = &src[((size_t)y1 * n_grid + x0) * n_embd];
                        const float * s11 = &src[((size_t)y1 * n_grid + x1) * n_embd];
                        float * d         = &dst[((size_t)y * out_w + x) * n_embd];
                        for (int c = 0; c < n_embd; ++c) {
                            d[c] = w00 * s00[c] + w01 * s01[c] + w10 * s10[c] + w11 * s11[c];
                        }
                    }
                }

                set_input_f32("hunyuanvl_pos_embd", dst);
            } break;
        case PROJECTOR_TYPE_LLAMA4:
            {
                // set the 2D positions
                int n_patches_per_col = image_size_width / patch_size;
                std::vector<int> pos_data(num_patches + 1, 0); // +1 for the [CLS] token
                // last pos is always kept 0, it's for CLS
                // dimension H
                for (int i = 0; i < num_patches; i++) {
                    pos_data[i] = (i / n_patches_per_col) + 1;
                }
                set_input_i32("pos_h", pos_data);
                // dimension W
                for (int i = 0; i < num_patches; i++) {
                    pos_data[i] = (i % n_patches_per_col) + 1;
                }
                set_input_i32("pos_w", pos_data);
            } break;
        case PROJECTOR_TYPE_GEMMA4A:
            {
                GGML_ASSERT(imgs.entries.size() == 1);
                const auto & img0 = imgs.entries.front();
                // Compute n_pos matching SSCP output: two stride-2 convs
                int n_pos = img0->nx;
                for (int i = 0; i < 2; i++) { n_pos = (n_pos - 1) / 2 + 1; }

                // Chunked local attention: blocked causal mask and RPE
                const int chunk_size   = 12;
                const int max_past     = 12;
                const int context_size = chunk_size + max_past;
                const int num_blocks   = (n_pos + chunk_size - 1) / chunk_size;

                // Blocked causal attention mask: [context_size, chunk_size, num_blocks]
                {
                    std::vector<float> mask(context_size * chunk_size * num_blocks, -1e9f);
                    for (int b = 0; b < num_blocks; b++) {
                        for (int q = 0; q < chunk_size; q++) {
                            int gq = b * chunk_size + q;
                            for (int k = 0; k < context_size; k++) {
                                int gk = b * chunk_size - max_past + k;
                                if (gq < n_pos && gk >= 0 && gk < n_pos && gk <= gq && (gq - gk) < max_past) {
                                    mask[k + q * context_size + b * context_size * chunk_size] = 0.0f;
                                }
                            }
                        }
                    }
                    set_input_f32("kq_mask", mask);
                }

                // Sinusoidal RPE: 13 positions [12, 11, ..., 0]
                {
                    const int n_embd = ctx->model.hparams.n_embd;
                    const int num_timescales = n_embd / 2;
                    const float log_timescale_increment = logf(10000.0f) / std::max(num_timescales - 1, 1);
                    const int rpe_len = max_past + 1;
                    std::vector<float> pos_emb(n_embd * rpe_len, 0.0f);
                    for (int p = 0; p < rpe_len; p++) {
                        float position = (float)(max_past - p);
                        for (int i = 0; i < num_timescales; i++) {
                            float inv_ts = expf(-(float)i * log_timescale_increment);
                            float scaled = position * inv_ts;
                            pos_emb[p * n_embd + i]                 = sinf(scaled);
                            pos_emb[p * n_embd + i + num_timescales] = cosf(scaled);
                        }
                    }
                    set_input_f32("pos_emb", pos_emb);
                }
            } break;
        case PROJECTOR_TYPE_LFM2A:
            {
                GGML_ASSERT(imgs.entries.size() == 1);
                const auto n_frames = clip_n_output_tokens(ctx, imgs.entries.front().get());

                auto d_model = 512;
                auto seq_len = n_frames * 2 - 1;
                std::vector<float> pos_emb(d_model*seq_len);
                std::vector<double> inv_freq(d_model / 2);
                for (size_t i = 0; i < inv_freq.size(); ++i) {
                    inv_freq[i] = std::exp(-(std::log(10000.0) / (float)d_model) * (2.0f * (float)(i)));
                }
                for (int64_t pos = 0; pos < seq_len; ++pos) {
                    for (size_t i = 0; i < inv_freq.size(); ++i) {
                        const float ang = (n_frames - pos - 1) * inv_freq[i];
                        pos_emb[pos*d_model + 2*i + 0] = sinf(ang);  // even
                        pos_emb[pos*d_model + 2*i + 1] = cosf(ang);  // odd
                    }
                }
                set_input_f32("pos_emb", pos_emb);
            } break;
        case PROJECTOR_TYPE_GRANITE_SPEECH:
            {
                const int context_size = ctx->model.hparams.audio_chunk_size;
                const int max_pos_emb  = ctx->model.hparams.audio_max_pos_emb;

                std::vector<int32_t> dists(context_size * context_size);
                for (int i = 0; i < context_size; i++) {
                    for (int j = 0; j < context_size; j++) {
                        int d = i - j;
                        if (d < -context_size) d = -context_size;
                        if (d >  context_size) d =  context_size;
                        dists[i * context_size + j] = d + max_pos_emb;
                    }
                }
                set_input_i32("attn_dists", dists);

                const int n_frames   = image_size_width;
                const int remainder  = n_frames % context_size;
                if (remainder > 0) {
                    const int num_blocks = (n_frames + context_size - 1) / context_size;
                    std::vector<float> mask(context_size * context_size * num_blocks, 0.0f);
                    const float neg_inf = -INFINITY;
                    const int last_block_offset = (num_blocks - 1) * context_size * context_size;
                    for (int q = 0; q < context_size; q++) {
                        for (int k = 0; k < context_size; k++) {
                            if (q >= remainder || k >= remainder) {
                                mask[last_block_offset + q * context_size + k] = neg_inf;
                            }
                        }
                    }
                    set_input_f32("attn_mask", mask);
                }
            } break;
        default:
            GGML_ABORT("Unknown projector type");
    }

    // ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
    ggml_backend_dev_t dev = ggml_backend_get_device(ctx->backend_cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) {
        auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            ggml_backend_set_n_threads_fn(ctx->backend_cpu, n_threads);
        }
    }

    auto status = ggml_backend_sched_graph_compute(ctx->sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("%s: ggml_backend_sched_graph_compute failed with error %d\n", __func__, status);
        return false;
    }

    // the last node is the embedding tensor
    ggml_tensor * embeddings = ggml_graph_node(gf, -1);

    // sanity check (only support batch size of 1 for now)
    const int n_tokens_out = embeddings->ne[1];
    const int expected_n_tokens_out = clip_n_output_tokens(ctx, imgs.entries[0].get());
    if (n_tokens_out != expected_n_tokens_out) {
        LOG_ERR("%s: expected output %d tokens, got %d\n", __func__, expected_n_tokens_out, n_tokens_out);
        GGML_ABORT("Invalid number of output tokens");
    }

    // copy the embeddings to the location passed by the user
    if (vec != nullptr) {
        ggml_backend_tensor_get(embeddings, vec, 0, ggml_nbytes(embeddings));
    }

    // Debug: dump final embeddings if MTMD_DEBUG_EMBEDDINGS is set
    if (ctx->debug_output_embeddings) {
        const int64_t n_embd = embeddings->ne[0];
        const int64_t n_tokens = embeddings->ne[1];
        std::vector<float> emb_data(n_embd * n_tokens);
        ggml_backend_tensor_get(embeddings, emb_data.data(), 0, ggml_nbytes(embeddings));

        LOG_INF("\n=== MTMD_DEBUG_EMBEDDINGS ===\n");
        LOG_INF("Shape: [%lld, %lld]\n", (long long)n_embd, (long long)n_tokens);

        // Print first few values of first token
        LOG_INF("Token 0 (first 16 values): ");
        for (int i = 0; i < std::min((int64_t)16, n_embd); i++) {
            LOG_INF("%.6f ", emb_data[i]);
        }
        LOG_INF("\n");

        // Print last few values of first token
        if (n_embd > 16) {
            LOG_INF("Token 0 (last 16 values):  ");
            for (int64_t i = n_embd - 16; i < n_embd; i++) {
                LOG_INF("%.6f ", emb_data[i]);
            }
            LOG_INF("\n");
        }

        // Compute and print statistics
        float sum = 0.0f, sum_sq = 0.0f, min_val = emb_data[0], max_val = emb_data[0];
        for (size_t i = 0; i < emb_data.size(); i++) {
            sum += emb_data[i];
            sum_sq += emb_data[i] * emb_data[i];
            min_val = std::min(min_val, emb_data[i]);
            max_val = std::max(max_val, emb_data[i]);
        }
        float mean = sum / emb_data.size();
        float variance = (sum_sq / emb_data.size()) - (mean * mean);
        LOG_INF("Stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f, sum=%.6f\n",
                mean, sqrtf(variance), min_val, max_val, sum);
        LOG_INF("=== END MTMD_DEBUG_EMBEDDINGS ===\n\n");
    }

    return true;
}

int clip_n_mmproj_embd(const struct clip_ctx * ctx) {
    switch (ctx->model.proj_type) {
        case PROJECTOR_TYPE_LDP:
            return ctx->model.mm_model_block_1_block_2_1_b->ne[0];
        case PROJECTOR_TYPE_LDPV2:
            return ctx->model.mm_model_peg_0_b->ne[0];
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_PHI4:
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_LIGHTONOCR:
        case PROJECTOR_TYPE_DOTS_OCR:
            return ctx->model.mm_2_w->ne[1];
        case PROJECTOR_TYPE_MLP_NORM:
            return ctx->model.mm_3_b->ne[0];
        case PROJECTOR_TYPE_MINICPMV:
            return ctx->model.mm_model_proj->ne[0];
        case PROJECTOR_TYPE_GLM_EDGE:
            return ctx->model.mm_model_mlp_3_w->ne[1];
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_JANUS_PRO:
        case PROJECTOR_TYPE_YOUTUVL:
            return ctx->model.mm_1_b->ne[0];
        case PROJECTOR_TYPE_QWEN3VL:
            // main path + deepstack paths
            return ctx->model.mm_1_b->ne[0] * (1 + ctx->model.n_deepstack_layers);
        case PROJECTOR_TYPE_STEP3VL:
            return ctx->model.mm_model_proj->ne[1];
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_GEMMA3NV:
            return ctx->model.mm_input_proj_w->ne[0];
        case PROJECTOR_TYPE_GEMMA4V:
            return ctx->model.mm_input_proj_w->ne[1];
        case PROJECTOR_TYPE_IDEFICS3:
            return ctx->model.mm_fc_w->ne[1];
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_MUSIC_FLAMINGO:
            return ctx->model.mm_2_w->ne[1];
        case PROJECTOR_TYPE_MERALION:
            return ctx->model.mm_3_w->ne[1]; // out_proj output dim
        case PROJECTOR_TYPE_INTERNVL:
        case PROJECTOR_TYPE_NEMOTRON_V2_VL:
            return ctx->model.mm_3_w->ne[1];
        case PROJECTOR_TYPE_LLAMA4:
            return ctx->model.mm_model_proj->ne[1];
        case PROJECTOR_TYPE_QWEN2A:
            return ctx->model.mm_fc_w->ne[1];
        case PROJECTOR_TYPE_QWEN3A:
            return ctx->model.mm_2_w->ne[1];
        case PROJECTOR_TYPE_GLMA:
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_KIMIVL:
        case PROJECTOR_TYPE_PADDLEOCR:
        case PROJECTOR_TYPE_KIMIK25:
        case PROJECTOR_TYPE_YASA2:
            return ctx->model.mm_2_w->ne[1];
        case PROJECTOR_TYPE_HUNYUANOCR:
        case PROJECTOR_TYPE_HUNYUANVL:
            return ctx->model.mm_model_proj->ne[1];
        case PROJECTOR_TYPE_COGVLM:
            return ctx->model.mm_4h_to_h_w->ne[1];
        case PROJECTOR_TYPE_DEEPSEEKOCR:
            return ctx->model.mm_fc_w->ne[1];
        case PROJECTOR_TYPE_LFM2A:
            return ctx->model.position_embeddings->ne[0];
        case PROJECTOR_TYPE_GEMMA4A:
            return ctx->model.hparams.projection_dim;
        case PROJECTOR_TYPE_GRANITE_SPEECH:
            return ctx->model.qf_proj_linear_w->ne[1];
        case PROJECTOR_TYPE_GLM4V:
            return ctx->model.mm_ffn_down_w->ne[1];
        default:
            GGML_ABORT("Unknown projector type");
    }
}

int clip_is_minicpmv(const struct clip_ctx * ctx) {
    // TODO: remove this function
    if (ctx->proj_type() == PROJECTOR_TYPE_MINICPMV) {
        return ctx->model.hparams.minicpmv_version;
    }
    return 0;
}

bool clip_is_glm(const struct clip_ctx * ctx) {
    // TODO: remove this function
    return ctx->proj_type() == PROJECTOR_TYPE_GLM_EDGE;
}

bool clip_is_llava(const struct clip_ctx * ctx) {
    return ctx->model.hparams.has_llava_projector;
}

bool clip_has_vision_encoder(const struct clip_ctx * ctx) {
    return ctx->model.modality == CLIP_MODALITY_VISION;
}

bool clip_has_audio_encoder(const struct clip_ctx * ctx) {
    return ctx->model.modality == CLIP_MODALITY_AUDIO;
}

bool clip_has_whisper_encoder(const struct clip_ctx * ctx) {
    switch (ctx->proj_type()) {
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_QWEN2A:
        case PROJECTOR_TYPE_QWEN3A:
        case PROJECTOR_TYPE_GLMA:
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_MERALION:
        case PROJECTOR_TYPE_MUSIC_FLAMINGO:
            return true;
        default:
            return false;
    }
}

bool clip_encode_float_image (struct clip_ctx * ctx, int n_threads, float * img, int h, int w, float * vec) {
    clip_image_f32 clip_img;
    clip_img.buf.resize(h * w * 3);
    for (int i = 0; i < h*w*3; i++)
    {
        clip_img.buf[i] = img[i];
    }
    clip_img.nx = w;
    clip_img.ny = h;
    clip_image_encode(ctx, n_threads, &clip_img, vec);
    return true;
}

//
// API used internally with mtmd
//

projector_type clip_get_projector_type(const struct clip_ctx * ctx) {
    return ctx->proj_type();
}

void clip_image_f32_batch_add_mel(struct clip_image_f32_batch * batch, int n_mel, int n_frames, float * mel) {
    clip_image_f32 * audio = new clip_image_f32;
    audio->nx = n_frames;
    audio->ny = n_mel;
    audio->buf.resize(n_frames * n_mel);
    std::memcpy(audio->buf.data(), mel, n_frames * n_mel * sizeof(float));

    batch->entries.push_back(clip_image_f32_ptr(audio));
    batch->is_audio = true;
}

const clip_hparams * clip_get_hparams(const struct clip_ctx * ctx) {
    return &ctx->model.hparams;
}

//
// API for debugging
//

void clip_set_debug_output_embeddings(clip_ctx * ctx, bool enable) {
    ctx->debug_output_embeddings = enable;
}
