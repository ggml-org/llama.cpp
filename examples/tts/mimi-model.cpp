#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "common.h"
#include "mimi-model.h"

#include <limits.h>
#include <vector>
#include <cinttypes>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <float.h>
#include <cmath>
#include <cstdarg>
#include <functional>
#include <array>

/**
 * Implementation of Kyutai's Mimi model using GGML.
 * Based on this research: https://github.com/ngxson/ggml-easy/blob/master/demo/kyutai-mimi.cpp
 *
 * NOTE: only decoder is working for now.
 *
 * Background:
 * - The audio codes can be generated using any Mimi-based model, for example: Moshi, Hibiki, Sesame, etc
 * - Audio codes must be in the order: N semantic codes followed by (N*31) acoustic codes
 *   (In other words, input matrix has shape 32 cols x N rows)
 *
 * How it works?
 * 1. Audio code passed to RVQ (mimi_residual_vector_quantizer) to get the latent code
 * 2. The latent code is passed to a mimi_conv_transpose_1d (depthwise) to upscale
 * 3. The upscaled code is passed to transformer, it converts N frames to N frames
 * 4. The output embeddings is then passed to SEANet (mimi_encoder_decoder) to get the final waveform
 * 5. Waveform is written to a file
 */

// copied from https://huggingface.co/kyutai/mimi/blob/main/config.json
struct mimi_config_t {
    bool causal = true;
    int sample_rate = 24000;
    int max_position_embeddings = 8000;
    int num_hidden_layers = 8;
    int n_embd = 512;
    int n_ffn = 2048;
    int n_head = 8;
    int n_head_kv = 8;
    int n_rot = 64;
    float norm_eps = 1e-5;
    float rope_theta = 10000.0f;
    int sliding_window = 250;
    std::array<int, 4> upsampling_ratio   = {8, 6, 5, 4};
    std::array<int, 4> downsampling_ratio = {4, 5, 6, 8}; // reverse of upsampling_ratio
    // vector quantizer
    float frame_rate = 12.5;
    int audio_channels = 1;
    int codebook_size = 2048;
    int codebook_dim = 256;
    int n_semantic_components = 1;
    int n_acoustic_components = 31;
    // decode
    float trim_right_ratio = 1.0f;
    int n_codes_per_frame = (sliding_window / 2) * (n_semantic_components + n_acoustic_components);
} mimi_config;

// Adapted from https://github.com/ngxson/ggml-easy/blob/master/ggml-easy.h
struct mimi_ggml_ctx {
    gguf_context * ctx_gguf = nullptr;
    ggml_context * ctx_data = nullptr;
    ggml_context * ctx_gf   = nullptr;

    // CPU-only for now, as many kernels are missing and we actually get less performance with GPU
    ggml_backend_t backend     = nullptr;
    ggml_backend_buffer_t buf  = nullptr;
    ggml_backend_sched_ptr sched;

    ggml_cgraph * gf = nullptr;
    std::vector<uint8_t> buf_compute_meta;
    int max_nodes = 16 * 1024;

    std::unordered_map<std::string, ggml_tensor *> tensors;

    mimi_ggml_ctx() {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        auto buft = ggml_backend_get_default_buffer_type(backend);
        sched.reset(
            ggml_backend_sched_new(&backend, &buft, 1, max_nodes, false)
        );
        buf_compute_meta.resize(max_nodes * ggml_tensor_overhead() + ggml_graph_overhead());
    }

    void load_gguf(const char * fname) {
        ggml_context * meta = nullptr;

        gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };

        ctx_gguf = gguf_init_from_file(fname, params);

        // load tensors
        const int n_tensors = gguf_get_n_tensors(ctx_gguf);

        std::vector<uint8_t> read_buf;
        ggml_init_params ggml_params = {
            /*.mem_size   =*/ (n_tensors + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };

        ctx_data = ggml_init(ggml_params);
        auto fin = std::ifstream(fname, std::ios::binary);
        if (!fin) {
            ggml_free(meta);
            throw std::runtime_error("cannot open model file for loading tensors");
        }

        // add tensors to context
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf, i);
            ggml_tensor * t = ggml_get_tensor(meta, name);
            ggml_tensor * cur = ggml_dup_tensor(ctx_data, t);
            ggml_set_name(cur, name);
            tensors.insert({name, cur});
        }

        // alloc memory and offload data
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_data, buft);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf, i);
            ggml_tensor * cur = ggml_get_tensor(ctx_data, name);
            const size_t offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
            // printf("%s: Loading tensor \"%s\"\n", __func__, name);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                ggml_free(meta);
                throw std::runtime_error(string_format("failed to seek for tensor: %s", name));
            }
            int num_bytes = ggml_nbytes(cur);
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
        printf("%s: Loaded %d tensors from %s\n", __func__, n_tensors, fname);
        fin.close();

        ggml_free(meta);
    }

    /**
     * Build a cgraph using the given builder function.
     *
     * The built cgraph will be stored in `ctx.gf`
     */
    void build_graph(std::function<void(ggml_context *, ggml_cgraph *)> builder_fn) {
        ggml_free(ctx_gf);
        struct ggml_init_params params = {
            /*.mem_size   =*/ buf_compute_meta.size(),
            /*.mem_buffer =*/ buf_compute_meta.data(),
            /*.no_alloc   =*/ true,
        };

        ctx_gf = ggml_init(params);
        ggml_backend_sched_reset(sched.get());
        gf = ggml_new_graph_custom(ctx_gf, max_nodes, false);

        builder_fn(ctx_gf, gf);
        ggml_backend_sched_alloc_graph(sched.get(), gf);
    }

    ggml_status compute() {
        ggml_status status = ggml_backend_sched_graph_compute(sched.get(), gf);
        return status;
    }

    void set_tensor_data(const std::string & name, const void * data) {
        ggml_tensor * t = ggml_get_tensor(ctx_gf, name.c_str());
        if (!t) {
            throw std::runtime_error(string_format("tensor not found: %s", name.c_str()));
        }
        ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
    }

    std::pair<ggml_tensor *, std::vector<uint8_t>> get_tensor_data(const std::string & name) {
        ggml_tensor * t = ggml_get_tensor(ctx_gf, name.c_str());
        if (!t) {
            throw std::runtime_error(string_format("tensor not found: %s", name.c_str()));
        }
        std::vector<uint8_t> data(ggml_nbytes(t));
        ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
        return std::make_pair(t, data);
    }

    ggml_tensor * get_weight(const char *fmt, ...) {
        std::vector<char> str(128);
        va_list va;
        va_start(va, fmt);
        vsnprintf(str.data(), 128, fmt, va);
        va_end(va);
        auto it = tensors.find(str.data());
        if (it == tensors.end()) {
            throw std::runtime_error(string_format("weight tensor not found: %s", str.data()));
        }
        return it->second;
    }

    ~mimi_ggml_ctx() {
        ggml_free(ctx_data);
        gguf_free(ctx_gguf);
        ggml_backend_buffer_free(buf);
    }
};

///////////////////////////////////////////////////////////////////////////
// extension to ggml.h
// TODO: add these ops to the library (ofc with a more optimized kernel)


// mode: (0) constant, (1) reflect, (2) replicate, (3) circular
// value is only used in "constant"
// only "constant" with 0.0f and "replicate" are implemented here
static ggml_tensor * ggml_pad_ext(ggml_context * ctx0, ggml_tensor * x, int mode,
        int64_t pad_left, int64_t pad_right, float value = 0.0f) {
    GGML_ASSERT(value == 0.0f); // we can technically use ggml_arange, but for simplication we only support 0.0f
    GGML_ASSERT(mode == 0 || mode == 2);
    if (pad_left > 0) {
        ggml_tensor * tmp = ggml_new_tensor_2d(ctx0, x->type, pad_left, x->ne[1]);
        if (mode == 0) {
            tmp = ggml_scale(ctx0, tmp, value);
        } else if (mode == 2) {
            ggml_tensor * elem = ggml_view_2d(ctx0, x, 1, x->ne[1], x->nb[1], 0); // get first column
            tmp = ggml_repeat(ctx0, elem, tmp);
        }
        x = ggml_concat(ctx0, tmp, x, 0);
    }
    if (pad_right > 0) {
        ggml_tensor * tmp = ggml_new_tensor_2d(ctx0, x->type, pad_right, x->ne[1]);
        if (mode == 0) {
            tmp = ggml_scale(ctx0, tmp, value);
        } else if (mode == 2) {
            int64_t last = x->ne[0] - 1;
            ggml_tensor * elem = ggml_view_2d(ctx0, x, 1, x->ne[1], x->nb[1], last * ggml_element_size(x)); // get last column
            tmp = ggml_repeat(ctx0, elem, tmp);
        }
        x = ggml_concat(ctx0, x, tmp, 0);
    }
    return x;
}




///////////////////////////////////////////////////////////////////////////
// MimiConv and MimiConvTranspose

static int64_t div_ceil(int64_t a, int64_t b) {
    return a / b + (a % b ? 1 : 0);
}

static ggml_tensor * mimi_conv_1d(ggml_context * ctx0, ggml_tensor * x,
        ggml_tensor * kernel, ggml_tensor * bias, int stride, int dilation, bool pad_zero = true) {
    int64_t kernel_size = (kernel->ne[0] - 1) * dilation + 1;
    int64_t p_total = kernel_size - stride; // padding total
    int64_t p_half = p_total / 2;

    int64_t n_frames = div_ceil(x->ne[0] - kernel_size + p_total, stride);
    int64_t ideal_len = n_frames * stride + kernel_size - p_total;
    int64_t p_extra = ideal_len - x->ne[0];

    int64_t p_right = (mimi_config.causal ? 0 : p_half) + p_extra;
    int64_t p_left = p_total - (mimi_config.causal ? 0 : p_half);

    x = ggml_pad_ext(ctx0, x, pad_zero ? 0 : 2, p_left, p_right);

    x = ggml_conv_1d(ctx0, kernel, x, stride, 0, dilation);
    if (bias) {
        x = ggml_add(ctx0, x, bias);
    }
    ggml_set_name(x, "mimi_conv_1d");
    return x;
}

static ggml_tensor * mimi_conv_transpose_1d(ggml_context * ctx0, ggml_tensor * x,
        ggml_tensor * kernel, ggml_tensor * bias, int stride, int dilation, bool depthwise) {
    GGML_ASSERT(x->ne[1] == kernel->ne[2]);
    int64_t n_rows = x->ne[1];
    int64_t kernel_size = kernel->ne[0];
    int64_t p_total = kernel_size - stride; // padding total

    int64_t p_right = mimi_config.causal
        ? (float)p_total / mimi_config.trim_right_ratio
        : p_total / 2;
    int64_t p_left = p_total - p_right;

    ggml_tensor * out = nullptr;

    if (depthwise) {
        for (int64_t ir = 0; ir < n_rows; ir++) {
            ggml_tensor * row = ggml_view_1d(ctx0, x,
                                            x->ne[0], ir*x->ne[0]*ggml_element_size(x));
            ggml_tensor * krn = ggml_view_1d(ctx0, kernel,
                                            kernel->ne[0], ir*kernel->ne[0]*ggml_element_size(kernel));
            row = ggml_conv_transpose_1d(ctx0, krn, row, stride, 0, dilation);
            // unpad (remove p_right and p_left columns)
            row = ggml_view_1d(ctx0, row, row->ne[0] - p_total, p_left*ggml_element_size(row));

            // TODO: concat can be slow, we should use ggml_view_1d/ggml_cpy to avoid realloc
            out = out ? ggml_concat(ctx0, out, row, 1) : row;
        }

    } else {
        out = ggml_conv_transpose_1d(ctx0, kernel, x, stride, 0, dilation);
        // unpad
        out = ggml_view_2d(ctx0, out,
            out->ne[0] - p_total, out->ne[1],
            out->nb[1], p_left*ggml_element_size(out));
    }

    if (bias) {
        out = ggml_add(ctx0, out, bias);
    }

    return out;
}



///////////////////////////////////////////////////////////////////////////

// based on MimiEncoder
// SEANet encoder as used by Mimi.
struct mimi_encoder_decoder {
    mimi_ggml_ctx & ctx;
    struct layer {
        bool is_elu = false;
        bool is_resnet = false;
        bool is_transposed_conv = false;
        ggml_tensor * conv_0_w = nullptr;
        ggml_tensor * conv_0_b = nullptr;
        ggml_tensor * conv_1_w = nullptr;
        ggml_tensor * conv_1_b = nullptr;
        int stride = 1;
    };
    std::vector<layer> layers;

    std::array<int, 4> repeated_pattern = {1, 4, 7, 10};

    mimi_encoder_decoder(mimi_ggml_ctx & ctx): ctx(ctx) {
        layers.push_back({
            .conv_0_w = ctx.get_weight("decoder.layers.0.conv.weight"),
            .conv_0_b = ctx.get_weight("decoder.layers.0.conv.bias"),
        });
        for (int i = 0; i < (int)repeated_pattern.size(); ++i) {
            int i_start = repeated_pattern[i];
            // upsampling layers
            layers.push_back({
                .is_elu = true, // layer (i_start)
            });
            layers.push_back({
                .is_transposed_conv = true,
                .conv_0_w = ctx.get_weight("decoder.layers.%d.conv.weight", i_start + 1),
                .conv_0_b = ctx.get_weight("decoder.layers.%d.conv.bias",   i_start + 1),
                .stride = mimi_config.upsampling_ratio[i],
            });
            // residual layers
            layers.push_back({
                .is_resnet = true,
                .conv_0_w = ctx.get_weight("decoder.layers.%d.block.1.conv.weight", i_start + 2),
                .conv_0_b = ctx.get_weight("decoder.layers.%d.block.1.conv.bias",   i_start + 2),
                .conv_1_w = ctx.get_weight("decoder.layers.%d.block.3.conv.weight", i_start + 2),
                .conv_1_b = ctx.get_weight("decoder.layers.%d.block.3.conv.bias",   i_start + 2),
            });
        }
        layers.push_back({
            .is_elu = true, // layer 13
        });
        layers.push_back({
            .conv_0_w = ctx.get_weight("decoder.layers.14.conv.weight"),
            .conv_0_b = ctx.get_weight("decoder.layers.14.conv.bias"),
        });
    }

    ggml_tensor * forward(ggml_context * ctx0, ggml_tensor * input) {
        ggml_tensor * x = input;

        for (auto & layer : layers) {
            if (layer.is_elu) {
                x = ggml_elu(ctx0, x);
            } else if (layer.is_resnet) {
                ggml_tensor * residual = x;
                x = ggml_elu(ctx0, x);
                x = mimi_conv_1d(ctx0, x, layer.conv_0_w, layer.conv_0_b, 1, 1);
                x = ggml_elu(ctx0, x);
                x = mimi_conv_1d(ctx0, x, layer.conv_1_w, layer.conv_1_b, 1, 1);
                x = ggml_add(ctx0, x, residual);
            } else {
                x = layer.is_transposed_conv
                    ? mimi_conv_transpose_1d(ctx0, x, layer.conv_0_w, layer.conv_0_b, layer.stride, 1, false)
                    : mimi_conv_1d(ctx0, x, layer.conv_0_w, layer.conv_0_b, layer.stride, 1);
            }
        }

        return x;
    }
};

struct mimi_transformer {
    struct layer {
        ggml_tensor * inp_norm_w = nullptr;
        ggml_tensor * inp_norm_b = nullptr;

        ggml_tensor * attn_q = nullptr;
        ggml_tensor * attn_k = nullptr;
        ggml_tensor * attn_v = nullptr;
        ggml_tensor * attn_o = nullptr;
        ggml_tensor * attn_post_norm_w = nullptr;
        ggml_tensor * attn_post_norm_b = nullptr;
        ggml_tensor * attn_layer_scale = nullptr;

        ggml_tensor * ffn_up = nullptr;
        ggml_tensor * ffn_down = nullptr;
        ggml_tensor * mlp_layer_scale = nullptr;
    };
    std::vector<layer> layers;

    mimi_transformer(mimi_ggml_ctx & ctx, const char * prefix, int n_layers) {
        for (int il = 0; il < n_layers; il++) {
            layers.push_back({
                .inp_norm_w = ctx.get_weight("%s_transformer.layers.%d.input_layernorm.weight", prefix, il),
                .inp_norm_b = ctx.get_weight("%s_transformer.layers.%d.input_layernorm.bias",   prefix, il),

                .attn_q           = ctx.get_weight("%s_transformer.layers.%d.self_attn.q_proj.weight",         prefix, il),
                .attn_k           = ctx.get_weight("%s_transformer.layers.%d.self_attn.k_proj.weight",         prefix, il),
                .attn_v           = ctx.get_weight("%s_transformer.layers.%d.self_attn.v_proj.weight",         prefix, il),
                .attn_o           = ctx.get_weight("%s_transformer.layers.%d.self_attn.o_proj.weight",         prefix, il),
                .attn_post_norm_w = ctx.get_weight("%s_transformer.layers.%d.post_attention_layernorm.weight", prefix, il),
                .attn_post_norm_b = ctx.get_weight("%s_transformer.layers.%d.post_attention_layernorm.bias",   prefix, il),
                .attn_layer_scale = ctx.get_weight("%s_transformer.layers.%d.self_attn_layer_scale.scale",     prefix, il),

                .ffn_up          = ctx.get_weight("%s_transformer.layers.%d.mlp.fc1.weight",        prefix, il),
                .ffn_down        = ctx.get_weight("%s_transformer.layers.%d.mlp.fc2.weight",        prefix, il),
                .mlp_layer_scale = ctx.get_weight("%s_transformer.layers.%d.mlp_layer_scale.scale", prefix, il),
            });
        }
    }

    ggml_tensor * forward(ggml_context * ctx0, ggml_tensor * input, ggml_tensor * inp_pos) {
        int n_tokens    = input->ne[1];
        ggml_tensor * x = input;

        auto layer_norm = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
            x = ggml_norm(ctx0, x, mimi_config.norm_eps);
            x = ggml_mul(ctx0, x, w);
            x = ggml_add(ctx0, x, b);
            return x;
        };

        ggml_tensor * residual = input;

        for (auto & layer : layers) {
            residual = x;

            // input layer norm
            x = layer_norm(x, layer.inp_norm_w, layer.inp_norm_b);

            // self attention
            {
                ggml_tensor * q = ggml_mul_mat(ctx0, layer.attn_q, x);
                ggml_tensor * k = ggml_mul_mat(ctx0, layer.attn_k, x);
                ggml_tensor * v = ggml_mul_mat(ctx0, layer.attn_v, x);

                int n_embd_head = mimi_config.n_embd / mimi_config.n_head;
                q = ggml_reshape_3d(ctx0, q, n_embd_head, mimi_config.n_head,    n_tokens);
                k = ggml_reshape_3d(ctx0, k, n_embd_head, mimi_config.n_head_kv, n_tokens);
                v = ggml_reshape_3d(ctx0, v, n_embd_head, mimi_config.n_head_kv, n_tokens);

                int n_rot = n_embd_head;
                q = ggml_rope_inplace(ctx0, q, inp_pos, n_rot, 0);
                q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));

                k = ggml_rope_inplace(ctx0, k, inp_pos, n_rot, 0);
                k = ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3));

                ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
                ggml_mul_mat_set_prec(kq, GGML_PREC_F32); // mimic behavior of llama.cpp
                kq = ggml_scale_inplace(ctx0, kq, 1.0f / std::sqrt(n_embd_head));
                ggml_tensor * kq_masked = ggml_diag_mask_inf_inplace(ctx0, kq, n_tokens);
                kq = ggml_soft_max_inplace(ctx0, kq_masked);

                v = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3));

                ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
                kqv = ggml_reshape_3d(ctx0, kqv, n_embd_head, n_tokens, mimi_config.n_head);
                kqv = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
                kqv = ggml_cont_2d(ctx0, kqv, mimi_config.n_embd, n_tokens);

                x = ggml_mul_mat(ctx0, layer.attn_o, kqv);
            }

            // residual
            x = ggml_mul(ctx0, x, layer.attn_layer_scale);
            x = ggml_add(ctx0, x, residual);

            residual = x;
            x = layer_norm(x, layer.attn_post_norm_w, layer.attn_post_norm_b);

            // mlp
            {
                x = ggml_mul_mat(ctx0, layer.ffn_up, x);
                x = ggml_gelu(ctx0, x);
                x = ggml_mul_mat(ctx0, layer.ffn_down, x);
            }

            // residual
            x = ggml_mul(ctx0, x, layer.mlp_layer_scale);
            x = ggml_add(ctx0, x, residual);
        }

        return x;
    }
};

struct mimi_residual_vector_quantizer {
    struct component {
        ggml_tensor * codebook;
    };

    ggml_tensor * semantic_inp_proj;
    std::vector<component> semantic_components;
    ggml_tensor * semantic_out_proj;

    ggml_tensor * acoustic_inp_proj;
    std::vector<component> acoustic_components;
    ggml_tensor * acoustic_out_proj;

    mimi_residual_vector_quantizer(mimi_ggml_ctx & ctx) {
        semantic_inp_proj = ctx.get_weight("quantizer.semantic_rvq.input_proj.weight");
        semantic_out_proj = ctx.get_weight("quantizer.semantic_rvq.output_proj.weight");
        for (int i = 0; i < mimi_config.n_semantic_components; i++) {
            semantic_components.push_back({
                .codebook = ctx.get_weight("quantizer.semantic_rvq.layers.%d.codebook",     i),
            });
        }
        acoustic_inp_proj = ctx.get_weight("quantizer.acoustic_rvq.input_proj.weight");
        acoustic_out_proj = ctx.get_weight("quantizer.acoustic_rvq.output_proj.weight");
        for (int i = 0; i < mimi_config.n_acoustic_components; i++) {
            acoustic_components.push_back({
                .codebook = ctx.get_weight("quantizer.acoustic_rvq.layers.%d.codebook",     i),
            });
        }
    }

    // the input has shape [n_codes, n_codes_per_embd]
    // first row is semantic, the rest are acoustic
    // example: [ [semantic], [acoustic1], [acoustic2], ... ]
    ggml_tensor * decode(ggml_context * ctx0, ggml_tensor * input) {
        GGML_ASSERT(input->type == GGML_TYPE_I32);

        size_t  n_semantic       = semantic_components.size();
        int64_t n_codes_per_embd = (n_semantic + acoustic_components.size());
        int64_t n_codes          = input->ne[0] / n_codes_per_embd;

        GGML_ASSERT(input->ne[0] % n_codes_per_embd == 0);

        ggml_tensor * out_s = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mimi_config.codebook_dim, n_codes);
        ggml_tensor * out_a = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mimi_config.codebook_dim, n_codes);
        out_s = ggml_scale(ctx0, out_s, 0.0f); // clear
        out_a = ggml_scale(ctx0, out_a, 0.0f); // clear

        for (size_t ir = 0; ir < (size_t)n_codes_per_embd; ir++) {
            ggml_tensor * row = ggml_view_1d(ctx0, input, n_codes, ir*n_codes*ggml_element_size(input));
            if (ir < n_semantic) {
                // semantic
                ggml_tensor * codebook = semantic_components[ir].codebook;
                ggml_tensor * embd = ggml_get_rows(ctx0, codebook, row);
                out_s = ggml_add(ctx0, out_s, embd);
            } else {
                // acoustic
                ggml_tensor * codebook = acoustic_components[ir-n_semantic].codebook;
                ggml_tensor * embd = ggml_get_rows(ctx0, codebook, row);
                out_a = ggml_add(ctx0, out_a, embd);
            }
        }

        out_s = ggml_mul_mat(ctx0, semantic_out_proj, out_s);
        out_a = ggml_mul_mat(ctx0, acoustic_out_proj, out_a);

        return ggml_add(ctx0, out_s, out_a);
    }
};


mimi_model::mimi_model(const char * fname, bool verbose) : verbose(verbose) {
    ctx.reset(new mimi_ggml_ctx());
    ctx->load_gguf(fname);

    // initialize components
    seanet_dec     .reset(new mimi_encoder_decoder(*ctx));
    transformer_dec.reset(new mimi_transformer(*ctx, "decoder", mimi_config.num_hidden_layers));
    quantizer      .reset(new mimi_residual_vector_quantizer(*ctx));
}

mimi_model::~mimi_model() {
}

std::vector<float> mimi_model::decode_frame(const std::vector<int> & codes, int & n_past) {
    // build cgraph
    int n_pos            = -1;
    int n_codes          = codes.size();
    int n_codes_per_embd = mimi_config.n_semantic_components + mimi_config.n_acoustic_components;
    GGML_ASSERT(n_codes % n_codes_per_embd == 0 && "number of codes must be a multiply of n_codes_per_embd");

    ctx->build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf) {
        ggml_tensor * inp_dec = ggml_new_tensor_1d(ctx_gf, GGML_TYPE_I32, n_codes);
        ggml_set_name(inp_dec, "inp_dec");
        ggml_set_input(inp_dec);

        // RVQ
        ggml_tensor * embeddings = quantizer->decode(ctx_gf, inp_dec);

        // upsample
        embeddings = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, embeddings));
        embeddings = mimi_conv_transpose_1d(ctx_gf, embeddings, ctx->get_weight("upsample.conv.weight"), nullptr, 2, 1, true);

        // transformer
        n_pos = embeddings->ne[0];
        ggml_tensor * pos_dec = ggml_new_tensor_1d(ctx_gf, GGML_TYPE_I32, n_pos);
        ggml_set_name(pos_dec, "pos_dec");
        ggml_set_input(pos_dec);
        embeddings = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, embeddings));
        embeddings = transformer_dec->forward(ctx_gf, embeddings, pos_dec);

        // SEANET decoder
        embeddings = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, embeddings));
        ggml_tensor * output = seanet_dec->forward(ctx_gf, embeddings);

        ggml_set_name(output, "output");
        ggml_set_output(output);
        ggml_build_forward_expand(gf, output);
    });

    // position data
    GGML_ASSERT(n_pos <= mimi_config.sliding_window);
    std::vector<int> pos_data(n_pos);
    for (int i = 0; i < (int)pos_data.size(); i++) {
        pos_data[i] = i + n_past;
    }
    if (verbose) {
        printf("%s: n_pos: %d, n_past: %d\n", __func__, n_pos, n_past);
    }
    n_past += n_pos;
    ctx->set_tensor_data("pos_dec", pos_data.data());

    // code data
    auto codes_T = mimi_model::transpose_input(codes);
    ctx->set_tensor_data("inp_dec", codes_T.data());

    ctx->compute();

    auto output = ctx->get_tensor_data("output");
    // auto output_tensor = output.first;
    auto output_data   = output.second;
    // printf("Output shape: [%lld, %lld]\n", output_tensor->ne[0], output_tensor->ne[1]);

    std::vector<float> wav_data(output_data.size() / sizeof(float));
    for (size_t i = 0; i < wav_data.size(); i++) {
        wav_data[i] = ((float *)output_data.data())[i];
    }

    return wav_data;
}

std::vector<float> mimi_model::decode(const std::vector<int> & codes) {
    std::vector<float> output;

    if (verbose) {
        printf("%s: n_codes: %zu\n", __func__, codes.size());
    }

    int64_t t_start = ggml_time_ms();
    int n_frames = 0;

    int n_past = 0;
    for (size_t i = 0; i < codes.size(); i += mimi_config.n_codes_per_frame) {
        size_t remaining = std::min((size_t)mimi_config.n_codes_per_frame, codes.size() - i);
        std::vector<int> frame(codes.begin() + i, codes.begin() + i + remaining);

        auto wav_data = decode_frame(frame, n_past);
        output.insert(output.end(), wav_data.begin(), wav_data.end());

        n_frames++;
    }

    int64_t t_end = ggml_time_ms();
    if (verbose) {
        printf("%s: n_frames: %d, time: %" PRId64 "ms, per_frame: %" PRId64 "ms\n", __func__, n_frames, t_end - t_start, (t_end - t_start) / n_frames);
    }

    return output;
}

std::vector<int> mimi_model::transpose_input(const std::vector<int> & codes) {
    int n_codes          = codes.size();
    int n_codes_per_embd = mimi_config.n_semantic_components + mimi_config.n_acoustic_components;
    GGML_ASSERT(n_codes % n_codes_per_embd == 0 && "number of codes must be a multiply of n_codes_per_embd");

    std::vector<int> codes_T(n_codes);
    for (int i = 0; i < n_codes / n_codes_per_embd; i++) {
        for (int j = 0; j < n_codes_per_embd; j++) {
            int src_idx = i * n_codes_per_embd + j;
            int dst_idx = j * (n_codes / n_codes_per_embd) + i;
            codes_T[dst_idx] = codes[src_idx];
        }
    }

    return codes_T;
}

int mimi_model::get_sample_rate() const {
    return mimi_config.sample_rate;
}
