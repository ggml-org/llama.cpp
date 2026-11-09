#include "models.h"

#include <algorithm>
#include <cmath>

static ggml_tensor * apertus_add_channel_bias(
        ggml_context * ctx0,
        ggml_tensor * cur,
        ggml_tensor * bias) {
    if (!bias) {
        return cur;
    }
    ggml_tensor * bias_4d = ggml_reshape_4d(ctx0, bias, 1, 1, bias->ne[0], 1);
    return ggml_add(ctx0, cur, bias_4d);
}

static ggml_tensor * apertus_mul_channel_weight(
        ggml_context * ctx0,
        ggml_tensor * cur,
        ggml_tensor * weight) {
    ggml_tensor * weight_4d = ggml_reshape_4d(ctx0, weight, 1, 1, weight->ne[0], 1);
    return ggml_mul(ctx0, cur, weight_4d);
}

static ggml_tensor * apertus_conv_2d(
        ggml_context * ctx0,
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * bias,
        int stride,
        int padding) {
    cur = ggml_conv_2d(ctx0, weight, cur, stride, stride, padding, padding, 1, 1);
    return apertus_add_channel_bias(ctx0, cur, bias);
}

static ggml_tensor * apertus_group_norm(
        ggml_context * ctx0,
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * bias) {
    cur = ggml_group_norm(ctx0, cur, 32, 1e-6f);
    cur = apertus_mul_channel_weight(ctx0, cur, weight);
    return apertus_add_channel_bias(ctx0, cur, bias);
}

static ggml_tensor * apertus_silu(ggml_context * ctx0, ggml_tensor * cur) {
    return ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, cur));
}

static ggml_tensor * apertus_build_resblock(
        ggml_context * ctx0,
        ggml_tensor * cur,
        const apertus_vision_resblock & block) {
    ggml_tensor * residual = cur;

    cur = apertus_group_norm(ctx0, cur, block.norm1_w, block.norm1_b);
    cur = apertus_silu(ctx0, cur);
    cur = apertus_conv_2d(ctx0, cur, block.conv1_w, block.conv1_b, 1, 1);
    cur = apertus_group_norm(ctx0, cur, block.norm2_w, block.norm2_b);
    cur = apertus_silu(ctx0, cur);
    cur = apertus_conv_2d(ctx0, cur, block.conv2_w, block.conv2_b, 1, 1);

    if (block.shortcut_w) {
        residual = apertus_conv_2d(ctx0, residual, block.shortcut_w, block.shortcut_b, 1, 0);
    }
    return ggml_add(ctx0, residual, cur);
}

static ggml_tensor * apertus_build_attention(
        ggml_context * ctx0,
        ggml_tensor * cur,
        const apertus_vision_attention & attention) {
    ggml_tensor * residual = cur;
    cur = apertus_group_norm(ctx0, cur, attention.norm_w, attention.norm_b);

    ggml_tensor * query = apertus_conv_2d(ctx0, cur, attention.q_w, attention.q_b, 1, 0);
    ggml_tensor * key   = apertus_conv_2d(ctx0, cur, attention.k_w, attention.k_b, 1, 0);
    ggml_tensor * value = apertus_conv_2d(ctx0, cur, attention.v_w, attention.v_b, 1, 0);

    const int64_t width = cur->ne[0];
    const int64_t height = cur->ne[1];
    const int64_t channels = cur->ne[2];
    const int64_t tokens = width * height;
    GGML_ASSERT(cur->ne[3] == 1);

    auto to_channel_tokens = [&](ggml_tensor * tensor) {
        tensor = ggml_reshape_2d(ctx0, tensor, tokens, channels);
        tensor = ggml_cont(ctx0, ggml_transpose(ctx0, tensor));
        return tensor;
    };

    query = to_channel_tokens(query);
    key = to_channel_tokens(key);
    value = to_channel_tokens(value);

    ggml_tensor * weights = ggml_mul_mat(ctx0, key, query);
    weights = ggml_scale(ctx0, weights, 1.0f / std::sqrt((float) channels));
    weights = ggml_soft_max(ctx0, weights);

    value = ggml_cont(ctx0, ggml_transpose(ctx0, value));
    cur = ggml_mul_mat(ctx0, value, weights);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = ggml_reshape_4d(ctx0, cur, width, height, channels, 1);
    cur = apertus_conv_2d(ctx0, cur, attention.out_w, attention.out_b, 1, 0);
    return ggml_add(ctx0, residual, cur);
}

ggml_cgraph * mtmd_graph_apertus_vision::build() {
    GGML_ASSERT(n_batch == 1);

    ggml_tensor * cur = build_inp_raw();
    cur = apertus_conv_2d(ctx0, cur, model.apertus_v_conv_in_w, model.apertus_v_conv_in_b, 1, 1);

    for (const auto & stage : model.apertus_v_stages) {
        for (size_t block_id = 0; block_id < stage.blocks.size(); ++block_id) {
            cur = apertus_build_resblock(ctx0, cur, stage.blocks[block_id]);
            if (!stage.attentions.empty()) {
                cur = apertus_build_attention(ctx0, cur, stage.attentions[block_id]);
            }
        }
        if (stage.downsample_w) {
            cur = ggml_pad(ctx0, cur, 1, 1, 0, 0);
            cur = apertus_conv_2d(ctx0, cur, stage.downsample_w, stage.downsample_b, 2, 0);
        }
    }

    cur = apertus_build_resblock(ctx0, cur, model.apertus_v_mid_block_1);
    cur = apertus_build_attention(ctx0, cur, model.apertus_v_mid_attention);
    cur = apertus_build_resblock(ctx0, cur, model.apertus_v_mid_block_2);
    cur = apertus_group_norm(ctx0, cur, model.apertus_v_norm_out_w, model.apertus_v_norm_out_b);
    cur = apertus_silu(ctx0, cur);
    cur = apertus_conv_2d(ctx0, cur, model.apertus_v_conv_out_w, model.apertus_v_conv_out_b, 1, 1);
    cur = apertus_conv_2d(ctx0, cur, model.apertus_v_quant_conv_w, model.apertus_v_quant_conv_b, 1, 0);

    const int64_t token_count = cur->ne[0] * cur->ne[1];
    const int64_t codebook_dim = model.apertus_v_codebook->ne[0];
    GGML_ASSERT(cur->ne[2] == codebook_dim);
    GGML_ASSERT(cur->ne[3] == 1);

    ggml_tensor * tokens = ggml_reshape_2d(ctx0, cur, token_count, codebook_dim);
    tokens = ggml_cont(ctx0, ggml_transpose(ctx0, tokens));

    static constexpr int64_t code_chunk_size = 64;
    ggml_tensor * codes = nullptr;
    for (int64_t offset = 0; offset < token_count; offset += code_chunk_size) {
        const int64_t chunk_size = std::min(code_chunk_size, token_count - offset);
        ggml_tensor * chunk = ggml_view_2d(
            ctx0, tokens, codebook_dim, chunk_size, tokens->nb[1], offset * tokens->nb[1]);
        ggml_tensor * logits = ggml_mul_mat(ctx0, model.apertus_v_codebook, chunk);
        ggml_tensor * chunk_codes = ggml_argmax(ctx0, logits);
        codes = codes ? ggml_concat(ctx0, codes, chunk_codes, 0) : chunk_codes;
    }

    ggml_set_name(codes, "apertus_vision_codes");
    cb(codes, "apertus_vision_codes", -1);
    ggml_build_forward_expand(gf, codes);
    return gf;
}
