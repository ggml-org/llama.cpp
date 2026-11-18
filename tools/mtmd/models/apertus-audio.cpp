#include "models.h"

#include <algorithm>
#include <vector>

static ggml_tensor * apertus_audio_add_channel_bias(
        ggml_context * ctx0,
        ggml_tensor * cur,
        ggml_tensor * bias) {
    GGML_ASSERT(bias != nullptr);
    GGML_ASSERT(cur->ne[1] == bias->ne[0]);
    return ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, bias, 1, bias->ne[0], 1));
}

static ggml_tensor * apertus_audio_conv_1d_f32(
        ggml_context * ctx0,
        ggml_tensor * weight,
        ggml_tensor * cur,
        int stride,
        int dilation) {
    ggml_tensor * columns = ggml_im2col(
        ctx0, weight, cur, stride, 0, 0, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * result = ggml_mul_mat(
        ctx0,
        ggml_reshape_2d(ctx0, columns, columns->ne[0], columns->ne[2] * columns->ne[1]),
        ggml_reshape_2d(ctx0, weight, weight->ne[0] * weight->ne[1], weight->ne[2]));
    return ggml_reshape_3d(ctx0, result, columns->ne[1], weight->ne[2], columns->ne[2]);
}

static ggml_tensor * apertus_audio_conv_1d(
        ggml_context * ctx0,
        ggml_tensor * cur,
        const apertus_audio_conv & conv,
        int stride,
        int dilation = 1) {
    GGML_ASSERT(conv.w != nullptr);
    GGML_ASSERT(conv.b != nullptr);
    GGML_ASSERT(stride > 0);
    GGML_ASSERT(dilation > 0);

    const int64_t length = cur->ne[0];
    const int64_t effective_kernel = (conv.w->ne[0] - 1) * dilation + 1;
    const int64_t padding_total = effective_kernel - stride;
    GGML_ASSERT(length > 0);
    GGML_ASSERT(padding_total >= 0);

    const int64_t extra_padding = (stride - length % stride) % stride;
    const int64_t padding_right = padding_total / 2 + extra_padding;
    const int64_t padding_left = padding_total - padding_total / 2;
    const int64_t max_padding = std::max(padding_left, padding_right);

    // WavTokenizer pads short waveforms with zeros so reflection stays within the input, then removes the suffix.
    int64_t temporary_padding = 0;
    if (length <= max_padding) {
        temporary_padding = max_padding - length + 1;
        cur = ggml_pad(ctx0, cur, temporary_padding, 0, 0, 0);
    }

    cur = ggml_pad_reflect_1d(ctx0, ggml_cont(ctx0, cur), padding_left, padding_right);
    if (temporary_padding > 0) {
        cur = ggml_view_3d(
            ctx0,
            cur,
            length + padding_left + padding_right,
            cur->ne[1],
            cur->ne[2],
            cur->nb[1],
            cur->nb[2],
            0);
        cur = ggml_cont(ctx0, cur);
    }

    // Keep im2col in F32: the F16 path can change codebook IDs.
    cur = apertus_audio_conv_1d_f32(ctx0, conv.w, cur, stride, dilation);
    return apertus_audio_add_channel_bias(ctx0, cur, conv.b);
}

static ggml_tensor * apertus_audio_concat_sequence(
        ggml_context * ctx0,
        const std::vector<ggml_tensor *> & columns,
        size_t begin,
        size_t end) {
    GGML_ASSERT(begin < end);
    if (end - begin == 1) {
        return columns[begin];
    }

    const size_t middle = begin + (end - begin) / 2;
    ggml_tensor * left = apertus_audio_concat_sequence(ctx0, columns, begin, middle);
    ggml_tensor * right = apertus_audio_concat_sequence(ctx0, columns, middle, end);
    return ggml_concat(ctx0, left, right, 1);
}

static ggml_tensor * apertus_audio_lstm(
        ggml_context * ctx0,
        ggml_tensor * sequence,
        const std::vector<apertus_audio_lstm_layer> & layers) {
    GGML_ASSERT(sequence->ne[2] == 1);
    GGML_ASSERT(!layers.empty());

    const int64_t hidden_size = sequence->ne[0];
    const int64_t sequence_length = sequence->ne[1];
    ggml_tensor * residual = sequence;

    for (const auto & layer : layers) {
        GGML_ASSERT(layer.weight_ih != nullptr);
        GGML_ASSERT(layer.weight_hh != nullptr);
        GGML_ASSERT(layer.bias_ih != nullptr);
        GGML_ASSERT(layer.bias_hh != nullptr);
        GGML_ASSERT(layer.weight_ih->ne[0] == hidden_size);
        GGML_ASSERT(layer.weight_hh->ne[0] == hidden_size);
        GGML_ASSERT(layer.weight_ih->ne[1] == 4 * hidden_size);
        GGML_ASSERT(layer.weight_hh->ne[1] == 4 * hidden_size);

        // Project all input steps at once. The recurrent projection stays sequential.
        ggml_tensor * input_gates = ggml_mul_mat(ctx0, layer.weight_ih, sequence);
        input_gates = ggml_add(ctx0, input_gates, layer.bias_ih);

        ggml_tensor * first_input = ggml_view_1d(ctx0, sequence, hidden_size, 0);
        ggml_tensor * hidden = ggml_scale(ctx0, first_input, 0.0f);
        ggml_tensor * cell = ggml_scale(ctx0, first_input, 0.0f);

        std::vector<ggml_tensor *> outputs;
        outputs.reserve(sequence_length);
        for (int64_t position = 0; position < sequence_length; ++position) {
            ggml_tensor * gates = ggml_view_1d(
                ctx0,
                input_gates,
                4 * hidden_size,
                position * input_gates->nb[1]);
            gates = ggml_add(ctx0, gates, ggml_mul_mat(ctx0, layer.weight_hh, hidden));
            gates = ggml_add(ctx0, gates, layer.bias_hh);

            ggml_tensor * input_gate = ggml_sigmoid(ctx0, ggml_view_1d(
                ctx0, gates, hidden_size, 0 * hidden_size * gates->nb[0]));
            ggml_tensor * forget_gate = ggml_sigmoid(ctx0, ggml_view_1d(
                ctx0, gates, hidden_size, 1 * hidden_size * gates->nb[0]));
            ggml_tensor * cell_gate = ggml_tanh(ctx0, ggml_view_1d(
                ctx0, gates, hidden_size, 2 * hidden_size * gates->nb[0]));
            ggml_tensor * output_gate = ggml_sigmoid(ctx0, ggml_view_1d(
                ctx0, gates, hidden_size, 3 * hidden_size * gates->nb[0]));

            cell = ggml_add(
                ctx0,
                ggml_mul(ctx0, forget_gate, cell),
                ggml_mul(ctx0, input_gate, cell_gate));
            hidden = ggml_mul(ctx0, output_gate, ggml_tanh(ctx0, cell));
            outputs.push_back(ggml_reshape_2d(ctx0, hidden, hidden_size, 1));
        }

        sequence = apertus_audio_concat_sequence(ctx0, outputs, 0, outputs.size());
    }

    return ggml_add(ctx0, sequence, residual);
}

ggml_cgraph * mtmd_graph_apertus_audio::build() {
    GGML_ASSERT(n_batch == 1);
    GGML_ASSERT(model.apertus_a_convs.size() == hparams.audio_downsample_ratios.size() + 2);
    GGML_ASSERT(model.apertus_a_resblocks.size() == hparams.audio_downsample_ratios.size());
    GGML_ASSERT(model.apertus_a_codebook != nullptr);

    ggml_tensor * cur = build_inp_raw(1);
    cur = apertus_audio_conv_1d(ctx0, cur, model.apertus_a_convs.front(), 1);

    for (size_t stage = 0; stage < hparams.audio_downsample_ratios.size(); ++stage) {
        const auto & block = model.apertus_a_resblocks[stage];
        ggml_tensor * residual = apertus_audio_conv_1d(ctx0, cur, block.shortcut, 1);

        ggml_tensor * block_out = ggml_elu(ctx0, cur);
        block_out = apertus_audio_conv_1d(ctx0, block_out, block.conv1, 1);
        block_out = ggml_elu(ctx0, block_out);
        block_out = apertus_audio_conv_1d(ctx0, block_out, block.conv2, 1);
        cur = ggml_add(ctx0, residual, block_out);

        cur = ggml_elu(ctx0, cur);
        cur = apertus_audio_conv_1d(
            ctx0,
            cur,
            model.apertus_a_convs[stage + 1],
            hparams.audio_downsample_ratios[stage]);
    }

    ggml_tensor * sequence = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    sequence = apertus_audio_lstm(ctx0, sequence, model.apertus_a_lstm_layers);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, sequence));

    cur = ggml_elu(ctx0, cur);
    cur = apertus_audio_conv_1d(ctx0, cur, model.apertus_a_convs.back(), 1);

    const int64_t token_count = cur->ne[0];
    const int64_t codebook_dim = model.apertus_a_codebook->ne[0];
    GGML_ASSERT(cur->ne[1] == codebook_dim);
    GGML_ASSERT(cur->ne[2] == 1);

    ggml_tensor * tokens = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    ggml_tensor * codebook_norm = ggml_sum_rows(ctx0, ggml_sqr(ctx0, model.apertus_a_codebook));
    codebook_norm = ggml_cont(ctx0, ggml_transpose(ctx0, codebook_norm));
    ggml_tensor * dot = ggml_mul_mat(ctx0, model.apertus_a_codebook, tokens);
    ggml_tensor * scores = ggml_sub(ctx0, ggml_scale(ctx0, dot, 2.0f), codebook_norm);
    ggml_tensor * codes = ggml_argmax(ctx0, scores);
    GGML_ASSERT(codes->ne[0] == token_count);

    ggml_set_name(codes, "apertus_audio_codes");
    cb(codes, "apertus_audio_codes", -1);
    ggml_build_forward_expand(gf, codes);
    return gf;
}
