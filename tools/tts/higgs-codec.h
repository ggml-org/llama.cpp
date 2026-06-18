#pragma once

#include "higgs-gguf.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace higgs_audio {

struct rvq_quantizer_weights {
    std::vector<float> codebook_embed;     // [codebook_size, codebook_dim]
    std::vector<float> project_in_weight;  // [codebook_dim, hidden_size]
    std::vector<float> project_in_bias;    // [codebook_dim]
    std::vector<float> project_out_weight; // [hidden_size, codebook_dim]
    std::vector<float> project_out_bias;   // [hidden_size]
};

struct rvq_decoder_weights {
    int num_quantizers = 8;
    int codebook_size  = 1024;
    int codebook_dim   = 64;
    int hidden_size    = 1024;
    int acoustic_size  = 256;

    std::vector<rvq_quantizer_weights> quantizers;
    std::vector<float> fc2_weight; // [acoustic_size, hidden_size]
    std::vector<float> fc2_bias;   // [acoustic_size]
};

struct dac_conv1d_weights {
    int in_channels  = 0;
    int out_channels = 0;
    int kernel       = 0;
    std::vector<float> weight; // [out_channels, in_channels, kernel]
    std::vector<float> bias;   // [out_channels]
};

struct dac_conv_transpose1d_weights {
    int in_channels     = 0;
    int out_channels    = 0;
    int kernel          = 0;
    int stride          = 0;
    int padding         = 0;
    int output_padding  = 0;
    std::vector<float> weight; // [in_channels, out_channels, kernel]
    std::vector<float> bias;   // [out_channels]
};

struct dac_residual_unit_weights {
    int channels = 0;
    int dilation = 1;
    std::vector<float> snake1_alpha; // [channels]
    std::vector<float> snake2_alpha; // [channels]
    dac_conv1d_weights conv1;
    dac_conv1d_weights conv2;
};

struct dac_decoder_block_weights {
    int in_channels  = 0;
    int out_channels = 0;
    int stride       = 0;
    std::vector<float> snake1_alpha; // [in_channels]
    dac_conv_transpose1d_weights conv_t1;
    dac_residual_unit_weights res_units[3];
};

struct dac_decoder_weights {
    int acoustic_size = 256;
    int output_channels = 1;
    std::vector<int> strides = { 8, 5, 4, 2, 3 };

    dac_conv1d_weights conv1;
    dac_decoder_block_weights blocks[5];
    std::vector<float> snake1_alpha; // [32]
    dac_conv1d_weights conv2;
};

struct dac_encoder_block_weights {
    int in_channels  = 0;
    int out_channels = 0;
    int stride       = 0;
    dac_residual_unit_weights res_units[3];
    std::vector<float> snake1_alpha; // [in_channels]
    dac_conv1d_weights conv1;
};

struct dac_encoder_weights {
    int input_channels = 1;
    int acoustic_size = 256;
    std::vector<int> strides = { 8, 5, 4, 2, 3 };

    dac_conv1d_weights conv1;
    dac_encoder_block_weights blocks[5];
    std::vector<float> snake1_alpha; // [2048]
    dac_conv1d_weights conv2;
};

inline std::vector<float> load_codec_tensor(
        const companion_file & file,
        const std::string & original_name,
        const size_t expected_count) {
    const auto info = file.codec_tensor_info(original_name);
    const auto data = file.read_tensor_f32(info);
    if (data.size() != expected_count) {
        throw std::runtime_error("unexpected tensor size for Higgs codec tensor: " + original_name);
    }
    return data;
}

inline dac_conv1d_weights load_dac_conv1d(
        const companion_file & file,
        const std::string & prefix,
        const int out_channels,
        const int in_channels,
        const int kernel);

inline std::vector<float> load_dac_alpha(
        const companion_file & file,
        const std::string & name,
        const int channels);

inline dac_residual_unit_weights load_dac_residual_unit(
        const companion_file & file,
        const std::string & prefix,
        const int channels,
        const int dilation);

inline rvq_decoder_weights load_rvq_decoder_weights(const companion_file & file) {
    rvq_decoder_weights weights;
    weights.quantizers.reserve((size_t) weights.num_quantizers);

    for (int i = 0; i < weights.num_quantizers; ++i) {
        const std::string prefix = "quantizer.quantizers." + std::to_string(i) + ".";

        rvq_quantizer_weights q;
        q.codebook_embed = load_codec_tensor(
                file,
                prefix + "codebook.embed",
                (size_t) weights.codebook_size * (size_t) weights.codebook_dim);
        q.project_in_weight = load_codec_tensor(
                file,
                prefix + "project_in.weight",
                (size_t) weights.codebook_dim * (size_t) weights.hidden_size);
        q.project_in_bias = load_codec_tensor(
                file,
                prefix + "project_in.bias",
                (size_t) weights.codebook_dim);
        q.project_out_weight = load_codec_tensor(
                file,
                prefix + "project_out.weight",
                (size_t) weights.hidden_size * (size_t) weights.codebook_dim);
        q.project_out_bias = load_codec_tensor(
                file,
                prefix + "project_out.bias",
                (size_t) weights.hidden_size);
        weights.quantizers.push_back(std::move(q));
    }

    weights.fc2_weight = load_codec_tensor(
            file,
            "fc2.weight",
            (size_t) weights.acoustic_size * (size_t) weights.hidden_size);
    weights.fc2_bias = load_codec_tensor(
            file,
            "fc2.bias",
            (size_t) weights.acoustic_size);

    return weights;
}

inline dac_encoder_weights load_dac_encoder_weights(const companion_file & file) {
    dac_encoder_weights weights;
    weights.conv1 = load_dac_conv1d(file, "acoustic_encoder.conv1", 64, 1, 7);

    const int in_channels[]  = {   64, 128, 256,  512, 1024 };
    const int out_channels[] = {  128, 256, 512, 1024, 2048 };
    const int dilations[]    = { 1, 3, 9 };

    for (int i = 0; i < 5; ++i) {
        const std::string prefix = "acoustic_encoder.block." + std::to_string(i);
        auto & block = weights.blocks[i];
        block.in_channels = in_channels[i];
        block.out_channels = out_channels[i];
        block.stride = weights.strides[(size_t) i];
        for (int r = 0; r < 3; ++r) {
            block.res_units[r] = load_dac_residual_unit(
                    file,
                    prefix + ".res_unit" + std::to_string(r + 1),
                    block.in_channels,
                    dilations[r]);
        }
        block.snake1_alpha = load_dac_alpha(file, prefix + ".snake1.alpha", block.in_channels);
        block.conv1 = load_dac_conv1d(
                file,
                prefix + ".conv1",
                block.out_channels,
                block.in_channels,
                2 * block.stride);
    }

    weights.snake1_alpha = load_dac_alpha(file, "acoustic_encoder.snake1.alpha", 2048);
    weights.conv2 = load_dac_conv1d(file, "acoustic_encoder.conv2", 256, 2048, 3);
    return weights;
}

inline dac_conv1d_weights load_dac_conv1d(
        const companion_file & file,
        const std::string & prefix,
        const int out_channels,
        const int in_channels,
        const int kernel) {
    dac_conv1d_weights conv;
    conv.in_channels = in_channels;
    conv.out_channels = out_channels;
    conv.kernel = kernel;
    conv.weight = load_codec_tensor(file, prefix + ".weight", (size_t) out_channels * (size_t) in_channels * (size_t) kernel);
    conv.bias = load_codec_tensor(file, prefix + ".bias", (size_t) out_channels);
    return conv;
}

inline dac_conv_transpose1d_weights load_dac_conv_transpose1d(
        const companion_file & file,
        const std::string & prefix,
        const int in_channels,
        const int out_channels,
        const int kernel,
        const int stride) {
    dac_conv_transpose1d_weights conv;
    conv.in_channels = in_channels;
    conv.out_channels = out_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.padding = (stride + 1) / 2;
    conv.output_padding = stride % 2;
    conv.weight = load_codec_tensor(file, prefix + ".weight", (size_t) in_channels * (size_t) out_channels * (size_t) kernel);
    conv.bias = load_codec_tensor(file, prefix + ".bias", (size_t) out_channels);
    return conv;
}

inline std::vector<float> load_dac_alpha(
        const companion_file & file,
        const std::string & name,
        const int channels) {
    return load_codec_tensor(file, name, (size_t) channels);
}

inline dac_residual_unit_weights load_dac_residual_unit(
        const companion_file & file,
        const std::string & prefix,
        const int channels,
        const int dilation) {
    dac_residual_unit_weights unit;
    unit.channels = channels;
    unit.dilation = dilation;
    unit.snake1_alpha = load_dac_alpha(file, prefix + ".snake1.alpha", channels);
    unit.snake2_alpha = load_dac_alpha(file, prefix + ".snake2.alpha", channels);
    unit.conv1 = load_dac_conv1d(file, prefix + ".conv1", channels, channels, 7);
    unit.conv2 = load_dac_conv1d(file, prefix + ".conv2", channels, channels, 1);
    return unit;
}

inline dac_decoder_weights load_dac_decoder_weights(const companion_file & file) {
    dac_decoder_weights weights;
    weights.conv1 = load_dac_conv1d(file, "acoustic_decoder.conv1", 1024, 256, 7);

    const int in_channels[]  = { 1024, 512, 256, 128, 64 };
    const int out_channels[] = {  512, 256, 128,  64, 32 };
    const int dilations[]    = { 1, 3, 9 };

    for (int i = 0; i < 5; ++i) {
        const std::string prefix = "acoustic_decoder.block." + std::to_string(i);
        auto & block = weights.blocks[i];
        block.in_channels = in_channels[i];
        block.out_channels = out_channels[i];
        block.stride = weights.strides[(size_t) i];
        block.snake1_alpha = load_dac_alpha(file, prefix + ".snake1.alpha", block.in_channels);
        block.conv_t1 = load_dac_conv_transpose1d(
                file,
                prefix + ".conv_t1",
                block.in_channels,
                block.out_channels,
                2 * block.stride,
                block.stride);

        for (int r = 0; r < 3; ++r) {
            block.res_units[r] = load_dac_residual_unit(
                    file,
                    prefix + ".res_unit" + std::to_string(r + 1),
                    block.out_channels,
                    dilations[r]);
        }
    }

    weights.snake1_alpha = load_dac_alpha(file, "acoustic_decoder.snake1.alpha", 32);
    weights.conv2 = load_dac_conv1d(file, "acoustic_decoder.conv2", 1, 32, 7);
    return weights;
}

inline std::vector<float> decode_rvq_hidden_frame(
        const rvq_decoder_weights & weights,
        const std::vector<int> & codes) {
    if ((int) codes.size() != weights.num_quantizers) {
        throw std::invalid_argument("RVQ code count does not match num_quantizers");
    }

    std::vector<float> hidden((size_t) weights.hidden_size, 0.0f);

    for (int qidx = 0; qidx < weights.num_quantizers; ++qidx) {
        const int code = codes[(size_t) qidx];
        if (code < 0 || code >= weights.codebook_size) {
            throw std::out_of_range("RVQ code is outside the tokenizer codebook range");
        }

        const auto & q = weights.quantizers[(size_t) qidx];
        const size_t code_offset = (size_t) code * (size_t) weights.codebook_dim;

        for (int out = 0; out < weights.hidden_size; ++out) {
            float v = q.project_out_bias[(size_t) out];
            const size_t w_off = (size_t) out * (size_t) weights.codebook_dim;
            for (int k = 0; k < weights.codebook_dim; ++k) {
                v += q.project_out_weight[w_off + (size_t) k] * q.codebook_embed[code_offset + (size_t) k];
            }
            hidden[(size_t) out] += v;
        }
    }

    return hidden;
}

inline std::vector<float> project_acoustic_frame(
        const rvq_decoder_weights & weights,
        const std::vector<float> & hidden) {
    if ((int) hidden.size() != weights.hidden_size) {
        throw std::invalid_argument("RVQ hidden size does not match fc2 input size");
    }

    std::vector<float> acoustic((size_t) weights.acoustic_size, 0.0f);
    for (int out = 0; out < weights.acoustic_size; ++out) {
        float v = weights.fc2_bias[(size_t) out];
        const size_t w_off = (size_t) out * (size_t) weights.hidden_size;
        for (int k = 0; k < weights.hidden_size; ++k) {
            v += weights.fc2_weight[w_off + (size_t) k] * hidden[(size_t) k];
        }
        acoustic[(size_t) out] = v;
    }
    return acoustic;
}

inline std::vector<float> decode_rvq_acoustic_latents(
        const rvq_decoder_weights & weights,
        const std::vector<std::vector<int>> & codec_frames) {
    std::vector<float> acoustic;
    acoustic.reserve(codec_frames.size() * (size_t) weights.acoustic_size);

    for (const auto & frame : codec_frames) {
        const auto hidden = decode_rvq_hidden_frame(weights, frame);
        const auto projected = project_acoustic_frame(weights, hidden);
        acoustic.insert(acoustic.end(), projected.begin(), projected.end());
    }

    return acoustic;
}

inline bool all_finite(const std::vector<float> & values) {
    for (const float v : values) {
        if (!std::isfinite(v)) {
            return false;
        }
    }
    return true;
}

inline std::vector<float> snake_1d(
        const std::vector<float> & input,
        const int channels,
        const int length,
        const std::vector<float> & alpha) {
    if ((int) alpha.size() != channels || (int) input.size() != channels * length) {
        throw std::invalid_argument("invalid Snake1d input shape");
    }

    std::vector<float> output(input.size());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int c = 0; c < channels; ++c) {
        const float a = alpha[(size_t) c];
        const float inv = 1.0f / (a + 1.0e-9f);
        for (int t = 0; t < length; ++t) {
            const size_t idx = (size_t) c * (size_t) length + (size_t) t;
            const float s = std::sin(a * input[idx]);
            output[idx] = input[idx] + inv * s * s;
        }
    }
    return output;
}

inline std::vector<float> conv1d_same(
        const std::vector<float> & input,
        const int length,
        const dac_conv1d_weights & conv,
        const int padding,
        const int dilation = 1) {
    if ((int) input.size() != conv.in_channels * length) {
        throw std::invalid_argument("invalid Conv1d input shape");
    }

    const int output_length = (length + 2 * padding - dilation * (conv.kernel - 1) - 1) + 1;
    if (output_length <= 0) {
        throw std::invalid_argument("invalid Conv1d output length");
    }

    std::vector<float> output((size_t) conv.out_channels * (size_t) output_length, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int oc = 0; oc < conv.out_channels; ++oc) {
        for (int t = 0; t < output_length; ++t) {
            float v = conv.bias[(size_t) oc];
            for (int ic = 0; ic < conv.in_channels; ++ic) {
                for (int k = 0; k < conv.kernel; ++k) {
                    const int in_t = t - padding + k * dilation;
                    if (in_t < 0 || in_t >= length) {
                        continue;
                    }
                    const size_t w_idx = ((size_t) oc * (size_t) conv.in_channels + (size_t) ic) * (size_t) conv.kernel + (size_t) k;
                    const size_t x_idx = (size_t) ic * (size_t) length + (size_t) in_t;
                    v += conv.weight[w_idx] * input[x_idx];
                }
            }
            output[(size_t) oc * (size_t) output_length + (size_t) t] = v;
        }
    }
    return output;
}

inline std::vector<float> conv1d_strided(
        const std::vector<float> & input,
        const int length,
        const dac_conv1d_weights & conv,
        const int padding,
        const int stride,
        const int dilation = 1) {
    if ((int) input.size() != conv.in_channels * length) {
        throw std::invalid_argument("invalid strided Conv1d input shape");
    }
    if (stride <= 0) {
        throw std::invalid_argument("invalid strided Conv1d stride");
    }

    const int output_length = (length + 2 * padding - dilation * (conv.kernel - 1) - 1) / stride + 1;
    if (output_length <= 0) {
        throw std::invalid_argument("invalid strided Conv1d output length");
    }

    std::vector<float> output((size_t) conv.out_channels * (size_t) output_length, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int oc = 0; oc < conv.out_channels; ++oc) {
        for (int t = 0; t < output_length; ++t) {
            float v = conv.bias[(size_t) oc];
            const int center = t * stride;
            for (int ic = 0; ic < conv.in_channels; ++ic) {
                for (int k = 0; k < conv.kernel; ++k) {
                    const int in_t = center - padding + k * dilation;
                    if (in_t < 0 || in_t >= length) {
                        continue;
                    }
                    const size_t w_idx = ((size_t) oc * (size_t) conv.in_channels + (size_t) ic) * (size_t) conv.kernel + (size_t) k;
                    const size_t x_idx = (size_t) ic * (size_t) length + (size_t) in_t;
                    v += conv.weight[w_idx] * input[x_idx];
                }
            }
            output[(size_t) oc * (size_t) output_length + (size_t) t] = v;
        }
    }
    return output;
}

inline std::vector<float> conv_transpose1d(
        const std::vector<float> & input,
        const int length,
        const dac_conv_transpose1d_weights & conv) {
    if ((int) input.size() != conv.in_channels * length) {
        throw std::invalid_argument("invalid ConvTranspose1d input shape");
    }

    const int output_length = (length - 1) * conv.stride - 2 * conv.padding + conv.kernel + conv.output_padding;
    if (output_length <= 0) {
        throw std::invalid_argument("invalid ConvTranspose1d output length");
    }

    std::vector<float> output((size_t) conv.out_channels * (size_t) output_length, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int oc = 0; oc < conv.out_channels; ++oc) {
        float * out = output.data() + (size_t) oc * (size_t) output_length;
        std::fill(out, out + output_length, conv.bias[(size_t) oc]);

        for (int ic = 0; ic < conv.in_channels; ++ic) {
            const float * in = input.data() + (size_t) ic * (size_t) length;
            const float * w = conv.weight.data() + ((size_t) ic * (size_t) conv.out_channels + (size_t) oc) * (size_t) conv.kernel;
            for (int t = 0; t < length; ++t) {
                const float x = in[(size_t) t];
                for (int k = 0; k < conv.kernel; ++k) {
                    const int out_t = t * conv.stride - conv.padding + k;
                    if (out_t < 0 || out_t >= output_length) {
                        continue;
                    }
                    out[(size_t) out_t] += w[(size_t) k] * x;
                }
            }
        }
    }

    return output;
}

inline std::vector<float> dac_residual_unit_forward(
        const dac_residual_unit_weights & unit,
        const std::vector<float> & input,
        const int length) {
    auto x = snake_1d(input, unit.channels, length, unit.snake1_alpha);
    x = conv1d_same(x, length, unit.conv1, 3 * unit.dilation, unit.dilation);

    const int after_conv1_length = (int) x.size() / unit.channels;
    x = snake_1d(x, unit.channels, after_conv1_length, unit.snake2_alpha);
    x = conv1d_same(x, after_conv1_length, unit.conv2, 0, 1);

    const int out_length = (int) x.size() / unit.channels;
    const int crop = (length - out_length) / 2;
    if (crop < 0) {
        throw std::runtime_error("DAC residual unit produced a longer output than input");
    }

    std::vector<float> output(x.size());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int c = 0; c < unit.channels; ++c) {
        for (int t = 0; t < out_length; ++t) {
            const size_t out_idx = (size_t) c * (size_t) out_length + (size_t) t;
            const size_t in_idx = (size_t) c * (size_t) length + (size_t) (t + crop);
            output[out_idx] = input[in_idx] + x[out_idx];
        }
    }
    return output;
}

inline std::vector<float> dac_decode_pcm(
        const dac_decoder_weights & weights,
        const std::vector<float> & acoustic_latents,
        const int frames) {
    if (frames <= 0 || (int) acoustic_latents.size() != frames * weights.acoustic_size) {
        throw std::invalid_argument("invalid DAC acoustic latent shape");
    }

    std::vector<float> x((size_t) weights.acoustic_size * (size_t) frames);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int t = 0; t < frames; ++t) {
        for (int c = 0; c < weights.acoustic_size; ++c) {
            x[(size_t) c * (size_t) frames + (size_t) t] =
                    acoustic_latents[(size_t) t * (size_t) weights.acoustic_size + (size_t) c];
        }
    }

    x = conv1d_same(x, frames, weights.conv1, 3, 1);
    int length = frames;
    int channels = 1024;

    for (int i = 0; i < 5; ++i) {
        const auto & block = weights.blocks[i];
        x = snake_1d(x, channels, length, block.snake1_alpha);
        x = conv_transpose1d(x, length, block.conv_t1);
        length = (int) x.size() / block.out_channels;
        channels = block.out_channels;

        for (const auto & unit : block.res_units) {
            x = dac_residual_unit_forward(unit, x, length);
            length = (int) x.size() / channels;
        }
    }

    x = snake_1d(x, channels, length, weights.snake1_alpha);
    x = conv1d_same(x, length, weights.conv2, 3, 1);
    return x;
}

inline std::vector<float> dac_encode_acoustic_latents(
        const dac_encoder_weights & weights,
        const std::vector<float> & pcm_mono) {
    if (pcm_mono.empty()) {
        return {};
    }

    std::vector<float> x = pcm_mono;
    x = conv1d_same(x, (int) pcm_mono.size(), weights.conv1, 3, 1);
    int length = (int) x.size() / 64;
    int channels = 64;

    for (int i = 0; i < 5; ++i) {
        const auto & block = weights.blocks[i];
        if (channels != block.in_channels) {
            throw std::runtime_error("DAC encoder channel accounting mismatch");
        }
        for (const auto & unit : block.res_units) {
            x = dac_residual_unit_forward(unit, x, length);
            length = (int) x.size() / channels;
        }
        x = snake_1d(x, channels, length, block.snake1_alpha);
        x = conv1d_strided(x, length, block.conv1, (block.stride + 1) / 2, block.stride, 1);
        channels = block.out_channels;
        length = (int) x.size() / channels;
    }

    x = snake_1d(x, channels, length, weights.snake1_alpha);
    x = conv1d_same(x, length, weights.conv2, 1, 1);
    length = (int) x.size() / weights.acoustic_size;

    std::vector<float> acoustic((size_t) length * (size_t) weights.acoustic_size);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int t = 0; t < length; ++t) {
        for (int c = 0; c < weights.acoustic_size; ++c) {
            acoustic[(size_t) t * (size_t) weights.acoustic_size + (size_t) c] =
                    x[(size_t) c * (size_t) length + (size_t) t];
        }
    }
    return acoustic;
}

inline std::vector<std::vector<int>> encode_rvq_codes(
        const rvq_decoder_weights & weights,
        const std::vector<float> & hidden_frames) {
    if (hidden_frames.empty()) {
        return {};
    }
    if (hidden_frames.size() % (size_t) weights.hidden_size != 0) {
        throw std::invalid_argument("RVQ encoder hidden frames do not align to hidden size");
    }
    if ((int) weights.quantizers.size() != weights.num_quantizers) {
        throw std::invalid_argument("RVQ quantizer count does not match num_quantizers");
    }
    if (weights.codebook_dim != 64) {
        throw std::runtime_error("native RVQ encoder currently expects codebook_dim=64");
    }

    const int frames = (int) (hidden_frames.size() / (size_t) weights.hidden_size);
    std::vector<float> residual = hidden_frames;
    std::vector<std::vector<int>> codes((size_t) frames, std::vector<int>((size_t) weights.num_quantizers, 0));

    for (int qidx = 0; qidx < weights.num_quantizers; ++qidx) {
        const auto & q = weights.quantizers[(size_t) qidx];
        if ((int) q.project_in_weight.size() != weights.codebook_dim * weights.hidden_size ||
                (int) q.project_in_bias.size() != weights.codebook_dim) {
            throw std::invalid_argument("invalid RVQ project_in tensor shape");
        }

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (int t = 0; t < frames; ++t) {
            float projected[64];
            for (int d = 0; d < weights.codebook_dim; ++d) {
                float v = q.project_in_bias[(size_t) d];
                const size_t w_off = (size_t) d * (size_t) weights.hidden_size;
                const size_t x_off = (size_t) t * (size_t) weights.hidden_size;
                for (int h = 0; h < weights.hidden_size; ++h) {
                    v += q.project_in_weight[w_off + (size_t) h] * residual[x_off + (size_t) h];
                }
                projected[d] = v;
            }

            int best_code = 0;
            float best_dist = std::numeric_limits<float>::infinity();
            for (int code = 0; code < weights.codebook_size; ++code) {
                const size_t e_off = (size_t) code * (size_t) weights.codebook_dim;
                float dist = 0.0f;
                for (int d = 0; d < weights.codebook_dim; ++d) {
                    const float diff = projected[d] - q.codebook_embed[e_off + (size_t) d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_code = code;
                }
            }
            codes[(size_t) t][(size_t) qidx] = best_code;
        }

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (int t = 0; t < frames; ++t) {
            const int code = codes[(size_t) t][(size_t) qidx];
            const size_t code_offset = (size_t) code * (size_t) weights.codebook_dim;
            const size_t frame_offset = (size_t) t * (size_t) weights.hidden_size;
            for (int out = 0; out < weights.hidden_size; ++out) {
                float v = q.project_out_bias[(size_t) out];
                const size_t w_off = (size_t) out * (size_t) weights.codebook_dim;
                for (int k = 0; k < weights.codebook_dim; ++k) {
                    v += q.project_out_weight[w_off + (size_t) k] * q.codebook_embed[code_offset + (size_t) k];
                }
                residual[frame_offset + (size_t) out] -= v;
            }
        }
    }

    return codes;
}

#include "higgs-hubert.h"

} // namespace higgs_audio
