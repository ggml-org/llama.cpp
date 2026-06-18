#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include "higgs-codec.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace higgs_audio {

struct dac_backend_deleter {
    void operator()(ggml_backend_t backend) const {
        if (backend) {
            ggml_backend_free(backend);
        }
    }
};

struct dac_backend_buffer_deleter {
    void operator()(ggml_backend_buffer_t buffer) const {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

struct dac_ggml_context_deleter {
    void operator()(ggml_context * ctx) const {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

class dac_decoder_backend {
public:
    explicit dac_decoder_backend(const std::string & requested_device) {
        ggml_backend_dev_t dev = select_device(requested_device);
        if (!dev) {
            throw std::runtime_error("no matching ggml backend device is available for Higgs DAC decoder");
        }

        backend.reset(ggml_backend_dev_init(dev, nullptr));
        if (!backend) {
            throw std::runtime_error("failed to initialize ggml backend device for Higgs DAC decoder");
        }
        backend_name = ggml_backend_dev_name(dev);
        const enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
        cache_graph = dev_type == GGML_BACKEND_DEVICE_TYPE_CPU;
        conv_weight_type = cache_graph ? GGML_TYPE_F16 : GGML_TYPE_F32;
    }

    const std::string & name() const {
        return backend_name;
    }

    std::vector<float> decode_pcm(
            const dac_decoder_weights & weights,
            const std::vector<float> & acoustic_latents,
            const int frames) {
        if (frames <= 0 || (int) acoustic_latents.size() != frames * weights.acoustic_size) {
            throw std::invalid_argument("invalid DAC acoustic latent shape");
        }

        graph_t & graph = get_or_build_graph(weights, frames);

        std::vector<float> input_chw((size_t) frames * (size_t) weights.acoustic_size);
        for (int t = 0; t < frames; ++t) {
            for (int c = 0; c < weights.acoustic_size; ++c) {
                input_chw[(size_t) t + (size_t) frames * (size_t) c] =
                        acoustic_latents[(size_t) t * (size_t) weights.acoustic_size + (size_t) c];
            }
        }
        ggml_backend_tensor_set(graph.input, input_chw.data(), 0, input_chw.size() * sizeof(float));

        const ggml_status status = ggml_backend_graph_compute(backend.get(), graph.graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Higgs DAC backend graph compute failed");
        }

        std::vector<float> pcm((size_t) graph.output->ne[0]);
        ggml_backend_tensor_get(graph.output, pcm.data(), 0, pcm.size() * sizeof(float));
        if (!cache_graph) {
            buffer.reset();
            cached_graph.reset();
            cached_weights = nullptr;
            cached_frames = 0;
        }
        return pcm;
    }

private:
    struct owned_tensor {
        ggml_tensor * tensor = nullptr;
        std::vector<float> data;
        std::vector<ggml_fp16_t> data_f16;
    };

    struct graph_t {
        std::unique_ptr<ggml_context, dac_ggml_context_deleter> ctx;
        ggml_cgraph * graph = nullptr;
        const dac_decoder_weights * weights = nullptr;
        int frames = 0;
        ggml_tensor * input = nullptr;
        ggml_tensor * output = nullptr;
        std::vector<owned_tensor> owned;
    };

    graph_t & get_or_build_graph(const dac_decoder_weights & weights, const int frames) {
        if (cached_graph && cached_weights == &weights && cached_frames == frames) {
            return *cached_graph;
        }

        buffer.reset();
        cached_graph.reset();
        cached_weights = nullptr;
        cached_frames = 0;

        cached_graph = std::make_unique<graph_t>();
        cached_graph->weights = &weights;
        cached_graph->frames = frames;
        build_graph(*cached_graph);

        buffer.reset(ggml_backend_alloc_ctx_tensors(cached_graph->ctx.get(), backend.get()));
        if (!buffer) {
            cached_graph.reset();
            throw std::runtime_error("failed to allocate Higgs DAC tensors on selected backend");
        }
        set_tensor_data(*cached_graph);
        cached_weights = &weights;
        cached_frames = frames;
        return *cached_graph;
    }

    static ggml_backend_dev_t select_device(const std::string & requested_device) {
        if (!requested_device.empty() && requested_device != "auto") {
            return ggml_backend_dev_by_name(requested_device.c_str());
        }

        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!dev) {
                continue;
            }
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU || type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                return dev;
            }
        }
        return nullptr;
    }

    ggml_tensor * add_owned_tensor(
            graph_t & graph,
            const std::vector<float> & data,
            const std::vector<int64_t> & ne,
            const ggml_type type = GGML_TYPE_F32) {
        owned_tensor owned;
        if (type == GGML_TYPE_F16) {
            owned.data_f16.resize(data.size());
            for (size_t i = 0; i < data.size(); ++i) {
                owned.data_f16[i] = ggml_fp32_to_fp16(data[i]);
            }
        } else {
            owned.data = data;
        }
        owned.tensor = ggml_new_tensor(graph.ctx.get(), type, (int) ne.size(), ne.data());
        graph.owned.push_back(std::move(owned));
        return graph.owned.back().tensor;
    }

    ggml_tensor * add_bias_tensor(graph_t & graph, const std::vector<float> & bias) {
        return add_owned_tensor(graph, bias, { 1, (int64_t) bias.size(), 1 });
    }

    ggml_tensor * add_alpha_tensor(graph_t & graph, const std::vector<float> & alpha) {
        return add_owned_tensor(graph, alpha, { 1, (int64_t) alpha.size(), 1 });
    }

    static std::vector<float> reorder_conv1d_weight(const dac_conv1d_weights & conv) {
        std::vector<float> out((size_t) conv.kernel * (size_t) conv.in_channels * (size_t) conv.out_channels);
        for (int oc = 0; oc < conv.out_channels; ++oc) {
            for (int ic = 0; ic < conv.in_channels; ++ic) {
                for (int k = 0; k < conv.kernel; ++k) {
                    const size_t src = ((size_t) oc * (size_t) conv.in_channels + (size_t) ic) * (size_t) conv.kernel + (size_t) k;
                    const size_t dst = (size_t) k
                            + (size_t) conv.kernel * ((size_t) ic + (size_t) conv.in_channels * (size_t) oc);
                    out[dst] = conv.weight[src];
                }
            }
        }
        return out;
    }

    static std::vector<float> reorder_conv_transpose1d_col2im_weight(const dac_conv_transpose1d_weights & conv) {
        std::vector<float> out((size_t) conv.in_channels * (size_t) conv.kernel * (size_t) conv.out_channels);
        for (int ic = 0; ic < conv.in_channels; ++ic) {
            for (int oc = 0; oc < conv.out_channels; ++oc) {
                for (int k = 0; k < conv.kernel; ++k) {
                    const size_t src = ((size_t) ic * (size_t) conv.out_channels + (size_t) oc) * (size_t) conv.kernel + (size_t) k;
                    const size_t dst = (size_t) ic
                            + (size_t) conv.in_channels * ((size_t) k + (size_t) conv.kernel * (size_t) oc);
                    out[dst] = conv.weight[src];
                }
            }
        }
        return out;
    }

    ggml_tensor * conv1d(graph_t & graph, ggml_tensor * x, const dac_conv1d_weights & conv, const int padding, const int dilation) {
        ggml_tensor * weight = add_owned_tensor(
                graph,
                reorder_conv1d_weight(conv),
                { conv.kernel, conv.in_channels, conv.out_channels },
                conv_weight_type);
        ggml_tensor * y = ggml_conv_1d(graph.ctx.get(), weight, x, 1, padding, dilation);
        ggml_tensor * bias = add_bias_tensor(graph, conv.bias);
        y = ggml_add(graph.ctx.get(), y, ggml_repeat(graph.ctx.get(), bias, y));
        return y;
    }

    ggml_tensor * conv_transpose1d(graph_t & graph, ggml_tensor * x, const dac_conv_transpose1d_weights & conv) {
        ggml_tensor * weight = add_owned_tensor(
                graph,
                reorder_conv_transpose1d_col2im_weight(conv),
                { conv.in_channels, conv.kernel * conv.out_channels },
                conv_weight_type);
        ggml_tensor * cols = ggml_mul_mat(graph.ctx.get(), weight, ggml_cont(graph.ctx.get(), ggml_transpose(graph.ctx.get(), x)));
        ggml_tensor * y = ggml_col2im_1d(graph.ctx.get(), cols, conv.stride, conv.out_channels, conv.padding);

        const int desired = (int) ((x->ne[0] - 1) * conv.stride - 2 * conv.padding + conv.kernel + conv.output_padding);
        if (desired <= 0 || desired < y->ne[0]) {
            throw std::runtime_error("invalid DAC ConvTranspose1d crop");
        }
        if (desired > y->ne[0]) {
            y = ggml_pad(graph.ctx.get(), y, desired - y->ne[0], 0, 0, 0);
        }

        ggml_tensor * bias = add_bias_tensor(graph, conv.bias);
        y = ggml_add(graph.ctx.get(), y, ggml_repeat(graph.ctx.get(), bias, y));
        return y;
    }

    ggml_tensor * snake(graph_t & graph, ggml_tensor * x, const std::vector<float> & alpha) {
        std::vector<float> inv(alpha.size());
        for (size_t i = 0; i < alpha.size(); ++i) {
            inv[i] = 1.0f / (alpha[i] + 1.0e-9f);
        }

        ggml_tensor * a = add_alpha_tensor(graph, alpha);
        ggml_tensor * inv_a = add_alpha_tensor(graph, inv);
        ggml_tensor * ax = ggml_mul(graph.ctx.get(), x, a);
        ggml_tensor * s = ggml_sin(graph.ctx.get(), ax);
        ggml_tensor * s2 = ggml_sqr(graph.ctx.get(), s);
        return ggml_add(graph.ctx.get(), x, ggml_mul(graph.ctx.get(), s2, inv_a));
    }

    ggml_tensor * residual_unit(graph_t & graph, ggml_tensor * input, const dac_residual_unit_weights & unit) {
        ggml_tensor * x = snake(graph, input, unit.snake1_alpha);
        x = conv1d(graph, x, unit.conv1, 3 * unit.dilation, unit.dilation);
        x = snake(graph, x, unit.snake2_alpha);
        x = conv1d(graph, x, unit.conv2, 0, 1);
        if (x->ne[0] != input->ne[0]) {
            const int crop = (int) ((input->ne[0] - x->ne[0]) / 2);
            input = ggml_view_3d(graph.ctx.get(), input, x->ne[0], input->ne[1], 1, input->nb[1], input->nb[2], (size_t) crop * (size_t) input->nb[0]);
        }
        return ggml_add(graph.ctx.get(), input, x);
    }

    void build_graph(graph_t & graph) {
        ggml_init_params params {
            /*.mem_size   =*/ 64 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        graph.ctx.reset(ggml_init(params));
        if (!graph.ctx) {
            throw std::runtime_error("failed to initialize Higgs DAC ggml context");
        }

        const auto & weights = *graph.weights;
        graph.input = ggml_new_tensor_3d(graph.ctx.get(), GGML_TYPE_F32, graph.frames, weights.acoustic_size, 1);
        ggml_tensor * x = conv1d(graph, graph.input, weights.conv1, 3, 1);

        for (int i = 0; i < 5; ++i) {
            const auto & block = weights.blocks[i];
            x = snake(graph, x, block.snake1_alpha);
            x = conv_transpose1d(graph, x, block.conv_t1);
            for (const auto & unit : block.res_units) {
                x = residual_unit(graph, x, unit);
            }
        }

        x = snake(graph, x, weights.snake1_alpha);
        x = conv1d(graph, x, weights.conv2, 3, 1);
        graph.output = ggml_cont(graph.ctx.get(), ggml_reshape_1d(graph.ctx.get(), x, x->ne[0]));

        graph.graph = ggml_new_graph(graph.ctx.get());
        ggml_build_forward_expand(graph.graph, graph.output);
        if (!ggml_backend_supports_op(backend.get(), graph.output)) {
            throw std::runtime_error("selected ggml backend does not support Higgs DAC graph output");
        }
    }

    void set_tensor_data(graph_t & graph) {
        for (const auto & owned : graph.owned) {
            if (!owned.data_f16.empty()) {
                ggml_backend_tensor_set(owned.tensor, owned.data_f16.data(), 0, owned.data_f16.size() * sizeof(ggml_fp16_t));
            } else {
                ggml_backend_tensor_set(owned.tensor, owned.data.data(), 0, owned.data.size() * sizeof(float));
            }
        }
    }

    std::unique_ptr<ggml_backend, dac_backend_deleter> backend;
    std::unique_ptr<ggml_backend_buffer, dac_backend_buffer_deleter> buffer;
    std::unique_ptr<graph_t> cached_graph;
    const dac_decoder_weights * cached_weights = nullptr;
    int cached_frames = 0;
    std::string backend_name;
    bool cache_graph = false;
    ggml_type conv_weight_type = GGML_TYPE_F32;
};

} // namespace higgs_audio
