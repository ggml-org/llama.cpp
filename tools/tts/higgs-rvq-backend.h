#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include "higgs-codec.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace higgs_audio {

struct rvq_backend_deleter {
    void operator()(ggml_backend_t backend) const {
        if (backend) {
            ggml_backend_free(backend);
        }
    }
};

struct rvq_backend_buffer_deleter {
    void operator()(ggml_backend_buffer_t buffer) const {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

struct rvq_ggml_context_deleter {
    void operator()(ggml_context * ctx) const {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

class rvq_decoder_backend {
public:
    explicit rvq_decoder_backend(const std::string & requested_device) {
        ggml_backend_dev_t dev = select_device(requested_device);
        if (!dev) {
            throw std::runtime_error("no matching ggml backend device is available for Higgs RVQ decoder");
        }

        backend.reset(ggml_backend_dev_init(dev, nullptr));
        if (!backend) {
            throw std::runtime_error("failed to initialize ggml backend device for Higgs RVQ decoder");
        }
        backend_name = ggml_backend_dev_name(dev);
    }

    const std::string & name() const {
        return backend_name;
    }

    std::vector<float> decode_acoustic_latents(
            const rvq_decoder_weights & weights,
            const std::vector<std::vector<int>> & codec_frames) {
        if (codec_frames.empty()) {
            return {};
        }
        validate(weights, codec_frames);

        graph_t graph;
        graph.weights = &weights;
        graph.frames = (int) codec_frames.size();
        graph.codec_frames = &codec_frames;
        build_graph(graph);

        buffer.reset(ggml_backend_alloc_ctx_tensors(graph.ctx.get(), backend.get()));
        if (!buffer) {
            throw std::runtime_error("failed to allocate Higgs RVQ tensors on selected backend");
        }
        set_tensor_data(graph);

        const ggml_status status = ggml_backend_graph_compute(backend.get(), graph.graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Higgs RVQ backend graph compute failed");
        }

        std::vector<float> acoustic((size_t) graph.frames * (size_t) weights.acoustic_size);
        ggml_backend_tensor_get(graph.output, acoustic.data(), 0, acoustic.size() * sizeof(float));
        buffer.reset();
        return acoustic;
    }

private:
    struct owned_float_tensor {
        ggml_tensor * tensor = nullptr;
        std::vector<float> data;
    };

    struct owned_i32_tensor {
        ggml_tensor * tensor = nullptr;
        std::vector<int32_t> data;
    };

    struct graph_t {
        std::unique_ptr<ggml_context, rvq_ggml_context_deleter> ctx;
        ggml_cgraph * graph = nullptr;
        const rvq_decoder_weights * weights = nullptr;
        const std::vector<std::vector<int>> * codec_frames = nullptr;
        int frames = 0;
        ggml_tensor * output = nullptr;
        std::vector<owned_float_tensor> owned_f32;
        std::vector<owned_i32_tensor> owned_i32;
    };

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

    static void validate(
            const rvq_decoder_weights & weights,
            const std::vector<std::vector<int>> & codec_frames) {
        if ((int) weights.quantizers.size() != weights.num_quantizers) {
            throw std::invalid_argument("RVQ quantizer count does not match num_quantizers");
        }
        if ((int) weights.fc2_weight.size() != weights.acoustic_size * weights.hidden_size ||
                (int) weights.fc2_bias.size() != weights.acoustic_size) {
            throw std::invalid_argument("invalid RVQ fc2 tensor shape");
        }

        for (int qidx = 0; qidx < weights.num_quantizers; ++qidx) {
            const auto & q = weights.quantizers[(size_t) qidx];
            if ((int) q.codebook_embed.size() != weights.codebook_size * weights.codebook_dim ||
                    (int) q.project_out_weight.size() != weights.hidden_size * weights.codebook_dim ||
                    (int) q.project_out_bias.size() != weights.hidden_size) {
                throw std::invalid_argument("invalid RVQ quantizer tensor shape");
            }
        }

        for (const auto & frame : codec_frames) {
            if ((int) frame.size() != weights.num_quantizers) {
                throw std::invalid_argument("RVQ code count does not match num_quantizers");
            }
            for (const int code : frame) {
                if (code < 0 || code >= weights.codebook_size) {
                    throw std::out_of_range("RVQ code is outside the tokenizer codebook range");
                }
            }
        }
    }

    ggml_tensor * add_float_tensor(graph_t & graph, const std::vector<float> & data, const std::vector<int64_t> & ne) {
        owned_float_tensor owned;
        owned.data = data;
        owned.tensor = ggml_new_tensor(graph.ctx.get(), GGML_TYPE_F32, (int) ne.size(), ne.data());
        graph.owned_f32.push_back(std::move(owned));
        return graph.owned_f32.back().tensor;
    }

    ggml_tensor * add_i32_tensor(graph_t & graph, const std::vector<int32_t> & data, const std::vector<int64_t> & ne) {
        owned_i32_tensor owned;
        owned.data = data;
        owned.tensor = ggml_new_tensor(graph.ctx.get(), GGML_TYPE_I32, (int) ne.size(), ne.data());
        graph.owned_i32.push_back(std::move(owned));
        return graph.owned_i32.back().tensor;
    }

    void build_graph(graph_t & graph) {
        ggml_init_params params {
            /*.mem_size   =*/ 16 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        graph.ctx.reset(ggml_init(params));
        if (!graph.ctx) {
            throw std::runtime_error("failed to initialize Higgs RVQ ggml context");
        }

        const auto & weights = *graph.weights;
        ggml_tensor * hidden = nullptr;
        for (int qidx = 0; qidx < weights.num_quantizers; ++qidx) {
            const auto & q = weights.quantizers[(size_t) qidx];
            std::vector<int32_t> ids((size_t) graph.frames);
            for (int t = 0; t < graph.frames; ++t) {
                ids[(size_t) t] = (int32_t) (*graph.codec_frames)[(size_t) t][(size_t) qidx];
            }

            ggml_tensor * codebook = add_float_tensor(
                    graph,
                    q.codebook_embed,
                    { weights.codebook_dim, weights.codebook_size });
            ggml_tensor * code_ids = add_i32_tensor(graph, ids, { graph.frames });
            ggml_tensor * embed = ggml_get_rows(graph.ctx.get(), codebook, code_ids);

            ggml_tensor * project_weight = add_float_tensor(
                    graph,
                    q.project_out_weight,
                    { weights.codebook_dim, weights.hidden_size });
            ggml_tensor * q_hidden = ggml_mul_mat(graph.ctx.get(), project_weight, embed);

            ggml_tensor * bias = add_float_tensor(
                    graph,
                    q.project_out_bias,
                    { weights.hidden_size, 1 });
            q_hidden = ggml_add(graph.ctx.get(), q_hidden, ggml_repeat(graph.ctx.get(), bias, q_hidden));
            hidden = hidden ? ggml_add(graph.ctx.get(), hidden, q_hidden) : q_hidden;
        }

        if (!hidden) {
            throw std::runtime_error("failed to build Higgs RVQ hidden graph");
        }

        ggml_tensor * fc2 = add_float_tensor(
                graph,
                weights.fc2_weight,
                { weights.hidden_size, weights.acoustic_size });
        ggml_tensor * acoustic = ggml_mul_mat(graph.ctx.get(), fc2, hidden);
        ggml_tensor * bias = add_float_tensor(
                graph,
                weights.fc2_bias,
                { weights.acoustic_size, 1 });
        acoustic = ggml_add(graph.ctx.get(), acoustic, ggml_repeat(graph.ctx.get(), bias, acoustic));
        graph.output = ggml_cont(graph.ctx.get(), acoustic);

        graph.graph = ggml_new_graph(graph.ctx.get());
        ggml_build_forward_expand(graph.graph, graph.output);
        if (!ggml_backend_supports_op(backend.get(), graph.output)) {
            throw std::runtime_error("selected ggml backend does not support Higgs RVQ graph output");
        }
    }

    void set_tensor_data(graph_t & graph) {
        for (const auto & owned : graph.owned_f32) {
            ggml_backend_tensor_set(owned.tensor, owned.data.data(), 0, owned.data.size() * sizeof(float));
        }
        for (const auto & owned : graph.owned_i32) {
            ggml_backend_tensor_set(owned.tensor, owned.data.data(), 0, owned.data.size() * sizeof(int32_t));
        }
    }

    std::unique_ptr<ggml_backend, rvq_backend_deleter> backend;
    std::unique_ptr<ggml_backend_buffer, rvq_backend_buffer_deleter> buffer;
    std::string backend_name;
};

} // namespace higgs_audio
