#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "higgs-sampler.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace higgs_audio {

struct backend_deleter {
    void operator()(ggml_backend_t backend) const {
        if (backend) {
            ggml_backend_free(backend);
        }
    }
};

struct backend_buffer_deleter {
    void operator()(ggml_backend_buffer_t buffer) const {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

struct ggml_context_deleter {
    void operator()(ggml_context * ctx) const {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

class codebook_backend {
public:
    codebook_backend(const codebook_layout & layout, const std::vector<float> & head_weight, const std::string & requested_device)
            : layout(layout), logits((std::size_t) layout.n_rows()) {
        if (layout.num_codebooks <= 0 || layout.codebook_size <= 0 || layout.n_embd <= 0) {
            throw std::invalid_argument("invalid codebook layout");
        }
        if (head_weight.size() != layout.n_weights()) {
            throw std::invalid_argument("head weight size does not match codebook layout");
        }

        ggml_backend_dev_t dev = select_device(requested_device);
        if (!dev) {
            throw std::runtime_error("no matching ggml backend device is available for Higgs codebook head");
        }

        backend.reset(ggml_backend_dev_init(dev, nullptr));
        if (!backend) {
            throw std::runtime_error("failed to initialize ggml backend device for Higgs codebook head");
        }

        backend_name = ggml_backend_dev_name(dev);

        ggml_init_params params {
            /*.mem_size   =*/ 4 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        ctx.reset(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to initialize Higgs codebook ggml context");
        }

        t_head   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, layout.n_embd, layout.n_rows());
        t_hidden = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, layout.n_embd);
        t_logits = ggml_mul_mat(ctx.get(), t_head, t_hidden);

        if (!t_head || !t_hidden || !t_logits) {
            throw std::runtime_error("failed to build Higgs codebook backend tensors");
        }

        ggml_set_name(t_head, "higgs.codebook_head");
        ggml_set_name(t_hidden, "higgs.hidden");
        ggml_set_name(t_logits, "higgs.logits");

        if (!ggml_backend_supports_op(backend.get(), t_logits)) {
            throw std::runtime_error("selected ggml backend does not support Higgs codebook matmul");
        }

        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        if (!buffer) {
            throw std::runtime_error("failed to allocate Higgs codebook tensors on selected backend");
        }
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(t_head, head_weight.data(), 0, head_weight.size() * sizeof(float));

        graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, t_logits);
    }

    const std::string & name() const {
        return backend_name;
    }

    const std::vector<float> & codebook_logits(const std::vector<float> & hidden) {
        if ((int) hidden.size() != layout.n_embd) {
            throw std::invalid_argument("hidden size does not match codebook layout");
        }

        ggml_backend_tensor_set(t_hidden, hidden.data(), 0, hidden.size() * sizeof(float));
        const ggml_status status = ggml_backend_graph_compute(backend.get(), graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Higgs codebook backend graph compute failed");
        }
        ggml_backend_tensor_get(t_logits, logits.data(), 0, logits.size() * sizeof(float));
        return logits;
    }

    std::vector<int> greedy_codebook_codes(const std::vector<float> & hidden) {
        const auto & values = codebook_logits(hidden);
        std::vector<int> codes((std::size_t) layout.num_codebooks, 0);
        for (int cb = 0; cb < layout.num_codebooks; ++cb) {
            const float * row = values.data() + (std::size_t) cb * (std::size_t) layout.codebook_size;
            int best_code = 0;
            float best = row[0];
            for (int code = 1; code < layout.codebook_size; ++code) {
                if (row[code] > best) {
                    best = row[code];
                    best_code = code;
                }
            }
            codes[(std::size_t) cb] = best_code;
        }
        return codes;
    }

    static std::vector<std::string> available_devices() {
        std::vector<std::string> names;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (dev) {
                names.emplace_back(ggml_backend_dev_name(dev));
            }
        }
        return names;
    }

private:
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

    codebook_layout layout;
    std::unique_ptr<ggml_backend, backend_deleter> backend;
    std::unique_ptr<ggml_context, ggml_context_deleter> ctx;
    std::unique_ptr<ggml_backend_buffer, backend_buffer_deleter> buffer;
    ggml_tensor * t_head   = nullptr;
    ggml_tensor * t_hidden = nullptr;
    ggml_tensor * t_logits = nullptr;
    ggml_cgraph * graph    = nullptr;
    std::vector<float> logits;
    std::string backend_name;
};

} // namespace higgs_audio
