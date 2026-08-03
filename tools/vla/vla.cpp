#include "vla.h"

#include "models/models.h"
#include "vla-impl.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

struct vla_context {
    std::unique_ptr<vla_model> model;
};

namespace {

std::string gguf_string(const gguf_context * ctx, const char * key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        return {};
    }
    return gguf_get_val_str(ctx, id);
}

int64_t gguf_u32(const gguf_context * ctx, const char * key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_UINT32) {
        return 0;
    }
    return gguf_get_val_u32(ctx, id);
}

} // namespace

bool vla_metadata_load(const char * path, vla_metadata & metadata) {
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * ctx = gguf_init_from_file(path, params);
    if (!ctx) {
        std::fprintf(stderr, "vla: failed to open GGUF '%s'\n", path);
        return false;
    }

    metadata.architecture     = gguf_string(ctx, "general.architecture");
    metadata.model_type       = gguf_string(ctx, "vla.model_type");
    metadata.state_dim        = gguf_u32(ctx, "vla.state_dim");
    metadata.control_dim      = gguf_u32(ctx, "vla.control_dim");
    metadata.control_horizon  = gguf_u32(ctx, "vla.control_horizon");
    metadata.conditioning_dim = gguf_u32(ctx, "vla.conditioning_dim");
    metadata.n_embodiments    = gguf_u32(ctx, "vla.n_embodiments");
    gguf_free(ctx);

    if (metadata.architecture != "vla") {
        std::fprintf(stderr, "vla: expected general.architecture=vla, got '%s'\n",
                metadata.architecture.c_str());
        return false;
    }
    if (metadata.model_type.empty()) {
        std::fprintf(stderr, "vla: missing vla.model_type\n");
        return false;
    }
    if (metadata.state_dim <= 0 || metadata.control_dim <= 0 ||
            metadata.control_horizon <= 0 || metadata.conditioning_dim <= 0 ||
            metadata.n_embodiments <= 0) {
        std::fprintf(stderr, "vla: invalid common dimensions in GGUF metadata\n");
        return false;
    }
    return true;
}

vla_context_params vla_context_params_default(void) {
    vla_context_params params = {
        /*.use_gpu  =*/ true,
        /*.n_threads =*/ (int32_t) std::max(1u, std::thread::hardware_concurrency() / 2),
    };
    return params;
}

vla_context * vla_init_from_file(
        const char *               path,
        const llama_model *        text_model,
        vla_context_params         params) {
    if (!path || path[0] == '\0') {
        std::fprintf(stderr, "vla: model path is empty\n");
        return nullptr;
    }

    vla_metadata metadata = {};
    if (!vla_metadata_load(path, metadata)) {
        return nullptr;
    }

    if (text_model && llama_model_n_embd_out(text_model) != metadata.conditioning_dim) {
        std::fprintf(stderr, "vla: text model output dimension %d does not match conditioning dimension %lld\n",
                llama_model_n_embd_out(text_model), (long long) metadata.conditioning_dim);
        return nullptr;
    }

    std::unique_ptr<vla_model> model;
    if (metadata.model_type == "minicpm_robot") {
        model = vla_model_minicpm_robot_create(path, params);
    } else {
        std::fprintf(stderr, "vla: unsupported model type '%s'\n", metadata.model_type.c_str());
        return nullptr;
    }
    if (!model) {
        return nullptr;
    }

    if (model->state_dim() != metadata.state_dim ||
            model->control_dim() != metadata.control_dim ||
            model->control_horizon() != metadata.control_horizon ||
            model->conditioning_dim() != metadata.conditioning_dim ||
            model->n_embodiments() != metadata.n_embodiments) {
        std::fprintf(stderr, "vla: model factory dimensions do not match common metadata\n");
        return nullptr;
    }

    auto * ctx = new vla_context();
    ctx->model = std::move(model);
    return ctx;
}

void vla_free(vla_context * ctx) {
    delete ctx;
}

const char * vla_model_type(const vla_context * ctx) {
    return ctx && ctx->model ? ctx->model->model_type() : nullptr;
}

int64_t vla_state_dim(const vla_context * ctx) {
    return ctx && ctx->model ? ctx->model->state_dim() : 0;
}

int64_t vla_control_dim(const vla_context * ctx) {
    return ctx && ctx->model ? ctx->model->control_dim() : 0;
}

int64_t vla_control_horizon(const vla_context * ctx) {
    return ctx && ctx->model ? ctx->model->control_horizon() : 0;
}

int64_t vla_conditioning_dim(const vla_context * ctx) {
    return ctx && ctx->model ? ctx->model->conditioning_dim() : 0;
}

int64_t vla_n_embodiments(const vla_context * ctx) {
    return ctx && ctx->model ? ctx->model->n_embodiments() : 0;
}

bool vla_predict(vla_context * ctx, const vla_input * input, vla_output * output) {
    if (!ctx || !ctx->model || !input || !output) {
        return false;
    }
    if (!input->embeddings || input->n_tokens <= 0 ||
            input->n_embd != ctx->model->conditioning_dim() ||
            !input->state || input->n_state != ctx->model->state_dim() ||
            (input->noise && input->n_noise !=
                ctx->model->control_horizon() * ctx->model->control_dim()) ||
            input->embodiment_id < 0 ||
            input->embodiment_id >= ctx->model->n_embodiments() ||
            !output->controls ||
            output->capacity < ctx->model->control_horizon() * ctx->model->control_dim()) {
        std::fprintf(stderr, "vla: invalid prediction input or output dimensions\n");
        return false;
    }
    return ctx->model->predict(*input, *output);
}
