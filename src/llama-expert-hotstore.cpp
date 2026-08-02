#include "llama-expert-hotstore.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"

#include <regex>

// matches the weight tensor of an expert tensor, e.g.:
//   blk.0.ffn_gate_exps.weight
//   blk.3.ffn_down_chexps.weight
// follows the same convention as LLM_FFN_EXPS_REGEX in common.h
static const std::regex g_re_exps_weight("blk\\.(\\d+)\\.ffn_(up|down|gate|gate_up)_(ch|)exps\\.weight");

llama_expert_hotstore::llama_expert_hotstore(
        const llama_model * model, int n_layers, int n_experts, int hot_s) :
    n_layers(n_layers),
    n_experts(n_experts),
    hot_s(hot_s),
    bytes_per_slot(n_layers, 0) {
    if (n_layers <= 0) {
        return;
    }

    for (const auto & [name, tensor] : llama_internal_get_tensor_map(model)) {
        std::smatch m;
        if (std::regex_search(name, m, g_re_exps_weight)) {
            const int il = std::stoi(m[1].str());
            if (il >= 0 && il < n_layers && tensor->ne[2] > 0) {
                // a slot holds nbytes/n_experts of this tensor
                bytes_per_slot[il] += ggml_nbytes(tensor) / (size_t) tensor->ne[2];
                entries.push_back({il, tensor, nullptr});
            }
        }
    }
}

bool llama_expert_hotstore::allocate(ggml_backend_buffer_type_t gpu_buft) {
    if (hot_s <= 0 || entries.empty()) {
        return false;
    }
    if (hot_s > n_experts) {
        throw std::runtime_error(format("%s: hot store S=%d exceeds n_experts=%d",
            __func__, hot_s, n_experts));
    }

    // a no_alloc context just for the hot tensor metadata
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * entries.size(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_context_ptr(ggml_init(params));
    if (!ctx) {
        LLAMA_LOG_ERROR("%s: hot store: failed to create ggml context\n", __func__);
        return false;
    }

    // one hot tensor per model expert tensor, trimmed to hot_s slots
    for (auto & e : entries) {
        e.dst = ggml_new_tensor_3d(ctx.get(), e.src->type, e.src->ne[0], e.src->ne[1], hot_s);
    }

    // check whether the buffer would fit before committing any VRAM
    const size_t need = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), gpu_buft);
    if (need == 0) {
        LLAMA_LOG_ERROR("%s: hot store: zero-sized buffer, disabled\n", __func__);
        ctx.reset();
        return false;
    }

    size_t free_mem = 0, total_mem = 0;
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(gpu_buft);
    if (dev) {
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
    }
    if (dev && free_mem < need) {
        throw std::runtime_error(format("%s: not enough memory to allocate the GPU hot store of %d slots (%zu MiB needed, %zu MiB free on %s)", 
            __func__, hot_s, need / (1024 * 1024), free_mem / (1024 * 1024),
            ggml_backend_dev_name(dev)));
    }

    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), gpu_buft);
    if (b == nullptr) {
        throw std::runtime_error(format("%s: unable to allocate hot store buffer of %d slots (%zu MiB)", 
            __func__, hot_s, need / (1024 * 1024)));
    }
    buf = ggml_backend_buffer_ptr(b);
    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    return true;
}

void llama_expert_hotstore::log() const {
    LLAMA_LOG("=== Expert hotstore sizing (S=%d) ===\n", hot_s);
    size_t total = 0;
    for (int il = 0; il < n_layers; il++) {
        total += bytes_per_slot[il];
        LLAMA_LOG("  layer %3d: bytes/slot = %zu\n", il, bytes_per_slot[il]);
    }
    LLAMA_LOG("  total bytes/slot across all layers = %zu (%zu MiB)\n",
        total, total / (1024 * 1024));
    if (buf) {
        LLAMA_LOG("  GPU hot store allocated: %s, %zu bytes (%zu MiB) for %d slots\n",
            ggml_backend_buffer_name(buf.get()),
            ggml_backend_buffer_get_size(buf.get()),
            ggml_backend_buffer_get_size(buf.get()) / (1024 * 1024),
            hot_s);
    } else if (hot_s > 0) {
        LLAMA_LOG("  hot store DISABLED (%d slots requested)\n", hot_s);
    }
}
