#include "llama-kv-stream-resident.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

struct llama_kv_stream_resident::implementation {
    llama_kv_stream_binding_view binding;
    std::shared_ptr<llama_kv_stream_content> content;
    ggml_backend_t backend = nullptr;
    ggml_context_ptr context;
    llama_kv_stream_policy_layout layout;
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> roots;
    std::vector<llama_kv_stream_rows> ranges;
    size_t active = 0, bytes = 0, calls = 0;
    uint64_t generation = 0, epoch = 0;
    bool initialized = false, valid = false, busy = false;

    // Storage encoding must match the fixed device layout; replacement alone cannot change it.
    bool compatible() const {
        const auto host = content->host();
        const auto & a = binding.config.shape;
        const auto & b = host->config().shape;
        return binding.cache_id == host->cache_id() && binding.config.layers == host->config().layers &&
            a.type_k == b.type_k && a.type_v == b.type_v && a.head_dim_k == b.head_dim_k &&
            a.head_dim_v == b.head_dim_v && a.heads == b.heads && a.page_tokens == b.page_tokens && a.alignment == b.alignment;
    }

    // Ordinary CUDA attention needs padded keys, but every padded row must remain resident.
    bool extent(size_t tokens, size_t & padded) const {
        if (!compatible() || !tokens || tokens > content->host()->config().context_tokens || tokens > size_t(INT32_MAX) - 255) return false;
        padded = (tokens + 255)/256*256;
        for (const auto & entry : layout.layers) if (padded > entry.planes.tokens) return false;
        return padded <= content->host()->layout().tokens;
    }

    // Describe token-major heads using the same strides as the original resident fast path.
    ggml_tensor descriptor(uint32_t layer, bool value, size_t padded) const {
        ggml_tensor tensor = *(value ? roots[layer].second : roots[layer].first);
        const auto & shape = binding.config.shape;
        const auto & planes = layout.layers[layer].planes;
        tensor.ne[0] = value ? shape.head_dim_v : shape.head_dim_k;
        tensor.ne[1] = int64_t(padded);
        tensor.ne[2] = shape.heads;
        tensor.ne[3] = 1;
        tensor.nb[1] = value ? planes.v_token_bytes : planes.k_token_bytes;
        tensor.nb[2] = value ? planes.v_row_bytes : planes.k_row_bytes;
        tensor.nb[3] = value ? planes.v_bytes : planes.k_bytes;
        return tensor;
    }
};

// External captures must be retired before their binding execution pins are returned.
llama_kv_stream_resident::~llama_kv_stream_resident() = default;

// Construct only metadata and borrowed tensor bindings; do not copy or clear device memory in the factory.
std::unique_ptr<llama_kv_stream_resident> llama_kv_stream_resident::create(
        const llama_kv_stream_binding_view & binding, std::shared_ptr<llama_kv_stream_content> content, ggml_backend_t backend) {
    if (!content || !backend || !binding.buffer || !binding.base || binding.capacity != binding.config.pool_bytes ||
            binding.capacity > ggml_backend_buffer_get_size(binding.buffer) ||
            binding.base != ggml_backend_buffer_get_base(binding.buffer) ||
            !ggml_backend_supports_buft(backend, ggml_backend_buffer_get_type(binding.buffer)) ||
            binding.initial_policy.budget.page.attention != ggml_kv_stream_attention::direct) return {};
    const auto & shape = binding.config.shape;
    if (shape.page_tokens <= 0 || shape.page_tokens % 256 || shape.head_dim_k > INT32_MAX ||
            shape.head_dim_v > INT32_MAX || shape.heads > INT32_MAX) return {};
    try {
        std::unique_ptr<llama_kv_stream_resident> result(new llama_kv_stream_resident);
        result->impl = std::make_unique<implementation>();
        auto & s = *result->impl;
        s.binding = binding; s.content = std::move(content); s.backend = backend;
        if (!s.compatible() || llama_kv_stream_policy_layout_make(binding.config, binding.initial_policy, 0, s.layout).status !=
                llama_kv_stream_policy_status::success) return {};
        const size_t layers = s.layout.layers.size();
        if (layers > (SIZE_MAX/ggml_tensor_overhead() - 1)/2) return {};
        s.context.reset(ggml_init({(2*layers + 1)*ggml_tensor_overhead(), nullptr, true}));
        if (!s.context) return {};
        s.roots.reserve(layers);
        s.ranges.reserve(2*layers);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            const auto & entry = s.layout.layers[layer];
            ggml_tensor * roots[2];
            for (int value = 0; value < 2; ++value) {
                const auto type = ggml_type(value ? shape.type_v : shape.type_k);
                const size_t dim = size_t(value ? shape.head_dim_v : shape.head_dim_k);
                const size_t bytes = value ? entry.planes.v_bytes : entry.planes.k_bytes;
                if (entry.planes.tokens > size_t(INT64_MAX)/size_t(shape.heads)/dim) return {};
                const size_t elements = entry.planes.tokens*size_t(shape.heads)*dim;
                // Flat roots naturally satisfy quantized matrix padding without adding bytes between the K/V planes.
                roots[value] = ggml_new_tensor_1d(s.context.get(), type, int64_t(elements));
                if (ggml_nbytes(roots[value]) != bytes ||
                        ggml_backend_buffer_get_alloc_size(binding.buffer, roots[value]) != bytes) return {};
                const size_t offset = entry.offset + (value ? entry.planes.v_offset : 0);
                if (offset > binding.capacity || bytes > binding.capacity - offset) return {};
                if (ggml_backend_tensor_alloc(binding.buffer, roots[value], static_cast<char *>(binding.base) + offset) != GGML_STATUS_SUCCESS) return {};
            }
            s.roots.emplace_back(roots[0], roots[1]);
            s.ranges.push_back({layer, ggml_kv_stream_operand::k, 0, 0});
            s.ranges.push_back({layer, ggml_kv_stream_operand::v, 0, 0});
        }
        return result;
    } catch (const std::bad_alloc &) {
        return {};
    }
}

// Complete older attention before overwriting its inputs; acknowledgement follows completed synchronous copies.
bool llama_kv_stream_resident::synchronize(size_t active_tokens) {
    auto & s = *impl;
    if (s.busy) return false;
    s.valid = false; s.bytes = s.calls = 0;
    size_t padded;
    if (!s.extent(active_tokens, padded)) return false;
    s.busy = true;
    struct guard { bool & busy; ~guard() { busy = false; } } operation{s.busy};
    ggml_backend_synchronize(s.backend);
    if (!s.initialized) {
        if (!s.content->reset_mirror()) return false;
        s.initialized = true;
    }
    for (auto & range : s.ranges) range.count = padded;
    if (!s.content->flush(s.ranges, [&](const llama_kv_stream_copy_span & span) {
        const bool value = span.rows.operand == ggml_kv_stream_operand::v;
        const auto & planes = s.layout.layers[span.rows.layer].planes;
        const size_t stride = value ? planes.v_token_bytes : planes.k_token_bytes;
        auto * root = value ? s.roots[span.rows.layer].second : s.roots[span.rows.layer].first;
        ggml_backend_tensor_set(root, span.data, span.rows.first*stride, span.bytes);
        s.bytes += span.bytes; ++s.calls;
        return true;
    })) return false;
    s.active = active_tokens;
    s.generation = s.content->generation();
    s.epoch = s.content->mirror_epoch();
    s.valid = true;
    return true;
}

// Physical address stability alone does not prove that the resident contents are current.
bool llama_kv_stream_resident::ready(size_t active_tokens) const noexcept {
    const auto & s = *impl;
    return !s.busy && s.valid && s.active == active_tokens && s.generation == s.content->generation() && s.epoch == s.content->mirror_epoch();
}
// Report completed uploads in the last synchronization attempt, not PCIe utilization.
size_t llama_kv_stream_resident::last_upload_bytes() const noexcept { return impl->bytes; }
size_t llama_kv_stream_resident::last_upload_calls() const noexcept { return impl->calls; }

// Validate stack descriptors before GGML constructors can assert on malformed attention metadata.
ggml_tensor * llama_kv_stream_resident::attention(ggml_context * context, uint32_t layer, ggml_tensor * q,
        ggml_tensor * mask, size_t active_tokens, float scale) {
    auto & s = *impl;
    size_t padded;
    if (s.busy || !context || !q || layer >= s.roots.size() || !std::isfinite(scale) || scale <= 0 ||
            !s.extent(active_tokens, padded) || (!mask && active_tokens != padded) ||
            (mask && !ggml_is_contiguous(mask))) return nullptr;
    auto k = s.descriptor(layer, false, padded), v = s.descriptor(layer, true, padded);
    ggml_tensor prototype = {};
    prototype.type = GGML_TYPE_F32; prototype.op = GGML_OP_FLASH_ATTN_EXT;
    prototype.ne[0] = v.ne[0]; prototype.ne[1] = q->ne[2]; prototype.ne[2] = q->ne[1]; prototype.ne[3] = 1;
    prototype.nb[0] = sizeof(float);
    for (int i = 0; i < 3; ++i) {
        if (prototype.ne[i] <= 0 || uint64_t(prototype.ne[i]) > SIZE_MAX/prototype.nb[i]) return nullptr;
        prototype.nb[i+1] = prototype.nb[i]*size_t(prototype.ne[i]);
    }
    prototype.src[0] = q; prototype.src[1] = &k; prototype.src[2] = &v; prototype.src[3] = mask;
    std::memcpy(prototype.op_params, &scale, sizeof(scale));
    ggml_flash_attn_ext_set_prec(&prototype, GGML_PREC_F32);
    const auto & shape = s.binding.config.shape;
    const ggml_kv_stream_attention_limits limits{shape.head_dim_k, shape.head_dim_v, 256, shape.page_tokens, shape.alignment};
    ggml_kv_stream_execution execution;
    if (ggml_kv_stream_attention_validate(&prototype, limits, s.binding.config.capabilities, padded, execution).status !=
            ggml_kv_stream_status::success || execution.attention != ggml_kv_stream_attention::direct ||
            !ggml_backend_supports_op(s.backend, &prototype)) return nullptr;
    auto * kt = ggml_view_4d(context, s.roots[layer].first, k.ne[0], k.ne[1], k.ne[2], 1, k.nb[1], k.nb[2], k.nb[3], 0);
    auto * vt = ggml_view_4d(context, s.roots[layer].second, v.ne[0], v.ne[1], v.ne[2], 1, v.nb[1], v.nb[2], v.nb[3], 0);
    if (ggml_backend_view_init(kt) != GGML_STATUS_SUCCESS || ggml_backend_view_init(vt) != GGML_STATUS_SUCCESS) return nullptr;
    auto * out = ggml_flash_attn_ext(context, q, kt, vt, mask, scale, 0, 0);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    return out;
}
