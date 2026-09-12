#include "llama-kv-stream-host.h"

#include <cstring>
#include <limits>
#include <new>

// Validate byte geometry before touching an allocator.
static bool host_plan(const llama_kv_stream_host_config & c, ggml_kv_stream_layout & planes,
        size_t & stride, size_t & bytes) {
    if (!c.cache_id || !c.layers || !c.context_tokens || c.shape.page_tokens <= 0 ||
            uint64_t(c.shape.page_tokens) > SIZE_MAX) return false;
    ggml_kv_stream_execution page;
    if (ggml_kv_stream_resolve(c.shape, c.capabilities, size_t(c.shape.page_tokens), page).status != ggml_kv_stream_status::success) return false;
    const size_t page_tokens = size_t(c.shape.page_tokens);
    const size_t tail = c.context_tokens % page_tokens;
    const size_t pad = tail ? page_tokens - tail : 0;
    if (c.context_tokens > SIZE_MAX - pad) return false;
    if (ggml_kv_stream_layout_make(c.shape, c.context_tokens + pad, planes).status != ggml_kv_stream_status::success) return false;
    const size_t remainder = planes.bytes % c.shape.alignment;
    const size_t alignment_pad = remainder ? c.shape.alignment - remainder : 0;
    if (planes.bytes > SIZE_MAX - alignment_pad) return false;
    stride = planes.bytes + alignment_pad;
    if (!stride || c.layers > SIZE_MAX / stride) return false;
    bytes = stride*c.layers;
    return bytes <= SIZE_MAX - (c.shape.alignment - 1);
}

// Resolve the registry-local ordinal without assuming it equals a physical GPU index.
static int host_device_index(ggml_backend_dev_t device) {
    if (!device) return -1;
    auto * reg = ggml_backend_dev_backend_reg(device);
    if (!reg) return -1;
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg) && i <= size_t(std::numeric_limits<int>::max()); ++i) {
        if (ggml_backend_reg_dev_get(reg, i) == device) return int(i);
    }
    return -1;
}

// Missing strict pinning support must not become a pageable allocation.
ggml_backend_buffer_type_t llama_kv_stream_host_buffer_type(ggml_backend_dev_t device) {
    const int index = host_device_index(device);
    if (index < 0) return nullptr;
    auto * reg = ggml_backend_dev_backend_reg(device);
    using factory_t = ggml_backend_buffer_type_t (*)(int);
    auto fn = reinterpret_cast<factory_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_host_buffer_type"));
    if (!fn) return nullptr;
    auto * type = fn(index);
    return type && ggml_backend_buft_is_host(type) && ggml_backend_buft_get_device(type) == device ? type : nullptr;
}

// The returned wrapper owns only its new registration and a retained reference to the caller's buffer.
ggml_backend_buffer_t llama_kv_stream_host_register(ggml_backend_dev_t device, ggml_backend_buffer_t owner) {
    auto * type = llama_kv_stream_host_buffer_type(device);
    if (!type || !owner) return nullptr;
    using register_t = ggml_backend_buffer_t (*)(int, ggml_backend_buffer_t);
    auto fn = reinterpret_cast<register_t>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(device), "ggml_backend_cuda_kv_host_buffer_register"));
    if (!fn) return nullptr;
    auto * buffer = fn(host_device_index(device), owner);
    if (buffer && ggml_backend_buffer_get_type(buffer) != type) {
        ggml_backend_buffer_free(buffer);
        return nullptr;
    }
    return buffer;
}

// Final host ownership is independent of the device arena and its generations.
llama_kv_stream_host::~llama_kv_stream_host() { ggml_backend_buffer_free(storage); }

// Allocate only after checked sizing, and clear only the canonical storage span.
std::shared_ptr<llama_kv_stream_host> llama_kv_stream_host::create(
        const llama_kv_stream_host_config & c, ggml_backend_buffer_type_t type) {
    ggml_kv_stream_layout planes;
    size_t stride, bytes;
    if (!type || !ggml_backend_buft_is_host(type) || !host_plan(c, planes, stride, bytes)) return {};
    try {
        using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
        buffer_ptr buffer(ggml_backend_buft_alloc_buffer(type, bytes + c.shape.alignment - 1), ggml_backend_buffer_free);
        auto host = from_buffer(c, type, buffer.get());
        if (host) std::memset(host->base, 0, host->storage_bytes);
        return host;
    } catch (const std::bad_alloc &) {
        return {};
    }
}

// Imported bytes are preserved; alignment slack is not KV payload or a device reserve.
std::shared_ptr<llama_kv_stream_host> llama_kv_stream_host::from_buffer(
        const llama_kv_stream_host_config & c, ggml_backend_buffer_type_t type, ggml_backend_buffer_t buffer) {
    if (!type || !buffer || ggml_backend_buffer_get_type(buffer) != type || !ggml_backend_buft_is_host(type)) return {};
    try {
        std::shared_ptr<llama_kv_stream_host> host(new llama_kv_stream_host);
        if (!host_plan(c, host->planes, host->layer_stride, host->storage_bytes)) return {};
        void * pointer = ggml_backend_buffer_get_base(buffer);
        const auto address = reinterpret_cast<uintptr_t>(pointer);
        if (!address) return {};
        const size_t remainder = address % c.shape.alignment;
        const size_t offset = remainder ? c.shape.alignment - remainder : 0;
        const size_t capacity = ggml_backend_buffer_get_size(buffer);
        if (offset > capacity || host->storage_bytes > capacity - offset ||
                offset > UINTPTR_MAX - address || host->storage_bytes > UINTPTR_MAX - address - offset) return {};
        host->cfg = c;
        host->base = static_cast<char *>(pointer) + offset;
        host->storage = ggml_backend_buffer_retain(buffer);
        return host;
    } catch (const std::bad_alloc &) {
        return {};
    }
}

// Stable host identity does not change when a device mirror is rebound.
uint64_t llama_kv_stream_host::cache_id() const noexcept { return cfg.cache_id; }
// The validated shape and original context capacity belong to the authoritative storage.
const llama_kv_stream_host_config & llama_kv_stream_host::config() const noexcept { return cfg; }
// Each layer has the same separately packed K and V planes.
const ggml_kv_stream_layout & llama_kv_stream_host::layout() const noexcept { return planes; }
// Include alignment between layers, not just live K/V payload.
size_t llama_kv_stream_host::stride() const noexcept { return layer_stride; }
// Exclude leading/trailing allocation slack from the canonical storage size.
size_t llama_kv_stream_host::bytes() const noexcept { return storage_bytes; }
// Borrowed buffer access requires retaining this storage owner.
ggml_backend_buffer_t llama_kv_stream_host::buffer() const noexcept { return storage; }
// All products and offsets were bounded when the owner was created.
bool llama_kv_stream_host::layer(uint32_t index, llama_kv_stream_host_layer & output) const noexcept {
    if (index >= cfg.layers) return false;
    auto * k = static_cast<char *>(base) + size_t(index)*layer_stride;
    output = {k, k + planes.v_offset};
    return true;
}
