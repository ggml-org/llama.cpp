#include "llama-kv-stream-capture.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-impl.h"
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <unordered_set>

namespace {
using lease_ptr = std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>;

// Check arbitrary strided/quantized ranges before trusting graph metadata or a borrowed address.
static bool tensor_range(const ggml_tensor & tensor, uintptr_t & start, size_t & bytes) {
    if (!tensor.buffer || !tensor.data || tensor.type < 0 || tensor.type >= GGML_TYPE_COUNT) return false;
    const auto block = ggml_blck_size(tensor.type);
    if (block <= 0 || tensor.ne[0] <= 0 || tensor.ne[0]%block || !(bytes = ggml_type_size(tensor.type))) return false;
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (tensor.ne[dim] <= 0 || uint64_t(tensor.ne[dim]) > SIZE_MAX) return false;
        const size_t elements = size_t(tensor.ne[dim])/(dim ? 1 : size_t(block));
        if (tensor.nb[dim] && elements-1 > (SIZE_MAX-bytes)/tensor.nb[dim]) return false;
        bytes += (elements-1)*tensor.nb[dim];
    }
    start = uintptr_t(tensor.data);
    const auto base = uintptr_t(ggml_backend_buffer_get_base(tensor.buffer));
    const size_t capacity = ggml_backend_buffer_get_size(tensor.buffer);
    return base && start >= base && start-base <= capacity && bytes <= capacity-(start-base) && start <= UINTPTR_MAX-bytes;
}

// Names and backend-owned extra data do not affect fixed graph storage/topology.
static bool unchanged(const ggml_tensor & a, const ggml_tensor & b) {
    constexpr size_t begin = offsetof(ggml_tensor,buffer), end = offsetof(ggml_tensor,name);
    constexpr size_t fields = sizeof(a.buffer)+sizeof(a.ne)+sizeof(a.nb)+sizeof(a.op)+sizeof(a.op_params)+sizeof(a.flags)+
        sizeof(a.src)+sizeof(a.view_src)+sizeof(a.view_offs)+sizeof(a.data);
    // Compare the contiguous metadata range only when it has no padding; retain a portable field-wise fallback.
    if constexpr (end-begin == fields)
        return a.type == b.type && !std::memcmp(reinterpret_cast<const char *>(&a)+begin,reinterpret_cast<const char *>(&b)+begin,fields);
    return a.type == b.type && a.buffer == b.buffer && a.data == b.data && a.op == b.op && a.flags == b.flags &&
        a.view_src == b.view_src && a.view_offs == b.view_offs &&
        !std::memcmp(a.ne,b.ne,sizeof(a.ne)) && !std::memcmp(a.nb,b.nb,sizeof(a.nb)) &&
        !std::memcmp(a.src,b.src,sizeof(a.src)) && !std::memcmp(a.op_params,b.op_params,sizeof(a.op_params));
}
static bool metadata_only(const ggml_tensor & tensor) {
    return tensor.op == GGML_OP_NONE || tensor.op == GGML_OP_VIEW || tensor.op == GGML_OP_RESHAPE ||
        tensor.op == GGML_OP_PERMUTE || tensor.op == GGML_OP_TRANSPOSE;
}
static bool same_stamp(const llama_kv_stream_capture_stamp & a, const llama_kv_stream_capture_stamp & b) {
    return a.buffer == b.buffer && a.binding_revision == b.binding_revision && a.residency_revision == b.residency_revision &&
        a.mirror_epoch == b.mirror_epoch && a.padded_tokens == b.padded_tokens;
}
}

struct llama_kv_stream_cuda_executor::implementation {
    ggml_backend_t backend;
    ggml_backend_cuda_graph_is_capturing_t capturing = nullptr;
    llama_memory_execution owner;
    llama_memory_cuda_executor native;
    llama_kv_stream_resident * resident = nullptr;
    llama_kv_stream_capture_stamp stamp;
    ggml_cgraph * graph = nullptr;
    std::vector<ggml_tensor *> nodes, leafs;
    std::vector<std::pair<ggml_tensor *,ggml_tensor>> tensors;
    std::vector<lease_ptr> retained_leases;
    std::vector<ggml_backend_memory_lease_t> bindings;
    std::vector<ggml_backend_buffer_ptr> buffers;
    std::vector<llama_memory_resource_id> resources;

    explicit implementation(ggml_backend_t backend) : backend(backend), native(backend) {
        if (backend) capturing = reinterpret_cast<ggml_backend_cuda_graph_is_capturing_t>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)),"ggml_backend_cuda_graph_is_capturing"));
    }
    bool in_capture() const { return capturing && capturing(backend); }
    // Do not compare arena generations: surviving persistent leases keep valid physical identities.
    bool matches(size_t active) const {
        llama_kv_stream_capture_stamp current;
        if (!resident || !graph || !resident->capture_state(backend,active,current) || !same_stamp(stamp,current) ||
                ggml_graph_n_nodes(graph) != int(nodes.size()) || graph->n_leafs != int(leafs.size())) return false;
        if (!nodes.empty() && (!graph->nodes || std::memcmp(graph->nodes,nodes.data(),nodes.size()*sizeof(ggml_tensor *)))) return false;
        if (!leafs.empty() && (!graph->leafs || std::memcmp(graph->leafs,leafs.data(),leafs.size()*sizeof(ggml_tensor *)))) return false;
        for (const auto & tensor : tensors) if (!unchanged(*tensor.first,tensor.second)) return false;
        return true;
    }
    // Native retirement must precede these releases, including the pin that owns resident tensor metadata.
    void clear() {
        graph = nullptr; resident = nullptr; nodes.clear(); leafs.clear(); tensors.clear();
        bindings.clear(); retained_leases.clear(); buffers.clear(); resources.clear(); owner.reset();
    }
};

llama_kv_stream_cuda_executor::llama_kv_stream_cuda_executor(ggml_backend_t backend) : impl(std::make_unique<implementation>(backend)) {}
llama_kv_stream_cuda_executor::~llama_kv_stream_cuda_executor() {
    const auto status = retire().status;
    GGML_ASSERT(status == llama_memory_executor_status::retired || status == llama_memory_executor_status::unchanged);
}
bool llama_kv_stream_cuda_executor::supported() const noexcept { return impl->native.supported() && impl->capturing; }

// Validate all storage and KV nodes before touching the native cache or taking the caller's pin.
bool llama_kv_stream_cuda_executor::bind(llama_kv_stream_resident & resident, llama_memory_execution & owner,
        ggml_cgraph * graph, const std::vector<ggml_backend_memory_lease_t> & bindings, size_t active) {
    auto & s = *impl;
    llama_kv_stream_capture_stamp stamp;
    if (!supported() || s.graph || s.in_capture() || owner.executable() != &resident || !graph || ggml_graph_n_nodes(graph) <= 0 ||
            !resident.ready(active) || !resident.capture_state(s.backend,active,stamp)) return false;
    try {
        std::vector<lease_ptr> leases;
        std::vector<llama_memory_resource_id> resources;
        bool kv = false;
        for (auto * lease : bindings) {
            ggml_backend_memory_region region;
            if (!ggml_backend_memory_lease_get_region(lease,&region)) return false;
            kv |= ggml_backend_memory_lease_buffer(lease) == stamp.buffer;
            leases.emplace_back(ggml_backend_memory_lease_retain(lease),ggml_backend_memory_lease_free);
            resources.push_back(region.id);
        }
        if (!kv) return false;
        std::vector<ggml_tensor *> nodes, leafs, pending;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) nodes.push_back(ggml_graph_node(graph,i));
        for (int i = 0; i < graph->n_leafs; ++i) leafs.push_back(graph->leafs[i]);
        pending = nodes; pending.insert(pending.end(),leafs.begin(),leafs.end());
        std::unordered_set<ggml_tensor *> seen;
        std::unordered_set<ggml_backend_buffer_t> retained;
        std::vector<std::pair<ggml_tensor *,ggml_tensor>> tensors;
        std::vector<ggml_backend_buffer_ptr> buffers;
        size_t attention = 0;
        for (size_t i = 0; i < pending.size(); ++i) {
            auto * tensor = pending[i];
            if (!tensor) return false;
            if (!seen.insert(tensor).second) continue;
            uintptr_t start; size_t bytes;
            if (!tensor_range(*tensor,start,bytes) || ggml_backend_buffer_is_host(tensor->buffer) ||
                    !ggml_backend_supports_buft(s.backend,ggml_backend_buffer_get_type(tensor->buffer))) return false;
            bool covered = false;
            for (auto * lease : bindings) {
                auto * buffer = ggml_backend_memory_lease_buffer(lease);
                const auto base = uintptr_t(ggml_backend_buffer_get_base(buffer));
                const auto size = ggml_backend_buffer_get_size(buffer);
                covered |= ggml_backend_buffer_get_type(buffer) == ggml_backend_buffer_get_type(tensor->buffer) &&
                    start >= base && start-base <= size && bytes <= size-(start-base);
            }
            const bool weight = !ggml_backend_buffer_is_view(tensor->buffer) &&
                ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
            const auto kv_base = uintptr_t(ggml_backend_buffer_get_base(stamp.buffer));
            const size_t kv_size = ggml_backend_buffer_get_size(stamp.buffer);
            const bool touches_kv = start >= kv_base ? start-kv_base < kv_size : bytes > kv_base-start;
            if ((!covered && !weight) || (!metadata_only(*tensor) && (touches_kv || weight))) return false;
            if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
                if (!resident.capture_attention(tensor,active)) return false;
                ++attention;
            }
            if (retained.insert(tensor->buffer).second) buffers.emplace_back(ggml_backend_buffer_retain(tensor->buffer));
            tensors.emplace_back(tensor,*tensor);
            for (auto * src : tensor->src) if (src) pending.push_back(src);
            if (tensor->view_src) pending.push_back(tensor->view_src);
        }
        if (!attention) return false;
        auto dependencies = bindings;
        if (!s.native.bind(graph,dependencies,stamp.residency_revision)) return false;
        s.owner = std::move(owner); s.resident = &resident; s.stamp = stamp; s.graph = graph;
        s.nodes = std::move(nodes); s.leafs = std::move(leafs); s.tensors = std::move(tensors);
        s.retained_leases = std::move(leases); s.bindings = std::move(dependencies);
        s.buffers = std::move(buffers); s.resources = std::move(resources);
        return true;
    } catch (const std::bad_alloc &) { return false; }
}

// Stale metadata retires the capture; dirty data alone waits for synchronization and can reuse it afterward.
ggml_status llama_kv_stream_cuda_executor::compute_async(size_t active) {
    auto & s = *impl;
    if (!supported() || s.in_capture()) return GGML_STATUS_FAILED;
    if (!s.matches(active)) { (void) retire(); return GGML_STATUS_FAILED; }
    if (!s.resident->ready(active)) return GGML_STATUS_FAILED;
    return s.native.compute_async(s.bindings,s.stamp.residency_revision);
}
bool llama_kv_stream_cuda_executor::drain() { return supported() && !impl->in_capture() && impl->native.drain(); }
void llama_kv_stream_cuda_executor::quiesce() noexcept { impl->native.quiesce(); }
llama_memory_executor_result llama_kv_stream_cuda_executor::retire() {
    if (impl->in_capture()) return {llama_memory_executor_status::drain_failed,{}};
    const auto result = impl->native.retire();
    if (result.status == llama_memory_executor_status::retired || result.status == llama_memory_executor_status::unchanged) impl->clear();
    return result;
}
llama_memory_executor_result llama_kv_stream_cuda_executor::retire_if_affected(const std::vector<llama_memory_resource_id> & resources) {
    for (auto id : resources) if (std::find(impl->resources.begin(),impl->resources.end(),id) != impl->resources.end()) return retire();
    return {};
}
bool llama_kv_stream_cuda_executor::is_captured() const { return impl->native.is_captured(); }
bool llama_kv_stream_cuda_executor::ready(size_t active) const {
    return supported() && !impl->in_capture() && impl->native.ready() && impl->matches(active) && impl->resident->ready(active);
}
