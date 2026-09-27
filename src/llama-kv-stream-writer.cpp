#include "llama-kv-stream-writer.h"
#include "llama-memory-executor-cuda.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <limits>

struct write_graph {
    ggml_backend_buffer_ptr source_owner;
    ggml_context_ptr context;
    const void * data = nullptr;
    ggml_type type;
    size_t width = 0, rows = 0;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    // Captures retire before metadata and source-buffer ownership disappear.
    std::unique_ptr<llama_memory_cuda_executor> execution;
};

struct llama_kv_stream_writer::implementation {
    ggml_backend_t backend;
    ggml_backend_buffer_t buffer;
    ggml_kv_stream_shape shape;
    bool cpu;
    size_t maximum = 0, capacity = 0, scratch_bytes = 0;
    ggml_context_ptr context;
    ggml_tensor * roots[2] = {};
    ggml_tensor * indices = nullptr;
    std::array<std::vector<std::unique_ptr<write_graph>>,2> plans;
    llama_kv_stream_write_stats last;

    size_t width(bool value) const { return size_t(value ? shape.head_dim_v : shape.head_dim_k)*size_t(shape.heads); }
    ggml_type type(bool value) const { return ggml_type(value ? shape.type_v : shape.type_k); }
    size_t row_bytes(bool value) const { return ggml_row_size(type(value), int64_t(width(value))); }

    // Immutable per-tile metadata remains valid until every queued launch reaches its completion event.
    write_graph * prepare(const ggml_tensor * source, bool value, size_t tile, size_t offset, size_t count) {
        auto * owner=source->view_src ? source->view_src->buffer : source->buffer;
        auto * data=static_cast<char *>(source->data)+offset*source->nb[1];
        auto & cache=plans[value];
        if (cache.size() <= tile) cache.resize(tile+1);
        auto & plan=cache[tile];
        if (plan && plan->source_owner.get() == owner && plan->data == data && plan->type == type(value) &&
                plan->rows == count && plan->width == width(value)) return plan.get();
        plan.reset();
        auto next=std::make_unique<write_graph>();
        next->source_owner.reset(ggml_backend_buffer_retain(owner));
        next->context.reset(ggml_init({65536,nullptr,true}));
        if (!next->context) return nullptr;
        next->data=data; next->type=type(value); next->rows=count; next->width=width(value);
        auto * input=ggml_new_tensor_2d(next->context.get(),GGML_TYPE_F32,int64_t(width(value)),int64_t(count));
        if (ggml_backend_tensor_alloc(owner,input,data) != GGML_STATUS_SUCCESS) return nullptr;
        auto * dst=ggml_view_2d(next->context.get(),roots[value],int64_t(width(value)),int64_t(capacity),row_bytes(value),0);
        auto * idx=ggml_view_1d(next->context.get(),indices,int64_t(count),0);
        if (ggml_backend_view_init(dst) != GGML_STATUS_SUCCESS || ggml_backend_view_init(idx) != GGML_STATUS_SUCCESS) return nullptr;
        next->output=ggml_set_rows(next->context.get(),dst,input,idx);
        if (!ggml_backend_supports_op(backend,next->output) || ggml_backend_view_init(next->output) != GGML_STATUS_SUCCESS) return nullptr;
        next->graph=ggml_new_graph_custom(next->context.get(),32,false);
        ggml_build_forward_expand(next->graph,next->output);
        if (!cpu) {
            next->execution=std::make_unique<llama_memory_cuda_executor>(backend);
            if (!next->execution->bind(next->graph,{},1)) return nullptr;
        }
        plan=std::move(next);
        return plan.get();
    }
};

// The cached executor drains and retires before roots in the shared scratch context are released.
llama_kv_stream_writer::~llama_kv_stream_writer() = default;

// Use actual backend padding costs, then reserve aligned indices after both possible encoded layouts.
std::unique_ptr<llama_kv_stream_writer> llama_kv_stream_writer::create(ggml_backend_t backend, ggml_backend_buffer_t buffer,
        size_t scratch_bytes, const ggml_kv_stream_shape & shape, size_t max_rows) {
    if (!backend || !buffer || !max_rows || max_rows > INT32_MAX || scratch_bytes > ggml_backend_buffer_get_size(buffer)) return {};
    ggml_kv_stream_layout row;
    if (ggml_kv_stream_layout_make(shape, 1, row).status != ggml_kv_stream_status::success) return {};
    const uint64_t max_elements = std::min(uint64_t(INT32_MAX), uint64_t(SIZE_MAX/sizeof(float)));
    const uint64_t dim = uint64_t(std::max(shape.head_dim_k, shape.head_dim_v));
    if (dim > max_elements/max_rows || uint64_t(shape.heads) > max_elements/max_rows/dim) return {};
    try {
        std::unique_ptr<llama_kv_stream_writer> writer(new llama_kv_stream_writer);
        writer->impl = std::make_unique<implementation>();
        auto & s = *writer->impl;
        s.backend = backend; s.buffer = buffer; s.shape = shape; s.maximum = max_rows;
        auto * device = ggml_backend_get_device(backend);
        s.cpu = device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU;
        if (!s.cpu) { llama_memory_cuda_executor probe(backend); if (!probe.supported()) return {}; }
        const size_t max_row = std::max(row.k_token_bytes, row.v_token_bytes);
        if (max_row > SIZE_MAX - sizeof(int64_t)) return {};
        s.capacity = std::min(max_rows, scratch_bytes/(max_row + sizeof(int64_t)));
        size_t index_offset = 0;
        while (s.capacity) {
            size_t encoded = 0;
            for (bool value : {false, true}) {
                ggml_tensor tensor = {};
                tensor.type = s.type(value);
                tensor.ne[0] = int64_t(s.width(value)*s.capacity);
                tensor.ne[1] = tensor.ne[2] = tensor.ne[3] = 1;
                tensor.nb[0] = ggml_type_size(tensor.type);
                tensor.nb[1] = tensor.nb[2] = tensor.nb[3] = s.row_bytes(value)*s.capacity;
                const size_t bytes = ggml_backend_buffer_get_alloc_size(buffer, &tensor);
                if (bytes < tensor.nb[1]) return {};
                encoded = std::max(encoded, bytes);
            }
            if (encoded <= SIZE_MAX - 127) {
                index_offset = (encoded + 127)/128*128;
                if (index_offset <= scratch_bytes && s.capacity <= (scratch_bytes - index_offset)/sizeof(int64_t)) break;
            }
            --s.capacity;
        }
        if (!s.capacity) return {};
        s.scratch_bytes = index_offset + s.capacity*sizeof(int64_t);
        s.context.reset(ggml_init({16384, nullptr, true}));
        if (!s.context) return {};
        auto * base = static_cast<char *>(ggml_backend_buffer_get_base(buffer));
        for (bool value : {false, true}) {
            s.roots[value] = ggml_new_tensor_1d(s.context.get(), s.type(value), int64_t(s.width(value)*s.capacity));
            if (ggml_backend_tensor_alloc(buffer, s.roots[value], base) != GGML_STATUS_SUCCESS) return {};
        }
        s.indices = ggml_new_tensor_1d(s.context.get(), GGML_TYPE_I64, int64_t(s.capacity));
        if (ggml_backend_tensor_alloc(buffer, s.indices, base + index_offset) != GGML_STATUS_SUCCESS) return {};
        std::vector<int64_t> indices(s.capacity);
        for (size_t i = 0; i < s.capacity; ++i) indices[i] = int64_t(i);
        ggml_backend_tensor_set(s.indices, indices.data(), 0, indices.size()*sizeof(int64_t));
        return writer;
    } catch (const std::bad_alloc &) { return {}; }
}

// Admit dense F32 source rows; ordered backend work may still be pending when submission begins.
bool llama_kv_stream_writer::matches_shape(const ggml_kv_stream_shape & shape) const noexcept {
    if (!impl) return false;
    const auto & actual = impl->shape;
    return actual.type_k == shape.type_k && actual.type_v == shape.type_v &&
        actual.head_dim_k == shape.head_dim_k && actual.head_dim_v == shape.head_dim_v &&
        actual.heads == shape.heads && actual.page_tokens == shape.page_tokens &&
        actual.alignment == shape.alignment;
}


bool llama_kv_stream_writer::accepts(const ggml_tensor * source, bool value) const {
    const auto & s = *impl;
    if (!source || source->type != GGML_TYPE_F32 || !source->data || source->ne[0] != int64_t(s.width(value)) ||
            source->ne[1] <= 0 || uint64_t(source->ne[1]) > s.maximum || source->ne[2] != 1 || source->ne[3] != 1 ||
            source->nb[0] != sizeof(float) || source->nb[1] != s.width(value)*sizeof(float)) return false;
    auto * buffer = source->view_src ? source->view_src->buffer : source->buffer;
    if (!buffer || !ggml_backend_supports_buft(s.backend, ggml_backend_buffer_get_type(buffer))) return false;
    const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
    const auto data = reinterpret_cast<uintptr_t>(source->data);
    const size_t size = ggml_backend_buffer_get_size(buffer);
    const size_t bytes = size_t(source->ne[1])*source->nb[1];
    const auto scratch = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(s.buffer));
    const bool overlaps = data >= scratch ? data-scratch < s.scratch_bytes : bytes > scratch-data;
    return !overlaps && data >= base && data - base <= size && bytes <= size - size_t(data - base);
}

bool llama_kv_stream_writer::generate_async(const ggml_tensor * source, bool value, void * host,
        const std::function<bool(const ggml_tensor *, size_t, size_t)> & publish,
        llama_kv_stream_writer_completion & completion) {
    auto & s=*impl;
    s.last={}; s.last.device_scratch_bytes=s.scratch_bytes; s.last.tile_rows=s.capacity;
    if (!host || !accepts(source,value) || completion.device || completion.host) return false;
    const size_t rows=size_t(source->ne[1]),stride=s.row_bytes(value);
    s.last.host_payload_bytes=rows*stride;
    const auto fail=[&] {
        ggml_backend_synchronize(s.backend);
        release_completed();
        return false;
    };
    try {
        size_t tile=0;
        for (size_t first=0; first < rows; first += std::min(s.capacity,rows-first)) {
            const size_t count=std::min(s.capacity,rows-first);
            auto * plan=s.prepare(source,value,tile++,first,count);
            if (!plan) return fail();
            ++s.last.graph_submissions;
            if (s.cpu) {
                if (ggml_backend_graph_compute(s.backend,plan->graph) != GGML_STATUS_SUCCESS) return fail();
            } else if (plan->execution->compute_async({},1) != GGML_STATUS_SUCCESS) return fail();
            ggml_backend_tensor_get_async(s.backend,plan->output,static_cast<char *>(host)+first*stride,0,count*stride);
            const bool published=!publish || publish(s.roots[value],first,count);
            s.last.d2h_bytes += count*stride; ++s.last.d2h_calls;
            if (!published) return fail();
            if (publish) { s.last.d2d_bytes += count*stride; ++s.last.d2d_calls; }
        }
        llama_kv_stream_writer_completion next;
        if (s.cpu) {
            next.device=llama_memory_completion::completed(s.backend);
            next.host=llama_memory_completion::completed(s.backend);
        } else {
            next.device=llama_memory_completion::create(s.backend);
            next.host=llama_memory_completion::create(s.backend);
            if (!next.device || !next.host || !next.device->record() || !next.host->record()) return fail();
        }
        if (!next.device || !next.host) return fail();
        completion=std::move(next);
        return true;
    } catch (...) {
        ggml_backend_synchronize(s.backend);
        release_completed();
        throw;
    }
}

bool llama_kv_stream_writer::generate(const ggml_tensor * source, bool value, void * host,
        const std::function<bool(const ggml_tensor *, size_t, size_t)> & publish) {
    llama_kv_stream_writer_completion completion;
    if (!generate_async(source,value,host,publish,completion)) return false;
    const bool device=completion.device->synchronize();
    const bool host_ready=completion.host->synchronize();
    release_completed();
    return device && host_ready;
}

void llama_kv_stream_writer::release_completed() noexcept {
    if (!impl || impl->cpu) return;
    for (auto & cache : impl->plans) for (auto & plan : cache)
        if (plan && plan->execution) plan->execution->release_completed();
}

llama_kv_stream_write_stats llama_kv_stream_writer::stats() const { return impl->last; }
