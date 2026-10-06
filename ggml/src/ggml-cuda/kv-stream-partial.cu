#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "kv-stream-partial.cuh"
#include "kv-stream-dispatch.cuh"
#include "kv-stream-attention-dispatch.h"
#include "kv-stream-tile.cuh"
#include "convert.cuh"
#include "fattn.cuh"
#include "../ggml-backend-impl.h"
#include "../ggml-backend-execution.h"
#include "../ggml-kv-stream.h"

#include <cmath>
#include <climits>
#include <cstring>
#include <utility>
#include <vector>

namespace {
using plan_status = ggml_cuda_kv_stream_plan_status;
using kernel_family = ggml_cuda_kv_stream_kernel_family;
using execution_style = ggml_cuda_kv_stream_execution_style;

// Verify serial metadata before calling stock selection, which assumes valid grouped-head dimensions.
static plan_status describe_metadata(const ggml_tensor * op, const ggml_kv_stream_span_plan_view * view,
        ggml_cuda_kv_stream_attention_metadata & metadata) {
    if (!op || op->op != GGML_OP_FLASH_ATTN_EXT || op->type != GGML_TYPE_F32 ||
            !op->src[0] || !op->src[1] || !op->src[2]) return plan_status::invalid_metadata;
    const auto * q = op->src[0], * k = op->src[1], * v = op->src[2];
    if (q->type != GGML_TYPE_F32 || q->ne[0] != k->ne[0] || k->ne[1] != v->ne[1] ||
            k->ne[2] != v->ne[2] || q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
            op->ne[0] != v->ne[0] || op->ne[1] != q->ne[2] || op->ne[2] != q->ne[1] || op->ne[3] != 1)
        return plan_status::invalid_metadata;
    for (const auto * tensor : {q, k, v, op}) for (int i = 0; i < 4; ++i)
        if (tensor->ne[i] <= 0 || tensor->ne[i] > INT32_MAX) return plan_status::invalid_metadata;
    if (view && (view->active_tokens > INT32_MAX || !view->count || !view->spans ||
            view->query_tokens != size_t(q->ne[1]) || view->shape.type_k != k->type || view->shape.type_v != v->type ||
            view->shape.head_dim_k != k->ne[0] || view->shape.head_dim_v != v->ne[0] || view->shape.heads != k->ne[2]))
        return plan_status::invalid_metadata;
    metadata = {int32_t(k->type), int32_t(v->type), k->ne[0], v->ne[0], q->ne[2], k->ne[2], q->ne[1],
        view ? int64_t(view->active_tokens) : k->ne[1], k->ne[1]};
    size_t payload = 0;
    const auto valid = ggml_cuda_kv_stream_attention_metadata_validate(metadata, payload);
    if (valid != plan_status::success) return valid;
    if (op->nb[0] != sizeof(float) || !ggml_is_contiguous(op) || ggml_nbytes(op) != payload) return plan_status::invalid_metadata;
    if (view) {
        size_t cursor = 0;
        for (size_t i = 0; i < view->count; ++i) {
            if (view->spans[i].token_begin != cursor || !view->spans[i].tokens ||
                    cursor > view->active_tokens || view->spans[i].tokens > view->active_tokens-cursor)
                return plan_status::invalid_metadata;
            cursor += view->spans[i].tokens;
        }
        if (cursor != view->active_tokens) return plan_status::invalid_metadata;
    }
    if (const auto * mask = op->src[3]) {
        if (mask->type != GGML_TYPE_F16 || mask->ne[0] < k->ne[1] || mask->ne[1] < q->ne[1] ||
                mask->ne[2] != 1 || mask->ne[3] != 1 || mask->nb[0] != sizeof(half)) return plan_status::invalid_metadata;
    }
    return plan_status::success;
}
// Host-driven partials cannot allocate or synchronize inside a capture.
static bool capture_active(ggml_backend_t backend) {
    if (!backend || !ggml_backend_is_cuda(backend)) return false;
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    const auto stream = ctx.streams[ctx.device][ctx.curr_stream_no];
    if (!stream) return false;
    ggml_cuda_set_device(ctx.device);
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(stream,&status));
    return status != cudaStreamCaptureStatusNone;
}

struct span { uintptr_t begin = 0, end = 0; };

// Check the touched tensor range before interpreting a borrowed device pointer.
static bool tensor_span(ggml_backend_t backend, const ggml_tensor * t, span & out) {
    if (!t || !t->data || !t->buffer || !ggml_backend_supports_buft(backend, ggml_backend_buffer_get_type(t->buffer)) ||
            ggml_backend_buffer_is_host(t->buffer)) return false;
    if (t->type < 0 || t->type >= GGML_TYPE_COUNT || ggml_blck_size(t->type) <= 0 || ggml_type_size(t->type) == 0 ||
            t->ne[0] <= 0 || t->ne[0] % ggml_blck_size(t->type)) return false;
    size_t bytes = ggml_type_size(t->type);
    for (int i = 0; i < 4; ++i) {
        if (t->ne[i] <= 0 || t->ne[i] > INT32_MAX || t->nb[i] > INT32_MAX) return false;
        const size_t elements = size_t(t->ne[i])/(i == 0 ? size_t(ggml_blck_size(t->type)) : 1);
        if (elements-1 > (SIZE_MAX-bytes)/(t->nb[i] ? t->nb[i] : 1)) return false;
        bytes += (elements-1)*t->nb[i];
    }
    const auto base = uintptr_t(ggml_backend_buffer_get_base(t->buffer));
    const auto data = uintptr_t(t->data);
    const size_t capacity = ggml_backend_buffer_get_size(t->buffer);
    if (data < base || data-base > capacity || bytes > capacity-(data-base) || data > UINTPTR_MAX-bytes ||
            data % (t->type == GGML_TYPE_F32 ? 4 : 2)) return false;
    out = {data, data+bytes};
    return true;
}

// Compare touched ranges, not owning buffers: independent tensors may share one allocation.
static bool overlap(span a, span b) { return a.begin < b.end && b.begin < a.end; }

// Query the same SET_ROWS admission used by the backend; storage alone does not imply a usable KV cache.
static ggml_kv_stream_capabilities capabilities(ggml_backend_t backend, int32_t key, int32_t value) {
    ggml_kv_stream_capabilities caps;
    caps.k.type = key; caps.v.type = value;
    if (!backend || !ggml_backend_is_cuda(backend)) return caps;
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    for (auto * side : {&caps.k,&caps.v}) {
        ggml_kv_stream_layout layout;
        if (ggml_kv_stream_layout_make({side->type,side->type,256,256,1,256,128},256,layout).status != ggml_kv_stream_status::success) continue;
        side->storage = true;
        auto type = ggml_type(side->type);
        ggml_tensor source = {}, index = {}, destination = {};
        source.type = GGML_TYPE_F32; index.type = GGML_TYPE_I64; destination.type = type;
        for (int i = 0; i < 4; ++i) {
            source.ne[i] = index.ne[i] = destination.ne[i] = 1;
            source.nb[i] = i ? 256*sizeof(float) : sizeof(float);
            index.nb[i] = sizeof(int64_t);
            destination.nb[i] = i == 0 ? ggml_type_size(type) : (i == 1 ? layout.k_row_bytes : layout.k_bytes);
        }
        source.ne[0] = destination.ne[0] = destination.ne[1] = 256;
        ggml_tensor write = destination;
        write.op = GGML_OP_SET_ROWS; write.src[0] = &source; write.src[1] = &index; write.src[2] = &destination;
        side->online_write = ggml_backend_supports_op(backend,&write);
        side->direct_attention = (side == &caps.k ? ggml_cuda_kv_stream_kernel(type,GGML_TYPE_F16) :
            ggml_cuda_kv_stream_kernel(GGML_TYPE_F16,type)) || ggml_cuda_kv_stream_kernel(type,type);
        side->convert_f16 = type == GGML_TYPE_F16 || ggml_get_to_fp16_cuda(type);
    }
    if (caps.k.storage && caps.v.storage) caps.direct_pair = ggml_cuda_kv_stream_kernel(ggml_type(key),ggml_type(value));
    caps.f16_attention = ggml_cuda_kv_stream_kernel(GGML_TYPE_F16,GGML_TYPE_F16);
    return caps;
}

// Admit only compiled native pairs with head size 256, one sequence, and no sinks/bias/softcap.
// Segmented execution validates physical K/V through its retained plan rather than prototype data pointers.
static bool supports_common(ggml_backend_t backend, const ggml_tensor * op, bool segmented) {
    if (!backend || !ggml_backend_is_cuda(backend) || !op || op->op != GGML_OP_FLASH_ATTN_EXT) return false;
    const auto * q = op->src[0], * k = op->src[1], * v = op->src[2], * mask = op->src[3];
    if (!q || !k || !v || !mask || q->type != GGML_TYPE_F32 ||
            mask->type != GGML_TYPE_F16 || op->type != GGML_TYPE_F32 ||
            mask->ne[2] != 1 || mask->ne[3] != 1 || mask->nb[0] != 2 ||
            k->ne[1] <= 0 || k->ne[1] > INT32_MAX || mask->nb[1] < size_t(k->ne[1])*2 || mask->nb[1]%2) return false;
    const auto caps = capabilities(backend,k->type,v->type);
    if (!caps.direct_pair) return false;
    span ranges[5];
    if (!tensor_span(backend,q,ranges[0]) || !tensor_span(backend,mask,ranges[3]) ||
            !tensor_span(backend,op,ranges[4])) return false;
    if (!segmented && (!tensor_span(backend,k,ranges[1]) || !tensor_span(backend,v,ranges[2]))) return false;
    for (int i : {0,3}) if (overlap(ranges[i],ranges[4])) return false;
    if (!segmented && (overlap(ranges[1],ranges[4]) || overlap(ranges[2],ranges[4]))) return false;
    if (ranges[0].begin%16 || q->nb[1]%16 || q->nb[2]%16 ||
            k->nb[1]%16 || k->nb[2]%16 || v->nb[1]%16 || v->nb[2]%16) return false;
    if (!segmented && (ranges[1].begin%16 || ranges[2].begin%16)) return false;
    if (q->ne[2] > 65535 || q->ne[1] > INT32_MAX/512/q->ne[2]) return false;
    float params[3]; std::memcpy(params, op->op_params, sizeof(params));
    if (!std::isfinite(params[0]) || params[0] <= 0 || params[1] != 0 || params[2] != 0) return false;
    // Sliced masks retain the full row pitch. Validate logical shape using a packed metadata clone.
    auto packed = *mask;
    for (int i = 1; i < 4; ++i) {
        if (size_t(packed.ne[i-1]) > SIZE_MAX/packed.nb[i-1]) return false;
        packed.nb[i] = packed.nb[i-1]*size_t(packed.ne[i-1]);
    }
    auto logical = *op; logical.src[3] = &packed;

    ggml_kv_stream_execution execution;
    return ggml_kv_stream_attention_validate(&logical, {256,256,256,256,128}, caps, size_t(k->ne[1]), execution).status == ggml_kv_stream_status::success;
}

static bool supports(ggml_backend_t backend, const ggml_tensor * op) {
    return supports_common(backend,op,false);
}

static bool supports_spans(ggml_backend_t backend, const ggml_tensor * op) {
    return supports_common(backend,op,true);
}

// Conversion operates on contiguous token-major planes, independently of their logical head/token axes.
static bool supports_conversion(ggml_backend_t backend, const ggml_tensor * source, const ggml_tensor * destination) {
    if (!source || !destination || destination->type != GGML_TYPE_F16 || source->ne[0] != 256 || source->ne[3] != 1) return false;
    const auto caps = capabilities(backend,source->type,source->type);
    if (!caps.k.online_write || !caps.k.convert_f16) return false;
    span src, dst;
    if (!tensor_span(backend,source,src) || !tensor_span(backend,destination,dst) ||
            overlap(src,dst) || src.begin%16 || dst.begin%16) return false;
    for (int i = 0; i < 4; ++i) if (source->ne[i] != destination->ne[i]) return false;
    for (const auto * tensor : {source,destination}) {
        ggml_kv_stream_layout layout;
        if (ggml_kv_stream_layout_make({tensor->type,tensor->type,256,256,tensor->ne[2],256,128},
                size_t(tensor->ne[1]),layout).status != ggml_kv_stream_status::success ||
                tensor->nb[0] != ggml_type_size(tensor->type) || tensor->nb[1] != layout.k_token_bytes ||
                tensor->nb[2] != layout.k_row_bytes) return false;
    }
    // Existing converter launchers use signed element counts in their grid calculations.
    return source->ne[1] <= INT32_MAX/256/source->ne[2];
}

// Reuse the backend's converter without graph allocation, pool scratch, or host round trips.
static bool convert(ggml_backend_t backend, const ggml_tensor * source, ggml_tensor * destination) {
    if (capture_active(backend) || !supports_conversion(backend,source,destination)) return false;
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    const int64_t elements = source->ne[0]*source->ne[1]*source->ne[2];
    if (source->type == GGML_TYPE_F16) {
        CUDA_CHECK(cudaMemcpyAsync(destination->data,source->data,size_t(elements)*sizeof(half),cudaMemcpyDeviceToDevice,ctx.stream()));
    } else {
        ggml_get_to_fp16_cuda(source->type)(source->data,static_cast<half *>(destination->data),elements,ctx.stream());
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    return true;
}

// Workspace is an explicit device allocation with no alias of any participating tensor.
static bool workspace_valid(ggml_backend_t backend, ggml_backend_buffer_t buffer, const ggml_kv_stream_block_layout & layout,
        const ggml_tensor * output, bool inputs) {
    if (!buffer || !backend || !ggml_backend_is_cuda(backend)) return false;
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    if (ggml_backend_buffer_get_type(buffer) != ggml_backend_cuda_device_buffer_type(ctx.device) ||
            ggml_backend_buffer_get_size(buffer) < layout.bytes) return false;
    const auto base = uintptr_t(ggml_backend_buffer_get_base(buffer));
    if (!base || base%128 || base > UINTPTR_MAX-layout.bytes) return false;
    const span scratch{base,base+layout.bytes};
    span touched;
    if (!tensor_span(backend, output, touched) || overlap(scratch,touched)) return false;
    if (inputs) for (int i = 0; i < 4; ++i)
        if (!tensor_span(backend, output->src[i], touched) || overlap(scratch,touched)) return false;
    return true;
}

// Export two native vector splits. A shifted maximum remains a valid common (m,L,U) coordinate.
static bool partial(ggml_backend_t backend, const ggml_tensor * op, ggml_backend_buffer_t buffer, bool second) {
    if (capture_active(backend) || !supports(backend, op)) return false;
    ggml_kv_stream_block_layout layout;
    if (ggml_kv_stream_block_layout_make(size_t(op->ne[1])*size_t(op->ne[2]), 256, layout).status != ggml_kv_stream_partial_status::success ||
            !workspace_valid(backend,buffer,layout,op,true)) return false;
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    auto * scratch = static_cast<char *>(ggml_backend_buffer_get_base(buffer)) + (second ? layout.second_offset : 0);
    const auto * q = op->src[0], * k = op->src[1], * v = op->src[2], * m = op->src[3];
    float scale; std::memcpy(&scale, op->op_params, sizeof(scale));
    for (size_t first = 0; first < size_t(q->ne[1]);) {
        ggml_kv_stream_query_tile tile;
        GGML_ASSERT(ggml_kv_stream_query_tile_make(size_t(q->ne[1]),size_t(q->ne[2]),first,tile));
        const ggml_cuda_kernel_launch_params launch({unsigned(tile.queries),2,unsigned(q->ne[2])}, {32,4,1}, 0, ctx.stream());
        ggml_cuda_kernel_launch(ggml_cuda_kv_stream_kernel(k->type,v->type), launch,
            (const char *)q->data+first*q->nb[1], (const char *)k->data, (const char *)v->data, (const char *)m->data+first*m->nb[1],
            (const char *)nullptr, (const int *)nullptr, (float *)scratch+tile.first_row*2*256,
            (float2 *)(scratch+layout.partial.meta_offset)+tile.first_row*2,
            scale, 0.0f, 1.0f, 1.0f, uint32_t(1), 0.0f,
            int32_t(q->ne[0]), init_fastdiv_values(tile.queries), int32_t(q->ne[2]), int32_t(1),
            int32_t(q->nb[1]), int32_t(q->nb[2]), int32_t(q->nb[3]),
            int32_t(k->ne[0]), int32_t(k->ne[1]), int32_t(k->ne[2]), int32_t(1),
            int32_t(k->nb[1]), int32_t(k->nb[2]), int64_t(k->nb[3]),
            int32_t(v->nb[1]), int32_t(v->nb[2]), int64_t(v->nb[3]),
            int32_t(m->ne[1]), int32_t(1), int32_t(1), int32_t(m->nb[1]), int32_t(m->nb[2]), int64_t(m->nb[3]));
        CUDA_CHECK(cudaGetLastError());
        first += tile.queries;
    }
    // One completion fence covers every reader of this K/V span, including a partial final tile.
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    return true;
}

// Use ordinary attention only when the graph allocator has reserved its backend-specific output extras.
static bool direct(ggml_backend_t backend, const ggml_tensor * op) {
    if (capture_active(backend) || !op || op->op != GGML_OP_FLASH_ATTN_EXT || !ggml_backend_supports_op(backend,op)) return false;
    ggml_cuda_kv_stream_attention_plan description;
    if (ggml_cuda_kv_stream_attention_plan_query(backend, op, nullptr, execution_style::native, description) != plan_status::success ||
            (description.requirements.output_extra_bytes && ggml_backend_execution_has_external_workspace(op))) return false;
    span output;
    if (!tensor_span(backend,op,output)) return false;
    const auto base = uintptr_t(ggml_backend_buffer_get_base(op->buffer)), address = uintptr_t(op->data);
    const size_t capacity = ggml_backend_buffer_get_size(op->buffer);
    const size_t required = description.output_allocation_bytes;
    if (address < base || address-base > capacity || required > capacity-(address-base) || address > UINTPTR_MAX-required) return false;
    output.end = address+required;
    for (int i = 0; i < 4; ++i) { span input; if (!tensor_span(backend,op->src[i],input) || overlap(input,output)) return false; }
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_flash_attn_ext_selected(ctx,const_cast<ggml_tensor *>(op),description.family);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream())); return true;
}

// Match native TG1/TG2 vector ownership before any resident or ring span is consumed.
static bool resume_plan(
        ggml_backend_t backend, int32_t key, int32_t value,
        uint32_t heads, uint32_t kv_heads, uint32_t queries,
        size_t tokens, ggml_kv_stream_resume_plan & output) {
    if (!backend || !ggml_backend_is_cuda(backend) || capture_active(backend) ||
            key < 0 || key >= GGML_TYPE_COUNT || value < 0 || value >= GGML_TYPE_COUNT ||
            !heads || !kv_heads || heads%kv_heads || heads > 65535 ||
            !queries || queries > 4 || !tokens || tokens > INT32_MAX ||
            !ggml_cuda_kv_stream_kernel(ggml_type(key),ggml_type(value))) return false;
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    const auto & device = ggml_cuda_info().devices[ctx.device];
    const int compiled_arch = ggml_cuda_highest_compiled_arch(device.cc);
    if (compiled_arch < 0) return false;
    // The wrapper admits the baseline; stock still chooses the executable family.
    ggml_tensor q{}, k{}, v{}, mask{}, op{};
    q.type = GGML_TYPE_F32; k.type = ggml_type(key); v.type = ggml_type(value); mask.type = GGML_TYPE_F16;
    const int64_t padded = int64_t((tokens+255)/256*256);
    q.ne[0] = k.ne[0] = v.ne[0] = 256; q.ne[1] = queries; q.ne[2] = heads;
    k.ne[1] = v.ne[1] = padded; k.ne[2] = v.ne[2] = kv_heads; mask.ne[0] = padded; mask.ne[1] = queries;
    for (auto * tensor : {&q, &k, &v, &mask}) {
        tensor->ne[3] = 1;
        if (tensor == &mask) tensor->ne[2] = 1;
        tensor->nb[0] = ggml_type_size(tensor->type);
        tensor->nb[1] = ggml_row_size(tensor->type, tensor->ne[0]);
        for (int i = 2; i < 4; ++i) tensor->nb[i] = tensor->nb[i-1]*size_t(tensor->ne[i-1]);
    }
    op.op = GGML_OP_FLASH_ATTN_EXT; op.src[0] = &q; op.src[1] = &k; op.src[2] = &v; op.src[3] = &mask;
    const auto family=ggml_cuda_flash_attn_ext_kernel_family(ctx.device,&op);
    if (family == kernel_family::tile) return ggml_cuda_kv_stream_tile_plan(
        ctx,ggml_type(key),ggml_type(value),heads,kv_heads,queries,tokens,output);
    auto native=ggml_cuda_kv_stream_vector_kernel(ggml_type(key),ggml_type(value),queries);
    auto resumed=ggml_cuda_kv_stream_resume_kernel(ggml_type(key),ggml_type(value),queries);
    if (!native || !resumed ||
            ggml_cuda_kv_stream_attention_select(device.cc,queries,ggml_type(key),ggml_type(value)) == ggml_cuda_kv_stream_attention_path::none ||
            family != kernel_family::vector ||
            (!ggml_is_quantized(ggml_type(key)) && !ggml_is_quantized(ggml_type(value)))) return false;
    int occupancy = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,native,128,0) != cudaSuccess) {
        (void) cudaGetLastError(); return false;
    }
    const int tiles = int((tokens+255)/256), wave = device.nsm*occupancy;
    if (!wave || !occupancy) return false;
    int splits = std::min(occupancy,tiles), best = 0;
    int64_t best_waves = 0;
    for (int candidate = splits; candidate <= tiles; ++candidate) {
        const int64_t blocks = int64_t(heads)*candidate, waves = (blocks+wave-1)/wave;
        const int efficiency = int(100*blocks/(waves*wave));
        if (best >= 95 && waves > best_waves) break;
        if (efficiency > best) {
            best = efficiency;
            best_waves = waves;
            splits = candidate;
        }
    }
    ggml_kv_stream_resume_plan next;
    if (!ggml_kv_stream_resume_layout_make(
            heads,queries,uint32_t(splits),
            ggml_cuda_kv_stream_vector_values_per_thread(ggml_type(value),compiled_arch),next)) return false;
    next.tokens = tokens;
    output = next;
    return true;
}

static bool resume_plan_valid(const ggml_kv_stream_resume_plan & plan, ggml_type value, int compiled_arch) {
    if (plan.resume_resident) return false;
    for (auto item : plan.kernel_config) if (item) return false;
    ggml_kv_stream_resume_plan checked;
    return plan.tokens && ggml_kv_stream_resume_layout_make(
        plan.heads,plan.queries,plan.splits,
        ggml_cuda_kv_stream_vector_values_per_thread(value,compiled_arch),checked) &&
        checked.bytes == plan.bytes && checked.state_bytes == plan.state_bytes &&
        checked.partial_offset == plan.partial_offset && checked.meta_offset == plan.meta_offset &&
        checked.values_per_thread == plan.values_per_thread;
}

// Validate common output, prototype, and workspace storage before any span launch can mutate scratch.
static bool resume_common(
        ggml_backend_t backend, const ggml_tensor * op, ggml_backend_buffer_t workspace,
        const ggml_kv_stream_resume_plan & plan, ggml_kv_resume_kernel_t & kernel,
        uintptr_t & base, span & scratch, bool segmented = false, bool metadata_checked = false) {
    if (capture_active(backend) || (!metadata_checked && !(segmented ? supports_spans(backend,op) : supports(backend,op))) || !workspace) return false;
    const auto * q = op->src[0], * k = op->src[1], * v = op->src[2];
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    const int compiled_arch = ggml_cuda_highest_compiled_arch(ggml_cuda_info().devices[ctx.device].cc);
    if (q->ne[1] != plan.queries || q->ne[2] != plan.heads ||
            compiled_arch < 0 || !resume_plan_valid(plan,v->type,compiled_arch)) return false;
    kernel = ggml_cuda_kv_stream_resume_kernel(k->type,v->type,plan.queries);
    base = uintptr_t(ggml_backend_buffer_get_base(workspace));
    const size_t capacity = ggml_backend_buffer_get_size(workspace);
    if (!kernel || !base || base%128 || base > UINTPTR_MAX-capacity || capacity < plan.bytes ||
            !ggml_backend_supports_buft(backend,ggml_backend_buffer_get_type(workspace)) ||
            ggml_backend_buffer_is_host(workspace)) return false;
    scratch = {base,base+capacity};
    span touched;
    if (!tensor_span(backend,op,touched) || overlap(scratch,touched)) return false;
    for (int i = 0; i < 4; ++i) {
        if (segmented && (i == 1 || i == 2)) continue;
        if (!tensor_span(backend,op->src[i],touched) || overlap(scratch,touched)) return false;
    }
    return true;
}

// Enqueue one physical span while retaining the same per-thread logical softmax state.
static bool resume_launch(
        ggml_backend_t backend, const ggml_tensor * op,
        const ggml_tensor & k, const ggml_tensor & v,
        ggml_backend_buffer_t workspace, const ggml_kv_stream_resume_plan & plan,
        ggml_kv_resume_kernel_t kernel, size_t first, size_t mask_tokens, bool last) {
    const auto * q = op->src[0];
    const auto * m = op->src[3];
    if (k.type != op->src[1]->type || v.type != op->src[2]->type ||
            k.ne[0] != 256 || v.ne[0] != 256 || k.ne[1] <= 0 || v.ne[1] != k.ne[1] ||
            k.ne[2] != op->src[1]->ne[2] || v.ne[2] != k.ne[2] ||
            size_t(k.ne[1]) > plan.tokens-first ||
            last != (first+size_t(k.ne[1]) == plan.tokens) ||
            mask_tokens < plan.tokens || mask_tokens > INT32_MAX) return false;
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    const auto base = uintptr_t(ggml_backend_buffer_get_base(workspace));
    float scale;
    std::memcpy(&scale,op->op_params,sizeof(scale));
    auto * partials = reinterpret_cast<float *>(base+plan.partial_offset);
    auto * meta = reinterpret_cast<float2 *>(base+plan.meta_offset);
    const ggml_cuda_kernel_launch_params launch({1,plan.splits,plan.heads},{32,4,1},0,ctx.stream());
    ggml_cuda_kernel_launch(kernel,launch,
        static_cast<const char *>(q->data),static_cast<const char *>(k.data),static_cast<const char *>(v.data),static_cast<const char *>(m->data),
        (const char *) nullptr,(const int *) nullptr,partials,meta,scale,0.0f,1.0f,1.0f,uint32_t(1),0.0f,
        int32_t(256),init_fastdiv_values(plan.queries),int32_t(q->ne[2]),int32_t(1),
        int32_t(q->nb[1]),int32_t(q->nb[2]),int32_t(q->nb[3]),
        int32_t(256),int32_t(mask_tokens),int32_t(k.ne[2]),int32_t(1),
        int32_t(k.nb[1]),int32_t(k.nb[2]),int64_t(k.nb[3]),
        int32_t(v.nb[1]),int32_t(v.nb[2]),int64_t(v.nb[3]),
        int32_t(m->ne[1]),int32_t(1),int32_t(1),
        int32_t(m->nb[1]),int32_t(m->nb[2]),int64_t(m->nb[3]),
        reinterpret_cast<float *>(base),int(first),int(first+k.ne[1]),first == 0,last);
    CUDA_CHECK(cudaGetLastError());
    if (last) {
        if (plan.splits > 1) {
            const ggml_cuda_kernel_launch_params combine(
                {plan.queries,plan.heads,1},{256,1,1},plan.splits*sizeof(float2),ctx.stream());
            ggml_cuda_kernel_launch(
                flash_attn_combine_results<256>,combine,partials,meta,
                static_cast<float *>(op->data),int(plan.splits));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(
                op->data,partials,ggml_nbytes(op),cudaMemcpyDeviceToDevice,ctx.stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}

// Legacy session entry point: span order belongs to the caller and launches remain asynchronous.
static bool resume(
        ggml_backend_t backend, const ggml_tensor * op, ggml_backend_buffer_t workspace,
        const ggml_kv_stream_resume_plan & plan, size_t tokens, size_t first, bool last) {
    if (!op || plan.tokens != tokens || first >= tokens) return false;
    if (plan.kernel_config[0] == 1) {
        if (capture_active(backend) || !supports(backend,op) || !workspace) return false;
        auto & ctx=*static_cast<ggml_backend_cuda_context *>(backend->context);
        const auto * q=op->src[0], * k=op->src[1], * v=op->src[2];
        if (q->ne[1] != plan.queries || q->ne[2] != plan.heads || k->ne[2] != plan.kernel_config[1] ||
                size_t(k->ne[1]) > tokens-first || last != (first+size_t(k->ne[1]) == tokens)) return false;
        const uintptr_t base=uintptr_t(ggml_backend_buffer_get_base(workspace));
        const size_t capacity=ggml_backend_buffer_get_size(workspace);
        if (!base || base%128 || capacity < plan.bytes || base > UINTPTR_MAX-capacity ||
                ggml_backend_buffer_is_host(workspace) ||
                !ggml_backend_supports_buft(backend,ggml_backend_buffer_get_type(workspace))) return false;
        const span scratch{base,base+capacity};
        span range;
        if (!tensor_span(backend,op,range) || overlap(scratch,range)) return false;
        for (int i=0;i<4;++i)
            if (!tensor_span(backend,op->src[i],range) || overlap(scratch,range)) return false;
        ggml_tensor mask=*op->src[3], logical=*op;
        const uintptr_t mask_base=uintptr_t(ggml_backend_buffer_get_base(mask.buffer)), mask_data=uintptr_t(mask.data);
        if (mask_data < mask_base || mask_data-mask_base < first*sizeof(half)) return false;
        mask.data=reinterpret_cast<void *>(mask_data-first*sizeof(half)); mask.ne[0]=int64_t(tokens);
        if (!tensor_span(backend,&mask,range) || overlap(scratch,range)) return false;
        logical.src[3]=&mask;
        ggml_tensor global_k=*k, global_v=*v, global_op=logical;
        global_k.ne[1]=global_v.ne[1]=int64_t(tokens);
        global_op.src[1]=&global_k; global_op.src[2]=&global_v;
        if (ggml_cuda_flash_attn_ext_kernel_family(ctx.device,&global_op) != kernel_family::tile) return false;
        const auto key_base=uintptr_t(ggml_backend_buffer_get_base(k->buffer));
        const auto value_base=uintptr_t(ggml_backend_buffer_get_base(v->buffer));
        const ggml_kv_stream_span source{k->buffer,v->buffer,first,size_t(k->ne[1]),
            uintptr_t(k->data)-key_base,uintptr_t(v->data)-value_base};
        const ggml_kv_stream_span_plan_view view{{k->type,v->type,256,256,int32_t(k->ne[2]),256,128},
            &source,1,tokens,size_t(q->ne[1])};
        return ggml_cuda_kv_stream_tile_launch(ctx,&logical,view,workspace,plan,first,first+size_t(k->ne[1]));
    }
    ggml_kv_resume_kernel_t kernel;
    uintptr_t base;
    span scratch;
    if (!resume_common(backend,op,workspace,plan,kernel,base,scratch)) return false;
    const auto * k = op->src[1], * v = op->src[2];
    span touched;
    if (!tensor_span(backend,k,touched) || overlap(scratch,touched) ||
            !tensor_span(backend,v,touched) || overlap(scratch,touched)) return false;
    if (plan.queries <= 2 && plan.tokens%256 == 0 && first%256 == 0 && size_t(k->ne[1])%256 == 0) {
        kernel = ggml_cuda_kv_stream_resume_aligned_kernel(k->type,v->type,plan.queries);
        if (!kernel) return false;
    }
    ggml_tensor global_mask = *op->src[3];
    const size_t mask_offset = first*sizeof(ggml_fp16_t);
    const auto mask_base = uintptr_t(ggml_backend_buffer_get_base(global_mask.buffer));
    const auto mask_data = uintptr_t(global_mask.data);
    if (mask_data < mask_base || mask_data-mask_base < mask_offset) return false;
    global_mask.data = reinterpret_cast<void *>(mask_data-mask_offset);
    global_mask.ne[0] = int64_t(plan.tokens);
    if (!tensor_span(backend,&global_mask,touched) || overlap(scratch,touched)) return false;
    ggml_tensor logical = *op;
    logical.src[3] = &global_mask;
    return resume_launch(
        backend,&logical,*k,*v,workspace,plan,kernel,first,plan.tokens,last);
}

// The stock vector body may address a full 128-token work tile inside each physical region.
static bool vector_spans_fast_eligible(const ggml_kv_stream_span_plan_view & view) {
    if (view.query_tokens != 1 || !view.count || view.count > 3) return false;
    for (size_t i = 0; i < view.count; ++i) {
        if (view.spans[i].token_begin%128 || (i+1 < view.count && view.spans[i].tokens%128)) return false;
    }
    return true;
}

static size_t vector_spans_physical_count(const ggml_kv_stream_span_plan_view & view) {
    if (!vector_spans_fast_eligible(view)) return view.count;
    const size_t tail = view.spans[view.count-1].tokens;
    return view.count + (tail >= 128 && tail%128 != 0);
}

// Reserve at most one padded vector tile so an exact final tail can use stock affine addressing.
static bool vector_spans_workspace(
        const ggml_kv_stream_span_plan_view & view,
        const ggml_kv_stream_resume_plan & plan, size_t & bytes) {
    bytes = plan.bytes;
    if (!vector_spans_fast_eligible(view) || view.spans[view.count-1].tokens%128 == 0) return true;
    ggml_kv_stream_layout tail;
    if (ggml_kv_stream_layout_make(view.shape,128,tail).status != ggml_kv_stream_status::success ||
            plan.bytes > SIZE_MAX-127) return false;
    const size_t offset = (plan.bytes+127)/128*128;
    if (tail.bytes > SIZE_MAX-offset) return false;
    bytes = offset+tail.bytes;
    return true;
}

// Plan bounded scratch for either the vector or F16 MMA span family.
struct streamed_plan {
    ggml_cuda_kv_stream_attention_plan description;
    ggml_kv_stream_resume_plan resume;
};

// Sizing and dispatch consume one stock-selected description; no payload or full-context staging is needed.
static plan_status describe_streamed(ggml_backend_t backend, const ggml_tensor * op,
        const ggml_kv_stream_span_plan_view & view, execution_style style, streamed_plan & output, bool metadata_checked = false) {
    ggml_cuda_kv_stream_attention_metadata metadata;
    const auto valid = describe_metadata(op, &view, metadata);
    if (valid != plan_status::success) return valid;
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    if (ggml_cuda_highest_compiled_arch(ggml_cuda_info().devices[ctx.device].cc) < 0) return plan_status::missing_stock_code;
    const auto family = ggml_cuda_flash_attn_ext_kernel_family(ctx.device, op);
    if (family == kernel_family::none) return plan_status::stock_unavailable;
    if (!op->src[3] || op->src[4] || metadata.head_dim_k != 256 || metadata.head_dim_v != 256 ||
            metadata.queries > 4) return plan_status::unsupported_geometry;
    if (op->src[3]->nb[1] < size_t(op->src[1]->ne[1])*sizeof(half) || op->src[3]->nb[1]%sizeof(half))
        return plan_status::unsupported_geometry;
    float params[3]; std::memcpy(params, op->op_params, sizeof(params));
    if (!std::isfinite(params[0]) || params[0] <= 0 || params[1] != 0 || params[2] != 0) return plan_status::unsupported_geometry;
    // Live entry points already checked these same immutable tensors; metadata-only queries must do it here.
    if (!metadata_checked) {
        const auto * q = op->src[0], * k = op->src[1], * v = op->src[2];
        if (q->nb[1]%16 || q->nb[2]%16 || k->nb[1]%16 || k->nb[2]%16 || v->nb[1]%16 || v->nb[2]%16 ||
                q->ne[2] > 65535 || q->ne[1] > INT32_MAX/512/q->ne[2]) return plan_status::unsupported_geometry;
        const auto caps = capabilities(backend, k->type, v->type);
        if (!caps.direct_pair) return plan_status::missing_streamed_implementation;
        auto packed = *op->src[3];
        for (int i = 1; i < 4; ++i) {
            if (size_t(packed.ne[i-1]) > SIZE_MAX/packed.nb[i-1]) return plan_status::overflow;
            packed.nb[i] = packed.nb[i-1]*size_t(packed.ne[i-1]);
        }
        auto logical = *op; logical.src[3] = &packed;
        ggml_kv_stream_execution checked;
        if (ggml_kv_stream_attention_validate(&logical, {256,256,256,256,128}, caps, size_t(k->ne[1]), checked).status !=
                ggml_kv_stream_status::success) return plan_status::unsupported_geometry;
    }
    if (family != kernel_family::tile && metadata.queries <= 2 && ggml_cuda_kv_stream_attention_select(ggml_cuda_info().devices[ctx.device].cc,
            uint32_t(metadata.queries), ggml_type(metadata.type_k), ggml_type(metadata.type_v)) == ggml_cuda_kv_stream_attention_path::none)
        return plan_status::missing_streamed_implementation;
    streamed_plan next;
    ggml_cuda_kv_stream_workspace_requirements requirements{true, 0, 0, 128, 0, true, true};
    if (family == kernel_family::vector) {
        if (!resume_plan(backend, metadata.type_k, metadata.type_v, uint32_t(metadata.query_heads),
                uint32_t(metadata.kv_heads), uint32_t(metadata.queries), view.active_tokens, next.resume))
            return plan_status::missing_streamed_implementation;
        requirements.scratch_bytes = next.resume.bytes;
        if (style == execution_style::spanned && !vector_spans_workspace(view, next.resume, requirements.scratch_bytes))
            return plan_status::overflow;
    } else if (family == kernel_family::tile) {
        if (!ggml_cuda_kv_stream_tile_plan(ctx,ggml_type(metadata.type_k),ggml_type(metadata.type_v),
                uint32_t(metadata.query_heads),uint32_t(metadata.kv_heads),uint32_t(metadata.queries),
                size_t(metadata.padded_tokens),next.resume)) return plan_status::unsupported_launch_resources;
        ggml_cuda_fattn_tile_resume_layout layout;
        if (!ggml_cuda_kv_stream_tile_layout(ctx,ggml_type(metadata.type_k),ggml_type(metadata.type_v),
                next.resume,layout,&requirements.shared_bytes)) return plan_status::unsupported_launch_resources;
        requirements.scratch_bytes=next.resume.bytes;
    } else if (family == kernel_family::mma) {
        if (style == execution_style::resumable ||
                (metadata.queries <= 2 && metadata.type_k == GGML_TYPE_F16 && metadata.type_v == GGML_TYPE_F16))
            return plan_status::missing_streamed_implementation;
        plan_status failure;
        if (!ggml_cuda_flash_attn_ext_mma_f16_spans_requirements(ctx, op, view.count, requirements, &failure)) return failure;
    } else return plan_status::missing_streamed_implementation;
    const ggml_cuda_kv_stream_plan_support support{true, true, true, true, true};
    const auto result = ggml_cuda_kv_stream_attention_plan_make(metadata, family, style, support, requirements, next.description);
    if (result == plan_status::success) output = next;
    return result;
}

static bool spans_workspace(
        ggml_backend_t backend, const ggml_tensor * op,
        ggml_kv_stream_span_plan_t span_plan, size_t & output) {
    ggml_kv_stream_span_plan_view view;
    if (!ggml_kv_stream_span_plan_get_view(span_plan,view) || !supports_spans(backend,op) ||
            !op->src[0] || op->src[0]->ne[1] != int64_t(view.query_tokens)) return false;
    streamed_plan selected;
    if (describe_streamed(backend, op, view, execution_style::spanned, selected, true) != plan_status::success) return false;
    output = selected.description.requirements.scratch_bytes;
    return true;
}

// Launch one stock-vector reduction over retained regions plus an optional padded tail tile.
static bool vector_spans_launch(
        ggml_backend_t backend, const ggml_tensor * op, ggml_backend_buffer_t workspace,
        const ggml_kv_stream_resume_plan & plan,
        const ggml_cuda_kv_span * spans, size_t count) {
    if (!spans || !count || count > 3) return false;
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    const auto base = uintptr_t(ggml_backend_buffer_get_base(workspace));
    const size_t capacity = ggml_backend_buffer_get_size(workspace);
    const auto * q = op->src[0], * k = op->src[1], * v = op->src[2], * mask = op->src[3];
    ggml_cuda_kv_span fixed[4] = {};
    size_t logical_tokens[4] = {};
    std::memcpy(fixed,spans,count*sizeof(*spans));
    for (size_t i = 0; i < count; ++i) logical_tokens[i] = spans[i].tokens;
    size_t fixed_count = count;
    auto & source_tail = fixed[count-1];
    const int tail_live = source_tail.tokens%128;
    if (tail_live) {
        if (plan.bytes > SIZE_MAX-127 || source_tail.k_token_stride <= 0 || source_tail.v_token_stride <= 0) return false;
        const size_t tail_offset = (plan.bytes+127)/128*128;
        const size_t k_bytes = size_t(source_tail.k_token_stride)*128;
        const size_t v_bytes = size_t(source_tail.v_token_stride)*128;
        if (k_bytes > SIZE_MAX-127) return false;
        const size_t v_offset = (k_bytes+127)/128*128;
        if (v_bytes > SIZE_MAX-v_offset || tail_offset > capacity || v_offset+v_bytes > capacity-tail_offset) return false;
        const int prefix = source_tail.tokens-tail_live;
        ggml_cuda_kv_span padded = source_tail;
        padded.first += prefix;
        padded.k += int64_t(prefix)*padded.k_token_stride;
        padded.v += int64_t(prefix)*padded.v_token_stride;
        padded.tokens = 128;
        auto * tail_k = reinterpret_cast<char *>(base+tail_offset);
        auto * tail_v = tail_k+v_offset;
        const size_t live_k = size_t(padded.k_token_stride)*tail_live;
        const size_t live_v = size_t(padded.v_token_stride)*tail_live;
        CUDA_CHECK(cudaMemcpyAsync(tail_k,padded.k,live_k,cudaMemcpyDeviceToDevice,ctx.stream()));
        CUDA_CHECK(cudaMemsetAsync(tail_k+live_k,0,k_bytes-live_k,ctx.stream()));
        CUDA_CHECK(cudaMemcpyAsync(tail_v,padded.v,live_v,cudaMemcpyDeviceToDevice,ctx.stream()));
        CUDA_CHECK(cudaMemsetAsync(tail_v+live_v,0,v_bytes-live_v,ctx.stream()));
        padded.k = tail_k;
        padded.v = tail_v;
        if (prefix) {
            source_tail.tokens = prefix;
            logical_tokens[count-1] = prefix;
            fixed[fixed_count] = padded;
            logical_tokens[fixed_count++] = tail_live;
        } else {
            source_tail = padded;
            logical_tokens[count-1] = tail_live;
        }
    }
    if (fixed_count > 4 || plan.splits < fixed_count) return false;
    auto kernel = ggml_cuda_kv_stream_vector_span_kernel(
        k->type,v->type,uint32_t(q->ne[1]),uint32_t(fixed_count));
    if (!kernel) return false;
    auto * device_spans = reinterpret_cast<ggml_cuda_kv_span *>(base);
    CUDA_CHECK(cudaMemcpyAsync(
        device_spans,fixed,fixed_count*sizeof(*fixed),cudaMemcpyHostToDevice,ctx.stream()));
    auto * partials = reinterpret_cast<float *>(base+plan.partial_offset);
    auto * meta = reinterpret_cast<float2 *>(base+plan.meta_offset);
    float scale;
    std::memcpy(&scale,op->op_params,sizeof(scale));
    int span_splits[4] = {};
    int remaining_splits = int(plan.splits);
    size_t remaining_tokens = plan.tokens;
    for (size_t i = 0; i < fixed_count; ++i) {
        const int remaining_spans = int(fixed_count-i);
        int selected = remaining_spans == 1 ? remaining_splits :
            std::max(1,int((int64_t(remaining_splits)*logical_tokens[i]+remaining_tokens/2)/remaining_tokens));
        selected = std::min(selected,remaining_splits-(remaining_spans-1));
        span_splits[i] = selected;
        remaining_splits -= selected;
        remaining_tokens -= logical_tokens[i];
    }
    if (remaining_splits) return false;
    const ggml_cuda_kernel_launch_params launch(
        {1,plan.splits,plan.heads},{32,4,1},0,ctx.stream());
    ggml_cuda_kernel_launch(kernel,launch,
        static_cast<const char *>(q->data),fixed[0].k,fixed[0].v,static_cast<const char *>(mask->data),
        (const char *) nullptr,(const int *) nullptr,
        plan.splits > 1 ? partials : static_cast<float *>(op->data),meta,
        scale,0.0f,1.0f,1.0f,uint32_t(4),0.0f,
        int32_t(q->ne[0]),init_fastdiv_values(q->ne[1]),int32_t(q->ne[2]),int32_t(q->ne[3]),
        int32_t(q->nb[1]),int32_t(q->nb[2]),int32_t(q->nb[3]),
        int32_t(k->ne[0]),int32_t(k->ne[1]),int32_t(k->ne[2]),int32_t(k->ne[3]),
        int32_t(k->nb[1]),int32_t(k->nb[2]),int64_t(k->nb[3]),
        int32_t(v->nb[1]),int32_t(v->nb[2]),int64_t(v->nb[3]),
        int32_t(mask->ne[1]),int32_t(mask->ne[2]),int32_t(mask->ne[3]),
        int32_t(mask->nb[1]),int32_t(mask->nb[2]),int64_t(mask->nb[3]),
        device_spans,int(fixed_count),span_splits[0],span_splits[1],span_splits[2],span_splits[3]);
    CUDA_CHECK(cudaGetLastError());
    if (plan.splits > 1) {
        const ggml_cuda_kernel_launch_params combine(
            {plan.queries,plan.heads,1},{256,1,1},plan.splits*sizeof(float2),ctx.stream());
        ggml_cuda_kernel_launch(
            flash_attn_combine_results<256>,combine,partials,meta,
            static_cast<float *>(op->data),int(plan.splits));
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}

// Exercise the same row-local dequantizer used by span-aware MMA without allocating conversion planes.
static bool convert_mma_rows(
        ggml_backend_t backend, const ggml_tensor * source, ggml_tensor * destination) {
    if (!backend || !ggml_backend_is_cuda(backend) || !source || !destination ||
            (source->type != GGML_TYPE_Q8_0 && source->type != GGML_TYPE_Q4_0) ||
            destination->type != GGML_TYPE_F16 || source->ne[0] != 256 ||
            source->ne[1] <= 0 || source->ne[1] > INT_MAX || source->ne[2] != 1 || source->ne[3] != 1 ||
            destination->ne[0] != source->ne[0] || destination->ne[1] != source->ne[1] ||
            destination->ne[2] != 1 || destination->ne[3] != 1) return false;
    span src, dst;
    if (!tensor_span(backend,source,src) || !tensor_span(backend,destination,dst) || overlap(src,dst) ||
            source->nb[0] != ggml_type_size(source->type) || destination->nb[0] != sizeof(half) ||
            source->nb[1] > INT64_MAX || destination->nb[1] > INT64_MAX) return false;
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    return ggml_cuda_flash_attn_ext_mma_convert_rows(
        ctx,source->type,static_cast<const char *>(source->data),static_cast<half *>(destination->data),
        int(source->ne[1]),int64_t(source->nb[1]),int64_t(destination->nb[1]));
}

static bool mma_workspace(ggml_backend_t backend, int32_t key, int32_t value,
        uint32_t heads, uint32_t kv_heads, size_t tokens, size_t spans, size_t & bytes) {
    if (!backend || !ggml_backend_is_cuda(backend) || capture_active(backend) ||
            key < 0 || key >= GGML_TYPE_COUNT || value < 0 || value >= GGML_TYPE_COUNT ||
            !heads || heads > 65535 || !kv_heads || kv_heads > heads || heads%kv_heads ||
            !tokens || tokens > INT32_MAX ||
            !spans || spans > 3) return false;
    ggml_tensor q={},k={},v={},mask={},op={};
    q.type=GGML_TYPE_F32;
    q.ne[0]=256; q.ne[1]=4; q.ne[2]=heads; q.ne[3]=1;
    k.type=ggml_type(key); v.type=ggml_type(value);
    for (auto * tensor : {&k,&v}) {
        tensor->ne[0]=256; tensor->ne[1]=int64_t(tokens);
        tensor->ne[2]=kv_heads; tensor->ne[3]=1;
    }
    mask.type=GGML_TYPE_F16;
    mask.ne[0]=int64_t(tokens); mask.ne[1]=4;
    mask.ne[2]=mask.ne[3]=1;
    op.op=GGML_OP_FLASH_ATTN_EXT; op.type=GGML_TYPE_F32;
    op.src[0]=&q; op.src[1]=&k; op.src[2]=&v; op.src[3]=&mask;
    const float scale=1.0f/16;
    std::memcpy(op.op_params,&scale,sizeof(scale));
    auto & ctx=*static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    size_t required=ggml_cuda_flash_attn_ext_mma_f16_spans_workspace(ctx,&op,spans);
    if (!required) return false;
    if (ggml_cuda_kv_stream_attention_select(ggml_cuda_info().devices[ctx.device].cc,
            2,ggml_type(key),ggml_type(value)) == ggml_cuda_kv_stream_attention_path::mma) {
        q.ne[1]=mask.ne[1]=2;
        const size_t tg2=ggml_cuda_flash_attn_ext_mma_f16_spans_workspace(ctx,&op,spans);
        if (!tg2) return false;
        required=std::max(required,tg2);
    }
    bytes=required;
    return true;
}

// Stage-7 entry point: validate every physical range before the first asynchronous launch.
static bool spans(
        ggml_backend_t backend, const ggml_tensor * op,
        ggml_kv_stream_span_plan_t span_plan, ggml_backend_buffer_t workspace) {
    ggml_kv_stream_span_plan_view view;
    if (!ggml_kv_stream_span_plan_get_view(span_plan,view) || !op ||
            view.query_tokens > UINT32_MAX || view.active_tokens > INT32_MAX) return false;
    const auto * q = op->src[0], * prototype_k = op->src[1], * prototype_v = op->src[2], * mask = op->src[3];
    if (!q || !prototype_k || !prototype_v || !mask ||
            q->ne[1] != int64_t(view.query_tokens) ||
            q->ne[0] != view.shape.head_dim_k ||
            prototype_k->type != view.shape.type_k || prototype_v->type != view.shape.type_v ||
            prototype_k->ne[0] != view.shape.head_dim_k ||
            prototype_v->ne[0] != view.shape.head_dim_v ||
            prototype_k->ne[2] != view.shape.heads || prototype_v->ne[2] != view.shape.heads ||
            mask->ne[0] < int64_t(view.active_tokens)) return false;

    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    streamed_plan selected;
    if (!supports_spans(backend, op) ||
            describe_streamed(backend, op, view, execution_style::spanned, selected, true) != plan_status::success) return false;
    if (selected.description.family == kernel_family::tile) {
        const size_t required=selected.resume.bytes;
        if (!workspace || ggml_backend_buffer_get_size(workspace) < required ||
                ggml_backend_buffer_is_host(workspace) ||
                !ggml_backend_supports_buft(backend,ggml_backend_buffer_get_type(workspace))) return false;
        const uintptr_t base=uintptr_t(ggml_backend_buffer_get_base(workspace));
        const size_t capacity=ggml_backend_buffer_get_size(workspace);
        if (!base || base%128 || base > UINTPTR_MAX-capacity) return false;
        const span scratch{base,base+capacity};
        span public_ranges[3];
        for (int i=0;i<3;++i) {
            const auto * tensor=i == 0 ? q : i == 1 ? mask : op;
            if (!tensor_span(backend,tensor,public_ranges[i]) || overlap(scratch,public_ranges[i])) return false;
        }
        ggml_cuda_fattn_tile_resume_layout layout;
        ggml_cuda_fattn_tile_resume_descriptor descriptor;
        if (!ggml_cuda_kv_stream_tile_layout(ctx,prototype_k->type,prototype_v->type,selected.resume,layout) ||
                !ggml_cuda_fattn_tile_resume_wave_make(view,layout,workspace,0,selected.resume.tokens,descriptor,true)) return false;
        for (size_t i=0;i<view.count;++i) {
            const auto & source=view.spans[i];
            if (ggml_backend_buffer_is_host(source.k_buffer) || ggml_backend_buffer_is_host(source.v_buffer) ||
                    !ggml_backend_supports_buft(backend,ggml_backend_buffer_get_type(source.k_buffer)) ||
                    !ggml_backend_supports_buft(backend,ggml_backend_buffer_get_type(source.v_buffer))) return false;
            const auto & physical=descriptor.table.spans[i];
            for (auto touched : {span{uintptr_t(physical.k),uintptr_t(physical.k)+size_t(physical.tokens)*size_t(physical.k_token_stride)},
                    span{uintptr_t(physical.v),uintptr_t(physical.v)+size_t(physical.tokens)*size_t(physical.v_token_stride)}}) {
                if (overlap(scratch,touched)) return false;
                for (auto other : public_ranges) if (overlap(other,touched)) return false;
            }
        }
        return ggml_cuda_kv_stream_tile_launch(ctx,op,view,workspace,selected.resume,0,selected.resume.tokens);
    }
    if (selected.description.family == kernel_family::mma) {
        const size_t required = selected.description.requirements.scratch_bytes;
        if (!workspace ||
                ggml_backend_buffer_get_size(workspace) < required ||
                ggml_backend_buffer_is_host(workspace) ||
                !ggml_backend_supports_buft(backend,ggml_backend_buffer_get_type(workspace))) return false;
        const auto scratch_base = uintptr_t(ggml_backend_buffer_get_base(workspace));
        if (!scratch_base || scratch_base%128 || scratch_base > UINTPTR_MAX-required) return false;
        const span scratch{scratch_base,scratch_base+required};
        span public_ranges[3];
        for (int i = 0; i < 3; ++i) {
            const ggml_tensor * tensor = i == 0 ? q : (i == 1 ? mask : op);
            if (!tensor_span(backend,tensor,public_ranges[i]) || overlap(scratch,public_ranges[i])) return false;
        }
        try {
            std::vector<ggml_cuda_kv_span> descriptors;
            descriptors.reserve(view.count);
            for (size_t i = 0; i < view.count; ++i) {
                const auto & source = view.spans[i];
                ggml_kv_stream_layout layout;
                if (ggml_kv_stream_layout_make(view.shape,source.tokens,layout).status !=
                        ggml_kv_stream_status::success) return false;
                const auto k_base = uintptr_t(ggml_backend_buffer_get_base(source.k_buffer));
                const auto v_base = uintptr_t(ggml_backend_buffer_get_base(source.v_buffer));
                if (!k_base || !v_base || ggml_backend_buffer_is_host(source.k_buffer) ||
                        ggml_backend_buffer_is_host(source.v_buffer) ||
                        !ggml_backend_supports_buft(backend,ggml_backend_buffer_get_type(source.k_buffer)) ||
                        !ggml_backend_supports_buft(backend,ggml_backend_buffer_get_type(source.v_buffer))) return false;
                const span kr{k_base+source.k_offset,k_base+source.k_offset+layout.k_bytes};
                const span vr{v_base+source.v_offset,v_base+source.v_offset+layout.v_bytes};
                for (const auto range : {kr,vr})
                    if (overlap(scratch,range) || overlap(public_ranges[0],range) ||
                            overlap(public_ranges[1],range) || overlap(public_ranges[2],range)) return false;
                descriptors.push_back({
                    reinterpret_cast<const char *>(kr.begin),reinterpret_cast<const char *>(vr.begin),
                    int32_t(source.token_begin),int32_t(source.tokens),
                    int64_t(layout.k_token_bytes),int64_t(layout.v_token_bytes),
                    int64_t(layout.k_row_bytes),int64_t(layout.v_row_bytes)});
            }
            auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
            return ggml_cuda_flash_attn_ext_mma_f16_spans(
                ctx,const_cast<ggml_tensor *>(op),descriptors.data(),descriptors.size(),
                reinterpret_cast<void *>(scratch_base),required);
        } catch (const std::bad_alloc &) {
            return false;
        }
    }

    const auto & plan = selected.resume;
    ggml_kv_resume_kernel_t kernel;
    uintptr_t base;
    span scratch;
    if (!resume_common(backend,op,workspace,plan,kernel,base,scratch,true,true)) return false;

    // Match the aligned native arithmetic; retain guards for every partial region.
    bool aligned = view.active_tokens%256 == 0;
    for (size_t i = 0; i < view.count; ++i) {
        aligned &= view.spans[i].token_begin%256 == 0 && view.spans[i].tokens%256 == 0;
    }
    if (aligned) {
        kernel = ggml_cuda_kv_stream_resume_aligned_kernel(op->src[1]->type,op->src[2]->type,plan.queries);
        if (!kernel) return false;
    }

    try {
        std::vector<std::pair<ggml_tensor,ggml_tensor>> physical;
        std::vector<ggml_cuda_kv_span> descriptors;
        physical.reserve(view.count);
        descriptors.reserve(view.count);
        span public_ranges[3];
        for (int i = 0; i < 3; ++i) {
            const ggml_tensor * tensor = i == 0 ? q : (i == 1 ? mask : op);
            if (!tensor_span(backend,tensor,public_ranges[i])) return false;
        }
        for (size_t i = 0; i < view.count; ++i) {
            const auto & source = view.spans[i];
            ggml_kv_stream_layout layout;
            if (ggml_kv_stream_layout_make(view.shape,source.tokens,layout).status !=
                    ggml_kv_stream_status::success) return false;
            ggml_tensor k = {}, v = {};
            for (auto * tensor : {&k,&v}) {
                tensor->ne[0] = tensor == &k ? view.shape.head_dim_k : view.shape.head_dim_v;
                tensor->ne[1] = int64_t(source.tokens);
                tensor->ne[2] = view.shape.heads;
                tensor->ne[3] = 1;
            }
            k.type = ggml_type(view.shape.type_k);
            k.nb[0] = ggml_type_size(k.type);
            k.nb[1] = layout.k_token_bytes;
            k.nb[2] = layout.k_row_bytes;
            k.nb[3] = layout.k_bytes;
            k.buffer = source.k_buffer;
            k.data = static_cast<uint8_t *>(ggml_backend_buffer_get_base(k.buffer)) + source.k_offset;
            v.type = ggml_type(view.shape.type_v);
            v.nb[0] = ggml_type_size(v.type);
            v.nb[1] = layout.v_token_bytes;
            v.nb[2] = layout.v_row_bytes;
            v.nb[3] = layout.v_bytes;
            v.buffer = source.v_buffer;
            v.data = static_cast<uint8_t *>(ggml_backend_buffer_get_base(v.buffer)) + source.v_offset;
            span range;
            if (!tensor_span(backend,&k,range) || overlap(scratch,range) ||
                    overlap(public_ranges[0],range) || overlap(public_ranges[1],range) ||
                    overlap(public_ranges[2],range) ||
                    !tensor_span(backend,&v,range) || overlap(scratch,range) ||
                    overlap(public_ranges[0],range) || overlap(public_ranges[1],range) ||
                    overlap(public_ranges[2],range)) return false;
            physical.emplace_back(k,v);
            descriptors.push_back({
                static_cast<const char *>(k.data),static_cast<const char *>(v.data),
                int32_t(source.token_begin),int32_t(source.tokens),
                int64_t(layout.k_token_bytes),int64_t(layout.v_token_bytes),
                int64_t(layout.k_row_bytes),int64_t(layout.v_row_bytes)});
        }
        if (vector_spans_fast_eligible(view) && plan.splits >= vector_spans_physical_count(view)) {
            return vector_spans_launch(
                backend,op,workspace,plan,descriptors.data(),descriptors.size());
        }
        if (mask->nb[1]%sizeof(ggml_fp16_t)) return false;
        const size_t mask_stride = mask->nb[1]/sizeof(ggml_fp16_t);
        for (size_t i = 0; i < physical.size(); ++i) {
            if (!resume_launch(
                    backend,op,physical[i].first,physical[i].second,workspace,plan,kernel,
                    view.spans[i].token_begin,mask_stride,i+1 == physical.size())) return false;
        }
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    }
}

// The surrounding attention TU uses fast math; preserve subnormal contract payloads across conversion.
static __device__ __forceinline__ double widen(float value) {
    double result;
    asm("cvt.f64.f32 %0, %1;" : "=d"(result) : "f"(value));
    return result;
}
// Explicit rounding also prevents fast-math publication from flushing a valid FP32 accumulator to zero.
static __device__ __forceinline__ float narrow(double value) {
    float result;
    asm("cvt.rn.f32.f64 %0, %1;" : "=f"(result) : "d"(value));
    return result;
}

// Validate all four contributions and fold or stage normalized values; never write public output here.
__global__ void merge_kernel(float * a, float2 * am, const float * b, const float2 * bm,
        float * values, unsigned * status, bool fold) {
    const size_t row = blockIdx.x;
    const size_t channel = threadIdx.x;
    __shared__ double weights[4], mass[4], total, reference;
    if (channel == 0) {
        const float2 meta[4] = {am[2*row],am[2*row+1],bm[2*row],bm[2*row+1]};
        double maximum = -INFINITY;
        bool valid = true;
        for (int i = 0; i < 4; ++i) {
            const double m = widen(meta[i].x), l = widen(meta[i].y);
            valid &= isfinite(l) && l >= 0 && (isfinite(m) || (l == 0 && m == -INFINITY));
            if (l > 0) maximum = fmax(maximum,m);
            mass[i] = l;
        }
        reference = maximum;
        total = 0;
        for (int i = 0; i < 4; ++i) {
            weights[i] = mass[i] > 0 ? exp(widen(meta[i].x)-maximum) : 0;
            total += mass[i]*weights[i];
        }
        if (!valid || !isfinite(total) || total > FLT_MAX) atomicOr(status,1u);
    }
    __syncthreads();
    const float u[4] = {a[(2*row)*256+channel],a[(2*row+1)*256+channel],b[(2*row)*256+channel],b[(2*row+1)*256+channel]};
    double numerator = 0;
    bool valid = true;
    for (int i = 0; i < 4; ++i) {
        const double term = widen(u[i]);
        valid &= isfinite(term) && (mass[i] != 0 || term == 0);
        numerator += term*weights[i];
    }
    const double value = fold ? numerator : (total > 0 ? numerator/total : 0);
    if (!valid || !isfinite(numerator) || fabs(numerator) > FLT_MAX || !isfinite(value) || fabs(value) > FLT_MAX) atomicOr(status,1u);
    if (fold) {
        a[2*row*256+channel] = narrow(numerator);
        a[(2*row+1)*256+channel] = 0;
        // All lanes must finish reading the old split before its metadata is replaced.
        __syncthreads();
        if (channel == 0) {
            am[2*row] = make_float2(narrow(reference),narrow(total));
            am[2*row+1] = make_float2(-INFINITY,0);
        }
    } else {
        values[row*256+channel] = narrow(value);
    }
}

// Share checked workspace handling between publication, incremental folding, and empty initialization.
static bool combine(ggml_backend_t backend, ggml_tensor * output, ggml_backend_buffer_t buffer, int action) {
    if (capture_active(backend)) return false;
    if (!output || output->type != GGML_TYPE_F32 || output->ne[0] != 256 || output->ne[1] <= 0 ||
            output->ne[2] <= 0 || output->ne[1] > INT32_MAX || output->ne[2] > INT32_MAX/512/output->ne[1] ||
            output->ne[3] != 1 || !ggml_is_contiguous(output)) return false;
    ggml_kv_stream_block_layout layout;
    if (ggml_kv_stream_block_layout_make(size_t(output->ne[1])*size_t(output->ne[2]),256,layout).status != ggml_kv_stream_partial_status::success ||
            !workspace_valid(backend,buffer,layout,output,false)) return false;
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    auto * raw = static_cast<char *>(ggml_backend_buffer_get_base(buffer));
    if (action >= 2) {
        CUDA_CHECK(cudaMemsetAsync(raw+(action == 3 ? layout.second_offset : 0),0,layout.partial.bytes,ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        return true;
    }
    auto * flag = reinterpret_cast<unsigned *>(raw+layout.status_offset);
    CUDA_CHECK(cudaMemsetAsync(flag,0,sizeof(unsigned),ctx.stream()));
    merge_kernel<<<unsigned(layout.partial.rows),256,0,ctx.stream()>>>(
        (float *)raw, (float2 *)(raw+layout.partial.meta_offset),
        (float *)(raw+layout.second_offset), (float2 *)(raw+layout.second_offset+layout.partial.meta_offset),
        (float *)(raw+layout.value_offset), flag, action == 1);
    CUDA_CHECK(cudaGetLastError());
    unsigned status;
    CUDA_CHECK(cudaMemcpyAsync(&status,flag,sizeof(status),cudaMemcpyDeviceToHost,ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    if (status) return false;
    if (action == 1) return true;
    CUDA_CHECK(cudaMemcpyAsync(output->data,raw+layout.value_offset,layout.value_bytes,cudaMemcpyDeviceToDevice,ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    return true;
}
// Only this operation publishes normalized values to the caller's output.
static bool merge(ggml_backend_t backend, ggml_tensor * output, ggml_backend_buffer_t buffer) {
    return combine(backend,output,buffer,0);
}
// Keep an unnormalized accumulator in the first export; no context-sized scratch growth.
static bool fold(ggml_backend_t backend, ggml_tensor * output, ggml_backend_buffer_t buffer) {
    return combine(backend,output,buffer,1);
}
// Zero mass and numerator encode an empty contribution regardless of its finite reference value.
static bool clear(ggml_backend_t backend, ggml_tensor * output, ggml_backend_buffer_t buffer, bool second) {
    return combine(backend,output,buffer,second ? 3 : 2);
}
} // namespace

// Describe owned workspace without inspecting payloads; stock backend-pool/launch details stay explicitly unknown.
ggml_cuda_kv_stream_plan_status ggml_cuda_kv_stream_attention_plan_query(
        ggml_backend_t backend, const ggml_tensor * op, const ggml_kv_stream_span_plan_view * view,
        ggml_cuda_kv_stream_execution_style style, ggml_cuda_kv_stream_attention_plan & output) {
    if (!backend || !ggml_backend_is_cuda(backend)) return plan_status::unsupported_device_features;
    if (style > execution_style::resumable || (style == execution_style::native ? view != nullptr : view == nullptr))
        return plan_status::invalid_metadata;
    if (capture_active(backend)) return plan_status::unsupported_launch_resources;
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    if (style != execution_style::native) {
        streamed_plan selected;
        const auto result = describe_streamed(backend, op, *view, style, selected);
        if (result == plan_status::success) output = selected.description;
        return result;
    }
    ggml_cuda_kv_stream_attention_metadata metadata;
    const auto valid = describe_metadata(op, nullptr, metadata);
    if (valid != plan_status::success) return valid;
    if (ggml_cuda_highest_compiled_arch(ggml_cuda_info().devices[ctx.device].cc) < 0) return plan_status::missing_stock_code;
    const auto family = ggml_cuda_flash_attn_ext_kernel_family(ctx.device, op);
    if (family == kernel_family::none) return plan_status::stock_unavailable;
    const size_t payload = ggml_nbytes(op);
    const size_t allocated = ggml_cuda_flash_attn_ext_selected_alloc_size(family, op);
    if (allocated < payload) return plan_status::overflow;
    const ggml_cuda_kv_stream_plan_support support{true, false, true, true, true};
    const ggml_cuda_kv_stream_workspace_requirements requirements{true, allocated-payload, 0, 128, 0, false, false};
    return ggml_cuda_kv_stream_attention_plan_make(metadata, family, style, support, requirements, output);
}

// Keep CUDA details behind the backend-neutral registry contract.
const ggml_kv_stream_partial_ops * ggml_cuda_kv_stream_partial_ops() {
    static_assert(sizeof(float2) == sizeof(ggml_kv_stream_partial_meta), "partial metadata ABI");
    static const ggml_kv_stream_partial_ops ops{10,supports,partial,merge,fold,clear,capabilities,supports_conversion,convert,direct,resume_plan,resume,spans,spans_workspace,convert_mma_rows,mma_workspace,
        [](ggml_backend_t backend,int32_t key,int32_t value,uint32_t heads,uint32_t kv_heads,uint32_t max_queries,size_t tokens,size_t & bytes) {
            if (!max_queries || max_queries > 4) return false;
            size_t required=0;
            for (uint32_t queries=1;queries<=max_queries;++queries) {
                ggml_kv_stream_resume_plan plan;
                if (resume_plan(backend,key,value,heads,kv_heads,queries,tokens,plan)) {
                    required=std::max(required,plan.bytes);
                    // MTP can end inside a vector tile even when the context capacity is page-aligned.
                    if (queries == 1 && !plan.kernel_config[0]) {
                        ggml_kv_stream_layout tail;
                        if (ggml_kv_stream_layout_make({key,value,256,256,int32_t(kv_heads),256,128},128,tail).status !=
                                ggml_kv_stream_status::success || plan.bytes > SIZE_MAX-127) return false;
                        const size_t offset=(plan.bytes+127)/128*128;
                        if (tail.bytes > SIZE_MAX-offset) return false;
                        required=std::max(required,offset+tail.bytes);
                    }
                }
            }
            if (max_queries >= 2) {
                size_t mma=0;
                if (mma_workspace(backend,key,value,heads,kv_heads,tokens,3,mma)) required=std::max(required,mma);
            }
            if (!required) return false;
            bytes=required; return true;
        }};
#ifdef GGML_CUDA_NO_FA
    GGML_UNUSED(ops);
    return nullptr;
#else
    return &ops;
#endif
}
#endif
