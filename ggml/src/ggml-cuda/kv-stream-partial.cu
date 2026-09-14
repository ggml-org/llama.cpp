#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "kv-stream-partial.cuh"
#include "kv-stream-dispatch.cuh"
#include "convert.cuh"
#include "fattn.cuh"
#include "../ggml-backend-impl.h"
#include "../ggml-kv-stream.h"

#include <cmath>
#include <climits>
#include <cstring>

namespace {
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
static bool supports(ggml_backend_t backend, const ggml_tensor * op) {
    if (!backend || !ggml_backend_is_cuda(backend) || !op || op->op != GGML_OP_FLASH_ATTN_EXT) return false;
    const auto * q = op->src[0], * k = op->src[1], * v = op->src[2], * mask = op->src[3];
    if (!q || !k || !v || !mask || q->type != GGML_TYPE_F32 ||
            mask->type != GGML_TYPE_F16 || op->type != GGML_TYPE_F32 ||
            mask->ne[2] != 1 || mask->ne[3] != 1 || mask->nb[0] != 2 ||
            k->ne[1] <= 0 || k->ne[1] > INT32_MAX || mask->nb[1] < size_t(k->ne[1])*2 || mask->nb[1]%2) return false;
    const auto caps = capabilities(backend,k->type,v->type);
    if (!caps.direct_pair) return false;
    span ranges[5];
    const ggml_tensor * tensors[] = {q,k,v,mask,op};
    for (int i = 0; i < 5; ++i) if (!tensor_span(backend, tensors[i], ranges[i])) return false;
    for (int i = 0; i < 4; ++i) if (overlap(ranges[i], ranges[4])) return false;
    for (int i = 0; i < 3; ++i) if (ranges[i].begin%16 || tensors[i]->nb[1]%16 || tensors[i]->nb[2]%16) return false;
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
    span output;
    if (!tensor_span(backend,op,output)) return false;
    for (int i = 0; i < 4; ++i) { span input; if (!tensor_span(backend,op->src[i],input) || overlap(input,output)) return false; }
    const auto base = uintptr_t(ggml_backend_buffer_get_base(op->buffer)), address = uintptr_t(op->data);
    const size_t capacity = ggml_backend_buffer_get_size(op->buffer);
    const size_t required = ggml_backend_buffer_get_alloc_size(op->buffer,op);
    if (address < base || address-base > capacity || required > capacity-(address-base)) return false;
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_flash_attn_ext(ctx,const_cast<ggml_tensor *>(op));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream())); return true;
}

// Match native TG1 vector ownership before any resident or ring span is consumed.
static bool resume_plan(ggml_backend_t backend, int32_t key, int32_t value, uint32_t heads, uint32_t kv_heads,
        size_t tokens, ggml_kv_stream_resume_plan & output) {
    if (!backend || !ggml_backend_is_cuda(backend) || capture_active(backend) || key < 0 || key >= GGML_TYPE_COUNT || value < 0 || value >= GGML_TYPE_COUNT ||
            !heads || !kv_heads || heads%kv_heads || heads > 65535 || !tokens || tokens%256 || tokens > INT32_MAX) return false;
    auto kernel = ggml_cuda_kv_stream_kernel(ggml_type(key),ggml_type(value));
    auto resumed = ggml_cuda_kv_stream_resume_kernel(ggml_type(key),ggml_type(value));
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    const auto & device = ggml_cuda_info().devices[ctx.device];
    // Native dispatch may select MMA for long unquantized TG1; keep that path on strict attention.
    if (!kernel || !resumed || device.cc < GGML_CUDA_CC_ADA_LOVELACE ||
            (!ggml_is_quantized(ggml_type(key)) && !ggml_is_quantized(ggml_type(value)))) return false;
    int occupancy = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,kernel,128,0));
    const int tiles = int(tokens/256), wave = device.nsm*occupancy;
    if (!wave || !occupancy) return false;
    int splits = std::min(occupancy,tiles), best = 0;
    int64_t best_waves = 0;
    for (int p = splits; p <= tiles; ++p) {
        const int64_t blocks = int64_t(heads)*p, waves = (blocks+wave-1)/wave;
        const int efficiency = int(100*blocks/(waves*wave));
        if (best >= 95 && waves > best_waves) break;
        if (efficiency > best) { best = efficiency; best_waves = waves; splits = p; }
    }
    return ggml_kv_stream_resume_layout_make(heads,uint32_t(splits),
        value == GGML_TYPE_F16 || value == GGML_TYPE_BF16 ? 32 : 8,output);
}

// Span order belongs to the session. Each completed call releases its input bytes for ring reuse.
static bool resume(ggml_backend_t backend, const ggml_tensor * op, ggml_backend_buffer_t workspace,
        const ggml_kv_stream_resume_plan & plan, size_t tokens, size_t first, bool last) {
    if (capture_active(backend) || !supports(backend,op) || !workspace || !tokens || tokens%256 || tokens > INT32_MAX || first >= tokens || first%256) return false;
    const auto * q = op->src[0], * k = op->src[1], * v = op->src[2], * m = op->src[3];
    if (q->ne[1] != 1 || q->ne[2] != plan.heads || k->ne[1]%256 || size_t(k->ne[1]) > tokens-first ||
            last != (first+size_t(k->ne[1]) == tokens)) return false;
    ggml_kv_stream_resume_plan checked;
    if (!ggml_kv_stream_resume_layout_make(plan.heads,plan.splits,
            v->type == GGML_TYPE_F16 || v->type == GGML_TYPE_BF16 ? 32 : 8,checked) ||
            checked.bytes != plan.bytes || checked.state_bytes != plan.state_bytes || checked.partial_offset != plan.partial_offset ||
            checked.meta_offset != plan.meta_offset || checked.values_per_thread != plan.values_per_thread) return false;
    auto kernel = ggml_cuda_kv_stream_resume_kernel(k->type,v->type);
    const auto base = uintptr_t(ggml_backend_buffer_get_base(workspace));
    const auto capacity = ggml_backend_buffer_get_size(workspace);
    if (!kernel || !base || base%128 || base > UINTPTR_MAX-capacity || capacity < plan.bytes ||
            !ggml_backend_supports_buft(backend,ggml_backend_buffer_get_type(workspace)) || ggml_backend_buffer_is_host(workspace)) return false;
    const span scratch{base,base+capacity}; span touched;
    if (!tensor_span(backend,op,touched) || overlap(scratch,touched)) return false;
    for (int i = 0; i < 4; ++i) if (!tensor_span(backend,op->src[i],touched) || overlap(scratch,touched)) return false;
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    float scale; std::memcpy(&scale,op->op_params,sizeof(scale));
    auto * partials = reinterpret_cast<float *>(base+plan.partial_offset);
    auto * meta = reinterpret_cast<float2 *>(base+plan.meta_offset);
    const ggml_cuda_kernel_launch_params launch({1,plan.splits,plan.heads},{32,4,1},0,ctx.stream());
    ggml_cuda_kernel_launch(kernel,launch,
        static_cast<const char *>(q->data),static_cast<const char *>(k->data),static_cast<const char *>(v->data),static_cast<const char *>(m->data),
        (const char *) nullptr,(const int *) nullptr,partials,meta,scale,0.0f,1.0f,1.0f,uint32_t(1),0.0f,
        int32_t(256),init_fastdiv_values(1),int32_t(q->ne[2]),int32_t(1),int32_t(q->nb[1]),int32_t(q->nb[2]),int32_t(q->nb[3]),
        int32_t(256),int32_t(tokens),int32_t(k->ne[2]),int32_t(1),int32_t(k->nb[1]),int32_t(k->nb[2]),int64_t(k->nb[3]),
        int32_t(v->nb[1]),int32_t(v->nb[2]),int64_t(v->nb[3]),int32_t(m->ne[1]),int32_t(1),int32_t(1),
        int32_t(m->nb[1]),int32_t(m->nb[2]),int64_t(m->nb[3]),reinterpret_cast<float *>(base),int(first),int(first+k->ne[1]),first == 0,last);
    CUDA_CHECK(cudaGetLastError());
    if (last) {
        if (plan.splits > 1) {
            const ggml_cuda_kernel_launch_params combine({1,plan.heads,1},{256,1,1},plan.splits*sizeof(float2),ctx.stream());
            ggml_cuda_kernel_launch(flash_attn_combine_results<256>,combine,partials,meta,static_cast<float *>(op->data),int(plan.splits));
        } else CUDA_CHECK(cudaMemcpyAsync(op->data,partials,ggml_nbytes(op),cudaMemcpyDeviceToDevice,ctx.stream()));
    }
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(ctx.stream())); return true;
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

// Keep CUDA details behind the backend-neutral registry contract.
const ggml_kv_stream_partial_ops * ggml_cuda_kv_stream_partial_ops() {
    static_assert(sizeof(float2) == sizeof(ggml_kv_stream_partial_meta), "partial metadata ABI");
    static const ggml_kv_stream_partial_ops ops{5,supports,partial,merge,fold,clear,capabilities,supports_conversion,convert,direct,resume_plan,resume};
#ifdef GGML_CUDA_NO_FA
    GGML_UNUSED(ops);
    return nullptr;
#else
    return &ops;
#endif
}
#endif
