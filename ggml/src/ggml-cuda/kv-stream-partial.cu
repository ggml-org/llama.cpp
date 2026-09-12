#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "kv-stream-partial.cuh"
#include "fattn-vec.cuh"
#include "../ggml-backend-impl.h"
#include "../ggml-kv-stream.h"

#include <cmath>
#include <climits>
#include <cstring>

namespace {
struct span { uintptr_t begin = 0, end = 0; };

// Check the touched tensor range before interpreting a borrowed device pointer.
static bool tensor_span(ggml_backend_t backend, const ggml_tensor * t, span & out) {
    if (!t || !t->data || !t->buffer || !ggml_backend_supports_buft(backend, ggml_backend_buffer_get_type(t->buffer)) ||
            ggml_backend_buffer_is_host(t->buffer)) return false;
    size_t bytes = ggml_type_size(t->type);
    for (int i = 0; i < 4; ++i) {
        if (t->ne[i] <= 0 || t->ne[i] > INT32_MAX || t->nb[i] > INT32_MAX ||
                size_t(t->ne[i]-1) > (SIZE_MAX-bytes)/(t->nb[i] ? t->nb[i] : 1)) return false;
        bytes += size_t(t->ne[i]-1)*t->nb[i];
    }
    const auto base = uintptr_t(ggml_backend_buffer_get_base(t->buffer));
    const auto data = uintptr_t(t->data);
    const size_t capacity = ggml_backend_buffer_get_size(t->buffer);
    if (data < base || data-base > capacity || bytes > capacity-(data-base) || data > UINTPTR_MAX-bytes ||
            data % ggml_type_size(t->type)) return false;
    out = {data, data+bytes};
    return true;
}

// Compare touched ranges, not owning buffers: independent tensors may share one allocation.
static bool overlap(span a, span b) { return a.begin < b.end && b.begin < a.end; }

// The first adapter admits F16, head size 256, one sequence, and no sinks/bias/softcap.
static bool supports(ggml_backend_t backend, const ggml_tensor * op) {
    if (!backend || !ggml_backend_is_cuda(backend) || !op || op->op != GGML_OP_FLASH_ATTN_EXT) return false;
    const auto * q = op->src[0], * k = op->src[1], * v = op->src[2], * mask = op->src[3];
    if (!q || !k || !v || !mask || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 ||
            v->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 || op->type != GGML_TYPE_F32 ||
            mask->ne[2] != 1 || mask->ne[3] != 1 || mask->nb[0] != 2 ||
            k->ne[1] <= 0 || k->ne[1] > INT32_MAX || mask->nb[1] < size_t(k->ne[1])*2 || mask->nb[1]%2) return false;
    span ranges[5];
    const ggml_tensor * tensors[] = {q,k,v,mask,op};
    for (int i = 0; i < 5; ++i) if (!tensor_span(backend, tensors[i], ranges[i])) return false;
    for (int i = 0; i < 4; ++i) if (overlap(ranges[i], ranges[4])) return false;
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
    const ggml_kv_stream_capabilities caps{{GGML_TYPE_F16,true,true,true,false}, {GGML_TYPE_F16,true,true,true,false},true,false};
    ggml_kv_stream_execution execution;
    return ggml_kv_stream_attention_validate(&logical, {256,256,256,256,128}, caps, size_t(k->ne[1]), execution).status == ggml_kv_stream_status::success;
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
    if (!supports(backend, op)) return false;
    ggml_kv_stream_block_layout layout;
    if (ggml_kv_stream_block_layout_make(size_t(op->ne[1])*size_t(op->ne[2]), 256, layout).status != ggml_kv_stream_partial_status::success ||
            !workspace_valid(backend,buffer,layout,op,true)) return false;
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    auto * scratch = static_cast<char *>(ggml_backend_buffer_get_base(buffer)) + (second ? layout.second_offset : 0);
    const auto * q = op->src[0], * k = op->src[1], * v = op->src[2], * m = op->src[3];
    float scale; std::memcpy(&scale, op->op_params, sizeof(scale));
    const ggml_cuda_kernel_launch_params launch({unsigned(q->ne[1]),2,unsigned(q->ne[2])}, {32,4,1}, 0, ctx.stream());
    ggml_cuda_kernel_launch(flash_attn_ext_vec<256,1,GGML_TYPE_F16,GGML_TYPE_F16,false>, launch,
        (const char *)q->data, (const char *)k->data, (const char *)v->data, (const char *)m->data,
        (const char *)nullptr, (const int *)nullptr, (float *)scratch, (float2 *)(scratch+layout.partial.meta_offset),
        scale, 0.0f, 1.0f, 1.0f, uint32_t(1), 0.0f,
        int32_t(q->ne[0]), init_fastdiv_values(q->ne[1]), int32_t(q->ne[2]), int32_t(1),
        int32_t(q->nb[1]), int32_t(q->nb[2]), int32_t(q->nb[3]),
        int32_t(k->ne[0]), int32_t(k->ne[1]), int32_t(k->ne[2]), int32_t(1),
        int32_t(k->nb[1]), int32_t(k->nb[2]), int64_t(k->nb[3]),
        int32_t(v->nb[1]), int32_t(v->nb[2]), int64_t(v->nb[3]),
        int32_t(m->ne[1]), int32_t(1), int32_t(1), int32_t(m->nb[1]), int32_t(m->nb[2]), int64_t(m->nb[3]));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    return true;
}

// Validate all four contributions and stage normalized values; never write the public output here.
__global__ void merge_kernel(const float * a, const float2 * am, const float * b, const float2 * bm,
        float * values, unsigned * status) {
    const size_t row = blockIdx.x;
    const size_t channel = threadIdx.x;
    __shared__ double weights[4], mass[4], total;
    if (channel == 0) {
        const float2 meta[4] = {am[2*row],am[2*row+1],bm[2*row],bm[2*row+1]};
        double maximum = -INFINITY;
        bool valid = true;
        for (int i = 0; i < 4; ++i) {
            const double m = meta[i].x, l = meta[i].y;
            valid &= isfinite(l) && l >= 0 && (isfinite(m) || (l == 0 && m == -INFINITY));
            if (l > 0) maximum = fmax(maximum,m);
            mass[i] = l;
        }
        total = 0;
        for (int i = 0; i < 4; ++i) {
            weights[i] = mass[i] > 0 ? exp(double(meta[i].x)-maximum) : 0;
            total += mass[i]*weights[i];
        }
        if (!valid || !isfinite(total) || total > FLT_MAX) atomicOr(status,1u);
    }
    __syncthreads();
    const float u[4] = {a[(2*row)*256+channel],a[(2*row+1)*256+channel],b[(2*row)*256+channel],b[(2*row+1)*256+channel]};
    double numerator = 0;
    bool valid = true;
    for (int i = 0; i < 4; ++i) {
        valid &= isfinite(u[i]) && (mass[i] != 0 || u[i] == 0);
        numerator += double(u[i])*weights[i];
    }
    const double value = total > 0 ? numerator/total : 0;
    if (!valid || !isfinite(numerator) || fabs(numerator) > FLT_MAX || !isfinite(value) || fabs(value) > FLT_MAX) atomicOr(status,1u);
    values[row*256+channel] = float(value);
}

// Publish only after device-wide validation of this result; the four-byte status copy is deliberate in v1.
static bool merge(ggml_backend_t backend, ggml_tensor * output, ggml_backend_buffer_t buffer) {
    if (!output || output->type != GGML_TYPE_F32 || output->ne[0] != 256 || output->ne[1] <= 0 ||
            output->ne[2] <= 0 || output->ne[1] > INT32_MAX || output->ne[2] > INT32_MAX/512/output->ne[1] ||
            output->ne[3] != 1 || !ggml_is_contiguous(output)) return false;
    ggml_kv_stream_block_layout layout;
    if (ggml_kv_stream_block_layout_make(size_t(output->ne[1])*size_t(output->ne[2]),256,layout).status != ggml_kv_stream_partial_status::success ||
            !workspace_valid(backend,buffer,layout,output,false)) return false;
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    auto * raw = static_cast<char *>(ggml_backend_buffer_get_base(buffer));
    auto * flag = reinterpret_cast<unsigned *>(raw+layout.status_offset);
    CUDA_CHECK(cudaMemsetAsync(flag,0,sizeof(unsigned),ctx.stream()));
    merge_kernel<<<unsigned(layout.partial.rows),256,0,ctx.stream()>>>(
        (float *)raw, (float2 *)(raw+layout.partial.meta_offset),
        (float *)(raw+layout.second_offset), (float2 *)(raw+layout.second_offset+layout.partial.meta_offset),
        (float *)(raw+layout.value_offset), flag);
    CUDA_CHECK(cudaGetLastError());
    unsigned status;
    CUDA_CHECK(cudaMemcpyAsync(&status,flag,sizeof(status),cudaMemcpyDeviceToHost,ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    if (status) return false;
    CUDA_CHECK(cudaMemcpyAsync(output->data,raw+layout.value_offset,layout.value_bytes,cudaMemcpyDeviceToDevice,ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    return true;
}
} // namespace

// Keep CUDA details behind the backend-neutral registry contract.
const ggml_kv_stream_partial_ops * ggml_cuda_kv_stream_partial_ops() {
    static_assert(sizeof(float2) == sizeof(ggml_kv_stream_partial_meta), "partial metadata ABI");
    static const ggml_kv_stream_partial_ops ops{1,supports,partial,merge};
#ifdef GGML_CUDA_NO_FA
    GGML_UNUSED(ops);
    return nullptr;
#else
    return &ops;
#endif
}
#endif
