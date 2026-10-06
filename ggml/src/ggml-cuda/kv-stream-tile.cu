#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "kv-stream-tile.cuh"
#include "fattn-tile.cuh"
#include "../ggml-backend-impl.h"
#include <cstring>

namespace {
static fattn_kernel_t encoded_kernel(ggml_type key, ggml_type value, uint32_t c1, uint32_t c2) {
    switch (key) {
#define KEY(K) case GGML_TYPE_##K: return ggml_cuda_kv_stream_tile_kernel_##K(value,c1,c2);
        KEY(F16) KEY(BF16) KEY(Q4_0) KEY(Q4_1) KEY(Q5_0) KEY(Q5_1) KEY(Q8_0)
#undef KEY
        default: return nullptr;
    }
}

static fattn_kernel_t native_kernel(uint32_t c1, uint32_t c2) {
#define TILE(C1, C2) if (c1 == C1 && c2 == C2) return flash_attn_tile<256,256,C1,C2,false>;
    TILE(2,1) TILE(4,1)
    TILE(1,2) TILE(2,2) TILE(4,2)
    TILE(1,4) TILE(2,4) TILE(4,4)
    TILE(1,8) TILE(2,8) TILE(4,8)
#undef TILE
    return nullptr;
}

// Follow stock's serial, masked head-256 column selection, including its padded GQA requirement.
static bool geometry(ggml_backend_cuda_context & ctx, uint32_t heads, uint32_t kv_heads,
        uint32_t queries, size_t tokens, ggml_cuda_fattn_tile_resume_geometry & g) {
    if (!heads || heads > 65535 || !kv_heads || heads%kv_heads || !queries || queries > 4 ||
            tokens < queries || tokens > INT32_MAX) return false;
    const int cc=ggml_cuda_info().devices[ctx.device].cc;
    const int compiled=ggml_cuda_highest_compiled_arch(cc);
    if (compiled < 610) return false;
    const uint32_t ratio=heads/kv_heads;
    uint32_t c2=1;
    if (tokens%FATTN_KQ_STRIDE == 0) for (uint32_t candidate : {8u,4u,2u})
        if (ratio%candidate == 0) {c2=candidate; break;}
    uint32_t columns=2;
    while (columns < queries*c2) columns*=2;
    if (columns > 32) return false;
    const uint32_t c1=columns/c2;
    g={256,heads,kv_heads,queries,c1,c2,
        uint32_t(ggml_cuda_fattn_tile_get_nthreads(256,256,columns,cc)),
        uint32_t(ggml_cuda_fattn_tile_get_nbatch_fa(256,256,columns,cc)),1,
        uint32_t(fast_fp16_available(cc) ? sizeof(half2) : sizeof(float2))};
    return native_kernel(c1,c2) != nullptr;
}
}

// Use the native kernel's occupancy and split search, not the resumable reader's register count.
bool ggml_cuda_kv_stream_tile_plan(ggml_backend_cuda_context & ctx, ggml_type key, ggml_type value,
        uint32_t heads, uint32_t kv_heads, uint32_t queries, size_t tokens, ggml_kv_stream_resume_plan & output) {
    ggml_cuda_fattn_tile_resume_geometry g;
    if (!geometry(ctx,heads,kv_heads,queries,tokens,g)) return false;
    auto resumed=encoded_kernel(key,value,g.ncols1,g.ncols2);
    auto native=native_kernel(g.ncols1,g.ncols2);
    if (!resumed || !native) return false;
    int occupancy=0, resume_occupancy=0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,native,g.nthreads,0) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resume_occupancy,resumed,g.nthreads,0) != cudaSuccess) {
        (void) cudaGetLastError(); return false;
    }
    if (!occupancy || !resume_occupancy) return false;
    const int tiles=int((tokens+g.nbatch_fa-1)/g.nbatch_fa);
    const int64_t dst_tiles=int64_t((queries+g.ncols1-1)/g.ncols1)*(heads/g.ncols2);
    const int64_t wave=int64_t(ggml_cuda_info().devices[ctx.device].nsm)*occupancy;
    if (!wave) return false;
    int splits=std::min(occupancy,tiles), best=0;
    int64_t best_waves=0;
    for (int candidate=splits;candidate<=tiles;++candidate) {
        const int64_t blocks=dst_tiles*candidate, waves=(blocks+wave-1)/wave;
        const int efficiency=int(100*blocks/(waves*wave));
        if (best >= 95 && waves > best_waves) break;
        if (efficiency > best) {best=efficiency; best_waves=waves; splits=candidate;}
    }
    g.splits=uint32_t(splits);
    ggml_cuda_fattn_tile_resume_layout layout;
    if (!ggml_cuda_fattn_tile_resume_layout_make(g,tokens,layout)) return false;
    ggml_kv_stream_resume_plan next;
    next.heads=heads; next.queries=queries; next.splits=g.splits; next.tokens=tokens;
    next.state_bytes=layout.state_bytes; next.partial_offset=layout.partial_offset;
    next.meta_offset=layout.meta_offset; next.bytes=layout.bytes;
    next.kernel_config[0]=1; next.kernel_config[1]=kv_heads;
    next.kernel_config[2]=g.ncols1; next.kernel_config[3]=g.ncols2;
    next.kernel_config[4]=uint32_t(ggml_cuda_highest_compiled_arch(ggml_cuda_info().devices[ctx.device].cc));
    next.resume_resident=true;
    output=next; return true;
}

// Reconstruct all offsets and compiled geometry before interpreting the backend-private resume configuration.
bool ggml_cuda_kv_stream_tile_layout(ggml_backend_cuda_context & ctx, ggml_type key, ggml_type value,
        const ggml_kv_stream_resume_plan & plan, ggml_cuda_fattn_tile_resume_layout & output, size_t * shared_bytes) {
    ggml_cuda_fattn_tile_resume_geometry g;
    if (plan.kernel_config[0] != 1 || !plan.resume_resident || plan.values_per_thread ||
            plan.kernel_config[4] != uint32_t(ggml_cuda_highest_compiled_arch(ggml_cuda_info().devices[ctx.device].cc)) ||
            !geometry(ctx,plan.heads,plan.kernel_config[1],plan.queries,plan.tokens,g) ||
            plan.kernel_config[2] != g.ncols1 || plan.kernel_config[3] != g.ncols2 ||
            !encoded_kernel(key,value,g.ncols1,g.ncols2)) return false;
    g.splits=plan.splits;
    ggml_cuda_fattn_tile_resume_layout layout;
    if (!ggml_cuda_fattn_tile_resume_layout_make(g,plan.tokens,layout) ||
            plan.bytes != layout.bytes || plan.state_bytes != layout.state_bytes ||
            plan.partial_offset != layout.partial_offset || plan.meta_offset != layout.meta_offset) return false;
    if (shared_bytes) {
        cudaFuncAttributes attributes{};
        if (cudaFuncGetAttributes(&attributes,encoded_kernel(key,value,g.ncols1,g.ncols2)) != cudaSuccess) {
            (void) cudaGetLastError(); return false;
        }
        *shared_bytes=attributes.sharedSizeBytes;
    }
    output=layout; return true;
}

bool ggml_cuda_kv_stream_tile_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * op,
        const ggml_kv_stream_span_plan_view & view, ggml_backend_buffer_t workspace,
        const ggml_kv_stream_resume_plan & plan, size_t first, size_t end) {
    const auto * q=op->src[0], * k=op->src[1], * v=op->src[2], * m=op->src[3];
    ggml_cuda_fattn_tile_resume_layout layout;
    ggml_cuda_fattn_tile_resume_descriptor descriptor;
    if (!ggml_cuda_kv_stream_tile_layout(ctx,k->type,v->type,plan,layout) ||
            !ggml_cuda_fattn_tile_resume_wave_make(view,layout,workspace,first,end,descriptor,true)) return false;
    auto * raw=static_cast<char *>(ggml_backend_buffer_get_base(workspace));
    auto kernel=encoded_kernel(k->type,v->type,layout.geometry.ncols1,layout.geometry.ncols2);
    float scale; std::memcpy(&scale,op->op_params,sizeof(scale));
    CUDA_CHECK(cudaMemcpyAsync(raw,&descriptor,sizeof(descriptor),cudaMemcpyHostToDevice,ctx.stream()));
    const auto & g=layout.geometry;
    const ggml_cuda_kernel_launch_params launch(
        {(g.queries+g.ncols1-1)/g.ncols1,g.splits,g.query_heads/g.ncols2},{32,g.nthreads/32,1},0,ctx.stream());
    ggml_cuda_kernel_launch(kernel,launch,
        static_cast<const char *>(q->data),raw,static_cast<const char *>(nullptr),static_cast<const char *>(m->data),
        static_cast<const char *>(nullptr),static_cast<const int *>(nullptr),reinterpret_cast<float *>(raw+layout.partial_offset),
        reinterpret_cast<float2 *>(raw+layout.meta_offset),scale,0.0f,1.0f,1.0f,uint32_t(1),0.0f,
        int32_t(256),init_fastdiv_values(g.queries),int32_t(g.query_heads),int32_t(1),
        int32_t(q->nb[1]),int32_t(q->nb[2]),int32_t(q->nb[3]),
        int32_t(256),int32_t(plan.tokens),int32_t(g.kv_heads),int32_t(1),
        int32_t(512*g.kv_heads),int32_t(512),int64_t(plan.tokens)*512*g.kv_heads,
        int32_t(512*g.kv_heads),int32_t(512),int64_t(plan.tokens)*512*g.kv_heads,
        int32_t(m->ne[1]),int32_t(1),int32_t(1),int32_t(m->nb[1]),int32_t(m->nb[2]),int64_t(m->nb[3]));
    CUDA_CHECK(cudaGetLastError());
    if (end == plan.tokens) {
        auto * final=reinterpret_cast<float *>(raw+layout.output_offset);
        if (g.splits > 1) {
            const ggml_cuda_kernel_launch_params combine({g.queries,g.query_heads,1},{256,1,1},g.splits*sizeof(float2),ctx.stream());
            ggml_cuda_kernel_launch(flash_attn_combine_results<256>,combine,
                reinterpret_cast<float *>(raw+layout.partial_offset),reinterpret_cast<float2 *>(raw+layout.meta_offset),final,int(g.splits));
            CUDA_CHECK(cudaGetLastError());
        } else CUDA_CHECK(cudaMemcpyAsync(final,raw+layout.partial_offset,layout.output_bytes,cudaMemcpyDeviceToDevice,ctx.stream()));
        CUDA_CHECK(cudaMemcpyAsync(op->data,final,layout.output_bytes,cudaMemcpyDeviceToDevice,ctx.stream()));
    }
    return true;
}
#endif
