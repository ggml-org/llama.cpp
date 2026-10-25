#pragma once

#include "kv-stream-span.h"
#include "fattn-mma-quant.cuh"
#include "fattn-mma-quant-chunk.cuh"
#include "fattn-mma-resume.h"

// Copy register fragments as integer words, without floating-point conversion or arithmetic.
template<size_t bytes>
static __device__ __forceinline__ void ggml_cuda_fattn_mma_checkpoint_copy(void * destination, const void * source) {
    static_assert(bytes%4 == 0, "checkpoint must contain complete words");
    auto * dst=static_cast<uint32_t *>(destination);
    const auto * src=static_cast<const uint32_t *>(source);
#pragma unroll
    for (size_t i=0;i<bytes/4;++i) dst[i]=src[i];
}

template<size_t thread_bytes>
static __device__ __forceinline__ char * ggml_cuda_fattn_mma_checkpoint_thread(
        const ggml_cuda_fattn_mma_resume_control & control, int slice, uint32_t block) {
    const size_t thread = threadIdx.y*blockDim.x+threadIdx.x;
    const size_t original_slot = size_t(slice)+block;
    const size_t slot = control.state_per_block ? block :
        control.state_ids ? control.state_ids[original_slot] : original_slot;
    return reinterpret_cast<char *>(uintptr_t(control.state)+(slot*blockDim.x*blockDim.y+thread)*thread_bytes);
}

// Resolve an ordered logical token without using physical span order.
static __device__ __forceinline__ int flash_attn_ext_f16_span_index(
        const ggml_cuda_kv_span * spans, int count, int token) {
    int index = 0;
    if (count > 1 && token >= spans[1].first) index = 1;
    if (count > 2 && token >= spans[2].first) index = 2;
    const auto & span = spans[index];
    return token >= span.first && token-span.first < span.tokens ? index : -1;
}

// Select one FP16 head row inside a validated physical interval.
static __device__ __forceinline__ const half2 * flash_attn_ext_f16_span_row(
        const ggml_cuda_kv_span * spans, int count, bool value, int head, int token, int feature) {
    const int index = flash_attn_ext_f16_span_index(spans, count, token);
    if (index < 0) return nullptr;
    const auto & span = spans[index];
    return reinterpret_cast<const half2 *>((value ? span.v : span.k)+
        int64_t(token-span.first)*(value ? span.v_token_stride : span.k_token_stride)+
        int64_t(head)*(value ? span.v_head_stride : span.k_head_stride))+feature;
}

// Resolve FP16 span rows into the stock swizzled shared-memory tile.
template<int stride_tile, int nwarps, int nbatch_fa, bool use_cp_async>
static __device__ __forceinline__ void flash_attn_ext_f16_load_span_tile(
        const ggml_cuda_kv_span * spans, int count, bool value, int head, int first, int feature,
        half2 * tile, int width, int live) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int index = flash_attn_ext_f16_span_index(spans, count, first);
    if (index >= 0 && live == nbatch_fa && nbatch_fa <= spans[index].tokens-(first-spans[index].first) &&
            (value ? spans[index].v_token_stride : spans[index].k_token_stride)/sizeof(half2) <= INT32_MAX) {
        const auto & span = spans[index];
        const half2 * source;
        // Keep the wider-query/smaller-tile route lean; holding this cached address increased TG8 spills.
        if constexpr (nbatch_fa >= 64) source=reinterpret_cast<const half2 *>((value ? span.v : span.k)+
            int64_t(first-span.first)*(value ? span.v_token_stride : span.k_token_stride)+
            int64_t(head)*(value ? span.v_head_stride : span.k_head_stride))+feature;
        else source=flash_attn_ext_f16_span_row(spans,count,value,head,first,feature);
        flash_attn_ext_f16_load_tile<stride_tile,nwarps,nbatch_fa,use_cp_async,false,false>(
            source, tile, width, int((value ? span.v_token_stride : span.k_token_stride)/sizeof(half2)), 0, live, nullptr);
        return;
    }
    constexpr int pairs_per_chunk = 16/sizeof(half2);
    const int chunks = width/pairs_per_chunk;
    const half2 zero[pairs_per_chunk] = {};
    unsigned int shared = 0;
    if constexpr (use_cp_async) {
        static_assert(warp_size == 32, "bad warp_size");
        shared = ggml_cuda_cvta_generic_to_shared(tile);
    }
    auto load = [&] __device__ (const int n) {
        const int stride_k = 32 >> n;
        const int begin = stride_k == 32 ? 0 : chunks-chunks%(2*stride_k);
        const int end = chunks-chunks%stride_k;
        const int stride_i = warp_size/stride_k;
        if (begin == end) return;
#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
            const int i = i0+threadIdx.y*stride_i+(stride_k == warp_size ? 0 : threadIdx.x/stride_k);
            if (i0+nwarps*stride_i > nbatch_fa && i >= nbatch_fa) break;
            const auto * source = i < live ? flash_attn_ext_f16_span_row(spans, count, value, head, first+i, feature) : nullptr;
#pragma unroll
            for (int k0 = begin; k0 < end; k0 += stride_k) {
                const int k = k0+(stride_k == warp_size ? threadIdx.x : threadIdx.x%stride_k);
                if constexpr (use_cp_async) {
                    if (source) cp_async_cg_16<64>(shared+swizzle<stride_tile*sizeof(half2),char>(
                        i*stride_tile*sizeof(half2)+k*16,i), source+k*pairs_per_chunk);
                    else ggml_cuda_memcpy_1<16>(swizzle<stride_tile>(tile, i*stride_tile+k*pairs_per_chunk, i), zero);
                } else ggml_cuda_memcpy_1<16>(swizzle<stride_tile>(tile, i*stride_tile+k*pairs_per_chunk, i),
                    source ? source+k*pairs_per_chunk : zero);
            }
        }
    };
    ggml_cuda_unroll<6>{}(load);
}

// Decode only the requested encoded features into the stock shared-memory layout.
template<ggml_type type, int stride, int warps, int rows>
static __device__ __forceinline__ void ggml_cuda_fattn_load_encoded_tile(
        const ggml_cuda_kv_span * spans, int count, bool value, int head, int first, int feature,
        half2 * tile, int width, int live, bool contiguous) {
    constexpr int warp = ggml_cuda_get_physical_warp_size();
    const int chunks = width/4;
    auto load = [&] __device__ (const int n) {
        const int step = 32 >> n, begin = step == 32 ? 0 : chunks-chunks%(2*step), end = chunks-chunks%step;
        const int stride_i = warp/step;
        if (begin == end) return;
#pragma unroll
        for (int i0 = 0; i0 < rows; i0 += warps*stride_i) {
            const int row = i0+threadIdx.y*stride_i+(step == warp ? 0 : threadIdx.x/step);
            if (i0+warps*stride_i > rows && row >= rows) break;
            const int index = row < live ? flash_attn_ext_f16_span_index(spans,count,first+row) : -1;
            const char * source = nullptr;
            if (index >= 0) {
                const auto & span = spans[index];
                source = (value ? span.v : span.k)+int64_t(first+row-span.first)*(value ? span.v_token_stride : span.k_token_stride)+
                    int64_t(head)*(value ? span.v_head_stride : span.k_head_stride);
            }
#pragma unroll
            for (int k0 = begin; k0 < end; k0 += step) {
                const int k = k0+(step == warp ? threadIdx.x : threadIdx.x%step);
                half2 decoded[4];
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) decoded[pair] = ggml_cuda_fattn_decode_pair<type>(source,2*feature+8*k+2*pair,contiguous);
                ggml_cuda_memcpy_1<16>(swizzle<stride>(tile,row*stride+k*4,row),decoded);
            }
        }
    };
    ggml_cuda_unroll<6>{}(load);
}

// Copy compressed rows into an existing tile region without applying the FP16 swizzle.
template<ggml_type type, int warps, int rows>
static __device__ __forceinline__ void ggml_cuda_fattn_prefetch_encoded_tile(
        const ggml_cuda_kv_span * spans, int count, bool value, int head, int first,
        char * staging, int full_width, int live) {
    constexpr int warp = ggml_cuda_get_physical_warp_size();
    if constexpr (type == GGML_TYPE_F32) {
        const int thread = threadIdx.y*warp+threadIdx.x;
        for (int i = thread; i < rows*full_width; i += warps*warp) {
            const int row = i/full_width, pair = i%full_width;
            const int index = row < live ? flash_attn_ext_f16_span_index(spans,count,first+row) : -1;
            const char * source = nullptr;
            if (index >= 0) {
                const auto & span = spans[index];
                source = (value ? span.v : span.k)+int64_t(first+row-span.first)*(value ? span.v_token_stride : span.k_token_stride)+
                    int64_t(head)*(value ? span.v_head_stride : span.k_head_stride);
            }
            reinterpret_cast<half2 *>(staging)[i] = source ?
                __float22half2_rn(reinterpret_cast<const float2 *>(source)[pair]) : __float22half2_rn(make_float2(0,0));
        }
        return;
    }
    const int bytes = ggml_cuda_fattn_encoded_traits<type>::row_bytes(full_width), chunks = bytes/16;
    const unsigned int shared = ggml_cuda_cvta_generic_to_shared(staging);
    const int4 zero = {};
    const int first_index=flash_attn_ext_f16_span_index(spans,count,first);
    const bool interior=first_index >= 0 && live == rows && rows <= spans[first_index].tokens-(first-spans[first_index].first);
    const char * base=nullptr;
    int64_t pitch=0;
    if (interior) {
        const auto & span=spans[first_index];
        pitch=value ? span.v_token_stride : span.k_token_stride;
        base=(value ? span.v : span.k)+int64_t(first-span.first)*pitch+
            int64_t(head)*(value ? span.v_head_stride : span.k_head_stride);
    }
    auto load = [&] __device__ (const int n) {
        const int step = warp >> n, begin = step == warp ? 0 : chunks-chunks%(2*step), end = chunks-chunks%step;
        const int stride_i = warp/step;
        if (begin == end) return;
#pragma unroll
        for (int i0 = 0; i0 < rows; i0 += warps*stride_i) {
            const int row = i0+threadIdx.y*stride_i+(step == warp ? 0 : threadIdx.x/step);
            if (i0+warps*stride_i > rows && row >= rows) break;
            const int index = row < live && !interior ? flash_attn_ext_f16_span_index(spans,count,first+row) : -1;
            const char * source = interior ? base+int64_t(row)*pitch : nullptr;
            if (!interior && index >= 0) {
                const auto & span = spans[index];
                source = (value ? span.v : span.k)+int64_t(first+row-span.first)*(value ? span.v_token_stride : span.k_token_stride)+
                    int64_t(head)*(value ? span.v_head_stride : span.k_head_stride);
            }
#pragma unroll
            for (int k0 = begin; k0 < end; k0 += step) {
                const int k = k0+(step == warp ? threadIdx.x : threadIdx.x%step);
                if (source) cp_async_cg_16<64>(shared+row*bytes+16*k,source+16*k);
                else ggml_cuda_memcpy_1<16>(staging+row*bytes+16*k,&zero);
            }
        }
    };
    ggml_cuda_unroll<6>{}(load);
}

// Expand completed compressed rows into the FP16 swizzle after their readers have retired.
template<ggml_type type, int stride, int warps, int rows, bool group_strided = false>
static __device__ __forceinline__ void ggml_cuda_fattn_finish_encoded_tile(
        const char * staging, half2 * tile, int full_width, int feature, int width, bool contiguous) {
    constexpr int warp = ggml_cuda_get_physical_warp_size();
    const int bytes = ggml_cuda_fattn_encoded_traits<type>::row_bytes(full_width), chunks = width/4;
    auto load = [&] __device__ (const int n) {
        const int step = warp >> n, begin = step == warp ? 0 : chunks-chunks%(2*step), end = chunks-chunks%step;
        const int stride_i = warp/step;
        if (begin == end) return;
#pragma unroll
        for (int i0 = 0; i0 < rows; i0 += warps*stride_i) {
            const int row = i0+threadIdx.y*stride_i+(step == warp ? 0 : threadIdx.x/step);
            if (i0+warps*stride_i > rows && row >= rows) break;
#pragma unroll
            for (int k0 = begin; k0 < end; k0 += step) {
                const int k = k0+(step == warp ? threadIdx.x : threadIdx.x%step);
                half2 decoded[4];
                if constexpr (group_strided) {
                    ggml_cuda_fattn_decode_strided_chunk8<type>(staging+row*bytes,2*feature+8*k,decoded);
                } else {
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair) decoded[pair] = ggml_cuda_fattn_decode_pair<type>(
                        staging+row*bytes,2*feature+8*k+2*pair,contiguous);
                }
                ggml_cuda_memcpy_1<16>(swizzle<stride>(tile,row*stride+k*4,row),decoded);
            }
        }
    };
    ggml_cuda_unroll<6>{}(load);
}

// Keep stock contiguous and sparse loads compile-time direct.
struct ggml_cuda_fattn_mma_contiguous_kv {
    static constexpr bool spanned = false;
    static constexpr bool staged = false;
    static constexpr bool resumable = false;
    static constexpr bool feature_pipelined = false;
    __device__ __forceinline__ const int32_t * mask_bounds() const { return nullptr; }
    __device__ __forceinline__ uint32_t block_index() const { return blockIdx.x; }
    __device__ __forceinline__ uint32_t block_count() const { return gridDim.x; }
    __device__ __forceinline__ ggml_cuda_fattn_mma_contiguous_kv with_head(int) const { return {}; }
    __device__ __forceinline__ ggml_cuda_fattn_mma_contiguous_kv with_slice(int) const { return {}; }
    __device__ __forceinline__ const half2 * offset(const half2 * source, int feature) const { return source+feature; }
    template<int stride, int warps, int rows, bool asynchronous, bool bounds, bool sparse>
    __device__ __forceinline__ void load(bool, const half2 * source, half2 * tile, int width, int row_stride,
            int first, int live, const int32_t * indices, int) const {
        flash_attn_ext_f16_load_tile<stride,warps,rows,asynchronous,bounds,sparse>(
            source, tile, width, row_stride, first, live, indices);
    }
};

// Change source addresses only; sparse attention is not implemented over spans.
template<ggml_type key = GGML_TYPE_F16, ggml_type value_type = GGML_TYPE_F16, bool resume = false>
struct ggml_cuda_fattn_mma_typed_span_kv {
    static constexpr bool spanned = true;
    static constexpr bool encoded = key != GGML_TYPE_F16 || value_type != GGML_TYPE_F16;
    static constexpr bool staged = encoded;
    static constexpr bool resumable = resume;
    static constexpr bool feature_pipelined = false;
    // F32 is rounded on ingress so every staged row fits the existing decoded tile.
    static constexpr ggml_type key_type = key == GGML_TYPE_F32 ? GGML_TYPE_F16 : key;
    static constexpr ggml_type v_type = value_type == GGML_TYPE_F32 ? GGML_TYPE_F16 : value_type;
    const ggml_cuda_kv_span * spans;
    int count;
    int head;
    uint32_t conversion_flags = 3;
    ggml_cuda_fattn_mma_resume_control control;
    __device__ __forceinline__ const int32_t * mask_bounds() const { return control.mask_bounds; }
    int slice = 0;
    __device__ __forceinline__ uint32_t block_index() const {
        if constexpr (resume) return control.block_ids ? control.block_ids[blockIdx.x] : blockIdx.x;
        return blockIdx.x;
    }
    __device__ __forceinline__ uint32_t block_count() const {
        if constexpr (resume) return control.native_blocks ? control.native_blocks : gridDim.x;
        return gridDim.x;
    }
    __device__ __forceinline__ ggml_cuda_fattn_mma_typed_span_kv with_head(int next) const {
        auto result=*this; result.head=next; return result;
    }
    __device__ __forceinline__ ggml_cuda_fattn_mma_typed_span_kv with_slice(int next) const {
        auto result=*this; result.slice=next; return result;
    }
    __device__ __forceinline__ const half2 * offset(const half2 *, int) const { return nullptr; }
    template<int stride, int warps, int rows, bool asynchronous, bool bounds, bool sparse>
    __device__ __forceinline__ void load(bool value, const half2 *, half2 * tile, int width, int,
            int first, int live, const int32_t *, int feature) const {
        static_assert(!sparse, "sparse attention over spans is not implemented");
        if constexpr (!encoded) flash_attn_ext_f16_load_span_tile<stride,warps,rows,asynchronous>(
            spans,count,value,head,first,feature,tile,width,live);
        else if constexpr (asynchronous && staged) {
            if (value) ggml_cuda_fattn_prefetch_encoded_tile<value_type,warps,rows>(
                spans,count,true,head,first,reinterpret_cast<char *>(tile),width,live);
            else ggml_cuda_fattn_prefetch_encoded_tile<key,warps,rows>(
                spans,count,false,head,first,reinterpret_cast<char *>(tile),width,live);
        } else {
            if (value) ggml_cuda_fattn_load_encoded_tile<value_type,stride,warps,rows>(
                spans,count,true,head,first,feature,tile,width,live,conversion_flags&2);
            else ggml_cuda_fattn_load_encoded_tile<key,stride,warps,rows>(
                spans,count,false,head,first,feature,tile,width,live,conversion_flags&1);
        }
    }
    template<int stride, int warps, int rows>
    __device__ __forceinline__ void finish_key(const char * raw,half2 * tile,int width) const {
        ggml_cuda_fattn_finish_encoded_tile<key_type,stride,warps,rows>(raw,tile,width,0,width,conversion_flags&1);
    }
    template<int stride, int warps, int rows>
    __device__ __forceinline__ void finish_value(const char * raw,half2 * tile,int width) const {
        ggml_cuda_fattn_finish_encoded_tile<v_type,stride,warps,rows>(raw,tile,width,0,width,conversion_flags&2);
    }
};

using ggml_cuda_fattn_mma_span_kv = ggml_cuda_fattn_mma_typed_span_kv<>;
