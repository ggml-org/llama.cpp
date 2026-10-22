#pragma once

#include "fattn-mma-f16.cuh"

#define GGML_CUDA_KV_RUNTIME_INLINE __device__ __forceinline__

enum class ggml_cuda_kv_load_stage { decode, prefetch, finish };

template<ggml_type format, ggml_cuda_kv_load_stage operation, int stride, int warps, int rows, bool dense>
static __device__ __forceinline__ void ggml_cuda_kv_runtime_typed_load(
        const ggml_cuda_kv_span * spans, int count, bool value, int head, int first, int feature,
        half2 * tile, int width, int live, bool contiguous, const char * raw, int full_width) {
    if constexpr (operation == ggml_cuda_kv_load_stage::decode) {
        ggml_cuda_fattn_load_encoded_tile<format,stride,warps,rows>(
            spans,count,value,head,first,feature,tile,width,live,contiguous);
    } else if constexpr (operation == ggml_cuda_kv_load_stage::prefetch) {
        ggml_cuda_fattn_prefetch_encoded_tile<format,warps,rows>(
            spans,count,value,head,first,reinterpret_cast<char *>(tile),width,live);
    } else {
        ggml_cuda_fattn_finish_encoded_tile<format,stride,warps,rows,dense>(
            raw,tile,full_width,feature,width,contiguous);
    }
}

// One uniform format choice per tile, not per decoded value.
template<ggml_cuda_kv_load_stage operation, int stride, int warps, int rows, bool dense>
static GGML_CUDA_KV_RUNTIME_INLINE void ggml_cuda_kv_runtime_load(
        ggml_type format, const ggml_cuda_kv_span * spans, int count, bool value, int head, int first,
        int feature, half2 * tile, int width, int live, bool contiguous, const char * raw = nullptr,
        int full_width = 0) {
    if constexpr (operation == ggml_cuda_kv_load_stage::finish) {
        if (format == GGML_TYPE_F32) format=GGML_TYPE_F16;
    }
#define FORMAT_CASE(F) case GGML_TYPE_##F: ggml_cuda_kv_runtime_typed_load<GGML_TYPE_##F,operation,stride,warps,rows,dense>( \
        spans,count,value,head,first,feature,tile,width,live,contiguous,raw,full_width); return;
    switch (format) {
        FORMAT_CASE(F32)
        FORMAT_CASE(F16)
        FORMAT_CASE(BF16)
        FORMAT_CASE(Q8_0)
        FORMAT_CASE(Q5_0)
        FORMAT_CASE(Q5_1)
        FORMAT_CASE(Q4_0)
        FORMAT_CASE(Q4_1)
        default: asm("trap;"); return;
    }
#undef FORMAT_CASE
}

template<bool encoded, bool resume, bool dense>
struct ggml_cuda_fattn_mma_runtime_span_kv {
    static constexpr bool spanned=true;
    static constexpr bool staged=encoded;
    static constexpr bool resumable=resume;
#if defined(CP_ASYNC_AVAILABLE) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    static constexpr bool feature_pipelined=!encoded && resume && !dense;
#else
    static constexpr bool feature_pipelined=false;
#endif
    // Every supported staging encoding fits the existing FP16 tile; F32 rounds on ingress.
    static constexpr ggml_type key_type=GGML_TYPE_F16, v_type=GGML_TYPE_F16;
    const ggml_cuda_kv_span * spans;
    int count, head;
    uint32_t conversion_flags;
    ggml_cuda_fattn_mma_resume_control control;
    int slice;
    ggml_type key_format, value_format;

    static __device__ __forceinline__ half2 * feature_buffer() {
        __shared__ __align__(16) half2 tile[32*128];
        return tile;
    }

    __device__ inline const int32_t * mask_bounds() const {
        if constexpr (dense) return nullptr;
        return control.mask_bounds;
    }
    __device__ inline uint32_t block_index() const {
        if constexpr (resume) return control.block_ids ? control.block_ids[blockIdx.x] : blockIdx.x;
        return blockIdx.x;
    }
    __device__ inline uint32_t block_count() const {
        if constexpr (resume) return control.native_blocks ? control.native_blocks : gridDim.x;
        return gridDim.x;
    }
    __device__ inline ggml_cuda_fattn_mma_runtime_span_kv with_head(int next) const {
        auto result=*this; result.head=next; return result;
    }
    __device__ inline ggml_cuda_fattn_mma_runtime_span_kv with_slice(int next) const {
        auto result=*this; result.slice=next; return result;
    }
    __device__ inline const half2 * offset(const half2 *, int) const { return nullptr; }

    template<int stride, int warps, int rows, bool asynchronous, bool bounds, bool sparse>
    __device__ inline void load(bool value, const half2 *, half2 * tile, int width, int,
            int first, int live, const int32_t *, int feature) const {
        static_assert(!sparse,"sparse attention over spans is not supported");
        if constexpr (!encoded) {
            flash_attn_ext_f16_load_span_tile<stride,warps,rows,asynchronous>(
                spans,count,value,head,first,feature,tile,width,live);
        } else {
            constexpr auto operation=asynchronous ? ggml_cuda_kv_load_stage::prefetch : ggml_cuda_kv_load_stage::decode;
            const bool contiguous=dense ? false : (conversion_flags&(value ? 2u : 1u)) != 0;
            ggml_cuda_kv_runtime_load<operation,stride,warps,rows,dense>(value ? value_format : key_format,
                spans,count,value,head,first,feature,tile,width,live,contiguous);
        }
    }
    template<int stride, int warps, int rows>
    __device__ inline void finish_key(const char * raw, half2 * tile, int width) const {
        ggml_cuda_kv_runtime_load<ggml_cuda_kv_load_stage::finish,stride,warps,rows,dense>(key_format,
            spans,count,false,head,0,0,tile,width,rows,dense ? false : (conversion_flags&1u) != 0,raw,width);
    }
    template<int stride, int warps, int rows>
    __device__ inline void finish_value(const char * raw, half2 * tile, int width) const {
        ggml_cuda_kv_runtime_load<ggml_cuda_kv_load_stage::finish,stride,warps,rows,dense>(value_format,
            spans,count,true,head,0,0,tile,width,rows,dense ? false : (conversion_flags&2u) != 0,raw,width);
    }
};

#undef GGML_CUDA_KV_RUNTIME_INLINE
