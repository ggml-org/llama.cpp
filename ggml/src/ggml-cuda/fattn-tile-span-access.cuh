#pragma once

#include "fattn-tile-access.cuh"
#include "fattn-tile-resume.h"
#include "fattn-mma-quant.cuh"

template<ggml_type type>
struct ggml_cuda_fattn_tile_span_rows {
    const ggml_cuda_fattn_tile_span_table * table;
    int first, feature_pair, head;
    bool value;

    // Resolve encoded rows without changing the source's stock conversion expression.
    template<int bytes>
    __device__ __forceinline__ void load(half2 * destination, int row, int pair, bool valid, const half2 * zero) const {
        static_assert(bytes%4 == 0,"half2 copies required");
        if (!valid) {ggml_cuda_memcpy_1<bytes>(destination,zero); return;}
        const int token = first+row;
        const char * source = nullptr;
        if (token >= 0 && token < table->tokens) {
            int index = 0;
            if (table->count > 1 && token >= table->spans[1].first) index = 1;
            if (table->count > 2 && token >= table->spans[2].first) index = 2;
            const auto & span = table->spans[index];
            if (token >= span.first && token-span.first < span.tokens)
                source = (value ? span.v : span.k)+int64_t(token-span.first)*(value ? span.v_token_stride : span.k_token_stride)+
                    int64_t(head)*(value ? span.v_head_stride : span.k_head_stride);
        }
#pragma unroll
        for (int i = 0; i < bytes/4; ++i) destination[i] = ggml_cuda_fattn_decode_pair<type>(
            source,2*(feature_pair+pair+i),value ? table->contiguous_v : table->contiguous_k);
    }
};

template<ggml_type key, ggml_type value>
struct ggml_cuda_fattn_tile_span_kv {
    const ggml_cuda_fattn_tile_span_table * table;
    int head;
    __device__ __forceinline__ ggml_cuda_fattn_tile_span_rows<key> k(int first, int feature_pair) const {
        return {table,first,feature_pair,head,false};
    }
    __device__ __forceinline__ ggml_cuda_fattn_tile_span_rows<value> v(int first) const {
        return {table,first,0,head,true};
    }
};
