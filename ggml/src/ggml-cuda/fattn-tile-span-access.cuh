#pragma once

#include "fattn-tile-access.cuh"
#include "fattn-tile-spans.h"
#include "dequantize.cuh"

template<ggml_type type>
struct ggml_cuda_fattn_tile_span_rows {
    const ggml_cuda_fattn_tile_span_table * table;
    int first, feature_pair, head;
    bool value;

    // Physical boundaries change only row addresses; each encoded value rounds to stock F16 before use.
    template<int bytes>
    __device__ __forceinline__ void load(half2 * destination, int row, int pair, bool valid, const half2 * zero) const {
        static_assert(bytes%4 == 0,"half2 copies required");
        static_assert(type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F32 ||
            type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
            type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1,"unsupported tile conversion");
        const int token = first+row;
        const char * source = nullptr;
        if (valid && token >= 0 && token < table->tokens) {
            int index = 0;
            if (table->count > 1 && token >= table->spans[1].first) index = 1;
            if (table->count > 2 && token >= table->spans[2].first) index = 2;
            const auto & span = table->spans[index];
            if (token >= span.first && token-span.first < span.tokens)
                source = (value ? span.v : span.k) + int64_t(token-span.first)*(value ? span.v_token_stride : span.k_token_stride) +
                    int64_t(head)*(value ? span.v_head_stride : span.k_head_stride);
        }
        if (!source) { ggml_cuda_memcpy_1<bytes>(destination,zero); return; }
        if constexpr (type == GGML_TYPE_F16) {
            ggml_cuda_memcpy_1<bytes>(destination,reinterpret_cast<const half2 *>(source)+feature_pair+pair);
        } else {
#pragma unroll
            for (int i = 0; i < bytes/4; ++i) {
                const int feature = 2*(feature_pair+pair+i);
                float2 decoded;
                if constexpr (type == GGML_TYPE_F32) {
                    decoded = reinterpret_cast<const float2 *>(source)[feature/2];
                } else if constexpr (type == GGML_TYPE_BF16) {
                    decoded = ggml_cuda_cast<float2>(reinterpret_cast<const nv_bfloat162 *>(source)[feature/2]);
                } else if constexpr (type == GGML_TYPE_Q8_0) {
                    dequantize_q8_0(source,feature/32,feature%32,decoded);
                } else if constexpr (type == GGML_TYPE_Q4_0) {
                    const auto & block = reinterpret_cast<const block_q4_0 *>(source)[feature/32];
                    const int offset = feature%32;
                    const int shift = offset < 16 ? 0 : 4;
                    const float x = (block.qs[offset%16]>>shift)&15;
                    const float y = (block.qs[(offset+1)%16]>>shift)&15;
                    const float d = __half2float(block.d);
                    // Stock contiguous and strided converters differ in their zero sign and expression order.
                    if (value ? table->contiguous_v : table->contiguous_k) {
                        const float m = -8*d;
                        decoded = make_float2(d*x+m,d*y+m);
                    } else decoded = make_float2((x-8)*d,(y-8)*d);
                } else {
                    // Stock scalar quant conversion produces two values sixteen positions apart.
                    float2 a, b;
                    const int offset = feature%32;
                    if constexpr (type == GGML_TYPE_Q4_1) {
                        dequantize_q4_1(source,feature/32,offset%16,a);
                        dequantize_q4_1(source,feature/32,(offset+1)%16,b);
                    } else if constexpr (type == GGML_TYPE_Q5_0) {
                        dequantize_q5_0(source,feature/32,offset%16,a);
                        dequantize_q5_0(source,feature/32,(offset+1)%16,b);
                    } else {
                        dequantize_q5_1(source,feature/32,offset%16,a);
                        dequantize_q5_1(source,feature/32,(offset+1)%16,b);
                    }
                    decoded = offset < 16 ? make_float2(a.x,b.x) : make_float2(a.y,b.y);
                }
                destination[i] = __float22half2_rn(decoded);
            }
        }
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
