#pragma once

#include "dequantize.cuh"

// Convert adjacent encoded values with the stock contiguous or strided expression.
template<ggml_type type>
static __device__ __forceinline__ half2 ggml_cuda_fattn_decode_pair(const char * row, int first, bool contiguous) {
    static_assert(type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F32 ||
        type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
        type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1,"unsupported stock tile conversion");
    if (!row) {
        const half zero = type == GGML_TYPE_Q5_0 || (type == GGML_TYPE_Q4_0 && !contiguous) ?
            __ushort_as_half(0x8000) : __ushort_as_half(0);
        return __halves2half2(zero,zero);
    }
    if constexpr (type == GGML_TYPE_F16) return reinterpret_cast<const half2 *>(row)[first/2];
    else if constexpr (type == GGML_TYPE_BF16) return __float22half2_rn(
        ggml_cuda_cast<float2>(reinterpret_cast<const nv_bfloat162 *>(row)[first/2]));
    else if constexpr (type == GGML_TYPE_F32) return __float22half2_rn(reinterpret_cast<const float2 *>(row)[first/2]);
    else if constexpr (type == GGML_TYPE_Q8_0) {
        const auto & block = reinterpret_cast<const block_q8_0 *>(row)[first/32];
        const int offset = first%32;
        if (contiguous) return __hmul2(make_half2(block.qs[offset],block.qs[offset+1]),__half2half2(block.d));
        float2 values;
        dequantize_q8_0(row,first/32,offset,values);
        return __float22half2_rn(values);
    } else {
        float2 a,b;
        const int offset = first%32;
        if constexpr (type == GGML_TYPE_Q4_0) {
            if (contiguous) {
                const auto & block = reinterpret_cast<const block_q4_0 *>(row)[first/32];
                const int shift = offset < 16 ? 0 : 4;
                const float d = __half2float(block.d), m = -8*d;
                const auto * q = block.qs+offset%16;
                return __floats2half2_rn(d*((q[0]>>shift)&15)+m,d*((q[1]>>shift)&15)+m);
            }
            dequantize_q4_0(row,first/32,offset%16,a);
            dequantize_q4_0(row,first/32,(offset+1)%16,b);
        } else if constexpr (type == GGML_TYPE_Q4_1) {
            dequantize_q4_1(row,first/32,offset%16,a);
            dequantize_q4_1(row,first/32,(offset+1)%16,b);
        } else if constexpr (type == GGML_TYPE_Q5_0) {
            dequantize_q5_0(row,first/32,offset%16,a);
            dequantize_q5_0(row,first/32,(offset+1)%16,b);
        } else {
            dequantize_q5_1(row,first/32,offset%16,a);
            dequantize_q5_1(row,first/32,(offset+1)%16,b);
        }
        return __floats2half2_rn(offset < 16 ? a.x : a.y,offset < 16 ? b.x : b.y);
    }
}

// Encoded block sizes are taken from the stock types, never from a particular K/V pair.
template<ggml_type type>
struct ggml_cuda_fattn_encoded_traits {
    static_assert(type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F32 ||
        type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
        type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1,"unsupported encoded cache type");
    static constexpr int block = type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F32 ? 1 : 32;
    static constexpr int bytes = type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 ? 2 :
        type == GGML_TYPE_F32 ? 4 : type == GGML_TYPE_Q8_0 ? sizeof(block_q8_0) :
        type == GGML_TYPE_Q4_0 ? sizeof(block_q4_0) : type == GGML_TYPE_Q4_1 ? sizeof(block_q4_1) :
        type == GGML_TYPE_Q5_0 ? sizeof(block_q5_0) : sizeof(block_q5_1);
    static constexpr __host__ __device__ int row_bytes(int pairs) { return 2*pairs/block*bytes; }
};
