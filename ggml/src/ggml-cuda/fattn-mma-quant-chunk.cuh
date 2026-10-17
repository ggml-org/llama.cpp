#pragma once

#include "fattn-mma-quant.cuh"

// Decode eight values with stock strided rounding; only aligned groups share one packed block.
template<ggml_type type>
static __device__ __forceinline__ void ggml_cuda_fattn_decode_strided_chunk8(
        const char * row, int first, half2 (&output)[4]) {
    if constexpr (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0) {
        if (row && first%8 == 0) {
            constexpr int bytes = ggml_cuda_fattn_encoded_traits<type>::bytes;
            const char * block = row+first/32*bytes;
            const float scale = __half2float(*reinterpret_cast<const half *>(block));
            const int offset = first%32;
            uint16_t packed[4];
            // Encoded blocks are only two-byte aligned: do not use an unaligned 64-bit load.
            ggml_cuda_memcpy_1<8,2>(packed,block+2+(type == GGML_TYPE_Q8_0 ? offset : offset%16));
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                float2 values;
                if constexpr (type == GGML_TYPE_Q8_0) {
                    values.x = float(int8_t(packed[pair]&255))*scale;
                    values.y = float(int8_t(packed[pair]>>8))*scale;
                } else {
                    const int shift = offset < 16 ? 0 : 4;
                    values.x = (float((packed[pair]>>shift)&15)-8.0f)*scale;
                    values.y = (float((packed[pair]>>(8+shift))&15)-8.0f)*scale;
                }
                output[pair] = __float22half2_rn(values);
            }
            return;
        }
    }
#pragma unroll
    for (int pair = 0; pair < 4; ++pair)
        output[pair] = ggml_cuda_fattn_decode_pair<type>(row,first+2*pair,false);
}
