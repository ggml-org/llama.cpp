#pragma once

#include "fattn-tile-span-access.cuh"
#include "fattn-tile-resume.h"

template<ggml_type key, ggml_type value>
struct ggml_cuda_fattn_tile_resume_kv : ggml_cuda_fattn_tile_span_kv<key,value> {
    using base = ggml_cuda_fattn_tile_span_kv<key,value>;
    __device__ ggml_cuda_fattn_tile_resume_kv(const ggml_cuda_fattn_tile_span_table * table,int head) : base{table,head} {}

    // The descriptor's leading table is pointer-interconvertible with the descriptor itself.
    __device__ __forceinline__ const ggml_cuda_fattn_tile_resume_control & control() const {
        return reinterpret_cast<const ggml_cuda_fattn_tile_resume_descriptor *>(this->table)->control;
    }

    // Copy native representations without reducing lanes or converting half accumulators to float.
    template<bool load, int cpw, typename Acc, int count>
    __device__ __forceinline__ void checkpoint(float (&maximum)[cpw],float (&sum)[cpw],Acc (&values)[count]) const {
        static_assert(count%cpw == 0,"invalid accumulator count");
        constexpr size_t query_bytes = 8+(count/cpw)*sizeof(Acc), thread_bytes = cpw*query_bytes;
        const size_t block = (size_t(blockIdx.z)*gridDim.x+blockIdx.x)*gridDim.y+blockIdx.y;
        const size_t thread = block*(blockDim.x*blockDim.y)+threadIdx.y*blockDim.x+threadIdx.x;
        char * saved = control().state+thread*thread_bytes;
#pragma unroll
        for (int q = 0; q < cpw; ++q) {
            char * row = saved+q*query_bytes;
            if constexpr (load) {
                ggml_cuda_memcpy_1<4>(&maximum[q],row);
                ggml_cuda_memcpy_1<4>(&sum[q],row+4);
            } else {
                ggml_cuda_memcpy_1<4>(row,&maximum[q]);
                ggml_cuda_memcpy_1<4>(row+4,&sum[q]);
            }
#pragma unroll
            for (int i = 0; i < count/cpw; ++i) {
                auto & accumulator = values[q*(count/cpw)+i];
                if constexpr (load) ggml_cuda_memcpy_1<sizeof(Acc)>(&accumulator,row+8+i*sizeof(Acc));
                else ggml_cuda_memcpy_1<sizeof(Acc)>(row+8+i*sizeof(Acc),&accumulator);
            }
        }
    }

    // First-wave initialization must define every lane, including splits with no tile in that wave.
    template<typename Acc, int count>
    __device__ __forceinline__ void initialize(Acc (&values)[count]) const {
#pragma unroll
        for (int i = 0; i < count; ++i) {
            if constexpr (std::is_same<Acc,half2>::value) values[i] = make_half2(0.0f,0.0f);
            else values[i] = make_float2(0.0f,0.0f);
        }
    }
};

template<ggml_type key, ggml_type value>
struct ggml_cuda_fattn_tile_resume_traits<ggml_cuda_fattn_tile_resume_kv<key,value>> {
    static constexpr bool enabled = true;
};
