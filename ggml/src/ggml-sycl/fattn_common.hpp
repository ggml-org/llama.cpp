#ifndef GGML_SYCL_FATTN_COMMON_HPP
#define GGML_SYCL_FATTN_COMMON_HPP

template<int N>
inline void ggml_sycl_memcpy(float* dst, const float* src) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        dst[i] = src[i];
    }
}

#endif // GGML_SYCL_FATTN_COMMON_HPP
