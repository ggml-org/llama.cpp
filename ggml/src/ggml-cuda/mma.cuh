// This file contains primitives that expose the tensor core PTX instructions for CUDA code.
// The primitives can be used in a similar way as the nvcuda::wmma interface but with a well-defined memory layout.
// The documentation for the PTX instructions can be found under:
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-multiply-accumulate-operation-using-mma-instruction
//
// Like with nvcuda::wmma there are three types of matrix tiles: A, B, and C with A @ B = C.
// A is a row-major matrix with shape I x K.
// B is a column-major matrix with shape K x J.
// C is a column-major matrix with shape I x J.
// Note that along their lowest dimension I, J, and K are measured in physical 32 bit elements instead of logical elements.
// The functions get_i, get_j, and get_k can be used to get the physical 32 bit index of the lth element of a thread within a tile.
// All matrix tiles have ne physical 32 bit elements per warp.
//
// As described in the documentation, all pointers for load_ldmatrix must be to shared memory and aligned to 16 bytes.

#include "common.cuh"


#if CUDART_VERSION >= 11800

static __device__ __forceinline__ int ggml_cuda_movmatrix(const int x) {
    int ret = 0;

#ifdef NEW_MMA_AVAILABLE
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
        : "+r"(ret) : "r"(x));
#else
    NO_DEVICE_CODE;
#endif // defined(NEW_MMA_AVAILABLE)
    return ret;
}

#else

static __device__ __forceinline__ int ggml_cuda_movmatrix(const int x) {
    // Imagine transposing row-major matrix to column-major matrix.
    const int src_i_low  = 2 * (threadIdx.x % 4);
    const int src_i_high = src_i_low + 1;
    const int src_j      = threadIdx.x / 4;

    const int src_laneid_low  = src_i_low  * 4 + src_j / 2;
    const int src_laneid_high = src_i_high * 4 + src_j / 2;

    const int shift_low  = ((src_j + 0) % 2) * 16;
    const int shift_high = ((src_j + 1) % 2) * 16;

    const int ret_low  = (__shfl_sync(0xFFFFFFFF, x, src_laneid_low,  WARP_SIZE) >> shift_low)  & 0x0000FFFF;
    const int ret_high = (__shfl_sync(0xFFFFFFFF, x, src_laneid_high, WARP_SIZE) << shift_high) & 0xFFFF0000;

    return ret_low | ret_high;
}

#endif // CUDART_VERSION >= 11800


template <typename T>
struct mma_A_I16K4 {
    static_assert(sizeof(T) == 4, "bad type size");

    static constexpr int I  = 16;
    static constexpr int K  = 4;
    static constexpr int ne = 2;

    T x[ne];

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = (l%2) * (I/2) + threadIdx.x / K;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  I);
        return ret;
    }

    static __device__ __forceinline__ int get_k(const int /* l */) {
        const int ret = threadIdx.x % K;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  K);
        return ret;
    }

    __device__ __forceinline__ void load_generic(const T * __restrict__ xs0, const int & stride) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_i(l)*stride + get_k(l)];
        }
    }

    __device__ __forceinline__ void load_ldmatrix(const T * __restrict__ xs0, const int & stride) {
#ifdef NEW_MMA_AVAILABLE
        int * xi = (int *) x;
        const int * xs = (const int *) xs0 + (threadIdx.x%I)*stride;
        asm("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "+r"(xi[0]), "+r"(xi[1])
            : "l"(xs));
#else
        load_generic(xs0, stride);
#endif // NEW_MMA_AVAILABLE
    }
};

template <typename T>
struct mma_A_I16K8 {
    static_assert(sizeof(T) == 4, "bad type size");

    static constexpr int I  = 16;
    static constexpr int K  = 8;
    static constexpr int ne = 4;

    T x[ne];

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = (l%2) * (I/2) + threadIdx.x / (K/2);
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  I);
        return ret;
    }

    static __device__ __forceinline__ int get_k(const int l) {
        const int ret = (l/2) * (K/2) + threadIdx.x % (K/2);
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  K);
        return ret;
    }

    __device__ __forceinline__ void load_generic(const T * __restrict__ xs0, const int & stride) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_i(l)*stride + get_k(l)];
        }
    }

    __device__ __forceinline__ void load_ldmatrix(const T * __restrict__ xs0, const int & stride) {
#ifdef NEW_MMA_AVAILABLE
        int * xi = (int * ) x;
        const int * xs = (const int *) xs0 + (threadIdx.x%I)*stride + (threadIdx.x/I)*(K/2);
        asm("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "+r"(xi[0]), "+r"(xi[1]), "+r"(xi[2]), "+r"(xi[3])
            : "l"(xs));
#else
        GGML_UNUSED(xs0);
        GGML_UNUSED(stride);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }

    __device__ __forceinline__ void load_ldmatrix_trans(const T * __restrict__ xs0, const int & stride) {
#ifdef NEW_MMA_AVAILABLE
        int * xi = (int * ) x;
        const int * xs = (const int *) xs0 + (threadIdx.x%I)*stride + (threadIdx.x/I)*(K/2);
        asm("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
            : "+r"(xi[0]), "+r"(xi[2]), "+r"(xi[1]), "+r"(xi[3])
            : "l"(xs));
#else
        GGML_UNUSED(xs0);
        GGML_UNUSED(stride);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }

    __device__ __forceinline__ void transpose() {
        int * xi  = (int *) x;
        xi[0] = ggml_cuda_movmatrix(xi[0]);

        const int tmp = ggml_cuda_movmatrix(xi[1]);
        xi[1] = ggml_cuda_movmatrix(xi[2]);
        xi[2] = tmp;

        xi[3] = ggml_cuda_movmatrix(xi[3]);
    }
};

template <typename T>
struct mma_B_J8K4 {
    static_assert(sizeof(T) == 4, "bad type size");

    static constexpr int J  = 8;
    static constexpr int K  = 4;
    static constexpr int ne = 1;

    T x[ne];

    static __device__ __forceinline__ int get_j(const int /* l */) {
        const int ret = threadIdx.x / K;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  J);
        return ret;
    }

    static __device__ __forceinline__ int get_k(const int /* l */) {
        const int ret = threadIdx.x % K;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  K);
        return ret;
    }

    __device__ __forceinline__ void load_generic(const T * __restrict__ xs0, const int & stride) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_j(l)*stride + get_k(l)];
        }
    }

    __device__ __forceinline__ void load_ldmatrix(const T * __restrict__ xs0, const int & stride) {
#ifdef NEW_MMA_AVAILABLE
        int * xi = (int *) x;
        const int * xs = (const int *) xs0 + (threadIdx.x%J)*stride;
        asm("ldmatrix.sync.aligned.m8n8.x1.b16 {%0}, [%1];"
            : "+r"(xi[0]) : "l"(xs));
#else
        load_generic(xs0, stride);
#endif // NEW_MMA_AVAILABLE
    }
};

template <typename T>
struct mma_B_J8K8 {
    static_assert(sizeof(T) == 4, "bad type size");

    static constexpr int J  = 8;
    static constexpr int K  = 8;
    static constexpr int ne = 2;

    T x[ne];

    static __device__ __forceinline__ int get_j(const int /* l */) {
        const int ret = threadIdx.x / (K/2);
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  J);
        return ret;
    }

    static __device__ __forceinline__ int get_k(const int l) {
        const int ret = l * (K/2) + threadIdx.x % (K/2);
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  K);
        return ret;
    }

    __device__ __forceinline__ void load_generic(const T * __restrict__ xs0, const int & stride) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_j(l)*stride + get_k(l)];
        }
    }

    __device__ __forceinline__ void load_ldmatrix(const T * __restrict__ xs0, const int & stride) {
#ifdef NEW_MMA_AVAILABLE
        int * xi = (int *) x;
        const int * xs = (const int *) xs0 + (threadIdx.x%J)*stride + ((threadIdx.x/J)*(K/2)) % K;
        asm("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "+r"(xi[0]), "+r"(xi[1])
            : "l"(xs));
#else
        load_generic(xs0, stride);
#endif // NEW_MMA_AVAILABLE
    }
};

template <typename T>
struct mma_C_I16J8 {};

template <>
struct mma_C_I16J8<int> {
    static constexpr int I  = 16;
    static constexpr int J  = 8;
    static constexpr int ne = 4;

    int x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = (l/2) * (I/2) + threadIdx.x / (J/2);
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  I);
        return ret;
    }

    static __device__ __forceinline__ int get_j(const int l) {
        const int ret = 2 * (threadIdx.x % (J/2)) + l%2;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  J);
        return ret;
    }

    __device__ __forceinline__ void mma(const mma_A_I16K4<int> & mma_A, const mma_B_J8K4<int> & mma_B) {
#ifdef NEW_MMA_AVAILABLE
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(x[0]), "+r"(x[1]), "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[0]), "r"(mma_A.x[1]), "r"(mma_B.x[0]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k16 mma instead:
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[0]), "+r"(x[1])
            : "r"(mma_A.x[0]), "r"(mma_B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[1]), "r"(mma_B.x[0]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        GGML_UNUSED(mma_A);
        GGML_UNUSED(mma_B);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }

    __device__ __forceinline__ void mma(const mma_A_I16K8<int> & mma_A, const mma_B_J8K8<int> & mma_B) {
#ifdef NEW_MMA_AVAILABLE
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(x[0]), "+r"(x[1]), "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[0]), "r"(mma_A.x[1]), "r"(mma_A.x[2]), "r"(mma_A.x[3]), "r"(mma_B.x[0]), "r"(mma_B.x[1]));
#else
        // On Turing m16n8k32 mma is not available, use 4x m8n8k16 mma instead:
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[0]), "+r"(x[1])
            : "r"(mma_A.x[0]), "r"(mma_B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[1]), "r"(mma_B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[0]), "+r"(x[1])
            : "r"(mma_A.x[2]), "r"(mma_B.x[1]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[3]), "r"(mma_B.x[1]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        GGML_UNUSED(mma_A);
        GGML_UNUSED(mma_B);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }
};

template <>
struct mma_C_I16J8<half2> {
    static constexpr int I  = 16;
    static constexpr int J  = 4;
    static constexpr int ne = 2;

    half2 x[ne] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = l * (I/2) + threadIdx.x / J;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  I);
        return ret;
    }

    static __device__ __forceinline__ int get_j(const int /* l */) {
        const int ret = threadIdx.x % J;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  J);
        return ret;
    }

    __device__ __forceinline__ void mma(const mma_A_I16K8<half2> & mma_A, const mma_B_J8K8<half2> & mma_B) {
#ifdef NEW_MMA_AVAILABLE
        int * Axi = (int *) mma_A.x;
        int * Bxi = (int *) mma_B.x;
        int * xi  = (int *) x;
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(xi[0]), "+r"(xi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(xi[0]), "+r"(xi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(xi[0]), "+r"(xi[1])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        GGML_UNUSED(mma_A);
        GGML_UNUSED(mma_B);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }

    __device__ __forceinline__ mma_B_J8K8<half2> to_mma_B() {
        mma_B_J8K8<half2> mma_B;

        int * xi   = (int *) x;
        int * Bxi  = (int *) mma_B.x;
        Bxi[0] = ggml_cuda_movmatrix(xi[0]);
        Bxi[1] = ggml_cuda_movmatrix(xi[1]);

        return mma_B;
    }
};

template <>
struct mma_C_I16J8<float> {
    static constexpr int I  = 16;
    static constexpr int J  = 8;
    static constexpr int ne = 4;

    float x[ne] = {0.0f, 0.0f, 0.0f, 0.0f};

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = (l/2) * (I/2) + threadIdx.x / (J/2);
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  I);
        return ret;
    }

    static __device__ __forceinline__ int get_j(const int l) {
        const int ret = 2 * (threadIdx.x % (J/2)) + l%2;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  J);
        return ret;
    }

    __device__ __forceinline__ void mma(const mma_A_I16K8<half2> & mma_A, const mma_B_J8K8<half2> & mma_B) {
#ifdef NEW_MMA_AVAILABLE
        int * Axi = (int *) mma_A.x;
        int * Bxi = (int *) mma_B.x;
        int * xi  = (int *) x;
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(xi[0]), "+r"(xi[1]), "+r"(xi[2]), "+r"(xi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(xi[0]), "+r"(xi[1]), "+r"(xi[2]), "+r"(xi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(xi[0]), "+r"(xi[1]), "+r"(xi[2]), "+r"(xi[3])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        GGML_UNUSED(mma_A);
        GGML_UNUSED(mma_B);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }

    __device__ __forceinline__ mma_B_J8K8<half2> to_mma_B() {
        mma_B_J8K8<half2> mma_B;
        mma_B.x[0] = make_half2(x[0], x[1]);
        mma_B.x[1] = make_half2(x[2], x[3]);

        int * Bxi  = (int *) mma_B.x;
        Bxi[0] = ggml_cuda_movmatrix(Bxi[0]);
        Bxi[1] = ggml_cuda_movmatrix(Bxi[1]);

        return mma_B;
    }

    __device__ __forceinline__ void load_generic(const float * __restrict__ xs0, const int & stride) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_j(l)*stride + get_i(l)];
        }
    }
};
