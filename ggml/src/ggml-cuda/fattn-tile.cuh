#include "common.cuh"
#include "fattn-common.cuh"

// nbatch_fa == number of KQ rows to process per iteration
// nbatch_K == number of K columns to load in parallel for KQ calculation

// TODO optimize kernel parameters for FP16 NVIDIA (P100)
// TODO optimize kernel parameters for head sizes 40, 72, 80, 96, 112

// The ROCm compiler cannot handle templating in __launch_bounds__.
// As a workaround, define a macro to package the kernel parameters as uint32_t:
#define GGML_CUDA_FATTN_TILE_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                          \
        static_assert((nthreads)          <= 512, "bad nthreads");                                    \
        static_assert((occupancy)         <=   8, "bad occupancy");                                   \
        static_assert((nbatch_fa)         <= 256, "bad nbatch_fa");                                   \
        static_assert((nbatch_K)          <= 256, "bad nbatch_K");                                    \
        return ((nthreads) << 0) | ((occupancy) << 10) | ((nbatch_fa) << 14) | ((nbatch_K) << 23);    \
    }                                                                                                 \

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_nvidia_fp16(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  2, 128, 3, 128,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2, 128,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 128, 3, 128,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  64,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  2,  64, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 256, 3, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  2, 128, 2,  64,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  4, 128, 3, 128,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  8, 256, 2,  64,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 16, 256, 2,  64,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 32, 256, 2,  64,  72)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  2, 128, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  64,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2, 128,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  2,  64, 2,  64,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  64,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  64,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 2,  64,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 2, 128,  24)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  2, 128, 3,  64,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  64,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 3,  64, 112)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  32, 112)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2, 128,  56)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  2,  64, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  2,  64, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  2,  64, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(320, 256, 16, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  2,  64, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2,  64,  64)

    return 0;
}

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_nvidia_fp32(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  2, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  4, 128, 3,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 128, 3,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  2,  64, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  4, 128, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  8, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 16, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 32, 256, 2,  32,  72)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  2,  64, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 2,  32,  48)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  2,  64, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2,  32,  56)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  2, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 3,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  8, 128, 3,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 128, 3,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  2, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  4, 128, 3,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  8, 256, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 16, 256, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 32, 256, 2,  32,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  2, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  4, 128, 3,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  8, 256, 2,  32, 256)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2,  32,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(320, 256, 16, 256, 2,  32,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  2,  64, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  4, 128, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  8, 256, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 2,  32,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  4, 128, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  8, 256, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2,  32,  64)

    return 0;
}

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_amd(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 64, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  2,  64, 3,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  4, 128, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 128, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 256, 2, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 64, 256, 2,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  2,  64, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  4, 128, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  8, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 16, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 32, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 64, 256, 2,  32,  72)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 64, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  2,  64, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 64, 256, 2,  32,  48)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  2,  64, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 64, 256, 2,  32,  56)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  2, 256, 2, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 2,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  8, 256, 2,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 256, 2,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 64, 256, 2,  64,  32)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  2, 256, 2, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  4, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 16, 256, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 32, 256, 2,  32,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  2, 256, 2, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  4, 256, 2,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  8, 256, 2,  64, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2,  32, 128)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(320, 256, 32, 512, 1, 128,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  2,  64, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 32, 512, 1, 128,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 32, 512, 1, 128,  64)

    return 0;
}

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_amd_rdna(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 32, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 40,  40, 64, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  2,  64, 8,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  4,  64, 8,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  8, 128, 5, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 16, 128, 5, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 32, 128, 4,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64, 64, 128, 5,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  2,  64, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  4, 128, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72,  8, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 16, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 32, 256, 2,  32,  72)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 72,  72, 64, 256, 2,  32,  72)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  2,  64, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  4, 128, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80,  8, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 16, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 32, 256, 2,  32,  40)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 80,  80, 64, 256, 2,  32,  40)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  2,  64, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  4, 128, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96,  8, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 16, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 32, 256, 2,  32,  48)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 96,  96, 64, 256, 2,  32,  48)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  2,  64, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  4, 128, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112,  8, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2,  32,  56)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 64, 256, 2,  32,  56)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  2,  64, 8,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 8,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  8, 128, 8,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 256, 3, 128, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 3, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 64, 256, 3,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  2,  64, 8,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  4, 128, 6,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128,  8, 128, 6,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 16, 256, 5,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 32, 256, 3,  64,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  2,  64, 8,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  4, 128, 6,  32, 256)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  8, 128, 6,  32, 256)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 5,  32, 256)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 3,  64, 128)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(320, 256, 32, 256, 2, 128,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  2,  64, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 4,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 32, 256, 2, 128,  64)

    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  4, 128, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512,  8, 256, 2,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 4,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 32, 256, 2, 128,  64)

    return 0;
}

static __host__ uint32_t ggml_cuda_fattn_tile_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA(cc)) {
            return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
        }
        return ggml_cuda_fattn_tile_get_config_amd(DKQ, DV, ncols);
    }
    if (fast_fp16_available(cc)) {
        return ggml_cuda_fattn_tile_get_config_nvidia_fp16(DKQ, DV, ncols);
    }
    return ggml_cuda_fattn_tile_get_config_nvidia_fp32(DKQ, DV, ncols);
}

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config(const int DKQ, const int DV, const int ncols) {
#ifdef GGML_USE_HIP
#ifdef RDNA
    return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
#else
    return ggml_cuda_fattn_tile_get_config_amd(DKQ, DV, ncols);
#endif // RDNA
#else
#ifdef FAST_FP16_AVAILABLE
    return ggml_cuda_fattn_tile_get_config_nvidia_fp16(DKQ, DV, ncols);
#else
    return ggml_cuda_fattn_tile_get_config_nvidia_fp32(DKQ, DV, ncols);
#endif // FAST_FP16_AVAILABLE
#endif // GGML_USE_HIP
}

// Allowlist for native (quantized or bf16) K/V: rows timed on gfx1201 (Radeon AI PRO R9700).
// A shape without a row converts K and V to f16 and uses the f16 rows above.
static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_quant_gfx1201(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  2, 256, 6, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE( 64,  64,  4, 128, 6, 128,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  2,  64, 6,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  4, 128, 6,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128,  8, 128, 6,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 256, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 3,  64,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  2,  64, 6,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  4, 128, 6,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256,  8, 128, 6,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 5,  32, 128)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2,  64,  64)

    return 0;
}

static __host__ uint32_t ggml_cuda_fattn_tile_get_config_quant(const int DKQ, const int DV, const int ncols, const int cc) {
    if (cc == GGML_CUDA_CC_OFFSET_AMD + 0x1201) { // gfx1201
        return ggml_cuda_fattn_tile_get_config_quant_gfx1201(DKQ, DV, ncols);
    }
    return 0;
}

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_quant(const int DKQ, const int DV, const int ncols) {
#if defined(GGML_USE_HIP) && defined(__gfx1201__)
    return ggml_cuda_fattn_tile_get_config_quant_gfx1201(DKQ, DV, ncols);
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols);
    return 0;
#endif // defined(GGML_USE_HIP) && defined(__gfx1201__)
}

// Block shapes (ncols1 Q columns x ncols2 Q heads) that read K/V natively, used by the host gate and by the kernel.
// A (2, 2) block only pays off at head size 128 or with bf16 K/V.
static constexpr __host__ __device__ bool ggml_cuda_fattn_tile_native_block(
        const int DKQ, const int ncols1, const int ncols2, const ggml_type type_K, const ggml_type type_V) {
    if (ncols1 == 2 && ncols2 == 2 && DKQ != 128 && !(type_K == GGML_TYPE_BF16 && type_V == GGML_TYPE_BF16)) {
        return false;
    }
    return ncols2 >= 2 && (ncols2 == 8 || ncols1*ncols2 <= 16);
}

static __host__ int ggml_cuda_fattn_tile_get_nthreads(const int DKQ, const int DV, const int ncols, const int cc, const bool is_quant = false) {
    const uint32_t config = is_quant ? ggml_cuda_fattn_tile_get_config_quant(DKQ, DV, ncols, cc) : ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols, cc);
    return (config >> 0) & ((1 << 10) - 1);
}

static __host__ int ggml_cuda_fattn_tile_get_occupancy(const int DKQ, const int DV, const int ncols, const int cc, const bool is_quant = false) {
    const uint32_t config = is_quant ? ggml_cuda_fattn_tile_get_config_quant(DKQ, DV, ncols, cc) : ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols, cc);
    return (config >> 10) & ((1 << 4) - 1);
}

static __host__ int ggml_cuda_fattn_tile_get_nbatch_fa(const int DKQ, const int DV, const int ncols, const int cc, const bool is_quant = false) {
    const uint32_t config = is_quant ? ggml_cuda_fattn_tile_get_config_quant(DKQ, DV, ncols, cc) : ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols, cc);
    return (config >> 14) & ((1 << 9) - 1);
}

static __host__ int ggml_cuda_fattn_tile_get_nbatch_K(const int DKQ, const int DV, const int ncols, const int cc, const bool is_quant = false) {
    const uint32_t config = is_quant ? ggml_cuda_fattn_tile_get_config_quant(DKQ, DV, ncols, cc) : ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols, cc);
    return (config >> 23) & ((1 << 9) - 1);
}

// GQA heads per block that have a kernel for this head size, see launch_fattn_tile_switch_ncols2.
static constexpr bool ggml_cuda_fattn_tile_ncols2_instantiated(const int DKQ, const int DV, const int ncols2) {
    if (DKQ == 320) {
        // Only used for Mistral Small 4 which has a GQA ratio of 32.
        // On AMD, simply use that GQA ratio with 32 columns / block since we always have enough SRAM.
        // On NVIDIA however, the tile kernel is only used for GPUs that can't use the mma kernel (Pascal and older).
        // Therefore, use a GQA ratio of 16 with 16 columns / block to stay below 48 kiB of SRAM / block.
#ifdef GGML_USE_HIP
        return ncols2 == 32;
#else
        return ncols2 == 16;
#endif // GGML_USE_HIP
    }
    if (DKQ == 576) {
        return ncols2 == 16 || ncols2 == 4;
    }
    if (DKQ == 192) {
        // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
        return ncols2 == 16 || ncols2 == 8;
    }
    return ncols2 == 8 || ncols2 == 4 || ncols2 == 2 || (ncols2 == 1 && DV <= 256);
}

// GQA heads per block, 0 if there is no kernel. Without pad_gqa ncols2 must divide the GQA ratio.
// With pad_gqa (native K/V) it is the smallest ncols2 that holds the ratio, else the one that pads the fewest columns.
static int ggml_cuda_fattn_tile_get_ncols2(
        const int DKQ, const int DV, const int gqa_ratio, const bool use_gqa_opt, const bool pad_gqa) {
    if (!use_gqa_opt) {
        return ggml_cuda_fattn_tile_ncols2_instantiated(DKQ, DV, 1) ? 1 : 0;
    }
    if (pad_gqa) {
        int ncols2_pad = 0;
        for (int ncols2 = 32; ncols2 >= gqa_ratio && ncols2 >= 1; ncols2 /= 2) {
            if (ggml_cuda_fattn_tile_ncols2_instantiated(DKQ, DV, ncols2)) {
                ncols2_pad = ncols2;
            }
        }
        if (ncols2_pad != 0) {
            return ncols2_pad;
        }
        int padding_best = INT_MAX;
        for (int ncols2 = 32; ncols2 >= 2; ncols2 /= 2) {
            const int padding = (gqa_ratio + ncols2 - 1)/ncols2*ncols2 - gqa_ratio;
            if (ggml_cuda_fattn_tile_ncols2_instantiated(DKQ, DV, ncols2) && padding < padding_best) {
                padding_best = padding;
                ncols2_pad   = ncols2;
            }
        }
        return ncols2_pad;
    }
    for (int ncols2 = 32; ncols2 >= 1; ncols2 /= 2) {
        if (ggml_cuda_fattn_tile_ncols2_instantiated(DKQ, DV, ncols2) && gqa_ratio % ncols2 == 0) {
            return ncols2;
        }
    }
    return 0;
}

// Q columns per block (ncols1 * ncols2) for a head size, GQA heads per block and Q rows (ne01).
static int ggml_cuda_fattn_tile_get_cols_per_block(
        const int DKQ, const int ncols2, const int ncols2_eff, const int64_t ne01) {
#ifdef GGML_USE_HIP
    if (DKQ <= 128 && ne01 > 32/ncols2_eff) {
        return 64;
    }
    if (ne01 > 16/ncols2_eff) {
        return 32;
    }
#else
    if (DKQ <= 256 && ne01 > 16/ncols2_eff) {
        return 32;
    }
#endif // GGML_USE_HIP
    if (ncols2 <= 16 && ne01 > 8/ncols2_eff) {
        return 16;
    }
    if (ncols2 <= 8 && ne01 > 4/ncols2_eff) {
        return 8;
    }
    if (ncols2 <= 4 && ne01 > 2/ncols2_eff) {
        return 4;
    }
    return 2;
}

// Block shape of a tile launch: ncols2 Q heads of one K/V head times ncols1 = cols_per_block / ncols2 Q columns.
struct ggml_cuda_fattn_tile_shape {
    int ncols2;         // 0 if there is no kernel for the head size and GQA ratio of the op
    int ncols2_eff;     // Q heads actually present per block, < ncols2 only when the GQA ratio is padded (native K/V)
    int cols_per_block;
};

// Used by the host gate and by the launcher so that both form the same block shape.
static ggml_cuda_fattn_tile_shape ggml_cuda_fattn_tile_get_shape(const ggml_tensor * dst, const int cc, const bool pad_gqa) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On NVIDIA (Pascal and older) the GQA optimizations seem to be detrimental in some cases.
    // However, for DKQ == 576, DV == 512 only the kernel variant with GQA optimizations is implemented.
    const bool nvidia = GGML_CUDA_CC_IS_NVIDIA(cc);
    const int gqa_limit = nvidia && gqa_ratio <= 4 && V->ne[0] <= 256 ? 16 : INT_MAX;
    const bool use_gqa_opt = mask && max_bias == 0.0f && Q->ne[1] <= gqa_limit && K->ne[1] % FATTN_KQ_STRIDE == 0;

    ggml_cuda_fattn_tile_shape shape;
    shape.ncols2         = ggml_cuda_fattn_tile_get_ncols2(K->ne[0], V->ne[0], gqa_ratio, use_gqa_opt, pad_gqa);
    shape.ncols2_eff     = shape.ncols2 < gqa_ratio ? shape.ncols2 : gqa_ratio;
    shape.cols_per_block = shape.ncols2 == 0 ? 0 :
        ggml_cuda_fattn_tile_get_cols_per_block(K->ne[0], shape.ncols2, shape.ncols2_eff, Q->ne[1]);
    return shape;
}

typedef void (* fattn_tile_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

fattn_tile_case_t ggml_cuda_get_fattn_tile_case(
        const int64_t head_size_kq, const int64_t head_size_v,
        const ggml_type type_K_in, const ggml_type type_V_in);

// Whether the tile kernel reads K and V of this op as stored (quantized or bf16) instead of converting them to f16.
static inline bool ggml_cuda_fattn_tile_use_native_kv(const int cc, const ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];

    const ggml_type type_K = K->type == GGML_TYPE_F32 ? GGML_TYPE_F16 : K->type;
    const ggml_type type_V = V->type == GGML_TYPE_F32 ? GGML_TYPE_F16 : V->type;
    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
        return false;
    }

    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        return false; // Logit softcap is only instantiated for f16 K/V, see ggml_cuda_flash_attn_ext_tile_case.
    }

    const ggml_cuda_fattn_tile_shape shape = ggml_cuda_fattn_tile_get_shape(dst, cc, true);
    if (shape.ncols2 < 2) {
        return false;
    }
    // The native rows were timed for decode, where all Q columns of the op fit in one block. Wider ops convert.
    const int max_cols = shape.ncols2 == 8 ? 32 : 16;
    if (Q->ne[1] * shape.ncols2_eff > max_cols) {
        return false;
    }
    if (!ggml_cuda_fattn_tile_native_block(K->ne[0], shape.cols_per_block / shape.ncols2, shape.ncols2, type_K, type_V)) {
        return false;
    }
    if (ggml_cuda_fattn_tile_get_config_quant(K->ne[0], V->ne[0], shape.cols_per_block, cc) == 0) {
        return false;
    }

    return ggml_cuda_get_fattn_tile_case(K->ne[0], V->ne[0], type_K, type_V) != nullptr;
}

// TODO: deduplicate with mma-f16
template<int warp_size, int nwarps, int I, int J, int J_padding, bool oob_check,
    ggml_type type_KV = GGML_TYPE_F16, bool is_quant_kv = false, typename T>
static __device__ __forceinline__ void flash_attn_tile_load_tile(
        const half2 * const __restrict__ KV, T * const __restrict__ tile_KV,
        const int stride_KV, const int i_sup, const int j_offset = 0) {
    constexpr int cpy_nb   = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne   = cpy_nb / 4;
    constexpr int spt      = std::is_same_v<T, half2> ? 2 : 1;
    constexpr int JT       = J / spt;
    constexpr int n_unroll = std::is_same_v<T, half2> ? 7 : 5;
    // Values dequantized per call; must not exceed the cpy_ne*spt values that one copy of tmp holds (cpy_ne == 2 on Pascal).
    constexpr int ne_deq   = (std::is_same_v<T, half2> &&
        (type_KV == GGML_TYPE_F16 || type_KV == GGML_TYPE_BF16 || type_KV == GGML_TYPE_Q8_0)) ? 2*cpy_ne :
        (spt*cpy_ne < 4 ? spt*cpy_ne : 4);
    using T_scalar = std::conditional_t<std::is_same_v<T, half2>, half, float>;
    constexpr dequantize_V_t deq = get_dequantize_V<type_KV, T_scalar, ne_deq>();

    auto load = [&] __device__ (const int n) {
        const int stride_j = warp_size >> n;

        if (stride_j == 0) {
            return;
        }

        const int j0_start = stride_j == warp_size ? 0 : (JT/cpy_ne) - (JT/cpy_ne) % (2*stride_j);
        const int j0_stop  =                             (JT/cpy_ne) - (JT/cpy_ne) % (1*stride_j);
        const int stride_i = warp_size / stride_j;

        if (j0_start == j0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_j == warp_size ? 0 : threadIdx.x / stride_j);

            if (i0 + nwarps*stride_i <= I || i < I) {
#pragma unroll
                for (int j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    const bool in_bounds = !oob_check || i < i_sup;

                    if constexpr (is_quant_kv) {
                        // j in units of T, one copy of cpy_ne values per thread, dequantized from the stored row.
                        const int j = j0*cpy_ne + (stride_j == warp_size ? threadIdx.x : threadIdx.x % stride_j)*cpy_ne;
                        const char * row_ptr = (const char *) KV + int64_t(i)*stride_KV;

                        __align__(16) T tmp[cpy_ne] = {};
                        if (in_bounds) {
#pragma unroll
                            for (int l = 0; l < cpy_ne; l += ne_deq/spt) {
                                deq(row_ptr, &tmp[l], j_offset + spt*(j + l));
                            }
                        }
                        ggml_cuda_memcpy_1<cpy_nb>(tile_KV + i*(JT + J_padding) + j, tmp);
                    } else if constexpr (std::is_same_v<T, half2>) {
                        const int j = j0*cpy_ne + (stride_j == warp_size ? threadIdx.x : threadIdx.x % stride_j)*cpy_ne;

                        const __align__(16) half2 zero[cpy_ne] = {{0.0f, 0.0f}};
                        ggml_cuda_memcpy_1<cpy_nb>(
                            tile_KV + i*(J/2 + J_padding) + j,
                            in_bounds ? KV + i*stride_KV + j : zero);
                    } else {
                        const int j = j0*(cpy_ne/2) + (stride_j == warp_size ? threadIdx.x : threadIdx.x % stride_j)*(cpy_ne/2);

                        const half2 zero[cpy_ne/2] = {{0.0f, 0.0f}};
                        __align__(16) half2 tmp_h2[cpy_ne/2];
                        ggml_cuda_memcpy_1<sizeof(tmp_h2)>(
                            tmp_h2, in_bounds ? KV + i*stride_KV + j : zero);

                        __align__(16) float2 tmp_f2[cpy_ne/2];
#pragma unroll
                        for (int l = 0; l < cpy_ne/2; ++l) {
                            tmp_f2[l] = __half22float2(tmp_h2[l]);
                        }
                        ggml_cuda_memcpy_1<sizeof(tmp_f2)>(tile_KV + i*(J + J_padding) + 2*j, tmp_f2);
                    }
                }
            }
        }
    };
    static_assert(J % 8 == 0, "bad J");
    static_assert(JT % cpy_ne == 0, "bad J");
    ggml_cuda_unroll<n_unroll>{}(load);
}

// Function that performs a single iteration in for the KQ matrix multiplication:
template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int nbatch_fa, int nbatch_K,
    bool use_logit_softcap, bool oob_check,
    ggml_type type_K = GGML_TYPE_F16, ggml_type type_V = GGML_TYPE_F16, typename T_vec_dot>
static __device__ __forceinline__ void flash_attn_tile_iter_KQ(
        T_vec_dot   * const Q_tmp,
        const half2 * const __restrict__ K_h2,
        T_vec_dot   * const KV_tmp,
        const int stride_K2,
        const int k_VKQ_0,
        const int k_VKQ_sup,
        const int k_KQ_0,
        float * KQ_acc,
        const bool warp_valid = true) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column
    constexpr bool is_quant_kv = type_K != GGML_TYPE_F16 || type_V != GGML_TYPE_F16;

    flash_attn_tile_load_tile<warp_size, nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check, type_K, is_quant_kv>(
        !is_quant_kv ? K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2 : (const half2 *)((const char *) K_h2 + int64_t(k_VKQ_0)*stride_K2),
        KV_tmp, stride_K2, k_VKQ_sup, k_KQ_0);
    __syncthreads();

    if constexpr (is_quant_kv) {
        if (!warp_valid) {
            if (k_KQ_0 + nbatch_K < DKQ) {
                __syncthreads();
            }
            return;
        }
    }

#ifdef FAST_FP16_AVAILABLE
    static_assert((nbatch_K/2) % cpy_ne == 0, "bad nbatch_K");
#pragma unroll
    for (int k_KQ_1 = 0; k_KQ_1 < nbatch_K/2; k_KQ_1 += cpy_ne) {
        __align__(16) half2 K_k[nbatch_fa/(np*warp_size)][cpy_ne];
        __align__(16) half2 Q_k[cpw][cpy_ne];
#else
    static_assert(nbatch_K % cpy_ne == 0, "bad nbatch_K");
#pragma unroll
    for (int k_KQ_1 = 0; k_KQ_1 < nbatch_K; k_KQ_1 += cpy_ne) {
        __align__(16) float K_k[nbatch_fa/(np*warp_size)][cpy_ne];
        __align__(16) float Q_k[cpw][cpy_ne];
#endif // FAST_FP16_AVAILABLE

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
            const int i_KQ = i_KQ_0 + (threadIdx.y % np)*warp_size + threadIdx.x;

#ifdef FAST_FP16_AVAILABLE
            ggml_cuda_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_size)], &KV_tmp[i_KQ*(nbatch_K/2 + cpy_ne) + k_KQ_1]);
#else
            ggml_cuda_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_size)], &KV_tmp[i_KQ*(nbatch_K   + cpy_ne) + k_KQ_1]);
#endif // FAST_FP16_AVAILABLE
        }
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int jc = jc0 + (threadIdx.y / np)*cpw;

#ifdef FAST_FP16_AVAILABLE
            ggml_cuda_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc*(DKQ/2) + k_KQ_0/2 + k_KQ_1]);
#else
            ggml_cuda_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc* DKQ    + k_KQ_0   + k_KQ_1]);
#endif // FAST_FP16_AVAILABLE
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
#pragma unroll
            for (int jc0 = 0; jc0 < cpw; ++jc0) {
#pragma unroll
                for (int k = 0; k < cpy_ne; ++k) {
                    ggml_cuda_mad(KQ_acc[i_KQ_0/(np*warp_size)*cpw + jc0], K_k[i_KQ_0/(np*warp_size)][k], Q_k[jc0][k]);
                }
            }
        }
    }

    if (k_KQ_0 + nbatch_K < DKQ) {
        __syncthreads(); // Sync not needed on last iteration.
    }
}

// Function that performs a single iteration of the main loop over up to nbatch_fa tokens.
template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int DV, int nbatch_fa, int nbatch_K,
    bool use_logit_softcap, bool oob_check,
    ggml_type type_K = GGML_TYPE_F16, ggml_type type_V = GGML_TYPE_F16,
    typename T_vec_dot, typename T_KQ, typename T_acc>
static __device__ __forceinline__ void flash_attn_tile_iter(
        T_vec_dot * const Q_tmp,
        const half2 * const __restrict__ K_h2,
        const half2 * const __restrict__ V_h2,
        const half  * const __restrict__ mask,
        const uint3 ne01,
        const float logit_softcap,
        const float slope,
        T_KQ      * const KQ,
        T_vec_dot * const KV_tmp,
        const int stride_K2,
        const int stride_V2,
        const int stride_mask,
        float * const KQ_max,
        float * const KQ_sum,
        T_acc * const VKQ,
        const int k_VKQ_0,
        const int k_VKQ_max,
        const int col_Q_0,
        const bool warp_valid = true,
        const int ncols2_eff = ncols2) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column
    constexpr bool is_quant_kv = type_K != GGML_TYPE_F16 || type_V != GGML_TYPE_F16;

    constexpr int DVp = (DV + 2*warp_size - 1) & ~(2*warp_size - 1); // DV padded to multiple of 2*warp_size.

    // KQ_cs == KQ chunk size, number of KQ values in j direction to store as one contiguous chunk in memory.
    // KQ is originally 2D but uses a Z-shaped 3D memory pattern like KQ[ncols/KQ_cs][DVp][KQ_cs].
#ifdef FAST_FP16_AVAILABLE
    constexpr int KQ_cs = cpw < 2*cpy_ne ? cpw : 2*cpy_ne;
#else
    constexpr int KQ_cs = cpw < 1*cpy_ne ? cpw : 1*cpy_ne;
#endif // FAST_FP16_AVAILABLE
    static_assert(cpw % KQ_cs == 0, "bad KQ_cs");
    const int k_VKQ_sup = k_VKQ_max - k_VKQ_0; // k supremum, only smaller k values have valid KV data

    float KQ_max_new[cpw];
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    float KQ_acc[nbatch_fa/(np*warp_size) * cpw] = {0.0f}; // Accumulators for KQ matrix multiplication.

    // KQ = K @ Q matrix multiplication:
    constexpr int nbatch_K_last = DKQ % nbatch_K;
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < DKQ - nbatch_K_last; k_KQ_0 += nbatch_K) {
        flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, use_logit_softcap, oob_check, type_K, type_V>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc, warp_valid);
    }
    if (nbatch_K_last > 0) {
        constexpr int k_KQ_0 = DKQ - nbatch_K_last;
        flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K_last, use_logit_softcap, oob_check, type_K, type_V>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc, warp_valid);
    }

    // Apply logit softcap + mask, update KQ_max:
    if (!is_quant_kv || warp_valid) {
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int j = fastmodulo(col_Q_0 + (jc0 + (threadIdx.y / np)*cpw)/(is_quant_kv ? ncols2_eff : ncols2), ne01);

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
                const int i_KQ = i_KQ_0 + (threadIdx.y % np)*warp_size + threadIdx.x;

#if defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)
                // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
                // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
                KQ_acc[i_KQ_0/(np*warp_size)*cpw + jc0] *= 4.0f;
#endif // defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)

                if (use_logit_softcap) {
                    KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] = logit_softcap * tanhf(KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0]);
                }

                if (!oob_check || i_KQ < k_VKQ_sup) {
                    KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] += (ncols2 > 1 || mask) ?
                        slope*__half2float(mask[j*stride_mask + k_VKQ_0 + i_KQ]) : 0.0f;

                    KQ_max_new[jc0] = fmaxf(KQ_max_new[jc0], KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] + FATTN_KQ_MAX_OFFSET);
                }
            }

            KQ_max_new[jc0] = warp_reduce_max<warp_size>(KQ_max_new[jc0]);
        }
    }

    if constexpr (np == 1) {
        __syncthreads();
    } else {
        static_assert(cpw == 1, "bad cpw");
        __shared__ float KQ_max_new_shared[nwarps];
        if (threadIdx.x == 0) {
            KQ_max_new_shared[threadIdx.y] = KQ_max_new[0];
        }
        __syncthreads();
        KQ_max_new[0] = KQ_max_new_shared[(threadIdx.y & ~(np-1)) + threadIdx.x % np];
        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
    }

    // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
    if (!is_quant_kv || warp_valid) {
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
#ifdef FAST_FP16_AVAILABLE
            __align__(16) half  tmp[nbatch_fa/(np*warp_size)][KQ_cs];
#else
            __align__(16) float tmp[nbatch_fa/(np*warp_size)][KQ_cs];
#endif // FAST_FP16_AVAILABLE

#pragma unroll
            for (int jc1 = 0; jc1 < KQ_cs; ++jc1) {
                const int jc = jc0 + jc1;

                const float KQ_max_scale = expf(KQ_max[jc] - KQ_max_new[jc]);
                KQ_max[jc] = KQ_max_new[jc];

                float KQ_sum_add = 0.0f;
#pragma unroll
                for (int i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
                    const float val = !oob_check || i0 + (threadIdx.y % np)*warp_size + threadIdx.x < static_cast<uint32_t>(k_VKQ_sup) ?
                        expf(KQ_acc[(i0/(np*warp_size))*cpw + jc] - KQ_max[jc]) : 0.0f;
                    KQ_sum_add += val;
                    tmp[i0/(np*warp_size)][jc1] = val;
                }
                KQ_sum[jc] = KQ_sum[jc]*KQ_max_scale + KQ_sum_add;

#ifdef FAST_FP16_AVAILABLE
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
                for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                    VKQ[jc*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
                }
#else
#pragma unroll
                for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                    VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].x *= KQ_max_scale;
                    VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].y *= KQ_max_scale;
                }
#endif // FAST_FP16_AVAILABLE
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
                const int i = i0 + (threadIdx.y % np)*warp_size + threadIdx.x;

                ggml_cuda_memcpy_1<sizeof(tmp[0])>(
                    KQ + (jc0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs))*(nbatch_fa*KQ_cs) + i*KQ_cs,
                    tmp[i0/(np*warp_size)]);
            }
        }
    }

    // VKQ = V @ KQ matrix multiplication:
    static_assert(DV <= DKQ, "bad DV");
    static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K*2/3) == 0), "bad nbatch_K");
    constexpr int nbatch_V = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K*2/3) * nbatch_fa / DV; // Number of V columns that fit in SRAM for K.
    static_assert(nbatch_fa % nbatch_V == 0, "bad nbatch_V");
    static_assert(nbatch_V % np == 0, "bad nbatch_V");
#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
        // For native (non-f16) K/V stride_V2 is a byte stride and the tile loader dequantizes the rows itself.
        flash_attn_tile_load_tile<warp_size, nwarps, nbatch_V, DV, 0, oob_check, type_V, is_quant_kv>(
            !is_quant_kv ? V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2 : (const half2 *)((const char *) V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2),
            KV_tmp, stride_V2, k_VKQ_sup - k0);
        __syncthreads();

        if (!is_quant_kv || warp_valid) {
#ifdef FAST_FP16_AVAILABLE
#pragma unroll
            for (int k1 = 0; k1 < nbatch_V; k1 += np) {
                __align__(16) half2 V_k[(DVp/2)/warp_size];
                __align__(16) half2 KQ_k[cpw];

                constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
                for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                    ggml_cuda_memcpy_1<cpy_ne_D*4>(&V_k[i0/warp_size], &KV_tmp[(k1 + threadIdx.y % np)*(DV/2) + i0 + threadIdx.x*cpy_ne_D]);
                }
#pragma unroll
                for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                    const int jc_KQ = jc_VKQ_0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs);

                    __align__(16) half tmp[KQ_cs];
                    ggml_cuda_memcpy_1<KQ_cs*sizeof(half)>(
                        &tmp, KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + threadIdx.y % np)*KQ_cs);
#pragma unroll
                    for (int jc_VKQ_1 = 0; jc_VKQ_1 < KQ_cs; ++jc_VKQ_1) {
                        KQ_k[jc_VKQ_0+jc_VKQ_1] = __half2half2(tmp[jc_VKQ_1]);
                    }
                }

#pragma unroll
                for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
#pragma unroll
                    for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                        VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size] += V_k[i0/warp_size]*KQ_k[jc_VKQ_0];
                    }
                }
            }
#else
#pragma unroll
            for (int k1 = 0; k1 < nbatch_V; k1 += np) {
                __align__(16) float2 V_k[(DVp/2)/warp_size];
                __align__(16) float  KQ_k[cpw];

                constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
                for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                    ggml_cuda_memcpy_1<cpy_ne_D*4>(&V_k[i0/(2*warp_size)], &KV_tmp[(k1 + threadIdx.y % np)*DV + i0 + threadIdx.x*cpy_ne_D]);
                }
#pragma unroll
                for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                    const int jc_KQ = jc_VKQ_0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs);

                    ggml_cuda_memcpy_1<KQ_cs*sizeof(float)>(
                        &KQ_k[jc_VKQ_0], KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + threadIdx.y % np)*KQ_cs);
                }

#pragma unroll
                for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
#pragma unroll
                    for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                        VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].x += V_k[i0/warp_size].x*KQ_k[jc_VKQ_0];
                        VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].y += V_k[i0/warp_size].y*KQ_k[jc_VKQ_0];
                    }
                }
            }
#endif // FAST_FP16_AVAILABLE
        }

        __syncthreads();
    }
}

// Whether this kernel instance reads K and V natively. A non-f16 instance that does not is compiled as an empty kernel.
static constexpr __host__ __device__ bool ggml_cuda_fattn_tile_kernel_is_quant(
        const int DKQ, const int DV, const int ncols1, const int ncols2, const ggml_type type_K, const ggml_type type_V) {
    return (type_K != GGML_TYPE_F16 || type_V != GGML_TYPE_F16) &&
        ggml_cuda_fattn_tile_native_block(DKQ, ncols1, ncols2, type_K, type_V) &&
        ggml_cuda_fattn_tile_get_config_quant(DKQ, DV, ncols1*ncols2) != 0;
}

// Kernel-side config accessors, usable in __launch_bounds__ in both compilation passes.
static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_kernel_config(
        const int DKQ, const int DV, const int ncols1, const int ncols2, const ggml_type type_K, const ggml_type type_V) {
    return ggml_cuda_fattn_tile_kernel_is_quant(DKQ, DV, ncols1, ncols2, type_K, type_V) ?
        ggml_cuda_fattn_tile_get_config_quant(DKQ, DV, ncols1*ncols2) : ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols1*ncols2);
}

static constexpr __host__ __device__ int ggml_cuda_fattn_tile_kernel_nthreads(
        const int DKQ, const int DV, const int ncols1, const int ncols2, const ggml_type type_K, const ggml_type type_V) {
    return (ggml_cuda_fattn_tile_kernel_config(DKQ, DV, ncols1, ncols2, type_K, type_V) >> 0) & ((1 << 10) - 1);
}

static constexpr __host__ __device__ int ggml_cuda_fattn_tile_kernel_occupancy(
        const int DKQ, const int DV, const int ncols1, const int ncols2, const ggml_type type_K, const ggml_type type_V) {
    return (ggml_cuda_fattn_tile_kernel_config(DKQ, DV, ncols1, ncols2, type_K, type_V) >> 10) & ((1 << 4) - 1);
}

static constexpr __host__ __device__ int ggml_cuda_fattn_tile_kernel_nbatch_fa(
        const int DKQ, const int DV, const int ncols1, const int ncols2, const ggml_type type_K, const ggml_type type_V) {
    return (ggml_cuda_fattn_tile_kernel_config(DKQ, DV, ncols1, ncols2, type_K, type_V) >> 14) & ((1 << 9) - 1);
}

static constexpr __host__ __device__ int ggml_cuda_fattn_tile_kernel_nbatch_K(
        const int DKQ, const int DV, const int ncols1, const int ncols2, const ggml_type type_K, const ggml_type type_V) {
    return (ggml_cuda_fattn_tile_kernel_config(DKQ, DV, ncols1, ncols2, type_K, type_V) >> 23) & ((1 << 9) - 1);
}

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap,
    ggml_type type_K = GGML_TYPE_F16, ggml_type type_V = GGML_TYPE_F16> // D == head size
__launch_bounds__(
    ggml_cuda_fattn_tile_kernel_nthreads (DKQ, DV, ncols1, ncols2, type_K, type_V),
    ggml_cuda_fattn_tile_kernel_occupancy(DKQ, DV, ncols1, ncols2, type_K, type_V))
static __global__ void flash_attn_tile(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        const int  * KV_rows_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;
    GGML_UNUSED(KV_rows_ptr);

    constexpr bool native_kv   = type_K != GGML_TYPE_F16 || type_V != GGML_TYPE_F16;
    constexpr bool is_quant_kv = ggml_cuda_fattn_tile_kernel_is_quant(DKQ, DV, ncols1, ncols2, type_K, type_V);
    constexpr ggml_type type_K_eff = is_quant_kv ? type_K : GGML_TYPE_F16;
    constexpr ggml_type type_V_eff = is_quant_kv ? type_V : GGML_TYPE_F16;

    // Skip unused kernel variants for faster compilation:
    if ((use_logit_softcap && !(DV == 128 || DV == 256 || DV == 512)) || (native_kv && !is_quant_kv)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    static_assert(ggml_cuda_fattn_tile_kernel_config(DKQ, DV, ncols1, ncols2, type_K, type_V) != 0, "kernel config not defined");

    constexpr int ncols     = ncols1*ncols2;
    constexpr int warp_size = 32;
    constexpr int nwarps    = ggml_cuda_fattn_tile_kernel_nthreads (DKQ, DV, ncols1, ncols2, type_K, type_V) / warp_size;
    constexpr int nbatch_fa = ggml_cuda_fattn_tile_kernel_nbatch_fa(DKQ, DV, ncols1, ncols2, type_K, type_V);
    constexpr int nbatch_K  = ggml_cuda_fattn_tile_kernel_nbatch_K (DKQ, DV, ncols1, ncols2, type_K, type_V);

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    // With native K/V ncols2 can exceed the GQA ratio (launch_fattn fold_gqa), gridDim.z is ceil(gqa_ratio/ncols2) * ne12 * ne03.
    const int ncols2_eff  = !is_quant_kv ? ncols2 : (ncols2 > (ne02/ne12) ? (ne02/ne12) : ncols2);
    const int ncols1_eff  = !is_quant_kv ? ncols1 : (ncols / ncols2_eff);
    const int ncols_valid = !is_quant_kv ? ncols  : (ncols1_eff * ncols2_eff);

    const int col_Q_0 = blockIdx.x * ncols1_eff; // Index of the first Q column for this CUDA block to work on.

    const int sequence = !is_quant_kv
        ? blockIdx.z / (ne02/ncols2)
        : blockIdx.z / (ne12 * ((ne02/ne12 + ncols2 - 1) / ncols2));
    const int head0 = !is_quant_kv
        ? blockIdx.z*ncols2 - sequence*ne02 // == blockIdx.z % (ne02/ncols2)
        : ((blockIdx.z - sequence * ne12 * ((ne02/ne12 + ncols2 - 1) / ncols2)) / ((ne02/ne12 + ncols2 - 1) / ncols2)) * (ne02/ne12)
            + (blockIdx.z % ((ne02/ne12 + ncols2 - 1) / ncols2)) * ncols2;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float * Q_f  = (const float *) (Q + nb03*sequence + nb02* head0);
    const half2 * K_h2 = (const half2 *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
    const half2 * V_h2 = (const half2 *) (V + nb23*sequence + nb22*(head0 / gqa_ratio)); // K and V have same shape

    const half * maskh = mask ? (const half *) (mask + nb33*(sequence % ne33)) : nullptr;

    // Native K/V rows are addressed in bytes because the row size depends on the quantization block.
    const int stride_K2   = is_quant_kv ? nb11 : int(nb11 / sizeof(half2));
    const int stride_V2   = is_quant_kv ? nb21 : int(nb21 / sizeof(half2));
    const int stride_mask = nb31 / sizeof(half);

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int cpw = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp.
    constexpr int np  = nwarps > ncols ? nwarps/ncols : 1; // Number of parallel warps per Q column.
    static_assert(cpw == 1 || np == 1, "bad cpw / np");
    static_assert(nbatch_fa % (np*warp_size) == 0, "nbatch_fa % (np*warp_size) != 0");

    constexpr int DKQp = (DKQ + 2*warp_size - 1) & ~(2*warp_size - 1); // DKQ padded to multiple of 2*warp_size.
    constexpr int DVp  = (DV  + 2*warp_size - 1) & ~(2*warp_size - 1); // DV  padded to multiple of 2*warp_size.

    // Q_tmp == SRAM buffer to hold Q data for the entire lifetime of the kernel.
    // KV_tmp == SRAM buffer to hold fragments of K/V data while iterating over ne11.
    //     KV_tmp is padded to avoid memory conflicts for K (cpy_ne) and OOB accesses for V (DVp-DV).
    // KQ == SRAM buffer to hold KQ fragments between KQ and VKQ matrix multiplications.
    // VKQ == Accumulators in registers for the final VKQ result.
#ifdef FAST_FP16_AVAILABLE
    __shared__ half2 Q_tmp[ncols * DKQ/2];
    __shared__ half2 KV_tmp[nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-DV];
    __shared__ half  KQ[ncols * nbatch_fa];
    __align__(16) half2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};
#else
    __shared__ float Q_tmp[ncols * DKQ];
    __shared__ float KV_tmp[nbatch_fa * (nbatch_K + cpy_ne) + DVp-DV];
    __shared__ float KQ[ncols * nbatch_fa];
    __align__(16) float2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};
#endif // FAST_FP16_AVAILABLE

    float KQ_max[cpw];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        KQ_max[j0/nwarps] = -FLT_MAX/2.0f;
    }
    float KQ_sum[cpw] = {0.0f};

    ggml_cuda_pdl_sync();

    // With folded GQA a block can have columns without a (query, head) pair. They load zero Q and are never stored.
    auto col_valid = [&] __device__ (const int jc) -> bool {
        if constexpr (!is_quant_kv) {
            return true;
        }
        return jc < ncols_valid && col_Q_0 + (jc / ncols2_eff) < int(ne01.z) &&
            (ncols2 == 1 || (head0 % gqa_ratio) + (jc % ncols2_eff) < gqa_ratio);
    };
    const int jc_warp_0 = (threadIdx.y / np) * cpw;
    const bool warp_valid = !is_quant_kv || (
        jc_warp_0 < ncols_valid &&
        col_Q_0 + (jc_warp_0 / ncols2_eff) < int(ne01.z) &&
        (cpw > ncols2_eff || (head0 % gqa_ratio) + (jc_warp_0 % ncols2_eff) < gqa_ratio));

    // Load Q data, convert to FP16 if fast:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + (threadIdx.y / np)*cpw;

        const int j = !is_quant_kv ? (jc / ncols2) : (jc / ncols2_eff);
        const int c = !is_quant_kv ? (jc % ncols2) : (jc % ncols2_eff);

        constexpr int cpy_ne_D = cpy_ne < DKQp/warp_size ? cpy_ne : DKQp/warp_size;

#pragma unroll
        for (int i0 = 0; i0 < DKQp; i0 += np*warp_size*cpy_ne_D) {
            if (i0 + np*warp_size*cpy_ne_D <= DKQ || i0 + (threadIdx.y % np)*(warp_size*cpy_ne_D) + threadIdx.x*cpy_ne_D < DKQ) {
                __align__(16) float tmp_f[cpy_ne_D] = {0.0f};
                if (col_valid(jc)) {
                    ggml_cuda_memcpy_1<sizeof(tmp_f)>
                        (tmp_f, &Q_f[c*(nb02/sizeof(float)) + fastmodulo(col_Q_0 + j, ne01)*(nb01/sizeof(float))
                                     + i0 + (threadIdx.y % np)*(warp_size*cpy_ne_D) + threadIdx.x*cpy_ne_D]);
                }

#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    tmp_f[i1] *= scale;
                }

#ifdef FAST_FP16_AVAILABLE
                __align__(16) half2 tmp_h2[cpy_ne_D/2];
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; i1 += 2) {
                    tmp_h2[i1/2] = make_half2(tmp_f[i1 + 0], tmp_f[i1 + 1]);
#if defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)
                    // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
                    // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
                    tmp_h2[i1/2] *= make_half2(0.25f, 0.25f);
#endif // defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)
                }
                ggml_cuda_memcpy_1<sizeof(tmp_h2)>(
                    &Q_tmp[jc*(DKQ/2) + i0/2 + (threadIdx.y % np)*(warp_size*cpy_ne_D/2) + threadIdx.x*(cpy_ne_D/2)],
                    tmp_h2);
#else
                ggml_cuda_memcpy_1<sizeof(tmp_f)>(
                    &Q_tmp[jc* DKQ    + i0   + (threadIdx.y % np)*(warp_size*cpy_ne_D)   + threadIdx.x* cpy_ne_D],
                    tmp_f);
#endif // FAST_FP16_AVAILABLE
            }
        }
    }

    __syncthreads();

    // Main loop over KV cache:
    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    if (ncols2 == 1) {
        // Branch with out-of-bounds checks.
        int k_VKQ_0 = blockIdx.y*nbatch_fa;
        while (k_VKQ_0 <= k_VKQ_max - nbatch_fa) {
            constexpr bool oob_check = false;
            flash_attn_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check, type_K_eff, type_V_eff>
                (Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp,
                stride_K2, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0, warp_valid, ncols2_eff);
            k_VKQ_0 += gridDim.y*nbatch_fa;
        }
        if (k_VKQ_0 < k_VKQ_max) {
            constexpr bool oob_check = true;
            flash_attn_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check, type_K_eff, type_V_eff>
                (Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp,
                stride_K2, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0, warp_valid, ncols2_eff);
        }
    } else {
        // Branch without out-of-bounds checks.
        for (int k_VKQ_0 = blockIdx.y*nbatch_fa; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nbatch_fa) {
            constexpr bool oob_check = false;
            flash_attn_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check, type_K_eff, type_V_eff>
                (Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp,
                stride_K2, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0, warp_valid, ncols2_eff);
        }
    }

#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_sum[jc0] = warp_reduce_sum<warp_size>(KQ_sum[jc0]);
    }

    if constexpr (np > 1) {
        static_assert(cpw == 1, "bad cpw");
        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, "KV_tmp too small");

#ifdef FAST_FP16_AVAILABLE
        half2 * VKQ_combine    = (half2 *) KV_tmp;
#else
        float * VKQ_combine    = (float *) KV_tmp;
#endif // FAST_FP16_AVAILABLE
        float * KQ_sum_combine = (float *) Q_tmp;

        if (threadIdx.y % np != 0) {
#ifdef FAST_FP16_AVAILABLE
            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(&VKQ_combine[threadIdx.y*(DVp/2) + i0 + threadIdx.x*cpy_ne_D], &VKQ[i0/warp_size]);
            }
#else
            constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(
                    &VKQ_combine[threadIdx.y*DVp + i0 + threadIdx.x*cpy_ne_D], ((const float *) VKQ) + i0/warp_size);
            }
#endif // FAST_FP16_AVAILABLE

            if (threadIdx.x == 0) {
                KQ_sum_combine[threadIdx.y] = KQ_sum[0];
            }

            return;
        }

        __syncthreads();

#pragma unroll
        for (int ip = 1; ip < np; ++ip) {
#ifdef FAST_FP16_AVAILABLE
            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                __align__(16) half2 tmp[cpy_ne_D];
                ggml_cuda_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(threadIdx.y + ip)*(DVp/2) + i0 + threadIdx.x*cpy_ne_D]);
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    VKQ[i0/warp_size + i1] += tmp[i1];
                }
            }
#else
            constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                __align__(16) float tmp[cpy_ne_D];
                ggml_cuda_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(threadIdx.y + ip)*DVp + i0 + threadIdx.x*cpy_ne_D]);
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    ((float *)VKQ)[i0/warp_size + i1] += tmp[i1];
                }
            }
#endif // FAST_FP16_AVAILABLE

            KQ_sum[0] += KQ_sum_combine[threadIdx.y + ip];
        }
    }

    // Attention sink: adjust KQ max and sum only for the first of all parallel blocks:
    if (sinks && blockIdx.y == 0) {
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int jc = jc0 + (threadIdx.y/np)*cpw;
            if (!col_valid(jc)) {
                continue;
            }
            const float sink = ((const float *) sinks)[head0 + (!is_quant_kv ? (jc % ncols2) : (jc % ncols2_eff))];

            float KQ_max_new_j = fmaxf(KQ_max[jc0], sink);
            const float KQ_max_scale = expf(KQ_max[jc0] - KQ_max_new_j);
            KQ_max[jc0] = KQ_max_new_j;

            const float val = expf(sink - KQ_max[jc0]);
            KQ_sum[jc0] = KQ_sum[jc0]*KQ_max_scale + val;

#ifdef FAST_FP16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size].x *= KQ_max_scale;
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size].y *= KQ_max_scale;
            }
#endif // FAST_FP16_AVAILABLE
        }
    }

    // Write back results:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + (threadIdx.y/np)*cpw;

        const int j = !is_quant_kv ? (jc / ncols2) : (jc / ncols2_eff);
        const int c = !is_quant_kv ? (jc % ncols2) : (jc % ncols2_eff);

        if constexpr (!is_quant_kv) {
            if (ncols1 > 1 && col_Q_0 + j >= int(ne01.z)) {
                return;
            }
        } else {
            if (!col_valid(jc)) {
                continue;
            }
        }

        const float scale = gridDim.y == 1 ? 1.0f/KQ_sum[jc0] : 1.0f;

        const int j_dst_unrolled = ((sequence*int(ne01.z) + col_Q_0 + j)*ne02 + head0 + c)*gridDim.y + blockIdx.y;

#ifdef FAST_FP16_AVAILABLE
        constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
            __align__(16) float2 tmp[cpy_ne_D];
#pragma unroll
            for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = __half22float2(VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size + i1]);
                tmp[i1].x *= scale;
                tmp[i1].y *= scale;
            }
            if (i0 + warp_size*cpy_ne_D <= DV/2 || i0 + threadIdx.x*cpy_ne_D < DV/2) {
                ggml_cuda_memcpy_1<sizeof(tmp)>(&dst[j_dst_unrolled*DV + 2*i0 + threadIdx.x*(2*cpy_ne_D)], tmp);
            }
        }
#else
        constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
            if (i0 + warp_size*cpy_ne_D <= DV || i0 + threadIdx.x*cpy_ne_D < DV) {
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].x *= scale;
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].y *= scale;
                }
                ggml_cuda_memcpy_1<cpy_ne_D*4>(
                    &dst[j_dst_unrolled*DV + i0 + threadIdx.x*cpy_ne_D],
                    &VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size)]);
            }
        }
#endif // FAST_FP16_AVAILABLE

        if (gridDim.y != 1 && threadIdx.x == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[jc0], KQ_sum[jc0]);
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, KV_rows_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

template <int DKQ, int DV, int ncols2, bool use_logit_softcap, ggml_type type_K, ggml_type type_V>
static void launch_fattn_tile_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cuda_fattn_tile_shape & shape, const bool use_native_kv) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const int id        = ggml_cuda_get_device();
    const int cc        = ggml_cuda_info().devices[id].cc;
    const int warp_size = 32;

    constexpr size_t nbytes_shared = 0;

    const bool fold_gqa   = use_native_kv;
    const bool need_f16_K = !use_native_kv || K->type != type_K;
    const bool need_f16_V = !use_native_kv || V->type != type_V;

    const int cols_per_block_sel = shape.cols_per_block;

#ifdef GGML_USE_HIP
    if constexpr (DKQ <= 128) {
        if (cols_per_block_sel == 64) {
            constexpr int cols_per_block = 64;
            const int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc, use_native_kv) / warp_size;
            const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc, use_native_kv);
            fattn_kernel_t fattn_kernel = flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, type_K, type_V>;
            launch_fattn<DV, cols_per_block/ncols2, ncols2>
                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, need_f16_K, need_f16_V, false, false, warp_size, false, fold_gqa);
            return;
        }
    }
#endif // GGML_USE_HIP

#ifndef GGML_USE_HIP
    if constexpr (DKQ <= 256)
#endif // GGML_USE_HIP
    {
        if (cols_per_block_sel == 32) {
            constexpr int cols_per_block = 32;
            const int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc, use_native_kv) / warp_size;
            const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc, use_native_kv);
            fattn_kernel_t fattn_kernel = flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, type_K, type_V>;
            launch_fattn<DV, cols_per_block/ncols2, ncols2>
                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, need_f16_K, need_f16_V, false, false, warp_size, false, fold_gqa);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (cols_per_block_sel == 16) {
            constexpr int cols_per_block = 16;
            const int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc, use_native_kv) / warp_size;
            const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc, use_native_kv);
            fattn_kernel_t fattn_kernel = flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, type_K, type_V>;
            launch_fattn<DV, cols_per_block/ncols2, ncols2>
                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, need_f16_K, need_f16_V, false, false, warp_size, false, fold_gqa);
            return;
        }
    }

    if constexpr (ncols2 <= 8) {
        if (cols_per_block_sel == 8) {
            constexpr int cols_per_block = 8;
            const int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc, use_native_kv) / warp_size;
            const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc, use_native_kv);
            fattn_kernel_t fattn_kernel = flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, type_K, type_V>;
            launch_fattn<DV, cols_per_block/ncols2, ncols2>
                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, need_f16_K, need_f16_V, false, false, warp_size, false, fold_gqa);
            return;
        }
    }

    if constexpr (ncols2 <= 4) {
        if (cols_per_block_sel == 4) {
            constexpr int cols_per_block = 4;
            const int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc, use_native_kv) / warp_size;
            const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc, use_native_kv);
            fattn_kernel_t fattn_kernel = flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, type_K, type_V>;
            launch_fattn<DV, cols_per_block/ncols2, ncols2>
                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, need_f16_K, need_f16_V, false, false, warp_size, false, fold_gqa);
            return;
        }
    }

    if constexpr (ncols2 <= 2) {
        constexpr int cols_per_block = 2;
        const int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc, use_native_kv) / warp_size;
        const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc, use_native_kv);
        fattn_kernel_t fattn_kernel = flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap, type_K, type_V>;
        launch_fattn<DV, cols_per_block/ncols2, ncols2>
            (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, need_f16_K, need_f16_V, false, false, warp_size, false, fold_gqa);
        return;
    }

    GGML_ABORT("fatal error");
}

template <int DKQ, int DV, bool use_logit_softcap, ggml_type type_K, ggml_type type_V>
static void launch_fattn_tile_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool use_native_kv = !use_logit_softcap && ggml_cuda_fattn_tile_use_native_kv(cc, dst);
    const ggml_cuda_fattn_tile_shape shape = ggml_cuda_fattn_tile_get_shape(dst, cc, use_native_kv);

#define GGML_CUDA_FATTN_TILE_LAUNCH_NCOLS2(ncols2_)                                                            \
    if constexpr (ggml_cuda_fattn_tile_ncols2_instantiated(DKQ, DV, ncols2_)) {                                 \
        if (shape.ncols2 == (ncols2_)) {                                                                        \
            launch_fattn_tile_switch_ncols1<DKQ, DV, ncols2_, use_logit_softcap, type_K, type_V>(ctx, dst, shape, use_native_kv); \
            return;                                                                                             \
        }                                                                                                       \
    }                                                                                                           \

    GGML_CUDA_FATTN_TILE_LAUNCH_NCOLS2(32)
    GGML_CUDA_FATTN_TILE_LAUNCH_NCOLS2(16)
    GGML_CUDA_FATTN_TILE_LAUNCH_NCOLS2( 8)
    GGML_CUDA_FATTN_TILE_LAUNCH_NCOLS2( 4)
    GGML_CUDA_FATTN_TILE_LAUNCH_NCOLS2( 2)
    GGML_CUDA_FATTN_TILE_LAUNCH_NCOLS2( 1)

#undef GGML_CUDA_FATTN_TILE_LAUNCH_NCOLS2

    GGML_ABORT("flash-attn tile: no kernel for head size %d with GQA ratio %d", DKQ, int(Q->ne[2] / K->ne[2]));
}

template <int DKQ, int DV, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_tile_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap, type_K, type_V>(ctx, dst);
    } else {
        if constexpr (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
            constexpr bool use_logit_softcap = true;
            launch_fattn_tile_switch_ncols2<DKQ, DV, use_logit_softcap, type_K, type_V>(ctx, dst);
        } else {
            GGML_ABORT("fatal error"); // Native K/V with logit softcap is not instantiated, the host converts K/V to f16.
        }
    }
}

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define DECL_FATTN_TILE_CASE(DKQ, DV, type_K, type_V)                             \
    template void ggml_cuda_flash_attn_ext_tile_case                              \
    <DKQ, DV, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

// extern so that the dispatch table in fattn-tile.cu does not instantiate the kernels a second time.
// The lists must match generate_cu_files.py, which has quantized pairs only for DKQ % 32 == 0.
#define EXTERN_DECL_FATTN_TILE_CASES_FP(type_K, type_V)     \
    extern DECL_FATTN_TILE_CASE( 40,  40, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE( 64,  64, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE( 72,  72, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE( 80,  80, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE( 96,  96, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE(112, 112, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE(128, 128, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE(192, 128, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE(256, 256, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE(320, 256, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE(512, 512, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE(576, 512, type_K, type_V);  \

#define EXTERN_DECL_FATTN_TILE_CASES_QUANT(type_K, type_V)  \
    extern DECL_FATTN_TILE_CASE( 64,  64, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE( 96,  96, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE(128, 128, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE(192, 128, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE(256, 256, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE(320, 256, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE(512, 512, type_K, type_V);  \
    extern DECL_FATTN_TILE_CASE(576, 512, type_K, type_V);  \

#define EXTERN_DECL_FATTN_TILE_CASES_ALL_V(type_K, CASES_F16, CASES_BF16)   \
    CASES_F16                          (type_K, GGML_TYPE_F16)              \
    EXTERN_DECL_FATTN_TILE_CASES_QUANT (type_K, GGML_TYPE_Q4_0)             \
    EXTERN_DECL_FATTN_TILE_CASES_QUANT (type_K, GGML_TYPE_Q4_1)             \
    EXTERN_DECL_FATTN_TILE_CASES_QUANT (type_K, GGML_TYPE_Q5_0)             \
    EXTERN_DECL_FATTN_TILE_CASES_QUANT (type_K, GGML_TYPE_Q5_1)             \
    EXTERN_DECL_FATTN_TILE_CASES_QUANT (type_K, GGML_TYPE_Q8_0)             \
    CASES_BF16                         (type_K, GGML_TYPE_BF16)             \

EXTERN_DECL_FATTN_TILE_CASES_ALL_V(GGML_TYPE_F16,  EXTERN_DECL_FATTN_TILE_CASES_FP,    EXTERN_DECL_FATTN_TILE_CASES_FP)
EXTERN_DECL_FATTN_TILE_CASES_ALL_V(GGML_TYPE_Q4_0, EXTERN_DECL_FATTN_TILE_CASES_QUANT, EXTERN_DECL_FATTN_TILE_CASES_QUANT)
EXTERN_DECL_FATTN_TILE_CASES_ALL_V(GGML_TYPE_Q4_1, EXTERN_DECL_FATTN_TILE_CASES_QUANT, EXTERN_DECL_FATTN_TILE_CASES_QUANT)
EXTERN_DECL_FATTN_TILE_CASES_ALL_V(GGML_TYPE_Q5_0, EXTERN_DECL_FATTN_TILE_CASES_QUANT, EXTERN_DECL_FATTN_TILE_CASES_QUANT)
EXTERN_DECL_FATTN_TILE_CASES_ALL_V(GGML_TYPE_Q5_1, EXTERN_DECL_FATTN_TILE_CASES_QUANT, EXTERN_DECL_FATTN_TILE_CASES_QUANT)
EXTERN_DECL_FATTN_TILE_CASES_ALL_V(GGML_TYPE_Q8_0, EXTERN_DECL_FATTN_TILE_CASES_QUANT, EXTERN_DECL_FATTN_TILE_CASES_QUANT)
EXTERN_DECL_FATTN_TILE_CASES_ALL_V(GGML_TYPE_BF16, EXTERN_DECL_FATTN_TILE_CASES_FP,    EXTERN_DECL_FATTN_TILE_CASES_FP)
