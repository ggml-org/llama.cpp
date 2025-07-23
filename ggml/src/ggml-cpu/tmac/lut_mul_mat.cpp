#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "lut_mul_mat.h"


#if defined(GGML_USE_TMAC)

namespace ggml::cpu::tmac {
    bool tensor_traits::work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) {
        if (ggml_tmac_can_mul_mat(op)) {
            size = ggml_backend_tmac_desired_wsize(op);
            return true;
        }
        return false;
    }

    bool tensor_traits::compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
        if (ggml_tmac_can_mul_mat(op)) {
            ggml_backend_tmac_mul_mat(params, op);
            return true;
        };
        return false;
    }
}  // namespace ggml::cpu::tmac


/****** T-MAC properties ******/
constexpr size_t kAllocAlignment = 64;
const int n_threads = 8;

static tmac_tensor_extra * tmac_tensor_extras = nullptr;
static size_t tmac_tensor_extras_index = 0;

struct tmac_run_single_kernel_settings {
    int32_t test_time_ms;
    int32_t M;
    int32_t N;
    int32_t K;

    int32_t n;

    struct tmac_kernel_config * kernel_config;
};

static bool initialized = false;
void tmac_init() {
    if (initialized) {
        return;
    }
    initialized = true;

    if (tmac_tensor_extras == nullptr) {
        tmac_tensor_extras = new tmac_tensor_extra[GGML_TMAC_MAX_NODES];
    }
    tmac_tensor_extras_index = 0;
}
void tmac_free() {
    // TODO
}

/****** T-MAC helper functions ******/
static inline bool is_tmac_2bit_type(enum ggml_type type) {
    return (
        type == GGML_TYPE_TMAC_BN_0 ||
        type == GGML_TYPE_TMAC_W2G64_0 ||
        type == GGML_TYPE_TMAC_W2G64_1 ||
        type == GGML_TYPE_TMAC_W2G128_0 ||
        type == GGML_TYPE_TMAC_W2G128_1
    );
}

static inline bool is_tmac_4bit_type(enum ggml_type type) {
    return (
        type == GGML_TYPE_TMAC_W4G64_0 ||
        type == GGML_TYPE_TMAC_W4G64_1 ||
        type == GGML_TYPE_TMAC_W4G128_0 ||
        type == GGML_TYPE_TMAC_W4G128_1
    );
}

bool is_tmac_type(enum ggml_type type) {
    return (
        is_tmac_2bit_type(type) ||
        is_tmac_4bit_type(type)
    );
}

bool is_type_supported(enum ggml_type type) {
    return (
        type == GGML_TYPE_Q4_0 ||
        type == GGML_TYPE_TQ1_0 ||
        type == GGML_TYPE_TQ2_0 ||
        is_tmac_2bit_type(type) ||
        is_tmac_4bit_type(type)
    );
}

bool ggml_tmac_can_mul_mat(const struct ggml_tensor * dst) {
    struct ggml_tensor * src0 = dst->src[0];
    struct ggml_tensor * src1 = dst->src[1];

    if (dst->op == GGML_OP_MUL_MAT &&
        (is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        strcmp(src0->name, "token_embd.weight") &&  // means not equal
        strcmp(src0->name, "output.weight")) {
        return true;
    }
    return false;
}

bool is_embedding_weight(const struct ggml_tensor * tensor) {
    return (strcmp(tensor->name, "token_embd.weight") == 0);
}

static inline int get_type_bits(enum ggml_type type) {
    if (is_tmac_2bit_type(type) || type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0) {
        return 2;
    } else if (is_tmac_4bit_type(type) || type == GGML_TYPE_Q4_0) {
        return 4;
    } else {
        return 0;
    }
}

static inline int get_type_group_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TMAC_BN_0:
            return -1;
        case GGML_TYPE_TMAC_W2G64_0:
        case GGML_TYPE_TMAC_W2G64_1:
        case GGML_TYPE_TMAC_W4G64_0:
        case GGML_TYPE_TMAC_W4G64_1:
            return 64;
        case GGML_TYPE_TMAC_W2G128_0:
        case GGML_TYPE_TMAC_W2G128_1:
        case GGML_TYPE_TMAC_W4G128_0:
        case GGML_TYPE_TMAC_W4G128_1:
            return 128;
        case GGML_TYPE_Q4_0:
            return 32;
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
            return 256;
        default:
            return 0;
    }
}

static inline bool get_type_has_zero_point(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TMAC_BN_0:
        case GGML_TYPE_TMAC_W2G64_0:
        case GGML_TYPE_TMAC_W4G64_0:
        case GGML_TYPE_TMAC_W2G128_0:
        case GGML_TYPE_TMAC_W4G128_0:
            return false;
        case GGML_TYPE_TMAC_W2G64_1:
        case GGML_TYPE_TMAC_W4G64_1:
        case GGML_TYPE_TMAC_W2G128_1:
        case GGML_TYPE_TMAC_W4G128_1:
            return true;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
            return false;
        default:
            return false;
    }
}

static inline bool get_type_is_one_scale(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TMAC_BN_0:
            return true;
        default:
            return false;
    }
}

static inline int ggml_tmac_get_scales_size(const struct tmac_kernel_config * kernel_config, int m, int k) {
    int scales_size;
    if (kernel_config->one_scale) {
        scales_size = 1;
    } else if (kernel_config->has_zero_point) {
        scales_size = m * k / kernel_config->q_group_size * 2;
    } else{
        scales_size = m * k / kernel_config->q_group_size;
    }
    return scales_size;
}


static void * aligned_malloc(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, kAllocAlignment);
#else
    void * ptr = nullptr;
    posix_memalign(&ptr, kAllocAlignment, size);
    return ptr;
#endif
}
    
static void aligned_free(void * ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}


/****** T-MAC meta model info ******/
static void init_tmac_kernel_config_from_tensor_type(enum ggml_type type, int M, struct tmac_kernel_config * kernel_config) {
    kernel_config->bits = get_type_bits(type);
    kernel_config->q_group_size = get_type_group_size(type);
    kernel_config->has_zero_point = get_type_has_zero_point(type);
    kernel_config->one_scale = get_type_is_one_scale(type);

    // Fixed features
    kernel_config->has_scale = true;
    kernel_config->g = 4;
    kernel_config->ngroups_per_elem = 8 / kernel_config->g;

    // Decide q_group_size for BN_0
    if (kernel_config->q_group_size == -1) {
        if (M % 256 == 0) {
            kernel_config->q_group_size = 64;
        } else if (M % 128 == 0) {
            kernel_config->q_group_size = 64;
        } else if (M % 64 == 0) {
            kernel_config->q_group_size = 64;
        } else if (M % 32 == 0) {
            kernel_config->q_group_size = 32;
        } else {
            GGML_LOG_ERROR("Unsupported M value. Expected multiple of 32, got %d. Please check all of the model weight shapes.\n", M);
        }
    }

    if (kernel_config->q_group_size % 64 == 0) {
        kernel_config->act_group_size = 64;
    } else if (kernel_config->q_group_size % 32 == 0) {
        kernel_config->act_group_size = 32;
    } else {
        GGML_LOG_ERROR("Unsupported activation group size: %d\n", kernel_config->q_group_size);
    }
    kernel_config->actk = kernel_config->act_group_size / kernel_config->g;

    // kfactor to be tuned
    // bm to be tuned
    kernel_config->simd_n_in = 16;
    kernel_config->simd_n_out = 8;

    kernel_config->chunk_n = 8;
}


/****** T-MAC configurations ******/
static std::unordered_map<std::string, struct tmac_kernel_config> final_tmac_kernel_config;
static std::string get_tmac_kernel_config_key(int M, int K, int bits) {
    return "M" + std::to_string(M) + "_K" + std::to_string(K) + "_b" + std::to_string(bits);
}
struct tmac_kernel_config * find_tmac_kernel_config(int M, int K, int bits)
{
    std::string key = get_tmac_kernel_config_key(M, K, bits);
    if (final_tmac_kernel_config.count(key) == 0) {
        return nullptr;
    }
    return &final_tmac_kernel_config[key];
}
static void insert_or_assign_tmac_kernel_config(int M, int K, int bits, struct tmac_kernel_config kernel_config)
{
    std::string key = get_tmac_kernel_config_key(M, K, bits);
    final_tmac_kernel_config.insert_or_assign(key, kernel_config);
}


static inline void ggml_tmac_forward_mul_mat(
        void * A, void * B, void * C, void * QLUT, void * LUT_Scales, void * LUT_Biases, void * Scales,
        int M, int N, int K, const struct tmac_kernel_config * kernel_config) {
    // Currently, scale is a must.
    assert(kernel_config->has_scale);
    // Currently, one_scale and has_zero_point are mutually exclusive.
    assert(!(kernel_config->one_scale && kernel_config->has_zero_point));

    int bits = kernel_config->bits;
    int bm = kernel_config->bm;
    int act_group_size = kernel_config->act_group_size;

    lut_ctor_int8_g4(B, LUT_Scales, LUT_Biases, QLUT, K, kernel_config);

    const int m = bm / bits;
    const int64_t chunk_size0 = m;

    for (int32_t chunk_outer = 0; chunk_outer < M/m; chunk_outer++) {
        /* One Block */
        const int64_t w_offset      = chunk_outer * m * K * bits / 8;
        const int64_t scales_offset = kernel_config->one_scale ? 0 : ggml_tmac_get_scales_size(kernel_config, m, K) * chunk_outer;

        for (int32_t n_outer = 0; n_outer < N; n_outer++) {
            const int64_t qlut_offset       = K * n_outer * 4;
            const int64_t lut_scales_offset = K / act_group_size * n_outer;
            const int64_t dst_offset        = M * n_outer + chunk_outer * chunk_size0;

            int8_t *lut = (int8_t *)QLUT + qlut_offset;
            uint8_t *a = (uint8_t *)A + w_offset;
            tmac_float_type *scales = (tmac_float_type *)Scales + scales_offset;
            tmac_float_type *lut_scales = (tmac_float_type *)LUT_Scales + lut_scales_offset;
            tmac_float_type *lut_biases = (tmac_float_type *)LUT_Biases + lut_scales_offset;
            tmac_float_type *act_output = (tmac_float_type *)C + dst_offset;

            qgemm_lut_int8_g4(a, lut, scales, lut_scales, lut_biases, act_output, bm, K, N, kernel_config);
        }  
        /* One Block */
    }
}

static void ggml_tmac_tune_single_kernel_config(const struct tmac_run_single_kernel_settings * const settings, double & elapsed_time) {
    if (settings->kernel_config->kfactor < settings->kernel_config->actk) {
        return;
    }

    const int test_time_ms = settings->test_time_ms;
    const int M = settings->M;
    const int N = settings->N;
    const int K = settings->K;
    const struct tmac_kernel_config * const kernel_config = settings->kernel_config;
    const int bits = kernel_config->bits;
    const int act_group_size = kernel_config->act_group_size;
    const int bm = kernel_config->bm;
    // const int m = bm / bits;
    const int scales_size = ggml_tmac_get_scales_size(kernel_config, M, K);

    std::chrono::duration<double> total_elapsed = std::chrono::duration<double>::zero();
    GGML_LOG_DEBUG("Run single kernel config: M=%d, N=%d, K=%d, bm=%d, kfactor=%d, actk=%d\n", M, N, K, bm, kernel_config->kfactor, kernel_config->actk);
    int n_try = 0;
    while (total_elapsed.count() < test_time_ms / 1000.0) {
        uint8_t    *A = new uint8_t[M * K * bits / 8];          // quantized weight
        tmac_float_type *B = new tmac_float_type[K * N];        // activation
        tmac_float_type *C = new tmac_float_type[M * N];        // output
        int8_t     *QLUT = new int8_t[K * N * 4];
        tmac_float_type *LUT_Scales = new tmac_float_type[K * N / act_group_size];
        tmac_float_type *LUT_Biases = new tmac_float_type[K * N / act_group_size];
        tmac_float_type *Scales = new tmac_float_type[scales_size];

        // multi-threading profiling
        auto start = std::chrono::high_resolution_clock::now();
        ggml_tmac_forward_mul_mat(A, B, C, QLUT, LUT_Scales, LUT_Biases, Scales,
            M, N, K, kernel_config);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        total_elapsed += elapsed;
        n_try++;

        delete[] A;
        delete[] B;
        delete[] C;
        delete[] QLUT;
        delete[] LUT_Scales;
        delete[] LUT_Biases;
        delete[] Scales;
    }

    elapsed_time = total_elapsed.count() / n_try * 1000.0;  // in ms
}

static void ggml_tmac_tune_kernel_config(const struct ggml_tensor * tensor, int M, int K) {
    const int bits = get_type_bits(tensor->type);
    struct tmac_kernel_config * existing_kcfg = find_tmac_kernel_config(M, K, bits);
    if (existing_kcfg != nullptr) {
        return;
    }

    struct tmac_kernel_config kernel_config;
    init_tmac_kernel_config_from_tensor_type(tensor->type, M, &kernel_config);

    // TODO: add more choices for prefilling?
    int N = 1;

    // search space
    std::vector<int> bms;
    if (bits == 1 || bits == 2 || bits == 4) {
        bms = {256, 512, 1024, 2048, 320, 640, 1280};
    } else if (bits == 3) {
        bms = {192, 384, 576, 768};
    }
    std::vector<int> bns = {8, 16, 32, 64};
    std::vector<int> kfactors = {8, 16};


    double min_time = 1e9;
    struct tmac_kernel_config best_kcfg = kernel_config;

    auto profile_based = [&]() {
        for (int bm: bms) {
            if (M % (bm/bits) != 0 || bm % bits != 0) {
                continue;
            }
            
            kernel_config.bm = bm;
            for (int n: bns) {
                if ((N >= n && N % n != 0) || (N < n && n != bns[0])) {
                    continue;
                }

                for (int kfactor: kfactors) {
                    if ((kfactor < kernel_config.actk) || (kfactor * kernel_config.g > kernel_config.q_group_size)) {
                        continue;
                    }

                    kernel_config.kfactor = kfactor;
                    // insert to dict for finding
                    insert_or_assign_tmac_kernel_config(M, K, bits, kernel_config);
                    struct tmac_run_single_kernel_settings settings = {
                        /* .test_time_ms = */ 5000,
                        /* .M = */ M,
                        /* .N = */ N,
                        /* .K = */ K,
                        /* .n = */ n,
                        /* .kernel_config = */ &kernel_config
                    };
                    double this_time;
                    ggml_tmac_tune_single_kernel_config(&settings, this_time);
                    if (this_time < min_time) {
                        min_time = this_time;
                        best_kcfg = kernel_config;
                    }
                }
            }
        };
    };
    auto rule_based = [&]() {
        float smallest_penalty = 1e9;
        for (int bm: bms) {
            if (M % (bm/bits) != 0 || bm % bits != 0) {
                continue;
            }
            int num_tiles = M / (bm/bits);
            int num_groups = (num_tiles + n_threads - 1) / n_threads;
            float penalty = 0.1 * num_groups + (num_groups - 1.0 * num_tiles / n_threads) / num_groups;
            if (penalty <= smallest_penalty) {
                smallest_penalty = penalty;
                best_kcfg.bm = bm;
            }
        }

        int largest_kfactor = 0;
        for (int kfactor: kfactors) {
            if ((kfactor < kernel_config.actk) || (kfactor * kernel_config.g > kernel_config.q_group_size)) {
                continue;
            }
            if (kfactor > largest_kfactor) {
                largest_kfactor = kfactor;
                best_kcfg.kfactor = kfactor;
            }
        }
    };
    rule_based();

    // Save the results
    insert_or_assign_tmac_kernel_config(M, K, bits, best_kcfg);
    GGML_LOG_INFO("Tuned kernel config: M=%d, N=%d, K=%d, bm=%d, kfactor=%d, bits=%d, actk=%d, g=%d, ngroups_per_elem=%d, q_group_size=%d, act_group_size=%d\n",
                    M, N, K, best_kcfg.bm, best_kcfg.kfactor, bits, best_kcfg.actk, best_kcfg.g, best_kcfg.ngroups_per_elem, best_kcfg.q_group_size, best_kcfg.act_group_size);
}



size_t ggml_backend_tmac_desired_wsize(const struct ggml_tensor * dst) {
    struct ggml_tensor * src0 = dst->src[0];
    struct ggml_tensor * src1 = dst->src[1];

    const size_t n = src0->ne[1];    // llama.cpp n
    const size_t k = src1->ne[0];    // k
    const size_t m = src1->ne[1];    // llama.cpp m
    const int bits = get_type_bits(src0->type);

    struct tmac_kernel_config * kernel_config = find_tmac_kernel_config(n, k, bits);
    if (kernel_config == nullptr) {
        ggml_tmac_tune_kernel_config(src0, n, k);
        kernel_config = find_tmac_kernel_config(n, k, bits);
    }
    const int lut_scales_size = k / kernel_config->act_group_size;

    size_t wsize = k * m * 4 * sizeof(int8_t) + lut_scales_size * m * 2 * sizeof(tmac_float_type);
    if (sizeof(tmac_float_type) == 2) {
        // Need fp32 to fp16 conversion
        wsize += std::max(k, n) * m * sizeof(tmac_float_type);
    }
    wsize = ((wsize - 1) / kAllocAlignment + 1) * kAllocAlignment;
    return wsize;
}

size_t ggml_tmac_get_nbytes(const struct ggml_tensor * tensor) {
    if (is_tmac_type(tensor->type)) {
        const int bits = get_type_bits(tensor->type);

        int k = tensor->ne[0];
        int m = tensor->ne[1];  // `n` in llama.cpp

        struct tmac_kernel_config * kernel_config = find_tmac_kernel_config(m, k, bits);
        if (kernel_config == nullptr) {
            ggml_tmac_tune_kernel_config(tensor, m, k);
            kernel_config = find_tmac_kernel_config(m, k, bits);
        }

        const int scales_size = ggml_tmac_get_scales_size(kernel_config, m, k);
        // Currently, always uses float16 to store scales or zero points
        size_t nbytes = k * m / 8 * bits + scales_size * sizeof(ggml_fp16_t);
        nbytes = GGML_PAD(nbytes, GGUF_DEFAULT_ALIGNMENT);
        return nbytes;
    } else {
        return ggml_nbytes(tensor);
    }
}

    


/****** T-MAC convert tensor ******/
static bool do_permutate(enum ggml_type type) {
    return true;
    // if (type == GGML_TYPE_I1 ||
    //     type == GGML_TYPE_I2 ||
    //     type == GGML_TYPE_I3 ||
    //     type == GGML_TYPE_I4) {
    //     // Add additional args to decide if permuted I2 or naive I2
    //     return false;
    // } else {
    //     return true;
    // }
}

struct BlockQ40TypeAccessor {
    using block_t = block_q4_0;

    static constexpr int BITS = 4;
    static constexpr int SIMD_LEN = 16;
    static constexpr int group_size = (sizeof(block_t) - sizeof(ggml_fp16_t)) * 8 / BITS;
    static constexpr int simd_n_elem = SIMD_LEN * 8 / BITS;

    static uint8_t get_q(const void * data, int idx) {
        const uint8_t * qs = (const uint8_t *) ((((const block_t *) data)[idx / group_size]).qs);
        int internal_idx = idx % group_size;
        const uint8_t * simd_qs = qs + internal_idx / simd_n_elem * SIMD_LEN;
        int simd_idx = internal_idx % simd_n_elem;
        return simd_qs[simd_idx % SIMD_LEN] >> (simd_idx / SIMD_LEN * BITS);
    }

    static tmac_float_type get_scale(const void * data, int idx) {
        ggml_fp16_t d = ((const block_t *) data)[idx / group_size].d;
        if constexpr (sizeof(tmac_float_type) == 2) {
            tmac_float_type * fp16dp = reinterpret_cast<tmac_float_type *>(&d);
            return *fp16dp;
        } else {
            return ggml_fp16_to_fp32(d);
        }
    }
};

struct BlockI2TypeAccessor {
    static constexpr int BITS = 2;
    static constexpr int n_elem = 8 / BITS;

    static uint8_t get_q(const void * data, int idx) {
        const uint8_t * qs = (const uint8_t *) data;
        int elem_idx = idx % n_elem;
        return qs[idx / n_elem] >> ((n_elem - 1 - elem_idx) * BITS);
    }

    static tmac_float_type get_scale(const void * data, int idx, int group_size) {
        ggml_fp16_t s = ((const ggml_fp16_t *) data)[idx / group_size];
        if constexpr (sizeof(tmac_float_type) == 2) {
            tmac_float_type * fp16dp = reinterpret_cast<tmac_float_type *>(&s);
            return *fp16dp;
        } else {
            return ggml_fp16_to_fp32(s);
        }
    }

    static tmac_float_type get_zero_point(const void * data, int idx, int group_size) {
        const ggml_fp16_t * zs = (const ggml_fp16_t *) data;
        ggml_fp16_t z = zs[idx / group_size];
        if constexpr (sizeof(tmac_float_type) == 2) {
            tmac_float_type * fp16dp = reinterpret_cast<tmac_float_type *>(&z);
            return *fp16dp;
        } else {
            return ggml_fp16_to_fp32(z);
        }
    }
};

struct BlockI4TypeAccessor {
    static constexpr int BITS = 4;
    static constexpr int n_elem = 8 / BITS;

    static uint8_t get_q(const void * data, int idx) {
        const uint8_t * qs = (const uint8_t *) data;
        int elem_idx = idx % n_elem;
        return qs[idx / n_elem] >> ((n_elem - 1 - elem_idx) * BITS);
    }

    static tmac_float_type get_scale(const void * data, int idx, int group_size) {
        const ggml_fp16_t * ss = (const ggml_fp16_t *) data;
        ggml_fp16_t s = ss[idx / group_size];
        if constexpr (sizeof(tmac_float_type) == 2) {
            tmac_float_type * fp16dp = reinterpret_cast<tmac_float_type *>(&s);
            return *fp16dp;
        } else {
            return ggml_fp16_to_fp32(s);
        }
    }

    static tmac_float_type get_zero_point(const void * data, int idx, int group_size) {
        const ggml_fp16_t * zs = (const ggml_fp16_t *) data;
        ggml_fp16_t z = zs[idx / group_size];
        if constexpr (sizeof(tmac_float_type) == 2) {
            tmac_float_type * fp16dp = reinterpret_cast<tmac_float_type *>(&z);
            return *fp16dp;
        } else {
            return ggml_fp16_to_fp32(z);
        }
    }
};


struct BlockTQ10TypeAccessor {
    using block_t = block_tq1_0;

    static constexpr int elements_qs = 5;    // 5 elements per byte
    static constexpr int elements_qh = 4;    // 4 elements per byte
    static constexpr int BITS = 2;
    static constexpr int group_size_qs = sizeof(((block_t *)0)->qs) * elements_qs;
    static constexpr int group_size_qh = sizeof(((block_t *)0)->qh) * elements_qh;
    static constexpr int group_size = group_size_qs + group_size_qh;
    static constexpr int SIMD_LEN_qs_1 = 32;
    static constexpr int SIMD_LEN_qs_2 = 16;
    static constexpr int SIMD_LEN_qh = 4;
    static constexpr int simd_n_elem_qs_1 = SIMD_LEN_qs_1 * elements_qs;        // 160
    static constexpr int simd_n_elem_qs_2 = SIMD_LEN_qs_2 * elements_qs;        // 80
    static constexpr int simd_n_elem_qh = SIMD_LEN_qh * elements_qh;            // 16

    static constexpr uint8_t pow3[5] = {1, 3, 9, 27, 81};

    static uint8_t get_q(const void * data, int idx) {
        const uint8_t * qs = (const uint8_t *) ((((const block_t *) data)[idx / group_size]).qs);
        uint8_t cur_qs;
        uint8_t trit;
        int internal_idx = idx % group_size;

        if (internal_idx < simd_n_elem_qs_1) {
            const int internal_offset = 0;
            const uint8_t * simd_qs = qs + internal_offset;
            int simd_idx = internal_idx;
            int simd_byte = simd_idx % SIMD_LEN_qs_1;
            int simd_trit = simd_idx / SIMD_LEN_qs_1;

            cur_qs = simd_qs[simd_byte] * pow3[simd_trit];
            trit = ((uint16_t) cur_qs * 3) >> 8;
        }
        else if (internal_idx < simd_n_elem_qs_1 + simd_n_elem_qs_2) {
            const int internal_offset = SIMD_LEN_qs_1;
            const uint8_t * simd_qs = qs + internal_offset;
            int simd_idx = internal_idx - simd_n_elem_qs_1;
            int simd_byte = simd_idx % SIMD_LEN_qs_2;
            int simd_trit = simd_idx / SIMD_LEN_qs_2;

            cur_qs = simd_qs[simd_byte] * pow3[simd_trit];
            trit = ((uint16_t) cur_qs * 3) >> 8;
        }
        else {
            const int internal_offset = SIMD_LEN_qs_1 + SIMD_LEN_qs_2;
            const uint8_t * simd_qs = qs + internal_offset;
            int simd_idx = internal_idx - simd_n_elem_qs_1 - simd_n_elem_qs_2;
            int simd_byte = simd_idx % SIMD_LEN_qh;
            int simd_trit = simd_idx / SIMD_LEN_qh;

            cur_qs = simd_qs[simd_byte] * pow3[simd_trit];
            trit = ((uint16_t) cur_qs * 3) >> 8;
        }

        return trit + 1;
    }

    static tmac_float_type get_scale(const void * data, int idx, int group_size) {
        ggml_fp16_t d = ((const block_t *) data)[idx / group_size].d;
        if constexpr (sizeof(tmac_float_type) == 2) {
            tmac_float_type * fp16dp = reinterpret_cast<tmac_float_type *>(&d);
            return *fp16dp;
        } else {
            return ggml_fp16_to_fp32(d);
        }
    }
};

struct BlockTQ20TypeAccessor {
    using block_t = block_tq2_0;

    static constexpr int BITS = 2;
    static constexpr int SIMD_LEN = 32;
    static constexpr int group_size = (sizeof(block_t) - sizeof(ggml_fp16_t)) * 8 / BITS;   // 256
    static constexpr int simd_n_elem = SIMD_LEN * 8 / BITS;                                 // 128

    static uint8_t get_q(const void * data, int idx) {
        const uint8_t * qs = (const uint8_t *) ((((const block_t *) data)[idx / group_size]).qs);
        int internal_idx = idx % group_size;
        const uint8_t * simd_qs = qs + internal_idx / simd_n_elem * SIMD_LEN;
        int simd_idx = internal_idx % simd_n_elem;
        return (simd_qs[simd_idx % SIMD_LEN] >> (simd_idx / SIMD_LEN * BITS)) + 1;
    }

    static tmac_float_type get_scale(const void * data, int idx, int group_size) {
        ggml_fp16_t d = ((const block_t *) data)[idx / group_size].d;
        if constexpr (sizeof(tmac_float_type) == 2) {
            tmac_float_type * fp16dp = reinterpret_cast<tmac_float_type *>(&d);
            return *fp16dp;
        } else {
            return ggml_fp16_to_fp32(d);
        }
    }
};

static inline void ggml_tmac_transform_tensor(struct ggml_tensor * tensor, const void * origin_data) {
    GGML_ASSERT(tensor->extra != nullptr);
    struct ggml::cpu::tmac::tensor_traits * tensor_extra = (struct ggml::cpu::tmac::tensor_traits *) tensor->extra;
    if (!(is_type_supported(tensor->type) && tensor_extra->get_tmac_tensor_extra(tensor->name) == nullptr)) {
        return;
    }

    const int bits = get_type_bits(tensor->type);
    int k = tensor->ne[0];
    int m = tensor->ne[1];  // `n` in llama.cpp

    struct tmac_kernel_config * kernel_config = find_tmac_kernel_config(m, k, bits);
    if (kernel_config == nullptr) {
        ggml_tmac_tune_kernel_config(tensor, m, k);
        kernel_config = find_tmac_kernel_config(m, k, bits);
    }

    // Currently, scale is a must.
    assert(kernel_config->has_scale);
    // Currently, one_scale and has_zero_point are mutually exclusive.
    assert(!(kernel_config->one_scale && kernel_config->has_zero_point));

    const int g = kernel_config->g;
    const int ngroups_per_elem = kernel_config->ngroups_per_elem;
    const int bm = kernel_config->bm;
    const int simd_n_in = kernel_config->simd_n_in;
    const int simd_n_out = kernel_config->simd_n_out;
    const int kfactor = kernel_config->kfactor;
    const int group_size = kernel_config->q_group_size;

    const int act_group_size = kernel_config->act_group_size;
    const int lut_scales_size = k / act_group_size;
    const int scales_size = ggml_tmac_get_scales_size(kernel_config, m, k);
    const int n_tile_num = m * bits / bm;

    GGML_LOG_DEBUG("Transforming tensor: %s (m: %d, k: %d, bits: %d)\n", tensor->name, m, k, bits);
    GGML_LOG_DEBUG("kcfg (bm=%d, simd_n_in=%d, simd_n_out=%d, kfactor=%d, group_size=%d, lut_scales_size=%d, scales_size=%d, n_tile_num=%d)\n",
        bm, simd_n_in, simd_n_out, kfactor, group_size, lut_scales_size, scales_size, n_tile_num);
    if (bm == 0) {
        if (!strcmp(tensor->name, "token_embd.weight") || !strcmp(tensor->name, "output.weight")) {
            GGML_LOG_WARN("Do not find kcfg for %s. Consider compiling T-MAC kernel for it if vocab size is a multiply of 128 or 320, detected %lld.\n", tensor->name, tensor->ne[1]);
            return;
        }
        else {
            // TODO: Instead of fatal error, try to avoid using t-mac?
            GGML_LOG_ERROR("Failed to find kcfg. Abort transforming\n");
            return;
        }
    }

    const int mgroup = ngroups_per_elem * simd_n_in;
    m = m * bits;

    uint8_t * qweights;
    tmac_float_type * scales;

    // TODO: if sizeof(tmac_float_type) <= sizeof(float), we can copy tensor->data to qweights and scales,
    //       and do permutation on tensor->data, finally aligned_free qweights and scales.
    if (do_permutate(tensor->type)) {
        scales = (tmac_float_type *) aligned_malloc(scales_size * sizeof(tmac_float_type));
        qweights = (uint8_t *) aligned_malloc(k * m / 8);
    } else {
        /* scales could be either float32 or float16, so inplace cast is feasible. */
        GGML_ASSERT(sizeof(tmac_float_type) <= sizeof(float));
        qweights = (uint8_t *) tensor->data;
        scales = (tmac_float_type *) (qweights + k * m / 8);
        float * i2_scales = (float * )(qweights + k * m / 8);
        for (int i = 0; i < scales_size; i++) {
            scales[i] = (tmac_float_type) i2_scales[i];
        }
    }

    struct tmac_tensor_extra * cur_tensor_extra = new tmac_tensor_extra({
        /* .lut_scales_size = */ lut_scales_size,
        /* .scales_size     = */ scales_size,
        /* .n_tile_num      = */ n_tile_num,
        /* .qweights        = */ qweights,
        /* .scales          = */ scales
    });
    tensor_extra->set_tmac_tensor_extra(tensor->name, cur_tensor_extra);

    if (do_permutate(tensor->type)) {
// for fast testing
// #define TMAC_EMPTY_WEIGHTS
#ifndef TMAC_EMPTY_WEIGHTS
        std::vector<std::thread> threads;
        const int n_threads = std::thread::hardware_concurrency();

        // TODO: optimize to accelerate weights loading
        uint8_t * buf2 = new uint8_t[m * k / g];
        memset(buf2, 0, m * k / g);

        // # (M // bits, K, bits)
        // w = np.stack([(w >> ib) & 1 for ib in range(bits)], axis=-1)
        // # (M // bits, K, bits) -> (M // bits, bits, K) -> (M // bits, bits, K // g, g) -> (M // bits, bits, K // g)
        // w = w.transpose(0, 2, 1).reshape(M // bits, bits, K // g, g)
        // w = sum([(w[:, :, :, ig] << ig) for ig in range(g)])
        threads.reserve(n_threads);
        auto parallel_worker_buf2 = [&](size_t start_index, size_t end_index) {
        for (int im = start_index; im < end_index; im++) {
            for (int ik = 0; ik < k; ik++) {
                uint8_t v;
                if (tensor->type == GGML_TYPE_Q4_0) {
                    v = BlockQ40TypeAccessor::get_q(origin_data, im * k + ik);
                } else if (is_tmac_2bit_type(tensor->type)) {
                    v = BlockI2TypeAccessor::get_q(origin_data, im * k + ik);
                } else if (is_tmac_4bit_type(tensor->type)) {
                    v = BlockI4TypeAccessor::get_q(origin_data, im * k + ik);
                } else if (tensor->type == GGML_TYPE_TQ1_0) {
                    v = BlockTQ10TypeAccessor::get_q(origin_data, im * k + ik);
                } else if (tensor->type == GGML_TYPE_TQ2_0) {
                    v = BlockTQ20TypeAccessor::get_q(origin_data, im * k + ik);
                } else {
                    GGML_LOG_ERROR("Unsupported type: %s\n", ggml_type_name(tensor->type));
                }

                for (int ib = 0; ib < bits; ib++) {
                    int new_im = im;
                    int new_ib = ib;
                    int new_ik = ik / g;
                    int shft_left = ik % g;
                    buf2[new_im * bits * k / g + new_ib * k / g + new_ik] += ((v >> ib) & 1) << shft_left;
                }
            }
        }
        };

        size_t start_index = 0;
        size_t chunk_size = m / bits / n_threads;
        for (size_t i = 0; i < n_threads; ++i) {
            size_t end_index = (i == n_threads - 1) ? m / bits : start_index + chunk_size;

            // Create and launch a thread
            threads.emplace_back(parallel_worker_buf2,
                                start_index,
                                end_index); // Pass the mutex array by reference

            start_index = end_index;
        }
        // Wait for all threads to complete
        for (std::thread& t : threads) {
            t.join();
        }
        threads.clear();

        // # 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
        // # for bits=3
        // # bit0: [0, 8), bit1: [8, 16), bit2: [16, 24), bit0: [24, 32)
        // # (M // bits // simd_n_float16, bits, simd_n_float16, K // g)
        // w = w.reshape(M // bits // simd_n_out, simd_n_out, bits, K // g).transpose(0, 2, 1, 3)
        // mgroup = ngroups_per_elem * simd_n_in
        // w = w.reshape(M // mgroup, ngroups_per_elem, simd_n_in, K // g).transpose(0, 2, 1, 3)
        // #             0        1             2             3                 4                  5
        // w = w.reshape(M // bm, bm // mgroup, simd_n_in, ngroups_per_elem, K // g // kfactor, kfactor).transpose(0, 4, 1, 5, 2, 3)
        // w = sum([(w[:, :, :, :, :, ng] << (ng * g)) for ng in range(ngroups_per_elem)])
        memset(qweights, 0, m * k / g / ngroups_per_elem);

        int c0_fac2 = k / g;
        int c0_fac1 = simd_n_out * c0_fac2;
        int c0_fac0 = bits * c0_fac1;

        int c1_nb2 = k / g;
        int c1_nb1 = simd_n_in * c1_nb2;
        int c1_nb0 = ngroups_per_elem * c1_nb1;
        int c1_fac2 = k / g;
        int c1_fac1 = ngroups_per_elem * c1_fac2;
        int c1_fac0 = simd_n_in * c1_fac1;


        int c2_nb4 = kfactor;
        int c2_nb3 = k / g / kfactor * c2_nb4;
        int c2_nb2 = ngroups_per_elem * c2_nb3;
        int c2_nb1 = simd_n_in * c2_nb2;
        int c2_nb0 = bm / mgroup * c2_nb1;
        int c2_fac3 = simd_n_in * ngroups_per_elem;
        int c2_fac2 = kfactor * c2_fac3;
        int c2_fac1 = bm / mgroup * c2_fac2;
        int c2_fac0 = k / g / kfactor * c2_fac1;

        threads.reserve(n_threads);
        auto parallel_worker_qweights = [&](size_t start_index, size_t end_index) {
        for (int im = start_index; im < end_index; im++) {
            for (int ib = 0; ib < bits; ib++) {
                for (int ik = 0; ik < k / g; ik++) {
                    // w = w.reshape(M // bits // simd_n_out, simd_n_out, bits, K // g).transpose(0, 2, 1, 3)
                    int new_im = im / simd_n_out;
                    int new_isno = im % simd_n_out;
                    int new_ib = ib;
                    int new_ik = ik;
                    int new_idx = new_im * c0_fac0 + new_ib * c0_fac1 + new_isno * c0_fac2 + new_ik;

                    // w = w.reshape(M // mgroup, ngroups_per_elem, simd_n_in, K // g).transpose(0, 2, 1, 3)
                    new_im = new_idx / c1_nb0;
                    int new_ing = (new_idx % c1_nb0) / c1_nb1;
                    int new_isni = (new_idx % c1_nb1) / c1_nb2;
                    new_ik = (new_idx % c1_nb2);
                    new_idx = new_im * c1_fac0 + new_isni * c1_fac1 + new_ing * c1_fac2 + new_ik;

                    // #             0        1             2             3                 4                  5
                    // w = w.reshape(M // bm, bm // mgroup, simd_n_in, ngroups_per_elem, K // g // kfactor, kfactor).transpose(0, 4, 1, 5, 2, 3)
                    new_im = new_idx / c2_nb0;
                    int new_ibm = (new_idx % c2_nb0) / c2_nb1;
                    new_isni = (new_idx % c2_nb1) / c2_nb2;
                    new_ing = (new_idx % c2_nb2) / c2_nb3;
                    new_ik = (new_idx % c2_nb3) / c2_nb4;
                    int new_ikf = (new_idx % c2_nb4);
                    new_idx = new_im * c2_fac0 +
                            new_ik * c2_fac1 +
                            new_ibm * c2_fac2 +
                            new_ikf * c2_fac3 +
                            new_isni * ngroups_per_elem +
                            new_ing;
                    new_idx = new_idx / ngroups_per_elem;

                    // w = sum([(w[:, :, :, :, :, ng] << (ng * g)) for ng in range(ngroups_per_elem)])
                    qweights[new_idx] += buf2[im * bits * k / g + ib * k / g + ik] << (new_ing * g);
                }
            }
        }
        };

        start_index = 0;
        chunk_size = m / bits / n_threads;
        for (size_t i = 0; i < n_threads; ++i) {
            size_t end_index = (i == n_threads - 1) ? m / bits : start_index + chunk_size;

            // Create and launch a thread
            threads.emplace_back(parallel_worker_qweights,
                                start_index,
                                end_index); // Pass the mutex array by reference

            start_index = end_index;
        }
        // Wait for all threads to complete
        for (std::thread& t : threads) {
            t.join();
        }
        threads.clear();

        const ggml_fp16_t * int_n_scales = (const ggml_fp16_t * ) ((const uint8_t *) origin_data + k * m / 8);
        const ggml_fp16_t * int_n_zero_points = int_n_scales + scales_size / 2;

        if (scales_size < m / bits) {  // BitNet-like scale (m_groups,)
            for (int i = 0; i < scales_size; i++) {
                scales[i] = BlockI2TypeAccessor::get_scale(int_n_scales, i, 1);
            }
        } else {
            // TODO: move if-else outside the loop
            // scales = scales.reshape(M // bm, bm // bits, K // group_size).transpose(0, 2, 1)
            for (int im = 0; im < m / bits; im += 1) {
                for (int ik = 0; ik < k; ik += group_size) {
                    tmac_float_type scale;
                    int idx = im * k + ik;
                    if (tensor->type == GGML_TYPE_Q4_0) {
                        scale = BlockQ40TypeAccessor::get_scale(origin_data, idx);
                    } else if (is_tmac_2bit_type(tensor->type)) {
                        scale = BlockI2TypeAccessor::get_scale(int_n_scales, idx, group_size);
                    } else if (is_tmac_4bit_type(tensor->type)) {
                        scale = BlockI4TypeAccessor::get_scale(int_n_scales, idx, group_size);
                    } else if (tensor->type == GGML_TYPE_TQ1_0) {
                        scale = BlockTQ10TypeAccessor::get_scale(origin_data, idx, group_size);
                    } else if (tensor->type == GGML_TYPE_TQ2_0) {
                        scale = BlockTQ20TypeAccessor::get_scale(origin_data, idx, group_size);
                    } else {
                        GGML_LOG_ERROR("Unsupported type for get_scale: %s\n", ggml_type_name(tensor->type));
                    }

                    tmac_float_type zero_point;
                    if (get_type_has_zero_point(tensor->type)) {
                        if (is_tmac_2bit_type(tensor->type)) {
                            zero_point = BlockI2TypeAccessor::get_zero_point(int_n_zero_points, idx, group_size);
                        } else if (is_tmac_4bit_type(tensor->type)) {
                            zero_point = BlockI4TypeAccessor::get_zero_point(int_n_zero_points, idx, group_size);
                        } else {
                            GGML_LOG_ERROR("Unsupported type for get_zero_point: %s\n", ggml_type_name(tensor->type));
                        }
                    }

                    idx = idx / group_size;
                    int nb1 = k / group_size;
                    int nb0 = bm / bits * nb1;
                    int new_im = idx / nb0;
                    int new_ibm = (idx % nb0) / nb1;
                    int new_ik = (idx % nb1);

                    if (get_type_has_zero_point(tensor->type)) {
                        int new_isimd = new_ibm % simd_n_out;
                        int new_idx_outer = new_im * bm / bits * k / group_size / simd_n_out
                                          + new_ik * bm / bits / simd_n_out
                                          + new_ibm / simd_n_out;
                        int new_idx_scale = new_idx_outer * (simd_n_out * 2) + new_isimd;
                        int new_idx_zero = new_idx_outer * (simd_n_out * 2) + simd_n_out + new_isimd;

                        scales[new_idx_scale] = scale;
                        scales[new_idx_zero] = zero_point;
                    } else {
                        int new_idx = new_im * bm / bits * k / group_size + new_ik * bm / bits + new_ibm;
                        scales[new_idx] = scale;
                    }
                }
            }
        }

        delete[] buf2;
#else
        memset(qweights, 0x88, k * m / 8);
        for (int i = 0; i < scales_size; i++) {
            scales[i] = 1.0f;
        }
#endif
    }  // if (do_permutate(tensor->type))
}

void ggml_backend_tmac_convert_weight(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0 && size == ggml_tmac_get_nbytes(tensor)); // only full tensor conversion is supported for now
    ggml_tmac_transform_tensor(tensor, data);
}


/****** T-MAC compute ******/


// m = batch_size
// n = output_dim
// t-mac llama.cpp n and m swapped
static inline void ggml_tmac_mul_mat_task_init(void * src1, void * qlut, void * lut_scales, void * lut_biases, int n, int k, int m, int bits) {
    struct tmac_kernel_config * kernel_config = find_tmac_kernel_config(n, k, bits);
    if (kernel_config == nullptr) {
        throw std::runtime_error("ggml_tmac_mul_mat_task_init: Failed to find kernel config for m" + std::to_string(n) + "_k" + std::to_string(k) + "_b" + std::to_string(bits));
    }
    lut_ctor_int8_g4(src1, lut_scales, lut_biases, qlut, k, kernel_config);
}

static inline void ggml_tmac_mul_mat_task_compute(void * src0, void * scales, void * qlut, void * lut_scales, void * lut_biases, void * dst, int n, int k, int m, int bits) {
    struct tmac_kernel_config * kernel_config = find_tmac_kernel_config(n, k, bits);
    if (kernel_config == nullptr) {
        GGML_LOG_INFO("Failed to find kernel config for m%d_k%d_b%d\n", n, k, bits);
        throw std::runtime_error("ggml_tmac_mul_mat_task_compute: Failed to find kernel config for m" + std::to_string(n) + "_k" + std::to_string(k) + "_b" + std::to_string(bits));
    }
    qgemm_lut_int8_g4(src0, qlut, scales, lut_scales, lut_biases, dst, kernel_config->bm, k, m, kernel_config);
}


void ggml_backend_tmac_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int bits = get_type_bits(src0->type);
    // src0: weight,     ne00 = k, ne01 = n
    // src1: activation, ne10 = k, ne11 = m
    char * wdata = (char *) (params->wdata);

    struct tmac_tensor_extra * wt = ((struct ggml::cpu::tmac::tensor_traits *)src0->extra)->get_tmac_tensor_extra(src0->name);
    char * cur_wdata = wdata;
    tmac_float_type * tmac_f_ptr = (tmac_float_type *) wdata;
    if (sizeof(tmac_float_type) == 2) {
        cur_wdata = wdata + MAX(ne10, ne01) * ne11 * sizeof(tmac_float_type);
    };
    int8_t * qlut = (int8_t *) cur_wdata;
    tmac_float_type * lut_scales = (tmac_float_type *) (qlut + ne10 * ne11 * 4);
    tmac_float_type * lut_biases = (tmac_float_type *) (lut_scales + wt->lut_scales_size * ne11);

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    tmac_float_type * act_input;
    if (sizeof(tmac_float_type) == 2) {
        act_input = tmac_f_ptr;
    } else {
        act_input = (tmac_float_type *) src1->data;
    }

    for (int ine11 = ith; ine11 < ne11; ine11 += nth) {
        if constexpr (sizeof(tmac_float_type) == 2) {
            // TODO: can we reuse the src1->data memory?
            ggml_fp32_to_fp16_row((const float *) src1->data + ne10 * ine11, (ggml_fp16_t *) act_input + ne10 * ine11, ne10);
        }
        ggml_tmac_mul_mat_task_init(act_input + ne10 * ine11,
                                    qlut + ne10 * ine11 * 4,
                                    lut_scales + wt->lut_scales_size * ine11,
                                    lut_biases + wt->lut_scales_size * ine11,
                                    ne01, ne00, 1, bits);
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        ggml_threadpool_atomic_store_explicit(params->threadpool, nth);
        // atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

    tmac_float_type * act_output = (tmac_float_type *) (dst->data);
    if constexpr (sizeof(tmac_float_type) == 2) {
        act_output = tmac_f_ptr;
    }

    const int n_tile_num = wt->n_tile_num;
    // Currently, T-MAC requires ne0 devisible by n_tile_num
    GGML_ASSERT(ne0 % n_tile_num == 0);

    const int64_t w_size       = ne00 * ne01 * bits / 8;
    const int64_t w_chunk_size = w_size / n_tile_num;

    const int64_t nr0 = ne0;
    const int64_t nr1 = ne1 * ne2 * ne3;

    // Adopt the same style with current llama.cpp impl
    // But different chunk size for 0/1 dim.
    // No scrap.
    const int chunk_size0 = ne0 / n_tile_num;
    const int chunk_size1 = 8;  // TODO: tune in T-MAC

    // nchunk0 == n_tile_num
    int64_t nchunk0 = (nr0 + chunk_size0 - 1) / chunk_size0;
    int64_t nchunk1 = (nr1 + chunk_size1 - 1) / chunk_size1;

    int64_t dr0 = chunk_size0;
    int64_t dr1 = chunk_size1;
#if defined(TMAC_RECHUNK)
    // Rechunk
    if ((nchunk1 == 1) && (nchunk0 > nth * 4)) {
        // dr0 should be divisible by chunk_size0
        dr0 = (ne0 / (nth * 4) / chunk_size0) * chunk_size0;
        nchunk0 = (nr0 + dr0 - 1) / dr0;
    }
#endif

    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end   = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end   = MIN(ir1_start + dr1, nr1);

        // inline ggml_compute_forward_mul_mat_one_chunk here for simplicity
        for (int64_t ichunk0 = ir0_start / chunk_size0; ichunk0 < ir0_end / chunk_size0; ichunk0++) {
            const int64_t w_offset      = ichunk0 * w_chunk_size;
            const int64_t scales_offset = ichunk0 * wt->scales_size / n_tile_num;

            for (int64_t ine11 = ir1_start; ine11 < ir1_end; ine11++) {
                const int64_t qlut_offset       = ne10 * ine11 * 4;
                const int64_t lut_scales_offset = wt->lut_scales_size * ine11;
                const int64_t dst_offset        = ne0 * ine11 + ichunk0 * chunk_size0;

                ggml_tmac_mul_mat_task_compute(wt->qweights + w_offset,
                                                wt->scales + scales_offset,
                                                qlut + qlut_offset,
                                                lut_scales + lut_scales_offset,
                                                lut_biases + lut_scales_offset,
                                                act_output + dst_offset,
                                                ne01, ne00, 1, bits);
                if constexpr (sizeof(tmac_float_type) == 2) {
                    ggml_fp16_to_fp32_row((const ggml_fp16_t *) act_output + dst_offset, (float *) dst->data + dst_offset, chunk_size0);
                }
            }
        }

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        // current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
        current_chunk = ggml_threadpool_atomic_fetch_add_explicit(params->threadpool, 1);
    }
    return;
}

#endif  // GGML_USE_TMAC