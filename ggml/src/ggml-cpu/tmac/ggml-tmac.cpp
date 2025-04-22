#include <vector>
#include <unordered_map>
#include <type_traits>
#include <iostream>
#include <fstream>
#include <chrono>
#include "../../../../common/json.hpp"
#include "../../../../common/log.h"

#include "ggml-cpu.h"
#include "ggml-tmac.h"
#include "../../ggml-quants.h"

#define GGML_TMAC_MAX_NODES 8192

constexpr size_t kAllocAlignment = 64;

static bool initialized = false;

static tmac_tensor_extra * tmac_tensor_extras = nullptr;

static size_t tmac_tensor_extras_index = 0;

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

static struct llama_model_tmac_meta * tmac_model_meta = nullptr;

static int n_threads = 4;
static int batch_size = 64;

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

void ggml_tmac_init(void) {
    LOG_INF("ggml_tmac_init\n");

    if (initialized) {
        return;
    }
    initialized = true;

    if (tmac_tensor_extras == nullptr) {
        tmac_tensor_extras = new tmac_tensor_extra[GGML_TMAC_MAX_NODES];
    }
    tmac_tensor_extras_index = 0;
}

void ggml_tmac_free(void) {
    LOG_INF("ggml_tmac_free\n");

    if (!initialized) {
        return;
    }
    initialized = false;

    for (size_t i = 0; i < tmac_tensor_extras_index; i++) {
        // aligned_free(tmac_tensor_extras[i].qweights);
        // aligned_free(tmac_tensor_extras[i].scales);
    }
    delete[] tmac_tensor_extras;
    tmac_tensor_extras = nullptr;
}

static bool is_type_supported(enum ggml_type type) {
    if (type == GGML_TYPE_Q4_0 ||
        type == GGML_TYPE_I1 ||
        type == GGML_TYPE_I2 ||
        type == GGML_TYPE_I3 ||
        type == GGML_TYPE_I4 ||
        type == GGML_TYPE_TQ1_0 ||
        type == GGML_TYPE_TQ2_0) {
        return true;
    } else {
        return false;
    }
}

static bool do_permutate(enum ggml_type type) {
    return true;
    if (type == GGML_TYPE_I1 ||
        type == GGML_TYPE_I2 ||
        type == GGML_TYPE_I3 ||
        type == GGML_TYPE_I4) {
        // Add additional args to decide if permuted I2 or naive I2
        return false;
    } else {
        return true;
    }
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
        if (sizeof(tmac_float_type) == 2) {
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
        const float * ss = (const float *) data;
        float s = ss[idx / group_size];
        return (tmac_float_type) s;
    }

    static tmac_float_type get_zero_point(const void * data, int idx, int group_size) {
        const float * zs = (const float *) data;
        float z = zs[idx / group_size];
        return (tmac_float_type) z;
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
        const float * ss = (const float *) data;
        float s = ss[idx / group_size];
        return (tmac_float_type) s;
    }

    static tmac_float_type get_zero_point(const void * data, int idx, int group_size) {
        const float * zs = (const float *) data;
        float z = zs[idx / group_size];
        return (tmac_float_type) z;
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
        if (sizeof(tmac_float_type) == 2) {
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
        if (sizeof(tmac_float_type) == 2) {
            tmac_float_type * fp16dp = reinterpret_cast<tmac_float_type *>(&d);
            return *fp16dp;
        } else {
            return ggml_fp16_to_fp32(d);
        }
    }
};

bool ggml_tmac_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        strcmp(src0->name, "token_embd.weight") &&  // means not equal
        strcmp(src0->name, "output.weight")) {
        return true;
    }
    return false;
}

size_t ggml_tmac_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t n = src0->ne[1];    // llama.cpp n
    const size_t k = src1->ne[0];    // k
    const size_t m = src1->ne[1];    // llama.cpp m
    const int bits = ggml_tmac_get_type_bits(src0->type);

    struct tmac_kernel_config * kernel_config = find_tmac_kernel_config(n, k, bits);
    if (kernel_config == nullptr) {
        ggml_tmac_tune_kernel_config(n, k);
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

// m = batch_size
// n = output_dim
// t-mac llama.cpp n and m swapped
void ggml_tmac_mul_mat_task_init(void * src1, void * qlut, void * lut_scales, void * lut_biases, int n, int k, int m, int bits) {
    struct tmac_kernel_config * kernel_config = find_tmac_kernel_config(n, k, bits);
    if (kernel_config == nullptr) {
        throw std::runtime_error("ggml_tmac_mul_mat_task_init: Failed to find kernel config for m" + std::to_string(n) + "_k" + std::to_string(k) + "_b" + std::to_string(bits));
    }
    lut_ctor_int8_g4(src1, lut_scales, lut_biases, qlut, k, kernel_config);
}

void ggml_tmac_mul_mat_task_compute(void * src0, void * scales, void * qlut, void * lut_scales, void * lut_biases, void * dst, int n, int k, int m, int bits) {
    struct tmac_kernel_config * kernel_config = find_tmac_kernel_config(n, k, bits);
    if (kernel_config == nullptr) {
        LOG_ERR("Failed to find kernel config for m%d_k%d_b%d\n", n, k, bits);
        throw std::runtime_error("ggml_tmac_mul_mat_task_compute: Failed to find kernel config for m" + std::to_string(n) + "_k" + std::to_string(k) + "_b" + std::to_string(bits));
    }
    qgemm_lut_int8_g4(src0, qlut, scales, lut_scales, lut_biases, dst, kernel_config->bm, k, m, kernel_config);
}

int ggml_tmac_get_scales_size(const struct tmac_kernel_config * kernel_config, int m, int k) {
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

size_t ggml_tmac_get_nbytes(const struct ggml_tensor * tensor) {
    const int bits = ggml_tmac_get_type_bits(tensor->type);

    int k = tensor->ne[0];
    int m = tensor->ne[1];  // `n` in llama.cpp

    struct tmac_kernel_config * kernel_config = find_tmac_kernel_config(m, k, bits);
    if (kernel_config == nullptr) {
        ggml_tmac_tune_kernel_config(m, k);
        kernel_config = find_tmac_kernel_config(m, k, bits);
    }

    const int scales_size = ggml_tmac_get_scales_size(kernel_config, m, k);
    // Currently, always uses float to store scales or zero points
    size_t nbytes = k * m / 8 * bits + scales_size * sizeof(float);
    // printf("ggml_tmac_get_nbytes: %s --- k=%d, m=%d, w=%d, sc=%d, nbytes: %zu\n", tensor->name, k, m, k * m / 8 * bits, scales_size, nbytes);
    return nbytes;
}

template<typename T>
void save_array_to_file(const char * tensor_name, const char * suffix, T * array, uint64_t size) {
    std::string w_filename = std::string(tensor_name) + "_" + std::string(suffix) + ".bin";
    std::ofstream outFile(w_filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Cannot open file for writing." << std::endl;
        return;
    }
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
    outFile.write(reinterpret_cast<const char*>(array), size * sizeof(T));
    outFile.close();
}


void ggml_tmac_transform_tensor(struct ggml_tensor * tensor) {
    if (!(is_type_supported(tensor->type) && tensor->extra == nullptr)) {
        return;
    }

    const int bits = ggml_tmac_get_type_bits(tensor->type);
    int k = tensor->ne[0];
    int m = tensor->ne[1];  // `n` in llama.cpp

    struct tmac_kernel_config * kernel_config = find_tmac_kernel_config(m, k, bits);
    if (kernel_config == nullptr) {
        ggml_tmac_tune_kernel_config(m, k);
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

    LOG_DBG("Transforming tensor: %s (m: %d, k: %d, bits: %d)\n", tensor->name, m, k, bits);
    LOG_DBG("kcfg (bm=%d, simd_n_in=%d, simd_n_out=%d, kfactor=%d, group_size=%d, lut_scales_size=%d, scales_size=%d, n_tile_num=%d)\n",
        bm, simd_n_in, simd_n_out, kfactor, group_size, lut_scales_size, scales_size, n_tile_num);
    if (bm == 0) {
        if (!strcmp(tensor->name, "token_embd.weight") || !strcmp(tensor->name, "output.weight")) {
            LOG_WRN("Do not find kcfg for %s. Consider compiling T-MAC kernel for it if vocab size is a multiply of 128 or 320, detected %lld.\n", tensor->name, tensor->ne[1]);
            return;
        }
        else {
            // TODO: Instead of fatal error, try to avoid using t-mac?
            LOG_ERR("Failed to find kcfg. Abort transforming\n");
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

    tensor->extra = tmac_tensor_extras + tmac_tensor_extras_index;
    tmac_tensor_extras[tmac_tensor_extras_index++] = {
        /* .lut_scales_size = */ lut_scales_size,
        /* .scales_size     = */ scales_size,
        /* .n_tile_num      = */ n_tile_num,
        /* .qweights        = */ qweights,
        /* .scales          = */ scales
    };

    if (do_permutate(tensor->type)) {
// for fast testing
// #define TMAC_EMPTY_WEIGHTS
#ifndef TMAC_EMPTY_WEIGHTS
        // TODO: optimize to accelerate weights loading
        uint8_t * buf2 = new uint8_t[m * k / g];
        memset(buf2, 0, m * k / g);

        // # (M // bits, K, bits)
        // w = np.stack([(w >> ib) & 1 for ib in range(bits)], axis=-1)
        // # (M // bits, K, bits) -> (M // bits, bits, K) -> (M // bits, bits, K // g, g) -> (M // bits, bits, K // g)
        // w = w.transpose(0, 2, 1).reshape(M // bits, bits, K // g, g)
        // w = sum([(w[:, :, :, ig] << ig) for ig in range(g)])
        for (int im = 0; im < m / bits; im++) {
            for (int ik = 0; ik < k; ik++) {
                uint8_t v;
                if (tensor->type == GGML_TYPE_Q4_0) {
                    v = BlockQ40TypeAccessor::get_q(tensor->data, im * k + ik);
                } else if (tensor->type == GGML_TYPE_I2) {
                    v = BlockI2TypeAccessor::get_q(tensor->data, im * k + ik);
                } else if (tensor->type == GGML_TYPE_I4) {
                    v = BlockI4TypeAccessor::get_q(tensor->data, im * k + ik);
                } else if (tensor->type == GGML_TYPE_TQ1_0) {
                    v = BlockTQ10TypeAccessor::get_q(tensor->data, im * k + ik);
                } else if (tensor->type == GGML_TYPE_TQ2_0) {
                    v = BlockTQ20TypeAccessor::get_q(tensor->data, im * k + ik);
                } else {
                    LOG_ERR("Unsupported type\n");
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
        for (int im = 0; im < m / bits; im++) {
            for (int ib = 0; ib < bits; ib++) {
                for (int ik = 0; ik < k / g; ik++) {
                    int new_im = im / simd_n_out;
                    int new_isno = im % simd_n_out;
                    int new_ib = ib;
                    int new_ik = ik;
                    // w = w.reshape(M // bits // simd_n_out, simd_n_out, bits, K // g).transpose(0, 2, 1, 3)
                    int new_idx = new_im * bits * simd_n_out * k / g + new_ib * simd_n_out * k / g + new_isno * k / g + new_ik;
                    // w = w.reshape(M // mgroup, ngroups_per_elem, simd_n_in, K // g).transpose(0, 2, 1, 3)
                    int nb2 = k / g;
                    int nb1 = simd_n_in * nb2;
                    int nb0 = ngroups_per_elem * nb1;
                    new_im = new_idx / nb0;
                    int new_ing = (new_idx % nb0) / nb1;
                    int new_isni = (new_idx % nb1) / nb2;
                    new_ik = (new_idx % nb2);
                    new_idx = new_im * ngroups_per_elem * simd_n_in * k / g + new_isni * ngroups_per_elem * k / g + new_ing * k / g + new_ik;
                    // #             0        1             2             3                 4                  5
                    // w = w.reshape(M // bm, bm // mgroup, simd_n_in, ngroups_per_elem, K // g // kfactor, kfactor).transpose(0, 4, 1, 5, 2, 3)
                    int nb4 = kfactor;
                    int nb3 = k / g / kfactor * nb4;
                    nb2 = ngroups_per_elem * nb3;
                    nb1 = simd_n_in * nb2;
                    nb0 = bm / mgroup * nb1;
                    new_im = new_idx / nb0;
                    int new_ibm = (new_idx % nb0) / nb1;
                    new_isni = (new_idx % nb1) / nb2;
                    new_ing = (new_idx % nb2) / nb3;
                    new_ik = (new_idx % nb3) / nb4;
                    int new_ikf = (new_idx % nb4);
                    new_idx = new_im * k / g / kfactor * bm / mgroup * kfactor * simd_n_in * ngroups_per_elem +
                            new_ik * bm / mgroup * kfactor * simd_n_in * ngroups_per_elem +
                            new_ibm * kfactor * simd_n_in * ngroups_per_elem +
                            new_ikf * simd_n_in * ngroups_per_elem +
                            new_isni * ngroups_per_elem +
                            new_ing;
                    new_idx = new_idx / ngroups_per_elem;
                    // w = sum([(w[:, :, :, :, :, ng] << (ng * g)) for ng in range(ngroups_per_elem)])
                    qweights[new_idx] += buf2[im * bits * k / g + ib * k / g + ik] << (new_ing * g);
                }
            }
        }

        const float * int_n_scales = (const float * ) ((const uint8_t *) tensor->data + k * m / 8);
        const float * int_n_zero_points = int_n_scales + scales_size / 2;

        if (scales_size < m / bits) {  // BitNet-like scale (m_groups,)
            for (int i = 0; i < scales_size; i++) {
                scales[i] = (tmac_float_type) int_n_scales[i];
            }
        } else {
            // scales = scales.reshape(M // bm, bm // bits, K // group_size).transpose(0, 2, 1)
            for (int im = 0; im < m / bits; im += 1) {
                for (int ik = 0; ik < k; ik += group_size) {
                    tmac_float_type scale;
                    int idx = im * k + ik;
                    if (tensor->type == GGML_TYPE_Q4_0) {
                        scale = BlockQ40TypeAccessor::get_scale(tensor->data, idx);
                    } else if (tensor->type == GGML_TYPE_I2) {
                        scale = BlockI2TypeAccessor::get_scale(int_n_scales, idx, group_size);
                    } else if (tensor->type == GGML_TYPE_I4) {
                        scale = BlockI4TypeAccessor::get_scale(int_n_scales, idx, group_size);
                    } else if (tensor->type == GGML_TYPE_TQ1_0) {
                        scale = BlockTQ10TypeAccessor::get_scale(tensor->data, idx, group_size);
                    } else if (tensor->type == GGML_TYPE_TQ2_0) {
                        scale = BlockTQ20TypeAccessor::get_scale(tensor->data, idx, group_size);
                    } else {
                        LOG_ERR("Unsupported type\n");
                    }

                    tmac_float_type zero_point;
                    if (tmac_model_meta->has_zero_point) {
                        if (tensor->type == GGML_TYPE_I2) {
                            zero_point = BlockI2TypeAccessor::get_zero_point(int_n_zero_points, idx, group_size);
                        } else if (tensor->type == GGML_TYPE_I4) {
                            zero_point = BlockI4TypeAccessor::get_zero_point(int_n_zero_points, idx, group_size);
                        } else {
                            LOG_ERR("Unsupported type\n");
                        }
                    }

                    idx = idx / group_size;
                    int nb1 = k / group_size;
                    int nb0 = bm / bits * nb1;
                    int new_im = idx / nb0;
                    int new_ibm = (idx % nb0) / nb1;
                    int new_ik = (idx % nb1);

                    if (tmac_model_meta->has_zero_point) {
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

int ggml_tmac_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_I1:
            return 1;
        case GGML_TYPE_I2:
            return 2;
        case GGML_TYPE_I3:
            return 3;
        case GGML_TYPE_I4:
            return 4;
        case GGML_TYPE_Q4_0:
            return 4;
        case GGML_TYPE_TQ1_0:
            return 2;
        case GGML_TYPE_TQ2_0:
            return 2;
        default:
            return 0;
    }
}

inline void ggml_tmac_forward_mul_mat(
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
    const int m = bm / bits;
    const int scales_size = ggml_tmac_get_scales_size(kernel_config, M, K);

    std::chrono::duration<double> total_elapsed = std::chrono::duration<double>::zero();
    LOG_DBG("Run single kernel config: M=%d, N=%d, K=%d, bm=%d, kfactor=%d, actk=%d\n", M, N, K, bm, kernel_config->kfactor, kernel_config->actk);
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

void ggml_tmac_tune_kernel_config(int M, int K) {
    const int bits = tmac_model_meta->bits;
    struct tmac_kernel_config * existing_kcfg = find_tmac_kernel_config(M, K, bits);
    if (existing_kcfg != nullptr) {
        return;
    }

    struct tmac_kernel_config kernel_config;
    {
        kernel_config.g = tmac_model_meta->g;
        kernel_config.ngroups_per_elem = tmac_model_meta->ngroups_per_elem;
        kernel_config.q_group_size = tmac_model_meta->q_group_size;
        kernel_config.act_group_size = tmac_model_meta->act_group_size;

        kernel_config.has_scale = tmac_model_meta->has_scale;
        // kfactor to be tuned
        kernel_config.bits = bits;
        kernel_config.actk = tmac_model_meta->actk;
        kernel_config.has_zero_point = tmac_model_meta->has_zero_point;
        kernel_config.one_scale = tmac_model_meta->one_scale;

        // bm to be tuned
        kernel_config.simd_n_in = 16;
        kernel_config.simd_n_out = 8;

        kernel_config.chunk_n = 8;
    }

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
    struct tmac_kernel_config best_kcfg;
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
                if (kfactor < kernel_config.actk) {
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
                LOG_INF("Tuned kernel config: [thread=%d] M=%d, N=%d, K=%d, bm=%d, n=%d, kfactor=%d, bits=%d, g=%d, ngroups_per_elem=%d, q_group_size=%d, act_group_size=%d\t TIME: %.4f ms\n",
                                n_threads, M, N, K, bm, n, kfactor, bits, kernel_config.g, kernel_config.ngroups_per_elem, kernel_config.q_group_size, kernel_config.act_group_size, this_time);
                if (this_time < min_time) {
                    min_time = this_time;
                    best_kcfg = kernel_config;
                }
            }
        }
    }

    // Save the results
    insert_or_assign_tmac_kernel_config(M, K, bits, best_kcfg);
}


/* GGML backend functions */

bool qtype_has_tmac_kernels(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_I1:
        case GGML_TYPE_I2:
        case GGML_TYPE_I3:
        case GGML_TYPE_I4:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
            return true;
        default:
            return false;
    }
}

bool tmac_meta_init(const char * tmac_meta_fname) {
    tmac_model_meta = new llama_model_tmac_meta();

    std::ifstream file(tmac_meta_fname);
    
    if (!file.is_open()) {
        LOG_ERR("Failed to open file: %s\n", tmac_meta_fname);
        return false;
    }

    // Parse the JSON content from the file 
    nlohmann::json j;
    try {
        file >> j;  // Read the JSON data into the object
    } catch (const nlohmann::json::parse_error& e) {
        LOG_ERR("JSON parsing error: %s\n", e.what());
        return false;
    }

    // Load the values into the struct
    tmac_model_meta->bits = j["bits"].get<int>();
    tmac_model_meta->q_group_size = j["group_size"].get<int>();
    tmac_model_meta->has_scale = j["has_scale"].get<bool>();
    tmac_model_meta->has_zero_point = j["has_zero_point"].get<bool>();
    tmac_model_meta->one_scale = j["one_scale"].get<bool>();
    tmac_model_meta->quant_method = new char[64];
    strncpy(tmac_model_meta->quant_method, j["quant_method"].get<std::string>().c_str(), sizeof(tmac_model_meta->quant_method) - 1);
    tmac_model_meta->quant_method[sizeof(tmac_model_meta->quant_method) - 1] = '\0';  // Ensure null termination

    // Fixed features
    tmac_model_meta->g = 4;
    tmac_model_meta->ngroups_per_elem = 8 / tmac_model_meta->g;
    if (tmac_model_meta->q_group_size % 64 == 0) {
        tmac_model_meta->act_group_size = 64;
    } else if (tmac_model_meta->q_group_size % 32 == 0) {
        tmac_model_meta->act_group_size = 32;
    } else {
        LOG_ERR("Unsupported activation group size: %d\n", tmac_model_meta->q_group_size);
    }
    tmac_model_meta->actk = tmac_model_meta->act_group_size / tmac_model_meta->g;

    return true;
}

