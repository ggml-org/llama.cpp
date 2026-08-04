#include "mmq.cuh"
#include "moe-mmq-mxfp8.cuh"
#include "moe-mmq-tma.cuh"
#include "unary.cuh"

#include <cstring>

#if CUDART_VERSION >= 12080

static constexpr int moe_tma_threads                    = 256;
static constexpr int moe_tma_warp_size                  = 32;
static constexpr int moe_tma_cooperative_rows           = 128;
static constexpr int moe_tma_specialized_rows           = 96;
static constexpr int moe_tma_specialized_consumer_warps = 6;
static constexpr int moe_tma_w13_rows                   = 64;
static constexpr int moe_tma_w13_consumer_warps         = 4;
static constexpr int moe_tma_record_ints                = ggml_cuda_moe_tma_record_bytes / sizeof(int);

static __device__ __forceinline__ void moe_tma_barrier_init(uint64_t * barrier) {
#    if defined(BLACKWELL_MMA_AVAILABLE)
    const uint32_t address = ggml_cuda_cvta_generic_to_shared(barrier);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(address) : "memory");
#    else
    GGML_UNUSED(barrier);
    NO_DEVICE_CODE;
#    endif
}

static __device__ __forceinline__ uint64_t moe_tma_barrier_arrive(uint64_t * barrier, uint32_t bytes) {
#    if defined(BLACKWELL_MMA_AVAILABLE)
    const uint32_t address = ggml_cuda_cvta_generic_to_shared(barrier);
    uint64_t       state;
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
                 : "=l"(state)
                 : "r"(address), "r"(bytes)
                 : "memory");
    return state;
#    else
    GGML_UNUSED_VARS(barrier, bytes);
    NO_DEVICE_CODE;
    return 0;
#    endif
}

static __device__ __forceinline__ void moe_tma_barrier_wait(uint64_t * barrier, uint64_t state) {
#    if defined(BLACKWELL_MMA_AVAILABLE)
    const uint32_t address  = ggml_cuda_cvta_generic_to_shared(barrier);
    uint32_t       complete = 0;
    while (!complete) {
        asm volatile(
            "{\n\t"
            ".reg .pred done;\n\t"
            "mbarrier.try_wait.acquire.cta.shared::cta.b64 done, [%1], %2;\n\t"
            "selp.b32 %0, 1, 0, done;\n\t"
            "}"
            : "=r"(complete)
            : "r"(address), "l"(state)
            : "memory");
    }
#    else
    GGML_UNUSED_VARS(barrier, state);
    NO_DEVICE_CODE;
#    endif
}

static __device__ __forceinline__ uint8_t moe_tma_e8m0_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    constexpr int fp4_e2m1_emax = 2;
    const int     shared_exp    = __float2int_rn(log2f(amax)) - fp4_e2m1_emax;
    return (uint8_t) max(0, min(254, shared_exp + 127));
}

static __device__ __forceinline__ uint8_t moe_tma_mxfp8_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    constexpr float e4m3_max = 448.0f;
    const int       exponent = __float2int_ru(log2f(amax / e4m3_max));
    return (uint8_t) max(0, min(254, exponent + 127));
}

template <bool compact_tail>
static __device__ __forceinline__ void moe_tma_load_weight(void *              dst,
                                                           const CUtensorMap * map,
                                                           int                 row,
                                                           int                 k_tile,
                                                           int                 expert,
                                                           int                 rows_padded,
                                                           int                 k_tiles,
                                                           uint32_t            bytes,
                                                           uint64_t *          barrier,
                                                           uint64_t *          token) {
#    if defined(BLACKWELL_MMA_AVAILABLE)
    *token                         = moe_tma_barrier_arrive(barrier, bytes);
    const uint32_t dst_address     = ggml_cuda_cvta_generic_to_shared(dst);
    const uint32_t barrier_address = ggml_cuda_cvta_generic_to_shared(barrier);
    if constexpr (compact_tail) {
        asm volatile(
            "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(dst_address),
            "l"(map), "r"(0), "r"(row), "r"(k_tile), "r"(expert), "r"(barrier_address)
            : "memory");
    } else {
        const int record = (expert * k_tiles + k_tile) * rows_padded + row;
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];" ::"r"(dst_address),
            "l"(map), "r"(0), "r"(record), "r"(barrier_address)
            : "memory");
    }
#    else
    GGML_UNUSED_VARS(dst, map, row, k_tile, expert, rows_padded, k_tiles, bytes, barrier, token);
    NO_DEVICE_CODE;
#    endif
}

static __device__ __forceinline__ void moe_tma_load_activation(void *       dst,
                                                               const void * src,
                                                               uint32_t     bytes,
                                                               uint64_t *   barrier,
                                                               uint64_t *   token) {
#    if defined(BLACKWELL_MMA_AVAILABLE)
    *token                             = moe_tma_barrier_arrive(barrier, bytes);
    const uint32_t     dst_address     = ggml_cuda_cvta_generic_to_shared(dst);
    const uint32_t     barrier_address = ggml_cuda_cvta_generic_to_shared(barrier);
    const uint64_t     src_address     = (uint64_t) src;
    constexpr uint32_t max_bulk_bytes  = 16 * 1024;
    const uint32_t     first           = min(bytes, max_bulk_bytes);
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst_address),
        "l"(src_address), "r"(first), "r"(barrier_address)
        : "memory");
    if (bytes > first) {
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                         dst_address + first),
                     "l"(src_address + first), "r"(bytes - first), "r"(barrier_address)
                     : "memory");
    }
#    else
    GGML_UNUSED_VARS(dst, src, bytes, barrier, token);
    NO_DEVICE_CODE;
#    endif
}

template <int I>
static __device__ __forceinline__ void moe_tma_load_tail(char *       dst,
                                                         const char * data,
                                                         int          expert,
                                                         int          row,
                                                         int          rows,
                                                         int          tail_blocks,
                                                         int64_t      expert_stride,
                                                         int64_t      tail_offset) {
    const int linear_thread = threadIdx.y * moe_tma_warp_size + threadIdx.x;
    const char * tail = data + (int64_t) expert * expert_stride + tail_offset;
    for (int index = linear_thread; index < I * tail_blocks; index += moe_tma_threads) {
        const int row_in_tile = index / tail_blocks;
        const int block = index - row_in_tile * tail_blocks;
        char * record = dst + row_in_tile * ggml_cuda_moe_tma_record_bytes;
        char * qs = record + block * (QK_MXFP4 / 2);
        uint8_t * scale = reinterpret_cast<uint8_t *>(record) + ggml_cuda_moe_tma_data_bytes + block;
        if (row + row_in_tile < rows) {
            const block_mxfp4 * value = reinterpret_cast<const block_mxfp4 *>(
                tail + ((int64_t) (row + row_in_tile) * tail_blocks + block) * sizeof(block_mxfp4));
            memcpy(qs, value->qs, QK_MXFP4 / 2);
            *scale = value->e;
        } else {
            memset(qs, 0, QK_MXFP4 / 2);
            *scale = 0;
        }
    }
}

template <int J, int I, int consumer_warps>
static __device__ __forceinline__ void moe_tma_vec_dot(const int * __restrict__ x,
                                                       const int * __restrict__ y,
                                                       float * __restrict__ sum,
                                                       int k00,
                                                       int nfrags_valid = MMQ_TILE_NE_K / 8) {
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int rows_per_warp = J >= 48 ? 32 : 16;
    constexpr int ntx           = rows_per_warp / tile_C::I;
    constexpr int nfrags        = MMQ_TILE_NE_K / tile_A::J;
    static_assert(I == consumer_warps * tile_C::I, "invalid TMA consumer shape");

    const int warp = threadIdx.y;
    y += (warp % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int *      x_qs   = x;
    const uint8_t *  x_sc   = reinterpret_cast<const uint8_t *>(x + 2 * MMQ_TILE_NE_K);
    const int *      y_qs   = y + 4;
    const uint32_t * y_sc   = reinterpret_cast<const uint32_t *>(y);
    const int        tidx_A = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
    const int        tidx_B = threadIdx.x / 4;
    const int        i0     = (warp / ntx) * rows_per_warp;

    tile_A   A[ntx][nfrags];
    uint32_t scale_A[ntx][nfrags];

#    pragma unroll
    for (int n = 0; n < ntx; ++n) {
#    pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            if (frag >= nfrags_valid) {
                continue;
            }
            const int k0  = k00 + frag * tile_A::J;
            const int row = i0 + n * tile_A::I;
            load_ldmatrix(A[n][frag], x_qs + row * moe_tma_record_ints + k0, moe_tma_record_ints);
            const uint8_t * scales = x_sc + (row + tidx_A) * ggml_cuda_moe_tma_record_bytes;
            const int       scale  = 2 * (k0 / tile_A::J);
            scale_A[n][frag]       = (uint32_t) scales[scale] | ((uint32_t) scales[scale + 1] << 8);
        }
    }

#    pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
        tile_B   B[nfrags];
        uint32_t scale_B[nfrags];

#    pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            if (frag >= nfrags_valid) {
                continue;
            }
            const int k0 = frag * tile_B::J;
            load_generic(B[frag], y_qs + j0 * MMQ_TILE_Y_K + k0, MMQ_TILE_Y_K);
            scale_B[frag] = y_sc[(j0 + tidx_B) * MMQ_TILE_Y_K + frag];
        }

#    pragma unroll
        for (int n = 0; n < ntx; ++n) {
#    pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                if (frag >= nfrags_valid) {
                    continue;
                }
                tile_C C = {};
                mma_block_scaled_fp4<GGML_TYPE_MXFP4>(C, A[n][frag], B[frag], scale_A[n][frag], scale_B[frag]);
#    pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += C.x[l];
                }
            }
        }
    }
}

template <int J, int I, int consumer_warps>
static __device__ __forceinline__ void moe_tma_vec_dot_mxfp8(const char * __restrict__ x,
                                                             const block_mxfp8_mmq * __restrict__ y,
                                                             float * __restrict__ sum,
                                                             int k_block) {
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int rows_per_warp = J >= 48 ? 32 : 16;
    constexpr int ntx           = rows_per_warp / tile_C::I;
    static_assert(I == consumer_warps * tile_C::I, "invalid TMA MXFP8 consumer shape");

    const int       warp   = threadIdx.y;
    const int       lane   = threadIdx.x;
    const int       group  = lane / 4;
    const int       thread = lane % 4;
    const int       i0     = (warp / ntx) * rows_per_warp;
    const uint8_t * x_qs   = reinterpret_cast<const uint8_t *>(x);
    const uint8_t * x_sc   = x_qs + ggml_cuda_moe_tma_data_bytes;

    tile_A   A[ntx];
    uint32_t scale_A[ntx];
#    pragma unroll
    for (int n = 0; n < ntx; ++n) {
        uint32_t * values = reinterpret_cast<uint32_t *>(A[n].x);
#    pragma unroll
        for (int r = 0; r < 4; ++r) {
            values[r] = 0;
        }
#    pragma unroll
        for (int element = 0; element < 16; ++element) {
            const int     row   = i0 + n * tile_C::I + group + ((element >= 4 && element < 8) || element >= 12 ? 8 : 0);
            const int     k     = k_block * 32 + thread * 4 + (element & 3) + (element >= 8 ? 16 : 0);
            const int     block = k / QK_MXFP4;
            const int     within = k % QK_MXFP4;
            const uint8_t packed = x_qs[row * ggml_cuda_moe_tma_record_bytes + block * (QK_MXFP4 / 2) + (within & 15)];
            const uint8_t code   = (packed >> (within >= 16 ? 4 : 0)) & 0x0F;
            values[element / 4] |= (uint32_t) (code << 2) << (8 * (element & 3));
        }
        const int scale_row = i0 + n * tile_C::I + lane / 4 + (lane % 2) * 8;
        scale_A[n]          = x_sc[scale_row * ggml_cuda_moe_tma_record_bytes + k_block];
    }

#    pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
        const int  j_base = j0 + (warp % ntx) * tile_C::J;
        tile_B     B;
        uint32_t * values = reinterpret_cast<uint32_t *>(B.x);
        values[0]         = 0;
        values[1]         = 0;
#    pragma unroll
        for (int element = 0; element < 8; ++element) {
            const int k = thread * 4 + (element & 3) + (element >= 4 ? 16 : 0);
            values[element / 4] |= (uint32_t) y[j_base + group].qs[k] << (8 * (element & 3));
        }
        const uint32_t scale_B = y[j_base + group].scale;

#    pragma unroll
        for (int n = 0; n < ntx; ++n) {
            tile_C C = {};
            mma_block_scaled_mxfp4_mxfp8(C, A[n], B, scale_A[n], scale_B);
#    pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                sum[(j0 / tile_C::J + n) * tile_C::ne + l] += C.x[l];
            }
        }
    }
}

template <int J, int I, int consumer_warps>
static __device__ __forceinline__ void moe_tma_write_back(const float * __restrict__ sum,
                                                          const int32_t * __restrict__ ids_dst,
                                                          float * __restrict__ dst,
                                                          int stride,
                                                          int i_max,
                                                          int j_max) {
    typedef tile<16, 8, int> tile_C;
    constexpr int            rows_per_warp = J >= 48 ? 32 : 16;
    constexpr int            ntx           = rows_per_warp / tile_C::I;
    static_assert(I == consumer_warps * tile_C::I, "invalid TMA writeback shape");

    const int warp = threadIdx.y;
    const int i0   = (warp / ntx) * (ntx * tile_C::I);
#    pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
#    pragma unroll
        for (int n = 0; n < ntx; ++n) {
#    pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (warp % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if (j <= j_max && i <= i_max) {
                    dst[ids_dst[j] * stride + i] = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
                }
            }
        }
    }
}

template <int J, int I, int consumer_warps>
static __device__ __forceinline__ void moe_tma_write_back_w2(const float * __restrict__ sum,
                                                             const int32_t * __restrict__ ids_dst,
                                                             const float * __restrict__ bias,
                                                             const float * __restrict__ route_weights,
                                                             float * __restrict__ dst,
                                                             int  expert,
                                                             int  width,
                                                             int  n_expert_used,
                                                             int  i_max,
                                                             int  j_max,
                                                             bool atomic_reduce) {
    typedef tile<16, 8, int> tile_C;
    constexpr int            rows_per_warp = J >= 48 ? 32 : 16;
    constexpr int            ntx           = rows_per_warp / tile_C::I;
    static_assert(I == consumer_warps * tile_C::I, "invalid TMA W2 writeback shape");

    const int warp = threadIdx.y;
    const int i0   = (warp / ntx) * (ntx * tile_C::I);
#    pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
#    pragma unroll
        for (int n = 0; n < ntx; ++n) {
#    pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (warp % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if (j <= j_max && i <= i_max) {
                    const int   route_row = ids_dst[j];
                    const float value =
                        __fadd_rn(sum[(j0 / tile_C::J + n) * tile_C::ne + l], bias[(int64_t) expert * width + i]);
                    const float weighted = __fmul_rn(value, route_weights[route_row]);
                    if (atomic_reduce) {
                        atomicAdd(dst + (int64_t) (route_row / n_expert_used) * width + i, weighted);
                    } else {
                        dst[(int64_t) route_row * width + i] = weighted;
                    }
                }
            }
        }
    }
}

template <int J, int I, int consumer_warps>
static __device__ __forceinline__ void moe_tma_store_shared(const float * __restrict__ sum,
                                                            float * __restrict__ dst,
                                                            int i_max,
                                                            int j_max) {
    typedef tile<16, 8, int> tile_C;
    constexpr int            rows_per_warp = J >= 48 ? 32 : 16;
    constexpr int            ntx           = rows_per_warp / tile_C::I;
    static_assert(I == consumer_warps * tile_C::I, "invalid TMA shared-store shape");

    const int warp = threadIdx.y;
    const int i0   = (warp / ntx) * (ntx * tile_C::I);
#    pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
#    pragma unroll
        for (int n = 0; n < ntx; ++n) {
#    pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (warp % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if (j <= j_max && i <= i_max) {
                    dst[j * I + i] = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
                }
            }
        }
    }
}

template <int J>
static __device__ __forceinline__ void moe_tma_store_swiglu(const float * __restrict__ sum,
                                                            const float * __restrict__ bias,
                                                            float * __restrict__ gate,
                                                            int expert,
                                                            int output_row,
                                                            int n_ff,
                                                            int j_max) {
    typedef tile<16, 8, int> tile_C;
    constexpr int            I             = moe_tma_w13_rows;
    constexpr int            rows_per_warp = J >= 48 ? 32 : 16;
    constexpr int            ntx           = rows_per_warp / tile_C::I;

    const int warp = threadIdx.y;
    const int i0   = (warp / ntx) * (ntx * tile_C::I);
#    pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
#    pragma unroll
        for (int n = 0; n < ntx; ++n) {
#    pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (warp % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if (j <= j_max && output_row + i < n_ff) {
                    const int64_t bias_base  = (int64_t) expert * 2 * n_ff;
                    const float   gate_value = __fadd_rn(gate[j * I + i], bias[bias_base + output_row + i]);
                    const float   up_value =
                        __fadd_rn(sum[(j0 / tile_C::J + n) * tile_C::ne + l], bias[bias_base + n_ff + output_row + i]);
                    gate[j * I + i] = __fadd_rn(ggml_cuda_op_swiglu_oai_single(gate_value, up_value), 0.0f);
                }
            }
        }
    }
}

template <int J>
static __device__ __forceinline__ void moe_tma_quantize_w13(const float * __restrict__ activation,
                                                            void * __restrict__ dst,
                                                            int64_t n_rows,
                                                            int     sorted_row_0,
                                                            int     output_row,
                                                            int     n_ff,
                                                            int     activation_q_ne0,
                                                            int     j_max) {
    constexpr int I = moe_tma_w13_rows;

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    for (int j = warp; j <= j_max; j += 8) {
        const int group         = lane / 4;
        const int lane_in_group = lane % 4;
        const int base          = group * 2;
        uint8_t   scales[2];
        char2     packed[2];

#    pragma unroll
        for (int b = 0; b < 2; ++b) {
            const float value = activation[j * I + b * 32 + lane];
            float       amax  = fabsf(value);
#    pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, moe_tma_warp_size));
            }

            const uint8_t   e      = moe_tma_e8m0_scale(amax);
            const float     inv    = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(e));
            const float     scaled = value * inv;
            const float     v0     = __shfl_sync(0xFFFFFFFF, scaled, base, moe_tma_warp_size);
            const float     v1     = __shfl_sync(0xFFFFFFFF, scaled, base + 16, moe_tma_warp_size);
            const float     v2     = __shfl_sync(0xFFFFFFFF, scaled, base + 1, moe_tma_warp_size);
            const float     v3     = __shfl_sync(0xFFFFFFFF, scaled, base + 17, moe_tma_warp_size);
            __nv_fp4x4_e2m1 fp4_packed(make_float4(v0, v1, v2, v3));
            scales[b] = e;
            packed[b] = *reinterpret_cast<char2 *>(&fp4_packed);
        }

        const int       k_block = output_row / QK_FP4_MMQ;
        const int       quad    = (output_row % QK_FP4_MMQ) / I;
        block_fp4_mmq * yb      = (block_fp4_mmq *) dst + (int64_t) k_block * n_rows + sorted_row_0 + j;
        char2 *         yqs     = reinterpret_cast<char2 *>(yb->qs);
        if (lane_in_group == 0) {
            yqs[quad * 16 + group]     = packed[0];
            yqs[quad * 16 + 8 + group] = packed[1];
        }
        if (lane == 0) {
            yb->d4[quad] = ((uint32_t) scales[1] << 8) | scales[0];
        }

        if (output_row + I == n_ff) {
            for (int padded = n_ff; padded < activation_q_ne0; padded += I) {
                const int       padded_block = padded / QK_FP4_MMQ;
                const int       padded_quad  = (padded % QK_FP4_MMQ) / I;
                block_fp4_mmq * padded_yb = (block_fp4_mmq *) dst + (int64_t) padded_block * n_rows + sorted_row_0 + j;
                char2 *         padded_qs = reinterpret_cast<char2 *>(padded_yb->qs);
                if (lane_in_group == 0) {
                    padded_qs[padded_quad * 16 + group]     = make_char2(0, 0);
                    padded_qs[padded_quad * 16 + 8 + group] = make_char2(0, 0);
                }
                if (lane == 0) {
                    padded_yb->d4[padded_quad] = 0;
                }
            }
        }
    }
}

template <int J>
static __device__ __forceinline__ void moe_tma_quantize_w13_mxfp8(const float * __restrict__ activation,
                                                                  void * __restrict__ dst,
                                                                  int64_t n_rows,
                                                                  int     sorted_row_0,
                                                                  int     output_row,
                                                                  int     n_ff,
                                                                  int     activation_q_ne0,
                                                                  int     j_max) {
    constexpr int I = moe_tma_w13_rows;

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    for (int work = warp; work < (j_max + 1) * 2; work += 8) {
        const int   j     = work / 2;
        const int   block = work % 2;
        const float value = activation[j * I + block * 32 + lane];
        float       amax  = fabsf(value);
#    pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, moe_tma_warp_size));
        }

        const uint8_t       scale = moe_tma_mxfp8_scale(amax);
        const float         inv   = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
        const __nv_fp8_e4m3 quantized(value * inv);
        const int           k_block = (output_row + block * 32) / 32;
        block_mxfp8_mmq *   yb      = (block_mxfp8_mmq *) dst + (int64_t) k_block * n_rows + sorted_row_0 + j;
        yb->qs[lane]                = quantized.__x;
        if (lane == 0) {
            yb->scale = scale;
        }

        if (block == 0 && output_row + I == n_ff) {
            for (int padded = n_ff; padded < activation_q_ne0; padded += 32) {
                block_mxfp8_mmq * padded_yb =
                    (block_mxfp8_mmq *) dst + (int64_t) (padded / 32) * n_rows + sorted_row_0 + j;
                padded_yb->qs[lane] = 0;
                if (lane == 0) {
                    padded_yb->scale = 0;
                }
            }
        }
    }
}

template <int J, bool mxfp8>
static __device__ __forceinline__ void moe_tma_accumulate_w13_tile(const int * y_base,
                                                                   int         k_tile,
                                                                   int         valid_k,
                                                                   int         ncols_y,
                                                                   const char * x_gate,
                                                                   const char * x_up,
                                                                   int *       y_tile,
                                                                   uint64_t *  barriers,
                                                                   uint64_t *  tokens,
                                                                   bool        wait_weights,
                                                                   float *     gate_sum,
                                                                   float *     up_sum) {
    constexpr int I              = moe_tma_w13_rows;
    constexpr int consumer_warps = moe_tma_w13_consumer_warps;
    constexpr int y_tile_bytes   = mxfp8 ? J * sizeof(block_mxfp8_mmq) : J * MMQ_TILE_Y_K * sizeof(int);
    constexpr int y_block_stride = sizeof(block_fp4_mmq) / sizeof(int);

    const bool consumer            = threadIdx.y < consumer_warps;
    const bool activation_producer = threadIdx.y == 6 && threadIdx.x == 0;
    const int  first_frags         = valid_k < 256 ? (valid_k + 63) / 64 : 4;
    const int  second_k            = valid_k > 256 ? valid_k - 256 : 0;
    const int  second_frags        = second_k < 256 ? (second_k + 63) / 64 : 4;
    const int  mxfp8_blocks        = valid_k < ggml_cuda_moe_tma_k ? (valid_k + 31) / 32 : 16;

    if constexpr (mxfp8) {
        __syncthreads();
        if (wait_weights) {
            moe_tma_barrier_wait(barriers, tokens[0]);
            moe_tma_barrier_wait(barriers + 1, tokens[1]);
        }
        __syncthreads();
#    pragma unroll
        for (int k_block = 0; k_block < mxfp8_blocks; ++k_block) {
            if (activation_producer) {
                const block_mxfp8_mmq * y_block =
                    reinterpret_cast<const block_mxfp8_mmq *>(y_base) + (int64_t) (k_tile * 16 + k_block) * ncols_y;
                moe_tma_load_activation(y_tile, y_block, y_tile_bytes, barriers + 2, tokens + 2);
            }
            __syncthreads();
            moe_tma_barrier_wait(barriers + 2, tokens[2]);
            __syncthreads();
            if (consumer) {
                moe_tma_vec_dot_mxfp8<J, I, consumer_warps>(
                    x_gate, reinterpret_cast<const block_mxfp8_mmq *>(y_tile), gate_sum, k_block);
                moe_tma_vec_dot_mxfp8<J, I, consumer_warps>(
                    x_up, reinterpret_cast<const block_mxfp8_mmq *>(y_tile), up_sum, k_block);
            }
            __syncthreads();
        }
        return;
    }

    if (activation_producer) {
        const int * y_first = y_base + ncols_y * (2 * k_tile * y_block_stride);
        moe_tma_load_activation(y_tile, y_first, y_tile_bytes, barriers + 2, tokens + 2);
    }
    __syncthreads();
    if (wait_weights) {
        moe_tma_barrier_wait(barriers, tokens[0]);
        moe_tma_barrier_wait(barriers + 1, tokens[1]);
    }
    moe_tma_barrier_wait(barriers + 2, tokens[2]);
    __syncthreads();
    if (consumer) {
        moe_tma_vec_dot<J, I, consumer_warps>(reinterpret_cast<const int *>(x_gate), y_tile, gate_sum, 0,
                                              first_frags);
        moe_tma_vec_dot<J, I, consumer_warps>(reinterpret_cast<const int *>(x_up), y_tile, up_sum, 0, first_frags);
    }
    __syncthreads();

    if (second_frags > 0) {
        if (activation_producer) {
            const int * y_second = y_base + ncols_y * (2 * k_tile * y_block_stride + y_block_stride);
            moe_tma_load_activation(y_tile, y_second, y_tile_bytes, barriers + 2, tokens + 2);
        }
        __syncthreads();
        moe_tma_barrier_wait(barriers + 2, tokens[2]);
        __syncthreads();
        if (consumer) {
            moe_tma_vec_dot<J, I, consumer_warps>(reinterpret_cast<const int *>(x_gate), y_tile, gate_sum,
                                                  MMQ_TILE_NE_K, second_frags);
            moe_tma_vec_dot<J, I, consumer_warps>(reinterpret_cast<const int *>(x_up), y_tile, up_sum,
                                                  MMQ_TILE_NE_K, second_frags);
        }
        __syncthreads();
    }
}

template <int J, bool mxfp8, bool compact_tail>
static __device__ __forceinline__ void moe_tma_accumulate_w13_pair(const CUtensorMap * weight_map,
                                                                   const int *         y_base,
                                                                   const char *        weight_data,
                                                                   int                 expert,
                                                                   int                 gate_row,
                                                                   int                 up_row,
                                                                   int                 rows,
                                                                   int                 rows_padded,
                                                                   int                 k_tiles,
                                                                   int                 tma_tiles,
                                                                   int                 tail_blocks,
                                                                   int64_t             expert_stride,
                                                                   int64_t             tail_offset,
                                                                   int                 logical_k,
                                                                   int                 ncols_y,
                                                                   char *              x_gate,
                                                                   char *              x_up,
                                                                   int *               y_tile,
                                                                   uint64_t *          barriers,
                                                                   uint64_t *          tokens,
                                                                   float *             gate_sum,
                                                                   float *             up_sum) {
    constexpr int I            = moe_tma_w13_rows;
    constexpr int x_tile_bytes = I * ggml_cuda_moe_tma_record_bytes;

    const bool gate_weight_producer = threadIdx.y == 4 && threadIdx.x == 0;
    const bool up_weight_producer   = threadIdx.y == 5 && threadIdx.x == 0;
    const int  full_tiles           = compact_tail ? tma_tiles : k_tiles;
    for (int k_tile = 0; k_tile < full_tiles; ++k_tile) {
        const int remaining    = logical_k - k_tile * ggml_cuda_moe_tma_k;
        const int valid_k      = remaining < ggml_cuda_moe_tma_k ? remaining : ggml_cuda_moe_tma_k;
        if (gate_weight_producer) {
            moe_tma_load_weight<compact_tail>(x_gate, weight_map, gate_row, k_tile, expert, rows_padded, k_tiles,
                                       x_tile_bytes, barriers, tokens);
        }
        if (up_weight_producer) {
            moe_tma_load_weight<compact_tail>(x_up, weight_map, up_row, k_tile, expert, rows_padded, k_tiles, x_tile_bytes,
                                       barriers + 1, tokens + 1);
        }
        moe_tma_accumulate_w13_tile<J, mxfp8>(y_base, k_tile, valid_k, ncols_y, x_gate, x_up, y_tile, barriers,
                                               tokens, true, gate_sum, up_sum);
    }

    if constexpr (compact_tail) {
        if (tail_blocks > 0) {
            moe_tma_load_tail<I>(x_gate, weight_data, expert, gate_row, rows, tail_blocks, expert_stride, tail_offset);
            moe_tma_load_tail<I>(x_up, weight_data, expert, up_row, rows, tail_blocks, expert_stride, tail_offset);
            __syncthreads();
            moe_tma_accumulate_w13_tile<J, mxfp8>(y_base, tma_tiles, tail_blocks * QK_MXFP4, ncols_y, x_gate, x_up,
                                                   y_tile, barriers, tokens, false, gate_sum, up_sum);
        }
    }
}

template <int J, bool mxfp8, bool compact_tail>
__launch_bounds__(moe_tma_threads, 1) static __global__ void moe_tma_w13_persistent(
    const __grid_constant__ CUtensorMap weight_map,
    const char * __restrict__ weight_data,
    const int * __restrict__ y,
    const int32_t * __restrict__ expert_bounds,
    const int32_t * __restrict__ tile_offsets,
    int32_t * __restrict__ work_counter,
    const float * __restrict__ bias,
    void * __restrict__ activation_q,
    int  experts,
    int  n_ff,
    int  rows_padded,
    int  k_tiles,
    int  tma_tiles,
    int  tail_blocks,
    int64_t expert_stride,
    int64_t tail_offset,
    int  logical_k,
    int  ncols_y,
    int  activation_q_ne0,
    bool output_tile_major) {
    constexpr int I            = moe_tma_w13_rows;
    constexpr int x_tile_bytes = I * ggml_cuda_moe_tma_record_bytes;
    constexpr int y_tile_bytes = mxfp8 ? J * sizeof(block_mxfp8_mmq) : J * MMQ_TILE_Y_K * sizeof(int);

    extern __shared__ __align__(128) int shared[];
    int *                 y_tile    = shared;
    char *                x_gate    = reinterpret_cast<char *>(y_tile) + y_tile_bytes;
    char *                x_up      = x_gate + x_tile_bytes;
    float *               gate_tile = reinterpret_cast<float *>(x_up + x_tile_bytes);

    __shared__ uint64_t barriers[3];
    __shared__ uint64_t tokens[3];
    __shared__ int      work_idx;
    __shared__ int      task_expert;
    __shared__ int      task_token_tile;
    __shared__ int      task_output_tile;
    __shared__ int      task_col_low;
    __shared__ int      task_col_high;

    const int linear_thread = threadIdx.y * moe_tma_warp_size + threadIdx.x;
    if (linear_thread < 3) {
        moe_tma_barrier_init(barriers + linear_thread);
    }
    __syncthreads();

    const int n_output_tiles = (n_ff + I - 1) / I;
    const int n_token_tiles  = tile_offsets[experts];
    const int n_work         = n_token_tiles * n_output_tiles;
    while (true) {
        if (linear_thread == 0) {
            work_idx = atomicAdd(work_counter, 1);
            if (work_idx < n_work) {
                const int token_tile = output_tile_major ? work_idx % n_token_tiles : work_idx / n_output_tiles;
                const int output_tile =
                    output_tile_major ? work_idx / n_token_tiles : work_idx - token_tile * n_output_tiles;
                int low  = 0;
                int high = experts;
                while (low < high) {
                    const int mid = (low + high) / 2;
                    if (tile_offsets[mid + 1] <= token_tile) {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }
                task_expert      = low;
                task_token_tile  = token_tile - tile_offsets[low];
                task_output_tile = output_tile;
                task_col_low     = expert_bounds[low];
                task_col_high    = expert_bounds[low + 1];
            }
        }
        __syncthreads();
        if (work_idx >= n_work) {
            return;
        }

        const int   tile_y_max_j   = min(J - 1, task_col_high - task_col_low - task_token_tile * J - 1);
        const int   output_row     = task_output_tile * I;
        const int   sorted_row_0   = task_col_low + task_token_tile * J;
        const int   y_block_stride = sizeof(block_fp4_mmq) / sizeof(int);
        const int * y_base =
            mxfp8 ? reinterpret_cast<const int *>(reinterpret_cast<const block_mxfp8_mmq *>(y) + sorted_row_0) :
                    y + sorted_row_0 * y_block_stride;
        float gate_sum[J / 2] = { 0.0f };
        float up_sum[J / 2]   = { 0.0f };
        moe_tma_accumulate_w13_pair<J, mxfp8, compact_tail>(
            &weight_map, y_base, weight_data, task_expert, output_row, output_row + n_ff, 2 * n_ff, rows_padded,
            k_tiles, tma_tiles, tail_blocks, expert_stride, tail_offset, logical_k, ncols_y, x_gate, x_up, y_tile,
            barriers, tokens, gate_sum, up_sum);
        if (threadIdx.y < moe_tma_w13_consumer_warps) {
            moe_tma_store_shared<J, I, moe_tma_w13_consumer_warps>(gate_sum, gate_tile, n_ff - output_row - 1,
                                                                   tile_y_max_j);
        }
        __syncthreads();

        if (threadIdx.y < moe_tma_w13_consumer_warps) {
            moe_tma_store_swiglu<J>(up_sum, bias, gate_tile, task_expert, output_row, n_ff, tile_y_max_j);
        }
        __syncthreads();

        if constexpr (mxfp8) {
            moe_tma_quantize_w13_mxfp8<J>(gate_tile, activation_q, ncols_y, sorted_row_0, output_row, n_ff,
                                          activation_q_ne0, tile_y_max_j);
        } else {
            moe_tma_quantize_w13<J>(gate_tile, activation_q, ncols_y, sorted_row_0, output_row, n_ff, activation_q_ne0,
                                    tile_y_max_j);
        }
        __syncthreads();
    }
}

template <int J, bool warp_specialized, bool mxfp8, bool compact_tail>
__launch_bounds__(moe_tma_threads, 1) static __global__ void moe_tma_persistent(
    const __grid_constant__ CUtensorMap weight_map,
    const char * __restrict__ weight_data,
    const int * __restrict__ y,
    const int32_t * __restrict__ ids_dst,
    const int32_t * __restrict__ expert_bounds,
    const int32_t * __restrict__ tile_offsets,
    int32_t * __restrict__ work_counter,
    float * __restrict__ dst,
    int  experts,
    int  rows,
    int  rows_padded,
    int  k_tiles,
    int  tma_tiles,
    int  tail_blocks,
    int64_t expert_stride,
    int64_t tail_offset,
    int  logical_k,
    int  ncols_y,
    int  stride_dst,
    bool output_tile_major,
    int  epilogue,
    const float * __restrict__ bias,
    const float * __restrict__ route_weights,
    float * __restrict__ epilogue_dst,
    int n_expert_used) {
    constexpr int I              = warp_specialized ? moe_tma_specialized_rows : moe_tma_cooperative_rows;
    constexpr int consumer_warps = warp_specialized ? moe_tma_specialized_consumer_warps : 8;
    constexpr int x_tile_bytes   = I * ggml_cuda_moe_tma_record_bytes;
    constexpr int y_tile_bytes   = mxfp8 ? J * sizeof(block_mxfp8_mmq) : J * MMQ_TILE_Y_K * sizeof(int);
    constexpr int y_block_stride = sizeof(block_q8_1_mmq) / sizeof(int);

    extern __shared__ __align__(128) int shared[];
    int32_t *             ids_shared = shared;
    int *                 y_tile     = shared + J;
    char *                x_tile_0   = reinterpret_cast<char *>(y_tile) + y_tile_bytes;
    char *                x_tile_1   = x_tile_0 + x_tile_bytes;

    __shared__ uint64_t barriers[3];
    __shared__ uint64_t tokens[3];
    __shared__ int      work_idx;
    __shared__ int      task_expert;
    __shared__ int      task_token_tile;
    __shared__ int      task_output_tile;
    __shared__ int      task_col_low;
    __shared__ int      task_col_high;

    const int  linear_thread       = threadIdx.y * moe_tma_warp_size + threadIdx.x;
    const bool consumer            = threadIdx.y < consumer_warps;
    const bool weight_producer     = warp_specialized ? (threadIdx.y == 6 && threadIdx.x == 0) : linear_thread == 0;
    const bool activation_producer = warp_specialized ? (threadIdx.y == 7 && threadIdx.x == 0) : linear_thread == 1;

    if (linear_thread < 3) {
        moe_tma_barrier_init(barriers + linear_thread);
    }
    __syncthreads();

    const int nty           = (rows + I - 1) / I;
    const int n_token_tiles = tile_offsets[experts];
    const int n_work        = n_token_tiles * nty;

    while (true) {
        if (linear_thread == 0) {
            work_idx = atomicAdd(work_counter, 1);
            if (work_idx < n_work) {
                const int token_tile  = output_tile_major ? work_idx % n_token_tiles : work_idx / nty;
                const int output_tile = output_tile_major ? work_idx / n_token_tiles : work_idx - token_tile * nty;
                int       low         = 0;
                int       high        = experts;
                while (low < high) {
                    const int mid = (low + high) / 2;
                    if (tile_offsets[mid + 1] <= token_tile) {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }
                task_expert      = low;
                task_token_tile  = token_tile - tile_offsets[low];
                task_output_tile = output_tile;
                task_col_low     = expert_bounds[low];
                task_col_high    = expert_bounds[low + 1];
            }
        }
        __syncthreads();
        if (work_idx >= n_work) {
            return;
        }

        const int tile_y_max_j = min(J - 1, task_col_high - task_col_low - task_token_tile * J - 1);
        for (int j = linear_thread; j < J; j += moe_tma_threads) {
            ids_shared[j] = j <= tile_y_max_j ? ids_dst[task_col_low + task_token_tile * J + j] : 0;
        }
        __syncthreads();

        const int   output_row   = task_output_tile * I;
        const int   sorted_row_0 = task_col_low + task_token_tile * J;
        const int   offset_y     = sorted_row_0 * y_block_stride;
        const int   i_max        = rows - output_row - 1;
        const int * y_base       = y + offset_y;
        const int   full_tiles   = compact_tail ? tma_tiles : k_tiles;

        if (weight_producer) {
            moe_tma_load_weight<compact_tail>(x_tile_0, &weight_map, output_row, 0, task_expert, rows_padded, k_tiles,
                                       x_tile_bytes, barriers, tokens);
        }
        if (!mxfp8 && activation_producer) {
            moe_tma_load_activation(y_tile, y_base, y_tile_bytes, barriers + 2, tokens + 2);
        }
        __syncthreads();
        moe_tma_barrier_wait(barriers, tokens[0]);
        if (!mxfp8) {
            moe_tma_barrier_wait(barriers + 2, tokens[2]);
        }
        __syncthreads();

        float  sum[J / 2] = { 0.0f };
        char * x_current  = x_tile_0;
        char * x_next     = x_tile_1;
        for (int k_tile = 0; k_tile < full_tiles; ++k_tile) {
            const int  remaining    = logical_k - k_tile * ggml_cuda_moe_tma_k;
            const int  valid_k      = remaining < ggml_cuda_moe_tma_k ? remaining : ggml_cuda_moe_tma_k;
            const int  first_frags  = valid_k < 256 ? (valid_k + 63) / 64 : 4;
            const int  second_k     = valid_k > 256 ? valid_k - 256 : 0;
            const int  second_frags = second_k < 256 ? (second_k + 63) / 64 : 4;
            const int  mxfp8_blocks = valid_k < ggml_cuda_moe_tma_k ? (valid_k + 31) / 32 : 16;
            const bool has_next     = k_tile + 1 < full_tiles;
            const int  next_stage   = (k_tile + 1) & 1;
            if (has_next && weight_producer) {
                moe_tma_load_weight<compact_tail>(x_next, &weight_map, output_row, k_tile + 1, task_expert, rows_padded,
                                           k_tiles, x_tile_bytes, barriers + next_stage, tokens + next_stage);
            }
            __syncthreads();

            if constexpr (mxfp8) {
#    pragma unroll
                for (int k_block = 0; k_block < mxfp8_blocks; ++k_block) {
                    if (activation_producer) {
                        const block_mxfp8_mmq * y_block = reinterpret_cast<const block_mxfp8_mmq *>(y) +
                                                          (int64_t) (k_tile * 16 + k_block) * ncols_y + sorted_row_0;
                        moe_tma_load_activation(y_tile, y_block, y_tile_bytes, barriers + 2, tokens + 2);
                    }
                    __syncthreads();
                    moe_tma_barrier_wait(barriers + 2, tokens[2]);
                    __syncthreads();
                    if (consumer) {
                        moe_tma_vec_dot_mxfp8<J, I, consumer_warps>(
                            x_current, reinterpret_cast<const block_mxfp8_mmq *>(y_tile), sum, k_block);
                    }
                    __syncthreads();
                }
                if (has_next) {
                    moe_tma_barrier_wait(barriers + next_stage, tokens[next_stage]);
                }
                __syncthreads();
            } else {
                if (consumer) {
                    moe_tma_vec_dot<J, I, consumer_warps>(reinterpret_cast<const int *>(x_current), y_tile, sum, 0,
                                                          first_frags);
                }
                __syncthreads();
                if (has_next) {
                    moe_tma_barrier_wait(barriers + next_stage, tokens[next_stage]);
                }
                __syncthreads();

                if (second_frags > 0) {
                    if (activation_producer) {
                        const int * y_second = y_base + ncols_y * (2 * k_tile * y_block_stride + y_block_stride);
                        moe_tma_load_activation(y_tile, y_second, y_tile_bytes, barriers + 2, tokens + 2);
                    }
                    __syncthreads();
                    moe_tma_barrier_wait(barriers + 2, tokens[2]);
                    __syncthreads();
                    if (consumer) {
                        moe_tma_vec_dot<J, I, consumer_warps>(reinterpret_cast<const int *>(x_current), y_tile, sum,
                                                              MMQ_TILE_NE_K, second_frags);
                    }
                    __syncthreads();
                }

                if (has_next && activation_producer) {
                    const int * y_first_next = y_base + ncols_y * (2 * (k_tile + 1) * y_block_stride);
                    moe_tma_load_activation(y_tile, y_first_next, y_tile_bytes, barriers + 2, tokens + 2);
                }
                __syncthreads();
                if (has_next) {
                    moe_tma_barrier_wait(barriers + 2, tokens[2]);
                }
                __syncthreads();
            }

            char * tmp = x_current;
            x_current  = x_next;
            x_next     = tmp;
        }

        if constexpr (compact_tail) {
            if (tail_blocks > 0) {
                char * x_tail = x_tile_0;
                moe_tma_load_tail<I>(x_tail, weight_data, task_expert, output_row, rows, tail_blocks, expert_stride,
                                     tail_offset);
                __syncthreads();

                const int valid_k = tail_blocks * QK_MXFP4;
                const int first_frags = valid_k < 256 ? (valid_k + 63) / 64 : 4;
                const int second_k = valid_k > 256 ? valid_k - 256 : 0;
                const int second_frags = second_k < 256 ? (second_k + 63) / 64 : 4;
                if constexpr (mxfp8) {
                    for (int k_block = 0; k_block < tail_blocks; ++k_block) {
                        if (activation_producer) {
                            const block_mxfp8_mmq * y_block = reinterpret_cast<const block_mxfp8_mmq *>(y) +
                                                              (int64_t) (tma_tiles * 16 + k_block) * ncols_y +
                                                              sorted_row_0;
                            moe_tma_load_activation(y_tile, y_block, y_tile_bytes, barriers + 2, tokens + 2);
                        }
                        __syncthreads();
                        moe_tma_barrier_wait(barriers + 2, tokens[2]);
                        __syncthreads();
                        if (consumer) {
                            moe_tma_vec_dot_mxfp8<J, I, consumer_warps>(
                                x_tail, reinterpret_cast<const block_mxfp8_mmq *>(y_tile), sum, k_block);
                        }
                        __syncthreads();
                    }
                } else {
                    if (activation_producer) {
                        const int * y_first = y_base + ncols_y * (2 * tma_tiles * y_block_stride);
                        moe_tma_load_activation(y_tile, y_first, y_tile_bytes, barriers + 2, tokens + 2);
                    }
                    __syncthreads();
                    moe_tma_barrier_wait(barriers + 2, tokens[2]);
                    __syncthreads();
                    if (consumer) {
                        moe_tma_vec_dot<J, I, consumer_warps>(reinterpret_cast<const int *>(x_tail), y_tile, sum, 0,
                                                              first_frags);
                    }
                    __syncthreads();

                    if (second_frags > 0) {
                        if (activation_producer) {
                            const int * y_second =
                                y_base + ncols_y * (2 * tma_tiles * y_block_stride + y_block_stride);
                            moe_tma_load_activation(y_tile, y_second, y_tile_bytes, barriers + 2, tokens + 2);
                        }
                        __syncthreads();
                        moe_tma_barrier_wait(barriers + 2, tokens[2]);
                        __syncthreads();
                        if (consumer) {
                            moe_tma_vec_dot<J, I, consumer_warps>(reinterpret_cast<const int *>(x_tail), y_tile, sum,
                                                                  MMQ_TILE_NE_K, second_frags);
                        }
                        __syncthreads();
                    }
                }
            }
        }

        if (consumer) {
            if (epilogue == GGML_CUDA_MOE_MMQ_EPILOGUE_W2_WEIGHTED ||
                epilogue == GGML_CUDA_MOE_MMQ_EPILOGUE_W2_ATOMIC) {
                moe_tma_write_back_w2<J, I, consumer_warps>(
                    sum, ids_shared, bias + output_row, route_weights, epilogue_dst + output_row, task_expert,
                    stride_dst, n_expert_used, i_max, tile_y_max_j, epilogue == GGML_CUDA_MOE_MMQ_EPILOGUE_W2_ATOMIC);
            } else {
                moe_tma_write_back<J, I, consumer_warps>(sum, ids_shared, dst + output_row, stride_dst, i_max,
                                                         tile_y_max_j);
            }
        }
        __syncthreads();
    }
}

template <int J, bool warp_specialized, bool compact_tail>
static void moe_tma_launch(const mmq_args &                args,
                           const ggml_cuda_moe_mmq_state & state,
                           const CUtensorMap &             map,
                           int32_t *                       work_counter,
                           cudaStream_t                    stream) {
    constexpr int I         = warp_specialized ? moe_tma_specialized_rows : moe_tma_cooperative_rows;
    const bool    mxfp8     = state.activation_format == GGML_CUDA_MOE_ACTIVATION_MXFP8;
    const int logical_k = compact_tail || state.tma_tail_elide ? (int) state.logical_k :
                                                            state.weight.k_tiles * ggml_cuda_moe_tma_k;
    const int     shared_bytes = J * sizeof(int32_t) +
                             (mxfp8 ? J * sizeof(block_mxfp8_mmq) : J * MMQ_TILE_Y_K * sizeof(int)) +
                             2 * I * ggml_cuda_moe_tma_record_bytes;
    CUDA_CHECK(cudaMemsetAsync(work_counter, 0, sizeof(int32_t), stream));
    if (mxfp8) {
        CUDA_SET_SHARED_MEMORY_LIMIT((moe_tma_persistent<J, warp_specialized, true, compact_tail>), shared_bytes);
        moe_tma_persistent<J, warp_specialized, true, compact_tail>
            <<<state.grid_blocks, dim3(32, 8, 1), shared_bytes, stream>>>(
                map, state.weight.data, args.y, args.ids_dst, args.expert_bounds, state.tile_offsets, work_counter,
                args.dst, args.nchannels_x, args.nrows_x, state.weight.rows_padded, state.weight.k_tiles,
                state.weight.tma_tiles, state.weight.tail_blocks, state.weight.expert_stride, state.weight.tail_offset,
                logical_k, args.ncols_y, args.nrows_dst, state.output_tile_major, state.epilogue, state.bias,
                state.route_weights, state.epilogue_dst, state.n_expert_used);
    } else {
        CUDA_SET_SHARED_MEMORY_LIMIT((moe_tma_persistent<J, warp_specialized, false, compact_tail>), shared_bytes);
        moe_tma_persistent<J, warp_specialized, false, compact_tail>
            <<<state.grid_blocks, dim3(32, 8, 1), shared_bytes, stream>>>(
                map, state.weight.data, args.y, args.ids_dst, args.expert_bounds, state.tile_offsets, work_counter,
                args.dst, args.nchannels_x, args.nrows_x, state.weight.rows_padded, state.weight.k_tiles,
                state.weight.tma_tiles, state.weight.tail_blocks, state.weight.expert_stride, state.weight.tail_offset,
                logical_k, args.ncols_y, args.nrows_dst, state.output_tile_major, state.epilogue, state.bias,
                state.route_weights, state.epilogue_dst, state.n_expert_used);
    }
}

template <int J, bool compact_tail>
static void moe_tma_w13_launch(const mmq_args &                args,
                               const ggml_cuda_moe_mmq_state & state,
                               const CUtensorMap &             map,
                               int32_t *                       work_counter,
                               cudaStream_t                    stream) {
    const bool mxfp8        = state.activation_format == GGML_CUDA_MOE_ACTIVATION_MXFP8;
    const int  logical_k    = compact_tail || state.tma_tail_elide ? (int) state.logical_k :
                                                                state.weight.k_tiles * ggml_cuda_moe_tma_k;
    const int  shared_bytes = (mxfp8 ? J * sizeof(block_mxfp8_mmq) : J * MMQ_TILE_Y_K * sizeof(int)) +
                             2 * moe_tma_w13_rows * ggml_cuda_moe_tma_record_bytes +
                             J * moe_tma_w13_rows * sizeof(float);
    CUDA_CHECK(cudaMemsetAsync(work_counter, 0, sizeof(int32_t), stream));
    if (mxfp8) {
        CUDA_SET_SHARED_MEMORY_LIMIT((moe_tma_w13_persistent<J, true, compact_tail>), shared_bytes);
        moe_tma_w13_persistent<J, true, compact_tail><<<state.grid_blocks, dim3(32, 8, 1), shared_bytes, stream>>>(
            map, state.weight.data, args.y, args.expert_bounds, state.tile_offsets, work_counter, state.bias,
            state.activation_q, args.nchannels_x, state.epilogue_width, state.weight.rows_padded,
            state.weight.k_tiles, state.weight.tma_tiles, state.weight.tail_blocks, state.weight.expert_stride,
            state.weight.tail_offset, logical_k, args.ncols_y, state.activation_q_ne0, state.output_tile_major);
    } else {
        CUDA_SET_SHARED_MEMORY_LIMIT((moe_tma_w13_persistent<J, false, compact_tail>), shared_bytes);
        moe_tma_w13_persistent<J, false, compact_tail><<<state.grid_blocks, dim3(32, 8, 1), shared_bytes, stream>>>(
            map, state.weight.data, args.y, args.expert_bounds, state.tile_offsets, work_counter, state.bias,
            state.activation_q, args.nchannels_x, state.epilogue_width, state.weight.rows_padded,
            state.weight.k_tiles, state.weight.tma_tiles, state.weight.tail_blocks, state.weight.expert_stride,
            state.weight.tail_offset, logical_k, args.ncols_y, state.activation_q_ne0, state.output_tile_major);
    }
}

#endif

bool ggml_cuda_moe_mmq_tma_supported(const ggml_cuda_moe_weight_view & weight,
                                     int                               tile_rows,
                                     bool                              warp_specialized,
                                     size_t                            smpbo,
                                     int                               epilogue) {
#if CUDART_VERSION >= 12080
    const bool w13  = epilogue == GGML_CUDA_MOE_MMQ_EPILOGUE_W13;
    const int  mode = w13 ? 2 : warp_specialized ? 1 : 0;
    const bool compact_tail = weight.layout == ggml_cuda_moe_weight_layout::tma_inplace;
    if ((weight.layout != ggml_cuda_moe_weight_layout::tma && !compact_tail) || weight.rows_padded <= 0 ||
        weight.k_tiles <= 0 ||
        !weight.tma_valid[mode] || (tile_rows != 32 && tile_rows != 64 && tile_rows != 128)) {
        return false;
    }
    if (compact_tail && (weight.tma_tiles <= 0 || weight.tma_tiles > weight.k_tiles || weight.tail_blocks < 0 ||
                  weight.tail_blocks >= ggml_cuda_moe_tma_k_blocks || weight.expert_stride <= 0 ||
                  weight.tail_offset <= 0)) {
        return false;
    }

    const int    output_rows  = w13              ? moe_tma_w13_rows :
                                warp_specialized ? moe_tma_specialized_rows :
                                                   moe_tma_cooperative_rows;
    const size_t shared_bytes = tile_rows * MMQ_TILE_Y_K * sizeof(int) +
                                2 * output_rows * ggml_cuda_moe_tma_record_bytes +
                                (w13 ? tile_rows * output_rows * sizeof(float) : tile_rows * sizeof(int32_t));
    return shared_bytes <= smpbo;
#else
    GGML_UNUSED_VARS(weight, tile_rows, warp_specialized, smpbo, epilogue);
    return false;
#endif
}

bool ggml_cuda_moe_mmq_tma(ggml_backend_cuda_context &     ctx,
                           const mmq_args &                args,
                           const ggml_cuda_moe_mmq_state & state,
                           cudaStream_t                    stream) {
#if CUDART_VERSION >= 12080
    const size_t smpbo = ggml_cuda_info().devices[ctx.device].smpbo;
    if (!ggml_cuda_moe_mmq_tma_supported(state.weight, state.tile_rows, state.tma_warp_specialized, smpbo,
                                         state.epilogue)) {
        return false;
    }
    if (state.tma_tail_elide &&
        (state.logical_k <= (state.weight.k_tiles - 1) * ggml_cuda_moe_tma_k ||
         state.logical_k > state.weight.k_tiles * ggml_cuda_moe_tma_k || state.logical_k % QK_MXFP4 != 0)) {
        return false;
    }

    const bool  w13  = state.epilogue == GGML_CUDA_MOE_MMQ_EPILOGUE_W13;
    const bool  compact_tail = state.weight.layout == ggml_cuda_moe_weight_layout::tma_inplace;
    const int   mode = w13 ? 2 : state.tma_warp_specialized ? 1 : 0;
    CUtensorMap map;
    static_assert(sizeof(map) == sizeof(state.weight.tma_map[mode]), "unexpected tensor-map storage size");
    memcpy(&map, state.weight.tma_map[mode], sizeof(map));

    ggml_cuda_pool_alloc<int32_t> work_counter(ctx.pool(), 1);
    if (w13) {
        if (state.bias == nullptr || state.activation_q == nullptr || state.epilogue_width <= 0 ||
            state.activation_q_ne0 < state.epilogue_width) {
            return false;
        }
        if (state.tile_rows == 32) {
            compact_tail ? moe_tma_w13_launch<32, true>(args, state, map, work_counter.get(), stream) :
                    moe_tma_w13_launch<32, false>(args, state, map, work_counter.get(), stream);
        } else if (state.tile_rows == 64) {
            compact_tail ? moe_tma_w13_launch<64, true>(args, state, map, work_counter.get(), stream) :
                    moe_tma_w13_launch<64, false>(args, state, map, work_counter.get(), stream);
        } else if (state.tile_rows == 128) {
            compact_tail ? moe_tma_w13_launch<128, true>(args, state, map, work_counter.get(), stream) :
                    moe_tma_w13_launch<128, false>(args, state, map, work_counter.get(), stream);
        } else {
            return false;
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    if ((state.epilogue == GGML_CUDA_MOE_MMQ_EPILOGUE_W2_WEIGHTED ||
         state.epilogue == GGML_CUDA_MOE_MMQ_EPILOGUE_W2_ATOMIC) &&
        (state.bias == nullptr || state.route_weights == nullptr || state.epilogue_dst == nullptr ||
         state.n_expert_used <= 0)) {
        return false;
    }

    const bool specialized = state.tma_warp_specialized;
    if (state.tile_rows == 32) {
        if (compact_tail) {
            specialized ? moe_tma_launch<32, true, true>(args, state, map, work_counter.get(), stream) :
                          moe_tma_launch<32, false, true>(args, state, map, work_counter.get(), stream);
        } else {
            specialized ? moe_tma_launch<32, true, false>(args, state, map, work_counter.get(), stream) :
                          moe_tma_launch<32, false, false>(args, state, map, work_counter.get(), stream);
        }
    } else if (state.tile_rows == 64) {
        if (compact_tail) {
            specialized ? moe_tma_launch<64, true, true>(args, state, map, work_counter.get(), stream) :
                          moe_tma_launch<64, false, true>(args, state, map, work_counter.get(), stream);
        } else {
            specialized ? moe_tma_launch<64, true, false>(args, state, map, work_counter.get(), stream) :
                          moe_tma_launch<64, false, false>(args, state, map, work_counter.get(), stream);
        }
    } else if (state.tile_rows == 128) {
        if (compact_tail) {
            specialized ? moe_tma_launch<128, true, true>(args, state, map, work_counter.get(), stream) :
                          moe_tma_launch<128, false, true>(args, state, map, work_counter.get(), stream);
        } else {
            specialized ? moe_tma_launch<128, true, false>(args, state, map, work_counter.get(), stream) :
                          moe_tma_launch<128, false, false>(args, state, map, work_counter.get(), stream);
        }
    } else {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    GGML_UNUSED_VARS(ctx, args, state, stream);
    return false;
#endif
}
