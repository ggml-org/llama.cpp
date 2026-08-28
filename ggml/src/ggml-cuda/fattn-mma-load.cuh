// Tile loaders for the MMA FlashAttention kernel.
// Each specialization of fattn_mma_tile_loader loads a tile of K or V data into shared memory,
// converting it to the f16 layout expected by the MMA pipeline if necessary.
// This file is included from fattn-mma-f16.cuh after the definition of flash_attn_ext_f16_load_tile.
//
// Loading is split in two steps to preserve the asynchronous copy pipeline:
//   - load():   either copies/converts the data to tile_KV directly (f16, or quantized without cp_async)
//               or issues cp_async copies of the raw quantized bytes into a staging buffer.
//   - finish(): converts staged raw bytes to f16 in tile_KV, called after the cp_async_wait_all() +
//               __syncthreads() that the pipeline already performs. Ends with one additional
//               __syncthreads() so that the converted tile is visible to all threads.
//               A no-op for f16 and for the synchronous path.

#pragma once

// Size in bytes of the units in which the K/V row stride is expressed for a given KV cache type.
static constexpr __host__ __device__ size_t fattn_mma_kv_granule(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return sizeof(half2);
        case GGML_TYPE_Q8_0:
            return sizeof(block_q8_0);
        case GGML_TYPE_Q4_0:
            return sizeof(block_q4_0);
        default:
            return 0;
    }
}

// Type of the K/V data pointers passed through the kernel: half2 for f16 data as in the kernel
// without fused dequantization (with an untyped pointer the compiler generates different code for
// the f16 kernels), raw bytes for quantized data.
template <ggml_type type>
using fattn_mma_kv_ptr_t = std::conditional_t<type == GGML_TYPE_F16, const half2 *, const void *>;

// Copies data from global to shared memory, ca == cache all levels.
// Both the src and dst pointers must be aligned to 8 bytes.
// Quantized KV rows are not 16-byte aligned (block_q8_0 is 34 bytes), 8 bytes is their max. alignment.
static __device__ __forceinline__ void cp_async_ca_8(const unsigned int dst, const void * src) {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
        : : "r"(dst), "l"(src));
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(src);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

template <ggml_type type>
struct fattn_mma_tile_loader;

template <int granule>
static __device__ __forceinline__ void fattn_mma_cp_async(const unsigned int dst, const void * src) {
    if constexpr (granule == 16) {
        cp_async_cg_16<64>(dst, src);
    } else {
        static_assert(granule == 8, "bad cp.async granule");
        cp_async_ca_8(dst, src);
    }
}

// Copies nbatch_fa rows of row_bytes raw bytes each from global memory (rows stride_row bytes
// apart) to the staging buffer (rows stride_staging bytes apart) with cp.async granules.
// Each warp pass covers R rows; the lane -> (row delta, offset) assignment is constant across
// passes so that the index divisions are hoisted out of the copy loop.
template <int granule, int row_bytes, int stride_staging, int nwarps, int nbatch_fa>
static __device__ __forceinline__ void fattn_mma_copy_rows_async(
        const char * const __restrict__ KV_bytes, const size_t stride_row, const unsigned int staging_32) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int gpr  = row_bytes/granule;    // granules per row
    constexpr int rmax = 2*warp_size/gpr;
    constexpr int R    = rmax >= 8 ? 8 : rmax >= 4 ? 4 : rmax >= 2 ? 2 : 1; // rows per warp pass
    static_assert(row_bytes % granule == 0, "partial granule");
    static_assert(gpr <= 2*warp_size, "granule too small");
    static_assert(nbatch_fa % R == 0, "bad rows per pass");

    const int l   = threadIdx.x;
    const int dr0 =  l              / gpr; // always a valid granule: R*gpr >= warp_size
    const int do0 =  l              % gpr;
    const int dr1 = (l + warp_size) / gpr; // valid for the first R*gpr - warp_size lanes
    const int do1 = (l + warp_size) % gpr;

    const size_t       goff0 = dr0*stride_row     + granule*do0;
    const size_t       goff1 = dr1*stride_row     + granule*do1;
    const unsigned int soff0 = dr0*stride_staging + granule*do0;
    const unsigned int soff1 = dr1*stride_staging + granule*do1;

#pragma unroll
    for (int p0 = 0; p0 < nbatch_fa/R; p0 += nwarps) {
        const int p = p0 + threadIdx.y;

        if (p0 + nwarps > nbatch_fa/R && p >= nbatch_fa/R) {
            break;
        }

        const char       * grow = KV_bytes   + (p*R)*stride_row;
        const unsigned int srow = staging_32 + (p*R)*stride_staging;

        fattn_mma_cp_async<granule>(srow + soff0, grow + goff0);
        if (l + warp_size < R*gpr) {
            fattn_mma_cp_async<granule>(srow + soff1, grow + goff1);
        }
    }
}

static __device__ __forceinline__ half2 fattn_mma_u32_as_half2(const uint32_t u) {
    half2 h;
    memcpy(&h, &u, sizeof(h));
    return h;
}

// Broadcasts the 16-bit block scale at even byte offset boff of word-aligned row ws to a half2.
static __device__ __forceinline__ half2 fattn_mma_load_scale(const uint32_t * __restrict__ ws, const int boff) {
    return fattn_mma_u32_as_half2(__byte_perm(ws[boff >> 2], 0, 0x1010 + (boff & 2)*0x1111));
}

// Maximum dynamic shared memory per block, used to decide whether the staged quantized pipeline
// still fits with 2 stages. The host and device versions MUST be consistent:
// they determine the shared memory layout on both sides.
static constexpr __device__ size_t fattn_mma_max_nbytes_shared_device() {
#if __CUDA_ARCH__ >= 1200
    return  99u*1024;
#elif __CUDA_ARCH__ >= 900
    return 227u*1024;
#elif __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 870
    return 163u*1024;
#elif __CUDA_ARCH__ >= 860
    return  99u*1024;
#elif __CUDA_ARCH__ >= 700
    return  64u*1024;
#else
    return  48u*1024;
#endif // __CUDA_ARCH__
}

static __host__ size_t fattn_mma_max_nbytes_shared_host(const int cc) {
    if (!GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return 64u*1024;
    }
    if (cc >= GGML_CUDA_CC_BLACKWELL) {
        return  99u*1024;
    }
    if (cc >= GGML_CUDA_CC_HOPPER) {
        return 227u*1024;
    }
    if (cc == GGML_CUDA_CC_AMPERE || cc == 870) {
        return 163u*1024;
    }
    if (cc >= 860) {
        return  99u*1024;
    }
    if (cc >= GGML_CUDA_CC_VOLTA) {
        return  64u*1024;
    }
    return 48u*1024;
}

template <>
struct fattn_mma_tile_loader<GGML_TYPE_F16> {
    // f16 data is loaded to tile_KV directly, no staging memory needed.
    static constexpr __host__ __device__ size_t nbytes_staging(int nbatch_fa, int nbatch2) {
        GGML_UNUSED(nbatch_fa); GGML_UNUSED(nbatch2);
        return 0;
    }

    // Transparent wrapper around flash_attn_ext_f16_load_tile, must optimize away completely.
    // row0 is in KV rows, col0 and D2 in half2, stride_KV in half2 per row.
    template <int stride_tile, bool swz, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check>
    static __device__ __forceinline__ void load(
            const void * const __restrict__ KV, half2 * const __restrict__ tile_KV, char * const __restrict__ staging,
            const int64_t row0, const int col0, const int D2, const int stride_KV, const int i_sup) {
        GGML_UNUSED(staging);
        flash_attn_ext_f16_load_tile<stride_tile, swz, nwarps, nbatch_fa, use_cp_async, oob_check>
            ((const half2 *) KV + row0*stride_KV + col0, tile_KV, D2, stride_KV, i_sup);
    }

    template <int stride_tile, bool swz, int nwarps, int nbatch_fa, bool use_cp_async>
    static __device__ __forceinline__ void finish(
            const char * const __restrict__ staging, half2 * const __restrict__ tile_KV, const int D2) {
        GGML_UNUSED(staging); GGML_UNUSED(tile_KV); GGML_UNUSED(D2);
    }
};

// Common load/finish machinery for quantized KV types with 32-element blocks.
// Self provides dequant_block (a whole block from a word-aligned staging row, used by the
// cp_async path) and dequant_chunk (4 elements, used by the synchronous path).
template <typename Self, typename block_t>
struct fattn_mma_tile_loader_quant {
    static constexpr int h2_per_block    = QK8_0/2; // 16, all supported quant types have 32-element blocks
    static constexpr int lanes_per_block = 8;       // sync path: each lane dequantizes 4 consecutive elements

    // Staging rows are padded to 16 bytes. A row slab of nbatch2 half2 covers 2*nbatch2/QK8_0 blocks.
    static constexpr __host__ __device__ size_t nbytes_staging_row(int nbatch2) {
        return ((2*nbatch2/QK8_0)*sizeof(block_t) + 15) & ~size_t(15);
    }
    static constexpr __host__ __device__ size_t nbytes_staging(int nbatch_fa, int nbatch2) {
        return nbatch_fa*nbytes_staging_row(nbatch2);
    }

    // Tile row length in half2: the swizzled layout is unpadded, the linear layout pads the
    // stride by 4 half2 against bank conflicts.
    template <int stride_tile, bool swz>
    static constexpr int tile_nbatch2 = swz ? stride_tile : stride_tile - 4;

    // Tile store through the same layout the ldmatrix consumers read. The swizzle remaps whole
    // 16-byte granules (address bits 4-6), so 8-byte-aligned chunks keep their position within
    // a granule.
    template <int stride_tile, bool swz, int nbytes>
    static __device__ __forceinline__ void store_tile(
            half2 * const __restrict__ tile_KV, const int row, const int col_h2, const void * const __restrict__ src) {
        static_assert(nbytes % 8 == 0 && nbytes <= 16, "bad tile store granularity");
        if constexpr (swz) {
            const int off = ggml_cuda_fattn_smem_swizzle::bytes_rc<stride_tile>(row, col_h2);
            ggml_cuda_memcpy_1<nbytes>((char *) tile_KV + off, src);
        } else {
            ggml_cuda_memcpy_1<nbytes>(tile_KV + row*stride_tile + col_h2, src);
        }
    }

    // row0 is in KV rows, col0 and D2 in half2, stride_KV in blocks per row.
    template <int stride_tile, bool swz, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check>
    static __device__ __forceinline__ void load(
            const void * const __restrict__ KV, half2 * const __restrict__ tile_KV, char * const __restrict__ staging,
            const int64_t row0, const int col0, const int D2, const int stride_KV, const int i_sup) {
        constexpr int warp_size = ggml_cuda_get_physical_warp_size();
        constexpr int nbatch2   = tile_nbatch2<stride_tile, swz>;
        static_assert(warp_size % lanes_per_block == 0, "bad lanes_per_block");
        static_assert(nbatch2 % 16 == 0, "slab boundaries not aligned to quant block boundaries");

        const block_t * KV_q = (const block_t *) KV + row0*stride_KV + col0/h2_per_block;

        if constexpr (use_cp_async) {
            static_assert(!oob_check, "OOB check not compatible with cp_async");
            // The cp_async path always loads whole rows (guaranteed by the host: nbatch == D/2),
            // so the row size in bytes is known at compile time.
            constexpr int row_bytes = (nbatch2/h2_per_block)*(int)sizeof(block_t);
            static_assert(row_bytes % 8 == 0, "row slab bytes not a multiple of 8");
            constexpr int stride_staging = (int) nbytes_staging_row(nbatch2);

            const unsigned int staging_32 = ggml_cuda_cvta_generic_to_shared(staging);
            const char * KV_bytes   = (const char *) KV_q;
            const size_t stride_row = stride_KV*sizeof(block_t);

            // 16-byte granules halve the copy loop but need 16-byte aligned rows in global memory.
            if constexpr (row_bytes % 16 == 0) {
                if (stride_row % 16 == 0 && (uintptr_t) KV_bytes % 16 == 0) {
                    fattn_mma_copy_rows_async<16, row_bytes, stride_staging, nwarps, nbatch_fa>(KV_bytes, stride_row, staging_32);
                } else {
                    fattn_mma_copy_rows_async< 8, row_bytes, stride_staging, nwarps, nbatch_fa>(KV_bytes, stride_row, staging_32);
                }
            } else {
                fattn_mma_copy_rows_async<8, row_bytes, stride_staging, nwarps, nbatch_fa>(KV_bytes, stride_row, staging_32);
            }
        } else {
            const int blocks_per_row = D2 / h2_per_block; // slab boundaries are block-aligned

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps) {
                const int i = i0 + threadIdx.y;

                if (i0 + nwarps > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

                for (int idx = threadIdx.x; idx < blocks_per_row*lanes_per_block; idx += warp_size) {
                    const int b = idx / lanes_per_block; // block within the row slab
                    const int l = idx % lanes_per_block; // 4-element chunk within the block

                    half2 val[2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
                    if (!oob_check || i < i_sup) {
                        Self::dequant_chunk((const char *) (KV_q + i*stride_KV + b), l, val);
                    }
                    store_tile<stride_tile, swz, 2*sizeof(half2)>(tile_KV, i, b*h2_per_block + 2*l, val);
                }
            }
        }
    }

    template <int stride_tile, bool swz, int nwarps, int nbatch_fa, bool use_cp_async>
    static __device__ __forceinline__ void finish(
            const char * const __restrict__ staging, half2 * const __restrict__ tile_KV, const int D2) {
        if constexpr (!use_cp_async) {
            GGML_UNUSED(staging); GGML_UNUSED(tile_KV); GGML_UNUSED(D2);
            return; // load() already converted directly to tile_KV
        } else {
            constexpr int warp_size = ggml_cuda_get_physical_warp_size();
            constexpr int nbatch2   = tile_nbatch2<stride_tile, swz>;
            constexpr size_t stride_staging = nbytes_staging_row(nbatch2);

            // The cp_async path always loads whole rows (guaranteed by the host: nbatch == D/2),
            // so the loop bounds are known at compile time. Each lane converts one whole block per
            // iteration: blocks are only 2-byte aligned inside the staging row, so the lane loads
            // the 4-byte words covering its block and extracts the misaligned payload with funnel
            // shifts instead of issuing narrow 2-byte shared memory loads.
            constexpr int blocks_per_row = nbatch2 / h2_per_block;
            constexpr int nthreads = nwarps*warp_size;
            GGML_UNUSED(D2);

            const int tid = threadIdx.y*warp_size + threadIdx.x;

#pragma unroll
            for (int idx = tid; idx < nbatch_fa*blocks_per_row; idx += nthreads) {
                const int i = idx / blocks_per_row;
                const int b = idx % blocks_per_row;

                const uint32_t * ws = (const uint32_t *) (staging + i*stride_staging);

                half2 val[h2_per_block];
                Self::dequant_block(ws, b*(int)sizeof(block_t), val);

#pragma unroll
                for (int s = 0; s < h2_per_block; s += 4) {
                    store_tile<stride_tile, swz, 4*sizeof(half2)>(tile_KV, i, b*h2_per_block + s, val + s);
                }
            }

            __syncthreads(); // Make the converted tile visible to all threads.
        }
    }
};

template <>
struct fattn_mma_tile_loader<GGML_TYPE_Q8_0>
        : fattn_mma_tile_loader_quant<fattn_mma_tile_loader<GGML_TYPE_Q8_0>, block_q8_0> {
    // For the int8 tensor core KQ path: instead of dequantizing, deinterleave the staged tile
    // into contiguous raw int8 qs (32 bytes per block, 16-byte aligned rows of stride_K8 bytes)
    // and a separate per-block f32 scale array dK (blocks_per_row scales per row).
    // The int8 tile has its own linear layout; swz only affects the tile-stride bookkeeping.
    template <int stride_tile, bool swz, int stride_K8, int nwarps, int nbatch_fa>
    static __device__ __forceinline__ void finish_imma(
            const char * const __restrict__ staging, char * const __restrict__ tile_K8, float * const __restrict__ dK) {
        constexpr int warp_size = ggml_cuda_get_physical_warp_size();
        constexpr int nbatch2   = tile_nbatch2<stride_tile, swz>;
        constexpr size_t stride_staging = nbytes_staging_row(nbatch2);
        constexpr int blocks_per_row = nbatch2 / h2_per_block;
        constexpr int nthreads = nwarps*warp_size;

        const int tid = threadIdx.y*warp_size + threadIdx.x;

#pragma unroll
        for (int idx = tid; idx < nbatch_fa*blocks_per_row; idx += nthreads) {
            const int i = idx / blocks_per_row;
            const int b = idx % blocks_per_row;

            const uint32_t * ws = (const uint32_t *) (staging + i*stride_staging);
            const int boff = b*(int)sizeof(block_q8_0);

            dK[i*blocks_per_row + b] = __ushort_as_half((unsigned short) (ws[boff >> 2] >> 8*(boff & 3)));

            const int j0 = (boff + 2) >> 2;    // first word of the qs bytes
            const int r  = 8*((boff + 2) & 3); // bit offset of the qs bytes within that word, 0 or 16

            __align__(16) uint32_t qs[QK8_0/4];
            uint32_t cur = ws[j0];
#pragma unroll
            for (int k = 0; k < QK8_0/4; ++k) {
                const uint32_t nxt = k < QK8_0/4 - 1 || r != 0 ? ws[j0 + k + 1] : 0;
                qs[k] = __funnelshift_r(cur, nxt, r);
                cur = nxt;
            }
#pragma unroll
            for (int k = 0; k < QK8_0/4; k += 4) {
                ggml_cuda_memcpy_1<16>(tile_K8 + i*stride_K8 + QK8_0*b + 4*k, qs + k);
            }
        }

        __syncthreads(); // Make the deinterleaved tile visible to all threads.
    }
    // Dequantizes the whole block at even byte offset boff of the word-aligned staging row ws
    // into 16 half2.
    // Each byte is placed in the mantissa of the f16 magic constant 1536 (0x6600) after flipping
    // its sign bit, so 0x6600|(q^0x80) == 1536 + 128 + q exactly; the bias subtraction is exact in
    // f16 and the multiply rounds the (in f32 exact) product q*d once, so the result is
    // bit-identical to the f32-multiply dequantize_q8_0 conversion path while avoiding all
    // I2F/F2F instructions.
    static __device__ __forceinline__ void dequant_block(const uint32_t * __restrict__ ws, const int boff, half2 * __restrict__ val) {
        const half2 d2   = fattn_mma_load_scale(ws, boff);
        const half2 bias = fattn_mma_u32_as_half2(0x66806680); // half2(1664, 1664)

        const int j0 = (boff + 2) >> 2;    // first word of the qs bytes
        const int r  = 8*((boff + 2) & 3); // bit offset of the qs bytes within that word, 0 or 16

        uint32_t cur = ws[j0];
#pragma unroll
        for (int k = 0; k < QK8_0/4; ++k) {
            // For the last word the load is needed (and in bounds, see nbytes_staging_row) only if
            // the qs bytes are misaligned; with r == 0 the funnel shift returns cur unchanged.
            const uint32_t nxt = k < QK8_0/4 - 1 || r != 0 ? ws[j0 + k + 1] : 0;
            const uint32_t q   = __funnelshift_r(cur, nxt, r) ^ 0x80808080;
            cur = nxt;

            const uint32_t h01 = __byte_perm(q, 0x66666666, 0x4140);
            const uint32_t h23 = __byte_perm(q, 0x66666666, 0x4342);

            val[2*k + 0] = __hmul2(__hsub2(fattn_mma_u32_as_half2(h01), bias), d2);
            val[2*k + 1] = __hmul2(__hsub2(fattn_mma_u32_as_half2(h23), bias), d2);
        }
    }

    // Dequantizes a 4-element chunk l of a block at src (2-byte aligned) into 2 half2.
    static __device__ __forceinline__ void dequant_chunk(const char * __restrict__ src, const int l, half2 * __restrict__ val) {
        half dh;
        ggml_cuda_memcpy_1<sizeof(half)>(&dh, src);
        const float d = dh;

        int8_t q[4];
        ggml_cuda_memcpy_1<2*sizeof(int8_t)>(q + 0, src + sizeof(half) + 4*l + 0);
        ggml_cuda_memcpy_1<2*sizeof(int8_t)>(q + 2, src + sizeof(half) + 4*l + 2);

        val[0] = make_half2(q[0]*d, q[1]*d);
        val[1] = make_half2(q[2]*d, q[3]*d);
    }
};

template <>
struct fattn_mma_tile_loader<GGML_TYPE_Q4_0>
        : fattn_mma_tile_loader_quant<fattn_mma_tile_loader<GGML_TYPE_Q4_0>, block_q4_0> {
    // For the int8 tensor core KQ path: deinterleave the staged tile into centered int8 qs
    // (nibble - 8 as signed bytes, 32 bytes per block, 16-byte aligned rows of stride_K8 bytes)
    // and a separate per-block f32 scale array dK. Centering during the expansion makes the
    // int8 tile indistinguishable from q8_0 downstream: no zero-point correction term needed.
    // The int8 tile has its own linear layout; swz only affects the tile-stride bookkeeping.
    template <int stride_tile, bool swz, int stride_K8, int nwarps, int nbatch_fa>
    static __device__ __forceinline__ void finish_imma(
            const char * const __restrict__ staging, char * const __restrict__ tile_K8, float * const __restrict__ dK) {
        constexpr int warp_size = ggml_cuda_get_physical_warp_size();
        constexpr int nbatch2   = tile_nbatch2<stride_tile, swz>;
        constexpr size_t stride_staging = nbytes_staging_row(nbatch2);
        constexpr int blocks_per_row = nbatch2 / h2_per_block;
        constexpr int nthreads = nwarps*warp_size;

        const int tid = threadIdx.y*warp_size + threadIdx.x;

#pragma unroll
        for (int idx = tid; idx < nbatch_fa*blocks_per_row; idx += nthreads) {
            const int i = idx / blocks_per_row;
            const int b = idx % blocks_per_row;

            const uint32_t * ws = (const uint32_t *) (staging + i*stride_staging);
            const int boff = b*(int)sizeof(block_q4_0);

            dK[i*blocks_per_row + b] = __ushort_as_half((unsigned short) (ws[boff >> 2] >> 8*(boff & 3)));

            const int j0 = (boff + 2) >> 2;    // first word of the qs bytes
            const int r  = 8*((boff + 2) & 3); // bit offset of the qs bytes within that word, 0 or 16

            __align__(16) uint32_t qs[QK4_0/4];
            uint32_t cur = ws[j0];
#pragma unroll
            for (int k = 0; k < QK4_0/8; ++k) {
                const uint32_t nxt = k < QK4_0/8 - 1 || r != 0 ? ws[j0 + k + 1] : 0;
                const uint32_t q   = __funnelshift_r(cur, nxt, r);
                cur = nxt;

                const uint32_t lo =  q       & 0x0F0F0F0F; // elements 4*k +  0 .. 4*k +  3
                const uint32_t hi = (q >> 4) & 0x0F0F0F0F; // elements 4*k + 16 .. 4*k + 19

                // nibble - 8 as signed bytes without a borrow between bytes: flip bit 3 and
                // fill the high nibble with 1s where n < 8 (0x08 * 0x1E == 0xF0, no carry).
                qs[k    ] = (lo ^ 0x08080808) | ((~lo & 0x08080808)*0x1E);
                qs[k + 4] = (hi ^ 0x08080808) | ((~hi & 0x08080808)*0x1E);
            }
#pragma unroll
            for (int k = 0; k < QK4_0/4; k += 4) {
                ggml_cuda_memcpy_1<16>(tile_K8 + i*stride_K8 + QK4_0*b + 4*k, qs + k);
            }
        }

        __syncthreads(); // Make the deinterleaved tile visible to all threads.
    }
    // q4_0 packs element j in the low nibble and element j+16 in the high nibble of byte j.
    // Dequantizes the whole block at even byte offset boff of the word-aligned staging row ws
    // into 16 half2.
    // Each nibble is placed in the mantissa of the f16 magic constant 1024 (0x6400), so
    // 0x6400|n == 1024 + n exactly; the bias subtraction of 1024 + 8 is exact in f16 and the
    // multiply rounds the (in f32 exact) product (n - 8)*d once, so the result is bit-identical
    // to the f32-multiply dequantize_q4_0 conversion path while avoiding all I2F/F2F instructions.
    static __device__ __forceinline__ void dequant_block(const uint32_t * __restrict__ ws, const int boff, half2 * __restrict__ val) {
        const half2 d2   = fattn_mma_load_scale(ws, boff);
        const half2 bias = fattn_mma_u32_as_half2(0x64086408); // half2(1032, 1032)

        const int j0 = (boff + 2) >> 2;    // first word of the qs bytes
        const int r  = 8*((boff + 2) & 3); // bit offset of the qs bytes within that word, 0 or 16

        uint32_t cur = ws[j0];
#pragma unroll
        for (int k = 0; k < QK4_0/8; ++k) {
            const uint32_t nxt = k < QK4_0/8 - 1 || r != 0 ? ws[j0 + k + 1] : 0;
            const uint32_t q   = __funnelshift_r(cur, nxt, r);
            cur = nxt;
            const uint32_t lo =  q       & 0x0F0F0F0F; // elements 4*k +  0 .. 4*k +  3
            const uint32_t hi = (q >> 4) & 0x0F0F0F0F; // elements 4*k + 16 .. 4*k + 19

            const uint32_t l01 = __byte_perm(lo, 0x64646464, 0x4140);
            const uint32_t l23 = __byte_perm(lo, 0x64646464, 0x4342);
            const uint32_t h01 = __byte_perm(hi, 0x64646464, 0x4140);
            const uint32_t h23 = __byte_perm(hi, 0x64646464, 0x4342);

            val[    2*k + 0] = __hmul2(__hsub2(fattn_mma_u32_as_half2(l01), bias), d2);
            val[    2*k + 1] = __hmul2(__hsub2(fattn_mma_u32_as_half2(l23), bias), d2);
            val[8 + 2*k + 0] = __hmul2(__hsub2(fattn_mma_u32_as_half2(h01), bias), d2);
            val[8 + 2*k + 1] = __hmul2(__hsub2(fattn_mma_u32_as_half2(h23), bias), d2);
        }
    }

    // Dequantizes a 4-element chunk l of a block at src (2-byte aligned) into 2 half2.
    static __device__ __forceinline__ void dequant_chunk(const char * __restrict__ src, const int l, half2 * __restrict__ val) {
        half dh;
        ggml_cuda_memcpy_1<sizeof(half)>(&dh, src);
        const float d = dh;

        const int shift = (l >> 2)*4; // chunks 0-3 = low nibbles, chunks 4-7 = high nibbles

        uint8_t q[4];
        ggml_cuda_memcpy_1<2*sizeof(uint8_t)>(q + 0, src + sizeof(half) + 4*(l & 3) + 0);
        ggml_cuda_memcpy_1<2*sizeof(uint8_t)>(q + 2, src + sizeof(half) + 4*(l & 3) + 2);

        val[0] = make_half2((((q[0] >> shift) & 0xF) - 8.0f)*d, (((q[1] >> shift) & 0xF) - 8.0f)*d);
        val[1] = make_half2((((q[2] >> shift) & 0xF) - 8.0f)*d, (((q[3] >> shift) & 0xF) - 8.0f)*d);
    }
};
// Total dynamic shared memory of the 2-stage pipeline incl. staging buffers.
// Must not underestimate the layout in flash_attn_ext_f16_process_tile /
// ggml_cuda_flash_attn_ext_mma_f16_case: the +4 padded strides are an upper bound that also
// covers the swizzled (unpadded) tile layout, and the host and device versions agree on it.
template <ggml_type type_K, ggml_type type_V>
static constexpr __host__ __device__ size_t fattn_mma_staged_nbytes_shared_2stage(
        const int DKQ, const int ncols1, const int ncols2,
        const int nbatch_fa, const int nbatch_K2, const int nbatch_V2, const bool Q_in_reg) {
    const size_t nbytes_KV      = nbatch_fa*(nbatch_K2 + 4 + nbatch_V2 + 4)*sizeof(half2);
    const size_t nbytes_mask    = ncols1*(nbatch_fa/2 + 4)*sizeof(half2);
    const size_t nbytes_Q       = ncols1*ncols2*(DKQ/2 + 4)*sizeof(half2);
    // Scratch for the in-kernel Q quantization of the int8 KQ path, placed after the Q tile:
    const size_t nbytes_Q_scr   = type_K != GGML_TYPE_F16 && Q_in_reg ? ncols1*ncols2*(QK8_0 + 4*(DKQ/QK8_0)) : 0;
    const size_t nbytes_staging_K = fattn_mma_tile_loader<type_K>::nbytes_staging(nbatch_fa, nbatch_K2);
    const size_t nbytes_staging_V = fattn_mma_tile_loader<type_V>::nbytes_staging(nbatch_fa, nbatch_V2);
    const size_t nbytes_staging   = nbytes_staging_K > nbytes_staging_V ? nbytes_staging_K : nbytes_staging_V;
    const size_t nbytes_KV_total = nbytes_KV + nbytes_mask + nbytes_staging;
    return Q_in_reg ? (nbytes_Q + nbytes_Q_scr > nbytes_KV_total ? nbytes_Q + nbytes_Q_scr : nbytes_KV_total) : nbytes_Q + nbytes_KV_total;
}

// nstages for a given KV type pair: same as for f16 unless the staging buffers for the raw
// quantized data would not fit in shared memory with 2 stages -> fall back to a single stage.
template <ggml_type type_K, ggml_type type_V>
static constexpr __device__ int fattn_mma_get_nstages_kv(const int DKQ, const int DV, const int ncols1, const int ncols2) {
    const int nstages = ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2);
    if ((type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) || nstages <= 1) {
        return nstages;
    }
    const int  ncols     = ncols1*ncols2;
    const int  nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols, type_K, type_V);
    const int  nbatch_K2 = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols, type_K, type_V);
    const int  nbatch_V2 = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols, type_K, type_V);
    const bool Q_in_reg  = ggml_cuda_fattn_mma_get_Q_in_reg (DKQ, DV, ncols, type_K, type_V);
    return fattn_mma_staged_nbytes_shared_2stage<type_K, type_V>(DKQ, ncols1, ncols2, nbatch_fa, nbatch_K2, nbatch_V2, Q_in_reg)
        <= fattn_mma_max_nbytes_shared_device() ? nstages : 1;
}

template <ggml_type type_K, ggml_type type_V>
static __host__ int fattn_mma_get_nstages_kv(const int DKQ, const int DV, const int ncols1, const int ncols2, const int cc) {
    const int nstages = ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2, cc);
    if ((type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) || nstages <= 1) {
        return nstages;
    }
    const int  ncols     = ncols1*ncols2;
    const int  nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols, cc, type_K, type_V);
    const int  nbatch_K2 = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols, cc, type_K, type_V);
    const int  nbatch_V2 = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols, cc, type_K, type_V);
    const bool Q_in_reg  = ggml_cuda_fattn_mma_get_Q_in_reg (DKQ, DV, ncols, cc, type_K, type_V);
    return fattn_mma_staged_nbytes_shared_2stage<type_K, type_V>(DKQ, ncols1, ncols2, nbatch_fa, nbatch_K2, nbatch_V2, Q_in_reg)
        <= fattn_mma_max_nbytes_shared_host(cc) ? nstages : 1;
}
