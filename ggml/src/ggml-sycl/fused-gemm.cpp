#include "fused-gemm.hpp"

#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <mutex>
#include <unordered_map>

namespace mx = sycl::ext::oneapi::experimental::matrix;

// XMX f16 tile: 8x16 (A) times 16x16 (B) into an 8x16 f32 accumulator
static constexpr int FG_TM = 8;
static constexpr int FG_TN = 16;
static constexpr int FG_TK = 16;

// a sub-group owns 16 rows (2 A tiles), stages 32 weight values per lane per step and
// walks its own K range; the sub-groups of a work-group are summed at the end
static constexpr int FG_SG_ROWS = 2 * FG_TM;
static constexpr int FG_BK      = QK4_NL;   // iq3_s splits its superblock into steps of this width
static constexpr int FG_BN      = 2 * FG_TN;
static constexpr int FG_KSPLIT  = 4;
static constexpr int FG_WG_SIZE = FG_KSPLIT * WARP_SIZE;
static_assert(FG_SG_ROWS == WARP_SIZE, "the A stage maps one lane to one row");
static_assert(2 * FG_BN == GGML_SYCL_FG_MAX_N, "header gate must match the tile width");

static size_t grouped_gemm_packed_capacity(size_t size) {
    size_t capacity = 1;
    while (capacity < size) {
        capacity *= 2;
    }
    return capacity;
}

static bool fused_gemm_f16_supported(dpct::queue_ptr stream) {
    try {
        const auto combinations = stream->get_device().get_info<
            sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        for (const auto & c : combinations) {
            if (c.atype == mx::matrix_type::fp16 && c.btype == mx::matrix_type::fp16 &&
                c.ctype == mx::matrix_type::fp32 && c.dtype == mx::matrix_type::fp32 &&
                (c.max_msize >= FG_TM || c.msize == FG_TM) &&
                (c.max_nsize >= FG_TN || c.nsize == FG_TN) &&
                (c.max_ksize >= FG_TK || c.ksize == FG_TK)) {
                return true;
            }
        }
    } catch (const sycl::exception &) {
    }
    return false;
}

// src1 [N][K] -> VNNI packed [K/2][Npad][2] so B tiles load straight from global memory
static void fused_gemm_pack_b(const sycl::half * y, sycl::half * packed, int N, int Npad, int K, dpct::queue_ptr stream) {
    const int kpairs = K / 2;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kpairs), [=](sycl::id<1> id) {
        const int idx = id[0];
        const int n   = idx / kpairs;
        const int kp  = idx - n * kpairs;
        sycl::half v0 = (sycl::half) 0.0f;
        sycl::half v1 = (sycl::half) 0.0f;
        if (n < N) {
            const sycl::half * src = y + (size_t) n * K + 2 * kp;
            v0 = src[0];
            v1 = src[1];
        }
        sycl::half * out = packed + ((size_t) kp * Npad + n) * 2;
        out[0] = v0;
        out[1] = v1;
    });
}

// A stage: one lane owns one row and decodes FG_BK values of it per k step, with every scale
// folded into the f16 so the mad below sees plain f16. One overload per weight format.
static __dpct_inline__ void fg_stage_a(const block_iq4_nl * __restrict__ xrow, const int kb, sycl::half2 * a) {
    const block_iq4_nl blk = xrow[kb];
    const float        d   = (float) blk.d;
#pragma unroll
    for (int j = 0; j < QK4_NL / 2; j += 2) {
        const uint8_t q0 = blk.qs[j];
        const uint8_t q1 = blk.qs[j + 1];
        a[j / 2]     = sycl::half2((sycl::half) (d * kvalues_iq4nl[q0 & 0xf]), (sycl::half) (d * kvalues_iq4nl[q1 & 0xf]));
        a[j / 2 + 8] = sycl::half2((sycl::half) (d * kvalues_iq4nl[q0 >> 4]),  (sycl::half) (d * kvalues_iq4nl[q1 >> 4]));
    }
}

// iq3_s: k step kb is sub-block kb % 8 of superblock kb / 8. The superblock is 110 bytes, so read
// only the fields of that sub-block instead of copying the block. Same decode as
// dequantize_block_iq3_s: grid entries are taken as dwords and the sign bit is a plain shift.
static __dpct_inline__ void fg_stage_a(const block_iq3_s * __restrict__ xrow, const int kb, sycl::half2 * a) {
    static_assert(QK_K == 256, "the iq3_s A stage assumes 8 sub-blocks per superblock");
    const block_iq3_s * blk = xrow + kb / (QK_K / 32);
    const int           ib8 = kb % (QK_K / 32);
    const uint8_t *     qs  = blk->qs + 8 * ib8;
    const int           qh  = blk->qh[ib8];
    const float         d   = (float) blk->d * (1 + 2 * ((blk->scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint32_t grid1 = iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 256)];
        const uint32_t grid2 = iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)];
        const int      signs = blk->signs[4 * ib8 + il];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float g1a = (float) ((grid1 >> (16 * j + 0)) & 0xff);
            const float g1b = (float) ((grid1 >> (16 * j + 8)) & 0xff);
            const float g2a = (float) ((grid2 >> (16 * j + 0)) & 0xff);
            const float g2b = (float) ((grid2 >> (16 * j + 8)) & 0xff);
            const int   s   = 2 * j;
            a[4 * il + j]     = sycl::half2((sycl::half) (d * ((signs & (1 << (s + 0))) ? -g1a : g1a)),
                                            (sycl::half) (d * ((signs & (1 << (s + 1))) ? -g1b : g1b)));
            a[4 * il + j + 2] = sycl::half2((sycl::half) (d * ((signs & (1 << (s + 4))) ? -g2a : g2a)),
                                            (sycl::half) (d * ((signs & (1 << (s + 5))) ? -g2b : g2b)));
        }
    }
}

// values per stored block, so a row of K values is K/qk blocks
template <typename block_q_t> struct fg_block_traits;
template <> struct fg_block_traits<block_iq4_nl> { static constexpr int qk = QK4_NL; };
template <> struct fg_block_traits<block_iq3_s>  { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq3_xxs> { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq4_xs>  { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq2_xxs> { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq2_xs>  { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq2_s>   { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq1_s>   { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq1_m>   { static constexpr int qk = QK_K; };

// The A stages below are the dequantize_block_iq* kernels rewritten for one k step. There a
// work-item handled one quarter (il) of one 32-wide sub-block (ib); here one lane produces the
// whole step, so il becomes a loop and ib is kb inside the superblock. Each quarter yields 8
// consecutive values, i.e. 4 half2 at a[4*il], so nothing larger than 8 floats is ever live.
#define FG_SUPERBLOCK(T)                                                         \
    static_assert(QK_K == 256, "the " #T " A stage assumes 8 sub-blocks per superblock"); \
    const T * blk = xrow + kb / (QK_K / 32);                                     \
    const int ib  = kb % (QK_K / 32)

static __dpct_inline__ void fg_pack_quarter(const float * __restrict__ t, sycl::half2 * a, int il) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        a[4 * il + j] = sycl::half2((sycl::half) t[2 * j], (sycl::half) t[2 * j + 1]);
    }
}

static __dpct_inline__ void fg_stage_a(const block_iq4_xs * __restrict__ xrow, const int kb, sycl::half2 * a) {
    FG_SUPERBLOCK(block_iq4_xs);
    // low nibbles fill the first half of the step, high nibbles the second, so the two halves
    // land at a[0..7] and a[8..15] and no quarter loop is needed
    const float d = (float) blk->d *
        ((((blk->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((blk->scales_h >> (2 * ib)) & 3) << 4)) - 32);
    const uint8_t * q4 = blk->qs + 16 * ib;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        a[j]     = sycl::half2((sycl::half) (d * kvalues_iq4nl[q4[2 * j] & 0xf]),
                               (sycl::half) (d * kvalues_iq4nl[q4[2 * j + 1] & 0xf]));
        a[8 + j] = sycl::half2((sycl::half) (d * kvalues_iq4nl[q4[2 * j] >> 4]),
                               (sycl::half) (d * kvalues_iq4nl[q4[2 * j + 1] >> 4]));
    }
}

static __dpct_inline__ void fg_stage_a(const block_iq3_xxs * __restrict__ xrow, const int kb, sycl::half2 * a) {
    FG_SUPERBLOCK(block_iq3_xxs);
    const uint8_t *  q3    = blk->qs + 8 * ib;
    const uint16_t * gas   = (const uint16_t *) (blk->qs + QK_K / 4) + 2 * ib;
    const uint32_t   aux32 = gas[0] | (gas[1] << 16);
    const float      d     = (float) blk->d * (0.5f + (aux32 >> 28)) * 0.5f;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid1 = (const uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        const uint8_t * grid2 = (const uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        const uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        float t[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            t[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
            t[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
        fg_pack_quarter(t, a, il);
    }
}

static __dpct_inline__ void fg_stage_a(const block_iq2_xxs * __restrict__ xrow, const int kb, sycl::half2 * a) {
    FG_SUPERBLOCK(block_iq2_xxs);
    const uint16_t * q2    = blk->qs + 4 * ib;
    const uint8_t *  aux8  = (const uint8_t *) q2;
    const uint32_t   aux32 = q2[2] | (q2[3] << 16);
    const float      d     = (float) blk->d * (0.5f + (aux32 >> 28)) * 0.25f;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid  = (const uint8_t *) (iq2xxs_grid + aux8[il]);
        const uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter(t, a, il);
    }
}

static __dpct_inline__ void fg_stage_a(const block_iq2_xs * __restrict__ xrow, const int kb, sycl::half2 * a) {
    FG_SUPERBLOCK(block_iq2_xs);
    const uint16_t * q2 = blk->qs + 4 * ib;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid  = (const uint8_t *) (iq2xs_grid + (q2[il] & 511));
        const float     d     = (float) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        const uint8_t   signs = ksigns_iq2xs[q2[il] >> 9];
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter(t, a, il);
    }
}

static __dpct_inline__ void fg_stage_a(const block_iq2_s * __restrict__ xrow, const int kb, sycl::half2 * a) {
    FG_SUPERBLOCK(block_iq2_s);
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid =
            (const uint8_t *) (iq2s_grid + (blk->qs[4 * ib + il] | ((blk->qh[ib] << (8 - 2 * il)) & 0x300)));
        const float   d     = (float) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        const uint8_t signs = blk->qs[QK_K / 8 + 4 * ib + il];
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter(t, a, il);
    }
}

static __dpct_inline__ void fg_stage_a(const block_iq1_s * __restrict__ xrow, const int kb, sycl::half2 * a) {
    FG_SUPERBLOCK(block_iq1_s);
    const float delta = blk->qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d     = (float) blk->d * (2 * ((blk->qh[ib] >> 12) & 7) + 1);
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        uint32_t       grid32[2];
        const int8_t * q = (const int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] | (((blk->qh[ib] >> (3 * il)) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter(t, a, il);
    }
}

static __dpct_inline__ void fg_stage_a(const block_iq1_m * __restrict__ xrow, const int kb, sycl::half2 * a) {
    FG_SUPERBLOCK(block_iq1_m);
    const uint16_t * sc = (const uint16_t *) blk->scales;
    iq1m_scale_t     scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const int   ib16  = 2 * ib + il / 2;
        const float d     = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);
        const float delta = blk->qh[2 * ib + il / 2] & (0x08 << (4 * (il % 2))) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
        uint32_t       grid32[2];
        const int8_t * q = (const int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] |
                                  (((blk->qh[2 * ib + il / 2] >> (4 * (il % 2))) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter(t, a, il);
    }
}

// one FG_SG_ROWS x FG_BN output tile: B columns [b0, b0 + FG_BN) of packed_b go to dst columns
// [n0, n1), n1 - n0 <= FG_BN
template <typename block_q_t>
[[sycl::reqd_sub_group_size(WARP_SIZE)]]
static void fused_dequant_gemm_tile(
    const block_q_t * __restrict__ x,
    const sycl::half * __restrict__ packed_b,
    float * __restrict__ dst,
    const int M, const int Npad, const int K, const int ldd,
    const int b0, const int n0, const int n1,
    sycl::local_accessor<sycl::half, 1> tile_a,
    sycl::local_accessor<float, 1> tile_c,
    const sycl::nd_item<2> & item) {
    const auto sg     = item.get_sub_group();
    const int  sg_id  = sg.get_group_id()[0];
    const int  lane   = sg.get_local_id()[0];
    const int  m0     = item.get_group(1) * FG_SG_ROWS;
    const int  nstep  = K / FG_BK;
    const int  a_base = sg_id * FG_SG_ROWS * FG_BK;
    const int  c_base = sg_id * FG_SG_ROWS * FG_BN;

    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, FG_TM, FG_TN> acc[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            mx::joint_matrix_fill(sg, acc[mt][nt], 0.0f);
        }
    }

    const int  row    = m0 + lane;
    const bool row_ok = row < M;
    const block_q_t * xrow = x + (size_t) (row_ok ? row : 0) * (K / fg_block_traits<block_q_t>::qk);
    sycl::half2 * a = (sycl::half2 *) &tile_a[a_base + lane * FG_BK];

    const auto b_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                sycl::access::decorated::no>(packed_b);
    const int b_stride = Npad * 2;

    const int kb_begin = (sg_id * nstep) / FG_KSPLIT;
    const int kb_end   = ((sg_id + 1) * nstep) / FG_KSPLIT;
    for (int kb = kb_begin; kb < kb_end; ++kb) {
        if (row_ok) {
            fg_stage_a(xrow, kb, a);
        } else {
#pragma unroll
            for (int j = 0; j < FG_BK / 2; ++j) {
                a[j] = sycl::half2((sycl::half) 0.0f, (sycl::half) 0.0f);
            }
        }
        sycl::group_barrier(sg);

#pragma unroll
        for (int kt = 0; kt < FG_BK / FG_TK; ++kt) {
            const int kp0 = (kb * FG_BK + kt * FG_TK) / 2;
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, FG_TK, FG_TN, mx::layout::ext_intel_packed> sub_b[2];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                mx::joint_matrix_load(sg, sub_b[nt], b_ptr + (size_t) kp0 * b_stride + (b0 + nt * FG_TN) * 2, b_stride);
            }
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, FG_TM, FG_TK, mx::layout::row_major> sub_a;
                mx::joint_matrix_load(sg, sub_a,
                    tile_a.get_multi_ptr<sycl::access::decorated::no>() + a_base + (mt * FG_TM) * FG_BK + kt * FG_TK,
                    FG_BK);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    mx::joint_matrix_mad(sg, acc[mt][nt], sub_a, sub_b[nt], acc[mt][nt]);
                }
            }
        }
        // the next step overwrites tile_a
        sycl::group_barrier(sg);
    }

#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            mx::joint_matrix_store(sg, acc[mt][nt],
                tile_c.get_multi_ptr<sycl::access::decorated::no>() + c_base + (mt * FG_TM) * FG_BN + nt * FG_TN,
                FG_BN, mx::layout::row_major);
        }
    }
    sycl::group_barrier(item.get_group());

    // sum the K splits; consecutive lanes write consecutive rows of one dst column
    for (int idx = item.get_local_linear_id(); idx < FG_SG_ROWS * FG_BN; idx += FG_WG_SIZE) {
        const int r = idx % FG_SG_ROWS;
        const int c = idx / FG_SG_ROWS;
        const int m = m0 + r;
        const int n = n0 + c;
        if (m < M && n < n1) {
            float sum = 0.0f;
#pragma unroll
            for (int s = 0; s < FG_KSPLIT; ++s) {
                sum += tile_c[s * FG_SG_ROWS * FG_BN + r * FG_BN + c];
            }
            dst[(size_t) n * ldd + m] = sum;
        }
    }
}

template <typename block_q_t>
[[sycl::reqd_sub_group_size(WARP_SIZE)]]
static void fused_dequant_gemm(
    const block_q_t * __restrict__ x,
    const sycl::half * __restrict__ packed_b,
    float * __restrict__ dst,
    const int M, const int N, const int Npad, const int K, const int ldd,
    sycl::local_accessor<sycl::half, 1> tile_a,
    sycl::local_accessor<float, 1> tile_c,
    const sycl::nd_item<2> & item) {
    const int n0 = item.get_group(0) * FG_BN;
    fused_dequant_gemm_tile<block_q_t>(x, packed_b, dst, M, Npad, K, ldd, n0, n0, N, tile_a, tile_c, item);
}

// grouped: work-group (t, mt) is tile t of the schedule; its B columns sit at t * FG_BN
template <typename block_q_t>
[[sycl::reqd_sub_group_size(WARP_SIZE)]]
static void grouped_dequant_gemm(
    const char * __restrict__ src0_base,
    const size_t expert_stride,
    const ggml_sycl_gg_tile * __restrict__ tiles,
    const sycl::half * __restrict__ packed_b,
    float * __restrict__ dst,
    const int M, const int Npad, const int K,
    sycl::local_accessor<sycl::half, 1> tile_a,
    sycl::local_accessor<float, 1> tile_c,
    const sycl::nd_item<2> & item) {
    const int t = item.get_group(0);
    const ggml_sycl_gg_tile tile = tiles[t];
    const block_q_t * x = (const block_q_t *) (src0_base + (size_t) tile.expert * expert_stride);
    fused_dequant_gemm_tile<block_q_t>(x, packed_b, dst, M, Npad, K, M, t * FG_BN, tile.n0, tile.n1, tile_a, tile_c, item);
}

template <typename block_q_t>
static void fused_dequant_gemm_launch(const void * src0, const sycl::half * packed, float * dst, const int M,
                                      const int N, const int Npad, const int K, const int ldd,
                                      const int64_t groups_n, const int64_t groups_m, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::half, 1> tile_a(FG_KSPLIT * FG_SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<float, 1>      tile_c(FG_KSPLIT * FG_SG_ROWS * FG_BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_n, groups_m * FG_WG_SIZE), sycl::range<2>(1, FG_WG_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                fused_dequant_gemm<block_q_t>((const block_q_t *) src0, packed, dst, M, N, Npad, K, ldd,
                                              tile_a, tile_c, item);
            });
    });
}

template <typename block_q_t>
static void grouped_dequant_gemm_launch(const char * src0_dd, const size_t expert_stride,
                                        const ggml_sycl_gg_tile * tiles_ptr, const sycl::half * packed, float * dst,
                                        const int M, const int Npad, const int K, const int64_t n_tiles,
                                        const int64_t groups_m, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::half, 1> tile_a(FG_KSPLIT * FG_SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<float, 1>      tile_c(FG_KSPLIT * FG_SG_ROWS * FG_BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_tiles, groups_m * FG_WG_SIZE), sycl::range<2>(1, FG_WG_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                grouped_dequant_gemm<block_q_t>(src0_dd, expert_stride, tiles_ptr, packed, dst, M, Npad, K,
                                                tile_a, tile_c, item);
            });
    });
}

// src1 f32 rows -> VNNI packed [K/2][n_tiles*FG_BN][2], tile t holds its rows [n0, n1) at columns
// t*FG_BN.., zero past n1. The column runs fastest so a sub-group writes one contiguous run.
static void grouped_gemm_pack_b(const float * y, sycl::half * packed, const ggml_sycl_gg_tile * tiles, int Npad, int K,
                                dpct::queue_ptr stream) {
    const int kpairs = K / 2;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kpairs), [=](sycl::id<1> id) {
        const size_t idx = id[0];
        const int    kp  = idx / Npad;
        const int    n   = idx - (size_t) kp * Npad;
        const ggml_sycl_gg_tile tile = tiles[n / FG_BN];
        const int    row = tile.n0 + n % FG_BN;
        sycl::half v0 = (sycl::half) 0.0f;
        sycl::half v1 = (sycl::half) 0.0f;
        if (row < tile.n1) {
            const float * src = y + (size_t) row * K + 2 * kp;
            v0 = (sycl::half) src[0];
            v1 = (sycl::half) src[1];
        }
        sycl::half * out = packed + ((size_t) kp * Npad + n) * 2;
        out[0] = v0;
        out[1] = v1;
    });
}

bool ggml_sycl_fused_dequant_gemm_f16_device_ok(dpct::queue_ptr stream) {
    // Cached per device, not once: on a mixed box the first caller's verdict is not the others'.
    static std::mutex                        mtx;
    static std::unordered_map<sycl::device, bool> known;
    const sycl::device                       dev = stream->get_device();
    std::lock_guard<std::mutex>              lock(mtx);
    const auto                               it = known.find(dev);
    if (it != known.end()) {
        return it->second;
    }
    const bool ok = fused_gemm_f16_supported(stream);
    known.emplace(dev, ok);
    return ok;
}

bool ggml_sycl_fused_dequant_gemm_f16(ggml_type src0_type, const void * src0, const sycl::half * src1_f16, float * dst,
                                      int64_t M, int64_t N, int64_t K, int64_t ldd, ggml_sycl_pool & pool,
                                      dpct::queue_ptr stream) {
    // every FG_BN columns dequantize A again, so wide N is left to the library GEMM
    if (!ggml_sycl_xmx_gather_type_enabled(src0_type)) {
        return false;
    }
    if (!ggml_sycl_fused_dequant_gemm_f16_shape_ok(src0_type, M, N, K, ldd)) {
        return false;
    }
    if (!ggml_sycl_fused_dequant_gemm_f16_device_ok(stream)) {
        return false;
    }

    const int64_t groups_n = (N + FG_BN - 1) / FG_BN;
    const int64_t groups_m = (M + FG_SG_ROWS - 1) / FG_SG_ROWS;
    const int     Npad     = (int) (groups_n * FG_BN);

    ggml_sycl_pool_alloc<sycl::half> packed_b(pool, (size_t) K * Npad);
    fused_gemm_pack_b(src1_f16, packed_b.get(), (int) N, Npad, (int) K, stream);

    const sycl::half * packed = packed_b.get();
    switch (src0_type) {
        case GGML_TYPE_IQ4_NL:
            fused_dequant_gemm_launch<block_iq4_nl>(src0, packed, dst, (int) M, (int) N, Npad, (int) K, (int) ldd,
                                                    groups_n, groups_m, stream);
            break;
        case GGML_TYPE_IQ3_S:
            fused_dequant_gemm_launch<block_iq3_s>(src0, packed, dst, (int) M, (int) N, Npad, (int) K, (int) ldd,
                                                   groups_n, groups_m, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            fused_dequant_gemm_launch<block_iq4_xs>(src0, packed, dst, (int) M, (int) N, Npad, (int) K, (int) ldd,
                                                    groups_n, groups_m, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            fused_dequant_gemm_launch<block_iq3_xxs>(src0, packed, dst, (int) M, (int) N, Npad, (int) K, (int) ldd,
                                                    groups_n, groups_m, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            fused_dequant_gemm_launch<block_iq2_xxs>(src0, packed, dst, (int) M, (int) N, Npad, (int) K, (int) ldd,
                                                    groups_n, groups_m, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            fused_dequant_gemm_launch<block_iq2_xs>(src0, packed, dst, (int) M, (int) N, Npad, (int) K, (int) ldd,
                                                    groups_n, groups_m, stream);
            break;
        case GGML_TYPE_IQ2_S:
            fused_dequant_gemm_launch<block_iq2_s>(src0, packed, dst, (int) M, (int) N, Npad, (int) K, (int) ldd,
                                                    groups_n, groups_m, stream);
            break;
        case GGML_TYPE_IQ1_S:
            fused_dequant_gemm_launch<block_iq1_s>(src0, packed, dst, (int) M, (int) N, Npad, (int) K, (int) ldd,
                                                    groups_n, groups_m, stream);
            break;
        case GGML_TYPE_IQ1_M:
            fused_dequant_gemm_launch<block_iq1_m>(src0, packed, dst, (int) M, (int) N, Npad, (int) K, (int) ldd,
                                                    groups_n, groups_m, stream);
            break;
        default:
            return false;
    }
    return true;
}

bool ggml_sycl_grouped_dequant_gemm_f16(ggml_type src0_type, const void * src0_base, size_t expert_stride,
                                        const float * src1, float * dst, const int64_t * expert_row_offsets,
                                        int64_t n_as, int64_t M, int64_t K, int64_t total_rows,
                                        std::vector<ggml_sycl_gg_tile> & tiles, ggml_sycl_pool & pool,
                                        dpct::queue_ptr stream) {
    int64_t n_active = 0;
    for (int64_t e = 0; e < n_as; ++e) {
        n_active += expert_row_offsets[e + 1] > expert_row_offsets[e];
    }
    if (!ggml_sycl_xmx_gather_type_enabled(src0_type)) {
        return false;
    }
    if (!ggml_sycl_grouped_dequant_gemm_f16_shape_ok(src0_type, M, K, total_rows, n_active)) {
        return false;
    }
    if (!ggml_sycl_fused_dequant_gemm_f16_device_ok(stream)) {
        return false;
    }

    // the host knows every slice, so it lays out the work-groups: no search on the device
    tiles.clear();
    for (int64_t e = 0; e < n_as; ++e) {
        const int64_t end = expert_row_offsets[e + 1];
        for (int64_t n0 = expert_row_offsets[e]; n0 < end; n0 += FG_BN) {
            tiles.push_back({ (int32_t) e, (int32_t) n0, (int32_t) std::min(n0 + FG_BN, end) });
        }
    }
    const int64_t n_tiles  = tiles.size();
    const int64_t groups_m = (M + FG_SG_ROWS - 1) / FG_SG_ROWS;
    const int     Npad     = (int) (n_tiles * FG_BN);

    ggml_sycl_pool_alloc<ggml_sycl_gg_tile> tiles_dev(pool, n_tiles);
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(tiles_dev.get(), tiles.data(), n_tiles * sizeof(ggml_sycl_gg_tile))));

    ggml_sycl_pool_alloc<sycl::half> packed_b(pool, grouped_gemm_packed_capacity((size_t) K * Npad));
    grouped_gemm_pack_b(src1, packed_b.get(), tiles_dev.get(), Npad, (int) K, stream);

    const sycl::half *       packed    = packed_b.get();
    const ggml_sycl_gg_tile * tiles_ptr = tiles_dev.get();
    const char *             src0_dd   = (const char *) src0_base;
    switch (src0_type) {
        case GGML_TYPE_IQ4_NL:
            grouped_dequant_gemm_launch<block_iq4_nl>(src0_dd, expert_stride, tiles_ptr, packed, dst, (int) M, Npad,
                                                      (int) K, n_tiles, groups_m, stream);
            break;
        case GGML_TYPE_IQ3_S:
            grouped_dequant_gemm_launch<block_iq3_s>(src0_dd, expert_stride, tiles_ptr, packed, dst, (int) M, Npad,
                                                     (int) K, n_tiles, groups_m, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            grouped_dequant_gemm_launch<block_iq4_xs>(src0_dd, expert_stride, tiles_ptr, packed, dst, (int) M, Npad,
                                                    (int) K, n_tiles, groups_m, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            grouped_dequant_gemm_launch<block_iq3_xxs>(src0_dd, expert_stride, tiles_ptr, packed, dst, (int) M, Npad,
                                                    (int) K, n_tiles, groups_m, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            grouped_dequant_gemm_launch<block_iq2_xxs>(src0_dd, expert_stride, tiles_ptr, packed, dst, (int) M, Npad,
                                                    (int) K, n_tiles, groups_m, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            grouped_dequant_gemm_launch<block_iq2_xs>(src0_dd, expert_stride, tiles_ptr, packed, dst, (int) M, Npad,
                                                    (int) K, n_tiles, groups_m, stream);
            break;
        case GGML_TYPE_IQ2_S:
            grouped_dequant_gemm_launch<block_iq2_s>(src0_dd, expert_stride, tiles_ptr, packed, dst, (int) M, Npad,
                                                    (int) K, n_tiles, groups_m, stream);
            break;
        case GGML_TYPE_IQ1_S:
            grouped_dequant_gemm_launch<block_iq1_s>(src0_dd, expert_stride, tiles_ptr, packed, dst, (int) M, Npad,
                                                    (int) K, n_tiles, groups_m, stream);
            break;
        case GGML_TYPE_IQ1_M:
            grouped_dequant_gemm_launch<block_iq1_m>(src0_dd, expert_stride, tiles_ptr, packed, dst, (int) M, Npad,
                                                    (int) K, n_tiles, groups_m, stream);
            break;
        default:
            return false;
    }
    return true;
}
