// SPDX-License-Identifier: MIT
// Xe-LPG+ execution-size-8 DPAS kernel, ported from xmx-mul-mat.
#pragma once
#include "common.hpp"

#ifdef GGML_SYCL_MMVQ_HAS_XMX
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/intel/info/device.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <type_traits>

namespace ggml_sycl_esimd_arl {
//
// Multi-column mul_mat_vec for reordered K-quant weights on the XMX engines (int8 DPAS).
// A thread owns 16 weight rows (B operand, one dword is 4 weights of a row), the q8_1 activation
// columns are the A operand. A super-block is 8 groups of 32, run as 2 parts of 4 groups. A quant
// type supplies its loads and unpack as a traits struct.
//
#ifndef XMX_ENABLE_SLM
#    define XMX_ENABLE_SLM 0
#endif
#ifndef XMX_ENABLE_TRANSPOSED_2D_LOADS
#    define XMX_ENABLE_TRANSPOSED_2D_LOADS 0
#endif
#ifndef XMX_ENABLE_GATHER_TRANSPOSE
#    define XMX_ENABLE_GATHER_TRANSPOSE 1
#endif

using sycl::ext::intel::esimd::simd;
constexpr int XMX_ROWS        = 16;  // weight rows per thread (DPAS N)
constexpr int XMX_COLUMN_TILE = 16;  // activation columns accumulated by one work-group
constexpr int XMX_FUSED_COLUMN_TILE = 8;  // limit live registers for the two GLU projections
constexpr int XMX_CHUNK       = 64;  // widest single launch, more columns run as repeated launches

// LSC gathers do not require native 2D block I/O. SLM can split K across work-items.
constexpr bool XMX_USE_SLM = XMX_ENABLE_SLM != 0;
// The software path can schedule several fused tiles in one grid. Keep the
// original chunk size when the grid shape controls K splitting or native loads.
constexpr int XMX_FUSED_CHUNK =
    XMX_USE_SLM || XMX_ENABLE_TRANSPOSED_2D_LOADS ? GGML_SYCL_XMX_GLU_MAX_COLS : XMX_CHUNK;

// threads that split K for one tile, sized so that the shared memory stays small
template <int NC> constexpr int xmx_n_split() {
    return NC <= 8 ? 32 : (NC == 16 ? 16 : 8);
}

inline bool xmx_supported(const void * vx, const int ncols) {
    // 2D block loads need a 64 byte aligned base
    return vx && ncols > 0 && ncols % QK_K == 0 && (uintptr_t) vx % 64 == 0;
}

constexpr bool xmx_supports_dpas8(sycl::ext::oneapi::experimental::architecture arch) {
    using sycl::ext::oneapi::experimental::architecture;
    return arch == architecture::intel_gpu_arl_h || arch == architecture::intel_gpu_acm_g10 ||
           arch == architecture::intel_gpu_acm_g11 || arch == architecture::intel_gpu_acm_g12;
}

constexpr int xmx_activation_scale_bytes(int ncols) {
    return ncols / QK8_1 * sizeof(sycl::half2);
}

constexpr int xmx_activation_source_row(int row, int valid_rows) {
    return row < valid_rows ? row : valid_rows - 1;
}

struct xmx_column_tile {
    int first;
    int width;
};

constexpr int xmx_column_tile_count(int columns) {
    return columns / XMX_COLUMN_TILE + (columns % XMX_COLUMN_TILE != 0);
}

constexpr int xmx_column_tile_width(int valid_columns) {
    return valid_columns <= XMX_COLUMN_TILE / 2 ? valid_columns : XMX_COLUMN_TILE;
}

// Each row tile uses one work-group per full or partial column tile.
constexpr xmx_column_tile xmx_column_tile_at(int columns, int tile) {
    const int first = tile * XMX_COLUMN_TILE;
    return {first, xmx_column_tile_width(columns - first)};
}

// Load 16 rows in DPAS B order: dword (kd, n) lands at [kd * 16 + n].
// Short rows (SR) skip the native 2D block message.
template <int W, bool SR>
ESIMD_INLINE simd<uint32_t, W * XMX_ROWS> xmx_load_t(const void * base, unsigned pitch, unsigned h, int x, int y) {
    using namespace sycl::ext::intel::esimd;
    if constexpr (!SR && XMX_ENABLE_TRANSPOSED_2D_LOADS) {
        return load_2d<uint32_t, W, XMX_ROWS, 1, true, false>((const uint32_t *) base, pitch, h, pitch, x, y);
    } else if constexpr (XMX_ENABLE_GATHER_TRANSPOSE) {
        simd<uint32_t, XMX_ROWS> rows(y, 1);
        rows.merge(h, rows > h);
        const simd<uint64_t, XMX_ROWS> offsets = convert<uint64_t>(rows) * (uint64_t(pitch) + 1);
        // One address per weight row. An LSC vector gather returns the first
        // dword from all lanes, then the second, and so on.
        return gather<uint32_t, W * XMX_ROWS, W>(static_cast<const uint32_t *>(base) + x, offsets);
    } else {
        simd<uint32_t, W * XMX_ROWS> r;
        for (int n = 0; n < XMX_ROWS; ++n) {
            const unsigned row                = y + n < (int) h ? y + n : h;
            const char *   p                  = (const char *) base + (size_t) row * (pitch + 1);
            // select<W, XMX_ROWS>(n) places this row's K dwords at
            // [kd * XMX_ROWS + row], which is the DPAS VNNI B layout.
            r.template select<W, XMX_ROWS>(n) = block_load<uint32_t, W>((const uint32_t *) p + x);
        }
        // The strided select above directly forms VNNI [k_dword][row] order.
        return r;
    }
}

// Xe-LPG+ bring-up keeps activation 2D messages opt-in. The row pitch is in
// uint32 elements; each row is loaded independently in the default path.
template <int W, int H, bool CLAMP = false>
ESIMD_INLINE simd<uint32_t, W * H> xmx_load_rows(const uint32_t * base, unsigned row_pitch, int x, int y, int ncv = 0) {
    simd<uint32_t, W * H> r;
#pragma unroll
    for (int row = 0; row < H; ++row) {
        // A is horizontal: each source row must occupy one contiguous W-wide span.
        const int source_row = CLAMP ? xmx_activation_source_row(y + row, ncv) : y + row;
        r.template select<W, 1>(row * W) = sycl::ext::intel::esimd::block_load<uint32_t, W>(
            base + (size_t) source_row * row_pitch + x);
    }
    return r;
}

template <int W, int H, bool CLAMP = false, bool COMBINE = false>
ESIMD_INLINE simd<uint32_t, 2 * W * H> xmx_load_rows_pair(const uint32_t * base, unsigned row_pitch, int x, int y,
                                                       int ncv = 0) {
    simd<uint32_t, 2 * W * H> r;
    // Both load strategies produce [group][row][K] for xmx_group.
    if constexpr (COMBINE) {
#pragma unroll
        for (int row = 0; row < H; ++row) {
            const int source_row = CLAMP ? xmx_activation_source_row(y + row, ncv) : y + row;
            auto values = sycl::ext::intel::esimd::block_load<uint32_t, 2 * W>(
                base + (size_t) source_row * row_pitch + x);
            r.template select<W, 1>(row * W) = values.template select<W, 1>(0);
            r.template select<W, 1>(H * W + row * W) = values.template select<W, 1>(W);
        }
    } else {
#pragma unroll
        for (int group = 0; group < 2; ++group) {
#pragma unroll
            for (int row = 0; row < H; ++row) {
                const int source_row = CLAMP ? xmx_activation_source_row(y + row, ncv) : y + row;
                r.template select<W, 1>(group * H * W + row * W) =
                    sycl::ext::intel::esimd::block_load<uint32_t, W>(
                        base + (size_t) source_row * row_pitch + x + group * W);
            }
        }
    }
    return r;
}

// per byte (q - zero) as int8 for q in 0..127, no borrow between bytes
ESIMD_INLINE simd<uint32_t, 128> xmx_bytes_sub(simd<uint32_t, 128> x, uint32_t zero_x4) {
    return ((x | 0x80808080u) - zero_x4) ^ 0x80808080u;
}

// How the scales of a quant type enter the DPAS epilogue of a group of 32 (see xmx_group)
enum class xmx_scale_kind {
    split,  // two scales per group, one per 16 wide half: the weights are masked to one half for each DPAS
    min,    // one scale and one min per group: one DPAS sums the activations for all output rows
    min16,  // a scale and a min per 16 wide half: the scales are folded into the DPAS operands
    plain,  // one scale per group
};

template <bool SR, xmx_scale_kind KIND, int MIN_ROW_BYTES> struct xmx_traits_base {
    static constexpr bool short_rows = SR;
    static constexpr int  min_2d_bytes_per_block = MIN_ROW_BYTES;
    static constexpr bool has_min    = KIND == xmx_scale_kind::min;
    static constexpr bool plain      = KIND == xmx_scale_kind::plain;
    static constexpr bool min16      = KIND == xmx_scale_kind::min16;
    static constexpr bool combine_activation_loads = false;
};

// The part of a context every quant type shares: the last row (the clamp of 2D loads), this thread's first row and
// the number of super-blocks per row. A type adds its array pointers and 2D load pitches.
struct xmx_ctx_base {
    unsigned h;
    int      row0;
    int      bpr;
};

ESIMD_INLINE void xmx_ctx_init(xmx_ctx_base & c, int nrows, int bpr, int row0) {
    c.h    = nrows - 1;
    c.row0 = row0;
    c.bpr  = bpr;
}

// pitch of a 2D load over an array with `bytes` bytes per super-block
constexpr unsigned xmx_pitch(int bpr, int bytes) {
    return bpr * bytes - 1;
}

// per group row scales of the 16 rows, one vector per group
template <int N> struct xmx_row_scales {
    simd<float, 16> v[N];

    ESIMD_INLINE simd<float, 16> get(int i) { return v[i]; }
};

// the 12 packed scale bytes of a super-block (Q3_K, Q4_K, Q5_K) of the 16 rows as three dwords
ESIMD_INLINE void xmx_load_sc12(const uint8_t *            sc,
                                const simd<uint32_t, 16> & bi,
                                simd<uint32_t, 16> &       w0,
                                simd<uint32_t, 16> &       w1,
                                simd<uint32_t, 16> &       w2) {
    using namespace sycl::ext::intel::esimd;
    w0 = gather<uint32_t, 16>((const uint32_t *) sc, bi * 12);
    w1 = gather<uint32_t, 16>((const uint32_t *) sc, bi * 12 + 4);
    w2 = gather<uint32_t, 16>((const uint32_t *) sc, bi * 12 + 8);
}

// d and dmin (a half2 per super-block, Q2_K, Q4_K, Q5_K) of the 16 rows
ESIMD_INLINE void xmx_load_dm(const sycl::half *          dm,
                              const simd<uint32_t, 16> & bi,
                              simd<float, 16> &          d,
                              simd<float, 16> &          dmin) {
    using namespace sycl::ext::intel::esimd;
    d    = convert<float>(gather<sycl::half, 16>(dm, bi * (2 * (uint32_t) sizeof(sycl::half))));
    dmin = convert<float>(
        gather<sycl::half, 16>(dm, bi * (2 * (uint32_t) sizeof(sycl::half)) + (uint32_t) sizeof(sycl::half)));
}

// Q6_K reorder layout: [ql: nb*128] [qh: nb*64] [scales int8: nb*16] [d: nb*2]
template <bool SR> struct xmx_traits_q6_k : xmx_traits_base<SR, xmx_scale_kind::split, QK_K / 16> {
    using short_row_traits = xmx_traits_q6_k<true>;

    struct ctx : xmx_ctx_base {
        const uint8_t *    ql;
        const uint8_t *    qh;
        const int8_t *     scales;
        const sycl::half * dh;
        unsigned           pl, ph, ps;
    };

    static ESIMD_INLINE ctx make(const void * vx, int nrows, int bpr, int row0) {
        const size_t nb = (size_t) nrows * bpr;
        ctx          c;
        xmx_ctx_init(c, nrows, bpr, row0);
        c.ql     = (const uint8_t *) vx;
        c.qh     = c.ql + nb * (QK_K / 2);
        c.scales = (const int8_t *) (c.qh + nb * (QK_K / 4));
        c.dh     = (const sycl::half *) (c.scales + nb * (QK_K / 16));
        c.pl     = xmx_pitch(bpr, QK_K / 2);
        c.ph     = xmx_pitch(bpr, QK_K / 4);
        c.ps     = xmx_pitch(bpr, QK_K / 16);
        return c;
    }

    // row scales of the 16 scale groups of a super-block
    struct scales_t {
        simd<int8_t, 256> sc8;
        simd<float, 16>   dv;

        ESIMD_INLINE simd<float, 16> get(int gr) {
            simd<int8_t, 16> s = sc8.template select<16, 4>(64 * (gr / 4) + (gr % 4));
            return sycl::ext::intel::esimd::convert<float>(s) * dv;
        }
    };

    // rows: clamped row index of each of the 16 lanes
    static ESIMD_INLINE scales_t load_scales(const ctx & c, int sb, const simd<uint32_t, 16> & rows) {
        using namespace sycl::ext::intel::esimd;
        scales_t s;
        s.dv = convert<float>(gather<sycl::half, 16>(c.dh, (rows * c.bpr + sb) * (uint32_t) sizeof(sycl::half)));
        simd<uint32_t, 64> raw = xmx_load_t<4, SR>(c.scales, c.ps, c.h, sb * 4, c.row0);
        s.sc8                  = raw.template bit_cast_view<int8_t>();
        return s;
    }

    struct part_t {
        simd<uint32_t, 128> a, b, h;
    };

    static ESIMD_INLINE part_t load_part(const ctx & c, int sb, int part) {
        part_t p;
        p.a = xmx_load_t<8, SR>(c.ql, c.pl, c.h, sb * 32 + part * 16, c.row0);
        p.b = xmx_load_t<8, SR>(c.ql, c.pl, c.h, sb * 32 + part * 16 + 8, c.row0);
        p.h = xmx_load_t<8, SR>(c.qh, c.ph, c.h, sb * 16 + part * 8, c.row0);
        return p;
    }

    static ESIMD_INLINE simd<uint32_t, 128> unpack(const part_t & p, int g, int /*part*/) {
        simd<uint32_t, 128> qs = (g & 1) ? p.b : p.a;
        simd<uint32_t, 128> x;
        if (g == 0) {
            x = (qs & 0x0F0F0F0Fu) | ((p.h << 4) & 0x30303030u);
        } else if (g == 1) {
            x = (qs & 0x0F0F0F0Fu) | ((p.h << 2) & 0x30303030u);
        } else if (g == 2) {
            x = ((qs >> 4) & 0x0F0F0F0Fu) | (p.h & 0x30303030u);
        } else {
            x = ((qs >> 4) & 0x0F0F0F0Fu) | ((p.h >> 2) & 0x30303030u);
        }
        return xmx_bytes_sub(x, 0x20202020u);
    }
};

// Q3_K reorder layout: [qs: nb*64] [hmask: nb*32] [scales: nb*12] [d: nb*2]
// chunk s of 32: 2 bits from qs byte 32*(s/4)+l at shift 2*(s%4), high bit is hmask[l] bit s, value is q - 4
template <bool SR> struct xmx_traits_q3_k : xmx_traits_base<SR, xmx_scale_kind::split, QK_K / 8> {
    using short_row_traits = xmx_traits_q3_k<true>;

    struct ctx : xmx_ctx_base {
        const uint8_t *    qs;
        const uint8_t *    hm;
        const uint8_t *    sc;
        const sycl::half * dh;
        unsigned           pq, ph;
    };

    static ESIMD_INLINE ctx make(const void * vx, int nrows, int bpr, int row0) {
        const size_t nb = (size_t) nrows * bpr;
        ctx          c;
        xmx_ctx_init(c, nrows, bpr, row0);
        c.qs = (const uint8_t *) vx;
        c.hm = c.qs + nb * (QK_K / 4);
        c.sc = c.hm + nb * (QK_K / 8);
        c.dh = (const sycl::half *) (c.sc + nb * 12);
        c.pq = xmx_pitch(bpr, QK_K / 4);
        c.ph = xmx_pitch(bpr, QK_K / 8);
        return c;
    }

    using scales_t = xmx_row_scales<16>;

    static ESIMD_INLINE scales_t load_scales(const ctx & c, int sb, const simd<uint32_t, 16> & rows) {
        using namespace sycl::ext::intel::esimd;
        scales_t           s;
        simd<uint32_t, 16> bi = rows * c.bpr + sb;
        simd<uint32_t, 16> w0, w1, w2;
        xmx_load_sc12(c.sc, bi, w0, w1, w2);
        simd<float, 16> dv = convert<float>(gather<sycl::half, 16>(c.dh, bi * (uint32_t) sizeof(sycl::half)));
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const simd<uint32_t, 16> lo   = (i % 8) < 4 ? w0 : w1;
            const int                bl   = i % 4;
            simd<uint32_t, 16>       nib  = (lo >> (8 * bl + (i < 8 ? 0 : 4))) & 0xFu;
            simd<uint32_t, 16>       hi   = ((w2 >> (8 * bl + 2 * (i / 4))) & 3u) << 4;
            simd<int, 16>            code = nib | hi;
            s.v[i]                        = convert<float>(code - 32) * dv;
        }
        return s;
    }

    struct part_t {
        simd<uint32_t, 128> q, h;
    };

    static ESIMD_INLINE part_t load_part(const ctx & c, int sb, int part) {
        part_t p;
        p.q = xmx_load_t<8, SR>(c.qs, c.pq, c.h, sb * 16 + part * 8, c.row0);
        p.h = xmx_load_t<8, SR>(c.hm, c.ph, c.h, sb * 8, c.row0);
        return p;
    }

    static ESIMD_INLINE simd<uint32_t, 128> unpack(const part_t & p, int g, int part) {
        const int           s = 4 * part + g;
        simd<uint32_t, 128> x = ((p.q >> (2 * g)) & 0x03030303u) | (((p.h >> s) & 0x01010101u) << 2);
        return xmx_bytes_sub(x, 0x04040404u);
    }
};

// Q4_K and Q5_K (QH) reorder layout: [qs: nb*128] ([qh: nb*32]) [scales: nb*12] [dm: nb*4]
// group j of 32: nibble of qs byte 32*(j/2)+l (low for even j), Q5_K adds 16 * bit j of qh[l]
// value is dall * sc_j * q - dmin * m_j, one scale and one min per group
template <bool QH, bool SR>
struct xmx_traits_q45_k : xmx_traits_base<SR, xmx_scale_kind::min, QH ? QK_K / 8 : QK_K / 2> {
    using short_row_traits = xmx_traits_q45_k<QH, true>;
    static constexpr bool combine_activation_loads = !QH;

    struct ctx : xmx_ctx_base {
        const uint8_t *    qs;
        const uint8_t *    qh;
        const uint8_t *    sc;
        const sycl::half * dm;
        unsigned           pq, ph;
    };

    static ESIMD_INLINE ctx make(const void * vx, int nrows, int bpr, int row0) {
        const size_t nb = (size_t) nrows * bpr;
        ctx          c;
        xmx_ctx_init(c, nrows, bpr, row0);
        c.qs = (const uint8_t *) vx;
        c.qh = c.qs + nb * (QK_K / 2);
        c.sc = QH ? c.qh + nb * (QK_K / 8) : c.qh;
        c.dm = (const sycl::half *) (c.sc + nb * 12);
        c.pq = xmx_pitch(bpr, QK_K / 2);
        c.ph = xmx_pitch(bpr, QK_K / 8);
        return c;
    }

    struct scales_t {
        xmx_row_scales<8> d;  // dall * scale of each group
        xmx_row_scales<8> m;  // dmin * min of each group
    };

    static ESIMD_INLINE scales_t load_scales(const ctx & c, int sb, const simd<uint32_t, 16> & rows) {
        using namespace sycl::ext::intel::esimd;
        scales_t           s;
        simd<uint32_t, 16> bi = rows * c.bpr + sb;
        simd<uint32_t, 16> w0, w1, w2;
        xmx_load_sc12(c.sc, bi, w0, w1, w2);
        simd<float, 16> d, dmin;
        xmx_load_dm(c.dm, bi, d, dmin);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int          jj = j % 4;
            simd<uint32_t, 16> sc, m;
            if (j < 4) {
                sc = (w0 >> (8 * jj)) & 63u;
                m  = (w1 >> (8 * jj)) & 63u;
            } else {
                sc = ((w2 >> (8 * jj)) & 0xFu) | (((w0 >> (8 * jj + 6)) & 3u) << 4);
                m  = ((w2 >> (8 * jj + 4)) & 0xFu) | (((w1 >> (8 * jj + 6)) & 3u) << 4);
            }
            s.d.v[j] = convert<float>(simd<int, 16>(sc)) * d;
            s.m.v[j] = convert<float>(simd<int, 16>(m)) * dmin;
        }
        return s;
    }

    struct part_t {
        simd<uint32_t, 128> a, b, h;
    };

    static ESIMD_INLINE part_t load_part(const ctx & c, int sb, int part) {
        part_t p;
        p.a = xmx_load_t<8, SR>(c.qs, c.pq, c.h, sb * 32 + part * 16, c.row0);
        p.b = xmx_load_t<8, SR>(c.qs, c.pq, c.h, sb * 32 + part * 16 + 8, c.row0);
        if constexpr (QH) {
            p.h = xmx_load_t<8, SR>(c.qh, c.ph, c.h, sb * 8, c.row0);
        }
        return p;
    }

    static ESIMD_INLINE simd<uint32_t, 128> unpack(const part_t & p, int g, int part) {
        simd<uint32_t, 128> qs = (g < 2) ? p.a : p.b;
        simd<uint32_t, 128> x  = (qs >> (4 * (g & 1))) & 0x0F0F0F0Fu;
        if constexpr (QH) {
            x |= ((p.h >> (4 * part + g)) & 0x01010101u) << 4;
        }
        return x;
    }
};

// Q2_K reorder layout: [qs: nb*64] [scales: nb*16] [dm: nb*4]
// chunk s of 32: 2 bits from qs byte 32*(s/4)+l at shift 2*(s%4), each 16 wide half has a 4 bit scale and min
// value is d * (code & 15) * q - dmin * (code >> 4)
template <bool SR> struct xmx_traits_q2_k : xmx_traits_base<SR, xmx_scale_kind::min16, QK_K / 16> {
    using short_row_traits = xmx_traits_q2_k<true>;

    struct ctx : xmx_ctx_base {
        const uint8_t *    qs;
        const uint8_t *    sc;
        const sycl::half * dm;
        unsigned           pq, ps;
    };

    static ESIMD_INLINE ctx make(const void * vx, int nrows, int bpr, int row0) {
        const size_t nb = (size_t) nrows * bpr;
        ctx          c;
        xmx_ctx_init(c, nrows, bpr, row0);
        c.qs = (const uint8_t *) vx;
        c.sc = c.qs + nb * (QK_K / 4);
        c.dm = (const sycl::half *) (c.sc + nb * (QK_K / 16));
        c.pq = xmx_pitch(bpr, QK_K / 4);
        c.ps = xmx_pitch(bpr, QK_K / 16);
        return c;
    }

    struct scales_t {
        simd<uint8_t, 256> code;  // 4 bit scale (low) and min (high) of each 16 wide half
        simd<float, 16>    d, dmin;

        ESIMD_INLINE simd<uint8_t, 16> byte(int gr) { return code.template select<16, 4>(64 * (gr / 4) + (gr % 4)); }
    };

    static ESIMD_INLINE scales_t load_scales(const ctx & c, int sb, const simd<uint32_t, 16> & rows) {
        using namespace sycl::ext::intel::esimd;
        scales_t           s;
        simd<uint32_t, 16> bi  = rows * c.bpr + sb;
        simd<uint32_t, 64> raw = xmx_load_t<4, SR>(c.sc, c.ps, c.h, sb * 4, c.row0);
        s.code                 = raw.template bit_cast_view<uint8_t>();
        xmx_load_dm(c.dm, bi, s.d, s.dmin);
        return s;
    }

    struct part_t {
        simd<uint32_t, 128> q;
    };

    static ESIMD_INLINE part_t load_part(const ctx & c, int sb, int part) {
        part_t p;
        p.q = xmx_load_t<8, SR>(c.qs, c.pq, c.h, sb * 16 + part * 8, c.row0);
        return p;
    }

    static ESIMD_INLINE simd<uint32_t, 128> unpack(const part_t & p, int g, int /*part*/) {
        return (p.q >> (2 * g)) & 0x03030303u;
    }
};

// Q8_0 reorder layout: [qs: nrows*K bytes] [d: one half per 32 weights]
// the int8 weights feed DPAS as they are, a group is one block with one scale
template <bool SR> struct xmx_traits_q8_0 : xmx_traits_base<SR, xmx_scale_kind::plain, QK_K> {
    using short_row_traits = xmx_traits_q8_0<true>;
    static constexpr bool combine_activation_loads = true;

    struct ctx : xmx_ctx_base {
        const uint8_t *    qs;
        const sycl::half * dh;
        unsigned           pq;
    };

    static ESIMD_INLINE ctx make(const void * vx, int nrows, int bpr, int row0) {
        ctx c;
        xmx_ctx_init(c, nrows, bpr, row0);
        c.qs = (const uint8_t *) vx;
        c.dh = (const sycl::half *) (c.qs + (size_t) nrows * bpr * QK_K);
        c.pq = xmx_pitch(bpr, QK_K);
        return c;
    }

    struct scales_t {
        xmx_row_scales<8> d;  // scale of each of the 8 blocks of a super-block
    };

    static ESIMD_INLINE scales_t load_scales(const ctx & c, int sb, const simd<uint32_t, 16> & rows) {
        using namespace sycl::ext::intel::esimd;
        constexpr int scales_per_block = QK_K / QK8_0;
        const auto offsets = (convert<uint64_t>(rows) * c.bpr + sb) * (scales_per_block * sizeof(sycl::half));
        auto packed = gather<uint32_t, 16 * (scales_per_block / 2), scales_per_block / 2>(
            reinterpret_cast<const uint32_t *>(c.dh), offsets);
        simd<sycl::half, 16 * scales_per_block> scales = packed.template bit_cast_view<sycl::half>();
        scales_t s;
        // Each packed dword contains two scales. The gather groups each dword across rows.
#pragma unroll
        for (int j = 0; j < scales_per_block; ++j) {
            s.d.v[j] = convert<float>(simd<sycl::half, 16>(scales.template select<16, 2>((j / 2) * 32 + j % 2)));
        }
        return s;
    }

    struct part_t {
        simd<uint32_t, 128> q[4];
    };

    static ESIMD_INLINE part_t load_part(const ctx & c, int sb, int part) {
        part_t p;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            p.q[g] = xmx_load_t<8, SR>(c.qs, c.pq, c.h, sb * 64 + 8 * (4 * part + g), c.row0);
        }
        return p;
    }

    static ESIMD_INLINE simd<uint32_t, 128> unpack(const part_t & p, int g, int /*part*/) { return p.q[g]; }
};

template <bool SR> using xmx_traits_q4_k = xmx_traits_q45_k<false, SR>;
template <bool SR> using xmx_traits_q5_k = xmx_traits_q45_k<true, SR>;

#ifdef GGML_SYCL_MMVQ_HAS_XMX
// Xe-LPG+ uses the execution-size-8 s8 x s8 -> int32 DPAS form. B is VNNI
// encoded as K x N (four adjacent K elements per dword); the logical tile has
// N=16 weight rows, so each term uses one operation for each 8-row half of B.
// The A operand remains the horizontal M x K activation matrix. This preserves
// the quantization epilogues and output layout while avoiding the
// execution-size-16 form used by the original Xe2-oriented code.
template <int HALF_INDEX>
ESIMD_INLINE void xmx_term_half(simd<int, XMX_ROWS> & C, const simd<int8_t, 512> & B, const simd<int8_t, 32> & A) {
    using namespace sycl::ext::intel::esimd;
    namespace xmx = sycl::ext::intel::esimd::xmx;

    constexpr int HALF = XMX_ROWS / 2;
    static_assert(XMX_ROWS == 16, "Xe-LPG+ tile split assumes 16 output rows");
    static_assert(HALF_INDEX >= 0 && HALF_INDEX < 2, "Invalid DPAS half");
    simd<int8_t, 512> Bcopy = B;
    // B is packed as eight K-dwords, each containing all 16 output rows.
    // The two execution-size-8 operations therefore gather a row half from
    // every K-dword; a contiguous 256-byte slice would mix K and N.
    simd<int8_t, HALF * 32> Bh;
#pragma unroll
    for (int kd = 0; kd < 8; ++kd) {
        Bh.template select<HALF * 4, 1>(kd * HALF * 4) =
            Bcopy.template select<HALF * 4, 1>(kd * XMX_ROWS * 4 + HALF_INDEX * HALF * 4);
    }
    simd<int, HALF> Ch = xmx::dpas<8, 1, int>(Bh, A);
    C.template select<HALF, 1>(HALF_INDEX * HALF) = Ch;
}

ESIMD_INLINE simd<float, XMX_ROWS> xmx_term(const simd<int8_t, 512> & B,
                                           const simd<int8_t, 32> & A,
                                           const simd<float, XMX_ROWS> & s) {
    simd<int, XMX_ROWS> C;
    xmx_term_half<0>(C, B, A);
    xmx_term_half<1>(C, B, A);
    return sycl::ext::intel::esimd::convert<float>(C) * s;
}

// one group of 32 of one weight set, accumulated into Cf for all the columns of the tile
template <typename T, int NC, typename A2T, typename D8T>
ESIMD_INLINE void xmx_group(simd<float, NC * XMX_ROWS> & Cf,
                            const typename T::part_t &   pd,
                            typename T::scales_t &       sc,
                            A2T &                        A2,
                            D8T &                        d8f,
                            simd<int8_t, 512> &          Bl,
                            simd<int8_t, 512> &          Bh,
                            const int                    g,
                            const int                    part) {
    using namespace sycl::ext::intel::esimd;

    constexpr int  NT    = XMX_ROWS;
    constexpr int  RC    = NC < 8 ? NC : 8;
    constexpr int  NB    = NC / RC;
    constexpr bool split = !T::has_min && !T::plain && !T::min16;  // two scales per group of 32
    constexpr bool pair  = (RC * 32) % 64 == 0;                    // a block of RC rows fills whole registers

    const int gi = 4 * part + g;

    simd<uint32_t, 128> x = T::unpack(pd, g, part);
    simd<int8_t, 512>   B = x.template bit_cast_view<int8_t>();
    if constexpr (split) {
        Bl.template select<256, 1>(0)   = B.template select<256, 1>(0);
        Bh.template select<256, 1>(256) = B.template select<256, 1>(256);
    }

    simd<float, NT> rd;
    simd<float, NT> rm;
    simd<float, NT> rs0;
    simd<float, NT> rs1;

    // Q2_K: scale and min of each 16 wide half are 4 bit codes, the weight is d * scl * q - dmin * mc. scl * q
    // is at most 45, so a dword multiply keeps the bytes apart and the scaled weights go to DPAS as int8. The
    // min term applies each half's row-specific code to a shared activation sum.
    simd<int8_t, 512> Bs;
    simd<int, NT> minimum0, minimum1;
    simd<uint32_t, 64> half_masks;
    if constexpr (T::min16) {
        // VNNI packs one K-dword per lane. Lane 0 sums the first 16
        // activations, and lane 1 sums the second 16.
        half_masks = 0;
#pragma unroll
        for (int kd = 0; kd < 8; ++kd) {
            half_masks[kd * 8 + (kd >= 4)] = 0x01010101u;
        }
        const simd<uint8_t, NT> c0  = sc.byte(2 * gi);
        const simd<uint8_t, NT> c1  = sc.byte(2 * gi + 1);
        simd<uint32_t, NT>      sl0 = convert<uint32_t>(simd<uint8_t, NT>(c0 & (uint8_t) 15));
        simd<uint32_t, NT>      sl1 = convert<uint32_t>(simd<uint8_t, NT>(c1 & (uint8_t) 15));
        minimum0 = convert<int>(simd<uint8_t, NT>(c0 >> (uint8_t) 4));
        minimum1 = convert<int>(simd<uint8_t, NT>(c1 >> (uint8_t) 4));
        simd<uint32_t, 128>     ws;
        ws.template select<64, 1>(0)  = x.template select<64, 1>(0) * sl0.template replicate<4>();
        ws.template select<64, 1>(64) = x.template select<64, 1>(64) * sl1.template replicate<4>();
        Bs                            = ws.template bit_cast_view<int8_t>();
    }
    if constexpr (T::plain) {
        rd = sc.d.get(gi);
    } else if constexpr (T::has_min) {
        rd = sc.d.get(gi);
        rm = sc.m.get(gi);
    } else if constexpr (T::min16) {
        // scales are applied through Bs and Bm
    } else {
        rs0 = sc.get(2 * gi);
        rs1 = sc.get(2 * gi + 1);
    }

#pragma unroll
    for (int b = 0; b < NB; ++b) {
        // Accumulate one column at a time so the DPAS results and scaled
        // terms need registers for only 16 rows at once.
#pragma unroll
        for (int m = 0; m < RC; ++m) {
            const int a_offset = (pair ? (g % 2) * RC * 8 : 0) + m * 8;
            const simd<int8_t, 32> A = A2[b].template select<8, 1>(a_offset).template bit_cast_view<int8_t>();
            auto C = Cf.template select<NT, 1>((b * RC + m) * NT);
            simd<float, NT> z;
            if constexpr (T::plain) {
                z = xmx_term(B, A, rd);
            } else if constexpr (T::has_min) {
                // DPAS against ones gives the exact integer sum of the activations.
                namespace xmx = sycl::ext::intel::esimd::xmx;
                const auto sums = xmx::dpas<8, 1, int>(simd<int8_t, 256>(1), A);
                // The sum is identical in every lane. Use one scalar for all output rows.
                const float sum = float(sums[0]);
                z = xmx_term(B, A, rd) - rm * sum;
            } else if constexpr (T::min16) {
                namespace xmx = sycl::ext::intel::esimd::xmx;
                const simd<int8_t, 256> mask_bytes = half_masks.template bit_cast_view<int8_t>();
                const auto sums = xmx::dpas<8, 1, int>(mask_bytes, A);
                const int sum0 = sums[0];
                const int sum1 = sums[1];
                const auto minimum = minimum0 * sum0 + minimum1 * sum1;
                z = xmx_term(Bs, A, sc.d) - convert<float>(minimum) * sc.dmin;
            } else {
                z = xmx_term(Bl, A, rs0) + xmx_term(Bh, A, rs1);
            }
            const float activation_scale = d8f[b][m * 8 + gi];
            C += z * activation_scale;
        }
    }
}

// activation of the GLU applied to the gate, uniform across the launch
template <int N> ESIMD_INLINE simd<float, N> xmx_glu_act(const simd<float, N> & gate, const int glu_op) {
    using namespace sycl::ext::intel::esimd;
    if (glu_op == GGML_GLU_OP_SWIGLU) {
        return gate / (1.0f + exp(-gate));
    }
    // GEGLU, the tanh is written with exp
    simd<float, N> a = 0.79788456f * gate * (1.0f + 0.044715f * gate * gate);
    simd<float, N> t = 1.0f - 2.0f / (exp(2.0f * a) + 1.0f);
    return 0.5f * gate * (1.0f + t);
}

// Load the (d, s) pairs and quantized bytes for the horizontal DPAS A operand.
// Groups with two scales (16 wide) mask the weights, so zeros in B drop the other half of K.
// KS is the most threads that split K for one tile, the launch may use fewer.
// FUSED computes up * act(gate) from the weights vx (up) and vg (gate).
template <typename T, int NC, int KS, bool FUSED, bool NATIVE_2D = XMX_ENABLE_TRANSPOSED_2D_LOADS,
          bool BOUNDED_COLUMNS = (NC > 8)>
ESIMD_INLINE void xmx_mul_mat(const void * vx,
                              const void * vg,
                              const char * vy,
                              float *      dst,
                              const int    ncols,
                              const int    nrows,
                              const int    ncv,
                              const int    stride_y_bytes,
                              const int    ldd,
                              const int    glu_op,
                              const int    tile,
                              const int    lid,
                              const int    ks) {
    using namespace sycl::ext::intel::esimd;

    constexpr int  NT    = XMX_ROWS;
    constexpr int  RC    = NC < 8 ? NC : 8;
    constexpr int  NB    = NC / RC;
    constexpr int  N_ACC = FUSED ? 2 : 1;
    constexpr bool pair  = (RC * 32) % 64 == 0;  // a block of RC rows fills whole registers
    // A last tile can be narrower than NC, including an 8-column GLU tile.
    // Native 2D loads use yh as their height bound. Software loads clamp
    // to the last valid column, and stores are bounded in either case.
    constexpr bool clamp_columns = !NATIVE_2D && BOUNDED_COLUMNS;

    if constexpr (KS > 1) {
        slm_init<N_ACC * KS * NC * NT * sizeof(float)>();
    }

    const int  bpr  = ncols / QK_K;
    const int  row0 = tile * NT;

    simd<uint32_t, NT> rows;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        rows[n] = row0 + n < nrows ? row0 + n : nrows - 1;
    }

    const uint32_t * yq  = (const uint32_t *) vy;
    const uint32_t * yd  = (const uint32_t *) (vy + ncols);
    const unsigned   yqw = ncols - 1;
    const unsigned   ydw = xmx_activation_scale_bytes(ncols) - 1;
    const unsigned   yh  = ncv - 1;
    const unsigned   yp  = stride_y_bytes - 1;

    simd<float, NC * NT> Cf  = 0.0f;
    simd<float, NC * NT> Cfg = 0.0f;  // gate, only used when fused
    // Run one projection at a time so only one set of weight vectors
    // and scales is live. Both results stay in this kernel through GLU.
    auto accumulate = [&](auto & acc, const void * weights) [[intel::sycl_explicit_simd]] {
        const auto c = T::make(weights, nrows, bpr, row0);
        simd<int8_t, 512>    Bl  = int8_t(0);
        simd<int8_t, 512>    Bh  = int8_t(0);

        for (int sb = lid; sb < bpr; sb += ks) {
            auto sc = T::load_scales(c, sb, rows);

            // d of the 8 groups of the super-block, [column * 8 + group]
            simd<float, 8 * RC> d8f[NB];
#pragma unroll
            for (int b = 0; b < NB; ++b) {
                simd<uint32_t, 8 * RC> ds;
                if constexpr (NATIVE_2D) {
                    ds = load_2d<uint32_t, 8, RC, 1, false, false>(yd, ydw, yh, yp, sb * 8, b * RC);
                } else {
                    ds = xmx_load_rows<8, RC, clamp_columns>(yd, stride_y_bytes / sizeof(uint32_t), sb * 8, b * RC, ncv);
                }
                simd<sycl::half, 16 * RC> dh = ds.template bit_cast_view<sycl::half>();
                d8f[b]                       = convert<float>(simd<sycl::half, 8 * RC>(dh.template select<8 * RC, 2>(0)));
            }

#pragma unroll
            for (int part = 0; part < 2; ++part) {
                const auto pd = T::load_part(c, sb, part);

                simd<uint32_t, 16 * RC> A2[NB];  // quants of two groups

#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const int gi = 4 * part + g;
                    if constexpr (pair) {
                        if (g % 2 == 0) {
#pragma unroll
                            for (int b = 0; b < NB; ++b) {
                                if constexpr (NATIVE_2D) {
                                    A2[b] = load_2d<uint32_t, 8, RC, 2, false, false>(yq, yqw, yh, yp,
                                                                                        sb * 64 + gi * 8, b * RC);
                                } else {
                                    A2[b] = xmx_load_rows_pair<8, RC, clamp_columns, T::combine_activation_loads>(
                                        yq, stride_y_bytes / sizeof(uint32_t), sb * 64 + gi * 8, b * RC, ncv);
                                }
                            }
                        }
                    } else {
#pragma unroll
                        for (int b = 0; b < NB; ++b) {
                            if constexpr (NATIVE_2D) {
                                A2[b].template select<RC * 8, 1>(0) =
                                    load_2d<uint32_t, 8, RC, 1, false, false>(yq, yqw, yh, yp,
                                                                               sb * 64 + gi * 8, b * RC);
                            } else {
                                A2[b].template select<RC * 8, 1>(0) =
                                    xmx_load_rows<8, RC, clamp_columns>(
                                        yq, stride_y_bytes / sizeof(uint32_t), sb * 64 + gi * 8, b * RC, ncv);
                            }
                        }
                    }

                    xmx_group<T, NC>(acc, pd, sc, A2, d8f, Bl, Bh, g, part);
                }
            }
        }
    };
    accumulate(Cf, vx);
    if constexpr (FUSED) {
        accumulate(Cfg, vg);
    }

    if constexpr (KS > 1) {
        slm_block_store<float, NC * NT>(lid * NC * NT * sizeof(float), Cf);
        if constexpr (FUSED) {
            slm_block_store<float, NC * NT>((KS + lid) * NC * NT * sizeof(float), Cfg);
        }
        barrier();
        if (lid != 0) {
            return;
        }
        Cf  = 0.0f;
        Cfg = 0.0f;
#pragma unroll 4
        for (int j = 0; j < ks; ++j) {
            Cf += slm_block_load<float, NC * NT>(j * NC * NT * sizeof(float));
            if constexpr (FUSED) {
                Cfg += slm_block_load<float, NC * NT>((KS + j) * NC * NT * sizeof(float));
            }
        }
    }

    if constexpr (FUSED) {
        Cf *= xmx_glu_act<NC * NT>(Cfg, glu_op);
    }

    if (row0 + NT <= nrows && ldd % 4 == 0 && (uintptr_t) dst % 16 == 0) {
#pragma unroll
        for (int m = 0; m < NC; ++m) {
            if (!BOUNDED_COLUMNS || m < ncv) {
                block_store<float, NT>(dst + (size_t) m * ldd + row0, Cf.template select<NT, 1>(m * NT));
            }
        }
    } else {
        // tail tile or unaligned output
        simd<uint32_t, NT> lane(0, 1);
        simd_mask<NT>      in_rows = (lane + row0) < (uint32_t) nrows;
#pragma unroll
        for (int m = 0; m < NC; ++m) {
            if (!BOUNDED_COLUMNS || m < ncv) {
                scatter<float, NT>(dst + (size_t) m * ldd + row0, lane * (uint32_t) sizeof(float),
                                   Cf.template select<NT, 1>(m * NT), in_rows);
            }
        }
    }
}

template <typename T, bool FUSED>
constexpr bool xmx_native_2d_geometry(uintptr_t vx, uintptr_t vg, uintptr_t vy, int ncols, int nrows, int ncv,
                                      int stride_y_bytes) {
    constexpr int max_extent = 1 << 24;
    const int scale_width = xmx_activation_scale_bytes(ncols);
    if (vx == 0 || vy == 0 || (FUSED && vg == 0) ||
        ncols <= 0 || ncols % QK_K != 0 || ncols > max_extent || scale_width < 64 ||
        nrows <= 0 || nrows > max_extent || ncv <= 0 || ncv > max_extent ||
        stride_y_bytes < int64_t(ncols) + scale_width || stride_y_bytes > max_extent ||
        stride_y_bytes % 16 != 0 || vy % 64 != 0) {
        return false;
    }

    // The activation scale plane starts at vy + ncols. The packed weight
    // layouts place every native 2D plane at a multiple of 64 bytes from vx.
    const int min_weight_width = ncols / QK_K * T::min_2d_bytes_per_block;
    return T::short_rows || (min_weight_width >= 64 && vx % 64 == 0 && (!FUSED || vg % 64 == 0));
}

// KS_MAX bounds the threads that split K for one tile, the launch uses as few as keep every thread to the same
// number of super-blocks as the busiest one
template <typename T, int NC, int KS_MAX, bool FUSED, bool NATIVE_2D, bool SPLIT_COLUMNS = false>
static void xmx_mul_mat_submit(const void *    vx,
                                  const void *    vg,
                                  const char *    vy,
                                  float *         dst,
                                  const int       ncols,
                                  const int       nrows,
                                  const int       ncv,
                                  const int       stride_y_bytes,
                                  const int       ldd,
                                  const int       glu_op,
                                  dpct::queue_ptr stream) {
    const size_t n_tiles = (nrows + XMX_ROWS - 1) / XMX_ROWS;
    const int    bpr     = ncols / QK_K;
    const int    rounds  = (bpr + KS_MAX - 1) / KS_MAX;
    const int    ks      = (bpr + rounds - 1) / rounds;
    constexpr auto properties = sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::grf_size_automatic };
    stream->submit([&](sycl::handler & cgh) {
        if constexpr (SPLIT_COLUMNS) {
            // Submit all column tiles together, including a bounded last tile.
            const size_t column_tiles = (ncv + NC - 1) / NC;
            cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_tiles * ks, column_tiles), sycl::range<2>(ks, 1)),
                             properties, [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
                                 const int column = (int) it.get_group(1) * NC;
                                 const int valid_columns = std::min(NC, ncv - column);
                                 xmx_mul_mat<T, NC, KS_MAX, FUSED, NATIVE_2D, true>(
                                     vx, vg, vy + (size_t) column * stride_y_bytes, dst + (size_t) column * ldd,
                                     ncols, nrows, valid_columns, stride_y_bytes, ldd, glu_op,
                                     (int) it.get_group(0), (int) it.get_local_id(0), ks);
                             });
        } else {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(n_tiles * ks), sycl::range<1>(ks)),
                             properties, [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                                 xmx_mul_mat<T, NC, KS_MAX, FUSED, NATIVE_2D>(
                                     vx, vg, vy, dst, ncols, nrows, ncv, stride_y_bytes, ldd, glu_op,
                                     (int) it.get_group(0), (int) it.get_local_id(0), ks);
                             });
        }
    });
}

// Submit exact-width tails of one to eight columns.
template <typename T, bool FUSED, bool SINGLE_COLUMN_TAIL>
static void xmx_mul_mat_submit_partial_columns(const void *    vx,
                                               const void *    vg,
                                               const char *    vy,
                                               float *         dst,
                                               const int       ncols,
                                               const int       nrows,
                                               const int       ncv,
                                               const int       stride_y_bytes,
                                               const int       ldd,
                                               const int       glu_op,
                                               dpct::queue_ptr stream) {
    const size_t row_tiles = (nrows + XMX_ROWS - 1) / XMX_ROWS;
    const size_t column_tiles = xmx_column_tile_count(ncv);
    constexpr auto properties = sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::grf_size_automatic };
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(row_tiles, column_tiles), sycl::range<2>(1, 1)),
                         properties, [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
                             const int column_tile = it.get_group(1);
                             const auto tile = SINGLE_COLUMN_TAIL
                                 ? xmx_column_tile{column_tile * XMX_COLUMN_TILE,
                                                   column_tile == ncv / XMX_COLUMN_TILE ? 1 : XMX_COLUMN_TILE}
                                 : xmx_column_tile_at(ncv, column_tile);
                             const char * y = vy + size_t(tile.first) * stride_y_bytes;
                             float * d = dst + size_t(tile.first) * ldd;
                             const int row_tile = it.get_group(0);
                             auto multiply = [&](auto width) [[intel::sycl_explicit_simd]] {
                                 constexpr int columns = decltype(width)::value;
                                 xmx_mul_mat<T, columns, 1, FUSED, false>(
                                     vx, vg, y, d, ncols, nrows, columns, stride_y_bytes, ldd, glu_op, row_tile, 0, 1);
                             };
                             if constexpr (SINGLE_COLUMN_TAIL) {
                                 if (tile.width == 1) {
                                     multiply(std::integral_constant<int, 1>{});
                                 } else {
                                     multiply(std::integral_constant<int, XMX_COLUMN_TILE>{});
                                 }
                             } else {
                                 switch (tile.width) {
                                     case XMX_COLUMN_TILE: multiply(std::integral_constant<int, XMX_COLUMN_TILE>{}); break;
                                     case 8: multiply(std::integral_constant<int, 8>{}); break;
                                     case 7: multiply(std::integral_constant<int, 7>{}); break;
                                     case 6: multiply(std::integral_constant<int, 6>{}); break;
                                     case 5: multiply(std::integral_constant<int, 5>{}); break;
                                     case 4: multiply(std::integral_constant<int, 4>{}); break;
                                     case 3: multiply(std::integral_constant<int, 3>{}); break;
                                     case 2: multiply(std::integral_constant<int, 2>{}); break;
                                 }
                             }
                         });
    });
}

template <typename T, int NC, int KS_MAX, bool FUSED>
static void xmx_mul_mat_launch_ks(const void *    vx,
                                  const void *    vg,
                                  const char *    vy,
                                  float *         dst,
                                  const int       ncols,
                                  const int       nrows,
                                  const int       ncv,
                                  const int       stride_y_bytes,
                                  const int       ldd,
                                  const int       glu_op,
                                  dpct::queue_ptr stream) {
    GGML_ASSERT(ncols > 0 && ncols % QK_K == 0);
    GGML_ASSERT(nrows > 0 && ncv > 0 && ncv <= NC && (NC > 8 || ncv == NC));
    GGML_ASSERT(vx && vy && dst && stream && (!FUSED || vg));
    GGML_ASSERT(stride_y_bytes >= int64_t(ncols) + xmx_activation_scale_bytes(ncols));
    GGML_ASSERT(stride_y_bytes % 4 == 0 && (uintptr_t) vy % 4 == 0 &&
                (uintptr_t) vx % 4 == 0 && (!FUSED || (uintptr_t) vg % 4 == 0));
    constexpr int bounded_glu_width = [] {
        if constexpr (FUSED && KS_MAX == 1 && !XMX_ENABLE_TRANSPOSED_2D_LOADS) {
            // Share the bounded four-column kernel across Q8_0 widths 2-4 with row loads.
            if constexpr (!XMX_ENABLE_GATHER_TRANSPOSE &&
                          std::is_same_v<T, xmx_traits_q8_0<false>> && NC >= 2 && NC <= 4) {
                return 4;
            }
            if constexpr (std::is_same_v<T, xmx_traits_q4_k<false>> && NC == 1) {
                return XMX_FUSED_COLUMN_TILE;
            }
        }
        return 0;
    }();
    constexpr int default_tile_columns = std::min(NC, FUSED ? XMX_FUSED_COLUMN_TILE : XMX_COLUMN_TILE);
    constexpr int tile_columns = bounded_glu_width ? bounded_glu_width : default_tile_columns;
    constexpr bool split_columns = NC > tile_columns || bounded_glu_width != 0;
    if constexpr (XMX_ENABLE_TRANSPOSED_2D_LOADS) {
        if (xmx_native_2d_geometry<T, FUSED>((uintptr_t) vx, (uintptr_t) vg, (uintptr_t) vy, ncols, nrows, ncv,
                                             stride_y_bytes)) {
            using sycl::ext::intel::esimd::info::device::has_2d_block_io_support;
            if (stream->get_device().get_info<has_2d_block_io_support>()) {
                xmx_mul_mat_submit<T, tile_columns, KS_MAX, FUSED, true, split_columns>(
                    vx, vg, vy, dst, ncols, nrows, ncv, stride_y_bytes, ldd, glu_op, stream);
                return;
            }
        }
    }
    using SoftwareTraits = std::conditional_t<XMX_ENABLE_TRANSPOSED_2D_LOADS, typename T::short_row_traits, T>;
    if constexpr (tile_columns == XMX_COLUMN_TILE && KS_MAX == 1) {
        const int tail_width = xmx_column_tile_width(ncv % XMX_COLUMN_TILE);
        if (tail_width == 1) {
            xmx_mul_mat_submit_partial_columns<SoftwareTraits, FUSED, true>(
                vx, vg, vy, dst, ncols, nrows, ncv, stride_y_bytes, ldd, glu_op, stream);
            return;
        }
        if (tail_width > 1 && tail_width < XMX_COLUMN_TILE) {
            xmx_mul_mat_submit_partial_columns<SoftwareTraits, FUSED, false>(
                vx, vg, vy, dst, ncols, nrows, ncv, stride_y_bytes, ldd, glu_op, stream);
            return;
        }
    }
    xmx_mul_mat_submit<SoftwareTraits, tile_columns, KS_MAX, FUSED, false, split_columns>(
        vx, vg, vy, dst, ncols, nrows, ncv, stride_y_bytes, ldd, glu_op, stream);
}

template <typename T, int NC, bool FUSED>
static void xmx_mul_mat_launch(const void *    vx,
                               const void *    vg,
                               const char *    vy,
                               float *         dst,
                               const int       ncols,
                               const int       nrows,
                               const int       ncv,
                               const int       stride_y_bytes,
                               const int       ldd,
                               const int       glu_op,
                               dpct::queue_ptr stream) {
    // the fused kernel keeps two sets of accumulators in shared memory, half the threads for the same size
    constexpr int ks_max = XMX_USE_SLM ? xmx_n_split<NC>() / (FUSED ? 2 : 1) : 1;
    xmx_mul_mat_launch_ks<T, NC, ks_max, FUSED>(vx, vg, vy, dst, ncols, nrows, ncv, stride_y_bytes, ldd, glu_op,
                                                stream);
}

// FUSED: vx and vg are the up and gate weights, dst gets up * act(gate) with glu_op a ggml_glu_op
template <typename T, bool FUSED = false>
static void xmx_mul_mat_ncols(const void *    vx,
                              const void *    vg,
                              const char *    vy,
                              float *         dst,
                              const int       ncols,
                              const int       nrows,
                              const int       ncols_dst,
                              const int       stride_y_bytes,
                              const int       ldd,
                              const int       glu_op,
                              dpct::queue_ptr stream) {
    constexpr int chunk_max = FUSED ? XMX_FUSED_CHUNK : XMX_CHUNK;
    for (int c0 = 0; c0 < ncols_dst; c0 += chunk_max) {
        const int    n  = std::min(chunk_max, ncols_dst - c0);
        const char * y  = vy + (size_t) c0 * stride_y_bytes;
        float *      d  = dst + (size_t) c0 * ldd;
        const int    nc = n > 32 ? 64 : (n > 16 ? 32 : (n > 8 ? 16 : n));
        switch (nc) {
#define XMX_CASE(N)                                                                                              \
    case N:                                                                                                      \
        if constexpr (N <= chunk_max) {                                                                           \
            xmx_mul_mat_launch<T, N, FUSED>(vx, vg, y, d, ncols, nrows, n, stride_y_bytes, ldd, glu_op, stream); \
        } else {                                                                                                 \
            GGML_ABORT("unsupported ncols_dst=%d for XMX MMV", n);                                                 \
        }                                                                                                        \
        break
            XMX_CASE(1);
            XMX_CASE(2);
            XMX_CASE(3);
            XMX_CASE(4);
            XMX_CASE(5);
            XMX_CASE(6);
            XMX_CASE(7);
            XMX_CASE(8);
            XMX_CASE(16);
            XMX_CASE(32);
            XMX_CASE(64);
#undef XMX_CASE
            default:
                GGML_ABORT("unsupported ncols_dst=%d for XMX MMV", n);
        }
    }
}

#endif // GGML_SYCL_MMVQ_HAS_XMX

} // namespace ggml_sycl_esimd_arl

#endif // GGML_SYCL_MMVQ_HAS_XMX
