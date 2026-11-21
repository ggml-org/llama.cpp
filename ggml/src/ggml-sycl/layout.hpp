#ifndef GGML_SYCL_LAYOUT_HPP
#define GGML_SYCL_LAYOUT_HPP

#include "ggml-common.h"

#include <cstddef>
#include <cstdint>

// The fields of a weight format's block in the order of its reorder (SoA) layout, where field f is a stream of
// seg[f].size bytes a block after the streams of the fields before it; seg[f].aos is its offset in the canonical
// block. A field with `join` set shares the stream of the field before it instead, interleaved a block at a
// time (iq3_s keeps d and scales in one stream). The joint_matrix and the ESIMD A stages read the fields of
// either layout through it, so one table per format keeps them and the reorder in step.
struct ggml_sycl_seg {
    int  aos;
    int  size;
    bool join = false;
};

template <typename block_q_t> struct ggml_sycl_layout;

// q8_0, 34 bytes a block
//   canonical block:  [d 2][qs 32]
//   reorder streams:  [qs][d], each over every block of the slice
template <> struct ggml_sycl_layout<block_q8_0> {
    enum { QS, D, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 2, 32 }, // qs: 32 int8 values
        { 0, 2 },  // d: f16 scale
    };
};

// q4_0, 18 bytes a block
//   canonical block:  [d 2][qs 16]
//   reorder streams:  [qs][d], each over every block of the slice
template <> struct ggml_sycl_layout<block_q4_0> {
    enum { QS, D, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 2, 16 }, // qs: 32 4-bit codes, two a byte
        { 0, 2 },  // d: f16 scale
    };
};

// q4_1, 20 bytes a block
//   canonical block:  [dm 4][qs 16]
//   reorder streams:  [qs][dm], each over every block of the slice
template <> struct ggml_sycl_layout<block_q4_1> {
    enum { QS, DM, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 4, 16 }, // qs: 32 4-bit codes, two a byte
        { 0, 4 },  // dm: f16 scale and f16 min
    };
};

// q5_0, 22 bytes a block
//   canonical block:  [d 2][qh 4][qs 16]
//   reorder streams:  [qs][qh][d], each over every block of the slice
template <> struct ggml_sycl_layout<block_q5_0> {
    enum { QS, QH, D, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 6, 16 }, // qs: low 4 bits of the 32 codes
        { 2, 4 },  // qh: fifth bit of each code
        { 0, 2 },  // d: f16 scale
    };
};

// q5_1, 24 bytes a block
//   canonical block:  [dm 4][qh 4][qs 16]
//   reorder streams:  [qs][qh][dm], each over every block of the slice
template <> struct ggml_sycl_layout<block_q5_1> {
    enum { QS, QH, DM, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 8, 16 }, // qs: low 4 bits of the 32 codes
        { 4, 4 },  // qh: fifth bit of each code
        { 0, 4 },  // dm: f16 scale and f16 min
    };
};

// q1_0, 18 bytes a block
//   canonical block:  [d 2][qs 16]
//   reorder streams:  [qs][d], each over every block of the slice
template <> struct ggml_sycl_layout<block_q1_0> {
    enum { QS, D, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 2, 16 }, // qs: 128 sign bits
        { 0, 2 },  // d: f16 scale
    };
};

// q2_0, 18 bytes a block
//   canonical block:  [d 2][qs 16]
//   reorder streams:  [qs][d], each over every block of the slice
template <> struct ggml_sycl_layout<block_q2_0> {
    enum { QS, D, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 2, 16 }, // qs: 64 2-bit codes
        { 0, 2 },  // d: f16 scale
    };
};

// mxfp4, 17 bytes a block
//   canonical block:  [e 1][qs 16]
//   reorder streams:  [qs][e], each over every block of the slice
template <> struct ggml_sycl_layout<block_mxfp4> {
    enum { QS, E, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 1, 16 }, // qs: 32 E2M1 codes, two a byte
        { 0, 1 },  // e: E8M0 scale
    };
};

// nvfp4, 36 bytes a block
//   canonical block:  [d 4][qs 32]
//   reorder streams:  [qs][d], each over every block of the slice
template <> struct ggml_sycl_layout<block_nvfp4> {
    enum { QS, D, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 4, 32 }, // qs: 64 E2M1 codes, two a byte
        { 0, 4 },  // d: 4 UE4M3 scales, one a 16-value sub-block
    };
};

// q2_K, 84 bytes a block
//   canonical block:  [scales 16][qs 64][dm 4]
//   reorder streams:  [qs][scales][dm], each over every block of the slice
template <> struct ggml_sycl_layout<block_q2_K> {
    enum { QS, SC, DM, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 16, 64 }, // qs: 256 2-bit codes
        { 0, 16 },  // scales: 4-bit scale and min per 16 values
        { 80, 4 },  // dm: f16 super-scale and super-min
    };
};

// q3_K, 110 bytes a block
//   canonical block:  [hmask 32][qs 64][scales 12][d 2]
//   reorder streams:  [qs][hmask][scales][d], each over every block of the slice
template <> struct ggml_sycl_layout<block_q3_K> {
    enum { QS, HM, SC, D, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 32, 64 }, // qs: low 2 bits of the 256 codes
        { 0, 32 },  // hmask: third bit of each code
        { 96, 12 }, // scales: 16 6-bit scales
        { 108, 2 }, // d: f16 super-scale
    };
};

// q4_K, 144 bytes a block
//   canonical block:  [dm 4][scales 12][qs 128]
//   reorder streams:  [qs][scales][dm], each over every block of the slice
template <> struct ggml_sycl_layout<block_q4_K> {
    enum { QS, SC, DM, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 16, 128 }, // qs: 256 4-bit codes
        { 4, 12 },   // scales: 6-bit scales and mins of the 8 sub-blocks
        { 0, 4 },    // dm: f16 super-scale and super-min
    };
};

// q5_K, 176 bytes a block
//   canonical block:  [dm 4][scales 12][qh 32][qs 128]
//   reorder streams:  [qs][qh][scales][dm], each over every block of the slice
template <> struct ggml_sycl_layout<block_q5_K> {
    enum { QS, QH, SC, DM, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 48, 128 }, // qs: low 4 bits of the 256 codes
        { 16, 32 },  // qh: fifth bit of each code
        { 4, 12 },   // scales: 6-bit scales and mins of the 8 sub-blocks
        { 0, 4 },    // dm: f16 super-scale and super-min
    };
};

// q6_K, 210 bytes a block
//   canonical block:  [ql 128][qh 64][scales 16][d 2]
//   reorder streams:  [ql][qh][scales][d], each over every block of the slice
template <> struct ggml_sycl_layout<block_q6_K> {
    enum { QL, QH, SC, D, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 0, 128 },  // ql: low 4 bits of the 256 codes
        { 128, 64 }, // qh: high 2 bits of each code
        { 192, 16 }, // scales: 16 int8 scales
        { 208, 2 },  // d: f16 super-scale
    };
};

// iq4_nl, 18 bytes a block
//   canonical block:  [d 2][qs 16]
//   reorder streams:  [qs][d], each over every block of the slice
template <> struct ggml_sycl_layout<block_iq4_nl> {
    enum { QS, D, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 2, 16 }, // qs: 32 4-bit codebook indices
        { 0, 2 },  // d: f16 scale
    };
};

// iq4_xs, 136 bytes a block
//   canonical block:  [d 2][scales_h 2][scales_l 4][qs 128]
//   reorder streams:  [qs][d][scales_h][scales_l], each over every block of the slice
template <> struct ggml_sycl_layout<block_iq4_xs> {
    enum { QS, D, SH, SL, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 8, 128 }, // qs: 256 4-bit codebook indices
        { 0, 2 },   // d: f16 super-scale
        { 2, 2 },   // scales_h: high 2 bits of the 8 sub-block scales
        { 4, 4 },   // scales_l: low 4 bits of the 8 sub-block scales
    };
};

// iq3_s, 110 bytes a block
//   canonical block:  [d 2][qs 64][qh 8][signs 32][scales 4]
//   reorder streams:  [qs][qh][signs][d scales], each over every block of the slice
template <> struct ggml_sycl_layout<block_iq3_s> {
    enum { QS, QH, SG, D, SC, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 2, 64 },        // qs: low 8 bits of the 64 grid indices
        { 66, 8 },        // qh: ninth bit of each grid index
        { 74, 32 },       // signs: one sign bit a value
        { 0, 2 },         // d: f16 super-scale
        { 106, 4, true }, // scales: 4-bit scale per 32 values; joins d's stream
    };
};

// iq3_xxs, 98 bytes a block
//   canonical block:  [d 2][qs 96]
//   reorder streams:  [qs][d], each over every block of the slice
template <> struct ggml_sycl_layout<block_iq3_xxs> {
    enum { QS, D, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 2, 96 }, // qs: 64 grid indices, then 8 sign/scale dwords
        { 0, 2 },  // d: f16 super-scale
    };
};

// iq2_xxs, 66 bytes a block
//   canonical block:  [d 2][qs 64]
//   reorder streams:  [qs][d], each over every block of the slice
template <> struct ggml_sycl_layout<block_iq2_xxs> {
    enum { QS, D, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 2, 64 }, // qs: per sub-block 4 grid indices and a sign/scale dword
        { 0, 2 },  // d: f16 super-scale
    };
};

// iq2_xs, 74 bytes a block
//   canonical block:  [d 2][qs 64][scales 8]
//   reorder streams:  [qs][scales][d], each over every block of the slice
template <> struct ggml_sycl_layout<block_iq2_xs> {
    enum { QS, SC, D, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 2, 64 }, // qs: 32 16-bit words, 9-bit grid index and 7 sign bits
        { 66, 8 }, // scales: two 4-bit scales per sub-block
        { 0, 2 },  // d: f16 super-scale
    };
};

// iq2_s, 82 bytes a block
//   canonical block:  [d 2][qs 64][qh 8][scales 8]
//   reorder streams:  [qs][qh][scales][d], each over every block of the slice
template <> struct ggml_sycl_layout<block_iq2_s> {
    enum { QS, QH, SC, D, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 2, 64 }, // qs: 32 grid index bytes, then 32 sign bytes
        { 66, 8 }, // qh: high 2 bits of the grid indices
        { 74, 8 }, // scales: two 4-bit scales per sub-block
        { 0, 2 },  // d: f16 super-scale
    };
};

// iq1_s, 50 bytes a block
//   canonical block:  [d 2][qs 32][qh 16]
//   reorder streams:  [qs][qh][d], each over every block of the slice
template <> struct ggml_sycl_layout<block_iq1_s> {
    enum { QS, QH, D, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 2, 32 },  // qs: low 8 bits of the 32 grid indices
        { 34, 16 }, // qh: per sub-block 3 high index bits a grid entry, scale and delta sign
        { 0, 2 },   // d: f16 super-scale
    };
};

// iq1_m, 56 bytes a block
//   canonical block:  [qs 32][qh 16][scales 8]
//   reorder streams:  [qs][qh][scales], each over every block of the slice
template <> struct ggml_sycl_layout<block_iq1_m> {
    enum { QS, QH, SC, N };

    // { offset in the canonical block, bytes a block }, in reorder stream order
    static constexpr ggml_sycl_seg seg[N] = {
        { 0, 32 },  // qs: low 8 bits of the 32 grid indices
        { 32, 16 }, // qh: high index bits and delta signs
        { 48, 8 },  // scales: 3-bit sub-block scales; their top nibbles form the f16 super-scale
    };
};

// the first field of field f's stream
template <typename L> constexpr int ggml_sycl_layout_first(int f) {
    while (f > 0 && L::seg[f].join) {
        --f;
    }
    return f;
}

// bytes a block of the streams before field f's stream in the reorder layout
template <typename L> constexpr int ggml_sycl_layout_pre(const int f) {
    int pre = 0;
    for (int i = 0; i < ggml_sycl_layout_first<L>(f); ++i) {
        pre += L::seg[i].size;
    }
    return pre;
}

// bytes a block of field f's stream, and f's offset in them
template <typename L> constexpr int ggml_sycl_layout_stride(const int f) {
    int stride = L::seg[ggml_sycl_layout_first<L>(f)].size;
    for (int i = ggml_sycl_layout_first<L>(f) + 1; i < L::N && L::seg[i].join; ++i) {
        stride += L::seg[i].size;
    }
    return stride;
}

template <typename L> constexpr int ggml_sycl_layout_off(const int f) {
    int off = 0;
    for (int i = ggml_sycl_layout_first<L>(f); i < f; ++i) {
        off += L::seg[i].size;
    }
    return off;
}

// the fields cover the block once: their sizes add up to it and none reaches past it
template <typename block_q_t> constexpr bool ggml_sycl_layout_ok() {
    using L = ggml_sycl_layout<block_q_t>;
    for (int i = 0; i < L::N; ++i) {
        if (L::seg[i].aos < 0 || L::seg[i].aos + L::seg[i].size > (int) sizeof(block_q_t)) {
            return false;
        }
    }
    int total = 0;
    for (int i = 0; i < L::N; ++i) {
        total += L::seg[i].size;
    }
    return total == (int) sizeof(block_q_t);
}

static_assert(ggml_sycl_layout_ok<block_q8_0>() && ggml_sycl_layout_ok<block_q4_0>() &&
                  ggml_sycl_layout_ok<block_q4_1>() && ggml_sycl_layout_ok<block_q5_0>() &&
                  ggml_sycl_layout_ok<block_q5_1>() && ggml_sycl_layout_ok<block_q1_0>() &&
                  ggml_sycl_layout_ok<block_q2_0>() && ggml_sycl_layout_ok<block_mxfp4>() &&
                  ggml_sycl_layout_ok<block_nvfp4>() && ggml_sycl_layout_ok<block_q2_K>() &&
                  ggml_sycl_layout_ok<block_q3_K>() && ggml_sycl_layout_ok<block_q4_K>() &&
                  ggml_sycl_layout_ok<block_q5_K>() && ggml_sycl_layout_ok<block_q6_K>() &&
                  ggml_sycl_layout_ok<block_iq4_nl>() && ggml_sycl_layout_ok<block_iq4_xs>() &&
                  ggml_sycl_layout_ok<block_iq3_s>() && ggml_sycl_layout_ok<block_iq3_xxs>() &&
                  ggml_sycl_layout_ok<block_iq2_xxs>() && ggml_sycl_layout_ok<block_iq2_xs>() &&
                  ggml_sycl_layout_ok<block_iq2_s>() && ggml_sycl_layout_ok<block_iq1_s>() &&
                  ggml_sycl_layout_ok<block_iq1_m>(),
              "a layout table does not cover its block");
static_assert(offsetof(block_q8_0, qs) == 2 && offsetof(block_q4_0, qs) == 2 && offsetof(block_q4_1, qs) == 4 &&
                  offsetof(block_q5_0, qh) == 2 && offsetof(block_q5_0, qs) == 6 && offsetof(block_q5_1, qh) == 4 &&
                  offsetof(block_q5_1, qs) == 8 && offsetof(block_q1_0, qs) == 2 && offsetof(block_q2_0, qs) == 2 &&
                  offsetof(block_mxfp4, qs) == 1 && offsetof(block_nvfp4, qs) == 4 && offsetof(block_q2_K, qs) == 16 &&
                  offsetof(block_q2_K, dm) == 80 && offsetof(block_q3_K, qs) == 32 &&
                  offsetof(block_q3_K, scales) == 96 && offsetof(block_q3_K, d) == 108 &&
                  offsetof(block_q4_K, scales) == 4 && offsetof(block_q4_K, qs) == 16 &&
                  offsetof(block_q5_K, qh) == 16 && offsetof(block_q5_K, qs) == 48 && offsetof(block_q6_K, qh) == 128 &&
                  offsetof(block_q6_K, scales) == 192 && offsetof(block_q6_K, d) == 208 &&
                  offsetof(block_iq4_nl, qs) == 2 && offsetof(block_iq4_xs, scales_h) == 2 &&
                  offsetof(block_iq4_xs, scales_l) == 4 && offsetof(block_iq4_xs, qs) == 8 &&
                  offsetof(block_iq3_s, qh) == 66 && offsetof(block_iq3_s, signs) == 74 &&
                  offsetof(block_iq3_s, scales) == 106 && offsetof(block_iq3_xxs, qs) == 2 &&
                  offsetof(block_iq2_xxs, qs) == 2 && offsetof(block_iq2_xs, scales) == 66 &&
                  offsetof(block_iq2_s, qh) == 66 && offsetof(block_iq2_s, scales) == 74 &&
                  offsetof(block_iq1_s, qh) == 34 && offsetof(block_iq1_m, qh) == 32 &&
                  offsetof(block_iq1_m, scales) == 48,
              "a layout table disagrees with its block");

// byte offset from the start of a slice of nblocks blocks of field f of block ib, in the reorder layout
template <typename L> constexpr int64_t ggml_sycl_layout_at(const int f, const int64_t ib, const int64_t nblocks) {
    return nblocks * ggml_sycl_layout_pre<L>(f) + ib * ggml_sycl_layout_stride<L>(f) + ggml_sycl_layout_off<L>(f);
}

#endif // GGML_SYCL_LAYOUT_HPP
