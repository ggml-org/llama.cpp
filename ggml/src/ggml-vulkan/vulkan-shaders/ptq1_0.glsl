#ifndef PTQ1_0_GLSL
#define PTQ1_0_GLSL

// Shared PTQ1_0 trit accessor. Lives in its own header because two consumers need it
// from different include chains: dequant_funcs.glsl (mul_mat_vec, get_rows,
// copy_from_quant) and mul_mm.comp (via mul_mm_funcs.glsl), and mul_mm.comp does not
// include dequant_funcs.glsl. Duplicating it would leave two copies that must stay in
// step with the CPU codec in ggml-quants.c, where a divergence shows up as wrong
// matmul results rather than a build error.
//
// Element order is not positional: a 16-byte chunk of qs where byte j carries
// elements t*16+j, then an 8-byte chunk carrying 80 + t*8 + (j-16), then qh at four
// trits per byte carrying 120 + t*2 + h. Trits come out by the base-3 remainder
// recurrence t = (v*3)>>8, v = (v*3)&0xFF.
// 3^n for n in [0,4]. Written as selects rather than a const array: a
// dynamically indexed local array can land in scratch memory on some drivers,
// and this sits on the hot path of every ternary matmul.
uint ptq1_0_pow3(uint n) {
    return n < 2u ? (n == 0u ? 1u : 3u)
                  : (n == 2u ? 9u : (n == 3u ? 27u : 81u));
}

float ptq1_0_trit(uint ib, uint a_offset, uint e) {
    uint b;
    uint n;
    if (e < 80u) {
        b = uint(data_a[a_offset + ib].qs[e & 15u]);
        n = e >> 4u;
    } else if (e < 120u) {
        const uint t = e - 80u;
        b = uint(data_a[a_offset + ib].qs[16u + (t & 7u)]);
        n = t >> 3u;
    } else {
        const uint t = e - 120u;
        b = uint(data_a[a_offset + ib].qh[t & 1u]);
        n = t >> 1u;
    }

    // Mod-256 multiplication is associative, so the repeated v = (v*3)&0xFF that
    // walks to trit n collapses to a single multiply by 3^n -- which is what
    // dequantize_row_ptq1_0 in ggml-quants.c already does. Verified identical over
    // all 256 byte values x 5 trit positions. The serial form cost up to 4
    // dependent multiplies per element with a trip count that varies by element
    // position, so lanes in a subgroup diverged; this is branch-free and uniform.
    const uint v = (b * ptq1_0_pow3(n)) & 0xFFu;
    return float(int((v * 3u) >> 8u) - 1);
}

// Decode four independent bytes at the same trit position. One aligned word
// load replaces four byte loads; mask before multiplying by 3, not afterwards.
vec4 ptq1_0_decode4(uint packed, uint power) {
    const uvec4 bytes = (uvec4(packed) >> uvec4(0u, 8u, 16u, 24u)) & 255u;
    return vec4((((bytes * power) & 255u) * 3u) >> 8u) - 1.0f;
}

// e must be even. All qs region/position boundaries are multiples of eight.
vec2 ptq1_0_trits2(uint ib, uint a_offset, uint e) {
    uint packed;
    uint n;
    if (e < 120u) {
        const uint q = e < 80u ? (e & 15u) : (16u + (e & 7u));
        n = e < 80u ? (e >> 4u) : ((e - 80u) >> 3u);
        packed = data_a_packed32[a_offset + ib].qs[q >> 2u] >> ((q & 2u) * 8u);
    } else {
        packed = uint(data_a_packed32[a_offset + ib].qh);
        n = (e - 120u) >> 1u;
    }
    const uvec2 bytes = uvec2(packed, packed >> 8u) & 255u;
    return vec2((((bytes * ptq1_0_pow3(n)) & 255u) * 3u) >> 8u) - 1.0f;
}

// e must be four-aligned. The qh tail repeats two bytes at successive powers.
vec4 ptq1_0_trits4(uint ib, uint a_offset, uint e) {
    if (e < 120u) {
        const uint q = e < 80u ? (e & 15u) : (16u + (e & 7u));
        const uint n = e < 80u ? (e >> 4u) : ((e - 80u) >> 3u);
        return ptq1_0_decode4(data_a_packed32[a_offset + ib].qs[q >> 2u], ptq1_0_pow3(n));
    }
    const uint packed = uint(data_a_packed32[a_offset + ib].qh);
    const uvec2 bytes = uvec2(packed, packed >> 8u) & 255u;
    const uvec4 powers = e == 120u ? uvec4(1u, 1u, 3u, 3u) : uvec4(9u, 9u, 27u, 27u);
    return vec4((((bytes.xyxy * powers) & 255u) * 3u) >> 8u) - 1.0f;
}

// Shared hot path for matvec and matmul: one region decision and one power
// selection per eight outputs, with just two word loads (one halfword for qh).
void ptq1_0_trits8(uint ib, uint a_offset, uint e, out vec4 lo, out vec4 hi) {
    if (e < 120u) {
        const uint q = e < 80u ? (e & 15u) : 16u;
        const uint n = e < 80u ? (e >> 4u) : ((e - 80u) >> 3u);
        const uint power = ptq1_0_pow3(n);
        lo = ptq1_0_decode4(data_a_packed32[a_offset + ib].qs[q >> 2u], power);
        hi = ptq1_0_decode4(data_a_packed32[a_offset + ib].qs[(q >> 2u) + 1u], power);
    } else {
        const uint packed = uint(data_a_packed32[a_offset + ib].qh);
        const uvec2 bytes = uvec2(packed, packed >> 8u) & 255u;
        lo = vec4((((bytes.xyxy * uvec4(1u, 1u, 3u, 3u)) & 255u) * 3u) >> 8u) - 1.0f;
        hi = vec4((((bytes.xyxy * uvec4(9u, 9u, 27u, 27u)) & 255u) * 3u) >> 8u) - 1.0f;
    }
}

#endif // PTQ1_0_GLSL
