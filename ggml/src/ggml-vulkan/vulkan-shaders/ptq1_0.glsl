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

#endif // PTQ1_0_GLSL
