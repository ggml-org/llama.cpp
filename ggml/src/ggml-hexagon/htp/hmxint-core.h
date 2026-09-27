// hmxint-core.h - integer-HMX Q4_0 matmul building blocks (HVX + HMX), shared by the libnative tests (x86) and the DSP build.
//
// y[m][n] = sum_k a[m][k] * w[n][k],  w = q4_0 (q in [-8,7], one fp16 scale d per 32 k).
// Weights: per group of HMXI_G blocks, per column c: dmax = max|d|, w8 = round(q * 16 * d / dmax)  (int8, |w8| <= 128)
//          HMX output scale for the group s = (dmax / 16) * 2^j_c   (fp16, j_c per column keeps |out| < 2^15)
// Activations: per token, a_u = trunc(a * 32767/amax + 32768.5) in [0, 65535] (offset binary, a16 via 2x1 row pairs)
// HMX .uh 2x1 output per (row tile, col tile, group): out = floor(v * s / 512), v = sum a_u * w8
// bias2 word per column = -128 * S_g (S_g = sum_k w8) removes the offset: v = sum a_q * w8 exactly.
// y = sa * 2^-j_c * 512 * (sum_g out_g - C_c),  C_c = -0.5 * n_groups (floor bias).
//
// Tile layouts (verified bit-exact on SM7750):
//   activation: 32 rows x 32 k, byte (r/4)*128 + k*4 + r%4, row 2t = low byte / 2t+1 = high byte of token t;
//               k-tiles every 2048 bytes (first 1024 used)
//   weight:     32 k x 32 cols int8, byte (k/4)*128 + c*4 + k%4; k-tiles every 1024 bytes
//   output:     .uh 2x1, 16 tokens x 32 cols int16 at (t/2)*64 + c*2 + t%2
#ifndef HMXINT_CORE_H
#define HMXINT_CORE_H

#include <stdint.h>
#include <string.h>
#include <math.h>

#ifndef HMXI_ENABLE
#define HMXI_ENABLE 1        // 0: route everything to the stock (FP16 HMX / HVX) kernels, for A/B on FP16-capable SoCs
#endif
#define HMXI_G         16    // q4_0 blocks (k-tiles) per group
#define HMXI_Q40_TILE  576   // tiled q4_0: 512 B nibbles + 32 fp16 scales

static inline HVX_Vector hmxi_ldu(const void * p) { HVX_Vector v; memcpy(&v, p, sizeof(v)); return v; }
static inline void       hmxi_stu(void * p, HVX_Vector v) { memcpy(p, &v, sizeof(v)); }
static inline uint32_t   hmxi_f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float      hmxi_h2f(uint16_t h) {   // IEEE fp16 -> fp32 (portable)
    uint32_t sign = (uint32_t) (h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023, u;
    if (e == 0) { float f = ldexpf((float) m, -24); return sign ? -f : f; }
    if (e == 31) u = sign | 0x7f800000 | (m << 13); else u = sign | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}

// fp16-like HMX scale encoding (sign, e = bits 14..10, m = bits 9..0, value 2^(e-15)(1+m/1024), no subnormals)
static inline uint16_t hmxi_enc(float s) {
    uint16_t sign = s < 0.0f ? 0x8000 : 0; s = fabsf(s);
    if (!(s > 0.0f)) return 0;
    int ex; float f = frexpf(s, &ex);
    int e = ex - 1 + 15, m = (int) ((2.0f * f - 1.0f) * 1024.0f + 0.5f);
    if (m == 1024) { m = 0; e++; }
    if (e < 0) return 0;
    if (e > 30) { e = 30; m = 1023; }
    return (uint16_t) (sign | (e << 10) | m);
}
static inline float hmxi_dec(uint16_t v) {
    if ((v & 0x7fff) == 0) return 0.0f;
    int e = (v >> 10) & 31, m = v & 1023; float r = ldexpf((float) (1024 + m), e - 25);
    return (v & 0x8000) ? -r : r;
}

// max |x| over n floats (n % 32 == 0)
static inline float hmxi_absmax(const float * x, int n) {
    HVX_Vector vmax = Q6_V_vzero(); const HVX_Vector mask = Q6_V_vsplat_R(0x7fffffff);
    for (int i = 0; i < n; i += 32) vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vand_VV(hmxi_ldu(x + i), mask));
    int32_t t[32]; hmxi_stu(t, vmax);
    int32_t m = 0; for (int i = 0; i < 32; i++) m = t[i] > m ? t[i] : m;
    float r; memcpy(&r, &m, 4); return r;
}

// Quantize tokens a (even) and b (odd) of row pair p over nkt k-tiles (32 k each) into activation tiles.
// a/b may be NULL (padding token -> a_u = 32768).
static inline void hmxi_act_pair(const float * a, const float * b, float ia, float ib, int nkt, uint8_t * tiles, int p) {
    const HVX_Vector vA = Q6_V_vsplat_R(hmxi_f2u(ia)), vB = Q6_V_vsplat_R(hmxi_f2u(ib));
    const HVX_Vector off = Q6_V_vsplat_R(0x47000080);   // 32768.5f
    const HVX_Vector zero = Q6_V_vzero();
#define HMXI_Q(x, s) Q6_Vw_equals_Vsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf((x), (s)), off)))
    for (int kt = 0; kt < nkt; kt += 2) {
        const int two = (kt + 1 < nkt);
        HVX_Vector a0 = a ? hmxi_ldu(a + kt * 32) : zero, a1 = (a && two) ? hmxi_ldu(a + kt * 32 + 32) : zero;
        HVX_Vector b0 = b ? hmxi_ldu(b + kt * 32) : zero, b1 = (b && two) ? hmxi_ldu(b + kt * 32 + 32) : zero;
        HVX_Vector ua = Q6_Vuh_vpack_VwVw_sat(HMXI_Q(a1, vA), HMXI_Q(a0, vA));
        HVX_Vector ub = Q6_Vuh_vpack_VwVw_sat(HMXI_Q(b1, vB), HMXI_Q(b0, vB));
        HVX_VectorPair s = Q6_W_vshuff_VVR(ub, ua, -2);
        *(HVX_Vector *) (tiles + (size_t) kt * 2048 + p * 128) = Q6_V_lo_W(s);
        if (two) *(HVX_Vector *) (tiles + (size_t) (kt + 1) * 2048 + p * 128) = Q6_V_hi_W(s);
    }
#undef HMXI_Q
}

// One tiled q4_0 tile -> int8 HMX weight tile, column c scaled by F15[c]/32768 * 16 (i.e. w = round(q * 16 * F15/32768)).
// Returns S + per-column sums of the int8 weights.
static inline HVX_Vector hmxi_cvt_wtile(const uint8_t * q, uint8_t * dst, HVX_Vector F_lo, HVX_Vector F_hi, HVX_Vector S) {
    const HVX_Vector m4 = Q6_Vb_vsplat_R(0x0F), v8 = Q6_Vb_vsplat_R(8), ones = Q6_Vb_vsplat_R(1);
    for (int g = 0; g < 4; g++) {
        HVX_Vector v  = hmxi_ldu(q + g * 128);
        HVX_Vector lo = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(v, m4), v8);
        HVX_Vector hi = Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(v, 4), v8);
        HVX_VectorPair P = Q6_W_vshuff_VVR(hi, lo, -1);
        for (int h = 0; h < 2; h++) {
            HVX_Vector X = h ? Q6_V_hi_W(P) : Q6_V_lo_W(P);
            HVX_Vector O = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vror_VR(X, 64), X, -2));
            HVX_VectorPair U = Q6_Wh_vunpack_Vb(O);
            HVX_Vector rlo = Q6_Vh_vmpy_VhVh_s1_rnd_sat(Q6_Vh_vasl_VhR(Q6_V_lo_W(U), 4), F_lo);
            HVX_Vector rhi = Q6_Vh_vmpy_VhVh_s1_rnd_sat(Q6_Vh_vasl_VhR(Q6_V_hi_W(U), 4), F_hi);
            HVX_Vector W = Q6_Vb_vpack_VhVh_sat(rhi, rlo);
            *(HVX_Vector *) (dst + (2 * g + h) * 128) = W;
            S = Q6_Vw_vadd_VwVw(S, Q6_Vw_vrmpy_VbVb(W, ones));
        }
    }
    return S;
}

// Per column of one column tile: exponent j_c such that (max|d| / 16) * 2^j_c <= 2^-7 over all ntiles (tiles contiguous).
// Returns the column max |d| as fp16 bits in dmax_bits[32]; j[32] receives j_c.
static inline void hmxi_col_scan(const uint8_t * qtiles, int ntiles, int * j, uint16_t * dmax_bits) {
    const HVX_Vector m7 = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector M = Q6_V_vzero();
    uint16_t sc[64] __attribute__((aligned(128)));
    memset(sc + 32, 0, 64);
    for (int t = 0; t < ntiles; t++) {
        memcpy(sc, qtiles + (size_t) t * HMXI_Q40_TILE + 512, 64);
        M = Q6_Vuh_vmax_VuhVuh(M, Q6_V_vand_VV(*(HVX_Vector *) sc, m7));
    }
    uint16_t mb[64] __attribute__((aligned(128)));
    *(HVX_Vector *) mb = M;
    for (int c = 0; c < 32; c++) {
        int e = (mb[c] >> 10) & 31, m = mb[c] & 1023;
        j[c] = mb[c] == 0 ? 0 : (m == 0 ? 12 - e : 11 - e);
        dmax_bits[c] = mb[c];
    }
}

// Convert one group (nb <= HMXI_G tiles along k) of one column tile (vectorized).
//   qtiles: first tiled q4_0 tile of the group (tiles contiguous along k, HMXI_Q40_TILE apart)
//   dst:    first int8 tile (1024 apart);  bias: 128 words = coarse block (64) + fine block (64)
//   j:      per-column exponent j_c (32 ints);  C: per-column correction accumulator (32 floats)
static inline void hmxi_cvt_group(const uint8_t * qtiles, int nb, uint8_t * dst, uint32_t * bias, const int * j, float * C) {
    uint16_t sc[64] __attribute__((aligned(128)));        // one block's scales (upper half zero); any nb works
    memset(sc + 32, 0, 64);
    const HVX_Vector m7 = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector M = Q6_V_vzero();
    for (int b = 0; b < nb; b++) {
        memcpy(sc, qtiles + (size_t) b * HMXI_Q40_TILE + 512, 64);
        M = Q6_Vuh_vmax_VuhVuh(M, Q6_V_vand_VV(*(HVX_Vector *) sc, m7));
    }
    uint16_t mb[64] __attribute__((aligned(128)));
    float invE[32] __attribute__((aligned(128))), invO[32] __attribute__((aligned(128)));
    *(HVX_Vector *) mb = M;
    for (int c = 0; c < 32; c++) {
        float dm = hmxi_h2f(mb[c]);
        float inv = dm > 0.0f ? 32768.0f / dm : 0.0f;
        if (c & 1) invO[c / 2] = inv; else invE[c / 2] = inv;
    }
    for (int i = 16; i < 32; i++) { invE[i] = 0.0f; invO[i] = 0.0f; }
    const HVX_Vector vIE = *(HVX_Vector *) invE, vIO = *(HVX_Vector *) invO;
    const HVX_Vector k7fff = Q6_V_vsplat_R(0x7fff), k112 = Q6_V_vsplat_R(112 << 23), sgn = Q6_V_vsplat_R(0x8000);
    const HVX_Vector zero = Q6_V_vzero(), vmaxF = Q6_V_vsplat_R(32767), vminF = Q6_V_vsplat_R(-32768);
    HVX_Vector S = Q6_V_vzero();
    for (int b = 0; b < nb; b++) {
        memcpy(sc, qtiles + (size_t) b * HMXI_Q40_TILE + 512, 64);
        HVX_VectorPair X = Q6_Wuw_vzxt_Vuh(*(HVX_Vector *) sc);      // lo = even columns, hi = odd columns (fp16 bits)
        HVX_Vector F[2];
        for (int h = 0; h < 2; h++) {
            HVX_Vector x   = h ? Q6_V_hi_W(X) : Q6_V_lo_W(X);
            HVX_Vector mag = Q6_V_vand_VV(x, k7fff);
            HVX_Vector f32 = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_vand_VV(x, sgn), 16), Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(mag, 13), k112));
            f32 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_Vw_vasr_VwR(mag, 10), zero), f32, zero);   // exponent 0 (zero/subnormal) -> 0
            HVX_Vector r = Q6_Vw_equals_Vsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(f32, h ? vIO : vIE)));
            F[h] = Q6_Vw_vmax_VwVw(Q6_Vw_vmin_VwVw(r, vmaxF), vminF);
        }
        HVX_Vector Fh = Q6_Vh_vshuffe_VhVh(F[1], F[0]);                       // column order c0..c31
        HVX_Vector V1 = Q6_V_lo_W(Q6_W_vshuff_VVR(Fh, Fh, -2));
        HVX_VectorPair P4 = Q6_W_vshuff_VVR(V1, V1, -4);                       // lo = c0..c15 x4, hi = c16..c31 x4
        S = hmxi_cvt_wtile(qtiles + (size_t) b * HMXI_Q40_TILE, dst + (size_t) b * 1024, Q6_V_lo_W(P4), Q6_V_hi_W(P4), S);
    }
    int32_t Sv[32] __attribute__((aligned(128)));
    *(HVX_Vector *) Sv = S;
    for (int c = 0; c < 32; c++) {
        int e = ((mb[c] >> 10) & 31) + j[c] - 4;                              // exponent of (dmax/16) * 2^j (<= 8)
        uint16_t co = (mb[c] == 0 || e < 0) ? 0 : (uint16_t) ((e << 10) | (mb[c] & 1023));
        uint16_t fi = co ? (uint16_t) (co + (8 << 10)) : 0;                   // fine = coarse * 256 (exponent <= 16)
        uint32_t b2 = (uint32_t) (-128 * Sv[c]);                              // bias2 adds x256 to v: cancels 32768*S
        bias[c] = co; bias[32 + c] = b2; bias[64 + c] = fi; bias[96 + c] = b2;
        C[c] += -0.5f;
    }
}

// Same, but the group is the whole k-chunk: the per-column exponent comes from this group's own max (no column pass).
//   shift: extra exponent reduction so the coarse store cannot overflow for nb > 16 tiles (16 << shift >= nb)
//   k512:  out, per column 512 * 2^-j / 256 for the reduce step
static inline void hmxi_cvt_group_whole(const uint8_t * qtiles, int nb, uint8_t * dst, uint32_t * bias, int shift, float * k512, float * C) {
    uint16_t sc[64] __attribute__((aligned(128)));        // one block's scales (upper half zero); any nb works
    memset(sc + 32, 0, 64);
    const HVX_Vector m7 = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector M = Q6_V_vzero();
    for (int b = 0; b < nb; b++) {
        memcpy(sc, qtiles + (size_t) b * HMXI_Q40_TILE + 512, 64);
        M = Q6_Vuh_vmax_VuhVuh(M, Q6_V_vand_VV(*(HVX_Vector *) sc, m7));
    }
    uint16_t mb[64] __attribute__((aligned(128)));
    float invE[32] __attribute__((aligned(128))), invO[32] __attribute__((aligned(128)));
    *(HVX_Vector *) mb = M;
    for (int c = 0; c < 32; c++) {
        float dm = hmxi_h2f(mb[c]);
        float inv = dm > 0.0f ? 32768.0f / dm : 0.0f;
        if (c & 1) invO[c / 2] = inv; else invE[c / 2] = inv;
    }
    for (int i = 16; i < 32; i++) { invE[i] = 0.0f; invO[i] = 0.0f; }
    const HVX_Vector vIE = *(HVX_Vector *) invE, vIO = *(HVX_Vector *) invO;
    const HVX_Vector k7fff = Q6_V_vsplat_R(0x7fff), k112 = Q6_V_vsplat_R(112 << 23), sgn = Q6_V_vsplat_R(0x8000);
    const HVX_Vector zero = Q6_V_vzero(), vmaxF = Q6_V_vsplat_R(32767), vminF = Q6_V_vsplat_R(-32768);
    HVX_Vector S = Q6_V_vzero();
    for (int b = 0; b < nb; b++) {
        memcpy(sc, qtiles + (size_t) b * HMXI_Q40_TILE + 512, 64);
        HVX_VectorPair X = Q6_Wuw_vzxt_Vuh(*(HVX_Vector *) sc);      // lo = even columns, hi = odd columns (fp16 bits)
        HVX_Vector F[2];
        for (int h = 0; h < 2; h++) {
            HVX_Vector x   = h ? Q6_V_hi_W(X) : Q6_V_lo_W(X);
            HVX_Vector mag = Q6_V_vand_VV(x, k7fff);
            HVX_Vector f32 = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_vand_VV(x, sgn), 16), Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(mag, 13), k112));
            f32 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_Vw_vasr_VwR(mag, 10), zero), f32, zero);   // exponent 0 (zero/subnormal) -> 0
            HVX_Vector r = Q6_Vw_equals_Vsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(f32, h ? vIO : vIE)));
            F[h] = Q6_Vw_vmax_VwVw(Q6_Vw_vmin_VwVw(r, vmaxF), vminF);
        }
        HVX_Vector Fh = Q6_Vh_vshuffe_VhVh(F[1], F[0]);                       // column order c0..c31
        HVX_Vector V1 = Q6_V_lo_W(Q6_W_vshuff_VVR(Fh, Fh, -2));
        HVX_VectorPair P4 = Q6_W_vshuff_VVR(V1, V1, -4);                       // lo = c0..c15 x4, hi = c16..c31 x4
        S = hmxi_cvt_wtile(qtiles + (size_t) b * HMXI_Q40_TILE, dst + (size_t) b * 1024, Q6_V_lo_W(P4), Q6_V_hi_W(P4), S);
    }
    int32_t Sv[32] __attribute__((aligned(128)));
    *(HVX_Vector *) Sv = S;
    for (int c = 0; c < 32; c++) {
        const int ec = (mb[c] >> 10) & 31, jc = (mb[c] == 0 ? 0 : ((mb[c] & 1023) == 0 ? 12 - ec : 11 - ec)) - shift;
        k512[c] = ldexpf(2.0f, -jc);
        int e = ec + jc - 4;                                                  // exponent of (dmax/16) * 2^j (= 7..8 - shift)
        uint16_t co = (mb[c] == 0 || e < 0) ? 0 : (uint16_t) ((e << 10) | (mb[c] & 1023));
        uint16_t fi = co ? (uint16_t) (co + (8 << 10)) : 0;                   // fine = coarse * 256 (exponent <= 16)
        uint32_t b2 = (uint32_t) (-128 * Sv[c]);                              // bias2 adds x256 to v: cancels 32768*S
        bias[c] = co; bias[32 + c] = b2; bias[64 + c] = fi; bias[96 + c] = b2;
        C[c] += -0.5f;
    }
}

// Sum the group outputs of one (row tile, col tile) and emit y for its 16 tokens x 32 cols.
//   stage: per group a coarse tile then a fine tile (each 1024 B; group stride 2048); fine = coarse * 256 wrapped mod 2^16:
//          true_fine = fine + 65536 * round((coarse*256 - fine) / 65536)
//   rows: 16 dst row pointers (NULL = skip); sa: 16 token scales; k512: per column 512 * 2^-j_c / 256; C: per column correction
//   ncols: valid columns in this tile (< 32 only for the last, padded tile)
static inline void hmxi_reduce_tile(const uint8_t * stage, int n_g, float * const * rows, const float * sa,
                                    const float * k512, const float * C, int accumulate, int ncols) {
    const HVX_Vector vK = hmxi_ldu(k512), vC = hmxi_ldu(C), half = Q6_V_vsplat_R(32768);
    for (int j = 0; j < 8; j++) {                 // vector j = tokens 2j (even halfwords) and 2j+1 (odd)
        HVX_Vector acc_e = Q6_V_vzero(), acc_o = Q6_V_vzero();
        for (int g = 0; g < n_g; g++) {
            HVX_VectorPair wc = Q6_Ww_vsxt_Vh(*(const HVX_Vector *) (stage + (size_t) g * 2048 + j * 128));
            HVX_VectorPair wf = Q6_Ww_vsxt_Vh(*(const HVX_Vector *) (stage + (size_t) g * 2048 + 1024 + j * 128));
            for (int h = 0; h < 2; h++) {
                HVX_Vector c8 = Q6_Vw_vasl_VwR(h ? Q6_V_hi_W(wc) : Q6_V_lo_W(wc), 8);
                HVX_Vector fi = h ? Q6_V_hi_W(wf) : Q6_V_lo_W(wf);
                HVX_Vector k  = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vsub_VwVw(c8, fi), half), 16);
                HVX_Vector t  = Q6_Vw_vadd_VwVw(fi, Q6_Vw_vasl_VwR(k, 16));
                if (h) acc_o = Q6_Vw_vadd_VwVw(acc_o, t); else acc_e = Q6_Vw_vadd_VwVw(acc_e, t);
            }
        }
        for (int h = 0; h < 2; h++) {
            float * row = rows[2 * j + h];
            if (!row) continue;
            HVX_Vector acc = h ? acc_o : acc_e;
            HVX_Vector t   = Q6_Vqf32_vsub_VsfVsf(Q6_Vsf_equals_Vw(acc), vC);
            HVX_Vector ks  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vK, Q6_V_vsplat_R(hmxi_f2u(sa[2 * j + h]))));
            HVX_Vector y   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(t), ks));
            if (ncols >= 32) {
                if (accumulate) y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(y, hmxi_ldu(row)));
                hmxi_stu(row, y);
            } else {                                  // partial column tile (dst narrower than the padded weight)
                float tmp[32]; hmxi_stu(tmp, y);
                for (int c = 0; c < ncols; c++) row[c] = accumulate ? row[c] + tmp[c] : tmp[c];
            }
        }
    }
}


// ---------------------------------------------------------------------------------------------------------------
// Generic f32/f16 weights (used by attention): W[c][k] given through a getter-free layout:
//   w(c, k) = src[c * cstride + k * kstride]  (float), c < n_valid_cols (others 0), k < kdim (multiple of 32)
// Per column: s0 = fp16_round_up(max|w| / 127), w8 = round(w / s0); HMX scale = s0 * 2^j (j per column, bound for the group).
// Emits int8 tiles for one column tile (tiles along k, 1024 B apart) and bias blocks per group of HMXI_G tiles
// (128 words each: coarse + fine), accumulates C (-0.5 per group) and returns per column k512 = 2^(1-j) and s0.
static inline uint16_t hmxi_f16_round_up_bits(float x) {        // smallest fp16 (normal) >= x > 0
    int ex; float f = frexpf(x, &ex); int e = ex - 1 + 15;
    int m = (int) ceilf((2.0f * f - 1.0f) * 1024.0f);
    if (m == 1024) { m = 0; e++; }
    if (e < 1) { e = 1; m = 0; }
    if (e > 30) { e = 30; m = 1023; }
    return (uint16_t) ((e << 10) | m);
}
static inline void hmxi_cvt_fw_coltile(const float * src, size_t cstride, size_t kstride, int n_valid_cols, int kdim,
                                        uint8_t * dst_tiles, uint32_t * bias, float * k512, float * C) {
    const int nkt = kdim / 32, ng = (nkt + HMXI_G - 1) / HMXI_G;
    float s0[32]; uint16_t s0b[32]; int jc[32];
    for (int c = 0; c < 32; c++) {
        float am = 0.0f;
        if (c < n_valid_cols) for (int k = 0; k < kdim; k++) { float v = fabsf(src[c * cstride + k * kstride]); am = v > am ? v : am; }
        if (am > 0.0f) { s0b[c] = hmxi_f16_round_up_bits(am / 127.0f); s0[c] = hmxi_h2f(s0b[c]); } else { s0b[c] = 0; s0[c] = 0.0f; }
        // coarse bound: Kg*65535*127 * s/512 < 2^15 with Kg = 32*HMXI_G  ->  s0 * 2^j <= 2^15*512 / (32*HMXI_G*65535*127)
        float lim = 16777216.0f / (32.0f * HMXI_G * 65535.0f * 127.0f);
        int j = 0; if (s0[c] > 0.0f) { int ex; frexpf(lim / s0[c], &ex); j = ex - 1; }
        jc[c] = j; k512[c] = ldexpf(2.0f, -j); C[c] = 0.0f;
    }
    for (int g = 0; g < ng; g++) {
        int kt0 = g * HMXI_G, nb = nkt - kt0 < HMXI_G ? nkt - kt0 : HMXI_G;
        int32_t S[32] = {0};
        for (int kt = kt0; kt < kt0 + nb; kt++) {
            int8_t * t = (int8_t *) (dst_tiles + (size_t) kt * 1024);
            for (int k = 0; k < 32; k++) for (int c = 0; c < 32; c++) {
                int w = 0;
                if (c < n_valid_cols && s0[c] > 0.0f) {
                    w = (int) lrintf(src[c * cstride + (size_t) (kt * 32 + k) * kstride] / s0[c]);
                    w = w > 127 ? 127 : (w < -127 ? -127 : w);
                }
                t[(k / 4) * 128 + c * 4 + (k % 4)] = (int8_t) w; S[c] += w;
            }
        }
        uint32_t * b = bias + (size_t) g * 128;
        for (int c = 0; c < 32; c++) {
            int e = ((s0b[c] >> 10) & 31) + jc[c];
            uint16_t co = (s0b[c] == 0 || e < 0) ? 0 : (uint16_t) ((e << 10) | (s0b[c] & 1023));
            if (e > 8) co = 0x2000 | (s0b[c] & 1023);                  // guard (should not happen)
            uint16_t fi = co ? (uint16_t) (co + (8 << 10)) : 0;
            uint32_t b2 = (uint32_t) (-128 * S[c]);
            b[c] = co; b[32 + c] = b2; b[64 + c] = fi; b[96 + c] = b2;
            C[c] += -0.5f;
        }
    }
    (void) s0b;
}

// Same as hmxi_cvt_fw_coltile, but the source is fp16 (IEEE half bits) with an optional per-k multiplier kmul[k]
// and a valid-k bound (k >= kvalid reads as 0). Used for attention K (kmul = 1/channel scale) and V^T.
static inline void hmxi_cvt_hw_coltile(const uint16_t * src, size_t cstride, size_t kstride, int n_valid_cols, int kdim, int kvalid,
                                        const float * kmul, uint8_t * dst_tiles, uint32_t * bias, float * k512, float * C) {
    const int nkt = kdim / 32, ng = (nkt + HMXI_G - 1) / HMXI_G;
    float s0[32], inv[32]; uint16_t s0b[32]; int jc[32];
    for (int c = 0; c < 32; c++) {
        float am = 0.0f;
        if (c < n_valid_cols) for (int k = 0; k < kvalid; k++) {
            float v = fabsf(hmxi_h2f(src[c * cstride + (size_t) k * kstride]) * (kmul ? kmul[k] : 1.0f)); am = v > am ? v : am;
        }
        if (am > 0.0f) { s0b[c] = hmxi_f16_round_up_bits(am / 127.0f); s0[c] = hmxi_h2f(s0b[c]); inv[c] = 1.0f / s0[c]; }
        else { s0b[c] = 0; s0[c] = 0.0f; inv[c] = 0.0f; }
        float lim = 16777216.0f / (32.0f * HMXI_G * 65535.0f * 127.0f);
        int j = 0; if (s0[c] > 0.0f) { int ex; frexpf(lim / s0[c], &ex); j = ex - 1; }
        jc[c] = j; k512[c] = ldexpf(2.0f, -j); C[c] = 0.0f;
    }
    for (int g = 0; g < ng; g++) {
        int kt0 = g * HMXI_G, nb = nkt - kt0 < HMXI_G ? nkt - kt0 : HMXI_G;
        int32_t S[32] = {0};
        for (int kt = kt0; kt < kt0 + nb; kt++) {
            int8_t * t = (int8_t *) (dst_tiles + (size_t) kt * 1024);
            for (int k = 0; k < 32; k++) {
                int kk = kt * 32 + k; float km = (kmul && kk < kvalid) ? kmul[kk] : 1.0f;
                for (int c = 0; c < 32; c++) {
                    int w = 0;
                    if (c < n_valid_cols && kk < kvalid && s0[c] > 0.0f) {
                        w = (int) lrintf(hmxi_h2f(src[c * cstride + (size_t) kk * kstride]) * km * inv[c]);
                        w = w > 127 ? 127 : (w < -127 ? -127 : w);
                    }
                    t[(k / 4) * 128 + c * 4 + (k % 4)] = (int8_t) w; S[c] += w;
                }
            }
        }
        uint32_t * b = bias + (size_t) g * 128;
        for (int c = 0; c < 32; c++) {
            int e = ((s0b[c] >> 10) & 31) + jc[c];
            uint16_t co = (s0b[c] == 0 || e < 0) ? 0 : (uint16_t) ((e << 10) | (s0b[c] & 1023));
            uint16_t fi = co ? (uint16_t) (co + (8 << 10)) : 0;
            uint32_t b2 = (uint32_t) (-128 * S[c]);
            b[c] = co; b[32 + c] = b2; b[64 + c] = fi; b[96 + c] = b2;
            C[c] += -0.5f;
        }
    }
}

#endif // HMXINT_CORE_H
