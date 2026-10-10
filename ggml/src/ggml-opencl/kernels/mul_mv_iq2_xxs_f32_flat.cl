#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// IQ2_XXS decode GEMV over the feature-major planes (size preserving, 66 == sizeof(block_iq2_xxs)):
//   src0_qs [row + (k/8)*m]    uchar  grid index, K = 8*grp .. +7
//   src0_sas[row + (k/32)*m]   uint   scale in bits 28..31, four 7-bit sign codes
//   src0_d  [row + (k/256)*m]  half   super-block scale
// K is split across IQ2XXS_MV_NSG subgroups; IQ2XXS_MV_R2 gives a lane two rows so qs is read a ushort
// at a time. A grid entry is eight bytes, so one lookup feeds two float4 dots (table as uint pairs).

#define QK_K 256

#ifndef IQ2XXS_MV_NSG
#define IQ2XXS_MV_NSG 8
#endif

constant uint iq2xxs_grid[512] = {
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x082b0808, 0x08080808,
    0x082b082b, 0x08080808, 0x082b2b08, 0x08080808, 0x082b2b2b, 0x08080808, 0x19080819, 0x08080808,
    0x19081908, 0x08080808, 0x19190808, 0x08080808, 0x19192b08, 0x08080808, 0x192b0819, 0x08080808,
    0x192b1908, 0x08080808, 0x2b080808, 0x08080808, 0x2b08082b, 0x08080808, 0x2b082b2b, 0x08080808,
    0x2b2b082b, 0x08080808, 0x08080819, 0x08080819, 0x08081908, 0x08080819, 0x08190808, 0x08080819,
    0x08191919, 0x08080819, 0x19080808, 0x08080819, 0x2b081908, 0x08080819, 0x2b192b08, 0x08080819,
    0x08080808, 0x0808082b, 0x0808082b, 0x0808082b, 0x082b082b, 0x0808082b, 0x2b08082b, 0x0808082b,
    0x08080819, 0x08081908, 0x08081908, 0x08081908, 0x08190808, 0x08081908, 0x082b0819, 0x08081908,
    0x082b1908, 0x08081908, 0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19082b08, 0x08081908,
    0x192b0808, 0x08081908, 0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b190808, 0x08081908,
    0x2b2b1908, 0x08081908, 0x08080808, 0x08081919, 0x0808082b, 0x08081919, 0x08082b08, 0x08081919,
    0x082b0808, 0x08081919, 0x1908192b, 0x08081919, 0x192b2b19, 0x08081919, 0x2b080808, 0x08081919,
    0x2b190819, 0x08081919, 0x08082b19, 0x0808192b, 0x08190808, 0x0808192b, 0x19080808, 0x0808192b,
    0x2b081908, 0x0808192b, 0x2b2b1908, 0x0808192b, 0x08080808, 0x08082b08, 0x08081919, 0x08082b08,
    0x08082b08, 0x08082b08, 0x08191908, 0x08082b08, 0x082b2b08, 0x08082b08, 0x19080819, 0x08082b08,
    0x19081908, 0x08082b08, 0x19190808, 0x08082b08, 0x1919082b, 0x08082b08, 0x2b082b08, 0x08082b08,
    0x08081908, 0x08082b19, 0x19080808, 0x08082b19, 0x0808082b, 0x08082b2b, 0x08191908, 0x08082b2b,
    0x08080819, 0x08190808, 0x08081908, 0x08190808, 0x08190808, 0x08190808, 0x082b0819, 0x08190808,
    0x19080808, 0x08190808, 0x192b0808, 0x08190808, 0x2b081908, 0x08190808, 0x2b190808, 0x08190808,
    0x2b191919, 0x08190808, 0x08080808, 0x08190819, 0x08082b08, 0x08190819, 0x082b0808, 0x08190819,
    0x19190808, 0x08190819, 0x19192b2b, 0x08190819, 0x2b080808, 0x08190819, 0x082b1908, 0x0819082b,
    0x19081919, 0x0819082b, 0x08080808, 0x08191908, 0x08082b08, 0x08191908, 0x082b0808, 0x08191908,
    0x082b1919, 0x08191908, 0x19082b19, 0x08191908, 0x2b080808, 0x08191908, 0x08192b08, 0x08191919,
    0x192b082b, 0x08191919, 0x08080808, 0x0819192b, 0x0819192b, 0x0819192b, 0x08080819, 0x08192b08,
    0x08081908, 0x08192b08, 0x08190808, 0x08192b08, 0x19080808, 0x08192b08, 0x2b080819, 0x08192b08,
    0x08080808, 0x08192b19, 0x08081919, 0x08192b19, 0x2b2b0808, 0x08192b19, 0x19190819, 0x08192b2b,
    0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08082b2b, 0x082b0808, 0x19081908, 0x082b0808,
    0x192b0819, 0x082b0808, 0x2b080808, 0x082b0808, 0x2b08082b, 0x082b0808, 0x082b2b19, 0x082b0819,
    0x19082b08, 0x082b0819, 0x08080808, 0x082b082b, 0x0808082b, 0x082b082b, 0x08080819, 0x082b1908,
    0x08081908, 0x082b1908, 0x08190808, 0x082b1908, 0x19080808, 0x082b1908, 0x1919192b, 0x082b1908,
    0x08080808, 0x082b1919, 0x19080819, 0x082b1919, 0x192b1908, 0x082b1919, 0x2b190808, 0x082b192b,
    0x08082b08, 0x082b2b08, 0x082b0808, 0x082b2b08, 0x2b191908, 0x082b2b08, 0x19081908, 0x082b2b2b,
    0x08080819, 0x19080808, 0x08081908, 0x19080808, 0x08190808, 0x19080808, 0x08192b08, 0x19080808,
    0x082b0819, 0x19080808, 0x082b1908, 0x19080808, 0x19080808, 0x19080808, 0x19082b08, 0x19080808,
    0x1919192b, 0x19080808, 0x192b0808, 0x19080808, 0x2b080819, 0x19080808, 0x2b081908, 0x19080808,
    0x2b190808, 0x19080808, 0x08080808, 0x19080819, 0x082b0808, 0x19080819, 0x192b0819, 0x19080819,
    0x2b080808, 0x19080819, 0x2b081919, 0x19080819, 0x08080819, 0x1908082b, 0x08190808, 0x1908082b,
    0x19082b08, 0x1908082b, 0x1919192b, 0x1908082b, 0x192b2b08, 0x1908082b, 0x08080808, 0x19081908,
    0x08082b08, 0x19081908, 0x082b0808, 0x19081908, 0x2b080808, 0x19081908, 0x2b192b19, 0x19081908,
    0x0819082b, 0x19081919, 0x082b1908, 0x19081919, 0x08080808, 0x1908192b, 0x08080819, 0x19082b08,
    0x08081908, 0x19082b08, 0x08190808, 0x19082b08, 0x19080808, 0x19082b08, 0x19081919, 0x19082b08,
    0x08080808, 0x19082b19, 0x19192b08, 0x19082b19, 0x192b0819, 0x19082b19, 0x2b08082b, 0x19082b19,
    0x19081919, 0x19082b2b, 0x2b190808, 0x19082b2b, 0x08080808, 0x19190808, 0x08082b08, 0x19190808,
    0x08190819, 0x19190808, 0x08192b19, 0x19190808, 0x082b0808, 0x19190808, 0x2b080808, 0x19190808,
    0x2b082b08, 0x19190808, 0x08081908, 0x19190819, 0x1908082b, 0x19190819, 0x2b2b1908, 0x19190819,
    0x2b190819, 0x1919082b, 0x2b190808, 0x19191908, 0x2b19082b, 0x19191908, 0x08082b2b, 0x19191919,
    0x08080819, 0x1919192b, 0x19191908, 0x1919192b, 0x08080808, 0x19192b08, 0x08190819, 0x19192b08,
    0x08192b19, 0x19192b08, 0x192b1908, 0x19192b08, 0x19080808, 0x19192b19, 0x08082b08, 0x19192b2b,
    0x08081908, 0x192b0808, 0x08190808, 0x192b0808, 0x19080808, 0x192b0808, 0x192b2b08, 0x192b0808,
    0x08080808, 0x192b0819, 0x19191919, 0x192b0819, 0x08192b08, 0x192b082b, 0x192b0808, 0x192b082b,
    0x08080808, 0x192b1908, 0x08081919, 0x192b1908, 0x08190808, 0x192b1919, 0x0819082b, 0x192b1919,
    0x2b081908, 0x192b1919, 0x1908082b, 0x192b2b08, 0x08080808, 0x2b080808, 0x0808082b, 0x2b080808,
    0x08082b2b, 0x2b080808, 0x19080819, 0x2b080808, 0x2b08082b, 0x2b080808, 0x08081908, 0x2b080819,
    0x08192b08, 0x2b080819, 0x19080808, 0x2b080819, 0x08190819, 0x2b08082b, 0x08080819, 0x2b081908,
    0x08081908, 0x2b081908, 0x08190808, 0x2b081908, 0x08191919, 0x2b081908, 0x19080808, 0x2b081908,
    0x192b0808, 0x2b081908, 0x08080808, 0x2b081919, 0x1908192b, 0x2b081919, 0x2b191908, 0x2b081919,
    0x08082b19, 0x2b08192b, 0x19080808, 0x2b08192b, 0x192b0808, 0x2b08192b, 0x0808082b, 0x2b082b08,
    0x08081908, 0x2b082b19, 0x08190819, 0x2b082b2b, 0x08081908, 0x2b190808, 0x08190808, 0x2b190808,
    0x082b1908, 0x2b190808, 0x19080808, 0x2b190808, 0x2b2b0819, 0x2b190808, 0x0819192b, 0x2b190819,
    0x2b080808, 0x2b190819, 0x19081919, 0x2b19082b, 0x08080808, 0x2b191908, 0x082b082b, 0x2b191908,
    0x19081908, 0x2b191908, 0x19190819, 0x2b191919, 0x2b080819, 0x2b192b08, 0x082b0808, 0x2b192b19,
    0x0808082b, 0x2b2b0808, 0x19190808, 0x2b2b0808, 0x2b081919, 0x2b2b0808, 0x08082b19, 0x2b2b0819,
    0x08080808, 0x2b2b082b, 0x08192b08, 0x2b2b1908, 0x19190808, 0x2b2b2b08, 0x08081908, 0x2b2b2b19
};

// ksigns_iq2xs[v] == v | ((popcount(v) & 1) << 7), so the sign byte is computed rather than read:
// byte-indexed __constant loads serialize on Adreno.
inline uint iq2xxs_signs(uint code7) {
    return code7 | ((uint)(popcount(code7) & 1u) << 7);
}

// Four grid bytes with their signs applied, as floats. base picks which nibble
// of the sign byte this half of the entry uses: 0 for the low uint, 4 for the
// high one.
inline float4 iq2xxs_vals(uint gv, uint sgv, uint base) {
    const uint  s   = sgv >> base;
    const uint4 sgn = (uint4)(s << 31, s << 30, s << 29, s << 28) & 0x80000000u;
    return as_float4(as_uint4(convert_float4(as_uchar4(gv))) ^ sgn);
}

#if IQ2XXS_MV_AIMG
#define IQ2XXS_YV(g) read_imagef(y_img, (int)(y_tex + (g)))
#else
#define IQ2XXS_YV(g) vload4((g), y)
#endif

// IQ2XXS_MV_AIMG=1: read the activation through an image1d_buffer view of src1 (no copy) instead of
// vload4. The read is wave-uniform and redundant across the workgroup's 64 rows, which the texture cache
// serves well; it pays off for these codebook types, whose per-weight decode is expensive, and not for
// linear quants such as IQ4_XS. Default on for X2-class devices; GGML_OPENCL_IQ2XXS_MV_AIMG overrides.
#ifndef IQ2XXS_MV_AIMG
#define IQ2XXS_MV_AIMG 0
#endif

kernel void kernel_mul_mv_iq2_xxs_f32_flat(
        __read_only image1d_buffer_t grid_img,
#if IQ2XXS_MV_AIMG
        // Declared only when the option is on, so host and kernel argument lists cannot disagree
        // (a mismatch would silently shift the weight planes).
        __read_only image1d_buffer_t y_img,
        uint y_off,
#endif
        global const uchar * src0_qs,
        global const uint  * src0_sas,
        global const half  * src0_d,
        global const float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,      // K
        int ne01,      // M, the number of output rows
        int ne10,      // activation row stride, == K
        int ne0        // dst row stride
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global float       *)((global char       *)dst  + offsetd);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;                  // super blocks along K

    const uint lid = get_local_id(0);           // lane
    const uint sgi = get_local_id(1);           // K-split slice
    const uint col = get_group_id(1);           // token

    global const float * y = src1 + (ulong)col * (uint)ne10;
#if IQ2XXS_MV_AIMG
    // src1 is f32 and the image is RGBA/f32, so one texel is four floats.
    const uint y_tex = y_off + col * ((uint)ne10 >> 2);
#endif

#define IQ2XXS_GRID(i) (read_imageui(grid_img, (int)(i)).x)

    const uint mh  = m >> 1;                        // rows per plane row, as pairs
    const uint j   = get_group_id(0) * 64u + lid;   // row pair index
    const uint row = j << 1;

    float sumf  = 0.f;
    float sumf1 = 0.f;

    if (j < mh) {
        global const ushort * qsu = (global const ushort *)src0_qs;

        for (uint ib = sgi; ib < nsb; ib += IQ2XXS_MV_NSG) {
            const half2 dh = vload2(j + ib * mh, src0_d);
            const float d0 = (float)dh.s0;
            const float d1 = (float)dh.s1;

            float acc0 = 0.f, acc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint  sub = ib * 8u + sb;
                const uint2 aux = vload2(j + sub * mh, src0_sas);   // row pair, one load

                const uint grp = sub * 4u;      // quant plane units of 8 weights
                const uint qsb = j + grp * mh;

                float a0 = 0.f, a1 = 0.f;
                for (uint u = 0; u < 4u; ++u) {
                    const uint qsv = (uint)qsu[qsb + u * mh];       // row pair, one load
                    const uint sg0 = iq2xxs_signs((aux.s0 >> (7u * u)) & 127u);
                    const uint sg1 = iq2xxs_signs((aux.s1 >> (7u * u)) & 127u);

                    const uint g0 = ( qsv       & 0xFFu) << 1;      // uint pair index
                    const uint g1 = ((qsv >> 8) & 0xFFu) << 1;

                    const float4 yl = IQ2XXS_YV(sub * 8u + u * 2u + 0u);
                    const float4 yh = IQ2XXS_YV(sub * 8u + u * 2u + 1u);

                    a0 += dot(yl, iq2xxs_vals(IQ2XXS_GRID(g0    ), sg0, 0u))
                        + dot(yh, iq2xxs_vals(IQ2XXS_GRID(g0 + 1), sg0, 4u));
                    a1 += dot(yl, iq2xxs_vals(IQ2XXS_GRID(g1    ), sg1, 0u))
                        + dot(yh, iq2xxs_vals(IQ2XXS_GRID(g1 + 1), sg1, 4u));
                }
                acc0 += (0.5f + (float)(aux.s0 >> 28)) * a0;
                acc1 += (0.5f + (float)(aux.s1 >> 28)) * a1;
            }
            sumf  += d0 * 0.25f * acc0;
            sumf1 += d1 * 0.25f * acc1;
        }
    }

#if IQ2XXS_MV_NSG > 1
    __local float2 part[IQ2XXS_MV_NSG][64];
    part[sgi][lid] = (float2)(sumf, sumf1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ2XXS_MV_NSG; ++s) {
        const float2 p = part[s][lid];
        sumf  += p.s0;
        sumf1 += p.s1;
    }
#endif

    if (j < mh) {
        vstore2((float2)(sumf, sumf1), 0, dst + (ulong)col * (uint)ne0 + row);
    }
}

kernel void kernel_iq2xxs_grid_export(global uint * out) {
    const uint i = get_global_id(0);
    if (i < 512u) {
        out[i] = iq2xxs_grid[i];
    }
}

// Fused ffn_gate + ffn_up + GLU for decode: one pass over the shared activation accumulates both dot
// products and the GLU runs in registers. The host admits only MUL_MAT/MUL_MAT/GLU with one shared f32
// activation, matching split weights and no swapped operands.

#define IQ2XXS_GLU_GEGLU_COEF_A   0.044715f
#define IQ2XXS_GLU_SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define IQ2XXS_GLU_SQRT_2_INV     0.70710678118654752440084436210484f
#define IQ2XXS_GLU_QUICK_COEF    -1.702f

// Op numbering and expressions match the q4_K fused path.
inline float iq2xxs_glu_apply(int glu_op, float g, float u) {
    float act;
    if (glu_op == 1) {        // GEGLU (tanh-approx gelu)
        act = 0.5f*g*(1.0f + tanh(IQ2XXS_GLU_SQRT_2_OVER_PI*g*(1.0f + IQ2XXS_GLU_GEGLU_COEF_A*g*g)));
    } else if (glu_op == 2) { // SWIGLU (silu)
        act = g / (1.0f + exp(-g));
    } else if (glu_op == 0) { // REGLU
        return g*u*(g > 0.0f);
    } else if (glu_op == 4) { // GEGLU_ERF
        act = 0.5f*g*(1.0f + erf(g*IQ2XXS_GLU_SQRT_2_INV));
    } else {                  // GEGLU_QUICK
        act = g*(1.0f/(1.0f + exp(IQ2XXS_GLU_QUICK_COEF*g)));
    }
    return act*u;
}

kernel void kernel_mul_mv_iq2_xxs_f32_flat_glu(
        __read_only image1d_buffer_t grid_img,
#if IQ2XXS_MV_AIMG
        // declared exactly when the host binds them; see the plain GEMV
        __read_only image1d_buffer_t y_img,
        uint y_off,
#endif
        global const uchar * g_qs,
        global const uint  * g_sas,
        global const half  * g_d,
        global const uchar * u_qs,
        global const uint  * u_sas,
        global const half  * u_d,
        global const float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne10,
        int ne0,
        int glu_op
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global float       *)((global char       *)dst  + offsetd);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;

    const uint lid = get_local_id(0);
    const uint sgi = get_local_id(1);
    const uint col = get_group_id(1);

    global const float * y = src1 + (ulong)col * (uint)ne10;
#if IQ2XXS_MV_AIMG
#if IQ2XXS_MV_AIMG
    const uint y_tex = y_off + col * ((uint)ne10 >> 2);
#endif
#endif

#define IQ2XXS_GGRID(i) (read_imageui(grid_img, (int)(i)).x)

    const uint mh  = m >> 1;
    const uint j   = get_group_id(0) * 64u + lid;
    const uint row = j << 1;

    float gs0 = 0.f, gs1 = 0.f, us0 = 0.f, us1 = 0.f;

    if (j < mh) {
        global const ushort * gqsu = (global const ushort *)g_qs;
        global const ushort * uqsu = (global const ushort *)u_qs;

        for (uint ib = sgi; ib < nsb; ib += IQ2XXS_MV_NSG) {
            const half2 gdh = vload2(j + ib * mh, g_d);
            const half2 udh = vload2(j + ib * mh, u_d);

            float gacc0 = 0.f, gacc1 = 0.f, uacc0 = 0.f, uacc1 = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint  sub  = ib * 8u + sb;
                const uint2 gaux = vload2(j + sub * mh, g_sas);
                const uint2 uaux = vload2(j + sub * mh, u_sas);

                const uint grp = sub * 4u;
                const uint qsb = j + grp * mh;

                float ga0 = 0.f, ga1 = 0.f, ua0 = 0.f, ua1 = 0.f;
                for (uint u = 0; u < 4u; ++u) {
                    // activation read once for both streams and both rows
                    const float4 yl = IQ2XXS_YV(sub * 8u + u * 2u + 0u);
                    const float4 yh = IQ2XXS_YV(sub * 8u + u * 2u + 1u);

                    const uint gqsv = (uint)gqsu[qsb + u * mh];
                    const uint gsg0 = iq2xxs_signs((gaux.s0 >> (7u * u)) & 127u);
                    const uint gsg1 = iq2xxs_signs((gaux.s1 >> (7u * u)) & 127u);
                    const uint gg0  = ( gqsv       & 0xFFu) << 1;
                    const uint gg1  = ((gqsv >> 8) & 0xFFu) << 1;
                    ga0 += dot(yl, iq2xxs_vals(IQ2XXS_GGRID(gg0    ), gsg0, 0u))
                         + dot(yh, iq2xxs_vals(IQ2XXS_GGRID(gg0 + 1), gsg0, 4u));
                    ga1 += dot(yl, iq2xxs_vals(IQ2XXS_GGRID(gg1    ), gsg1, 0u))
                         + dot(yh, iq2xxs_vals(IQ2XXS_GGRID(gg1 + 1), gsg1, 4u));

                    const uint uqsv = (uint)uqsu[qsb + u * mh];
                    const uint usg0 = iq2xxs_signs((uaux.s0 >> (7u * u)) & 127u);
                    const uint usg1 = iq2xxs_signs((uaux.s1 >> (7u * u)) & 127u);
                    const uint ug0  = ( uqsv       & 0xFFu) << 1;
                    const uint ug1  = ((uqsv >> 8) & 0xFFu) << 1;
                    ua0 += dot(yl, iq2xxs_vals(IQ2XXS_GGRID(ug0    ), usg0, 0u))
                         + dot(yh, iq2xxs_vals(IQ2XXS_GGRID(ug0 + 1), usg0, 4u));
                    ua1 += dot(yl, iq2xxs_vals(IQ2XXS_GGRID(ug1    ), usg1, 0u))
                         + dot(yh, iq2xxs_vals(IQ2XXS_GGRID(ug1 + 1), usg1, 4u));
                }
                gacc0 += (0.5f + (float)(gaux.s0 >> 28)) * ga0;
                gacc1 += (0.5f + (float)(gaux.s1 >> 28)) * ga1;
                uacc0 += (0.5f + (float)(uaux.s0 >> 28)) * ua0;
                uacc1 += (0.5f + (float)(uaux.s1 >> 28)) * ua1;
            }
            gs0 += (float)gdh.s0 * 0.25f * gacc0;
            gs1 += (float)gdh.s1 * 0.25f * gacc1;
            us0 += (float)udh.s0 * 0.25f * uacc0;
            us1 += (float)udh.s1 * 0.25f * uacc1;
        }
    }

#if IQ2XXS_MV_NSG > 1
    __local float4 gpart[IQ2XXS_MV_NSG][64];
    gpart[sgi][lid] = (float4)(gs0, gs1, us0, us1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgi != 0) {
        return;
    }
    for (uint s = 1; s < IQ2XXS_MV_NSG; ++s) {
        const float4 p = gpart[s][lid];
        gs0 += p.s0; gs1 += p.s1; us0 += p.s2; us1 += p.s3;
    }
#endif

    if (j < mh) {
        global float * o = dst + (ulong)col * (uint)ne0 + row;
        o[0] = iq2xxs_glu_apply(glu_op, gs0, us0);
        o[1] = iq2xxs_glu_apply(glu_op, gs1, us1);
    }
#undef IQ2XXS_GGRID
}
