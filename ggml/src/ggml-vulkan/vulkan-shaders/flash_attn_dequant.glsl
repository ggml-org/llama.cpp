// Asymmetric K/V flash attention: aliased SSBO views of bindings 1 (K) and 2 (V)
// covering every supported FA element type, plus an uber dequantize4() that
// switches on FaTypeK / FaTypeV. After spec-constant specialization the driver
// folds away every path except the one matching the K/V type for this pipeline.
//
// Included by flash_attn.comp and flash_attn_cm1.comp. Not included by
// flash_attn_cm2.comp, which has its own buffer_reference-based decode path.
//
// We use macros (rather than per-quant decode functions taking a struct) on
// purpose: the FA shaders don't enable GL_EXT_shader_explicit_arithmetic_types_float16
// when FLOAT16 isn't defined, which makes float16-containing struct values
// illegal to return from / pass to functions. Macros expand inline where the
// float16 stays in storage and is converted to FLOAT_TYPE at use.

// F32 is fed as a vec4 "block" (4 floats), matching what dequant_funcs_cm2.glsl
// does for F32 in the cm2 shader. FaBlockBytesK/V == 16 for F32.
layout (binding = 1) readonly buffer K_PACKED_F32  { vec4 data[]; }                k_packed_f32;
layout (binding = 2) readonly buffer V_PACKED_F32  { vec4 data[]; }                v_packed_f32;

layout (binding = 1) readonly buffer K_PACKED_Q4_0 { block_q4_0_packed16 data[]; } k_packed_q4_0;
layout (binding = 2) readonly buffer V_PACKED_Q4_0 { block_q4_0_packed16 data[]; } v_packed_q4_0;
layout (binding = 1) readonly buffer K_PACKED_Q4_1 { block_q4_1_packed16 data[]; } k_packed_q4_1;
layout (binding = 2) readonly buffer V_PACKED_Q4_1 { block_q4_1_packed16 data[]; } v_packed_q4_1;
layout (binding = 1) readonly buffer K_PACKED_Q5_0 { block_q5_0_packed16 data[]; } k_packed_q5_0;
layout (binding = 2) readonly buffer V_PACKED_Q5_0 { block_q5_0_packed16 data[]; } v_packed_q5_0;
layout (binding = 1) readonly buffer K_PACKED_Q5_1 { block_q5_1_packed16 data[]; } k_packed_q5_1;
layout (binding = 2) readonly buffer V_PACKED_Q5_1 { block_q5_1_packed16 data[]; } v_packed_q5_1;
layout (binding = 1) readonly buffer K_PACKED_Q8_0 { block_q8_0_packed16 data[]; } k_packed_q8_0;
layout (binding = 2) readonly buffer V_PACKED_Q8_0 { block_q8_0_packed16 data[]; } v_packed_q8_0;
layout (binding = 1) readonly buffer K_PACKED_ROCMFP4 { block_rocmfp4 data[]; } k_packed_rocmfp4;
layout (binding = 2) readonly buffer V_PACKED_ROCMFP4 { block_rocmfp4 data[]; } v_packed_rocmfp4;
layout (binding = 1) readonly buffer K_PACKED_ROCMFP4_FAST { block_rocmfp4_fast data[]; } k_packed_rocmfp4_fast;
layout (binding = 2) readonly buffer V_PACKED_ROCMFP4_FAST { block_rocmfp4_fast data[]; } v_packed_rocmfp4_fast;

layout (binding = 1) readonly buffer K_PACKED_BF16 { u16vec4 data[]; } k_packed_bf16;
layout (binding = 2) readonly buffer V_PACKED_BF16 { u16vec4 data[]; } v_packed_bf16;

// Q4_1 and Q5_1 packed32 views: aliased to the same memory as the packed16
// views, used by the MMQ K-side hot path for fast 4-uint loads.
layout (binding = 1) readonly buffer K_PACKED_Q4_1_P32 { block_q4_1_packed32 data[]; } k_packed_q4_1_p32;
layout (binding = 1) readonly buffer K_PACKED_Q5_1_P32 { block_q5_1_packed32 data[]; } k_packed_q5_1_p32;

int8_t fa_rocmfp4_code_i8(uint q) {
    switch (q & 0xFu) {
        case  0u: return int8_t(  0);
        case  1u: return int8_t(  1);
        case  2u: return int8_t(  2);
        case  3u: return int8_t(  3);
        case  4u: return int8_t(  4);
        case  5u: return int8_t(  6);
        case  6u: return int8_t(  8);
        case  7u: return int8_t( 10);
        case  8u: return int8_t(  0);
        case  9u: return int8_t( -1);
        case 10u: return int8_t( -2);
        case 11u: return int8_t( -3);
        case 12u: return int8_t( -4);
        case 13u: return int8_t( -6);
        case 14u: return int8_t( -8);
        default:  return int8_t(-10);
    }
}

FLOAT_TYPE fa_rocmfp4_code_value(uint q) {
    return FLOAT_TYPE(fa_rocmfp4_code_i8(q));
}

int32_t fa_rocmfp4_pack4_i8(uint vui) {
    return pack32(i8vec4(fa_rocmfp4_code_i8( vui        & 0xFu),
                         fa_rocmfp4_code_i8((vui >>  8) & 0xFu),
                         fa_rocmfp4_code_i8((vui >> 16) & 0xFu),
                         fa_rocmfp4_code_i8((vui >> 24) & 0xFu)));
}

FLOAT_TYPE fa_rocmfp4_ue4m3_to_fp_half(uint8_t x) {
    const uint u = uint(x);
    if (u == 0u || u == 127u || u == 255u) {
        return FLOAT_TYPE(0.0);
    }

    const uint exp = (u >> 3) & 15u;
    const uint man = u & 7u;
    if (exp == 0u) {
        return FLOAT_TYPE(float(man) * (1.0 / 1024.0));
    }

    const uint bits = (exp + 119u) << 23 | (man << 20);
    return FLOAT_TYPE(uintBitsToFloat(bits));
}

// Per-quant decode bodies are expanded once for the K view set and once for
// the V view set. The macros take the buffer name as a parameter.
#define FA_DEQUANT4_F32(BUF) \
    return FLOAT_TYPEV4(BUF.data[a_offset + ib]);

#define FA_DEQUANT4_Q4_0(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles - FLOAT_TYPE(8.0f));                  \
}

#define FA_DEQUANT4_Q4_1(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * nibbles                                        \
         + FLOAT_TYPE(BUF.data[a_offset + ib].m);                                                 \
}

#define FA_DEQUANT4_Q5_0(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    uint qh = uint(BUF.data[a_offset + ib].qh[0])                                                 \
            | (uint(BUF.data[a_offset + ib].qh[1]) << 16);                                        \
    FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs)       & 1, (qh >> (iqs + 1)) & 1,                  \
                                   (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1)                  \
                      * FLOAT_TYPE(16.0f);                                                        \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles + hb - FLOAT_TYPE(16.0f));            \
}

#define FA_DEQUANT4_Q5_1(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    uint qh = BUF.data[a_offset + ib].qh;                                                         \
    FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs)       & 1, (qh >> (iqs + 1)) & 1,                  \
                                   (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1)                  \
                      * FLOAT_TYPE(16.0f);                                                        \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles + hb)                                 \
         + FLOAT_TYPE(BUF.data[a_offset + ib].m);                                                 \
}

#define FA_DEQUANT4_Q8_0(BUF) {                                                                   \
    const i8vec2 v0 = unpack8(int32_t(BUF.data[a_offset + ib].qs[iqs / 2    ])).xy;               \
    const i8vec2 v1 = unpack8(int32_t(BUF.data[a_offset + ib].qs[iqs / 2 + 1])).xy;               \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);          \
}

#define FA_DEQUANT4_BF16(BUF) \
    return FLOAT_TYPEV4(bf16_to_fp32(uvec4(BUF.data[(a_offset + ib) / 4])));

#define FA_DEQUANT4_ROCMFP4(BUF) {                                                                \
    const uint qbase = iqs & 0xFu;                                                                \
    uint vui = pack32(u8vec4(BUF.data[a_offset + ib].qs[qbase + 0u],                              \
                             BUF.data[a_offset + ib].qs[qbase + 1u],                              \
                             BUF.data[a_offset + ib].qs[qbase + 2u],                              \
                             BUF.data[a_offset + ib].qs[qbase + 3u]));                            \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui >>= shift;                                                                                \
    const uint half_idx = (iqs & 0x10) != 0 ? 1u : 0u;                                            \
    const FLOAT_TYPE d = fa_rocmfp4_ue4m3_to_fp_half(BUF.data[a_offset + ib].e[half_idx]);        \
    return d * FLOAT_TYPEV4(fa_rocmfp4_code_value( vui        & 0xF),                             \
                            fa_rocmfp4_code_value((vui >>  8) & 0xF),                             \
                            fa_rocmfp4_code_value((vui >> 16) & 0xF),                             \
                            fa_rocmfp4_code_value((vui >> 24) & 0xF));                            \
}

#define FA_DEQUANT4_ROCMFP4_FAST(BUF) {                                                           \
    const uint qbase = iqs & 0xFu;                                                                \
    uint vui = pack32(u8vec4(BUF.data[a_offset + ib].qs[qbase + 0u],                              \
                             BUF.data[a_offset + ib].qs[qbase + 1u],                              \
                             BUF.data[a_offset + ib].qs[qbase + 2u],                              \
                             BUF.data[a_offset + ib].qs[qbase + 3u]));                            \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui >>= shift;                                                                                \
    const FLOAT_TYPE d = fa_rocmfp4_ue4m3_to_fp_half(BUF.data[a_offset + ib].e);                  \
    return d * FLOAT_TYPEV4(fa_rocmfp4_code_value( vui        & 0xF),                             \
                            fa_rocmfp4_code_value((vui >>  8) & 0xF),                             \
                            fa_rocmfp4_code_value((vui >> 16) & 0xF),                             \
                            fa_rocmfp4_code_value((vui >> 24) & 0xF));                            \
}

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        switch (FaTypeK) {
            case FA_TYPE_F32:  FA_DEQUANT4_F32 (k_packed_f32)
            case FA_TYPE_Q4_0: FA_DEQUANT4_Q4_0(k_packed_q4_0)
            case FA_TYPE_Q4_1: FA_DEQUANT4_Q4_1(k_packed_q4_1)
            case FA_TYPE_Q5_0: FA_DEQUANT4_Q5_0(k_packed_q5_0)
            case FA_TYPE_Q5_1: FA_DEQUANT4_Q5_1(k_packed_q5_1)
            case FA_TYPE_Q8_0: FA_DEQUANT4_Q8_0(k_packed_q8_0)
            case FA_TYPE_BF16: FA_DEQUANT4_BF16(k_packed_bf16)
            case FA_TYPE_BF16: FA_DEQUANT4_BF16(k_packed_bf16)
            case FA_TYPE_Q4_0_ROCMFP4:      FA_DEQUANT4_ROCMFP4(k_packed_rocmfp4)
            case FA_TYPE_Q4_0_ROCMFP4_FAST: FA_DEQUANT4_ROCMFP4_FAST(k_packed_rocmfp4_fast)
        }
    } else {
        switch (FaTypeV) {
            case FA_TYPE_F32:  FA_DEQUANT4_F32 (v_packed_f32)
            case FA_TYPE_Q4_0: FA_DEQUANT4_Q4_0(v_packed_q4_0)
            case FA_TYPE_Q4_1: FA_DEQUANT4_Q4_1(v_packed_q4_1)
            case FA_TYPE_Q5_0: FA_DEQUANT4_Q5_0(v_packed_q5_0)
            case FA_TYPE_Q5_1: FA_DEQUANT4_Q5_1(v_packed_q5_1)
            case FA_TYPE_Q8_0: FA_DEQUANT4_Q8_0(v_packed_q8_0)
            case FA_TYPE_BF16: FA_DEQUANT4_BF16(v_packed_bf16)
            case FA_TYPE_BF16: FA_DEQUANT4_BF16(v_packed_bf16)
            case FA_TYPE_Q4_0_ROCMFP4:      FA_DEQUANT4_ROCMFP4(v_packed_rocmfp4)
            case FA_TYPE_Q4_0_ROCMFP4_FAST: FA_DEQUANT4_ROCMFP4_FAST(v_packed_rocmfp4_fast)
        }
    }
    return FLOAT_TYPEV4(0);
}
