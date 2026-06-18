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

#if !defined(DATA_A_TURBO3_0)
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
#endif  // !DATA_A_TURBO3_0

layout (binding = 1) readonly buffer K_PACKED_BF16 { u16vec4 data[]; } k_packed_bf16;
layout (binding = 2) readonly buffer V_PACKED_BF16 { u16vec4 data[]; } v_packed_bf16;

// TurboQuant K/V views. Explicit std430 + restrict because the turbo blocks have
// a different (larger) stride than the q4/q5/q8 views aliased at the same binding
// (turbo3 is a 50-byte block), so the driver must not assume a uniform stride.
// Graph applies forward WHT to Q pre-attention and inverse WHT to FA output, so
// dequant just returns centroid * norm.
layout (binding = 1, std430) restrict readonly buffer K_PACKED_TURBO2_0 { block_turbo2_0 data[]; } k_packed_turbo2_0;
layout (binding = 2, std430) restrict readonly buffer V_PACKED_TURBO2_0 { block_turbo2_0 data[]; } v_packed_turbo2_0;
layout (binding = 1, std430) restrict readonly buffer K_PACKED_TURBO3_0 { block_turbo3_0 data[]; } k_packed_turbo3_0;
layout (binding = 2, std430) restrict readonly buffer V_PACKED_TURBO3_0 { block_turbo3_0 data[]; } v_packed_turbo3_0;
layout (binding = 1, std430) restrict readonly buffer K_PACKED_TURBO4_0 { block_turbo4_0 data[]; } k_packed_turbo4_0;
layout (binding = 2, std430) restrict readonly buffer V_PACKED_TURBO4_0 { block_turbo4_0 data[]; } v_packed_turbo4_0;

// Q4_1 and Q5_1 packed32 views: aliased to the same memory as the packed16
// views, used by the MMQ K-side hot path for fast 4-uint loads.
layout (binding = 1) readonly buffer K_PACKED_Q4_1_P32 { block_q4_1_packed32 data[]; } k_packed_q4_1_p32;
layout (binding = 1) readonly buffer K_PACKED_Q5_1_P32 { block_q5_1_packed32 data[]; } k_packed_q5_1_p32;

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

// TurboQuant2 dequant: 2-bit centroids, no signs byte. All 4 elements share
// qs byte (iqs/4) at iqs%4==0.
#define FA_DEQUANT4_TURBO2_0(BUF) {                                                               \
    const float c[4] = float[4](-0.133462, -0.039994, 0.039994, 0.133462);                        \
    const float norm = float(BUF.data[a_offset + ib].norm);                                       \
    const uint qs_byte = uint(BUF.data[a_offset + ib].qs[iqs / 4]);                               \
    const uint i0 = (qs_byte     ) & 0x3u;                                                         \
    const uint i1 = (qs_byte >> 2) & 0x3u;                                                         \
    const uint i2 = (qs_byte >> 4) & 0x3u;                                                         \
    const uint i3 = (qs_byte >> 6) & 0x3u;                                                         \
    return FLOAT_TYPE(norm) * FLOAT_TYPEV4(c[i0], c[i1], c[i2], c[i3]);                           \
}

// TurboQuant3 dequant: per-element centroid lookup. iqs is in [0, 128) at vec4
// alignment (iqs%4 == 0), so all 4 elements share qs byte (iqs/4) and signs
// byte (iqs/8). No iWHT here, the graph handles rotation outside FA.
#define FA_DEQUANT4_TURBO3_0(BUF) {                                                               \
    const float c[8] = float[8](                                                                  \
        -0.190685, -0.117832, -0.065717, -0.021460,                                               \
         0.021460,  0.065717,  0.117832,  0.190685);                                              \
    const float norm = float(BUF.data[a_offset + ib].norm);                                       \
    const uint qs_byte  = uint(BUF.data[a_offset + ib].qs[iqs / 4]);                              \
    const uint sgn_byte = uint(BUF.data[a_offset + ib].signs[iqs / 8]);                           \
    const uint base = iqs & 0x7u;                                                                 \
    const uint i0 = ((qs_byte     ) & 0x3) | (((sgn_byte >> (base    )) & 0x1u) << 2);            \
    const uint i1 = ((qs_byte >> 2) & 0x3) | (((sgn_byte >> (base + 1)) & 0x1u) << 2);            \
    const uint i2 = ((qs_byte >> 4) & 0x3) | (((sgn_byte >> (base + 2)) & 0x1u) << 2);            \
    const uint i3 = ((qs_byte >> 6) & 0x3) | (((sgn_byte >> (base + 3)) & 0x1u) << 2);            \
    return FLOAT_TYPE(norm) * FLOAT_TYPEV4(c[i0], c[i1], c[i2], c[i3]);                           \
}

// TurboQuant4 dequant: 4-bit indices, 2 per byte. iqs%4==0 means the 4
// elements span 2 consecutive qs bytes (each holds 2 nibbles).
#define FA_DEQUANT4_TURBO4_0(BUF) {                                                               \
    const float c[16] = float[16](                                                                \
        -0.173926, -0.117195, -0.089527, -0.068756,                                               \
        -0.051262, -0.035597, -0.020989, -0.006938,                                               \
         0.006938,  0.020989,  0.035597,  0.051262,                                               \
         0.068756,  0.089527,  0.117195,  0.173926);                                              \
    const float norm = float(BUF.data[a_offset + ib].norm);                                       \
    const uint b0 = uint(BUF.data[a_offset + ib].qs[iqs / 2    ]);                                \
    const uint b1 = uint(BUF.data[a_offset + ib].qs[iqs / 2 + 1]);                                \
    const uint i0 = (b0     ) & 0xFu;                                                              \
    const uint i1 = (b0 >> 4) & 0xFu;                                                              \
    const uint i2 = (b1     ) & 0xFu;                                                              \
    const uint i3 = (b1 >> 4) & 0xFu;                                                              \
    return FLOAT_TYPE(norm) * FLOAT_TYPEV4(c[i0], c[i1], c[i2], c[i3]);                           \
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
            case FA_TYPE_TURBO2_0: FA_DEQUANT4_TURBO2_0(k_packed_turbo2_0)
            case FA_TYPE_TURBO3_0: FA_DEQUANT4_TURBO3_0(k_packed_turbo3_0)
            case FA_TYPE_TURBO4_0: FA_DEQUANT4_TURBO4_0(k_packed_turbo4_0)
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
            case FA_TYPE_TURBO2_0: FA_DEQUANT4_TURBO2_0(v_packed_turbo2_0)
            case FA_TYPE_TURBO3_0: FA_DEQUANT4_TURBO3_0(v_packed_turbo3_0)
            case FA_TYPE_TURBO4_0: FA_DEQUANT4_TURBO4_0(v_packed_turbo4_0)
        }
    }
    return FLOAT_TYPEV4(0);
}
