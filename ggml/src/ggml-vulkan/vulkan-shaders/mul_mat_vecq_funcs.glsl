#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include "types.glsl"

// =====================================================================
// Shared memory staging: byte-level read helpers for raw u64 shmem
// =====================================================================
#if defined(USE_SHMEM_STAGING)

int32_t shmem_read_i32(uint byte_off) {
    uint w = byte_off >> 3u;
    uint shift = (byte_off & 7u) << 3u;
    if (shift <= 32u)
        return int32_t(shmem_raw[w] >> shift);
    else
        return int32_t((shmem_raw[w] >> shift) | (shmem_raw[w + 1u] << (64u - shift)));
}

float shmem_read_f16(uint byte_off) {
    uint w = byte_off >> 3u;
    uint shift = (byte_off & 7u) << 3u;
    uint raw;
    if (shift <= 48u)
        raw = uint(shmem_raw[w] >> shift);
    else
        raw = uint((shmem_raw[w] >> shift) | (shmem_raw[w + 1u] << (64u - shift)));
    return unpackHalf2x16(raw & 0xFFFFu).x;
}

uint shmem_read_u8(uint byte_off) {
    uint w = byte_off >> 3u;
    uint shift = (byte_off & 7u) << 3u;
    return uint(shmem_raw[w] >> shift) & 0xFFu;
}

uint shmem_read_u16(uint byte_off) {
    uint w = byte_off >> 3u;
    uint shift = (byte_off & 7u) << 3u;
    uint raw;
    if (shift <= 48u)
        raw = uint(shmem_raw[w] >> shift);
    else
        raw = uint((shmem_raw[w] >> shift) | (shmem_raw[w + 1u] << (64u - shift)));
    return raw & 0xFFFFu;
}

#endif // USE_SHMEM_STAGING

// =====================================================================
// Per-type get_dm / repack / mul_q8_1 functions
// =====================================================================

#if defined(DATA_A_Q4_0) || defined(DATA_A_Q5_0) || defined(DATA_A_Q8_0) || defined(DATA_A_IQ1_S) || defined(DATA_A_IQ2_XXS) || defined(DATA_A_IQ2_XS) || defined(DATA_A_IQ2_S) || defined(DATA_A_IQ3_XXS) || defined(DATA_A_IQ3_S) || defined(DATA_A_IQ4_XS) || defined(DATA_A_IQ4_NL)
FLOAT_TYPE get_dm(uint ib) {
#if !defined(USE_SHMEM_STAGING)
    return FLOAT_TYPE(data_a[ib].d);
#else
    return FLOAT_TYPE(shmem_read_f16(ib));
#endif
}
#endif

#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1)
FLOAT_TYPE_VEC2 get_dm(uint ib) {
#if !defined(USE_SHMEM_STAGING)
    return FLOAT_TYPE_VEC2(data_a_packed32[ib].dm);
#else
    return FLOAT_TYPE_VEC2(unpackHalf2x16(uint(shmem_read_i32(ib))));
#endif
}
#endif

#if defined(DATA_A_MXFP4)
FLOAT_TYPE get_dm(uint ib) {
#if !defined(USE_SHMEM_STAGING)
    return FLOAT_TYPE(e8m0_to_fp32(data_a[ib].e));
#else
    return FLOAT_TYPE(e8m0_to_fp32(uint8_t(shmem_read_u8(ib))));
#endif
}
#endif

#if defined(DATA_A_Q2_K)
// Q2_K (84B): scales@0(16B), qs@16(64B), dm@80(4B)
FLOAT_TYPE_VEC2 get_dm(uint ib) {
#if !defined(USE_SHMEM_STAGING)
    const uint ib_k = ib / 8;
    return FLOAT_TYPE_VEC2(data_a_packed32[ib_k].dm);
#else
    return FLOAT_TYPE_VEC2(unpackHalf2x16(uint(shmem_read_i32(ib + 80))));
#endif
}
#endif

// Each iqs value maps to a 32-bit integer
#if defined(DATA_A_Q4_0)
// Q4_0 (18B): d@0(2B), qs@2(16B)
i32vec2 repack(uint ib, uint iqs) {
#if !defined(USE_SHMEM_STAGING)
    const u16vec2 quants = u16vec2(data_a_packed16[ib].qs[iqs * 2    ],
                                   data_a_packed16[ib].qs[iqs * 2 + 1]);
    const uint32_t vui = pack32(quants);
#else
    const uint32_t vui = uint32_t(shmem_read_i32(ib + 2 + iqs * 4));
#endif
    return i32vec2( vui       & 0x0F0F0F0F,
                   (vui >> 4) & 0x0F0F0F0F);
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(da * (float(q_sum) * dsb.x - (8 / sum_divisor) * dsb.y));
}
#endif

#if defined(DATA_A_Q4_1)
// Q4_1 (20B): dm@0(4B), qs@4(16B)
i32vec2 repack(uint ib, uint iqs) {
#if !defined(USE_SHMEM_STAGING)
    const uint32_t vui = data_a_packed32[ib].qs[iqs];
#else
    const uint32_t vui = uint32_t(shmem_read_i32(ib + 4 + iqs * 4));
#endif
    return i32vec2( vui       & 0x0F0F0F0F,
                   (vui >> 4) & 0x0F0F0F0F);
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const vec2 dma, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(float(q_sum) * dma.x * dsb.x + dma.y * dsb.y / sum_divisor);
}
#endif

#if defined(DATA_A_Q5_0)
// Q5_0 (22B): d@0(2B), qh@2(4B), qs@6(16B)
i32vec2 repack(uint ib, uint iqs) {
#if !defined(USE_SHMEM_STAGING)
    const u16vec2 quants = u16vec2(data_a_packed16[ib].qs[iqs * 2    ],
                                   data_a_packed16[ib].qs[iqs * 2 + 1]);
    const uint32_t vui = pack32(quants);
    const int32_t qh = int32_t((uint32_t(data_a_packed16[ib].qh[1]) << 16 | data_a_packed16[ib].qh[0]) >> (4 * iqs));
#else
    const uint32_t vui = uint32_t(shmem_read_i32(ib + 6 + iqs * 4));
    const int32_t qh = shmem_read_i32(ib + 2) >> (4 * int(iqs));
#endif
    const int32_t v0 = int32_t(vui & 0x0F0F0F0F)
                     | ((qh & 0xF) * 0x02040810) & 0x10101010; // (0,1,2,3) -> (4,12,20,28)

    const int32_t v1 = int32_t((vui >> 4) & 0x0F0F0F0F)
                     | (((qh >> 16) & 0xF) * 0x02040810) & 0x10101010; // (16,17,18,19) -> (4,12,20,28)

    return i32vec2(v0, v1);
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(da * (float(q_sum) * dsb.x - (16 / sum_divisor) * dsb.y));
}
#endif

#if defined(DATA_A_Q5_1)
// Q5_1 (24B): dm@0(4B), qh@4(4B), qs@8(16B)
i32vec2 repack(uint ib, uint iqs) {
#if !defined(USE_SHMEM_STAGING)
    const u16vec2 quants = u16vec2(data_a_packed16[ib].qs[iqs * 2    ],
                                   data_a_packed16[ib].qs[iqs * 2 + 1]);
    const uint32_t vui = pack32(quants);
    const int32_t qh = int32_t(data_a_packed32[ib].qh >> (4 * iqs));
#else
    const uint32_t vui = uint32_t(shmem_read_i32(ib + 8 + iqs * 4));
    const int32_t qh = shmem_read_i32(ib + 4) >> (4 * int(iqs));
#endif
    const int32_t v0 = int32_t(vui & 0x0F0F0F0F)
                     | ((qh & 0xF) * 0x02040810) & 0x10101010; // (0,1,2,3) -> (4,12,20,28)

    const int32_t v1 = int32_t((vui >> 4) & 0x0F0F0F0F)
                     | (((qh >> 16) & 0xF) * 0x02040810) & 0x10101010; // (16,17,18,19) -> (4,12,20,28)

    return i32vec2(v0, v1);
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const vec2 dma, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(float(q_sum) * dma.x * dsb.x + dma.y * dsb.y / sum_divisor);
}
#endif

#if defined(DATA_A_Q8_0)
// Q8_0 (34B): d@0(2B), qs@2(32B)
int32_t repack(uint ib, uint iqs) {
#if !defined(USE_SHMEM_STAGING)
    return pack32(i16vec2(data_a_packed16[ib].qs[iqs * 2    ],
                          data_a_packed16[ib].qs[iqs * 2 + 1]));
#else
    return shmem_read_i32(ib + 2 + iqs * 4);
#endif
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(float(q_sum) * da * dsb.x);
}
#endif

#if defined(DATA_A_MXFP4)
// MXFP4 (17B): e@0(1B), qs@1(16B)
i32vec2 repack(uint ib, uint iqs) {
#if !defined(USE_SHMEM_STAGING)
    const uint32_t qs = pack32(u8vec4(data_a[ib].qs[iqs * 4    ],
                                      data_a[ib].qs[iqs * 4 + 1],
                                      data_a[ib].qs[iqs * 4 + 2],
                                      data_a[ib].qs[iqs * 4 + 3]));
#else
    const uint32_t qs = uint32_t(shmem_read_i32(ib + 1 + iqs * 4));
#endif

    const u8vec4 i_a0 = unpack8( qs       & 0x0F0F0F0F);
    const u8vec4 i_a1 = unpack8((qs >> 4) & 0x0F0F0F0F);

    return i32vec2(pack32(i8vec4(kvalues_mxfp4[i_a0.x], kvalues_mxfp4[i_a0.y], kvalues_mxfp4[i_a0.z], kvalues_mxfp4[i_a0.w])),
                   pack32(i8vec4(kvalues_mxfp4[i_a1.x], kvalues_mxfp4[i_a1.y], kvalues_mxfp4[i_a1.z], kvalues_mxfp4[i_a1.w])));
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(da * dsb.x * float(q_sum) * 0.5);
}
#endif

#if defined(DATA_A_QUANT_LEGACY) || defined(DATA_A_MXFP4)
FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t q_sum = 0;
#if QUANT_R == 2
    const i32vec2 data_a_qs = repack(ib_a, iqs);
    q_sum += dotPacked4x8EXT(data_a_qs.x,
                             cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(data_a_qs.y,
                             cache_b_qs[1]);
#else
    int32_t data_a_qs = repack(ib_a, iqs * 2);
    q_sum += dotPacked4x8EXT(data_a_qs,
                             cache_b_qs[0]);
    data_a_qs = repack(ib_a, iqs * 2 + 1);
    q_sum += dotPacked4x8EXT(data_a_qs,
                             cache_b_qs[1]);
#endif

    // 2 quants per call => divide sums by 8/2 = 4
    return mul_q8_1(q_sum, get_dm(ib_a), cache_b_ds, 4);
}
#endif

#if defined(DATA_A_Q2_K)
// 4-byte loads for Q2_K blocks (84 bytes)
// Q2_K: scales@0(16B), qs@16(64B), dm@80(4B)
i32vec4 repack4(uint ib, uint iqs) {
#if !defined(USE_SHMEM_STAGING)
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;

    return i32vec4((data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x03030303,
                   (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x03030303,
                   (data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x03030303,
                   (data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x03030303);
#else
    const uint qs_idx = (iqs / 32) * 8 + (iqs % 8);
    const uint qs_shift = ((iqs % 32) / 8) * 2;

    return i32vec4((uint32_t(shmem_read_i32(ib + 16 + (qs_idx    ) * 4)) >> qs_shift) & 0x03030303,
                   (uint32_t(shmem_read_i32(ib + 16 + (qs_idx + 1) * 4)) >> qs_shift) & 0x03030303,
                   (uint32_t(shmem_read_i32(ib + 16 + (qs_idx + 2) * 4)) >> qs_shift) & 0x03030303,
                   (uint32_t(shmem_read_i32(ib + 16 + (qs_idx + 3) * 4)) >> qs_shift) & 0x03030303);
#endif
}

uint8_t get_scale(uint ib, uint iqs) {
#if !defined(USE_SHMEM_STAGING)
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    return data_a[ib_k].scales[iqs_k / 4];
#else
    return uint8_t(shmem_read_u8(ib + iqs / 4));
#endif
}

FLOAT_TYPE mmvq_dot_product(uint ib_a, uint iqs) {
    int32_t sum_d = 0;
    int32_t sum_m = 0;

#if !defined(USE_SHMEM_STAGING)
    const uint iqs_adj = iqs * 4;
#else
    const uint iqs_adj = iqs;
#endif

    const i32vec4 qs_a = repack4(ib_a, iqs_adj);
    const uint8_t scale = get_scale(ib_a, iqs_adj);
    const vec2 dm = vec2(get_dm(ib_a));
    const int32_t scale_m = int32_t(scale >> 4) * 0x01010101; // Duplicate 8-bit value across 32-bits.

    sum_d += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]) * (scale & 0xF);
    sum_m += dotPacked4x8EXT(scale_m, cache_b_qs[0]);

    sum_d += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]) * (scale & 0xF);
    sum_m += dotPacked4x8EXT(scale_m, cache_b_qs[1]);

    sum_d += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]) * (scale & 0xF);
    sum_m += dotPacked4x8EXT(scale_m, cache_b_qs[2]);

    sum_d += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]) * (scale & 0xF);
    sum_m += dotPacked4x8EXT(scale_m, cache_b_qs[3]);

    return FLOAT_TYPE(float(cache_b_ds.x) * (float(dm.x) * float(sum_d) - float(dm.y) * float(sum_m)));
}
#endif

#if defined(DATA_A_Q3_K)
// 2-byte loads for Q3_K blocks (110 bytes)
// Q3_K: hmask@0(32B), qs@32(64B), scales@96(12B), d@108(2B)
i32vec4 repack4(uint ib, uint iqs) {
#if !defined(USE_SHMEM_STAGING)
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;
    const uint hm_shift = iqs_k / 8;

    // bitwise OR to add 4 if hmask is set, subtract later
    const i8vec2 vals00 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2    ] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2    ] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals01 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 1] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 1] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals10 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 2] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 2] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals11 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 3] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 3] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals20 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 4] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 4] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals21 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 5] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 5] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals30 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 6] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 6] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals31 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 7] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 7] >> hm_shift) & uint16_t(0x0101)) << 2));
#else
    const uint qs_idx = (iqs / 32) * 8 + (iqs % 8);
    const uint qs_shift = ((iqs % 32) / 8) * 2;
    const uint hm_shift = iqs / 8;
    const uint hm_iqs = iqs % 8;

    // hmask: uint16[16] at byte 0, qs: uint16[32] at byte 32
    const i8vec2 vals00 = unpack8(int16_t((uint16_t(shmem_read_u16(ib + 32 + qs_idx * 4        )) >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((uint16_t(shmem_read_u16(ib + hm_iqs * 4               )) >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals01 = unpack8(int16_t((uint16_t(shmem_read_u16(ib + 32 + qs_idx * 4 + 2    )) >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((uint16_t(shmem_read_u16(ib + hm_iqs * 4 + 2           )) >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals10 = unpack8(int16_t((uint16_t(shmem_read_u16(ib + 32 + qs_idx * 4 + 4    )) >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((uint16_t(shmem_read_u16(ib + hm_iqs * 4 + 4           )) >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals11 = unpack8(int16_t((uint16_t(shmem_read_u16(ib + 32 + qs_idx * 4 + 6    )) >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((uint16_t(shmem_read_u16(ib + hm_iqs * 4 + 6           )) >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals20 = unpack8(int16_t((uint16_t(shmem_read_u16(ib + 32 + qs_idx * 4 + 8    )) >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((uint16_t(shmem_read_u16(ib + hm_iqs * 4 + 8           )) >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals21 = unpack8(int16_t((uint16_t(shmem_read_u16(ib + 32 + qs_idx * 4 + 10   )) >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((uint16_t(shmem_read_u16(ib + hm_iqs * 4 + 10          )) >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals30 = unpack8(int16_t((uint16_t(shmem_read_u16(ib + 32 + qs_idx * 4 + 12   )) >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((uint16_t(shmem_read_u16(ib + hm_iqs * 4 + 12          )) >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals31 = unpack8(int16_t((uint16_t(shmem_read_u16(ib + 32 + qs_idx * 4 + 14   )) >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((uint16_t(shmem_read_u16(ib + hm_iqs * 4 + 14          )) >> hm_shift) & uint16_t(0x0101)) << 2));
#endif

    return i32vec4(pack32(i8vec4(vals00.x, vals00.y, vals01.x, vals01.y) - int8_t(4)),
                   pack32(i8vec4(vals10.x, vals10.y, vals11.x, vals11.y) - int8_t(4)),
                   pack32(i8vec4(vals20.x, vals20.y, vals21.x, vals21.y) - int8_t(4)),
                   pack32(i8vec4(vals30.x, vals30.y, vals31.x, vals31.y) - int8_t(4)));
}

float get_d_scale(uint ib, uint iqs) {
#if !defined(USE_SHMEM_STAGING)
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint is = iqs_k / 4;

    const int8_t scale = int8_t(((data_a[ib_k].scales[is % 8      ] >> (4 * (is / 8))) & 0x0F0F) |
                               (((data_a[ib_k].scales[8 + (is % 4)] >> (2 * (is / 4))) & 0x0303) << 4));
    return float(data_a[ib_k].d) * float(scale - 32);
#else
    const uint is = iqs / 4;
    uint s0 = shmem_read_u8(ib + 96 + (is % 8));
    uint s1 = shmem_read_u8(ib + 96 + 8 + (is % 4));
    const int8_t scale = int8_t(((s0 >> (4 * (is / 8))) & 0x0F) |
                                (((s1 >> (2 * (is / 4))) & 0x03) << 4));
    float d = shmem_read_f16(ib + 108);
    return d * float(scale - 32);
#endif
}

FLOAT_TYPE mmvq_dot_product(uint ib_a, uint iqs) {
    int32_t q_sum = 0;

#if !defined(USE_SHMEM_STAGING)
    const uint iqs_adj = iqs * 4;
#else
    const uint iqs_adj = iqs;
#endif

    const i32vec4 qs_a = repack4(ib_a, iqs_adj);
    const float d_scale = get_d_scale(ib_a, iqs_adj);

    q_sum += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]);

    return FLOAT_TYPE(float(cache_b_ds.x) * d_scale * float(q_sum));
}
#endif

#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
// 4-byte loads for Q4_K blocks (144 bytes) and Q5_K blocks (176 bytes)
// Q4_K: dm@0(4B), scales@4(12B), qs@16(128B)
// Q5_K: dm@0(4B), scales@4(12B), qh@16(32B), qs@48(128B)
i32vec4 repack4(uint ib, uint iqs) {
#if !defined(USE_SHMEM_STAGING)
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 16) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 16) / 8) * 4;

#if defined(DATA_A_Q4_K)
    const uint32_t vals0 = (data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x0F0F0F0F;
    const uint32_t vals1 = (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x0F0F0F0F;
    const uint32_t vals2 = (data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x0F0F0F0F;
    const uint32_t vals3 = (data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x0F0F0F0F;

    return i32vec4(vals0, vals1, vals2, vals3);
#else // defined(DATA_A_Q5_K)
    const uint qh_idx = iqs;
    const uint qh_shift = iqs_k / 8;

    return i32vec4(((data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x0F0F0F0F) |
                  (((data_a_packed32[ib_k].qh[qh_idx    ] >> qh_shift) & 0x01010101) << 4),
                   ((data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x0F0F0F0F) |
                  (((data_a_packed32[ib_k].qh[qh_idx + 1] >> qh_shift) & 0x01010101) << 4),
                   ((data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x0F0F0F0F) |
                  (((data_a_packed32[ib_k].qh[qh_idx + 2] >> qh_shift) & 0x01010101) << 4),
                   ((data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x0F0F0F0F) |
                  (((data_a_packed32[ib_k].qh[qh_idx + 3] >> qh_shift) & 0x01010101) << 4));
#endif

#else
    const uint qs_idx = (iqs / 16) * 8 + (iqs % 8);
    const uint qs_shift = ((iqs % 16) / 8) * 4;

#if defined(DATA_A_Q4_K)
    // qs@16
    return i32vec4((uint32_t(shmem_read_i32(ib + 16 + (qs_idx    ) * 4)) >> qs_shift) & 0x0F0F0F0F,
                   (uint32_t(shmem_read_i32(ib + 16 + (qs_idx + 1) * 4)) >> qs_shift) & 0x0F0F0F0F,
                   (uint32_t(shmem_read_i32(ib + 16 + (qs_idx + 2) * 4)) >> qs_shift) & 0x0F0F0F0F,
                   (uint32_t(shmem_read_i32(ib + 16 + (qs_idx + 3) * 4)) >> qs_shift) & 0x0F0F0F0F);
#else // Q5_K: qh@16, qs@48
    const uint qh_idx = iqs % 8;
    const uint qh_shift = iqs / 8;

    return i32vec4(((uint32_t(shmem_read_i32(ib + 48 + (qs_idx    ) * 4)) >> qs_shift) & 0x0F0F0F0F) |
                  (((uint32_t(shmem_read_i32(ib + 16 + (qh_idx    ) * 4)) >> qh_shift) & 0x01010101) << 4),
                   ((uint32_t(shmem_read_i32(ib + 48 + (qs_idx + 1) * 4)) >> qs_shift) & 0x0F0F0F0F) |
                  (((uint32_t(shmem_read_i32(ib + 16 + (qh_idx + 1) * 4)) >> qh_shift) & 0x01010101) << 4),
                   ((uint32_t(shmem_read_i32(ib + 48 + (qs_idx + 2) * 4)) >> qs_shift) & 0x0F0F0F0F) |
                  (((uint32_t(shmem_read_i32(ib + 16 + (qh_idx + 2) * 4)) >> qh_shift) & 0x01010101) << 4),
                   ((uint32_t(shmem_read_i32(ib + 48 + (qs_idx + 3) * 4)) >> qs_shift) & 0x0F0F0F0F) |
                  (((uint32_t(shmem_read_i32(ib + 16 + (qh_idx + 3) * 4)) >> qh_shift) & 0x01010101) << 4));
#endif
#endif
}

vec2 get_dm_scale(uint ib, uint iqs) {
#if !defined(USE_SHMEM_STAGING)
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint is = iqs_k / 8;

    u8vec2 scale_dm;
    if (is < 4) {
        scale_dm = u8vec2(data_a[ib_k].scales[is] & 0x3F, data_a[ib_k].scales[is + 4] & 0x3F);
    } else {
        scale_dm = u8vec2((data_a[ib_k].scales[is+4] & 0xF) | ((data_a[ib_k].scales[is-4] & 0xC0) >> 2),
                          (data_a[ib_k].scales[is+4] >>  4) | ((data_a[ib_k].scales[is  ] & 0xC0) >> 2));
    }

    return FLOAT_TYPE_VEC2(data_a_packed32[ib_k].dm) * FLOAT_TYPE_VEC2(scale_dm);
#else
    // scales@4(12B), dm@0(4B)
    const uint is = iqs / 8;

    u8vec2 scale_dm;
    if (is < 4) {
        scale_dm = u8vec2(shmem_read_u8(ib + 4 + is) & 0x3F,
                          shmem_read_u8(ib + 4 + is + 4) & 0x3F);
    } else {
        uint s_is4  = shmem_read_u8(ib + 4 + is + 4);
        uint s_ism4 = shmem_read_u8(ib + 4 + is - 4);
        uint s_is   = shmem_read_u8(ib + 4 + is);
        scale_dm = u8vec2((s_is4 & 0xF) | ((s_ism4 & 0xC0) >> 2),
                          (s_is4 >> 4) | ((s_is & 0xC0) >> 2));
    }

    vec2 dm = unpackHalf2x16(uint(shmem_read_i32(ib)));
    return dm * vec2(scale_dm);
#endif
}

FLOAT_TYPE mmvq_dot_product(uint ib_a, uint iqs) {
    int32_t q_sum = 0;

#if !defined(USE_SHMEM_STAGING)
    const uint iqs_adj = iqs * 4;
#else
    const uint iqs_adj = iqs;
#endif

    const i32vec4 qs_a = repack4(ib_a, iqs_adj);
    const vec2 dm_scale = get_dm_scale(ib_a, iqs_adj);

    q_sum += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]);

    return FLOAT_TYPE(float(cache_b_ds.x) * float(dm_scale.x) * float(q_sum) - float(dm_scale.y) * float(cache_b_ds.y / 2));
}
#endif

#if defined(DATA_A_Q6_K)
// 2-byte loads for Q6_K blocks (210 bytes)
// Q6_K: ql@0(128B), qh@128(64B), scales@192(16B), d@208(2B)
i32vec4 repack4(uint ib, uint iqs) {
#if !defined(USE_SHMEM_STAGING)
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint ql_idx = (iqs_k / 32) * 16 + iqs_k % 16;
    const uint ql_shift = ((iqs_k % 32) / 16) * 4;

    const uint qh_idx = (iqs_k / 32) * 8 + iqs;
    const uint qh_shift = ((iqs_k % 32) / 8) * 2;

    const i8vec2 vals00 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2    ] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2    ] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals01 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 1] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 1] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals10 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 2] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 2] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals11 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 3] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 3] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals20 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 4] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 4] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals21 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 5] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 5] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals30 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 6] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 6] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals31 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 7] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 7] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);

#else
    const uint ql_idx = (iqs / 32) * 16 + iqs % 16;
    const uint ql_shift = ((iqs % 32) / 16) * 4;

    const uint qh_iqs = iqs % 8;
    const uint qh_idx = (iqs / 32) * 8 + qh_iqs;
    const uint qh_shift = ((iqs % 32) / 8) * 2;

    // ql: uint16[64] at byte 0, qh: uint16[32] at byte 128
    const i8vec2 vals00 = (unpack8(int16_t((uint16_t(shmem_read_u16(ib + ql_idx * 4        )) >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((uint16_t(shmem_read_u16(ib + 128 + qh_idx * 4        )) >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals01 = (unpack8(int16_t((uint16_t(shmem_read_u16(ib + ql_idx * 4 + 2    )) >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((uint16_t(shmem_read_u16(ib + 128 + qh_idx * 4 + 2    )) >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals10 = (unpack8(int16_t((uint16_t(shmem_read_u16(ib + ql_idx * 4 + 4    )) >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((uint16_t(shmem_read_u16(ib + 128 + qh_idx * 4 + 4    )) >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals11 = (unpack8(int16_t((uint16_t(shmem_read_u16(ib + ql_idx * 4 + 6    )) >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((uint16_t(shmem_read_u16(ib + 128 + qh_idx * 4 + 6    )) >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals20 = (unpack8(int16_t((uint16_t(shmem_read_u16(ib + ql_idx * 4 + 8    )) >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((uint16_t(shmem_read_u16(ib + 128 + qh_idx * 4 + 8    )) >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals21 = (unpack8(int16_t((uint16_t(shmem_read_u16(ib + ql_idx * 4 + 10   )) >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((uint16_t(shmem_read_u16(ib + 128 + qh_idx * 4 + 10   )) >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals30 = (unpack8(int16_t((uint16_t(shmem_read_u16(ib + ql_idx * 4 + 12   )) >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((uint16_t(shmem_read_u16(ib + 128 + qh_idx * 4 + 12   )) >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals31 = (unpack8(int16_t((uint16_t(shmem_read_u16(ib + ql_idx * 4 + 14   )) >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((uint16_t(shmem_read_u16(ib + 128 + qh_idx * 4 + 14   )) >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
#endif

    return i32vec4(pack32(i8vec4(vals00.x, vals00.y, vals01.x, vals01.y)),
                   pack32(i8vec4(vals10.x, vals10.y, vals11.x, vals11.y)),
                   pack32(i8vec4(vals20.x, vals20.y, vals21.x, vals21.y)),
                   pack32(i8vec4(vals30.x, vals30.y, vals31.x, vals31.y)));
}

float get_d_scale(uint ib, uint iqs) {
#if !defined(USE_SHMEM_STAGING)
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;
    return float(data_a[ib_k].d) * float(data_a[ib_k].scales[iqs_k / 4]);
#else
    float d = shmem_read_f16(ib + 208);
    int8_t scale = int8_t(shmem_read_u8(ib + 192 + iqs / 4));
    return d * float(scale);
#endif
}

FLOAT_TYPE mmvq_dot_product(uint ib_a, uint iqs) {
    int32_t q_sum = 0;

#if !defined(USE_SHMEM_STAGING)
    const uint iqs_adj = iqs * 4;
#else
    const uint iqs_adj = iqs;
#endif

    const i32vec4 qs_a = repack4(ib_a, iqs_adj);
    const float d_scale = get_d_scale(ib_a, iqs_adj);

    q_sum += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]);

    return FLOAT_TYPE(float(cache_b_ds.x) * float(d_scale) * float(q_sum));
}
#endif

#if defined(DATA_A_IQ1_S)
// IQ1_S (50B): d@0(2B), qs@2(32B), qh@34(16B)
void repack8(uint ib, uint iqs, out i32vec4 out0, out i32vec4 out1) {
#if !defined(USE_SHMEM_STAGING)
    const uint ib32 = iqs / 32;

    const uint qh = data_a[ib].qh[ib32];

    const uint qs16_0 = data_a_packed16[ib].qs[(4 * ib32 + 0) / 2];
    const uint qs16_1 = data_a_packed16[ib].qs[(4 * ib32 + 2) / 2];
#else
    // ib = byte offset, iqs = sub-block index (0..7)
    const uint qh = shmem_read_u16(ib + 34 + iqs * 2);
    const uint qs16_0 = shmem_read_u16(ib + 2 + iqs * 4);
    const uint qs16_1 = shmem_read_u16(ib + 2 + iqs * 4 + 2);
#endif

    const uint qs0 = qs16_0 & 0xFF;
    const uint qs1 = qs16_0 >> 8;
    const uint qs2 = qs16_1 & 0xFF;
    const uint qs3 = qs16_1 >> 8;

    const uint hi0 = bitfieldExtract(qh, 3 * int(0), 3);
    const uint hi1 = bitfieldExtract(qh, 3 * int(1), 3);
    const uint hi2 = bitfieldExtract(qh, 3 * int(2), 3);
    const uint hi3 = bitfieldExtract(qh, 3 * int(3), 3);

    const int32_t grid0 = int32_t(iq1s_grid_gpu[qs0 | (hi0 << 8)]);
    const int32_t grid1 = int32_t(iq1s_grid_gpu[qs1 | (hi1 << 8)]);
    const int32_t grid2 = int32_t(iq1s_grid_gpu[qs2 | (hi2 << 8)]);
    const int32_t grid3 = int32_t(iq1s_grid_gpu[qs3 | (hi3 << 8)]);

    out0 = i32vec4((grid0 >> 0) & 0x0F0F0F0F,
                   (grid0 >> 4) & 0x0F0F0F0F,
                   (grid1 >> 0) & 0x0F0F0F0F,
                   (grid1 >> 4) & 0x0F0F0F0F);
    out1 = i32vec4((grid2 >> 0) & 0x0F0F0F0F,
                   (grid2 >> 4) & 0x0F0F0F0F,
                   (grid3 >> 0) & 0x0F0F0F0F,
                   (grid3 >> 4) & 0x0F0F0F0F);
}

vec2 get_dm(uint ib, uint iqs) {
#if !defined(USE_SHMEM_STAGING)
    const uint ib32 = iqs / 32;

    const uint qh = data_a[ib].qh[ib32];

    const float d = float(data_a[ib].d);
#else
    // ib = byte offset, iqs = sub-block index (0..7)
    const uint qh = shmem_read_u16(ib + 34 + iqs * 2);

    const float d = shmem_read_f16(ib);
#endif
    const float delta = ((qh & 0x8000) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;
    const float dl = d * float(2 * bitfieldExtract(qh, 12, 3) + 1);

    // the -1 cancels out the bias in iq1s_grid_gpu
    return FLOAT_TYPE_VEC2(dl, dl * (delta - 1));
}

FLOAT_TYPE mmvq_dot_product(uint ib_a, uint iqs) {
    int32_t q_sum = 0;

#if !defined(USE_SHMEM_STAGING)
    const uint ib_k = ib_a / 8;
    const uint iqs_k = (ib_a % 8) * 32 + iqs * 32;
#else
    // ib_a = byte offset to block, iqs = sub-block index (0..7)
    const uint ib_k = ib_a;
    const uint iqs_k = iqs;
#endif

    i32vec4 qs_a0;
    i32vec4 qs_a1;
    repack8(ib_k, iqs_k, qs_a0, qs_a1);

    const vec2 dm = get_dm(ib_k, iqs_k);

    q_sum += dotPacked4x8EXT(qs_a0.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a0.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a0.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a0.w, cache_b_qs[3]);
    q_sum += dotPacked4x8EXT(qs_a1.x, cache_b_qs[4]);
    q_sum += dotPacked4x8EXT(qs_a1.y, cache_b_qs[5]);
    q_sum += dotPacked4x8EXT(qs_a1.z, cache_b_qs[6]);
    q_sum += dotPacked4x8EXT(qs_a1.w, cache_b_qs[7]);

    return FLOAT_TYPE(float(cache_b_ds.x) * float(dm.x) * float(q_sum) + float(dm.y) * float(cache_b_ds.y));
}
#endif

#if defined(DATA_A_IQ1_M)
// IQ1_M (56B): qs@0(32B), qh@32(16B), scales@48(8B)
FLOAT_TYPE mmvq_dot_product(uint ib_a, uint iqs) {
#if !defined(USE_SHMEM_STAGING)
    const uint ib_k = ib_a / 8;
    const uint iqs_k = (ib_a % 8) * 32 + iqs * 32;

    const uint ib32 = iqs_k / 32;
    const uint ib64 = ib32 / 2;

    const uint16_t[4] scales = data_a[ib_k].scales;
    const u16vec4 s = u16vec4(scales[0], scales[1], scales[2], scales[3]) >> 12;
    const float d = float(unpackHalf2x16(s.x | (s.y << 4) | (s.z << 8) | (s.w << 12)).x);

    const uint qs32 = data_a_packed32[ib_k].qs[ib32];
    const uint qh16 = data_a_packed16[ib_k].qh[ib32];

    float sum = 0;
    const uint sc = data_a[ib_k].scales[ib64];
#else
    // ib_a = byte offset to block, iqs = sub-block index (0..7)
    const uint ib32 = iqs;
    const uint ib64 = ib32 / 2;

    // scales: uint16_t[4] at byte 48
    const u16vec4 s = u16vec4(shmem_read_u16(ib_a + 48),
                              shmem_read_u16(ib_a + 50),
                              shmem_read_u16(ib_a + 52),
                              shmem_read_u16(ib_a + 54)) >> 12;
    const float d = float(unpackHalf2x16(s.x | (s.y << 4) | (s.z << 8) | (s.w << 12)).x);

    // qs: uint32_t[8] at byte 0
    const uint qs32 = uint(shmem_read_i32(ib_a + ib32 * 4));
    // qh: uint16_t[8] at byte 32
    const uint qh16 = shmem_read_u16(ib_a + 32 + ib32 * 2);

    float sum = 0;
    // scales[ib64]: uint16_t at byte 48
    const uint sc = shmem_read_u16(ib_a + 48 + ib64 * 2);
#endif
    [[unroll]] for (int l = 0; l < 4; ++l) {
        const uint ib16 = 2 * ib32 + l / 2;
        const float dl = d * (2 * bitfieldExtract(sc, 3 * int(ib16 & 3), 3) + 1);
        const uint qh = qh16 >> (4 * l);
        const uint qs = (qs32 >> (8 * l)) & 0xFF;
        const float delta = ((qh & 8) != 0) ? -IQ1M_DELTA : IQ1M_DELTA;

        const int32_t grid = int32_t(iq1s_grid_gpu[qs | ((qh & 7) << 8)]);

        int32_t q_sum = 0;
        q_sum += dotPacked4x8EXT((grid >> 0) & 0x0F0F0F0F, cache_b_qs[2 * l + 0]);
        q_sum += dotPacked4x8EXT((grid >> 4) & 0x0F0F0F0F, cache_b_qs[2 * l + 1]);

        int32_t y_sum = 0;
        y_sum += dotPacked4x8EXT(int(0x01010101), cache_b_qs[2 * l + 0]);
        y_sum += dotPacked4x8EXT(int(0x01010101), cache_b_qs[2 * l + 1]);

        // the -1 cancels out the bias in iq1s_grid_gpu
        sum += dl * (q_sum + y_sum * (delta - 1));
    }
    sum *= float(cache_b_ds.x);

    return sum;
}
#endif
