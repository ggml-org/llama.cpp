#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include "types.glsl"

// Each iqs value maps to a 32-bit integer

#if defined(DATA_A_Q4_0) || defined(DATA_A_Q4_1)
// 2-byte loads for Q4_0 blocks (18 bytes)
// 4-byte loads for Q4_1 blocks (20 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
#ifdef DATA_A_Q4_0
    const uint32_t vui = pack32(u16vec2(data_a_packed16[ib].qs[iqs * 2],
                                        data_a_packed16[ib].qs[iqs * 2 + 1]));
#else // DATA_A_Q4_1
    const uint32_t vui = data_a_packed32[ib].qs[iqs];
#endif

    uint32_t lo4 = vui & 0x0F0F0F0F;
    uint32_t hi4 = (vui >> 4) & 0x0F0F0F0F;

    // subtract 8 from each byte
    lo4 = ((lo4 | 0x80808080) - 0x08080808) ^ 0x80808080;
    hi4 = ((hi4 | 0x80808080) - 0x08080808) ^ 0x80808080;

    buf_a_qs[buf_ib * shmem_stride + iqs    ] = lo4;
    buf_a_qs[buf_ib * shmem_stride + iqs + 4] = hi4;

    if (iqs == 0) {
#ifdef DATA_A_Q4_0
        buf_a_d[buf_ib] = FLOAT_TYPE(data_a_packed16[ib].d);
#else // DATA_A_Q4_1
#endif
    }
}
#endif

#if defined(DATA_A_Q5_0) || defined(DATA_A_Q5_1)
// 2-byte loads for Q5_0 blocks (22 bytes)
// 4-byte loads for Q5_1 blocks (24 bytes)
}
#endif

#if defined(DATA_A_Q8_0)
// 2-byte loads for Q8_0 blocks (34 bytes)
#endif

#if defined(DATA_A_MXFP4)
// 1-byte loads for mxfp4 blocks (17 bytes)
#endif

// For k-quants, ib and iqs still assume 32-wide blocks, but k-quants are 256-wide
// iqs still refers to a 32-bit integer, meaning 0..7 for 32-wide quants
#if defined(DATA_A_Q2_K)
// 4-byte loads for Q2_K blocks (84 bytes)
#endif

#if defined(DATA_A_Q3_K)
// 2-byte loads for Q3_K blocks (110 bytes)
#endif

#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
// 4-byte loads for Q4_K blocks (144 bytes) and Q5_K blocks (176 bytes)
#endif

#if defined(DATA_A_Q6_K)
// 2-byte loads for Q6_K blocks (210 bytes)
#endif

void block_b_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_outer = ib / 4;
    const uint ib_inner = ib % 4;

    if (iqs == 0) {
        // Divide by TK for matmul scale application
        buf_b_d[buf_ib] = data_b[ib_outer].ds[ib_inner].x;
    }

    const ivec4 values = data_b[ib_outer].qs[ib_inner * 2 + iqs];
    buf_b_qs[buf_ib * shmem_stride + iqs * 4    ] = values.x;
    buf_b_qs[buf_ib * shmem_stride + iqs * 4 + 1] = values.y;
    buf_b_qs[buf_ib * shmem_stride + iqs * 4 + 2] = values.z;
    buf_b_qs[buf_ib * shmem_stride + iqs * 4 + 3] = values.w;
}
