#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// ifairy_relu2
//------------------------------------------------------------------------------

static inline float ifairy_bf16_to_f32(ushort x) {
    return as_float(((uint) x) << 16);
}

static inline ushort ifairy_f32_to_bf16(float x) {
    const uint u = as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        // Quiet NaNs to match ggml_compute_fp32_to_bf16.
        return (ushort) ((u >> 16) | 64u);
    }
    return (ushort) ((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
}

static inline uint ifairy_pack_bf16_pair(float real_v, float imag_v) {
    const uint r = (uint) ifairy_f32_to_bf16(real_v);
    const uint i = (uint) ifairy_f32_to_bf16(imag_v);
    return r | (i << 16);
}

kernel void kernel_ifairy_relu2(
        global uint * src0,
        ulong offset0,
        global uint * dst,
        ulong offsetd
) {
    src0 = (global uint *) ((global char *) src0 + offset0);
    dst  = (global uint *) ((global char *) dst  + offsetd);

    const uint gid = get_global_id(0);

    const uint v = src0[gid];
    const ushort r_bf16 = (ushort) (v & 0xFFFFu);
    const ushort i_bf16 = (ushort) (v >> 16);

    // If both real/imag are negative, output packed zero.
    if (((r_bf16 & i_bf16) & (ushort) 0x8000u) != 0) {
        dst[gid] = 0u;
        return;
    }

    const float r = ifairy_bf16_to_f32(r_bf16);
    const float i = ifairy_bf16_to_f32(i_bf16);

    dst[gid] = ifairy_pack_bf16_pair(r * r, i * i);
}
