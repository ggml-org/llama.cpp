#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Pack B f32 [N][K] into f16 64x16 tiles (column-major, ld 16) at (tn * (K/16) + ks) * 1024.
// Columns >= N are zero. A workgroup (64, 4) packs 64 columns x 64 k.
#define PACK_LDS (64 + 8)

kernel void kernel_pack_b_f16_coopmat(
        global const char * src1,
        ulong               offset1,
        global half       * dst,
        int                 K,
        int                 N
) {
    local half tile[64 * PACK_LDS];

    const int l  = get_local_id(1) * 64 + get_local_id(0);
    const int tn = get_group_id(0);
    const int kt = get_group_id(1);
    const int k0 = kt * 64;

    global const float * src = (global const float *)(src1 + offset1);

    for (int j = 0; j < 4; j++) {
        const int f = j * 256 + l;
        const int r = f >> 4;
        const int c = (f & 15) * 4;
        const int n = tn * 64 + r;
        float4 v = (float4)(0.0f);
        if (n < N && k0 + c < K) {
            v = vload4(0, src + (ulong)n * K + k0 + c);
        }
        vstore_half4(v, 0, tile + r * PACK_LDS + c);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    global half * out = dst + ((ulong)tn * (K / 16) + kt * 4) * 1024;
    for (int j = 0; j < 2; j++) {
        const int q  = j * 256 + l;
        const int ks = q >> 7;
        const int n  = (q >> 1) & 63;
        const int h  = q & 1;
        if (kt * 4 + ks < K / 16) {
            half8 v = vload8(0, tile + n * PACK_LDS + ks * 16 + h * 8);
            vstore8(v, 0, out + q * 8);
        }
    }
}
