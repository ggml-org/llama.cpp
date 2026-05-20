#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_uniform_load : enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_constant_load : enable

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void adreno_xmem_pack_src_f32(
    __global const void * src_void,
    ulong offset,
    __write_only image2d_t src_img,
    int K,
    int N) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int kpack = K / 4;

    if (x >= N || y >= kpack) {
        return;
    }

    __global const float * src = (__global const float *)((__global const char *)src_void + offset);
    const int base = x*K + y*4;
    const half4 v = (half4)((half)src[base + 0], (half)src[base + 1], (half)src[base + 2], (half)src[base + 3]);
    write_imageh(src_img, (int2)(x, y), v);
}

__kernel void adreno_xmem_prepack_weight_f16(
    __global half4 * dst,
    __global const void * src_void,
    ulong offset,
    int K,
    int M,
    int kpack,
    int npack,
    int os) {
    const int linear = get_global_id(0);
    const int total = kpack*npack;
    if (linear >= total) {
        return;
    }

    __global const half * src = (__global const half *)((__global const char *)src_void + offset);

    const int dst_ogroup = linear % os;
    const int dst_o_sp_i = linear / os;
    const int dst_i = dst_o_sp_i % kpack;
    const int dst_o = dst_o_sp_i / kpack;
    const int o_slice = dst_o*os + dst_ogroup;
    const int k_base = dst_i*4;

    half4 w0 = (half4)(0.0h);
    half4 w1 = (half4)(0.0h);
    half4 w2 = (half4)(0.0h);
    half4 w3 = (half4)(0.0h);

    const int o0 = o_slice*4 + 0;
    const int o1 = o_slice*4 + 1;
    const int o2 = o_slice*4 + 2;
    const int o3 = o_slice*4 + 3;

    if (k_base + 0 < K) {
        if (o0 < M) w0.s0 = src[o0*K + k_base + 0];
        if (o1 < M) w0.s1 = src[o1*K + k_base + 0];
        if (o2 < M) w0.s2 = src[o2*K + k_base + 0];
        if (o3 < M) w0.s3 = src[o3*K + k_base + 0];
    }
    if (k_base + 1 < K) {
        if (o0 < M) w1.s0 = src[o0*K + k_base + 1];
        if (o1 < M) w1.s1 = src[o1*K + k_base + 1];
        if (o2 < M) w1.s2 = src[o2*K + k_base + 1];
        if (o3 < M) w1.s3 = src[o3*K + k_base + 1];
    }
    if (k_base + 2 < K) {
        if (o0 < M) w2.s0 = src[o0*K + k_base + 2];
        if (o1 < M) w2.s1 = src[o1*K + k_base + 2];
        if (o2 < M) w2.s2 = src[o2*K + k_base + 2];
        if (o3 < M) w2.s3 = src[o3*K + k_base + 2];
    }
    if (k_base + 3 < K) {
        if (o0 < M) w3.s0 = src[o0*K + k_base + 3];
        if (o1 < M) w3.s1 = src[o1*K + k_base + 3];
        if (o2 < M) w3.s2 = src[o2*K + k_base + 3];
        if (o3 < M) w3.s3 = src[o3*K + k_base + 3];
    }

    dst[linear*4 + 0] = w0;
    dst[linear*4 + 1] = w1;
    dst[linear*4 + 2] = w2;
    dst[linear*4 + 3] = w3;
}

__attribute__((qcom_max_concurrent_subgroups(12)))
__kernel void kernel_gemm_xmem_f16_f32_os8(
    __constant half8 * weights_buffer __attribute__((sub_group_uniform)),
    __constant half8 * xmem_buffer __attribute__((max_constant_size((6144)))),
    __read_only image2d_t src_img,
    __write_only image2d_t dst_img,
    int N,
    int npack,
    int kpack) {
    const int X = get_group_id(1)*get_local_size(0) + get_local_id(0);
    const int Z = get_group_id(0)*get_local_size(2) + get_local_id(2);

    if (X >= N || Z*8 >= npack) {
        return;
    }

    half4 r0 = (half4)(0.0h);
    half4 r1 = (half4)(0.0h);
    half4 r2 = (half4)(0.0h);
    half4 r3 = (half4)(0.0h);
    half4 r4 = (half4)(0.0h);
    half4 r5 = (half4)(0.0h);
    half4 r6 = (half4)(0.0h);
    half4 r7 = (half4)(0.0h);

    int f_offset = Z*kpack*32;
    int subgroup_id = (int)(0x1F & qcom_get_physical_sub_group_id());
    subgroup_id = subgroup_id % 12;
    const int c_offset = subgroup_id*32;
    __constant half16 * weights_cache = (__constant half16 *)&xmem_buffer[c_offset];

    int coord_s = 0;
    do {
        const half4 src0 = read_imageh(src_img, smp_zero, (int2)(X, coord_s));
        coord_s++;
        const half4 src1 = read_imageh(src_img, smp_zero, (int2)(X, coord_s));
        coord_s++;

        qcom_sub_group_constant_load8(xmem_buffer, weights_buffer, c_offset, f_offset >> 1, 32);
        f_offset += 64;
        qcom_sub_group_sync(QCOM_CLK_CONST_LOAD_SYNC);

        r0 += src0.x * weights_cache[0].s0123;
        r0 += src0.y * weights_cache[0].s4567;
        r0 += src0.z * weights_cache[0].s89ab;
        r0 += src0.w * weights_cache[0].scdef;
        r1 += src0.x * weights_cache[1].s0123;
        r1 += src0.y * weights_cache[1].s4567;
        r1 += src0.z * weights_cache[1].s89ab;
        r1 += src0.w * weights_cache[1].scdef;
        r2 += src0.x * weights_cache[2].s0123;
        r2 += src0.y * weights_cache[2].s4567;
        r2 += src0.z * weights_cache[2].s89ab;
        r2 += src0.w * weights_cache[2].scdef;
        r3 += src0.x * weights_cache[3].s0123;
        r3 += src0.y * weights_cache[3].s4567;
        r3 += src0.z * weights_cache[3].s89ab;
        r3 += src0.w * weights_cache[3].scdef;
        r4 += src0.x * weights_cache[4].s0123;
        r4 += src0.y * weights_cache[4].s4567;
        r4 += src0.z * weights_cache[4].s89ab;
        r4 += src0.w * weights_cache[4].scdef;
        r5 += src0.x * weights_cache[5].s0123;
        r5 += src0.y * weights_cache[5].s4567;
        r5 += src0.z * weights_cache[5].s89ab;
        r5 += src0.w * weights_cache[5].scdef;
        r6 += src0.x * weights_cache[6].s0123;
        r6 += src0.y * weights_cache[6].s4567;
        r6 += src0.z * weights_cache[6].s89ab;
        r6 += src0.w * weights_cache[6].scdef;
        r7 += src0.x * weights_cache[7].s0123;
        r7 += src0.y * weights_cache[7].s4567;
        r7 += src0.z * weights_cache[7].s89ab;
        r7 += src0.w * weights_cache[7].scdef;

        r0 += src1.x * weights_cache[8].s0123;
        r0 += src1.y * weights_cache[8].s4567;
        r0 += src1.z * weights_cache[8].s89ab;
        r0 += src1.w * weights_cache[8].scdef;
        r1 += src1.x * weights_cache[9].s0123;
        r1 += src1.y * weights_cache[9].s4567;
        r1 += src1.z * weights_cache[9].s89ab;
        r1 += src1.w * weights_cache[9].scdef;
        r2 += src1.x * weights_cache[10].s0123;
        r2 += src1.y * weights_cache[10].s4567;
        r2 += src1.z * weights_cache[10].s89ab;
        r2 += src1.w * weights_cache[10].scdef;
        r3 += src1.x * weights_cache[11].s0123;
        r3 += src1.y * weights_cache[11].s4567;
        r3 += src1.z * weights_cache[11].s89ab;
        r3 += src1.w * weights_cache[11].scdef;
        r4 += src1.x * weights_cache[12].s0123;
        r4 += src1.y * weights_cache[12].s4567;
        r4 += src1.z * weights_cache[12].s89ab;
        r4 += src1.w * weights_cache[12].scdef;
        r5 += src1.x * weights_cache[13].s0123;
        r5 += src1.y * weights_cache[13].s4567;
        r5 += src1.z * weights_cache[13].s89ab;
        r5 += src1.w * weights_cache[13].scdef;
        r6 += src1.x * weights_cache[14].s0123;
        r6 += src1.y * weights_cache[14].s4567;
        r6 += src1.z * weights_cache[14].s89ab;
        r6 += src1.w * weights_cache[14].scdef;
        r7 += src1.x * weights_cache[15].s0123;
        r7 += src1.y * weights_cache[15].s4567;
        r7 += src1.z * weights_cache[15].s89ab;
        r7 += src1.w * weights_cache[15].scdef;
    } while (coord_s < kpack);

    int coord_s_out = Z*8;
    if (coord_s_out < npack) { write_imageh(dst_img, (int2)(X, coord_s_out), r0); coord_s_out++; }
    if (coord_s_out < npack) { write_imageh(dst_img, (int2)(X, coord_s_out), r1); coord_s_out++; }
    if (coord_s_out < npack) { write_imageh(dst_img, (int2)(X, coord_s_out), r2); coord_s_out++; }
    if (coord_s_out < npack) { write_imageh(dst_img, (int2)(X, coord_s_out), r3); coord_s_out++; }
    if (coord_s_out < npack) { write_imageh(dst_img, (int2)(X, coord_s_out), r4); coord_s_out++; }
    if (coord_s_out < npack) { write_imageh(dst_img, (int2)(X, coord_s_out), r5); coord_s_out++; }
    if (coord_s_out < npack) { write_imageh(dst_img, (int2)(X, coord_s_out), r6); coord_s_out++; }
    if (coord_s_out < npack) { write_imageh(dst_img, (int2)(X, coord_s_out), r7); }
}

__kernel void adreno_xmem_store_dst_f32(
    __read_only image2d_t dst_img,
    __global void * dst_void,
    ulong offset,
    int M,
    int N) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int npack = (M + 3) / 4;

    if (x >= N || y >= npack) {
        return;
    }

    __global float * dst = (__global float *)((__global char *)dst_void + offset);
    const half4 hv = read_imageh(dst_img, smp_zero, (int2)(x, y));
    const int m = y*4;
    if (m + 0 < M) dst[x*M + m + 0] = (float)hv.s0;
    if (m + 1 < M) dst[x*M + m + 1] = (float)hv.s1;
    if (m + 2 < M) dst[x*M + m + 2] = (float)hv.s2;
    if (m + 3 < M) dst[x*M + m + 3] = (float)hv.s3;
}
