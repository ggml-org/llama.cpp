#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_dot_product8 : enable
#pragma OPENCL EXTENSION cl_qcom_local_memory_control : enable

//------------------------------------------------------------------------------
// activation quantize+pack: one work-item per (token, 32-K block).
//------------------------------------------------------------------------------
kernel void kernel_gemm_dot8_q8_0_pack_act(
        global uint4 * act,
        global float2 * params,
        global const void * src_void,
        ulong offset,
        int K,
        int N,
        int nblk
) {
    const int lin = get_global_id(0);
    if (lin >= N*nblk) {
        return;
    }
    const int t   = lin / nblk;
    const int blk = lin % nblk;

    global const float * src = (global const float *)((global const char *)src_void + offset);
    global const float4 * row = (global const float4 *)(src + (long)t*K + blk*32);

    const float4 v0 = row[0], v1 = row[1], v2 = row[2], v3 = row[3];
    const float4 v4 = row[4], v5 = row[5], v6 = row[6], v7 = row[7];
    float4 am = fmax(fmax(fmax(fabs(v0), fabs(v1)), fmax(fabs(v2), fabs(v3))),
                     fmax(fmax(fabs(v4), fabs(v5)), fmax(fabs(v6), fabs(v7))));
    const float amax = fmax(fmax(am.x, am.y), fmax(am.z, am.w));
    const float d_a = amax / 127.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;

    const int4 q0 = convert_int4_rte(v0*inv), q1 = convert_int4_rte(v1*inv);
    const int4 q2 = convert_int4_rte(v2*inv), q3 = convert_int4_rte(v3*inv);
    const int4 q4 = convert_int4_rte(v4*inv), q5 = convert_int4_rte(v5*inv);
    const int4 q6 = convert_int4_rte(v6*inv), q7 = convert_int4_rte(v7*inv);
    const int4 s4 = q0+q1+q2+q3+q4+q5+q6+q7;
    const int sum = s4.x + s4.y + s4.z + s4.w;

    uint4 ta, tb;
    ta.x = as_uint(convert_char4_sat(q0)); ta.y = as_uint(convert_char4_sat(q1));
    ta.z = as_uint(convert_char4_sat(q2)); ta.w = as_uint(convert_char4_sat(q3));
    tb.x = as_uint(convert_char4_sat(q4)); tb.y = as_uint(convert_char4_sat(q5));
    tb.z = as_uint(convert_char4_sat(q6)); tb.w = as_uint(convert_char4_sat(q7));
    act[(long)(blk*2 + 0)*N + t] = ta;
    act[(long)(blk*2 + 1)*N + t] = tb;

    params[(long)blk*N + t] = (float2)(d_a, -128.0f*d_a*(float)sum);
}

//------------------------------------------------------------------------------
// Q8_0 GEMM using dot8 + local memory broadcast
// uses the same memory layout as gemm_noshuffle_q8_0_f32
//------------------------------------------------------------------------------
static inline int4 dot8x4(int4 acc, uint4 w, uint s) {
    acc.x = qcom_dot8_acc(s, w.x, acc.x);
    acc.y = qcom_dot8_acc(s, w.y, acc.y);
    acc.z = qcom_dot8_acc(s, w.z, acc.z);
    acc.w = qcom_dot8_acc(s, w.w, acc.w);
    return acc;
}

kernel void kernel_gemm_dot8_q8_0_f32_lmbcast(
        global const uint4 * q_nosh,
        global const uint4 * act,
        global const half * d_trans,
        global const float2 * params,
        global void * dst_void,
        ulong offsetd,
        int N,
        int M,
        int nblk
) {
    // One full 32-K block (64 uint4 = two 16-K tiles) gathered per work-group pass.
    __local uint4 wl[64] __attribute__((qcom_local_memory_control(0)));

    const int lid = get_local_id(0);
    const int X   = get_group_id(0)*get_local_size(0) + lid;  // token
    const int Z   = get_group_id(1);                          // 32-channel group
    const int Xr  = (X < N) ? X : (N - 1);                    // clamped for OOB lanes

    const int Mu = M >> 2;     // M in uint4 (channels/4); M%32==0 so M%4==0
    const int th = lid >> 5;   // tile half (0/1) within the 32-K block
    const int qp = (lid >> 3) & 3;
    const int Mb = lid & 7;

    float4 f0 = 0;
    float4 f1 = 0;
    float4 f2 = 0;
    float4 f3 = 0;
    float4 f4 = 0;
    float4 f5 = 0;
    float4 f6 = 0;
    float4 f7 = 0;

    int ks = 0;
    for (int blk = 0; blk < nblk; blk++) {
        int4 r0 = 0;
        int4 r1 = 0;
        int4 r2 = 0;
        int4 r3 = 0;
        int4 r4 = 0;
        int4 r5 = 0;
        int4 r6 = 0;
        int4 r7 = 0;

        const uint4 sa = act[(long)(ks + 0)*N + Xr];   // first 16-K tile codes
        const uint4 sb = act[(long)(ks + 1)*N + Xr];   // second 16-K tile codes
        ks += 2;

        // cooperative gather from the noshuffle layout
        const int kq = blk*8 + th*4 + qp;
        wl[lid] = q_nosh[(long)kq*Mu + Z*8 + Mb] ^ (uint4)0x80808080u;
        barrier(CLK_LOCAL_MEM_FENCE);

        // First 16-K tile (wl[0..31]); broadcast-read with subgroup-uniform index.
        r0 = dot8x4(r0, wl[0], sa.x);
        r1 = dot8x4(r1, wl[1], sa.x);
        r2 = dot8x4(r2, wl[2], sa.x);
        r3 = dot8x4(r3, wl[3], sa.x);
        r4 = dot8x4(r4, wl[4], sa.x);
        r5 = dot8x4(r5, wl[5], sa.x);
        r6 = dot8x4(r6, wl[6], sa.x);
        r7 = dot8x4(r7, wl[7], sa.x);
        r0 = dot8x4(r0, wl[8],  sa.y);
        r1 = dot8x4(r1, wl[9],  sa.y);
        r2 = dot8x4(r2, wl[10], sa.y);
        r3 = dot8x4(r3, wl[11], sa.y);
        r4 = dot8x4(r4, wl[12], sa.y);
        r5 = dot8x4(r5, wl[13], sa.y);
        r6 = dot8x4(r6, wl[14], sa.y);
        r7 = dot8x4(r7, wl[15], sa.y);
        r0 = dot8x4(r0, wl[16], sa.z);
        r1 = dot8x4(r1, wl[17], sa.z);
        r2 = dot8x4(r2, wl[18], sa.z);
        r3 = dot8x4(r3, wl[19], sa.z);
        r4 = dot8x4(r4, wl[20], sa.z);
        r5 = dot8x4(r5, wl[21], sa.z);
        r6 = dot8x4(r6, wl[22], sa.z);
        r7 = dot8x4(r7, wl[23], sa.z);
        r0 = dot8x4(r0, wl[24], sa.w);
        r1 = dot8x4(r1, wl[25], sa.w);
        r2 = dot8x4(r2, wl[26], sa.w);
        r3 = dot8x4(r3, wl[27], sa.w);
        r4 = dot8x4(r4, wl[28], sa.w);
        r5 = dot8x4(r5, wl[29], sa.w);
        r6 = dot8x4(r6, wl[30], sa.w);
        r7 = dot8x4(r7, wl[31], sa.w);

        // Second 16-K tile (wl[32..63]).
        r0 = dot8x4(r0, wl[32], sb.x);
        r1 = dot8x4(r1, wl[33], sb.x);
        r2 = dot8x4(r2, wl[34], sb.x);
        r3 = dot8x4(r3, wl[35], sb.x);
        r4 = dot8x4(r4, wl[36], sb.x);
        r5 = dot8x4(r5, wl[37], sb.x);
        r6 = dot8x4(r6, wl[38], sb.x);
        r7 = dot8x4(r7, wl[39], sb.x);
        r0 = dot8x4(r0, wl[40], sb.y);
        r1 = dot8x4(r1, wl[41], sb.y);
        r2 = dot8x4(r2, wl[42], sb.y);
        r3 = dot8x4(r3, wl[43], sb.y);
        r4 = dot8x4(r4, wl[44], sb.y);
        r5 = dot8x4(r5, wl[45], sb.y);
        r6 = dot8x4(r6, wl[46], sb.y);
        r7 = dot8x4(r7, wl[47], sb.y);
        r0 = dot8x4(r0, wl[48], sb.z);
        r1 = dot8x4(r1, wl[49], sb.z);
        r2 = dot8x4(r2, wl[50], sb.z);
        r3 = dot8x4(r3, wl[51], sb.z);
        r4 = dot8x4(r4, wl[52], sb.z);
        r5 = dot8x4(r5, wl[53], sb.z);
        r6 = dot8x4(r6, wl[54], sb.z);
        r7 = dot8x4(r7, wl[55], sb.z);
        r0 = dot8x4(r0, wl[56], sb.w);
        r1 = dot8x4(r1, wl[57], sb.w);
        r2 = dot8x4(r2, wl[58], sb.w);
        r3 = dot8x4(r3, wl[59], sb.w);
        r4 = dot8x4(r4, wl[60], sb.w);
        r5 = dot8x4(r5, wl[61], sb.w);
        r6 = dot8x4(r6, wl[62], sb.w);
        r7 = dot8x4(r7, wl[63], sb.w);

        // fold: out += d * (d_a*rawdot + (-128*d_a*sum))
        const float2 ap = params[(long)blk*N + Xr];
        const float4 c1 = (float4)(ap.x);
        const float4 c2 = (float4)(ap.y);
        global const half * dp = d_trans + (long)blk*M + Z*32;
        float4 t;
        t = mad(convert_float4(r0), c1, c2); f0 = mad(vload_half4(0, dp), t, f0);
        t = mad(convert_float4(r1), c1, c2); f1 = mad(vload_half4(1, dp), t, f1);
        t = mad(convert_float4(r2), c1, c2); f2 = mad(vload_half4(2, dp), t, f2);
        t = mad(convert_float4(r3), c1, c2); f3 = mad(vload_half4(3, dp), t, f3);
        t = mad(convert_float4(r4), c1, c2); f4 = mad(vload_half4(4, dp), t, f4);
        t = mad(convert_float4(r5), c1, c2); f5 = mad(vload_half4(5, dp), t, f5);
        t = mad(convert_float4(r6), c1, c2); f6 = mad(vload_half4(6, dp), t, f6);
        t = mad(convert_float4(r7), c1, c2); f7 = mad(vload_half4(7, dp), t, f7);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (X < N) {
        global float * dst = (global float *)((global char *)dst_void + offsetd);
        global float * o = dst + (long)X*M + Z*32;
        vstore4(f0, 0, o);
        vstore4(f1, 1, o);
        vstore4(f2, 2, o);
        vstore4(f3, 3, o);
        vstore4(f4, 4, o);
        vstore4(f5, 5, o);
        vstore4(f6, 6, o);
        vstore4(f7, 7, o);
    }
}
