#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

__attribute__((qcom_reqd_sub_group_size("half")))
__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void kernel_fused_qknorm_rope_f32(
        global const char * src,
        ulong4              offsets,
        ulong4              strides,
        global const char * weight_q,
        ulong               weight_q_offset,
        global const char * weight_k,
        ulong               weight_k_offset,
        global const char * pe,
        ulong               pe_offset,
        ulong               pe_stride,
        global char       * dst_q,
        ulong               dst_q_offset,
        global char       * dst_k,
        ulong               dst_k_offset,
        global char       * dst_v,
        ulong               dst_v_offset,
        int                 d_head,
        int                 n_head,
        int                 n_tokens,
        int                 dst_tokens,
        int                 token_offset,
        float               eps_q,
        float               eps_k,
        float               scale_k,
        float               scale_v) {
    const int head  = get_group_id(0);
    const int token = get_group_id(1) % n_tokens;
    const int batch = get_group_id(1) / n_tokens;
    const int lane  = get_local_id(0);
    const int pairs = d_head / 2;
    const ulong row = head * strides.s1 + token * strides.s2 + batch * strides.s3;

    global const float * q = (global const float *) (src + offsets.s0 + row);
    global const float * k = (global const float *) (src + offsets.s1 + row);
    global const float * v = (global const float *) (src + offsets.s2 + row);
    global const float * wq = (global const float *) (weight_q + weight_q_offset);
    global const float * wk = (global const float *) (weight_k + weight_k_offset);
    global const float * r = (global const float *) (pe + pe_offset + (token_offset + token) * pe_stride);

    const float2 q0 = lane < pairs ? vload2(lane, q) : (float2) (0.0f);
    const float2 k0 = lane < pairs ? vload2(lane, k) : (float2) (0.0f);
    float sq = dot(q0, q0);
    float sk = dot(k0, k0);
    for (int p = lane + 64; p < pairs; p += 64) {
        const float2 qp = vload2(p, q);
        const float2 kp = vload2(p, k);
        sq += dot(qp, qp);
        sk += dot(kp, kp);
    }
    const float iq = rsqrt(sub_group_reduce_add(sq) / d_head + eps_q);
    const float ik = rsqrt(sub_group_reduce_add(sk) / d_head + eps_k);
    const ulong out = ((ulong) batch * n_head + head) * dst_tokens * d_head +
                      (ulong) (token_offset + token) * d_head;

    global float * oq = (global float *) (dst_q + dst_q_offset) + out;
    global half  * ok = (global half  *) (dst_k + dst_k_offset) + out;
    global half  * ov = (global half  *) (dst_v + dst_v_offset) + out;
    for (int p = lane; p < pairs; p += 64) {
        const float2 qn = (p == lane ? q0 : vload2(p, q)) * iq * vload2(p, wq);
        const float2 kn = (p == lane ? k0 : vload2(p, k)) * ik * vload2(p, wk);
        const float4 rot = vload4(p, r);
        const float2 qr = qn.x * rot.xz + qn.y * rot.yw;
        const float2 kr = kn.x * rot.xz + kn.y * rot.yw;
        vstore2(qr, p, oq);
        vstore2(convert_half2(kr * scale_k), p, ok);
        vstore2(convert_half2(vload2(p, v) * scale_v), p, ov);
    }
}
