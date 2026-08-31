// Fused MoE combine epilogue: replaces the router-weight MUL + the (n_expert_used-1)
// cross-expert ADD chain with ONE weighted-sum-across-experts pass.
//   dst[row, tok] = sum_e experts[row, e, tok] * weights[0, e, tok]
// experts: [n_embd, n_expert_used, n_tokens] f32 (contiguous after down-proj GEMM)
// weights: [1, n_expert_used, n_tokens] f32
// dst:     [n_embd, n_tokens] f32
// One read of experts + one write of dst (eliminates the intermediate weighted
// buffer and the k-1 elementwise add round-trips). Vectorized float4 over rows.
// strides e1/e2/w1/w2/d1 are in ELEMENTS (floats).

// Same weighted sum, with the per-expert bias add folded in.
//
// The MoE down projection's bias is applied by an in-place add_id whose only
// consumer is this combine, so it costs a full read plus a full write of a
// tensor that is read once more immediately afterwards. Reading the raw matmul
// output here and adding the bias row while it is already in registers removes
// that pass. Kept as a separate kernel so the unfused path is untouched.
__kernel void kernel_moe_combine_bias_f32(
        __global const char * e_buf, ulong off_e,
        __global const char * w_buf, ulong off_w,
        __global const char * b_buf, ulong off_b,   // per-expert bias rows
        __global const char * i_buf, ulong off_i,   // expert ids
        __global       char * d_buf, ulong off_d,
        int  n_embd4,            // n_embd / 4
        int  k,                  // n_expert_used
        int  n_tokens,
        uint e1, uint e2,        // experts strides (elements): per-expert, per-token
        uint w1, uint w2,        // weights strides (elements)
        uint d1,                 // dst per-token stride (elements)
        ulong nb_b1,             // bias row stride (bytes)
        ulong nb_i1)             // ids row stride (bytes) - ids is a view, not packed
{
    const uint r4  = get_global_id(0);
    const uint tok = get_global_id(1);
    if (r4 >= (uint)n_embd4 || tok >= (uint)n_tokens) return;

    __global const float * E = (__global const float *)(e_buf + off_e) + tok*e2 + r4*4u;
    __global const float * W = (__global const float *)(w_buf + off_w) + tok*w2;
    __global const char  * B = b_buf + off_b;
    __global const char  * I = i_buf + off_i + (ulong)tok*nb_i1;

    float4 acc = (float4)(0.0f);
    for (int e = 0; e < k; ++e) {
        const int i11 = *((__global const int *)(I + (ulong)e*sizeof(int)));
        __global const float * Brow = (__global const float *)(B + (ulong)i11*nb_b1) + r4*4u;
        const float4 v = vload4(0, E + (uint)e*e1) + vload4(0, Brow);
        acc = mad(v, (float4)(W[(uint)e*w1]), acc);
    }

    __global float * D = (__global float *)(d_buf + off_d) + tok*d1 + r4*4u;
    vstore4(acc, 0, D);
}

__kernel void kernel_moe_combine_f32(
        __global const char * e_buf, ulong off_e,
        __global const char * w_buf, ulong off_w,
        __global       char * d_buf, ulong off_d,
        int  n_embd4,            // n_embd / 4
        int  k,                  // n_expert_used
        int  n_tokens,
        uint e1, uint e2,        // experts strides (elements): per-expert, per-token
        uint w1, uint w2,        // weights strides (elements)
        uint d1)                 // dst per-token stride (elements)
{
    const uint r4  = get_global_id(0);
    const uint tok = get_global_id(1);
    if (r4 >= (uint)n_embd4 || tok >= (uint)n_tokens) return;

    __global const float * E = (__global const float *)(e_buf + off_e) + tok*e2 + r4*4u;
    __global const float * W = (__global const float *)(w_buf + off_w) + tok*w2;

    float4 acc = (float4)(0.0f);
    for (int e = 0; e < k; ++e) {
        acc = mad(vload4(0, E + (uint)e*e1), (float4)(W[(uint)e*w1]), acc);
    }

    __global float * D = (__global float *)(d_buf + off_d) + tok*d1 + r4*4u;
    vstore4(acc, 0, D);
}
