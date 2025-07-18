#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void mul_mat_f16_f32_image(
    __read_only image2d_t A_img,
    __read_only image2d_t B_img,
    __global float* C_buf,
    const ulong c_offset,
    const int M,
    const int N,
    const int K
) {
    const int n_4_idx = get_global_id(0);
    const int m_idx = get_global_id(1);

    const int n_base = n_4_idx << 2;

    if (n_base >= N || m_idx >= M) {
        return;
    }

    float4 c_vals = (float4)(0.0f);
    const int K_4 = (K + 3) / 4;

    for (int k_4_idx = 0; k_4_idx < K_4; ++k_4_idx) {
        int k_base = k_4_idx << 2;

        float4 a_vals = convert_float4(read_imageh(A_img, SAMPLER, (int2)(k_4_idx, m_idx)));

        if (k_base < K) {
            float4 b0 = convert_float4(read_imageh(B_img, SAMPLER, (int2)(n_4_idx, k_base + 0)));
            c_vals = mad(a_vals.x, b0, c_vals);
        }
        if (k_base + 1 < K) {
            float4 b1 = convert_float4(read_imageh(B_img, SAMPLER, (int2)(n_4_idx, k_base + 1)));
            c_vals = mad(a_vals.y, b1, c_vals);
        }
        if (k_base + 2 < K) {
            float4 b2 = convert_float4(read_imageh(B_img, SAMPLER, (int2)(n_4_idx, k_base + 2)));
            c_vals = mad(a_vals.z, b2, c_vals);
        }
        if (k_base + 3 < K) {
            float4 b3 = convert_float4(read_imageh(B_img, SAMPLER, (int2)(n_4_idx, k_base + 3)));
            c_vals = mad(a_vals.w, b3, c_vals);
        }
    }

    __global float* C = (__global float*)((__global char*)C_buf + c_offset);

    if (n_base + 3 < N) {
        C[(n_base + 0) * M + m_idx] = c_vals.x;
        C[(n_base + 1) * M + m_idx] = c_vals.y;
        C[(n_base + 2) * M + m_idx] = c_vals.z;
        C[(n_base + 3) * M + m_idx] = c_vals.w;
    } else {
        if (n_base < N) C[n_base * M + m_idx] = c_vals.x;
        if (n_base + 1 < N) C[(n_base + 1) * M + m_idx] = c_vals.y;
        if (n_base + 2 < N) C[(n_base + 2) * M + m_idx] = c_vals.z;
    }
}
