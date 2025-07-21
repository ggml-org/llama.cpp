#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void pack_b_for_image(
    __global const float* src_b,
    const ulong b_offset,
    __write_only image2d_t dest_img,
    const int K,
    const int N
) {
    const int n_4_idx = get_global_id(0);
    const int k_idx = get_global_id(1);

    const int n_base = n_4_idx << 2;

    if (n_base >= N || k_idx >= K) {
        return;
    }

    __global const float* b_ptr = (__global const float*)((__global const char*)src_b + b_offset);

    half4 vals;
    vals.x = convert_half(b_ptr[n_base * K + k_idx]);
    vals.y = (n_base + 1 < N) ? convert_half(b_ptr[(n_base + 1) * K + k_idx]) : (half)0.0h;
    vals.z = (n_base + 2 < N) ? convert_half(b_ptr[(n_base + 2) * K + k_idx]) : (half)0.0h;
    vals.w = (n_base + 3 < N) ? convert_half(b_ptr[(n_base + 3) * K + k_idx]) : (half)0.0h;

    write_imageh(dest_img, (int2)(n_4_idx, k_idx), vals);
}
