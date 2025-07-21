#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void pack_a_for_image(
    __global const half* src_a,
    const ulong a_offset,
    __write_only image2d_t dest_img,
    const int M,
    const int K
) {
    const int k_4_idx = get_global_id(0);
    const int m_idx = get_global_id(1);

    const int k_base = k_4_idx << 2;

    if (k_base >= K || m_idx >= M) {
        return;
    }

    __global const half* a_ptr = (__global const half*)((__global const char*)src_a + a_offset);
    const int a_idx_base = m_idx * K + k_base;

    half4 vals;
    vals.x = a_ptr[a_idx_base];
    vals.y = (k_base + 1 < K) ? a_ptr[a_idx_base + 1] : (half)0.0h;
    vals.z = (k_base + 2 < K) ? a_ptr[a_idx_base + 2] : (half)0.0h;
    vals.w = (k_base + 3 < K) ? a_ptr[a_idx_base + 3] : (half)0.0h;

    write_imageh(dest_img, (int2)(k_4_idx, m_idx), vals);
}
