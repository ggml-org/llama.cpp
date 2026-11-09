__kernel void kernel_arange_f32(
        __global float * dst,
        ulong offsetd,
        int n,
        float start,
        float step
) {
    dst = (global float*)((global char*)dst + offsetd);

    const int i = get_global_id(0);
    if (i < n) {
        dst[i] = start + step * i;
    }
}
