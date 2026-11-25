#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_fwht(
        global const char * src,
        ulong offset_src,
        global char * dst,
        ulong offset_dst,
        int n,
        ulong ne1,
        ulong ne2,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        float scale,
        local float * work) {
    const ulong row = get_group_id(1);
    const ulong i1 = row % ne1;
    const ulong i2 = row / ne1 % ne2;
    const ulong i3 = row / ne1 / ne2;
    global const SRC_TYPE * x = (global const SRC_TYPE *)(src + offset_src + i1*nb1 + i2*nb2 + i3*nb3);
    global float * y = (global float *)(dst + offset_dst) + row*n;
    const int tid = get_local_id(0);
    const int nth = get_local_size(0);

    for (int i = tid; i < n; i += nth) {
        work[i] = (float)x[i] * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int h = 1; h < n; h *= 2) {
        for (int i = tid; i < n/2; i += nth) {
            const int j = 2*i - (i & (h - 1));
            const float a = work[j];
            const float b = work[j + h];
            work[j] = a + b;
            work[j + h] = a - b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = tid; i < n; i += nth) {
        y[i] = work[i];
    }
}
