kernel void kernel_l2_norm_f32(
        global void * src0,
        ulong offset0,
        global void * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        ulong db1,
        ulong db2,
        ulong db3,
        float eps,
        local float * sum
) {
    src0 = (global void*)((global char*)src0 + offset0);
    dst = (global void*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float * x = (global float *) ((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);
    global float * y = (global float *) ((global char *) dst + i03*db3 + i02*db2 + i01*db1);

    float sumf = 0;

    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

    for (int i00 = lid; i00 < ne00; i00 += lsize) {
        sumf += x[i00] * x[i00];
    }
    sum[lid] = sumf;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize / 2; stride > 0; stride /= 2) {
       if (lid < stride) {
           sum[lid] += sum[lid + stride];
       }
       barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float scale = 1.0f/fmax(sqrt(sum[0]), eps);

    for (int i00 = lid; i00 < ne00; i00 += lsize) {
        y[i00] = x[i00] * scale;
    }
}
