#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Store 32-element blocks as separate int8 values and half scales.

kernel void kernel_fa_q8_rows_f16(
        global const char * src, ulong off_src, ulong nb1, ulong nb2,
        global char * dq, global half * dd,
        int ne0, int nrow, int nhead
) {
    const int nblk = ne0 / 32;
    const int gid  = get_global_id(0);
    if (gid >= nblk*nrow*nhead) {
        return;
    }
    const int b = gid % nblk;
    const int r = (gid / nblk) % nrow;
    const int h = gid / (nblk*nrow);

    global const half * p = (global const half *)(src + off_src + (ulong)h*nb2 + (ulong)r*nb1) + b*32;
    float v[32];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 x = convert_float4(vload4(i, p));
        v[i*4+0] = x.s0;
        v[i*4+1] = x.s1;
        v[i*4+2] = x.s2;
        v[i*4+3] = x.s3;
        amax = fmax(amax, fmax(fmax(fabs(x.s0), fabs(x.s1)), fmax(fabs(x.s2), fabs(x.s3))));
    }
    const float d  = amax / 127.0f;
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    const size_t row = (size_t)h*nrow + r;
    dd[row*nblk + b] = (half)d;
    global char * q = dq + row*ne0 + b*32;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        q[i] = (char)clamp((int)rint(v[i]*id), -127, 127);
    }
}

kernel void kernel_fa_q8_rows_f32(
        global const char * src, ulong off_src, ulong nb1, ulong nb2,
        global char * dq, global half * dd,
        int ne0, int nrow, int nhead
) {
    const int nblk = ne0 / 32;
    const int gid  = get_global_id(0);
    if (gid >= nblk*nrow*nhead) {
        return;
    }
    const int b = gid % nblk;
    const int r = (gid / nblk) % nrow;
    const int h = gid / (nblk*nrow);

    global const float * p = (global const float *)(src + off_src + (ulong)h*nb2 + (ulong)r*nb1) + b*32;
    float v[32];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 x = vload4(i, p);
        v[i*4+0] = x.s0;
        v[i*4+1] = x.s1;
        v[i*4+2] = x.s2;
        v[i*4+3] = x.s3;
        amax = fmax(amax, fmax(fmax(fabs(x.s0), fabs(x.s1)), fmax(fabs(x.s2), fabs(x.s3))));
    }
    const float d  = amax / 127.0f;
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    const size_t row = (size_t)h*nrow + r;
    dd[row*nblk + b] = (half)d;
    global char * q = dq + row*ne0 + b*32;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        q[i] = (char)clamp((int)rint(v[i]*id), -127, 127);
    }
}
