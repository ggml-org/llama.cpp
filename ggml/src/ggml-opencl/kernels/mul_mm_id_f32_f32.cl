#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define BM 64
#define BN 32
#define BK 16
#define TM 4
#define TN 4

kernel void kernel_mul_mm_id_f32_f32(
        global const char * src0, ulong offset0,
        global const char * src1, ulong offset1,
        global const int * routes,
        global const ushort * experts,
        global const int * total_tiles,
        global char * dst, ulong offsetd,
        int ncols, int nrows, int ne11, int nused, int src0_f16,
        ulong nb01, ulong nb02, ulong nb11, ulong nb12, ulong nb1, ulong nb2) {
    const int tile = get_group_id(1);
    if (tile >= total_tiles[0]) {
        return;
    }
    const int tid = get_local_id(0);
    const int row = get_group_id(0)*BM;
    const int mr = tid % (BM/TM)*TM;
    const int nr = tid / (BM/TM)*TN;
    src0 += offset0 + experts[tile]*nb02;
    src1 += offset1;
    dst += offsetd;
    local float a[BM*BK];
    local float b[BN*BK];
    float sums[TM*TN] = {0.0f};

    for (int base = 0; base < ncols; base += BK) {
        for (int i = tid; i < BM*BK/4; i += 128) {
            const int r = i/(BK/4);
            const int k = base + i%(BK/4)*4;
            float4 value = 0.0f;
            if (row + r < nrows && k + 3 < ncols) {
                global const char * arow = src0 + (row + r)*nb01;
                value = src0_f16 ? convert_float4(vload4(0, (global const half *) arow + k)) : vload4(0, (global const float *) arow + k);
            } else if (row + r < nrows) {
                float tail[4] = {0.0f};
                for (int j = 0; j < 4 && k + j < ncols; ++j) {
                    global const char * arow = src0 + (row + r)*nb01;
                    tail[j] = src0_f16 ? (float) ((global const half *) arow)[k + j] : ((global const float *) arow)[k + j];
                }
                value = (float4) (tail[0], tail[1], tail[2], tail[3]);
            }
            a[(i%(BK/4)*4 + 0)*BM + r] = value.s0;
            a[(i%(BK/4)*4 + 1)*BM + r] = value.s1;
            a[(i%(BK/4)*4 + 2)*BM + r] = value.s2;
            a[(i%(BK/4)*4 + 3)*BM + r] = value.s3;
        }
        for (int i = tid; i < BN*BK/4; i += 128) {
            const int route = routes[tile*BN + i/(BK/4)];
            const int k = base + i%(BK/4)*4;
            float4 value = 0.0f;
            if (route >= 0) {
                global const float * brow = (global const float *) (src1 + (route%nused%ne11)*nb11 + (route/nused)*nb12);
                if (k + 3 < ncols) {
                    value = vload4(0, brow + k);
                } else {
                    float tail[4] = {0.0f};
                    for (int j = 0; j < 4 && k + j < ncols; ++j) {
                        tail[j] = brow[k + j];
                    }
                    value = (float4) (tail[0], tail[1], tail[2], tail[3]);
                }
            }
            b[(i%(BK/4)*4 + 0)*BN + i/(BK/4)] = value.s0;
            b[(i%(BK/4)*4 + 1)*BN + i/(BK/4)] = value.s1;
            b[(i%(BK/4)*4 + 2)*BN + i/(BK/4)] = value.s2;
            b[(i%(BK/4)*4 + 3)*BN + i/(BK/4)] = value.s3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                const float bv = b[k*BN + nr + n];
                #pragma unroll
                for (int m = 0; m < TM; ++m) {
                    sums[n*TM + m] = mad(a[k*BM + mr + m], bv, sums[n*TM + m]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int n = 0; n < TN; ++n) {
        const int route = routes[tile*BN + nr + n];
        if (route >= 0) {
            global float * out = (global float *) (dst + (route%nused)*nb1 + (route/nused)*nb2);
            for (int m = 0; m < TM && row + mr + m < nrows; ++m) {
                out[row + mr + m] = sums[n*TM + m];
            }
        }
    }
}
