// Integer storage preserves the mask bits without FP16 conversion.
#if MASK_F16
#define MASK_T ushort
#define MASK_V ushort8
#define MASK_NV 8
#define MASK_NINF 0xfc00
#define MASK_STORE vstore8
#else
#define MASK_T uint
#define MASK_V uint4
#define MASK_NV 4
#define MASK_NINF 0xff800000u
#define MASK_STORE vstore4
#endif

kernel void kernel_top_k_mask_fill(
        global char * dst,
        ulong         offsetd,
        ulong         n
) {
    global MASK_T * out = (global MASK_T *) (dst + offsetd);
    const ulong c = get_global_id(0)*MASK_NV;
    if (c + MASK_NV <= n) {
        MASK_STORE((MASK_V) (MASK_NINF), 0, out + c);
    } else {
        for (ulong j = c; j < n; ++j) {
            out[j] = MASK_NINF;
        }
    }
}

kernel void kernel_top_k_mask(
        global const char * mask,
        ulong               offsetm,
        global const char * top_k,
        ulong               offsetk,
        global const char * score,
        ulong               offsets,
        global const char * blocks,
        ulong               offsetb,
        global const char * extra,
        ulong               offsete,
        global       char * dst,
        ulong               offsetd,
        ulong nem0,
        ulong nem1,
        ulong nem2,
        ulong nem3,
        ulong nbm1,
        ulong nbm2,
        ulong nbm3,
        ulong nek0,
        ulong nbk0,
        ulong nbk1,
        ulong nbk2,
        ulong nbk3,
        ulong nes0,
        ulong nbs0,
        ulong nbs1,
        ulong nbs2,
        ulong nbs3,
        ulong neb0,
        ulong neb1,
        ulong nbb0,
        ulong nbb1,
        ulong nee0,
        ulong nbe0,
        ulong nbe1,
        ulong nbe2,
        ulong nbe3,
        ulong ned1,
        ulong ned2,
        ulong nbd1,
        ulong nbd2,
        ulong nbd3,
        int   fill
) {
    const ulong row = get_group_id(1);
    const ulong i1 = row % ned1;
    const ulong i2 = (row / ned1) % ned2;
    const ulong i3 = row / (ned1*ned2);
    const ulong tid = get_local_id(0);
    const ulong nth = get_local_size(0);
    const ulong n_pick = blocks ? neb1 : score ? nes0 : nem0;

    global const MASK_T * mask_row = (global const MASK_T *) (mask + offsetm
        + (i1 % nem1)*nbm1 + (i2 % nem2)*nbm2 + (i3 % nem3)*nbm3);
    global MASK_T * dst_row = (global MASK_T *) (dst + offsetd + i1*nbd1 + i2*nbd2 + i3*nbd3);
    global const char * top_k_row = top_k + offsetk + i1*nbk1 + i2*nbk2 + i3*nbk3;

    if (fill) {
        for (ulong c = tid; c < nem0/MASK_NV; c += nth) {
            MASK_STORE((MASK_V) (MASK_NINF), c, dst_row);
        }
        for (ulong c = nem0/MASK_NV*MASK_NV + tid; c < nem0; c += nth) {
            dst_row[c] = MASK_NINF;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // Repeated cells copy the same mask value, as in the CUDA and Metal kernels.
    for (ulong j = tid; j < nek0 + nee0; j += nth) {
        if (j >= nek0) {
            const int c = *(global const int *) (extra + offsete + (j - nek0)*nbe0 + i1*nbe1 + i2*nbe2 + i3*nbe3);
            if (c >= 0 && (ulong) c < nem0) {
                dst_row[c] = mask_row[c];
            }
            continue;
        }

        const int p = *(global const int *) (top_k_row + j*nbk0);
        if (p < 0 || (ulong) p >= n_pick) {
            continue;
        }
        if (score && !(*(global const float *) (score + offsets + p*nbs0 + i1*nbs1 + i2*nbs2 + i3*nbs3) > -INFINITY)) {
            continue;
        }
        if (blocks) {
            for (ulong k = 0; k < neb0; ++k) {
                const int c = *(global const int *) (blocks + offsetb + k*nbb0 + p*nbb1);
                if (c >= 0 && (ulong) c < nem0) {
                    dst_row[c] = mask_row[c];
                }
            }
        } else if ((ulong) p < nem0) {
            dst_row[p] = mask_row[p];
        }
    }
}
