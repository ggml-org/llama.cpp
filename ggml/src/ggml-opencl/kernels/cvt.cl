//------------------------------------------------------------------------------
// This file is contains kernels for data conversion.
// These kernels are used when loading the model, so its performance is less
// important.
//------------------------------------------------------------------------------
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#define QK4_0                   32
#define QR4_0                   2
#define QK4_1                   32
#define QR4_1                   2
#define QK5_0                   32
#define QR5_0                   2
#define QK5_1                   32
#define QR5_1                   2
#define QK8_0                   32
#define QR8_0                   1
#define QK_K                    256
#define K_QUANTS_PER_ITERATION  2

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

//------------------------------------------------------------------------------
// block_q4_0
//------------------------------------------------------------------------------
struct block_q4_0
{
    half d;
    uint8_t qs[QK4_0 / 2];
};

//------------------------------------------------------------------------------
// kernel_convert_block_q4_0
// Convert the block_q4_0 format to 2 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q4_0(
    global struct block_q4_0 * src0,
    global uchar * dst_q,
    global half  * dst_d
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK4_0/2; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q4_0(
    global uchar * src_q,
    global half  * src_d,
    global struct block_q4_0 * dst
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) src_d + get_global_id(0);

    b->d = *d;
    for (int i = 0; i < QK4_0/2; ++i) {
        b->qs[i] = q[i];
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_q4_0_noshuffle
// Flatten q4_0 weights and unshuffle the bits
//------------------------------------------------------------------------------

kernel void kernel_convert_block_q4_0_noshuffle(
    global struct block_q4_0 * src0,
    global uchar * dst_q,
    global half  * dst_d
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;
    for (int i = 0; i < QK4_0/4; ++i) {
        uchar x0 = b->qs[2*i + 0];
        uchar x1 = b->qs[2*i + 1];

        q[i + 0      ] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
        q[i + QK4_0/4] = convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);

#ifdef ADRENO_GPU
        // Workaround for adreno - must have the following printf statement for
        // the kernel to work properly. Otherwise it produces incorrect result.
        // convert_uchar above also seems necessary.
        // Compare against a large number so that it does not print anything.
        // get_sub_group_local_id() also works.
        if (get_global_id(0) == 65536*4096) {
            printf("%04x - %02x\n", *(global ushort*)d, ((x0 & 0xF0) >> 4) | (x1 & 0xF0));
        }
#endif
    }
}

//------------------------------------------------------------------------------
// block_mxfp4
//------------------------------------------------------------------------------
#define QK_MXFP4 32
struct block_mxfp4 {
    uchar e; // E8M0
    uchar qs[QK_MXFP4 / 2];
};

//------------------------------------------------------------------------------
// kernel_convert_block_mxfp4
// Convert the block_mxfp4 format to 2 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_mxfp4(
    global struct block_mxfp4 * src0,
    global uchar * dst_q,
    global uchar * dst_e
) {
    global struct block_mxfp4 * b = (global struct block_mxfp4 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK_MXFP4 / 2 * get_global_id(0);
    global uchar * e = (global uchar *) dst_e + get_global_id(0);

    *e = b->e;

    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_mxfp4(
    global uchar * src_q,
    global half  * src_e,
    global struct block_mxfp4 * dst
) {
    global struct block_mxfp4 * b = (global struct block_mxfp4 *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK_MXFP4 / 2 * get_global_id(0);
    global uchar * e = (global uchar *) src_e + get_global_id(0);

    b->e = *e;
    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
        b->qs[i] = q[i];
    }
}

//------------------------------------------------------------------------------
// block_q8_0
//------------------------------------------------------------------------------
typedef struct {
    half d;       // delta
    char qs[QK8_0]; // quants
} block_q8_0;

kernel void kernel_convert_block_q8_0(
    global block_q8_0 * src0,
    global uchar * dst_q,
    global half  * dst_d
) {
    global block_q8_0 * b = (global block_q8_0 *) src0 + get_global_id(0);
    global uchar      * q = (global uchar *) dst_q + QK8_0*get_global_id(0);
    global half       * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK8_0; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q8_0(
    global uchar * src_q,
    global half  * src_d,
    global block_q8_0 * dst
) {
    global block_q8_0 * b = (global block_q8_0 *) dst + get_global_id(0);
    global uchar      * q = (global uchar *) src_q + QK8_0*get_global_id(0);
    global half       * d = (global half *) src_d + get_global_id(0);

    b->d = *d;
    for (int i = 0; i < QK8_0; ++i) {
        b->qs[i] = q[i];
    }
}
