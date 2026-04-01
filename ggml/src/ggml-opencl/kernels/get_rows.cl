#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

#define QK4_0                   32

//------------------------------------------------------------------------------
// block_q4_0
//------------------------------------------------------------------------------
struct block_q4_0
{
    half d;
    uint8_t qs[QK4_0 / 2];
};


//------------------------------------------------------------------------------
// dequantize_q4_0_f32, dequantize_q4_0_f16
//------------------------------------------------------------------------------
void dequantize_q4_0_f32(global struct block_q4_0 * xb, short il, float16 * reg) {
    global ushort * qs = ((global ushort *)xb + 1);
    float d1 = il ? (xb->d / 16.h) : xb->d;
    float d2 = d1 / 256.f;
    float md = -8.h * xb->d;
    ushort mask0 = il ? 0x00F0 : 0x000F;
    ushort mask1 = mask0 << 8;

    reg->s0 = d1 * (qs[0] & mask0) + md;
    reg->s1 = d2 * (qs[0] & mask1) + md;

    reg->s2 = d1 * (qs[1] & mask0) + md;
    reg->s3 = d2 * (qs[1] & mask1) + md;

    reg->s4 = d1 * (qs[2] & mask0) + md;
    reg->s5 = d2 * (qs[2] & mask1) + md;

    reg->s6 = d1 * (qs[3] & mask0) + md;
    reg->s7 = d2 * (qs[3] & mask1) + md;

    reg->s8 = d1 * (qs[4] & mask0) + md;
    reg->s9 = d2 * (qs[4] & mask1) + md;

    reg->sa = d1 * (qs[5] & mask0) + md;
    reg->sb = d2 * (qs[5] & mask1) + md;

    reg->sc = d1 * (qs[6] & mask0) + md;
    reg->sd = d2 * (qs[6] & mask1) + md;

    reg->se = d1 * (qs[7] & mask0) + md;
    reg->sf = d2 * (qs[7] & mask1) + md;
}


//------------------------------------------------------------------------------
// get_rows
//
// Dispatch layout (all three kernels):
//   dim 0 : ne10 * num_chunks groups, local_size threads each
//           num_chunks = ceil(ne_per_row / local_size)
//             ne_per_row = ne00/4 for f32/f16 (float4 vectorized)
//             ne_per_row = ne00/16 for q4_0 (float16 block per thread)
//   dim 1 : ne11
//   dim 2 : ne12
//
// Within each workgroup:
//   - grp / num_chunks -> i10 (row index, workgroup-uniform)
//   - grp % num_chunks -> chunk (which slice of the row, workgroup-uniform)
//   - chunk * local_size + lid -> ind (element index in ne_per_row units)
//   - thread 0 reads r from src1 and stores the row byte offsets in
//     local memory so every other thread avoids a redundant global load
//
// f32/f16: each thread processes 4 contiguous elements (float4 / half4→float4).
//          Requires ne00 % 4 == 0, which holds for all practical embedding dims.
//------------------------------------------------------------------------------
kernel void kernel_get_rows_f32(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    local ulong src_row_off;
    local ulong dst_row_off;

    int lsz    = (int)get_local_size(0);
    int grp    = (int)get_group_id(0);
    int i11    = (int)get_group_id(1);
    int i12    = (int)get_group_id(2);
    int lid    = (int)get_local_id(0);

    // Each thread handles 4 floats (float4).  ne00_4 is the number of float4
    // elements per row; both the dispatch and this kernel use ne00_4 as the
    // work unit so num_chunks stays workgroup-uniform (no per-lane divides).
    int ne00_4 = ne00 / 4;
    int num_chunks = (ne00_4 + lsz - 1) / lsz;
    int i10   = grp / num_chunks;
    int chunk = grp % num_chunks;

    // Thread 0 reads r from src1 (one global load per workgroup instead of lsz)
    // and precomputes the row byte offsets shared by all threads.
    if (lid == 0 && i10 < ne10) {
        int r = ((global int *)((global char *)src1
                    + (ulong)i12*nb12 + (ulong)i11*nb11 + (ulong)i10*nb10))[0];
        src_row_off = (ulong)r * nb01 + (ulong)i11 * nb02 + (ulong)i12 * nb03;
        dst_row_off = (ulong)i12 * nb3 + (ulong)i11 * nb2 + (ulong)i10 * nb1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int ind = chunk * lsz + lid;
    if (i10 >= ne10 || ind >= ne00_4) return;

    ((global float4 *)((global char *)dst + dst_row_off))[ind] =
        ((global const float4 *)((global const char *)src0 + src_row_off))[ind];
}

kernel void kernel_get_rows_f16(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    local ulong src_row_off;
    local ulong dst_row_off;

    int lsz    = (int)get_local_size(0);
    int grp    = (int)get_group_id(0);
    int i11    = (int)get_group_id(1);
    int i12    = (int)get_group_id(2);
    int lid    = (int)get_local_id(0);

    // Each thread converts 4 halves to 4 floats (half4 → float4).
    int ne00_4 = ne00 / 4;
    int num_chunks = (ne00_4 + lsz - 1) / lsz;
    int i10   = grp / num_chunks;
    int chunk = grp % num_chunks;

    if (lid == 0 && i10 < ne10) {
        int r = ((global int32_t *)((global char *)src1
                    + (ulong)i12*nb12 + (ulong)i11*nb11 + (ulong)i10*nb10))[0];
        src_row_off = (ulong)r * nb01 + (ulong)i11 * nb02 + (ulong)i12 * nb03;
        dst_row_off = (ulong)i12 * nb3 + (ulong)i11 * nb2 + (ulong)i10 * nb1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int ind = chunk * lsz + lid;
    if (i10 >= ne10 || ind >= ne00_4) return;

    global const half4 *src_row = (global const half4 *)((global const char *)src0 + src_row_off);
    global float4      *dst_row = (global float4 *)((global char *)dst + dst_row_off);
    half4 h = src_row[ind];
    dst_row[ind] = (float4)((float)h.x, (float)h.y, (float)h.z, (float)h.w);
}

kernel void kernel_get_rows_q4_0(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    local ulong src_row_off;
    local ulong dst_row_off;

    const int NL = 2;
    const int ne00_16 = ne00 / 16;  // number of float16 output blocks per row

    int lsz  = (int)get_local_size(0);
    int grp  = (int)get_group_id(0);
    int i11  = (int)get_group_id(1);
    int i12  = (int)get_group_id(2);
    int lid  = (int)get_local_id(0);

    int num_chunks = (ne00_16 + lsz - 1) / lsz;
    int i10   = grp / num_chunks;
    int chunk = grp % num_chunks;

    if (lid == 0 && i10 < ne10) {
        int r = ((global int32_t *)((global char *)src1
                    + (ulong)i12*nb12 + (ulong)i11*nb11 + (ulong)i10*nb10))[0];
        src_row_off = (ulong)r * nb01 + (ulong)i11 * nb02 + (ulong)i12 * nb03;
        dst_row_off = (ulong)i12 * nb3 + (ulong)i11 * nb2 + (ulong)i10 * nb1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int ind = chunk * lsz + lid;
    if (i10 >= ne10 || ind >= ne00_16) return;

    float16 temp;
    dequantize_q4_0_f32(
        ((global struct block_q4_0 *)((global const char *)src0 + src_row_off)) + ind/NL,
        ind%NL, &temp);
    *(((global float16 *)((global char *)dst + dst_row_off)) + ind) = temp;
}
