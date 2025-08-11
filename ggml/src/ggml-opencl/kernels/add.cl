#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// add
//------------------------------------------------------------------------------

// general-purpose kernel for addition of two tensors
// pros: works for non-contiguous tensors, supports broadcast across dims 1, 2 and 3
// cons: not very efficient
kernel void kernel_add(
        global char * src0,
        ulong  offset0,
        global char * src1,
        ulong  offset1,
        global char * dst,
        ulong  offsetd,
        int   ne00,
        int   ne01,
        int   ne02,
        int   ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int   ne10,
        int   ne11,
        int   ne12,
        int   ne13,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int   ne0,
        int   ne1,
        int   ne2,
        int   ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst = dst + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int i13 = i03 % ne13;
    int i12 = i02 % ne12;
    int i11 = i01 % ne11;

    global char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    global char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    global char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const int i10 = i0 % ne10;
        *((global float *)(dst_ptr + i0*nb0)) = *((global float *)(src0_ptr + i0*nb00)) + *((global float *)(src1_ptr + i10*nb10));
    }
}

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_add_row(
        global float4 * src0,
        ulong  offset0,
        global float4 * src1,
        ulong  offset1,
        global float4 * dst,
        ulong  offsetd,
        int ne
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    src1 = (global float4*)((global char*)src1 + offset1);
    dst = (global float4*)((global char*)dst + offsetd);

    // This performs better than using %.
    uint gid = get_global_id(0);
    uint idx1 = gid - (gid/ne)*ne; // get_global_id(0) % ne
    dst[gid] = src0[gid] + src1[idx1];
}

kernel void kernel_add_f16(
        global char * src0,
        ulong  offset0,
        global char * src1,
        ulong  offset1,
        global char * dst,
        ulong  offsetd,
        int   ne00,
        int   ne01,
        int   ne02,
        int   ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int   ne10,
        int   ne11,
        int   ne12,
        int   ne13,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int   ne0,
        int   ne1,
        int   ne2,
        int   ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst = dst + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int i13 = i03 % ne13;
    int i12 = i02 % ne12;
    int i11 = i01 % ne11;

    global char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    global char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    global char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const int i10 = i0 % ne10;
        *((global half *)(dst_ptr + i0*nb0)) = *((global half *)(src0_ptr + i0*nb00)) + *((global half *)(src1_ptr + i10*nb10));
    }
}

kernel void kernel_add_row_f16(
        global half4 * src0,
        ulong  offset0,
        global half4 * src1,
        ulong  offset1,
        global half4 * dst,
        ulong  offsetd,
        int ne
) {
    src0 = (global half4*)((global char*)src0 + offset0);
    src1 = (global half4*)((global char*)src1 + offset1);
    dst = (global half4*)((global char*)dst + offsetd);

    // This performs better than using %.
    uint gid = get_global_id(0);
    uint idx1 = gid - (gid/ne)*ne; // get_global_id(0) % ne
    dst[gid] = src0[gid] + src1[idx1];
}
