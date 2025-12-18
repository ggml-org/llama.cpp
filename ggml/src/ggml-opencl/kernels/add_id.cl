#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// add_id
//------------------------------------------------------------------------------
kernel void kernel_add_id(
    global char * src0,
    ulong         offset0,
    global char * src1,
    ulong         offset1,
    global char * src2,
    ulong         offset2,
    global char * dst,
    ulong         offsetd,
    ulong         nb01,
    ulong         nb02,
    ulong         nb11,
    ulong         nb21,
    int           ne0,
    int           ne1
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    src2 = (global char*)((global char*)src2 + offset2);
    dst  = (global char*)((global char*)dst  + offsetd);

    int i1 = get_group_id(0);
    int i2 = get_group_id(1);

    const int i11 = *((global const int *) (src2 + i1*sizeof(int) + i2*nb21));

    const size_t nb1 = ne0 * sizeof(float);
    const size_t nb2 = ne1 * nb1;

    global float * dst_row  = (global float *)((global char *)dst  + i1*nb1 + i2*nb2);
    global float * src0_row = (global float *)((global char *)src0 + i1*nb01 + i2*nb02);
    global float * src1_row = (global float *)((global char *)src1 + i11*nb11);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        dst_row[i0] = src0_row[i0] + src1_row[i0];
    }
}
