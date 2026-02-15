#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// expm1
//------------------------------------------------------------------------------
kernel void kernel_expm1_f32_nd(
        global void * p_src0_base,
        ulong off_src0_abs,
        global void * p_dst_base,
        ulong off_dst_abs,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        int ne13,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13
) {
    int i0 = get_global_id(0);
    int i1 = get_global_id(1);
    int i2 = get_global_id(2);

    if (i0 < ne10 && i1 < ne11 && i2 < ne12) {
        for (int i3 = 0; i3 < ne13; ++i3) {
            ulong src_offset_in_tensor = (ulong)i0*nb00 + (ulong)i1*nb01 + (ulong)i2*nb02 + (ulong)i3*nb03;
            global const float *src_val_ptr = (global const float *)((global char *)p_src0_base + off_src0_abs + src_offset_in_tensor);

            ulong dst_offset_in_tensor = (ulong)i0*nb10 + (ulong)i1*nb11 + (ulong)i2*nb12 + (ulong)i3*nb13;
            global float *dst_val_ptr = (global float *)((global char *)p_dst_base + off_dst_abs + dst_offset_in_tensor);

            *dst_val_ptr = exp(*src_val_ptr) - 1;
        }
    }
}

kernel void kernel_expm1_f16_nd(
        global void * p_src0_base,
        ulong off_src0_abs,
        global void * p_dst_base,
        ulong off_dst_abs,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        int ne13,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13
) {
    int i0 = get_global_id(0);
    int i1 = get_global_id(1);
    int i2 = get_global_id(2);

    if (i0 < ne10 && i1 < ne11 && i2 < ne12) {
        for (int i3 = 0; i3 < ne13; ++i3) {
            ulong src_offset_in_tensor = (ulong)i0*nb00 + (ulong)i1*nb01 + (ulong)i2*nb02 + (ulong)i3*nb03;
            global const half *src_val_ptr = (global const half *)((global char *)p_src0_base + off_src0_abs + src_offset_in_tensor);

            ulong dst_offset_in_tensor = (ulong)i0*nb10 + (ulong)i1*nb11 + (ulong)i2*nb12 + (ulong)i3*nb13;
            global half *dst_val_ptr = (global half *)((global char *)p_dst_base + off_dst_abs + dst_offset_in_tensor);

            *dst_val_ptr = exp(*src_val_ptr) - 1;
        }
    }
}
