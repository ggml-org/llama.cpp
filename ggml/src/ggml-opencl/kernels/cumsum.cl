#pragma OPENCL EXTENSION cl_khr_fp16 : enable
//------------------------------------------------------------------------------
// cumsum
//------------------------------------------------------------------------------
#define MAX_SUBGROUPS 16
kernel void kernel_cumsum_blk(
        global char * src0,
        ulong offset0,
        global char * tmp,
        global char * dst,
        ulong offsetd,
        int   ne00,
        int   ne01,
        int   ne02,
        int   ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        uint net0,
        uint net1,
        uint net2
) {
    src0 = src0 + offset0;
    dst  = dst + offsetd;

    const int i3 = get_group_id(2);
    const int i2 = get_group_id(1);
    const int i1 = get_group_id(0);

    const int nth = get_local_size(0);
    const int tid = get_local_id(0);

    const uint sg_size = get_sub_group_size();
    const uint sg_id = get_sub_group_id();
    const uint sg_lid = get_sub_group_local_id();

    const int ib = i1 / ne01;
    const int i00 = ib * nth;
    const int i01 = i1 % ne01;
    const int i02 = i2;
    const int i03 = i3;

    global const float * src0_row = (global const float *)(src0 + i03*nb03 + i02*nb02 + i01*nb01);
    global       float * tmp_row  = (global float *)tmp + net0 * i01 + net0 * net1 * i02 + net0 * net1 * net2 * i03;
    global       float * dst_row  = (global float *)dst + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    __local float partial[MAX_SUBGROUPS];

    float v = 0.0f;
    if(i00 + tid < ne00){
        v = src0_row[i00 + tid];
    }

    float s = sub_group_scan_inclusive_add(v);
    if(sg_lid == sg_size - 1){
        partial[sg_id] = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(sg_id == 0){
        float x = 0.0f;
        if(sg_lid < get_num_sub_groups()) x = partial[sg_lid];
        float ex = sub_group_scan_exclusive_add(x);
        if(sg_lid < get_num_sub_groups()) partial[sg_lid] = ex;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    s += partial[sg_id];

    if(i00 + tid < ne00){
        dst_row[i00 + tid] = s;
    }
    if(ne00 > nth && tid == nth - 1){
        tmp_row[ib] = s;
    }
}

kernel void kernel_cumsum_add(
        global char * tmp,
        global char * dst,
        ulong offsetd,
        int   ne00,
        int   ne01,
        int   ne02,
        int   ne03,
        uint nbt0,
        uint nbt1,
        uint nbt2,
        uint nbt3
) {
    dst  = dst + offsetd;

    const int i3 = get_group_id(2);
    const int i2 = get_group_id(1);
    const int i1 = get_group_id(0);

    const int nth = get_local_size(0);
    const int tid = get_local_id(0);

    const int ib = i1 / ne01;
    if(ib == 0){
        return;
    }
    const int i00 = ib * nth;
    const int i01 = i1 % ne01;
    const int i02 = i2;
    const int i03 = i3;

    global float * tmp_row  = (global float *)(tmp + nbt1 * i01 + nbt2 * i02 + nbt3 * i03);
    global float * dst_row  = (global float *)dst + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
    
    if(i00 + tid < ne00){
        dst_row[i00 + tid] += tmp_row[ib - 1];
    }
}
