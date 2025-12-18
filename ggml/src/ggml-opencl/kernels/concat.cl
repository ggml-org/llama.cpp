kernel void kernel_concat_f32_contiguous(
    global const char * p_src0, ulong off_src0,
    global const char * p_src1, ulong off_src1,
    global char * p_dst, ulong off_dst,
    int d_ne00, int d_ne01, int d_ne02, // src0->ne[0..2] for the slice
    int d_ne10, int d_ne11, int d_ne12, // src1->ne[0..2] for the slice (d_ne1X must match d_ne0X on non-concat axes)
    int d_ne0,  int d_ne1,  int d_ne2,  // dst->ne[0..2] for the slice
    int dim
) {
    global const float * src0 = (global const float*)((global char*)p_src0 + off_src0);
    global const float * src1 = (global const float*)((global char*)p_src1 + off_src1);
    global float * dst        = (global float*)((global char*)p_dst + off_dst);

    int i0 = get_global_id(0); // Index along dst's 0th dimension
    int i1 = get_global_id(1); // Index along dst's 1st dimension
    int i2 = get_global_id(2); // Index along dst's 2nd dimension

    if (i0 >= d_ne0 || i1 >= d_ne1 || i2 >= d_ne2) {
        return;
    }

    ulong dst_idx = (ulong)i2 * d_ne0 * d_ne1 + (ulong)i1 * d_ne0 + i0;
    ulong src_idx;

    if (dim == 0) {
        if (i0 < d_ne00) { // Data from src0
            src_idx = (ulong)i2 * d_ne00 * d_ne01 + (ulong)i1 * d_ne00 + i0;
            dst[dst_idx] = src0[src_idx];
        } else { // Data from src1
            src_idx = (ulong)i2 * d_ne10 * d_ne11 + (ulong)i1 * d_ne10 + (i0 - d_ne00);
            dst[dst_idx] = src1[src_idx];
        }
    } else if (dim == 1) {
        if (i1 < d_ne01) { // Data from src0
            src_idx = (ulong)i2 * d_ne00 * d_ne01 + (ulong)i1 * d_ne00 + i0;
            dst[dst_idx] = src0[src_idx];
        } else { // Data from src1
            src_idx = (ulong)i2 * d_ne10 * d_ne11 + (ulong)(i1 - d_ne01) * d_ne10 + i0;
            dst[dst_idx] = src1[src_idx];
        }
    } else if (dim == 2) {
        if (i2 < d_ne02) { // Data from src0
            src_idx = (ulong)i2 * d_ne00 * d_ne01 + (ulong)i1 * d_ne00 + i0;
            dst[dst_idx] = src0[src_idx];
        } else { // Data from src1

            src_idx = (ulong)(i2 - d_ne02) * d_ne10 * d_ne11 + (ulong)i1 * d_ne10 + i0;
            dst[dst_idx] = src1[src_idx];
        }
    }
}

kernel void kernel_concat_f32_non_contiguous(
    global const char * p_src0, ulong off_src0,
    global const char * p_src1, ulong off_src1,
    global char * p_dst, ulong off_dst,

    long ne00, long ne01, long ne02, long ne03,
    ulong nb00, ulong nb01, ulong nb02, ulong nb03,

    ulong nb10, ulong nb11, ulong nb12, ulong nb13, // Strides for src1

    long d_ne0, long d_ne1, long d_ne2, long d_ne3,
    ulong d_nb0, ulong d_nb1, ulong d_nb2, ulong d_nb3,
    int dim
) {
    global const char * src0_base = p_src0 + off_src0;
    global const char * src1_base = p_src1 + off_src1;
    global char * dst_base        = p_dst + off_dst;

    long current_i1 = get_global_id(0); // Index for dst_dim_1
    long current_i2 = get_global_id(1); // Index for dst_dim_2
    long current_i3 = get_global_id(2); // Index for dst_dim_3

    if (current_i1 >= d_ne1 || current_i2 >= d_ne2 || current_i3 >= d_ne3) {
        return;
    }

    global const float * x_val_ptr;
    global float * y_val_ptr;

    for (long current_i0 = 0; current_i0 < d_ne0; ++current_i0) {
        bool use_src0;
        long s_i0 = current_i0, s_i1 = current_i1, s_i2 = current_i2, s_i3 = current_i3;

        if (dim == 0) {
            use_src0 = (current_i0 < ne00);
            if (!use_src0) { s_i0 = current_i0 - ne00; }
        } else if (dim == 1) {
            use_src0 = (current_i1 < ne01);
            if (!use_src0) { s_i1 = current_i1 - ne01; }
        } else if (dim == 2) {
            use_src0 = (current_i2 < ne02);
            if (!use_src0) { s_i2 = current_i2 - ne02; }
        } else { // dim == 3
            use_src0 = (current_i3 < ne03);
            if (!use_src0) { s_i3 = current_i3 - ne03; }
        }

        if (use_src0) {
            x_val_ptr = (global const float *)(src0_base + (ulong)s_i3*nb03 + (ulong)s_i2*nb02 + (ulong)s_i1*nb01 + (ulong)s_i0*nb00);
        } else {
            x_val_ptr = (global const float *)(src1_base + (ulong)s_i3*nb13 + (ulong)s_i2*nb12 + (ulong)s_i1*nb11 + (ulong)s_i0*nb10);
        }

        y_val_ptr = (global float *)(dst_base + (ulong)current_i3*d_nb3 + (ulong)current_i2*d_nb2 + (ulong)current_i1*d_nb1 + (ulong)current_i0*d_nb0);
        *y_val_ptr = *x_val_ptr;
    }
}
