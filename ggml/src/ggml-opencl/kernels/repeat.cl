kernel void kernel_repeat(
    global const char * src0_data_in,
    global       char * dst_data_in,
    ulong src0_offset,
    ulong dst_offset,
    int src0_ne0, int src0_ne1, int src0_ne2, int src0_ne3,
    ulong src0_nb0, ulong src0_nb1, ulong src0_nb2, ulong src0_nb3,
    int dst_ne0, int dst_ne1, int dst_ne2, int dst_ne3,
    ulong dst_nb0, ulong dst_nb1, ulong dst_nb2, ulong dst_nb3
) {
    global const char * src0_data = src0_data_in + src0_offset;
    global       char * dst_data  = dst_data_in + dst_offset;

    const int d3 = get_global_id(2);
    const int d2 = get_global_id(1);
    const int d1 = get_global_id(0);

    if (d3 >= dst_ne3 || d2 >= dst_ne2 || d1 >= dst_ne1) {
        return;
    }

    const int s3 = d3 % src0_ne3;
    const int s2 = d2 % src0_ne2;
    const int s1 = d1 % src0_ne1;

    const global char * p_src0_slice = src0_data + (ulong)s3*src0_nb3 + (ulong)s2*src0_nb2 + (ulong)s1*src0_nb1;
    global char * p_dst_slice  = dst_data  + (ulong)d3*dst_nb3 + (ulong)d2*dst_nb2 + (ulong)d1*dst_nb1;

    for (int d0 = 0; d0 < dst_ne0; ++d0) {
        // Determine source index for dimension 0 based on tiling/broadcasting.
        const int s0 = d0 % src0_ne0;

        const global char * restrict current_src_el_ptr = p_src0_slice + (ulong)s0*src0_nb0;
        global char * restrict current_dst_el_ptr  = p_dst_slice  + (ulong)d0*dst_nb0;
        for (int k = 0; k < src0_nb0; ++k) {
            current_dst_el_ptr[k] = current_src_el_ptr[k];
        }
    }
}
