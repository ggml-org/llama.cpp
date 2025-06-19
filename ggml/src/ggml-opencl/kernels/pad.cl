kernel void kernel_pad(
        global const void * src0_ptr,
        ulong src0_offset,
        global void * dst_ptr,
        ulong dst_offset,
        int s_ne0, int s_ne1, int s_ne2,
        int d_ne0, int d_ne1, int d_ne2
) {
    global const float * src0 = (global const float *)((global const char *)src0_ptr + src0_offset);
    global float * dst = (global float *)((global char *)dst_ptr + dst_offset);

    int nidx   = get_global_id(0);
    int idx_d1 = get_group_id(1);
    int idx_d2 = get_group_id(2);

    if (nidx >= d_ne0) {
        return;
    }

    int dst_el_offset = nidx + idx_d1 * d_ne0 + idx_d2 * d_ne0 * d_ne1;

    bool in_src_bounds = (nidx < s_ne0) && (idx_d1 < s_ne1) && (idx_d2 < s_ne2);

    if (in_src_bounds) {
        int src_el_offset = nidx + idx_d1 * s_ne0 + idx_d2 * s_ne0 * s_ne1;
        dst[dst_el_offset] = src0[src_el_offset];
    } else {
        dst[dst_el_offset] = 0.0f;
    }
}
