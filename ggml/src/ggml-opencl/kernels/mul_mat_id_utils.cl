kernel void kernel_moe_zero_dst(
    global const char * ids,
    ulong offset_ids,
    ulong nb_ids1,
    global char * dst,
    ulong offset_dst,
    ulong nb1,
    ulong nb2,
    int ne0
) {
    const size_t i1 = get_group_id(0);
    const size_t i2 = get_group_id(1);
    const int expert = *((global const int *)(ids + offset_ids + i1*sizeof(int) + i2*nb_ids1));
    if (expert != -1) {
        return;
    }

    global float * row = (global float *)(dst + offset_dst + i1*nb1 + i2*nb2);
    for (size_t i0 = get_local_id(2); i0 < (size_t)ne0; i0 += get_local_size(2)) {
        row[i0] = 0.0f;
    }
}
