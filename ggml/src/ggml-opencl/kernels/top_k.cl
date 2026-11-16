#ifdef cl_khr_subgroups
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

uint top_k_key(float value) {
    uint bits = as_uint(value);
    if ((bits & 0x7fffffffu) == 0) {
        bits = 0;
    }
    return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}

uint4 top_k_keys(global const char * src, int i, int n, ulong nb0) {
    if (nb0 == sizeof(float) && i + 3 < n) {
        const float4 v = vload4(0, (global const float *) src + i);
        return (uint4) (top_k_key(v.s0), top_k_key(v.s1), top_k_key(v.s2), top_k_key(v.s3));
    }
    return (uint4) (
        i     < n ? top_k_key(*(global const float *) (src + (ulong)  i     *nb0)) : 0,
        i + 1 < n ? top_k_key(*(global const float *) (src + (ulong) (i + 1)*nb0)) : 0,
        i + 2 < n ? top_k_key(*(global const float *) (src + (ulong) (i + 2)*nb0)) : 0,
        i + 3 < n ? top_k_key(*(global const float *) (src + (ulong) (i + 3)*nb0)) : 0);
}

void top_k_store(global int * dst, global uint2 * candidates, global const char * src,
        ulong nb0, int indexed, int base, int i, uint pos) {
    const int index = indexed ? *(global const int *) (src + (ulong) i*nb0 + sizeof(float)) : base + i;
    if (candidates) {
        candidates[pos] = (uint2) (as_uint(*(global const float *) (src + (ulong) i*nb0)), index);
    } else {
        dst[pos] = index;
    }
}

void top_k_select(
        global const char * src,
        global int * dst,
        global uint2 * candidates,
        ulong nb00,
        int ne00,
        int k,
        int base,
        int indexed,
        local uint * hist
) {
    const int tid = get_local_id(0);
    const int nth = get_local_size(0);
    if (k == ne00) {
        for (int i = tid; i < ne00; i += nth) {
            top_k_store(dst, candidates, src, nb00, indexed, base, i, i);
        }
        return;
    }

    uint prefix = 0;
    uint mask = 0;
    uint rank = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += nth) {
            hist[i] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 4*tid; i < ne00; i += 4*nth) {
            const uint4 v = top_k_keys(src, i, ne00, nb00);
            const uint keys[4] = {v.s0, v.s1, v.s2, v.s3};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint key = keys[j];
                if (i + j < ne00 && (key & mask) == prefix) {
                    atomic_inc(hist + ((key >> shift) & 255));
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid == 0) {
            uint above = 0;
            for (int b = 255; b >= 0; --b) {
                uint count = hist[b];
                if (above + count >= rank) {
                    hist[256] = b;
                    hist[257] = rank - above;
                    hist[258] = count;
                    break;
                }
                above += count;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        prefix |= hist[256] << shift;
        rank = hist[257];
        const bool done = hist[258] == rank;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (done) {
            // The whole bucket is selected, so its low bits do not matter.
            break;
        }
        mask |= 255u << shift;
    }

    uint greater = 0;
    uint equal = 0;
    for (int i = 4*tid; i < ne00; i += 4*nth) {
        const uint4 v = top_k_keys(src, i, ne00, nb00);
        const uint keys[4] = {v.s0, v.s1, v.s2, v.s3};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            greater += i + j < ne00 && keys[j] > prefix;
            equal += i + j < ne00 && keys[j] == prefix;
        }
    }
    uint pos_greater;
    uint pos_equal;
#ifdef cl_khr_subgroups
    const uint sg = get_sub_group_id();
    const uint nsg = get_num_sub_groups();
    const uint total_greater = sub_group_reduce_add(greater);
    const uint total_equal = sub_group_reduce_add(equal);
    pos_greater = sub_group_scan_exclusive_add(greater);
    pos_equal = sub_group_scan_exclusive_add(equal);
    if (get_sub_group_local_id() == 0) {
        hist[sg] = total_greater;
        hist[nsg + sg] = total_equal;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = 0; i < nsg; ++i) {
        pos_equal += hist[i];
        if (i < sg) {
            pos_greater += hist[i];
            pos_equal += hist[nsg + i];
        }
    }
#else
    hist[tid] = greater;
    hist[nth + tid] = equal;
    barrier(CLK_LOCAL_MEM_FENCE);
    pos_greater = 0;
    pos_equal = 0;
    for (int i = 0; i < nth; ++i) {
        pos_equal += hist[i];
        if (i < tid) {
            pos_greater += hist[i];
            pos_equal += hist[nth + i];
        }
    }
#endif
    for (int i = 4*tid; i < ne00; i += 4*nth) {
        const uint4 v = top_k_keys(src, i, ne00, nb00);
        const uint keys[4] = {v.s0, v.s1, v.s2, v.s3};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (i + j < ne00) {
                if (keys[j] > prefix) {
                    top_k_store(dst, candidates, src, nb00, indexed, base, i + j, pos_greater++);
                } else if (keys[j] == prefix && pos_equal < k) {
                    top_k_store(dst, candidates, src, nb00, indexed, base, i + j, pos_equal++);
                }
            }
        }
    }
}

kernel void kernel_top_k_f32(
        global const char * src,
        ulong offset0,
        global int * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int k_max,
        local uint * hist,
        global uint2 * candidates,
        int tile_width
) {
    const ulong row = get_group_id(1);
    const int part = get_group_id(0);
    const int parts = get_num_groups(0);
    const int base = part*tile_width;
    const int n = min(ne00 - base, tile_width);
    const int k = min(k_max, n);
    const int out_width = (parts - 1)*k_max + min(k_max, ne00 - (parts - 1)*tile_width);
    src += offset0 + (row % ne01)*nb01 + (row / ne01 % ne02)*nb02 + (row / ne01 / ne02)*nb03 + (ulong) base*nb00;
    if (candidates) {
        candidates += row*out_width + part*k_max;
    } else {
        dst = (global int *) ((global char *) dst + offsetd) + row*out_width + part*k_max;
    }
    top_k_select(src, dst, candidates, nb00, n, k, base, 0, hist);
}

kernel void kernel_top_k_merge_f32(
        global const uint2 * candidates,
        global int * dst,
        ulong offsetd,
        int n_candidates,
        int k,
        local uint * hist
) {
    const ulong row = get_group_id(1);
    candidates += row*n_candidates;
    dst = (global int *) ((global char *) dst + offsetd) + row*k;
    top_k_select((global const char *) candidates, dst, 0, sizeof(uint2), n_candidates, k, 0, 1, hist);
}
