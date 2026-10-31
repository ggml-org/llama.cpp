#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void flash_attn_kv_pad_f16(
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * k_pad_void,
    global void * v_pad_void,
    const int n_kv,
    const int n_head_kv,
    const int n_batch,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3
) {
    const int row_idx = get_global_id(0);
    const int head_kv_idx = get_global_id(1);
    const int batch_idx = get_global_id(2);

    if (row_idx >= BLOCK_N || head_kv_idx >= n_head_kv || batch_idx >= n_batch) {
        return;
    }

    const int tail_start = n_kv - (n_kv % BLOCK_N);
    const int src_row_idx = tail_start + row_idx;

    const global char * k_src = (const global char *) k_void + k_offset;
    const global char * v_src = (const global char *) v_void + v_offset;
    global char * k_pad = (global char *) k_pad_void;
    global char * v_pad = (global char *) v_pad_void;

    const ulong k_dst_offset = ((ulong) batch_idx * (ulong) n_head_kv + (ulong) head_kv_idx) * ((ulong) BLOCK_N * k_nb1) + (ulong) row_idx * k_nb1;
    const ulong v_dst_offset = ((ulong) batch_idx * (ulong) n_head_kv + (ulong) head_kv_idx) * ((ulong) BLOCK_N * v_nb1) + (ulong) row_idx * v_nb1;

    if (src_row_idx < n_kv) {
        const ulong k_src_offset = (ulong) batch_idx * k_nb3 + (ulong) head_kv_idx * k_nb2 + (ulong) src_row_idx * k_nb1;
        const ulong v_src_offset = (ulong) batch_idx * v_nb3 + (ulong) head_kv_idx * v_nb2 + (ulong) src_row_idx * v_nb1;

        for (ulong i = 0; i < k_nb1; ++i) {
            k_pad[k_dst_offset + i] = k_src[k_src_offset + i];
        }
        for (ulong i = 0; i < v_nb1; ++i) {
            v_pad[v_dst_offset + i] = v_src[v_src_offset + i];
        }
    } else {
        for (ulong i = 0; i < k_nb1; ++i) {
            k_pad[k_dst_offset + i] = 0;
        }
        for (ulong i = 0; i < v_nb1; ++i) {
            v_pad[v_dst_offset + i] = 0;
        }
    }
}

__kernel void flash_attn_mask_pad_f16(
    const global void * mask_void, ulong mask_offset,
    global void * mask_pad_void,
    const int n_q,
    const int n_kv,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3
) {
    const int col_idx = get_global_id(0);
    const int q_row = get_global_id(1);
    const int mask_slice = get_global_id(2);

    if (col_idx >= BLOCK_N || q_row >= n_q || mask_slice >= mask_ne2 * mask_ne3) {
        return;
    }

    const int tail_start = n_kv - (n_kv % BLOCK_N);
    const int src_col_idx = tail_start + col_idx;
    const int mask_head_idx = mask_slice % mask_ne2;
    const int mask_batch_idx = mask_slice / mask_ne2;

    const global char * mask_src_base = (const global char *) mask_void + mask_offset +
        (ulong) mask_batch_idx * mask_nb3 +
        (ulong) mask_head_idx * mask_nb2 +
        (ulong) q_row * mask_nb1;
    const global half * mask_src = (const global half *) mask_src_base;

    global half * mask_pad = (global half *) mask_pad_void;
    const ulong dst_idx =
        (((ulong) mask_batch_idx * (ulong) mask_ne2 + (ulong) mask_head_idx) * (ulong) n_q + (ulong) q_row) * (ulong) BLOCK_N +
        (ulong) col_idx;

    mask_pad[dst_idx] = src_col_idx < n_kv ? mask_src[src_col_idx] : (half) (-INFINITY);
}

// Per-KV-tile mask class. 0=all -inf (skip tile), 1=mixed (apply mask),
// 2=all zero, no -inf (skip mask lookup). Causal diagonal tiles are class 1.
__kernel void flash_attn_blk_f16(
    const global void * mask_void, ulong mask_offset,
    global char * blk,
    const int n_q,
    const int n_kv,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3
) {
    const int kv_block_idx = get_global_id(0);
    const int q_block_idx = get_global_id(1);
    const int mask_slice = get_global_id(2);

    const int n_q_blocks = (n_q + BLOCK_M - 1) / BLOCK_M;
    const int n_kv_blocks = (n_kv + BLOCK_N - 1) / BLOCK_N;
    if (kv_block_idx >= n_kv_blocks || q_block_idx >= n_q_blocks || mask_slice >= mask_ne2 * mask_ne3) {
        return;
    }

    const int mask_head_idx = mask_slice % mask_ne2;
    const int mask_batch_idx = mask_slice / mask_ne2;
    const int q_start = q_block_idx * BLOCK_M;
    const int k_start = kv_block_idx * BLOCK_N;
    const int q_count = min(BLOCK_M, n_q - q_start);
    const int k_count = min(BLOCK_N, n_kv - k_start);

    const half neg_max_half = (half) (-65504.0f);
    char has_unmasked = 0;
    char has_masked = 0;
    char has_nonzero = 0;

    const global char * mask_base = (const global char *) mask_void + mask_offset +
        (ulong) mask_batch_idx * mask_nb3 +
        (ulong) mask_head_idx * mask_nb2;

    for (int qi = 0; qi < q_count; ++qi) {
        const global half * mask_row = (const global half *) (mask_base + (ulong) (q_start + qi) * mask_nb1) + k_start;
        for (int ki = 0; ki < k_count; ++ki) {
            const half v = mask_row[ki];
            if (v <= neg_max_half) {
                has_masked = 1;
            } else {
                has_unmasked = 1;
                if (v != (half) 0.0f) {
                    has_nonzero = 1;
                }
            }
        }
        if (has_masked && has_unmasked) break;  // mixed tile — short-circuit.
    }

    char res;
    if (has_unmasked == 0) {
        res = 0;
    } else if (has_masked || has_nonzero) {
        res = 1;
    } else {
        res = 2;
    }

    blk[((ulong) mask_slice * (ulong) n_q_blocks + (ulong) q_block_idx) * (ulong) n_kv_blocks + (ulong) kv_block_idx] = res;
}

// V transpose for the decomposed prefill: FA gets V as [dv, n_kv, n_head_kv], the KQV GEMM
// (mul_mm_f16_f32_kqv) wants [n_kv, dv] with n_kv contiguous. Tiled through local memory so reads
// and writes are both coalesced; the +1 row padding avoids bank conflicts.
#define FA_VT_TILE 32
#define FA_VT_ROWS 8
__kernel void flash_attn_v_transpose_f16(
    const global void * v_void, ulong v_offset,
    global void * vt_void,
    const int n_kv,
    const int dv,
    const int n_head_kv,
    const ulong v_nb1,
    const ulong v_nb2,
    const ulong v_nb3
) {
    __local half tile[FA_VT_TILE][FA_VT_TILE + 1];

    const int d0 = get_group_id(0) * FA_VT_TILE;
    const int k0 = get_group_id(1) * FA_VT_TILE;
    const int head_kv_idx = get_group_id(2) % n_head_kv;
    const int batch_idx   = get_group_id(2) / n_head_kv;

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    const global char * v_head = (const global char *) v_void + v_offset +
        (ulong) batch_idx * v_nb3 + (ulong) head_kv_idx * v_nb2;

    for (int r = 0; r < FA_VT_TILE; r += FA_VT_ROWS) {
        const int d  = d0 + lx;
        const int kv = k0 + ly + r;
        half val = (half) 0.0f;
        if (d < dv && kv < n_kv) {
            val = ((const global half *) (v_head + (ulong) kv * v_nb1))[d];
        }
        tile[ly + r][lx] = val;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    global half * vt_head = (global half *) vt_void +
        ((ulong) batch_idx * (ulong) n_head_kv + (ulong) head_kv_idx) * (ulong) n_kv * (ulong) dv;

    for (int r = 0; r < FA_VT_TILE; r += FA_VT_ROWS) {
        const int kv = k0 + lx;
        const int d  = d0 + ly + r;
        if (d < dv && kv < n_kv) {
            vt_head[(ulong) d * (ulong) n_kv + (ulong) kv] = tile[lx][ly + r];
        }
    }
}

// V transpose that emits int8 + a per-32-block scale instead of f16, for the dp4a KQV path.
// A tile row (FA_VT_TILE = 32 along kv, the contiguous output axis) is exactly one quantisation
// block, so the scales come from the existing staging. Lane t owns block d = d0+t and reads
// tile[0..31][t], a conflict-free column walk of the padded LDS tile.
__kernel void flash_attn_v_transpose_q8(
    const global void * v_void, ulong v_offset,
    global void * vt_q_void,            // char, [n_head_kv][dv][n_kv]
    global void * vt_d_void,            // half, [n_head_kv][dv][n_kv/32]
    const int n_kv,
    const int dv,
    const int n_head_kv,
    const ulong v_nb1,
    const ulong v_nb2,
    const ulong v_nb3,
    const int kv_pitch                  // bytes per V^T row, >= n_kv, multiple of 32
) {
    __local half tile[FA_VT_TILE][FA_VT_TILE + 1];

    const int d0 = get_group_id(0) * FA_VT_TILE;
    const int k0 = get_group_id(1) * FA_VT_TILE;
    const int head_kv_idx = get_group_id(2) % n_head_kv;
    const int batch_idx   = get_group_id(2) / n_head_kv;

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    const global char * v_head = (const global char *) v_void + v_offset +
        (ulong) batch_idx * v_nb3 + (ulong) head_kv_idx * v_nb2;

    for (int r = 0; r < FA_VT_TILE; r += FA_VT_ROWS) {
        const int d  = d0 + lx;
        const int kv = k0 + ly + r;
        half val = (half) 0.0f;
        if (d < dv && kv < n_kv) {
            val = ((const global half *) (v_head + (ulong) kv * v_nb1))[d];
        }
        tile[ly + r][lx] = val;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int lid  = ly * FA_VT_TILE + lx;
    const int nblk = kv_pitch / FA_VT_TILE;

    if (lid < FA_VT_TILE) {
        const int d = d0 + lid;
        if (d < dv) {
            const ulong hbase = ((ulong) batch_idx * (ulong) n_head_kv + (ulong) head_kv_idx);

            global char * qrow = (global char *) vt_q_void + hbase*(ulong)kv_pitch*(ulong)dv + (ulong) d * (ulong) kv_pitch;
            global half * drow = (global half *) vt_d_void + hbase*(ulong)nblk*(ulong)dv + (ulong) d * (ulong) nblk;

            float v[FA_VT_TILE];
            float amax = 0.0f;

            for (int i = 0; i < FA_VT_TILE; ++i) {
                v[i] = (float) tile[i][lid];
                amax = fmax(amax, fabs(v[i]));
            }

            const float dq = amax / 127.0f;
            const float id = amax > 0.0f ? 127.0f / amax : 0.0f;

            drow[k0 / FA_VT_TILE] = (half) dq;

            for (int i = 0; i < FA_VT_TILE; ++i) {
                const int kv = k0 + i;
                if (kv < n_kv) {
                    qrow[kv] = (char) clamp((int) rint(v[i] * id), -127, 127);
                }
            }
        }
    }
}

// The same int8 V^T from a q8_0 V cache. The cache blocks run along dv and the KQV contraction
// runs along n_kv, so the scales have to be rebuilt along n_kv; the q8_0 values are expanded in
// registers (the tile holds d*q as float) and requantised straight into V^T, never through an
// f16 copy of V. A tile column (FA_VT_TILE = 32 consecutive d) is exactly one q8_0 block, so
// each row load is one scale plus one byte per lane.
__kernel void flash_attn_v_transpose_q8_q8_0(
    const global void * v_void, ulong v_offset,
    global void * vt_q_void,            // char, [n_head_kv][dv][n_kv]
    global void * vt_d_void,            // half, [n_head_kv][dv][n_kv/32]
    const int n_kv,
    const int dv,
    const int n_head_kv,
    const ulong v_nb1,
    const ulong v_nb2,
    const ulong v_nb3,
    const int kv_pitch                  // bytes per V^T row, >= n_kv, multiple of 32
) {
    __local float tile[FA_VT_TILE][FA_VT_TILE + 1];

    const int d0 = get_group_id(0) * FA_VT_TILE;
    const int k0 = get_group_id(1) * FA_VT_TILE;
    const int head_kv_idx = get_group_id(2) % n_head_kv;
    const int batch_idx   = get_group_id(2) / n_head_kv;

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    const global char * v_head = (const global char *) v_void + v_offset +
        (ulong) batch_idx * v_nb3 + (ulong) head_kv_idx * v_nb2;

    for (int r = 0; r < FA_VT_TILE; r += FA_VT_ROWS) {
        const int d  = d0 + lx;
        const int kv = k0 + ly + r;
        float val = 0.0f;
        if (d < dv && kv < n_kv) {
            const global char * blk = v_head + (ulong) kv * v_nb1 + (ulong) (d / 32) * 34;
            val = vload_half(0, (const global half *) blk) * (float) blk[2 + (d % 32)];
        }
        tile[ly + r][lx] = val;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int lid  = ly * FA_VT_TILE + lx;
    const int nblk = kv_pitch / FA_VT_TILE;

    if (lid < FA_VT_TILE) {
        const int d = d0 + lid;
        if (d < dv) {
            const ulong hbase = ((ulong) batch_idx * (ulong) n_head_kv + (ulong) head_kv_idx);

            global char * qrow = (global char *) vt_q_void + hbase*(ulong)kv_pitch*(ulong)dv + (ulong) d * (ulong) kv_pitch;
            global half * drow = (global half *) vt_d_void + hbase*(ulong)nblk*(ulong)dv + (ulong) d * (ulong) nblk;

            float amax = 0.0f;
            for (int i = 0; i < FA_VT_TILE; ++i) {
                amax = fmax(amax, fabs(tile[i][lid]));
            }

            const float dq = amax / 127.0f;
            const float id = amax > 0.0f ? 127.0f / amax : 0.0f;

            drow[k0 / FA_VT_TILE] = (half) dq;

            for (int i = 0; i < FA_VT_TILE; ++i) {
                const int kv = k0 + i;
                if (kv < n_kv) {
                    qrow[kv] = (char) clamp((int) rint(tile[i][lid] * id), -127, 127);
                }
            }
        }
    }
}
