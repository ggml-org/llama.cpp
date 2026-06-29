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
