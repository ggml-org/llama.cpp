#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void kernel_repack_mask_for_wmm(
    const global half* mask_buf,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    global half* mask_buf_padded,
    const ulong mask_nb1_padded,
    const ulong mask_nb2_padded,
    const ulong mask_nb3_padded
) {
    int col   = get_global_id(0);   // 0 .. n_kv
    int row   = get_global_id(1);   // 0 .. n_q
    int slice = get_global_id(2);   // 0 .. (n_head * n_batch)

    int head_idx  = slice % mask_ne2;
    int batch_idx = slice / mask_ne2;

    ulong src_off = (ulong)batch_idx * mask_nb3 + (ulong)head_idx * mask_nb2 + (ulong)row * mask_nb1;
    ulong dst_off = (ulong)batch_idx * mask_nb3_padded + (ulong)head_idx * mask_nb2_padded + (ulong)row * mask_nb1_padded;

    mask_buf_padded[dst_off / 2 + col] = mask_buf[src_off / 2 + col];
}

__kernel void kernel_repack_q_for_wmm(
    const global float* q_buf,
    const ulong q_nb1,
    const ulong q_nb2,
    const ulong q_nb3,
    const int n_head,
    __write_only image3d_t img_q_wmm
) {
    int k4    = get_global_id(0);
    int row   = get_global_id(1);
    int slice = get_global_id(2);
    int batch_idx = slice / n_head;
    int head_idx  = slice % n_head;


    ulong elem_off = (batch_idx * q_nb3 + head_idx * q_nb2 + row * q_nb1) / 4 + (ulong)k4 * 4;
    float4 v = vload4(elem_off / 4, q_buf);

    write_imageh(img_q_wmm, (int4)(row, slice, k4, 0), convert_half4(v));
}

__kernel void kernel_repack_k_for_wmm(
    const global half* k_buf,
    const ulong k_nb1,
    const ulong k_nb2,
    const ulong k_nb3,
    const int n_head_kv,
    const int n_kv,
    __write_only image3d_t img_k_wmm
) {
    int kk    = get_global_id(0);
    int row4  = get_global_id(1);
    int slice = get_global_id(2);
    int batch_idx   = slice / n_head_kv;
    int head_kv_idx = slice % n_head_kv;

    ulong base = batch_idx * k_nb3 + head_kv_idx * k_nb2;
    int row0 = row4 * 4;
    half4 v;
    v.x = (row0 + 0 < n_kv) ? k_buf[(base + (ulong)(row0 + 0) * k_nb1) / 2 + kk] : (half)0;
    v.y = (row0 + 1 < n_kv) ? k_buf[(base + (ulong)(row0 + 1) * k_nb1) / 2 + kk] : (half)0;
    v.z = (row0 + 2 < n_kv) ? k_buf[(base + (ulong)(row0 + 2) * k_nb1) / 2 + kk] : (half)0;
    v.w = (row0 + 3 < n_kv) ? k_buf[(base + (ulong)(row0 + 3) * k_nb1) / 2 + kk] : (half)0;

    write_imageh(img_k_wmm, (int4)(kk, row4, slice, 0), v);
}

__kernel void kernel_repack_v_for_wmm(
      const global half* v_buf,
      const ulong v_nb1,
      const ulong v_nb2,
      const ulong v_nb3,
      const int n_head_kv,
      __write_only image3d_t img_v_wmm
) {
    int hdim4   = get_global_id(0);   // now fastest — walks contiguous memory
    int row     = get_global_id(1);
    int slice   = get_global_id(2);
    int batch_idx   = slice / n_head_kv;
    int head_kv_idx = slice % n_head_kv;

    ulong row_off = batch_idx * v_nb3 + head_kv_idx * v_nb2 + (ulong)row * v_nb1;
    half4 v = vload4((row_off / 2 + (ulong)hdim4 * 4) / 4, v_buf);

    write_imageh(img_v_wmm, (int4)(row, hdim4, slice, 0), v);
}
