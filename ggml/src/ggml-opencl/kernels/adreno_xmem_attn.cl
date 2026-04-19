#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_uniform_load : enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_constant_load : enable

#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4

__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE  | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

inline int round_up_div(int x, int y) {
    return (x + y - 1) / y;
}

inline half4 rotate_tail_mask(half4 in_mask) {
    return (half4)(in_mask.y, in_mask.z, in_mask.w, in_mask.x);
}

__kernel void adreno_xmem_attn_q_f32_to_img_scaled(
        const global void * src_void,
        ulong src_offset,
        write_only image2d_t dst_image2d,
        const float scale,
        const int d_head,
        const int n_q,
        const int n_head,
        const int n_batch,
        const ulong src_nb1,
        const ulong src_nb2,
        const ulong src_nb3) {
    const int x = get_global_id(0);
    const int flat_h = get_global_id(1);
    const int d = get_global_id(2);

    const int heads_total = n_head * n_batch;
    const int kpack = d_head / 4;

    if (x >= n_q || flat_h >= heads_total || d >= kpack) {
        return;
    }

    const int batch = flat_h / n_head;
    const int head  = flat_h % n_head;
    const int c = d * 4;

    const global char * src_base = (const global char *) src_void + src_offset;
    const global float * row_ptr = (const global float *) (src_base + batch * src_nb3 + head * src_nb2 + x * src_nb1);

    half4 out = (half4)(0.0h);
    out.x = convert_half(row_ptr[c + 0] * scale);
    if (c + 1 < d_head) out.y = convert_half(row_ptr[c + 1] * scale);
    if (c + 2 < d_head) out.z = convert_half(row_ptr[c + 2] * scale);
    if (c + 3 < d_head) out.w = convert_half(row_ptr[c + 3] * scale);

    write_imageh(dst_image2d, (int2)(x, flat_h * kpack + d), out);
}

__kernel void adreno_xmem_attn_kv_f32_to_img_gqa(
        const global void * src_void,
        ulong src_offset,
        write_only image2d_t dst_image2d,
        const int d_head,
        const int n_kv,
        const int n_head,
        const int n_head_kv,
        const int n_batch,
        const ulong src_nb1,
        const ulong src_nb2,
        const ulong src_nb3) {
    const int x = get_global_id(0);
    const int flat_h = get_global_id(1);
    const int d = get_global_id(2);

    const int heads_total = n_head * n_batch;
    const int kpack = d_head / 4;

    if (x >= n_kv || flat_h >= heads_total || d >= kpack) {
        return;
    }

    const int batch = flat_h / n_head;
    const int head  = flat_h % n_head;
    const int head_kv = head / (n_head / n_head_kv);
    const int c = d * 4;

    const global char * src_base = (const global char *) src_void + src_offset;
    const global float * row_ptr = (const global float *) (src_base + batch * src_nb3 + head_kv * src_nb2 + x * src_nb1);

    half4 out = (half4)(0.0h);
    out.x = convert_half(row_ptr[c + 0]);
    if (c + 1 < d_head) out.y = convert_half(row_ptr[c + 1]);
    if (c + 2 < d_head) out.z = convert_half(row_ptr[c + 2]);
    if (c + 3 < d_head) out.w = convert_half(row_ptr[c + 3]);

    write_imageh(dst_image2d, (int2)(x, flat_h * kpack + d), out);
}

__kernel void adreno_xmem_attn_kv_f16_to_img_gqa(
        const global void * src_void,
        ulong src_offset,
        write_only image2d_t dst_image2d,
        const int d_head,
        const int n_kv,
        const int n_head,
        const int n_head_kv,
        const int n_batch,
        const ulong src_nb1,
        const ulong src_nb2,
        const ulong src_nb3) {
    const int x = get_global_id(0);
    const int flat_h = get_global_id(1);
    const int d = get_global_id(2);

    const int heads_total = n_head * n_batch;
    const int kpack = d_head / 4;

    if (x >= n_kv || flat_h >= heads_total || d >= kpack) {
        return;
    }

    const int batch = flat_h / n_head;
    const int head  = flat_h % n_head;
    const int head_kv = head / (n_head / n_head_kv);
    const int c = d * 4;

    const global char * src_base = (const global char *) src_void + src_offset;
    const global half * row_ptr = (const global half *) (src_base + batch * src_nb3 + head_kv * src_nb2 + x * src_nb1);

    half4 out = (half4)(0.0h);
    out.x = row_ptr[c + 0];
    if (c + 1 < d_head) out.y = row_ptr[c + 1];
    if (c + 2 < d_head) out.z = row_ptr[c + 2];
    if (c + 3 < d_head) out.w = row_ptr[c + 3];

    write_imageh(dst_image2d, (int2)(x, flat_h * kpack + d), out);
}

__kernel void adreno_xmem_attn_img_to_f32(
        global void * dst_void,
        ulong dst_offset,
        read_only image2d_t src_image2d,
        const int d_head,
        const int n_q,
        const int n_head,
        const int n_batch,
        const ulong dst_nb1,
        const ulong dst_nb2,
        const ulong dst_nb3) {
    const int x = get_global_id(0);
    const int flat_h = get_global_id(1);
    const int d = get_global_id(2);

    const int heads_total = n_head * n_batch;
    const int kpack = d_head / 4;

    if (x >= n_q || flat_h >= heads_total || d >= kpack) {
        return;
    }

    const int batch = flat_h / n_head;
    const int head  = flat_h % n_head;
    const int c = d * 4;

    global char * dst_base = (global char *) dst_void + dst_offset;
    global float * row_ptr = (global float *) (dst_base + batch * dst_nb3 + x * dst_nb2 + head * dst_nb1);

    const half4 in_value = read_imageh(src_image2d, smp_zero, (int2)(x, flat_h * kpack + d));
    row_ptr[c + 0] = convert_float(in_value.x);
    if (c + 1 < d_head) row_ptr[c + 1] = convert_float(in_value.y);
    if (c + 2 < d_head) row_ptr[c + 2] = convert_float(in_value.z);
    if (c + 3 < d_head) row_ptr[c + 3] = convert_float(in_value.w);
}

__kernel void adreno_xmem_attn_k_gather(
        global half4 * dst_tensor_buffer,
        read_only image2d_t src_tensor_image2d,
        const int4 shared_int4_0,
        const int4 shared_int4_1) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int S = get_global_id(2);
  if (X >= shared_int4_0.w || Y >= shared_int4_0.y || S >= shared_int4_0.z) {
    return;
  }
  half temps[4];
  temps[0] = (half)(0.f);
  temps[1] = (half)(0.f);
  temps[2] = (half)(0.f);
  temps[3] = (half)(0.f);
  for (int i = 0; i < 4; ++i) {
    int dst_channel = S * 4 + i;
    if (dst_channel < shared_int4_0.x) {
      int s_y = Y;
      int s_x = dst_channel;
      int s_c = X;
      {
        int slice_coord_TMP = (s_c) / 4;
        int sub_ch_coord_TMP = (s_c) % 4;
        half4 src_TMP = read_imageh(src_tensor_image2d, smp_zero, (int2)((s_x), ((s_y) * shared_int4_1.x + (slice_coord_TMP))));
        temps[i] = (half[4]){src_TMP.x, src_TMP.y, src_TMP.z, src_TMP.w}[sub_ch_coord_TMP];
      };
    }
  }
  half4 result;
  result.x = temps[0];
  result.y = temps[1];
  result.z = temps[2];
  result.w = temps[3];
  dst_tensor_buffer[(((S) * shared_int4_0.y + (Y)) * shared_int4_0.w + (X))] = result;
}

__kernel void adreno_xmem_attn_pack_k(
        global half4 * dst_tensor_buffer,
        read_only image1d_buffer_t src_image_buffer,
        const int4 shared_int4_0,
        const int4 shared_int4_1,
        const int4 shared_int4_2,
        const half4 shared_half4_0) {
  int linear_index = get_global_id(0);
  if (linear_index >= shared_int4_0.y) return;
  if (get_global_id(1) != 0) return;
  if (get_global_id(2) != 0) return;
  int dst_o_sp_i_ogroup = linear_index;
  int dst_ogroup = dst_o_sp_i_ogroup % shared_int4_0.x;
  int dst_o_sp_i = dst_o_sp_i_ogroup / shared_int4_0.x;
  int dst_i = dst_o_sp_i % shared_int4_0.z;
  int dst_o_sp = dst_o_sp_i / shared_int4_0.z;
  int dst_sp = dst_o_sp % shared_int4_1.x;
  int dst_o = dst_o_sp / shared_int4_1.x;
  int i_slice = dst_i;
  int o_slice = dst_o * shared_int4_0.x + dst_ogroup;
  int spatial_linear = dst_sp;
  int W = spatial_linear % shared_int4_1.y;
  int H = spatial_linear / shared_int4_1.y;
  half4 w0 = (half4)(0);
  half4 w1 = (half4)(0);
  half4 w2 = (half4)(0);
  half4 w3 = (half4)(0);

  if (i_slice * 4 < shared_int4_0.w && o_slice < shared_int4_1.w) {
    w0 = read_imageh(src_image_buffer, (((o_slice) * shared_int4_1.z + (W)) * shared_int4_2.x + (i_slice * 4)));
  }
  if (i_slice * 4 + 1 < shared_int4_0.w && o_slice < shared_int4_1.w) {
    w1 = read_imageh(src_image_buffer, (((o_slice) * shared_int4_1.z + (W)) * shared_int4_2.x + (i_slice * 4 + 1)));
  }
  if (i_slice * 4 + 2 < shared_int4_0.w && o_slice < shared_int4_1.w) {
    w2 = read_imageh(src_image_buffer, (((o_slice) * shared_int4_1.z + (W)) * shared_int4_2.x + (i_slice * 4 + 2)));
  }
  if (i_slice * 4 + 3 < shared_int4_0.w && o_slice < shared_int4_1.w) {
    w3 = read_imageh(src_image_buffer, (((o_slice) * shared_int4_1.z + (W)) * shared_int4_2.x + (i_slice * 4 + 3)));
  }
  if (o_slice == shared_int4_1.w - 1) {
    half4 mask = (half4)(shared_half4_0.y, shared_half4_0.z, shared_half4_0.w, shared_half4_0.x);
    w0 *= mask;
    w1 *= mask;
    w2 *= mask;
    w3 *= mask;
  }
  half4 r0 = w0;
  half4 r1 = w1;
  half4 r2 = w2;
  half4 r3 = w3;
  dst_tensor_buffer[linear_index * 4 + 0] = r0;
  dst_tensor_buffer[linear_index * 4 + 1] = r1;
  dst_tensor_buffer[linear_index * 4 + 2] = r2;
  dst_tensor_buffer[linear_index * 4 + 3] = r3;
}

__attribute__((qcom_max_concurrent_subgroups(12)))
__kernel void adreno_xmem_attn_qk_gemm(
        global half4 * dst_tensor_buffer,
        constant half8 * weights_buffer __attribute__((sub_group_uniform)),
        constant half8 * xmem_buffer __attribute__((max_constant_size((6144)))),
        read_only image2d_t src_tensor_image2d,
        const int4 shared_int4_0,
        const int4 shared_int4_1,
        const int4 shared_int4_2) {
  int X = get_group_id(1) * get_local_size(0) + get_local_id(0);
  int Y = get_group_id(2) * get_local_size(1) + get_local_id(1);
  int Z = get_group_id(0) * get_local_size(2) + get_local_id(2);
  if (X >= shared_int4_0.z || Y >= shared_int4_0.x) return;
  if (Z * 8 >= shared_int4_0.y) return;

  half4 r0 = (half4)(0.f);
  half4 r1 = (half4)(0.f);
  half4 r2 = (half4)(0.f);
  half4 r3 = (half4)(0.f);
  half4 r4 = (half4)(0.f);
  half4 r5 = (half4)(0.f);
  half4 r6 = (half4)(0.f);
  half4 r7 = (half4)(0.f);
  int x_coord = mad24(X, shared_int4_2.y, shared_int4_1.y);
  int y_coord = mad24(Y, shared_int4_2.z, shared_int4_1.z);
  int coord_x, coord_y, coord_s;
  int f_offset = (Z * shared_int4_1.w + Y) * shared_int4_1.x * 32;

  int subgroup_id = (int)((0x1F & qcom_get_physical_sub_group_id()));
  subgroup_id = subgroup_id % 12;
  int c_offset = mul24(subgroup_id, shared_int4_0.w);
  __constant half16 * weights_cache = (__constant half16 *) &xmem_buffer[c_offset];
  coord_y = Y;
  coord_x = X;
  coord_s = 0;
  do {
    half4 src0 = read_imageh(src_tensor_image2d, smp_zero, (int2)((coord_x), ((coord_y) * shared_int4_2.x + (coord_s))));
    coord_s++;
    half4 src1 = read_imageh(src_tensor_image2d, smp_zero, (int2)((coord_x), ((coord_y) * shared_int4_2.x + (coord_s))));
    coord_s++;
    qcom_sub_group_constant_load8(xmem_buffer, weights_buffer, c_offset, f_offset >> 1, 32);
    f_offset += 64;
    qcom_sub_group_sync(QCOM_CLK_CONST_LOAD_SYNC);
    r0 += src0.x * weights_cache[0].s0123;
    r0 += src0.y * weights_cache[0].s4567;
    r0 += src0.z * weights_cache[0].s89ab;
    r0 += src0.w * weights_cache[0].scdef;
    r1 += src0.x * weights_cache[1].s0123;
    r1 += src0.y * weights_cache[1].s4567;
    r1 += src0.z * weights_cache[1].s89ab;
    r1 += src0.w * weights_cache[1].scdef;
    r2 += src0.x * weights_cache[2].s0123;
    r2 += src0.y * weights_cache[2].s4567;
    r2 += src0.z * weights_cache[2].s89ab;
    r2 += src0.w * weights_cache[2].scdef;
    r3 += src0.x * weights_cache[3].s0123;
    r3 += src0.y * weights_cache[3].s4567;
    r3 += src0.z * weights_cache[3].s89ab;
    r3 += src0.w * weights_cache[3].scdef;
    r4 += src0.x * weights_cache[4].s0123;
    r4 += src0.y * weights_cache[4].s4567;
    r4 += src0.z * weights_cache[4].s89ab;
    r4 += src0.w * weights_cache[4].scdef;
    r5 += src0.x * weights_cache[5].s0123;
    r5 += src0.y * weights_cache[5].s4567;
    r5 += src0.z * weights_cache[5].s89ab;
    r5 += src0.w * weights_cache[5].scdef;
    r6 += src0.x * weights_cache[6].s0123;
    r6 += src0.y * weights_cache[6].s4567;
    r6 += src0.z * weights_cache[6].s89ab;
    r6 += src0.w * weights_cache[6].scdef;
    r7 += src0.x * weights_cache[7].s0123;
    r7 += src0.y * weights_cache[7].s4567;
    r7 += src0.z * weights_cache[7].s89ab;
    r7 += src0.w * weights_cache[7].scdef;
    r0 += src1.x * weights_cache[8].s0123;
    r0 += src1.y * weights_cache[8].s4567;
    r0 += src1.z * weights_cache[8].s89ab;
    r0 += src1.w * weights_cache[8].scdef;
    r1 += src1.x * weights_cache[9].s0123;
    r1 += src1.y * weights_cache[9].s4567;
    r1 += src1.z * weights_cache[9].s89ab;
    r1 += src1.w * weights_cache[9].scdef;
    r2 += src1.x * weights_cache[10].s0123;
    r2 += src1.y * weights_cache[10].s4567;
    r2 += src1.z * weights_cache[10].s89ab;
    r2 += src1.w * weights_cache[10].scdef;
    r3 += src1.x * weights_cache[11].s0123;
    r3 += src1.y * weights_cache[11].s4567;
    r3 += src1.z * weights_cache[11].s89ab;
    r3 += src1.w * weights_cache[11].scdef;
    r4 += src1.x * weights_cache[12].s0123;
    r4 += src1.y * weights_cache[12].s4567;
    r4 += src1.z * weights_cache[12].s89ab;
    r4 += src1.w * weights_cache[12].scdef;
    r5 += src1.x * weights_cache[13].s0123;
    r5 += src1.y * weights_cache[13].s4567;
    r5 += src1.z * weights_cache[13].s89ab;
    r5 += src1.w * weights_cache[13].scdef;
    r6 += src1.x * weights_cache[14].s0123;
    r6 += src1.y * weights_cache[14].s4567;
    r6 += src1.z * weights_cache[14].s89ab;
    r6 += src1.w * weights_cache[14].scdef;
    r7 += src1.x * weights_cache[15].s0123;
    r7 += src1.y * weights_cache[15].s4567;
    r7 += src1.z * weights_cache[15].s89ab;
    r7 += src1.w * weights_cache[15].scdef;
  } while (coord_s < shared_int4_2.x);

  coord_s = mul24(Z, 8);
  coord_x = X;
  coord_y = Y;
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r0);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_2.x + (0))));
    }
    dst_tensor_buffer[(((coord_s) * shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r1);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_2.x + (0))));
    }
    dst_tensor_buffer[(((coord_s) * shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r2);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_2.x + (0))));
    }
    dst_tensor_buffer[(((coord_s) * shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r3);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_2.x + (0))));
    }
    dst_tensor_buffer[(((coord_s) * shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r4);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_2.x + (0))));
    }
    dst_tensor_buffer[(((coord_s) * shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r5);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_2.x + (0))));
    }
    dst_tensor_buffer[(((coord_s) * shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r6);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_2.x + (0))));
    }
    dst_tensor_buffer[(((coord_s) * shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r7);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_2.x + (0))));
    }
    dst_tensor_buffer[(((coord_s) * shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
    coord_s++;
  }
}

__kernel void adreno_xmem_attn_softmax_reduce_basic(
        read_only image1d_buffer_t src_tensor_image_buffer,
        write_only image2d_t dst_tensor_image2d,
        const int4 shared_int4_0,
        const int4 shared_int4_1) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= shared_int4_0.z || Y >= shared_int4_0.x) return;
  float sum = 0.0f;
  int end_channel = shared_int4_0.w;
  int end_slice = (end_channel + 3) / 4;
  int start_channel = 0;
  int start_slice = start_channel / 4;
  bool need_per_channels_check = start_channel % 4 != 0 || end_channel % 4 != 0;
  float maximum;
  {
    int slice_coord_TMP = (start_channel) / 4;
    int sub_ch_coord_TMP = (start_channel) % 4;
    float4 src_TMP = read_imagef(src_tensor_image_buffer, (((slice_coord_TMP) * shared_int4_1.x + (Y)) * shared_int4_1.y + (X)));
    maximum = (float[4]){src_TMP.x, src_TMP.y, src_TMP.z, src_TMP.w}[sub_ch_coord_TMP];
  };
  for (int d = start_slice; d < end_slice; d += 1) {
    float4 mask_dot = (float4)(1.f);
    float4 src = read_imagef(src_tensor_image_buffer, (((d) * shared_int4_1.x + (Y)) * shared_int4_1.y + (X)));
    if (need_per_channels_check && (d == start_slice || d == end_slice - 1)) {
      if (d * 4 + 0 < start_channel || d * 4 + 0 >= end_channel) { mask_dot.x = 0.f; src.x = maximum; }
      if (d * 4 + 1 < start_channel || d * 4 + 1 >= end_channel) { mask_dot.y = 0.f; src.y = maximum; }
      if (d * 4 + 2 < start_channel || d * 4 + 2 >= end_channel) { mask_dot.z = 0.f; src.z = maximum; }
      if (d * 4 + 3 < start_channel || d * 4 + 3 >= end_channel) { mask_dot.w = 0.f; src.w = maximum; }
    }
    float new_max = max(src.x, src.y);
    new_max = max(new_max, src.z);
    new_max = max(new_max, src.w);
    new_max = max(new_max, maximum);
    float scale = native_exp(maximum - new_max);
    maximum = new_max;
    sum *= scale;
    float4 exp_res = native_exp(src - maximum);
    sum += dot(mask_dot, exp_res);
  }
  float inv_sum = 1.0f / sum;
  half4 result;
  result.x = convert_half(inv_sum);
  result.y = convert_half(maximum);
  write_imageh(dst_tensor_image2d, (int2)((X), ((Y) * shared_int4_0.y + (0))), result);
}

__kernel void adreno_xmem_attn_softmax_apply_basic(
        global half4 * dst_tensor_buffer,
        read_only image1d_buffer_t src_tensor_image_buffer,
        read_only image2d_t src_tensor_1_image2d,
        const int4 shared_int4_0,
        const int4 shared_int4_1) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= shared_int4_0.z || Y >= shared_int4_0.x || Z >= shared_int4_0.y) return;
  half4 src = read_imageh(src_tensor_image_buffer, (((Z) * shared_int4_1.x + (Y)) * shared_int4_1.y + (X)));
  {
    half4 src_final;
    {
      {
        half4 exp_val = read_imageh(src_tensor_1_image2d, smp_zero, (int2)(((X)), (((Y)) * shared_int4_0.w + (0))));
        src_final = exp(src - exp_val.y) * exp_val.x;
      }
    }
    dst_tensor_buffer[(((Z) * shared_int4_0.x + (Y)) * shared_int4_0.z + (X))] = src_final;
  };
}

__kernel void adreno_xmem_attn_mask_scores(
        global half4 * dst_score_tensor_buffer,
        read_only image1d_buffer_t src_score_image_buffer,
        global const half * mask,
        const int n_q,
        const int n_kv,
        const int heads_total,
        const int n_head,
        const int is_causal,
        const ulong mask_nb1,
        const ulong mask_nb2,
        const ulong mask_nb3,
        const int mask_ne2,
        const int mask_ne3) {
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    const int npack = round_up_div(n_kv, 4);
    if (X >= n_q || Y >= heads_total || Z >= npack) {
        return;
    }

    const int head = Y % n_head;
    const int batch = Y / n_head;
    const int mask_head_idx = mask ? (head % mask_ne2) : 0;
    const int mask_batch_idx = mask ? (batch % mask_ne3) : 0;
    const global half * mask_row = mask ? (const global half *) ((const global char *) mask + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2 + X * mask_nb1) : 0;

    const half4 score = read_imageh(src_score_image_buffer, ((Z * heads_total + Y) * n_q + X));
    float vals[4] = {
        convert_float(score.x),
        convert_float(score.y),
        convert_float(score.z),
        convert_float(score.w),
    };

    for (int lane = 0; lane < 4; ++lane) {
        const int k_idx = Z * 4 + lane;
        if (k_idx >= n_kv || (is_causal && k_idx > (n_kv - n_q + X))) {
            vals[lane] = -INFINITY;
        } else if (mask) {
            vals[lane] += convert_float(mask_row[k_idx]);
        }
    }

    dst_score_tensor_buffer[((Z * heads_total + Y) * n_q + X)] = (half4)(
            convert_half(vals[0]),
            convert_half(vals[1]),
            convert_half(vals[2]),
            convert_half(vals[3]));
}

__kernel void adreno_xmem_attn_mask_all_zero(
        global int * dst_flag,
        global const half * mask,
        const ulong mask_offset,
        const int mask_elems) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const global half * mask_ptr = (const global half *) ((const global char *) mask + mask_offset);

    int found = 0;
    for (int i = lid; i < mask_elems; i += lsize) {
        found |= mask_ptr[i] != (half) 0.0h;
    }

    __local int reduced[256];
    reduced[lid] = found;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            reduced[lid] |= reduced[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        dst_flag[0] = reduced[0];
    }
}

__kernel void adreno_xmem_attn_softmax_reduce_masked(
        read_only image1d_buffer_t src_tensor_image_buffer,
        write_only image2d_t dst_tensor_image2d,
        global const half * mask,
        const ulong mask_offset,
        const int has_mask,
        const int n_q,
        const int n_kv,
        const int heads_total,
        const int n_head,
        const int is_causal,
        const ulong mask_nb1,
        const ulong mask_nb2,
        const ulong mask_nb3,
        const int mask_ne2,
        const int mask_ne3) {
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    if (X >= n_q || Y >= heads_total) {
        return;
    }

    const int head = Y % n_head;
    const int batch = Y / n_head;
    const global char * mask_base = has_mask ? ((const global char *) mask + mask_offset) : 0;
    const int mask_head_idx = has_mask ? (head % mask_ne2) : 0;
    const int mask_batch_idx = has_mask ? (batch % mask_ne3) : 0;
    const global half * mask_row = has_mask ? (const global half *) (mask_base + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2 + X * mask_nb1) : 0;

    float maximum = -INFINITY;
    const int npack = round_up_div(n_kv, 4);
    for (int d = 0; d < npack; ++d) {
        const half4 packed_scores = read_imageh(src_tensor_image_buffer, ((d * heads_total + Y) * n_q + X));
        float4 scores = convert_float4(packed_scores);
        for (int lane = 0; lane < 4; ++lane) {
            const int k_idx = d * 4 + lane;
            float score = lane == 0 ? scores.x : lane == 1 ? scores.y : lane == 2 ? scores.z : scores.w;
            if (k_idx >= n_kv || (is_causal && k_idx > (n_kv - n_q + X))) {
                score = -INFINITY;
            } else if (has_mask) {
                score += convert_float(mask_row[k_idx]);
            }
            maximum = fmax(maximum, score);
        }
    }

    float sum = 0.0f;
    if (!isfinite(maximum)) {
        write_imageh(dst_tensor_image2d, (int2)(X, Y), (half4)((half)0.0h, (half)0.0h, (half)0.0h, (half)0.0h));
        return;
    }

    for (int d = 0; d < npack; ++d) {
        const half4 packed_scores = read_imageh(src_tensor_image_buffer, ((d * heads_total + Y) * n_q + X));
        float4 scores = convert_float4(packed_scores);
        for (int lane = 0; lane < 4; ++lane) {
            const int k_idx = d * 4 + lane;
            float score = lane == 0 ? scores.x : lane == 1 ? scores.y : lane == 2 ? scores.z : scores.w;
            if (k_idx >= n_kv || (is_causal && k_idx > (n_kv - n_q + X))) {
                continue;
            }
            if (has_mask) {
                score += convert_float(mask_row[k_idx]);
            }
            sum += native_exp(score - maximum);
        }
    }

    const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    write_imageh(dst_tensor_image2d, (int2)(X, Y), (half4)(convert_half(inv_sum), convert_half(maximum), (half)0.0h, (half)0.0h));
}

__kernel void adreno_xmem_attn_softmax_apply_masked(
        global half4 * dst_tensor_buffer,
        read_only image1d_buffer_t src_tensor_image_buffer,
        read_only image2d_t src_stats_image2d,
        global const half * mask,
        const ulong mask_offset,
        const int has_mask,
        const int n_q,
        const int n_kv,
        const int heads_total,
        const int n_head,
        const int is_causal,
        const ulong mask_nb1,
        const ulong mask_nb2,
        const ulong mask_nb3,
        const int mask_ne2,
        const int mask_ne3) {
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    const int npack = round_up_div(n_kv, 4);
    if (X >= n_q || Y >= heads_total || Z >= npack) {
        return;
    }

    const int head = Y % n_head;
    const int batch = Y / n_head;
    const global char * mask_base = has_mask ? ((const global char *) mask + mask_offset) : 0;
    const int mask_head_idx = has_mask ? (head % mask_ne2) : 0;
    const int mask_batch_idx = has_mask ? (batch % mask_ne3) : 0;
    const global half * mask_row = has_mask ? (const global half *) (mask_base + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2 + X * mask_nb1) : 0;

    const half4 src = read_imageh(src_tensor_image_buffer, ((Z * heads_total + Y) * n_q + X));
    const half4 stats = read_imageh(src_stats_image2d, smp_zero, (int2)(X, Y));

    half4 out = (half4)(0.0h);
    const float inv_sum = convert_float(stats.x);
    const float maximum = convert_float(stats.y);
    if (inv_sum > 0.0f) {
        const float src_vals[4] = {
            convert_float(src.x), convert_float(src.y), convert_float(src.z), convert_float(src.w)
        };
        float out_vals[4] = {0, 0, 0, 0};
        for (int lane = 0; lane < 4; ++lane) {
            const int k_idx = Z * 4 + lane;
            if (k_idx >= n_kv || (is_causal && k_idx > (n_kv - n_q + X))) {
                out_vals[lane] = 0.0f;
                continue;
            }
            float score = src_vals[lane];
            if (has_mask) {
                score += convert_float(mask_row[k_idx]);
            }
            out_vals[lane] = native_exp(score - maximum) * inv_sum;
        }
        out = (half4)(convert_half(out_vals[0]), convert_half(out_vals[1]), convert_half(out_vals[2]), convert_half(out_vals[3]));
    }

    dst_tensor_buffer[((Z * heads_total + Y) * n_q + X)] = out;
}

__kernel void adreno_xmem_attn_pack_v(
        global half4 * dst_tensor_buffer,
        read_only image2d_t src_image2d,
        const int4 shared_int4_0,
        const int4 shared_int4_1,
        const half4 shared_half4_0) {
  int linear_index = get_global_id(0);
  if (linear_index >= shared_int4_0.y) return;
  if (get_global_id(1) != 0) return;
  if (get_global_id(2) != 0) return;
  int dst_o_sp_i_ogroup = linear_index;
  int dst_ogroup = dst_o_sp_i_ogroup % shared_int4_0.x;
  int dst_o_sp_i = dst_o_sp_i_ogroup / shared_int4_0.x;
  int dst_i = dst_o_sp_i % shared_int4_0.z;
  int dst_o_sp = dst_o_sp_i / shared_int4_0.z;
  int dst_sp = dst_o_sp % shared_int4_1.x;
  int dst_o = dst_o_sp / shared_int4_1.x;
  int i_slice = dst_i;
  int o_slice = dst_o * shared_int4_0.x + dst_ogroup;
  int spatial_linear = dst_sp;
  int W = spatial_linear % shared_int4_1.y;
  int H = spatial_linear / shared_int4_1.y;
  half4 w0 = (half4)(0);
  half4 w1 = (half4)(0);
  half4 w2 = (half4)(0);
  half4 w3 = (half4)(0);

  if (i_slice * 4 < shared_int4_0.w && o_slice < shared_int4_1.z) {
    w0 = read_imageh(src_image2d, smp_zero, (int2)((i_slice * 4), ((W) * shared_int4_1.z + (o_slice))));
  }
  if (i_slice * 4 + 1 < shared_int4_0.w && o_slice < shared_int4_1.z) {
    w1 = read_imageh(src_image2d, smp_zero, (int2)((i_slice * 4 + 1), ((W) * shared_int4_1.z + (o_slice))));
  }
  if (i_slice * 4 + 2 < shared_int4_0.w && o_slice < shared_int4_1.z) {
    w2 = read_imageh(src_image2d, smp_zero, (int2)((i_slice * 4 + 2), ((W) * shared_int4_1.z + (o_slice))));
  }
  if (i_slice * 4 + 3 < shared_int4_0.w && o_slice < shared_int4_1.z) {
    w3 = read_imageh(src_image2d, smp_zero, (int2)((i_slice * 4 + 3), ((W) * shared_int4_1.z + (o_slice))));
  }
  if (o_slice == shared_int4_1.z - 1) {
    half4 mask = (half4)(shared_half4_0.y, shared_half4_0.z, shared_half4_0.w, shared_half4_0.x);
    w0 *= mask;
    w1 *= mask;
    w2 *= mask;
    w3 *= mask;
  }
  half4 r0 = w0;
  half4 r1 = w1;
  half4 r2 = w2;
  half4 r3 = w3;
  dst_tensor_buffer[linear_index * 4 + 0] = r0;
  dst_tensor_buffer[linear_index * 4 + 1] = r1;
  dst_tensor_buffer[linear_index * 4 + 2] = r2;
  dst_tensor_buffer[linear_index * 4 + 3] = r3;
}

__attribute__((qcom_max_concurrent_subgroups(12)))
__kernel void adreno_xmem_attn_pv_gemm(
        constant half8 * weights_buffer __attribute__((sub_group_uniform)),
        constant half8 * xmem_buffer __attribute__((max_constant_size((6144)))),
        read_only image1d_buffer_t src_tensor_image_buffer,
        write_only image2d_t dst_tensor_image2d,
        const int4 shared_int4_0,
        const int4 shared_int4_1,
        const int4 shared_int4_2,
        const int4 shared_int4_3) {
  int X = get_group_id(1) * get_local_size(0) + get_local_id(0);
  int Y = get_group_id(2) * get_local_size(1) + get_local_id(1);
  int Z = get_group_id(0) * get_local_size(2) + get_local_id(2);
  if (X >= shared_int4_0.z || Y >= shared_int4_0.x) return;
  if (Z * 8 >= shared_int4_0.y) return;

  half4 r0 = (half4)(0.f);
  half4 r1 = (half4)(0.f);
  half4 r2 = (half4)(0.f);
  half4 r3 = (half4)(0.f);
  half4 r4 = (half4)(0.f);
  half4 r5 = (half4)(0.f);
  half4 r6 = (half4)(0.f);
  half4 r7 = (half4)(0.f);
  int x_coord = mad24(X, shared_int4_2.w, shared_int4_1.y);
  int y_coord = mad24(Y, shared_int4_3.x, shared_int4_1.z);
  int coord_x, coord_y, coord_s;
  int f_offset = (Z * shared_int4_1.w + Y) * shared_int4_1.x * 32;

  int subgroup_id = (int)((0x1F & qcom_get_physical_sub_group_id()));
  subgroup_id = subgroup_id % 12;
  int c_offset = mul24(subgroup_id, shared_int4_0.w);
  __constant half16 * weights_cache = (__constant half16 *)&xmem_buffer[c_offset];
  coord_y = Y;
  coord_x = X;
  int addr = (((0) * shared_int4_1.w + (coord_y)) * shared_int4_2.z + (coord_x));
  int dz = shared_int4_2.x;
  coord_s = 0;
  do {
    half4 src0 = read_imageh(src_tensor_image_buffer, addr); addr += dz;
    coord_s++;
    half4 src1 = read_imageh(src_tensor_image_buffer, addr); addr += dz;
    coord_s++;
    qcom_sub_group_constant_load8(xmem_buffer, weights_buffer, c_offset, f_offset >> 1, 32);
    f_offset += 64;
    qcom_sub_group_sync(QCOM_CLK_CONST_LOAD_SYNC);
    r0 += src0.x * weights_cache[0].s0123;
    r0 += src0.y * weights_cache[0].s4567;
    r0 += src0.z * weights_cache[0].s89ab;
    r0 += src0.w * weights_cache[0].scdef;
    r1 += src0.x * weights_cache[1].s0123;
    r1 += src0.y * weights_cache[1].s4567;
    r1 += src0.z * weights_cache[1].s89ab;
    r1 += src0.w * weights_cache[1].scdef;
    r2 += src0.x * weights_cache[2].s0123;
    r2 += src0.y * weights_cache[2].s4567;
    r2 += src0.z * weights_cache[2].s89ab;
    r2 += src0.w * weights_cache[2].scdef;
    r3 += src0.x * weights_cache[3].s0123;
    r3 += src0.y * weights_cache[3].s4567;
    r3 += src0.z * weights_cache[3].s89ab;
    r3 += src0.w * weights_cache[3].scdef;
    r4 += src0.x * weights_cache[4].s0123;
    r4 += src0.y * weights_cache[4].s4567;
    r4 += src0.z * weights_cache[4].s89ab;
    r4 += src0.w * weights_cache[4].scdef;
    r5 += src0.x * weights_cache[5].s0123;
    r5 += src0.y * weights_cache[5].s4567;
    r5 += src0.z * weights_cache[5].s89ab;
    r5 += src0.w * weights_cache[5].scdef;
    r6 += src0.x * weights_cache[6].s0123;
    r6 += src0.y * weights_cache[6].s4567;
    r6 += src0.z * weights_cache[6].s89ab;
    r6 += src0.w * weights_cache[6].scdef;
    r7 += src0.x * weights_cache[7].s0123;
    r7 += src0.y * weights_cache[7].s4567;
    r7 += src0.z * weights_cache[7].s89ab;
    r7 += src0.w * weights_cache[7].scdef;
    r0 += src1.x * weights_cache[8].s0123;
    r0 += src1.y * weights_cache[8].s4567;
    r0 += src1.z * weights_cache[8].s89ab;
    r0 += src1.w * weights_cache[8].scdef;
    r1 += src1.x * weights_cache[9].s0123;
    r1 += src1.y * weights_cache[9].s4567;
    r1 += src1.z * weights_cache[9].s89ab;
    r1 += src1.w * weights_cache[9].scdef;
    r2 += src1.x * weights_cache[10].s0123;
    r2 += src1.y * weights_cache[10].s4567;
    r2 += src1.z * weights_cache[10].s89ab;
    r2 += src1.w * weights_cache[10].scdef;
    r3 += src1.x * weights_cache[11].s0123;
    r3 += src1.y * weights_cache[11].s4567;
    r3 += src1.z * weights_cache[11].s89ab;
    r3 += src1.w * weights_cache[11].scdef;
    r4 += src1.x * weights_cache[12].s0123;
    r4 += src1.y * weights_cache[12].s4567;
    r4 += src1.z * weights_cache[12].s89ab;
    r4 += src1.w * weights_cache[12].scdef;
    r5 += src1.x * weights_cache[13].s0123;
    r5 += src1.y * weights_cache[13].s4567;
    r5 += src1.z * weights_cache[13].s89ab;
    r5 += src1.w * weights_cache[13].scdef;
    r6 += src1.x * weights_cache[14].s0123;
    r6 += src1.y * weights_cache[14].s4567;
    r6 += src1.z * weights_cache[14].s89ab;
    r6 += src1.w * weights_cache[14].scdef;
    r7 += src1.x * weights_cache[15].s0123;
    r7 += src1.y * weights_cache[15].s4567;
    r7 += src1.z * weights_cache[15].s89ab;
    r7 += src1.w * weights_cache[15].scdef;
  } while (coord_s < shared_int4_2.y);

  coord_s = mul24(Z, 8);
  coord_x = X;
  coord_y = Y;
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r0);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image_buffer, (((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0)));
    }
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r1);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image_buffer, (((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0)));
    }
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r2);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image_buffer, (((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0)));
    }
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r3);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image_buffer, (((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0)));
    }
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r4);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image_buffer, (((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0)));
    }
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r5);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image_buffer, (((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0)));
    }
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r6);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image_buffer, (((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0)));
    }
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r7);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image_buffer, (((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0)));
    }
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
}
