#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#pragma OPENCL EXTENSION cl_qcom_subgroup_uniform_load: enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_constant_load: enable
__attribute__((qcom_max_concurrent_subgroups(12)))
MAIN_FUNCTION(__constant half8* weights_buffer  __attribute__((sub_group_uniform)),
  __constant half8* xmem_buffer  __attribute__((max_constant_size((6144)))),
  __read_only image1d_buffer_t src_tensor_image_buffer,
  __write_only image2d_t dst_tensor_image2d,
  int4 shared_int4_0,
  int4 shared_int4_1,
  int4 shared_int4_2,
  int4 shared_int4_3) {
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
  __constant half16* weights_cache = (__constant half16*)&xmem_buffer[c_offset];

  coord_y = Y;
  coord_x = X;
      int addr = (((0) * shared_int4_1.w + (coord_y)) * shared_int4_2.z + (coord_x));
      int dz = shared_int4_2.x;
      coord_s = 0;
      do {
        half4 src0 = read_imageh(src_tensor_image_buffer, addr); addr += dz;
; coord_s++;
        half4 src1 = read_imageh(src_tensor_image_buffer, addr); addr += dz;
; coord_s++;
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
