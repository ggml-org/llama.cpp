#include <sycl/sycl.hpp>
#include "common.hpp"
#include "add-id.hpp"

static void add_id_kernel(
    const float* src0,
    const float* src1,
    const int32_t* src2,
    float* dst,
    int64_t ne0,
    int64_t ne1,
    size_t nb01,
    size_t nb02,
    size_t nb11,
    size_t nb21,
    sycl::nd_item<3> item_ct1) {
  const int64_t i1 = item_ct1.get_group(2);
  const int64_t i2 = item_ct1.get_group(1);

  const int i11 =
      *(const int32_t*)((const char*)src2 + i1 * sizeof(int32_t) + i2 * nb21);

  const size_t nb1 = ne0 * sizeof(float);
  const size_t nb2 = ne1 * nb1;

  float* dst_row = (float*)((char*)dst + i1 * nb1 + i2 * nb2);
  const float* src0_row =
      (const float*)((const char*)src0 + i1 * nb01 + i2 * nb02);
  const float* src1_row = (const float*)((const char*)src1 + i11 * nb11);

  for (int64_t i0 = item_ct1.get_local_id(2); i0 < ne0;
       i0 += item_ct1.get_local_range(2)) {
    dst_row[i0] = src0_row[i0] + src1_row[i0];
  }
}

void ggml_sycl_add_id(ggml_backend_sycl_context& ctx, ggml_sycl::sycl_tensor dst) {
  auto src0 = dst.src(0);
  auto src1 = dst.src(1);
  auto src2 = dst.src(2);

  const int64_t ne0 = dst.ne(0);
  const int64_t ne1 = dst.ne(1);
  const int64_t ne2 = dst.ne(2);
  const size_t nb00 = src0.nb(0);
  const size_t nb01 = src0.nb(1);
  const size_t nb02 = src0.nb(2);
  const size_t nb10 = src1.nb(0);
  const size_t nb11 = src1.nb(1);
  const size_t nb20 = src2.nb(0);
  const size_t nb21 = src2.nb(1);

  GGML_ASSERT(dst.type() == GGML_TYPE_F32);
  GGML_ASSERT(src0.type() == GGML_TYPE_F32);
  GGML_ASSERT(src1.type() == GGML_TYPE_F32);
  GGML_ASSERT(src2.type() == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == sizeof(float));
  GGML_ASSERT(nb10 == sizeof(float));
  GGML_ASSERT(nb20 == sizeof(int32_t));

  sycl::queue& q = *ctx.stream();

  const float* src0_d = src0.resolve_as<const float>();
  const float* src1_d = src1.resolve_as<const float>();
  const int32_t* src2_d = src2.resolve_as<const int32_t>();
  float* dst_d = dst.resolve_as<float>();

  int threads = std::min((int)ne0, 768);  // cols

  q.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, ne2, ne1) * sycl::range<3>(1, 1, threads),
            sycl::range<3>(1, 1, threads)),
        [=](sycl::nd_item<3> item_ct1) {
          add_id_kernel(
              src0_d,
              src1_d,
              src2_d,
              dst_d,
              ne0,
              ne1,
              nb01,
              nb02,
              nb11,
              nb21,
              item_ct1);
        });
}
