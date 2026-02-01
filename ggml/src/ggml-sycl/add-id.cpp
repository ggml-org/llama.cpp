#include <sycl/sycl.hpp>
#include <cstring>  // for std::memcpy
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

void ggml_sycl_add_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
  const ggml_tensor* src0 = dst->src[0];
  const ggml_tensor* src1 = dst->src[1];
  const ggml_tensor* src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == sizeof(float));
  GGML_ASSERT(nb10 == sizeof(float));
  GGML_ASSERT(nb20 == sizeof(int32_t));

  sycl::queue& q = *ctx.stream();

  const float* src0_d = (const float*)src0->data;
  const int32_t* src2_d = (const int32_t*)src2->data;
  float* dst_d = (float*)dst->data;

  // Check if src1 needs staging - GPU can only directly access device memory
  // CRITICAL: Stage ALL non-device pointers due to Level Zero driver bug that reports
  // mmap'd memory as "shared" (type=3) instead of "unknown" (type=0), causing DEVICE_LOST
  const sycl::usm::alloc src1_type = sycl::get_pointer_type(src1->data, q.get_context());
  const bool src1_needs_staging = (src1_type != sycl::usm::alloc::device);

  const float* src1_d = nullptr;
  void* src1_staging = nullptr;
  void* host_staging = nullptr;
  size_t src1_bytes = 0;
  const int runtime_device = ggml_sycl_get_device_id_from_queue(q);
  sycl::event copy_event;

  if (src1_needs_staging) {
    // Non-device source - stage to device memory via pinned host buffer
    // Calculate total size of src1 tensor
    src1_bytes = ggml_nbytes(src1);

    // Allocate pinned host staging buffer
    host_staging = ggml_sycl_malloc_host_tracked_bytes(src1_bytes, q, "add_id:host_staging");
    if (!host_staging) {
      GGML_LOG_ERROR("[ADD_ID] Failed to allocate host staging for mmap src1 (%zu bytes)\n", src1_bytes);
      return;
    }

    // CPU memcpy from mmap to pinned host
    std::memcpy(host_staging, src1->data, src1_bytes);

    // Allocate device staging buffer
    ggml_sycl::unified_cache_add_runtime_bytes(runtime_device, src1_bytes);
    src1_staging = ggml_sycl_malloc_device(src1_bytes, q, "add_id:device_staging");
    if (!src1_staging) {
      ggml_sycl::unified_cache_sub_runtime_bytes(runtime_device, src1_bytes);
      GGML_LOG_ERROR("[ADD_ID] Failed to allocate device staging for mmap src1 (%zu bytes)\n", src1_bytes);
      ggml_sycl_free_host_tracked_bytes(host_staging, src1_bytes, q);
      return;
    }

    // DMA from pinned host to device - NO .wait()! Use event chaining instead
    copy_event = q.memcpy(src1_staging, host_staging, src1_bytes);

    src1_d = (const float*)src1_staging;
    GGML_SYCL_DEBUG("[ADD_ID] Staged non-device src1 to device (%zu bytes, type=%d)\n", src1_bytes, (int)src1_type);
  } else {
    // Device memory - use directly
    src1_d = (const float*)src1->data;
  }

  int threads = std::min((int)ne00, 768);  // cols

  // Submit kernel with dependency on copy event if staging was needed
  sycl::event kernel_event;
  if (src1_needs_staging) {
    // Kernel depends on copy completing
    kernel_event = q.submit([&](sycl::handler& cgh) {
      cgh.depends_on(copy_event);
      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, ne02, ne01) * sycl::range<3>(1, 1, threads),
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
    });
  } else {
    // No staging needed, submit kernel directly
    kernel_event = q.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, ne02, ne01) * sycl::range<3>(1, 1, threads),
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

  // Async cleanup via host_task - NO .wait()! Cleanup after kernel completes
  if (src1_staging) {
    q.submit([&](sycl::handler& cgh) {
      cgh.depends_on(kernel_event);
      cgh.host_task([src1_staging, host_staging, runtime_device, src1_bytes, &q]() {
        ggml_sycl::unified_cache_sub_runtime_bytes(runtime_device, src1_bytes);
        sycl::free(src1_staging, q);
        ggml_sycl_free_host_tracked_bytes(host_staging, src1_bytes, q);
      });
    });
  }
}
