#include "convert.hpp"
#include "dequantize.hpp"
#include "presets.hpp"

#if defined(__INTEL_LLVM_COMPILER)
    #if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
        #include <sycl/ext/oneapi/bfloat16.hpp>
        #define GGML_SYCL_HAS_BF16
    #endif
#endif

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k,
                             const sycl::nd_item<3> &item_ct1) {
    const int64_t i = 2 * (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2));

    if (i >= k) {
        return;
    }

    const int64_t ib = i/qk; // block index
    const int64_t iqs = (i%qk)/qr; // quant index
    const int64_t iybs = i - i%qk; // y block start index
    const int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x();
    y[iybs + iqs + y_offset] = v.y();
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_sycl(const void *__restrict__ vx,
                                  dst_t *__restrict__ y, const int64_t k,
                                  dpct::queue_ptr stream) {
    const int64_t num_blocks = (k + 2*SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / (2*SYCL_DEQUANTIZE_BLOCK_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, num_blocks) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_block<qk, qr, dequantize_kernel>(vx, y, k, item_ct1);
            });
    }
}

template <typename dst_t>
static void dequantize_row_q2_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q2_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q2_K(vx, y, item_ct1);
                             });
    }

#endif
}

template <typename dst_t>
static void dequantize_row_q3_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q3_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q3_K(vx, y, item_ct1);
                             });
    }
#endif
}

template <typename dst_t>
static void dequantize_row_q4_0_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb32 = k / 32;
    const int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_0(vx, y, nb32, item_ct1);
                             });
    }
}

template <typename dst_t>
static void dequantize_row_q4_0_sycl_reorder(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    int constexpr WARP_K = WARP_SIZE * QK4_0;
    const int n_warp = (k + WARP_K - 1) / WARP_K;
    GGML_ASSERT(k % 2 == 0);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) *
        sycl::range<3>(1, 1, WARP_SIZE),
        sycl::range<3>(1, 1, WARP_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            dequantize_block_q4_0_reorder(vx, y, k, item_ct1);
        });

}

template <typename dst_t>
static void dequantize_row_q4_1_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb32 = k / 32;
    const int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_1(vx, y, nb32, item_ct1);
                             });
    }
}


template <typename dst_t>
static void dequantize_row_q4_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), cgh);
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_K(vx, y, get_pointer(scale_local_acc), item_ct1);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_q4_K_sycl_reorder(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    const size_t  local_size  = 32;
    const size_t  global_size = nb * local_size;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
                         [=](sycl::nd_item<1> item_ct1) {
                             dequantize_block_q4_K_reorder(vx, y, get_pointer(scale_local_acc), item_ct1, nb);
                         });
    });
}

template <typename dst_t>
static void dequantize_row_q5_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q5_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q5_K(vx, y, item_ct1);
                             });
    }

#endif
}

template <typename dst_t>
static void dequantize_row_q6_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q6_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q6_K(vx, y, item_ct1);
                             });
    }

#endif
}

template <typename dst_t>
static void dequantize_row_q6_K_sycl_reorder(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) { dequantize_block_q6_K_reorder(vx, y, item_ct1, nb); });
}

template <typename dst_t>
static void dequantize_row_iq1_s_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq1_s(
                                     vx, y, item_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq1_m_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq1_m(
                                     vx, y, item_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_xxs_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_xxs(
                                     vx, y, item_ct1, iq2xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_xs_sycl(const void *vx, dst_t *y, const int64_t k,
                                       dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_xs(
                                     vx, y, item_ct1, iq2xs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_s_sycl(const void *vx, dst_t *y, const int64_t k,
                                      dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_s(vx, y, item_ct1);
                             });
        });
    }
}


template <typename dst_t>
static void dequantize_row_iq3_xxs_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq3_xxs(
                                     vx, y, item_ct1, iq3xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq3_s_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq3_s(
                                     vx, y, item_ct1, kmask_iq2xs, iq3s_grid);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq4_xs_sycl(const void *vx, dst_t *y, const int64_t k,
                                       dpct::queue_ptr stream) {
    const int64_t nb = (k + QK_K - 1) / QK_K;
#if QK_K == 64
    dequantize_row_iq4_nl_sycl(vx, y, k, stream);
#else
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_block_iq4_xs(vx, y, item_ct1);
                      });
            });
      }
#endif
}

template <typename dst_t>
static void dequantize_row_iq4_nl_sycl(const void *vx, dst_t *y, const int64_t k,
                                       dpct::queue_ptr stream) {
    const int64_t nb = (k + QK_K - 1) / QK_K;
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_block_iq4_nl(vx, y, item_ct1);
                      });
            });
      }
}

template <typename dst_t>
static void dequantize_row_mxfp4_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_mxfp4(vx, y, item_ct1);
        });
}

template <typename src_t, typename dst_t>
static void convert_unary_nc(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne00, const int64_t ne01,
                          const int64_t ne02, const int64_t s01, const int64_t s02, const int64_t s03,
                          const sycl::nd_item<3> & item_ct1) {

    const int64_t work_group_size = item_ct1.get_local_range(2);
    const int64_t global_id       = item_ct1.get_local_id(2) + work_group_size * item_ct1.get_group(2);

    const int64_t i01 = item_ct1.get_group(1);
    const int64_t i02 = item_ct1.get_group(0) % ne02;
    const int64_t i03 = item_ct1.get_group(0) / ne02;

    // make each work-item deal with more elements since sycl global range can not exceed max int
    const src_t * x = static_cast<const src_t *>(vx);
    const int64_t ix = i03 * s03 + i02 * s02 + i01 * s01;
    const int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00;

#pragma unroll
    for (int64_t i00 = global_id; i00 < ne00; i00 += work_group_size * item_ct1.get_group_range(2)) {
        y[iy + i00] = static_cast<dst_t>(x[ix + i00]);
    }
}

template <typename src_t, typename dst_t>
static void convert_unary_nc_sycl(const void * __restrict__ vx, dst_t * __restrict__ y,
                                  const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
                                  const int64_t s01, const int64_t s02, const int64_t s03, dpct::queue_ptr queue) {
    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    sycl::range<3> global_size(ne02 * ne03, ne01, ceil_div(ne00, SYCL_DEQUANTIZE_BLOCK_SIZE));

    // decrease global range when it exceeds the max int
    // TODO: Downsample logic is separated from the kernel, a rewrite is desirable
    int64_t        downsized_workgroup = downsample_sycl_global_range(global_size[0], SYCL_DEQUANTIZE_BLOCK_SIZE);
    sycl::range<3> workgroup_size(1, 1, downsized_workgroup);

    queue->parallel_for(sycl::nd_range<3>(global_size * workgroup_size, workgroup_size), [=](sycl::nd_item<3> item_ct1) {
        convert_unary_nc<src_t>(vx, y, ne00, ne01, ne02, s01, s02, s03, item_ct1);
    });
}

template <typename src_t, typename dst_t>
static void convert_unary_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr queue) {
    convert_unary_nc_sycl<src_t>(vx, y, k, 1, 1, 1, k, k, k, queue);
}

// =============================================================================
// Q4_0 COALESCED REORDER KERNEL
// =============================================================================
// Reorders Q4_0 data from AoS (block_q4_0 structs) to a warp-coalesced layout
// where adjacent threads in a warp (32 threads) access adjacent memory addresses.
//
// Input layout (AoS - Array of Structures):
//   Each block_q4_0: [d:2 bytes][qs[0..15]:16 bytes] = 18 bytes per block
//   Blocks are stored contiguously: [block0][block1]...[blockN]
//
// Output layout (Coalesced):
//   Tiles of WARP_SIZE (32) blocks where:
//   - qs bytes are interleaved: byte[i] of block[b] at offset [tile * 512 + i * 32 + b]
//   - d values grouped after qs: [tile * (512 + 64) + 512 + b * 2]
//
//   This ensures thread t accesses consecutive memory addresses:
//   Thread 0: bytes 0, 32, 64, ...
//   Thread 1: bytes 1, 33, 65, ...
//   etc.
// =============================================================================

// GPU kernel for Q4_0 AoS to Coalesced conversion
static void reorder_q4_0_aos_to_coalesced_kernel(
    const block_q4_0 * __restrict__ src,
    uint8_t * __restrict__ dst,
    const int blocks_per_row,
    const int nrows,
    const sycl::nd_item<2> & item)
{
    constexpr int TILE_BLOCKS = WARP_SIZE;  // 32 blocks per tile
    constexpr int QS_BYTES_PER_BLOCK = QK4_0 / 2;  // 16 bytes of qs per block
    constexpr int D_BYTES_PER_BLOCK = sizeof(sycl::half);  // 2 bytes for scale

    const int row = item.get_global_id(0);
    const int tid = item.get_local_id(1);      // 0-31 within tile (thread in warp)
    const int tile = item.get_group(1);        // Which tile

    if (row >= nrows) return;

    const int block_idx = tile * TILE_BLOCKS + tid;
    if (block_idx >= blocks_per_row) return;

    // Source: AoS layout - each block_q4_0 is 18 bytes
    const block_q4_0 * src_block = &src[row * blocks_per_row + block_idx];

    // Calculate tile dimensions
    const int tiles_per_row = (blocks_per_row + TILE_BLOCKS - 1) / TILE_BLOCKS;
    constexpr int qs_bytes_per_tile = TILE_BLOCKS * QS_BYTES_PER_BLOCK;  // 512 bytes
    constexpr int d_bytes_per_tile = TILE_BLOCKS * D_BYTES_PER_BLOCK;    // 64 bytes
    constexpr int bytes_per_tile = qs_bytes_per_tile + d_bytes_per_tile; // 576 bytes

    // Destination offsets - use int64_t to avoid overflow for large tensors
    const int64_t row_offset = (int64_t)row * tiles_per_row * bytes_per_tile;
    const int64_t tile_qs_base = row_offset + (int64_t)tile * bytes_per_tile;
    const int64_t tile_d_base = tile_qs_base + qs_bytes_per_tile;

    // Copy qs bytes - interleaved by thread index for coalesced access
    // Thread tid writes its 16 qs bytes at positions [i * 32 + tid] for i in [0,15]
    for (int i = 0; i < QS_BYTES_PER_BLOCK; i++) {
        dst[tile_qs_base + i * TILE_BLOCKS + tid] = src_block->qs[i];
    }

    // Copy d value (scale) - grouped at end of tile
    // Thread tid writes its d at position [tile_d_base + tid * 2]
    *(sycl::half *)(dst + tile_d_base + tid * D_BYTES_PER_BLOCK) = src_block->d;
}

// Host function to launch Q4_0 AoS to Coalesced reorder
void reorder_q4_0_aos_to_coalesced_sycl(
    const void * src,
    void * dst,
    int64_t ne00,  // number of elements per row
    int64_t ne01,  // number of rows
    dpct::queue_ptr stream)
{
    GGML_ASSERT(ne00 % QK4_0 == 0);  // Must be multiple of block size

    const int blocks_per_row = ne00 / QK4_0;
    const int nrows = ne01;
    const int tiles_per_row = (blocks_per_row + WARP_SIZE - 1) / WARP_SIZE;

    // Launch kernel: one work-item per block, organized into warp-sized tiles
    stream->parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(nrows, tiles_per_row * WARP_SIZE),
            sycl::range<2>(1, WARP_SIZE)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            reorder_q4_0_aos_to_coalesced_kernel(
                (const block_q4_0 *)src,
                (uint8_t *)dst,
                blocks_per_row,
                nrows,
                item);
        });
}

// =============================================================================
// Q4_0 COALESCED to SoA REVERSE CONVERSION (for compatibility/debugging)
// =============================================================================
// This reverses the coalesced layout back to SoA layout if needed.
// Not typically used in the hot path, but useful for testing.

static void reorder_q4_0_coalesced_to_soa_kernel(
    const uint8_t * __restrict__ src,
    uint8_t * __restrict__ dst_qs,
    sycl::half * __restrict__ dst_d,
    const int blocks_per_row,
    const int nrows,
    const sycl::nd_item<2> & item)
{
    constexpr int TILE_BLOCKS = WARP_SIZE;
    constexpr int QS_BYTES_PER_BLOCK = QK4_0 / 2;
    constexpr int D_BYTES_PER_BLOCK = sizeof(sycl::half);

    const int row = item.get_global_id(0);
    const int tid = item.get_local_id(1);
    const int tile = item.get_group(1);

    if (row >= nrows) return;

    const int block_idx = tile * TILE_BLOCKS + tid;
    if (block_idx >= blocks_per_row) return;

    // Calculate tile dimensions (same as forward conversion)
    const int tiles_per_row = (blocks_per_row + TILE_BLOCKS - 1) / TILE_BLOCKS;
    constexpr int qs_bytes_per_tile = TILE_BLOCKS * QS_BYTES_PER_BLOCK;
    constexpr int d_bytes_per_tile = TILE_BLOCKS * D_BYTES_PER_BLOCK;
    constexpr int bytes_per_tile = qs_bytes_per_tile + d_bytes_per_tile;

    // Source offsets (coalesced layout) - use int64_t to avoid overflow
    const int64_t row_offset = (int64_t)row * tiles_per_row * bytes_per_tile;
    const int64_t tile_qs_base = row_offset + (int64_t)tile * bytes_per_tile;
    const int64_t tile_d_base = tile_qs_base + qs_bytes_per_tile;

    // Destination: SoA layout - use int64_t for offsets
    // qs: all quants contiguous, then all d values
    const int64_t dst_qs_offset = (int64_t)row * blocks_per_row * QS_BYTES_PER_BLOCK + block_idx * QS_BYTES_PER_BLOCK;
    const int64_t dst_d_idx = (int64_t)row * blocks_per_row + block_idx;

    // Read interleaved qs and write to contiguous SoA
    for (int i = 0; i < QS_BYTES_PER_BLOCK; i++) {
        dst_qs[dst_qs_offset + i] = src[tile_qs_base + i * TILE_BLOCKS + tid];
    }

    // Read d value and write to d array
    dst_d[dst_d_idx] = *(const sycl::half *)(src + tile_d_base + tid * D_BYTES_PER_BLOCK);
}

// Host function to launch Q4_0 Coalesced to SoA conversion
void reorder_q4_0_coalesced_to_soa_sycl(
    const void * src,
    void * dst,  // SoA format: [all qs bytes][all d values]
    int64_t ne00,
    int64_t ne01,
    dpct::queue_ptr stream)
{
    GGML_ASSERT(ne00 % QK4_0 == 0);

    const int blocks_per_row = ne00 / QK4_0;
    const int nrows = ne01;
    const int tiles_per_row = (blocks_per_row + WARP_SIZE - 1) / WARP_SIZE;

    // SoA layout: qs bytes first, then d values
    uint8_t * dst_qs = (uint8_t *)dst;
    sycl::half * dst_d = (sycl::half *)((uint8_t *)dst + nrows * blocks_per_row * (QK4_0 / 2));

    stream->parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(nrows, tiles_per_row * WARP_SIZE),
            sycl::range<2>(1, WARP_SIZE)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            reorder_q4_0_coalesced_to_soa_kernel(
                (const uint8_t *)src,
                dst_qs,
                dst_d,
                blocks_per_row,
                nrows,
                item);
        });
}


to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst, bool full_tensor) {
    // SoA-aware reorder kernels compute d_offset from k parameter.
    // This only works when k == full tensor size. For row slices, use standard kernels.
    // Only SOA layout has reorder dequantization kernels (no COALESCED version).
    const bool use_reorder = full_tensor && dst->src[0]->extra &&
        ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.is_soa();

    switch (type) {
        case GGML_TYPE_Q4_0:
            if (use_reorder) {
                return dequantize_row_q4_0_sycl_reorder;
            } else {
                return dequantize_block_sycl<QK4_0, QR4_0, dequantize_q4_0>;
            }
        case GGML_TYPE_Q4_1:
            return dequantize_block_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_sycl;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_sycl;
        case GGML_TYPE_Q4_K:
            if (use_reorder) {
                return dequantize_row_q4_K_sycl_reorder;
            } else {
                return dequantize_row_q4_K_sycl;
            }
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_sycl;
        case GGML_TYPE_Q6_K:
            if (use_reorder) {
                return dequantize_row_q6_K_sycl_reorder;
            } else {
                return dequantize_row_q6_K_sycl;
            }
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_sycl;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_sycl;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_sycl;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_sycl;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_sycl;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_sycl;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_sycl;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_sycl;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_sycl;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_sycl;
        case GGML_TYPE_F32:
            return convert_unary_sycl<float>;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            return convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        default:
            return nullptr;
    }
}

to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor *dst, bool full_tensor) {
    // SoA-aware reorder kernels compute d_offset from k parameter.
    // This only works when k == full tensor size. For row slices, use standard kernels.
    // Only SOA layout has reorder dequantization kernels (no COALESCED version).
    const bool use_reorder = full_tensor && dst->src[0]->extra &&
        ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.is_soa();

    switch (type) {
        case GGML_TYPE_Q4_0:
            if (use_reorder) {
                return dequantize_row_q4_0_sycl_reorder;
            } else {
                return dequantize_row_q4_0_sycl;
            }
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_sycl;
        case GGML_TYPE_Q5_0:
            return dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_sycl;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_sycl;
        case GGML_TYPE_Q4_K:
            if (use_reorder) {
                return dequantize_row_q4_K_sycl_reorder;
            } else {
                return dequantize_row_q4_K_sycl;
            }
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_sycl;
        case GGML_TYPE_Q6_K:
            if (use_reorder) {
                return dequantize_row_q6_K_sycl_reorder;
            } else {
                return dequantize_row_q6_K_sycl;
            }
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_sycl;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_sycl;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_sycl;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_sycl;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_sycl;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_sycl;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_sycl;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_sycl;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_sycl;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_sycl;
        case GGML_TYPE_F16:
            return convert_unary_sycl<sycl::half>;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            return convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        default:
            return nullptr;
    }
}

to_fp16_nc_sycl_t get_to_fp16_nc_sycl(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_nc_sycl<float>;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            return convert_unary_nc_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        default:
            return nullptr;
    }
}
