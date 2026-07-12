#include "cpy.hpp"

#include <float.h>
#include <vector>

#include "dequantize.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml.h"


static void cpy_1_f32_f32(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    float *       dsti = (float *) cdsti;

    *dsti = *xi;
}

static void cpy_1_f32_f16(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    sycl::half *  dsti = (sycl::half *) cdsti;

    *dsti = sycl::vec<float, 1>(*xi).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

static void cpy_1_f16_f16(const char * cxi, char * cdsti) {
    const sycl::half * xi   = (const sycl::half *) cxi;
    sycl::half *       dsti = (sycl::half *) cdsti;

    *dsti = *xi;
}

static void cpy_1_f16_f32(const char * cxi, char * cdsti) {
    const sycl::half * xi   = (const sycl::half *) cxi;
    float *            dsti = (float *) cdsti;

    *dsti = *xi;
}

static void cpy_1_i16_i16(const char * cxi, char * cdsti) {
    const int16_t * xi   = (const int16_t *) cxi;
    int16_t *       dsti = (int16_t *) cdsti;

    *dsti = *xi;
}

static void cpy_1_i32_i32(const char * cxi, char * cdsti) {
    const int32_t * xi   = (const int32_t *) cxi;
    int32_t *       dsti = (int32_t *) cdsti;

    *dsti = *xi;
}

static void cpy_1_f32_i32(const char * cxi, char * cdsti) {
    const float * xi     = (const float *) cxi;
    int32_t *     dsti   = (int32_t *) cdsti;

    *dsti = (int32_t) *xi;
}

static void cpy_1_i32_f32(const char * cxi, char * cdsti) {
    const int32_t * xi   = (const int32_t *) cxi;
    float *         dsti = (float *) cdsti;

    *dsti = (float) *xi;
}

#ifdef GGML_SYCL_HAS_BF16
static void cpy_1_f32_bf16(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(*xi);
}

static void cpy_1_bf16_f32(const char * cxi, char * cdsti) {
    const sycl::ext::oneapi::bfloat16 * xi   = (const sycl::ext::oneapi::bfloat16 *) cxi;
    float *            dsti = (float *) cdsti;

    *dsti = static_cast<float>(*xi);
}

static void cpy_1_bf16_bf16(const char * cxi, char * cdsti) {
    const sycl::ext::oneapi::bfloat16 * xi   = (const sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::ext::oneapi::bfloat16 *       dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = *xi;
}

static void cpy_1_f16_bf16(const char * cxi, char * cdsti) {
    const sycl::half * xi   = (const sycl::half *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(static_cast<float>(*xi));
}

static void cpy_1_bf16_f16(const char * cxi, char * cdsti) {
    const sycl::ext::oneapi::bfloat16 * xi   = (const sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::half *            dsti = (sycl::half *) cdsti;

    *dsti = sycl::half(static_cast<float>(*xi));
}
#endif

template <cpy_kernel_t cpy_1>
static void cpy_f32_f16(const char * cx, char * cdst, const int ne, const int ne00, const int ne01, const int ne02,
                        const int nb00, const int nb01, const int nb02, const int nb03, const int ne10, const int ne11,
                        const int ne12, const int nb10, const int nb11, const int nb12, const int nb13,
                        const sycl::nd_item<3> & item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    if (i >= ne) {
        return;
    }

    // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int i03      = i / (ne00 * ne01 * ne02);
    const int i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int i13        = i / (ne10 * ne11 * ne12);
    const int i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}


/* quantized type same copy */
template<typename T>
static void cpy_blck_q_q(const char * cxi, char * cdsti) {
    const T * xi = (const T *) cxi;
    T * dsti = (T *) cdsti;
    *dsti = *xi;
}


static void cpy_blck_q8_0_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *) (cdsti);

    for (int j = 0; j < QK8_0; j += 2) {
        dfloat2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j)     = dq.x();
        *(cdstf + j + 1) = dq.y();
    }
}



template <dequantize_kernel_t dequant, int qk> static void cpy_blck_q_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *) (cdsti);

    for (int j = 0; j < qk / 2; j++) {
        dfloat2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j)          = dq.x();
        *(cdstf + j + qk / 2) = dq.y();
    }
}


template <typename T, int qk>
static void cpy_q_q(const char * cx, char * cdst, const int ne, const int ne00, const int ne01, const int ne02,
                      const int nb00, const int nb01, const int nb02, const int nb03, const int ne10, const int ne11,
                      const int ne12, const int nb10, const int nb11, const int nb12, const int nb13,
                      const sycl::nd_item<3> & item_ct1) {
    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    if (i >= ne) {
        return;
    }

    const int i03      = i / (ne00 * ne01 * ne02);
    const int i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;


    const int i13        = i / (ne10 * ne11 * ne12);
    const int i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck_q_q<T>(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, int qk>
static void cpy_f32_q(const char * cx, char * cdst, const int ne, const int ne00, const int ne01, const int ne02,
                      const int nb00, const int nb01, const int nb02, const int nb03, const int ne10, const int ne11,
                      const int ne12, const int nb10, const int nb11, const int nb12, const int nb13,
                      const sycl::nd_item<3> & item_ct1) {
    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    if (i >= ne) {
        return;
    }


    const int i03      = i / (ne00 * ne01 * ne02);
    const int i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int i13        = i / (ne10 * ne11 * ne12);
    const int i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, int qk>
static void cpy_q_f32(const char * cx, char * cdst, const int ne, const int ne00, const int ne01, const int ne02,
                      const int nb00, const int nb01, const int nb02, const int nb03, const int ne10, const int ne11,
                      const int ne12, const int nb10, const int nb11, const int nb12, const int nb13,
                      const sycl::nd_item<3> & item_ct1) {
    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    if (i >= ne) {
        return;
    }

    const int i03      = i / (ne00 * ne01 * ne02);
    const int i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int i13        = i / (ne10 * ne11 * ne12);
    const int i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

static void ggml_cpy_f16_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_f16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_f32_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_f32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_f32_f16_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_f32_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_f32_i32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_f32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_i32_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_i32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_f32_q8_0_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    const int num_blocks = ne / QK8_0;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q8_0_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ne;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_q_f32<cpy_blck_q8_0_f32, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_f32_q4_0_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    const int num_blocks = ne / QK4_0;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q4_0_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

static void ggml_cpy_f32_q4_1_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    const int num_blocks = ne / QK4_1;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q4_1_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

static void ggml_cpy_f32_q5_0_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    const int num_blocks = ne / QK5_0;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q5_0_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

static void ggml_cpy_f32_q5_1_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    const int num_blocks = ne / QK5_1;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q5_1_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

static void ggml_cpy_mxfp4_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                    const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                    const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                    const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_f32<cpy_blck_q_f32<dequantize_mxfp4, QK_MXFP4>, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                             nb01, nb02, nb03, ne10, ne11, ne12,
                                                                             nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_f32_iq4_nl_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                     const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                     const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                     const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    const int num_blocks = ne / QK4_NL;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
            cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                   ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void cpy_blck_f16_q4_0(const char * cxi, char * cdsti) {
    const sycl::half * xi = (const sycl::half *) cxi;
    float xf[QK4_0];

    for (int j = 0; j < QK4_0; ++j) {
        xf[j] = (float) xi[j];
    }

    cpy_blck_f32_q4_0((const char *) xf, cdsti);
}

static void cpy_blck_f16_q4_1(const char * cxi, char * cdsti) {
    const sycl::half * xi = (const sycl::half *) cxi;
    float xf[QK4_1];

    for (int j = 0; j < QK4_1; ++j) {
        xf[j] = (float) xi[j];
    }

    cpy_blck_f32_q4_1((const char *) xf, cdsti);
}

static void cpy_blck_f16_q5_0(const char * cxi, char * cdsti) {
    const sycl::half * xi = (const sycl::half *) cxi;
    float xf[QK5_0];

    for (int j = 0; j < QK5_0; ++j) {
        xf[j] = (float) xi[j];
    }

    cpy_blck_f32_q5_0((const char *) xf, cdsti);
}

static void ggml_cpy_f16_q4_0_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    const int num_blocks = ne / QK4_0;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f16_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_f16_q4_1_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    const int num_blocks = ne / QK4_1;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f16_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_f16_q5_0_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    const int num_blocks = ne / QK5_0;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f16_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static bool ggml_sycl_is_quantized_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static bool ggml_sycl_can_quantize_rows_sycl(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

template <typename SrcScalar>
static inline float ggml_sycl_src_to_f32(const SrcScalar & x) {
    return (float) x;
}

#ifdef GGML_SYCL_HAS_BF16
template <>
inline float ggml_sycl_src_to_f32<sycl::ext::oneapi::bfloat16>(const sycl::ext::oneapi::bfloat16 & x) {
    return static_cast<float>(x);
}

template <>
inline float ggml_sycl_src_to_f32<ggml_bf16_t>(const ggml_bf16_t & x) {
    union {
        uint32_t u32;
        float f32;
    } value;

    value.u32 = (uint32_t) x.bits << 16;
    return value.f32;
}
#endif

template <typename SrcScalar, cpy_kernel_t quantize_block, int qk>
static void ggml_sycl_quantize_rows_q(const char * cx, char * cdst, const int64_t ne,
                                      const int64_t ne00, const int64_t ne01, const int64_t ne02,
                                      const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                      const int64_t ne10, const int64_t ne11, const int64_t ne12,
                                      const size_t nb10, const size_t nb11, const size_t nb12, const size_t nb13,
                                      queue_ptr stream) {
    GGML_ASSERT(ne % qk == 0);
    GGML_ASSERT(ne00 % qk == 0);

    const int64_t total_blocks = ne / qk;
    constexpr int block_size = 256;
    const int64_t grid_size = ceil_div(total_blocks, (int64_t) block_size);

    stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> item_ct1) {
        const int64_t block_idx = item_ct1.get_global_linear_id();
        if (block_idx >= total_blocks) {
            return;
        }

        const int64_t i = block_idx * qk;

        const int64_t i03 = i / (ne00 * ne01 * ne02);
        const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
        const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
        const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
        const size_t x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

        const int64_t i13 = i / (ne10 * ne11 * ne12);
        const int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
        const int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
        const int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
        const size_t dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

        float xf[qk];
        if (nb00 == sizeof(SrcScalar)) {
            const SrcScalar * src_row = (const SrcScalar *) (cx + x_offset);
            for (int j = 0; j < qk; ++j) {
                xf[j] = ggml_sycl_src_to_f32(src_row[j]);
            }
        } else {
            for (int j = 0; j < qk; ++j) {
                const SrcScalar * src_val = (const SrcScalar *) (cx + x_offset + j * nb00);
                xf[j] = ggml_sycl_src_to_f32(*src_val);
            }
        }

        quantize_block((const char *) xf, cdst + dst_offset);
    });
}

template <typename SrcScalar>
static void ggml_sycl_quantize_rows_sycl(const char * cx, char * cdst, const ggml_tensor * src0, const ggml_tensor * src1,
                                         const int64_t ne, const int64_t ne00, const int64_t ne01, const int64_t ne02,
                                         const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                                         const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10,
                                         const size_t nb11, const size_t nb12, const size_t nb13, queue_ptr stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);

    switch (src1->type) {
        case GGML_TYPE_Q8_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            break;
        case GGML_TYPE_Q1_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            break;
        case GGML_TYPE_Q5_1:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            break;
        case GGML_TYPE_Q5_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            break;
        case GGML_TYPE_Q4_1:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            break;
        case GGML_TYPE_Q4_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                               nb01, nb02, nb03, ne10, ne11, ne12,
                                                                               nb10, nb11, nb12, nb13, stream);
            break;
        case GGML_TYPE_MXFP4:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            break;
        case GGML_TYPE_NVFP4:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            break;
        default:
            GGML_ABORT("unsupported quantized target type in sycl quantizer src1->type=%s\n",
                       ggml_type_name(src1->type));
    }
}

static void ggml_cpy_f16_f16_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_f16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_i16_i16_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_i16_i16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_i32_i32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_i32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_q8_0_q8_0(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


static void ggml_cpy_q5_0_q5_0(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


static void ggml_cpy_q5_1_q5_1(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


static void ggml_cpy_q4_0_q4_0(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


static void ggml_cpy_q4_1_q4_1(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {

   const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
   stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_q1_0_q1_0(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_mxfp4_mxfp4(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_nvfp4_nvfp4(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_q2_K_q2_K(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q2_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_q3_K_q3_K(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q3_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_q4_K_q4_K(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q4_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_q5_K_q5_K(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q5_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_q6_K_q6_K(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q6_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq2_xxs_iq2_xxs(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_iq2_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq2_xs_iq2_xs(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_iq2_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq2_s_iq2_s(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_iq2_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq3_xxs_iq3_xxs(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_iq3_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq1_s_iq1_s(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_iq1_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq1_m_iq1_m(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_iq1_m, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq4_nl_iq4_nl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq3_s_iq3_s(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_iq3_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_iq4_xs_iq4_xs(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_iq4_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

#ifdef GGML_SYCL_HAS_BF16
static void ggml_cpy_f32_bf16_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            cpy_f32_f16<cpy_1_f32_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_bf16_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            cpy_f32_f16<cpy_1_bf16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_bf16_bf16_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            cpy_f32_f16<cpy_1_bf16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                            nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_f16_bf16_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            cpy_f32_f16<cpy_1_f16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_bf16_f16_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            cpy_f32_f16<cpy_1_bf16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}
#endif

void ggml_sycl_cpy(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1) try {
    // Unlike other operators ggml_sycl_cpy takes 2 distinct tensors instead of a dst ggml_tensor and rely on its src field
    GGML_SYCL_DEBUG("ggml_sycl_cpy: src0->type=%s, src1->type=%s\n",
        ggml_type_name(src0->type), ggml_type_name(src1->type));
    scope_op_debug_print scope_dbg_print(__func__, src1, /*num_src=*/0, debug_get_tensor_str("\tsrc0", src0));
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    GGML_TENSOR_BINARY_OP_LOCALS01;

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    char * src0_ddc = (char *) src0->data;
    char * src1_ddc = (char *) src1->data;
    if ((src0->type == src1->type) && (ggml_is_contiguous(src0) && ggml_is_contiguous(src1))) {
        GGML_SYCL_DEBUG("%s: memcpy path\n", __func__);
        main_stream->memcpy(src1_ddc, src0_ddc, ggml_nbytes(src0));
    } else if (src0->type == GGML_TYPE_F32 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<float>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00, nb01,
                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<sycl::half>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00,
                                                 nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                 main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } else if (src0->type == GGML_TYPE_BF16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<ggml_bf16_t>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02,
                                                  nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                  nb12, nb13, main_stream);
#endif
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_f32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f16_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f16_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f16_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16) {
        ggml_cpy_i16_i16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_i32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_i32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_mxfp4_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                 nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_q8_0_q8_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_q5_0_q5_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_q5_1_q5_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_q4_0_q4_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_q4_1_q4_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q1_0 && src1->type == GGML_TYPE_Q1_0) {
        ggml_cpy_q1_0_q1_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_MXFP4) {
        ggml_cpy_mxfp4_mxfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_NVFP4 && src1->type == GGML_TYPE_NVFP4) {
        ggml_cpy_nvfp4_nvfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q2_K && src1->type == GGML_TYPE_Q2_K) {
        ggml_cpy_q2_K_q2_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q3_K && src1->type == GGML_TYPE_Q3_K) {
        ggml_cpy_q3_K_q3_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_K && src1->type == GGML_TYPE_Q4_K) {
        ggml_cpy_q4_K_q4_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_K && src1->type == GGML_TYPE_Q5_K) {
        ggml_cpy_q5_K_q5_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q6_K && src1->type == GGML_TYPE_Q6_K) {
        ggml_cpy_q6_K_q6_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ2_XXS && src1->type == GGML_TYPE_IQ2_XXS) {
        ggml_cpy_iq2_xxs_iq2_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ2_XS && src1->type == GGML_TYPE_IQ2_XS) {
        ggml_cpy_iq2_xs_iq2_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ2_S && src1->type == GGML_TYPE_IQ2_S) {
        ggml_cpy_iq2_s_iq2_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ3_XXS && src1->type == GGML_TYPE_IQ3_XXS) {
        ggml_cpy_iq3_xxs_iq3_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ1_S && src1->type == GGML_TYPE_IQ1_S) {
        ggml_cpy_iq1_s_iq1_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ1_M && src1->type == GGML_TYPE_IQ1_M) {
        ggml_cpy_iq1_m_iq1_m(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ4_NL && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_iq4_nl_iq4_nl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ3_S && src1->type == GGML_TYPE_IQ3_S) {
        ggml_cpy_iq3_s_iq3_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_IQ4_XS && src1->type == GGML_TYPE_IQ4_XS) {
        ggml_cpy_iq4_xs_iq4_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f32_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_bf16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_bf16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_bf16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
#endif
    } else {
        GGML_LOG_ERROR("%s: unsupported type combination (%s to %s)\n", __func__, ggml_type_name(src0->type),
                       ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_cpy(ctx, dst->src[0], dst);
}
