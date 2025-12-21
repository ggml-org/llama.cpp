#include "binbcast.hpp"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>
#include <sycl/sycl.hpp>

#include "ggml.h"

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast(const src0_t * src0, const src1_t * src1, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        /*int s0, */ int s1,  int s2,  int s3,
        /*int s00,*/ int s01, int s02, int s03,
        /*int s10,*/ int s11, int s12, int s13,
        const sycl::nd_item<3> &item_ct1) {
    const int i0s = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    const int i1 = (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1));
    const int i2 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) /
                   ne3;
    const int i3 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) %
                   ne3;

    if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    for (int i0 = i0s; i0 < ne0;
         i0 += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        const int i10 = i0 % ne10;
        dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0] : 0.0f, (float)src1_row[i10]);
    }
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast_unravel(const src0_t * src0, const src1_t * src1, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        /*int s0, */ int s1,  int s2,  int s3,
        /*int s00,*/ int s01, int s02, int s03,
        /*int s10,*/ int s11, int s12, int s13,
        const sycl::nd_item<3> &item_ct1) {

    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    const int i3 = i/(ne2*ne1*ne0);
    const int i2 = (i/(ne1*ne0)) % ne2;
    const int i1 = (i/ne0) % ne1;
    const int i0 = i % ne0;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    const int i10 = i0 % ne10;
    dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0] : 0.0f, (float)src1_row[i10]);
}


template<float (*bin_op)(const float, const float)>
struct bin_bcast_sycl {
    template <typename src0_t, typename src1_t, typename dst_t>
    void operator()(const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd, const int64_t ne00,
                    const int64_t ne01, const int64_t ne02, const int64_t ne03, const int64_t ne10, const int64_t ne11,
                    const int64_t ne12, const int64_t ne13, const int64_t ne0, const int64_t ne1, const int64_t ne2,
                    const int64_t ne3, const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                    const size_t nb10, const size_t nb11, const size_t nb12, const size_t nb13, const size_t nb0,
                    const size_t nb1, const size_t nb2, const size_t nb3, const bool src0_is_contiguous,
                    const bool src1_is_contiguous, const bool dst_is_contiguous, queue_ptr stream) {
        int nr0 = ne10 / ne0;
        int nr1 = ne11/ne1;
        int nr2 = ne12/ne2;
        int nr3 = ne13/ne3;

        int nr[4] = { nr0, nr1, nr2, nr3 };

        // collapse dimensions until first broadcast dimension
        int64_t cne[] = {ne0, ne1, ne2, ne3};
        int64_t cne0[] = {ne00, ne01, ne02, ne03};
        int64_t cne1[] = {ne10, ne11, ne12, ne13};
        size_t cnb[] = {nb0, nb1, nb2, nb3};
        size_t cnb0[] = {nb00, nb01, nb02, nb03};
        size_t cnb1[] = {nb10, nb11, nb12, nb13};
        auto collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        auto collapse_nb = [](size_t cnb[], int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        if (src0_is_contiguous && src1_is_contiguous && dst_is_contiguous) {
            for (int i = 0; i < 4; i++) {
                if (nr[i] != 1) {
                    break;
                }
                if (i > 0) {
                    collapse_nb(cnb, cne);
                    collapse_nb(cnb0, cne0);
                    collapse_nb(cnb1, cne1);
                    collapse(cne);
                    collapse(cne0);
                    collapse(cne1);
                }
            }
        }
        {
            int64_t ne0 = cne[0];
            int64_t ne1 = cne[1];
            int64_t ne2 = cne[2];
            int64_t ne3 = cne[3];

            int64_t ne10 = cne1[0];
            int64_t ne11 = cne1[1];
            int64_t ne12 = cne1[2];
            int64_t ne13 = cne1[3];

            size_t nb0 = cnb[0];
            size_t nb1 = cnb[1];
            size_t nb2 = cnb[2];
            size_t nb3 = cnb[3];

            size_t nb00 = cnb0[0];
            size_t nb01 = cnb0[1];
            size_t nb02 = cnb0[2];
            size_t nb03 = cnb0[3];

            size_t nb10 = cnb1[0];
            size_t nb11 = cnb1[1];
            size_t nb12 = cnb1[2];
            size_t nb13 = cnb1[3];

            size_t s0 = nb0 / sizeof(dst_t);
            size_t s1 = nb1 / sizeof(dst_t);
            size_t s2 = nb2 / sizeof(dst_t);
            size_t s3 = nb3 / sizeof(dst_t);

            size_t s10 = nb10 / sizeof(src1_t);
            size_t s11 = nb11 / sizeof(src1_t);
            size_t s12 = nb12 / sizeof(src1_t);
            size_t s13 = nb13 / sizeof(src1_t);

            size_t s00 = nb00 / sizeof(src0_t);
            size_t s01 = nb01 / sizeof(src0_t);
            size_t s02 = nb02 / sizeof(src0_t);
            size_t s03 = nb03 / sizeof(src0_t);

            GGML_UNUSED(s00);

            GGML_ASSERT(nb0 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb1 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb2 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb3 % sizeof(dst_t) == 0);

            GGML_ASSERT(nb00 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb01 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb02 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb03 % sizeof(src0_t) == 0);

            GGML_ASSERT(nb10 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb11 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb12 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb13 % sizeof(src1_t) == 0);

            GGML_ASSERT(s0 == 1);
            GGML_ASSERT(s10 == 1);

            const int block_size = 128;

            int64_t hne0 = std::max(ne0/2LL, 1LL);

            sycl::range<3> block_dims(1, 1, 1);
            block_dims[2] = std::min<unsigned int>(hne0, block_size);
            block_dims[1] = std::min<unsigned int>(
                ne1, block_size / (unsigned int)block_dims[2]);
            block_dims[0] = std::min(
                std::min<unsigned int>(
                    ne2 * ne3, block_size / (unsigned int)block_dims[2] /
                                   (unsigned int)block_dims[1]),
                64U);

            sycl::range<3> block_nums(
                (ne2 * ne3 + block_dims[0] - 1) / block_dims[0],
                (ne1 + block_dims[1] - 1) / block_dims[1],
                (hne0 + block_dims[2] - 1) / block_dims[2]);

            if (block_nums[0] > 65535) {
                // this is the maximum number of blocks in z direction, fallback to 1D grid kernel
                int block_num = (ne0*ne1*ne2*ne3 + block_size - 1) / block_size;
                {
                    dpct::has_capability_or_fail(stream->get_device(),
                                                 {sycl::aspect::fp16});

                    stream->parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, block_num) *
                                              sycl::range<3>(1, 1, block_size),
                                          sycl::range<3>(1, 1, block_size)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_bin_bcast_unravel<bin_op>(
                                src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3,
                                ne10, ne11, ne12, ne13, s1, s2, s3, s01, s02,
                                s03, s11, s12, s13, item_ct1);
                        });
                }
            } else {
                /*
                DPCT1049:16: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_bin_bcast<bin_op>(src0_dd, src1_dd, dst_dd, ne0, ne1,
                                            ne2, ne3, ne10, ne11, ne12, ne13,
                                            s1, s2, s3, s01, s02, s03, s11, s12, s13,
                                            item_ct1);
                    });
            }
        }
    }
};

template <class op>
inline void ggml_sycl_op_bin_bcast(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                   ggml_tensor * dst) {
    dpct::queue_ptr main_stream = ctx.stream();
    GGML_TENSOR_BINARY_OP_LOCALS

    // Use device-specific data pointers for TP support
    const int device = ctx.device;
    void * src0_d = ggml_sycl_get_data_ptr(src0, device);
    void * src1_d = ggml_sycl_get_data_ptr(src1, device);
    void * dst_d  = ggml_sycl_get_data_ptr(dst, device);

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()((const float *) src0_d, (const float *) src1_d, (float *) dst_d, ne00, ne01, ne02, ne03, ne10,
             ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3,
             ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_contiguous(dst), main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()((const sycl::half *) src0_d, (const sycl::half *) src1_d, (sycl::half *) dst_d, ne00, ne01,
             ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13,
             nb0, nb1, nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_contiguous(dst),
             main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        op()((const sycl::half *) src0_d, (const float *) src1_d, (sycl::half *) dst_d, ne00, ne01, ne02,
             ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1,
             nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_contiguous(dst), main_stream);
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
        op()((const int32_t *) src0_d, (const int32_t *) src1_d, (int32_t *) dst_d, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_contiguous(dst), main_stream);
    } else if (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && dst->type == GGML_TYPE_I16) {
        op()((const int16_t *) src0_d, (const int16_t *) src1_d, (int16_t *) dst_d, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_contiguous(dst), main_stream);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__, ggml_type_name(dst->type),
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

inline void ggml_sycl_op_add(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_add>>(ctx, dst->src[0], dst->src[1], dst);
}

inline void ggml_sycl_op_sub(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_sub>>(ctx, dst->src[0], dst->src[1], dst);
}

inline void ggml_sycl_op_mul(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_mul>>(ctx, dst->src[0], dst->src[1], dst);
}

inline void ggml_sycl_op_div(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_div>>(ctx, dst->src[0], dst->src[1], dst);
}

inline void ggml_sycl_op_repeat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_repeat>>(ctx, dst, dst->src[0], dst);
}


void ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_add(ctx, dst);
}

void ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_sub(ctx, dst);
}

void ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ggml_sycl_op_mul(ctx, dst);

    // Cache FFN norm output for TP: the GGML scheduler may reuse this buffer
    // before device 1 can access it. Cache immediately after MUL completes.
    if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1 &&
        strncmp(dst->name, "ffn_norm-", 9) == 0) {
        int layer = atoi(dst->name + 9);
        size_t size = ggml_nbytes(dst);
        // IMPORTANT: Use device-specific pointer for TP mode!
        void * dst_ptr = ggml_sycl_get_data_ptr(dst, ctx.device);
        // DEBUG: Check if MUL runs for batch=1
        static int mul_b1_dbg = 0;
        if (g_ggml_sycl_tp_debug && dst->ne[1] == 1 && layer == 0 && mul_b1_dbg++ < 5) {
            ctx.stream()->wait();
            float check[4];
            ctx.stream()->memcpy(check, dst_ptr, 4*sizeof(float)).wait();
            fprintf(stderr, "TP DEBUG MUL ffn_norm-0 batch=1: caching dst_ptr=%p dst[0..3]=[%f,%f,%f,%f]\n",
                    dst_ptr, check[0], check[1], check[2], check[3]);
        }
        ggml_sycl_tp_cache_ffn_norm(layer, dst_ptr, dst->ne[0], dst->ne[1],
                                     size, ctx.stream());
    }
}

void ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_div(ctx, dst);
}

void ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat(ctx, dst);
}

// Specialized ADD1 kernel: dst[i] = src0[i] + scalar
// Much more efficient than generic broadcast for single-scalar addition
template<typename T>
static void k_add1(
    const T * __restrict__ src0,
    const T scalar,
    T * __restrict__ dst,
    const int64_t n,
    const sycl::nd_item<3> & item) {

    const int64_t i = item.get_global_id(2);
    if (i >= n) return;

    dst[i] = src0[i] + scalar;
}

// ADD1 operation: add a single scalar to all elements
// Optimized path when src1 has exactly 1 element
void ggml_sycl_add1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_nelements(src1) == 1);

    const int device = ctx.device;
    dpct::queue_ptr stream = ctx.stream();

    const int64_t n = ggml_nelements(src0);

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        const float * src0_d = (const float *) ggml_sycl_get_data_ptr(src0, device);
        float * dst_d = (float *) ggml_sycl_get_data_ptr(dst, device);

        // Load scalar from device memory
        float scalar;
        stream->memcpy(&scalar, ggml_sycl_get_data_ptr(src1, device), sizeof(float)).wait();

        const int block_size = 256;
        const int num_blocks = (n + block_size - 1) / block_size;

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item) {
                k_add1(src0_d, scalar, dst_d, n, item);
            });
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        const sycl::half * src0_d = (const sycl::half *) ggml_sycl_get_data_ptr(src0, device);
        sycl::half * dst_d = (sycl::half *) ggml_sycl_get_data_ptr(dst, device);

        // Load scalar from device memory (src1 is F32)
        float scalar_f32;
        stream->memcpy(&scalar_f32, ggml_sycl_get_data_ptr(src1, device), sizeof(float)).wait();
        sycl::half scalar = sycl::half(scalar_f32);

        const int block_size = 256;
        const int num_blocks = (n + block_size - 1) / block_size;

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item) {
                k_add1(src0_d, scalar, dst_d, n, item);
            });
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        const sycl::half * src0_d = (const sycl::half *) ggml_sycl_get_data_ptr(src0, device);
        sycl::half * dst_d = (sycl::half *) ggml_sycl_get_data_ptr(dst, device);

        // Load scalar from device memory
        sycl::half scalar;
        stream->memcpy(&scalar, ggml_sycl_get_data_ptr(src1, device), sizeof(sycl::half)).wait();

        const int block_size = 256;
        const int num_blocks = (n + block_size - 1) / block_size;

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item) {
                k_add1(src0_d, scalar, dst_d, n, item);
            });
    } else {
        // Fallback to generic broadcast for unsupported types
        ggml_sycl_op_add(ctx, dst);
    }
}

// Fused MUL + ADD kernel: dst = x * scale + bias
// Optimized for the common scale+bias pattern in normalization
template<typename T>
static void k_mul_add_fused(
    const T * __restrict__ x,
    const T * __restrict__ scale,
    const T * __restrict__ bias,
    T * __restrict__ dst,
    const int64_t ne0,
    const int64_t ne1,
    const int64_t ne_scale0,
    const int64_t ne_bias0,
    const sycl::nd_item<3> & item) {

    const int64_t i0 = item.get_global_id(2);
    const int64_t i1 = item.get_global_id(1);

    if (i0 >= ne0 || i1 >= ne1) {
        return;
    }

    const int64_t idx = i1 * ne0 + i0;
    const int64_t scale_idx = i0 % ne_scale0;
    const int64_t bias_idx = i0 % ne_bias0;

    dst[idx] = x[idx] * scale[scale_idx] + bias[bias_idx];
}

// Fused MUL + ADD operation
// Pattern: mul_node = x * scale, add_node = mul_node + bias
// Fused: dst = x * scale + bias
void ggml_sycl_op_mul_add_fused(ggml_backend_sycl_context & ctx,
                                 ggml_tensor * mul_node,
                                 ggml_tensor * add_node) {
    GGML_ASSERT(mul_node->op == GGML_OP_MUL);
    GGML_ASSERT(add_node->op == GGML_OP_ADD);

    // Get input tensors
    // MUL: src[0] = x, src[1] = scale
    // ADD: src[0] = mul_result, src[1] = bias
    ggml_tensor * x = mul_node->src[0];
    ggml_tensor * scale = mul_node->src[1];
    ggml_tensor * bias = add_node->src[1];

    // Handle case where scale/bias might be swapped
    if (ggml_nelements(x) < ggml_nelements(scale)) {
        std::swap(x, scale);
    }
    if (add_node->src[0] != mul_node && add_node->src[1] == mul_node) {
        // ADD has form: bias + mul_result, swap to get mul_result + bias
        bias = add_node->src[0];
    }

    // Verify types
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(bias->type == GGML_TYPE_F32);
    GGML_ASSERT(add_node->type == GGML_TYPE_F32);

    // Get data pointers
    const int device = ctx.device;
    const float * x_d = (const float *) ggml_sycl_get_data_ptr(x, device);
    const float * scale_d = (const float *) ggml_sycl_get_data_ptr(scale, device);
    const float * bias_d = (const float *) ggml_sycl_get_data_ptr(bias, device);
    float * dst_d = (float *) ggml_sycl_get_data_ptr(add_node, device);

    dpct::queue_ptr stream = ctx.stream();

    const int64_t ne0 = x->ne[0];
    const int64_t ne1 = ggml_nrows(x);
    const int64_t ne_scale0 = scale->ne[0];
    const int64_t ne_bias0 = bias->ne[0];

    // Launch kernel
    const int block_size = 256;
    const int grid_x = (ne0 + block_size - 1) / block_size;
    const int grid_y = ne1;

    sycl::range<3> block_dims(1, 1, block_size);
    sycl::range<3> grid_dims(1, grid_y, grid_x * block_size);

    stream->parallel_for(
        sycl::nd_range<3>(grid_dims, block_dims),
        [=](sycl::nd_item<3> item) {
            k_mul_add_fused(x_d, scale_d, bias_d, dst_d, ne0, ne1, ne_scale0, ne_bias0, item);
        });
}

