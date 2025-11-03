#include "pad_reflect_1d.hpp"

void pad_reflect_1d_f32(const float* src,float* dst,
    const int64_t ne0, const int64_t ne02, const int p0, const int p1,
    const int64_t nb0, const int64_t nb1, const int64_t nb2, const int64_t nb3,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const sycl::nd_item<3> &item_ct1){

    const int i0 = item_ct1.get_group(0) * SYCL_CONCAT_BLOCK_SIZE + item_ct1.get_local_id(0);
    const int i1 = item_ct1.get_group(1);
    const int g2 = item_ct1.get_group(2);
    const int i2 = g2 % ne02;
    const int i3 = g2 / ne02;

    if (i0 >= p0 + ne0 + p1) return;

    int t = i0 - p0;
    int period = 2 * ne0 -2;
    int m = t % period;
    m += (m < 0) * period;
    int center = ne0 -1;
    int srci0 = center - abs(center - m);

    int offest_src = i3*nb3 + i2*nb2 + i1*nb1 + srci0*nb0;
    int offest_dst =  i3*nb03 +  i2*nb02 +  i1*nb01 +  i0*nb00;
    dst[offest_dst] = src[offest_src];

}

void ggml_sycl_op_pad_reflect_1d(ggml_backend_sycl_context& ctx, ggml_tensor* dst){

    const ggml_tensor * src0 = dst->src[0];
    queue_ptr           stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *) dst->op_params;
    const int p0 = opts[0];
    const int p1 = opts[1];

    const int64_t ne0 = src0->ne[0];

    const int64_t ne00 = dst->ne[0];
    const int64_t ne01 = dst->ne[1];
    const int64_t ne02 = dst->ne[2];
    const int64_t ne03 = dst->ne[3];

    const int64_t nb00 = dst->nb[0];
    const int64_t nb01 = dst->nb[1];
    const int64_t nb02 = dst->nb[2];
    const int64_t nb03 = dst->nb[3];
    const int64_t nb0 = src0->nb[0];
    const int64_t nb1 = src0->nb[1];
    const int64_t nb2 = src0->nb[2];
    const int64_t nb3 = src0->nb[3];

    int num_blocks = (ne00 + SYCL_CONCAT_BLOCK_SIZE - 1) / SYCL_CONCAT_BLOCK_SIZE;
    sycl::range<3> global(num_blocks * SYCL_CONCAT_BLOCK_SIZE, ne01, ne02*ne03);
    sycl::range<3> local(SYCL_CONCAT_BLOCK_SIZE, 1, 1);

    stream->parallel_for(
        sycl::nd_range<3>(global,
                            local),
        [=](sycl::nd_item<3> item_ct1) { pad_reflect_1d_f32(
            (const float *) src0->data, (float *) dst->data,
            ne0, ne02, p0, p1,
            nb0, nb1, nb2, nb3,
            nb00, nb01, nb02, nb03
            , item_ct1);
         });
}
