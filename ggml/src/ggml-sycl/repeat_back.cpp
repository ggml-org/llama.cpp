#include "repeat_back.hpp"

#include "common.hpp"

void ggml_sycl_op_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const float * src0_dd = (const float *) dst->src[0]->data;
    float *       dst_dd  = (float *) dst->data;

    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    const int64_t ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1], ne02 = dst->src[0]->ne[2],
                  ne03 = dst->src[0]->ne[3];

    const int nr0 = (int) (ne00 / ne0);
    const int nr1 = (int) (ne01 / ne1);
    const int nr2 = (int) (ne02 / ne2);
    const int nr3 = (int) (ne03 / ne3);

    const size_t total      = ne0 * ne1 * ne2 * ne3;
    const int    BLOCK_SIZE = 256;
    const int    num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    queue_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            const size_t i = item_ct1.get_global_linear_id();
            if (i >= total) {
                return;
            }

            const int i0 = i % ne0;
            const int i1 = (i / ne0) % ne1;
            const int i2 = (i / (ne0 * ne1)) % ne2;
            const int i3 = i / (ne0 * ne1 * ne2);

            float acc = 0.0f;

            for (int j3 = 0; j3 < nr3; ++j3) {
                for (int j2 = 0; j2 < nr2; ++j2) {
                    for (int j1 = 0; j1 < nr1; ++j1) {
                        for (int j0 = 0; j0 < nr0; ++j0) {
                            acc += src0_dd[(i0 + j0 * ne0) + (i1 + j1 * ne1) * ne00 + (i2 + j2 * ne2) * ne00 * ne01 +
                                           (i3 + j3 * ne3) * ne00 * ne01 * ne02];
                        }
                    }
                }
            }

            dst_dd[i] = acc;
        });
}
