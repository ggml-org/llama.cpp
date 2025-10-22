#include "roll.hpp"
#include "common.hpp"

using namespace sycl;

static void kernel_roll_multi_axis(queue &q, const ggml_tensor *src, ggml_tensor *dst,
                                    int shift0, int shift1, int shift2, int shift3) {
    if (!src || !dst) throw std::runtime_error("null tensor");
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
        throw std::runtime_error("only F32 supported in SYCL roll");

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];
    const int ne2 = dst->ne[2];
    const int ne3 = dst->ne[3];

    if (ne0 != src->ne[0] || ne1 != src->ne[1] || ne2 != src->ne[2] || ne3 != src->ne[3])
        throw std::runtime_error("src/dst shape mismatch");


    const int sh0 = ne0 > 0 ? ((int)shift0 % ne0 + ne0) % ne0 : 0;
    const int sh1 = ne1 > 0 ? ((int)shift1 % ne1 + ne1) % ne1 : 0;
    const int sh2 = ne2 > 0 ? ((int)shift2 % ne2 + ne2) % ne2 : 0;
    const int sh3 = ne3 > 0 ? ((int)shift3 % ne3 + ne3) % ne3 : 0;


    const int shNe0 = ne0 - sh0;
    const int shNe1 = ne1 - sh1;
    const int shNe2 = ne2 - sh2;
    const int shNe3 = ne3 - sh3;

    const float *src_d = (const float*) src->data;
    float *dst_d = (float*) dst->data;

    if (!src_d || !dst_d) throw std::runtime_error("null data pointers");

    q.submit([&](handler &h) {
        range<3> r((size_t)ne3, (size_t)ne2, (size_t)ne1);
        h.parallel_for(r, [=](id<3> idx) {
            const int i3 = (int)idx[0];
            const int i2 = (int)idx[1];
            const int i1 = (int)idx[2];

            for (int i0 = 0; i0 < ne0; i0++) {
                const int idx_dst = i0 + i1 * ne0 + i2 * ne0 * ne1 + i3 * ne0 * ne1 * ne2;


                const int src_i0 = (i0 + shNe0) % ne0;
                const int src_i1 = (i1 + shNe1) % ne1;
                const int src_i2 = (i2 + shNe2) % ne2;
                const int src_i3 = (i3 + shNe3) % ne3;

                const int idx_src = src_i0 + src_i1 * ne0 +
                                        src_i2 * ne0 * ne1 + src_i3 * ne0 * ne1 * ne2;

                dst_d[idx_dst] = src_d[idx_src];
            }
        });
    });
}

void ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const ggml_tensor *src = dst->src[0];

    const int32_t *params = (const int32_t *)dst->op_params;
    const int shift0 = params[0];
    const int shift1 = params[1];
    const int shift2 = params[2];
    const int shift3 = params[3];


    if (shift0 == 0 && shift1 == 0 && shift2 == 0 && shift3 == 0) {
        const size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(dst->data, src->data, nb)));
        return;
    }

    try {
        queue *q = ctx.stream();
        kernel_roll_multi_axis(*q, src, dst, shift0, shift1, shift2, shift3);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[SYCL-ROLL] ERROR: %s\n", e.what());
        throw;
    }
}
