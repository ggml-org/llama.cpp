#include "ggml-dsp.h"

inline static void ggmlhexagon_dsp_add_f32 (const int n, float * z, const float * x, const float * y) {
    HVX_Vector * va;
    HVX_Vector * vb;
    HVX_Vector * vc;
    HVX_Vector qf32;
    const int FLOATS_PER_VECTOR = 128 / sizeof(float);
    const int block  = n / FLOATS_PER_VECTOR;
    const int left   = n % FLOATS_PER_VECTOR;
    const int blocks = block * FLOATS_PER_VECTOR;

    if ((((uintptr_t)z | (uintptr_t)x | (uintptr_t)y) % ALIGN_128_BYTE) != 0) {
        GGMLHEXAGON_LOG_DEBUG("memaddress mismatch alignment 128 bytes z:%p x:%p y:%p", z, x, y);
        for (size_t i = 0; i < n; ++i)
            z[i] = x[i] + y[i];

        return;
    }

    va = (HVX_Vector *)x;
    vb = (HVX_Vector *)y;
    vc = (HVX_Vector *)z;
    for (size_t i = 0; i < block; ++i) {
        //*vc++ = Q6_Vsf_vadd_VsfVsf(*va++, *vb++);
        qf32 = Q6_Vqf32_vadd_VsfVsf(*va++, *vb++);
        *vc = Q6_Vsf_equals_Vqf32(qf32);
        vc++;
    }

    if (left > 0) {
        for (size_t i = 0; i < left; ++i)
            z[i + blocks] = x[i + blocks] + y[i + blocks];
    }
}

static void ggml_compute_forward_add_f32(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    uint64_t start_time = ggml_time_us();

    memcpy(dst->ne, src1->ne, 16);
    memcpy(dst->nb, src1->nb, 16);
    ggmlhexagon_dump_tensor(src0, 1);
    ggmlhexagon_dump_tensor(src1, 1);
    ggmlhexagon_dump_tensor(dst, 1);

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    const int ith = 0;
    const int nth = 1;

    const int nr  = ggml_nrows(src0);
    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    const int dr = (nr + nth - 1)/nth;
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);
    if (nb10 == sizeof(float)) {
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int32_t i03 = ir/(ne02*ne01);
            const int32_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int32_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int32_t i13 = i03 % ne13;
            const int32_t i12 = i02 % ne12;
            const int32_t i11 = i01 % ne11;
            const int32_t nr0 = ne00 / ne10;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
            float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);
            for (int32_t r = 0; r < nr0; ++r) {
                ggmlhexagon_dsp_add_f32(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        }
    } else {
        // src1 is not contiguous
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int32_t i03 = ir/(ne02*ne01);
            const int32_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int32_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int32_t i13 = i03 % ne13;
            const int32_t i12 = i02 % ne12;
            const int32_t i11 = i01 % ne11;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

            for (int32_t i0 = 0; i0 < ne0; ++i0) {
                const int32_t i10 = i0 % ne10;
                float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10);

                dst_ptr[i0] = src0_ptr[i0] + *src1_ptr;
            }
        }
    }

    uint64_t end_time = ggml_time_us();
    uint64_t duration = (end_time - start_time);
    GGMLHEXAGON_LOG_DEBUG("duration %llu us", duration);
#if !GGMLHEXAGON_DEBUG
    UNUSED(duration);
#endif

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
}

//FIXME: why failed with test-backend-ops when disable ion rpc mempool
int ggmlop_dsp_add(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst)
{
    GGMLHEXAGON_LOG_DEBUG("enter %s\n", __func__);
    switch (src0->type) {
        case GGML_TYPE_F32:
        {
            if (src1->type == GGML_TYPE_F32) {
                ggml_compute_forward_add_f32(src0, src1, dst);
            } else {
                GGML_ABORT("fatal error");
            }
            break;
        }
        default:
        {
            GGML_ABORT("fatal error");
        }
    }
    GGMLHEXAGON_LOG_DEBUG("leave %s\n", __func__);
    return 0;
}
