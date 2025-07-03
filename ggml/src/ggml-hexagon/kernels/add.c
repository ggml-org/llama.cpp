#include "ggml-dsp.h"

static inline void l2fetch(const void * p, uint32_t stride,
                           uint32_t width, uint32_t height,
                           uint32_t dir) {
    uint64_t control = HEXAGON_V64_CREATE_H(dir, stride, width, height);
    __asm__ __volatile__ (" l2fetch(%0,%1) " : :"r"(p),"r"(control));
}

static inline void ggmlhexagon_dsp_add_f32(const int n, float * GGML_RESTRICT z, const float * GGML_RESTRICT x, const float * GGML_RESTRICT y) {
    HVX_Vector * va;
    HVX_Vector * vb;
    HVX_Vector * vc;
    HVX_Vector qf32;
    const size_t FLOATS_PER_VECTOR = 128 / sizeof(float);
    const size_t block  = n / FLOATS_PER_VECTOR;
    const size_t left   = n % FLOATS_PER_VECTOR;
    const size_t blocks = block * FLOATS_PER_VECTOR;

    if ((((uintptr_t)z | (uintptr_t)x | (uintptr_t)y) % ALIGN_128_BYTE) != 0) {
        GGMLHEXAGON_LOG_DEBUG("memaddress mismatch alignment 128 bytes z:%p x:%p y:%p", z, x, y);
        for (size_t i = 0; i < n; ++i)
            z[i] = x[i] + y[i];

        return;
    }

    va = (HVX_Vector *)x;
    vb = (HVX_Vector *)y;
    vc = (HVX_Vector *)z;
    //unroll is better but need more carefully check for various cases and I think DSP also don't like branch predication
    for (size_t i = 0; i < block; ++i) {
        l2fetch(va + VLEN, VLEN, VLEN, 1, 0);
        l2fetch(vb + VLEN, VLEN, VLEN, 1, 0);
        //*vc++ = Q6_Vsf_vadd_VsfVsf(*va++, *vb++);
        qf32 = Q6_Vqf32_vadd_VsfVsf(*va++, *vb++);
        *vc++ = Q6_Vsf_equals_Vqf32(qf32);
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

    const int rank = ggml_n_dims(src0);
    if (1 == rank) {
        //element-wise addition with vector
        const size_t len = src0->ne[0];
        float * dst_ptr  = (float *) (dst->data);
        float * src0_ptr = (float *) (src0->data);
        float * src1_ptr = (float *) (src1->data);
        ggmlhexagon_dsp_add_f32(len, dst_ptr, src0_ptr, src1_ptr);
        return;
    }

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

int ggmlop_dsp_add(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s\n", __func__);
    ggml_compute_forward_add_f32(src0, src1, dst);
    GGMLHEXAGON_LOG_DEBUG("leave %s\n", __func__);
    return 0;
}
