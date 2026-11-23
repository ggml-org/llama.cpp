#pragma OPENCL EXTENSION cl_khr_fp16 : enable

inline float xielu(float x, float alpha_n, float alpha_p, float beta, float eps) {
    return x > 0.0f ? alpha_p*x*x + beta*x : (expm1(fmin(x, eps)) - x)*alpha_n + beta*x;
}

#define XIELU_KERNEL(NAME, TYPE)                                                      \
kernel void kernel_xielu_##NAME(                                                      \
        global const char * src0, ulong offset0,                                      \
        global       char * dst,  ulong offsetd,                                      \
        int ne00, ulong nb00, ulong nb01, ulong nb02, ulong nb03,                     \
        ulong nb0, ulong nb1, ulong nb2, ulong nb3,                                   \
        float alpha_n, float alpha_p, float beta, float eps) {                        \
    const size_t i3 = get_group_id(2);                                                \
    const size_t i2 = get_group_id(1);                                                \
    const size_t i1 = get_group_id(0);                                                \
    src0 += offset0 + i3*nb03 + i2*nb02 + i1*nb01;                                    \
    dst  += offsetd + i3*nb3  + i2*nb2  + i1*nb1;                                     \
    for (size_t i0 = get_local_id(0); i0 < (size_t) ne00; i0 += get_local_size(0)) {  \
        const float x = *(global const TYPE *)(src0 + i0*nb00);                       \
        *(global TYPE *)(dst + i0*nb0) = xielu(x, alpha_n, alpha_p, beta, eps);       \
    }                                                                                 \
}

XIELU_KERNEL(f32, float)
XIELU_KERNEL(f16, half)
