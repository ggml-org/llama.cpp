#include "common.h"

template<uint32_t ttype>
bool _ggml_vec_tri_cmp(const int i, const int r);

template<>
bool _ggml_vec_tri_cmp</* GGML_TRI_TYPE_LOWER */ 3>(const int i, const int r) {
    return i < r;
}

template<>
bool _ggml_vec_tri_cmp</* GGML_TRI_TYPE_LOWER_DIAG */ 2>(const int i, const int r) {
    return i <= r;
}

template<>
bool _ggml_vec_tri_cmp</* GGML_TRI_TYPE_UPPER */ 1>(const int i, const int r) {
    return i > r;
}

template<>
bool _ggml_vec_tri_cmp</* GGML_TRI_TYPE_UPPER_DIAG */ 0>(const int i, const int r) {
    return i >= r;
}

template<typename T, int ttype>
kernel void kernel_tri(
        constant ggml_metal_kargs_tri & args,
        device const char * src0,
        device const char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    if (i3 >= args.ne03 || i2 >= args.ne02 || i1 >= args.ne01) {
        return;
    }

    device const T * src_row = (device const T *) ((device const char *) src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03);
    device       T * dst_row = (device       T *) ((device       char *) dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3);

    // Each thread is a single element of the row if ne00 < max threads per
    // threadgroup, so this will loop once for each index that this thread is
    // responsible for
    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        // Use the comparison as a mask for branchless
        dst_row[i0] = static_cast<T>(_ggml_vec_tri_cmp<ttype>(i0, i1)) * src_row[i0];
    }
}

typedef decltype(kernel_tri<float, 0>) kernel_tri_t;

template [[host_name("kernel_tri_f32_0")]] kernel kernel_tri_t kernel_tri<float, 0>;
template [[host_name("kernel_tri_f32_1")]] kernel kernel_tri_t kernel_tri<float, 1>;
template [[host_name("kernel_tri_f32_2")]] kernel kernel_tri_t kernel_tri<float, 2>;
template [[host_name("kernel_tri_f32_3")]] kernel kernel_tri_t kernel_tri<float, 3>;
template [[host_name("kernel_tri_f16_0")]] kernel kernel_tri_t kernel_tri<half, 0>;
template [[host_name("kernel_tri_f16_1")]] kernel kernel_tri_t kernel_tri<half, 1>;
template [[host_name("kernel_tri_f16_2")]] kernel kernel_tri_t kernel_tri<half, 2>;
template [[host_name("kernel_tri_f16_3")]] kernel kernel_tri_t kernel_tri<half, 3>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_tri_bf16_0")]] kernel kernel_tri_t kernel_tri<bfloat, 0>;
template [[host_name("kernel_tri_bf16_1")]] kernel kernel_tri_t kernel_tri<bfloat, 1>;
template [[host_name("kernel_tri_bf16_2")]] kernel kernel_tri_t kernel_tri<bfloat, 2>;
template [[host_name("kernel_tri_bf16_3")]] kernel kernel_tri_t kernel_tri<bfloat, 3>;
#endif
