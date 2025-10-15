#include "common.h"
#include "dequantize.h"

template<typename src0_t, typename src1_t>
kernel void kernel_out_prod_impl(
    constant ggml_metal_kargs_out_prod & args,
    device const char * src0,
    device const char * src1,
    device       char * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    const int dps2 = args.ne02 > 0 ? args.ne2 / args.ne02 : 1;
    const int dps3 = args.ne03 > 0 ? args.ne3 / args.ne03 : 1;

    const int i02 = args.ne02 > 0 ? i2 / dps2 : 0;
    const int i03 = args.ne03 > 0 ? i3 / dps3 : 0;

    device const char * src0_base = src0 + i02*args.nb02 + i03*args.nb03;
    device const char * src1_base = src1 + i1*args.nb10 + i2*args.nb12 + i3*args.nb13;
    device       char * dst_base  = dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        float acc = 0.0f;

        for (int i01 = 0; i01 < args.ne01; ++i01) {
            device const char * src0_row = src0_base + i01*args.nb01;
            const float v0 = (float) *((device const src0_t *)(src0_row + i0*args.nb00));
            const float v1 = (float) *((device const src1_t *)(src1_base + i01*args.nb11));

            acc += v0 * v1;
        }

        *((device float *)(dst_base + i0*args.nb0)) = acc;
    }
}

typedef decltype(kernel_out_prod_impl<float, float>) kernel_out_prod_f32_t;
typedef decltype(kernel_out_prod_impl<half,  float>) kernel_out_prod_f16_f32_t;
typedef decltype(kernel_out_prod_impl<float, half >) kernel_out_prod_f32_f16_t;
typedef decltype(kernel_out_prod_impl<half,  half >) kernel_out_prod_f16_t;

template [[host_name("kernel_out_prod_f32")]]      kernel kernel_out_prod_f32_t      kernel_out_prod_impl<float, float>;
template [[host_name("kernel_out_prod_f16_f32")]]  kernel kernel_out_prod_f16_f32_t  kernel_out_prod_impl<half,  float>;
template [[host_name("kernel_out_prod_f32_f16")]]  kernel kernel_out_prod_f32_f16_t  kernel_out_prod_impl<float, half>;
template [[host_name("kernel_out_prod_f16")]]      kernel kernel_out_prod_f16_t      kernel_out_prod_impl<half,  half>;

template <typename src1_t>
kernel void kernel_out_prod_q8_0_impl(
    constant ggml_metal_kargs_out_prod & args,
    device const char * src0,
    device const char * src1,
    device       char * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    const int dps2 = args.ne02 > 0 ? args.ne2 / args.ne02 : 1;
    const int dps3 = args.ne03 > 0 ? args.ne3 / args.ne03 : 1;

    const int i02 = args.ne02 > 0 ? i2 / dps2 : 0;
    const int i03 = args.ne03 > 0 ? i3 / dps3 : 0;

    device const char * src0_base = src0 + i02*args.nb02 + i03*args.nb03;
    device const char * src1_base = src1 + i1*args.nb10 + i2*args.nb12 + i3*args.nb13;
    device       char * dst_base  = dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int ib = i0 / QK8_0;
        const int ix = i0 % QK8_0;

        float acc = 0.0f;

        for (int i01 = 0; i01 < args.ne01; ++i01) {
            device const char * src0_row_char = src0_base + i01*args.nb01;
            device const block_q8_0 * src0_row = (device const block_q8_0 *) src0_row_char;
            // read the two needed fields through the device pointer instead of
            // copying the whole 34-byte block per iteration: the copy made each
            // thread pull the full block while using 3 bytes of it, defeating
            // coalescing across the threadgroup and multiplying memory traffic
            // ~30x. On slow-bandwidth phone GPUs the resulting command buffers
            // exceeded the OS watchdog during finetuning backward passes
            // (kIOGPUCommandBufferCallbackErrorHang). Same failure class the
            // Vulkan backend already works around with tiled OUT_PROD shaders
            // ("avoid VK_ERROR_DEVICE_LOST due to slow threads").
            device const block_q8_0 * blk = src0_row + ib;

            const float v0 = (float) blk->d * (float) blk->qs[ix];

            device const src1_t * src1_row = (device const src1_t *)(src1_base + i01*args.nb11);
            const float v1 = (float) src1_row[0];

            acc += v0 * v1;
        }

        *((device float *)(dst_base + i0*args.nb0)) = acc;
    }
}

typedef decltype(kernel_out_prod_q8_0_impl<float>) kernel_out_prod_q8_0_f32_t;
typedef decltype(kernel_out_prod_q8_0_impl<half >) kernel_out_prod_q8_0_f16_t;

template [[host_name("kernel_out_prod_q8_0_f32")]] kernel kernel_out_prod_q8_0_f32_t kernel_out_prod_q8_0_impl<float>;
template [[host_name("kernel_out_prod_q8_0_f16")]] kernel kernel_out_prod_q8_0_f16_t kernel_out_prod_q8_0_impl<half>;

template <typename src1_t>
kernel void kernel_out_prod_q4_0_impl(
    constant ggml_metal_kargs_out_prod & args,
    device const char * src0,
    device const char * src1,
    device       char * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    const int dps2 = args.ne02 > 0 ? args.ne2 / args.ne02 : 1;
    const int dps3 = args.ne03 > 0 ? args.ne3 / args.ne03 : 1;

    const int i02 = args.ne02 > 0 ? i2 / dps2 : 0;
    const int i03 = args.ne03 > 0 ? i3 / dps3 : 0;

    device const char * src0_base = src0 + i02*args.nb02 + i03*args.nb03;
    device const char * src1_base = src1 + i1*args.nb10 + i2*args.nb12 + i3*args.nb13;
    device       char * dst_base  = dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int ib = i0 / QK4_0;
        const int ix = i0 % QK4_0;
        const int iq = ix % (QK4_0 / 2);
        const bool upper = ix >= (QK4_0 / 2);

        float acc = 0.0f;

        for (int i01 = 0; i01 < args.ne01; ++i01) {
            device const char * src0_row_char = src0_base + i01*args.nb01;
            device const block_q4_0 * src0_row = (device const block_q4_0 *) src0_row_char;
            // field access through the device pointer instead of a whole-block
            // copy - see the q8_0 kernel above for the rationale.
            device const block_q4_0 * blk = src0_row + ib;

            const uint8_t q = blk->qs[iq];
            const int nibble = upper ? (q >> 4) : (q & 0x0F);
            const float v0 = ((float) blk->d) * ((float) nibble - 8.0f);

            device const src1_t * src1_row = (device const src1_t *)(src1_base + i01*args.nb11);
            const float v1 = (float) src1_row[0];

            acc += v0 * v1;
        }

        *((device float *)(dst_base + i0*args.nb0)) = acc;
    }
}

typedef decltype(kernel_out_prod_q4_0_impl<float>) kernel_out_prod_q4_0_f32_t;
typedef decltype(kernel_out_prod_q4_0_impl<half >) kernel_out_prod_q4_0_f16_t;

template [[host_name("kernel_out_prod_q4_0_f32")]] kernel kernel_out_prod_q4_0_f32_t kernel_out_prod_q4_0_impl<float>;
template [[host_name("kernel_out_prod_q4_0_f16")]] kernel kernel_out_prod_q4_0_f16_t kernel_out_prod_q4_0_impl<half>;
