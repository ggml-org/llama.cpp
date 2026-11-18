#if !defined(GGML_CONV2D_PUSH_CONSTANTS)
#define GGML_CONV2D_PUSH_CONSTANTS

// Push constants of conv2d_mm.comp and conv_halo.comp for CONV_2D and
// CONV_TRANSPOSE_2D, matching vk_op_conv2d_push_constants. Filter
// [KW, KH, IC, OC] ([KW, KH, OC, IC] for CONV_TRANSPOSE_2D), input
// [IW, IH, IC, N], output [OW, OH, OC, N].
layout(push_constant) uniform parameter {
    // I/O channels, batch size
    uint OC;
    uint IC;
    uint N;

    // Tensor spatial sizes: input, output
    uint IW;
    uint IH;
    uint OW;
    uint OH;

    // Strides in elements
    uint nb01;
    uint nb02;
    uint nb03;

    uint nb11;
    uint nb12;
    uint nb13;

    uint nb1;
    uint nb2;
    uint nb3;

    // fastdiv helper values
    uint OWmp;   uint OWL;
    uint OWOHmp; uint OWOHL;

    // element offsets for misaligned buffer bindings
    uint knl_offset;
    uint src_offset;
    uint dst_offset;
} p;

#endif
