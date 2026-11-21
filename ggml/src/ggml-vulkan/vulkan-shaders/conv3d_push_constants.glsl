#if !defined(GGML_CONV3D_PUSH_CONSTANTS)
#define GGML_CONV3D_PUSH_CONSTANTS

// Push constants of conv3d_mm.comp and conv_halo.comp for CONV_3D, matching
// vk_op_conv3d_push_constants. Filter [KW, KH, KD, IC * OC], input
// [IW, IH, ID, IC * N], output [OW, OH, OD, OC * N].
layout(push_constant) uniform parameter {
    // I/O channels, batch size
    uint OC;
    uint IC;
    uint N;

    // Tensor spatial sizes: input, output
    uint IW;
    uint IH;
    uint ID;
    uint OW;
    uint OH;
    uint OD;

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
    uint OWmp;     uint OWL;
    uint OWOHmp;   uint OWOHL;
    uint OWOHODmp; uint OWOHODL;

    // element offsets for misaligned buffer bindings
    uint knl_offset;
    uint src_offset;
    uint dst_offset;
} p;

#endif
