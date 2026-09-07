#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// add_id(gate) + add_id(up) + swiglu_oai, fused
//
// gpt-oss-class MoE FFNs run three full passes over the same
// [n_ff, n_expert_used, n_tokens] f32 tensor: a per-expert bias add on the gate
// matmul output, the same on the up matmul output, then swiglu_oai over the
// two. Both bias adds are in-place, so each costs a full read plus a full write
// of a tensor that is only read once more. Folding them into the swiglu pass
// leaves two reads and one write instead of six passes.
//
// Grouping matches kernel_add_id: group 0 = expert slot (i1), group 1 = token
// (i2). For a contiguous destination that addressing is identical to the flat
// row walk kernel_swiglu_oai uses, since row i1 + i2*ne1 sits at
// i1*nb1 + i2*ne1*nb1.
//------------------------------------------------------------------------------
kernel void kernel_add_id_add_id_swiglu_oai(
    global char * src_g,
    ulong         offset_g,
    global char * src_gb,
    ulong         offset_gb,
    global char * src_u,
    ulong         offset_u,
    global char * src_ub,
    ulong         offset_ub,
    global char * src_ids,
    ulong         offset_ids,
    global char * dst,
    ulong         offsetd,
    ulong         nb01_g,
    ulong         nb02_g,
    ulong         nb01_u,
    ulong         nb02_u,
    ulong         nb11_g,
    ulong         nb11_u,
    ulong         nb21,
    ulong         nbd1,
    ulong         nbd2,
    int           ne0,
    float         limit,
    float         alpha
) {
    src_g   = (global char *)(src_g   + offset_g);
    src_gb  = (global char *)(src_gb  + offset_gb);
    src_u   = (global char *)(src_u   + offset_u);
    src_ub  = (global char *)(src_ub  + offset_ub);
    src_ids = (global char *)(src_ids + offset_ids);
    dst     = (global char *)(dst     + offsetd);

    const int i1 = get_group_id(0);
    const int i2 = get_group_id(1);

    // The ids tensor is a view into a [n_expert, n_tokens] buffer, so its row
    // stride is nb21 and the k selected ids are NOT contiguous per token.
    const int i11 = *((global const int *) (src_ids + i1*sizeof(int) + i2*nb21));

    global const float * g_row  = (global const float *)(src_g  + i1*nb01_g + i2*nb02_g);
    global const float * u_row  = (global const float *)(src_u  + i1*nb01_u + i2*nb02_u);
    global const float * gb_row = (global const float *)(src_gb + i11*nb11_g);
    global const float * ub_row = (global const float *)(src_ub + i11*nb11_u);
    global       float * d_row  = (global       float *)(dst    + i1*nbd1   + i2*nbd2);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        float x0 = g_row[i0] + gb_row[i0];
        float x1 = u_row[i0] + ub_row[i0];

        x0 = min(x0, limit);
        x1 = max(min(x1, limit), -limit);

        float out_glu = x0 / (1.0f + exp(-x0 * alpha));
        out_glu = out_glu * (1.0f + x1);

        d_row[i0] = out_glu;
    }
}
