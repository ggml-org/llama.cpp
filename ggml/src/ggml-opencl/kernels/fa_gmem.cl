////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// online softmax, across blocks of a grid, four rows at a time
// inspired by: https://medium.com/data-science-collective/online-softmax-to-flash-attention-and-why-it-matters-9d676e7c50a8
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma OPENCL EXTENSION cl_khr_fp16 : enable


/////////////////////////////////////////////////////////////////////////////////////
// Fused scale + max pass. Reads raw X0 and updates
// SM.m_i / m_i_minus_1 / d_i_minus_1 in a single kernel -- replaces the old
// scaled-score intermediate. Sum and output recompute the scaled raw scores, and
// output writes only packed-half probabilities for MM1.
//
// Thread mapping matches online_softmax_sum below (and the old online_softmax_max_sum
// this replaces): local_id(0) (size LOCAL_M) walks adjacent m_4 row-groups, local_id(1)
// (size LOCAL_N) partitions the N-reduction, combined via __local memory + a
// local-only barrier rather than sub_group_reduce (LOCAL_M and LOCAL_N can share one
// wave, and a subgroup reduction would incorrectly mix lanes from different m_4
// row-groups). LOCAL_M*LOCAL_N need not stay a full wavefront now that this kernel also
// carries scaling work -- see FA_GMEM.h's fusedSoftmaxLocalM/N for the
// current sweep result.
//
__kernel void online_softmax_max (
                            __global const float * matrixX0_in,
                            uint matrixX0_in_stride,
                            float scale,
                            uint blockM,
                            uint blockN,
                            __global float * shared_per_row_data,
                            uint per_row_data_stride_elements,
                            uint init_per_row_data
)
{
    int local_m = get_local_id(0);
    int local_n = get_local_id(1);
    int local_size_n = get_local_size(1);
    int m_4 = get_global_id(0);
    uint num_row_groups = (blockM + 3) / 4;
    bool valid = (uint)m_4 < num_row_groups;
    uint remainderM = blockM % 4;
    bool full_m4 = (m_4*4 + 4) <= (int)blockM;

    // upper-bounded by the LOCAL_M*LOCAL_N dispatch invariant (see FA_GMEM.cpp)
    __local float4 local_max[64];
    int local_idx = local_m * local_size_n + local_n;

    struct fa_per_row_data * prd = (struct fa_per_row_data *)(shared_per_row_data + m_4*per_row_data_stride_elements);

    if (valid && local_n == 0) {
        if (init_per_row_data) {
            // once per row
            prd->m_i_minus_1 = (float4)(-MAXFLOAT);
            prd->d_i_minus_1 = (float4)(0);
        } else {
            // once per block (GWS)
            prd->m_i_minus_1 = prd->m_i;
            prd->d_i_minus_1 = prd->d_i;
        }
    }
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);

    // per-work_item max across this lane's LOCAL_N-wide partition of the row
    float4 max_per_thread4 = (float4)(-MAXFLOAT);
    if (valid) {
        for (int n = local_n; n < blockN; n += local_size_n) {
            // X0 is M-first (M contiguous).
            const float * mX0_in = matrixX0_in + (n*matrixX0_in_stride + m_4*4);

            float4 values4 = (float4)(-MAXFLOAT);
            if (full_m4) {
                values4 = vload4(0, mX0_in);
            } else {
                switch (remainderM) {
                    case 1: values4.s0 = *mX0_in; break;
                    case 2: values4.s01 = vload2(0, mX0_in); break;
                    case 3: values4.s01 = vload2(0, mX0_in); values4.s2 = *(mX0_in + 2); break;
                }
            }

            float4 outputs4 = values4 * scale;

            // lanes beyond remainderM were never computed/written above; pin them to
            // -MAXFLOAT explicitly (rather than relying on values4's scaled sentinel) so the
            // max reduction below can't be corrupted regardless of scale's sign/magnitude
            if (!full_m4) {
                switch (remainderM) {
                    case 1: outputs4.s1 = -MAXFLOAT; outputs4.s2 = -MAXFLOAT; outputs4.s3 = -MAXFLOAT; break;
                    case 2: outputs4.s2 = -MAXFLOAT; outputs4.s3 = -MAXFLOAT; break;
                    case 3: outputs4.s3 = -MAXFLOAT; break;
                }
            }

            max_per_thread4 = select(outputs4, max_per_thread4, isless(outputs4, max_per_thread4));
        }
    }

    // combine this row-group's LOCAL_N partial maxes via local memory
    local_max[local_idx] = max_per_thread4;
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    if (valid && local_n == 0) {
        float4 max_per_workgroup4 = local_max[local_idx];
        for (int i = 1; i < local_size_n; i++) {
            float4 other = local_max[local_m * local_size_n + i];
            max_per_workgroup4 = select(other, max_per_workgroup4, isless(other, max_per_workgroup4));
        }
        prd->m_i = max(prd->m_i_minus_1, max_per_workgroup4);
    }
}

/////////////////////////////////////////////////////////////////////////////////////
// Sum-only pass: the second half of the old online_softmax_max_sum, unchanged except
// that the max pass and its m_i_minus_1/d_i_minus_1 rotation moved to
// online_softmax_max above (this kernel now only consumes prd->m_i, it no
// longer produces it). Kept as its own kernel rather than folded further because
// mathematically combining the max and sum loops requires an extra exp() per element
// and empirically loses -- same reasoning that kept them separate before the fusion.
__kernel void online_softmax_sum (
                            __global const float * matrixX0_in,
                            uint matrixX0_in_stride,
                            float scale,
                            uint blockM,
                            uint blockN,
                            __global float * shared_per_row_data,
                            uint per_row_data_stride_elements
)
{
    int local_m = get_local_id(0);
    int local_n = get_local_id(1);
    int local_size_n = get_local_size(1);
    int m_4 = get_global_id(0);
    uint num_row_groups = (blockM + 3) / 4;
    bool valid = (uint)m_4 < num_row_groups;

    // upper-bounded by the LOCAL_M*LOCAL_N dispatch invariant (see FA_GMEM.cpp)
    __local float4 local_sum[64];
    int local_idx = local_m * local_size_n + local_n;

    struct fa_per_row_data * prd = (struct fa_per_row_data *)(shared_per_row_data + m_4*per_row_data_stride_elements);

    // per-work_item sum across this lane's LOCAL_N-wide partition of the row
    float4 sum_per_thread4 = (float4)(0);
    if (valid) {
        float4 m_i = prd->m_i;
        for (int n = local_n; n < blockN; n += local_size_n) {
            const float * mX0_in = matrixX0_in + (n*matrixX0_in_stride + m_4*4);

            float4 values4 = (float4)(0.0f);
            if ((m_4*4 + 4) <= blockM) {
                values4 = vload4(0, mX0_in);
            } else {
                uint remainderM = blockM % 4;
                switch (remainderM) {
                    case 1:
                        values4.s0 = *mX0_in;
                        break;
                    case 2:
                        values4.s01 = vload2(0, mX0_in);
                        break;
                    case 3:
                        values4.s01 = vload2(0, mX0_in);
                        values4.s2 = *(mX0_in + 2);
                        break;
                }
            }

            values4 *= scale;

            sum_per_thread4 += native_exp(values4 - m_i);
        }
    }

    // combine this row-group's LOCAL_N partial sums via local memory
    local_sum[local_idx] = sum_per_thread4;
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    if (valid && local_n == 0) {
        float4 sum_per_workgroup4 = (float4)(0);
        for (int i = 0; i < local_size_n; i++) {
            sum_per_workgroup4 += local_sum[local_m * local_size_n + i];
        }
        prd->d_i = prd->d_i_minus_1 * native_exp(prd->m_i_minus_1 - prd->m_i);
        prd->d_i += sum_per_workgroup4;
    }
}

/////////////////////////////////////////////////////////////////////////////////////
__kernel void flash_attn_f32_f16_gmem (
                            __global const float * matrixX0_in,
                            uint matrixX0_in_stride,
                            float scale,
                            // MM1's A operand is always FP16ACC (unconditional, see FA_GMEM.cpp's
                            // "-DFP16ACC" build option comment) -- narrow to packed half here, once,
                            // instead of leaving it for mm_load_a (which since ada937a assumes its A
                            // input is already narrowed by a copy_block_a_padded-style step, a
                            // precondition this kernel must now satisfy directly since it has no such
                            // separate copy step).
                            __global half * matrixX0_out,
                            uint matrixX0_out_stride,
                            uint blockM,
                            uint blockN,
                            __global const float * shared_per_row_data,
                            uint per_row_data_stride_elements
)
{
    int m = get_global_id(0);
    int n = get_global_id(1);

    const float * mX0_in0 =  matrixX0_in + ((4 * n) + 0) * matrixX0_in_stride + 4*m;
    const float * mX0_in1 =  matrixX0_in + ((4 * n) + 1) * matrixX0_in_stride + 4*m;
    const float * mX0_in2 =  matrixX0_in + ((4 * n) + 2) * matrixX0_in_stride + 4*m;
    const float * mX0_in3 =  matrixX0_in + ((4 * n) + 3) * matrixX0_in_stride + 4*m;

    struct fa_per_row_data * prd = (struct fa_per_row_data *)(shared_per_row_data + m*per_row_data_stride_elements);

    half * mX0_out0 = matrixX0_out + ((4 * m) + 0) * matrixX0_out_stride + 4*n;
    half * mX0_out1 = matrixX0_out + ((4 * m) + 1) * matrixX0_out_stride + 4*n;
    half * mX0_out2 = matrixX0_out + ((4 * m) + 2) * matrixX0_out_stride + 4*n;
    half * mX0_out3 = matrixX0_out + ((4 * m) + 3) * matrixX0_out_stride + 4*n;

    if (n*4 >= (int)blockN) {
        // all-padding group; MM1's K-loop reads in fixed TILESIZE_K-wide chunks, so it can
        // read up to the next multiple of 8 past blockN (the actual N0K1 for this block) --
        // zero-fill so that overread stays finite (matches copy_xpose_block_b_padded's V padding).
        vstore4((half4)(0), 0, mX0_out0);
        vstore4((half4)(0), 0, mX0_out1);
        vstore4((half4)(0), 0, mX0_out2);
        vstore4((half4)(0), 0, mX0_out3);
        return;
    }

    float4 inv_d_i4 = native_recip(prd->d_i);

    if ((m*4 + 4) <= blockM) {
        float4 values0 = vload4(0, mX0_in0);
        float4 values1 = vload4(0, mX0_in1);
        float4 values2 = vload4(0, mX0_in2);
        float4 values3 = vload4(0, mX0_in3);

        values0 *= scale;
        values1 *= scale;
        values2 *= scale;
        values3 *= scale;

        float4 exp_diffs0 = native_exp(values0 - prd->m_i);
        float4 exp_diffs1 = native_exp(values1 - prd->m_i);
        float4 exp_diffs2 = native_exp(values2 - prd->m_i);
        float4 exp_diffs3 = native_exp(values3 - prd->m_i);

        float4 outputs0 = exp_diffs0 * inv_d_i4;
        float4 outputs1 = exp_diffs1 * inv_d_i4;
        float4 outputs2 = exp_diffs2 * inv_d_i4;
        float4 outputs3 = exp_diffs3 * inv_d_i4;

        if (4*n + 1 >= blockN) outputs1 = (float4)(0.0f);
        if (4*n + 2 >= blockN) outputs2 = (float4)(0.0f);
        if (4*n + 3 >= blockN) outputs3 = (float4)(0.0f);

        // xpose on write so A is K-first for MM1
        vstore4(convert_half4((float4)(outputs0.s0, outputs1.s0, outputs2.s0, outputs3.s0)), 0, mX0_out0);
        vstore4(convert_half4((float4)(outputs0.s1, outputs1.s1, outputs2.s1, outputs3.s1)), 0, mX0_out1);
        vstore4(convert_half4((float4)(outputs0.s2, outputs1.s2, outputs2.s2, outputs3.s2)), 0, mX0_out2);
        vstore4(convert_half4((float4)(outputs0.s3, outputs1.s3, outputs2.s3, outputs3.s3)), 0, mX0_out3);
    } else {
        uint remainderM = blockM % 4;
        switch (remainderM) {
            case 1:
                {
                    float value0 = *mX0_in0;
                    float value1 = *mX0_in1;
                    float value2 = *mX0_in2;
                    float value3 = *mX0_in3;

                    value0 *= scale;
                    value1 *= scale;
                    value2 *= scale;
                    value3 *= scale;

                    float exp_diff0 = native_exp(value0 - prd->m_i.s0);
                    float exp_diff1 = native_exp(value1 - prd->m_i.s0);
                    float exp_diff2 = native_exp(value2 - prd->m_i.s0);
                    float exp_diff3 = native_exp(value3 - prd->m_i.s0);
                    float output0 = exp_diff0 * inv_d_i4.s0;
                    float output1 = exp_diff1 * inv_d_i4.s0;
                    float output2 = exp_diff2 * inv_d_i4.s0;
                    float output3 = exp_diff3 * inv_d_i4.s0;

                    if (4*n + 1 >= blockN) output1 = 0.0f;
                    if (4*n + 2 >= blockN) output2 = 0.0f;
                    if (4*n + 3 >= blockN) output3 = 0.0f;

                    vstore4(convert_half4((float4)(output0, output1, output2, output3)), 0, mX0_out0);
                }
                break;
            case 2:
                {
                    float2 values0 = vload2(0, mX0_in0);
                    float2 values1 = vload2(0, mX0_in1);
                    float2 values2 = vload2(0, mX0_in2);
                    float2 values3 = vload2(0, mX0_in3);

                    values0 *= scale;
                    values1 *= scale;
                    values2 *= scale;
                    values3 *= scale;

                    float2 exp_diffs0 = native_exp(values0 - prd->m_i.s01);
                    float2 exp_diffs1 = native_exp(values1 - prd->m_i.s01);
                    float2 exp_diffs2 = native_exp(values2 - prd->m_i.s01);
                    float2 exp_diffs3 = native_exp(values3 - prd->m_i.s01);
                    float2 outputs0 = exp_diffs0 * inv_d_i4.s01;
                    float2 outputs1 = exp_diffs1 * inv_d_i4.s01;
                    float2 outputs2 = exp_diffs2 * inv_d_i4.s01;
                    float2 outputs3 = exp_diffs3 * inv_d_i4.s01;

                    if (4*n + 1 >= blockN) outputs1 = (float2)(0.0f);
                    if (4*n + 2 >= blockN) outputs2 = (float2)(0.0f);
                    if (4*n + 3 >= blockN) outputs3 = (float2)(0.0f);

                    vstore4(convert_half4((float4)(outputs0.s0, outputs1.s0, outputs2.s0, outputs3.s0)), 0, mX0_out0);
                    vstore4(convert_half4((float4)(outputs0.s1, outputs1.s1, outputs2.s1, outputs3.s1)), 0, mX0_out1);
                }
                break;
            case 3:
                {
                    float2 values0 = vload2(0, mX0_in0);
                    float2 values1 = vload2(0, mX0_in1);
                    float2 values2 = vload2(0, mX0_in2);
                    float2 values3 = vload2(0, mX0_in3);
                    float value0 = *(mX0_in0 + 2);
                    float value1 = *(mX0_in1 + 2);
                    float value2 = *(mX0_in2 + 2);
                    float value3 = *(mX0_in3 + 2);

                    values0 *= scale;
                    values1 *= scale;
                    values2 *= scale;
                    values3 *= scale;
                    value0 *= scale;
                    value1 *= scale;
                    value2 *= scale;
                    value3 *= scale;

                    float2 exp_diffs0 = native_exp(values0 - prd->m_i.s01);
                    float2 exp_diffs1 = native_exp(values1 - prd->m_i.s01);
                    float2 exp_diffs2 = native_exp(values2 - prd->m_i.s01);
                    float2 exp_diffs3 = native_exp(values3 - prd->m_i.s01);
                    float2 outputs0 = exp_diffs0 * inv_d_i4.s01;
                    float2 outputs1 = exp_diffs1 * inv_d_i4.s01;
                    float2 outputs2 = exp_diffs2 * inv_d_i4.s01;
                    float2 outputs3 = exp_diffs3 * inv_d_i4.s01;
                    float exp_diff0 = native_exp(value0 - prd->m_i.s2);
                    float exp_diff1 = native_exp(value1 - prd->m_i.s2);
                    float exp_diff2 = native_exp(value2 - prd->m_i.s2);
                    float exp_diff3 = native_exp(value3 - prd->m_i.s2);
                    float output0 = exp_diff0 * inv_d_i4.s2;
                    float output1 = exp_diff1 * inv_d_i4.s2;
                    float output2 = exp_diff2 * inv_d_i4.s2;
                    float output3 = exp_diff3 * inv_d_i4.s2;

                    if (4*n + 1 >= blockN) { outputs1 = (float2)(0.0f); output1 = 0.0f; }
                    if (4*n + 2 >= blockN) { outputs2 = (float2)(0.0f); output2 = 0.0f; }
                    if (4*n + 3 >= blockN) { outputs3 = (float2)(0.0f); output3 = 0.0f; }

                    vstore4(convert_half4((float4)(outputs0.s0, outputs1.s0, outputs2.s0, outputs3.s0)), 0, mX0_out0);
                    vstore4(convert_half4((float4)(outputs0.s1, outputs1.s1, outputs2.s1, outputs3.s1)), 0, mX0_out1);
                    vstore4(convert_half4((float4)(output0, output1, output2, output3)), 0, mX0_out2);
                }
                break;
        }
    }
}

static float4 fa_load_query_rows(__global const float * src, uint remaining) {
    if (remaining >= 4) { return vload4(0, src); }
    float4 value = (float4)(0.0f);
    value.s0 = src[0];
    if (remaining >= 2) { value.s1 = src[1]; }
    if (remaining >= 3) { value.s2 = src[2]; }
    return value;
}

static void fa_store_query_rows(float4 value, __global float * dst, uint remaining) {
    if (remaining >= 4) { vstore4(value, 0, dst); return; }
    dst[0] = value.s0;
    if (remaining >= 2) { dst[1] = value.s1; }
    if (remaining >= 3) { dst[2] = value.s2; }
}

// Normalize the previous output before an ordinary MM adds the next block.
__kernel void online_softmax_rescale_output(
        __global float * output, uint output_stride,
        __global const float * shared_per_row_data, uint per_row_data_stride_elements,
        uint blockM, uint head_dim) {
    uint group = get_global_id(0);
    uint column = get_global_id(1);
    uint row = 4 * group;
    if (row >= blockM || column >= head_dim) { return; }
    __global const struct fa_per_row_data * prd = (__global const struct fa_per_row_data *)(shared_per_row_data + group * per_row_data_stride_elements);
    float4 previous_sum = prd->d_i_minus_1;
    float4 max_scale = native_exp(prd->m_i_minus_1 - prd->m_i);
    float4 inverse_sum = native_recip(prd->d_i);
    __global float * dst = output + column * output_stride + row;
    float4 value = fa_load_query_rows(dst, blockM - row) * previous_sum * max_scale * inverse_sum;
    fa_store_query_rows(value, dst, blockM - row);
}

#ifdef FP16KV
#define FA_V4 half4
#else
#define FA_V4 float4
#endif

// V is N-first in DDR and K-first, packed half, in GMEM.
__kernel void copy_xpose_block_b_padded(
        __global const FA_V4 * input, uint input_stride,
        __global half * output, uint output_stride,
        uint head_dim, uint actual_kv) {
    uint kv = 4 * get_global_id(0);
    uint column = 4 * get_global_id(1);
    if (column >= head_dim) { return; }
    float4 row0 = kv + 0 < actual_kv ? convert_float4(input[(kv + 0) * (input_stride / 4) + column / 4]) : (float4)(0.0f);
    float4 row1 = kv + 1 < actual_kv ? convert_float4(input[(kv + 1) * (input_stride / 4) + column / 4]) : (float4)(0.0f);
    float4 row2 = kv + 2 < actual_kv ? convert_float4(input[(kv + 2) * (input_stride / 4) + column / 4]) : (float4)(0.0f);
    float4 row3 = kv + 3 < actual_kv ? convert_float4(input[(kv + 3) * (input_stride / 4) + column / 4]) : (float4)(0.0f);
    vstore4(convert_half4((float4)(row0.s0, row1.s0, row2.s0, row3.s0)), 0, output + (column + 0) * output_stride + kv);
    vstore4(convert_half4((float4)(row0.s1, row1.s1, row2.s1, row3.s1)), 0, output + (column + 1) * output_stride + kv);
    vstore4(convert_half4((float4)(row0.s2, row1.s2, row2.s2, row3.s2)), 0, output + (column + 2) * output_stride + kv);
    vstore4(convert_half4((float4)(row0.s3, row1.s3, row2.s3, row3.s3)), 0, output + (column + 3) * output_stride + kv);
}

// Scatter M-first C into llama.cpp's [head_dim, heads, queries] output.
__kernel void copy_xpose_permute_block_c(
        __global const float * input, uint input_stride,
        __global float * output, uint output_stride,
        uint blockM, uint m_block, uint actual_m, uint head, uint heads) {
    uint row = 4 * get_global_id(0);
    uint column = 4 * get_global_id(1);
    if (row >= actual_m) { return; }
    float4 col0 = fa_load_query_rows(input + (column + 0) * input_stride + row, actual_m - row);
    float4 col1 = fa_load_query_rows(input + (column + 1) * input_stride + row, actual_m - row);
    float4 col2 = fa_load_query_rows(input + (column + 2) * input_stride + row, actual_m - row);
    float4 col3 = fa_load_query_rows(input + (column + 3) * input_stride + row, actual_m - row);
    __global float * dst = output + (blockM * m_block + row) * heads * output_stride + head * output_stride + column;
    vstore4((float4)(col0.s0, col1.s0, col2.s0, col3.s0), 0, dst);
    if (row + 1 < actual_m) { vstore4((float4)(col0.s1, col1.s1, col2.s1, col3.s1), 0, dst + heads * output_stride); }
    if (row + 2 < actual_m) { vstore4((float4)(col0.s2, col1.s2, col2.s2, col3.s2), 0, dst + 2 * heads * output_stride); }
    if (row + 3 < actual_m) { vstore4((float4)(col0.s3, col1.s3, col2.s3, col3.s3), 0, dst + 3 * heads * output_stride); }
}

#undef FA_V4

#ifdef FP16A
#define FA_Q8 half8
#else
#define FA_Q8 float8
#endif

#ifdef FP16KV
#define FA_K8 half8
#else
#define FA_K8 float8
#endif

// Q and K retain their input types for MM0.
__kernel void copy_block_a_padded(
        __global const FA_Q8 * matrixA, __global FA_Q8 * blockA_padded,
        uint TOTAL_K, uint LINE_STRIDE_MATRIX_A_IN_ELEMENTS) {
    int k = get_global_id(0);
    int m = get_global_id(1);
    blockA_padded[m*LINE_STRIDE_MATRIX_A_IN_ELEMENTS/8 + k] = matrixA[m*TOTAL_K/8 + k];
}

__kernel void copy_block_b_padded(
        __global const FA_K8 * matrixB_in, uint matrixB_in_stride,
        __global FA_K8 * matrixB_out, uint matrixB_out_stride, uint actual_rows) {
    int k = get_global_id(0);
    int n = get_global_id(1);
    if (n >= (int)actual_rows) {
        matrixB_out[n*matrixB_out_stride/8 + k] = (FA_K8)(0);
    } else {
        matrixB_out[n*matrixB_out_stride/8 + k] = matrixB_in[n*matrixB_in_stride/8 + k];
    }
}

#undef FA_Q8
#undef FA_K8
