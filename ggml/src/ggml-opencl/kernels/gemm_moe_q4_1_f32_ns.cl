#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_uniform_load: enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_constant_load: enable
#pragma OPENCL EXTENSION cl_qcom_extra_vector_types : enable

#define TILESIZE_K 16
#define TILESIZE_M 64
#define TILESIZE_N 32


#define dequantize_q4_1(q4, a_f16, scale, m) \
    a_f16.s0 = (half)(q4.s0 & 0x000F) * scale + m; \
    a_f16.s1 = (half)((q4.s0 & 0x00F0) >> 4) * scale + m; \
    a_f16.s2 = (half)((q4.s0 & 0x0F00) >> 8) * scale + m; \
    a_f16.s3 = (half)((q4.s0 & 0xF000) >> 12) * scale + m; \
    a_f16.s4 = (half)(q4.s1 & 0x000F) * scale + m; \
    a_f16.s5 = (half)((q4.s1 & 0x00F0) >> 4) * scale + m; \
    a_f16.s6 = (half)((q4.s1 & 0x0F00) >> 8) * scale + m; \
    a_f16.s7 = (half)((q4.s1 & 0xF000) >> 12) * scale + m; \
    a_f16.s8 = (half)(q4.s2 & 0x000F) * scale + m; \
    a_f16.s9 = (half)((q4.s2 & 0x00F0) >> 4) * scale + m; \
    a_f16.sa = (half)((q4.s2 & 0x0F00) >> 8) * scale + m; \
    a_f16.sb = (half)((q4.s2 & 0xF000) >> 12) * scale + m; \
    a_f16.sc = (half)(q4.s3 & 0x000F) * scale + m; \
    a_f16.sd = (half)((q4.s3 & 0x00F0) >> 4) * scale + m; \
    a_f16.se = (half)((q4.s3 & 0x0F00) >> 8) * scale + m; \
    a_f16.sf = (half)((q4.s3 & 0xF000) >> 12) * scale + m; \


#define dotx16_reduce8(a_reg, b_lm, c_reg, lm_offset) \
    acc.s0 = dot(a_reg.s0123, b_lm[lm_offset + 0]); \
    acc.s1 = dot(a_reg.s0123, b_lm[lm_offset + 1]); \
    acc.s2 = dot(a_reg.s0123, b_lm[lm_offset + 2]); \
    acc.s3 = dot(a_reg.s0123, b_lm[lm_offset + 3]); \
    acc.s4 = dot(a_reg.s0123, b_lm[lm_offset + 4]); \
    acc.s5 = dot(a_reg.s0123, b_lm[lm_offset + 5]); \
    acc.s6 = dot(a_reg.s0123, b_lm[lm_offset + 6]); \
    acc.s7 = dot(a_reg.s0123, b_lm[lm_offset + 7]); \
    acc.s8 = dot(a_reg.s0123, b_lm[lm_offset + 8]); \
    acc.s9 = dot(a_reg.s0123, b_lm[lm_offset + 9]); \
    acc.sa = dot(a_reg.s0123, b_lm[lm_offset + 10]); \
    acc.sb = dot(a_reg.s0123, b_lm[lm_offset + 11]); \
    acc.sc = dot(a_reg.s0123, b_lm[lm_offset + 12]); \
    acc.sd = dot(a_reg.s0123, b_lm[lm_offset + 13]); \
    acc.se = dot(a_reg.s0123, b_lm[lm_offset + 14]); \
    acc.sf = dot(a_reg.s0123, b_lm[lm_offset + 15]); \
    acc.s0 += dot(a_reg.s4567, b_lm[lm_offset + 32]); \
    acc.s1 += dot(a_reg.s4567, b_lm[lm_offset + 33]); \
    acc.s2 += dot(a_reg.s4567, b_lm[lm_offset + 34]); \
    acc.s3 += dot(a_reg.s4567, b_lm[lm_offset + 35]); \
    acc.s4 += dot(a_reg.s4567, b_lm[lm_offset + 36]); \
    acc.s5 += dot(a_reg.s4567, b_lm[lm_offset + 37]); \
    acc.s6 += dot(a_reg.s4567, b_lm[lm_offset + 38]); \
    acc.s7 += dot(a_reg.s4567, b_lm[lm_offset + 39]); \
    acc.s8 += dot(a_reg.s4567, b_lm[lm_offset + 40]); \
    acc.s9 += dot(a_reg.s4567, b_lm[lm_offset + 41]); \
    acc.sa += dot(a_reg.s4567, b_lm[lm_offset + 42]); \
    acc.sb += dot(a_reg.s4567, b_lm[lm_offset + 43]); \
    acc.sc += dot(a_reg.s4567, b_lm[lm_offset + 44]); \
    acc.sd += dot(a_reg.s4567, b_lm[lm_offset + 45]); \
    acc.se += dot(a_reg.s4567, b_lm[lm_offset + 46]); \
    acc.sf += dot(a_reg.s4567, b_lm[lm_offset + 47]); \
    c_reg.lo += convert_float8(acc.lo); \
    c_reg.hi += convert_float8(acc.hi); \
    acc.s0 = dot(a_reg.s89ab, b_lm[lm_offset + 64]); \
    acc.s1 = dot(a_reg.s89ab, b_lm[lm_offset + 65]); \
    acc.s2 = dot(a_reg.s89ab, b_lm[lm_offset + 66]); \
    acc.s3 = dot(a_reg.s89ab, b_lm[lm_offset + 67]); \
    acc.s4 = dot(a_reg.s89ab, b_lm[lm_offset + 68]); \
    acc.s5 = dot(a_reg.s89ab, b_lm[lm_offset + 69]); \
    acc.s6 = dot(a_reg.s89ab, b_lm[lm_offset + 70]); \
    acc.s7 = dot(a_reg.s89ab, b_lm[lm_offset + 71]); \
    acc.s8 = dot(a_reg.s89ab, b_lm[lm_offset + 72]); \
    acc.s9 = dot(a_reg.s89ab, b_lm[lm_offset + 73]); \
    acc.sa = dot(a_reg.s89ab, b_lm[lm_offset + 74]); \
    acc.sb = dot(a_reg.s89ab, b_lm[lm_offset + 75]); \
    acc.sc = dot(a_reg.s89ab, b_lm[lm_offset + 76]); \
    acc.sd = dot(a_reg.s89ab, b_lm[lm_offset + 77]); \
    acc.se = dot(a_reg.s89ab, b_lm[lm_offset + 78]); \
    acc.sf = dot(a_reg.s89ab, b_lm[lm_offset + 79]); \
    acc.s0 += dot(a_reg.scdef, b_lm[lm_offset + 96]); \
    acc.s1 += dot(a_reg.scdef, b_lm[lm_offset + 97]); \
    acc.s2 += dot(a_reg.scdef, b_lm[lm_offset + 98]); \
    acc.s3 += dot(a_reg.scdef, b_lm[lm_offset + 99]); \
    acc.s4 += dot(a_reg.scdef, b_lm[lm_offset + 100]); \
    acc.s5 += dot(a_reg.scdef, b_lm[lm_offset + 101]); \
    acc.s6 += dot(a_reg.scdef, b_lm[lm_offset + 102]); \
    acc.s7 += dot(a_reg.scdef, b_lm[lm_offset + 103]); \
    acc.s8 += dot(a_reg.scdef, b_lm[lm_offset + 104]); \
    acc.s9 += dot(a_reg.scdef, b_lm[lm_offset + 105]); \
    acc.sa += dot(a_reg.scdef, b_lm[lm_offset + 106]); \
    acc.sb += dot(a_reg.scdef, b_lm[lm_offset + 107]); \
    acc.sc += dot(a_reg.scdef, b_lm[lm_offset + 108]); \
    acc.sd += dot(a_reg.scdef, b_lm[lm_offset + 109]); \
    acc.se += dot(a_reg.scdef, b_lm[lm_offset + 110]); \
    acc.sf += dot(a_reg.scdef, b_lm[lm_offset + 111]); \
    c_reg.lo += convert_float8(acc.lo); \
    c_reg.hi += convert_float8(acc.hi); \


__attribute__((qcom_wave_pair_mode(1))) // 1=force single 2=force pair
kernel void kernel_gemm_moe_q4_1_f32_ns(
        __read_only  image1d_buffer_t src0_q,
        __global     half *           src0_d,
        __global     half *           src0_m,
        __read_only  image1d_buffer_t src1,
        __global     uint *           src2,
        __global     ushort *         src2_emap,
        __write_only image1d_buffer_t dst,
        __global     int *            total_tiles,
        uint ne00,
        uint ne01
) {
    uint block_id_m = get_global_id(1); // m_tile
    uint block_id_n = get_global_id(2); // n_tile

    // Boundary check
    if (((get_global_id(0) + block_id_m * TILESIZE_M) >= ne01) || (block_id_n >= total_tiles[0])) {
        return;
    }

    __private half16 reg_a;
    __private float32 reg_c = (float32)(0);
    __local half4 shared_b[128];

    const ushort expert_id = src2_emap[block_id_n];

    const uint row = block_id_m * TILESIZE_M;
    const uint col = block_id_n * TILESIZE_N;

    uint sub_block_id_m = get_local_id(0);
    uint2 b_global_offset;
    b_global_offset.x = ((sub_block_id_m & 3) << 2) + (sub_block_id_m >> 2) * ne00;
    b_global_offset.y = b_global_offset.x + (16 * ne00);
    uint2 b_local_offset;
    b_local_offset.x = (sub_block_id_m & 3) * 32 + (sub_block_id_m >> 2);
    b_local_offset.y = b_local_offset.x + 16;

    // Loop along K axis, 32 elements (one block) for each iteration, divided into 2 sub-blocks
    for (uint step = 0; step < ne00; step += TILESIZE_K * 2) {
        // First sub-block
        uint q_sub_offset = row + ((ne01 * step) >> 3) + ((expert_id * ne00 * ne01) >> 3);
        uint s_sub_offset = row + ((ne01 * step) >> 5) + ((expert_id * ne00 * ne01) >> 5);
        uint b_sub_offset = col * ne00 + step;

        // Load scale and m for current Q4_1 block
        uint sm_offset = s_sub_offset + get_global_id(0);
        half s = src0_d[sm_offset];
        half m = src0_m[sm_offset];

        // Load 16 q (64-bits) in transposed layout
        uint2 q4x16;
        q4x16.x = read_imageui(src0_q, q_sub_offset + sub_block_id_m).x;
        q4x16.y = read_imageui(src0_q, q_sub_offset + sub_block_id_m + ne01).x;

        // Load 16x32 floats from matrix B, each fiber out of 64 in a sub-group loads 8 elements
        float8 bx8_f32;
        bx8_f32.lo = read_imagef(src1, (b_sub_offset + b_global_offset.x) / 4);
        bx8_f32.hi = read_imagef(src1, (b_sub_offset + b_global_offset.y) / 4);
        // Convert to half and store to LM to share within the subgroup
        half8 bx8_f16 = convert_half8(bx8_f32);
        shared_b[b_local_offset.x] = bx8_f16.lo;
        shared_b[b_local_offset.y] = bx8_f16.hi;

        // Dequantization
        dequantize_q4_1(as_ushort4(q4x16), reg_a, s, m);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        // 32 16x16 fp16 dot product with 8 elements reduction for better precision
        half16 acc;
        dotx16_reduce8(reg_a, shared_b, reg_c.lo, 0);
        dotx16_reduce8(reg_a, shared_b, reg_c.hi, 16);

        // Repeat for second sub-block
        uint half_step = step + TILESIZE_K;
        q_sub_offset = row + ((ne01 * half_step) >> 3) + ((expert_id * ne00 * ne01) >> 3);
        b_sub_offset = col * ne00 + half_step;

        // Load next 16 q (64-bits) in transposed layout
        q4x16.x = read_imageui(src0_q, q_sub_offset + sub_block_id_m).x;
        q4x16.y = read_imageui(src0_q, q_sub_offset + sub_block_id_m + ne01).x;

        // Load 16x32 floats from matrix B, each fiber out of 64 in a sub-group loads 8 elements
        bx8_f32.lo = read_imagef(src1, (b_sub_offset + b_global_offset.x) / 4);
        bx8_f32.hi = read_imagef(src1, (b_sub_offset + b_global_offset.y) / 4);
        // Convert to half and store to LM to share within the subgroup
        bx8_f16 = convert_half8(bx8_f32);
        shared_b[b_local_offset.x] = bx8_f16.lo;
        shared_b[b_local_offset.y] = bx8_f16.hi;

        // Dequantization
        dequantize_q4_1(as_ushort4(q4x16), reg_a, s, m);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        // 32 16x16 fp16 dot product with 3-levels reduction for better precision
        dotx16_reduce8(reg_a, shared_b, reg_c.lo, 0);
        dotx16_reduce8(reg_a, shared_b, reg_c.hi, 16);
    }

    // Load poster router and share in LM
    __local uint out_idx[TILESIZE_N];

    if (get_local_id(0) < TILESIZE_N) {
        uint idx = src2[block_id_n * TILESIZE_N + get_local_id(0)];
        if (idx == 0xFFFFFFFF) {
            idx = src2[block_id_n * TILESIZE_N + 0];
        }
        out_idx[get_local_id(0)] = idx * ne01;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Scatter results back to original position in output grid
    uint m_offset = row + get_local_id(0);

    write_imagef(dst, out_idx[1] + m_offset, (reg_c.s1));
    write_imagef(dst, out_idx[2] + m_offset, (reg_c.s2));
    write_imagef(dst, out_idx[3] + m_offset, (reg_c.s3));
    write_imagef(dst, out_idx[4] + m_offset, (reg_c.s4));
    write_imagef(dst, out_idx[5] + m_offset, (reg_c.s5));
    write_imagef(dst, out_idx[6] + m_offset, (reg_c.s6));
    write_imagef(dst, out_idx[7] + m_offset, (reg_c.s7));
    write_imagef(dst, out_idx[8] + m_offset, (reg_c.s8));
    write_imagef(dst, out_idx[9] + m_offset, (reg_c.s9));
    write_imagef(dst, out_idx[10] + m_offset, (reg_c.sa));
    write_imagef(dst, out_idx[11] + m_offset, (reg_c.sb));
    write_imagef(dst, out_idx[12] + m_offset, (reg_c.sc));
    write_imagef(dst, out_idx[13] + m_offset, (reg_c.sd));
    write_imagef(dst, out_idx[14] + m_offset, (reg_c.se));
    write_imagef(dst, out_idx[15] + m_offset, (reg_c.sf));
    write_imagef(dst, out_idx[16] + m_offset, (reg_c.sg));
    write_imagef(dst, out_idx[17] + m_offset, (reg_c.sh));
    write_imagef(dst, out_idx[18] + m_offset, (reg_c.si));
    write_imagef(dst, out_idx[19] + m_offset, (reg_c.sj));
    write_imagef(dst, out_idx[20] + m_offset, (reg_c.sk));
    write_imagef(dst, out_idx[21] + m_offset, (reg_c.sl));
    write_imagef(dst, out_idx[22] + m_offset, (reg_c.sm));
    write_imagef(dst, out_idx[23] + m_offset, (reg_c.sn));
    write_imagef(dst, out_idx[24] + m_offset, (reg_c.so));
    write_imagef(dst, out_idx[25] + m_offset, (reg_c.sp));
    write_imagef(dst, out_idx[26] + m_offset, (reg_c.sq));
    write_imagef(dst, out_idx[27] + m_offset, (reg_c.sr));
    write_imagef(dst, out_idx[28] + m_offset, (reg_c.ss));
    write_imagef(dst, out_idx[29] + m_offset, (reg_c.st));
    write_imagef(dst, out_idx[30] + m_offset, (reg_c.su));
    write_imagef(dst, out_idx[31] + m_offset, (reg_c.sv));

    // Store zero padding parts to the index of first output in tile, override correct result in the end
    barrier(CLK_GLOBAL_MEM_FENCE);
    write_imagef(dst, out_idx[0] + m_offset, (reg_c.s0));
}
