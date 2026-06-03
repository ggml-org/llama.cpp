#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK_IFAIRY_Q16 256
#define QK_IFAIRY64   64

static inline float ifairy_bf16_to_f32(ushort h) {
    return as_float(((uint) h) << 16);
}

static inline ushort ifairy_f32_to_bf16(float x) {
    uint bits = as_uint(x);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return (ushort) ((bits >> 16) | 64U);
    }
    return (ushort) ((bits + (0x7fffU + ((bits >> 16) & 1U))) >> 16);
}

static inline int ifairy_clamp_i32(int v, int lo, int hi) {
    return min(hi, max(lo, v));
}

static inline void ifairy64_code_to_sign(uint code, int * wr, int * wi) {
    if (code == 0U) {
        *wr = -1;
        *wi = 0;
    } else if (code == 1U) {
        *wr = 1;
        *wi = 0;
    } else if (code == 2U) {
        *wr = 0;
        *wi = -1;
    } else {
        *wr = 0;
        *wi = 1;
    }
}

static inline uint ifairy64_pack_bf16_pair(float real, float imag) {
    const uint r = (uint) ifairy_f32_to_bf16(real);
    const uint i = (uint) ifairy_f32_to_bf16(imag);
    return r | (i << 16);
}

/**
 * Converts packed-BF16 iFairy activations from F32 carrier format to a SoA
 * q16 staging layout. The q buffer stores real and imaginary int8 planes for
 * each 256-value block; d stores the corresponding fp16 scale pair.
 */
kernel void kernel_ifairy_q16_quantize_block127(
        global char * src,
        ulong         offset,
        global char * act_q,
        global half * act_d,
        int           k,
        int           n,
        ulong         nb10,
        ulong         nb11,
        local float * tmp_real,
        local float * tmp_imag
) {
    src = src + offset;

    const int block = get_group_id(0);
    const int col   = get_group_id(1);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    const int blocks_per_col = k / QK_IFAIRY_Q16;

    if (col >= n || block >= blocks_per_col) {
        return;
    }

    float max_real = 1.0e-5f;
    float max_imag = 1.0e-5f;
    for (int j = lid; j < QK_IFAIRY_Q16; j += lsize) {
        const int k_idx = block * QK_IFAIRY_Q16 + j;
        const uint pair = *((global uint *) (src + (ulong) col * nb11 + (ulong) k_idx * nb10));
        const float xr = ifairy_bf16_to_f32((ushort) (pair & 0xffffU));
        const float xi = ifairy_bf16_to_f32((ushort) (pair >> 16));
        max_real = fmax(max_real, fabs(xr));
        max_imag = fmax(max_imag, fabs(xi));
    }

    tmp_real[lid] = max_real;
    tmp_imag[lid] = max_imag;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            tmp_real[lid] = fmax(tmp_real[lid], tmp_real[lid + stride]);
            tmp_imag[lid] = fmax(tmp_imag[lid], tmp_imag[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float scale_real = tmp_real[0] / 127.0f;
    const float scale_imag = tmp_imag[0] / 127.0f;
    const float iscale_real = 1.0f / scale_real;
    const float iscale_imag = 1.0f / scale_imag;
    const int block_index = col * blocks_per_col + block;

    if (lid == 0) {
        vstore_half(scale_real, 0, act_d + block_index * 2 + 0);
        vstore_half(scale_imag, 0, act_d + block_index * 2 + 1);
    }

    const int q_base = block_index * (2 * QK_IFAIRY_Q16);
    for (int j = lid; j < QK_IFAIRY_Q16; j += lsize) {
        const int k_idx = block * QK_IFAIRY_Q16 + j;
        const uint pair = *((global uint *) (src + (ulong) col * nb11 + (ulong) k_idx * nb10));
        const float xr = ifairy_bf16_to_f32((ushort) (pair & 0xffffU));
        const float xi = ifairy_bf16_to_f32((ushort) (pair >> 16));

        const int qr = ifairy_clamp_i32((int) rint(xr * iscale_real), -127, 127);
        const int qi = ifairy_clamp_i32((int) rint(xi * iscale_imag), -127, 127);

        act_q[q_base + j] = (char) qr;
        act_q[q_base + QK_IFAIRY_Q16 + j] = (char) qi;
    }
}

static inline void ifairy64_accumulate_block(
        global uchar * w_q,
        global half  * w_d,
        global char  * act_q,
        global half  * act_d,
        int            nb64,
        int            nbq,
        int            row,
        int            col,
        int            wb,
        float        * acc_real,
        float        * acc_imag
) {
    const int w_block = row * nb64 + wb;
    const int act_block = wb >> 2;
    const int act_base = (wb & 3) * QK_IFAIRY64;
    const int act_index = col * nbq + act_block;
    const int q_base = act_index * (2 * QK_IFAIRY_Q16) + act_base;

    int sum_ac = 0;
    int sum_ad = 0;
    int sum_bc = 0;
    int sum_bd = 0;

    for (int j = 0; j < QK_IFAIRY64; ++j) {
        const int lane = j & 15;
        const int part = j >> 4;
        const uint packed = (uint) w_q[w_block * 16 + lane];
        const uint code = (packed >> (2 * part)) & 3U;

        int wr;
        int wi;
        ifairy64_code_to_sign(code, &wr, &wi);

        const int xr = (int) act_q[q_base + j];
        const int xi = (int) act_q[q_base + QK_IFAIRY_Q16 + j];

        sum_ac += xr * wr;
        sum_ad += xi * wr;
        sum_bc += xr * wi;
        sum_bd += xi * wi;
    }

    const float w_real = vload_half(w_block * 2 + 0, w_d);
    const float w_imag = vload_half(w_block * 2 + 1, w_d);
    const float x_real = vload_half(act_index * 2 + 0, act_d);
    const float x_imag = vload_half(act_index * 2 + 1, act_d);

    *acc_real += w_real * x_real * (float) sum_ac + w_imag * x_imag * (float) sum_bd;
    *acc_imag += w_imag * x_real * (float) sum_bc - w_real * x_imag * (float) sum_ad;
}

/**
 * Correctness-first iFairy64 matmul. One work-group computes one output
 * element from SoA IFAIRY64 weights and SoA q16 activation staging buffers.
 */
kernel void kernel_ifairy64_mul_mat_f32_q16(
        global uchar * w_q,
        global half  * w_d,
        global char  * act_q,
        global half  * act_d,
        global char  * dst,
        ulong         offsetd,
        int           k,
        int           m,
        int           n,
        ulong         nb0,
        ulong         nb1,
        local float * tmp_real,
        local float * tmp_imag
) {
    dst = dst + offsetd;

    const int row = get_group_id(0);
    const int col = get_group_id(1);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int nb64 = k / QK_IFAIRY64;
    const int nbq = k / QK_IFAIRY_Q16;

    float acc_real = 0.0f;
    float acc_imag = 0.0f;

    if (row < m && col < n) {
        for (int wb = lid; wb < nb64; wb += lsize) {
            ifairy64_accumulate_block(w_q, w_d, act_q, act_d, nb64, nbq, row, col, wb, &acc_real, &acc_imag);
        }
    }

    tmp_real[lid] = acc_real;
    tmp_imag[lid] = acc_imag;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            tmp_real[lid] += tmp_real[lid + stride];
            tmp_imag[lid] += tmp_imag[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0 && row < m && col < n) {
        *((global uint *) (dst + (ulong) col * nb1 + (ulong) row * nb0)) =
            ifairy64_pack_bf16_pair(tmp_real[0], tmp_imag[0]);
    }
}
