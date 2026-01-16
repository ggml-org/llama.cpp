#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hex-dma.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"

#define MM_SPAD_SRC0_NROWS 16
#define MM_SPAD_SRC1_NROWS 16
#define MM_SPAD_DST_NROWS  2

struct htp_matmul_type {
    const char * type;
    void (*vec_dot)(const int n, float * restrict s, const void * restrict vx, const void * restrict vy);
    void (*vec_dot_rx2)(const int n, float * restrict s, const void * restrict vx, uint32_t vx_row_size, const void * restrict vy);
};

// vdelta control to replicate first 4x fp32 values across lanes
static const uint8_t __attribute__((aligned(128))) repl_4x_f32[128] = {
    0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10,
    0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20,
    0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x04,
    0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x40, 0x40, 0x40, 0x40,
    0x44, 0x44, 0x44, 0x44, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20, 0x20, 0x20, 0x04, 0x04,
    0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10,
};

// vdelta control to replicate and interleave first 8x fp32 values across lanes
static const uint8_t __attribute__((aligned(128))) repl_interleave_8x_f32[128] = {
    0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x00, 0x00, 0x00,
    0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20,
    0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20, 0x20, 0x20, 0x04,
    0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x40, 0x40, 0x40, 0x40,
    0x44, 0x44, 0x44, 0x44, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x40, 0x40, 0x40, 0x40, 0x44, 0x44, 0x44,
    0x44, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20, 0x20, 0x20, 0x04, 0x04,
    0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20, 0x20, 0x20,
};

// vdelta control to replicate first fp32 value across all elements
static const uint8_t __attribute__((aligned(128))) repl_1x_f32[128] = {
    0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10,
    0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20, 0x20, 0x20, 0x04, 0x04,
    0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08,
    0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08,
    0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04,
    0x04, 0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10,
    0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
};

// vdelta control to replicate first fp16 value across all elements
static const uint8_t __attribute__((aligned(128))) repl_1x_f16[128] = {
    0x00, 0x00, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x10, 0x10, 0x02,
    0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x20, 0x20, 0x02, 0x02, 0x04, 0x04,
    0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08,
    0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x40, 0x40, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02,
    0x04, 0x04, 0x02, 0x02, 0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02,
    0x02, 0x20, 0x20, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x10, 0x10,
    0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
};

// vdelta control to replicate first fp16 value across all elements
static const uint8_t __attribute__((aligned(128))) repl_2x_f16[128] = {
    0x00, 0x00, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x20, 0x20, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x00, 0x00, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x20, 0x20, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
};

// vdelta control to expand first 32 e8m0 values into 32 uint32 elements
static const uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
    0x00, 0x00, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00, 0x00,
    0x00, 0x11, 0x10, 0x10, 0x10, 0x02, 0x00, 0x04, 0x00, 0x01, 0x02, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04,
    0x00, 0x00, 0x22, 0x20, 0x20, 0x20, 0x21, 0x22, 0x20, 0x24, 0x04, 0x00, 0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x02,
    0x00, 0x04, 0x00, 0x11, 0x12, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08,
    0x01, 0x02, 0x00, 0x04, 0x44, 0x40, 0x40, 0x40, 0x41, 0x40, 0x40, 0x40, 0x42, 0x40, 0x44, 0x40, 0x41, 0x42, 0x48,
    0x48, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x12, 0x10, 0x10, 0x10, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00,
    0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x22, 0x20, 0x24, 0x20, 0x21, 0x22, 0x20, 0x20,
};

static const uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};

// q4x4x2 and q8x4x2 are the flat q4/8_0 formats where all quants are stored first followed by all scales

static inline size_t q8x4x2_row_size(uint32_t ne) {
    // ensures perfect alignment of quants and full row
    const uint32_t qk = QK_Q8_0x4x2;
    const uint32_t nb = (ne + qk - 1) / qk;
    return hex_round_up(ne + nb * 8 * sizeof(__fp16), 128);
}

static inline HVX_Vector_x8 hvx_vec_load_q4x4x8(const uint8_t * restrict ptr) {
    const HVX_Vector * restrict vptr = (const HVX_Vector *) ptr;

    HVX_Vector v0_1 = vptr[0];  // first 256 elements (128 bytes)
    HVX_Vector v2_3 = vptr[1];  // ...
    HVX_Vector v4_5 = vptr[2];  // ...
    HVX_Vector v6_7 = vptr[3];  // ...

    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    HVX_Vector v0 = Q6_V_vand_VV(v0_1, mask_h4);  // & 0x0F
    HVX_Vector v1 = Q6_Vub_vlsr_VubR(v0_1, 4);    // >> 4
    HVX_Vector v2 = Q6_V_vand_VV(v2_3, mask_h4);  // & 0x0F
    HVX_Vector v3 = Q6_Vub_vlsr_VubR(v2_3, 4);    // >> 4
    HVX_Vector v4 = Q6_V_vand_VV(v4_5, mask_h4);  // & 0x0F
    HVX_Vector v5 = Q6_Vub_vlsr_VubR(v4_5, 4);    // >> 4
    HVX_Vector v6 = Q6_V_vand_VV(v6_7, mask_h4);  // & 0x0F
    HVX_Vector v7 = Q6_Vub_vlsr_VubR(v6_7, 4);    // >> 4

    // Convert uint4 to int4 (i.e. x - 8)
    const HVX_Vector i8 = Q6_Vb_vsplat_R(8);
    v0                  = Q6_Vb_vsub_VbVb(v0, i8);
    v1                  = Q6_Vb_vsub_VbVb(v1, i8);
    v2                  = Q6_Vb_vsub_VbVb(v2, i8);
    v3                  = Q6_Vb_vsub_VbVb(v3, i8);
    v4                  = Q6_Vb_vsub_VbVb(v4, i8);
    v5                  = Q6_Vb_vsub_VbVb(v5, i8);
    v6                  = Q6_Vb_vsub_VbVb(v6, i8);
    v7                  = Q6_Vb_vsub_VbVb(v7, i8);

    HVX_Vector_x8 r = { v0, v1, v2, v3, v4, v5, v6, v7 };
    return r;
}

static inline HVX_Vector_x8 hvx_vec_load_mxfp4x4x8(const uint8_t * restrict ptr) {
    const HVX_Vector * restrict vptr = (const HVX_Vector *) ptr;

    HVX_Vector v0_1 = vptr[0];  // first 256 elements (128 bytes)
    HVX_Vector v2_3 = vptr[1];  // ...
    HVX_Vector v4_5 = vptr[2];  // ...
    HVX_Vector v6_7 = vptr[3];  // ...

    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    HVX_Vector v0 = Q6_V_vand_VV(v0_1, mask_h4);  // & 0x0F
    HVX_Vector v1 = Q6_Vub_vlsr_VubR(v0_1, 4);    // >> 4
    HVX_Vector v2 = Q6_V_vand_VV(v2_3, mask_h4);  // & 0x0F
    HVX_Vector v3 = Q6_Vub_vlsr_VubR(v2_3, 4);    // >> 4
    HVX_Vector v4 = Q6_V_vand_VV(v4_5, mask_h4);  // & 0x0F
    HVX_Vector v5 = Q6_Vub_vlsr_VubR(v4_5, 4);    // >> 4
    HVX_Vector v6 = Q6_V_vand_VV(v6_7, mask_h4);  // & 0x0F
    HVX_Vector v7 = Q6_Vub_vlsr_VubR(v6_7, 4);    // >> 4

    HVX_Vector lut = *(const HVX_Vector *) kvalues_mxfp4_lut;
    v0             = Q6_Vb_vlut32_VbVbI(v0, lut, 0);
    v1             = Q6_Vb_vlut32_VbVbI(v1, lut, 0);
    v2             = Q6_Vb_vlut32_VbVbI(v2, lut, 0);
    v3             = Q6_Vb_vlut32_VbVbI(v3, lut, 0);
    v4             = Q6_Vb_vlut32_VbVbI(v4, lut, 0);
    v5             = Q6_Vb_vlut32_VbVbI(v5, lut, 0);
    v6             = Q6_Vb_vlut32_VbVbI(v6, lut, 0);
    v7             = Q6_Vb_vlut32_VbVbI(v7, lut, 0);

    HVX_Vector_x8 r = { v0, v1, v2, v3, v4, v5, v6, v7 };
    return r;
}

static inline HVX_Vector_x8 hvx_vec_load_q8x4x8(const uint8_t * restrict ptr) {
    const HVX_Vector * restrict vptr = (const HVX_Vector *) ptr;

    HVX_Vector v0 = vptr[0];  // first  128 vals
    HVX_Vector v1 = vptr[1];  // ...
    HVX_Vector v2 = vptr[2];  // ...
    HVX_Vector v3 = vptr[3];  // ...
    HVX_Vector v4 = vptr[4];  // ...
    HVX_Vector v5 = vptr[5];  // ...
    HVX_Vector v6 = vptr[6];  // ...
    HVX_Vector v7 = vptr[7];  // ...

    HVX_Vector_x8 r = { v0, v1, v2, v3, v4, v5, v6, v7 };
    return r;
}

static inline HVX_Vector_x4 hvx_vec_load_x4_f16(const uint8_t * restrict ptr) {
    const HVX_Vector * restrict vptr = (const HVX_Vector *) ptr;

    HVX_Vector v0 = vptr[0];  // first  64 vals
    HVX_Vector v1 = vptr[1];  // second 64 vals
    HVX_Vector v2 = vptr[2];  // third  64 vals
    HVX_Vector v3 = vptr[3];  // forth  64 vals

    HVX_Vector_x4 r = { v0, v1, v2, v3 };
    return r;
}

static inline HVX_Vector_x4 hvx_vec_load_x4_f32_as_f16(const uint8_t * restrict ptr) {
    const HVX_VectorPair * restrict vptr = (const HVX_VectorPair *) ptr;

    HVX_VectorPair v0 = vptr[0];  // first  64 vals
    HVX_VectorPair v1 = vptr[1];  // second 64 vals
    HVX_VectorPair v2 = vptr[2];  // third  64 vals
    HVX_VectorPair v3 = vptr[3];  // forth  64 vals

    HVX_Vector vq0_lo = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(v0), Q6_V_vzero());
    HVX_Vector vq0_hi = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(v0), Q6_V_vzero());
    HVX_Vector vq1_lo = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(v1), Q6_V_vzero());
    HVX_Vector vq1_hi = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(v1), Q6_V_vzero());
    HVX_Vector vq2_lo = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(v2), Q6_V_vzero());
    HVX_Vector vq2_hi = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(v2), Q6_V_vzero());
    HVX_Vector vq3_lo = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(v3), Q6_V_vzero());
    HVX_Vector vq3_hi = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(v3), Q6_V_vzero());

    HVX_Vector vh0 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vq0_hi, vq0_lo));
    HVX_Vector vh1 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vq1_hi, vq1_lo));
    HVX_Vector vh2 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vq2_hi, vq2_lo));
    HVX_Vector vh3 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vq3_hi, vq3_lo));

    // vcombine does a shuffle, use vdeal to undo

    HVX_Vector_x4 r = { Q6_Vh_vdeal_Vh(vh0), Q6_Vh_vdeal_Vh(vh1), Q6_Vh_vdeal_Vh(vh2), Q6_Vh_vdeal_Vh(vh3) };
    return r;
}

// Reduce multiply 1024 x 1024 int8 elements (32x q4/8 blocks in 8x HVX vectors).
// Accumulate each block into a single int32 value.
// Return a single HVX vector with 32x int32 accumulators.
// This version is parameterized to support less than 1024 elements.
// if() checks are optimized out at compile time -- make sure to pass N as a constexpr.

static inline HVX_Vector hvx_vec_rmpy_x8_n(HVX_Vector_x8 x, HVX_Vector_x8 y, unsigned int n) {
    HVX_Vector r0 = Q6_V_vsplat_R(0);
    HVX_Vector r1 = Q6_V_vsplat_R(0);
    HVX_Vector r2 = Q6_V_vsplat_R(0);
    HVX_Vector r3 = Q6_V_vsplat_R(0);
    HVX_Vector r4 = Q6_V_vsplat_R(0);
    HVX_Vector r5 = Q6_V_vsplat_R(0);
    HVX_Vector r6 = Q6_V_vsplat_R(0);
    HVX_Vector r7 = Q6_V_vsplat_R(0);

    HVX_VectorPair p3;
    HVX_VectorPair p2;
    HVX_VectorPair p1;
    HVX_VectorPair p0;

    if (n >=  128) { r0 = Q6_Vw_vrmpy_VbVb(x.v[0], y.v[0]); }
    if (n >=  256) { r1 = Q6_Vw_vrmpy_VbVb(x.v[1], y.v[1]); }
    if (n >=  384) { r2 = Q6_Vw_vrmpy_VbVb(x.v[2], y.v[2]); }
    if (n >=  512) { r3 = Q6_Vw_vrmpy_VbVb(x.v[3], y.v[3]); }
    if (n >=  640) { r4 = Q6_Vw_vrmpy_VbVb(x.v[4], y.v[4]); }
    if (n >=  768) { r5 = Q6_Vw_vrmpy_VbVb(x.v[5], y.v[5]); }
    if (n >=  896) { r6 = Q6_Vw_vrmpy_VbVb(x.v[6], y.v[6]); }
    if (n >= 1024) { r7 = Q6_Vw_vrmpy_VbVb(x.v[7], y.v[7]); }

    if (n >=  128) { p0 = Q6_W_vdeal_VVR(r1, r0, -4); }
    if (n >=  384) { p1 = Q6_W_vdeal_VVR(r3, r2, -4); }
    if (n >=  640) { p2 = Q6_W_vdeal_VVR(r5, r4, -4); }
    if (n >=  896) { p3 = Q6_W_vdeal_VVR(r7, r6, -4); }

    if (n >=  128) { r0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(p0), Q6_V_hi_W(p0)); }
    if (n >=  384) { r1 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(p1), Q6_V_hi_W(p1)); }
    if (n >=  640) { r2 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(p2), Q6_V_hi_W(p2)); }
    if (n >=  896) { r3 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(p3), Q6_V_hi_W(p3)); }

    if (n >=  128) { p0 = Q6_W_vdeal_VVR(r1, r0, -4); }
    if (n >=  640) { p1 = Q6_W_vdeal_VVR(r3, r2, -4); }

    if (n >=  128) { r0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(p0), Q6_V_hi_W(p0)); }
    if (n >=  640) { r1 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(p1), Q6_V_hi_W(p1)); }

    if (n >=  128) { p0 = Q6_W_vdeal_VVR(r1, r0, -4); }
    if (n >=  128) { r0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(p0), Q6_V_hi_W(p0)); }

    return r0;
}

static inline HVX_Vector hvx_vec_rmpy_x8_full(HVX_Vector_x8 x, HVX_Vector_x8 y) {
    return hvx_vec_rmpy_x8_n(x, y, 1024);
}

// Handle most common cases of tensors not multiple of 1024.
static inline HVX_Vector hvx_vec_rmpy_x8_nloe(HVX_Vector_x8 x, HVX_Vector_x8 y, unsigned int n) {
    if (n <= 256) { return hvx_vec_rmpy_x8_n(x, y, 256); };
    if (n <= 512) { return hvx_vec_rmpy_x8_n(x, y, 512); };
    if (n <= 768) { return hvx_vec_rmpy_x8_n(x, y, 768); };
    return hvx_vec_rmpy_x8_n(x, y, 1024);
}

static void vec_dot_q4x4x2_q8x4x2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % 32 == 0);  // min sub-block size
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_Q4_0x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 2;                                  // 32x __fp16
    const uint32_t x_qblk_size = qk / 2;                                     // int4
    const uint32_t x_qrow_size = n / 2;                                      // int4 (not padded)

    const uint32_t y_dblk_size = 8 * 4 * 2;                                  // 32x __fp16
    const uint32_t y_qblk_size = qk;                                         // int8
    const uint32_t y_qrow_size = n;                                          // int8 (not padded)

    const uint8_t * restrict r0_x_q = ((const uint8_t *) vx + 0);            // quants first
    const uint8_t * restrict r0_x_d = ((const uint8_t *) vx + x_qrow_size);  // then scales

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);               // quants first
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);     // then scales

    // Row sum (qf32)
    HVX_Vector r0_sum = Q6_V_vsplat_R(0);

    // Multiply and accumulate into int32.
    // Compute combined scale (fp32).
    // Apply scale to acc and accumulate into the row sum (qf32).

    const uint32_t nb   = n / qk;  // num full blocks
    const uint32_t nloe = n % qk;  // num leftover elemements

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q4x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
    }

    // Process leftovers, we still load full 4x4x2 block but zero out unused scales/blocks
    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q4x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r0_q, vy_q, nloe));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));

        // Zero out unused scales
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
    }

    // Reduce and convert into fp32
    r0_sum = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(r0_sum));

    hvx_vec_store_u(&s[0], 4, r0_sum);
}

static void vec_dot_q4x4x2_q8x4x2_rx2(const int n,
                                      float * restrict s,
                                      const void * restrict vx,
                                      uint32_t vx_row_size,
                                      const void * restrict vy) {
    assert(n % 32 == 0);  // min sub-block size
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_Q4_0x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 2;                                                        // 32x __fp16
    const uint32_t x_qblk_size = qk / 2;                                                           // int4
    const uint32_t x_qrow_size = n / 2;                                                            // int4 (not padded)

    const uint32_t y_dblk_size = 8 * 4 * 2;                                                        // 32x __fp16
    const uint32_t y_qblk_size = qk;                                                               // int8
    const uint32_t y_qrow_size = n;                                                                // int8 (not padded)

    const uint8_t * restrict r0_x_q = ((const uint8_t *) (vx + (0 * vx_row_size)) + 0);            // quants first
    const uint8_t * restrict r0_x_d = ((const uint8_t *) (vx + (0 * vx_row_size)) + x_qrow_size);  // then scales

    const uint8_t * restrict r1_x_q = ((const uint8_t *) (vx + (1 * vx_row_size)) + 0);            // quants first
    const uint8_t * restrict r1_x_d = ((const uint8_t *) (vx + (1 * vx_row_size)) + x_qrow_size);  // then scales

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);                                     // quants first
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);                           // then scales

    // Row sum (qf32)
    HVX_Vector r0_sum = Q6_V_vsplat_R(0);
    HVX_Vector r1_sum = Q6_V_vsplat_R(0);

    // Multiply and accumulate into int32.
    // Compute combined scale (fp32).
    // Apply scale to acc and accumulate into the row sum (qf32).

    const uint32_t nb   = n / qk;  // num full blocks
    const uint32_t nloe = n % qk;  // num leftover elemements

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q4x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_q4x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy_q));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));
        HVX_Vector r1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r1_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy_d)));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
        r1_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r1_sum, r1_fa);
    }

    // Process leftovers, we still load full 4x4x2 block but zero out unused scales/blocks
    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q4x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_q4x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r0_q, vy_q, nloe));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r1_q, vy_q, nloe));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));
        HVX_Vector r1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r1_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy_d)));

        // Zero out unused scales
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);
        r1_dd                = Q6_V_vand_QV(bmask, r1_dd);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
        r1_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r1_sum, r1_fa);
    }

    // Convert into fp32 and reduce
    r0_sum = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(r0_sum));
    r1_sum = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(r1_sum));
    HVX_VectorPair p0 = Q6_W_vshuff_VVR(r1_sum, r0_sum, 4);

    hvx_vec_store_u(&s[0], 8, Q6_V_lo_W(p0));
}

static void vec_dot_q8x4x2_q8x4x2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % 32 == 0);  // min sub-block size
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_Q4_0x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 2;                                  // 32x __fp16
    const uint32_t x_qblk_size = qk;                                         // int8
    const uint32_t x_qrow_size = n;                                          // int8 (not padded)

    const uint32_t y_dblk_size = 8 * 4 * 2;                                  // 32x __fp16
    const uint32_t y_qblk_size = qk;                                         // int8
    const uint32_t y_qrow_size = n;                                          // int8 (not padded)

    const uint8_t * restrict r0_x_q = ((const uint8_t *) vx + 0);            // quants first
    const uint8_t * restrict r0_x_d = ((const uint8_t *) vx + x_qrow_size);  // then scales

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);               // quants first
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);     // then scales

    // Row sum (qf32)
    HVX_Vector r0_sum = Q6_V_vsplat_R(0);

    // Multiply and accumulate into int32.
    // Compute combined scale (fp32).
    // Apply scale to acc and accumulate into the row sum (qf32).

    const uint32_t nb   = n / qk;  // num full blocks
    int32_t        nloe = n % qk;  // num leftover elemements (must be signed)

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q8x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
    }

    // Process leftovers, we still load full 4x4x2 block but zero out unused scales/blocks
    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q8x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r0_q, vy_q, nloe));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));

        // Zero out unused scales
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
    }

    // Reduce and convert into fp32
    r0_sum = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(r0_sum));

    hvx_vec_store_u(&s[0], 4, r0_sum);
}

static void vec_dot_q8x4x2_q8x4x2_rx2(const int n,
                                      float * restrict s,
                                      const void * restrict vx,
                                      uint32_t vx_row_size,
                                      const void * restrict vy) {
    assert(n % 32 == 0);  // min sub-block size
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_Q4_0x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 2;                                                        // 32x __fp16
    const uint32_t x_qblk_size = qk;                                                               // int8
    const uint32_t x_qrow_size = n;                                                                // int8 (not padded)

    const uint32_t y_dblk_size = 8 * 4 * 2;                                                        // 32x __fp16
    const uint32_t y_qblk_size = qk;                                                               // int8
    const uint32_t y_qrow_size = n;                                                                // int8 (not padded)

    const uint8_t * restrict r0_x_q = ((const uint8_t *) (vx + (0 * vx_row_size)) + 0);            // quants first
    const uint8_t * restrict r0_x_d = ((const uint8_t *) (vx + (0 * vx_row_size)) + x_qrow_size);  // then scales

    const uint8_t * restrict r1_x_q = ((const uint8_t *) (vx + (1 * vx_row_size)) + 0);            // quants first
    const uint8_t * restrict r1_x_d = ((const uint8_t *) (vx + (1 * vx_row_size)) + x_qrow_size);  // then scales

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);                                     // quants first
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);                           // then scales

    // Row sum (qf32)
    HVX_Vector r0_sum = Q6_V_vsplat_R(0);
    HVX_Vector r1_sum = Q6_V_vsplat_R(0);

    // Multiply and accumulate into int32.
    // Compute combined scale (fp32).
    // Apply scale to acc and accumulate into the row sum (qf32).

    const uint32_t nb   = n / qk;  // num full blocks
    int32_t        nloe = n % qk;  // num leftover elemements (must be signed)

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q8x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_q8x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy_q));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));
        HVX_Vector r1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r1_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy_d)));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
        r1_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r1_sum, r1_fa);
    }

    // Process leftovers, we still load full 4x4x2 block but zero out unused scales/blocks
    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_q8x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_q8x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r0_q, vy_q, nloe));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_nloe(r1_q, vy_q, nloe));

        HVX_Vector vy_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (y_d + i * y_dblk_size));
        HVX_Vector r0_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r0_x_d + i * x_dblk_size));
        HVX_Vector r1_d = Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (r1_x_d + i * x_dblk_size));

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r0_d, vy_d)));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(r1_d, vy_d)));

        // Zero out unused scales
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);
        r1_dd                = Q6_V_vand_QV(bmask, r1_dd);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
        r1_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r1_sum, r1_fa);
    }

    // Convert into fp32 and reduce
    r0_sum = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(r0_sum));
    r1_sum = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(r1_sum));
    HVX_VectorPair p0 = Q6_W_vshuff_VVR(r1_sum, r0_sum, 4);

    hvx_vec_store_u(&s[0], 8, Q6_V_lo_W(p0));
}

static void vec_dot_mxfp4x4x2_q8x4x2(const int n,
                                     float * restrict s,
                                     const void * restrict vx,
                                     const void * restrict vy) {
    assert(n % 32 == 0);  // min sub-block size
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_MXFP4x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 1;                                  // 32x e8m0
    const uint32_t x_qblk_size = qk / 2;                                     // fp4
    const uint32_t x_qrow_size = n / 2;                                      // fp4 (not padded)

    const uint32_t y_dblk_size = 8 * 4 * 2;                                  // 32x __fp16
    const uint32_t y_qblk_size = qk;                                         // int8
    const uint32_t y_qrow_size = n;                                          // int8 (not padded)

    const uint8_t * restrict r0_x_q = ((const uint8_t *) vx + 0);            // quants first
    const uint8_t * restrict r0_x_d = ((const uint8_t *) vx + x_qrow_size);  // then scales

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);               // quants first
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);     // then scales

    // Row sum (qf32)
    HVX_Vector r0_sum = Q6_V_vsplat_R(0);

    // Multiply and accumulate into int32.
    // Compute combined scale (fp32).
    // Apply scale to acc and accumulate into the row sum (qf32).

    const uint32_t nb   = n / qk;  // num full blocks
    int32_t        nloe = n % qk;  // num leftover elemements (must be signed)

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_mxfp4x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));

        HVX_Vector vy_d = *(const HVX_UVector *) (y_d + i * y_dblk_size);
        HVX_Vector r0_d = *(const HVX_UVector *) (r0_x_d + i * x_dblk_size);

        // Convert vy_d from fp16 to fp32 while applying 0.5 scaling which is used for e8m0 halving
        HVX_Vector half = Q6_Vh_vsplat_R(0x3800);  // 0.5 in fp16
        vy_d            = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vy_d), half));
        vy_d            = Q6_Vsf_equals_Vqf32(vy_d);

        // Convert rX_d scales from e8m0 to fp32
        // Expand and zero-pad 32x uint8 e8m0 values to uint32s : 0 0 0 0, 0 0 0 1, 0 0 0 2, ...
        // Left shift with zero fill to create FP32
        // FIXME: might need to handle zero as a special case (see ggml-cpu code)
        HVX_Vector expand    = *(const HVX_Vector *) expand_x32_e8m0;
        HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);
        r0_d                 = Q6_V_vdelta_VV(r0_d, expand);
        r0_d                 = Q6_V_vand_VV(r0_d, e8m0_mask);
        r0_d                 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0_d, vy_d));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
    }

    // Process leftovers
    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_mxfp4x4x8(r0_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));

        HVX_Vector vy_d = *(const HVX_UVector *) (y_d + i * y_dblk_size);
        HVX_Vector r0_d = *(const HVX_UVector *) (r0_x_d + i * x_dblk_size);

        // Convert vy_d from fp16 to fp32 while applying 0.5 scaling which is used for e8m0 halving
        HVX_Vector half = Q6_Vh_vsplat_R(0x3800);  // 0.5 in fp16
        vy_d            = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vy_d), half));
        vy_d            = Q6_Vsf_equals_Vqf32(vy_d);

        // Convert rX_d scales from e8m0 to fp32
        // Expand and zero-pad 32x uint8 e8m0 values to uint32s : 0 0 0 0, 0 0 0 1, 0 0 0 2, ...
        // Left shift with zero fill to create FP32
        // FIXME: might need to handle zero as a special case (see ggml-cpu code)
        HVX_Vector expand    = *(const HVX_Vector *) expand_x32_e8m0;
        HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);
        r0_d                 = Q6_V_vdelta_VV(r0_d, expand);
        r0_d                 = Q6_V_vand_VV(r0_d, e8m0_mask);
        r0_d                 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0_d, vy_d));

        // Zero-out unused scales
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
    }

    // Reduce and convert into fp32
    r0_sum = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(r0_sum));

    hvx_vec_store_u(&s[0], 4, r0_sum);
}

static void vec_dot_mxfp4x4x2_q8x4x2_rx2(const int n,
                                         float * restrict s,
                                         const void * restrict vx,
                                         uint32_t vx_row_size,
                                         const void * restrict vy) {
    assert(n % 32 == 0);  // min sub-block size
    assert((unsigned long) vx % 128 == 0);
    assert((unsigned long) vy % 128 == 0);

    const uint32_t qk = QK_MXFP4x4x2 * 4;

    const uint32_t x_dblk_size = 8 * 4 * 1;                                                        // 32x e8m0
    const uint32_t x_qblk_size = qk / 2;                                                           // fp4
    const uint32_t x_qrow_size = n / 2;                                                            // fp4 (not padded)

    const uint32_t y_dblk_size = 8 * 4 * 2;                                                        // 32x __fp16
    const uint32_t y_qblk_size = qk;                                                               // int8
    const uint32_t y_qrow_size = n;                                                                // int8 (not padded)

    const uint8_t * restrict r0_x_q = ((const uint8_t *) (vx + (0 * vx_row_size)) + 0);            // quants first
    const uint8_t * restrict r0_x_d = ((const uint8_t *) (vx + (0 * vx_row_size)) + x_qrow_size);  // then scales

    const uint8_t * restrict r1_x_q = ((const uint8_t *) (vx + (1 * vx_row_size)) + 0);            // quants first
    const uint8_t * restrict r1_x_d = ((const uint8_t *) (vx + (1 * vx_row_size)) + x_qrow_size);  // then scales

    const uint8_t * restrict y_q = ((const uint8_t *) vy + 0);                                     // quants first
    const uint8_t * restrict y_d = ((const uint8_t *) vy + y_qrow_size);                           // then scales

    // Row sum (qf32)
    HVX_Vector r0_sum = Q6_V_vsplat_R(0);
    HVX_Vector r1_sum = Q6_V_vsplat_R(0);

    // Multiply and accumulate into int32.
    // Compute combined scale (fp32).
    // Apply scale to acc and accumulate into the row sum (qf32).

    const uint32_t nb   = n / qk;  // num full blocks
    int32_t        nloe = n % qk;  // num leftover elemements (must be signed)

    uint32_t i = 0;
    for (; i < nb; i++) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_mxfp4x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_mxfp4x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy_q));

        HVX_Vector vy_d = *(const HVX_UVector *) (y_d + i * y_dblk_size);
        HVX_Vector r0_d = *(const HVX_UVector *) (r0_x_d + i * x_dblk_size);
        HVX_Vector r1_d = *(const HVX_UVector *) (r1_x_d + i * x_dblk_size);

        // Convert vy_d from fp16 to fp32 while applying 0.5 scaling which is used for e8m0 halving
        HVX_Vector half = Q6_Vh_vsplat_R(0x3800);  // 0.5 in fp16
        vy_d            = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vy_d), half));
        vy_d            = Q6_Vsf_equals_Vqf32(vy_d);

        // Convert rX_d scales from e8m0 to fp32
        // Expand and zero-pad 32x uint8 e8m0 values to uint32s : 0 0 0 0, 0 0 0 1, 0 0 0 2, ...
        // Left shift with zero fill to create FP32
        // FIXME: might need to handle zero as a special case (see ggml-cpu code)
        HVX_Vector expand    = *(const HVX_Vector *) expand_x32_e8m0;
        HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);
        r0_d                 = Q6_V_vdelta_VV(r0_d, expand);
        r0_d                 = Q6_V_vand_VV(r0_d, e8m0_mask);
        r0_d                 = Q6_Vw_vasl_VwR(r0_d, 23);
        r1_d                 = Q6_V_vdelta_VV(r1_d, expand);
        r1_d                 = Q6_V_vand_VV(r1_d, e8m0_mask);
        r1_d                 = Q6_Vw_vasl_VwR(r1_d, 23);

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0_d, vy_d));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r1_d, vy_d));

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
        r1_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r1_sum, r1_fa);
    }

    // Process leftovers
    if (nloe) {
        HVX_Vector_x8 vy_q = hvx_vec_load_q8x4x8(y_q + i * y_qblk_size);
        HVX_Vector_x8 r0_q = hvx_vec_load_mxfp4x4x8(r0_x_q + i * x_qblk_size);
        HVX_Vector_x8 r1_q = hvx_vec_load_mxfp4x4x8(r1_x_q + i * x_qblk_size);

        HVX_Vector r0_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r0_q, vy_q));
        HVX_Vector r1_ia = Q6_Vsf_equals_Vw(hvx_vec_rmpy_x8_full(r1_q, vy_q));

        HVX_Vector vy_d = *(const HVX_UVector *) (y_d + i * y_dblk_size);
        HVX_Vector r0_d = *(const HVX_UVector *) (r0_x_d + i * x_dblk_size);
        HVX_Vector r1_d = *(const HVX_UVector *) (r1_x_d + i * x_dblk_size);

        // Convert vy_d from fp16 to fp32 while applying 0.5 scaling which is used for e8m0 halving
        HVX_Vector half = Q6_Vh_vsplat_R(0x3800);  // 0.5 in fp16
        vy_d            = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vy_d), half));
        vy_d            = Q6_Vsf_equals_Vqf32(vy_d);

        // Convert rX_d scales from e8m0 to fp32
        // Expand and zero-pad 32x uint8 e8m0 values to uint32s : 0 0 0 0, 0 0 0 1, 0 0 0 2, ...
        // Left shift with zero fill to create FP32
        // FIXME: might need to handle zero as a special case (see ggml-cpu code)
        HVX_Vector expand    = *(const HVX_Vector *) expand_x32_e8m0;
        HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);
        r0_d                 = Q6_V_vdelta_VV(r0_d, expand);
        r0_d                 = Q6_V_vand_VV(r0_d, e8m0_mask);
        r0_d                 = Q6_Vw_vasl_VwR(r0_d, 23);
        r1_d                 = Q6_V_vdelta_VV(r1_d, expand);
        r1_d                 = Q6_V_vand_VV(r1_d, e8m0_mask);
        r1_d                 = Q6_Vw_vasl_VwR(r1_d, 23);

        HVX_Vector r0_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0_d, vy_d));
        HVX_Vector r1_dd = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r1_d, vy_d));

        // Zero-out unused scales
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe / 8);
        r0_dd                = Q6_V_vand_QV(bmask, r0_dd);
        r1_dd                = Q6_V_vand_QV(bmask, r1_dd);

        HVX_Vector r0_fa = Q6_Vqf32_vmpy_VsfVsf(r0_ia, r0_dd);
        HVX_Vector r1_fa = Q6_Vqf32_vmpy_VsfVsf(r1_ia, r1_dd);

        r0_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r0_sum, r0_fa);
        r1_sum = Q6_Vqf32_vadd_Vqf32Vqf32(r1_sum, r1_fa);
    }

    // Convert into fp32 and reduce
    r0_sum = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(r0_sum));
    r1_sum = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(r1_sum));
    HVX_VectorPair p0 = Q6_W_vshuff_VVR(r1_sum, r0_sum, 4);

    hvx_vec_store_u(&s[0], 8, Q6_V_lo_W(p0));
}

static void vec_dot_f16_f16_aa(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const HVX_Vector * restrict x = (const HVX_Vector *) vx;
    const HVX_Vector * restrict y = (const HVX_Vector *) vy;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_Vector rsum = Q6_V_vsplat_R(0);

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; i++) {
        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(x[i], y[i]);
        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector x_hf = Q6_V_vand_QV(bmask, x[i]);
        HVX_Vector y_hf = Q6_V_vand_QV(bmask, y[i]);

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(x_hf, y_hf);
        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    rsum = Q6_Vsf_equals_Vqf32(hvx_vec_reduce_sum_qf32(rsum));
    hvx_vec_store_u(&s[0], 4, rsum);
}

static void vec_dot_f16_f16_aa_rx2(const int n,
                                float * restrict s,
                                const void * restrict vx,
                                uint32_t vx_row_size,
                                const void * restrict vy) {
    const HVX_Vector * restrict x0 = (const HVX_Vector *) vx;
    const HVX_Vector * restrict x1 = (const HVX_Vector *) ((const uint8_t *) vx + vx_row_size);
    const HVX_Vector * restrict y  = (const HVX_Vector *) vy;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    HVX_Vector rsum0 = Q6_V_vsplat_R(0);
    HVX_Vector rsum1 = Q6_V_vsplat_R(0);

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        HVX_Vector y_hf = y[i];
        HVX_VectorPair xy0_qf = Q6_Wqf32_vmpy_VhfVhf(x0[i], y_hf);
        HVX_VectorPair xy1_qf = Q6_Wqf32_vmpy_VhfVhf(x1[i], y_hf);

        rsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(rsum0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy0_qf), Q6_V_hi_W(xy0_qf)));
        rsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(rsum1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy1_qf), Q6_V_hi_W(xy1_qf)));
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector x0_hf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_Vector x1_hf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_Vector y_hf  = Q6_V_vand_QV(bmask, y[i]);

        HVX_VectorPair xy0_qf = Q6_Wqf32_vmpy_VhfVhf(x0_hf, y_hf);
        HVX_VectorPair xy1_qf = Q6_Wqf32_vmpy_VhfVhf(x1_hf, y_hf);

        rsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(rsum0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy0_qf), Q6_V_hi_W(xy0_qf)));
        rsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(rsum1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy1_qf), Q6_V_hi_W(xy1_qf)));
    }

    rsum0 = Q6_Vsf_equals_Vqf32(hvx_vec_reduce_sum_qf32(rsum0));
    rsum1 = Q6_Vsf_equals_Vqf32(hvx_vec_reduce_sum_qf32(rsum1));
    HVX_VectorPair p0 = Q6_W_vshuff_VVR(rsum1, rsum0, 4);

    hvx_vec_store_u(&s[0], 8, Q6_V_lo_W(p0));
}

static void vec_dot_f16_f16_uu(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const HVX_UVector * restrict x = (const HVX_UVector *) vx;
    const HVX_UVector * restrict y = (const HVX_UVector *) vy;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_Vector rsum = Q6_V_vsplat_R(0);

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; i++) {
        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(x[i], y[i]);
        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector x_hf = Q6_V_vand_QV(bmask, x[i]);
        HVX_Vector y_hf = Q6_V_vand_QV(bmask, y[i]);

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(x_hf, y_hf);
        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    rsum = Q6_Vsf_equals_Vqf32(hvx_vec_reduce_sum_qf32(rsum));
    hvx_vec_store_u(&s[0], 4, rsum);
}

static void vec_dot_f16_f32_uu(const int n, float * restrict s, const void * restrict x, const void * restrict y) {
    const HVX_UVector * restrict vx = (const HVX_UVector * restrict) x;
    const HVX_UVector * restrict vy = (const HVX_UVector * restrict) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    const HVX_Vector zero = Q6_V_vsplat_R(0);

    HVX_Vector       rsum = Q6_V_vsplat_R(0);

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; i++) {
        // Load y (fp32) and convert into fp16
        HVX_Vector y0_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+0], zero);  // 32 elements
        HVX_Vector y1_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+1], zero);  // 32 elements
        HVX_Vector y_hf  = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(y1_qf, y0_qf)));

        // Load x (fp16)
        HVX_Vector x_hf  = vx[i];

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(x_hf, y_hf);

        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    if (nloe) {
        // Load y (fp32) and convert into fp16
        HVX_Vector y0_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+0], zero);  // 32 elements
        HVX_Vector y1_qf = Q6_Vqf32_vsub_VsfVsf(vy[i*2+1], zero);  // 32 elements
        HVX_Vector y_hf  = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(y1_qf, y0_qf)));

        // Load x (fp16)
        HVX_Vector x_hf  = vx[i];

        // Zero-out unused elements
        // Note that we need to clear both x and y because they may contain NANs
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        x_hf = Q6_V_vand_QV(bmask, x_hf);
        y_hf = Q6_V_vand_QV(bmask, y_hf);

        HVX_VectorPair xy_qf = Q6_Wqf32_vmpy_VhfVhf(x_hf, y_hf);

        rsum = Q6_Vqf32_vadd_Vqf32Vqf32(rsum, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(xy_qf),  Q6_V_hi_W(xy_qf)));
    }

    rsum = Q6_Vsf_equals_Vqf32(hvx_vec_reduce_sum_qf32(rsum));
    hvx_vec_store_u(&s[0], 4, rsum);
}

#define htp_matmul_tensors_preamble    \
    struct htp_tensor * restrict src0    = &octx->src0;      \
    struct htp_tensor * restrict src1    = &octx->src1;      \
    struct htp_tensor * restrict src2    = &octx->src2;      \
    struct htp_tensor * restrict dst     = &octx->dst;       \
    struct htp_spad * restrict src0_spad = &octx->src0_spad; \
    struct htp_spad * restrict src1_spad = &octx->src1_spad; \
    struct htp_spad * restrict dst_spad  = &octx->dst_spad;  \
                                                             \
    const uint32_t ne00 = src0->ne[0]; \
    const uint32_t ne01 = src0->ne[1]; \
    const uint32_t ne02 = src0->ne[2]; \
    const uint32_t ne03 = src0->ne[3]; \
                                       \
    const uint32_t ne10 = src1->ne[0]; \
    const uint32_t ne11 = src1->ne[1]; \
    const uint32_t ne12 = src1->ne[2]; \
    const uint32_t ne13 = src1->ne[3]; \
                                       \
    const uint32_t ne20 = src2->ne[0]; \
    const uint32_t ne21 = src2->ne[1]; \
    const uint32_t ne22 = src2->ne[2]; \
    const uint32_t ne23 = src2->ne[3]; \
                                       \
    const uint32_t ne0 = dst->ne[0];   \
    const uint32_t ne1 = dst->ne[1];   \
    const uint32_t ne2 = dst->ne[2];   \
    const uint32_t ne3 = dst->ne[3];   \
                                       \
    const uint32_t nb00 = src0->nb[0]; \
    const uint32_t nb01 = src0->nb[1]; \
    const uint32_t nb02 = src0->nb[2]; \
    const uint32_t nb03 = src0->nb[3]; \
                                       \
    const uint32_t nb10 = src1->nb[0]; \
    const uint32_t nb11 = src1->nb[1]; \
    const uint32_t nb12 = src1->nb[2]; \
    const uint32_t nb13 = src1->nb[3]; \
                                       \
    const uint32_t nb0 = dst->nb[0];   \
    const uint32_t nb1 = dst->nb[1];   \
    const uint32_t nb2 = dst->nb[2];   \
    const uint32_t nb3 = dst->nb[3];

#define htp_matmul_preamble            \
    htp_matmul_tensors_preamble;       \
    dma_queue *dma_queue           = octx->ctx->dma[ith];         \
    uint32_t src0_nrows_per_thread = octx->src0_nrows_per_thread;

// *** matmul with support for 4d tensors and full broadcasting

static void matmul_4d(struct htp_matmul_type * mt, struct htp_ops_context * octx, uint32_t nth, uint32_t ith) {
    htp_matmul_preamble;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const uint32_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const uint32_t nr1 = ne1 * ne2 * ne3;

    // distribute the thread work across the inner or outer loop based on which one is larger
    uint32_t nchunk0 = nr0 > nr1 ? nth : 1;  // parallelize by src0 rows
    uint32_t nchunk1 = nr0 > nr1 ? 1 : nth;  // parallelize by src1 rows

    // The number of elements in each chunk
    const uint32_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const uint32_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    uint32_t current_chunk = ith;

    const uint32_t ith0 = current_chunk % nchunk0;
    const uint32_t ith1 = current_chunk / nchunk0;

    const uint32_t ir0_start = dr0 * ith0;
    const uint32_t ir0_end   = MIN(ir0_start + dr0, nr0);

    const uint32_t ir1_start = dr1 * ith1;
    const uint32_t ir1_end   = MIN(ir1_start + dr1, nr1);

    // no work for this thread
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    // block-tiling attempt
    const uint32_t blck_0 = 64;
    const uint32_t blck_1 = 64;

    for (uint32_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (uint32_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (uint32_t ir1 = iir1; ir1 < MIN(iir1 + blck_1, ir1_end); ir1++) {
                const uint32_t i13 = fastdiv(ir1, &octx->mm_div_ne12_ne1);
                const uint32_t i12 = fastdiv(ir1 - i13 * ne12 * ne1, &octx->mm_div_ne1);
                const uint32_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const uint32_t i03 = fastdiv(i13, &octx->mm_div_r3);
                const uint32_t i02 = fastdiv(i12, &octx->mm_div_r2);

                const uint32_t i1 = i11;
                const uint32_t i2 = i12;
                const uint32_t i3 = i13;

                const uint8_t * restrict src0_base = (const uint8_t *) src0->data + (0 + i02 * nb02 + i03 * nb03);
                const uint8_t * restrict src1_col  = (const uint8_t *) src1->data + (i11 * nb11 + i12 * nb12 + i13 * nb13);
                float * dst_col = (float *) ((uint8_t * restrict) dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                const uint32_t ir0_block_end = MIN(iir0 + blck_0, ir0_end);
                for (uint32_t ir0 = iir0; ir0 < ir0_block_end; ir0++) {
                    const uint8_t * restrict src0_row = src0_base + ir0 * nb01;
                    mt->vec_dot(ne00, &dst_col[ir0], src0_row, src1_col);
                }
            }
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "matmul-4d %d/%d: %ux%ux%ux%u (%u:%u %u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0_start, ir0_end, ir1_start, ir1_end, src1->ne[0],
         src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

// src1 tensor is already in VTCM spad
static void matmul_2d(struct htp_matmul_type * mt, struct htp_ops_context * octx, uint32_t nth, uint32_t ith) {
    htp_matmul_preamble;

    const uint32_t src0_nrows = ne01 * ne02 * ne03;  // src0 rows
    const uint32_t src1_nrows = ne11 * ne12 * ne13;  // src1 rows

    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src1_row_size = nb11;

    const size_t src0_stride = src0_spad->stride;
    const size_t src1_stride = src1_spad->stride;

    // Per-thread VTCM scratchpads for all tensors
    // Note that the entire src1 tensor is already in VTCM
    // For other tensors we allocate N rows per thread, padded to HVX vector size
    uint8_t * restrict spad_dst  = dst_spad->data + dst_spad->size_per_thread * ith;
    uint8_t * restrict spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;
    uint8_t * restrict src1_data = src1_spad->data;

    volatile uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;

    // Prefill spad with src0 rows
    #pragma unroll(4)
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const int is0 = (ir0 - src0_start_row);
        if (is0 >= MM_SPAD_SRC0_NROWS) {
            break;
        }
        dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, 2);
    }

    // Process src0 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;

        #pragma unroll(2)
        for (uint32_t ir1 = 0; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
            float * restrict dst_row          = (float *) (dst->data + (ir1 * dst_row_size));
            mt->vec_dot_rx2(ne00, &dst_row[ir0], ss0, src0_stride, src1_col);
        }

        // Prefetch next (n + spad_nrows) row
        const int pr0 = (ir0 + MM_SPAD_SRC0_NROWS);
        const int is0 = (pr0 - src0_start_row) % MM_SPAD_SRC0_NROWS;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        const int is0 = (ir0 - src0_start_row);
        dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, 1);
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;

        #pragma unroll(2)
        for (uint32_t ir1 = 0; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
            float * restrict dst_row          = (float *) (dst->data + (ir1 * dst_row_size));
            mt->vec_dot(ne00, &dst_row[ir0], ss0, src1_col);
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "matmul-%s %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", mt->type, ith, nth,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0], src1->ne[1],
         src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

// q8x4x2 src1 tensor is already in VTCM spad
static void matvec_2d(struct htp_matmul_type * mt, struct htp_ops_context * octx, uint32_t nth, uint32_t ith) {
    htp_matmul_preamble;

    const uint32_t src0_nrows = ne01;

    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src1_row_size = nb11;

    const size_t src0_stride = src0_spad->stride;
    const size_t src1_stride = src1_spad->stride;

    // Per-thread VTCM scratchpads for all tensors
    // Note that the entire src1 tensor is already in VTCM
    // For other tensors we allocate N rows per thread, padded to HVX vector size
    uint8_t * spad_dst  = dst_spad->data + dst_spad->size_per_thread * ith;
    uint8_t * spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;
    uint8_t * src1_data = src1_spad->data;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    float * tmp = (float *) spad_dst;

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;
    const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
    float * restrict dst_col          = (float *) dst->data;

    // Prefill spad with 2x src0 rows
    #pragma unroll(2)
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint32_t is0 = (ir0 - src0_start_row);
        if (is0 >= MM_SPAD_SRC0_NROWS) {
            break;
        }
        dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, 2);
    }

    // Process src0 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        mt->vec_dot_rx2(ne00, &tmp[ir0 - src0_start_row], ss0, src0_stride, src1_col);

        // Prefetch next (n + spad_nrows) row
        const uint32_t pr0 = (ir0 + MM_SPAD_SRC0_NROWS);
        const uint32_t is0 = (pr0 - src0_start_row) % MM_SPAD_SRC0_NROWS;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        const uint32_t ir0 = src0_end_row_x2;
        const uint32_t is0 = (ir0 - src0_start_row);
        dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, 1);
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        mt->vec_dot(ne00, &tmp[ir0 - src0_start_row], ss0, src1_col);
    }

    hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, src0_end_row - src0_start_row);

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "matvec-%s %u/%u: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", mt->type, ith, nth,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0], src1->ne[1],
         src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * ids->ne[0] * ids->ne[1] + (i1)]

struct mmid_row_mapping {
    uint32_t i1;
    uint32_t i2;
};

// src1 tensor is already in VTCM spad
static void matmul_id(struct htp_matmul_type * mt, struct htp_ops_context * octx, uint32_t nth, uint32_t ith) {
    htp_matmul_preamble;

    struct htp_tensor * restrict     ids = &octx->src2;
    struct htp_spad * restrict src2_spad = &octx->src2_spad;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint32_t src0_nrows = ne01;  // src0 rows per expert
    const uint32_t src1_nrows = ne11;

    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    const uint32_t n_ids = ids->ne[0];  // n_expert_used
    const uint32_t n_as  = ne02;        // n_expert

    const size_t matrix_row_counts_size = n_as * sizeof(uint32_t);
    const size_t matrix_row_map_size    = n_as * ids->ne[0] * ids->ne[1] * sizeof(struct mmid_row_mapping);

    const uint32_t *                matrix_row_counts = (const uint32_t *) src2_spad->data + 0;
    const struct mmid_row_mapping * matrix_rows       = (const void *) src2_spad->data + matrix_row_counts_size;

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src1_row_size = q8x4x2_row_size(ne10);

    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    // Per-thread VTCM scratchpads for all tensors
    // Note that the entire src1 tensor is already in VTCM
    // For other tensors we allocate N rows per thread, padded to HVX vector size
    uint8_t * restrict spad_dst  = dst_spad->data + dst_spad->size_per_thread * ith;
    uint8_t * restrict spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;
    uint8_t * restrict src1_data = src1_spad->data;

    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const uint8_t * src0_row = (const uint8_t *) src0->data + (0 + cur_a * nb02 + 0);

        // Prefill spad with src0 rows
        #pragma unroll(4)
        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const int is0 = (ir0 - src0_start_row);
            if (is0 >= MM_SPAD_SRC0_NROWS) {
                break;
            }
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + ir0 * src0_row_size),
                           src0_row_size_padded, src0_row_size, 2);
        }

        // Process src0 rows
        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;

            for (uint32_t cid = 0; cid < cne1; ++cid) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                const int               rm1         = row_mapping.i1;  // expert idx
                const int               rm2         = row_mapping.i2;  // token idx

                const uint32_t ir1 = src1_nrows == 1 ? 0 : rm1;        // src1 row idx
                const uint8_t * restrict src1_col =
                    (const uint8_t *) (src1_data + (ir1 + rm2 * ne11 + 0) * src1_row_size);
                float * dst_row = (float *) (dst->data + (rm1 * nb1 + rm2 * nb2 + 0));

                mt->vec_dot_rx2(ne00, &dst_row[ir0], ss0, src0_row_size_padded, src1_col);
            }

            // Prefetch next (n + spad_nrows) row
            const int pr0 = (ir0 + MM_SPAD_SRC0_NROWS);
            const int is0 = (pr0 - src0_start_row) % MM_SPAD_SRC0_NROWS;
            if (pr0 < src0_end_row_x2) {
                dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + pr0 * src0_row_size),
                               src0_row_size_padded, src0_row_size, 2);
            }
        }

        // Process the last row (if any)
        if (src0_end_row != src0_end_row_x2) {
            uint32_t       ir0 = src0_end_row_x2;
            const uint32_t is0 = (ir0 - src0_start_row);
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + ir0 * src0_row_size),
                           src0_row_size_padded, src0_row_size, 1);
            const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;

            for (uint32_t cid = 0; cid < cne1; ++cid) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                const int               rm1         = row_mapping.i1;  // expert idx
                const int               rm2         = row_mapping.i2;  // token idx

                const uint32_t ir1 = src1_nrows == 1 ? 0 : rm1;        // src1 row idx
                const uint8_t * restrict src1_col =
                    (const uint8_t *) (src1_data + (ir1 + rm2 * ne11 + 0) * src1_row_size);
                float * dst_row = (float *) (dst->data + (rm1 * nb1 + rm2 * nb2 + 0));

                mt->vec_dot(ne00, &dst_row[ir0], ss0, src1_col);
            }
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "matmul-id-%s %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u usec %u\n", mt->type,
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0],
         src1->ne[1], src1->ne[2], src1->ne[3], ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

// src1 tensor is already in VTCM spad
static void matvec_id(struct htp_matmul_type * mt, struct htp_ops_context * octx, uint32_t nth, uint32_t ith) {
    htp_matmul_preamble;

    struct htp_tensor * restrict     ids = &octx->src2;
    struct htp_spad * restrict src2_spad = &octx->src2_spad;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint32_t src0_nrows = ne01;  // src0 rows per expert

    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    assert(ne13 % ne03 == 0);

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src1_row_size = q8x4x2_row_size(ne10);

    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    const uint32_t n_aids = src2->ne[0];  // num activated experts
    const uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCM scratchpads for all tensors
    // Note that the entire src1 tensor is already in VTCM
    // For other tensors we allocate N rows per thread, padded to HVX vector size
    uint8_t * restrict spad_dst  = dst_spad->data + dst_spad->size_per_thread * ith;
    uint8_t * restrict spad_src0 = src0_spad->data + src0_spad->size_per_thread * ith;
    uint8_t * restrict src1_data = src1_spad->data;

    for (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        const uint32_t eid = *(const int32_t *) ((const uint8_t *) src2->data + ie1 * src2->nb[0]);
        assert(eid < n_ids);

        const uint8_t * restrict src0_row = (const uint8_t *) src0->data + eid * nb02;
        const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
        float * restrict dst_row          = (float *) (dst->data + ie1 * nb1);

        // Prefill spad with src0 rows
        #pragma unroll(4)
        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const int is0 = (ir0 - src0_start_row);
            if (is0 >= MM_SPAD_SRC0_NROWS) {
                break;
            }
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + ir0 * src0_row_size),
                           src0_row_size_padded, src0_row_size, 2);
        }

        // Process src0 rows
        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
            mt->vec_dot_rx2(ne00, &dst_row[ir0], ss0, src0_row_size_padded, src1_col);

            // Prefetch next (n + spad_nrows) row
            const int pr0 = (ir0 + MM_SPAD_SRC0_NROWS);
            const int is0 = (pr0 - src0_start_row) % MM_SPAD_SRC0_NROWS;
            if (pr0 < src0_end_row_x2) {
                dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + pr0 * src0_row_size),
                               src0_row_size_padded, src0_row_size, 2);
            }
        }

        // Process the last row (if any)
        if (src0_end_row != src0_end_row_x2) {
            uint32_t       ir0 = src0_end_row_x2;
            const uint32_t is0 = (ir0 - src0_start_row);
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(spad_src0 + is0 * src0_row_size_padded, src0_row + ir0 * src0_row_size),
                           src0_row_size_padded, src0_row_size, 1);
            const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
            mt->vec_dot(ne00, &dst_row[ir0], ss0, src1_col);
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "matvec-id-%s %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u usec %u\n", mt->type,
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0],
         src1->ne[1], src1->ne[2], src1->ne[3], src2->ne[0], src2->ne[1], src2->ne[2], src2->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

// *** dynamic quant

static inline void quantize_block_f32_q8x1(float * restrict x, uint8_t * restrict y_q, uint8_t * restrict y_d) {
    assert((unsigned long) x % 128 == 0);
    assert((unsigned long) y_q % 128 == 0);

    HVX_Vector * vx = (HVX_Vector *) x;
    HVX_Vector zero   = Q6_V_vsplat_R(0);

    // Use reduce max fp32 to find max(abs(e)) first
    HVX_Vector vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
    HVX_Vector vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
    HVX_Vector vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
    HVX_Vector vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));
    // Load and convert into QF32
    HVX_Vector vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);  // 32 elements
    HVX_Vector vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);  // 32 elements
    HVX_Vector vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);  // 32 elements
    HVX_Vector vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);  // 32 elements

    // Convert to QF32
    HVX_Vector vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
    HVX_Vector vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);
    HVX_Vector vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
    HVX_Vector vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

    // Combine and convert to fp16
    HVX_Vector vmax01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
    HVX_Vector vmax23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));

    // Convert into fp16
    HVX_Vector vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_Vector vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    // Replicate first fp16 scale across all lanes
    HVX_Vector ctrl = *(const HVX_Vector *) repl_2x_f16;
    vmax01_hf         = Q6_V_vdelta_VV(vmax01_hf, ctrl);
    vmax23_hf         = Q6_V_vdelta_VV(vmax23_hf, ctrl);

    HVX_Vector vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_Vector vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

    hvx_vec_store_u(y_d + 0, 2, vd01_hf);
    HVX_Vector rotated_vd_hf = Q6_V_vror_VR(vd01_hf, 64);
    hvx_vec_store_u(y_d + 2, 2, rotated_vd_hf);

    hvx_vec_store_u(y_d + 4, 2, vd23_hf);
    rotated_vd_hf = Q6_V_vror_VR(vd23_hf, 64);
    hvx_vec_store_u(y_d + 6, 2, rotated_vd_hf);

    // Divide input by the scale
    HVX_Vector vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
    HVX_Vector vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
    vx01_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
    vx23_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    // Convert to int8
    HVX_Vector vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_Vector vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_Vector vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    *(HVX_Vector *) y_q = vx_i8;
}

static inline void quantize_block_f32_q8x2(float * restrict x, uint8_t * restrict y_q, uint8_t * restrict y_d) {
    assert((unsigned long) x % 128 == 0);
    assert((unsigned long) y_q % 128 == 0);

    HVX_Vector * vx = (HVX_Vector *) x;

    // Load and convert into QF32
    HVX_Vector zero   = Q6_V_vsplat_R(0);
    HVX_Vector vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);  // 32 elements
    HVX_Vector vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);  // 32 elements
    HVX_Vector vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);  // 32 elements
    HVX_Vector vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);  // 32 elements

    // Convert into fp16
    HVX_Vector vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_Vector vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    // Compute max and scale
    HVX_Vector vmax01_hf = hvx_vec_reduce_max_f16(hvx_vec_abs_f16(vx01_hf));
    HVX_Vector vmax23_hf = hvx_vec_reduce_max_f16(hvx_vec_abs_f16(vx23_hf));

    // Replicate first fp16 scale across all lanes
    HVX_Vector ctrl = *(const HVX_Vector *) repl_1x_f16;
    vmax01_hf         = Q6_V_vdelta_VV(vmax01_hf, ctrl);
    vmax23_hf         = Q6_V_vdelta_VV(vmax23_hf, ctrl);

    HVX_Vector vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_Vector vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

    hvx_vec_store_u(y_d + 0, 4, vd01_hf);
    hvx_vec_store_u(y_d + 4, 4, vd23_hf);

    // Divide input by the scale
    HVX_Vector vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
    HVX_Vector vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
    vx01_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
    vx23_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    // Convert to int8
    HVX_Vector vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_Vector vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_Vector vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    *(HVX_Vector *) y_q = vx_i8;
}

static inline void quantize_block_f32_q8x4(float * restrict x, uint8_t * restrict y_q, uint8_t * restrict y_d) {
    assert((unsigned long) x % 128 == 0);
    assert((unsigned long) y_q % 128 == 0);

    HVX_Vector * vx = (HVX_Vector *) x;

    // Load and convert into QF32
    HVX_Vector zero   = Q6_V_vsplat_R(0);
    HVX_Vector vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);  // 32 elements
    HVX_Vector vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);  // 32 elements
    HVX_Vector vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);  // 32 elements
    HVX_Vector vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);  // 32 elements

    // Convert into fp16
    HVX_Vector vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_Vector vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    // Compute max and scale
    HVX_Vector vmax_hf = hvx_vec_reduce_max_f16(hvx_vec_abs_f16(vx01_hf));
    vmax_hf            = hvx_vec_reduce_max2_f16(hvx_vec_abs_f16(vx23_hf), vmax_hf);

    // Replicate first fp16 scale across all lanes
    HVX_Vector ctrl = *(const HVX_Vector *) repl_1x_f16;
    vmax_hf         = Q6_V_vdelta_VV(vmax_hf, ctrl);

    HVX_Vector vd_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd_hf   = Q6_Vhf_equals_Vqf16(vd_qf16);

    *(HVX_UVector *) y_d = vd_hf;

    // Divide input by the scale
    HVX_Vector vd_inv_hf = hvx_vec_inverse_f16(vd_hf);
    vx01_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd_inv_hf));
    vx23_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd_inv_hf));

    // Convert to int8
    HVX_Vector vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_Vector vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_Vector vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    *(HVX_Vector *) y_q = vx_i8;
}

// Overrides input x
static void quantize_row_f32_q8x4x2(float * restrict x, uint8_t * restrict y, uint32_t k) {
    assert(k % 32 == 0);
    const uint32_t qk = QK_Q8_0x4x2;
    const uint32_t nb = (k + qk - 1) / qk;

    const uint32_t qrow_size = k;              // int8

    const uint32_t dblk_size = 8 * 2;          // 8x __fp16
    const uint32_t qblk_size = QK_Q8_0x4x2;    // int8

    uint8_t * restrict y_q = (y + 0);          // quants first
    uint8_t * restrict y_d = (y + qrow_size);  // then scales

    // Temp scales override input since we're working off of the aligned temp buffer in VTCM
    uint8_t * restrict t_d = (uint8_t *) x;

    for (uint32_t i = 0; i < nb; i++) {
#if FP32_QUANTIZE_GROUP_SIZE == 32
        quantize_block_f32_q8x1(x + (i*2 + 0) * qk/2, y_q + (i*2 + 0) * qblk_size/2, t_d + (i*2 + 0) * dblk_size/2);
        quantize_block_f32_q8x1(x + (i*2 + 1) * qk/2, y_q + (i*2 + 1) * qblk_size/2, t_d + (i*2 + 1) * dblk_size/2);
#elif FP32_QUANTIZE_GROUP_SIZE == 64
        quantize_block_f32_q8x2(x + (i*2 + 0) * qk/2, y_q + (i*2 + 0) * qblk_size/2, t_d + (i*2 + 0) * dblk_size/2);
        quantize_block_f32_q8x2(x + (i*2 + 1) * qk/2, y_q + (i*2 + 1) * qblk_size/2, t_d + (i*2 + 1) * dblk_size/2);
#elif FP32_QUANTIZE_GROUP_SIZE == 128
        quantize_block_f32_q8x4(x + (i*2 + 0) * qk/2, y_q + (i*2 + 0) * qblk_size/2, t_d + (i*2 + 0) * dblk_size/2);
        quantize_block_f32_q8x4(x + (i*2 + 1) * qk/2, y_q + (i*2 + 1) * qblk_size/2, t_d + (i*2 + 1) * dblk_size/2);
#else
#error "FP32_QUANTIZE_GROUP_SIZE must be 32, 64, or 128"
#endif
    }

    // now copy the scales into final location
    hvx_copy_f16_ua(y_d, t_d, nb * 8);
}

static void quantize_f32_q8x4x2(const struct htp_tensor * src,
                                 uint8_t * restrict dst,
                                 struct htp_spad * spad,
                                 uint32_t          nth,
                                 uint32_t          ith,
                                 uint32_t          nrows_per_thread) {

    uint64_t t1 = HAP_perf_get_qtimer_count();

    const uint32_t ne0 = src->ne[0];
    const uint32_t ne1 = src->ne[1];
    const uint32_t ne2 = src->ne[2];
    const uint32_t ne3 = src->ne[3];

    const uint32_t nrows = ne1 * ne2 * ne3;                             // total n_rows

    const uint32_t ir_first = nrows_per_thread * ith;                   // first row
    const uint32_t ir_last  = MIN(ir_first + nrows_per_thread, nrows);  // last row

    const size_t src_row_size = src->nb[1];
    const size_t dst_row_size = q8x4x2_row_size(ne0);

    uint8_t * restrict src_data = (uint8_t *) src->data + (src_row_size * ir_first);
    uint8_t * restrict dst_data = (uint8_t *) dst + (dst_row_size * ir_first);
    uint8_t * restrict tmp_data = (uint8_t *) spad->data + (spad->size_per_thread * ith);

    const size_t src_row_size_padded = hex_round_up(src_row_size, QK_Q8_0x4x2 * sizeof(float));
    memset(tmp_data, 0, src_row_size_padded);  // zero-out temp row data for padding

    for (uint32_t i = ir_first; i < ir_last; ++i) {
        hex_l2fetch(src_data, src_row_size, src_row_size, 2);
        hvx_copy_f32_aa(tmp_data, src_data, ne0);

        // FARF(HIGH, "quantize-q8x4-row: %u\n", i);
        quantize_row_f32_q8x4x2((float *) tmp_data, dst_data, ne0);
        dst_data += dst_row_size;
        src_data += src_row_size;
    }

    uint64_t t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "quantize-f32-q8x4: %u/%u : n-rows %u (%u:%u) row-size %u -> %u usec %u\n", ith, nth, nrows, ir_first,
         ir_last, src_row_size, dst_row_size, (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void quantize_f32_f16(const struct htp_tensor * src, uint8_t * restrict dst, uint32_t nth, uint32_t ith,
                              uint32_t nrows_per_thread, uint32_t dst_stride) {

    uint64_t t1 = HAP_perf_get_qtimer_count();

    const uint32_t ne0 = src->ne[0];
    const uint32_t ne1 = src->ne[1];
    const uint32_t ne2 = src->ne[2];
    const uint32_t ne3 = src->ne[3];

    const uint32_t nrows = ne1 * ne2 * ne3;                             // total n_rows

    const uint32_t ir_first = nrows_per_thread * ith;                   // first row
    const uint32_t ir_last  = MIN(ir_first + nrows_per_thread, nrows);  // last row

    const size_t src_row_size = ne0 * sizeof(float);
    const size_t src_stride   = src->nb[1];

    uint8_t * restrict src_data = (uint8_t *) src->data + (src_stride * ir_first);
    uint8_t * restrict dst_data = (uint8_t *) dst       + (dst_stride * ir_first);

    for (uint32_t i = ir_first; i < ir_last; ++i) {
        hex_l2fetch(src_data, src_row_size, src_stride, 2);
        hvx_copy_f16_f32_au(dst_data, src_data, ne0);

        dst_data += dst_stride;
        src_data += src_stride;
    }

    uint64_t t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "quantize-f32-f16: %u/%u : n-rows %u (%u:%u) row-size %u (%u) -> %u usec %u\n", ith, nth, nrows, ir_first,
        ir_last, src_row_size, src_stride, dst_stride, (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

// TODO just a plain copy that should be done via the DMA during the Op setup
static void quantize_f16_f16(const struct htp_tensor * src, uint8_t * restrict dst, uint32_t nth, uint32_t ith,
                              uint32_t nrows_per_thread, uint32_t dst_stride) {

    uint64_t t1 = HAP_perf_get_qtimer_count();

    const uint32_t ne0 = src->ne[0];
    const uint32_t ne1 = src->ne[1];
    const uint32_t ne2 = src->ne[2];
    const uint32_t ne3 = src->ne[3];

    const uint32_t nrows = ne1 * ne2 * ne3;                             // total n_rows

    const uint32_t ir_first = nrows_per_thread * ith;                   // first row
    const uint32_t ir_last  = MIN(ir_first + nrows_per_thread, nrows);  // last row

    const size_t src_row_size = ne0 * sizeof(float);
    const size_t src_stride   = src->nb[1];

    uint8_t * restrict src_data = (uint8_t *) src->data + (src_stride * ir_first);
    uint8_t * restrict dst_data = (uint8_t *) dst       + (dst_stride * ir_first);

    for (uint32_t i = ir_first; i < ir_last; ++i) {
        hex_l2fetch(src_data, src_row_size, src_stride, 2);
        hvx_copy_f16_au(dst_data, src_data, ne0);

        dst_data += dst_stride;
        src_data += src_stride;
    }

    uint64_t t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "quantize-f16-f16: %u/%u : n-rows %u (%u:%u) row-size %u (%u) -> %u usec %u\n", ith, nth, nrows, ir_first,
        ir_last, src_row_size, src_stride, dst_stride, (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void htp_quantize_f32_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;
    quantize_f32_q8x4x2(&octx->src1, octx->src1_spad.data, &octx->src0_spad, n, i, octx->src1_nrows_per_thread);
}

static void htp_quantize_f32_f16(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;
    quantize_f32_f16(&octx->src1, octx->src1_spad.data, n, i, octx->src1_nrows_per_thread, octx->src1_spad.stride);
}

static void htp_quantize_f16_f16(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;
    quantize_f16_f16(&octx->src1, octx->src1_spad.data, n, i, octx->src1_nrows_per_thread, octx->src1_spad.stride);
}

// ** matmul/matvec callbacks for worker_pool

static void htp_matvec_2d_q4x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "q4x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_q4x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_q4x4x2_q8x4x2_rx2;

    matvec_2d(&mt, octx, n, i);
}

static void htp_matmul_2d_q4x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "q4x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_q4x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_q4x4x2_q8x4x2_rx2;

    matmul_2d(&mt, octx, n, i);
}

static void htp_matvec_2d_q8x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "q8x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_q8x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_q8x4x2_q8x4x2_rx2;

    matvec_2d(&mt, octx, n, i);
}

static void htp_matmul_2d_q8x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "q8x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_q8x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_q8x4x2_q8x4x2_rx2;

    matmul_2d(&mt, octx, n, i);
}

static void htp_matvec_2d_mxfp4x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "mxfp4x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_mxfp4x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_mxfp4x4x2_q8x4x2_rx2;

    matvec_2d(&mt, octx, n, i);
}

static void htp_matmul_2d_mxfp4x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "mxfp4x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_mxfp4x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_mxfp4x4x2_q8x4x2_rx2;

    matmul_2d(&mt, octx, n, i);
}

static void htp_matvec_2d_f16_f16(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "f16-f16";
    mt.vec_dot     = vec_dot_f16_f16_aa;
    mt.vec_dot_rx2 = vec_dot_f16_f16_aa_rx2;

    matvec_2d(&mt, octx, n, i);
}

static void htp_matmul_2d_f16_f16(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "f16-f16";
    mt.vec_dot     = vec_dot_f16_f16_aa;
    mt.vec_dot_rx2 = vec_dot_f16_f16_aa_rx2;

    matmul_2d(&mt, octx, n, i);
}

static void htp_matmul_4d_f16_f32(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "f16-f32";
    mt.vec_dot     = vec_dot_f16_f32_uu;

    matmul_4d(&mt, octx, n, i);
}

static void htp_matmul_4d_f16_f16(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "f16-f16";
    mt.vec_dot     = vec_dot_f16_f16_uu;

    matmul_4d(&mt, octx, n, i);
}

// ** matmul-id callbacks for worker_pool

static void htp_matvec_id_q4x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "q4x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_q4x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_q4x4x2_q8x4x2_rx2;

    matvec_id(&mt, octx, n, i);
}

static void htp_matmul_id_q4x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "q4x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_q4x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_q4x4x2_q8x4x2_rx2;

    matmul_id(&mt, octx, n, i);
}

static void htp_matvec_id_q8x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "q8x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_q8x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_q8x4x2_q8x4x2_rx2;

    matvec_id(&mt, octx, n, i);
}

static void htp_matmul_id_q8x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "q8x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_q8x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_q8x4x2_q8x4x2_rx2;

    matmul_id(&mt, octx, n, i);
}

static void htp_matvec_id_mxfp4x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "mxfp4x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_mxfp4x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_mxfp4x4x2_q8x4x2_rx2;

    matvec_id(&mt, octx, n, i);
}

static void htp_matmul_id_mxfp4x4x2_q8x4x2(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = data;

    struct htp_matmul_type mt;
    mt.type        = "mxfp4x4x2-q8x4x2";
    mt.vec_dot     = vec_dot_mxfp4x4x2_q8x4x2;
    mt.vec_dot_rx2 = vec_dot_mxfp4x4x2_q8x4x2_rx2;

    matmul_id(&mt, octx, n, i);
}

// ** main matmul entry point

static inline bool htp_is_permuted(const struct htp_tensor * t) {
    return t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

int op_matmul(struct htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    const char * op_type;

    const uint32_t src0_nrows = ne01 * ne02 * ne03;
    const uint32_t src1_nrows = ne11 * ne12 * ne13;

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;
    size_t       src1_row_size = nb11;

    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);
    size_t       src1_row_size_padded;

    worker_callback_t quant_job_func;
    worker_callback_t matmul_job_func;

    bool need_quant = !(octx->flags & HTP_OPFLAGS_SKIP_QUANTIZE);

    switch (src0->type) {
        case HTP_TYPE_Q4_0:
            op_type        = "q4x4x2-f32";
            quant_job_func = htp_quantize_f32_q8x4x2;
            if (src1_nrows > 1) {
                matmul_job_func = htp_matmul_2d_q4x4x2_q8x4x2;
            } else {
                matmul_job_func = htp_matvec_2d_q4x4x2_q8x4x2;
            }

            src1_row_size = q8x4x2_row_size(ne10);  // row size post quantization

            // Entire src1 tensor is placed into the VTCM
            // For other tensors we allocate N rows per thread, padded to HVX vector size

            octx->dst_spad.size_per_thread  = hex_round_up(MM_SPAD_DST_NROWS * dst_row_size, 256);
            octx->src0_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
            octx->src1_spad.size_per_thread = hex_round_up(src1_row_size * src1_nrows, 256);

            // src0 spad is also used in dynamic quantizer to store padded src1 rows
            src1_row_size_padded = hex_round_up(src1_row_size, QK_Q8_0x4x2 * sizeof(float));
            if (octx->src0_spad.size_per_thread < src1_row_size_padded) {
                octx->src0_spad.size_per_thread = src1_row_size_padded;
            }

            octx->src1_spad.size = octx->src1_spad.size_per_thread;
            octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;
            break;

        case HTP_TYPE_Q8_0:
            op_type        = "q8x4x2-f32";
            quant_job_func = htp_quantize_f32_q8x4x2;
            if (src1_nrows > 1) {
                matmul_job_func = htp_matmul_2d_q8x4x2_q8x4x2;
            } else {
                matmul_job_func = htp_matvec_2d_q8x4x2_q8x4x2;
            }

            src1_row_size = q8x4x2_row_size(ne10);  // row size post quantization

            // Entire src1 tensor is placed into the VTCM
            // For other tensors we allocate N rows per thread, padded to HVX vector size

            octx->dst_spad.size_per_thread  = hex_round_up(MM_SPAD_DST_NROWS * dst_row_size, 256);
            octx->src0_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
            octx->src1_spad.size_per_thread = hex_round_up(src1_row_size * src1_nrows, 256);

            // src0 spad is also used in dynamic quantizer to store padded src1 rows
            src1_row_size_padded = hex_round_up(src1_row_size, QK_Q8_0x4x2 * sizeof(float));
            if (octx->src0_spad.size_per_thread < src1_row_size_padded) {
                octx->src0_spad.size_per_thread = src1_row_size_padded;
            }

            octx->src1_spad.size = octx->src1_spad.size_per_thread;
            octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;
            break;

        case HTP_TYPE_MXFP4:
            op_type        = "mxfp4x4x2-f32";
            quant_job_func = htp_quantize_f32_q8x4x2;
            if (src1_nrows > 1) {
                matmul_job_func = htp_matmul_2d_mxfp4x4x2_q8x4x2;
            } else {
                matmul_job_func = htp_matvec_2d_mxfp4x4x2_q8x4x2;
            }

            src1_row_size = q8x4x2_row_size(ne10);  // row size post quantization

            // Entire src1 tensor is placed into the VTCM
            // For other tensors we allocate N rows per thread, padded to HVX vector size

            octx->dst_spad.size_per_thread  = hex_round_up(MM_SPAD_DST_NROWS * dst_row_size, 256);
            octx->src0_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
            octx->src1_spad.size_per_thread = hex_round_up(src1_row_size * src1_nrows, 256);

            // src0 spad is also used in dynamic quantizer to store padded src1 rows
            src1_row_size_padded = hex_round_up(src1_row_size, QK_Q8_0x4x2 * sizeof(float));
            if (octx->src0_spad.size_per_thread < src1_row_size_padded) {
                octx->src0_spad.size_per_thread = src1_row_size_padded;
            }

            octx->src1_spad.size = octx->src1_spad.size_per_thread;
            octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;
            break;

        case HTP_TYPE_F16:
            {
                // Try optimized f16-f16 path first (src1 in VTCM)
                const size_t f16_src1_row_size  = hex_round_up(ne10 * 2, 128);
                const size_t f16_src1_spad_size = hex_round_up(f16_src1_row_size * src1_nrows, 256);
                const size_t f16_src0_spad_size = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256) * octx->n_threads;
                const size_t f16_dst_spad_size  = hex_round_up(MM_SPAD_DST_NROWS  * dst_row_size, 256) * octx->n_threads;

                const size_t f16_total_size = f16_src1_spad_size + f16_src0_spad_size + f16_dst_spad_size;

                // Default matmul implementation does not support multi-batch src0 (N-vs-N broadcasting).
                // It only supports 1-vs-N broadcasting (src0 is 2D) or standard 2D matmul.
                const bool is_batched  = (ne02 > 1) || (ne03 > 1);
                const bool is_permuted = htp_is_permuted(&octx->src0) || htp_is_permuted(&octx->src1);

                if (!is_batched && !is_permuted && f16_total_size <= octx->ctx->vtcm_size) {
                    // Optimized path
                    op_type        = "f16-f16";
                    quant_job_func = (src1->type == HTP_TYPE_F32) ? htp_quantize_f32_f16 : htp_quantize_f16_f16;
                    if (src1_nrows > 1) {
                        matmul_job_func = htp_matmul_2d_f16_f16;
                    } else {
                        matmul_job_func = htp_matvec_2d_f16_f16;
                    }

                    src1_row_size = f16_src1_row_size; // row size post quantization

                    octx->dst_spad.size_per_thread  = hex_round_up(MM_SPAD_DST_NROWS * dst_row_size, 256);
                    octx->src0_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
                    octx->src1_spad.size_per_thread = hex_round_up(src1_row_size * src1_nrows, 256);

                    octx->src1_spad.size = octx->src1_spad.size_per_thread;
                    octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
                    octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;
                } else {
                    // Fallback to f16/f32 (DDR) if src1 doesn't fit in VTCM or broadcasting is required
                    quant_job_func  = NULL;
                    if (src1->type == HTP_TYPE_F32) {
                        op_type         = "f16-f32";
                        matmul_job_func = htp_matmul_4d_f16_f32;
                    } else {
                        op_type         = "f16-f16";
                        matmul_job_func = htp_matmul_4d_f16_f16;
                    }

                    src1_row_size = nb11; // original row size in DDR

                    octx->dst_spad.size_per_thread  = hex_round_up(MM_SPAD_DST_NROWS * dst_row_size, 256);
                    octx->src0_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size, 256);
                    octx->src1_spad.size_per_thread = hex_round_up(MM_SPAD_SRC1_NROWS * src1_row_size, 256);

                    octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
                    octx->src1_spad.size = octx->src1_spad.size_per_thread * octx->n_threads;
                    octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;

                    // Init fastdiv for matmul_4d (supports broadcasting)
                    octx->mm_div_ne12_ne1 = init_fastdiv_values(src1->ne[2] * dst->ne[1]);
                    octx->mm_div_ne1      = init_fastdiv_values(dst->ne[1]);
                    octx->mm_div_r2       = init_fastdiv_values(src1->ne[2] / src0->ne[2]);
                    octx->mm_div_r3       = init_fastdiv_values(src1->ne[3] / src0->ne[3]);

                    need_quant = false;
                }
            }
            break;

        default:
            return HTP_STATUS_NO_SUPPORT;
    }

    // VTCM scratchpads for all tensors
    size_t spad_size = octx->src1_spad.size + octx->src0_spad.size + octx->dst_spad.size;

    FARF(HIGH, "matmul-%s : src0-spad-size %u src1-spad-size %u dst-spad-size %u (%zu)\n", op_type,
         octx->src0_spad.size, octx->src1_spad.size, octx->dst_spad.size, spad_size);

    FARF(HIGH, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", op_type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->data, src1->data, dst->data);

    // Make sure the reserved vtcm size is sufficient
    if (octx->ctx->vtcm_size < spad_size) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type,
             octx->ctx->vtcm_size, spad_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;

    octx->src0_nrows_per_thread = (src0_nrows + octx->n_threads - 1) / octx->n_threads;
    octx->src0_nrows_per_thread += (octx->src0_nrows_per_thread & 1);  // round up to even

    octx->src0_spad.stride = src0_row_size_padded;
    octx->src1_spad.stride = src1_row_size;

    if (need_quant) {
        // Run quant jobs
        const uint32_t n_quant_jobs = MIN(src1_nrows, octx->n_threads);
        octx->src1_nrows_per_thread = (src1_nrows + n_quant_jobs - 1) / n_quant_jobs;
        worker_pool_run_func(octx->ctx->worker_pool, quant_job_func, octx, n_quant_jobs);
    }

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        // Run matmul jobs
        const uint32_t n_matmul_jobs = octx->n_threads;
        worker_pool_run_func(octx->ctx->worker_pool, matmul_job_func, octx, n_matmul_jobs);
    }

    return HTP_STATUS_OK;
}

// ** main matmul-id entry point

int op_matmul_id(struct htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    struct htp_tensor * restrict ids = &octx->src2;

    const char * op_type;

    worker_callback_t quant_job_func;
    worker_callback_t matmul_id_job_func;

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;

    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    const uint32_t src0_nrows = ne01;  // per expert
    const uint32_t src1_nrows = ne11 * ne12 * ne13;

    size_t src1_row_size;
    size_t src1_row_size_padded;

    // row groups
    const int n_ids = ids->ne[0];  // n_expert_used
    const int n_as  = ne02;        // n_expert

    size_t matrix_row_counts_size = n_as * sizeof(uint32_t);
    size_t matrix_row_map_size    = n_as * ids->ne[0] * ids->ne[1] * sizeof(struct mmid_row_mapping);

    switch (src0->type) {
        case HTP_TYPE_Q4_0:
            op_type        = "q4x2x2-f32";
            quant_job_func = htp_quantize_f32_q8x4x2;
            src1_row_size  = q8x4x2_row_size(ne10);  // row size post quantization
            if (src1_nrows > 1) {
                matmul_id_job_func = htp_matmul_id_q4x4x2_q8x4x2;
            } else {
                matmul_id_job_func = htp_matvec_id_q4x4x2_q8x4x2;
            }

            // Entire src1 tensor is placed into the VTCM
            // For other tensors we allocate N rows per thread, padded to HVX vector size
            octx->dst_spad.size_per_thread  = hex_round_up(MM_SPAD_DST_NROWS * dst_row_size, 256);
            octx->src0_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
            octx->src1_spad.size_per_thread = hex_round_up(src1_row_size * src1_nrows, 256);
            octx->src2_spad.size_per_thread = hex_round_up(matrix_row_counts_size + matrix_row_map_size, 256);

            // src0 spad is also used in dynamic quantizer to store padded src1 rows
            src1_row_size_padded = hex_round_up(src1_row_size, QK_Q8_0x4x2 * sizeof(float));
            if (octx->src0_spad.size_per_thread < src1_row_size_padded) {
                octx->src0_spad.size_per_thread = src1_row_size_padded;
            }

            octx->src2_spad.size = octx->src2_spad.size_per_thread;
            octx->src1_spad.size = octx->src1_spad.size_per_thread;
            octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;
            break;

        case HTP_TYPE_Q8_0:
            op_type        = "q8x2x2-f32";
            quant_job_func = htp_quantize_f32_q8x4x2;
            src1_row_size  = q8x4x2_row_size(ne10);  // row size post quantization
            if (src1_nrows > 1) {
                matmul_id_job_func = htp_matmul_id_q8x4x2_q8x4x2;
            } else {
                matmul_id_job_func = htp_matvec_id_q8x4x2_q8x4x2;
            }

            // Entire src1 tensor is placed into the VTCM
            // For other tensors we allocate N rows per thread, padded to HVX vector size
            octx->dst_spad.size_per_thread  = hex_round_up(MM_SPAD_DST_NROWS * dst_row_size, 256);
            octx->src0_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
            octx->src1_spad.size_per_thread = hex_round_up(src1_row_size * src1_nrows, 256);
            octx->src2_spad.size_per_thread = hex_round_up(matrix_row_counts_size + matrix_row_map_size, 256);

            // src0 spad is also used in dynamic quantizer to store padded src1 rows
            src1_row_size_padded = hex_round_up(src1_row_size, QK_Q8_0x4x2 * sizeof(float));
            if (octx->src0_spad.size_per_thread < src1_row_size_padded) {
                octx->src0_spad.size_per_thread = src1_row_size_padded;
            }

            octx->src2_spad.size = octx->src2_spad.size_per_thread;
            octx->src1_spad.size = octx->src1_spad.size_per_thread;
            octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;
            break;

        case HTP_TYPE_MXFP4:
            op_type        = "mxfp4x2x2-f32";
            quant_job_func = htp_quantize_f32_q8x4x2;
            src1_row_size  = q8x4x2_row_size(ne10);  // row size post quantization
            if (src1_nrows > 1) {
                matmul_id_job_func = htp_matmul_id_mxfp4x4x2_q8x4x2;
            } else {
                matmul_id_job_func = htp_matvec_id_mxfp4x4x2_q8x4x2;
            }

            // Entire src1 tensor is placed into the VTCM
            // For other tensors we allocate N rows per thread, padded to HVX vector size
            octx->dst_spad.size_per_thread  = hex_round_up(MM_SPAD_DST_NROWS * dst_row_size, 256);
            octx->src0_spad.size_per_thread = hex_round_up(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
            octx->src1_spad.size_per_thread = hex_round_up(src1_row_size * src1_nrows, 256);
            octx->src2_spad.size_per_thread = hex_round_up(matrix_row_counts_size + matrix_row_map_size, 256);

            // src0 spad is also used in dynamic quantizer to store padded src1 rows
            src1_row_size_padded = hex_round_up(src1_row_size, QK_Q8_0x4x2 * sizeof(float));
            if (octx->src0_spad.size_per_thread < src1_row_size_padded) {
                octx->src0_spad.size_per_thread = src1_row_size_padded;
            }

            octx->src2_spad.size = octx->src2_spad.size_per_thread;
            octx->src1_spad.size = octx->src1_spad.size_per_thread;
            octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
            octx->dst_spad.size  = octx->dst_spad.size_per_thread * octx->n_threads;
            break;

        default:
            return HTP_STATUS_NO_SUPPORT;
    }

    size_t spad_size = octx->src2_spad.size + octx->src1_spad.size + octx->src0_spad.size + octx->dst_spad.size;

    FARF(HIGH, "matmul-id-%s : src0-spad-size %u src1-spad-size %u src2-spad-size %u dst-spad-size %u (%zu)\n", op_type,
         octx->src0_spad.size, octx->src1_spad.size, octx->src2_spad.size, octx->dst_spad.size, spad_size);

    FARF(HIGH, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->data,
         src1->data, dst->data);

    // Make sure the reserved vtcm size is sufficient
    if (octx->ctx->vtcm_size < spad_size) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type,
             octx->ctx->vtcm_size, spad_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->src2_spad.data = octx->src1_spad.data + octx->src1_spad.size;
    octx->dst_spad.data  = octx->src2_spad.data + octx->src2_spad.size;

    octx->src0_nrows_per_thread = (src0_nrows + octx->n_threads - 1) / octx->n_threads;
    octx->src0_nrows_per_thread += (octx->src0_nrows_per_thread & 1);  // round up to even

    if (src1_nrows > 1) {
        // initialize matrix_row_counts and map
        uint32_t *                matrix_row_counts = (uint32_t *) octx->src2_spad.data + 0;
        struct mmid_row_mapping * matrix_rows       = (void *) octx->src2_spad.data + matrix_row_counts_size;

        memset(matrix_row_counts, 0, n_as * sizeof(uint32_t));

        // group rows by src0 matrix
        for (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {  // token idx
            for (uint32_t id = 0; id < n_ids; ++id) {         // expert idx
                const uint32_t i02 =
                    *(const uint32_t *) ((const uint8_t *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) { id, iid1 };
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // Setup worker pool callbacks
    if (!(octx->flags & HTP_OPFLAGS_SKIP_QUANTIZE)) {
        // Run quant jobs
        const uint32_t n_quant_jobs = MIN(src1_nrows, octx->n_threads);
        octx->src1_nrows_per_thread = (src1_nrows + n_quant_jobs - 1) / n_quant_jobs;
        worker_pool_run_func(octx->ctx->worker_pool, quant_job_func, octx, n_quant_jobs);
    }

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        // Run matmul-id jobs
        const uint32_t n_matmul_jobs = octx->n_threads;
        worker_pool_run_func(octx->ctx->worker_pool, matmul_id_job_func, octx, n_matmul_jobs);
    }

    return HTP_STATUS_OK;
}
