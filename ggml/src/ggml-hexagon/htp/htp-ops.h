#ifndef HTP_OPS_H
#define HTP_OPS_H

#include "htp-ctx.h"
#include "htp-msg.h"
#include "worker-pool.h"

#include <assert.h>
#include <stdint.h>

#include <hex-fastdiv.h>

// ggml-common.h must be included prior to this header

struct htp_spad {
    uint8_t * data;
    size_t    stride;
    size_t    size;
    size_t    size_per_thread;
};

struct htp_ops_context {
    struct htp_context * ctx;

    enum htp_op op;
    int32_t     op_params[HTP_MAX_OP_PARAMS / sizeof(int32_t)];

    struct htp_tensor src0;
    struct htp_tensor src1;
    struct htp_tensor src2;
    struct htp_tensor src3;
    struct htp_tensor src4;
    struct htp_tensor dst;

    struct htp_spad src0_spad;
    struct htp_spad src1_spad;
    struct htp_spad src2_spad;
    struct htp_spad src3_spad;
    struct htp_spad dst_spad;

    worker_pool_context_t * wpool;      // worker pool
    uint32_t                n_threads;  // num threads

    uint32_t src0_nrows_per_thread;
    uint32_t src1_nrows_per_thread;

    struct fastdiv_values src0_div1;  // fastdiv values for ne1
    struct fastdiv_values src0_div2;  // fastdiv values for ne2
    struct fastdiv_values src0_div3;  // fastdiv values for ne3
    struct fastdiv_values src0_div21; // fastdiv values for ne2 * ne1

    struct fastdiv_values src1_div1;  // fastdiv values for ne1
    struct fastdiv_values src1_div2;  // fastdiv values for ne2
    struct fastdiv_values src1_div3;  // fastdiv values for ne3
    struct fastdiv_values src1_div21; // fastdiv values for ne2 * ne1

    struct fastdiv_values src3_div1;  // fastdiv values for ne1
    struct fastdiv_values src3_div2;  // fastdiv values for ne2
    struct fastdiv_values src3_div3;  // fastdiv values for ne3
    struct fastdiv_values src3_div21; // fastdiv values for ne2 * ne1

    struct fastdiv_values broadcast_rk2;
    struct fastdiv_values broadcast_rk3;
    struct fastdiv_values broadcast_rv2;
    struct fastdiv_values broadcast_rv3;

    struct fastdiv_values mm_div_ne12_ne1; // fastdiv values for ne12 * ne1
    struct fastdiv_values mm_div_ne1;      // fastdiv values for ne1
    struct fastdiv_values mm_div_r2;       // fastdiv values for ne12 / ne02
    struct fastdiv_values mm_div_r3;       // fastdiv values for ne13 / ne03

    struct fastdiv_values set_rows_div_ne12; // fastdiv values for ne12
    struct fastdiv_values set_rows_div_ne11; // fastdiv values for ne11

    struct fastdiv_values get_rows_div_ne10;      // fastdiv values for ne10
    struct fastdiv_values get_rows_div_ne10_ne11; // fastdiv values for ne10 * ne11

    struct fastdiv_values cpy_div_ne01; // fastdiv values for ne01
    struct fastdiv_values cpy_div_ne02; // fastdiv values for ne02
    struct fastdiv_values cpy_div_ne03; // fastdiv values for ne03

    struct fastdiv_values cpy_rshp_div_n0;       // fastdiv values for ne00
    struct fastdiv_values cpy_rshp_div_n1n0;     // fastdiv values for ne00*ne01
    struct fastdiv_values cpy_rshp_div_n2n1n0;   // fastdiv values for ne00*ne01*ne02

    uint32_t flags;
};

int op_matmul(struct htp_ops_context * octx);
int op_matmul_id(struct htp_ops_context * octx);
int op_binary(struct htp_ops_context * octx);
int op_unary(struct htp_ops_context * octx);
int op_activations(struct htp_ops_context * octx);
int op_softmax(struct htp_ops_context * octx);
int op_add_id(struct htp_ops_context * octx);
int op_rope(struct htp_ops_context * octx);
int op_flash_attn_ext(struct htp_ops_context * octx);
int op_set_rows(struct htp_ops_context * octx);
int op_get_rows(struct htp_ops_context * octx);
int op_cpy(struct htp_ops_context * octx);

#endif /* HTP_OPS_H */
