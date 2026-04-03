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
    uint32_t  stride;
    uint32_t  size;
    uint32_t  size_per_thread;
};

#define HTP_OP_MAX_SPADS (HTP_OP_MAX_INPUTS+1)
#define HTP_OP_DST_SPAD  (HTP_OP_MAX_INPUTS)

struct htp_ops_context {
    struct htp_context * ctx;

    enum htp_op_code    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];

    const struct htp_tensor * src[HTP_OP_MAX_INPUTS];
    const struct htp_tensor * dst;

    // TODO convert these to an array
    struct htp_spad src0_spad;
    struct htp_spad src1_spad;
    struct htp_spad src2_spad;
    struct htp_spad src3_spad;
    struct htp_spad dst_spad;

    uint32_t n_threads;
    uint32_t flags;
};

int op_matmul(struct htp_ops_context * octx);
int op_matmul_id(struct htp_ops_context * octx);
int op_binary(struct htp_ops_context * octx);
int op_unary(struct htp_ops_context * octx);
int op_sum_rows(struct htp_ops_context * octx);
int op_activations(struct htp_ops_context * octx);
int op_softmax(struct htp_ops_context * octx);
int op_add_id(struct htp_ops_context * octx);
int op_rope(struct htp_ops_context * octx);
int op_flash_attn_ext(struct htp_ops_context * octx);
int op_set_rows(struct htp_ops_context * octx);
int op_get_rows(struct htp_ops_context * octx);
int op_cpy(struct htp_ops_context * octx);
int op_repeat(struct htp_ops_context * octx);
int op_argsort(struct htp_ops_context * octx);
int op_ssm_conv(struct htp_ops_context * octx);
int op_cumsum(struct htp_ops_context * octx);

#endif /* HTP_OPS_H */
