#ifndef HTP_FUSION_H
#define HTP_FUSION_H

#include "htp-common.h"

enum htp_fusion_flags {
    HTP_FUSE_ALLREDUCE_ADD   = (1 << 1),  // 2
    HTP_FUSE_RMS_NORM_MUL    = (1 << 2),  // 4
    HTP_FUSE_MUL_MAT_ADD     = (1 << 3),  // 8
    HTP_FUSE_MUL_MAT_NX      = (1 << 4),  // 16
    HTP_FUSE_MUL_MAT_ID_NX   = (1 << 5),  // 32
    HTP_FUSE_GDN_CPY         = (1 << 6),  // 64
    HTP_FUSE_REPEAT_GET_ROWS = (1 << 7),  // 128
    HTP_FUSE_UNARY_MUL_GLU   = (1 << 8),  // 256
    HTP_FUSE_MUL_MUL_SCALAR  = (1 << 9),  // 512
    HTP_FUSE_BINARY_NX       = (1 << 10), // 1024
    HTP_FUSE_RMS_NORM_SCALE  = (1 << 11), // 2048
    HTP_FUSE_SSM_CONV_SILU   = (1 << 12), // 4096
    HTP_FUSE_MUL_MAT_GLU     = (1 << 13), // 8192
};

static inline bool htp_fusion_enabled(int flag) {
    if (opt_opfusion <= 0) return false;
    if (opt_opfusion == 1) return true; // 1 enables all
    return (opt_opfusion & flag) != 0;
}

bool htp_fusion_can_stack(const struct ggml_tensor * a);
bool htp_fusion_can_stack(const struct ggml_tensor * a, const struct ggml_tensor * b);
int  htp_fusion_max(const struct ggml_cgraph * graph, int idx);
void htp_fusion_tag_graph(struct ggml_cgraph * graph);
bool htp_fusion_try_fuse(const struct ggml_hexagon_session * sess, struct htp_opnode & prev, const struct htp_opnode & cur);

#endif // HTP_FUSION_H
