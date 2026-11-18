#ifndef HTP_LOWER_H
#define HTP_LOWER_H

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_cgraph;
struct ggml_hexagon_session;
struct htp_opnode;

bool        htp_rows_stride(const int64_t * ne, const size_t * nb, size_t * stride);
bool        htp_matmul_can_collapse(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
bool        htp_matmul_add_can_collapse(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * src2, const struct ggml_tensor * dst);
ggml_tensor htp_tensor_collapse_rows(const struct ggml_tensor * t);

bool htp_dsv4_hc_pre_can_collapse(const struct ggml_tensor * x, const struct ggml_tensor * weights, const struct ggml_tensor * dst);
void htp_dsv4_hc_pre_collapse_tensors(const struct ggml_tensor * x, const struct ggml_tensor * weights, const struct ggml_tensor * dst,
                                      struct ggml_tensor * x_c, struct ggml_tensor * w_c, struct ggml_tensor * d_c);

bool htp_dsv4_hc_post_can_collapse(const struct ggml_tensor * x, const struct ggml_tensor * residual, const struct ggml_tensor * post, const struct ggml_tensor * comb, const struct ggml_tensor * dst);
void htp_dsv4_hc_post_collapse_tensors(const struct ggml_tensor * x, const struct ggml_tensor * residual, const struct ggml_tensor * post, const struct ggml_tensor * comb, const struct ggml_tensor * dst,
                                       struct ggml_tensor * x_c, struct ggml_tensor * r_c, struct ggml_tensor * p_c, struct ggml_tensor * c_c, struct ggml_tensor * d_c);

bool htp_lightning_indexer_can_collapse(const struct ggml_tensor * q, const struct ggml_tensor * k, const struct ggml_tensor * w, const struct ggml_tensor * m, const struct ggml_tensor * dst);
void htp_lightning_indexer_collapse_tensors(const struct ggml_tensor * q, const struct ggml_tensor * k, const struct ggml_tensor * w, const struct ggml_tensor * m, const struct ggml_tensor * dst,
                                            struct ggml_tensor * q_c, struct ggml_tensor * k_c, struct ggml_tensor * w_c, struct ggml_tensor * m_c, struct ggml_tensor * dst_c);

void htp_lower_opnode(const struct ggml_hexagon_session * sess, htp_opnode & node, struct ggml_tensor * scratch = nullptr);
void htp_lower_graph(const struct ggml_hexagon_session * sess, const struct ggml_cgraph * graph, std::vector<htp_opnode> & nodes);
void htp_lower_reset_graph(const struct ggml_cgraph * graph);
void htp_unpack_tensor_lowered_f16(void * dst, const struct ggml_tensor * t, size_t offset, size_t size);

#endif // HTP_LOWER_H
