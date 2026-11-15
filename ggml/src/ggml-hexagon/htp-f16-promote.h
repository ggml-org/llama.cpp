#ifndef HTP_F16_PROMOTE_H
#define HTP_F16_PROMOTE_H

#include <vector>

struct ggml_cgraph;
struct ggml_hexagon_session;
struct htp_opnode;

void htp_f16_promote(const ggml_hexagon_session * sess, const ggml_cgraph * graph, std::vector<htp_opnode> & nodes);

#endif // HTP_F16_PROMOTE_H
