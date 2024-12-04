#include "ggml-graph-iterator.h"
#include "ggml-backend-impl.h"

std::shared_ptr<ov::frontend::tensorflow::ggml::GgmlOvGraphIterator> get_ggml_graph_iterator(struct ggml_cgraph * cgraph);

enum ggml_status openvino_frontend_compute (ggml_backend_t backend, struct ggml_cgraph * cgraph);
