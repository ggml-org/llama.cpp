#include "ggml-ov-frontend-utils.h"
#include "ggml-backend-impl.h"
#include <openvino/frontend/manager.hpp>

using ov::frontend::tensorflow::ggml::GgmlOvGraphIterator;

std::shared_ptr<GgmlOvGraphIterator> get_ggml_graph_iterator(struct ggml_cgraph * cgraph) {
    return std::make_shared<GgmlOvGraphIterator>(cgraph);
}

static ov::frontend::FrontEnd::Ptr get_ggml_frontend() {
    ov::frontend::FrontEnd::Ptr front_end = nullptr;
    auto fem = ov::frontend::FrontEndManager();
    std::string fe_so_path = "/home/yumeng/Code/ov-ggml-frontend/openvino/bin/intel64/Release/libopenvino_ggml_frontend.so";
    fem.register_front_end("ggml", fe_so_path);
    front_end = fem.load_by_framework("ggml");
    return front_end;
}

enum ggml_status openvino_frontend_compute (ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    // Get GGML Frontend 
    auto front_end = get_ggml_frontend();
    if (!front_end) {
        GGML_LOG_ERROR("GGML FrontEnd is not initialized \n");
        return GGML_STATUS_FAILED;
    } else {
        GGML_LOG_ERROR("GGML FrontEnd is initialized \n");
    }

    auto ggml_graph_iterator = get_ggml_graph_iterator(cgraph);
    std::shared_ptr<ov::frontend::tensorflow::GraphIterator> graph_iterator = ggml_graph_iterator;
    GGML_LOG_ERROR("Decoder count in current GraphIterator: %s\n", std::to_string(graph_iterator->size()).c_str());
    
    // Load GraphIterator -> InputModel
    ov::frontend::InputModel::Ptr input_model = front_end->load(graph_iterator);
    if (!input_model) {
        GGML_LOG_ERROR("\nInput Model is not loaded \n");
        return GGML_STATUS_FAILED;
    } else {
        GGML_LOG_ERROR("\nInput Model loaded \n");
    }

    // TODO: Convert InputModel -> ov::Model 
    // std::shared_ptr<ov::Model> model = front_end->convert(input_model);
    // if (!model) {
    //     GGML_LOG_ERROR("Model is not converted");
    // }

    // TODO: Compute

    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}
