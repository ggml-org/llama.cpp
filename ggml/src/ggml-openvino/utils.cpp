#include "utils.h"
#include "ggml-backend-impl.h"
#include <openvino/frontend/manager.hpp>
#include <openvino/openvino.hpp>

using ov::frontend::tensorflow::ggml::GgmlOvGraphIterator;

std::shared_ptr<GgmlOvGraphIterator> get_ggml_graph_iterator(struct ggml_cgraph * cgraph) {
    return std::make_shared<GgmlOvGraphIterator>(cgraph);
}

std::map<std::string, ov::Tensor> get_ggml_graph_input_tensors(std::shared_ptr<GgmlOvGraphIterator> ggml_graph_iterator) {
    std::map<std::string, ov::Tensor> input_tensors;
    auto input_names = ggml_graph_iterator->get_input_names();
    ggml_graph_iterator->reset();
    for (; !ggml_graph_iterator->is_end(); ggml_graph_iterator->next()) {
        auto decoder = std::dynamic_pointer_cast<GgmlOvDecoder>(ggml_graph_iterator->get_decoder()); 
        for (size_t inp = 0; inp < decoder->get_input_size(); ++inp) {
            if (std::find(input_names.begin(), input_names.end(), decoder->get_input_name(inp)) != input_names.end()) {
                auto input_data = decoder->get_input_ggml_tensor(inp)->data;
                #ifdef GGML_OPENVINO_DEBUG
                    printf("Subgraph input %d: %g\n", inp, *(double*)(input_data));
                #endif
                ov::Tensor input_tensor = ov::Tensor(decoder->get_input_type(inp), decoder->get_input_shape(inp).to_shape(), input_data);
                input_tensors[decoder->get_input_name(inp)] = input_tensor;
            }
        }
    }
    return input_tensors;
}

std::map<std::string, ov::Tensor> get_ggml_graph_output_tensors(std::shared_ptr<GgmlOvGraphIterator> ggml_graph_iterator) {
    std::map<std::string, ov::Tensor> output_tensors;
    auto output_names = ggml_graph_iterator->get_output_names();
    ggml_graph_iterator->reset();
    for (; !ggml_graph_iterator->is_end(); ggml_graph_iterator->next()) {
        auto decoder = std::dynamic_pointer_cast<GgmlOvDecoder>(ggml_graph_iterator->get_decoder()); 
        for (size_t inp = 0; inp < decoder->get_output_size(); ++inp) {
            if (std::find(output_names.begin(), output_names.end(), decoder->get_output_name(inp)) != output_names.end()) {
                auto output_data = decoder->get_output_ggml_tensor(inp)->data;
                #ifdef GGML_OPENVINO_DEBUG
                    printf("Output %d: %g\n", inp, *(double*)(output_data));
                #endif
                ov::Tensor output_tensor = ov::Tensor(decoder->get_output_type(inp), decoder->get_output_shape(inp).to_shape(), output_data);
                output_tensors[decoder->get_output_name(inp)] = output_tensor;
            }
        }
    }
    return output_tensors;
}


static ov::frontend::FrontEnd::Ptr get_ggml_frontend() {
    ov::frontend::FrontEnd::Ptr front_end = nullptr;
    auto fem = ov::frontend::FrontEndManager();
    std::string fe_so_path;
#ifdef GGML_OV_FRONTEND
    fe_so_path = GGML_OV_FRONTEND;
#endif
    fem.register_front_end("ggml", fe_so_path);
    front_end = fem.load_by_framework("ggml");
    return front_end;
}

enum ggml_status openvino_frontend_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ov::Core core;
    auto devices = core.get_available_devices();
    // Get GGML Frontend 
    auto front_end = get_ggml_frontend();
    if (!front_end) {
        GGML_LOG_ERROR("GGML FrontEnd is not initialized \n");
        return GGML_STATUS_FAILED;
    } else {
        #ifdef GGML_OPENVINO_DEBUG
            GGML_LOG_INFO("GGML FrontEnd is initialized \n");
        #endif
    }

    auto ggml_graph_iterator = get_ggml_graph_iterator(cgraph);
    std::shared_ptr<ov::frontend::tensorflow::GraphIterator> graph_iterator = ggml_graph_iterator;
    
    // Load GraphIterator -> InputModel
    ov::frontend::InputModel::Ptr input_model = front_end->load(graph_iterator);
    if (!input_model) {
        GGML_LOG_ERROR("Input Model is not loaded \n");
        return GGML_STATUS_FAILED;
    } else {
        #ifdef GGML_OPENVINO_DEBUG
            GGML_LOG_INFO("Input Model loaded \n");
        #endif
    }

    // Convert InputModel -> ov::Model 
    std::shared_ptr<ov::Model> model = front_end->convert(input_model);
    if (!model) {
        GGML_LOG_ERROR("Model is not converted \n");
    } else {
        #ifdef GGML_OPENVINO_DEBUG
            GGML_LOG_INFO("Model converted \n");
        #endif
    }


    //  Loading a model to the device
    ov::CompiledModel compiled_model = core.compile_model(model);

    // Create infer request
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // Get input tensor
    auto input_names = ggml_graph_iterator->get_input_names();
    auto input_tensors = get_ggml_graph_input_tensors(ggml_graph_iterator);

    // Set input tensor
    for (size_t i = 0; i < input_names.size(); i++) {
        infer_request.set_input_tensor(i, input_tensors[input_names[i]]);
    }

    // Set output tensor

    auto output_names = ggml_graph_iterator->get_output_names();
    auto output_tensors = get_ggml_graph_output_tensors(ggml_graph_iterator);
    for (size_t i = 0; i < output_names.size(); i++) {
        infer_request.set_output_tensor(i, output_tensors[output_names[i]]);
    }

    infer_request.infer();
    
    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}
