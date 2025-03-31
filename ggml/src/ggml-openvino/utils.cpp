#include "utils.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include <openvino/frontend/manager.hpp>
#include <openvino/openvino.hpp>
#include <fstream>

using ov::frontend::ggml::GgmlDecoder;

std::shared_ptr<GgmlOvDecoder> get_ggml_decoder(struct ggml_cgraph * cgraph, const int32_t start_index, const int32_t end_index) {
    return std::make_shared<GgmlOvDecoder>(nullptr, cgraph, start_index, end_index);
}

std::vector<std::pair<std::string, ov::Tensor>> get_ggml_graph_input_tensors(std::shared_ptr<GgmlOvDecoder> ggml_decoder, bool flag) {
    std::vector<std::pair<std::string, ov::Tensor>> input_tensors;
    auto input_names = ggml_decoder->get_input_names();
    size_t op_iter = 0;
    for (size_t inp = 0; inp < input_names.size(); ++inp) {
        auto name = input_names[inp];
        std::string op_node_name = ggml_decoder->get_op_node_name(name, op_iter++);
        // auto node_op_name = ggml_decoder->get_node_op_name(name);
        auto input_data = ggml_decoder->get_input_ggml_tensor(name)->data;
        auto check_if_contiguous = ggml_is_contiguous(ggml_decoder->get_input_ggml_tensor(name));
        #ifdef GGML_OPENVINO_DEBUG
            printf("Subgraph input %d: %g\n", inp, *(double*)(input_data));
        #endif
        ov::Tensor input_tensor;
        ov::Shape input_shape = ggml_decoder->get_input_shape(name).to_shape();

        ov::element::Type input_type = ggml_decoder->get_input_type(name);
        size_t element_size = input_type.size();
        std::vector<size_t> input_stride = ggml_decoder->get_input_stride(name);
        if (op_node_name == "CONT" && input_shape[0] == 1 // Except for the kqv_merge node
            && (input_shape[1] != 1 && flag || input_shape[2]*element_size!=input_stride[1])
        ) {
            const size_t num_rows    = static_cast<size_t>(ggml_decoder->get_input_shape(name).to_shape()[1]);
            const size_t dim2        = static_cast<size_t>(ggml_decoder->get_input_shape(name).to_shape()[0]);
            size_t phys_stride = static_cast<size_t>(input_stride[1]) / element_size;
            ov::Shape input_shape = { dim2, num_rows, phys_stride }; // {1, 7, 9216 }
            input_tensor = ov::Tensor(ggml_decoder->get_input_type(name), input_shape, input_data);
        } else {
            input_tensor = ov::Tensor(ggml_decoder->get_input_type(name), ggml_decoder->get_input_shape(name).to_shape(), input_data);
        }

        // input_tensors[name] = input_tensor;
        input_tensors.emplace_back(name, input_tensor);
    }
    // std::cout << "input_names.size(): " << input_names.size() << std::endl;
    return input_tensors;
}

std::map<std::string, void*> get_ggml_graph_output_dst(std::shared_ptr<GgmlOvDecoder> ggml_decoder) {
    std::map<std::string, void*> output_tensors;
    auto output_names = ggml_decoder->get_output_names();
    for (size_t inp = 0; inp < output_names.size(); ++inp) {
        auto name = output_names[inp];
        auto output_data = ggml_decoder->get_output_ggml_tensor(name)->data;
        #ifdef GGML_OPENVINO_DEBUG
            printf("Output %d: %g\n", inp, *(double*)(output_data));
        #endif
        output_tensors[name] = output_data;
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

enum ggml_status openvino_frontend_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph, const int32_t start_index, const int32_t end_index, bool flag) {
    static ov::Core core;
    // auto devices = core.get_available_devices();
    // Get GGML Frontend 
    static auto front_end = get_ggml_frontend();
    if (!front_end) {
        GGML_LOG_ERROR("GGML FrontEnd is not initialized \n");
        return GGML_STATUS_FAILED;
    } else {
        #ifdef GGML_OPENVINO_DEBUG
            GGML_LOG_INFO("GGML FrontEnd is initialized \n");
        #endif
    }
    auto ggml_decoder = get_ggml_decoder(cgraph, start_index, end_index);
    std::shared_ptr<ov::frontend::DecoderBase> graph_decoder = ggml_decoder;
    // Load GraphIterator -> InputModel
    ov::frontend::InputModel::Ptr input_model = front_end->load(graph_decoder);
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

    ov::CompiledModel compiled_model = core.compile_model(model);

    // Create infer request
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // Get input tensor
    auto input_names = ggml_decoder->get_input_names();
    auto input_tensors = get_ggml_graph_input_tensors(ggml_decoder, flag);

    // Set input tensor
    for (size_t i = 0; i < input_names.size(); i++) {
        infer_request.set_input_tensor(i, input_tensors.at(i).second);
    }

    infer_request.infer();

    // Set dst data for outputs
    auto output_names = ggml_decoder->get_output_names();
    auto output_tensors = get_ggml_graph_output_dst(ggml_decoder);
    for (size_t i = 0; i < output_names.size(); i++) {
        auto output_tensor = infer_request.get_output_tensor(i);
        std::memcpy(output_tensors[output_names[i]], output_tensor.data(), output_tensor.get_byte_size());
        #ifdef GGML_OPENVINO_DEBUG
            printf("Output %s after: %g\n", output_names[i].c_str(), *(double*)(output_tensor.data()));
        #endif
    }

    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}
