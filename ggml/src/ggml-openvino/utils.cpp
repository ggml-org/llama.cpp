#include "utils.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <openvino/core/graph_util.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/openvino.hpp>

#include "ggml-impl.h"
#include "ggml.h"

std::shared_ptr<GgmlOvDecoder> get_ggml_decoder(struct ggml_cgraph* cgraph) {
    return std::make_shared<GgmlOvDecoder>(nullptr, cgraph);
}

ov::Tensor get_ggml_graph_input_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, std::string& name) {
    auto* input_data = ggml_decoder->get_input_ggml_tensor(name)->data;
    ov::Tensor input_tensor;
    ov::Shape input_shape = ggml_decoder->get_input_shape(name).to_shape();
    std::vector<size_t> input_stride = ggml_decoder->get_input_stride(name);
    input_tensor = ov::Tensor(ggml_decoder->get_input_type(name), input_shape, input_data);
    return input_tensor;
}

std::map<std::string, void*> get_ggml_graph_output_dst(std::shared_ptr<GgmlOvDecoder> ggml_decoder) {
    std::map<std::string, void*> output_tensors;
    auto output_names = ggml_decoder->get_model_output_names();
    for (size_t inp = 0; inp < output_names.size(); ++inp) {
        auto name = output_names[inp];
        const auto* tensor = ggml_decoder->get_output_ggml_tensor(name);
        auto* output_data = tensor->view_src ? tensor->view_src->data : tensor->data;
        output_tensors[name] = output_data;
    }
    return output_tensors;
}

static ov::frontend::FrontEnd::Ptr get_ggml_frontend() {
    auto fem = ov::frontend::FrontEndManager();
    std::string fe_so_path;
#ifdef GGML_OV_FRONTEND
    fe_so_path = GGML_OV_FRONTEND;
#endif
    fem.register_front_end("ggml", fe_so_path);
    front_end = fem.load_by_framework("ggml");
    return front_end;
}

enum ggml_status openvino_frontend_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    auto start_time = ggml_time_us();

    static ov::Core core;
    auto* cache_dir = getenv("GGML_OPENVINO_CACHE_DIR");
    if (cache_dir) {
        core.set_property(ov::cache_dir(cache_dir));
    }

    // auto devices = core.get_available_devices();
    static auto front_end = get_ggml_frontend();
    if (!front_end) {
        GGML_LOG_ERROR("GGML FrontEnd is not initialized \n");
        return GGML_STATUS_FAILED;
    }

    auto ggml_decoder = get_ggml_decoder(cgraph);
    std::shared_ptr<ov::frontend::DecoderBase> graph_decoder = ggml_decoder;
    ov::frontend::InputModel::Ptr input_model = front_end->load(graph_decoder);
    if (!input_model) {
        GGML_LOG_ERROR("Input Model is not loaded \n");
        return GGML_STATUS_FAILED;
    }

    std::shared_ptr<ov::Model> model = front_end->convert(input_model);
    auto conversion_end_time = ggml_time_us();

    if (getenv("GGML_OPENVINO_DUMP_IR")) {
        char timestamped_filename[64];
        auto timestamp = (long long)ggml_time_us();
        snprintf(timestamped_filename, sizeof(timestamped_filename), "model_%lld.xml", timestamp);
        ov::serialize(model, timestamped_filename);
    }

    if (!model) {
        GGML_LOG_ERROR("Model is not converted \n");
    }

    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    auto compile_end_time = ggml_time_us();

    ov::InferRequest infer_request = compiled_model.create_infer_request();
    auto infer_request_start_time = ggml_time_us();

    auto input_names = ggml_decoder->get_input_names();
    auto ov_params = model->get_parameters();
    for (size_t i = 0; i < ov_params.size(); i++) {
        auto param_name = ov_params[i]->get_friendly_name();
        auto input_tensor = get_ggml_graph_input_tensor(ggml_decoder, param_name);

        if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
            std::cout << "Input name: " << param_name << ", Input shape: " << input_tensor.get_shape()
                      << ", Address: " << input_tensor.data() << std::endl;
            switch (input_tensor.get_element_type()) {
            case ov::element::f32:
                std::cout << *(float*)(input_tensor.data()) << std::endl;
                break;
            case ov::element::f16:
                std::cout << ov::float16::from_bits(*(uint16_t*)(input_tensor.data())) << std::endl;
                break;
            case ov::element::i32:
                std::cout << *(int32_t*)(input_tensor.data()) << std::endl;
                break;
            case ov::element::i64:
                std::cout << *(int64_t*)(input_tensor.data()) << std::endl;
                break;
            default:
                break;
            }
        }
        infer_request.set_input_tensor(i, input_tensor);
    }
    auto input_end_time = ggml_time_us();

    infer_request.infer();
    auto infer_end_time = ggml_time_us();

    auto output_names = ggml_decoder->get_model_output_names();
    auto output_tensors = get_ggml_graph_output_dst(ggml_decoder);
    for (size_t i = 0; i < output_names.size(); i++) {
        auto output_tensor = infer_request.get_output_tensor(i);
        std::memcpy(output_tensors[output_names[i]], output_tensor.data(), output_tensor.get_byte_size());

        if (getenv("GGML_OPENVINO_DEBUG_OUTPUT")) {
            std::cout << "Output name: " << output_names[i] << ", Output shape: " << output_tensor.get_shape()
                      << ", Address: " << output_tensors[output_names[i]] << std::endl;
            switch (output_tensor.get_element_type()) {
            case ov::element::f32:
                std::cout << *(float*)(output_tensors[output_names[i]]) << std::endl;
                break;
            case ov::element::f16:
                std::cout << ov::float16::from_bits(*(uint16_t*)(output_tensors[output_names[i]])) << std::endl;
                break;
            default:
                break;
            }
        }
    }
    auto end_time = ggml_time_us();

    if (getenv("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("GGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph conversion Time: %ld ms \n", (conversion_end_time - start_time) / 1000);
        GGML_LOG_INFO("  - Graph compile Time: %ld ms \n", (compile_end_time - conversion_end_time) / 1000);
        GGML_LOG_INFO("  - Graph InferRequest created Time: %ld ms \n",
                      (infer_request_start_time - compile_end_time) / 1000);
        GGML_LOG_INFO("  - Graph Input Time: %ld ms \n", (input_end_time - infer_request_start_time) / 1000);
        GGML_LOG_INFO("  - Graph Inference Time: %ld ms \n", (infer_end_time - input_end_time) / 1000);
        GGML_LOG_INFO("  - Graph Output Time: %ld ms \n", (end_time - infer_end_time) / 1000);
    }

    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}
