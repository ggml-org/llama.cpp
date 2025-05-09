#include "utils.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <openvino/core/graph_util.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/tensor.hpp>
#include <unordered_map>

#include "ggml-impl.h"
#include "ggml.h"
#include "openvino/frontend.hpp"
#include "openvino/input_model.hpp"

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
    auto front_end = fem.load_by_framework("ggml");
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
    // static auto front_end = get_ggml_frontend();
    // if (!front_end) {
    //     GGML_LOG_ERROR("GGML FrontEnd is not initialized \n");
    //     return GGML_STATUS_FAILED;
    // }

    using CachedItem = std::pair<std::shared_ptr<ov::Model>, ov::CompiledModel>;
    static std::unordered_map<struct ggml_cgraph*, CachedItem> compiled_cache;

    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    int64_t conversion_end_time;
    int64_t compile_end_time;

    auto ggml_decoder = get_ggml_decoder(cgraph);
    auto it = compiled_cache.find(cgraph);
    if (it != compiled_cache.end()) {
        model = it->second.first;
        conversion_end_time = ggml_time_us();

        compiled_model = it->second.second;
        compile_end_time = ggml_time_us();
    } else {
        // std::shared_ptr<ov::frontend::DecoderBase> graph_decoder = ggml_decoder;
        // ov::frontend::InputModel::Ptr input_model = front_end->load(graph_decoder);
        // if (!input_model) {
        //     GGML_LOG_ERROR("Input Model is not loaded \n");
        //     return GGML_STATUS_FAILED;
        // }

        // model = front_end->convert(input_model);

        ov::frontend::InputModel::Ptr input_model = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder);
        model = ov::frontend::ggml::FrontEnd::convert(input_model);

        conversion_end_time = ggml_time_us();

        if (getenv("GGML_OPENVINO_DUMP_IR")) {
            char timestamped_filename[64];
            auto timestamp = (long long)ggml_time_us();
            snprintf(timestamped_filename, sizeof(timestamped_filename), "model_%lld.xml", timestamp);
            ov::serialize(model, timestamped_filename);
        }

        if (!model) {
            GGML_LOG_ERROR("Model is not converted \n");
        }
        compiled_model = core.compile_model(model, "CPU");
        compile_end_time = ggml_time_us();

        compiled_cache[cgraph] = std::make_pair(model, compiled_model);
    }

    ov::InferRequest infer_request = compiled_model.create_infer_request();

    auto ov_params = model->get_parameters();
    for (size_t i = 0; i < ov_params.size(); i++) {
        auto param_name = ov_params[i]->get_friendly_name();
        ov::Tensor input_tensor;
        if (ggml_decoder->get_model_extra_inputs().find(param_name) != ggml_decoder->get_model_extra_inputs().end()) {
            input_tensor = *ggml_decoder->get_model_extra_input_values().at(param_name);
        } else {
            input_tensor = get_ggml_graph_input_tensor(ggml_decoder, param_name);
        }
        infer_request.set_input_tensor(i, input_tensor);

        if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
            print_input_tensor_info(param_name, input_tensor);
        }
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
            print_output_tensor_info(output_names[i], output_tensor, output_tensors);
        }
    }
    auto end_time = ggml_time_us();

    if (getenv("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("GGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph conversion Time: %ld ms \n", (conversion_end_time - start_time) / 1000);
        GGML_LOG_INFO("  - Graph compile Time: %ld ms \n", (compile_end_time - conversion_end_time) / 1000);
        GGML_LOG_INFO("  - Graph Input Time: %ld ms \n", (input_end_time - compile_end_time) / 1000);
        GGML_LOG_INFO("  - Graph Inference Time: %ld ms \n", (infer_end_time - input_end_time) / 1000);
        GGML_LOG_INFO("  - Graph Output Time: %ld ms \n", (end_time - infer_end_time) / 1000);
    }

    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}

size_t checksum(const void* data, size_t size) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    size_t sum = 0;
    for (size_t i = 0; i < size; ++i) {
        sum += bytes[i];
    }
    return sum;
}

void print_input_tensor_info(const std::string& name, const ov::Tensor& tensor) {
    std::cout << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.data()
              << std::endl;
    switch (tensor.get_element_type()) {
    case ov::element::f32:
        std::cout << *(float*)(tensor.data()) << std::endl;
        break;
    case ov::element::f16:
        std::cout << ov::float16::from_bits(*(uint16_t*)(tensor.data())) << std::endl;
        break;
    case ov::element::i32:
        std::cout << *(int32_t*)(tensor.data()) << std::endl;
        break;
    case ov::element::i64:
        std::cout << *(int64_t*)(tensor.data()) << std::endl;
        break;
    default:
        break;
    }
}

void print_output_tensor_info(const std::string& name,
                              const ov::Tensor& tensor,
                              std::map<std::string, void*>& output_dst) {
    std::cout << "Output name: " << name << ", Output shape: " << tensor.get_shape()
              << ", Address: " << output_dst[name] << std::endl;
    switch (tensor.get_element_type()) {
    case ov::element::f32:
        std::cout << *(float*)(tensor.data()) << std::endl;
        std::cout << checksum(tensor.data(), tensor.get_byte_size()) << std::endl;
        break;
    case ov::element::f16:
        std::cout << ov::float16::from_bits(*(uint16_t*)(tensor.data())) << std::endl;
        std::cout << checksum(tensor.data(), tensor.get_byte_size()) << std::endl;
        break;
    default:
        break;
    }
}