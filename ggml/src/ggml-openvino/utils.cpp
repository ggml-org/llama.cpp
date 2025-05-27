#include "utils.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <openvino/core/any.hpp>
#include <openvino/core/graph_util.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <openvino/runtime/tensor.hpp>
#include <unordered_map>

#include "ggml-impl.h"
#include "ggml.h"
#include "openvino/frontend.hpp"
#include "openvino/input_model.hpp"

std::shared_ptr<GgmlOvDecoder> get_ggml_decoder(struct ggml_cgraph* cgraph, bool is_static, bool is_first_token) {
    return std::make_shared<GgmlOvDecoder>(nullptr, cgraph, is_static, is_first_token);
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
    static ov::Core core;
    static bool is_first_token = true;

    static std::string device = getenv("GGML_OPENVINO_DEVICE") ? getenv("GGML_OPENVINO_DEVICE") : "";
    if (device.empty()) {
        // Prefer GPU over CPU
        for (const auto& dev : core.get_available_devices()) {
            device = dev;
            if (device == "GPU")
                break;
        }
    }

    bool is_static = device == "NPU" ? true : false;
    ov::AnyMap config;
    if (is_static) {
        config = {
            {"NPU_COMPILATION_MODE_PARAMS", "compute-layers-with-higher-precision=ReduceMean"},
            {"NPU_USE_NPUW", "YES"},
            {"NPUW_DEVICES", "NPU"},
            {"NPUW_FOLD", "YES"},
            // {"NPU_COMPILER_TYPE", "MLIR"},
        };
    }

    auto start_time = ggml_time_us();

    auto* cache_dir = getenv("GGML_OPENVINO_CACHE_DIR");
    if (cache_dir && !is_static) {
        core.set_property(ov::cache_dir(cache_dir));
    }

    // For CPU and GPU, there is only one compiled model, so only use the first element of the pair
    // For NPU, there are prefill model and kvcache model (This is the ideal approach, but not implemented yet,
    // currently recompile for every token)
    using CachedItem = std::pair<std::shared_ptr<ov::Model>, std::pair<ov::CompiledModel, ov::CompiledModel>>;
    static std::unordered_map<struct ggml_cgraph*, CachedItem> compiled_cache;

    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model_prefill;
    ov::CompiledModel compiled_model_kvcache;
    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;

    auto ggml_decoder = get_ggml_decoder(cgraph, is_static, is_first_token);
    decoder_end_time = ggml_time_us();

    auto it = compiled_cache.find(cgraph);
    if (it != compiled_cache.end() && !is_static) {
        model = it->second.first;
        conversion_end_time = ggml_time_us();

        compiled_model_prefill = it->second.second.first;
        compiled_model_kvcache = it->second.second.second;
        compile_end_time = ggml_time_us();
    } else {
        ov::frontend::InputModel::Ptr input_model = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder);
        model = ov::frontend::ggml::FrontEnd::convert(input_model);

        conversion_end_time = ggml_time_us();

        if (getenv("GGML_OPENVINO_DUMP_IR")) {
            char timestamped_filename[64];
            auto timestamp = (long long)ggml_time_us();
            snprintf(timestamped_filename, sizeof(timestamped_filename), "model_%lld.xml", timestamp);
            ov::serialize(model, timestamped_filename);
        }

        compiled_model_prefill = core.compile_model(model, device, config);
        compile_end_time = ggml_time_us();

        compiled_cache[cgraph] = std::make_pair(model, std::make_pair(compiled_model_prefill, compiled_model_kvcache));
    }

    ov::InferRequest infer_request;
    if (!is_static) {
        infer_request = compiled_model_prefill.create_infer_request();
    } else {
        infer_request = compiled_model_prefill.create_infer_request();
        // if (is_first_token) {
        //     infer_request = compiled_model_prefill.create_infer_request();
        // } else {
        //     infer_request = compiled_model_kvcache.create_infer_request();
        // }
    }

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

    is_first_token = false;

    if (getenv("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("GGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder Time: %ld ms \n", (decoder_end_time - start_time) / 1000);
        GGML_LOG_INFO("  - Graph conversion Time: %ld ms \n", (conversion_end_time - decoder_end_time) / 1000);
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
