#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
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
#include <vector>

#include "ggml-impl.h"
#include "ggml.h"
#include "openvino/frontend.hpp"
#include "openvino/input_model.hpp"

std::shared_ptr<GgmlOvDecoder> get_ggml_decoder(struct ggml_cgraph* cgraph, bool is_static, bool is_first_token) {
    return std::make_shared<GgmlOvDecoder>(nullptr, cgraph, is_static, is_first_token);
}

ov::Tensor convert_ggml_input_to_ov(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                    const std::string& name) {
  auto *input_data = ggml_decoder->get_input_ggml_tensor(name)->data;
  ov::Tensor input_tensor;
  ov::Shape input_shape = ggml_decoder->get_input_shape(name).to_shape();
  std::vector<size_t> input_stride = ggml_decoder->get_input_stride(name);
  input_tensor =
      ov::Tensor(ggml_decoder->get_input_type(name), input_shape, input_data);
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

    static std::string device = getenv("GGML_OPENVINO_DEVICE") ? getenv("GGML_OPENVINO_DEVICE") : "";
    if (device.empty()) {
      const std::vector<std::string> preferred_device = {"GPU", "CPU", "NPU"};
      const auto available_devices = core.get_available_devices();
      for (const auto& dev : preferred_device) {
        if (std::find(available_devices.begin(), available_devices.end(),
                      dev) != available_devices.end()) {
          device = dev;
          break;
        }
      }
    }

    bool is_static = device == "NPU" ? true : false;
    ov::AnyMap config;
    if (device == "NPU") {
      config = get_npu_config();
    }

    auto start_time = ggml_time_us();

    auto* cache_dir = getenv("GGML_OPENVINO_CACHE_DIR");
    if (cache_dir && !is_static) {
        core.set_property(ov::cache_dir(cache_dir));
    }

    // CPU and GPU will only use cache_prefill
    using CachedItem = std::pair<std::shared_ptr<ov::Model>, ov::CompiledModel>;
    static std::unordered_map<struct ggml_cgraph*, CachedItem> compiled_cache_prefill;
    static std::unordered_map<struct ggml_cgraph*, CachedItem> compiled_cache_kvcache;

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;

    bool is_first_token = is_prefill(cgraph);

    auto it = compiled_cache_prefill.find(cgraph);
    if (it != compiled_cache_prefill.end()) {
        ggml_decoder = get_ggml_decoder(cgraph, is_static, false);
        decoder_end_time = ggml_time_us();

        if (is_static) {
            if (is_first_token) {
                model          = compiled_cache_prefill[cgraph].first;
                compiled_model = compiled_cache_prefill[cgraph].second;
            } else {
                model          = compiled_cache_kvcache[cgraph].first;
                compiled_model = compiled_cache_kvcache[cgraph].second;
            }
        } else {
            model = it->second.first;
            compiled_model = it->second.second;
        }
        conversion_end_time = ggml_time_us();
        compile_end_time = conversion_end_time;
    } else {
        if (is_static) {
            ggml_decoder = get_ggml_decoder(cgraph, is_static, true);
            auto ggml_decoder_kvcache = get_ggml_decoder(cgraph, is_static, false);
            decoder_end_time = ggml_time_us();

            auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder);
            auto input_model_kvcache = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder_kvcache);

            model = ov::frontend::ggml::FrontEnd::convert(input_model);
            auto model_kvcache = ov::frontend::ggml::FrontEnd::convert(input_model_kvcache);
            conversion_end_time = ggml_time_us();

            compiled_model = core.compile_model(model, device, config);
            auto compiled_model_kvcache = core.compile_model(model_kvcache, device, config);
            compile_end_time = ggml_time_us();

            compiled_cache_prefill[cgraph] = std::make_pair(model, compiled_model);
            compiled_cache_kvcache[cgraph] = std::make_pair(model_kvcache, compiled_model_kvcache);

            if (getenv("GGML_OPENVINO_DUMP_IR")) {
                char timestamped_filename[64];
                auto timestamp = (long long)ggml_time_us();
                snprintf(timestamped_filename, sizeof(timestamped_filename), "model_prefill_%lld.xml", timestamp);
                ov::serialize(model, timestamped_filename);
                snprintf(timestamped_filename, sizeof(timestamped_filename), "model_kvcache_%lld.xml", timestamp);
                ov::serialize(model_kvcache, timestamped_filename);
            }
        } else {
            ggml_decoder = get_ggml_decoder(cgraph, is_static, true);
            decoder_end_time = ggml_time_us();

            auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder);
            model = ov::frontend::ggml::FrontEnd::convert(input_model);
            conversion_end_time = ggml_time_us();

            compiled_model = core.compile_model(model, device, config);
            compile_end_time = ggml_time_us();
            compiled_cache_prefill[cgraph] = std::make_pair(model, compiled_model);

            if (getenv("GGML_OPENVINO_DUMP_IR")) {
                char timestamped_filename[64];
                auto timestamp = (long long)ggml_time_us();
                snprintf(timestamped_filename, sizeof(timestamped_filename), "model_%lld.xml", timestamp);
                ov::serialize(model, timestamped_filename);
            }
        }
    }
    auto infer_request = compiled_model.create_infer_request();

    auto ov_params = model->get_parameters();
    for (size_t i = 0; i < ov_params.size(); i++) {
        auto param_name = ov_params[i]->get_friendly_name();
        auto input_tensor = get_ov_input_tensor(ggml_decoder, param_name);
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

ov::AnyMap get_npu_config() {
    ov::AnyMap config = {
        { "NPU_COMPILATION_MODE_PARAMS", "compute-layers-with-higher-precision=ReduceMean" },
        { "NPU_USE_NPUW",                "YES"                                             },
        { "NPUW_DEVICES",                "NPU"                                             },
        { "NPUW_FOLD",                   "YES"                                             },
        { "NPUW_HOST_GATHER",            "YES"                                             },
        { "NPUW_DQ",                     "YES"                                             },
        { "NPUW_FUNCALL_ASYNC",          "YES"                                             },
        { "NPUW_WEIGHTS_BANK",           "shared"                                          },
        // Option 'CACHE_DIR' is not supported with MLIR compiler type
        // {"NPUW_CACHE_DIR", getenv("GGML_OPENVINO_CACHE_DIR") ? getenv("GGML_OPENVINO_CACHE_DIR") : ""},
        { "NPU_COMPILER_TYPE",           "MLIR"                                            },
    };
    return config;
}

ov::Tensor get_ov_input_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                               const std::string& param_name) {
  bool is_static = ggml_decoder->is_static();
  bool is_first_token = ggml_decoder->is_first_token();

  ov::Tensor input_tensor;
  if (ggml_decoder->get_model_extra_inputs().find(param_name) !=
      ggml_decoder->get_model_extra_inputs().end()) {
    input_tensor = *ggml_decoder->get_model_extra_input_values().at(param_name);

  } else if (!is_static) {
    input_tensor = convert_ggml_input_to_ov(ggml_decoder, param_name);

  } else {
    if (param_name == "inp_tokens" || param_name == "inp_pos") {
      if (is_first_token) {
        size_t max_token_len = ggml_decoder->get_max_token_len();
        const auto *input_tensor_ggml =
            ggml_decoder->get_input_ggml_tensor(param_name);
        std::vector<int32_t> padded_data =
            pad_input<int32_t>(input_tensor_ggml, 1, max_token_len, 0);
        input_tensor =
            ov::Tensor(ov::element::i32, ov::Shape{1, 1, max_token_len});
        auto *data_ptr = input_tensor.data<int32_t>();
        std::copy(padded_data.begin(), padded_data.end(), data_ptr);
      } else {
        input_tensor = convert_ggml_input_to_ov(ggml_decoder, param_name);
      }

    } else if (param_name == "KQ_mask") {
      size_t max_token_len = ggml_decoder->get_max_token_len();
      const auto *input_tensor_ggml =
          ggml_decoder->get_input_ggml_tensor(param_name);
      if (is_first_token) {
        std::vector<float> padded_data = pad_input<float>(
            input_tensor_ggml, max_token_len, max_token_len, -INFINITY);
        set_zero_diagonal(padded_data, max_token_len);
        input_tensor = ov::Tensor(ov::element::f32,
                                  ov::Shape{1, max_token_len, max_token_len});
        auto *data_ptr = input_tensor.data<float>();
        std::copy(padded_data.begin(), padded_data.end(), data_ptr);
      } else {
        std::vector<float> padded_data =
            pad_input<float>(input_tensor_ggml, 1, max_token_len, -INFINITY);
        input_tensor =
            ov::Tensor(ov::element::f32, ov::Shape{1, 1, max_token_len});
        auto *data_ptr = input_tensor.data<float>();
        std::copy(padded_data.begin(), padded_data.end(), data_ptr);
      }

    } else {
      input_tensor = convert_ggml_input_to_ov(ggml_decoder, param_name);
    }
  }
  return input_tensor;
}

size_t checksum(const void* data, size_t size) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    size_t sum = 0;
    for (size_t i = 0; i < size; ++i) {
        sum += (uint8_t)i;
        sum += bytes[i];
    }
    return sum;
}

// Suppress deprecation warning for ov::Tensor::data()
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

void print_input_tensor_info(const std::string& name, const ov::Tensor& tensor) {
    std::cout << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.data()
              << std::endl;
    switch (tensor.get_element_type()) {
    case ov::element::f32:
      std::cout << *(tensor.data<float>()) << std::endl;
      break;
    case ov::element::f16:
      std::cout << ov::float16::from_bits(*(tensor.data<uint16_t>()))
                << std::endl;
      break;
    case ov::element::i32:
      std::cout << *(tensor.data<int32_t>()) << std::endl;
      break;
    case ov::element::i64:
      std::cout << *(tensor.data<int64_t>()) << std::endl;
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
      std::cout << *(tensor.data<float>()) << std::endl;
      std::cout << checksum(tensor.data(), tensor.get_byte_size()) << std::endl;
      break;
    case ov::element::f16:
      std::cout << ov::float16::from_bits(*(tensor.data<uint16_t>()))
                << std::endl;
      std::cout << checksum(tensor.data(), tensor.get_byte_size()) << std::endl;
      break;
    default:
        break;
    }
}

#pragma GCC diagnostic pop

void set_zero_diagonal(std::vector<float>& matrix, size_t dim) {
    for (size_t i = 0; i < dim; ++i) {
        matrix[i * dim + i] = 0.0f;
    }
}

bool is_prefill(struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            auto* src = op->src[j];
            if (src == nullptr) {
                break;
            }
            if (std::string(src->name) == "inp_tokens") {
                return src->ne[0] != 1;
            }
        }
    }
    GGML_LOG_ERROR("is_prefill: inp_tokens not found in cgraph");
    throw std::runtime_error("is_prefill: inp_tokens not found in cgraph");
}
