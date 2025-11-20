#include "utils.h"

#include "ggml-impl.h"
#include "ggml-openvino/ggml-decoder.h"
#include "ggml.h"
#include "openvino/frontend.hpp"
#include "openvino/input_model.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <openvino/core/any.hpp>
#include <openvino/core/graph_util.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/runtime/tensor.hpp>
#include <string>
#include <unordered_map>
#include <vector>

// Suppress deprecation warning for ov::Tensor::data()
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

enum ggml_status openvino_frontend_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    static ov::Core core;

    auto get_device = [&] {
        std::string device = getenv("GGML_OPENVINO_DEVICE") ? getenv("GGML_OPENVINO_DEVICE") : "CPU";
        auto available_devices = core.get_available_devices();
        if (std::find(available_devices.begin(), available_devices.end(), device) == available_devices.end()) {
            GGML_LOG_WARN("GGML OpenVINO Backend: device %s is not available, fallback to CPU\n", device.c_str());
            device = "CPU";
        }
        return device;
    };
    static std::string device = get_device();
    bool is_static = device == "NPU" ? true : false;

    ov::AnyMap config;

    if (getenv("GGML_OPENVINO_DUMP_CGRAPH")) {
        std::string filename = "cgraph.txt";
        GgmlOvDecoder::dump_cgraph(cgraph, filename);
    }

    if (is_naive(cgraph)) {
        return naive_compute(cgraph, core, device, config);
    }

    auto start_time = ggml_time_us();

    auto * cache_dir = getenv("GGML_OPENVINO_CACHE_DIR");
    if (cache_dir && !is_static) {
        core.set_property(ov::cache_dir(cache_dir));
    }

    static std::mutex cache_mutex;
    static std::unordered_map<ggml_cgraph *, std::shared_ptr<ov::InferRequest>> infer_request_cache;
    static std::unordered_map<ggml_cgraph *, std::vector<std::string>> ov_input_names_cache;
    static std::unordered_map<ggml_cgraph *, std::vector<std::string>> ov_output_names_cache;

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;

    {
        std::lock_guard<std::mutex> lock(cache_mutex);

        auto it = infer_request_cache.find(cgraph);
        if (it != infer_request_cache.end()) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights, is_static);
            decoder_end_time = ggml_time_us();

            infer_request = infer_request_cache[cgraph];
            conversion_end_time = ggml_time_us();
            compile_end_time = conversion_end_time;
        } else {
            std::shared_ptr<ov::Model> model;
            auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, get_types_to_requant(device));

            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights, is_static);
            decoder_end_time = ggml_time_us();

            auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder);
            model = ov::frontend::ggml::FrontEnd::convert(input_model);
            ggml_decoder->clear_model_weights();
            conversion_end_time = ggml_time_us();

            if (getenv("GGML_OPENVINO_DUMP_IR")) {
                char timestamped_filename[64];
                auto timestamp = (long long) ggml_time_us();
                snprintf(timestamped_filename, sizeof(timestamped_filename), "model_%lld.xml", timestamp);
                ov::serialize(model, timestamped_filename);
            }

            auto compiled_model = core.compile_model(model, device, get_ov_compile_config(device));
            compile_end_time = ggml_time_us();
            infer_request_cache[cgraph] = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
            infer_request = infer_request_cache[cgraph];

            std::vector<std::string> ov_input_names;
            std::vector<std::string> ov_output_names;
            for (const auto & ov_param : model->get_parameters()) {
                ov_input_names.push_back(ov_param->get_friendly_name());
            }
            for (const auto & ov_output : model->get_results()) {
                ov_output_names.push_back(ov_output->get_friendly_name());
            }
            ov_input_names_cache[cgraph] = ov_input_names;
            ov_output_names_cache[cgraph] = ov_output_names;

            // Set output tensors (for NPU) and kvcache i/o tensors once and for all
            for (size_t i = 0; i < ov_output_names.size(); i++) {
                auto output_name = ov_output_names[i];
                if (is_static || output_name.find("cache") == 0) {
                    auto output_tensor = get_ov_output_tensor(ggml_decoder, ov_output_names[i]);
                    infer_request->set_output_tensor(i, output_tensor);
                }
            }
            for (size_t i = 0; i < ov_input_names.size(); i++) {
                auto param_name = ov_input_names[i];
                if (param_name.find("cache") == 0) {
                    ov::Tensor input_tensor;
                    if (is_static) {
                        input_tensor = get_ov_input_tensor_static(ggml_decoder, param_name, 0, 0);
                    } else {
                        input_tensor = get_ov_input_tensor(ggml_decoder, param_name);
                    }
                    infer_request->set_input_tensor(i, input_tensor);
                }
            }
        }
    }

    auto ov_input_names = ov_input_names_cache[cgraph];
    auto ov_output_names = ov_output_names_cache[cgraph];

    if (!is_static) {
        for (size_t i = 0; i < ov_input_names.size(); i++) {
            auto param_name = ov_input_names[i];
            if (param_name.find("cache") == 0) {
                continue;
            }
            auto input_tensor = get_ov_input_tensor(ggml_decoder, param_name);
            infer_request->set_input_tensor(i, input_tensor);

            if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
                print_input_tensor_info(param_name, input_tensor);
            }
        }

        for (size_t i = 0; i < ov_output_names.size(); i++) {
            auto output_tensor = get_ov_output_tensor(ggml_decoder, ov_output_names[i]);
            infer_request->set_output_tensor(i, output_tensor);
        }

        infer_request->infer();
        infer_end_time = ggml_time_us();

        if (getenv("GGML_OPENVINO_DEBUG_OUTPUT")) {
            for (size_t i = 0; i < ov_output_names.size(); i++) {
                const auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.data());
            }
        }
    } else {
        auto input_len = ggml_decoder->get_input_len();
        for (int j = 0; j < input_len; j++) {
            for (size_t i = 0; i < ov_input_names.size(); i++) {
                auto param_name = ov_input_names[i];
                if (param_name.find("cache") == 0) {
                    continue;
                }
                auto input_tensor = get_ov_input_tensor_static(ggml_decoder, param_name, j, input_len);
                infer_request->set_input_tensor(i, input_tensor);

                if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
                    const auto input_tensor = infer_request->get_input_tensor(i);
                    print_input_tensor_info(param_name, input_tensor);
                }
            }

            infer_request->infer();

            if (getenv("GGML_OPENVINO_DEBUG_OUTPUT")) {
                for (size_t i = 0; i < ov_output_names.size(); i++) {
                    const auto output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.data());
                }
            }
        }
        infer_end_time = ggml_time_us();
    }

    if (getenv("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder Time: %ld ms \n", (decoder_end_time - start_time) / 1000);
        GGML_LOG_INFO("  - Graph conversion Time: %ld ms \n", (conversion_end_time - decoder_end_time) / 1000);
        GGML_LOG_INFO("  - Graph compile Time: %ld ms \n", (compile_end_time - conversion_end_time) / 1000);
        GGML_LOG_INFO("  - Graph Inference Time: %ld ms \n", (infer_end_time - compile_end_time) / 1000);
    }

    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}

ov::AnyMap get_ov_compile_config(const std::string & device) {
    ov::AnyMap config;
    if (device == "NPU") {
        config = {
            {"NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES"   },
            {"NPU_USE_NPUW",                      "YES"   },
            {"NPUW_DEVICES",                      "NPU"   },
            {"NPUW_FOLD",                         "YES"   },
            {"NPUW_WEIGHTS_BANK",                 "shared"},
            {"NPUW_FUNCALL_FOR_ALL",              "YES"   },
            {"NPUW_FUNCALL_ASYNC",                "YES"   },
            {"NPUW_DQ",                           "YES"   },
            {"NPUW_DQ_FULL",                      "NO"    },
        };
        if (auto * cache_dir = getenv("GGML_OPENVINO_CACHE_DIR"); cache_dir) {
            config["NPUW_CACHE_DIR"] = cache_dir;
        }
    }
    return config;
}

std::map<ggml_type, ExtraQuantType> get_types_to_requant(const std::string & device) {
    if (device == "NPU") {
        return {
            {GGML_TYPE_Q4_0, ExtraQuantType::Q4_0_128},
            {GGML_TYPE_Q4_1, ExtraQuantType::Q4_0_128},
            {GGML_TYPE_Q4_K, ExtraQuantType::Q4_0_128},
            {GGML_TYPE_Q6_K, ExtraQuantType::F16     },
            {GGML_TYPE_Q5_K, ExtraQuantType::F16     },
        };
    }
    if (device == "GPU") {
        return {
            // gs16 will be supported on openvino-2025.4
            {GGML_TYPE_Q6_K, ExtraQuantType::Q8_0_32},
        };
    }
    return {};
}

bool is_naive(ggml_cgraph * cgraph) {
    constexpr int naive_graph_size_threshold = 20;
    return cgraph->n_nodes < naive_graph_size_threshold;
}

enum ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config) {
    if (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        return GGML_STATUS_SUCCESS;
    }
    if (cgraph->nodes[0]->op == GGML_OP_FLASH_ATTN_EXT) {
        return GGML_STATUS_FAILED;
    }

    auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);
    auto decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(decoder);
    auto naive = true;
    auto model = ov::frontend::ggml::FrontEnd::convert(input_model, naive);
    if (getenv("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }
    auto infer_request = core.compile_model(model, device, config).create_infer_request();

    auto ov_params = model->get_parameters();
    for (size_t i = 0; i < ov_params.size(); i++) {
        auto param_name = ov_params[i]->get_friendly_name();
        auto input_tensor = get_ov_input_tensor(decoder, param_name);
        infer_request.set_input_tensor(i, input_tensor);
    }

    auto ov_results = model->get_results();
    for (size_t i = 0; i < ov_results.size(); i++) {
        auto result_name = ov_results[i]->get_friendly_name();
        auto output_tensor = get_ov_output_tensor(decoder, result_name);
        infer_request.set_output_tensor(i, output_tensor);
    }

    infer_request.infer();
    return GGML_STATUS_SUCCESS;
}

namespace {
ov::Tensor convert_ggml_input_to_ov(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & name) {
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(name);
    auto * input_data = ggml_tensor->data;
    ov::Shape input_shape;
    if (ggml_tensor->op == GGML_OP_VIEW) {
        // This case is added to make test-backend-ops work
        input_shape = ggml_decoder->get_graph_input_shape(ggml_tensor, ggml_tensor->view_src).to_shape();
    } else {
        input_shape = ggml_decoder->get_input_shape(name).to_shape();
    }
    auto input_tensor = ov::Tensor(ggml_decoder->get_input_type(name), input_shape, input_data);
    return input_tensor;
}
}  // namespace

ov::Tensor get_ov_input_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & param_name) {
    ov::Tensor input_tensor;
    if (ggml_decoder->get_model_extra_inputs().find(param_name) != ggml_decoder->get_model_extra_inputs().end()) {
        input_tensor = *ggml_decoder->get_model_extra_input_values().at(param_name);
    } else {
        input_tensor = convert_ggml_input_to_ov(ggml_decoder, param_name);
    }
    return input_tensor;
}

ov::Tensor get_ov_input_tensor_static(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                      const std::string & param_name,
                                      int j,
                                      int input_len) {
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(param_name);
    const auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    if (param_name == "inp_pos" || param_name == "inp_tokens" || op->op == GGML_OP_SET_ROWS) {
        ov::Shape input_shape = {1, 1, 1, 1};
        ov::Tensor input_tensor(ggml_decoder->get_input_type(param_name), input_shape);
        // copy the j-th value from ggml_tensor
        size_t element_size = ggml_type_size(ggml_tensor->type);
        void * input_data = (char *) ggml_tensor->data + j * element_size;
        std::memcpy(input_tensor.data(), input_data, element_size);
        return input_tensor;
    }

    if (param_name == "inp_out_ids") {
        ov::Shape input_shape = {1, 1, 1, 1};
        ov::Tensor input_tensor(ggml_decoder->get_input_type(param_name), input_shape);
        if (ggml_tensor->ne[0] == 0) {
            *input_tensor.data<int32_t>() = 0;
        } else if (ggml_tensor->ne[0] == 1) {
            if (j == input_len - 1) {
                *input_tensor.data<int32_t>() = *((int32_t *) ggml_tensor->data);
            } else {
                *input_tensor.data<int32_t>() = 0;
            }
        } else {
            throw std::runtime_error("Static graph inp_out_ids unexpected ne[0] > 1");
        }
        return input_tensor;
    }

    if (param_name.find("KQ_mask") == 0) {
        size_t context_size = ggml_decoder->get_ctx_size();
        const auto * input_tensor_ggml = ggml_decoder->get_input_ggml_tensor(param_name);
        std::vector<float> padded_data = pad_input<float>(input_tensor_ggml, input_len, context_size, -INFINITY);
        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_size});
        // copy the j-th row of padded_data
        auto * data_ptr = input_tensor.data<float>();
        std::copy(padded_data.begin() + j * context_size, padded_data.begin() + (j + 1) * context_size, data_ptr);
        return input_tensor;
    }

    return get_ov_input_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_output_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & result_name) {
    auto * ggml_tensor = ggml_decoder->get_output_ggml_tensor(result_name);
    auto output_type = ggml_decoder->get_output_type(result_name);
    ov::Shape output_shape;
    output_shape = ggml_decoder->get_output_shape(result_name).to_shape();

    if (ggml_decoder->is_static() && result_name == "result_output") {
        output_shape[1] = 1;
    }

    ov::Tensor output_tensor(output_type, output_shape, ggml_tensor->data);
    return output_tensor;
}

size_t checksum(const void * data, size_t size) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    size_t sum = 0;
    for (size_t i = 0; i < size; ++i) {
        sum += (uint8_t) i;
        sum += bytes[i];
    }
    return sum;
}

void print_input_tensor_info(const std::string & name, const ov::Tensor & tensor) {
    std::cout << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.data()
              << std::endl;
    switch (tensor.get_element_type()) {
    case ov::element::f32:
        std::cout << *(tensor.data<float>()) << std::endl;
        break;
    case ov::element::f16:
        std::cout << *(tensor.data<ov::float16>()) << std::endl;
        break;
    case ov::element::i32:
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int32_t>()[i] << " ";
        }
        std::cout << std::endl;
        break;
    case ov::element::i64:
        std::cout << *(tensor.data<int64_t>()) << std::endl;
        break;
    default:
        break;
    }
}

void print_output_tensor_info(const std::string & name, const ov::Tensor & tensor, void * output_dst) {
    std::cout << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << std::endl;

    auto print_float_stats = [](const std::string & type_name, size_t size, auto get_value) {
        if (size == 0) {
            return;
        }

        float first = get_value(0);
        float min = first;
        float max = first;
        double sum = first;

        for (size_t i = 1; i < size; ++i) {
            float v = get_value(i);
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
            sum += v;
        }
        double mean = sum / size;

        std::cout << std::right << std::setw(6) << type_name << std::right << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << std::endl;
        std::cout << std::right << std::setw(6) << "" << std::right << std::setw(12) << first << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << mean << std::endl;
    };

    switch (tensor.get_element_type()) {
    case ov::element::f32: {
        const float * data = tensor.data<float>();
        size_t size = tensor.get_size();
        print_float_stats("[f32]", size, [data](size_t i) { return data[i]; });
        break;
    }
    case ov::element::f16: {
        const ov::float16 * data = tensor.data<ov::float16>();
        size_t size = tensor.get_size();
        print_float_stats("[f16]", size, [data](size_t i) { return static_cast<float>(data[i]); });
        break;
    }
    default:
        break;
    }
}

void set_zero_diagonal(std::vector<float> & matrix, size_t dim) {
    for (size_t i = 0; i < dim; ++i) {
        matrix[i * dim + i] = 0.0f;
    }
}

const ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            auto * src = op->src[j];
            if (src == nullptr) {
                break;
            }
            if (std::string(src->name) == "inp_pos") {
                return src;
            }
        }
    }
    GGML_LOG_ERROR("get_inp_pos_tensor: inp_pos not found in cgraph");
    throw std::runtime_error("get_inp_pos_tensor: inp_pos not found in cgraph");
}

bool get_is_first_token(const ggml_tensor * inp_pos) {
    return *(int32_t *) inp_pos->data == 0;
}

#pragma GCC diagnostic pop
