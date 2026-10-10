#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-openvino/ggml-decoder.h"
#include "ggml-openvino/openvino/node_context.h"
#include "ggml-openvino/openvino/op_table.h"

#include <cstdio>
#include <numeric>
#include <openvino/op/parameter.hpp>
#include <openvino/openvino.hpp>

static bool test_repeat(bool is_static, int64_t factor) {
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true}));
    GGML_ASSERT(ctx);
    auto * weights = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 8);
    auto * tokens = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 2);
    ggml_set_input(tokens);
    auto * input = ggml_get_rows(ctx.get(), weights, tokens);
    auto * output = ggml_repeat_4d(ctx.get(), input, 8, 2 * factor, 1, 1);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
    GGML_ASSERT(buffer);

    ModelParams model_params{};
    ComputeParams compute_params{};
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    auto decoder = std::make_shared<GgmlOvDecoder>(graph, model_params, compute_params, model_weights, is_static);
    const int repeat_index = ggml_graph_n_nodes(graph) - 1;
    auto tensor_map = std::make_shared<ov::frontend::ggml::TensorMap>();
    auto parameter = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{1, 1, is_static ? 2 : -1, 4});
    tensor_map->emplace(decoder->get_input_names(repeat_index).at(0), parameter);
    ov::frontend::ggml::NodeContext context(decoder, tensor_map, repeat_index);
    GGML_ASSERT(context.get_op_dynamic_dim() == 1);
    auto result = ov::frontend::ggml::op::translate_repeat(context);
    auto model = std::make_shared<ov::Model>(result, ov::ParameterVector{parameter});
    auto request = ov::Core().compile_model(model, "CPU").create_infer_request();

    for (size_t extent : {2, 1, 3, 5}) {
        if (is_static && extent != 2) {
            continue;
        }
        ov::Tensor data(ov::element::f32, {1, 1, extent, 4});
        std::iota(data.data<float>(), data.data<float>() + data.get_size(), 1.0f);
        request.set_input_tensor(data);
        request.infer();
        const auto actual = request.get_output_tensor();
        // The captured multiplier must scale runtime tokens, even past the captured output extent.
        const ov::Shape expected_shape{1, 1, extent * size_t(factor), 8};
        if (actual.get_shape() != expected_shape) {
            std::fprintf(stderr, "REPEAT static=%d factor=%lld extent=%zu: wrong output shape\n",
                         is_static, (long long) factor, extent);
            return false;
        }
        for (size_t row = 0; row < expected_shape[2]; ++row) {
            for (size_t col = 0; col < expected_shape[3]; ++col) {
                if (actual.data<const float>()[row * 8 + col] != data.data<float>()[(row % extent) * 4 + col % 4]) {
                    std::fprintf(stderr, "REPEAT: wrong value at row=%zu col=%zu\n", row, col);
                    return false;
                }
            }
        }
    }
    return true;
}

int main() {
    for (bool is_static : {false, true}) {
        for (int64_t factor : {1, 2}) {
            if (!test_repeat(is_static, factor)) {
                return 1;
            }
        }
    }
    std::puts("test-openvino-repeat: OK");
    return 0;
}
