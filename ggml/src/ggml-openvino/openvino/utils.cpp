#include "utils.h"

#include "ggml-impl.h"

#include <cmath>
#include <cstddef>
#include <ctime>
#include <memory>
#include <openvino/op/add.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/split.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/transpose.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

std::string getCurrentTime() {
    std::time_t now = std::time(nullptr);
    char buf[100];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return buf;
}

void num_inputs_check(const NodeContext & context, size_t min_inputs, size_t max_inputs) {
    auto input_size = context.get_input_size();
    FRONT_END_OP_CONVERSION_CHECK(input_size >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(input_size <= max_inputs, "Got more inputs than expected");
}

int non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    int dim = nb.size() - 1;
    size_t bytes = nb[dim];
    for (int i = dim; i > 0; i--) {
        bytes *= ne[i];
        if (bytes != nb[i - 1]) {
            return i;
        }
    }
    return 0;
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf> & shape,
                                         const std::vector<int> & dims) {
    using namespace ov::op;
    const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node> & node, const std::vector<int> & dims) {
    return get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(node), dims);
}

OutputVector rename_outputs_with_suffix(const OutputVector & outputs, const std::string & suffix) {
    for (const auto & output : outputs) {
        auto node = output.get_node_shared_ptr();
        std::string name = node->get_friendly_name();
        name += "_";
        name += suffix;
        node->set_friendly_name(name);
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    return outputs;
}

namespace {
ov::Output<ov::Node> rope_yarn_ramp_mix(int n_dims, const float corr_dims[2], float ext_factor) {
    int half_n_dims = n_dims / 2;
    std::vector<float> dim_ids_vec(half_n_dims);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0);
    auto dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) half_n_dims}, dim_ids_vec);
    auto corr_low = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    auto corr_high = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    auto denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::Subtract>(corr_high, corr_low),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    auto ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(dim_ids, corr_low), denom);
    auto ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    auto one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    auto ramp_inverted = std::make_shared<ov::op::v1::Subtract>(one, ramp_clamped);
    auto ext_factor_node = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_factor});
    auto ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_factor_node);
    return ramp_mix;
}

float ggml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float) M_PI)) / (2 * logf(base));
}

void ggml_rope_yarn_corr_dims(int n_dims,
                              int n_ctx_orig,
                              float freq_base,
                              float beta_fast,
                              float beta_slow,
                              float dims[2]) {
    float start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<float>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight,
                                                           bool imrope,
                                                           bool stateful) {
    if (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } else if (imrope) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, true);
        auto pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } else {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    const int n_dims = rope_params[1];
    const size_t n_dims_half = n_dims >> 1;
    const int n_ctx_orig = rope_params[4];
    memcpy(&freq_base, rope_params + 5, sizeof(float));
    memcpy(&freq_scale, rope_params + 6, sizeof(float));
    memcpy(&ext_factor, rope_params + 7, sizeof(float));
    memcpy(&attn_factor, rope_params + 8, sizeof(float));
    memcpy(&beta_fast, rope_params + 9, sizeof(float));
    memcpy(&beta_slow, rope_params + 10, sizeof(float));

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    std::vector<float> factor(n_dims_half);

    Output<Node> freq_factors;

    Output<Node> theta;
    float mscale = attn_factor;
    if (imrope) {
        std::vector<int64_t> gather_indices(n_dims_half);
        for (size_t j = 0; j < n_dims_half; j++) {
            gather_indices[j] = j % 3;
            factor[j] = std::pow(theta_scale, j);
        }
        auto gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{n_dims_half}, gather_indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        auto factor_const = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_dims_half}, factor);
        theta = std::make_shared<ov::op::v1::Multiply>(inp_pos, factor_const);
    } else {
        float corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        factor[0] = 1.0f;
        for (size_t i = 1; i < factor.size(); i++) {
            factor[i] = theta_scale * factor[i - 1];
        }
        if (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, factor.size()}, factor);
        } else {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, factor.size()}, factor);
        }
        if (rope_freqs_weight) {
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_freqs_weight);
        }

        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        if (ext_factor == 0.0f) {
            theta = theta_interp;
        } else {
            auto ramp_mix = rope_yarn_ramp_mix(n_dims, corr_dims, ext_factor);
            Output<Node> one;
            if (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } else {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            }
            auto one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);

            theta =
                std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                                                  std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
        }
    }

    Output<Node> cos_theta = std::make_shared<ov::op::v0::Cos>(theta);
    Output<Node> sin_theta = std::make_shared<ov::op::v0::Sin>(theta);

    if (!imrope) {
        auto mscale_node = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_node);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_node);
    }

    return std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_input(const NodeContext & context, int input_index, int slice_len) {
    // Only works for VIEW operations that slice at the lowest dimension
    // If the VIEW also reshape the result, `slice_len` should be provided
    auto input = context.get_input(input_index);
    auto * op_params = (size_t *) context.get_input_op_params(input_index);
    auto src1_stride = context.get_input_stride(input_index);

    int64_t split_addr = op_params[0] / src1_stride[3];
    if (slice_len == 0) {
        slice_len = context.get_input_shape(input_index)[3].get_length();
    }
    int64_t slice_end = split_addr + slice_len;

    auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {split_addr});
    auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_end});
    auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    auto sliced = std::make_shared<ov::op::v8::Slice>(input, begin, end, stride, axes);
    return sliced;
}

ov::Output<ov::Node> process_view_input_new(const NodeContext & context, int input_index) {
    auto input = context.get_input(input_index);

    // Check if this input has view inputs
    size_t view_input_size = context.get_view_input_size(input_index);
    if (view_input_size == 0) {
        // No view inputs, return the input as is
        return input;
    }

    // If translate_view already resolved this VIEW (produced a Slice), the input
    // will already have the expected shape — skip re-slicing.
    auto expected_ov_shape = context.get_view_input_ov_shape(input_index, 0);
    auto actual_shape = input.get_partial_shape();
    if (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank() == actual_shape.rank()) {
        bool shapes_match = true;
        for (int64_t i = 0; i < expected_ov_shape.rank().get_length(); ++i) {
            if (!expected_ov_shape[i].is_static() || !actual_shape[i].is_static()) {
                shapes_match = false;
                break;
            }
            if (expected_ov_shape[i] != actual_shape[i]) {
                shapes_match = false;
                break;
            }
        }
        if (shapes_match) {
            return input;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    if (context.is_static() && view_input_size == 1) {
        auto view_stride_v = context.get_view_input_stride(input_index, 0);
        auto view_src_stride_v = context.get_view_input_src_stride(input_index, 0);
        auto view_ggml_shape = context.get_view_input_ggml_shape(input_index, 0);
        auto view_src_ggml_shape = context.get_view_input_src_ggml_shape(input_index, 0);
        auto view_offset = context.get_view_input_offset(input_index, 0);
        auto view_src_offset = context.get_view_input_src_offset(input_index, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<int> diff_dims;
        if (view_src_ggml_shape.size() == ndims) {
            for (size_t i = 0; i < ndims; ++i) {
                if (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<int>(i));
                }
            }
        }

        if (diff_dims.size() == 1) {
            int split_dim = diff_dims[0];
            int64_t num_splits = static_cast<int64_t>(view_src_ggml_shape[split_dim]);
            int64_t chunk_size = static_cast<int64_t>(view_ggml_shape[split_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            if (chunk_size == 1 && num_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                bool suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                if (suffix_ok) {
                    for (size_t i = static_cast<size_t>(split_dim) + 1; i < ndims; ++i) {
                        if (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = false;
                            break;
                        }
                    }
                }

                if (suffix_ok && view_src_stride_v[split_dim] > 0) {
                    size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
                    int64_t split_index = static_cast<int64_t>(relative_offset / view_src_stride_v[split_dim]);

                    if (split_index >= 0 && split_index < num_splits) {
                        auto src_node = input.get_node_shared_ptr();
                        std::string rt_key = "split_dim_" + std::to_string(split_dim);
                        auto & rt_info = src_node->get_rt_info();

                        if (rt_info.find(rt_key) == rt_info.end()) {
                            auto axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(split_dim)});
                            auto split_node =
                                std::make_shared<ov::op::v1::Split>(input, axis_const, static_cast<size_t>(num_splits));
                            split_node->set_friendly_name(src_node->get_friendly_name() + "_split");
                            rt_info[rt_key] = split_node;
                        }

                        auto split_node = rt_info[rt_key].as<std::shared_ptr<ov::op::v1::Split>>();
                        return split_node->output(static_cast<size_t>(split_index));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    auto process_single_view =
        [](ov::Output<ov::Node> current, size_t view_offset, const std::vector<size_t> & view_stride,
           const ov::Shape & view_ggml_shape, const ov::PartialShape & view_ov_shape, const std::string & view_name,
           size_t view_src_offset, const std::vector<size_t> & view_src_stride, const ov::Shape & view_src_ggml_shape,
           const ov::PartialShape & view_src_ov_shape, const std::string & view_src_name) -> ov::Output<ov::Node> {
        auto build_reshape_pattern = [](const ov::PartialShape & target_ov_shape,
                                        const ov::Shape & target_ggml_shape) -> std::vector<int64_t> {
            const size_t ndims = target_ggml_shape.size();
            std::vector<int64_t> reshape_pattern(ndims);
            size_t dynamic_dims = 0;

            if (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(ndims)) {
                for (size_t i = 0; i < ndims; ++i) {
                    if (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } else {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } else {
                dynamic_dims = 2;
            }

            if (dynamic_dims > 1) {
                for (size_t i = 0; i < ndims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            return reshape_pattern;
        };

        auto build_prefix_tail_reshape_pattern = [](const ov::PartialShape & target_ov_shape,
                                                    const ov::Shape & target_ggml_shape, size_t prefix_dims,
                                                    int64_t tail_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_pattern(prefix_dims + 1);
            size_t dynamic_dims = 0;

            if (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(target_ggml_shape.size())) {
                for (size_t i = 0; i < prefix_dims; ++i) {
                    if (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } else {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } else {
                dynamic_dims = 2;
            }

            if (dynamic_dims > 1) {
                for (size_t i = 0; i < prefix_dims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            reshape_pattern[prefix_dims] = tail_dim;
            return reshape_pattern;
        };

        bool same_stride = view_stride.size() == view_src_stride.size();
        if (same_stride) {
            for (size_t i = 0; i < view_stride.size(); ++i) {
                if (view_stride[i] != view_src_stride[i]) {
                    same_stride = false;
                    break;
                }
            }
        }

        bool same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        if (same_ggml_shape) {
            for (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                if (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = false;
                    break;
                }
            }
        }

        if (same_stride && same_ggml_shape) {
            return current;
        }

        if (same_stride) {
            const size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
            const size_t ndims = view_stride.size();

            std::vector<int> diff_dims;
            if (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                for (size_t i = 0; i < ndims; ++i) {
                    if (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<int>(i));
                    }
                }
            }

            if (diff_dims.size() == 1) {
                const int slice_dim = diff_dims[0];
                const int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                if (view_stride[slice_dim] > 0 && relative_offset % view_stride[slice_dim] == 0) {
                    const int64_t begin_val = static_cast<int64_t>((relative_offset / view_stride[slice_dim]) %
                                                                   static_cast<size_t>(dim_size));
                    const int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);

                    if (begin_val >= 0 && end_val <= dim_size) {
                        auto sliced = std::make_shared<ov::op::v8::Slice>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

                        if (view_ov_shape.is_static()) {
                            auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                                false);
                            reshaped->set_friendly_name(view_name);
                            return reshaped;
                        }

                        sliced->set_friendly_name(view_name);
                        return sliced;
                    }
                }

                int64_t tail_src_elems = 1;
                int64_t tail_dst_elems = 1;
                for (size_t i = slice_dim; i < ndims; ++i) {
                    tail_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    tail_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                const size_t elem_stride = view_stride[ndims - 1];
                int64_t tail_begin = 0;
                if (elem_stride > 0) {
                    tail_begin =
                        static_cast<int64_t>((relative_offset / elem_stride) % static_cast<size_t>(tail_src_elems));
                }
                const int64_t tail_end = tail_begin + tail_dst_elems;

                if (tail_begin >= 0 && tail_end <= tail_src_elems) {
                    std::vector<int64_t> flat_shape;
                    for (int i = 0; i < slice_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(tail_src_elems);
                    const size_t flat_ndims = flat_shape.size();

                    auto flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), false);

                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_end}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

                    if (view_ov_shape.is_static()) {
                        auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                            false);
                        reshaped->set_friendly_name(view_name);
                        return reshaped;
                    }

                    sliced->set_friendly_name(view_name);
                    return sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> step(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t remaining_offset = relative_offset;
            for (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                if (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(remaining_offset / view_stride[i]);
                    remaining_offset %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            bool in_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            if (in_bounds) {
                for (size_t i = 0; i < ndims; ++i) {
                    if (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        in_bounds = false;
                        break;
                    }
                }
            }

            if (in_bounds && remaining_offset == 0) {
                auto sliced = std::make_shared<ov::op::v8::Slice>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->set_friendly_name(view_name);
                return sliced;
            }
        } else {
            bool same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            const size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;

            if (same_rank) {
                const size_t ndims = view_ggml_shape.size();
                std::vector<int> diff_dims;
                for (size_t i = 0; i < ndims; ++i) {
                    if (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<int>(i));
                    }
                }

                if (diff_dims.size() == 1) {
                    const size_t slice_dim = static_cast<size_t>(diff_dims[0]);
                    bool suffix_stride_match = true;
                    for (size_t i = slice_dim + 1; i < ndims; ++i) {
                        if (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_match = false;
                            break;
                        }
                    }

                    if (suffix_stride_match && view_src_stride[slice_dim] > 0 &&
                        relative_offset % view_src_stride[slice_dim] == 0) {
                        const int64_t begin_val = static_cast<int64_t>(relative_offset / view_src_stride[slice_dim]);
                        const int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);
                        const int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                        if (begin_val >= 0 && end_val <= dim_size) {
                            auto sliced = std::make_shared<ov::op::v8::Slice>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(slice_dim)}));
                            sliced->set_friendly_name(view_name);
                            return sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            if (same_rank) {
                for (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            bool same_num_elements = same_rank && view_elems == src_elems;

            if (same_rank && relative_offset == 0 && same_num_elements) {
                auto reshape_pattern = build_reshape_pattern(view_ov_shape, view_ggml_shape);

                auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()}, reshape_pattern),
                    false);
                reshaped->set_friendly_name(view_name);
                return reshaped;
            }

            if (same_rank) {
                const size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                bool is_regular_slice = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> step(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t remaining_offset = relative_offset;

                if (is_regular_slice) {
                    for (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        if (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_slice = false;
                            break;
                        }

                        step[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        if (step[i] <= 0) {
                            is_regular_slice = false;
                            break;
                        }

                        begin[i] = static_cast<int64_t>(remaining_offset / view_src_stride[i]);
                        remaining_offset %= view_src_stride[i];

                        if (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            continue;
                        }

                        end[i] = begin[i] + step[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        if (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_slice = false;
                            break;
                        }
                    }
                }

                if (is_regular_slice && remaining_offset == 0) {
                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->set_friendly_name(view_name);
                    return sliced;
                }

                const size_t elem_stride = view_src_stride.back();
                const bool aligned_offset = elem_stride > 0 && relative_offset % elem_stride == 0;

                if (aligned_offset) {
                    size_t suffix_start = 0;
                    size_t expected_stride = elem_stride;
                    for (int i = static_cast<int>(ndims) - 1; i >= 0; --i) {
                        if (view_stride[i] != expected_stride) {
                            suffix_start = static_cast<size_t>(i + 1);
                            break;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t prefix_elems = 1;
                    size_t suffix_elems = 1;
                    for (size_t i = 0; i < suffix_start; ++i) {
                        prefix_elems *= view_ggml_shape[i];
                    }
                    for (size_t i = suffix_start; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    if (prefix_elems > 0 && src_elems % prefix_elems == 0) {
                        const size_t src_tail_elems = src_elems / prefix_elems;
                        const int64_t tail_begin = static_cast<int64_t>(relative_offset / elem_stride);
                        const int64_t tail_end = tail_begin + static_cast<int64_t>(suffix_elems);

                        if (tail_begin >= 0 && tail_end <= static_cast<int64_t>(src_tail_elems)) {
                            auto prefix_tail_pattern = build_prefix_tail_reshape_pattern(
                                view_ov_shape, view_ggml_shape, suffix_start, static_cast<int64_t>(src_tail_elems));

                            auto prefix_tail = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {prefix_tail_pattern.size()},
                                                             prefix_tail_pattern),
                                false);

                            ov::Output<ov::Node> selected = prefix_tail;
                            if (tail_begin != 0 || tail_end != static_cast<int64_t>(src_tail_elems)) {
                                selected = std::make_shared<ov::op::v8::Slice>(
                                    prefix_tail, ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_begin}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_end}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {static_cast<int64_t>(suffix_start)}));
                            }

                            auto reshape_pattern = build_reshape_pattern(view_ov_shape, view_ggml_shape);
                            auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                                selected,
                                ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()},
                                                             reshape_pattern),
                                false);
                            reshaped->set_friendly_name(view_name);
                            return reshaped;
                        }
                    }
                }
            }

            return current;
        }

        (void) view_name;
        (void) view_src_ov_shape;
        (void) view_src_name;

        return current;
    };

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = input;

    // Process each view in reverse order (from base to current)
    for (int view_idx = view_input_size - 1; view_idx >= 0; view_idx--) {
        auto view_offset = context.get_view_input_offset(input_index, view_idx);
        auto view_stride = context.get_view_input_stride(input_index, view_idx);
        auto view_ggml_shape = context.get_view_input_ggml_shape(input_index, view_idx);
        auto view_ov_shape = context.get_view_input_ov_shape(input_index, view_idx);
        auto view_name = context.get_view_input_name(input_index, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        auto view_src_offset = context.get_view_input_src_offset(input_index, view_idx);
        auto view_src_stride = context.get_view_input_src_stride(input_index, view_idx);
        auto view_src_ggml_shape = context.get_view_input_src_ggml_shape(input_index, view_idx);
        auto view_src_ov_shape = context.get_view_input_src_ov_shape(input_index, view_idx);
        auto view_src_name = context.get_view_input_src_name(input_index, view_idx);
        // print source view info
        // std::cout << "View " << view_idx << ": source name = " << view_src_name
        //           << ", source offset = " << view_src_offset << ", source stride = [" << view_src_stride[0] << ","
        //           << view_src_stride[1] << "," << view_src_stride[2] << "," << view_src_stride[3]
        //           << "], source ggml shape = [" << view_src_ggml_shape[0] << "," << view_src_ggml_shape[1] << ","
        //           << view_src_ggml_shape[2] << "," << view_src_ggml_shape[3]
        //           << "], source ov shape = " << view_src_ov_shape << std::endl;

        current = process_single_view(current, view_offset, view_stride, view_ggml_shape, view_ov_shape, view_name,
                                      view_src_offset, view_src_stride, view_src_ggml_shape, view_src_ov_shape,
                                      view_src_name);
    }

    return current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
