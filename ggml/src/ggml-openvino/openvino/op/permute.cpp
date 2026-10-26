#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/op/variadic_split.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_permute(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case != 0, "Unsupported PERMUTE case");
    // op_case 1 is trivial permute
    // op_case 2 is to permute Q. It has a preceding VIEW that reshapes Q to restore the sequqence dimension
    // op_case 3 4 it to permute KV cache in the default layout
    // op_case 5 6 is to permute V cache when `-fa off`, where v_trans=true

    ov::Output<Node> res;
    ov::Output<Node> src;
    if (op_case == 3 || op_case == 4 || op_case == 5 || op_case == 6) {
        src = context.get_input(0);
    } else {
        src = process_view_input_new(context, 0);
    }
    std::vector<int64_t> perm_values{0, 2, 1, 3};
    const int32_t * op_params = context.get_output_op_params();
    if (op_params != nullptr) {
        for (size_t input_axis = 0; input_axis < perm_values.size(); ++input_axis) {
            const size_t output_axis = static_cast<size_t>(op_params[input_axis]);
            perm_values[perm_values.size() - 1 - output_axis] =
                static_cast<int64_t>(perm_values.size() - 1 - input_axis);
        }
    }
    if (context.is_stateful() && (op_case == 5 || op_case == 6)) {
        // `-fa off` reads V transposed. The stateful V cache is token-major, [1, seq, n_heads_kv, head_size]
        const auto output_shape = context.get_output_shape().to_shape();
        const int64_t n_heads = output_shape[1];
        const int64_t head_size = output_shape[2];
        const auto & state_shape = src.get_partial_shape();
        const bool single_head = n_heads == 1 && state_shape[2].get_length() == 1;
        if (!single_head && state_shape[2].get_length() * state_shape[3].get_length() != n_heads * head_size) {
            // llama.cpp pads the V cache rows to the widest layer: keep this layer's rows
            auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto rows = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{0, 0, -1}), true);
            auto sliced = std::make_shared<ov::op::v8::Slice>(
                rows, zero, ov::op::v0::Constant::create(ov::element::i64, {1}, {n_heads * head_size}), one,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));
            src = std::make_shared<ov::op::v1::Reshape>(
                sliced,
                ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, n_heads, head_size}),
                true);
        }
        // keep V as [1, n_heads_kv, kv, head_size]: MUL_MAT then multiplies without transposing it
        res = std::make_shared<ov::op::v1::Transpose>(
            src, ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3}));
        if (single_head && state_shape[3].get_length() != head_size) {
            // the padded rows, as above. After the Transpose, which then cancels the state relayout, so the Slice reads
            // the KV cache in place: between the two they become copies of the cache, recompiled for every length
            res = std::make_shared<ov::op::v8::Slice>(res, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                                                      ov::op::v0::Constant::create(ov::element::i64, {1}, {head_size}),
                                                      ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                                      ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
        }
        res.get_node()->get_rt_info()[GGML_OV_V_TOKEN_MAJOR] = true;
    } else if (op_case == 1 || context.is_stateful()) {
        if (op_case == 2 && src.get_partial_shape().rank() == 3) {
            // Stateful RoPE lifts Q to rank 4 (see translate_rope). Do the same for Q of a layer without RoPE (NoPE)
            src = std::make_shared<ov::op::v0::Unsqueeze>(src,
                                                          ov::op::v0::Constant::create(ov::element::i64, Shape{}, {0}));
        }
        // The stateful path carries hidden-state tensors in a rank-3 layout (the
        // leading batch dim is dropped, e.g. Gemma4's per-layer-embedding path). The
        // perm above is rank-4; when the actual input is rank-3, drop the batch axis
        // (perm[0], which is always the identity 0 here) and shift the rest down by 1
        // so the transpose order matches the input rank.
        std::vector<int64_t> perm_used = perm_values;
        const auto & src_ps = src.get_partial_shape();
        if (src_ps.rank().is_static() && src_ps.rank().get_length() == 3 && perm_values.size() == 4 &&
            perm_values[0] == 0) {
            perm_used = {perm_values[1] - 1, perm_values[2] - 1, perm_values[3] - 1};
        }
        auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{perm_used.size()}, perm_used);
        res = std::make_shared<ov::op::v1::Transpose>(src, perm);
    } else if (op_case == 2) {
        auto perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_values);
        auto output_shape = context.get_output_shape().to_shape();
        auto n_heads = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[1]});
        auto head_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        auto n_seq_active = context.has_input("n_seq_active") ?
                                context.get_input("n_seq_active") :
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[0]});
        auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

        auto new_shape =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, n_heads, head_size}, 0);

        // // Alternative
        // auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        // auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, zero, zero}, 0);

        auto reshaped = std::make_shared<ov::op::v1::Reshape>(src, new_shape, true);
        res = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    } else if ((op_case == 3 || op_case == 4) && is_kvcache_4d_shape(src.get_partial_shape())) {
        // cache [1, ctx, n_heads_kv, head_size] (see set_rows): take the first attention_size tokens with
        // VariadicSplit, the read side of the GPU plugin's StatelessKVFusion. The fused StatelessKV then hands
        // attention a view of those tokens, with no copy
        FRONT_END_OP_CONVERSION_CHECK((perm_values == std::vector<int64_t>{0, 2, 1, 3}),
                                      "Unsupported permutation of a 4-D KV cache");
        Output<Node> attention_size;
        if (!context.has_input("attention_size")) {
            attention_size =
                ov::op::v0::Constant::create(ov::element::i64, {1}, {context.get_output_shape().to_shape()[2]});
        } else if (op_case == 3) {
            attention_size = context.get_input("attention_size");
        } else {
            attention_size = context.get_input("attention_size_swa");
        }
        auto split_lengths = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{attention_size, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1})}, 0);
        auto tokens = std::make_shared<ov::op::v1::VariadicSplit>(
            src, ov::op::v0::Constant::create(ov::element::i64, {}, {1}), split_lengths);
        res = std::make_shared<ov::op::v1::Transpose>(tokens->output(0),
                                                      ov::op::v0::Constant::create(ov::element::i64, {4}, perm_values));
    } else {
        auto perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_values);
        auto cache_shape = src.get_partial_shape();
        auto output_shape = context.get_output_shape().to_shape();
        int64_t head_size = output_shape[3];
        int64_t n_heads = output_shape[1];
        if (op_case == 5 || op_case == 6) {
            head_size = output_shape[2];
            n_heads = output_shape[1];
        }
        int64_t ctx_per_seq = cache_shape[2].is_static() ? cache_shape[2].get_length() : -1;
        int64_t n_seq = cache_shape[1].get_length();

        Output<Node> attention_size;
        if (!context.has_input("attention_size")) {
            // the attended tokens are the last axis of the transposed V cache (`-fa off`)
            const size_t tokens_axis = (op_case == 5 || op_case == 6) ? 3 : 2;
            attention_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[tokens_axis]});
        } else if (op_case == 3 || op_case == 5) {
            attention_size = context.get_input("attention_size");
        } else {
            attention_size = context.get_input("attention_size_swa");
        }

        Output<Node> seq_active_start;
        Output<Node> seq_active_end;
        if (context.has_input("seq_active_start")) {
            seq_active_start = context.get_input("seq_active_start");
            seq_active_end = context.get_input("seq_active_end");
        } else {
            int64_t n_seq_active = output_shape[0];
            size_t offset = *((size_t *) context.get_input_op_params(0));
            int64_t seq_active_start_val = offset / context.get_input_stride(0)[0];
            int64_t seq_active_end_val = seq_active_start_val + n_seq_active;
            seq_active_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_active_start_val});
            seq_active_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_active_end_val});
        }

        // 1. reshape to [n_seq, ctx_per_seq, n_heads, head_size] (for `-fa off` [n_seq, n_heads, head_size, ctx_per_seq])
        // 2. slice out the active sequences
        // 3. slice out the attention part in each sequence
        // 4. permute (skip for `-fa off`)
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        if ((op_case == 3 || op_case == 4) && n_seq == 1 && perm_values == std::vector<int64_t>{0, 2, 1, 3}) {
            // one sequence: slice the cache [1, 1, ctx, n_heads * head_size] on its context axis, then split the
            // heads. The reshape after the slice also restates the head count and head size: the GPU plugin loses
            // all dimensions of a slice with a runtime end, and SDPA would then run its reference kernel
            auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
            auto sliced = std::make_shared<ov::op::v8::Slice>(src, zero, attention_size, one, two);
            if (n_heads == 1) {
                res = std::make_shared<ov::op::v1::Reshape>(
                    sliced,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, head_size}),
                    false);
            } else {
                auto split = std::make_shared<ov::op::v1::Reshape>(
                    sliced,
                    ov::op::v0::Constant::create(ov::element::i64, {4},
                                                 std::vector<int64_t>{1, -1, n_heads, head_size}),
                    false);
                res = std::make_shared<ov::op::v1::Transpose>(split, perm);
            }
        } else if (op_case == 3 || op_case == 4) {
            auto src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, ctx_per_seq, n_heads, head_size}),
                false);
            ov::Output<ov::Node> after_seq_slice;
            if (n_seq == 1) {
                after_seq_slice = src_reshaped;
            } else {
                after_seq_slice =
                    std::make_shared<ov::op::v8::Slice>(src_reshaped, seq_active_start, seq_active_end, one, zero);
            }
            auto slice2 = std::make_shared<ov::op::v8::Slice>(after_seq_slice, zero, attention_size, one, one);
            // restate the head count and head size, see above
            auto restated = std::make_shared<ov::op::v1::Reshape>(
                slice2,
                ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, -1, n_heads, head_size}),
                true);
            res = std::make_shared<ov::op::v1::Transpose>(restated, perm);
        } else {
            auto three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            // the transposed V cache holds one row per value element, [n_seq, rows, ctx]. A model whose layers have
            // different V sizes allocates the rows for the widest layer (gemma-4: 512, sliding-window layers use the
            // first 256), so keep this layer's n_heads * head_size rows before splitting them into heads
            const int64_t cache_rows = cache_shape[3].get_length();
            ov::Output<ov::Node> v_rows = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {3}, {n_seq, cache_rows, ctx_per_seq}), false);
            if (cache_rows != n_heads * head_size) {
                v_rows = std::make_shared<ov::op::v8::Slice>(
                    v_rows, zero, ov::op::v0::Constant::create(ov::element::i64, {1}, {n_heads * head_size}), one, one);
            }
            auto src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                v_rows, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, n_heads, head_size, ctx_per_seq}),
                false);
            ov::Output<ov::Node> after_seq_slice;
            if (n_seq == 1) {
                after_seq_slice = src_reshaped;
            } else {
                after_seq_slice =
                    std::make_shared<ov::op::v8::Slice>(src_reshaped, seq_active_start, seq_active_end, one, zero);
            }
            auto slice2 = std::make_shared<ov::op::v8::Slice>(after_seq_slice, zero, attention_size, one, three);
            res = slice2;
        }
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
