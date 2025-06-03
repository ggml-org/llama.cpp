#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/transpose.hpp>
#include <vector>

#include "../node_context.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_mulmat(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2, "Unsupported MULMAT case");

    ov::Output<Node> res;

    if (op_case == 1) {
        auto src0 = context.get_input(0);
        auto src1 = std::make_shared<ov::op::v0::Convert>(context.get_input(1), context.get_input_type(0));
        auto result_lp = std::make_shared<ov::op::v0::MatMul>(src1, src0, false, true);
        res = std::make_shared<ov::op::v0::Convert>(result_lp, context.get_output_type(0));
    } else {
        /*
        Two cases here:
        -  21: [    96,    32,    32,     1] VIEW                 k-0                 [ 2,  6144,   192,  6144]
                [ 196608,     1,     1,     1]          0: NONE        cache_k_l0         [ 2, 393216, 393216, 393216]
        -  22: [    96,     7,    32,     1] PERMUTE              q-0                 [ 4, 12288,   384, 86016]
                [    96,    32,     7,     1]          0: SCALE       Qcur-0             [ 4,   384, 12288, 86016]
        -  23: [    32,     7,    32,     1] MUL_MAT              kq-0                [ 4,   128,   896, 28672]
                [    96,    32,    32,     1]          0: VIEW        k-0                [ 2,  6144,   192,  6144]
                [    96,     7,    32,     1]          1: PERMUTE     q-0                [ 4, 12288,   384, 86016]

        -  20: [    32,    96,    32,     1] VIEW                 v-0                 [ 2,   128, 12288, 393216]
                [ 196608,     1,     1,     1]          0: NONE        cache_v_l0         [ 2, 393216, 393216, 393216]
        -  25: [    96,     7,    32,     1] MUL_MAT              kqv-0               [ 4,   384,  2688, 86016]
                [    32,    96,    32,     1]          0: VIEW        v-0                [ 2,   128, 12288, 393216]
                [    32,     7,    32,     1]          1: SOFT_MAX    kq_soft_max_ext-0  [ 4,   128,   896, 28672]

        For case 1, for src0, Reshape + Slice + Transpose
        For case 2, for src0, Reshape + Slice
        */
        ov::Output<ov::Node> A;
        ov::Output<ov::Node> B;

        auto src0 = context.get_input(0);
        auto src0_shape = context.get_input_shape(0).to_shape();
        auto src0_stride = context.get_input_stride(0);
        auto permuted = is_permuted(src0_stride);
        auto token_dim = permuted ? 0 : 2;

        auto attention_size = context.get_input("attention_size");

        auto src0_perm = argsort_descend(src0_stride);
        auto src0_original_shape_ = permute(src0_shape, src0_perm);
        std::vector<int64_t> src0_original_shape(src0_original_shape_.begin(), src0_original_shape_.end());

        if (context.is_static()) {
            attention_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX});
        }
        src0_original_shape[token_dim] = -1;

        auto src0_slice_shape = src0_original_shape;
        src0_slice_shape.erase(src0_slice_shape.begin() + token_dim);

        auto src0_reshape_shape =
            ov::op::v0::Constant::create(ov::element::i64, {src0_original_shape.size()}, src0_original_shape);
        auto src0_reshape = std::make_shared<ov::op::v1::Reshape>(src0, src0_reshape_shape, false);

        std::shared_ptr<ov::Node> slice_end;
        if (permuted) {
            slice_end = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{attention_size, ov::op::v0::Constant::create(ov::element::i64, {2}, src0_slice_shape)},
                0);
        } else {
            slice_end = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{ov::op::v0::Constant::create(ov::element::i64, {2}, src0_slice_shape), attention_size},
                0);
        }
        auto slice_start = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>(3, 0));
        auto slice_step = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>(3, 1));
        auto src0_slice = std::make_shared<ov::op::v8::Slice>(src0_reshape, slice_start, slice_end, slice_step);

        if (permuted) {
            B = std::make_shared<ov::op::v1::Transpose>(
                src0_slice,
                ov::op::v0::Constant::create(ov::element::i64, {src0_perm.size()}, src0_perm));
        } else {
            B = src0_slice;
        }

        A = std::make_shared<ov::op::v0::Convert>(context.get_input(1), context.get_input_type(0));

        int64_t num_heads = context.get_input_shape(1).to_shape()[0];
        int64_t num_heads_kv = src0_shape[0];
        int64_t kv_num_heads_factor = num_heads / num_heads_kv;
        if (kv_num_heads_factor > 1) {
            auto num_heads_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{num_heads});
            auto num_heads_kv_node =
                ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{num_heads_kv});
            auto B_shape_last_two = get_dimensions(B.get_node_shared_ptr(), {1, 2});

            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            std::shared_ptr<ov::Node> new_B_shape =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector{num_heads_kv_node, one, B_shape_last_two}, 0);
            B = std::make_shared<ov::op::v1::Reshape>(B, new_B_shape, false);

            B = std::make_shared<ov::op::v0::Concat>(ov::OutputVector(kv_num_heads_factor, B), 1);
            new_B_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{num_heads_node, B_shape_last_two}, 0);
            B = std::make_shared<ov::op::v1::Reshape>(B, new_B_shape, false);
        }

        auto result_lp = std::make_shared<ov::op::v0::MatMul>(A, B, false, true);
        res = std::make_shared<ov::op::v0::Convert>(result_lp, context.get_output_type(0));
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
