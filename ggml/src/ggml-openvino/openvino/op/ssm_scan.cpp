// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from OpenVINO's GGUF frontend (src/frontends/gguf/src/op/ssm_scan.cpp).

#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
#include "selective_ssm.hpp"

#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/softplus.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_ssm_scan(const NodeContext & context) {
    num_inputs_check(context, 7, 7);
    using namespace ov::op;

    // ggml: s [slots,H,D,S], x [B,T,H,D], dt [1,B,T,H], A [1,1,H,1],
    // B/C [B,T,G,S], ids [1,1,1,B]. Only Mamba2 has scalar decay per head.
    const auto a_shape = context.get_input_shape(3);
    FRONT_END_OP_CONVERSION_CHECK(a_shape.rank() == 4 && a_shape[0] == 1 && a_shape[1] == 1 && a_shape[3] == 1,
                                  "SSM_SCAN: SelectiveSSM requires Mamba2 scalar decay per head; "
                                  "Mamba1 state-wise decay is not supported");
    const auto x = process_view_input_new(context, 1);
    const auto flat = v0::Constant::create(element::i64, {1}, {-1});
    const auto axis0 = v0::Constant::create(element::i64, {}, {0});
    const auto ids = std::make_shared<v1::Reshape>(process_view_input_new(context, 6), flat, false);
    const auto state = std::make_shared<v8::Gather>(process_view_input_new(context, 0), ids, axis0);
    const auto a = std::make_shared<v1::Reshape>(process_view_input_new(context, 3), flat, false);
    const auto dt_shape = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(x),
                                                       v0::Constant::create(element::i64, {3}, {0, 1, 2}),
                                                       axis0);
    const auto dt =
        std::make_shared<v4::SoftPlus>(std::make_shared<v1::Reshape>(process_view_input_new(context, 2), dt_shape, false));
    const auto ssm = std::make_shared<internal::SelectiveSSM>(a,
                                                              dt,
                                                              process_view_input_new(context, 4),
                                                              x,
                                                              process_view_input_new(context, 5),
                                                              state);
    // Pack directly in GGML layout so whole-part views can select either Concat input.
    const auto packed_shape = v0::Constant::create(element::i64, {4}, {1, 1, 1, -1});
    const auto y = std::make_shared<v1::Reshape>(ssm->output(0), packed_shape, false);
    const auto s = std::make_shared<v1::Reshape>(ssm->output(1), packed_shape, false);
    const auto result = std::make_shared<v0::Concat>(OutputVector{y, s}, 3);
    result->get_rt_info()[GGML_OV_SSM_SCAN_PACKED] = true;
    return rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

