// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Local mirror of OpenVINO's internal ov::op::internal::GatherMatmul op.
//
// The op class body (validate_and_infer_types / clone_with_new_inputs) is
// provided by the linked libopenvino.so; only the declaration is needed here so
// the backend can construct the node directly (same approach as GatedDeltaNet).
// The class layout must stay in sync with
//   openvino/src/common/transformations/include/ov_ops/gather_matmul.hpp
//
// \note GatherMatmul op class is under development and subject to change.

#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::internal {

class OPENVINO_API GatherMatmul : public ov::op::Op {
public:
    OPENVINO_OP("GatherMatmul")

    GatherMatmul() = default;

    GatherMatmul(const ov::Output<Node>& A,
                 const ov::Output<Node>& B,
                 const ov::Output<Node>& indices,
                 const ov::Output<Node>& bias);

    GatherMatmul(const ov::Output<Node>& A, const ov::Output<Node>& B, const ov::Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    void validate_and_infer_types() override;

private:
    // the weights matrix B is expected to have the transposed form [group, N, K]
    static constexpr bool transp_a = false;
    static constexpr bool transp_b = true;
};

}  // namespace ov::op::internal
