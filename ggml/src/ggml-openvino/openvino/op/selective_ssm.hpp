// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Local mirror of OpenVINO's ov::op::internal::SelectiveSSM declaration
// (src/core/dev_api/openvino/op/selective_ssm.hpp). The implementation comes from the linked
// OpenVINO runtime; the class layout must stay in sync with the runtime's declaration.
#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::internal {
/// Mamba2 selective state-space recurrence (arXiv:2405.21060).
/// dA = exp(A * dt), dtB = dt * B; per token t:
/// dBx_t = x_t (x) dtB_t, state_t = state_{t-1} * dA_t + dBx_t, y_t = sum(state_t * C_t).
/// B and C are grouped and shared across heads: head h reads group h / (num_heads / num_groups).
class OPENVINO_API SelectiveSSM : public ov::op::Op {
public:
    OPENVINO_OP("SelectiveSSM")

    SelectiveSSM() = default;
    /// \param A (Negative) log-decay rates per head [num_heads].
    /// \param dt Per-token, per-head time steps [batch_size, seq_len, num_heads].
    /// \param B Grouped input projection [batch_size, seq_len, num_groups, state_size].
    /// \param x Input hidden states [batch_size, seq_len, num_heads, head_dim].
    /// \param C Grouped output projection [batch_size, seq_len, num_groups, state_size].
    /// \param recurrent_state Initial state [batch_size, num_heads, head_dim, state_size].
    SelectiveSSM(const Output<Node> & A,
                 const Output<Node> & dt,
                 const Output<Node> & B,
                 const Output<Node> & x,
                 const Output<Node> & C,
                 const Output<Node> & recurrent_state);

    /// \param args Input tensor vector in order: A, dt, B, x, C, recurrent_state.
    explicit SelectiveSSM(const ov::OutputVector & args);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor & visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector & new_args) const override;
};

}  // namespace ov::op::internal
