// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::internal {
/// \note GatedDeltaNet op class is under development and subject to change
///
/// \brief Operator performing Gated Delta Net computation
/// \ingroup ov_ops_cpp_api
class OPENVINO_API GatedDeltaNet : public ov::op::Op {
public:
    OPENVINO_OP("GatedDeltaNet");

    GatedDeltaNet() = default;
    /// \brief Constructs a GatedDeltaNet operation.
    ///
    /// \param query Query tensor input.
    /// \param key Key tensor input.
    /// \param value Value tensor input.
    /// \param recurrent_state Initial recurrent state tensor.
    /// \param gate Gate tensor controlling state decay/update.
    /// \param beta Beta tensor scaling the delta update.
    /// \param fuse_qk_l2norm Enables fusing q/k L2-normalization into this op.
    /// \param q_l2_norm_eps Epsilon used for query L2-normalization when fusion is enabled.
    /// \param k_l2_norm_eps Epsilon used for key L2-normalization when fusion is enabled.
    GatedDeltaNet(const Output<Node>& query,
                  const Output<Node>& key,
                  const Output<Node>& value,
                  const Output<Node>& recurrent_state,
                  const Output<Node>& gate,
                  const Output<Node>& beta,
                  const bool fuse_qk_l2norm = false,
                  const float q_l2_norm_eps = 1e-6F,
                  const float k_l2_norm_eps = 1e-6F);

    /// \brief Constructs a GatedDeltaNet operation from input vector.
    ///
    /// \param args Input tensor vector in order: query, key, value, recurrent_state, gate, beta.
    /// \param fuse_qk_l2norm Enables fusing q/k L2-normalization into this op.
    /// \param q_l2_norm_eps Epsilon used for query L2-normalization when fusion is enabled.
    /// \param k_l2_norm_eps Epsilon used for key L2-normalization when fusion is enabled.
    GatedDeltaNet(const ov::OutputVector& args,
                  const bool fuse_qk_l2norm = false,
                  const float q_l2_norm_eps = 1e-6F,
                  const float k_l2_norm_eps = 1e-6F);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool get_fuse_qk_l2norm() const {
        return m_fuse_qk_l2norm;
    }
    float get_q_l2_norm_eps() const {
        return m_q_l2_norm_eps;
    }
    float get_k_l2_norm_eps() const {
        return m_k_l2_norm_eps;
    }

private:
    bool m_fuse_qk_l2norm = false;
    float m_q_l2_norm_eps = 1e-6F;
    float m_k_l2_norm_eps = 1e-6F;
};

}  // namespace ov::op::internal
