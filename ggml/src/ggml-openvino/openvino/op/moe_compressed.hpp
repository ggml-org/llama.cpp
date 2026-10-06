// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Local mirror of OpenVINO's internal ov::op::internal::MOE and MOECompressed ops.
//
// The class bodies are provided by the linked libopenvino.so; only the declarations are
// needed here so the backend can construct the node directly (same approach as
// GatherMatmul and GatedDeltaNet). The class layout must stay in sync with
//   openvino/src/core/dev_api/openvino/op/moe.hpp
//   openvino/src/common/transformations/include/ov_ops/moe_compressed.hpp
//
// \note MOE op classes are under development and subject to change.

#pragma once

#include <optional>

#include "openvino/core/type/element_type.hpp"
#include "openvino/op/op.hpp"

namespace ov::op::internal {

class OPENVINO_API MOE : public ov::op::Op {
public:
    OPENVINO_OP("MOE")

    MOE() = default;

    MOE(const OutputVector & args) : Op(args) {}

    enum class Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };

    enum class Activation_type { SWIGLU, GEGLU_TANH, GEGLU_ERF };

    struct Config {
        Expert_type expert_type{ Expert_type::GEMM2_BIAS_SWIGLU_CLAMP };
        float expert_alpha{ 0.0f };
        float expert_beta{ 1.0f };
        size_t gate_idx{ 0 };
        Activation_type activation_type{ Activation_type::SWIGLU };
    };

    MOE(const OutputVector & args, const Config & config);

    const Config & get_config() const;
    void set_config(const Config & config);

    bool visit_attributes(AttributeVisitor & visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector & new_args) const override;

private:
    Config m_config;
};

class OPENVINO_API MOECompressed : public MOE {
public:
    OPENVINO_OP("MOECompressed", "", ov::op::internal::MOE)

    MOECompressed() = default;

    struct Config : public MOE::Config {
        size_t hidden_size = 0;
        size_t inter_size = 0;
        size_t num_expert = 0;
        size_t num_shared_expert = 0;
        size_t top_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group)
        size_t group_size = 0;
        bool has_batch_dim = false;
        bool has_zp = false;
        ov::element::Type out_type = ov::element::dynamic;
        std::optional<float> scale_factor;
    };

    MOECompressed(const OutputVector & args, const Config & config);

    const Config & get_config() const { return m_config; }

    void set_scale_factor(float scale_factor) { m_config.scale_factor = scale_factor; }

    bool visit_attributes(AttributeVisitor & visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector & new_args) const override;

protected:
    Config m_config;
};

}  // namespace ov::op::internal
