// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "openvino/op/op.hpp"

namespace ov::op::internal {
/// Qwen3.5/Qwen3.6 llama.cpp convolution and generic cache_r maintenance.
/// Inputs: original cache, active rows, extra rows, reset begin/length,
/// extra offsets, QKV [1,B,C,T], windows and depthwise weights.
/// Nine inputs; output 0 is convolution [1,B,T,C].
/// Output 1 is the complete updated [1,1,S,C*history] cache.
/// GPU execution may reuse a caller's identical input/output USM allocation only
/// when it is the exclusive cache reader. Required original rows are gathered into
/// device scratch before any reset, reorder or window write, preserving cycles,
/// duplicate sources and later-window precedence. Other bindings retain immutable
/// input semantics. Preserved rows retain the existing computation-precision rounding.
/// A failed in-place inference may partially modify the cache; the caller must
/// clear or restore it before retrying.
class OPENVINO_API RecurrentConvCacheUpdate : public Op {
public:
    OPENVINO_OP("RecurrentConvCacheUpdate", "internal");
    RecurrentConvCacheUpdate() = default;
    RecurrentConvCacheUpdate(const OutputVector & args, int64_t history);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor &) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector &) const override;

    int64_t get_history() const { return m_history; }

private:
    int64_t m_history = 0;
};

/// Qwen3.5/Qwen3.6 llama.cpp GDN and generic cache_s maintenance.
/// Twelve inputs: Q/K/V, cache, gate/beta,
/// active/extra rows, reset begin/length, extra offsets, ordered write control.
/// Outputs are attention [B,T,H,V] and the complete [1,1,S,H*V*K] updated cache.
/// snapshot_count saves only the newest min(T,count) states, newest first.
/// Required active/nonidentity rows are staged on device before computation;
/// identity reorder sources need no scratch copy.
/// GPU input/output aliasing uses staged original rows as described above;
/// distinct allocations keep the immutable-source implementation. Runtime maps,
/// reset, duplicate sources, cycles and overlapping snapshot writes are supported.
class OPENVINO_API RecurrentGatedDeltaNetUpdate : public Op {
public:
    OPENVINO_OP("RecurrentGatedDeltaNetUpdate", "internal");
    RecurrentGatedDeltaNetUpdate() = default;
    RecurrentGatedDeltaNetUpdate(const OutputVector & args,
                                 bool fuse_qk_l2norm = false,
                                 float q_l2_norm_eps = 1e-6f,
                                 float k_l2_norm_eps = 1e-6f,
                                 int64_t snapshot_count = 1);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor &) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector &) const override;

    bool get_fuse_qk_l2norm() const { return m_fuse_qk_l2norm; }

    float get_q_l2_norm_eps() const { return m_q_l2_norm_eps; }

    float get_k_l2_norm_eps() const { return m_k_l2_norm_eps; }

    int64_t get_snapshot_count() const { return m_snapshot_count; }

private:
    bool m_fuse_qk_l2norm = false;
    float m_q_l2_norm_eps = 1e-6f, m_k_l2_norm_eps = 1e-6f;
    int64_t m_snapshot_count = 1;
};
}  // namespace ov::op::internal
