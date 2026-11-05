// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "openvino/op/op.hpp"

namespace ov::op::internal {
/// Qwen3.5/Qwen3.6 llama.cpp convolution and generic cache_r maintenance.
/// Inputs: original cache, active rows, extra rows, reset begin/length,
/// extra offsets, QKV [1,B,C,T], windows and depthwise weights.
/// Nine inputs; output 0 is convolution [1,B,T,C].
/// Cache input 0 is a writable GPU RemoteTensor, updated in place.
/// Other inputs and output 0 must not overlap its allocation.
/// The cache must be a Parameter or an exclusive Reshape alias; no other op
/// may read it in the same inference. Required original rows are staged before
/// reset, reorder and window writes, preserving cycles and duplicate sources.
/// A failed inference may partially modify the cache; clear or restore it before retrying.
class OPENVINO_API RecurrentConvCacheUpdate : public Op {
public:
    OPENVINO_OP("RecurrentConvCacheUpdate", "internal")
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
/// The only output is attention [B,T,H,V]. Cache input 3 is updated in place
/// under the same writable-input contract as RecurrentConvCacheUpdate.
/// snapshot_count saves the newest min(T,count) states in cache, newest first.
/// Active/nonidentity rows are staged before writes; identity rows need no copy.
class OPENVINO_API RecurrentGatedDeltaNetUpdate : public Op {
public:
    OPENVINO_OP("RecurrentGatedDeltaNetUpdate", "internal")
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
