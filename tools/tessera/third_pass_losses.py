#!/usr/bin/env python3
"""Training losses for KV-compression-aware Engram and draft adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


def _mean_square(student: mx.array, teacher: mx.array) -> mx.array:
    return mx.mean(mx.square(student.astype(mx.float32) - teacher.astype(mx.float32)))


def logit_kl(
    student: mx.array,
    teacher: mx.array,
    temperature: float = 1.0,
) -> mx.array:
    student_scaled = student.astype(mx.float32) / temperature
    teacher_scaled = teacher.astype(mx.float32) / temperature
    teacher_log_probs = teacher_scaled - mx.logsumexp(
        teacher_scaled, axis=-1, keepdims=True
    )
    student_log_probs = student_scaled - mx.logsumexp(
        student_scaled, axis=-1, keepdims=True
    )
    teacher_probs = mx.exp(teacher_log_probs)
    return (
        mx.mean(
            mx.sum(
                teacher_probs * (teacher_log_probs - student_log_probs),
                axis=-1,
            )
        )
        * temperature
        * temperature
    )


def next_token_cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    logits = logits.astype(mx.float32)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    selected = mx.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    return -mx.mean(selected)


def draft_acceptance_loss(
    target_logits: mx.array,
    draft_logits: mx.array,
) -> mx.array:
    target_log_probs = target_logits.astype(mx.float32) - mx.logsumexp(
        target_logits.astype(mx.float32), axis=-1, keepdims=True
    )
    draft_log_probs = draft_logits.astype(mx.float32) - mx.logsumexp(
        draft_logits.astype(mx.float32), axis=-1, keepdims=True
    )
    target_probs = mx.exp(target_log_probs)
    draft_probs = mx.exp(draft_log_probs)
    overlap = mx.sum(mx.minimum(target_probs, draft_probs), axis=-1)
    return -mx.mean(mx.log(mx.maximum(overlap, 1e-8)))


@dataclass(frozen=True)
class LossWeights:
    next_token: float = 1.0
    logit_kl: float = 1.0
    hidden: float = 0.25
    attention: float = 0.25
    draft_acceptance: float = 0.5
    router_kl: float = 1.0
    router_margin: float = 0.5
    expert_output: float = 0.5


def router_margin_loss(
    student: mx.array,
    teacher: mx.array,
    top_k: int,
) -> mx.array:
    if top_k <= 0 or top_k >= teacher.shape[-1]:
        raise ValueError("router top-k must be between zero and the expert count")
    teacher_order = mx.argsort(teacher.astype(mx.float32), axis=-1)
    selected = teacher_order[..., -top_k:]
    rejected = teacher_order[..., -top_k - 1:-top_k]
    selected_student = mx.take_along_axis(
        student.astype(mx.float32), selected, axis=-1
    )
    rejected_student = mx.take_along_axis(
        student.astype(mx.float32), rejected, axis=-1
    )
    teacher_selected = mx.take_along_axis(
        teacher.astype(mx.float32), selected, axis=-1
    )
    teacher_rejected = mx.take_along_axis(
        teacher.astype(mx.float32), rejected, axis=-1
    )
    required_margin = mx.maximum(
        mx.min(teacher_selected, axis=-1, keepdims=True) - teacher_rejected,
        0.0,
    )
    student_margin = mx.min(
        selected_student, axis=-1, keepdims=True
    ) - rejected_student
    return mx.mean(mx.maximum(required_margin - student_margin, 0.0))


def joint_loss(
    student: dict[str, mx.array],
    teacher: dict[str, mx.array],
    targets: mx.array,
    weights: LossWeights = LossWeights(),
) -> tuple[mx.array, dict[str, mx.array]]:
    terms = {
        "next_token": next_token_cross_entropy(student["logits"], targets),
        "logit_kl": logit_kl(student["logits"], teacher["logits"]),
        "hidden": _mean_square(student["hidden"], teacher["hidden"]),
        "attention": _mean_square(student["attention"], teacher["attention"]),
        "draft_acceptance": draft_acceptance_loss(
            student["logits"], student["draft_logits"]
        ),
    }
    if "router_logits" in student and "router_logits" in teacher:
        terms["router_kl"] = logit_kl(
            student["router_logits"], teacher["router_logits"]
        )
        terms["router_margin"] = router_margin_loss(
            student["router_logits"],
            teacher["router_logits"],
            int(teacher.get("router_top_k", 1)),
        )
    if "expert_output" in student and "expert_output" in teacher:
        terms["expert_output"] = _mean_square(
            student["expert_output"], teacher["expert_output"]
        )
    total = (
        weights.next_token * terms["next_token"]
        + weights.logit_kl * terms["logit_kl"]
        + weights.hidden * terms["hidden"]
        + weights.attention * terms["attention"]
        + weights.draft_acceptance * terms["draft_acceptance"]
    )
    if "router_kl" in terms:
        total = (
            total
            + weights.router_kl * terms["router_kl"]
            + weights.router_margin * terms["router_margin"]
        )
    if "expert_output" in terms:
        total = total + weights.expert_output * terms["expert_output"]
    return total, terms
