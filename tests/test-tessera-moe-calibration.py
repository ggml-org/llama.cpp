#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from tools.tessera.moe_calibration import (
    RouterAccumulator,
    allocate_expert_residuals,
    coverage_aware_scores,
    residual_policy,
)
from tools.tessera.third_pass_losses import LossWeights, joint_loss


class MoECalibrationTests(unittest.TestCase):
    def test_router_accumulator_keeps_only_sufficient_statistics(self):
        accumulator = RouterAccumulator(layer=3, experts=4)
        accumulator.update(
            np.array([[4.0, 3.0, 1.0, 0.0], [0.0, 1.0, 4.0, 3.0]]),
            top_k=2,
            expert_output_error=np.array([[0.1, 0.2], [0.3, 0.4]]),
        )
        evidence = accumulator.evidence()
        self.assertEqual(sum(item.selected for item in evidence), 4)
        self.assertEqual({item.expert for item in evidence}, {0, 1, 2, 3})
        self.assertFalse(hasattr(evidence[0], "token"))
        self.assertFalse(hasattr(evidence[0], "logits"))

    def test_global_budget_protects_rare_experts(self):
        accumulator = RouterAccumulator(layer=0, experts=4)
        accumulator.update(
            np.tile(np.array([[4.0, 3.0, 1.0, 0.0]]), (32, 1)),
            top_k=2,
        )
        scores = coverage_aware_scores(accumulator.evidence(), prior_strength=8)
        allocation = allocate_expert_residuals(scores, total_fraction=0.01)
        self.assertAlmostEqual(sum(allocation.values()), 0.04, places=8)
        self.assertTrue(all(value >= 0.0001 for value in allocation.values()))
        policy = residual_policy(
            accumulator.evidence(), total_fraction=0.01, prior_strength=8
        )
        self.assertEqual(
            policy["schema"], "llama.tessera.moe-residual-policy.v1"
        )
        self.assertEqual(set(policy["layers"]["0"]["experts"]), {"0", "1", "2", "3"})

    def test_router_repair_terms_are_differentiable(self):
        teacher_logits = mx.array([[[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]]])
        student_logits = teacher_logits + 0.1
        teacher = {
            "logits": teacher_logits,
            "hidden": mx.ones((1, 2, 4)),
            "attention": mx.ones((1, 2, 4)),
            "router_logits": teacher_logits,
            "router_top_k": 1,
            "expert_output": mx.ones((1, 2, 4)),
        }
        student = {
            "logits": student_logits,
            "draft_logits": student_logits,
            "hidden": teacher["hidden"],
            "attention": teacher["attention"],
            "router_logits": teacher_logits - mx.array(
                [[[0.2, 0.0, 0.0], [0.0, 0.2, 0.0]]]
            ),
            "expert_output": teacher["expert_output"] + 0.1,
        }
        targets = mx.array([[0, 1]])
        total, terms = joint_loss(student, teacher, targets, LossWeights())
        mx.eval(total, terms)
        self.assertIn("router_kl", terms)
        self.assertIn("router_margin", terms)
        self.assertIn("expert_output", terms)
        gradient = mx.grad(
            lambda logits: joint_loss(
                {**student, "router_logits": logits},
                teacher,
                targets,
            )[0]
        )(student["router_logits"])
        self.assertTrue(np.all(np.isfinite(np.array(gradient))))


if __name__ == "__main__":
    unittest.main()
