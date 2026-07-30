#!/usr/bin/env python3

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from tools.tessera.kv_compression_mlx import (
    SUPPORTED_TYPES,
    hadamard,
    quantize_dequantize,
    straight_through_cache,
)
from tools.tessera.third_pass_losses import (
    LossWeights,
    joint_loss,
    logit_kl,
)
from tools.tessera.third_pass_trainer import ThirdPassTrainer


REFERENCE = os.environ.get("TESSERA_KV_REFERENCE")


class KVCompressionTests(unittest.TestCase):
    def test_hadamard_is_orthonormal(self):
        values = mx.arange(128, dtype=mx.float32).reshape(2, 64) / 17
        restored = hadamard(hadamard(values, 64), 64)
        np.testing.assert_allclose(np.array(restored), np.array(values), atol=2e-5)

    @unittest.skipUnless(REFERENCE, "native KV reference executable is not configured")
    def test_mlx_matches_native_reference(self):
        rng = np.random.default_rng(20260728)
        random_values = (rng.standard_normal((5, 128)) * 1.7).astype(np.float32)
        boundaries = np.stack(
            (
                np.zeros(128, dtype=np.float32),
                np.linspace(-4, 4, 128, dtype=np.float32),
                np.tile(
                    np.array([-3, 3, -3, 3], dtype=np.float32),
                    32,
                ),
            )
        )
        values = np.concatenate((random_values, boundaries), axis=0)
        for rotation in (0, 64):
            for cache_type in sorted(SUPPORTED_TYPES - {"f16", "bf16"}):
                with self.subTest(rotation=rotation, cache_type=cache_type):
                    with tempfile.TemporaryDirectory() as directory:
                        input_path = Path(directory) / "input.raw"
                        output_path = Path(directory) / "output.raw"
                        values.tofile(input_path)
                        subprocess.run(
                            [
                                REFERENCE,
                                cache_type,
                                str(values.shape[0]),
                                "128",
                                str(rotation),
                                input_path,
                                output_path,
                            ],
                            check=True,
                        )
                        native = np.fromfile(output_path, dtype=np.float32).reshape(
                            values.shape
                        )
                    simulated = mx.array(values)
                    if rotation:
                        simulated = hadamard(simulated, rotation)
                    simulated = np.array(
                        quantize_dequantize(simulated, cache_type)
                    )
                    np.testing.assert_array_equal(simulated, native)

    def test_straight_through_cache_has_finite_gradient(self):
        values = mx.arange(64, dtype=mx.float32).reshape(1, 64) / 19
        gradient = mx.grad(
            lambda source: mx.sum(straight_through_cache(source, "q4_1", 64))
        )(values)
        self.assertTrue(np.all(np.isfinite(np.array(gradient))))
        self.assertGreater(float(np.max(np.abs(np.array(gradient)))), 0)


class ThirdPassLossTests(unittest.TestCase):
    def fixtures(self):
        teacher_logits = mx.array(
            [[[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]]], dtype=mx.float32
        )
        student_logits = teacher_logits + mx.array(
            [[[0.1, -0.1, 0.0], [-0.1, 0.0, 0.1]]], dtype=mx.float32
        )
        teacher = {
            "logits": teacher_logits,
            "hidden": mx.ones((1, 2, 4)),
            "attention": mx.ones((1, 2, 4)) * 0.5,
        }
        student = {
            "logits": student_logits,
            "draft_logits": student_logits - 0.2,
            "hidden": teacher["hidden"] + 0.1,
            "attention": teacher["attention"] - 0.1,
        }
        return student, teacher, mx.array([[0, 1]])

    def test_identical_logits_have_zero_kl(self):
        _, teacher, _ = self.fixtures()
        self.assertLess(float(logit_kl(teacher["logits"], teacher["logits"])), 1e-6)

    def test_joint_loss_reports_all_terms_and_gradients(self):
        student, teacher, targets = self.fixtures()
        total, terms = joint_loss(student, teacher, targets, LossWeights())
        mx.eval(total, terms)
        self.assertTrue(np.isfinite(float(total)))
        self.assertEqual(
            set(terms),
            {"next_token", "logit_kl", "hidden", "attention", "draft_acceptance"},
        )
        gradient = mx.grad(
            lambda logits: joint_loss(
                {**student, "logits": logits},
                teacher,
                targets,
            )[0]
        )(student["logits"])
        self.assertTrue(np.all(np.isfinite(np.array(gradient))))

    def test_training_harness_updates_a_model(self):
        class ToyStudent(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(4, 3)

            def __call__(self, batch):
                logits = self.projection(batch["hidden"])
                return {
                    "logits": logits,
                    "draft_logits": logits - 0.1,
                    "hidden": batch["hidden"],
                    "attention": batch["attention"],
                }

        model = ToyStudent()
        trainer = ThirdPassTrainer(
            model,
            optim.AdamW(learning_rate=1e-3),
        )
        batch = {
            "hidden": mx.ones((1, 2, 4)),
            "attention": mx.ones((1, 2, 4)) * 0.5,
            "targets": mx.array([[0, 1]]),
        }
        teacher = {
            "logits": mx.zeros((1, 2, 3)),
            "hidden": batch["hidden"],
            "attention": batch["attention"],
        }
        metrics = trainer.step(batch, teacher)
        self.assertEqual(metrics["step"], 1)
        self.assertTrue(np.isfinite(metrics["loss"]))


if __name__ == "__main__":
    unittest.main()
