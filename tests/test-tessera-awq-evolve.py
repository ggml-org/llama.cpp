#!/usr/bin/env python3

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tools" / "tessera" / "awq-evolve.py"
SPEC = importlib.util.spec_from_file_location("tessera_awq_evolve", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
import sys
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TesseraEvolutionTest(unittest.TestCase):
    def make_layer(self):
        rng = np.random.default_rng(7)
        weight = rng.normal(size=(12, 16)).astype(np.float32)
        activations = rng.normal(size=(32, 16)).astype(np.float32)
        activations[:, 3] *= 8.0
        second = np.mean(np.square(activations), axis=0)
        fourth = np.mean(np.power(activations, 4), axis=0)
        return MODULE.Layer(
            "blk.0.attn_q.weight",
            "attention",
            weight,
            activations[:24],
            activations[24:],
            second,
            fourth,
            np.max(np.abs(activations), axis=0),
        )

    def test_transfer_ledger_prioritizes_late_survivors_and_tracks_savings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transfer.json"
            path.write_text(json.dumps({
                "schema": "llama.tessera.progressive-observer-ledger.v1",
                "checkpoint_chunk": 128,
                "tensors": {
                    "early.weight": {"frozen": True, "frozen_at": 32},
                    "late.weight": {"frozen": True, "frozen_at": 112},
                    "active.weight": {"frozen": False},
                },
            }))
            priorities, saved = MODULE.load_transfer_prior(path)
        self.assertLess(priorities["early.weight"], priorities["late.weight"])
        self.assertEqual(priorities["active.weight"], 1.0)
        self.assertGreater(saved, 0.0)

    def test_evolution_is_deterministic_and_finite(self):
        layer = self.make_layer()
        first, first_score, details = MODULE.evolve([layer], 3, 6, 2, 17, 0.01, None)
        second, second_score, _ = MODULE.evolve([layer], 3, 6, 2, 17, 0.01, None)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first_score.fitness, second_score.fitness)
        self.assertTrue(np.isfinite(first_score.fitness))
        self.assertGreater(len(details["archive"]), 0)

    def test_reconstruction_bounds_degenerate_importance_channels(self):
        layer = self.make_layer()
        importance = layer.second_moment.copy()
        importance[:5] = [0.0, 1e-30, 1e-12, np.inf, np.nan]
        candidate = MODULE.Candidate(1.0, 0.9, 0.005, 0.2, 0.3)
        reconstructed = MODULE._ternary_reconstruct(
            layer.weight,
            candidate,
            importance,
        )
        self.assertTrue(np.all(np.isfinite(reconstructed)))

    def test_checkpoint_resume_matches_full_run(self):
        layer = self.make_layer()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.json"
            MODULE.evolve([layer], 2, 6, 2, 29, 0.01, checkpoint)
            resumed, resumed_score, _ = MODULE.evolve([layer], 4, 6, 2, 29, 0.01, checkpoint)
            full, full_score, _ = MODULE.evolve([layer], 4, 6, 2, 29, 0.01, None)
            self.assertEqual(resumed, full)
            self.assertAlmostEqual(resumed_score.fitness, full_score.fitness)

    def test_progressive_evaluation_promotes_only_a_subset(self):
        prototype = self.make_layer()
        layers = []
        for index in range(8):
            layers.append(MODULE.Layer(
                f"blk.{index}.attn_q.weight",
                prototype.family,
                prototype.weight * (1.0 + index * 0.01),
                prototype.train_activations,
                prototype.heldout_activations,
                prototype.second_moment,
                prototype.fourth_moment,
                prototype.max_abs,
            ))
        population = [
            MODULE.Candidate(0.1 + index * 0.05, 0.9, 0.005, 0.2, 0.3)
            for index in range(12)
        ]
        config = MODULE.ProgressiveConfig(
            screen_fraction=0.25,
            refine_fraction=0.50,
            promotion_margin=0.0,
            diversity_slots=0,
        )
        cache = {}
        validated, work = MODULE.progressive_evaluate_population(
            population, layers, config, cache)
        self.assertEqual(work["screened"], len(population))
        self.assertLess(work["validated"], len(population))
        self.assertEqual(len(validated), work["validated"])
        cached, cached_work = MODULE.progressive_evaluate_population(
            population, layers, config, cache)
        self.assertEqual(validated, cached)
        self.assertEqual(work, cached_work)

    def test_policy_is_accepted_by_existing_schema(self):
        candidate = MODULE.Candidate(0.4, 0.9, 0.005, 0.2, 0.3)
        score = MODULE.Score(1.0, 1.0, 1.0, 0.005, 3.0)
        override = MODULE.Candidate(0.6, 0.95, 0.01, 0.4, 0.7)
        policy = MODULE.build_policy(
            {"attention": (candidate, score)},
            {"seed": 1},
            overrides={"blk.7.attn_q.weight": (override, score)},
        )
        self.assertEqual(policy["schema"], "llama.speculative.calibration-policy.v1")
        self.assertEqual(policy["tensor_families"]["attention"]["awq_alpha"], 0.4)
        entries = list(policy["tensor_families"].values())
        self.assertEqual(entries[0]["match"], ["blk.7.attn_q.weight"])
        self.assertEqual(entries[0]["awq_alpha"], 0.6)

    def test_fitness_penalizes_a_sensitive_layer(self):
        ordinary = self.make_layer()
        sensitive = self.make_layer()
        sensitive.name = "blk.31.attn_q.weight"
        sensitive.second_moment *= 100.0
        candidate = MODULE.Candidate(0.4, 0.9, 0.005, 0.2, 0.3)
        score = MODULE.evaluate(candidate, [ordinary, sensitive])
        no_guard = MODULE.evaluate(
            candidate,
            [ordinary, sensitive],
            worst_layer_weight=0.0,
        )
        self.assertGreater(score.worst_layer_error, 0.0)
        self.assertGreater(score.fitness, no_guard.fitness)

    def test_cached_layer_scores_match_uncached_evaluation(self):
        first = self.make_layer()
        second = self.make_layer()
        second.name = "blk.31.attn_q.weight"
        second.second_moment *= 1.25
        candidate = MODULE.Candidate(0.4, 0.9, 0.005, 0.2, 0.3)
        uncached = MODULE._evaluate_uncached(candidate, [first, second])
        cache = {}
        score = MODULE._cached_evaluate(candidate, [first], "screen", cache)
        full = MODULE._cached_evaluate(candidate, [first, second], "full", cache)
        self.assertTrue(np.isfinite(score.fitness))
        self.assertAlmostEqual(full.fitness, uncached.fitness, places=7)
        entries = cache[MODULE._candidate_key(candidate)]
        self.assertEqual(len([key for key in entries if key.startswith("layer:")]), 2)
        checkpoint_cache = MODULE._checkpoint_score_cache(cache)
        self.assertTrue(any(
            key.startswith("layer:")
            for entry in checkpoint_cache.values()
            for key in entry
        ))

    def test_batched_mlx_projection_scores_match_scalar_scores(self):
        layer = self.make_layer()
        candidates = [
            MODULE.Candidate(0.1 + index * 0.13, 0.82 + index * 0.03,
                             0.002 + index * 0.001, 0.2, 0.3 + index * 0.1)
            for index in range(4)
        ]
        expected = [MODULE._evaluate_layer(candidate, layer) for candidate in candidates]
        actual = MODULE._evaluate_layer_batch(candidates, layer)
        for scalar, batched in zip(expected, actual):
            self.assertAlmostEqual(scalar.train_error, batched.train_error, places=6)
            self.assertAlmostEqual(scalar.heldout_error, batched.heldout_error, places=6)
            self.assertAlmostEqual(
                scalar.tail_error, batched.tail_error, delta=1e-5)

    def test_batched_population_cache_matches_scalar_cache(self):
        first = self.make_layer()
        second = self.make_layer()
        second.name = "blk.31.attn_q.weight"
        candidates = [
            MODULE.Candidate(0.1 + index * 0.1, 0.9, 0.004, 0.2, 0.3)
            for index in range(5)
        ]
        batched = MODULE._cached_evaluate_population(
            candidates, [first, second], "full", {}, 3)
        scalar = [MODULE._cached_evaluate(candidate, [first, second], "full", {})
                  for candidate in candidates]
        for left, right in zip(batched, scalar):
            self.assertAlmostEqual(left.fitness, right.fitness, places=6)

    def test_residual_allocator_honors_global_budget(self):
        ordinary = self.make_layer()
        sensitive = self.make_layer()
        sensitive.name = "blk.31.attn_q.weight"
        sensitive.weight[:, 3] *= 8.0
        candidate = MODULE.Candidate(0.4, 0.9, 0.005, 0.2, 0.3)
        allocated, record = MODULE.allocate_residual_budget(
            [ordinary, sensitive],
            {},
            candidate,
            0.005,
        )
        mean_fraction = sum(
            item[0].outlier_fraction for item in allocated.values()
        ) / len(allocated)
        self.assertLessEqual(mean_fraction, 0.005 + 1e-12)
        self.assertLessEqual(record["allocated_fraction"], 0.005 + 1e-12)
        self.assertGreaterEqual(
            allocated[sensitive.name][0].outlier_fraction,
            allocated[ordinary.name][0].outlier_fraction,
        )

    def test_cli_writes_resumable_policy(self):
        layer = self.make_layer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layers = root / "layers"
            layers.mkdir()
            np.savez(
                layers / "attention.npz",
                name=np.asarray(layer.name),
                family=np.asarray(layer.family),
                weight=layer.weight,
                train_activations=layer.train_activations,
                heldout_activations=layer.heldout_activations,
            )
            np.savez(
                layers / "attention-sensitive.npz",
                name=np.asarray("blk.31.attn_q.weight"),
                family=np.asarray(layer.family),
                weight=layer.weight * 1.5,
                train_activations=layer.train_activations,
                heldout_activations=layer.heldout_activations,
            )
            output = root / "policy.json"
            checkpoint = root / "checkpoint.json"
            subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--layers", str(layers),
                    "--output", str(output),
                    "--checkpoint", str(checkpoint),
                    "--generations", "2",
                    "--population", "4",
                    "--islands", "2",
                    "--max-rows", "16",
                    "--max-tokens", "32",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            policy = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(policy["schema"], "llama.speculative.calibration-policy.v1")
            self.assertIn("attention", policy["tensor_families"])
            self.assertTrue(any(
                name.startswith("override:")
                for name in policy["tensor_families"]
            ))
            self.assertTrue(checkpoint.with_suffix(".attention.json").exists())


if __name__ == "__main__":
    unittest.main()
