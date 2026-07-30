#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tools" / "tessera" / filename
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORPUS = load_script("tessera_clean_corpus", "build-calibration-corpus.py")
SAMPLER = load_script("tessera_moe_sampler_test", "moe-sampler.py")


class MoESamplerTests(unittest.TestCase):
    def test_clean_room_corpus_is_deterministic_and_balanced(self):
        first = CORPUS.build_records(640)
        second = CORPUS.build_records(640)
        self.assertEqual(first, second)
        self.assertEqual(len(first), sum(CORPUS.CATEGORY_COUNTS.values()))
        self.assertEqual(len({record["id"] for record in first}), len(first))
        self.assertEqual(
            {record["category"] for record in first},
            set(CORPUS.CATEGORY_COUNTS),
        )
        self.assertTrue(all(
            record["origin"] == "tribunus.dev-clean-room-procedural"
            for record in first
        ))

    def test_stratified_sampler_covers_every_category(self):
        records = CORPUS.build_records(640)
        selected = SAMPLER.stratified_take(records, set(), 128, 640, 0)
        self.assertEqual(len(selected), 128)
        self.assertEqual(
            {record["category"] for record in selected},
            set(CORPUS.CATEGORY_COUNTS),
        )

    def test_adaptive_sampler_requires_coverage_and_stability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = CORPUS.build_records(640)
            index = root / "samples.jsonl"
            index.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            state = root / "state.json"
            batch = root / "batch.txt"
            SAMPLER.initialize(argparse.Namespace(
                index=str(index),
                state=str(state),
                batch_output=str(batch),
                seed=640,
                initial_samples=128,
                step_samples=128,
                max_samples=1024,
                minimum_expert_count=16,
                coverage_percentile=5.0,
                stability_p95=0.02,
                stable_rounds=2,
            ))
            original = SAMPLER.observer_snapshot
            SAMPLER.observer_snapshot = lambda path, gguf: (
                np.full(32, 20.0),
                {"blk.0.ffn": [1.0, 0.8, 1.2, 1.0]},
            )
            try:
                for _ in range(3):
                    SAMPLER.advance(argparse.Namespace(
                        state=str(state),
                        imatrix=str(root / "round.gguf"),
                        batch_output=str(batch),
                        gguf_py="unused",
                    ))
            finally:
                SAMPLER.observer_snapshot = original
            result = json.loads(state.read_text(encoding="utf-8"))
            self.assertTrue(result["complete"])
            self.assertEqual(
                result["stop_reason"], "coverage-and-observer-stability"
            )
            self.assertEqual(len(result["history"]), 3)
            self.assertEqual(len(result["selected_ids"]), 384)

    def test_stability_detects_distribution_change(self):
        median, p95 = SAMPLER.stability(
            {"x": [1.0, 1.0, 1.0]},
            {"x": [1.0, 1.5, 0.5]},
        )
        self.assertGreater(median, 0.0)
        self.assertGreaterEqual(p95, median)


if __name__ == "__main__":
    unittest.main()
