#!/usr/bin/env python3

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).parent / "perf" / "bench_spec.py"
SPEC = importlib.util.spec_from_file_location("bench_spec", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load bench_spec harness: {MODULE_PATH}")
BENCH_SPEC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH_SPEC
SPEC.loader.exec_module(BENCH_SPEC)


class BenchSpecEvidenceTests(unittest.TestCase):
    def test_analyze_native_response_accepts_target_argmax_tokens(self) -> None:
        response = {
            "content": "ok",
            "tokens": [11, 12],
            "timings": {"predicted_n": 2},
            "completion_probabilities": [
                {"id": 11, "logprob": -0.1, "top_logprobs": [{"id": 11, "logprob": -0.1}]},
                {"id": 12, "logprob": -0.2, "top_logprobs": [{"id": 12, "logprob": -0.2}]},
            ],
        }

        evidence = BENCH_SPEC.analyze_native_response(response)

        self.assertTrue(evidence["verifier_invariant_ok"])
        self.assertEqual(evidence["token_ids"], [11, 12])
        self.assertEqual(evidence["verifier_rows"], 2)
        self.assertEqual(evidence["verifier_failures"], [])
        self.assertEqual(len(evidence["token_sha256"]), 64)

    def test_analyze_native_response_accepts_verified_draft_without_top_logprobs(self) -> None:
        response = {
            "content": "draft",
            "tokens": [17],
            "completion_probabilities": [
                {"id": 17, "logprob": -0.3, "top_logprobs": []},
            ],
        }

        evidence = BENCH_SPEC.analyze_native_response(response)

        self.assertTrue(evidence["verifier_invariant_ok"])
        self.assertEqual(evidence["verifier_failures"], [])

    def test_analyze_native_response_rejects_non_argmax_and_nonfinite_logits(self) -> None:
        response = {
            "content": "bad",
            "tokens": [21, 22],
            "completion_probabilities": [
                {"id": 21, "logprob": -0.1, "top_logprobs": [{"id": 99, "logprob": -0.05}]},
                {"id": 22, "logprob": math.nan, "top_logprobs": [{"id": 22, "logprob": math.nan}]},
            ],
        }

        evidence = BENCH_SPEC.analyze_native_response(response)

        self.assertFalse(evidence["verifier_invariant_ok"])
        self.assertEqual([failure["reason"] for failure in evidence["verifier_failures"]], [
            "generated_token_not_target_argmax",
            "nonfinite_target_logprob",
        ])

    def test_stress_mode_includes_target_only_and_hard_off_arms(self) -> None:
        with mock.patch.object(BENCH_SPEC, "MODE", "stress"), mock.patch.object(BENCH_SPEC, "KV", "q8_0"):
            arms = BENCH_SPEC.build_arms()

        self.assertEqual([arm["name"] for arm in arms], [
            "none-q8_0",
            "deadoff0-q8_0",
            "deadoff3-q8_0",
        ])

    def test_scan_log_records_hard_off_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = Path(tmpdir) / "server.log"
            logpath.write_text(
                "common_speculative_impl_ngram_mod: 3 dead ngram-mod fires - disabling for seq 0\n",
                encoding="utf-8",
            )

            scan = BENCH_SPEC.scan_log(logpath)

        self.assertEqual(len(scan["hard_off_lines"]), 1)
        self.assertIn("disabling for seq 0", scan["hard_off_lines"][0])

    def test_parse_rejection_records_tracks_generated_positions_per_task(self) -> None:
        text = "\n".join([
            "slot update_slots: id 0 | task 7 | accepted 1/48 draft tokens",
            "slot update_slots: id 0 | task 7 | accepted 48/48 draft tokens",
            "slot update_slots: id 0 | task 7 | accepted 37/48 draft tokens",
        ])

        records = BENCH_SPEC.parse_rejection_records(text)

        self.assertEqual([record["rejection_position"] for record in records], [1, 88])
        self.assertEqual([record["task"] for record in records], [7, 7])


if __name__ == "__main__":
    unittest.main()
