#!/usr/bin/env python3
"""Unit tests for the canonical server slot state roundtrip harness."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).parent / "perf" / "server_state_roundtrip.py"
HARNESS = None
if SCRIPT_PATH.is_file():
    SPEC = importlib.util.spec_from_file_location("server_state_roundtrip", SCRIPT_PATH)
    if SPEC is None or SPEC.loader is None:
        raise RuntimeError(f"cannot load state roundtrip harness: {SCRIPT_PATH}")
    HARNESS = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(HARNESS)


class HarnessPresenceTests(unittest.TestCase):
    def test_state_roundtrip_harness_exists(self) -> None:
        self.assertTrue(SCRIPT_PATH.is_file(), f"missing state roundtrip harness: {SCRIPT_PATH}")


@unittest.skipUnless(SCRIPT_PATH.is_file(), "state roundtrip harness not implemented")

class StateRoundtripTests(unittest.TestCase):
    def test_exact_p5_protocol_checks_cross_slot_continuation(self) -> None:
        calls: list[tuple[str, str, dict[str, Any] | None]] = []
        responses = iter(
            [
                {"content": "\nA: Paris", "tokens_cached": 12},
                {"id_slot": 1, "filename": "state.bin", "n_saved": 12, "n_written": 836576},
                {"content": "\nBerlin", "timings": {"prompt_n": 2}},
                {"id_slot": 0, "filename": "state.bin", "n_restored": 12, "n_read": 836576},
                {"content": "\nBerlin", "timings": {"prompt_n": 2}},
                {"content": "\nBerlin", "timings": {"prompt_n": 1}},
            ]
        )

        def request_json(method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
            calls.append((method, path, payload))
            return next(responses)

        result = HARNESS.exercise_state_roundtrip(
            request_json,
            filename="state.bin",
            expected_tokens=12,
            expected_bytes=836576,
        )

        self.assertTrue(result["pass"])
        self.assertEqual(result["failures"], [])
        self.assertEqual(
            [(method, path) for method, path, _ in calls],
            [
                ("POST", "/completion"),
                ("POST", "/slots/1?action=save"),
                ("POST", "/completion"),
                ("POST", "/slots/0?action=restore"),
                ("POST", "/completion"),
                ("POST", "/completion"),
            ],
        )
        self.assertEqual(calls[0][2]["id_slot"], 1)
        self.assertEqual(calls[4][2]["id_slot"], 0)
        self.assertEqual(calls[5][2]["id_slot"], 1)

    def test_byte_count_mismatch_fails_closed(self) -> None:
        responses = iter(
            [
                {"content": "\nA: Paris", "tokens_cached": 12},
                {"id_slot": 1, "filename": "state.bin", "n_saved": 12, "n_written": 1},
                {"content": "\nBerlin", "timings": {"prompt_n": 2}},
                {"id_slot": 0, "filename": "state.bin", "n_restored": 12, "n_read": 1},
                {"content": "\nBerlin", "timings": {"prompt_n": 2}},
                {"content": "\nBerlin", "timings": {"prompt_n": 1}},
            ]
        )

        def request_json(method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
            del method, path, payload
            return next(responses)

        result = HARNESS.exercise_state_roundtrip(
            request_json,
            filename="state.bin",
            expected_tokens=12,
            expected_bytes=836576,
        )

        self.assertFalse(result["pass"])
        self.assertIn("save byte count", result["failures"])
        self.assertIn("restore byte count", result["failures"])


if __name__ == "__main__":
    unittest.main()
