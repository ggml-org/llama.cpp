from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "dory_oracle_check.py"
SPEC = importlib.util.spec_from_file_location("dory_oracle_check", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
oracle_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle_check)


def make_reference(tmp_path: Path, **overrides) -> dict:
    payload = {
        "quantity": "logprobs",
        "prompt": "What is 17 multiplied by 23? The answer is",
        "prompt_ids": [7493, 1395],
        "ids_loops4": [1032, 1051],
        "positions": [{"position": 0, "top": [[1032, -0.1], [2007, -5.0]]}],
    }
    payload.update(overrides)
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return oracle_check.load_reference(path)


def test_reference_pairing_rejects_a_different_prompt(tmp_path: Path) -> None:
    reference = make_reference(tmp_path)

    reason = oracle_check.reference_oracle_mismatch(reference, "The capital of France is", [1784, 8961], [6993])

    assert "oracle prompt_ids" in reason
    assert "reference prompt" in reason


def test_reference_pairing_rejects_different_expected_ids(tmp_path: Path) -> None:
    reference = make_reference(tmp_path)

    reason = oracle_check.reference_oracle_mismatch(
        reference,
        "What is 17 multiplied by 23? The answer is",
        [7493, 1395],
        [1032, 9999],
    )

    assert "position 1" in reason
    assert "9999" in reason


def test_reference_pairing_accepts_matching_data(tmp_path: Path) -> None:
    reference = make_reference(tmp_path)

    reason = oracle_check.reference_oracle_mismatch(
        reference,
        "What is 17 multiplied by 23? The answer is",
        [7493, 1395],
        [1032, 1051],
    )

    assert reason == ""


def report_reference() -> dict:
    return {
        "path": "reference.json",
        "quantity": "logprobs",
        "positions": {0: [(1032, -0.1), (2007, -5.0)]},
        "provenance": "pytest",
        "self_observed": False,
    }


def test_disjoint_top_k_is_loud_data_not_a_crash(capsys: pytest.CaptureFixture[str]) -> None:
    rows = [{"position": 0, "actual": [(6993, -0.2), (2549, -4.0)], "why": ""}]

    rc = oracle_check.report_logits(
        rows,
        report_reference(),
        0.5,
        "pytest",
        Path("oracle.json"),
        "ids_loops4",
        [[(6993, -0.2), (2549, -4.0)]],
        None,
    )

    captured = capsys.readouterr()
    assert rc == oracle_check.EXIT_PLUMBING == 3
    assert "EMPTY COMPARISON" in captured.out
    assert "no shared token ids" in captured.out
    assert "Traceback" not in captured.err


def test_zero_rows_cannot_be_a_silent_pass(capsys: pytest.CaptureFixture[str]) -> None:
    rc = oracle_check.report_logits(
        [],
        report_reference(),
        0.5,
        "pytest",
        Path("oracle.json"),
        "ids_loops4",
        [],
        None,
    )

    captured = capsys.readouterr()
    assert rc == oracle_check.EXIT_PLUMBING == 3
    assert "no position produced a non-empty numerical comparison" in captured.out


def test_nonempty_matching_comparison_still_passes(capsys: pytest.CaptureFixture[str]) -> None:
    actual = [(1032, -0.1), (2007, -5.0)]
    rows = [{"position": 0, "actual": actual, "why": ""}]

    rc = oracle_check.report_logits(
        rows,
        report_reference(),
        0.5,
        "pytest",
        Path("oracle.json"),
        "ids_loops4",
        [actual],
        None,
    )

    captured = capsys.readouterr()
    assert rc == oracle_check.EXIT_MATCH == 0
    assert "VERDICT: PASS -- 1 positions" in captured.out
