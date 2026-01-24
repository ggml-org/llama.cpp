import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODULE_DIR = os.path.join(ROOT, "tools", "sycl-kernel-bench")
sys.path.insert(0, MODULE_DIR)

import validate_matrix  # noqa: E402


@pytest.fixture(scope="module")
def matrix():
    path = os.path.join(MODULE_DIR, "test_matrix.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_generate_configs_count_in_range(matrix):
    configs = validate_matrix.generate_configs(matrix)
    runnable = [c for c in configs if not c.get("skip", False)]
    limits = matrix["limits"]
    assert len(runnable) >= limits["min_configs"]
    assert len(runnable) <= limits["max_configs"]


def test_no_duplicate_configs(matrix):
    configs = validate_matrix.generate_configs(matrix)
    duplicates = validate_matrix.find_duplicate_configs(configs)
    assert duplicates == []


def test_validation_passes(matrix):
    result = validate_matrix.validate_matrix(matrix)
    assert result["errors"] == []


def test_estimates_are_positive(matrix):
    configs = validate_matrix.generate_configs(matrix)
    for cfg in configs:
        if cfg.get("skip", False):
            continue
        estimate = validate_matrix.estimate_runtime_s(cfg, matrix["estimation"])
        assert estimate > 0.0


def test_expected_failures_are_recorded(matrix):
    configs = validate_matrix.generate_configs(matrix)
    expected = [c for c in configs if c.get("expected_failure", False)]
    assert expected
