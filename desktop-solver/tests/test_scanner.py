"""Tests for the filesystem scanner."""

from __future__ import annotations

from pathlib import Path

import pytest

from desktop_solver import scanner
from desktop_solver.scanner import FoundModel


def _touch(p: Path) -> None:
    """Create an empty placeholder GGUF file. We never need real bytes — the
    scanner only reads filenames and stat() results."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()


def _fake(name: str, size_gb: float) -> FoundModel:
    import time
    return FoundModel(path=Path(f"/fake/{name}"),
                      size_bytes=int(size_gb * 1024 ** 3),
                      mtime=time.time())


def test_finds_gguf_in_root(tmp_path):
    _touch(tmp_path / "model.gguf")
    found = scanner.scan(roots=[str(tmp_path)], max_seconds=5)
    assert len(found) == 1
    assert found[0].path.name == "model.gguf"


def test_finds_nested_gguf(tmp_path):
    _touch(tmp_path / "deep" / "in" / "tree" / "Qwen2.5-Math-7B.gguf")
    found = scanner.scan(roots=[str(tmp_path)], max_seconds=5)
    assert any(m.path.name == "Qwen2.5-Math-7B.gguf" for m in found)


def test_skips_skip_dirs(tmp_path):
    _touch(tmp_path / ".git" / "in-git.gguf")
    _touch(tmp_path / "node_modules" / "in-nm.gguf")
    _touch(tmp_path / "__pycache__" / "in-pyc.gguf")
    _touch(tmp_path / "good.gguf")
    found = scanner.scan(roots=[str(tmp_path)], max_seconds=5)
    names = {m.path.name for m in found}
    assert names == {"good.gguf"}


def test_skips_hidden_but_descends_dot_cache(tmp_path):
    _touch(tmp_path / ".hidden" / "x.gguf")
    _touch(tmp_path / ".cache" / "y.gguf")
    found = scanner.scan(roots=[str(tmp_path)], max_seconds=5)
    names = {m.path.name for m in found}
    assert names == {"y.gguf"}


def test_no_models_returns_empty(tmp_path):
    (tmp_path / "data").mkdir()
    found = scanner.scan(roots=[str(tmp_path)], max_seconds=5)
    assert found == []


def test_dedupes_via_resolve(tmp_path):
    real = tmp_path / "real" / "model.gguf"
    _touch(real)
    link = tmp_path / "alias"
    link.symlink_to(tmp_path / "real")
    found = scanner.scan(roots=[str(tmp_path)], max_seconds=5)
    assert len(found) == 1


def test_auto_pick_prefers_math_in_name():
    candidates = [
        _fake("qwen2.5-math-7b-instruct-q4_k_m.gguf", 4.0),
        _fake("generic-7b-q4_k_m.gguf",               4.0),
    ]
    best = scanner.auto_pick(candidates, ram_bytes=16 * 1024 ** 3)
    assert best is not None
    assert "math" in best.name.lower()


def test_auto_pick_prefers_q4_over_q8_when_budget_tight():
    candidates = [
        _fake("model-q4_k_m.gguf", 4.0),
        _fake("model-q8_0.gguf",   7.0),
    ]
    best = scanner.auto_pick(candidates, ram_bytes=8 * 1024 ** 3)
    assert best is not None
    assert "q4" in best.name.lower()


def test_auto_pick_returns_none_on_empty():
    assert scanner.auto_pick([], ram_bytes=8 * 1024 ** 3) is None


def test_scan_handles_missing_root(tmp_path):
    found = scanner.scan(
        roots=[str(tmp_path / "does-not-exist"), str(tmp_path)],
        max_seconds=5,
    )
    assert found == []


def test_score_orders_math_above_chat():
    found = sorted(
        [_fake("math-7b.gguf", 4.0), _fake("chat-7b.gguf", 4.0)],
        key=lambda m: scanner.score(m, ram_bytes=8 * 1024 ** 3),
        reverse=True,
    )
    assert "math" in found[0].name.lower()


def test_system_ram_bytes_is_positive():
    assert scanner.system_ram_bytes() > 0
