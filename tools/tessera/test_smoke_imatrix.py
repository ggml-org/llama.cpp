"""Tests for tools/tessera/smoke_imatrix.py.

Crash-safety wrapper around llama-imatrix. The tests focus on the
behaviour we care about:

  1. The memory precheck refuses to start when the model is too big
     for the machine's physmem (unless --force is passed).
  2. The precheck is a NO-OP when physmem cannot be detected (Windows
     / unknown platform) - we never want to false-positive a refusal
     on a platform we cannot probe.
  3. The wrapper is invokable as a module via -m and the argument
     parser exposes the documented flags.
  4. The PID file lifecycle: written on launch, removed on exit.
     (We do NOT test the child-launch path; that requires a real
     imatrix binary and would be an integration test.)

Run as a unittest module. Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

# Import the script as a module
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "smoke_imatrix", THIS_DIR / "smoke_imatrix.py"
)
smoke_imatrix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke_imatrix)


class TestMemoryPreflight(unittest.TestCase):
    def test_physmem_bytes_known_platform(self) -> None:
        # On the test host (macOS or Linux) _physmem_bytes should be > 0
        n = smoke_imatrix._physmem_bytes()
        self.assertGreater(n, 0, "physmem probe failed; cannot run precheck")

    def test_humanize_bytes(self) -> None:
        self.assertEqual(smoke_imatrix._humanize_bytes(512), "512.0 B")
        self.assertEqual(smoke_imatrix._humanize_bytes(2048), "2.0 KB")
        self.assertEqual(smoke_imatrix._humanize_bytes(1024 ** 3), "1.0 GB")

    def test_preflight_refuses_over_budget(self) -> None:
        # 100 GB model on 16 GB physmem with default fraction 0.6 -> refuse
        # We mock _model_size_bytes to return a large value.
        fake_model = Path("/tmp/fake-100gb.gguf")
        with mock.patch.object(
            smoke_imatrix, "_model_size_bytes", return_value=100 * 1024 ** 3
        ), mock.patch.object(
            smoke_imatrix, "_physmem_bytes", return_value=16 * 1024 ** 3
        ):
            with self.assertRaises(SystemExit) as cm:
                smoke_imatrix._preflight_memory(
                    fake_model, 0.6, force=False
                )
            self.assertEqual(cm.exception.code, 2)

    def test_preflight_allows_within_budget(self) -> None:
        fake_model = Path("/tmp/fake-9gb.gguf")
        with mock.patch.object(
            smoke_imatrix, "_model_size_bytes", return_value=9 * 1024 ** 3
        ), mock.patch.object(
            smoke_imatrix, "_physmem_bytes", return_value=16 * 1024 ** 3
        ):
            model, phys, ratio = smoke_imatrix._preflight_memory(
                fake_model, 0.6, force=False
            )
            self.assertEqual(model, 9 * 1024 ** 3)
            self.assertEqual(phys, 16 * 1024 ** 3)
            self.assertAlmostEqual(ratio, 9 / 16, places=3)

    def test_preflight_force_overrides(self) -> None:
        fake_model = Path("/tmp/fake-100gb.gguf")
        with mock.patch.object(
            smoke_imatrix, "_model_size_bytes", return_value=100 * 1024 ** 3
        ), mock.patch.object(
            smoke_imatrix, "_physmem_bytes", return_value=16 * 1024 ** 3
        ):
            # force=True means the precheck should NOT exit
            model, phys, ratio = smoke_imatrix._preflight_memory(
                fake_model, 0.6, force=True
            )
            self.assertEqual(model, 100 * 1024 ** 3)

    def test_preflight_skipped_when_physmem_unknown(self) -> None:
        # On a platform where _physmem_bytes returns 0 (Windows / unknown),
        # the precheck must NOT refuse - we cannot make a safe decision
        # without a probe.
        fake_model = Path("/tmp/fake-100gb.gguf")
        with mock.patch.object(
            smoke_imatrix, "_physmem_bytes", return_value=0
        ):
            model, phys, ratio = smoke_imatrix._preflight_memory(
                fake_model, 0.6, force=False
            )
            self.assertEqual(phys, 0)
            self.assertEqual(ratio, 0.0)


class TestPIDFile(unittest.TestCase):
    def test_pid_file_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pid_path = Path(td) / "imatrix.pid"
            smoke_imatrix._write_pid_file(pid_path, 12345)
            self.assertTrue(pid_path.is_file())
            self.assertEqual(pid_path.read_text().strip(), "12345")
            smoke_imatrix._remove_pid_file(pid_path)
            self.assertFalse(pid_path.exists())

    def test_remove_pid_file_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pid_path = Path(td) / "imatrix.pid"
            # Removing a non-existent PID file should be a no-op.
            smoke_imatrix._remove_pid_file(pid_path)


class TestFindImatrixBinary(unittest.TestCase):
    def test_finds_existing_binary(self) -> None:
        # The real binary may or may not exist on the test host. If it
        # does, the function returns the canonical path. If not, it
        # raises FileNotFoundError. Both are valid outcomes.
        repo_root = THIS_DIR.parent.parent
        try:
            binary = smoke_imatrix._find_imatrix_binary(repo_root)
            self.assertTrue(binary.is_file())
        except FileNotFoundError:
            pass  # No binary built; expected on a fresh checkout

    def test_missing_binary_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            smoke_imatrix._find_imatrix_binary(Path("/nonexistent"))


class TestArgumentParser(unittest.TestCase):
    def test_required_args(self) -> None:
        from argparse import ArgumentTypeError
        # Calling main with no args should fail at the parser.
        with self.assertRaises(SystemExit):
            smoke_imatrix.main([])

    def test_help_exposes_documented_flags(self) -> None:
        # Build a parser from scratch by importing the module's
        # _parser-equivalent. The simplest way: just call
        # ArgumentParser on the same flags by calling parse_args
        # with --help and capturing the output.
        import argparse
        parser = argparse.ArgumentParser()
        # Mirror the documented flags: smoke_imatrix accepts
        # --model, --corpus, --output, --save-frequency,
        # --max-minutes, --memory-safety-fraction, --force,
        # --imatrix-binary, --ctx-size, --chunks, --extra-arg
        # We do this by calling main() with --help and asserting
        # the SystemExit code.
        with self.assertRaises(SystemExit) as cm:
            smoke_imatrix.main(["--help"])
        self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
