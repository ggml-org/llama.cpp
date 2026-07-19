#!/usr/bin/env python3
"""Unit tests for the Arc A770 MMVQ geometry sweep orchestrator."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).with_name("sweep-a770-mmvq-geometry.py")
SPEC = importlib.util.spec_from_file_location("sweep_a770_mmvq_geometry", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load geometry sweep: {SCRIPT_PATH}")
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


class GeometrySweepTests(unittest.TestCase):
    def test_added_suffix_handles_full_and_truncated_history(self) -> None:
        self.assertEqual(SWEEP.added_suffix(["a", "b"], ["a", "b", "c"]), ["c"])
        self.assertEqual(SWEEP.added_suffix(["a", "b"], ["b", "c"]), ["c"])
        self.assertEqual(SWEEP.added_suffix(["a"], ["x"]), ["x"])

    # ------------------------------------------------------------------ #
    # Defect 1: parse_gate_fail - substring must not accept 10 GATE-FAIL  #
    # ------------------------------------------------------------------ #

    def test_parse_gate_fail_exact_zero_passes(self) -> None:
        self.assertTrue(SWEEP.parse_gate_fail("== summary: 0 GATE-FAIL"))

    def test_parse_gate_fail_nonzero_rejects(self) -> None:
        self.assertFalse(SWEEP.parse_gate_fail("== summary: 3 GATE-FAIL"))

    def test_parse_gate_fail_10_is_not_zero(self) -> None:
        """Sub-string '0' inside '10' must NOT match - verify boundary logic."""
        self.assertFalse(
            SWEEP.parse_gate_fail("== summary: 10 GATE-FAIL"),
        )

    def test_parse_gate_fail_empty_returns_false(self) -> None:
        self.assertFalse(SWEEP.parse_gate_fail(""))
        self.assertFalse(SWEEP.parse_gate_fail("no summary line"))

    # ------------------------------------------------------------------ #
    # Existing: require_sole_tenancy                                     #
    # ------------------------------------------------------------------ #

    def test_empty_fuser_exit_one_proves_idle(self) -> None:
        proc = subprocess.CompletedProcess(["fuser"], 1, "", "")
        with (
            mock.patch.object(SWEEP, "run", return_value=proc),
            mock.patch.object(SWEEP, "require_tool", return_value="fuser"),
        ):
            SWEEP.require_sole_tenancy("/dev/dri/renderD128")

    def test_holder_or_fuser_error_blocks_gpu_work(self) -> None:
        cases = ((0, "1234 llama-bench"), (1, "permission denied"))
        for returncode, output in cases:
            with self.subTest(returncode=returncode, output=output):
                proc = subprocess.CompletedProcess(["fuser"], returncode, output, "")
                with (
                    mock.patch.object(SWEEP, "run", return_value=proc),
                    mock.patch.object(SWEEP, "require_tool", return_value="fuser"),
                ):
                    with self.assertRaises(SWEEP.SweepError):
                        SWEEP.require_sole_tenancy("/dev/dri/renderD128")

    def test_build_directory_encodes_geometry_and_tag(self) -> None:
        self.assertEqual(
            SWEEP.build_dir(Path("/builds"), "abc123", 4, 32),
            Path("/builds/build-p58-y4-sg32-abc123"),
        )

    def test_cmake_cache_accepts_requested_flags_among_ambient_flags(
        self,
    ) -> None:
        requested = "-DGGML_SYCL_MMV_Y=4 -DGGML_SYCL_MMVQ_NUM_SUBGROUPS=32"
        cache = (
            "CMAKE_CXX_FLAGS:STRING=-O2 -g "
            "-DGGML_SYCL_MMVQ_NUM_SUBGROUPS=32 -DGGML_SYCL_MMV_Y=4\n"
        )
        self.assertTrue(SWEEP.cmake_cache_has_flags(cache, requested))
        self.assertFalse(
            SWEEP.cmake_cache_has_flags(
                cache,
                "-DGGML_SYCL_MMV_Y=8 -DGGML_SYCL_MMVQ_NUM_SUBGROUPS=32",
            )
        )

    # ------------------------------------------------------------------ #
    # Defect 3: selector propagation into benchmark subprocesses         #
    # ------------------------------------------------------------------ #

    def test_benchmark_matrix_propagates_level_zero_selector(self) -> None:
        """Benchmark subprocesses receive ONEAPI_DEVICE_SELECTOR=level_zero:0
        and each result records it as evidence."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tag = "abc123"
            manifest_id = f"{tag}-def456"
            source = root / "src"
            source.mkdir()
            harness = source / "scripts/bench-a770-fork-unique.py"
            harness.parent.mkdir(parents=True)
            harness.write_text("#! dummy harness", encoding="utf-8")

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "source": str(source),
                        "tag": tag,
                        "manifest_identity": manifest_id,
                        "correctness": [
                            {
                                "y": y,
                                "subgroups": sg,
                                "valid": True,
                                "manifest_identity": manifest_id,
                            }
                            for y, sg in SWEEP.CELLS
                        ],
                    }
                ),
                encoding="utf-8",
            )

            model_path = root / "model.bin"
            model_path.write_bytes(b"x" * 100)

            args = SimpleNamespace(
                build_root=root,
                render_node="/dev/dri/renderD128",
                timeout=60,
                repetitions=2,
                source=source,
                out_root=root,
                model=[f"test={model_path}"],
                baseline_y=1,
                baseline_subgroups=16,
                model_layers=32,
                query_heads=32,
                head_dim=128,
                tag=tag,
            )

            call_envs: list[dict[str, str]] = []

            def capture_run(argv, **kwargs):
                env_arg = kwargs.get("env")
                if env_arg is not None:
                    call_envs.append(dict(env_arg))
                fake = subprocess.CompletedProcess(argv, 0, "", "")
                return fake

            with (
                mock.patch.object(SWEEP, "CELLS", [(1, 16)]),
                mock.patch.object(SWEEP, "require_sole_tenancy"),
                mock.patch.object(SWEEP, "run", side_effect=capture_run),
            ):
                results = SWEEP.benchmark_matrix(args, manifest_id)

            self.assertGreater(len(call_envs), 0)
            for env in call_envs:
                self.assertEqual(
                    env["ONEAPI_DEVICE_SELECTOR"],
                    "level_zero:0",
                    msg="benchmark env missing level_zero:0",
                )

            for r in results:
                self.assertEqual(r["oneapi_device_selector"], "level_zero:0")

    # ------------------------------------------------------------------ #
    # Defect 5: reject duplicate model labels                             #
    # ------------------------------------------------------------------ #

    def test_duplicate_model_labels_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.bin"
            path.write_bytes(b"data")
            raw = [f"gpt={path}", f"gpt={path}"]
            with self.assertRaisesRegex(
                SWEEP.SweepError,
                r"duplicate model label",
            ):
                SWEEP.parse_models(raw)

    def test_unique_model_labels_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path_a = Path(td) / "a.bin"
            path_b = Path(td) / "b.bin"
            path_a.write_bytes(b"a")
            path_b.write_bytes(b"b")
            models = SWEEP.parse_models([f"x={path_a}", f"y={path_b}"])
            self.assertEqual(sorted(models.keys()), ["x", "y"])

    # ------------------------------------------------------------------ #
    # Defect 6: exclude invalid geometry (work-group > 1024)              #
    # ------------------------------------------------------------------ #

    def test_cell_valid_1x16_within_limit(self) -> None:
        self.assertTrue(SWEEP._cell_valid(1, 16))  # 1*16*32 = 512

    def test_cell_valid_4x4_within_limit(self) -> None:
        self.assertTrue(SWEEP._cell_valid(4, 4))  # 4*4*32 = 512

    def test_cell_valid_4x8_at_boundary(self) -> None:
        self.assertTrue(SWEEP._cell_valid(4, 8))  # 4*8*32 = 1024

    def test_cell_invalid_2x17_exceeds_limit(self) -> None:
        self.assertFalse(SWEEP._cell_valid(2, 17))  # 2*17*32 = 1088

    def test_cell_invalid_4x9_exceeds_limit(self) -> None:
        self.assertFalse(SWEEP._cell_valid(4, 9))  # 4*9*32 = 1152

    def test_cell_invalid_2x32_exceeds_limit(self) -> None:
        self.assertFalse(SWEEP._cell_valid(2, 32))  # 2*32*32 = 2048

    def test_cells_excludes_invalid_configurations(self) -> None:
        """The exported CELLS tuple contains ONLY work-groups ≤ 1024."""
        for y, sg in SWEEP.CELLS:
            size = y * sg * 32
            self.assertLessEqual(
                size,
                SWEEP.A770_MAX_WORK_ITEMS,
                msg=f"CELLS contains {y}x{sg} = {size} > {SWEEP.A770_MAX_WORK_ITEMS}",
            )

    def test_cells_contains_default_1x16(self) -> None:
        self.assertIn((1, 16), SWEEP.CELLS)

    def test_all_possible_configs_defined(self) -> None:
        """_ALL_CELL_CONFIGS enumerates every product of MMV_Y x subgroups."""
        expected = [(y, sg) for y in (1, 2, 4) for sg in (4, 8, 16, 32)]
        self.assertEqual(set(SWEEP._ALL_CELL_CONFIGS), set(expected))

    # ------------------------------------------------------------------ #
    # Defect 2: benchmark fail closed without matching correctness       #
    # ------------------------------------------------------------------ #

    def test_benchmark_without_manifest_raises(self) -> None:
        """benchmark-only mode refuses when manifest doesn't exist."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tag = "abc123"
            source = root / "src"
            source.mkdir()
            harness = source / "scripts/bench-a770-fork-unique.py"
            harness.parent.mkdir(parents=True)
            harness.write_text("#!/bin/sh\necho ok", encoding="utf-8")
            model_path = root / "model.bin"
            model_path.write_bytes(b"x" * 100)

            # out_root is set but no manifest.json written there
            out_dir = root / "output"
            out_dir.mkdir()

            args = SimpleNamespace(
                build_root=root,
                render_node="/dev/dri/renderD128",
                timeout=60,
                repetitions=2,
                source=source,
                out_root=out_dir,
                model=[f"m={model_path}"],
                baseline_y=1,
                baseline_subgroups=16,
                model_layers=32,
                query_heads=32,
                head_dim=128,
                tag=tag,
            )

            with self.assertRaisesRegex(
                SWEEP.SweepError,
                r"no manifest\.json found",
            ):
                SWEEP.benchmark_matrix(args, tag)

    def test_benchmark_wrong_identity_raises(self) -> None:
        """Manifest-level identity mismatches between manifest header
        and correctness-record bodies blocks benchmark."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tag = "abc123"
            source = root / "src"
            source.mkdir()
            harness = source / "scripts/bench-a770-fork-unique.py"
            harness.parent.mkdir(parents=True)
            harness.write_text("#!/bin/sh\necho ok", encoding="utf-8")
            model_path = root / "model.bin"
            model_path.write_bytes(b"x" * 100)

            out_dir = root / "output"
            out_dir.mkdir()
            # Header says "stale-identity" but records carry "fresh-identity"
            manifest_path = out_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "source": str(source),
                        "tag": tag,
                        "manifest_identity": "stale-identity",
                        "correctness": [
                            {
                                "y": y,
                                "subgroups": sg,
                                "valid": True,
                                "manifest_identity": "fresh-identity",
                            }
                            for y, sg in SWEEP.CELLS
                        ],
                    }
                ),
                encoding="utf-8",
            )

            args = SimpleNamespace(
                build_root=root,
                render_node="/dev/dri/renderD128",
                timeout=60,
                repetitions=2,
                source=source,
                out_root=out_dir,
                model=[f"m={model_path}"],
                baseline_y=1,
                baseline_subgroups=16,
                model_layers=32,
                query_heads=32,
                head_dim=128,
                tag=tag,
            )

            with (
                self.assertRaisesRegex(
                    SWEEP.SweepError,
                    r"no valid correctness record",
                ),
                mock.patch.object(SWEEP, "require_sole_tenancy"),
            ):
                SWEEP.benchmark_matrix(args, "any-id")

    def test_benchmark_missing_correctness_for_geometry_raises(self) -> None:
        """Missing correctness record for one geometry blocks benchmark."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tag = "abc123"
            source = root / "src"
            source.mkdir()
            harness = source / "scripts/bench-a770-fork-unique.py"
            harness.parent.mkdir(parents=True)
            harness.write_text("#!/bin/sh\necho ok", encoding="utf-8")
            model_path = root / "model.bin"
            model_path.write_bytes(b"x" * 100)

            out_dir = root / "output"
            out_dir.mkdir()
            manifest_path = out_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "source": str(source),
                        "tag": tag,
                        "manifest_identity": tag,
                        "correctness": [
                            {
                                "y": 1,
                                "subgroups": 16,
                                "valid": True,
                                "manifest_identity": tag,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            args = SimpleNamespace(
                build_root=root,
                render_node="/dev/dri/renderD128",
                timeout=60,
                repetitions=2,
                source=source,
                out_root=out_dir,
                model=[f"m={model_path}"],
                baseline_y=1,
                baseline_subgroups=16,
                model_layers=32,
                query_heads=32,
                head_dim=128,
                tag=tag,
            )

            with (
                self.assertRaisesRegex(
                    SWEEP.SweepError,
                    r"no valid correctness record",
                ),
                mock.patch.object(SWEEP, "require_sole_tenancy"),
            ):
                SWEEP.benchmark_matrix(args, tag)

    # ------------------------------------------------------------------ #
    # Defect 4: dirty working-tree diverges from clean HEAD identity     #
    # ------------------------------------------------------------------ #

    def test_source_tag_includes_dirty_fingerprint(self) -> None:
        """A dirty working tree produces a different tag than a clean one."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            root.joinpath(".git").mkdir()

            subprocess.run(
                ["git", "init"],
                cwd=root,
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=root,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=root,
                capture_output=True,
                check=True,
            )

            commit_file = root / "file.txt"
            commit_file.write_text("initial")
            subprocess.run(
                ["git", "add", "."],
                cwd=root,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "initial"],
                cwd=root,
                capture_output=True,
                check=True,
            )

            clean_tag = SWEEP.source_tag(root)

            root.joinpath("dirty.txt").write_text("dirt")
            dirty_tag = SWEEP.source_tag(root)

            self.assertNotEqual(clean_tag, dirty_tag)

            root.joinpath("dirty.txt").unlink()
            commit_file.write_text("changed-one", encoding="utf-8")
            first_dirty_content_tag = SWEEP.source_tag(root)
            commit_file.write_text("changed-two", encoding="utf-8")
            second_dirty_content_tag = SWEEP.source_tag(root)
            self.assertNotEqual(
                first_dirty_content_tag,
                second_dirty_content_tag,
                msg="dirty identity must include content, not only status",
            )

    def test_source_tag_identical_when_no_dirty_files(self) -> None:
        """Identical clean trees produce identical tags."""
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            for root in (Path(td1), Path(td2)):
                subprocess.run(
                    ["git", "init"],
                    cwd=root,
                    capture_output=True,
                    check=False,
                )
                subprocess.run(
                    ["git", "config", "user.email", "t@t.com"],
                    cwd=root,
                    capture_output=True,
                    check=True,
                )
                subprocess.run(
                    ["git", "config", "user.name", "T"],
                    cwd=root,
                    capture_output=True,
                    check=True,
                )
                root.joinpath("file.txt").write_text("same")
                subprocess.run(
                    ["git", "add", "."],
                    cwd=root,
                    capture_output=True,
                    check=True,
                )
                subprocess.run(
                    ["git", "commit", "-m", "msg"],
                    cwd=root,
                    capture_output=True,
                    check=True,
                )

            tag1 = SWEEP.source_tag(Path(td1))
            tag2 = SWEEP.source_tag(Path(td2))
            self.assertEqual(tag1, tag2)

    # ------------------------------------------------------------------ #
    # Correctness matrix persists identity                                #
    # ------------------------------------------------------------------ #

    def test_correctness_env_clears_ambient_fork_knobs(self) -> None:
        controlled = (
            "GGML_SYCL_FA_XMX",
            "GGML_SYCL_FA_FORCE_VEC_STANDARD",
            "GGML_SYCL_FA_Q8_GQA_TILE",
            "GGML_SYCL_Q8_KV_QUANTS_FIRST",
            "LLAMA_ENABLE_INNERQ",
            "TURBO_LAYER_ADAPTIVE",
            "TURBO_AUTO_ASYMMETRIC",
        )
        ambient = {name: "1" for name in controlled}
        ambient["ONEAPI_DEVICE_SELECTOR"] = "ambient:9"
        with mock.patch.dict(SWEEP.os.environ, ambient):
            env = SWEEP._correctness_env()
        for name in controlled:
            self.assertNotIn(name, env)
        self.assertEqual(env["ONEAPI_DEVICE_SELECTOR"], "level_zero:0")

    def test_correctness_persists_identity_fields(self) -> None:
        """Each correctness record carries manifest_identity for verification."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tag = "abc123"
            mid = "abc123-clean"
            # Build dir uses manifest_identity for the path component
            out = SWEEP.build_dir(root, mid, 1, 4)
            binary = out / "bin/test-sycl-turbo-correctness"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"test")

            source_dir = root / "src"
            source_dir.mkdir(exist_ok=True)

            args = SimpleNamespace(
                build_root=root,
                render_node="/dev/dri/renderD128",
                timeout=60,
                source=source_dir,
            )

            proc = subprocess.CompletedProcess(
                ["/usr/bin/timeout"], 0, "== summary: 0 GATE-FAIL\n", ""
            )
            with (
                mock.patch.object(SWEEP, "CELLS", [(1, 4)]),
                mock.patch.object(SWEEP, "require_sole_tenancy"),
                mock.patch.object(SWEEP, "dmesg_faults", side_effect=[[], []]),
                mock.patch.object(SWEEP, "run", return_value=proc),
                mock.patch.object(
                    SWEEP, "require_tool", return_value="/usr/bin/timeout"
                ),
            ):
                results = SWEEP.correctness_matrix(args, tag, mid)
            self.assertTrue(results[0]["valid"])
            self.assertEqual(results[0]["manifest_identity"], mid)

    def test_correctness_detects_10_gate_fail_as_failure(self) -> None:
        """A summary containing '10 GATE-FAIL' must NOT be treated as valid."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tag = "abc123"
            out = SWEEP.build_dir(root, tag, 1, 4)
            binary = out / "bin/test-sycl-turbo-correctness"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"test")

            source_dir = root / "src"
            source_dir.mkdir(exist_ok=True)

            args = SimpleNamespace(
                build_root=root,
                render_node="/dev/dri/renderD128",
                timeout=60,
                source=source_dir,
            )
            proc = subprocess.CompletedProcess(
                ["/usr/bin/timeout"], 0, "== summary: 10 GATE-FAIL\n", ""
            )
            with (
                mock.patch.object(SWEEP, "CELLS", [(1, 4)]),
                mock.patch.object(SWEEP, "require_sole_tenancy"),
                mock.patch.object(SWEEP, "dmesg_faults", side_effect=[[], []]),
                mock.patch.object(SWEEP, "run", return_value=proc),
                mock.patch.object(
                    SWEEP, "require_tool", return_value="/usr/bin/timeout"
                ),
            ):
                with self.assertRaises(SWEEP.SweepError):
                    SWEEP.correctness_matrix(args, tag, tag)

    def test_default_output_root_uses_manifest_identity(self) -> None:
        args = SimpleNamespace(out_root=None, tag=None)

        self.assertEqual(
            SWEEP.output_root(args, "source-identity"),
            Path("/tmp/a770-mmvq-geometry-source-identity"),
        )

    def test_main_rejects_two_repetitions_before_campaign_work(self) -> None:
        args = SimpleNamespace(
            phase="all",
            source=Path("/tmp/source"),
            build_root=Path("/tmp/build"),
            tag=None,
            out_root=None,
            jobs=1,
            parallel_builds=1,
            repetitions=2,
        )

        with mock.patch.object(SWEEP, "parse_args", return_value=args):
            self.assertEqual(SWEEP.main(), 2)

    def test_all_phase_persists_correctness_before_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out_root = root / "results"
            args = SimpleNamespace(
                phase="all",
                source=root,
                build_root=root,
                tag="identity",
                out_root=out_root,
                jobs=1,
                parallel_builds=1,
                repetitions=3,
            )
            correctness = [
                {
                    "y": 1,
                    "subgroups": 4,
                    "valid": True,
                    "manifest_identity": "identity",
                }
            ]

            def benchmark_after_persisted_correctness(_args, _identity):
                manifest = json.loads(
                    (out_root / "manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["correctness"], correctness)
                return []

            with (
                mock.patch.object(SWEEP, "parse_args", return_value=args),
                mock.patch.object(SWEEP, "CELLS", [(1, 4)]),
                mock.patch.object(SWEEP, "build_matrix", return_value=[]),
                mock.patch.object(
                    SWEEP,
                    "correctness_matrix",
                    return_value=correctness,
                ),
                mock.patch.object(
                    SWEEP,
                    "benchmark_matrix",
                    side_effect=benchmark_after_persisted_correctness,
                ),
            ):
                self.assertEqual(SWEEP.main(), 0)


if __name__ == "__main__":
    unittest.main()
