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
        self.assertEqual(evidence["verifier_rows_with_argmax"], 2)
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
        # accepted, but the summary shows that the row carried no argmax evidence
        self.assertEqual(evidence["verifier_rows_with_argmax"], 0)

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

    def test_gpu_fault_re_covers_i915_and_xe(self) -> None:
        faults = [
            "[  812.101] i915 0000:03:00.0: [drm] GPU HANG: ecode 12:1:85dffffb, in llama-server [4242]",
            "[  812.102] i915 0000:03:00.0: [drm] Resetting rcs0 for preemption time out",
            "[  812.103] xe 0000:03:00.0: [drm] GT0: Engine reset: engine_class=rcs, logical_mask: 0x1",
            "[  812.104] xe 0000:03:00.0: [drm] GT0: GuC load failed",
            "[  812.105] xe 0000:03:00.0: [drm] device lost",
            # matched by no other word of the pattern: spaced and past-tense timeouts, a wedged device
            "[  812.109] xe 0000:03:00.0: [drm] Tile0: GT0: Timedout job: seqno=7811, lrc_seqno=7811, flags=0x20",
            "[  812.110] i915 0000:03:00.0: Fence expiration time out i915-0000:03:00.0:test-backend-op[3758008]",
            "[  812.111] xe 0000:03:00.0: [drm] Tile0: GT0: timed out waiting for the engine to idle",
            "[  812.112] xe 0000:03:00.0: [drm] device wedged, needs recovery",
        ]
        clean = [
            "[  812.106] pci 0000:03:00.0: reset complete",
            "[  812.107] usb 1-2: reset high-speed USB device number 3 using xhci_hcd",
            "[  812.108] xe 0000:03:00.0: [drm] Found dg2/g10 (device ID 56a0) display version 13.00",
        ]

        for line in faults:
            self.assertIsNotNone(BENCH_SPEC.GPU_FAULT_RE.search(line), line)
        for line in clean:
            self.assertIsNone(BENCH_SPEC.GPU_FAULT_RE.search(line), line)

    def test_render_driver_names_the_bound_kernel_driver(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            drm = Path(root) / "class" / "drm"
            driver = Path(root) / "bus" / "pci" / "drivers" / "xe"
            (drm / "renderD128" / "device").mkdir(parents=True)
            driver.mkdir(parents=True)
            (drm / "renderD128" / "device" / "driver").symlink_to(driver)

            self.assertEqual(BENCH_SPEC.render_driver("/dev/dri/renderD128", str(drm)), "xe")
            self.assertIsNone(BENCH_SPEC.render_driver("/dev/dri/renderD129", str(drm)))

    def test_build_hashes_cover_every_library_of_the_build(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            bin_dir = Path(root)
            exe = bin_dir / "llama-server"
            exe.write_bytes(b"server")
            (bin_dir / "libllama-server-impl.so").write_bytes(b"impl")
            (bin_dir / "libllama.so.0.5.0").write_bytes(b"llama a")
            (bin_dir / "libllama.so.0").symlink_to("libllama.so.0.5.0")
            (bin_dir / "libllama.so").symlink_to("libllama.so.0")
            (bin_dir / "test-foo").write_bytes(b"not loaded by the server")

            hashes = BENCH_SPEC.build_hashes(exe)
            # one entry per file: the symlinks name the same library
            self.assertEqual(sorted(hashes), ["libllama-server-impl.so", "libllama.so.0.5.0", "llama-server"])
            # a change confined to a core library shows up although the server binary is unchanged
            (bin_dir / "libllama.so.0.5.0").write_bytes(b"llama b")
            changed = BENCH_SPEC.build_hashes(exe)
            self.assertEqual(changed["llama-server"], hashes["llama-server"])
            self.assertNotEqual(changed["libllama.so.0.5.0"], hashes["libllama.so.0.5.0"])

    def test_kmsg_lines_since_returns_only_appended_lines(self) -> None:
        self.assertEqual(BENCH_SPEC.kmsg_lines_since(["a", "b"], ["a", "b", "c"]), ["c"])
        self.assertEqual(BENCH_SPEC.kmsg_lines_since(["a", "b"], ["a", "b"]), [])
        # the ring buffer dropped its oldest lines while the run appended new ones
        self.assertEqual(BENCH_SPEC.kmsg_lines_since(["a", "b", "c"], ["c", "d", "e"]), ["d", "e"])
        self.assertEqual(BENCH_SPEC.kmsg_lines_since(["a", "b", "a", "b"], ["b", "a", "b", "c"]), ["c"])

    def test_kmsg_lines_since_is_indeterminate_without_overlap(self) -> None:
        # wrapped past the first read, cleared, or nothing to anchor on
        self.assertIsNone(BENCH_SPEC.kmsg_lines_since(["a", "b"], ["x", "y"]))
        self.assertIsNone(BENCH_SPEC.kmsg_lines_since(["a", "b"], []))
        self.assertIsNone(BENCH_SPEC.kmsg_lines_since([], ["a"]))

    def test_new_gpu_faults_survives_eviction_of_an_identical_old_fault(self) -> None:
        fault = "xe 0000:03:00.0: [drm] GT0: Engine reset: engine_class=rcs"
        before = [fault, "usb 1-2: new device"]
        after = ["usb 1-2: new device", fault]

        # counting fault lines sees one before and one after and reports nothing new
        self.assertEqual(BENCH_SPEC.new_gpu_faults(before, after), [fault])
        self.assertEqual(BENCH_SPEC.new_gpu_faults(before, before), [])
        self.assertIsNone(BENCH_SPEC.new_gpu_faults(None, after))
        self.assertIsNone(BENCH_SPEC.new_gpu_faults(before, None))
        self.assertIsNone(BENCH_SPEC.new_gpu_faults(before, ["unrelated"]))

    def test_gpu_holders_only_clears_an_idle_render_node(self) -> None:
        def fuser(returncode: int, stdout: str = "", stderr: str = "") -> mock.Mock:
            return mock.Mock(returncode=returncode, stdout=stdout, stderr=stderr)

        with mock.patch.object(BENCH_SPEC.subprocess, "run", return_value=fuser(1)):
            self.assertEqual(BENCH_SPEC.gpu_holders(), [])
        with mock.patch.object(BENCH_SPEC.subprocess, "run",
                               return_value=fuser(0, " 4242", "/dev/dri/renderD128:")):
            self.assertTrue(BENCH_SPEC.gpu_holders())
        # fuser also exits 1 for a node that does not exist; only stderr tells that apart
        with mock.patch.object(BENCH_SPEC.subprocess, "run",
                               return_value=fuser(1, "", "Specified filename /dev/dri/renderD129 does not exist.")):
            self.assertTrue(BENCH_SPEC.gpu_holders())
        with mock.patch.object(BENCH_SPEC.subprocess, "run", return_value=fuser(0)):
            self.assertTrue(BENCH_SPEC.gpu_holders())
        with mock.patch.object(BENCH_SPEC.subprocess, "run", side_effect=OSError("no fuser")):
            self.assertTrue(BENCH_SPEC.gpu_holders())

    def test_launch_problems_names_what_makes_a_launch_unusable(self) -> None:
        good = {"error": None, "spec_stats_missing": False,
                "prompts": [{"id": "p", "tg_median": 12.0, "all_verifier_invariants_ok": True}]}
        self.assertEqual(BENCH_SPEC.launch_problems(good), [])

        self.assertEqual(BENCH_SPEC.launch_problems({"error": "health_timeout", "prompts": []}), ["health_timeout"])
        # a speculative arm that never drafted measures something else than it claims
        self.assertEqual(BENCH_SPEC.launch_problems({**good, "spec_stats_missing": True}), ["no draft statistics"])
        # a generated token that is not the target's argmax, or a non-finite log-probability
        bad_tokens = {**good, "prompts": [{"id": "p", "tg_median": 12.0, "all_verifier_invariants_ok": False}]}
        self.assertEqual(BENCH_SPEC.launch_problems(bad_tokens), ["p: response failed the target-argmax verifier"])
        no_timing = {**good, "prompts": [{"id": "p", "tg_median": None, "all_verifier_invariants_ok": True}]}
        self.assertEqual(BENCH_SPEC.launch_problems(no_timing), ["p: no timings"])

    def test_library_path_has_no_empty_entry(self) -> None:
        self.assertEqual(BENCH_SPEC.library_path("/build/bin", "/opt/lib:/usr/lib"), "/build/bin:/opt/lib:/usr/lib")
        # a trailing ":" would make the loader search the current directory
        self.assertEqual(BENCH_SPEC.library_path("/build/bin", ""), "/build/bin")

    def test_run_ab_refuses_a_launch_count_below_one(self) -> None:
        arms = [{"name": "a", "server_bin": "/nonexistent/a/llama-server"},
                {"name": "b", "server_bin": "/nonexistent/b/llama-server"}]
        for launches in (0, -1):
            with mock.patch.object(BENCH_SPEC, "LAUNCHES", launches):
                self.assertEqual(BENCH_SPEC.run_ab(arms, []), BENCH_SPEC.EXIT_USAGE)

    def test_launch_order_is_balanced_only_for_an_even_count(self) -> None:
        self.assertEqual(BENCH_SPEC.launch_order(4), [(0, 1), (1, 0), (0, 1), (1, 0)])
        for launches, balanced in ((1, False), (2, True), (3, False), (4, True)):
            first = [pair[0] for pair in BENCH_SPEC.launch_order(launches)]
            self.assertEqual(first.count(0) == first.count(1), balanced, launches)

    def test_ab_exit_code_fails_closed_on_an_unevaluated_fault_gate(self) -> None:
        self.assertEqual(BENCH_SPEC.ab_exit_code(True, []), 0)
        self.assertEqual(BENCH_SPEC.ab_exit_code(True, ["xe 0000:03:00.0: [drm] GT0: Engine reset"]), 1)
        self.assertEqual(BENCH_SPEC.ab_exit_code(True, None), 1)
        self.assertEqual(BENCH_SPEC.ab_exit_code(False, []), 1)


if __name__ == "__main__":
    unittest.main()
