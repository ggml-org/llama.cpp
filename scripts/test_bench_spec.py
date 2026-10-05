#!/usr/bin/env python3

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
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

    def test_analyze_native_response_rejects_rows_without_argmax_evidence(self) -> None:
        # the server fills the top list of accepted draft tokens too, so a row without one proves nothing
        for top in (None, [], [{"logprob": -0.3}], [{"id": "17", "logprob": -0.3}]):
            row = {"id": 17, "logprob": -0.3}
            if top is not None:
                row["top_logprobs"] = top
            response = {"content": "draft", "tokens": [17], "completion_probabilities": [row]}

            evidence = BENCH_SPEC.analyze_native_response(response)

            self.assertFalse(evidence["verifier_invariant_ok"], top)
            self.assertEqual(evidence["verifier_failures"],
                             [{"index": 0, "token": 17, "reason": "missing_target_argmax"}], top)
            self.assertEqual(evidence["verifier_rows_with_argmax"], 0, top)

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
            # pr_notice without a device prefix: the driver name comes after the event
            "[  812.110] Fence expiration time out i915-0000:03:00.0:test-backend-op[3758008]:1a!",
            "[  812.111] xe 0000:03:00.0: [drm] Tile0: GT0: timed out waiting for the engine to idle",
            "[  812.112] xe 0000:03:00.0: [drm] device wedged, needs recovery",
            # formats from the xe and i915 modules of kernel 7.3 with values filled in
            "[  812.113] xe 0000:03:00.0: [drm] Tile0: GT0: Fault response: Unsuccessful -EACCES",
            "[  812.114] xe 0000:03:00.0: [drm] PageFault Queue (0) full, shouldn't be possible",
            "[  812.115] xe 0000:03:00.0: [drm] *ERROR* [CRTC:88:pipe A] DSB 0 GTT fault",
            "[  812.116] i915 0000:03:00.0: [drm] context llama-server[4242]: guilty 1, banned",
            "[  812.117] xe 0000:03:00.0: [drm] *ERROR* Tile0: GT0: GSC ER timed-out",
        ]
        clean = [
            "[  812.106] pci 0000:03:00.0: reset complete",
            "[  812.107] usb 1-2: reset high-speed USB device number 3 using xhci_hcd",
            "[  812.108] xe 0000:03:00.0: [drm] Found dg2/g10 (device ID 56a0) display version 13.00",
            # "fault" inside another word is no fault
            "[  812.118] xe 0000:03:00.0: [drm] Using default_page_size: 64KiB",
        ]

        for line in faults:
            self.assertTrue(BENCH_SPEC.is_gpu_fault(line), line)
        for line in clean:
            self.assertFalse(BENCH_SPEC.is_gpu_fault(line), line)

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

    def test_workload_mismatches_name_pairs_that_generated_different_tokens(self) -> None:
        def launch(*prompts: tuple[str, list[str]]) -> dict:
            return {"prompts": [{"id": pid, "runs": [{"token_sha256": h} for h in hashes]} for pid, hashes in prompts]}

        a = [launch(("p1", ["x", "x"]), ("p2", ["y"])), launch(("p1", ["x", "x"]), ("p2", ["y"]))]
        same = [launch(("p2", ["y"]), ("p1", ["x", "x"])), launch(("p1", ["x", "x"]), ("p2", ["y"]))]
        self.assertEqual(BENCH_SPEC.workload_mismatches(a, same, ["p1", "p2"]), [])

        # the second pair generated another stream for p1, so its delta mixes speed with workload
        other = [launch(("p1", ["x", "x"]), ("p2", ["y"])), launch(("p1", ["x", "z"]), ("p2", ["y"]))]
        self.assertEqual(BENCH_SPEC.workload_mismatches(a, other, ["p1", "p2"]), [{"launch": 1, "prompt": "p1"}])
        # a prompt missing on one side does not count as matched
        self.assertEqual(BENCH_SPEC.workload_mismatches(a, [launch(("p1", ["x", "x"])), a[1]], ["p1", "p2"]),
                         [{"launch": 0, "prompt": "p2"}])

    def test_argmax_evidence_counts_rows_over_every_run(self) -> None:
        launches = [{"prompts": [{"id": "p1", "runs": [{"verifier_rows": 4, "verifier_rows_with_argmax": 1},
                                                       {"verifier_rows": 4, "verifier_rows_with_argmax": 4}]}]},
                    {"prompts": [{"id": "p1", "runs": [{"verifier_rows": 2, "verifier_rows_with_argmax": 0}]}]}]
        self.assertEqual(BENCH_SPEC.argmax_evidence(launches), {"rows": 10, "rows_with_argmax": 5})
        # a summary from before the count has no count: unknown, not zero rows with evidence
        launches[1]["prompts"][0]["runs"][0].pop("verifier_rows_with_argmax")
        self.assertEqual(BENCH_SPEC.argmax_evidence(launches), {"rows": 10, "rows_with_argmax": None})

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
        # an empty entry anywhere would make the loader search the current directory
        self.assertEqual(BENCH_SPEC.library_path("/build/bin", ""), "/build/bin")
        for inherited in ("/opt/lib:", ":/opt/lib", "/opt/lib::"):
            self.assertEqual(BENCH_SPEC.library_path("/build/bin", inherited), "/build/bin:/opt/lib", inherited)
        self.assertEqual(BENCH_SPEC.library_path("/build/bin", "/opt/lib::/other"), "/build/bin:/opt/lib:/other")

    def test_run_arm_fails_the_launch_when_the_warmup_fails(self) -> None:
        arm = {"name": "a", "kv": "q8_0", "spec_label": "none"}
        prompts = [{"id": "p1", "messages": [{"role": "user", "content": "hi"}]}]
        good = {"content": "ok", "tokens": [11], "timings": {"predicted_n": 1},
                "completion_probabilities": [{"id": 11, "logprob": -0.1, "top_logprobs": [{"id": 11, "logprob": -0.1}]}]}
        bad = {"content": "ok", "tokens": [11], "timings": {"predicted_n": 1},
               "completion_probabilities": [{"id": 11, "logprob": -0.1, "top_logprobs": [{"id": 12, "logprob": -0.05}]}]}

        def launch(warmup: Any) -> tuple[dict, mock.Mock]:
            run_prompt = mock.Mock(return_value={"id": "p1", "tg_median": 10.0, "accept_rate_median": None,
                                                 "completion_tokens": 1, "draft_reported": False})
            with mock.patch.multiple(BENCH_SPEC, start_server=mock.Mock(), stop_server=mock.Mock(),
                                     wait_health=mock.Mock(return_value=True),
                                     apply_chat_template=mock.Mock(return_value="prompt"),
                                     post_completion=mock.Mock(**warmup), run_prompt=run_prompt,
                                     scan_log=mock.Mock(return_value={})), \
                    mock.patch.object(BENCH_SPEC.time, "sleep"):
                return BENCH_SPEC.run_arm(arm, prompts), run_prompt

        # a request error: the first measured request would absorb the JIT work the warmup removes
        result, run_prompt = launch({"side_effect": OSError("connection reset")})
        self.assertIn("warmup", result["error"])
        run_prompt.assert_not_called()
        self.assertTrue(BENCH_SPEC.launch_problems(result))
        # a response that fails the verifier fails the launch as a measured one would
        result, run_prompt = launch({"return_value": bad})
        self.assertIn("warmup", result["error"])
        run_prompt.assert_not_called()
        # a clean warmup goes on to the measured prompts
        result, run_prompt = launch({"return_value": good})
        self.assertIsNone(result["error"])
        run_prompt.assert_called_once()

    def test_run_ab_refuses_equal_arm_names(self) -> None:
        # both arms would share one launch list, and one arm's launches could stand in for the other's
        arms = [{"name": "x", "server_bin": "/nonexistent/a/llama-server"},
                {"name": "x", "server_bin": "/nonexistent/b/llama-server"}]
        self.assertEqual(BENCH_SPEC.run_ab(arms, []), BENCH_SPEC.EXIT_USAGE)

    def test_scan_log_takes_flash_attention_evidence_from_runtime_lines_only(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            log = Path(root) / "server.log"
            log.write_text("load_model: loading model '/models/Qwen3.8-Flash-Next-IQ1_M.gguf'\n"
                           "llama_context: flash_attn    = enabled\n", encoding="utf-8")
            self.assertEqual(BENCH_SPEC.scan_log(log)["fa_lines"], ["llama_context: flash_attn    = enabled"])

    def test_resolved_libraries_parse_ldd(self) -> None:
        ldd = ("\tlinux-vdso.so.1 (0x00007ffd)\n"
               "\tlibllama.so.0 => /build/bin/libllama.so.0 (0x00007f00)\n"
               "\tlibggml-sycl.so.0 => not found\n"
               "\t/lib64/ld-linux-x86-64.so.2 (0x00007f01)\n")
        with mock.patch.object(BENCH_SPEC.subprocess, "run", return_value=mock.Mock(returncode=0, stdout=ldd)):
            self.assertEqual(BENCH_SPEC.resolved_libraries(Path("/build/bin/llama-server"), "/build/bin"),
                             {"libllama.so.0": "/build/bin/libllama.so.0", "libggml-sycl.so.0": "not found"})
        with mock.patch.object(BENCH_SPEC.subprocess, "run", side_effect=OSError("no ldd")):
            self.assertIsNone(BENCH_SPEC.resolved_libraries(Path("/build/bin/llama-server"), "/build/bin"))

    def test_build_hashes_follow_core_libraries_resolved_elsewhere(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            bin_dir, elsewhere = Path(root) / "bin", Path(root) / "lib"
            bin_dir.mkdir()
            elsewhere.mkdir()
            exe = bin_dir / "llama-server"
            exe.write_bytes(b"server")
            (elsewhere / "libllama.so.0").write_bytes(b"llama")
            (elsewhere / "libsycl.so.8").write_bytes(b"runtime")
            resolved = {"libllama.so.0": str(elsewhere / "libllama.so.0"), "libsycl.so.8": str(elsewhere / "libsycl.so.8")}

            hashes = BENCH_SPEC.build_hashes(exe, resolved)
            # llama and ggml code loaded from outside the build is hashed under its path, the runtime is not
            self.assertIn(str(elsewhere / "libllama.so.0"), hashes)
            self.assertNotIn(str(elsewhere / "libsycl.so.8"), hashes)

    def test_run_ab_refuses_a_launch_count_that_cannot_balance_the_order(self) -> None:
        arms = [{"name": "a", "server_bin": "/nonexistent/a/llama-server"},
                {"name": "b", "server_bin": "/nonexistent/b/llama-server"}]
        # an odd number of AB pairs leaves one arm earlier on average, so drift biases the delta
        for launches in (0, -1, 1, 3):
            with mock.patch.object(BENCH_SPEC, "LAUNCHES", launches), \
                    mock.patch.object(BENCH_SPEC, "resolved_libraries",
                                      side_effect=AssertionError("ran past the launch count check")):
                self.assertEqual(BENCH_SPEC.run_ab(arms, []), BENCH_SPEC.EXIT_USAGE, launches)

    def test_run_ab_refuses_a_driver_other_than_xe_or_i915(self) -> None:
        arms = [{"name": "a", "server_bin": "/nonexistent/a/llama-server"},
                {"name": "b", "server_bin": "/nonexistent/b/llama-server"}]
        # a run that cannot be told xe from i915 is no baseline for either (AGENTS.md, "Kernel Driver"), and the
        # fault gate knows only those two drivers: another GPU's faults would pass it
        for driver in (None, "amdgpu", "nouveau"):
            with mock.patch.object(BENCH_SPEC, "LAUNCHES", 2), \
                    mock.patch.object(BENCH_SPEC, "resolved_libraries", return_value={}), \
                    mock.patch.object(BENCH_SPEC, "render_driver", return_value=driver), \
                    mock.patch.object(BENCH_SPEC, "dmesg_lines", side_effect=AssertionError("ran past the driver check")):
                self.assertEqual(BENCH_SPEC.run_ab(arms, []), BENCH_SPEC.EXIT_USAGE, driver)

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
