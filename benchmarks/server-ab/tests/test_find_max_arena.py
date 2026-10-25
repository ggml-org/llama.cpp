import importlib.util
import json
import os
from pathlib import Path
import tempfile
import socket
import shlex
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "find-max-arena.py"
SPEC = importlib.util.spec_from_file_location("arena_finder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FINDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINDER)


class ArenaFinderTests(unittest.TestCase):
    def args(self, *extra):
        return FINDER.arguments(["--model", "model.gguf", "--context-size", "8192", *extra])

    def test_preflight_caps_history_but_keeps_native_capacity_and_requested_decode(self):
        args = self.args("--context-size", "262144", "--probe-mode", "preflight",
                         "--spec-mode", "mtp", "--model-draft", "assistant.gguf", "--mmproj", "vision.gguf")
        self.assertEqual(FINDER.probe_text_tokens(args), 2048)
        self.assertEqual(args.decode_tokens, 256)
        command = FINDER.server_command(args, 1024)
        self.assertEqual(command[command.index("--ctx-size") + 1], "262144")
        self.assertIn("--model-draft", command)
        self.assertIn("--mmproj", command)
        self.assertNotIn("--probe-mode", command)
        self.assertNotIn("--no-warmup", command)

    def test_preflight_exercises_wide_batches_and_never_exceeds_requested_history(self):
        for options, expected in [(("--batch-size", "1280", "--ubatch-size", "1280"), 2560),
                                  (("--batch-size", "1280", "--ubatch-size", "0"), 2560),
                                  (("--prompt-tokens", "17"), 17),
                                  (("--context-size", "512", "--decode-tokens", "8"), 489)]:
            with self.subTest(options=options):
                self.assertEqual(FINDER.probe_text_tokens(self.args("--probe-mode", "preflight", *options)), expected)
        self.assertEqual(FINDER.probe_text_tokens(self.args()), 7921)

    def test_selected_validation_is_explicit_and_requires_preflight_search(self):
        self.assertEqual(self.args().probe_mode, "full")
        self.assertFalse(self.args().validate_selected)
        self.assertTrue(self.args("--probe-mode", "preflight", "--validate-selected").validate_selected)
        with self.assertRaises(SystemExit):
            self.args("--validate-selected")

    def test_preflight_executes_a_real_request_and_rejects_runtime_failures_after_ready(self):
        args = self.args("--probe-mode", "preflight", "--context-size", "262144", "--decode-tokens", "8")
        with tempfile.TemporaryDirectory() as temporary:
            args.output = Path(temporary)
            args.prefill_text = Path(temporary) / "corpus.txt"
            args.prefill_text.write_text("corpus")
            def request(url, body: dict | None = None, **kwargs):
                if url.endswith("/tokenize"):
                    return {"tokens": list(range(4096))}
                assert body is not None
                self.assertEqual(len(body["prompt"]), 2048)
                self.assertEqual(body["n_predict"], 8)
                return {"tokens": list(range(8)), "timings": {"prompt_n": 2048, "predicted_n": 8}}
            with patch.object(FINDER.subprocess, "Popen") as popen, patch.object(FINDER.SWEEP, "wait_ready"), \
                    patch.object(FINDER.SWEEP, "stop_process") as stop, patch.object(FINDER, "request_json", side_effect=request):
                self.assertTrue(FINDER.probe(args, 1000)["fits"])
                stop.assert_called_once_with(popen.return_value)
            with patch.object(FINDER.subprocess, "Popen"), patch.object(FINDER.SWEEP, "wait_ready"), \
                    patch.object(FINDER.SWEEP, "stop_process"), \
                    patch.object(FINDER, "request_json", side_effect=RuntimeError("runtime kernel rejected")):
                with self.assertRaisesRegex(RuntimeError, "runtime kernel rejected"):
                    FINDER.probe(args, 1001)

    def test_preflight_does_not_truncate_or_skip_an_explicit_image_request(self):
        args = self.args("--probe-mode", "preflight", "--mmproj", "vision.gguf", "--image", "image.png", "--decode-tokens", "8")
        with tempfile.TemporaryDirectory() as temporary:
            args.output = Path(temporary)
            args.image = [Path(temporary) / "image.png"]
            args.image[0].write_bytes(b"fixture image")
            def request(url, body: dict | None = None, **kwargs):
                self.assertTrue(url.endswith("/completion"))
                assert body is not None
                self.assertEqual(len(body["prompt"]["multimodal_data"]), 1)
                self.assertEqual(body["n_predict"], 8)
                return {"tokens": list(range(8)), "timings": {"prompt_n": 70, "predicted_n": 8}}
            with patch.object(FINDER.subprocess, "Popen"), patch.object(FINDER.SWEEP, "wait_ready"), \
                    patch.object(FINDER.SWEEP, "stop_process"), patch.object(FINDER, "request_json", side_effect=request):
                self.assertTrue(FINDER.probe(args, 1000)["fits"])

    def test_preflight_report_cannot_claim_full_length_or_unexecuted_vision(self):
        args = self.args("--probe-mode", "preflight", "--context-size", "262144", "--mmproj", "vision.gguf")
        with tempfile.TemporaryDirectory() as temporary:
            args.output = Path(temporary)
            def probe(config, candidate):
                return {"arena_mib": candidate, "fits": candidate <= 1000,
                        "timings": {"prompt_n": 2048, "predicted_n": 256}}
            with patch.object(FINDER, "probe", side_effect=probe):
                report = FINDER.fit(args)
        self.assertEqual(report["probe_mode"], "preflight")
        self.assertEqual(report["prompt_tokens"], 2048)
        self.assertEqual(report["requested_prompt_tokens"], 261873)
        self.assertFalse(report["full_workload_validated"])
        self.assertFalse(report["vision_execution_validated"])

    def test_selected_full_validation_runs_once_in_separate_logs(self):
        args = self.args("--probe-mode", "preflight", "--validate-selected", "--context-size", "262144")
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            args.output = Path(temporary)
            def probe(config, candidate):
                calls.append((config.probe_mode, candidate, config.output))
                return {"arena_mib": candidate, "fits": candidate <= 1000,
                        "timings": {"prompt_n": FINDER.probe_text_tokens(config), "predicted_n": 256}}
            with patch.object(FINDER, "probe", side_effect=probe):
                report = FINDER.fit(args)
            self.assertTrue(report["full_workload_validated"])
            self.assertEqual(report["selected_validation"]["timings"]["prompt_n"], 261873)
            self.assertEqual([mode for mode, _, _ in calls].count("full"), 1)
            self.assertEqual(calls[-1], ("full", 1000, args.output / "selected-validation"))
            self.assertTrue((args.output / "selected-validation/record.json").is_file())

    def test_selected_runtime_failure_never_publishes_a_successful_fit(self):
        args = self.args("--probe-mode", "preflight", "--validate-selected")
        with tempfile.TemporaryDirectory() as temporary:
            args.output = Path(temporary)
            def probe(config, candidate):
                return {"arena_mib": candidate, "fits": config.probe_mode != "full" and candidate <= 1000}
            with patch.object(FINDER, "probe", side_effect=probe):
                with self.assertRaisesRegex(RuntimeError, "selected.*full"):
                    FINDER.fit(args)
            self.assertFalse((args.output / "result.json").exists())
            self.assertTrue((args.output / "probes.jsonl").is_file())
            self.assertTrue((args.output / "selected-validation/record.json").is_file())

    def test_preflight_rejects_partial_or_non_speculative_runtime_responses(self):
        args = self.args("--probe-mode", "preflight", "--spec-mode", "mtp", "--decode-tokens", "8")
        for response in ({"tokens": [1], "timings": {"prompt_n": 2048, "predicted_n": 8, "draft_n": 3}},
                         {"tokens": list(range(8)), "timings": {"prompt_n": 2047, "predicted_n": 8, "draft_n": 3}},
                         {"tokens": list(range(8)), "timings": {"prompt_n": 2048, "predicted_n": 8, "draft_n": 0}}):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as temporary:
                args.output = Path(temporary)
                args.prefill_text = args.output / "corpus.txt"
                args.prefill_text.write_text("corpus")
                def request(url, *unused, **kwargs):
                    return {"tokens": list(range(2048))} if url.endswith("/tokenize") else response
                with patch.object(FINDER.subprocess, "Popen"), patch.object(FINDER.SWEEP, "wait_ready"), \
                        patch.object(FINDER.SWEEP, "stop_process"), patch.object(FINDER, "request_json", side_effect=request):
                    with self.assertRaises(RuntimeError):
                        FINDER.probe(args, 1000)

    def test_preflight_never_classifies_a_device_fault_as_an_oom_capacity_boundary(self):
        args = self.args("--probe-mode", "preflight")
        for fault in ("an illegal memory access was encountered", "misaligned address", "device-side assert triggered", "unspecified launch failure"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                args.output = Path(temporary)
                def launch(*unused, **kwargs):
                    kwargs["stdout"].write("CUDA error: " + fault + "\ncudaMalloc failed: out of memory\n")
                    return object()
                with patch.object(FINDER.subprocess, "Popen", side_effect=launch), \
                        patch.object(FINDER.SWEEP, "wait_ready", side_effect=RuntimeError("failed execution")), \
                        patch.object(FINDER.SWEEP, "stop_process"):
                    with self.assertRaisesRegex(RuntimeError, "device fault"):
                        FINDER.probe(args, 1000)

    def test_selected_runtime_exception_is_preserved_and_never_published(self):
        args = self.args("--probe-mode", "preflight", "--validate-selected")
        with tempfile.TemporaryDirectory() as temporary:
            args.output = Path(temporary)
            def probe(config, candidate):
                if config.probe_mode == "full":
                    raise RuntimeError("unsupported full-runtime kernel")
                return {"arena_mib": candidate, "fits": candidate <= 1000}
            with patch.object(FINDER, "probe", side_effect=probe):
                with self.assertRaisesRegex(RuntimeError, "unsupported full-runtime kernel"):
                    FINDER.fit(args)
            self.assertFalse((args.output / "result.json").exists())
            record = json.loads((args.output / "selected-validation/record.json").read_text())
            self.assertFalse(record["fits"])
            self.assertIn("unsupported full-runtime kernel", record["error"])

    @unittest.skipUnless(os.name == "posix", "POSIX listener restart contract")
    def test_completed_probe_socket_does_not_block_the_next_probe(self):
        with socket.socket() as listener, socket.socket() as client:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            address = listener.getsockname()
            client.settimeout(5)
            client.connect(address)
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(5)
                connection.shutdown(socket.SHUT_WR)
                self.assertEqual(client.recv(1), b"")
                client.close()
                self.assertEqual(connection.recv(1), b"")
        with socket.socket() as closed:
            with self.assertRaises(OSError):
                closed.bind(address)
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args("--port", str(address[1]))
            args.output = Path(temporary)
            with patch.object(FINDER.subprocess, "Popen", side_effect=RuntimeError("after preflight")) as popen:
                with self.assertRaisesRegex(RuntimeError, "after preflight"):
                    FINDER.probe(args, 1024)
                popen.assert_called_once()

    def test_mtp_allows_an_optional_external_assistant_with_inherited_cache_types(self):
        args=self.args("--spec-mode","mtp","--model-draft","assistant.gguf","--cache-type-k","q5_1","--cache-type-v","q8_0")
        command=FINDER.server_command(args,1024)
        self.assertEqual(command[command.index("--model-draft")+1],"assistant.gguf")
        self.assertEqual(command[command.index("--spec-draft-ngl")+1],"999")
        self.assertNotIn("--spec-draft-type-k",command)
        self.assertNotIn("--spec-draft-type-v",command)
        self.assertNotIn("--model-draft",FINDER.server_command(self.args("--spec-mode","mtp"),1024))
        with self.assertRaises(SystemExit):
            self.args("--spec-mode","none","--model-draft","assistant.gguf")

    def test_fitted_report_records_the_external_assistant(self):
        args=self.args("--spec-mode","mtp","--model-draft","assistant.gguf")
        with tempfile.TemporaryDirectory() as temp:
            args.output=Path(temp)
            with patch.object(FINDER.SWEEP,"find_max_arena_mib",return_value=1024):
                report=FINDER.fit(args)
        self.assertEqual(report["model_draft"],"assistant.gguf")
        self.assertIn("assistant.gguf",report["shell_command"])

    def test_persistent_cache_quota_rejection_keeps_its_exact_lower_bound(self):
        args=self.args()
        with tempfile.TemporaryDirectory() as temp:
            args.output=Path(temp)
            def reject(*command,**kwargs):
                kwargs["stdout"].write("persistent local cache does not fit the declared shared arena: insufficient arena quota; required=788880128, available=536870912, additional=252009216 bytes; phase=1, resource=6\n")
                return object()
            with patch.object(FINDER.subprocess,"Popen",side_effect=reject), \
                    patch.object(FINDER.SWEEP,"wait_ready",side_effect=RuntimeError("startup")), \
                    patch.object(FINDER.SWEEP,"stop_process"),self.assertRaises(FINDER.SWEEP.ArenaTooSmallError) as failure:
                FINDER.probe(args,512)
            self.assertEqual(failure.exception.minimum_mib,753)

    def test_initial_arena_is_an_optional_guess_not_a_final_arena_setting(self):
        self.assertEqual(self.args().initial_arena_mib, 2240)
        self.assertEqual(self.args("--initial-arena-mib", "1000").initial_arena_mib, 1000)
        with self.assertRaises(SystemExit):
            self.args("--arena-mib", "1000")

    def test_commands_cover_target_mtp_dflash_and_vision_without_draft_overrides(self):
        for mode, depth in (("none", 0), ("mtp", 3), ("dflash", 7)):
            args = self.args("--spec-mode", mode, *(["--model-draft", "draft.gguf"] if mode == "dflash" else []))
            command = FINDER.server_command(args, 1234)
            self.assertEqual(command[command.index("--shared-device-memory-mib")+1], "1234")
            self.assertEqual(args.draft_length, depth)
            self.assertEqual(command[command.index("--spec-type")+1], {"none":"none","mtp":"draft-mtp","dflash":"draft-dflash"}[mode])
            self.assertNotIn("--spec-draft-type-k", command)
            self.assertNotIn("--spec-draft-type-v", command)
        for mode in ("none", "mtp"):
            args = self.args("--spec-mode", mode, "--mmproj", "vision.gguf")
            command = FINDER.server_command(args, 2240)
            self.assertIn("--mmproj", command)
            self.assertNotIn("--no-mmproj", command)
            self.assertIn("--no-context-shift", command)

    def test_invalid_combinations_are_rejected_before_starting_a_server(self):
        for options in (("--spec-mode","mtp","--draft-length","4"),
                ("--spec-mode","none","--draft-length","3"), ("--spec-mode","dflash"),
                ("--spec-mode","dflash","--model-draft","draft","--mmproj","vision","--image","x"),
                ("--mmproj","vision","--prompt-file","text"), ("--image","x"), ("--prompt-tokens","8192"),
                ("--batch-size","0"), ("--decode-tokens","0"), ("--initial-arena-mib","0"),
                ("--request-timeout","nan"), ("--port","70000")):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                self.args(*options)

    def test_cache_choices_and_environment_do_not_enable_uvm_or_hidden_overrides(self):
        args = self.args("--context-size", "512", "--decode-tokens", "8", "--ubatch-size", "0")
        self.assertEqual(args.context_size, 512)
        for key in FINDER.SWEEP.KV_TYPES:
            args = self.args("--cache-type-k", key, "--cache-type-v", "q5_1")
            command = FINDER.server_command(args, 1000)
            self.assertEqual(command[command.index("--cache-type-k")+1], key)
        with patch.dict(FINDER.os.environ, {"GGML_CUDA_ENABLE_UNIFIED_MEMORY":"1","GGML_CUDA_DISABLE_GRAPHS":"1","LLAMA_ARG_MODEL":"wrong","KEEP_ME":"yes"}):
            env = FINDER.environment()
        self.assertFalse("GGML_CUDA_ENABLE_UNIFIED_MEMORY" in env)
        self.assertFalse("GGML_CUDA_DISABLE_GRAPHS" in env)
        self.assertFalse("LLAMA_ARG_MODEL" in env)
        self.assertEqual(env["KEEP_ME"], "yes")

    def test_text_and_multiple_image_payloads_keep_the_workload_explicit(self):
        args = self.args()
        payload = FINDER.payload(args, [1,2,3], [])
        self.assertEqual(payload["prompt"], [1,2,3])
        self.assertFalse(payload["cache_prompt"])
        args = self.args("--mmproj", "vision", "--image", "a", "--image", "b")
        payload = FINDER.payload(args, "Inspect <__media__> and <__media__>", ["one", "two"])
        self.assertEqual(payload["prompt"]["multimodal_data"], ["one", "two"])
        with self.assertRaisesRegex(ValueError, "marker"):
            FINDER.payload(args, "no markers", ["one", "two"])

    def test_output_validation_rejects_partial_generation_and_missing_drafts(self):
        args = self.args("--decode-tokens", "8")
        valid = {"tokens":list(range(8)), "timings":{"predicted_n":8,"prompt_n":7921}}
        FINDER.validate_result(args, valid, 7921)
        for result in ({"tokens":[1],"timings":{"predicted_n":8,"prompt_n":7921}},
                {"tokens":list(range(8)),"timings":{"predicted_n":8,"prompt_n":7920}}):
            with self.assertRaises(RuntimeError):
                FINDER.validate_result(args, result, 7921)
        args = self.args("--spec-mode", "mtp", "--decode-tokens", "8")
        with self.assertRaisesRegex(RuntimeError, "draft"):
            FINDER.validate_result(args, valid, 7921)

    def test_search_only_accepts_tested_workloads_and_preserves_probe_reports(self):
        args = self.args("--initial-arena-mib", "1000")
        with tempfile.TemporaryDirectory() as temp:
            args.output = Path(temp)
            def probe(config, candidate):
                if candidate < 979:
                    raise FINDER.SWEEP.ArenaTooSmallError(979)
                return {"arena_mib":candidate,"fits":candidate <= 1066,"log":str(candidate)}
            with patch.object(FINDER, "probe", side_effect=probe):
                report = FINDER.fit(args)
            self.assertEqual(report["arena_mib"], 1066)
            self.assertEqual(report["initial_arena_mib"], 1000)
            self.assertEqual(report["next_mib_rejected"], 1067)
            self.assertTrue((args.output/"result.json").is_file())
            launch = shlex.split(report["shell_command"])
            self.assertEqual(launch[0], "env")
            self.assertIn("GGML_CUDA_ENABLE_UNIFIED_MEMORY", launch)
            self.assertFalse(report["vision_execution_validated"])
            records = [json.loads(line) for line in (args.output/"probes.jsonl").read_text().splitlines()]
            self.assertTrue(any(row.get("arena_mib") == 1067 and not row.get("fits") for row in records))
            with patch.object(FINDER, "probe", side_effect=RuntimeError("kernel unsupported")), self.assertRaisesRegex(RuntimeError,"kernel unsupported"):
                FINDER.fit(args)

    def test_probe_stops_only_its_child_and_does_not_hide_non_oom_errors(self):
        args = self.args("--decode-tokens", "8", "--prompt-tokens", "4")
        with tempfile.TemporaryDirectory() as temp:
            args.output = Path(temp)
            args.prefill_text = Path(temp)/"corpus.txt"
            args.prefill_text.write_text("article text")
            def request(url, body: dict | None = None, **kwargs):
                if url.endswith("/tokenize"):
                    return {"tokens":[1,2,3,4]}
                return {"tokens":list(range(8)),"timings":{"prompt_n":4,"predicted_n":8}}
            with patch.object(FINDER.subprocess,"Popen") as popen, patch.object(FINDER.SWEEP,"wait_ready"), \
                    patch.object(FINDER.SWEEP,"stop_process") as stop, patch.object(FINDER,"request_json",side_effect=request):
                self.assertTrue(FINDER.probe(args,1000)["fits"])
                stop.assert_called_once_with(popen.return_value)
            with patch.object(FINDER.subprocess,"Popen") as popen, patch.object(FINDER.SWEEP,"wait_ready",side_effect=RuntimeError("bad kernel")), \
                    patch.object(FINDER.SWEEP,"stop_process") as stop, self.assertRaisesRegex(RuntimeError,"bad kernel"):
                FINDER.probe(args,1001)
            stop.assert_called_once_with(popen.return_value)

    def test_an_occupied_port_cannot_be_mistaken_for_the_new_server(self):
        args = self.args()
        for reusable in ((False, True) if os.name == "posix" else (False,)):
            with self.subTest(reusable=reusable), socket.socket() as listener, tempfile.TemporaryDirectory() as temp:
                if reusable:
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", 0))
                listener.listen()
                args.port = listener.getsockname()[1]
                args.output = Path(temp)
                with patch.object(FINDER.subprocess, "Popen") as popen, self.assertRaisesRegex(RuntimeError, "port"):
                    FINDER.probe(args, 1000)
                popen.assert_not_called()

    def test_memory_failures_keep_lower_quota_and_upper_oom_bounds_distinct(self):
        args = self.args()
        with tempfile.TemporaryDirectory() as temp:
            args.output = Path(temp)
            def allocate(*command, **kwargs):
                kwargs["stdout"].write("CUDA error: out of memory\n")
                return object()
            with patch.object(FINDER.subprocess,"Popen",side_effect=allocate), patch.object(FINDER.SWEEP,"wait_ready",side_effect=RuntimeError("startup")), \
                    patch.object(FINDER.SWEEP,"stop_process"):
                self.assertFalse(FINDER.probe(args,1000)["fits"])
            def too_small(*command, **kwargs):
                kwargs["stdout"].write("shared arena quota insufficient: required=503316480 bytes\n")
                return object()
            with patch.object(FINDER.subprocess,"Popen",side_effect=too_small), patch.object(FINDER.SWEEP,"wait_ready",side_effect=RuntimeError("startup")), \
                    patch.object(FINDER.SWEEP,"stop_process"), self.assertRaises(FINDER.SWEEP.ArenaTooSmallError) as failure:
                FINDER.probe(args,400)
            self.assertEqual(failure.exception.minimum_mib,480)

    def test_search_does_not_claim_an_untested_upper_neighbor(self):
        args = self.args("--initial-arena-mib", "1000")
        with tempfile.TemporaryDirectory() as temp:
            args.output = Path(temp)
            def search(start, probe):
                self.assertTrue(probe(1000))
                return 1000
            with patch.object(FINDER.SWEEP,"find_max_arena_mib",side_effect=search), \
                    patch.object(FINDER,"probe",return_value={"arena_mib":1000,"fits":True}):
                self.assertIsNone(FINDER.fit(args)["next_mib_rejected"])

    def test_projector_only_configuration_fits_text_without_loading_an_image(self):
        args = self.args("--mmproj","projector.gguf","--prompt-tokens","4","--decode-tokens","8")
        with tempfile.TemporaryDirectory() as temp:
            args.output = Path(temp)
            args.prefill_text = Path(temp)/"corpus.txt"
            args.prefill_text.write_text("article")
            def request(url, body: dict | None = None, **kwargs):
                if url.endswith("/tokenize"):
                    return {"tokens":[1,2,3,4]}
                assert body is not None
                self.assertEqual(body["prompt"], [1,2,3,4])
                return {"tokens":list(range(8)),"timings":{"prompt_n":4,"predicted_n":8}}
            with patch.object(FINDER.subprocess,"Popen"), patch.object(FINDER.SWEEP,"wait_ready"), \
                    patch.object(FINDER.SWEEP,"stop_process"), patch.object(FINDER,"request_json",side_effect=request):
                self.assertTrue(FINDER.probe(args,1000)["fits"])


if __name__ == "__main__":
    unittest.main()
