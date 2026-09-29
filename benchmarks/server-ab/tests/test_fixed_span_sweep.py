import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "run-fixed-span-sweep.py"
SPEC = importlib.util.spec_from_file_location("fixed_span_sweep", SCRIPT)
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


class MtpSweepTests(unittest.TestCase):
    def test_default_sweep_includes_full_native_context(self):
        with patch("sys.argv", [str(SCRIPT), "--model", "model.gguf"]):
            args = SWEEP.arguments()
        self.assertEqual(args.max_context, 262144)
        contexts = list(range(args.min_context, args.max_context + 1, args.context_step))
        self.assertEqual(contexts[0], 8192)
        self.assertEqual(contexts[-1], 262144)
        self.assertEqual(len(contexts), 32)
        self.assertEqual(args.output.name, "fixed-span-8k-256k")

    def test_full_context_keeps_shared_prompt_and_verification_inside_capacity(self):
        with patch("sys.argv", [str(SCRIPT), "--model", "model.gguf", "--mtp-lengths", "0,1,2,3"]):
            args = SWEEP.arguments()
        prompt = SWEEP.prompt_tokens_for_context(262144, args.decode_tokens, args.mtp_lengths)
        self.assertEqual(prompt, 261884)
        self.assertEqual(prompt + args.decode_tokens + max(args.mtp_lengths) + 1, 262144)
        for length in args.mtp_lengths:
            command = SWEEP.server_command(args, 262144, length)
            self.assertEqual(command[command.index("--ctx-size") + 1], "262144")

    def test_auto_max_is_opt_in_and_uses_arena_as_seed(self):
        with patch("sys.argv", [str(SCRIPT), "--model", "model.gguf"]):
            fixed = SWEEP.arguments()
        self.assertFalse(fixed.auto_max_arena)
        self.assertEqual(fixed.arena_mib, 2368)
        with patch("sys.argv", [str(SCRIPT), "--model", "model.gguf", "--auto-max-arena", "--arena-mib", "2048"]):
            automatic = SWEEP.arguments()
        self.assertTrue(automatic.auto_max_arena)
        self.assertEqual(automatic.arena_mib, 2048)

    def test_auto_max_search_finds_exact_mib_from_below_and_above(self):
        for seed in (1024, 4096):
            tried = []
            def probe(arena_mib):
                tried.append(arena_mib)
                return arena_mib <= 2273
            self.assertEqual(SWEEP.find_max_arena_mib(seed, probe), 2273)
            self.assertIn(2274, tried)
            self.assertEqual(len(tried), len(set(tried)))

    def test_auto_max_search_reports_no_viable_arena_and_propagates_errors(self):
        with self.assertRaisesRegex(RuntimeError, "no allocatable arena"):
            SWEEP.find_max_arena_mib(1024, lambda arena_mib: False)
        def broken_probe(arena_mib):
            raise ValueError("not an OOM")
        with self.assertRaisesRegex(ValueError, "not an OOM"):
            SWEEP.find_max_arena_mib(1024, broken_probe)

    def test_auto_max_searches_each_mtp_mode_independently(self):
        args = SimpleNamespace(arena_mib=100, auto_max_arena=True)
        def fits(_args, _context, mtp_length, arena_mib, _logs, _url, *_extra):
            return arena_mib <= (120 if mtp_length == 0 else 110)
        with patch.object(SWEEP, "probe_arena", side_effect=fits):
            self.assertEqual(SWEEP.arena_for_point(args, 8192, 0, Path("/tmp/logs"), "http://localhost"), 120)
            self.assertEqual(SWEEP.arena_for_point(args, 8192, 3, Path("/tmp/logs"), "http://localhost"), 110)
        args.auto_max_arena = False
        with patch.object(SWEEP, "probe_arena", side_effect=AssertionError("fixed mode must not probe")):
            self.assertEqual(SWEEP.arena_for_point(args, 8192, 3, Path("/tmp/logs"), "http://localhost"), 100)

    def test_probe_classifies_oom_but_not_other_startup_errors(self):
        args = SimpleNamespace(server=Path("/build/llama-server"), model=Path("/models/model.gguf"),
                               batch_size=256, ubatch_size=256, arena_mib=100, port=1246, uvm=False)
        class FakeProcess:
            def poll(self):
                return None
        with tempfile.TemporaryDirectory() as temp:
            logs = Path(temp)
            def oom_popen(_command, *, stdout, **_kwargs):
                stdout.write("CUDA error: out of memory\n")
                stdout.flush()
                return FakeProcess()
            with patch.object(SWEEP.subprocess, "Popen", side_effect=oom_popen), \
                 patch.object(SWEEP, "wait_ready", side_effect=RuntimeError("server exited")), \
                 patch.object(SWEEP, "stop_process"):
                self.assertFalse(SWEEP.probe_arena(args, 8192, 0, 2500, logs, "http://localhost"))
            with patch.object(SWEEP.subprocess, "Popen", return_value=FakeProcess()), \
                 patch.object(SWEEP, "wait_ready", side_effect=RuntimeError("bad model")), \
                 patch.object(SWEEP, "stop_process"):
                with self.assertRaisesRegex(RuntimeError, "bad model"):
                    SWEEP.probe_arena(args, 8192, 0, 2500, logs, "http://localhost")

    def test_prompt_prefix_probe_uses_full_length_and_stops_after_progress(self):
        class FakeResponse:
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
            def __iter__(self):
                for processed in (0, 256, 2048):
                    yield f'data: {{"prompt_progress": {{"processed": {processed}}}}}\n'.encode()
                raise AssertionError("probe should stop after first 2048 tokens")
        prompt = list(range(8000))
        with patch.object(SWEEP.urllib.request, "urlopen", return_value=FakeResponse()) as open_url:
            SWEEP.probe_prompt_prefix("http://localhost", prompt, 256, 2048)
        payload = json.loads(open_url.call_args.args[0].data)
        self.assertEqual(len(payload["prompt"]), 8000)
        self.assertEqual(payload["n_predict"], 256)
        self.assertTrue(payload["return_progress"])
        self.assertFalse(payload["cache_prompt"])

    def test_prompt_prefix_probe_rejects_early_eof(self):
        class FakeResponse:
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
            def __iter__(self):
                yield b'data: {"prompt_progress": {"processed": 256}}\n'
        with patch.object(SWEEP.urllib.request, "urlopen", return_value=FakeResponse()):
            with self.assertRaisesRegex(RuntimeError, "before prompt probe reached"):
                SWEEP.probe_prompt_prefix("http://localhost", [1] * 8000, 256, 2048)

    def test_arena_probe_uses_real_context_prompt_and_caches_tokenization(self):
        args = SimpleNamespace(server=Path("/build/llama-server"), model=Path("/models/model.gguf"),
                               batch_size=256, ubatch_size=256, arena_mib=100, port=1246, uvm=False,
                               decode_tokens=256, mtp_lengths=(0, 3))
        class FakeProcess:
            def poll(self):
                return None
        cache = {}
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(SWEEP.subprocess, "Popen", return_value=FakeProcess()), \
                 patch.object(SWEEP, "wait_ready"), \
                 patch.object(SWEEP, "stop_process"), \
                 patch.object(SWEEP, "request_json", return_value={"tokens": list(range(9000))}) as tokenize, \
                 patch.object(SWEEP, "stream_completion", return_value={"timings": {"predicted_n": 4}}) as completion, \
                 patch.object(SWEEP, "probe_prompt_prefix") as prefill:
                self.assertTrue(SWEEP.probe_arena(args, 8192, 3, 2400, Path(temp), "http://localhost", "article", cache))
            self.assertEqual(tokenize.call_count, 1)
            self.assertEqual(completion.call_count, 1)
            self.assertEqual(len(completion.call_args.args[1]["prompt"]), 256)
            self.assertEqual(len(prefill.call_args.args[1]), 7932)
            self.assertEqual(prefill.call_args.args[2:], (256, 2048))
            self.assertEqual(len(cache["tokens"]), 9000)

    def test_full_workload_probe_checks_all_requested_tokens(self):
        args = SimpleNamespace(server=Path("/build/llama-server"), model=Path("/models/model.gguf"),
                               batch_size=256, ubatch_size=256, arena_mib=100, port=1246, uvm=False,
                               decode_tokens=256, mtp_lengths=(3,))
        class FakeProcess:
            def poll(self):
                return None
        def completion_result(_url, payload, **_kwargs):
            return {"timings": {"prompt_n": len(payload["prompt"]), "predicted_n": payload["n_predict"]},
                    "tokens": list(range(payload["n_predict"]))}
        def request_result(url, *_args, **_kwargs):
            return {"tokens": list(range(9000))} if url.endswith("/tokenize") else {"content": "text"}
        saved = {}
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(SWEEP.subprocess, "Popen", return_value=FakeProcess()), \
                 patch.object(SWEEP, "wait_ready"), \
                 patch.object(SWEEP, "stop_process"), \
                 patch.object(SWEEP, "request_json", side_effect=request_result), \
                 patch.object(SWEEP, "stream_completion", side_effect=completion_result) as completion, \
                 patch.object(SWEEP, "probe_prompt_prefix") as prefix:
                self.assertTrue(SWEEP.probe_arena(args, 8192, 3, 2400, Path(temp), "http://localhost",
                                                  "article", {"tokens": None}, full_workload=True, result_cache=saved))
            self.assertEqual(completion.call_count, 2)
            self.assertEqual(len(completion.call_args_list[1].args[1]["prompt"]), 7932)
            self.assertEqual(completion.call_args_list[1].args[1]["n_predict"], 256)
            self.assertEqual(saved[2400]["result"]["timings"]["predicted_n"], 256)
            self.assertEqual(saved[2400]["first_text"], "text")
            prefix.assert_not_called()

    def test_auto_max_refines_fast_bound_with_full_workload(self):
        args = SimpleNamespace(arena_mib=100, auto_max_arena=True)
        def fits(_args, _context, _mode, arena_mib, _logs, _url, _article, _cache, full_workload=False, result_cache=None):
            return arena_mib <= (113 if full_workload else 120)
        with patch.object(SWEEP, "probe_arena", side_effect=fits) as probe:
            result = SWEEP.arena_for_point(args, 8192, 3, Path("/tmp/logs"), "http://localhost", "article", {})
        self.assertEqual(result, 113)
        self.assertTrue(any(call.kwargs.get("full_workload") for call in probe.call_args_list))

    def test_arena_probe_exercises_decode_to_prefill_reentry(self):
        args = SimpleNamespace(server=Path("/build/llama-server"), model=Path("/models/model.gguf"),
                               batch_size=256, ubatch_size=256, arena_mib=100, port=1246, uvm=False,
                               decode_tokens=256)
        class FakeProcess:
            def poll(self):
                return None
        def completion_result(_url, payload, **_kwargs):
            return {"timings": {"predicted_n": payload["n_predict"]}}
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(SWEEP.subprocess, "Popen", return_value=FakeProcess()), \
                 patch.object(SWEEP, "wait_ready"), \
                 patch.object(SWEEP, "stop_process"), \
                 patch.object(SWEEP, "stream_completion", side_effect=completion_result) as completion:
                self.assertTrue(SWEEP.probe_arena(args, 8192, 0, 2400, Path(temp), "http://localhost"))
            self.assertEqual(completion.call_count, 2)
            first = completion.call_args_list[0].args[1]
            second = completion.call_args_list[1].args[1]
            self.assertFalse(first["cache_prompt"])
            self.assertFalse(second["cache_prompt"])
            self.assertGreater(len(second["prompt"]), len(first["prompt"]))
            self.assertEqual(first["n_predict"], 4)
            self.assertEqual(second["n_predict"], 256)

    def test_arena_probe_recognizes_only_capacity_failures(self):
        self.assertTrue(SWEEP.is_arena_capacity_failure("CUDA error: out of memory"))
        self.assertTrue(SWEEP.is_arena_capacity_failure("cudaMalloc failed: out of memory"))
        self.assertFalse(SWEEP.is_arena_capacity_failure("Prompt contains invalid tokens"))
        copy_failure = "CUDA error: invalid resource handle\n#4 copy_queue::~copy_queue()\n#11 llama_kv_stream_model::acquire_mtp_layer()"
        self.assertTrue(SWEEP.is_arena_capacity_failure(copy_failure))
        self.assertFalse(SWEEP.is_arena_capacity_failure("CUDA error: invalid resource handle"))

    def test_default_server_is_the_release_build(self):
        with patch("sys.argv", [str(SCRIPT), "--model", "model.gguf"]):
            args = SWEEP.arguments()
        self.assertEqual(args.server, SWEEP.ROOT / "build-device-memory-infra-cuda-release/bin/llama-server")

    def test_uvm_flag_controls_the_server_environment(self):
        inherited = {"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1", "GGML_CUDA_PREFER_MODEL_WEIGHTS": "1", "OTHER_OPTION": "kept"}
        with patch.dict("os.environ", inherited, clear=True):
            disabled = SWEEP.server_environment(False)
            self.assertNotIn("GGML_CUDA_ENABLE_UNIFIED_MEMORY", disabled)
            self.assertNotIn("GGML_CUDA_PREFER_MODEL_WEIGHTS", disabled)
            self.assertEqual(disabled["OTHER_OPTION"], "kept")
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(SWEEP.server_environment(True)["GGML_CUDA_ENABLE_UNIFIED_MEMORY"], "1")

    def test_parse_mtp_lengths(self):
        self.assertEqual(SWEEP.parse_mtp_lengths("0,1,2,4"), (0, 1, 2, 4))
        for value in ("", "1,", "1,1", "-1", "5", "x"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                SWEEP.parse_mtp_lengths(value)

    def test_server_command_enables_only_requested_mtp_length(self):
        args = SimpleNamespace(
            server=Path("/build/llama-server"), model=Path("/models/model.gguf"),
            batch_size=256, ubatch_size=256, arena_mib=2688, port=1246,
        )
        baseline = SWEEP.server_command(args, 8192, 0)
        self.assertNotIn("--spec-type", baseline)
        self.assertNotIn("--kv-stream-auxiliary-layers", baseline)
        mtp = SWEEP.server_command(args, 8192, 3)
        self.assertEqual(mtp[mtp.index("--spec-type") + 1], "draft-mtp")
        self.assertEqual(mtp[mtp.index("--spec-draft-n-max") + 1], "3")
        self.assertEqual(mtp[mtp.index("--kv-stream-auxiliary-layers") + 1], "1")
        probe = SWEEP.server_command(args, 8192, 3, arena_mib=2337)
        self.assertEqual(probe[probe.index("--kv-stream-arena-mib") + 1], "2337")

    def test_checkpoint_opt_out_is_only_forwarded_to_mtp_server(self):
        args = SimpleNamespace(
            server=Path("/build/llama-server"), model=Path("/models/model.gguf"),
            batch_size=256, ubatch_size=256, arena_mib=2240, port=1246,
            no_kv_stream_rs_rollback=True,
        )
        self.assertNotIn("--no-kv-stream-rs-rollback", SWEEP.server_command(args, 8192, 0))
        self.assertIn("--no-kv-stream-rs-rollback", SWEEP.server_command(args, 8192, 3))
        args.no_kv_stream_rs_rollback = False
        self.assertNotIn("--no-kv-stream-rs-rollback", SWEEP.server_command(args, 8192, 3))

    def test_series_and_logs_keep_lengths_separate(self):
        rows = [
            {"context_capacity": context, "mtp_length": length, "decode_tps": 20 + length}
            for context in (8192, 16384) for length in (0, 2)
        ]
        self.assertEqual(
            [row["context_capacity"] for row in SWEEP.series(rows, 2)],
            [8192, 16384],
        )
        self.assertNotEqual(SWEEP.log_name(8192, 2688, 0), SWEEP.log_name(8192, 2688, 2))

    def test_mtp_sweep_reserves_shared_draft_headroom(self):
        self.assertEqual(SWEEP.prompt_tokens_for_context(8192, 256, (0,)), 7936)
        self.assertEqual(SWEEP.prompt_tokens_for_context(8192, 256, (1, 2, 3, 4)), 7931)

    def test_last_request_acceptance_is_selected(self):
        log = (
            "draft acceptance = 0.50000 ( 2 accepted / 4 generated)\n"
            "draft acceptance = 0.75000 ( 6 accepted / 8 generated)\n"
        )
        self.assertEqual(SWEEP.parse_draft_acceptance(log), (6, 8))
        self.assertEqual(SWEEP.parse_draft_acceptance("no speculative data"), (None, None))
        self.assertEqual(
            SWEEP.parse_draft_acceptance(log + "prompt eval time = 100 ms / 256 tokens\n"),
            (None, None),
        )


if __name__ == "__main__":
    unittest.main()
