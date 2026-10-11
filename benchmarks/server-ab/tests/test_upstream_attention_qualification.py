import importlib.util
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[2] / "run-upstream-attention-qualification.py"
SPEC = importlib.util.spec_from_file_location("qualification", SCRIPT)
QUALIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFY)


class Commands(unittest.TestCase):
    def test_stock_keeps_matching_draft_quant_without_arena(self):
        args = SimpleNamespace(server=Path("server"), model=Path("model"), batch_size=256,
                               ubatch_size=256, arena_mib=2240, port=12351)
        command = QUALIFY.command(args, 8192, 3, 2240, stock=True)
        self.assertNotIn("--kv-stream-arena-mib", command)
        self.assertEqual(command[command.index("--spec-draft-type-k")+1], "q8_0")
        self.assertEqual(command[command.index("--spec-draft-type-v")+1], "q4_0")
        self.assertIn("--spec-draft-n-max", command)

    def test_adaptive_retains_arena_and_no_mtp_control(self):
        args = SimpleNamespace(server=Path("server"), model=Path("model"), batch_size=256,
                               ubatch_size=256, arena_mib=2240, port=12351)
        command = QUALIFY.command(args, 131072, 0, 2100)
        self.assertEqual(command[command.index("--kv-stream-arena-mib")+1], "2100")
        self.assertNotIn("--spec-type", command)


class Collection(unittest.TestCase):
    def run_fixture(self, measured, text, stock=False, reuse=None):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), port=12351, server=Path("server"),
                                   model=Path("model"), batch_size=256, ubatch_size=256, arena_mib=2240,
                                   variant="fixture", stock=stock, decode_tokens=256)
            args.verify_reuse = reuse is not None
            def launch(*unused, **kwargs):
                kwargs["stdout"].write(text)
                kwargs["stdout"].flush()
                return object()
            with patch.object(QUALIFY.subprocess, "Popen", side_effect=launch), \
                    patch.object(QUALIFY, "wait_for_health"), \
                    patch.object(QUALIFY, "stream_completion", side_effect=[{"tokens": [1]*4}, measured]+(reuse or [])), \
                    patch.object(QUALIFY.SWEEP, "stop_process") as stop:
                result = QUALIFY.run_point(args,8192,0,2240,{"tokens": [1]*8192})
                stop.assert_called_once()
                return result

    def measured(self):
        return {"tokens": list(range(256)), "content": "text", "timings": {"prompt_n":7932,
                "predicted_n":256, "prompt_per_second":1000, "predicted_per_second":40,
                "prompt_ms":7932, "predicted_ms":6400}}

    def test_complete_response_keeps_token_ids_and_exact_layout(self):
        text = "memory_phase: phase=decode kv_pool=104857600 resident_pages=20 ring_slots=8 active_pages=31\n"
        text += "attention: accepted KV layout copied 12.5 MiB in 4 H2D calls\n"
        result = self.run_fixture(self.measured(),text)
        self.assertEqual(result["status"],"ok")
        self.assertEqual(len(result["token_ids"]),256)
        self.assertEqual(result["decode_kv_pool_mib"],100)
        self.assertEqual(result["h2d_mib_per_eval"],12.5)
        self.assertLessEqual(result["started_unix"], result["finished_unix"])
        self.assertIn("--kv-stream-arena-mib", result["command"])
        self.assertFalse(result["uvm"])

    def test_missing_layout_does_not_discard_finished_tokens(self):
        result = self.run_fixture(self.measured(),"")
        self.assertEqual(result["status"],"failed")
        self.assertEqual(len(result["token_ids"]),256)

    def test_stock_does_not_require_adaptive_telemetry(self):
        self.assertEqual(self.run_fixture(self.measured(),"",stock=True)["status"],"ok")

    def test_incomplete_response_is_not_a_success_or_silent_retry(self):
        result = self.measured()
        result["tokens"].pop()
        self.assertEqual(self.run_fixture(result,"")["status"],"failed")

    def test_capacity_error_is_distinct_from_execution_error(self):
        result = self.measured()
        result["tokens"] = []
        self.assertEqual(self.run_fixture(result,"cudaMalloc failed: out of memory")["status"],"capacity_failure")

    def test_serial_reuse_and_fresh_replay_are_compared(self):
        continuation = {"tokens": list(range(32)), "timings": {"predicted_n":32, "cache_n":7936}}
        result = self.run_fixture(self.measured(), "", stock=True, reuse=[continuation,continuation])
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["reuse_matches_fresh"])
        self.assertEqual(result["reuse_cache_tokens"], 7936)

    def test_serial_reuse_difference_is_retained_for_matching_stock_comparison(self):
        cached = {"tokens": list(range(32)), "timings": {"predicted_n":32, "cache_n":7936}}
        fresh = {"tokens": [2]*32, "timings": {"predicted_n":32}}
        result = self.run_fixture(self.measured(), "", stock=True, reuse=[cached,fresh])
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["reuse_matches_fresh"])
        self.assertEqual(len(result["token_ids"]), 256)

    def test_serial_reuse_incomplete_response_still_fails(self):
        cached = {"tokens": list(range(31)), "timings": {"predicted_n":31, "cache_n":7936}}
        fresh = {"tokens": list(range(32)), "timings": {"predicted_n":32}}
        self.assertEqual(self.run_fixture(self.measured(), "", stock=True, reuse=[cached,fresh])["status"], "failed")

    def test_measured_acceptance_is_not_overwritten_by_later_requests(self):
        measured = self.measured()
        measured["timings"].update(draft_n=200, draft_n_accepted=150)
        result = self.run_fixture(measured, "draft acceptance = 0.5 ( 5 accepted / 10 generated)", stock=True)
        self.assertEqual(result["draft_accepted"], 150)
        self.assertEqual(result["draft_generated"], 200)


class MaximumArena(unittest.TestCase):
    def test_maximum_requires_full_workload_and_reuses_accepted_result(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), arena_mib=1000, port=12351,
                                   prefill_text=Path(directory)/"corpus")
            args.prefill_text.write_text("corpus")
            seen = []
            def measured(unused, context, mtp, arena, cache):
                seen.append(arena)
                return {"status":"ok" if arena <= 999 else "capacity_failure", "arena_mib":arena}
            with patch.object(QUALIFY.SWEEP, "probe_arena", side_effect=lambda *a, **k:a[3] <= 1000), \
                    patch.object(QUALIFY, "run_point", side_effect=measured):
                result = QUALIFY.run_max_point(args, 8192, 3, {})
            self.assertEqual(result["arena_mib"], 999)
            self.assertEqual(len(seen), len(set(seen)))
            self.assertTrue((args.output/"arena-probes.jsonl").exists())

    def test_execution_failure_is_not_treated_as_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), arena_mib=1000, port=12351,
                                   prefill_text=Path(directory)/"corpus")
            args.prefill_text.write_text("corpus")
            log = Path(directory)/"failed.log"
            log.write_text("kernel assertion")
            with patch.object(QUALIFY.SWEEP, "probe_arena", side_effect=lambda *a, **k:a[3] <= 1000), \
                    patch.object(QUALIFY, "run_point", return_value={"status":"failed", "log":str(log)}):
                with self.assertRaisesRegex(RuntimeError, "execution failure"):
                    QUALIFY.run_max_point(args, 8192, 3, {})


class MatchingControl(unittest.TestCase):
    def row(self):
        return {"context":8192, "mtp":3, "prompt_tokens":7932, "decode_tokens":256,
                "token_ids":list(range(256)), "reuse_token_ids":list(range(32)),
                "fresh_token_ids":[2]*32, "reuse_cache_tokens":7936}

    def test_matching_stock_modes_pass_even_when_cached_and_fresh_differ(self):
        self.assertTrue(all(QUALIFY.compare_matching_rows(self.row(),self.row()).values()))

    def test_cached_drift_against_stock_is_detected(self):
        actual, stock = self.row(), self.row()
        actual["reuse_token_ids"][10] = 77
        result = QUALIFY.compare_matching_rows(actual, stock)
        self.assertFalse(result["reuse_token_ids"])
        self.assertTrue(result["fresh_token_ids"])

    def test_different_cache_boundary_or_query_profile_is_not_comparable(self):
        for field in ("mtp", "context", "prompt_tokens", "decode_tokens", "reuse_cache_tokens"):
            actual, stock = self.row(), self.row()
            actual[field] += 1
            with self.assertRaises(ValueError):
                QUALIFY.compare_matching_rows(actual,stock)

    def test_missing_or_incomplete_outputs_cannot_pass(self):
        for field in ("token_ids", "reuse_token_ids", "fresh_token_ids"):
            actual, stock = self.row(), self.row()
            actual[field].pop()
            with self.assertRaises(ValueError):
                QUALIFY.compare_matching_rows(actual,stock)

    def test_missing_profile_fields_are_not_assumed_equal(self):
        for field in ("context", "mtp", "prompt_tokens"):
            actual, stock = self.row(), self.row()
            del actual[field]
            del stock[field]
            with self.assertRaises(ValueError):
                QUALIFY.compare_matching_rows(actual,stock)


if __name__ == "__main__":
    unittest.main()
