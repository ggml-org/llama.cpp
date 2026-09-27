import argparse
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[2] / "run-fixed-span-sweep.py"
SPEC = importlib.util.spec_from_file_location("fixed_span_sweep", SCRIPT)
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


class MtpSweepTests(unittest.TestCase):
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
