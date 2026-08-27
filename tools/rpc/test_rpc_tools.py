#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RPC_TOOLS = ROOT / "tools" / "rpc"


def load_tool(name):
    path = RPC_TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchRpcRemoteTests(unittest.TestCase):
    def test_decode_only_comparison_omits_prompt_delta(self):
        bench_rpc_remote = load_tool("bench_rpc_remote")
        summary = {
            "cases": {
                "base": {
                    "metrics": {
                        "decode": {"avg_tokens_per_second": 10.0},
                    },
                    "elapsed_sec": 20.0,
                },
                "patch": {
                    "metrics": {
                        "decode": {"avg_tokens_per_second": 11.0},
                    },
                    "elapsed_sec": 18.0,
                },
            },
        }

        bench_rpc_remote.build_comparison(summary)

        self.assertEqual(summary["measured_metric_kinds"], ["decode"])
        self.assertNotIn("prompt_avg_ts_delta_pct", summary)
        self.assertAlmostEqual(summary["decode_avg_ts_delta_pct"], 10.0)
        self.assertAlmostEqual(summary["elapsed_delta_pct"], -10.0)


class SuggestRpcPlacementCliTests(unittest.TestCase):
    def run_suggest(self, *extra):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace.stderr"
            trace_path.write_text("", encoding="utf-8")
            cmd = [
                sys.executable,
                str(RPC_TOOLS / "suggest_rpc_placement.py"),
                "--trace-stderr",
                str(trace_path),
                "--device",
                "RPC0/RPC1",
                "--tensor-split",
                "1/1",
                *extra,
            ]
            return subprocess.run(cmd, text=True, capture_output=True, check=False)

    def test_sweep_device_weight_preserves_user_order_and_dedupes(self):
        result = self.run_suggest(
            "--sweep-device-weight",
            "RPC0=2/4/4/8",
            "--max-candidates",
            "8",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)

        summary = json.loads(result.stdout)
        candidates = summary["candidates"]
        self.assertEqual(
            [candidate["tensor_split"] for candidate in candidates],
            ["1/1", "2/1", "4/1", "8/1"],
        )
        sweep_candidates = [
            candidate for candidate in candidates
            if candidate.get("generation") == "device-weight-sweep"
        ]
        self.assertTrue(sweep_candidates)
        self.assertTrue(all(
            candidate["trace_cost_source"] == "not-estimated"
            for candidate in sweep_candidates
        ))

    def test_sweep_device_weight_rejects_unknown_device(self):
        result = self.run_suggest("--sweep-device-weight", "RPC2=2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown device", result.stderr)

    def test_sweep_device_weight_rejects_zero_weight(self):
        result = self.run_suggest("--sweep-device-weight", "RPC0=0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("positive weights", result.stderr)


if __name__ == "__main__":
    unittest.main()
