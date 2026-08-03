#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl


TOOL = Path(__file__).parents[1] / "tools" / "tessera" / "evidence-store.py"


class EvidenceStoreTest(unittest.TestCase):
    def test_acceptance_and_evolution_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "store"
            telemetry = root / "events.jsonl"
            # Mix v1 (legacy adapter) and v3 (default unified schema) records
            # to verify both are accepted by the consumer.
            telemetry.write_text(
                "\n".join([
                    json.dumps({
                        "schema": "llama.dflash.acceptance.v1",
                        "drafted": 3,
                        "accepted": 2,
                        "confidence": [0.9, 0.8, 0.3],
                    }),
                    json.dumps({
                        "schema": "llama.spec_calib.v3",
                        "drafted": 3,
                        "accepted": 2,
                        "confidence": [0.9, 0.8, 0.3],
                    }),
                ]) + "\n",
                encoding="utf-8",
            )
            checkpoint = root / "search.attention.json"
            checkpoint.write_text(json.dumps({
                "schema": "llama.tessera.awq-evolution.v1",
                "generation": 0,
                "history": [{
                    "generation": 0,
                    "candidate": {
                        "alpha": 0.4,
                        "clip": 0.9,
                        "outlier_fraction": 0.005,
                        "moment_mix": 0.2,
                        "tail_guard": 0.3,
                    },
                    "score": {
                        "train_error": 0.1,
                        "heldout_error": 0.2,
                        "tail_error": 0.3,
                        "size_cost": 0.005,
                        "fitness": 0.5,
                    },
                }],
                "archive": {},
            }), encoding="utf-8")
            subprocess.run([
                sys.executable, str(TOOL), "ingest-acceptance",
                "--store", str(store), "--run-id", "pilot",
                "--telemetry", str(telemetry),
            ], check=True, capture_output=True, text=True)
            subprocess.run([
                sys.executable, str(TOOL), "ingest-evolution",
                "--store", str(store), "--run-id", "pilot",
                "--checkpoint-glob", str(checkpoint),
            ], check=True, capture_output=True, text=True)
            summary = root / "summary.parquet"
            subprocess.run([
                sys.executable, str(TOOL), "summarize",
                "--store", str(store), "--run-id", "pilot",
                "--output", str(summary),
            ], check=True, capture_output=True, text=True)
            # Two events written above (one v1, one v3), 3 positions each.
            self.assertEqual(pl.scan_parquet(store / "acceptance" / "*.parquet").select(pl.len()).collect().item(), 2)
            self.assertEqual(pl.scan_parquet(store / "acceptance_position" / "*.parquet").select(pl.len()).collect().item(), 6)
            kinds = set(pl.read_parquet(summary)["kind"].to_list())
            self.assertEqual(kinds, {"acceptance", "evolution"})


if __name__ == "__main__":
    unittest.main()
