#!/usr/bin/env python3
# Copyright 2026 Georgi Gerganov & llama.cpp Authors.
# SPDX-License-Identifier: MIT

import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../scripts/production_debt.py",
)
spec = importlib.util.spec_from_file_location("llamacpp_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["llamacpp_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtInferenceGate = production_debt_mod.ProductionDebtInferenceGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtInferenceGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtInferenceGate(
            never_equate_intent_to_approval=True,
            max_acceptable_ldi=12.0,
        )

    def test_clean_inference_passes_readiness(self) -> None:
        report = self.gate.evaluate_inference_run(
            model_id="llama-3-70b-instruct-q4_k_m.gguf",
            allocated_kv_cache_bytes=2147483648,
            peak_dequant_buffer_bytes=2250000000,
            prompt_eval_latency_ms=32.5,
            context_shift_thrashes=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.ldi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_inference_fails_debt(self) -> None:
        report = self.gate.evaluate_inference_run(
            model_id="uncalibrated_gguf_model",
            allocated_kv_cache_bytes=2147483648,
            peak_dequant_buffer_bytes=6000000000,  # 2.79x dequant sprawl
            prompt_eval_latency_ms=180.0,  # High prompt eval latency
            context_shift_thrashes=3,  # 3 context shift thrashes
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.ldi_score, 50.0)
        self.assertIn("HIGH_DEQUANT_MEMORY_SPRAWL_2.79X", report.critical_smells)
        self.assertIn("HIGH_PROMPT_EVAL_LATENCY_180.0MS", report.critical_smells)
        self.assertIn("DETECTED_3_KV_CACHE_CONTEXT_THRASHES", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_GGUF_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_inference_run("run-1")
        self.gate.evaluate_inference_run("run-2")
        self.gate.evaluate_inference_run("run-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
