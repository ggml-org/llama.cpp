#!/usr/bin/env python3
"""Host-only tests for vision memory/latency report accounting."""

import re
import unittest
from measure_adaptive_vision import group_phases, parse_phases, parse_reloads, sample_window


class Measurements(unittest.TestCase):
    def test_lazy_reload_cost_is_separate_from_phase_restoration(self):
        text = "I KV_reload: bytes=4096 calls=2 padded_rows=256 drain_us=3 upload_us=4 begin_us=120 end_us=130\n"
        self.assertEqual(parse_reloads(text)[0]["upload_us"], 4)
        with self.assertRaises(ValueError):
            parse_reloads("KV_reload: bytes=10\n")

    def test_phase_records_keep_distinct_transfer_and_restore_costs(self):
        text = "I vision_phase: parent=1024 weights=400 compute=100 host_compute=20 grants=512 before_kv=800 resumed_kv=750 borrowed_after=0 suspended_after=0 diagnostics=1 measure_us=2 suspend_us=3 loan_us=4 projector_reload_us=5 encode_us=6 release_us=7 text_resume_us=8 begin_us=100 end_us=135\n"
        records = parse_phases(text)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["projector_reload_us"], 5)
        self.assertEqual(records[0]["text_resume_us"], 8)
        self.assertEqual(records[0]["grants"], 512)
        self.assertEqual(parse_phases("unrelated startup log\n"), [])
        for key, value in (("grants", "1025"), ("borrowed_after", "1"), ("suspended_after", "1"),
                           ("diagnostics", "0"), ("encode_us", "-1"), ("end_us", "99")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse_phases(re.sub(rf"\b{key}=\d+", f"{key}={value}", text))

    def test_incomplete_or_failed_accounting_is_not_reported_as_success(self):
        with self.assertRaises(ValueError):
            parse_phases("vision_phase: parent=1024\n")
        good = "vision_phase: " + " ".join(f"{key}=1" for key in (
            "parent weights compute host_compute grants before_kv resumed_kv borrowed_after suspended_after diagnostics measure_us suspend_us loan_us projector_reload_us encode_us release_us text_resume_us begin_us end_us".split()))
        with self.assertRaises(ValueError):
            parse_phases(good)

    def test_sample_window_does_not_invent_a_peak_without_samples(self):
        samples = [(10, 100), (20, 110), (30, 120)]
        self.assertEqual(sample_window(samples, 15, 25), {"samples": 1, "min_mib": 110, "peak_mib": 110})
        self.assertEqual(sample_window(samples, 31, 40), {"samples": 0, "min_mib": None, "peak_mib": None})

    def test_multiple_encoded_batches_belong_to_one_request(self):
        phases = [{"begin_us": 12, "end_us": 15}, {"begin_us": 20, "end_us": 25}, {"begin_us": 35, "end_us": 38}]
        rows = [{"begin_us": 10, "end_us": 30}, {"begin_us": 32, "end_us": 40}]
        self.assertEqual(group_phases(phases, rows), [phases[:2], phases[2:]])
        with self.assertRaises(ValueError):
            group_phases(phases, rows[:1])
        with self.assertRaises(ValueError):
            group_phases([], rows)
        with self.assertRaises(ValueError):
            group_phases([phases[0], phases[2]], [rows[0], {"begin_us": 11, "end_us": 26}])


if __name__ == "__main__":
    unittest.main()
