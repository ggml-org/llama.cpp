import unittest

from rx7900xtx_bench.metrics import counter_deltas, summarize_samples, summarize_values


class MetricsTest(unittest.TestCase):
    def test_counter_deltas(self):
        before = {
            "llamacpp:spec_decode_num_draft_tokens_total": 10.0,
            "llamacpp:spec_decode_num_accepted_tokens_total": 8.0,
            "llamacpp:spec_decode_num_drafts_total": 4.0,
        }
        after = {
            "llamacpp:spec_decode_num_draft_tokens_total": 18.0,
            "llamacpp:spec_decode_num_accepted_tokens_total": 14.0,
            "llamacpp:spec_decode_num_drafts_total": 7.0,
        }
        self.assertEqual(counter_deltas(before, after)["llamacpp:spec_decode_num_accepted_tokens_total"], 6.0)

    def test_value_summary(self):
        summary = summarize_values([10.0, 20.0, 30.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["mean"], 20.0)
        self.assertEqual(summary["median"], 20.0)

    def test_samples_include_reported_throughputs(self):
        summary = summarize_samples([{
            "end_to_end_output_tps": 10.0,
            "client_delivery_tps": 8.0,
            "ttft_ms": 100.0,
        }])
        self.assertEqual(summary["end_to_end_output_tps"]["mean"], 10.0)
        self.assertEqual(summary["client_delivery_tps"]["mean"], 8.0)


if __name__ == "__main__":
    unittest.main()
