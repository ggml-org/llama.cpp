import unittest

from rx7900xtx_bench.workloads import common_suffix


class WorkloadsTest(unittest.TestCase):
    def test_common_suffix_preserves_template_tail(self):
        self.assertEqual(common_suffix([1, 2, 3, 4], [9, 3, 4]), [3, 4])

    def test_common_suffix_rejects_unrelated_sequences(self):
        with self.assertRaises(RuntimeError):
            common_suffix([1, 2], [3, 4])


if __name__ == "__main__":
    unittest.main()
