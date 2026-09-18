"""CPU-only tests for the runner and failure handling; not GPU execution."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import run_moe_suite as suite


class MoeSuiteTests(unittest.TestCase):
    def test_suite_sizes_and_unique_names(self):
        for name, count in [('smoke', 2), ('quick', 3), ('full', 7)]:
            rows = suite.cases_for(name)
            self.assertEqual(len(rows), count)
            self.assertEqual(len({c['name'] for c in rows}), count)

    def test_legal_full_shapes(self):
        for c in suite.cases_for('full'):
            for k, split in [(c['k'], c['gate_split']), (c['hidden'], c['down_split'])]:
                self.assertEqual(k % 256, 0)
                self.assertEqual((k // 256) % split, 0)
            self.assertTrue(1 <= c['tokens'] <= 8)
            self.assertGreaterEqual(c['experts'], 8)

    def test_real_capsules_are_all_three(self):
        c = next(c for c in suite.cases_for('quick') if c.get('real'))
        caps = {k: Path(k + '.mw2') for k in ('gate', 'up', 'down')}
        a = suite.case_args(c, caps, 28, 0)
        for k in caps:
            self.assertEqual(a[a.index('--' + k) + 1], str(caps[k]))
        self.assertEqual(a[a.index('--host-waits') + 1], 'both')

    def test_smoke_does_not_use_real_weights(self):
        c = suite.cases_for('smoke')[0]
        caps = {k: Path(k + '.mw2') for k in ('gate', 'up', 'down')}
        a = suite.case_args(c, caps, 28, 0)
        self.assertNotIn('--gate', a)
        self.assertEqual(a[a.index('--repeats') + 1], '5')

    def test_out_of_order_case_and_mixed_precision_group(self):
        rows = suite.cases_for('full')
        c = next(c for c in rows if c['order'] == 0)
        a = suite.case_args(c, {}, 28, 0)
        self.assertEqual(a[a.index('--in-order') + 1], '0')
        c = next(c for c in rows if c.get('down_group'))
        a = suite.case_args(c, {}, 28, 0)
        self.assertEqual(a[a.index('--groups') + 1], '32')
        self.assertEqual(a[a.index('--down-group') + 1], '128')

    def test_dry_run_does_not_require_a_gpu_executable(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()) as text:
            rc = suite.main(['--exe', str(Path(d) / 'missing.exe'), '--out', d, '--suite', 'full', '--dry-run'])
            self.assertEqual(rc, 0)
            self.assertEqual(len(json.loads(text.getvalue())), 7)
            self.assertFalse((Path(d) / 'suite_status.json').exists())

    def test_missing_capsule_fails_and_preserves_status(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()):
            root = Path(d); exe = root / 'fake.exe'; exe.write_bytes(b'not executed')
            out = root / 'out'
            rc = suite.main(['--exe', str(exe), '--out', str(out), '--capsule-dir', str(root / 'missing')])
            status = json.loads((out / 'suite_status.json').read_text())
            self.assertEqual(rc, 1)
            self.assertFalse(status['passed'])
            self.assertIn('missing capsule', status['error'])

    def test_mocked_runner_aggregates_and_rejects_numerical_failure(self):
        # Only the orchestration is exercised. No executable/GPU is invoked.
        for passed in [True, False]:
            with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                root = Path(d); exe = root / 'fake.exe'; exe.write_bytes(b'not executed')
                out = root / 'out'
                def fake(cmd, log):
                    target = Path(cmd[cmd.index('--out') + 1]); target.mkdir(parents=True)
                    (target / 'summary.csv').write_text('variant,kernel_pass,wall_median_us\nmock,' + ('1' if passed else '0') + ',10\n')
                with patch.object(suite, 'run_logged', side_effect=fake):
                    rc = suite.main(['--exe', str(exe), '--out', str(out), '--suite', 'smoke'])
                self.assertEqual(rc, 0 if passed else 1)
                status = json.loads((out / 'suite_status.json').read_text())
                self.assertEqual(status['passed'], passed)
                if passed:
                    with (out / 'all_moe_summary.csv').open() as inp:
                        self.assertEqual(len(list(csv.DictReader(inp))), 2)


if __name__ == '__main__':
    unittest.main()
