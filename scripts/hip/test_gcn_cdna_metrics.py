#!/usr/bin/env python3

import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import yaml

SCRIPT = Path(__file__).with_name('gcn-cdna-metrics.py')
spec = importlib.util.spec_from_file_location('gcn_cdna_metrics', SCRIPT)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def write_records(directory, filename, kernels):
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for name, values in kernels.items():
            for key, value in values.items():
                f.write('--- !Analysis\n')
                yaml.safe_dump({'Pass': 'kernel-resource-usage', 'Name': key, 'Function': name,
                                'DebugLoc': {'File': 'wkv.cu', 'Line': 69, 'Column': 1},
                                'Args': [{'String': key + ': '}, {key: str(value)}]}, f, sort_keys=False)
                f.write('...\n')
    return path


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / 'baseline'
        self.pr = self.root / 'pr'
        self.base.mkdir()
        self.pr.mkdir()

    def compare(self, **kwargs):
        with redirect_stdout(io.StringIO()) as out:
            result = metrics.check_comparison(metrics.parse_remarks(self.base), metrics.parse_remarks(self.pr),
                                              'baseline', 'PR', **kwargs)
        return result, out.getvalue()

    def test_same_source_location_does_not_swap_kernel_specializations(self):
        write_records(self.pr, 'kernels.opt.yaml', {'_Zkernel64': {'NumVGPR': 64, 'VGPRSpill': 360},
                                                    '_Zkernel128': {'NumVGPR': 64, 'VGPRSpill': 0}})
        data = metrics.parse_remarks(self.pr)
        self.assertEqual(data['_Zkernel64']['vgpr_spill'], [360])
        self.assertEqual(data['_Zkernel128']['vgpr_spill'], [0])
        self.assertEqual(data['_Zkernel64']['location'], 'wkv.cu:69')

    def test_vgpr_limit_and_allowlist(self):
        write_records(self.pr, 'kernels.opt.yaml', {'bad': {'NumVGPR': 255, 'VGPRSpill': 2},
                                                    'allowed': {'NumVGPR': 256, 'VGPRSpill': 4},
                                                    'good': {'NumVGPR': 256, 'VGPRSpill': 0}})
        metrics.IGNORED_VGPR.add('allowed')
        self.addCleanup(metrics.IGNORED_VGPR.discard, 'allowed')
        with redirect_stdout(io.StringIO()) as out:
            code = metrics.check_vgpr(metrics.parse_remarks(self.pr), 'PR')
        self.assertEqual(code, 1)
        self.assertIn('allowed - Total VGPRs: 260 (256 + 4) [IGNORED]', out.getvalue())
        self.assertIn('bad - Total VGPRs: 257 (255 + 2)', out.getvalue())
        self.assertNotIn('good - Total VGPRs', out.getvalue())

    def test_occupancy_spill_regressions_and_tolerance(self):
        write_records(self.base, 'a.opt.yaml', {'same': {'Occupancy': 5, 'VGPRSpill': 0, 'SGPRSpill': 1}})
        write_records(self.pr, 'a.opt.yaml', {'same': {'Occupancy': 4, 'VGPRSpill': 1, 'SGPRSpill': 0}})
        code, output = self.compare()
        self.assertEqual(code, 1)
        self.assertIn('Occupancy [waves/SIMD]: 5 -> 4 (REGRESSION)', output)
        self.assertIn('VGPRs Spill: 0 -> 1 (REGRESSION)', output)
        self.assertIn('SGPRs Spill: 1 -> 0 (improved)', output)
        code, output = self.compare(tolerance=1)
        self.assertEqual(code, 1)
        self.assertNotIn('Occupancy [waves/SIMD]: 5 -> 4 (REGRESSION)', output)

    def test_duplicate_kernel_uses_worst_per_metric(self):
        write_records(self.base, 'a.opt.yaml', {'same': {'Occupancy': 6, 'VGPRSpill': 0}})
        write_records(self.base, 'sub/b.opt.yaml', {'same': {'Occupancy': 5, 'VGPRSpill': 2}})
        write_records(self.pr, 'a.opt.yaml', {'same': {'Occupancy': 6, 'VGPRSpill': 1}})
        code, output = self.compare()
        self.assertEqual(code, 0)
        self.assertIn('VGPRs Spill: 2 -> 1 (improved)', output)
        self.assertIn('Occupancy [waves/SIMD]: 5 -> 6 (improved)', output)

    def test_added_removed_deltas_and_markdown(self):
        write_records(self.base, 'a.opt.yaml', {'removed': {'Occupancy': 5, 'VGPRSpill': 2}, 'same': {'Occupancy': 4}})
        write_records(self.pr, 'a.opt.yaml', {'added': {'Occupancy': 6, 'SGPRSpill': 1}, 'same': {'Occupancy': 5}})
        tsv = self.root / 'diff.tsv'
        report = self.root / 'report.md'
        code, _ = self.compare(dump=tsv)
        self.assertEqual(code, 0)
        self.assertIn('occupancy\tadded\twkv.cu:69\t\t6\tadded', tsv.read_text())
        meta, rows = metrics.read_deltas(tsv)
        metrics.write_report(report, meta, rows)
        result = report.read_text()
        self.assertIn('# HIP kernel metrics comparison', result)
        self.assertIn('## Occupancy [waves/SIMD] improvements (1)', result)
        self.assertIn('## Added kernels (1)', result)
        self.assertIn('## Removed kernels (1)', result)
        self.assertIn('| wkv.cu:69 | ' + chr(96) + 'same' + chr(96) + ' | 4 | 5 |', result)

    def test_ignore_only_comparison_not_vgpr(self):
        write_records(self.base, 'a.opt.yaml', {'same': {'Occupancy': 5}})
        write_records(self.pr, 'a.opt.yaml', {'same': {'Occupancy': 1}})
        code, output = self.compare(ignored={'same'})
        self.assertEqual(code, 0)
        self.assertIn('Kernels compared: 0', output)

    def test_vgpr_only_records_are_not_compared(self):
        write_records(self.base, 'a.opt.yaml', {'vgpr_only': {'NumVGPR': 64}, 'same': {'Occupancy': 4}})
        write_records(self.pr, 'a.opt.yaml', {'vgpr_only': {'NumVGPR': 128}, 'same': {'Occupancy': 4}})
        code, output = self.compare()
        self.assertEqual(code, 0)
        self.assertIn('Kernels compared: 1 (baseline 1, PR 1)', output)

    def test_missing_or_invalid_records_fail(self):
        with self.assertRaisesRegex(ValueError, 'No optimization records'):
            metrics.parse_remarks(self.pr)
        write_records(self.pr, 'a.opt.yaml', {'same': {'Occupancy': 'invalid'}})
        with self.assertRaisesRegex(ValueError, 'Invalid Occupancy value'):
            metrics.parse_remarks(self.pr)


if __name__ == '__main__':
    unittest.main()
