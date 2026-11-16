#!/usr/bin/env python3

import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml

SCRIPT = Path(__file__).with_name('gcn-cdna-metrics.py')
spec = importlib.util.spec_from_file_location('gcn_cdna_metrics', SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f'Cannot load {SCRIPT}')
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
        self.assertIn('Occupancy [waves/SIMD]: 5 -> 4 (WARNING)', output)
        self.assertIn('VGPRs Spill: 0 -> 1 (REGRESSION)', output)
        self.assertIn('SGPRs Spill: 1 -> 0 (improved)', output)
        code, output = self.compare(tolerance=1)
        self.assertEqual(code, 1)
        self.assertNotIn('Occupancy [waves/SIMD]: 5 -> 4 (WARNING)', output)

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

    def test_occupancy_drop_with_more_vgprs_blocks(self):
        write_records(self.base, 'a.opt.yaml', {'same': {'Occupancy': 5, 'NumVGPR': 64}})
        write_records(self.pr, 'a.opt.yaml', {'same': {'Occupancy': 4, 'NumVGPR': 96}})
        tsv = self.root / 'diff.tsv'
        code, output = self.compare(dump=tsv)
        self.assertEqual(code, 1)
        self.assertIn('Occupancy [waves/SIMD]: 5 -> 4, VGPRs: 64 -> 96 (REGRESSION)', output)
        self.assertIn('1 blocking regression(s), 0 warning(s)', output)
        self.assertIn('occupancy\tsame\twkv.cu:69\t5\t4\tregression', tsv.read_text())

    def test_occupancy_drop_with_fewer_vgprs_warns(self):
        write_records(self.base, 'a.opt.yaml', {'same': {'Occupancy': 5, 'NumVGPR': 96}})
        write_records(self.pr, 'a.opt.yaml', {'same': {'Occupancy': 4, 'NumVGPR': 64}})
        tsv = self.root / 'diff.tsv'
        report = self.root / 'report.md'
        code, output = self.compare(dump=tsv)
        self.assertEqual(code, 0)
        self.assertIn('Occupancy [waves/SIMD]: 5 -> 4 (WARNING)', output)
        self.assertIn('0 blocking regression(s), 1 warning(s)', output)
        self.assertIn('occupancy\tsame\twkv.cu:69\t5\t4\twarning', tsv.read_text())
        meta, rows = metrics.read_deltas(tsv)
        metrics.write_report(report, meta, rows)
        self.assertIn('## Occupancy [waves/SIMD] warnings (1)', report.read_text())
        self.assertIn('| wkv.cu:69 | `same` | 5 | 4 |', report.read_text())

    def test_occupancy_drop_without_vgpr_increase_warns(self):
        write_records(self.base, 'a.opt.yaml', {'unchanged': {'Occupancy': 5, 'NumVGPR': 64},
                                                'missing': {'Occupancy': 5}})
        write_records(self.pr, 'a.opt.yaml', {'unchanged': {'Occupancy': 4, 'NumVGPR': 64},
                                              'missing': {'Occupancy': 4}})
        code, output = self.compare()
        self.assertEqual(code, 0)
        self.assertEqual(output.count('(WARNING)'), 2)
        self.assertIn('0 blocking regression(s), 2 warning(s)', output)

    def test_vgpr_and_sgpr_spill_increases_block_independently(self):
        write_records(self.base, 'a.opt.yaml', {'vgpr': {'VGPRSpill': 0}, 'sgpr': {'SGPRSpill': 0}})
        write_records(self.pr, 'a.opt.yaml', {'vgpr': {'VGPRSpill': 1}, 'sgpr': {'SGPRSpill': 1}})
        code, output = self.compare()
        self.assertEqual(code, 1)
        self.assertIn('vgpr - VGPRs Spill: 0 -> 1 (REGRESSION)', output)
        self.assertIn('sgpr - SGPRs Spill: 0 -> 1 (REGRESSION)', output)
        self.assertIn('2 blocking regression(s), 0 warning(s)', output)

    def test_vgpr_limit_blocks_comparison_and_appears_in_report(self):
        write_records(self.base, 'a.opt.yaml', {'same': {'NumVGPR': 256, 'VGPRSpill': 0}})
        write_records(self.pr, 'a.opt.yaml', {'same': {'NumVGPR': 256, 'VGPRSpill': 1}})
        tsv = self.root / 'diff.tsv'
        report = self.root / 'report.md'
        code, output = self.compare(dump=tsv)
        self.assertEqual(code, 1)
        self.assertIn('Total VGPRs: 257 (256 + 1)', output)
        self.assertIn('2 blocking regression(s), 0 warning(s)', output)
        self.assertIn('vgpr_total\tsame\twkv.cu:69\t256\t257\tregression', tsv.read_text())
        meta, rows = metrics.read_deltas(tsv)
        metrics.write_report(report, meta, rows)
        self.assertIn('## Total VGPRs (limit 256) regressions (1)', report.read_text())
        self.assertIn('| wkv.cu:69 | `same` | 256 | 257 |', report.read_text())

    def test_existing_vgpr_allowlist_preserves_comparison_checks(self):
        write_records(self.base, 'a.opt.yaml', {'allowed': {'NumVGPR': 255, 'VGPRSpill': 3}})
        write_records(self.pr, 'a.opt.yaml', {'allowed': {'NumVGPR': 255, 'VGPRSpill': 4}})
        metrics.IGNORED_VGPR.add('allowed')
        self.addCleanup(metrics.IGNORED_VGPR.discard, 'allowed')
        code, output = self.compare()
        self.assertEqual(code, 1)
        self.assertIn('Total VGPRs: 259 (255 + 4) [IGNORED]', output)
        self.assertIn('VGPRs Spill: 3 -> 4 (REGRESSION)', output)
        self.assertIn('1 blocking regression(s), 0 warning(s)', output)

    def test_comparison_ignore_does_not_disable_vgpr_limit(self):
        write_records(self.base, 'a.opt.yaml', {'ignored': {'NumVGPR': 256, 'VGPRSpill': 0}})
        write_records(self.pr, 'a.opt.yaml', {'ignored': {'NumVGPR': 256, 'VGPRSpill': 1}})
        code, output = self.compare(ignored={'ignored'})
        self.assertEqual(code, 1)
        self.assertIn('Kernels compared: 0 (baseline 0, PR 0)', output)
        self.assertIn('Total VGPRs: 257 (256 + 1)', output)

    def test_occupancy_tolerance_applies_to_blocking_rule(self):
        write_records(self.base, 'a.opt.yaml', {'same': {'Occupancy': 5, 'NumVGPR': 64}})
        write_records(self.pr, 'a.opt.yaml', {'same': {'Occupancy': 4, 'NumVGPR': 96}})
        code, output = self.compare(tolerance=1)
        self.assertEqual(code, 0)
        self.assertNotIn('Occupancy [waves/SIMD]: 5 -> 4', output)
        self.assertIn('No regressions found', output)

    def test_all_rules_complete_and_return_error_after_dump(self):
        write_records(self.base, 'a.opt.yaml', {'occb': {'Occupancy': 5, 'NumVGPR': 64},
                                                'occw': {'Occupancy': 5, 'NumVGPR': 80},
                                                'spill': {'SGPRSpill': 0}})
        write_records(self.pr, 'a.opt.yaml', {'occb': {'Occupancy': 4, 'NumVGPR': 96},
                                              'occw': {'Occupancy': 4, 'NumVGPR': 64},
                                              'spill': {'SGPRSpill': 1}})
        tsv = self.root / 'diff.tsv'
        code, output = self.compare(dump=tsv)
        self.assertEqual(code, 1)
        self.assertIn('occb - Occupancy [waves/SIMD]: 5 -> 4, VGPRs: 64 -> 96 (REGRESSION)', output)
        self.assertIn('occw - Occupancy [waves/SIMD]: 5 -> 4 (WARNING)', output)
        self.assertIn('spill - SGPRs Spill: 0 -> 1 (REGRESSION)', output)
        self.assertIn('2 blocking regression(s), 1 warning(s)', output)
        self.assertTrue(tsv.is_file())

    def test_cli_returns_success_for_warnings_and_generates_report_after_failure(self):
        write_records(self.base, 'a.opt.yaml', {'same': {'Occupancy': 5, 'NumVGPR': 96}})
        write_records(self.pr, 'a.opt.yaml', {'same': {'Occupancy': 4, 'NumVGPR': 64}})
        tsv = self.root / 'diff.tsv'
        report = self.root / 'report.md'
        argv = ['gcn-cdna-metrics.py', 'compare', str(self.base), str(self.pr), '--dump', str(tsv)]
        with patch('sys.argv', argv), redirect_stdout(io.StringIO()):
            self.assertEqual(metrics.main(), 0)
        self.assertIn('warning', tsv.read_text())

        write_records(self.pr, 'a.opt.yaml', {'same': {'Occupancy': 4, 'NumVGPR': 128}})
        with patch('sys.argv', argv), redirect_stdout(io.StringIO()):
            self.assertEqual(metrics.main(), 1)
        self.assertIn('regression', tsv.read_text())
        with patch('sys.argv', ['gcn-cdna-metrics.py', 'report', str(tsv), '-o', str(report)]), redirect_stdout(io.StringIO()):
            self.assertEqual(metrics.main(), 0)
        self.assertIn('## Occupancy [waves/SIMD] regressions (1)', report.read_text())

    def test_missing_or_invalid_records_fail(self):
        with self.assertRaisesRegex(ValueError, 'No optimization records'):
            metrics.parse_remarks(self.pr)
        write_records(self.pr, 'a.opt.yaml', {'same': {'Occupancy': 'invalid'}})
        with self.assertRaisesRegex(ValueError, 'Invalid Occupancy value'):
            metrics.parse_remarks(self.pr)


if __name__ == '__main__':
    unittest.main()
