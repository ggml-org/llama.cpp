import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('suite',ROOT/'tools/run_suite.py')
suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)

class SuitePlanTests(unittest.TestCase):
    def test_quick_has_both_maple_shapes(self):
        c=suite.cases_for('quick')
        kinds={x.get('model_kind') for x in c}
        self.assertTrue({'gate-up','down'}<=kinds)
    def test_all_shapes_are_legal(self):
        cases=suite.cases_for('full')
        self.assertEqual(len(cases),len({c['name'] for c in cases}))
        for c in cases:
            self.assertEqual(c['k']%256,0);self.assertEqual((c['k']//256)%c['split'],0)
            self.assertEqual(c['m']%8,0);self.assertLessEqual(c['tokens'],8)
    def test_real_weight_path_not_mixed_with_synthetic_pair(self):
        c=next(x for x in suite.cases_for('quick') if x.get('model_kind')=='gate-up')
        args=suite.case_args(c,{'gate':Path('gate.mw2'),'up':Path('up.mw2')})
        self.assertIn('--weights2',args);self.assertNotIn('--synthetic-pair',args)
    def test_collect_refuses_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'summary.csv'
            p.write_text('variant,kernel_pass,quant_pass\nbad,1,0\n')
            with self.assertRaises(RuntimeError):suite.collect({'name':'c'},Path(d))
    def test_collect_and_aggregate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'summary.csv').write_text('variant,kernel_pass,quant_pass\na,1,1\n')
            c=suite.cases_for('smoke')[0];rows=suite.collect(c,p);suite.write_all(rows,p/'all.csv')
            with (p/'all.csv').open() as inp:r=list(csv.DictReader(inp))
            self.assertEqual(r[0]['case'],c['name']);self.assertEqual(r[0]['variant'],'a')

if __name__=='__main__':unittest.main()
