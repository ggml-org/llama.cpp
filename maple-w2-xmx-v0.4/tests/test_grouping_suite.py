import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run_grouping_suite as g

class GroupSuiteTests(unittest.TestCase):
    def test_full_qs(self):
        p=g.cases_for('full');self.assertEqual([c['tokens'] for c in p if c.get('model_shape')],list(g.QS))
        self.assertEqual(len(p),14)
    def test_two_smokes(self):
        p=g.cases_for('smoke');self.assertEqual(len(p),2);self.assertEqual(p[1]['tokens'],13);self.assertEqual(p[1]['order'],0)
    def test_quick(self):self.assertEqual(len(g.cases_for('quick')),7)
    def test_fixed_parameters(self):
        for c in g.cases_for('full'):
            a=g.case_args(c,{},28,0);d=dict(zip(a[::2],a[1::2]))
            self.assertEqual(d['--gate-split'],'1');self.assertEqual(d['--down-split'],'1');self.assertEqual(d['--local'],'4')
            self.assertNotIn('--groups',d);self.assertNotIn('--host-waits',d)
    def test_capsules(self):
        caps={k:Path(k+'.mw2') for k in ('gate','up','down')}
        self.assertNotIn('--gate',g.case_args(g.cases_for('full')[0],caps,28,0))
        self.assertIn('--gate',g.case_args(g.cases_for('full')[-1],caps,28,0))
    def test_pair_stats(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'s.csv'
            p.write_text('round,phase,variant,wall_us\n1,repeat,baseline,100\n1,repeat,x-expert-grouped,80\n2,repeat,baseline,90\n2,repeat,x-expert-grouped,110\n',encoding='utf-8')
            r=g.paired_stats(p);self.assertEqual(r['paired_rounds'],2);self.assertEqual(r['grouped_wins'],1);self.assertEqual(r['paired_saved_median_us'],0)
    def test_unpaired_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'s.csv';p.write_text('round,phase,variant,wall_us\n1,repeat,baseline,100\n',encoding='utf-8')
            with self.assertRaises(ValueError):g.paired_stats(p)
    def test_sources_cover_core(self):
        s=g.sources();self.assertIn('src/expert_grouping.cpp',s);self.assertIn('src/maple_w2a8.cpp',s);self.assertIn('tools/grouping_compare.cpp',s)
    def test_missing_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.exe';p.write_bytes(b'not a real executable')
            with self.assertRaises(ValueError):g.verify_build(p)
    def test_stale_binary_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.exe';p.write_bytes(b'stub');m={'version':'0.4','source_sha256':g.sources(),'executable_sha256':'wrong'}
            (p.parent/'grouping-build.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):g.verify_build(p)
    def test_stale_source_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.exe';p.write_bytes(b'stub');m={'version':'0.4','source_sha256':{},'executable_sha256':g.sha256(p)}
            (p.parent/'grouping-build.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):g.verify_build(p)
    def test_host_no_wait_in_group_builder(self):
        s=(ROOT/'src/expert_grouping.cpp').read_text();self.assertNotIn('.wait(',s);self.assertNotIn('.wait_and_throw(',s)
        self.assertNotIn('malloc_',s);self.assertNotIn('memcpy(',s)
    def test_original_output_job(self):
        s=(ROOT/'src/maple_w2a8.cpp').read_text();self.assertIn('job=size_t(mapped);',s)
        self.assertIn('const size_t oi=job*p.shape.m+rt*8;',s)
        self.assertIn('const size_t xr=p.per_selection?job:job/p.topk;',s)
    def test_only_two_benchmark_arms(self):
        s=(ROOT/'tools/grouping_compare.cpp').read_text();self.assertIn('for(bool grouped:{false,true})',s)
        self.assertIn('vars[(r+j)%2]',s)
        self.assertNotIn('MoePath::a16;',s)
    def test_glu_and_a8_math_unchanged(self):
        before=(ROOT/'evidence/v03_a8_compute_excerpt.txt').read_text()
        after=(ROOT/'src/maple_w2a8.cpp').read_text()
        region=after[after.index('const size_t xr=p.per_selection?job:job/p.topk;'):after.index('} else if(rt==0 && sp==0)')]
        self.assertEqual(before,region)
        before=(ROOT/'evidence/v03_moe_epilogue_excerpt.txt').read_text()
        after=(ROOT/'src/maple_moe.cpp').read_text()
        region=after[after.index('static sycl::event epilogue('):after.index('MoeRun enqueue_moe(')]
        self.assertEqual(before,region)
if __name__=='__main__':unittest.main()
