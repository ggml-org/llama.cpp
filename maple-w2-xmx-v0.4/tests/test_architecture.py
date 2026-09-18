# SPDX-License-Identifier: MIT
import argparse
import array
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run_architecture_suite as suite
import analyze_architecture_results as analysis
import architecture_common as common
import audit_quant_dump as audit
class ArchitectureTests(unittest.TestCase):
    def test_scope_counts(self):
        self.assertEqual(len(suite.cases_for('smoke')),2)
        self.assertEqual(len(suite.cases_for('quick')),24)
        self.assertEqual(len(suite.cases_for('full')),len(suite.QS)*6)
    def test_cmd_shape(self):
        a=argparse.Namespace(exe=Path('a.exe'),out=Path('out'),device='A750',repeats=28,dump_contract=True,gate=None,up=None,down=None)
        for c in suite.cases_for('quick'):
            args=suite.command_for(c,a);v=dict(zip(args[1::2],args[2::2]))
            self.assertEqual(v['--tokens'],str(c['q']));self.assertEqual(v['--in-order'],'0')
            self.assertEqual(v['--candidate'],'grouped');self.assertEqual(v['--gate-tile'],str(c['gt']))
            self.assertEqual(v['--dump-contract'],str(int(c['q']==184)))
            self.assertNotIn('--freeze-input',v)
    def test_capsules_not_on_small_smoke(self):
        a=argparse.Namespace(exe=Path('a'),out=Path('o'),device='A750',repeats=2,dump_contract=False,
                            gate=Path('g'),up=Path('u'),down=Path('d'))
        self.assertNotIn('--gate',suite.command_for(suite.cases_for('smoke')[0],a))
        self.assertIn('--gate',suite.command_for(suite.cases_for('quick')[0],a))
    def test_pair_stats(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'samples.csv';p.write_text('round,phase,variant,wall_us\n1,repeat,baseline,100\n1,repeat,candidate,80\n2,repeat,baseline,120\n2,repeat,candidate,100\n')
            s=analysis.paired_stats(p);self.assertEqual(s['pairs'],2);self.assertEqual(s['candidate_win_rate'],1)
            self.assertAlmostEqual(s['paired_gain_lower_median'],1-100/120)
    def test_missing_or_duplicate_or_nonfinite(self):
        for content in ('1,repeat,baseline,100\n','1,repeat,baseline,100\n1,repeat,baseline,90\n',
                        '1,repeat,baseline,nan\n1,repeat,candidate,2\n'):
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/'s';p.write_text('round,phase,variant,wall_us\n'+content)
                with self.assertRaises(ValueError):analysis.paired_stats(p)
    def test_stale_stamp(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.exe';p.write_bytes(b'test-only-not-executable')
            with self.assertRaises(ValueError):common.verify_build(p)
            (p.parent/'architecture-build.json').write_text(json.dumps({'version':common.VERSION,'source_sha256':{},'executable_sha256':common.sha256(p)}))
            with self.assertRaises(ValueError):common.verify_build(p)
    def test_source_map(self):
        hashes=common.sources();self.assertIn('src/maple_w2a8_grouped.cpp',hashes);self.assertIn('tools/architecture_compare.cpp',hashes)
    def test_scalar_dump(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);x=[127.,110.5,111.5,-110.5,-111.5]+[0.]*27
            (p/'input.f32').write_bytes(array.array('f',x).tobytes())
            (p/'input.scales.f32').write_bytes(array.array('f',[1.]).tobytes())
            q=[int(round(v)) for v in x];(p/'input.q8').write_bytes(array.array('b',q).tobytes())
            r=audit.inspect(p,'input');self.assertTrue(r['pass']);self.assertEqual(r['cpu_code_differences'],0)
            q[6]=1;(p/'input.q8').write_bytes(array.array('b',q).tobytes());self.assertFalse(audit.inspect(p,'input')['pass'])
    def test_no_arbitrary_global_reuse_on(self):
        s=(ROOT/'include/maple_moe.hpp').read_text();self.assertIn('gate_tokens_per_tile=1,down_tokens_per_tile=1',s)
        self.assertIn('overlap_grouping_quant=false',s)
    def test_new_backend_preserves_dpas_repeat(self):
        s=(ROOT/'src/maple_w2a8_grouped.cpp').read_text();self.assertIn('dpas<8,1,',s)
        self.assertNotIn('dpas<8,2,',s);self.assertNotIn('malloc_',s)
        self.assertNotIn('.wait_and_throw(',s)
if __name__=='__main__':unittest.main()
