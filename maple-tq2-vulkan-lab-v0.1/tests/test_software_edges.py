#!/usr/bin/env python3
"""Supplemental Linux Mesa execution tests. NOT Vulkan or native INT8 validation."""
from pathlib import Path
import sys,tempfile,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import reference as ref
from make_fixture import Plan
from glsl_software_oracle import Oracle

class SoftwareEdges(unittest.TestCase):
    def run_qkv(self,q8,reuse):
        with tempfile.TemporaryDirectory(prefix='tq2-qkv-') as directory:
            root=Path(directory);rng=np.random.default_rng(102)
            n,k=5,512;dims=(33,17,15);offsets=(2,6,10)
            x=rng.normal(size=(n,k)).astype(np.float32)
            # Include code 3 (+2); it is representable even if a native ternary
            # packer normally emits only 0/1/2. Block scales vary independently.
            weights=[]
            for m in dims:
                codes=rng.integers(-1,3,(1,m,k),dtype=np.int8)
                scales=rng.uniform(.005,.1,(1,m,k//256)).astype(np.float16)
                scales[:,::7,0]=0
                weights.append(ref.pack_codes(codes,scales))
            p=Plan(root,q8);p.buffer('control',16);p.zero('control')
            for i,(w,off) in enumerate(zip(weights,offsets)):
                payload=b'\xa5'*off+w.data.tobytes()
                p.buffer('w'+str(i),len(payload),payload)
            p.buffer('input',x.nbytes,x);inp='input';sc='dummy'
            if q8:
                p.buffer('qx',n*k);p.buffer('sx',n*k//32*4)
                p.dispatch('quantize_q8g32',(n*k//32,1,1),[k,n],['input','qx','sx','control'])
                inp='qx';sc='sx'
            total=sum(dims);p.buffer('out',n*total*4)
            p.dispatch(f'project_q{int(q8)}_f2_r{int(reuse)}',(((n+3)//4)*((total+31)//32),1,1),
                       [k,total,n,1,1,0,0,dims[0],dims[1],*offsets],
                       ['w0','w1','w2',inp,sc,'dummy','dummy','control','out'])
            p.dump('out','out.actual.bin');p.dump('control','control.actual.bin');p.save()
            oracle=Oracle()
            try:oracle.run(root,ROOT/'shaders')
            finally:oracle.close()
            expected=np.concatenate([ref.dense(w,x,q8=q8) for w in weights],axis=1)
            actual=np.fromfile(root/'out.actual.bin',dtype='<f4').reshape(n,total)
            np.testing.assert_allclose(actual,expected,rtol=3e-4,atol=3e-4)
            self.assertEqual(np.fromfile(root/'control.actual.bin',dtype='<u4')[2],0)
            print('EDGE PASS',q8,reuse,'max_abs',float(np.max(np.abs(actual-expected))))
    def test_f32_reuse(self):self.run_qkv(False,True)
    def test_q8_reuse(self):self.run_qkv(True,True)
    def test_f32_no_reuse(self):self.run_qkv(False,False)
    def test_q8_no_reuse(self):self.run_qkv(True,False)

if __name__=='__main__':unittest.main()
