import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
import reference as r

class ReferenceTests(unittest.TestCase):
    def setUp(self): self.rng=np.random.default_rng(1976)
    def test_pack_all_codes_and_variable_scales(self):
        c=self.rng.integers(-1,3,(3,7,2048),dtype=np.int8)
        d=self.rng.uniform(.01,.3,(3,7,8)).astype(np.float32)
        w=r.pack_codes(c,d)
        np.testing.assert_array_equal(w.codes(),c)
        expected=(c.reshape(3,7,8,256).astype(np.float32)*d.astype(np.float16).astype(np.float32)[...,None]).reshape(3,7,2048)
        np.testing.assert_array_equal(w.decode(),expected)
    def test_bitplane_boundaries(self):
        c=np.zeros((1,1,512),np.int8)
        for index in (0,31,32,63,64,95,96,127,128,159,160,191,192,223,224,255,256,511):
            c.fill(0);c[0,0,index]=-1;w=r.pack_codes(c,np.ones((1,1,2),np.float32));np.testing.assert_array_equal(w.codes(),c)
    def test_q8_zero_rounding_and_finiteness(self):
        x=np.zeros((2,256),np.float32);x[1,:4]=[-127,-.5,.5,127]
        q,d=r.q8g32(x);np.testing.assert_array_equal(q[1,:4],[-127,-1,1,127]);self.assertTrue(np.all(q[0]==0));self.assertTrue(np.all(d[0]==0))
        with self.assertRaises(ValueError):r.q8g32(np.full((1,256),np.nan,np.float32))
    def test_int8_dot_matches_reconstructed_values(self):
        w=r.random_tq2(self.rng,1,33,512,code3=True);x=self.rng.normal(size=(13,512)).astype(np.float32)
        q,d=r.q8g32(x);xd=(q.reshape(13,-1,32).astype(np.float32)*d[...,None]).reshape(x.shape)
        np.testing.assert_allclose(r.dense(w,x,q8=True),xd@w.decode()[0].T,rtol=3e-5,atol=2e-5)
    def test_bucket_permutation_and_capacity(self):
        for n in (1,2,4,8,13,184,665,2048):
            ids=self.rng.integers(0,256,(n,8),dtype=np.uint32)
            offsets,a=r.bucket(ids,256);self.assertEqual(int(offsets[-1]),ids.size)
            np.testing.assert_array_equal(np.sort(a),np.arange(ids.size))
            for e in range(256):self.assertTrue(np.all(ids.reshape(-1)[a[offsets[e]:offsets[e+1]]]==e))
            for rows in (1,31,32,33,512,2048):self.assertLessEqual(len(r.make_jobs(offsets,rows)),r.job_capacity(ids.size,256,rows))
    def test_skew_and_empty_experts(self):
        ids=np.full((184,8),255,np.uint32);o,a=r.bucket(ids,256);self.assertEqual(int(o[255]),0);self.assertEqual(int(o[256]),ids.size)
        with self.assertRaises(ValueError):r.bucket(np.array([[256]],np.uint32),256)
    def test_router_ties_and_order(self):
        z=np.zeros((2,256),np.float32)
        ids,s=r.route(z);np.testing.assert_array_equal(ids[0],np.arange(8));np.testing.assert_array_equal(s,.125)
        ids,s=r.route(z,tie_high=True,ascending=True);np.testing.assert_array_equal(ids[0],np.arange(248,256));np.testing.assert_allclose(s.sum(1),1)
    def test_router_extreme_values(self):
        z=np.full((1,256),-10000,np.float32);z[0,237]=10000;ids,s=r.route(z);self.assertEqual(int(ids[0,0]),237);self.assertEqual(float(s[0,0]),1.0)
    def test_one_sided_gate_clamp(self):
        x=r.clamped_swiglu(np.array([1,10,-10],np.float32),np.array([-20,20,20],np.float32))
        self.assertLess(abs(x[0]),1e-6);self.assertGreater(x[1],48);self.assertLess(x[2],-48)
    def test_virtual_qkv_fusion(self):
        ws=[r.random_tq2(self.rng,1,m,256) for m in (65,33,17)]
        w=r.TQ2(np.concatenate([x.data for x in ws],axis=1),256)
        x=self.rng.normal(size=(13,256)).astype(np.float32)
        for q8 in (False,True):np.testing.assert_allclose(r.dense(w,x,q8=q8),np.concatenate([r.dense(a,x,q8=q8) for a in ws],axis=1),rtol=2e-5,atol=2e-5)
    def test_up_gate_fusion(self):
        ws=[r.random_tq2(self.rng,7,33,256) for _ in range(2)];ids=self.rng.integers(0,7,(13,4),dtype=np.uint32);x=self.rng.normal(size=(13,256)).astype(np.float32)
        w=r.TQ2(np.concatenate([v.data for v in ws],axis=1),256)
        for q8 in (False,True):
            both=r.project_routed(w,x,ids,q8=q8);a=r.clamped_swiglu(both[:,:33],both[:,33:]);b=r.clamped_swiglu(*(r.project_routed(v,x,ids,q8=q8) for v in ws));np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-5)
    def test_down_input_is_per_assignment(self):
        w=r.random_tq2(self.rng,4,5,256);ids=np.array([[3,1],[1,2]],np.uint32);x=self.rng.normal(size=(4,256)).astype(np.float32)
        out=r.project_routed(w,x,ids,per_assignment=True)
        for a,e in enumerate(ids.reshape(-1)):np.testing.assert_allclose(out[a],r.dense(w.expert(int(e)),x[a:a+1])[0],atol=2e-5,rtol=2e-5)
    def test_ordered_expert_reduce(self):
        y=self.rng.normal(size=(24,7)).astype(np.float32);s=self.rng.random((3,8),dtype=np.float32);s/=s.sum(1,keepdims=True)
        np.testing.assert_allclose(r.reduce_experts(y,s),(y.reshape(3,8,7)*s[:,:,None]).sum(1),rtol=1e-6,atol=1e-6)

if __name__=='__main__':unittest.main(verbosity=2)
