#!/usr/bin/env python3
"""Create a reproducible, one-layer MoE + QKV Vulkan replay fixture.

The plan is a standalone test workload, NOT an injection into llama-server.
Each optimization has an A/B control. CPU references are generated independently.
"""
from pathlib import Path
import argparse,hashlib,json,math,struct
import numpy as np
import reference as ref

class Plan:
    def __init__(self, root: Path, require_dot: bool):
        self.root=root.resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.lines=['MAPLE_TQ2_PLAN 1',f'require_dot {int(require_dot)}'];self.sizes={}
        self.buffer('dummy',16)
    def buffer(self,name,size,data=None):
        if not name.replace('_','').isalnum() or name in self.sizes: raise ValueError('invalid/duplicate buffer name')
        size=(int(size)+3)//4*4
        if size<=0 or size>=2**32: raise ValueError('buffer must fit 32-bit shader-relative indexing')
        self.sizes[name]=size
        filename='-'
        if data is not None:
            raw=data if isinstance(data,bytes) else np.ascontiguousarray(data).tobytes()
            if len(raw)>size: raise ValueError('input does not fit buffer')
            filename=name+'.input.bin';(self.root/filename).write_bytes(raw+b'\0'*(size-len(raw)))
        self.lines.append(f'buffer {name} {size} "{filename}"')
    def zero(self,name,size=None,offset=0):
        size=self.sizes[name] if size is None else size
        if offset%4 or size%4 or offset+size>self.sizes[name]:raise ValueError('bad fill range')
        self.lines.append(f'fill {name} {offset} {size} 0')
    def dispatch(self,kernel,grid,push,bindings):
        if any(int(v)!=v or v<0 or v>=2**32 for v in push):raise ValueError('invalid push word')
        if len(push)>32 or len(bindings)>9:raise ValueError('push/binding limit exceeded')
        # Linear kernels flatten 2D workgroup ids; keep every dimension <=65535.
        if grid[1:]==(1,1) and grid[0]>65535:
            grid=(65535,(grid[0]+65534)//65535,1)
        if any(x not in self.sizes for x in bindings):raise ValueError('unknown binding')
        if any(g<1 or g>65535 for g in grid):raise ValueError('dispatch dimension outside portable limits; chunk the workload')
        self.lines.append('dispatch '+kernel+' '+' '.join(map(str,grid))+' '+str(len(push))+' '+' '.join(map(str,push))+' '+str(len(bindings))+' '+' '.join(bindings))
    def dump(self,name,filename):self.lines.append(f'dump {name} "{filename}"')
    def save(self):
        (self.root/'run.plan').write_text('\n'.join(self.lines)+'\n',encoding='utf-8')
        return sum(self.sizes.values())


def create(a):
    if not (1<=a.topk<=8 and a.topk<=a.experts<=256):raise ValueError('1<=topk<=8, topk<=experts<=256 required')
    if min(a.hidden,a.ffn)<=0 or a.hidden%256 or a.ffn%256:raise ValueError('hidden and ffn must be positive multiples of 256')
    if a.tokens<1 or a.tokens*a.topk>=2**24:raise ValueError('invalid tokens/assignments')
    rng=np.random.default_rng(a.seed)
    n,k,h,e,s=a.tokens,a.hidden,a.ffn,a.experts,a.topk;A=n*s
    # Limit fixture size before allocating arrays. This is not a whole-model load.
    estimate=(3*e*k*h*66//256)+A*(h+k)*4+e*k*4
    if estimate>a.max_fixture_mib*2**20:raise ValueError('fixture exceeds --max-fixture-mib budget')
    p=Plan(a.out,a.q8)
    x=rng.normal(0,0.5,(n,k)).astype(np.float32)
    rw=rng.normal(0,0.02,(e,k)).astype(np.float32)
    up=ref.random_tq2(rng,e,h,k);gate=ref.random_tq2(rng,e,h,k);down=ref.random_tq2(rng,e,k,h)
    p.buffer('input',x.nbytes,x);p.buffer('router_w',rw.nbytes,rw)
    for name,w in [('up',up),('gate',gate),('down',down)]:p.buffer(name,w.data.nbytes,w.data)
    p.buffer('control',16);p.zero('control')
    p.buffer('ids',A*4);p.buffer('scores',A*4)
    if a.full_router:
        p.dispatch('router_fused',(n,1,1),[n,e,s,int(a.tie_high),int(a.ascending),k],['input','router_w','ids','scores','control'])
    else:
        p.buffer('logits',n*e*4)
        p.dispatch('router_logits',((e+7)//8,n,1),[n,e,k],['input','router_w','logits','control'])
        p.dispatch('router_select',(n,1,1),[n,e,s,int(a.tie_high),int(a.ascending)],['logits','ids','scores','control'])
        p.dump('logits','logits.actual.bin')
    grouped=a.route=='bucket' or (a.route=='auto' and A>=64)
    if grouped:
        p.buffer('counts',e*4);p.zero('counts');p.buffer('offsets',(e+1)*4);p.buffer('cursors',e*4);p.buffer('sorted',A*4)
        p.dispatch('bucket_count',((A+255)//256,1,1),[A,e],['ids','counts','control'])
        p.dispatch('bucket_prefix',(1,1,1),[e],['counts','offsets','cursors'])
        p.dispatch('bucket_scatter',((A+255)//256,1,1),[A,e],['ids','offsets','cursors','sorted','control'])
        capacity=max(ref.job_capacity(A,e,h),ref.job_capacity(A,e,k))
        p.buffer('jobs',capacity*16)
        p.dump('offsets','offsets.actual.bin');p.dump('sorted','sorted.actual.bin')
    else:capacity=0
    def quantize(name,rows,width):
        if not a.q8:return name,'dummy'
        qname=name+'_q';sname=name+'_s'
        p.buffer(qname,rows*width);p.buffer(sname,rows*(width//32)*4)
        p.dispatch('quantize_q8g32',(rows*(width//32),1,1),[width,rows],[name,qname,sname,'control'])
        return qname,sname
    def project(name,w0,w1,inp,sc,width,rows,out_rows,fusion,per_assignment=False):
        # rows: input row count; assignment output always A.
        p.buffer(name,A*out_rows*4)
        schedule=(2 if a.persistent else 3) if grouped else 1
        if grouped:
            # Keep errors from earlier stages; reset only cursor and njobs.
            p.zero('control',8)
            p.dispatch('build_jobs',(1,1,1),[e,out_rows,capacity],['offsets','jobs','control'])
            groups=a.workgroups if a.persistent else capacity
        else:groups=A*((out_rows+31)//32)
        p.dispatch(f'project_q{int(a.q8)}_f{fusion}_r{int(a.reuse)}',(groups,1,1),
                   [width,out_rows,rows,s,e,schedule,int(per_assignment),0,0,0,0,0],
                   [w0,w1,'dummy',inp,sc,'sorted' if grouped else 'ids','jobs' if grouped else 'dummy','control',name])
    inp,ins=quantize('input',n,k)
    if a.fuse_upgate:project('intermediate','up','gate',inp,ins,k,n,h,1)
    else:
        project('up_out','up','dummy',inp,ins,k,n,h,0);project('gate_out','gate','dummy',inp,ins,k,n,h,0)
        p.buffer('intermediate',A*h*4)
        p.dispatch('swiglu',((A*h+255)//256,1,1),[A*h],['up_out','gate_out','intermediate'])
    mid,ms=quantize('intermediate',A,h)
    project('expert_out','down','dummy',mid,ms,h,A,k,0,per_assignment=True)
    p.buffer('moe_out',n*k*4)
    p.dispatch('reduce_experts',((n*k+255)//256,1,1),[n,k,s],['expert_out','scores','moe_out'])
    # Dense QKV projection is separate from the MoE path. No attention/KV mutation.
    nq,nk,nv=k,k//4,k//4
    qs=[ref.random_tq2(rng,1,dim,k) for dim in (nq,nk,nv)]
    for name,w in zip(('wq','wk','wv'),qs):p.buffer(name,w.data.nbytes,w.data)
    for name,dim in zip(('q_out','k_out','v_out'),(nq,nk,nv)):p.buffer(name,n*dim*4)
    if a.fuse_qkv:
        total=nq+nk+nv;p.buffer('qkv_out',n*total*4)
        p.dispatch(f'project_q{int(a.q8)}_f2_r{int(a.reuse)}',(((n+3)//4)*((total+31)//32),1,1),
                   [k,total,n,1,1,0,0,nq,nk,0,0,0],['wq','wk','wv',inp,ins,'dummy','dummy','control','qkv_out'])
        p.dispatch('split_qkv',((n*total+255)//256,1,1),[n,nq,nk,nv],['qkv_out','q_out','k_out','v_out'])
    else:
        for name,wname,dim in zip(('q_out','k_out','v_out'),('wq','wk','wv'),(nq,nk,nv)):
            p.dispatch(f'project_q{int(a.q8)}_f0_r{int(a.reuse)}',(((n+3)//4)*((dim+31)//32),1,1),
                       [k,dim,n,1,1,0,0,0,0,0,0,0],[wname,'dummy','dummy',inp,ins,'dummy','dummy','control',name])
    for name in ('ids','scores','intermediate','expert_out','moe_out','q_out','k_out','v_out','control'):p.dump(name,name+'.actual.bin')
    footprint=p.save()
    logits=x@rw.T;ids,scores=ref.route(logits,s,tie_high=a.tie_high,ascending=a.ascending)
    mid_ref=ref.clamped_swiglu(ref.project_routed(up,x,ids,q8=a.q8),ref.project_routed(gate,x,ids,q8=a.q8))
    eo=ref.project_routed(down,mid_ref,ids,per_assignment=True,q8=a.q8)
    expected={'ids':ids,'scores':scores,'intermediate':mid_ref,'expert_out':eo,'moe_out':ref.reduce_experts(eo,scores),
              'logits':logits,**{name:ref.dense(w,x,q8=a.q8) for name,w in zip(('q_out','k_out','v_out'),qs)}}
    for name,data in expected.items():data.tofile(p.root/(name+'.expected.bin'))
    meta={'seed':a.seed,'tokens':n,'hidden':k,'ffn':h,'experts':e,'topk':s,'q8':a.q8,'grouped':grouped,
          'persistent':a.persistent,'workgroups':a.workgroups,'reuse':a.reuse,'fuse_upgate':a.fuse_upgate,
          'fuse_qkv':a.fuse_qkv,'full_router':a.full_router,'tie_high':a.tie_high,'ascending':a.ascending,
          'device_payload_bytes':footprint,'expected':{key:{'shape':list(val.shape),'dtype':str(val.dtype)} for key,val in expected.items()}}
    (p.root/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'Created {p.root}/run.plan; device buffer payload {footprint/2**20:.2f} MiB')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('fixtures/demo'))
    ap.add_argument('--tokens',type=int,default=13);ap.add_argument('--hidden',type=int,default=256);ap.add_argument('--ffn',type=int,default=256)
    ap.add_argument('--experts',type=int,default=256);ap.add_argument('--topk',type=int,default=8);ap.add_argument('--seed',type=int,default=918)
    ap.add_argument('--q8',action='store_true');ap.add_argument('--route',choices=['auto','direct','bucket'],default='auto')
    ap.add_argument('--persistent',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--workgroups',type=int,default=128,help='experimental persistent grid, not a tuned A750 result')
    ap.add_argument('--reuse',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--fuse-upgate',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--fuse-qkv',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--full-router',action='store_true',help='experimental all-in-one router; default uses parallel GEMV + fused selection')
    ap.add_argument('--tie-high',action='store_true');ap.add_argument('--ascending',action='store_true')
    ap.add_argument('--max-fixture-mib',type=int,default=512)
    a=ap.parse_args()
    if not 1<=a.workgroups<=65535:ap.error('invalid --workgroups')
    try:create(a)
    except ValueError as exc:ap.error(str(exc))
if __name__=='__main__':main()
