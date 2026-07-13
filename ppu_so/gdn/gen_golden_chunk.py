# Golden for FLA chunked gated delta rule (prefill). Dumps inputs + ALL intermediates, so the C++/AOT
# orchestration can be validated stage-by-stage. chunk_size=64 (fused kkt+solve path).
import os, sys
import torch
FLA = os.environ.get("FLA_ROOT")
sys.path.insert(0, FLA)
from fla.ops.utils import chunk_local_cumsum
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o
from fla.ops.utils.constant import RCP_LN2

OUT = os.path.join(os.path.dirname(__file__), "golden_chunk"); os.makedirs(OUT, exist_ok=True)
def dump(nm, t):
    if t is None: print(f"{nm}: None"); return
    t.detach().contiguous().cpu().float().numpy().tofile(os.path.join(OUT, nm+".bin"))
    print(f"{nm:10s} {tuple(t.shape)} {t.dtype}")

torch.manual_seed(0)
dev="cuda"
B,T,H,HV,K,V = 1, 256, 4, 4, 128, 128   # T multiple of chunk 64; non-GVA first
BT=64
scale = 1.0/(K**0.5)
q = torch.randn(B,T,H,K, device=dev, dtype=torch.float32)
k = torch.randn(B,T,H,K, device=dev, dtype=torch.float32)
q = torch.nn.functional.normalize(q, p=2, dim=-1)   # real GDN L2-normalizes q,k
k = torch.nn.functional.normalize(k, p=2, dim=-1)
v = torch.randn(B,T,HV,V, device=dev, dtype=torch.float32)
g_raw = -torch.rand(B,T,HV, device=dev, dtype=torch.float32)*0.5   # log-space gate (pre-cumsum)
beta  = torch.rand(B,T,HV, device=dev, dtype=torch.float32)
h0 = torch.randn(B,HV,K,V, device=dev, dtype=torch.float32)*0.1

# Stage 1: cumsum (scale RCP_LN2 -> base-2 exp downstream)
g = chunk_local_cumsum(g_raw, chunk_size=BT, scale=RCP_LN2, cu_seqlens=None, output_dtype=torch.float32)
# Stage 2: WY (fused kkt+solve) + recompute -> w,u,A
w, u, A = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta, cu_seqlens=None, chunk_size=BT)
# Stage 3: inter-chunk state -> h, v_new, final_state
h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
    k=k, w=w, u=u, g=g, gk=None, initial_state=h0, output_final_state=True,
    chunk_size=BT, save_new_value=True, state_v_first=False, cu_seqlens=None)
# Stage 4: output
o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, state_v_first=False, cu_seqlens=None, chunk_size=BT)

for nm,t in [("q",q),("k",k),("v",v),("g_raw",g_raw),("beta",beta),("h0",h0),
             ("g",g),("A",A),("w",w),("u",u),("h",h),("v_new",v_new),("o",o),("final_state",final_state)]:
    dump(nm,t)
with open(os.path.join(OUT,"config.txt"),"w") as f:
    for kk,vv in dict(B=B,T=T,H=H,HV=HV,K=K,V=V,BT=BT,scale=scale,NT=T//BT).items(): f.write(f"{kk}={vv}\n")
print("NT(chunks)=",T//BT,"  h.shape=",tuple(h.shape))
print("OK ->", OUT)
