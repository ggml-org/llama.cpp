# Golden generator for FLA fused_recurrent_gated_delta_rule (decode path).
# Dumps inputs + outputs + the pinned kernel config to ppu_so/gdn/golden/, so the C++/AOT
# implementation can be validated bit-for-bit against FLA's own kernel.
import os, sys, struct
import torch

FLA = os.environ.get("FLA_ROOT", "/tmp/claude-0/-root/b4d24e49-75af-4442-836b-b20da6e6712c/scratchpad/fla_probe")
sys.path.insert(0, FLA)
from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd

OUT = os.path.join(os.path.dirname(__file__), "golden")
os.makedirs(OUT, exist_ok=True)

def dump(name, t):
    a = t.detach().contiguous().cpu().float().numpy()
    a.tofile(os.path.join(OUT, name + ".bin"))
    return a.shape

torch.manual_seed(0)
dev = "cuda"
# small but non-trivial: exercise the recurrence over T>1, GVA (HV>H)
B, T, H, HV, K, V = 1, 8, 2, 4, 128, 128
scale = 1.0 / (K ** 0.5)

q  = torch.randn(B, T, H,  K, device=dev, dtype=torch.float32)
k  = torch.randn(B, T, H,  K, device=dev, dtype=torch.float32)
v  = torch.randn(B, T, HV, V, device=dev, dtype=torch.float32)
# g: log-space decay in [B,T,HV], use small negative (typical gate). beta in (0,1).
g    = (-torch.rand(B, T, HV, device=dev, dtype=torch.float32) * 0.5)
beta = torch.rand(B, T, HV, device=dev, dtype=torch.float32)
h0   = torch.randn(B, HV, K, V, device=dev, dtype=torch.float32) * 0.1

o, ht = fused_recurrent_gated_delta_rule_fwd(
    q=q, k=k, v=v, g=g, gk=None, gv=None, beta=beta,
    A_log=None, dt_bias=None, scale=scale,
    initial_state=h0, output_final_state=True,
    use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False, state_v_first=False, cu_seqlens=None,
)

for nm, t in [("q",q),("k",k),("v",v),("g",g),("beta",beta),("h0",h0),("o",o),("ht",ht)]:
    print(f"{nm:4s} {tuple(t.shape)}  -> {dump(nm, t)}")

# pinned config (matches the launcher's derivation)
import triton
BK = triton.next_power_of_2(K)
BV = min(8, triton.next_power_of_2(V))
NV = triton.cdiv(V, BV)
cfg = dict(B=B, T=T, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, NV=NV, scale=scale)
with open(os.path.join(OUT, "config.txt"), "w") as f:
    for kk, vv in cfg.items():
        f.write(f"{kk}={vv}\n")
print("config:", cfg)
print("OK -> golden dumped to", OUT)
