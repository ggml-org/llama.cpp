#!/usr/bin/env python3
# Build the chunked GDN forward chain from FLA's Triton kernels using the JIT-compiled cubins (NOT triton.tools.compile,
# whose AOT path miscompiles the large wgmma kernels). For each of the 5 kernels: JIT-compile it via the normal
# `kernel[grid](...)` path (produces a correct cubin), dump the cubin + record its launch metadata (grid, block, smem,
# and the ORDERED surviving-param list — Triton prunes None args from the cubin's parameter list, so the launcher must
# pass exactly the survivors + T + a trailing global_scratch pointer).
#
# Emits into cubin_chunk/: <kernel>.cubin (5) + manifest.json.  Usage: FLA_ROOT=... python3 gen_chunk_jit.py <H> <HV> <S>
import os, sys, json
import numpy as np, torch

FLA = os.environ["FLA_ROOT"]; sys.path.insert(0, FLA)
import triton
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.cumsum import chunk_local_cumsum_scalar_kernel
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd_kernel
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_kernel_h_blockdim64
from fla.ops.common.chunk_o import chunk_fwd_kernel_o

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "cubin_chunk"); os.makedirs(OUT, exist_ok=True)

def unwrap(k):
    while not isinstance(k, triton.runtime.JITFunction) and hasattr(k, "fn"): k = k.fn
    return k

def dump(compiled, key):
    cubin = compiled.asm["cubin"]
    open(os.path.join(OUT, key + ".cubin"), "wb").write(cubin)
    m = compiled.metadata
    return dict(key=key, name=m.name, cubin=key + ".cubin",
                block=m.num_warps * 32, smem=int(m.shared),
                cluster=list(m.cluster_dims))

def main():
    H, HV, S = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    K = V = S; BT = 64; dev = "cuda"
    # a golden-shaped set of inputs just to trigger JIT compilation (values irrelevant to the cubin)
    T = 256; NT = T // BT; B = 1
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(B,T,H,K,device=dev), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B,T,H,K,device=dev), dim=-1)
    v = torch.randn(B,T,HV,V,device=dev); beta = torch.rand(B,T,HV,device=dev)
    g_raw = -torch.rand(B,T,HV,device=dev)*0.5; h0 = torch.randn(B,HV,K,V,device=dev)*0.1
    g = chunk_local_cumsum(g_raw, chunk_size=BT, scale=RCP_LN2, output_dtype=torch.float32)

    man = {"H":H,"HV":HV,"S":S,"BT":BT,"kernels":{}}

    # 1. cumsum  (compile a fresh call so we capture the compiled object)
    ku = unwrap(chunk_local_cumsum_scalar_kernel)
    go = torch.empty_like(g_raw)
    c = ku[(triton.cdiv(T,BT), B*HV)](g_raw, go, RCP_LN2, None, None, T, B, HV, BT, False, True, False, False)
    man["kernels"]["cumsum"] = {**dump(c,"cumsum"),
        "grid":["(T+63)/64", HV, 1], "args":["g_raw","g","@scale","@T"]}

    # 2. kkt+solve -> A
    A = torch.zeros(B,T,HV,BT,device=dev)
    BC=16; BK=1
    while BK < K: BK <<= 1
    kk = unwrap(chunk_gated_delta_rule_fwd_kkt_solve_kernel)
    c = kk[(triton.cdiv(T,BT), B*HV)](k, g, beta, A, None, None, T, H, HV, K, BT, BC, BK, True, False)
    man["kernels"]["kkt"] = {**dump(c,"kkt"), "grid":["(T+63)/64", HV, 1], "args":["k","g","beta","A","@T"]}

    # 3. recompute w,u
    w = torch.empty(B,T,HV,K,device=dev); u = torch.empty(B,T,HV,V,device=dev)
    BV=1
    while BV < V: BV <<= 1
    wu = unwrap(recompute_w_u_fwd_kernel)
    c = wu[(triton.cdiv(T,BT), B*HV)](k, v, beta, w, u, A, g, None, None, T, H, HV, K, V, BT, BK, BV, True, False)
    man["kernels"]["wu"] = {**dump(c,"wu"), "grid":["(T+63)/64", HV, 1], "args":["k","v","beta","w","u","A","g","@T"]}

    # 4. fwd_h  (BV=32, num_stages=1 keeps smem modest; correctness is tiling-independent)
    h = torch.empty(B,NT,HV,K,V,device=dev); vnew = torch.empty_like(u); ht = torch.zeros(B,HV,K,V,device=dev)
    BVh=32
    hk = unwrap(chunk_gated_delta_rule_fwd_kernel_h_blockdim64)
    c = hk[(triton.cdiv(V,BVh), B*HV)](k, u, w, vnew, g, None, h, h0, ht, None, None, T, H, HV, K, V, BT, BVh,
                                        True, False, True, True, True, False, False, num_warps=4, num_stages=1)
    man["kernels"]["h"] = {**dump(c,"h"), "grid":[triton.cdiv(V,BVh), HV, 1],
        "args":["k","u","w","v_new","g","h","h0","ht","@T"]}

    # 5. fwd_o  (BK=64, BV=32, num_stages=1 -> modest smem)
    o = torch.empty_like(u); scale = 1.0/(K**0.5)
    BKo = 64 if K % 64 == 0 else BK; BVo = 32
    ok = unwrap(chunk_fwd_kernel_o)
    c = ok[(triton.cdiv(T,BT), B*HV)](q, k, vnew, h, g, None, o, None, None, scale, T, H, HV, K, V, BT, BKo, BVo,
                                       True, False, False, False, num_warps=4, num_stages=1)
    man["kernels"]["o"] = {**dump(c,"o"), "grid":["(T+63)/64", HV, 1], "args":["q","k","v_new","h","g","o","@scale","@T"]}

    json.dump(man, open(os.path.join(OUT,"manifest.json"),"w"), indent=2)
    for kk_,vv in man["kernels"].items():
        print(f"  {kk_:8s} smem={vv['smem']:6d} block={vv['block']} grid={vv['grid']} nargs={len(vv['args'])}")
    print("manifest ->", os.path.join(OUT,"manifest.json"))

if __name__ == "__main__":
    main()
