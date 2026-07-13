#!/usr/bin/env python3
# AOT-compile the FLA chunked gated-delta-rule forward kernel chain into embedded-cubin C (torch/python/triton-free).
# Chain (chunk_size=64): chunk_local_cumsum_scalar -> fwd_kkt_solve -> recompute_w_u -> fwd_h(blockdim64) -> fwd_o.
# Autotune block sizes only affect SPEED (not the result), so we pin a valid config per kernel.
#
# This script only does the AOT step per kernel (generic helper). The C++ orchestration lives in gdn_chunk.c
# (host-side buffer alloc + grids + launch sequence). Usage:  FLA_ROOT=... python3 aot_chunk.py <H> <HV> <S>
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FLA  = os.environ["FLA_ROOT"]
OUT  = os.path.join(HERE, "aot_chunk"); os.makedirs(OUT, exist_ok=True)

def shim(modpath, kname):
    p = os.path.join(OUT, f"_shim_{kname}.py")
    with open(p, "w") as f:
        f.write(f"import sys; sys.path.insert(0,{FLA!r})\n"
                f"import triton\n"
                f"from {modpath} import {kname} as _k\n"
                f"_u=_k\n"
                f"while not isinstance(_u,triton.runtime.JITFunction) and hasattr(_u,'fn'): _u=_u.fn\n"
                f"{kname}=_u\n")
    return p

def aot(modpath, kname, outname, signature, grid, num_warps=4, num_stages=2):
    sp = shim(modpath, kname)
    cmd = [sys.executable, "-m", "triton.tools.compile", "--kernel-name", kname,
           "--out-name", outname, "-o", os.path.join(OUT, outname),
           "--num-warps", str(num_warps), "--num-stages", str(num_stages),
           "--grid", grid, "--signature", signature, sp]
    subprocess.run(cmd, check=True, env=dict(os.environ, PYTHONPATH=FLA))
    cf = [f for f in os.listdir(OUT) if f.startswith(outname + ".") and f.endswith(".c")][0]
    hf = cf[:-2] + ".h"
    for fn in (cf, hf):  # fp32-scalar ABI fix (read THEN write -- one-liner would truncate before reading)
        fp = os.path.join(OUT, fn)
        txt = open(fp).read().replace("double scale", "float scale")
        open(fp, "w").write(txt)
    suf = cf[len(outname)+1:-2]
    print(f"  AOT {outname:16s} suffix={suf}")
    return suf

def next_pow2(x):
    p=1
    while p<x: p<<=1
    return p

def main():
    H, HV, S = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    K = V = S
    BT = 64
    P = "*fp32:16"; Pi = "*i32"
    sufs = {}

    # --- stage 1: cumsum scalar. args: s,o,scale,cu_seqlens,chunk_indices,T,B,H,BT,REVERSE,HAS_SCALE,IS_VARLEN,HEAD_FIRST
    sufs["cumsum"] = aot("fla.ops.utils.cumsum", "chunk_local_cumsum_scalar_kernel", "gdn_cumsum",
        f"{P},{P},fp32,{Pi},{Pi},i32,1,{HV},{BT},0,1,0,0",
        grid="(T+63)/64,%d,1" % HV, num_warps=1, num_stages=1)

    # --- stage 2: fused kkt+solve. args: k,g,beta,A,cu_seqlens,chunk_indices,T,H,HV,K,BT,BC,BK,USE_G,IS_VARLEN
    BC = 16; BK = next_pow2(K)
    sufs["kkt"] = aot("fla.ops.gated_delta_rule.chunk_fwd", "chunk_gated_delta_rule_fwd_kkt_solve_kernel", "gdn_kkt",
        f"{P},{P},{P},{P},{Pi},{Pi},i32,{H},{HV},{K},{BT},{BC},{BK},1,0",
        grid="(T+63)/64,%d,1" % HV, num_warps=4, num_stages=2)

    # --- stage 3: recompute_w_u. args: k,v,beta,w,u,A,g,cu_seqlens,chunk_indices,T,H,HV,K,V,BT,BK,BV,USE_G,IS_VARLEN
    BV = next_pow2(V)
    sufs["wu"] = aot("fla.ops.gated_delta_rule.wy_fast", "recompute_w_u_fwd_kernel", "gdn_wu",
        f"{P},{P},{P},{P},{P},{P},{P},{Pi},{Pi},i32,{H},{HV},{K},{V},{BT},{BK},{BV},1,0",
        grid="(T+63)/64,%d,1" % HV, num_warps=4, num_stages=2)

    # --- stage 4: fwd_h blockdim64. args: k,v,w,v_new,g,gk,h,h0,ht,cu_seqlens,chunk_offsets,T,H,HV,K,V,BT,BV,
    #     USE_G,USE_GK,USE_INITIAL_STATE,STORE_FINAL_STATE,SAVE_NEW_VALUE,STATE_V_FIRST,IS_VARLEN
    # num_stages=1 + BV=32 keeps smem at 32KB (<48KB, no opt-in): Triton's AOT launcher OOBs for the >128KB-smem
    # wgmma configs (works in the python JIT, fails standalone), so we stay under that ceiling. Correctness is
    # tiling-independent (validated: every config is bit-exact in python).
    BVh = 32
    NVh = (V + BVh - 1)//BVh
    sufs["h"] = aot("fla.ops.common.chunk_delta_h", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", "gdn_h",
        f"{P},{P},{P},{P},{P},{P},{P},{P},{P},{Pi},{Pi},i32,{H},{HV},{K},{V},{BT},{BVh},1,0,1,1,1,0,0",
        grid="%d,%d,1" % (NVh, HV), num_warps=4, num_stages=1)

    # --- stage 5: fwd_o. args: q,k,v,h,g,g_gamma,o,cu_seqlens,chunk_indices,scale,T,H,HV,K,V,BT,BK,BV,
    #     USE_G,USE_G_GAMMA,STATE_V_FIRST,IS_VARLEN
    # BK=64, BV=32, num_stages=1 -> smem 40KB (<48KB, no opt-in), under the AOT-failing ceiling. (BK<K just tiles K.)
    BKo = 64 if K % 64 == 0 else next_pow2(K)
    BVo = 32
    sufs["o"] = aot("fla.ops.common.chunk_o", "chunk_fwd_kernel_o", "gdn_o",
        f"{P},{P},{P},{P},{P},{P},{P},{Pi},{Pi},fp32,i32,{H},{HV},{K},{V},{BT},{BKo},{BVo},1,0,0,0",
        grid="(T+63)/64,%d,1" % HV, num_warps=4, num_stages=1)

    import json
    json.dump({"H":H,"HV":HV,"S":S,"BT":BT,"sufs":sufs}, open(os.path.join(OUT,"meta.json"),"w"), indent=2)
    print("meta ->", os.path.join(OUT,"meta.json"))

if __name__ == "__main__":
    main()
