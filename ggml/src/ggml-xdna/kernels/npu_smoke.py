"""Standalone check of one ggml-xdna block-GEMM kernel on the NPU through pyxrt.

Mirrors exactly what ggml-xdna.cpp does (same BO groups, opcode and layouts), so a pass here
validates kernel + dispatch ABI + numerics independently of the ggml build.

    python npu_smoke.py [kernel-name] [iterations]

pyxrt ships with the NPU driver (Windows: C:\\Windows\\System32\\AMD\\pyxrt.pyd, built for the
Python version noted in its import error; Linux: /opt/xilinx/xrt/python).
"""
import os
import re
import sys
import time

import numpy as np

if os.name == "nt":
    sys.path.insert(0, os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "AMD"))
else:
    sys.path.insert(0, os.path.join(os.environ.get("XILINX_XRT", "/opt/xilinx/xrt"), "python"))
import pyxrt  # noqa: E402

KDIR = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1] if len(sys.argv) > 1 else "mm_bf16_f32_M512_K512_N128"
M, K, N = map(int, re.match(r"mm_bf16_f32_M(\d+)_K(\d+)_N(\d+)", name).groups())
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def f32_to_bf16_bits(x):
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16          # round-to-nearest-even, as ggml does
    return u.astype(np.uint16)


def bf16_bits_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


insts = np.fromfile(os.path.join(KDIR, name + ".insts.bin"), dtype=np.uint32)
print(f"kernel {name}: M={M} K={K} N={N}, {len(insts)} instruction words")

dev = pyxrt.device(0)
print("device:", dev.get_info(pyxrt.xrt_info_device.name))
xclbin = pyxrt.xclbin(os.path.join(KDIR, name + ".xclbin"))
kname = [k.get_name() for k in xclbin.get_kernels() if k.get_name().startswith("MLIR_AIE")][0]
dev.register_xclbin(xclbin)
ctx = pyxrt.hw_context(dev, xclbin.get_uuid())
kernel = pyxrt.kernel(ctx, kname)

# argument layout of an IRON runtime sequence: (opcode, instr, n_instr, A, B, C)
bo_instr = pyxrt.bo(dev, insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
bo_a = pyxrt.bo(dev, M * K * 2, pyxrt.bo.host_only, kernel.group_id(3))
bo_b = pyxrt.bo(dev, N * K * 2, pyxrt.bo.host_only, kernel.group_id(4))
bo_c = pyxrt.bo(dev, N * M * 4, pyxrt.bo.host_only, kernel.group_id(5))

rng = np.random.default_rng(0)
A = rng.standard_normal((M, K)).astype(np.float32)   # weight rows
B = rng.standard_normal((N, K)).astype(np.float32)   # activation rows
Ab, Bb = f32_to_bf16_bits(A), f32_to_bf16_bits(B)
ref = bf16_bits_to_f32(Bb) @ bf16_bits_to_f32(Ab).T   # [N x M] = C[t][r]

bo_instr.write(insts.tobytes(), 0)
bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
bo_a.write(Ab.tobytes(), 0)
bo_b.write(Bb.tobytes(), 0)
bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
bo_b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

opcode = 3
times = []
for _ in range(iters):
    t0 = time.perf_counter()
    run = kernel(opcode, bo_instr, len(insts), bo_a, bo_b, bo_c)
    state = run.wait()
    times.append(time.perf_counter() - t0)
    if "COMPLETED" not in str(state) and int(state) != 4:
        print("run state:", state)

bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
C = np.frombuffer(bo_c.read(N * M * 4, 0), dtype=np.float32).reshape(N, M)

err = np.abs(C - ref)
nmse = float((err ** 2).sum() / ((ref ** 2).sum() + 1e-9))
t = float(np.median(times))
print(f"max abs err {err.max():.4g}  NMSE {nmse:.3g}")
print(f"median launch {t*1e6:.0f} us  ({2.0*M*K*N / t / 1e9:.1f} GFLOPS incl. dispatch; min {min(times)*1e6:.0f} us)")
print("RESULT:", "PASS" if nmse < 1e-4 else "FAIL")
sys.exit(0 if nmse < 1e-4 else 1)
