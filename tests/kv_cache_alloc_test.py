import sys, ctypes, os
hip = None
for _p in [r"C:\Program Files\AMD\ROCm.1inmdhip64.dll"]:
    if os.path.exists(_p):
        try: hip = ctypes.CDLL(_p); break
        except OSError: pass
if hip is None: print("[SKIP] HIP not found"); sys.exit(0)
def chk(r):
    e,*rest = r if hasattr(r,"__iter__") else (r,[])
    if int(e)!=0: raise RuntimeError("HIP err "+str(int(e)))
    return rest[0] if len(rest)==1 else rest
print("BEATEK_ROCm KV Alloc Test")
chk(hip.hipSetDevice(0))
e,ptr = hip.hipMalloc(512*1024*1024); chk(e)
print("  [PASS] hipMalloc 512MiB at", hex(int(ptr)))
chk(hip.hipFree(ptr))
e,stream = hip.hipStreamCreate(); chk(e)
e,kv = hip.hipMalloc(512*1024*1024); chk(e)
chk(hip.hipDeviceSynchronize())
chk(hip.hipStreamSynchronize(stream))
chk(hip.hipFree(kv))
chk(hip.hipStreamDestroy(stream))
print("  [PASS] Stream sync after alloc succeeded")
print("ALL TESTS PASSED")
