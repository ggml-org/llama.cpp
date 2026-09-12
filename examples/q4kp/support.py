"""Load only the synthetic test library built by this checkout."""
import ctypes
import os
from pathlib import Path
import unittest

_dll_directories = []

def library():
    path = Path(os.environ["Q4KP_TEST_LIBRARY"]).resolve()
    if hasattr(os, "add_dll_directory"):
        # Python does not search PATH for transitive DLL dependencies.
        for directory in (path.parent, Path(os.environ.get("Q4KP_TEST_RUNTIME_DIR", path.parent))):
            _dll_directories.append(os.add_dll_directory(str(directory)))
    dll = ctypes.CDLL(str(path))
    if not dll.q4kp_test_cpu_supported():
        raise unittest.SkipTest("AVX2, BMI2, FMA and F16C required")
    dll.q4kp_recode.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    dll.q4kp_recode.restype = ctypes.c_int
    args = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    for name in ("q4kp_gemv", "q4kp_gemm", "q4kp_original_gemv", "q4kp_original_gemm",
                 "q4kp_scalar_gemm", "q4kp_vnni_gemv", "q4kp_wide_gemv"):
        fn = getattr(dll, name)
        fn.argtypes, fn.restype = args, None
    dll.q4kp_scalar_gemv.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_int]
    dll.q4kp_scalar_gemv.restype = None
    return dll
