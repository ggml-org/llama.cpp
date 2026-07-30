"""Thin NumPy bridge for Tessera's Apple Accelerate CPU kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "apple_accelerate.cpp"
_DEFAULT_LIBRARY = _ROOT / ".build" / "libtessera_accelerate.dylib"


def build_library(output: Path = _DEFAULT_LIBRARY) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "xcrun",
            "clang++",
            "-std=c++17",
            "-O3",
            "-dynamiclib",
            str(_SOURCE),
            "-framework",
            "Accelerate",
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


class AccelerateBackend:
    def __init__(self, library: Path | None = None, build: bool = True):
        path = Path(
            library
            or os.environ.get("TESSERA_ACCELERATE_LIBRARY", _DEFAULT_LIBRARY)
        )
        if build and (
            not path.exists()
            or path.stat().st_mtime < _SOURCE.stat().st_mtime
        ):
            build_library(path)
        self._library = ctypes.CDLL(str(path))
        pointer = ctypes.POINTER(ctypes.c_float)
        i8_pointer = ctypes.POINTER(ctypes.c_int8)
        self._error = self._library.tessera_accelerate_weighted_square_error
        self._error.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self._error.restype = ctypes.c_int
        self._lanes = self._library.tessera_accelerate_lane_targets
        self._lanes.argtypes = [
            pointer,
            i8_pointer,
            pointer,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self._lanes.restype = ctypes.c_int

    @staticmethod
    def _f32(values: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(values, dtype=np.float32)

    def weighted_square_error(
        self,
        weights: np.ndarray,
        reconstructed: np.ndarray,
        importance: np.ndarray | None,
    ) -> np.ndarray:
        weights_f32 = self._f32(weights)
        reconstructed_f32 = self._f32(reconstructed)
        if weights_f32.shape != reconstructed_f32.shape or weights_f32.ndim != 2:
            raise ValueError("weights and reconstructed must be matching matrices")
        importance_squared = (
            self._f32(np.square(importance, dtype=np.float32))
            if importance is not None
            else None
        )
        if importance_squared is not None and importance_squared.shape != (
            weights_f32.shape[1],
        ):
            raise ValueError("importance width does not match matrix width")
        output = np.empty_like(weights_f32)
        pointer = ctypes.POINTER(ctypes.c_float)
        status = self._error(
            weights_f32.ctypes.data_as(pointer),
            reconstructed_f32.ctypes.data_as(pointer),
            (
                importance_squared.ctypes.data_as(pointer)
                if importance_squared is not None
                else None
            ),
            output.ctypes.data_as(pointer),
            weights_f32.shape[0],
            weights_f32.shape[1],
        )
        if status != 0:
            raise RuntimeError(f"Accelerate error kernel failed with {status}")
        return output

    def lane_targets(
        self,
        weights: np.ndarray,
        ternary: np.ndarray,
        lane_width: int = 20,
    ) -> np.ndarray:
        weights_f32 = self._f32(weights)
        ternary_i8 = np.ascontiguousarray(ternary, dtype=np.int8)
        if weights_f32.shape != ternary_i8.shape or weights_f32.ndim != 2:
            raise ValueError("weights and ternary must be matching matrices")
        if weights_f32.shape[1] % lane_width:
            raise ValueError("matrix width must be divisible by lane width")
        output = np.empty(
            (weights_f32.shape[0], weights_f32.shape[1] // lane_width),
            dtype=np.float32,
        )
        pointer = ctypes.POINTER(ctypes.c_float)
        status = self._lanes(
            weights_f32.ctypes.data_as(pointer),
            ternary_i8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            output.ctypes.data_as(pointer),
            weights_f32.shape[0],
            weights_f32.shape[1],
            lane_width,
        )
        if status != 0:
            raise RuntimeError(f"Accelerate lane kernel failed with {status}")
        return output
