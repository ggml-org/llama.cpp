"""NumPy bridge for Tessera's fixed-shape multifunction ANE quantizer asset."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "apple_ane_quantizer.mm"
_DEFAULT_LIBRARY = _ROOT / ".build" / "libtessera_ane_quantizer.dylib"
_VALID_ROWS = (64, 256, 1024)


def build_library(output: Path = _DEFAULT_LIBRARY) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "xcrun",
            "clang++",
            "-std=c++17",
            "-O3",
            "-dynamiclib",
            "-fobjc-arc",
            str(_SOURCE),
            "-framework",
            "CoreML",
            "-framework",
            "Foundation",
            "-framework",
            "Accelerate",
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


class ANEQuantizerBackend:
    def __init__(
        self,
        model: Path,
        library: Path | None = None,
        build: bool = True,
    ):
        model = Path(model)
        if model.suffix != ".mlmodelc" or not model.is_dir():
            raise ValueError(f"expected a compiled .mlmodelc directory: {model}")
        path = Path(
            library or os.environ.get("TESSERA_ANE_LIBRARY", _DEFAULT_LIBRARY)
        )
        if build and (
            not path.exists() or path.stat().st_mtime < _SOURCE.stat().st_mtime
        ):
            build_library(path)
        self._library = ctypes.CDLL(str(path))
        self._library.tessera_ane_create.argtypes = [ctypes.c_char_p]
        self._library.tessera_ane_create.restype = ctypes.c_void_p
        self._library.tessera_ane_destroy.argtypes = [ctypes.c_void_p]
        self._library.tessera_ane_last_error.argtypes = [ctypes.c_void_p]
        self._library.tessera_ane_last_error.restype = ctypes.c_char_p
        half = ctypes.POINTER(ctypes.c_uint16)
        self._lane = self._library.tessera_ane_lane_targets
        self._lane.argtypes = [
            ctypes.c_void_p, half, half, half, ctypes.c_size_t
        ]
        self._lane.restype = ctypes.c_int
        self._residual = self._library.tessera_ane_residual_score
        self._residual.argtypes = [
            ctypes.c_void_p, half, half, half, half, half, ctypes.c_size_t
        ]
        self._residual.restype = ctypes.c_int
        f32 = ctypes.POINTER(ctypes.c_float)
        self._lane_exact = (
            self._library.tessera_coreml_lane_targets_exact
        )
        self._lane_exact.argtypes = [
            ctypes.c_void_p, f32, f32, f32, ctypes.c_size_t
        ]
        self._lane_exact.restype = ctypes.c_int
        self._residual_exact = (
            self._library.tessera_coreml_residual_score_exact
        )
        self._residual_exact.argtypes = [
            ctypes.c_void_p, f32, f32, f32, f32, f32, ctypes.c_size_t
        ]
        self._residual_exact.restype = ctypes.c_int
        self._handle = self._library.tessera_ane_create(
            os.fsencode(str(model.resolve()))
        )
        if not self._handle:
            raise RuntimeError("failed to create ANE quantizer backend")

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.tessera_ane_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def _f16(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        result = np.ascontiguousarray(values, dtype=np.float16)
        if result.shape != shape:
            raise ValueError(f"expected {shape}, got {result.shape}")
        return result

    @staticmethod
    def _pointer(values: np.ndarray):
        return values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))

    @staticmethod
    def _f32(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        result = np.ascontiguousarray(values, dtype=np.float32)
        if result.shape != shape:
            raise ValueError(f"expected {shape}, got {result.shape}")
        return result

    @staticmethod
    def _f32_pointer(values: np.ndarray):
        return values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def _check(self, status: int) -> None:
        if status:
            detail = self._library.tessera_ane_last_error(self._handle)
            raise RuntimeError(
                f"ANE quantizer failed with {status}: "
                f"{detail.decode(errors='replace') if detail else 'unknown error'}"
            )

    def lane_targets_ane(
        self, weights: np.ndarray, ternary: np.ndarray
    ) -> np.ndarray:
        if weights.ndim != 2 or weights.shape[1] != 640:
            raise ValueError("ANE pages must have shape [rows, 640]")
        rows = weights.shape[0]
        if rows not in _VALID_ROWS:
            raise ValueError(f"ANE row count must be one of {_VALID_ROWS}")
        weights_f16 = self._f16(weights, (rows, 640))
        ternary_f16 = self._f16(ternary, (rows, 640))
        output = np.empty((rows, 32), dtype=np.float16)
        self._check(self._lane(
            self._handle,
            self._pointer(weights_f16),
            self._pointer(ternary_f16),
            self._pointer(output),
            rows,
        ))
        return output.astype(np.float32)

    def lane_targets(
        self, weights: np.ndarray, ternary: np.ndarray
    ) -> np.ndarray:
        """Return the canonical FP32 result through the exact Core ML function."""
        if weights.ndim != 2 or weights.shape[1] != 640:
            raise ValueError("Core ML pages must have shape [rows, 640]")
        rows = weights.shape[0]
        if rows not in _VALID_ROWS:
            raise ValueError(f"Core ML row count must be one of {_VALID_ROWS}")
        weights_f32 = self._f32(weights, (rows, 640))
        ternary_f32 = self._f32(ternary, (rows, 640))
        output = np.empty((rows, 32), dtype=np.float32)
        self._check(self._lane_exact(
            self._handle,
            self._f32_pointer(weights_f32),
            self._f32_pointer(ternary_f32),
            self._f32_pointer(output),
            rows,
        ))
        return output

    def residual_score_ane(
        self,
        weights: np.ndarray,
        ternary: np.ndarray,
        lane_scale: np.ndarray,
        importance: np.ndarray,
    ) -> np.ndarray:
        if weights.ndim != 2 or weights.shape[1] != 640:
            raise ValueError("ANE pages must have shape [rows, 640]")
        rows = weights.shape[0]
        if rows not in _VALID_ROWS:
            raise ValueError(f"ANE row count must be one of {_VALID_ROWS}")
        weights_f16 = self._f16(weights, (rows, 640))
        ternary_f16 = self._f16(ternary, (rows, 640))
        lane_f16 = self._f16(lane_scale, (rows, 32))
        importance_f16 = self._f16(importance, (640,))
        output = np.empty((rows, 640), dtype=np.float16)
        self._check(self._residual(
            self._handle,
            self._pointer(weights_f16),
            self._pointer(ternary_f16),
            self._pointer(lane_f16),
            self._pointer(importance_f16),
            self._pointer(output),
            rows,
        ))
        return output.astype(np.float32)

    def residual_score(
        self,
        weights: np.ndarray,
        ternary: np.ndarray,
        lane_scale: np.ndarray,
        importance: np.ndarray,
    ) -> np.ndarray:
        """Return the canonical FP32 result through the exact Core ML function."""
        if weights.ndim != 2 or weights.shape[1] != 640:
            raise ValueError("Core ML pages must have shape [rows, 640]")
        rows = weights.shape[0]
        if rows not in _VALID_ROWS:
            raise ValueError(f"Core ML row count must be one of {_VALID_ROWS}")
        weights_f32 = self._f32(weights, (rows, 640))
        ternary_f32 = self._f32(ternary, (rows, 640))
        lane_f32 = self._f32(lane_scale, (rows, 32))
        importance_f32 = self._f32(importance, (640,))
        output = np.empty((rows, 640), dtype=np.float32)
        self._check(self._residual_exact(
            self._handle,
            self._f32_pointer(weights_f32),
            self._f32_pointer(ternary_f32),
            self._f32_pointer(lane_f32),
            self._f32_pointer(importance_f32),
            self._f32_pointer(output),
            rows,
        ))
        return output
