"""Estimate per-layer HIGGS alpha_l from a GGUF model (thin C++ wrapper).

Phase 3.5 of the iPhone ANE demo: this module is a thin wrapper
around the C++ binary ``tessera-higgs-proxy``. The same C++ module
that powers the rest of the L5 / calibration / imatrix path now
produces the alpha sidecar too. The NumPy implementation at
``tools/ane-mtp/estimate_higgs_alpha.py`` is kept as the dev /
test fallback when the C++ binary is not built.

The math is the HIGGS Linearity Theorem (Malinovskii et al.,
arXiv:2411.17525, NAACL 2025). The C++ proxy is the
"ranking-grade cross-check" for Algorithm 3 (the perturbation
sweep). The two estimators agree on the layer ranking (K/V
high, FFN low) by construction. See
``docs/tessera-higgs-estimator.md`` and the architect's
research spine at ``docs/research-higgs-2026-07-30.md``.

Pipeline
--------

1. If the C++ binary ``tessera-higgs-proxy`` is on ``PATH``
   (look-up via ``shutil.which``), subprocess it with the
   same CLI surface as the original NumPy script.
2. If the binary is NOT on PATH (e.g. a dev environment without
   a C++ build), fall back to the in-process NumPy path.
   The fallback is logged as a WARNING so the operator sees
   that the sidecar was produced by the slower, dev-only
   path, and the sidecar's ``measurement`` field is stamped
   ``offline_ternary_mse_numpy_fallback`` (the C++ binary
   stamps ``offline_ternary_mse`` or ``uniform_fallback`` or
   ``l1_kernel_dequant``; the ``_numpy_fallback`` value is
   unique to this fallback path so a future consumer can tell
   the two paths apart).
3. The sidecar JSON shape (``ane.alpha-coefficients.v1``) is
   identical across both paths; a sidecar produced by either
   path is interchangeable at the byte level modulo a
   documented float-repr tolerance.

Usage
-----

    python3 estimate_higgs_alpha.py \\
        --gguf /path/to/model.gguf \\
        --output /path/to/model.alpha-coefficients.v1.json

The sidecar is the wire format between this estimator and the
iOS app's ANE dispatch (Phase 2's streaming layer).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Sequence

# The CLI surface is unchanged from the original NumPy script
# (Phase 3 of the iPhone ANE demo). Only the implementation
# behind it is new.

SIDECAR_SCHEMA = "ane.alpha-coefficients.v1"
DEFAULT_MIN_PARAMS_FOR_ESTIMATE = 1_000_000_000
ALPHA_FLOOR_FRACTION_OF_MEAN = 1.0e-3

# The C++ binary name. The Python wrapper looks it up via
# shutil.which(); if found on PATH, the subprocess path is
# taken. Otherwise, the in-process NumPy fallback runs.
CPP_BINARY_NAME = "tessera-higgs-proxy"

# The t_squared_source enum values the C++ binary stamps.
# The Python wrapper stamps ``offline_ternary_mse_numpy_fallback``
# for the in-process fallback so the consumer can tell the two
# paths apart. The four canonical values are documented in
# docs/tessera-higgs-estimator.md.
TSQ_SOURCE_CPP_DEFAULT = "offline_ternary_mse"
TSQ_SOURCE_CPP_FALLBACK = "uniform_fallback"
TSQ_SOURCE_CPP_L1 = "l1_kernel_dequant"
TSQ_SOURCE_PYTHON_FALLBACK = "offline_ternary_mse_numpy_fallback"

logger = logging.getLogger("tessera.estimate_higgs_alpha")


def _find_cpp_binary() -> str | None:
    """Locate the C++ ``tessera-higgs-proxy`` binary on PATH.

    Returns the absolute path to the binary, or None if the
    binary is not on PATH. The C++ binary is the production
    path; the NumPy fallback is the dev / test path.
    """
    return shutil.which(CPP_BINARY_NAME)


def _run_cpp_binary(gguf_path: Path, output_path: Path, *,
                    report_path: Path | None = None,
                    bundle_name: str | None = None,
                    min_params_for_estimate: int = DEFAULT_MIN_PARAMS_FOR_ESTIMATE,
                    alpha_floor_fraction: float = ALPHA_FLOOR_FRACTION_OF_MEAN,
                    verbose: bool = False) -> int:
    """Subprocess the C++ binary. Returns the exit code.

    The C++ binary writes the sidecar and (optionally) the
    report itself; the wrapper just captures stdout / stderr
    and the exit code. On non-zero exit, the caller falls
    back to the in-process NumPy path so a misconfigured
    build does not silently lose the sidecar.
    """
    bin_path = _find_cpp_binary()
    if bin_path is None:
        raise FileNotFoundError(
            f"{CPP_BINARY_NAME} not on PATH; cannot use the C++ path")

    cmd = [bin_path,
           "--gguf", str(gguf_path),
           "--output", str(output_path),
           "--min-params-for-estimate", str(min_params_for_estimate),
           "--alpha-floor-fraction", str(alpha_floor_fraction)]
    if report_path is not None:
        cmd += ["--report", str(report_path)]
    if bundle_name is not None:
        cmd += ["--bundle-name", bundle_name]
    if verbose:
        cmd += ["--verbose"]

    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        # shutil.which returned a path but exec failed (very rare,
        # e.g. permission denied or the binary is corrupt).
        logger.error("C++ binary %s could not be executed: %s", bin_path, exc)
        return 1

    # The C++ binary writes its run summary to stderr; surface it
    # so the wrapper's caller sees the same diagnostics as a
    # direct C++ invocation.
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return proc.returncode


def _stamp_numpy_fallback_measurement(sidecar_path: Path) -> None:
    """Replace the ``measurement`` field in the sidecar with
    ``offline_ternary_mse_numpy_fallback`` to flag the sidecar
    as produced by the in-process NumPy path.

    The NumPy path stamps the same value as the C++ default
    (``offline_ternary_mse``), which is correct for the math
    but loses the producer identity. The wrapper post-processes
    the sidecar to stamp the producer so a future consumer
    can audit the path.
    """
    import json
    with sidecar_path.open("r", encoding="utf-8") as f:
        sidecar = json.load(f)
    sidecar["measurement"] = TSQ_SOURCE_PYTHON_FALLBACK
    # Per-tensor t_squared_source: leave as-is (the NumPy
    # estimator stamps "offline_ternary_mse" per tensor, which
    # is also true for the fallback path).
    with sidecar_path.open("w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
        f.write("\n")


def _run_numpy_fallback(gguf_path: Path, output_path: Path, *,
                        report_path: Path | None = None,
                        bundle_name: str | None = None,
                        min_params_for_estimate: int = DEFAULT_MIN_PARAMS_FOR_ESTIMATE,
                        alpha_floor_fraction: float = ALPHA_FLOOR_FRACTION_OF_MEAN,
                        verbose: bool = False) -> int:
    """Run the in-process NumPy path. The dev / test fallback.

    Imports ``tools/ane-mtp/estimate_higgs_alpha.py`` and
    calls its ``main()`` with the same argv the wrapper
    would have built. Stamps the sidecar with the
    ``offline_ternary_mse_numpy_fallback`` discriminator
    after the run.
    """
    warnings.warn(
        "C++ binary not found, using in-process NumPy fallback",
        RuntimeWarning,
        stacklevel=2,
    )
    logger.warning(
        "C++ binary %s not found, using in-process NumPy fallback",
        CPP_BINARY_NAME,
    )

    ane_mtp = Path(__file__).resolve().parent.parent / "ane-mtp"
    numpy_path = ane_mtp / "estimate_higgs_alpha.py"
    if not numpy_path.is_file():
        logger.error(
            "NumPy fallback unavailable: %s not found. The C++ "
            "binary is not on PATH and the dev-fallback NumPy "
            "module is missing; install one or build the C++ "
            "target (tessera-higgs-proxy).",
            numpy_path,
        )
        return 1

    argv = [
        "--gguf", str(gguf_path),
        "--output", str(output_path),
        "--min-params-for-estimate", str(min_params_for_estimate),
        "--alpha-floor-fraction", str(alpha_floor_fraction),
    ]
    if report_path is not None:
        argv += ["--report", str(report_path)]
    if bundle_name is not None:
        argv += ["--bundle-name", bundle_name]
    if verbose:
        argv += ["--verbose"]

    # The NumPy module is at tools/ane-mtp/estimate_higgs_alpha.py;
    # this wrapper is at tools/tessera/estimate_higgs_alpha.py. Both
    # share the same module name, so a plain `import` would resolve
    # to this file. Use importlib to load the NumPy module by its
    # absolute path under a unique module name.
    import importlib.util
    numpy_module_name = "_tessera_higgs_alpha_numpy_fallback"
    if numpy_module_name in sys.modules:
        numpy_impl = sys.modules[numpy_module_name]
    else:
        spec = importlib.util.spec_from_file_location(
            numpy_module_name, str(numpy_path))
        numpy_impl = importlib.util.module_from_spec(spec)
        sys.modules[numpy_module_name] = numpy_impl
        spec.loader.exec_module(numpy_impl)

    rc = numpy_impl.main(argv)
    if rc != 0:
        return rc
    _stamp_numpy_fallback_measurement(output_path)
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="estimate_higgs_alpha",
        description=(
            "Estimate per-layer HIGGS alpha_l from a GGUF model. "
            "The estimator is L1-agnostic: today it uses the offline "
            "ternary MSE as the t_l^2 proxy; a future swap to the L1 "
            "kernel-dequant output is a one-function-call change. "
            "This wrapper prefers the C++ binary on PATH and falls "
            "back to the in-process NumPy path otherwise. See "
            "docs/tessera-higgs-estimator.md for the math and the "
            "sidecar JSON shape."),
    )
    parser.add_argument("--gguf", type=Path, required=True,
                        help="path to the source GGUF (the model the "
                             "alpha is being estimated for). The sidecar "
                             "is keyed off this file's content hash.")
    parser.add_argument("--output", type=Path, required=True,
                        help="output path for the alpha-coefficients "
                             "sidecar JSON. The conventional name is "
                             "<bundle>.alpha-coefficients.v1.json.")
    parser.add_argument("--report", type=Path, default=None,
                        help="optional path for a human-readable "
                             "markdown report. Default: alongside the "
                             "sidecar as <sidecar-stem>.report.md.")
    parser.add_argument("--bundle-name", type=str, default=None,
                        help="override the bundle name in the sidecar "
                             "(default: the .gguf file's stem).")
    parser.add_argument("--min-params-for-estimate", type=int,
                        default=DEFAULT_MIN_PARAMS_FOR_ESTIMATE,
                        help="parameter count threshold below which the "
                             "estimator returns uniform alpha (default: "
                             "1B, the architect's design-doc gate).")
    parser.add_argument("--alpha-floor-fraction", type=float,
                        default=ALPHA_FLOOR_FRACTION_OF_MEAN,
                        help="positive floor on alpha as a fraction of "
                             "the post-normalization mean (default: 1e-3).")
    parser.add_argument("--verbose", action="store_true",
                        help="print a one-line summary per tensor (off "
                             "by default; the sidecar is the durable "
                             "record).")
    parser.add_argument("--json-report", type=Path, default=None,
                        help="(deprecated) alias for --output. Kept for "
                             "backward compatibility with the original "
                             "NumPy CLI surface.")
    args = parser.parse_args(argv)
    if not args.gguf.is_file():
        parser.error(f"GGUF not found: {args.gguf}")
    if args.min_params_for_estimate < 0:
        parser.error("--min-params-for-estimate must be >= 0")
    if not (0.0 < args.alpha_floor_fraction < 1.0):
        parser.error("--alpha-floor-fraction must be in (0, 1)")
    if args.json_report is not None and args.output is None:
        args.output = args.json_report
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")

    cpp_bin = _find_cpp_binary()
    if cpp_bin is not None:
        rc = _run_cpp_binary(
            args.gguf, args.output,
            report_path=args.report,
            bundle_name=args.bundle_name,
            min_params_for_estimate=args.min_params_for_estimate,
            alpha_floor_fraction=args.alpha_floor_fraction,
            verbose=args.verbose,
        )
        if rc == 0:
            return 0
        # Non-zero exit: fall through to the NumPy path. This
        # handles the case where the binary is on PATH but
        # crashed (e.g. the GGUF is corrupt). A warning makes
        # the fallback visible.
        logger.warning(
            "C++ binary %s exited with rc=%d; falling back to "
            "in-process NumPy path", cpp_bin, rc)
    else:
        logger.info(
            "C++ binary %s not on PATH; using in-process NumPy "
            "fallback", CPP_BINARY_NAME)

    return _run_numpy_fallback(
        args.gguf, args.output,
        report_path=args.report,
        bundle_name=args.bundle_name,
        min_params_for_estimate=args.min_params_for_estimate,
        alpha_floor_fraction=args.alpha_floor_fraction,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
