#!/usr/bin/env python3
"""
l3_sidecar_v3_reader.py

Tessera L1 / L1.5 dequant sidecar reader (v1 / v2 / v3 schema).

Reads a `.dequant.f32` (L1) or `.act.dequant.f32` (L1.5) file produced
by the runtime dequant instrumentation in ggml-cpu / ggml-cuda /
ggml-metal. Dispatches on the file version field so the same reader
handles v1, v2, and v3 files. Reads the optional companion
`.provenance.json` sidecar written next to the data file.

Schema (v3, all multi-byte fields little-endian):

    offset              size   field
    ------              ----   --------------------------------------
          0                4   magic              = b"TDQT"
          4                4   version            = uint32 (= 1, 2, 3)
          8                8   rows               = int64
         16                8   cols               = int64
         24                4   dtype              = uint32 (0 = F32)
    -- end v1 header (28 bytes) ---------------------------------------
         28                4   outlier_threshold  = float32   (v2+)
         32                8   outlier_count_total= int64      (v2+)
    -- end v2 header (40 bytes) ---------------------------------------
         40           R * 4   row_outlier_count   = int32 per row (v2+)
    -- end v2 per-row strip ------------------------------------------
    40 + R*4           R * 24   row_v3_meta (24 bytes per row, v3+):
                           8       timing_ns            uint64
                           4       kernel_id            uint32
                           4       dispatch_count       uint32
                           8       reserved             uint64
    -- end v3 per-row strip ------------------------------------------
    40 + R*4 + R*24   R * C*4   data                = F32 row-major

Reader behavior:
  - v1 file: data at offset 28; v2 and v3 fields are absent and
    reported as None (or zero, see `mode` arg).
  - v2 file: v2 fields read, v3 fields absent and reported as None.
  - v3 file: all fields read.

Usage:

    from l3_sidecar_v3_reader import read_sidecar, read_provenance

    sidecar = read_sidecar("path/to/<name>.dequant.f32")
    print(sidecar["version"], sidecar["rows"], sidecar["cols"])
    if sidecar["version"] >= 2:
        print("total outliers:", sidecar["outlier_count_total"])
        print("per-row:", sidecar["row_outlier_count"][:5])
    if sidecar["version"] >= 3:
        print("per-row timing[0:3] (ns):", sidecar["row_timing_ns"][:3])
        print("kernel_id[0:3]:", sidecar["row_kernel_id"][:3])
    data = sidecar["data"]    # numpy.ndarray shape (rows, cols), dtype float32

    prov = read_provenance(sidecar["provenance_path"])
    print(prov["kernel_version"], prov["tessera_main_tip"], prov["created_at"])
"""

import argparse
import json
import os
import struct
import sys

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

MAGIC = b"TDQT"
DTYPE_F32 = 0
DTYPE_F16 = 1
DTYPE_BF16 = 2

DTYPE_NAME = {DTYPE_F32: "F32", DTYPE_F16: "F16", DTYPE_BF16: "BF16"}


def _err(msg, *args):
    sys.stderr.write("l3_sidecar_v3_reader: " + (msg % args) + "\n")


class SidecarReadError(Exception):
    pass


def _read_exact(f, n):
    buf = f.read(n)
    if len(buf) != n:
        raise SidecarReadError(
            "short read: wanted %d bytes, got %d" % (n, len(buf)))
    return buf


def _read_header(f):
    """Read magic + version + rows + cols + dtype. Returns a dict."""
    head = _read_exact(f, 28)
    magic = head[0:4]
    if magic != MAGIC:
        raise SidecarReadError(
            "bad magic: %r (want %r)" % (magic, MAGIC))
    version, = struct.unpack("<I", head[4:8])
    rows,    = struct.unpack("<q", head[8:16])
    cols,    = struct.unpack("<q", head[16:24])
    dtype,   = struct.unpack("<I", head[24:28])
    if dtype not in DTYPE_NAME:
        raise SidecarReadError("unknown dtype: %d" % dtype)
    if rows < 0 or cols < 0:
        raise SidecarReadError(
            "negative shape: rows=%d cols=%d" % (rows, cols))
    return {
        "magic":   magic,
        "version": int(version),
        "rows":    int(rows),
        "cols":    int(cols),
        "dtype":   int(dtype),
    }


def _read_v2_header(f):
    """Read the v2 header additions (threshold + total) at offset 28."""
    blob = _read_exact(f, 12)
    threshold, = struct.unpack("<f", blob[0:4])
    total,     = struct.unpack("<q", blob[4:12])
    return threshold, total


def _read_v2_strip(f, rows):
    """Read the v2 per-row outlier_count strip at offset 40."""
    nbytes = rows * 4
    if nbytes == 0:
        return []
    blob = _read_exact(f, nbytes)
    return list(struct.unpack("<%di" % rows, blob))


def _read_v3_strip(f, rows):
    """Read the v3 per-row meta strip (24 bytes per row). Returns
    (timing_ns, kernel_id, dispatch_count, reserved) as lists."""
    timing_ns      = []
    kernel_id      = []
    dispatch_count = []
    reserved       = []
    if rows == 0:
        return timing_ns, kernel_id, dispatch_count, reserved
    nbytes = rows * 24
    blob = _read_exact(f, nbytes)
    for i in range(rows):
        o = i * 24
        t_ns,  = struct.unpack("<Q", blob[o + 0:o + 8])
        k_id,  = struct.unpack("<I", blob[o + 8:o + 12])
        d_cnt, = struct.unpack("<I", blob[o + 12:o + 16])
        rsv,   = struct.unpack("<Q", blob[o + 16:o + 24])
        timing_ns.append(int(t_ns))
        kernel_id.append(int(k_id))
        dispatch_count.append(int(d_cnt))
        reserved.append(int(rsv))
    return timing_ns, kernel_id, dispatch_count, reserved


def _read_data(f, rows, cols, dtype):
    """Read the F32 data block (assumes dtype == F32)."""
    if dtype != DTYPE_F32:
        raise SidecarReadError(
            "dtype %s not yet supported by this reader" % DTYPE_NAME[dtype])
    n_els = rows * cols
    if n_els == 0:
        if HAVE_NUMPY:
            return np.zeros((rows, cols), dtype=np.float32)
        return []
    nbytes = n_els * 4
    blob = _read_exact(f, nbytes)
    if HAVE_NUMPY:
        arr = np.frombuffer(blob, dtype="<f4", count=n_els)
        return arr.reshape(rows, cols).copy()
    # Fall back to a flat list of Python floats.
    return list(struct.unpack("<%df" % n_els, blob))


def _resolve_provenance_path(data_path):
    """Return the conventional path for the provenance sidecar."""
    return data_path + ".provenance.json"


def read_sidecar(path, mode="auto", provenance=True):
    """Read a sidecar file. `mode` is one of:
        - "auto":  dispatch on the file's version field (default)
        - "v1":    read a v1 file (data at offset 28), even if the
                   file is actually v2 or v3 (in which case the
                   caller gets garbage as the F32 data -- this is
                   the "naive v1 reader" path used by the backward-
                   compat test)
        - "v2":    read a v2 file (data at offset 40 + R*4); the v3
                   per-row strip is treated as opaque padding
        - "v3":    read a v3 file (data at offset 40 + R*28)
    Returns a dict; see the module docstring.

    Backward-compat contract: a v3 reader (mode="v3") reading a v1 or
    v2 file produces zeros for the missing v2/v3 fields (not None).
    This is what the spec asks for: "a v3 reader reading a v1 file
    should produce zeros for the v2/v3 fields".
    """
    if mode not in ("auto", "v1", "v2", "v3"):
        raise ValueError("mode must be one of auto|v1|v2|v3")
    with open(path, "rb") as f:
        h = _read_header(f)
        actual_version = h["version"]
        rows = h["rows"]
        cols = h["cols"]
        dtype = h["dtype"]

        # Decide which version the caller wants to use.
        if mode == "auto":
            want_version = actual_version
        else:
            want_version = {"v1": 1, "v2": 2, "v3": 3}[mode]

        # Fields that are populated only for v2+. Initialize to zero
        # so a v1 file read by a v3 reader returns zeros (the
        # backward-compat contract).
        threshold = 0.0
        outlier_count_total = 0
        row_outlier_count = [0] * rows
        # Fields that are populated only for v3. Same: zero-init.
        row_timing_ns      = [0] * rows
        row_kernel_id      = [0] * rows
        row_dispatch_count = [0] * rows
        row_reserved       = [0] * rows

        # The data offset depends on the version we are reading as.
        if want_version == 1:
            # v1 reader: data at offset 28. We don't read the v2/v3
            # fields -- they are absent in v1, and in v2/v3 files
            # they are opaque padding that the v1 reader is allowed
            # to skip.
            data_off = 28
        elif want_version == 2:
            # v2 reader: data at offset 40 + R*4. We read the v2
            # fields, then the v2 strip, and treat the v3 strip
            # (if present) as opaque padding.
            if actual_version >= 2:
                threshold, outlier_count_total = _read_v2_header(f)
                row_outlier_count = _read_v2_strip(f, rows)
            data_off = 40 + rows * 4
            if actual_version >= 3:
                # Skip the v3 per-row strip (treated as opaque).
                _read_exact(f, rows * 24)
        elif want_version == 3:
            # v3 reader: data at offset 40 + R*4 + R*24. We read the
            # v2 fields, the v2 strip, and the v3 strip.
            if actual_version >= 2:
                threshold, outlier_count_total = _read_v2_header(f)
                row_outlier_count = _read_v2_strip(f, rows)
            if actual_version >= 3:
                (row_timing_ns, row_kernel_id,
                 row_dispatch_count, row_reserved) = _read_v3_strip(f, rows)
            data_off = 40 + rows * 4 + rows * 24
        else:
            raise SidecarReadError("unsupported want_version: %d" % want_version)

        # Seek to the data offset and read.
        f.seek(data_off)
        data = _read_data(f, rows, cols, dtype)

        # The file's actual version is exposed so the caller can
        # dispatch on it (e.g., a v3 reader reading a v1 file knows
        # the v2/v3 fields are zeros, not data).
        result = {
            "path":              path,
            "magic":             h["magic"],
            "version":           int(want_version),  # the version we read as
            "actual_version":    int(actual_version),
            "rows":              int(rows),
            "cols":              int(cols),
            "dtype":             int(dtype),
            "dtype_name":        DTYPE_NAME[dtype],
            "data":              data,
            "outlier_threshold": threshold,
            "outlier_count_total": outlier_count_total,
            "row_outlier_count": row_outlier_count,
            "row_timing_ns":     row_timing_ns,
            "row_kernel_id":     row_kernel_id,
            "row_dispatch_count": row_dispatch_count,
            "row_reserved":      row_reserved,
        }
        if provenance:
            prov_path = _resolve_provenance_path(path)
            if os.path.exists(prov_path):
                result["provenance_path"] = prov_path
                try:
                    with open(prov_path, "r") as pf:
                        result["provenance"] = json.load(pf)
                except (OSError, ValueError) as e:
                    _err("failed to read provenance %s: %s", prov_path, e)
                    result["provenance"] = None
            else:
                result["provenance"] = None
        return result


def read_provenance(path):
    """Read a `.provenance.json` sidecar. Returns a dict; raises on error."""
    with open(path, "r") as f:
        return json.load(f)


def _format_summary(s):
    """Format a one-line summary of a sidecar read result."""
    parts = [
        "version=%d" % s["version"],
        "actual_version=%d" % s["actual_version"],
        "shape=(%d,%d)" % (s["rows"], s["cols"]),
        "dtype=%s" % s["dtype_name"],
    ]
    if s["version"] >= 2:
        parts.append("threshold=%.4f" % s["outlier_threshold"])
        parts.append("outliers_total=%d" % s["outlier_count_total"])
    if s["version"] >= 3 and s["row_timing_ns"]:
        total = sum(s["row_timing_ns"])
        parts.append("timing_total_ns=%d" % total)
    if s.get("provenance"):
        p = s["provenance"]
        parts.append("kernel=%s" % p.get("kernel_version", "?"))
        parts.append("main_tip=%s" % p.get("tessera_main_tip", "?"))
        parts.append("created_at=%s" % p.get("created_at", "?"))
    return " ".join(parts)


def _main(argv):
    p = argparse.ArgumentParser(
        description="Read a Tessera L1 / L1.5 dequant sidecar.")
    p.add_argument("path", help="path to <name>.dequant.f32 (or .act.dequant.f32)")
    p.add_argument("--mode", default="auto", choices=("auto", "v1", "v2", "v3"),
                   help="read the file as this version (default: auto)")
    p.add_argument("--no-data", action="store_true",
                   help="skip reading the F32 data block (faster, just header/strips)")
    p.add_argument("--no-provenance", action="store_true",
                   help="skip reading the .provenance.json sidecar")
    p.add_argument("--summary", action="store_true",
                   help="print a one-line summary and exit")
    args = p.parse_args(argv)

    sidecar = read_sidecar(
        args.path, mode=args.mode, provenance=not args.no_provenance)
    if args.no_data:
        sidecar["data"] = None
    if args.summary:
        print(_format_summary(sidecar))
        return 0
    # Otherwise print the full structure as a Python repr.
    import pprint
    safe = {k: v for k, v in sidecar.items() if k != "data"}
    if sidecar["data"] is None or not HAVE_NUMPY:
        if isinstance(sidecar["data"], list) and len(sidecar["data"]) > 32:
            safe["data"] = sidecar["data"][:32] + ["...(truncated)"]
        else:
            safe["data"] = sidecar["data"]
    else:
        arr = sidecar["data"]
        safe["data_shape"] = arr.shape
        safe["data_dtype"] = str(arr.dtype)
        safe["data_first_8"] = arr.reshape(-1)[:8].tolist()
    pprint.pprint(safe, width=120)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
