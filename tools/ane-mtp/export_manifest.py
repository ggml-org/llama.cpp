#!/usr/bin/env python3
"""Emit the ane_state_layout.v1 manifest sidecar for an MTP / DFlash /
hybrid ANE bundle.

The multifunction ANE architecture pivot (integrate/ane-iosurface-state)
requires every .mlmodelc to be shipped with an ane_state_layout.v1.json
sidecar. The gemma4 prefill bundle already does this; the MTP, DFlash,
and hybrid bundles (built by tools/ane-mtp/export-gemma4-mtp.py and
its sibling scripts) currently do not.

This script is the post-export adapter that fills the gap. It reads
the source GGUF, materializes the embedded .mlmodelc to a temp
directory, reads its metadata.json, and emits the ane_state_layout.v1
manifest next to the user-supplied .mlpackage. The manifest is
validated against the state_layout schema before being written.

Marked experimental: the JSON shape may evolve as F4.5 (Phase 0
profile) and downstream consumers (the Studio UI, the runtime
reader) land. The script writes an "_experimental: true" field on
the emitted manifest so downstream consumers can detect the
unstable schema.

Usage:
    python3 export_manifest.py \\
        --gguf MTP/mtp-gemma-4-12b-it-BF16.gguf \\
        --mlpackage MTP/batch-1.mlpackage \\
        --output MTP/batch-1.ane_state.v1.json

The --mlpackage directory is only used to anchor the output path
(the sidecar is written next to it). The script does not modify
the .mlpackage; the materialize step extracts the .mlmodelc to a
temp directory and reads its metadata.json from there.

The script is ungated: it works on any of the existing bundles
(MTP, DFlash, hybrid, prefill). The validation step uses
state_layout.py's StateLayout.from_dict to catch malformed
manifests before the sidecar is written.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Make sibling modules importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent))

from emit_manifest_from_mlmodelc import (  # noqa: E402
    build_manifest as build_manifest_from_mlmodelc,
)
from state_layout import StateLayout  # noqa: E402


# GGUF key prefix for the MTP / DFlash / hybrid ANE bundle. The
# multifunction bundle's files are embedded as tensors under this
# prefix; the C++ runtime reads them via gguf_get_tensor in
# common/ane-mtp.mm.
MTP_KEY_PREFIX = "mtp.ane"

# The manifest's "experimental" flag. Downstream consumers should
# refuse manifests with this flag set to true until the schema
# stabilizes. The flag is set on every emit from this script
# because the schema is still in flux (F4.5 is the next
# stabilization point).
EXPERIMENTAL_FLAG = True


def find_bundle_prefix(gguf_keys: list[str]) -> str | None:
    """Return the GGUF key prefix for the multifunction bundle.

    Looks for any key starting with "mtp.ane.bucket." (the
    per-bucket layout) or "mtp.ane.bundle." (the multifunction
    layout). Returns the first match, or None if the GGUF
    doesn't carry an ANE bundle.

    The per-bucket layout prefixes a batch number to every
    subkey (mtp.ane.bucket.1.file_count). The multifunction
    layout uses the bare prefix (mtp.ane.bundle.file_count).
    The helper returns the longest dotted prefix that
    distinguishes the bundle; for the bucket case this
    includes the batch number, for the bundle case it does
    not.
    """
    # Subkeys that the multifunction bundle may use. When the
    # part after "mtp.ane.bundle." is one of these, the prefix
    # is the bare "mtp.ane.bundle" (no number suffix).
    BUNDLE_SUBKEYS = {"file_count", "file", "functions",
                      "context_length", "sync_chunk"}
    for key in gguf_keys:
        if key.startswith("mtp.ane.bucket."):
            # "mtp.ane.bucket.1.file_count" -> "mtp.ane.bucket.1"
            head, _, _ = key[len("mtp.ane.bucket."):].partition(".")
            return "mtp.ane.bucket." + head
        if key.startswith("mtp.ane.bundle."):
            # "mtp.ane.bundle.file_count" -> "mtp.ane.bundle"
            head, _, _ = key[len("mtp.ane.bundle."):].partition(".")
            if head in BUNDLE_SUBKEYS or not head:
                return "mtp.ane.bundle"
            return "mtp.ane.bundle." + head
    return None


def materialize_mlmodelc(
        gguf_path: Path,
        prefix: str,
        staging: Path) -> Path:
    """Materialize the .mlmodelc from the embedded GGUF tensors.

    Reads the per-file tensors (prefix.file.NNNN) and writes each
    to the staging directory at the path declared by the
    corresponding prefix.file.NNNN.path string kv. The result
    is a directory tree that Core ML recognizes as a .mlmodelc.
    """
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise SystemExit(
            f"failed to import gguf: {exc}. Install gguf-py "
            "(pip install gguf) or run from the llama.cpp "
            "tools/ane-mtp directory where gguf is on PYTHONPATH."
        ) from exc
    reader = GGUFReader(str(gguf_path))
    # GGUFReader indexes fields by the full dotted key. For
    # string fields the value lives in field.parts[-1] (a
    # memmap of bytes that decodes to the UTF-8 string). For
    # numeric fields field.data[0] is the scalar. The writer's
    # add_string / add_uint32 routes to the right shape; the
    # reader's kv helper handles both.
    def kv_get(key: str):
        field = reader.fields.get(key)
        if field is None:
            return None
        if not field.types:
            return None
        # GGUFValueType.STRING == 8 (the int value is stable
        # across gguf-py versions; the symbol import is
        # version-fragile).
        if int(field.types[0]) == 8:
            if not field.parts:
                return None
            last = field.parts[-1]
            if hasattr(last, "tobytes"):
                return bytes(last.tobytes()).decode("utf-8")
            return bytes(last).decode("utf-8")
        if len(field.data) == 0:
            return None
        return field.data[0]
    # GGUFReader.tensors is a list (not a dict). Build a
    # name->tensor lookup for the materialize loop.
    tensor_by_name = {t.name: t for t in reader.tensors}
    file_count_key = f"{prefix}.file_count"
    fc = kv_get(file_count_key)
    if fc is None:
        raise SystemExit(
            f"GGUF {gguf_path} has no {file_count_key}; cannot "
            f"materialize the .mlmodelc"
        )
    try:
        file_count = int(fc)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"GGUF {file_count_key} is not an integer: {exc}"
        ) from exc
    if file_count <= 0:
        raise SystemExit(
            f"GGUF {file_count_key} = {file_count} (no files to "
            f"materialize)"
        )
    written = 0
    for index in range(file_count):
        file_key = f"{prefix}.file.{index:04d}"
        path_key = f"{prefix}.file.{index:04d}.path"
        tensor = tensor_by_name.get(file_key)
        if tensor is None:
            raise SystemExit(
                f"GGUF missing embedded tensor {file_key}"
            )
        relative = kv_get(path_key)
        if relative is None:
            raise SystemExit(
                f"GGUF missing path kv {path_key}"
            )
        try:
            relative = relative.decode("utf-8")
        except AttributeError:
            relative = str(relative)
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # The tensor is stored as int8 (GGUF's writer enforces
        # signed dtypes). Cast back to unsigned bytes for write;
        # the .mlmodelc files are binary blobs so the sign
        # doesn't matter.
        data = tensor.data
        if hasattr(data, "tobytes"):
            # numpy array path.
            data = data.tobytes()
        else:
            data = bytes(data)
        target.write_bytes(data)
        written += 1
    if written != file_count:
        raise SystemExit(
            f"materialized {written} files, expected {file_count}"
        )
    return staging


def load_gguf_keys(gguf_path: Path) -> list[str]:
    """Return the dotted-key list for the GGUF header."""
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise SystemExit(f"failed to import gguf: {exc}") from exc
    reader = GGUFReader(str(gguf_path))
    return list(reader.fields.keys())


def build_manifest_from_gguf(
        gguf_path: Path,
        mlpackage_path: Path,
        bundle_name: str | None) -> tuple[dict, Path]:
    """Build the ane_state_layout.v1 manifest from the source GGUF.

    Returns (manifest_dict, mlmodelc_dir). The caller writes the
    manifest to disk and the caller is responsible for cleaning up
    mlmodelc_dir (a temp directory).
    """
    keys = load_gguf_keys(gguf_path)
    prefix = find_bundle_prefix(keys)
    if prefix is None:
        raise SystemExit(
            f"GGUF {gguf_path} has no mtp.ane.bucket.* or "
            f"mtp.ane.bundle.* keys; not an ANE multifunction bundle"
        )
    staging = tempfile.mkdtemp(prefix="ane-export-manifest-")
    staging_path = Path(staging)
    try:
        mlmodelc_dir = materialize_mlmodelc(gguf_path, prefix, staging_path)
        bundle_stem = bundle_name or mlpackage_path.stem
        manifest = build_manifest_from_mlmodelc(mlmodelc_dir, bundle_stem)
        # Mark the manifest as experimental so downstream
        # consumers can detect the unstable schema. The flag is
        # additive: the runtime reader ignores unknown fields.
        manifest["_experimental"] = EXPERIMENTAL_FLAG
        return manifest, mlmodelc_dir
    finally:
        # The mlmodelc_dir is returned to the caller; the
        # caller is responsible for cleanup. We do not delete
        # the staging directory here.
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the ane_state_layout.v1 manifest sidecar for an "
            "ANE multifunction bundle (MTP / DFlash / hybrid)."
        ),
    )
    parser.add_argument(
        "--gguf", type=Path, required=True,
        help="path to the source GGUF (the multifunction bundle's "
             "metadata carrier; the .mlmodelc is embedded as tensors)",
    )
    parser.add_argument(
        "--mlpackage", type=Path, required=True,
        help="path to the .mlpackage directory; the manifest is "
             "written next to this. The .mlpackage is not modified.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="output path for the manifest JSON (default: "
             "<mlpackage_stem>.ane_state.v1.json in the .mlpackage's "
             "parent directory)",
    )
    parser.add_argument(
        "--bundle-name", type=str, default=None,
        help="override the bundle name in the emitted manifest "
             "(default: the .mlpackage's stem)",
    )
    parser.add_argument(
        "--keep-staging", action="store_true",
        help="do not delete the materialized .mlmodelc after "
             "the manifest is written (useful for debugging)",
    )
    args = parser.parse_args()
    if not args.gguf.is_file():
        raise SystemExit(f"GGUF not found: {args.gguf}")
    if not args.mlpackage.is_dir():
        raise SystemExit(f"mlpackage not found: {args.mlpackage}")
    output = args.output or (args.mlpackage.parent /
        f"{args.mlpackage.stem}.ane_state.v1.json")
    manifest, mlmodelc_dir = build_manifest_from_gguf(
        args.gguf, args.mlpackage, args.bundle_name)
    # Validate before write: StateLayout.from_dict raises on
    # schema violations (slot alignment, name uniqueness,
    # function references, dead-state detection, etc.).
    try:
        StateLayout.from_dict(manifest)
    except (ValueError, KeyError, TypeError) as exc:
        if not args.keep_staging:
            shutil.rmtree(mlmodelc_dir.parent, ignore_errors=True)
        raise SystemExit(
            f"manifest validation failed: {exc}\n"
            f"the manifest was NOT written to {output}"
        ) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.keep_staging:
        shutil.rmtree(mlmodelc_dir.parent, ignore_errors=True)
    print(f"wrote {output}", file=sys.stderr)
    print(f"  bundle_name: {manifest['bundle_name']}", file=sys.stderr)
    print(f"  state_size_bytes: {manifest['state_size_bytes']}",
          file=sys.stderr)
    print(f"  model_type: {manifest['model_type']}", file=sys.stderr)
    print(f"  slots: {len(manifest['slots'])}", file=sys.stderr)
    print(f"  functions: {len(manifest['functions'])}", file=sys.stderr)
    print(f"  experimental: {manifest['_experimental']}", file=sys.stderr)


if __name__ == "__main__":
    main()
