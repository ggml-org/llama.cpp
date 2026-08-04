"""Polars-typed reader for Tessera analytical outputs.

The C++ helper ``common/tessera-analytical-io.{h,cpp}`` writes
analytical outputs as NDJSON, one record per line, conformant to a
schema file in ``common/schemas/<name>.schema.json``. This module is
the Python half of the same contract: it reads the NDJSON file and
returns a polars DataFrame with the column types pinned by the
schema's ``polars_schema`` field.

Schemas (mirrored on the C++ side in common/tessera-analytical-io.cpp):

* ``l3_outlier``     — per-tensor L3 outlier stats from
                       ``tools/tessera/l3_outlier_report.py``
* ``l4_probe``       — per-tensor L4 E2E probe report (consumed by
                       ``tools/tessera/l5_orchestrator.py``)
* ``l5_plan``        — per-tensor L5 IterQuant requant plan
                       (``tools/tessera/l5_orchestrator.py:write_history``)
* ``per_layer_error``— per-tensor L1/L1.5 sidecar relative error from
                       ``tools/tessera/per_layer_error_table.py``

Companion to the polars-integration scout
(``docs/tessera-polars-integration-scout.md``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import polars as pl

THIS_FILE = Path(__file__).resolve()
# tools/tessera/_analytical_io.py -> repo root = parents[2]
REPO_ROOT = THIS_FILE.parents[2]
DEFAULT_SCHEMA_DIR = REPO_ROOT / "common" / "schemas"

# Hard-coded list of known schema names. Mirrored on the C++ side in
# common/tessera-analytical-io.cpp:k_known_schemas. Keep them in sync.
KNOWN_SCHEMAS: tuple[str, ...] = (
    "l3_outlier",
    "l4_probe",
    "l5_plan",
    "per_layer_error",
)

# Mapping from polars-dtype-string to the polars dtype object. Polars
# itself does not parse dtype strings out of the box; the C++ side
# records the canonical string ("Int64", "Float32", ...) in the
# schema JSON, and this dict is the bridge.
_POLARS_DTYPES: Mapping[str, pl.DataType] = {
    "String":   pl.String,
    "Boolean":  pl.Boolean,
    "Int8":     pl.Int8,
    "Int16":    pl.Int16,
    "Int32":    pl.Int32,
    "Int64":    pl.Int64,
    "UInt8":    pl.UInt8,
    "UInt16":   pl.UInt16,
    "UInt32":   pl.UInt32,
    "UInt64":   pl.UInt64,
    "Float32":  pl.Float32,
    "Float64":  pl.Float64,
    "Date":     pl.Date,
    "Datetime": pl.Datetime,
}


def _schema_dir() -> Path:
    """Resolve the schema directory. Search order:

    1. ``TESSERA_SCHEMA_DIR`` environment variable (installed builds)
    2. The repo-relative default ``common/schemas``

    Returns the first existing path; raises if neither exists.
    """
    env = os.environ.get("TESSERA_SCHEMA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
    if DEFAULT_SCHEMA_DIR.is_dir():
        return DEFAULT_SCHEMA_DIR
    raise FileNotFoundError(
        f"no schema directory found; tried {env!r} and {DEFAULT_SCHEMA_DIR}"
    )


def schema_path(name: str) -> Path:
    """Return the full path to a named schema file. Raises if the
    name is not in KNOWN_SCHEMAS or the file is missing on disk."""
    if name not in KNOWN_SCHEMAS:
        raise ValueError(
            f"unknown schema name {name!r}; expected one of {KNOWN_SCHEMAS}"
        )
    p = _schema_dir() / f"{name}.schema.json"
    if not p.is_file():
        raise FileNotFoundError(f"schema file not found: {p}")
    return p


def _load_schema(name: str) -> dict:
    """Load a schema JSON file."""
    with schema_path(name).open("r", encoding="utf-8") as f:
        return json.load(f)


def polars_schema(name: str) -> dict[str, pl.DataType]:
    """Return the polars-typed schema for a named analytical output.

    Columns absent from the schema's ``polars_schema`` are left to
    polars' inference (the underlying ``read_ndjson`` is told to
    infer them). Columns present in the schema are pinned to the
    declared polars dtype so a C++ writer that emits a sloppy
    int-as-float does not silently change the polars dtype downstream.
    """
    schema = _load_schema(name)
    raw = schema.get("polars_schema", {})
    out: dict[str, pl.DataType] = {}
    for col, dtype_str in raw.items():
        if dtype_str not in _POLARS_DTYPES:
            raise ValueError(
                f"schema {name!r} declares unknown polars dtype "
                f"{dtype_str!r} for column {col!r}"
            )
        out[col] = _POLARS_DTYPES[dtype_str]
    return out


def read_analytical(
    path: Path | str,
    schema: str,
    *,
    infer_schema_length: int = 10_000,
) -> pl.DataFrame:
    """Read an NDJSON file produced by
    ``tessera::analytical::write_record`` and return a polars
    DataFrame with columns typed per the schema.

    Args:
        path: path to the NDJSON file.
        schema: schema name (one of ``KNOWN_SCHEMAS``).
        infer_schema_length: rows polars scans to infer types for
            columns absent from the schema (default 10k; the polars
            ``read_ndjson`` default is 100).

    Returns:
        A ``pl.DataFrame`` with the schema-pinned columns typed and
        the rest inferred.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NDJSON file not found: {path}")
    pschema = polars_schema(schema)
    return pl.read_ndjson(
        path,
        schema=pschema,
        infer_schema_length=infer_schema_length,
    )


def known_schemas() -> tuple[str, ...]:
    """Return the tuple of known schema names."""
    return KNOWN_SCHEMAS
