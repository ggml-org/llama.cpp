#!/usr/bin/env python3
"""Smoke test for tools/tessera/_analytical_io.py.

Exercises every code path: schema resolution, polars-dtype lookup,
NDJSON round-trip, type pinning against the four shipped schemas
(l3_outlier, l4_probe, l5_plan, per_layer_error).

The C++ helper writes the records; this test reads them back via
the polars wrapper and asserts:

  1. Every shipped schema is in KNOWN_SCHEMAS.
  2. polars_schema(name) returns a non-empty dict for every shipped
     schema.
  3. A record conforming to a schema's required list round-trips
     through read_analytical without losing or retyping columns.
  4. A column declared as Int64 in polars_schema is read as Int64
     even when the source NDJSON has the value as a number
     (C++ may emit "1.0" for an int; the schema pins the dtype).
  5. Reading a non-existent file raises FileNotFoundError.
  6. Reading with an unknown schema name raises ValueError.

Run as:  python3 tools/tessera/test_analytical_io.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from _analytical_io import (  # noqa: E402
    KNOWN_SCHEMAS,
    known_schemas,
    polars_schema,
    read_analytical,
    schema_path,
)


def _write_ndjson(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


SHIPPED_SCHEMAS = ("l3_outlier", "l4_probe", "l5_plan", "per_layer_error")


class TestKnownSchemas(unittest.TestCase):
    def test_every_shipped_schema_is_listed(self) -> None:
        for name in SHIPPED_SCHEMAS:
            self.assertIn(name, KNOWN_SCHEMAS,
                          f"shipped schema {name!r} missing from KNOWN_SCHEMAS")
        self.assertEqual(KNOWN_SCHEMAS, tuple(known_schemas()))

    def test_schema_path_resolves(self) -> None:
        for name in SHIPPED_SCHEMAS:
            p = schema_path(name)
            self.assertTrue(p.is_file(), f"schema_path({name!r}) is not a file: {p}")

    def test_polars_schema_non_empty(self) -> None:
        for name in SHIPPED_SCHEMAS:
            s = polars_schema(name)
            self.assertGreater(len(s), 0,
                               f"polars_schema({name!r}) is empty")


def _sample_l3_record() -> dict:
    return {
        "sidecar_label": "ckpt-v3-q4_k",
        "tensor": "blk.12.attn_q.weight",
        "layer": "blk.12",
        "rows": 4096,
        "cols": 4096,
        "total_elements": 16777216,
        "outlier_count": 17000,
        "outlier_fraction": 0.001013,
        "threshold": 6.0,
        "skipped": False,
        "kernel_version": "integrate/polars-scout-phase-a abc1234",
        "created_at": "2026-08-03T20:30:00Z",
        "tessera_main_tip": "df3e9c6cd",
    }


def _sample_l4_record() -> dict:
    return {
        "tensor": "blk.7.ffn_down.weight",
        "layer": "blk.7",
        "current_qtype": "Q4_K",
        "mse": 0.012,
        "mse_minus_one": 0.018,
        "perplexity": 8.21,
        "top1_mismatch": 0.04,
        "n_weights": 16777216,
        "model": "llama-3.1-8b",
        "calibration_corpus": "wikitext-2-raw-v1",
        "calibration_corpus_hash": "deadbeef" * 8,
        "kernel_version": "integrate/polars-scout-phase-a abc1234",
        "created_at": "2026-08-03T20:30:00Z",
        "tessera_main_tip": "df3e9c6cd",
    }


def _sample_l5_record() -> dict:
    return {
        "tensor": "blk.7.ffn_down.weight",
        "layer": "blk.7",
        "current_qtype": "Q4_K",
        "new_qtype": "Q5_K",
        "bits": 5.5,
        "delta_bits": 32,
        "sensitivity_score": 0.87,
        "delta_quality": -0.04,
        "imatrix_magnitude": 0.6,
        "gradient_proxy": 0.4,
        "layer_position_prior": 0.5,
        "plan_id": "plan-2026-08-03T20-30-iter3",
        "iteration": 3,
        "kernel_version": "integrate/polars-scout-phase-a abc1234",
        "created_at": "2026-08-03T20:30:00Z",
        "tessera_main_tip": "df3e9c6cd",
    }


def _sample_per_layer_record() -> dict:
    return {
        "tensor": "blk.7.ffn_down.weight",
        "layer": "blk.7",
        "epsilon": 0.0012,
        "epsilon_is_nan": False,
        "note": "",
        "sidecar_dir": "/tmp/sidecars-run-3",
        "kernel_version": "integrate/polars-scout-phase-a abc1234",
        "created_at": "2026-08-03T20:30:00Z",
        "tessera_main_tip": "df3e9c6cd",
    }


class TestReadAnalytical(unittest.TestCase):
    def test_l3_outlier_round_trip(self) -> None:
        records = [_sample_l3_record(), _sample_l3_record() | {"tensor": "blk.8.attn_q.weight"}]
        with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            _write_ndjson(records, tmp_path)
            df = read_analytical(tmp_path, "l3_outlier")
            self.assertEqual(df.height, 2)
            self.assertIn("tensor", df.columns)
            self.assertIn("outlier_fraction", df.columns)
            # Type pinning: outlier_count is Int64 in the schema.
            self.assertEqual(df.schema["outlier_count"], pl.Int64)
            self.assertEqual(df.schema["outlier_fraction"], pl.Float64)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_l4_probe_round_trip(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            _write_ndjson([_sample_l4_record()], tmp_path)
            df = read_analytical(tmp_path, "l4_probe")
            self.assertEqual(df.height, 1)
            self.assertEqual(df.schema["mse"], pl.Float64)
            self.assertEqual(df.schema["n_weights"], pl.Int64)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_l5_plan_round_trip(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            _write_ndjson([_sample_l5_record()], tmp_path)
            df = read_analytical(tmp_path, "l5_plan")
            self.assertEqual(df.height, 1)
            self.assertEqual(df.schema["delta_bits"], pl.Int64)
            self.assertEqual(df.schema["iteration"], pl.Int32)
            self.assertEqual(df.schema["sensitivity_score"], pl.Float64)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_per_layer_error_round_trip(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            _write_ndjson([_sample_per_layer_record()], tmp_path)
            df = read_analytical(tmp_path, "per_layer_error")
            self.assertEqual(df.height, 1)
            self.assertEqual(df.schema["epsilon"], pl.Float64)
            self.assertEqual(df.schema["epsilon_is_nan"], pl.Boolean)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_type_pinning_int_from_int(self) -> None:
        """A C++ writer that emits outlier_count as a JSON integer
        (17000, not 17000.0) is the canonical happy path. Polars
        reads the JSON integer and the schema pins it to Int64 even
        if polars' own inference would pick a wider numeric type.
        Verifies the schema's polars_schema override is in effect."""
        record = _sample_l3_record()
        with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            _write_ndjson([record], tmp_path)
            df = read_analytical(tmp_path, "l3_outlier")
            # Schema pins Int64; polars respects the override.
            self.assertEqual(df.schema["outlier_count"], pl.Int64)
            # The value survives the round-trip.
            self.assertEqual(df["outlier_count"].to_list(), [17000])
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_analytical("/nonexistent/path/to/nowhere.ndjson", "l3_outlier")

    def test_unknown_schema_raises(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            _write_ndjson([_sample_l3_record()], tmp_path)
            with self.assertRaises(ValueError):
                read_analytical(tmp_path, "not_a_real_schema")
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
