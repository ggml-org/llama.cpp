"""Exporter CLI.

``tessera export <entity_id> --format <fmt>``

Exports a single ``graph_entity`` from the data layer to the
given format. Output goes to a file in the current directory
(or to ``--output <path>`` when given).

The CLI mirrors the importer's: it writes one JSON line per
result to stdout, then a summary line. Exit code 0 on success,
2 on at least one failure, 1 on internal error.

Usage:

    python3 -m tools.tessera.exporters.cli export <entity_id> --format md
    python3 -m tools.tessera.exporters.cli export <entity_id> --format pdf --output /tmp/x.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence

from .pipeline import ExportPipeline, make_default_pipeline


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="tessera-export",
        description="Tessera productivity exporter",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_export = sub.add_parser("export", help="Export one entity")
    p_export.add_argument("entity_id", type=str, help="The graph_entity UUID")
    p_export.add_argument(
        "--format",
        "-f",
        type=str,
        required=True,
        choices=["pdf", "docx", "xlsx", "pptx", "html", "md", "eml"],
    )
    p_export.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Where to write the output (default: ./<entity_id>.<ext>)",
    )
    p_export.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        help="Default output directory (when --output is not set).",
    )
    p_export.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the data layer (use a minimal AST instead).",
    )
    p_export.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging to stderr.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    pipeline = make_default_pipeline()
    pipeline.output_dir = args.output_dir
    if args.dry_run:
        pipeline.client.dry_run = True

    try:
        result = pipeline.export(
            args.entity_id, args.format, output_path=args.output
        )
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"error: {e}\n")
        return 1

    _emit_event(
        {
            "event": "export_ok",
            "entity_id": result.entity_id,
            "format": result.format,
            "output_path": str(result.output_path),
            "size_bytes": result.size_bytes,
            "sha256": result.sha256,
            "receipt_id": result.receipt.receipt_id,
            "elapsed_seconds": round(result.elapsed_seconds, 4),
        }
    )
    _emit_event(
        {
            "event": "summary",
            "ok": 1,
            "failed": 0,
            "elapsed": round(result.elapsed_seconds, 4),
        }
    )
    return 0


def _emit_event(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
