"""Importer CLI.

``tessera import <path> [<path> ...]``

The CLI is the script-runner entry point. The Swift side
shells out to it (``python -m tools.tessera.importers.cli
import <files>``) and parses the JSON output. The HTTP API
on the Swift side wraps the same pipeline; the CLI is the
canonical surface for unit tests and for ad-hoc imports
from the terminal.

Output shape (stdout, one JSON line per file, then a final
summary line):

.. code-block:: json

    {"event": "import_ok", "path": "...", "format": "docx",
     "parser": "python-docx", "entities": [{"entity_id": "...",
     "entity_type": "document"}], "receipt_ids": ["..."],
     "elapsed_seconds": 0.34}
    {"event": "import_failed", "path": "...", "reason": "..."}
    {"event": "summary", "ok": 3, "failed": 1, "elapsed": 1.2}

Exit codes:

* ``0`` — all files imported (failures may exist; check
  the summary line).
* ``1`` — internal error (bad arguments, data layer
  unreachable, etc.).
* ``2`` — at least one file failed to import. The Swift
  side surfaces this as a non-fatal warning.

Usage examples:

    python3 -m tools.tessera.importers.cli import file.docx
    python3 -m tools.tessera.importers.cli import *.eml
    python3 -m tools.tessera.importers.cli import --media-dir /tmp/img file.docx

Environment:

* ``TESSERA_DATA_LAYER_URL`` — Swift HTTP API root.
* ``TESSERA_DATA_LAYER_DRY_RUN`` — ``1`` (default) means
  no HTTP calls; ``0`` enables the real client.
* ``TESSERA_RECEIPT_SIGNING_KEY`` / ``TESSERA_RECEIPT_SIGNING_KEY_PATH`` —
  ed25519 seed for receipt signing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence

from .data_layer_client import make_default_client
from .pipeline import ImportPipeline, ImportFailure, ImportSuccess


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="tessera-import",
        description="Tessera productivity importer",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_import = sub.add_parser("import", help="Import one or more files")
    p_import.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to import. Directories are walked recursively.",
    )
    p_import.add_argument(
        "--media-dir",
        type=Path,
        default=None,
        help="Where extracted images are written (default: skip images).",
    )
    p_import.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort the batch on the first failure (default: continue).",
    )
    p_import.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the data layer (parse + build AST only).",
    )
    p_import.add_argument(
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

    if args.command != "import":
        parser.error(f"unknown command: {args.command}")
        return 2

    client = make_default_client()
    if args.dry_run:
        client.dry_run = True
    pipeline = ImportPipeline(
        client,
        media_dir=args.media_dir,
        fail_fast=args.fail_fast,
    )
    result = pipeline.import_paths(list(args.paths))

    return _emit(result)


def _emit(result: Any) -> int:
    """Write per-file events + a summary to stdout. Returns the exit code."""
    for s in result.successes:
        _emit_event(
            {
                "event": "import_ok",
                "path": str(s.path),
                "format": s.format,
                "parser": s.parser,
                "entities": [
                    {"entity_id": e.entity_id, "entity_type": e.entity_type}
                    for e in s.entities
                ],
                "receipt_ids": [r.receipt_id for r in s.receipts],
                "elapsed_seconds": round(s.elapsed_seconds, 4),
            }
        )
    for f in result.failures:
        _emit_event(
            {
                "event": "import_failed",
                "path": str(f.path),
                "reason": f.reason,
            }
        )
    _emit_event(
        {
            "event": "summary",
            "ok": len(result.successes),
            "failed": len(result.failures),
            "elapsed": round(result.total_elapsed_seconds, 4),
        }
    )
    sys.stdout.flush()
    if result.failures:
        return 2
    return 0


def _emit_event(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
