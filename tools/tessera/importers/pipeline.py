"""Importer pipeline.

The pipeline is the orchestration layer that ties together
``format_detector`` + the format-specific ``parsers/*`` +
``ast_builder`` + ``data_layer_client`` + ``receipt_emitter``.

Each step is idempotent: re-running the pipeline on the same
file produces the same AST (modulo UUIDs, which are
deterministic when seeded, and timestamps). A failed parse
is logged and skipped — the rest of the batch continues.

The pipeline is a value type: it has no state beyond the
client references. It's safe to call from concurrent
coroutines in the same process as long as the underlying
data layer is thread-safe (the Swift ``TesseraDataLayer`` is
an actor, so yes).

Why a pipeline (not just a function per format)?

* The CLI / HTTP API both invoke the same orchestration; the
  pipeline is the unit-test boundary for "does the whole
  thing work?".
* The pipeline owns the failure-handling policy: which
  exceptions are recoverable (parser timeout → log + skip),
  which are fatal (data layer unreachable → bail).
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import ast_builder, ast_schema
from .data_layer_client import DataLayerClient, ImportResult
from .format_detector import Detection, Format, detect
from .intermediate import IntermediateDocument
from .receipt_emitter import (
    ReceiptRecord,
    emit_import_receipt,
    hash_bytes,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ImportFailure:
    """One file that failed to import."""

    path: Path
    reason: str
    traceback: str = ""


@dataclass
class ImportSuccess:
    """One file that imported successfully."""

    path: Path
    format: str
    parser: str
    entities: list[ImportResult]
    receipts: list[ReceiptRecord]
    elapsed_seconds: float


@dataclass
class PipelineResult:
    """The aggregate result of an import run (one or more files)."""

    successes: list[ImportSuccess] = field(default_factory=list)
    failures: list[ImportFailure] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.failures


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class ImportPipeline:
    """One configured import pipeline.

    `client` is the data-layer client (the dry-run client is
    the default). `media_dir` is the directory the parsers
    write extracted image bytes to; when None, images are
    skipped. `fail_fast` aborts on the first failure (used by
    the unit tests; the CLI defaults to continue-on-error so
    a single bad file doesn't take down a batch).
    """

    def __init__(
        self,
        client: DataLayerClient,
        *,
        media_dir: Optional[Path] = None,
        fail_fast: bool = False,
    ) -> None:
        self.client = client
        self.media_dir = media_dir
        self.fail_fast = fail_fast

    # -----------------------------------------------------------------------
    # Single file
    # -----------------------------------------------------------------------

    def import_file(self, path: Path) -> ImportSuccess:
        """Import a single file.

        The detection step chooses a parser; the parser produces
        one or more ``IntermediateDocument``s; the AST builder
        turns them into ``DocumentAST``s; the data layer
        client persists them as ``graph_entity`` rows; the
        receipt emitter appends an import receipt per entity.

        Returns the success record. Raises on the first
        unrecoverable error when ``fail_fast`` is set; the CLI
        catches and reports.
        """
        t0 = time.monotonic()
        log.info("import_file: %s", path)
        if not path.exists():
            raise FileNotFoundError(path)
        detection = detect(path)
        intermediates = self._parse(path, detection)
        if not intermediates:
            raise ValueError(f"no content extracted from {path}")
        # The base_title is the path stem; the AST builder
        # only uses it as a suffix when the intermediate
        # carries no inferred title (no leading heading).
        # For emails with a Subject, the subject becomes the
        # heading and the title; the file stem is reserved
        # for documents whose content doesn't include a
        # natural title.
        asts = ast_builder.build_many(
            intermediates, base_title=path.stem, override_title=False
        )
        return self._persist_and_emit(path, detection, asts, t0)

    # -----------------------------------------------------------------------
    # Batch
    # -----------------------------------------------------------------------

    def import_paths(self, paths: list[Path]) -> PipelineResult:
        """Import a batch of paths (files or directories).

        Directories are walked recursively. Each file is
        imported in sequence; failures are collected and the
        batch continues unless ``fail_fast`` is set.
        """
        t0 = time.monotonic()
        result = PipelineResult()
        files = list(_expand_paths(paths))
        for p in files:
            try:
                result.successes.append(self.import_file(p))
            except Exception as e:  # noqa: BLE001
                log.warning("import failed: %s: %s", p, e)
                result.failures.append(
                    ImportFailure(
                        path=p,
                        reason=str(e),
                        traceback=traceback.format_exc(),
                    )
                )
                if self.fail_fast:
                    break
        result.total_elapsed_seconds = time.monotonic() - t0
        return result

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _parse(
        self, path: Path, detection: Detection
    ) -> list[IntermediateDocument]:
        """Dispatch to the right parser based on the detection."""
        fmt = detection.format
        media = self.media_dir / path.stem if self.media_dir else None
        if fmt is Format.DOCX:
            from .parsers import docx as _docx

            return [_docx.parse_docx(path, media_dir=media)]
        if fmt is Format.XLSX:
            from .parsers import xlsx as _xlsx

            return _xlsx.parse_xlsx(path)
        if fmt is Format.PPTX:
            from .parsers import pptx as _pptx

            return _pptx.parse_pptx(path, media_dir=media)
        if fmt is Format.PDF:
            from .parsers import pdf as _pdf

            return [_pdf.parse_pdf(path)]
        if fmt is Format.EML:
            from .parsers import email as _email

            return [_email.parse_eml(path)]
        if fmt is Format.MSG:
            from .parsers import email as _email

            return [_email.parse_msg(path)]
        if fmt is Format.MBOX:
            from .parsers import email as _email

            return _email.parse_mbox(path)
        if fmt is Format.HTML:
            from .parsers import html as _html

            return [_html.parse_html(path)]
        if fmt is Format.MHTML:
            from .parsers import html as _html

            return [_html.parse_mhtml(path)]
        if fmt is Format.MARKDOWN:
            from .parsers import markdown as _md

            return [_md.parse_markdown(path)]
        if fmt is Format.PANDOC:
            from .parsers import pandoc as _pandoc

            return [_pandoc.parse_pandoc(path)]
        raise ValueError(f"unsupported format: {fmt}")

    def _persist_and_emit(
        self,
        path: Path,
        detection: Detection,
        asts: list[ast_schema.DocumentAST],
        t0: float,
    ) -> ImportSuccess:
        """Persist each AST as a graph_entity and append a receipt."""
        entities: list[ImportResult] = []
        receipts: list[ReceiptRecord] = []
        for ast in asts:
            body_str = json.dumps(ast.to_json(), ensure_ascii=False, separators=(",", ":"))
            title = ast.meta.get("title") or path.stem
            # Multi-document: append a per-doc index to the
            # label so the user can tell them apart.
            label = title
            if len(asts) > 1:
                label = f"{title} [{len(entities) + 1}/{len(asts)}]"
            entity_type, subtype = _entity_type_for(path, detection)
            result = self.client.create_entity(
                entity_type=entity_type,
                label=label,
                body=body_str,
                source_url=str(path),
                subtype=subtype,
            )
            entities.append(result)
            content_hash = hash_bytes(ast.canonical_bytes())
            receipt = emit_import_receipt(
                entity_id=result.entity_id,
                source_path=path,
                format_detected=detection.format.value,
                parser_used=_parser_name(detection.format),
                ast_content_hash=content_hash,
                body_size_bytes=len(body_str.encode("utf-8")),
                block_count=len(ast.blocks),
            )
            self.client.append_receipt(
                entity_id=result.entity_id,
                receipt_type=receipt.receipt_type,
                payload=receipt.payload,
                signature=receipt.signature,
            )
            receipts.append(receipt)

        elapsed = time.monotonic() - t0
        return ImportSuccess(
            path=path,
            format=detection.format.value,
            parser=_parser_name(detection.format),
            entities=entities,
            receipts=receipts,
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parser_name(fmt: Format) -> str:
    return {
        Format.DOCX: "python-docx",
        Format.XLSX: "openpyxl",
        Format.PPTX: "python-pptx",
        Format.PDF: "pdftotext",
        Format.EML: "mailbox+email",
        Format.MSG: "mailbox+email",
        Format.MBOX: "mailbox",
        Format.HTML: "beautifulsoup4",
        Format.MHTML: "beautifulsoup4",
        Format.MARKDOWN: "markdown-it-py",
        Format.PANDOC: "pandoc",
    }.get(fmt, "unknown")


def _entity_type_for(path: Path, detection: Detection) -> tuple[str, Optional[str]]:
    """Map a format to a ``graph_entity`` entity_type + subtype.

    The entity type is what the productivity surface uses to
    group the document: ``"document"`` for prose (the editor
    treats it as a doc), ``"spreadsheet"`` for XLSX,
    ``"presentation"`` for PPTX, ``"email"`` for EML / MBOX /
    MSG. Subtype is a finer tag (``"doc"``, ``"sheet"``,
    ``"slide"``).
    """
    mapping = {
        Format.DOCX: ("document", "doc"),
        Format.XLSX: ("spreadsheet", "sheet"),
        Format.PPTX: ("presentation", "slide"),
        Format.PDF: ("document", "doc"),
        Format.EML: ("email", "email"),
        Format.MSG: ("email", "email"),
        Format.MBOX: ("email", "email"),
        Format.HTML: ("document", "html"),
        Format.MHTML: ("document", "html"),
        Format.MARKDOWN: ("document", "md"),
        Format.PANDOC: ("document", "generic"),
    }
    return mapping.get(detection.format, ("document", "unknown"))


def _expand_paths(paths: list[Path]) -> list[Path]:
    """Expand a list of paths (files and directories) into a list of files.

    Directories are walked recursively. Non-existent paths are
    silently dropped (the caller logs the per-file failure).
    The order is filesystem order; we don't sort so a v2
    change can be deterministic with a flag.
    """
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            for child in p.rglob("*"):
                if child.is_file():
                    out.append(child)
        elif p.is_file():
            out.append(p)
    return out
