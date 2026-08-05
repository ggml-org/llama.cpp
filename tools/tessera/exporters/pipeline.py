"""Exporter pipeline.

The pipeline is the orchestration layer that ties together
``data_layer_client`` (to fetch the AST) + ``ast_to_intermediate``
+ the per-format ``builders/*`` + ``receipt_emitter``.

The pipeline is a value type. It's the unit-test boundary for
"does the whole export work?".

The flow:

1. Fetch the entity's body (AST JSON) and meta from the data
   layer. The body is the ``DocumentAST`` (as JSON).
2. Decode the AST to ``DocumentAST``.
3. Convert to ``IntermediateDocument`` via ``ast_to_intermediate``.
4. Dispatch to the per-format builder.
5. Compute the output file's SHA-256 and emit an export
   receipt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from tools.tessera.importers.ast_schema import DocumentAST
from tools.tessera.importers.data_layer_client import DataLayerClient, make_default_client
from tools.tessera.importers.intermediate import IntermediateDocument
from tools.tessera.importers.receipt_emitter import ReceiptRecord, emit_export_receipt
from . import ast_to_intermediate

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format enum
# ---------------------------------------------------------------------------


SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {"pdf", "docx", "xlsx", "pptx", "html", "md", "eml"}
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ExportSuccess:
    """One successful export."""

    entity_id: str
    format: str
    output_path: Path
    size_bytes: int
    sha256: str
    receipt: ReceiptRecord
    elapsed_seconds: float


@dataclass
class ExportFailure:
    """One failed export."""

    entity_id: str
    format: str
    reason: str
    traceback: str = ""


@dataclass
class ExportResult:
    """The aggregate result of one or more exports."""

    successes: list[ExportSuccess] = field(default_factory=list)
    failures: list[ExportFailure] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.failures


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class ExportPipeline:
    """One configured export pipeline.

    `client` is the data-layer client. `output_dir` is where
    files are written (default: a temp directory). `fail_fast`
    aborts on the first failure.
    """

    def __init__(
        self,
        client: DataLayerClient,
        *,
        output_dir: Optional[Path] = None,
        fail_fast: bool = False,
    ) -> None:
        self.client = client
        self.output_dir = output_dir or Path.cwd() / "exports"
        self.fail_fast = fail_fast

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def export(
        self,
        entity_id: str,
        target_format: str,
        output_path: Optional[Path] = None,
    ) -> ExportSuccess:
        """Export one entity to the given format.

        `output_path` overrides the default output location.
        The pipeline returns the success record; failures raise.
        """
        if target_format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"unsupported format: {target_format!r}; expected one of {sorted(SUPPORTED_FORMATS)}"
            )

        t0 = time.monotonic()
        log.info("export: %s -> %s", entity_id, target_format)

        # 1) Fetch the AST from the data layer
        body = self.client.get_entity_body(entity_id)
        if body is None:
            # Dry-run: the client returns None. Build a
            # minimal AST from the meta so the rest of the
            # pipeline still runs.
            meta = self.client.get_entity_meta(entity_id)
            ast = _minimal_ast_from_meta(meta, entity_id)
        else:
            ast = DocumentAST.from_json(json.loads(body))

        # 2) Convert to intermediate
        intermediate = ast_to_intermediate.ast_to_intermediate(ast)

        # 3) Build the file
        if output_path is None:
            output_path = self._default_output_path(entity_id, target_format)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _build(intermediate, target_format, output_path)

        # 4) Compute the SHA-256 of the output bytes
        sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
        size = output_path.stat().st_size

        # 5) Emit a receipt
        receipt = emit_export_receipt(
            entity_id=entity_id,
            output_path=output_path,
            target_format=target_format,
            output_size_bytes=size,
            output_sha256=f"sha256:{sha}",
        )
        self.client.append_receipt(
            entity_id=entity_id,
            receipt_type=receipt.receipt_type,
            payload=receipt.payload,
            signature=receipt.signature,
        )
        elapsed = time.monotonic() - t0
        return ExportSuccess(
            entity_id=entity_id,
            format=target_format,
            output_path=output_path,
            size_bytes=size,
            sha256=f"sha256:{sha}",
            receipt=receipt,
            elapsed_seconds=elapsed,
        )

    def export_batch(
        self,
        items: list[tuple[str, str]],
    ) -> ExportResult:
        """Export a batch of (entity_id, format) pairs.

        `items` is a list of ``(entity_id, format)`` tuples.
        Each item is exported in sequence; failures are
        collected and the batch continues.
        """
        t0 = time.monotonic()
        result = ExportResult()
        for entity_id, fmt in items:
            try:
                result.successes.append(self.export(entity_id, fmt))
            except Exception as e:  # noqa: BLE001
                log.warning("export failed: %s -> %s: %s", entity_id, fmt, e)
                result.failures.append(
                    ExportFailure(
                        entity_id=entity_id,
                        format=fmt,
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

    def _default_output_path(self, entity_id: str, target_format: str) -> Path:
        ext = {
            "pdf": "pdf",
            "docx": "docx",
            "xlsx": "xlsx",
            "pptx": "pptx",
            "html": "html",
            "md": "md",
            "eml": "eml",
        }[target_format]
        return self.output_dir / f"{entity_id}.{ext}"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _build(
    intermediate: IntermediateDocument, target_format: str, output: Path
) -> None:
    """Dispatch to the per-format builder."""
    if target_format == "md":
        from .builders import markdown as _md

        output.write_text(_md.build_markdown(intermediate), encoding="utf-8")
        return
    if target_format == "html":
        from .builders import html as _html

        output.write_text(_html.build_html(intermediate), encoding="utf-8")
        return
    if target_format == "eml":
        from .builders import email as _email

        output.write_text(_email.build_eml(intermediate), encoding="utf-8")
        return
    if target_format == "pdf":
        from .builders import pdf as _pdf

        _pdf.build_pdf(intermediate, output)
        return
    if target_format == "docx":
        from .builders import docx as _docx

        _docx.build_docx(intermediate, output)
        return
    if target_format == "xlsx":
        from .builders import xlsx as _xlsx

        _xlsx.build_xlsx(intermediate, output)
        return
    if target_format == "pptx":
        # PPTX is multi-slide; the export pipeline takes
        # one entity per export. v1 stores a single slide per
        # document; for multi-slide decks, the importer
        # creates one entity per slide and the user exports
        # each. v2 will add a "deck" entity type that groups
        # slides and exports them as one .pptx.
        from .builders import pptx as _pptx

        _pptx.build_pptx([intermediate], output)
        return
    raise ValueError(f"unsupported format: {target_format!r}")


def _minimal_ast_from_meta(meta: dict[str, Any], entity_id: str) -> DocumentAST:
    """Build a minimal AST for the dry-run / no-data-layer case.

    The AST is a single paragraph with the entity label. This
    lets the export pipeline run end-to-end in tests and CI
    without a live data layer. The block id is a uuid4; the
    entity id is recorded in the AST's meta.
    """
    from tools.tessera.importers.ast_schema import (
        Block,
        DocumentAST,
        new_block_id,
        run_to_json,
    )
    import uuid as _uuid

    label = str(meta.get("label", "(dry-run)"))
    block = Block(
        id=new_block_id(),
        type="heading",
        attributes={"level": 1},
        content=[run_to_json(label)],
    )
    ast = DocumentAST(blocks={block.id: block}, rootChildren=[block.id])
    # If entity_id looks like a UUID, use it as the title-meta
    # for traceability; otherwise store the raw value.
    try:
        _uuid.UUID(entity_id)
        ast.meta["entity_id"] = entity_id
    except ValueError:
        ast.meta["entity_id_raw"] = entity_id
    return ast


def make_default_pipeline() -> ExportPipeline:
    """Build the default export pipeline for the current process."""
    return ExportPipeline(make_default_client())
