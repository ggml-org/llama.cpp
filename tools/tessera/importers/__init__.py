"""Tessera productivity importer (Phase 4).

This package is the Python side of the importer pipeline
described in ``docs/tessera-productivity-import-export-design.md``.
The Swift side calls into it via the HTTP API; the CLI is
the script-runner entry point used in unit tests and
ad-hoc terminal imports.

The top-level symbols are the ones the Swift side and the
CLI need; everything else is internal.
"""

from .ast_schema import (
    BLOCK_TYPES,
    ANNOTATION_TAGS,
    Block,
    DocumentAST,
    new_block_id,
)
from .intermediate import (
    IntermediateBlock,
    IntermediateDocument,
    IntermediateInlineRun,
)
from .data_layer_client import DataLayerClient, ImportResult, make_default_client
from .format_detector import Detection, Format, detect
from .pipeline import (
    ImportFailure,
    ImportPipeline,
    ImportSuccess,
    PipelineResult,
)
from .receipt_emitter import (
    ReceiptRecord,
    emit_export_receipt,
    emit_import_receipt,
    hash_bytes,
)

__all__ = [
    # ast_schema
    "BLOCK_TYPES",
    "ANNOTATION_TAGS",
    "Block",
    "DocumentAST",
    "IntermediateBlock",
    "IntermediateDocument",
    "IntermediateInlineRun",
    "new_block_id",
    # data layer
    "DataLayerClient",
    "ImportResult",
    "make_default_client",
    # format detector
    "Detection",
    "Format",
    "detect",
    # pipeline
    "ImportFailure",
    "ImportPipeline",
    "ImportSuccess",
    "PipelineResult",
    # receipts
    "ReceiptRecord",
    "emit_export_receipt",
    "emit_import_receipt",
    "hash_bytes",
]
