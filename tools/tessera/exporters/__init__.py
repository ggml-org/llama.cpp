"""Tessera productivity exporter (Phase 4).

The exporter is the inverse of the importer: a ``DocumentAST``
from the data layer is rendered to a file in the target
format. The architecture mirrors the importer's:

* ``ast_to_intermediate`` -- AST -> intermediate.
* ``builders/<format>`` -- per-format builders (markdown,
  html, email, docx, xlsx, pptx, pdf).
* ``pipeline`` -- orchestration (fetch, build, emit
  receipt).
* ``cli`` -- script-runner entry point.
"""

from . import ast_to_intermediate
from .pipeline import (
    ExportFailure,
    ExportPipeline,
    ExportResult,
    ExportSuccess,
    SUPPORTED_FORMATS,
    make_default_pipeline,
)

__all__ = [
    "ast_to_intermediate",
    "ExportFailure",
    "ExportPipeline",
    "ExportResult",
    "ExportSuccess",
    "SUPPORTED_FORMATS",
    "make_default_pipeline",
]
