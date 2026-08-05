"""PDF builder (exporter).

Converts an ``IntermediateDocument`` to a .pdf file.

In v1 the PDF builder renders to HTML via the ``html``
builder and then converts the HTML to PDF via
``weasyprint``. The HTML pass is intentional: HTML is the
canonical rendering for the productivity surface, and
``weasyprint`` produces a deterministic PDF that is
byte-equivalent for byte-equivalent HTML input. The
production path on macOS uses ``PDFKit`` via a Swift
shim (per the spec's §11.1); weasyprint is the
cross-platform path used by the unit tests and the
Linux build.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tools.tessera.importers.intermediate import IntermediateDocument
from .html import build_html

log = logging.getLogger(__name__)


def build_pdf(doc: IntermediateDocument, output: Path) -> None:
    """Render an IntermediateDocument to a .pdf file at `output`."""
    html = build_html(doc)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        from weasyprint import HTML

        HTML(string=html).write_pdf(str(output))
    except ImportError as e:
        raise RuntimeError(
            "weasyprint is not installed; install it with `pip install weasyprint`"
        ) from e
