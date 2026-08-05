"""Per-format builders for the exporter.

Each builder takes an ``IntermediateDocument`` (or a list of
them, for PPTX) and writes a file in the target format. The
builders are independent of the data layer and the pipeline;
they're testable in isolation.

Adding a new format:

1. Add the format string to ``pipeline.SUPPORTED_FORMATS``.
2. Add the builder module here.
3. Add a dispatch case in ``pipeline._build``.
4. Add the ``--format`` choice to ``cli.py``.
"""

from . import docx, email, html, markdown, pdf, pptx, xlsx

__all__ = ["docx", "email", "html", "markdown", "pdf", "pptx", "xlsx"]
