"""Format-specific parsers for the importer.

Each parser is a module that exposes a ``parse_<format>``
function (or ``parse_<format>`` plus helpers) and produces
one or more ``IntermediateDocument``s. The pipeline's
``ImportPipeline._parse`` dispatches based on the detected
format.

The parsers are independent of the AST schema: they
produce ``IntermediateDocument`` and ``IntermediateBlock``
shapes (see ``intermediate.py``), and the AST builder
(``ast_builder.py``) promotes them to ``DocumentAST``.

Adding a new format:

1. Add the format to ``format_detector.Format``.
2. Add an extension mapping in ``format_detector.EXTENSION_MAP``.
3. Add the parser module under ``parsers/`` with a
   ``parse_<format>(path)`` function.
4. Wire it into ``ImportPipeline._parse``.

Punted-on formats (v1):

* `.doc` (legacy Word) — best handled via Pandoc.
* `.rtf` — handled by Pandoc.
* `.odt` / `.epub` — handled by Pandoc.
* `.ics` (calendar) — v2.
"""

from . import docx, email, html, markdown, pandoc, pdf, pptx, xlsx

__all__ = ["docx", "email", "html", "markdown", "pandoc", "pdf", "pptx", "xlsx"]
