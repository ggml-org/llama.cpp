"""EML builder (exporter).

Converts an ``IntermediateDocument`` to an RFC 5322 EML string
suitable for hand-off to Apple Mail (via the share sheet) or
for direct file export.

The exporter takes the document's meta (which the email
importer populated with ``from`` / ``to`` / ``subject`` / ``date``
/ etc.) and the body's blocks (paragraphs and headings) and
emits a single-part text/plain message. Multi-part / HTML
bodies are punted to v2.

The exporter is the inverse of the email importer: an email
imported and re-exported produces an EML that round-trips
through the importer again with the same headers and an
equivalent body.
"""

from __future__ import annotations

import logging
from email.message import EmailMessage
from email.utils import format_datetime
from datetime import datetime, timezone

from tools.tessera.importers.intermediate import (
    IntermediateBlock,
    IntermediateDocument,
)

log = logging.getLogger(__name__)


def build_eml(doc: IntermediateDocument) -> str:
    """Render an IntermediateDocument to an EML string."""
    msg = EmailMessage()
    meta = doc.meta
    msg["From"] = str(meta.get("from", "unknown@example.com"))
    msg["To"] = str(meta.get("to", ""))
    if meta.get("cc"):
        msg["Cc"] = str(meta["cc"])
    msg["Subject"] = str(meta.get("subject", "(no subject)"))
    if meta.get("date"):
        try:
            # ``format_datetime`` parses common RFC 2822 / 5322
            # forms. If it can't parse, fall through to the raw
            # string.
            msg["Date"] = str(meta["date"])
        except (TypeError, ValueError):
            pass
    else:
        msg["Date"] = format_datetime(datetime.now(timezone.utc))

    body = _render_body(doc)
    msg.set_content(body)
    return msg.as_string()


def _render_body(doc: IntermediateDocument) -> str:
    """Render the body blocks to a plain-text string."""
    parts: list[str] = []
    for b in doc.blocks:
        if b.type == "heading":
            parts.append("".join(r.text for r in b.runs))
            parts.append("")  # blank line after heading
        elif b.type == "paragraph":
            parts.append("".join(r.text for r in b.runs))
        elif b.type == "divider":
            parts.append("---")
        else:
            parts.append("".join(r.text for r in b.runs))
    return "\n".join(parts).strip() + "\n"
