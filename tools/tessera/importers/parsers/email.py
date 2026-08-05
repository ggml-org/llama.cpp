"""Email parser (EML, MSG, MBOX).

Reads .eml, .msg, and .mbox files using Python's stdlib
``mailbox`` and ``email`` modules. Returns one
``IntermediateDocument`` per message.

The v1 email importer is intentionally simple:

* Headers are stored in the document's ``meta`` dict (not in
  blocks). The agent can read them from the AST.
* The body is parsed as a single string. Plain-text bodies
  become one paragraph per line; HTML bodies are stripped to
  plain text via the stdlib ``html.parser``.
* Attachments are recorded as ``meta["attachments"]`` (a list
  of ``{"filename": ..., "content_type": ..., "size": ...}``).
  The bytes are NOT extracted in v1; the agent can re-fetch
  them from the original .eml/.mbox on demand.
* Each message is one document; multi-message .mbox files
  produce a list of documents. The CLI writes each as a
  separate ``email`` entity.

Punted (v1):

* Threading (In-Reply-To / References headers). Stored in
  ``meta`` but not surfaced as ``entity_links`` in v1.
* Calendar invites (.ics). Stored as attachments.
* S/MIME signature verification. Punted to v2.
"""

from __future__ import annotations

import email
import email.policy
import logging
import mailbox
import re
from email.message import EmailMessage
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ..intermediate import (
    IntermediateBlock,
    IntermediateDocument,
    IntermediateInlineRun,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EML / MSG (single message)
# ---------------------------------------------------------------------------


def parse_eml(path: Path) -> IntermediateDocument:
    """Parse a .eml or .msg file (single message).

    Uses the modern email policy (``EmailMessage``) so we get
    proper Unicode handling and structured access to multipart
    bodies.
    """
    log.debug("parse_eml: %s", path)
    with path.open("rb") as f:
        msg = email.message_from_binary_file(f, policy=email.policy.default)
    return _message_to_doc(msg)


def _message_to_doc(msg: EmailMessage) -> IntermediateDocument:
    """Convert an ``EmailMessage`` to an ``IntermediateDocument``."""
    meta: dict[str, Any] = {
        "from": _decode(msg.get("From", "")),
        "to": _decode(msg.get("To", "")),
        "cc": _decode(msg.get("Cc", "")),
        "subject": _decode(msg.get("Subject", "")),
        "date": _decode(msg.get("Date", "")),
        "message_id": _decode(msg.get("Message-ID", "")),
        "in_reply_to": _decode(msg.get("In-Reply-To", "")),
        "references": _decode(msg.get("References", "")),
        "headers": {k: _decode(v) for k, v in msg.items()},
    }

    body_text = _best_text_body(msg)
    blocks: list[IntermediateBlock] = []

    if meta["subject"]:
        blocks.append(
            IntermediateBlock(
                type="heading",
                attrs={"level": 1},
                runs=[IntermediateInlineRun(text=meta["subject"])],
            )
        )

    for line in body_text.splitlines():
        if not line.strip():
            continue
        blocks.append(
            IntermediateBlock(
                type="paragraph",
                runs=[IntermediateInlineRun(text=line)],
            )
        )

    # Attachments (no bytes in v1; just metadata).
    attachments: list[dict[str, Any]] = []
    for part in msg.walk():
        if part.is_attachment():
            attachments.append(
                {
                    "filename": part.get_filename(),
                    "content_type": part.get_content_type(),
                    "size": len(part.get_payload(decode=True) or b""),
                }
            )
    meta["attachments"] = attachments

    return IntermediateDocument(blocks=blocks, meta=meta)


def _best_text_body(msg: EmailMessage) -> str:
    """Return the best text/plain body, falling back to stripped text/html."""
    # First pass: text/plain part.
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            payload = part.get_content()
            if isinstance(payload, str):
                return payload
    # Second pass: strip text/html to plain text.
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            html = part.get_content()
            if isinstance(html, str):
                return _strip_html(html)
    return ""


class _HTMLStripper(HTMLParser):
    """Minimal HTML to plain text converter.

    Strips tags, decodes entities, and preserves block-level
    breaks as newlines. Sufficient for email body text; not a
    general HTML-to-text engine.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0  # inside <script> / <style>

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: ARG002
        if tag in ("script", "style"):
            self._skip_depth += 1
            return
        if tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in ("p", "div", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        # Collapse runs of whitespace within a line; preserve
        # paragraph breaks.
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.splitlines()]
        return "\n".join(ln for ln in lines if ln)


def _strip_html(html: str) -> str:
    s = _HTMLStripper()
    s.feed(html)
    return s.text()


def _decode(s: Any) -> str:
    """Decode an email header value to a plain string.

    email.policy.default decodes encoded-word sequences for us,
    so most headers are already unicode. We coerce to str and
    strip whitespace.
    """
    if s is None:
        return ""
    if isinstance(s, bytes):
        return s.decode("utf-8", errors="replace").strip()
    return str(s).strip()


# ---------------------------------------------------------------------------
# MBOX (multiple messages)
# ---------------------------------------------------------------------------


def parse_mbox(path: Path) -> list[IntermediateDocument]:
    """Parse a .mbox file (Apple Mail export) into a list of IntermediateDocuments.

    ``mailbox.mbox`` returns a mailbox object you can iterate; each
    item is an ``mboxMessage`` (a subclass of ``email.message.Message``).
    We convert each to an ``EmailMessage`` via the modern policy so
    the header decoding is consistent with ``parse_eml``.
    """
    log.debug("parse_mbox: %s", path)
    mbox = mailbox.mbox(str(path))
    docs: list[IntermediateDocument] = []
    for msg in mbox:
        # Convert the legacy mboxMessage to a modern EmailMessage
        # with the default policy so we get the same header
        # decoding as parse_eml.
        payload = msg.as_bytes()
        try:
            modern = email.message_from_bytes(payload, policy=email.policy.default)
        except Exception as e:  # noqa: BLE001
            log.warning("parse_mbox: skip msg: %s: %s", path, e)
            continue
        if not isinstance(modern, EmailMessage):
            continue
        docs.append(_message_to_doc(modern))
    mbox.close()
    return docs


def parse_msg(path: Path) -> IntermediateDocument:
    """Parse a .msg file (Outlook format).

    .msg is a Microsoft OLE format. Python's stdlib doesn't
    parse it natively, so we treat .msg as opaque for v1 and
    log a warning. v2 will vendor the ``olefile`` library or
    the ``extract-msg`` package and re-emit the message body.

    For now we delegate to ``parse_eml`` when the file happens
    to be a valid RFC 5322 message (some tools save .msg as
    RFC 5322 text); otherwise we return a document with a
    "format not supported" stub.
    """
    log.debug("parse_msg: %s", path)
    try:
        return parse_eml(path)
    except Exception as e:  # noqa: BLE001
        log.warning("parse_msg: %s: %s", path, e)
        return IntermediateDocument(
            blocks=[
                IntermediateBlock(
                    type="paragraph",
                    runs=[
                        IntermediateInlineRun(
                            text=f"(could not parse .msg file: {e})"
                        )
                    ],
                )
            ],
            meta={"format_error": str(e)},
        )
