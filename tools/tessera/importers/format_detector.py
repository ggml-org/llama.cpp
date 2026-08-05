"""Detect the format of a file to import.

The detector tries (in order):

1. **Magic bytes** — the most reliable signal. ZIP/OOXML files
   start with ``PK\\x03\\x04``, PDFs with ``%PDF-``, MHTML starts
   with ``MIME-Version:``, EML and MBOX are plain text starting
   with a header line, and so on.
2. **File extension** — secondary signal. Covers the cases the
   magic-bytes check misses (Markdown, HTML, email-with-extension,
   JSON, etc.).
3. **Filename-only fallback** — last resort. We never throw on
   "unknown" because the importer has a Pandoc bridge that accepts
   any input Pandoc can read; we just tag it as ``"pandoc"`` and
   let Pandoc take a swing at it.

The detector returns a ``Format`` enum-like value. The importer
pipeline uses it to pick a parser; the parser's ``accepts()`` method
re-validates the choice (e.g. an .html file containing bytes that
look like MHTML is re-routed to the MHTML parser).

Why a separate module:

* Single point of truth for "what format is this file?". The
  detector is cheap (one syscall + a peek) and is called before
  any heavy import work; the failure case is "unknown format",
  which is handled by falling through to Pandoc.
* The detector is pure-Python and has no third-party deps (only
  the stdlib ``pathlib`` and ``zipfile``-compatible reads). This
  matters because the detector is called on every import attempt
  including ones that fail to open the file at all.
"""

from __future__ import annotations

import enum
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format enum
# ---------------------------------------------------------------------------


class Format(str, enum.Enum):
    """The set of formats the importer v1 understands.

    Each value is the format's wire name (matches the file extension
    when one is canonical). The `pandoc` value is the fallback
    catch-all for anything the dedicated parsers don't recognise;
    it's routed to the Pandoc bridge.

    String enum so the JSON CLI output is the lowercase tag.
    """

    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    PDF = "pdf"
    EML = "eml"
    MSG = "msg"
    MBOX = "mbox"
    HTML = "html"
    MHTML = "mhtml"
    MARKDOWN = "md"
    PANDOC = "pandoc"  # fallback for anything else


# Map file extensions to Format. Lower-cased. The detector tries the
# extension first when the magic-bytes check is ambiguous.
EXTENSION_MAP: dict[str, Format] = {
    ".docx": Format.DOCX,
    ".xlsx": Format.XLSX,
    ".pptx": Format.PPTX,
    ".pdf": Format.PDF,
    ".eml": Format.EML,
    ".msg": Format.MSG,
    ".mbox": Format.MBOX,
    ".html": Format.HTML,
    ".htm": Format.HTML,
    ".mhtml": Format.MHTML,
    ".mht": Format.MHTML,
    ".md": Format.MARKDOWN,
    ".markdown": Format.MARKDOWN,
}


@dataclass(frozen=True)
class Detection:
    """The result of a detection call.

    `format` is the detected format. `via` is the detection path
    that produced the result, used for telemetry + logs. `hint` is
    an optional extra note (e.g. "fallback: no extension match").
    """

    format: Format
    via: str
    hint: Optional[str] = None


# ---------------------------------------------------------------------------
# Magic-byte signatures
# ---------------------------------------------------------------------------

# We read the first N bytes; the constant is sized to cover the
# longest signature we care about (DOCX/XLSX/PPTX all have
# signature-like strings inside the ZIP central directory, but
# the 4-byte ZIP header is enough for OOXML). MHTML and EML need a
# little more text to be unambiguous.
_HEADER_SIZE = 4096


def _read_header(path: Path) -> bytes:
    """Return the first ``_HEADER_SIZE`` bytes, never raising on read error."""
    try:
        with path.open("rb") as f:
            return f.read(_HEADER_SIZE)
    except OSError as e:
        log.debug("read_header: %s: %s", path, e)
        return b""


# ZIP / OOXML. .docx, .xlsx, .pptx, and the .jar-style OOXML
# extensions all start with the same 4 bytes.
_ZIP_MAGIC = b"PK\x03\x04"

# PDF starts with the %PDF- magic.
_PDF_MAGIC = b"%PDF-"

# MHTML starts with the multipart MIME header.
_MHTML_MAGIC = b"MIME-Version:"

# EML starts with a "From " line (per RFC 5322), or any other
# header line; the canonical sentinel is "Return-Path:" or
# "From ". We check for the "From " prefix as the most common.
_EML_MAGIC_PREFIX = b"From "

# MBOX is a sequence of "From " lines; the file starts with "From ".
# Distinguishing EML from MBOX by magic is unreliable, so we
# also use the extension. The detector falls through to the
# extension when the magic is ambiguous.
_MBOX_FIRST_LINE = re.compile(rb"^From [^\n]+\n(?:[A-Z][\w-]+:)", re.MULTILINE)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect(path: Path) -> Detection:
    """Detect the format of `path`.

    Tries magic bytes first, then extension, then falls through to
    the Pandoc catch-all. Never throws; an unreadable file returns
    ``Format.PANDOC`` with a "fallback" note so the importer at
    least gets a chance to produce a useful error.
    """
    header = _read_header(path)

    # 1) Magic bytes --------------------------------------------------------
    if header.startswith(_PDF_MAGIC):
        return Detection(Format.PDF, via="magic", hint="PDF header")

    if header.startswith(_MHTML_MAGIC):
        return Detection(Format.MHTML, via="magic", hint="MIME-Version header")

    if header.startswith(_ZIP_MAGIC):
        # ZIP could be DOCX, XLSX, PPTX, or a plain ZIP. Probe the
        # ZIP's content-type marker to disambiguate; fall back to
        # the extension.
        kind = _zip_kind(path, header)
        if kind is not None:
            return Detection(kind, via="magic-zip", hint="[Content_Types].xml marker")
        ext_kind = _ext_kind(path)
        if ext_kind in (Format.DOCX, Format.XLSX, Format.PPTX):
            return Detection(ext_kind, via="extension", hint="ZIP, extension match")
        return Detection(Format.PANDOC, via="fallback", hint="ZIP, no Office marker")

    if header.startswith(_EML_MAGIC_PREFIX):
        # EML vs MBOX: MBOX is also a sequence of "From " lines,
        # but a single-message EML file has only one "From " line
        # and a non-zero count of header lines below it. We treat
        # MBOX as "more than one From-line OR explicit .mbox".
        mbox_count = len(re.findall(rb"^From [^\n]+\n", header))
        ext_kind = _ext_kind(path)
        if ext_kind is Format.MBOX or mbox_count >= 2:
            return Detection(Format.MBOX, via="magic-mbox", hint=f"{mbox_count} From-lines")
        return Detection(Format.EML, via="magic", hint="single From-line")

    # 2) Extension ---------------------------------------------------------
    ext_kind = _ext_kind(path)
    if ext_kind is not None and header:
        return Detection(ext_kind, via="extension", hint="ext match")
    if ext_kind is not None:
        # The file is missing or unreadable but has a known
        # extension. Report the extension's format so the
        # importer's error path is well-typed.
        return Detection(ext_kind, via="extension", hint="ext match (file unreadable)")

    # 3) Fallback: Pandoc --------------------------------------------------
    return Detection(Format.PANDOC, via="fallback", hint="no ext / no magic")


def _ext_kind(path: Path) -> Optional[Format]:
    return EXTENSION_MAP.get(path.suffix.lower())


def _zip_kind(path: Path, header: bytes) -> Optional[Format]:
    """Return the Format of an OOXML ZIP, or None if it isn't Office.

    Reads the ZIP's central directory looking for the Office-specific
    ``[Content_Types].xml`` content type. The heuristic:

    * ``wordprocessingml`` => DOCX
    * ``spreadsheetml`` => XLSX
    * ``presentationml`` => PPTX

    Falls back to ``None`` when the ZIP can't be opened or the
    content type isn't found.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            # Read [Content_Types].xml if present; that's the OOXML
            # canonical way to identify the format.
            if "[Content_Types].xml" in names:
                with zf.open("[Content_Types].xml") as f:
                    content = f.read(4096).decode("utf-8", errors="replace")
                if "wordprocessingml" in content:
                    return Format.DOCX
                if "spreadsheetml" in content:
                    return Format.XLSX
                if "presentationml" in content:
                    return Format.PPTX
            # No [Content_Types].xml - might be a non-Office ZIP.
            # The extension is a reasonable fallback at this point.
            for name in names:
                if name.startswith("word/") and name.endswith(".xml"):
                    return Format.DOCX
                if name.startswith("xl/") and name.endswith(".xml"):
                    return Format.XLSX
                if name.startswith("ppt/") and name.endswith(".xml"):
                    return Format.PPTX
    except (zipfile.BadZipFile, OSError) as e:
        log.debug("zip_kind: %s: %s", path, e)
    return None
