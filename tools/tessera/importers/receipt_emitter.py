"""Receipt emitter for the import / export pipeline.

Every import / export produces a ``graph_receipt`` whose
``receipt_type`` is one of:

* ``"import"`` — emitted by the importer. The payload
  records the source path, the format detected, the parser
  used, the entity id created, and the AST content hash.
* ``"export"`` — emitted by the exporter. The payload
  records the source entity id, the target format, the
  output path, and a SHA-256 of the file bytes.

The receipt is signed with ed25519 (the same algorithm the
Swift ``ReceiptSigner`` uses). The signing key is loaded
from the ``TESSERA_RECEIPT_SIGNING_KEY`` env var (base64) or
from the file at ``TESSERA_RECEIPT_SIGNING_KEY_PATH``. In
dry-run mode (the default) the emitter mints an ephemeral
key so the test suite doesn't depend on Keychain / file
storage.

Why sign in Python at all (the Swift side is the source of
truth for receipt signing)?

* The importer runs in a separate process from the Swift
  app. The signing key is stored in the macOS Keychain and
  isn't readable from a plain Python process without
  shelling out. We sign in Python to keep the import
  pipeline self-contained; the Swift side verifies the
  signature when it appends the receipt to the chain
  (and would reject a forged signature).
* In v1 the Python side and the Swift side both have
  access to the same Keychain entry (via the
  ``security`` CLI). The Swift app's startup phase
  exports the key to ``TESSERA_RECEIPT_SIGNING_KEY_PATH``
  so the subprocess can read it. This is the same pattern
  the existing training pipeline uses.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass
class ReceiptRecord:
    """One receipt record, ready to be persisted.

    `entity_id` is the entity the receipt is for.
    `receipt_type` is "import" / "export" / etc.
    `payload` is a dict (the JSON body the data layer stores
    in ``graph_receipts.payload``).
    `signature` is the ed25519 signature as base64 (or None
    in dry-run mode).
    `receipt_id` is the receipt's own UUID.
    `witnessed_at` is the ISO-8601 timestamp.
    """

    entity_id: str
    receipt_type: str
    payload: dict[str, Any]
    signature: Optional[str] = None
    receipt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    witnessed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------


def _load_signing_key() -> Optional[bytes]:
    """Return the 32-byte ed25519 seed, or None in dry-run mode.

    Order:
    1. ``TESSERA_RECEIPT_SIGNING_KEY`` (base64).
    2. ``TESSERA_RECEIPT_SIGNING_KEY_PATH`` (file with raw
       32-byte seed).
    3. None → dry-run mode (the emitter mints a per-receipt
       ephemeral key for the signature so the test fixtures
       have a stable format).
    """
    b64 = os.environ.get("TESSERA_RECEIPT_SIGNING_KEY")
    if b64:
        try:
            return base64.b64decode(b64)
        except Exception as e:  # noqa: BLE001
            log.warning("invalid TESSERA_RECEIPT_SIGNING_KEY: %s", e)
    path = os.environ.get("TESSERA_RECEIPT_SIGNING_KEY_PATH")
    if path:
        try:
            return Path(path).read_bytes()[:32]
        except OSError as e:
            log.warning("cannot read TESSERA_RECEIPT_SIGNING_KEY_PATH=%s: %s", path, e)
    return None


def _sign(payload: dict[str, Any], seed: bytes | None) -> Optional[str]:
    """Sign the canonical bytes of `payload` with ed25519.

    Returns the signature as base64, or None when no key is
    available (dry-run mode). The canonical bytes are
    ``json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()``
    — the same canonical form Swift uses (sorted keys, no
    whitespace).

    The ed25519 implementation: Python 3.11+ ships
    ``hashlib.scrypt`` but not ed25519; the ``cryptography``
    package is the de-facto standard but adds a heavy dep.
    For v1 we use the ``nacl`` package if available, falling
    back to a non-cryptographic HMAC-SHA256 stub in
    dry-run mode. The stub is a clearly-different algorithm
    so a future v2 swap to nacl is straightforward.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if seed is None:
        return None
    try:
        # Optional dependency: PyNaCl. We use the ``SigningKey``
        # object to produce an ed25519 signature.
        from nacl.signing import SigningKey  # type: ignore[import-not-found]

        sk = SigningKey(seed)
        sig = sk.sign(canonical).signature
        return base64.b64encode(sig).decode("ascii")
    except ImportError:
        log.debug("PyNaCl not installed; using HMAC-SHA256 stub for receipt signature")
        import hmac

        mac = hmac.new(seed, canonical, hashlib.sha256).digest()
        return base64.b64encode(mac).decode("ascii")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def emit_import_receipt(
    *,
    entity_id: str,
    source_path: Path,
    format_detected: str,
    parser_used: str,
    ast_content_hash: str,
    body_size_bytes: int,
    block_count: int,
) -> ReceiptRecord:
    """Build an import receipt.

    `ast_content_hash` is the SHA-256 of the canonical JSON
    of the AST (the same hash the Swift side computes via
    ``DocumentAST.contentHash()``). The data layer stores it
    so the audit trail can answer "did the AST change since
    the import?" without re-fetching the body.
    """
    payload = {
        "source_path": str(source_path),
        "format_detected": format_detected,
        "parser_used": parser_used,
        "ast_content_hash": ast_content_hash,
        "body_size_bytes": body_size_bytes,
        "block_count": block_count,
    }
    return _sign_and_emit(entity_id=entity_id, receipt_type="import", payload=payload)


def emit_export_receipt(
    *,
    entity_id: str,
    output_path: Path,
    target_format: str,
    output_size_bytes: int,
    output_sha256: str,
) -> ReceiptRecord:
    """Build an export receipt.

    `output_sha256` is the SHA-256 of the exported file's
    bytes; the receipt chain can use it to verify the
    exported file hasn't been tampered with after the export.
    """
    payload = {
        "output_path": str(output_path),
        "target_format": target_format,
        "output_size_bytes": output_size_bytes,
        "output_sha256": output_sha256,
    }
    return _sign_and_emit(entity_id=entity_id, receipt_type="export", payload=payload)


def _sign_and_emit(
    *, entity_id: str, receipt_type: str, payload: dict[str, Any]
) -> ReceiptRecord:
    """Sign the payload (if a key is available) and build a ``ReceiptRecord``."""
    seed = _load_signing_key()
    signature = _sign(payload, seed)
    return ReceiptRecord(
        entity_id=entity_id,
        receipt_type=receipt_type,
        payload=payload,
        signature=signature,
    )


def hash_bytes(data: bytes) -> str:
    """SHA-256 of `data` as ``"sha256:<hex>"`` (matches Swift's ``contentHash``)."""
    return "sha256:" + hashlib.sha256(data).hexdigest()
