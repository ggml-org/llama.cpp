"""HTTP client for the data layer.

The Python importer / exporter talks to the Swift app's HTTP
API (``TesseraStudio/Sources/TesseraStudioMac/API/ImportExportAPI.swift``)
rather than directly to Postgres. The HTTP boundary is the
seam that keeps "no raw SQL in Python" an invariant.

API contract (mirrors what the Swift side exposes):

* ``POST /v1/import`` with a multipart body: the file under
  ``file`` plus an optional ``format_hint`` form field. Returns
  ``{"entity_id": "..."}`` on success. The Swift side runs
  the actual import pipeline (calls back into Python via the
  CLI) and persists the result.

  Why a CLI in addition to HTTP? The CLI is the unit-test
  surface; the HTTP API is the user-facing seam. The Swift
  side shells out to ``python -m tools.tessera.importers.cli``
  for unit-testable work and exposes the same pipeline over
  HTTP for the macOS / iOS apps. v1 uses the CLI directly;
  the HTTP path is used when the data layer is on a different
  host (test environment, future cloud) or when the import is
  triggered by a non-app client (e.g. the watch folder).

* ``POST /v1/export`` with a JSON body: ``{"entity_id": "...",
  "format": "..."}``. Returns the file bytes with a
  ``Content-Disposition: attachment`` header.

The client is a thin ``httpx`` wrapper. It supports dry-run
mode (``dry_run=True``) where it returns a synthetic UUID
without making a network call; that's what the unit tests use.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass
class ImportResult:
    """The result of a successful import.

    `entity_id` is the UUID of the new ``graph_entity``.
    `entity_type` is what the data layer recorded (e.g.
    "document" / "email" / "sheet"). `body` is the AST as a
    JSON string (when ``return_body=True`` was passed to the
    client; otherwise None).
    """

    entity_id: str
    entity_type: str
    body: Optional[str] = None


class DataLayerClient:
    """HTTP client for the Tessera data layer.

    `base_url` is the Swift HTTP API root (default
    ``http://127.0.0.1:8787``). `dry_run=True` makes every
    call a no-op that returns a synthetic UUID; that's the
    mode the unit tests and the import CLI use when no data
    layer is available (CI runs, offline development).
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8787",
        *,
        dry_run: bool = False,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.dry_run = dry_run
        self.timeout = timeout
        # The httpx import is deferred so the module is
        # importable in environments without httpx (the venv
        # test for "import everything" doesn't need it).
        self._client: Any = None

    def _http(self) -> Any:
        if self._client is None:
            import httpx  # type: ignore[import-not-found]

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    # -----------------------------------------------------------------------
    # Import
    # -----------------------------------------------------------------------

    def create_entity(
        self,
        entity_type: str,
        label: str,
        body: str,
        *,
        source_url: Optional[str] = None,
        subtype: Optional[str] = None,
    ) -> ImportResult:
        """Create a new ``graph_entity`` row.

        The Swift side maps this to ``TesseraDataLayer.upsertEntity``.
        `body` is the AST as a JSON string; the Swift side stores
        it as JSONB.
        """
        if self.dry_run:
            return ImportResult(
                entity_id=str(uuid.uuid4()),
                entity_type=entity_type,
                body=body,
            )

        client = self._http()
        payload: dict[str, Any] = {
            "entity_type": entity_type,
            "label": label,
            "body": body,
        }
        if source_url is not None:
            payload["source_url"] = source_url
        if subtype is not None:
            payload["subtype"] = subtype
        resp = client.post(f"{self.base_url}/v1/entities", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return ImportResult(
            entity_id=str(data["entity_id"]),
            entity_type=str(data.get("entity_type", entity_type)),
        )

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    def get_entity_body(self, entity_id: str) -> Optional[str]:
        """Return the body of a ``graph_entity``, or None if not found.

        Used by the exporter to fetch the AST JSON for an
        entity id. The Swift side maps this to
        ``TesseraDataLayer.getEntity(id:)``.
        """
        if self.dry_run:
            return None
        client = self._http()
        resp = client.get(f"{self.base_url}/v1/entities/{entity_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.text

    def get_entity_meta(self, entity_id: str) -> dict[str, Any]:
        """Return the metadata of a ``graph_entity`` (label, type, etc.)."""
        if self.dry_run:
            return {
                "entity_id": entity_id,
                "entity_type": "document",
                "label": "(dry-run)",
            }
        client = self._http()
        resp = client.get(f"{self.base_url}/v1/entities/{entity_id}/meta")
        resp.raise_for_status()
        return dict(resp.json())

    # -----------------------------------------------------------------------
    # Receipts
    # -----------------------------------------------------------------------

    def append_receipt(
        self,
        entity_id: str,
        receipt_type: str,
        payload: dict[str, Any],
        signature: Optional[str] = None,
    ) -> str:
        """Append a receipt to the chain and return its id.

        The Swift side calls ``TesseraDataLayer.appendReceiptToChain``.
        `signature` is the ed25519 signature as a base64 string
        (the Swift signer produces it; for v1 the Python
        importer signs in-process and passes the result here).
        """
        if self.dry_run:
            return str(uuid.uuid4())
        client = self._http()
        body: dict[str, Any] = {
            "entity_id": entity_id,
            "receipt_type": receipt_type,
            "payload": payload,
        }
        if signature is not None:
            body["signature"] = signature
        resp = client.post(f"{self.base_url}/v1/receipts", json=body)
        resp.raise_for_status()
        return str(resp.json()["receipt_id"])

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None


def make_default_client() -> DataLayerClient:
    """Build the default client for the current process.

    Honours the ``TESSERA_DATA_LAYER_URL`` env var
    (default ``http://127.0.0.1:8787``) and
    ``TESSERA_DATA_LAYER_DRY_RUN`` (default: ``1`` when
    unset, so the importer is testable without a running data
    layer). The Swift app's startup sets
    ``TESSERA_DATA_LAYER_DRY_RUN=0`` before invoking the
    subprocess.
    """
    base = os.environ.get("TESSERA_DATA_LAYER_URL", "http://127.0.0.1:8787")
    dry = os.environ.get("TESSERA_DATA_LAYER_DRY_RUN", "1") not in ("0", "false", "False")
    return DataLayerClient(base_url=base, dry_run=dry)
