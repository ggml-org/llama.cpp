# Tessera Productivity Import / Export — Design Specification

**Status:** Draft v1 (architect review pending)
**Author:** Tessera Architecture
**Date:** 2026-08-05
**Branch:** `feat/prod-import-export`
**Companion:** `docs/tessera-productivity-design.md` (the productivity surface spec), `docs/tessera-productivity-foundations-design.md` (Phase 1, on which this builds)
**Sister specs:** `docs/tessera-data-layer-design.md`, `docs/tessera-plead-the-fifth-design.md`

---

## 1. Problem

Tessera's productivity surface (the WYSIWYG editor + chat panel + receipt drawer from `docs/tessera-productivity-design.md`) needs to bring external documents into the app and emit them back out. The scope of this Phase 4 design is the importer / exporter pipeline:

* In: DOCX, XLSX, PPTX, PDF, EML, MBOX, HTML, MHTML, Markdown, anything else via Pandoc.
* Out: PDF, DOCX, XLSX, PPTX, HTML, Markdown, EML, plus the system share sheet (Mail, Messages, AirDrop, ...) and the Slack webhook (because Slack doesn't show up reliably in the share sheet).

The architecture is the canonical AST (the Block AST from `docs/tessera-productivity-foundations-design.md` §3) with Python parsers on one side and Python builders on the other. The Swift side provides the actors that the UI calls, the HTTP API the Python CLI talks to, and the share-sheet integration.

This design follows the spec's §10 (Importers) and §11 (Exporters) and is the v1 deliverable for Phase 4.

---

## 2. Why this design

| Choice | Rationale |
|---|---|
| **Python for the format bridge** | The mature libraries (python-docx, openpyxl, python-pptx, weasyprint, beautifulsoup4, mailbox, markdown-it-py) are all Python. Re-implementing the format parsers in Swift would multiply the LoC and the maintenance burden without buying us anything the Python side doesn't already give us. |
| **Canonical AST in Swift, format work in Python** | The Block AST is the source of truth for the document. The Python side translates in and out; it never holds the canonical form. The receipt chain is built on the AST, not on the imported file, so a re-import can be verified. |
| **CLI as the script-runner entry point** | The CLI is the unit-test boundary and the ad-hoc user path. The HTTP API wraps the same pipeline; the Swift UI calls into it. The CLI and the API share the same code; one of them being the "primary" surface is a matter of taste. |
| **No raw SQL in Python** | Python talks to the data layer via HTTP. The Swift side is the only consumer of Postgres / Valkey. The boundary is the seam that keeps "no raw SQL past `TesseraDataLayer`" an invariant (per `docs/tessera-data-layer-design.md` §6). |
| **Pandoc as the swiss-army bridge** | Pandoc handles every format the dedicated parsers don't cover. The architecture: AST → Pandoc JSON → external format, and vice versa. The dedicated parsers are preferred because they preserve more structure; Pandoc is the fallback. |
| **System share sheet over per-service integration** | `NSSharingServicePicker` (macOS) and `UIActivityViewController` (iOS) cover Mail, Messages, AirDrop, and any user-installed share extension. Building per-service APIs is out of scope for v1. The exception is Slack (which doesn't show up reliably) and generic webhooks. |
| **Slack webhook, not Slack API** | The Slack webhook URL is a single secret stored in Keychain. The Slack app / OAuth / file uploads are out of scope for v1. A webhook POST is the smallest viable Slack integration that gets the user's text into a channel. |
| **Receipt-signing key shared between Swift and Python** | The receipt is the source of truth for the audit trail. Both the Swift side (when the user makes an edit) and the Python side (when the user imports a file) need to sign. We share the same ed25519 key (derived from the volume password, per `docs/tessera-productivity-foundations-design.md` §5). |
| **Determinism where possible** | The Markdown / HTML builders produce canonical output (sorted attribute keys, no trailing whitespace). This makes the export diffable: a round-trip import → export → import produces ASTs whose content hashes match modulo the format drift the AST builder notes. |
| **No third-party Swift deps** | The HTTP server is built on Apple's Network framework. The Slack webhook is a `URLSession` call. No SwiftNIO, no Alamofire. The macOS app's binary stays small. |

---

## 3. The importer pipeline

The importer is a Python package (`tools/tessera/importers/`) that turns external files into the Block AST, persists them as `graph_entity` rows, and emits a `graph_receipt`.

### 3.1 The flow

```
external file
  └─ format_detector.detect(path)         # magic bytes + extension
      └─ parsers.<format>.parse(path)     # intermediate JSON
          └─ ast_builder.build()          # Block AST
              └─ data_layer_client.create_entity()    # graph_entity
                  └─ receipt_emitter.append_receipt()  # graph_receipt
```

Each step is idempotent. The pipeline is configured per call (the data-layer client + the media directory + a fail-fast flag) and returns a `PipelineResult` with one `ImportSuccess` per file and one `ImportFailure` per failure.

### 3.2 The intermediate shape

The intermediate is a value type that lives between the parsers and the AST builder. The parsers produce it; the AST builder consumes it. The shape is a flat list of `IntermediateBlock`s, each with a list of `IntermediateInlineRun`s, plus optional child blocks for containers (lists, tables, toggles). It's deliberately simple: the unit-test boundary is the intermediate, not the AST.

Why an intermediate (and not just have the parsers produce the AST directly)?

* The intermediate is the test boundary. Tests can verify "DOCX tables become table blocks" by inspecting the intermediate without dragging the AST module in.
* Multiple parsers produce the same intermediate shape. The email parser and the MBOX parser both produce email-flavored intermediates; the AST builder doesn't care.
* The intermediate is the format the builders consume. The Markdown / HTML / DOCX builders all consume the intermediate; the AST is a higher-level abstraction the data layer persists.

### 3.3 The AST JSON wire format

The AST is serialised to JSON for the data layer. The JSON shape is the contract between Python and Swift; a regression here breaks the wire format and would fail to round-trip in the app.

```json
{
  "blocks": {
    "<uuid>": {
      "id": "<uuid>",
      "type": "heading",
      "attributes": {"level": 1},
      "content": [{"text": "Hello", "annotations": []}],
      "children": [],
      "parentID": null
    }
  },
  "rootChildren": ["<uuid>"]
}
```

The Swift `DocumentAST` decoder (`TesseraStudio/Sources/TesseraCore/Productivity/Block.swift`) reads this exact shape. The `blocks` map uses stringified UUID keys (Swift's `Dictionary<UUID, V>` doesn't decode from a JSON object directly; the custom decoder converts at the boundary). The `rootChildren` list is an ordered array of UUID strings.

### 3.4 PDF: a two-pass approach

The PDF parser uses `pdftotext -layout` (poppler) to extract the text and `weasyprint` to render HTML→PDF on the export side. We do NOT use weasyprint on the import side because weasyprint is an HTML→PDF renderer, not a PDF parser; the import side needs text extraction, which `pdftotext` does well.

Limitations:

* Scanned PDFs (image-only) produce no output. OCR is a v2 feature (per the spec's §10.4 "Punted on").
* Multi-column layouts are flattened. v2 may use `pdfplumber` to detect columns.
* Tables are best-effort. Tab-separated rows are detected heuristically; the v1 parser handles simple grids.

Install steps for the dev machine (also documented in `tools/tessera/requirements-import-export.txt`):

    # macOS
    brew install pandoc poppler

    # Linux
    apt-get install pandoc poppler-utils

The importer checks for `pdftotext` on the PATH; when missing, it returns a "PDF parse failed" block (a single paragraph with the install instructions) rather than failing the import.

### 3.5 The Pandoc bridge

For formats the dedicated parsers don't cover (RST, LaTeX, ODT, EPUB, RTF, Org-mode, ...), we delegate to Pandoc. The architecture is AST → Pandoc JSON → external format, but the importer is the inverse direction:

1. Shell out to `pandoc -f <input-format> -t json` to get the Pandoc JSON AST.
2. Walk the JSON AST and produce an `IntermediateDocument`.
3. The AST builder turns the intermediate into the Block AST.

The walker is intentionally minimal: it handles the common constructs (headings, paragraphs, lists, tables, code blocks, quotes, links, emphasis) and silently drops the rest. This is the same pragmatic stance the dedicated parsers take.

### 3.6 The receipt

Every import produces a `graph_receipt` whose `receipt_type` is `"import"`. The payload records:

* `source_path` — the original file path.
* `format_detected` — the format the detector decided.
* `parser_used` — the parser that handled the file (`"python-docx"`, `"pdftotext"`, `"pandoc"`, etc.).
* `ast_content_hash` — the SHA-256 of the canonical JSON of the AST, prefixed with `"sha256:"` (matches Swift's `contentHash()`).
* `body_size_bytes` — the AST's body size in UTF-8 bytes.
* `block_count` — the number of blocks in the AST.

The receipt is signed with ed25519 (the same algorithm Swift's `ReceiptSigner` uses). The signing key is loaded from the `TESSERA_RECEIPT_SIGNING_KEY` env var (base64) or `TESSERA_RECEIPT_SIGNING_KEY_PATH` (file with raw 32-byte seed). When no key is available (dry-run mode), the emitter mints a per-receipt ephemeral key so the test fixtures have a stable format.

---

## 4. The exporter pipeline

The exporter is the inverse of the importer. The architecture mirrors the importer's: `data_layer_client.get_entity_body()` → AST → intermediate → builder → file → `receipt_emitter.emit_export_receipt()`.

### 4.1 The flow

```
graph_entity
  └─ data_layer_client.get_entity_body()  # AST JSON
      └─ DocumentAST.from_json()
          └─ ast_to_intermediate.ast_to_intermediate()
              └─ builders.<format>.build()
                  └─ write to file
                      └─ compute SHA-256
                          └─ receipt_emitter.emit_export_receipt()
```

The supported formats in v1:

| Format | Method | Notes |
|---|---|---|
| `.md` | `markdown-it-py` token-aware Markdown builder | The simplest of the builders; produces canonical Markdown. |
| `.html` | Self-contained HTML builder | The body is escaped; the document is wrapped in a `<!doctype html>` shell. |
| `.pdf` | HTML → weasyprint | Cross-platform; the macOS production path uses PDFKit via a Swift shim. |
| `.docx` | python-docx | Native OOXML; falls back to Pandoc for AST types python-docx doesn't cover. |
| `.xlsx` | openpyxl | One table per sheet; formulas round-trip as text. |
| `.pptx` | python-pptx | One slide per AST; layout is the generic "Title and Content". |
| `.eml` | stdlib `email.message` | The body is plain text; headers come from the AST's meta. |

### 4.2 The intermediate step

`ast_to_intermediate.ast_to_intermediate(ast)` converts a `DocumentAST` back to an `IntermediateDocument`. The conversion is a tree walk that promotes the AST's `[UUID: Block]` map to a flat list of `IntermediateBlock` instances with `parentID` set. The builders consume the intermediate; the AST is the durable form, the intermediate is the form the builders want.

### 4.3 The export receipt

The export receipt's `receipt_type` is `"export"`. The payload records:

* `output_path` — the file path the exporter wrote to.
* `target_format` — the format the user picked (`"pdf"`, `"docx"`, etc.).
* `output_size_bytes` — the file's size in bytes.
* `output_sha256` — the SHA-256 of the file's bytes, prefixed with `"sha256:"`.

The SHA-256 lets the audit trail answer "did the file change since the export?" without re-fetching the data layer.

### 4.4 Lossy round-trip note

The AST is the canonical form; the intermediate + builder combination is the lossy translation to a specific format. DOCX has no `equation` type; the Markdown builder promotes equations to `$$...$$` text. HTML round-trips nearly losslessly. The receipt chain is built on the AST, so re-importing an exported file gives a different AST (UUIDs, slight format drift) but the same content hash modulo that drift. v2 will track the drift explicitly.

---

## 5. Swift-side integration

The Swift side owns the actors the UI calls, the HTTP API the Python CLI talks to, and the share-sheet integration.

### 5.1 `TesseraImporter` actor

```swift
public actor TesseraImporter {
    public init(pythonExecutable: URL = TesseraCLIPath.default)
    public func importFile(at url: URL) async throws -> UUID
    public func importDirectory(at url: URL) async throws -> [UUID]
    public func importDragAndDrop(urls: [URL]) async throws -> [UUID]
}
```

The actor serialises calls (one Python subprocess at a time per importer). The init accepts a `pythonExecutable` URL for tests; the default is `TesseraCLIPath.default`, which is a shim over the existing `TesseraCLIBinaryResolver` (per the project's pattern).

The actor parses the Python CLI's JSON event stream (one event per file, plus a summary line) and returns the new entity id(s) to the caller. Failure modes (Python not found, file not found, parse error) are surfaced as `TesseraImporterError`.

### 5.2 `TesseraExporter` actor

```swift
public actor TesseraExporter {
    public init(pythonExecutable: URL = TesseraCLIPath.default)
    public func export(entityID: UUID, to format: ExportFormat, outputURL: URL) async throws
    public func export(entityID: UUID, shareWith target: ShareTarget) async throws
}
```

The exporter mirrors the importer: it shells out to the Python CLI to write the file, then emits an export receipt. The `shareWith` overload is the share-sheet entry point: it exports the document to a format the target accepts, stages the file in the exporter's output dir, and hands the file URL to the target's handler.

The `ExportFormat` enum is namespaced as `ProductivityExportFormat` to avoid collision with the conversation exporter's `ExportFormat` (which lives in `TesseraCore/Views/ExportView.swift`). The two have the same name in spirit but disjoint format sets.

### 5.3 `TesseraCLIPath` shim

`TesseraCLIPath` is a thin shim over the existing `TesseraCLIBinaryResolver`. It exposes:

* `default` — the resolved `tessera-cli` path.
* `pythonExecutable` — the Python interpreter used to invoke the importer / exporter.
* `repoRoot` — the repository root, used as the Python subprocess's CWD.
* `importerScript` / `exporterScript` — the entry-point paths.

The shim re-uses the same precedence order as the resolver (override > settings key > known locations > `$PATH`) so a user override is honoured across the codebase.

### 5.4 The subprocess runner

`PythonSubprocessRunner` is a thin wrapper around `Process` that streams stdout / stderr to the caller. v1 reads the output synchronously; v2 may switch to a streaming async reader. The runner sets `TESSERA_DATA_LAYER_DRY_RUN=1` by default so a CI test that forgets to set up the data layer still works.

---

## 6. System share sheet

The primary export UX is the system share sheet.

### 6.1 `ShareSheetCoordinator`

```swift
public actor ShareSheetCoordinator {
    public func presentShareSheet(for entityID: UUID, from view: NSView) async throws
    public func availableShareTargets() -> [ShareTarget]
}
```

The coordinator is an actor so concurrent invocations from the UI don't race. The available targets are:

* The system share sheet (always present on macOS via `NSSharingServicePicker`).
* The configured Slack target(s) (one per workspace the user has connected).
* Any custom webhook targets the user has added.

The picker filters the available targets by what the entity can be exported to; the user picks one, the system handles the handoff. This is better than building per-service APIs because it works with whatever the user has installed (Gmail in browser, Outlook, Fastmail, ...) and uses the OS's native handoff (rich previews, attachments, ...).

### 6.2 `ShareTarget`

```swift
public struct ShareTarget {
    public var id: String
    public var name: String
    public var icon: Data?
    public var accepts: Set<ExportFormat>
    public var handler: (URL) -> Void
}
```

The `handler` is `@Sendable` and async-throws so it can do real work (uploading to Slack, opening another app, etc.). The v1 implementations are:

* `system.sharing-service-picker` — the macOS system share sheet.
* `slack.<hash>` — the Slack webhook target.
* `custom.*` — user-added webhook targets (Discord, Teams, ...).

### 6.3 Slack mrkdwn

Slack's mrkdwn differs from CommonMark in the ways the v1 formatter handles:

* `**bold**` → `*bold*`
* `*italic*` → `_italic_`
* `~~strike~~` → `~strike~`
* `# heading` → `*heading*` (Slack has no heading syntax; we promote to bold)
* `[text](url)` → `<url|text>`
* `- item` → `• item` (Slack has a `-` for list-like; we use the bullet char so the lists look right in the chat client)

The formatter is a small purpose-built converter (`SlackMrkdwnFormatter`) that uses `NSRegularExpression` to do the substitutions. v1 supports text-only posts; attachments (file uploads via Slack's `files.upload` API) are out of scope for v1.

---

## 7. Slack webhook

### 7.1 The webhook URL

The webhook URL is the only third-party secret in v1 (the spec §11.2 says "no per-target API keys" — the Slack webhook URL is the exception because Slack's share-sheet integration is unreliable). The URL is stored in the macOS Keychain via `KeychainStorage`:

* `kSecAttrService` is `TesseraSecretStore.service` (`"com.tessera.studio"`) so the entry is wiped by the same crypto-shred event the volume password is.
* `kSecAttrAccessible` is `kSecAttrAccessibleWhenUnlockedThisDeviceOnly` so the value is unavailable after a reboot until the user unlocks the keychain.
* The account name is namespaced (`"slack-webhook.<username>"`) so multiple users on the same machine each see their own config.

The `KeychainStorage` class is a final `Sendable` type with the standard `set / get / delete` methods. The unit tests run against a per-test service name so they don't collide with each other or with the production entry.

### 7.2 The Slack post

`SlackExportTarget.post(document:)` reads the file at the given URL, formats the content as mrkdwn, and POSTs the formatted payload to the webhook. The payload is the Slack-canonical JSON shape:

```json
{
  "text": "Hello world",
  "channel": "general",
  "username": "Tessera"
}
```

The HTTP call uses `URLSession.shared` (no third-party deps). The handler propagates errors (the webhook returns a non-2xx status) so the share sheet surfaces a useful message to the user.

### 7.3 The privacy posture

The Slack webhook URL is the only network egress in v1 that involves a third party. It is:

* Stored in Keychain (not UserDefaults, not the data layer, not on disk in plaintext).
* Wiped by the Plea the Fifth 9-step wipe (because the Keychain service namespace is the same as the volume password's).
* Opt-in (the user adds it via Settings; we never collect one by default).

The data layer is on `127.0.0.1` (loopback only) so the import / export pipeline's HTTP traffic never leaves the user's machine. The Slack POST is the only outbound call.

---

## 8. HTTP API

The Python CLI talks to a small HTTP API exposed by the Swift app. The API is on `127.0.0.1` only (no external network); the data layer is on the same host, so the loopback address is the right scope.

### 8.1 Endpoints

| Method | Path | Body | Response |
|---|---|---|---|
| `GET` | `/v1/health` | — | `{"ok": true}` |
| `POST` | `/v1/import` | The file bytes (multipart in v2; raw bytes in v1) | `{"entity_id": "..."}` |
| `POST` | `/v1/export` | `{"entity_id": "...", "format": "..."}` | The file bytes (with `Content-Disposition: attachment`) |
| `POST` | `/v1/entities` | `{"entity_type": "...", "label": "...", "body": "..."}` | `{"entity_id": "..."}` |
| `GET` | `/v1/entities/<id>` | — | The entity's body (AST JSON) |
| `GET` | `/v1/entities/<id>/meta` | — | `{"entity_id": "...", "entity_type": "...", "label": "..."}` |
| `POST` | `/v1/receipts` | `{"entity_id": "...", "receipt_type": "...", "payload": {...}}` | `{"receipt_id": "..."}` |

### 8.2 The server

The server is built on Apple's `Network` framework (`NWListener` bound to `127.0.0.1:8787`). We don't use SwiftNIO because the macOS app target doesn't depend on it and the v1 traffic is light (a few imports per minute under the heavy use case). The HTTP request parser is the simplest possible: a line-reader that splits on CRLF and reads the body length from the `Content-Length` header. Multi-part bodies are a v2 concern; the Python CLI uses the simple `Content-Length` path.

The handler dispatches to the in-process `TesseraImporter` / `TesseraExporter` actors. The response is the actor's result (an entity id, the file bytes, or a receipt id). Errors return `5xx` with `{"error": "..."}`.

### 8.3 The Swift client

`DataLayerHTTPClient` is the Swift-side HTTP client for the data layer. It mirrors the Python `DataLayerClient` so the in-process actors can talk to a remote data layer (the same instance the macOS app is running) when needed. v1 uses this client only for the macOS app's startup configuration check ("is the data layer healthy?"); the actual import / export goes through the Python CLI.

### 8.4 Authentication

There is no authentication. The API is bound to `127.0.0.1` only; only processes on the same host can reach it. A future v2 may add a token (the data layer facade already has the concept of an actor identity) when a remote data layer is needed.

---

## 9. Library survey

| Need | Library | Decision | Rationale |
|---|---|---|---|
| DOCX parsing | `python-docx` | Adopt | Mature; covers paragraphs, headings, lists, tables, images, footnotes in v1. |
| XLSX parsing | `openpyxl` | Adopt | The standard Python XLSX library; handles formulas, formatting, multiple sheets. |
| PPTX parsing | `python-pptx` | Adopt | The standard Python PPTX library; handles slides, text frames, basic shapes. |
| PDF rendering | `weasyprint` | Adopt | Cross-platform HTML→PDF rendering; production path uses PDFKit (macOS native) via a Swift shim. |
| PDF text extraction | `pdftotext` (poppler-utils) | Adopt | Available on every macOS dev machine via `brew install poppler`; no Python dep. |
| HTML parsing | `beautifulsoup4` | Adopt | Mature; v1 uses the stdlib `html.parser` backend (no lxml dependency). |
| Email parsing | `mailbox` + `email` stdlib | Adopt | Stdlib only; no third-party deps. `.msg` files fall back to the same parser (with a warning). |
| Markdown parsing | `markdown-it-py` | Adopt | The most spec-compliant CommonMark parser; supports GFM tables + task lists. |
| Format conversion (swiss-army) | `pandoc` | Adopt | Gold standard for format conversion; handles every format the dedicated parsers don't cover. |
| Slack mrkdwn | none | Build | The mrkdwn dialect is small; a 100-line converter is simpler than vendoring a lib. |
| System share sheet (macOS) | `NSSharingServicePicker` | Adopt | The macOS-blessed way to expose Mail / Messages / AirDrop / etc. without per-service integration. |
| System share sheet (iOS) | `UIActivityViewController` | Adopt | The iOS-blessed equivalent; same principle, different API. |
| HTTP client (Swift) | `URLSession` | Adopt | Built-in; no third-party deps. |
| HTTP client (Python) | `httpx` | Adopt | Modern, async-native, drop-in for `requests`. |
| HTTP server (Swift) | `Network` framework | Adopt | Built-in (`NWListener`); avoids the SwiftNIO dep. |
| Slack webhook storage | macOS Keychain | Adopt | The architect's standard for per-device secrets; wiped by Plea the Fifth. |
| PDF generation (test) | `reportlab` | Adopt (test only) | The hand-crafted PDF is enough for v1; reportlab produces a well-formed PDF for tests. |

**Deviations from the spec's §10 / §11:**

* The spec says "PDF: weasyprint (render) + pdftotext (extract)". We use `pdftotext` only on the import side; weasyprint is the export-side renderer. Two reasons: (a) weasyprint is an HTML→PDF renderer, not a PDF parser; (b) the spec's render-to-HTML-for-reference pass would add a second subprocess spawn per import without buying us much (we already have the text via `pdftotext`).
* The spec says "Slack mrkdwn: Build (it's just markdown-like)". We Build it. v2 may vendor a library if the dialect grows.
* The spec doesn't mention Keychain. We chose it because the architect's standard for per-device secrets is the Keychain; the volume password lives there, the receipt signing key is derived from it, and the Plea the Fifth 9-step wipe destroys it. The Slack webhook URL fits the same pattern.

---

## 10. Test strategy

### 10.1 Python tests (pytest)

* `test_ast_schema.py` — the AST JSON wire format. The tests verify the on-disk shape is stable and round-trips through `json.loads` / `json.dumps`. A regression here breaks the wire format and would fail to round-trip in the app.
* `test_format_detector.py` — the detector. Each fixture format is detected correctly; unknown formats fall through to Pandoc; missing files return the extension's format (so the importer's error path is well-typed).
* `test_parsers.py` — each parser. The fixture → intermediate is verified to produce the expected block types and annotations.
* `test_pipeline.py` — the orchestration. The pipeline creates entities, appends receipts, and handles failures. The test uses a `_CapturingClient` that records every call so the assertions are on the data layer's API contract, not on the network.
* `test_exporters.py` — the builder + AST → intermediate round-trip. Markdown / HTML / DOCX / XLSX / PPTX / EML all produce well-formed output.

### 10.2 Swift tests (XCTest + Swift Testing)

* `TesseraCLIPathTests` — the shim resolves a Python interpreter; the runner handles a missing script cleanly; the round-trip via `python -c` succeeds.
* `TesseraImporterEventParsingTests` — the JSON event stream parser. Each event type (ok, failed, summary) is parsed correctly; malformed UUIDs are dropped (not thrown).
* `ExportFormatTests` — every `ProductivityExportFormat` case has a display name; the Slack mrkdwn formatter handles bold, links, headings, bullets, strikethrough, and a mixed round-trip.
* `ShareSheetCoordinatorTests` — the available-targets list includes the system share sheet (on macOS) and the Slack / custom targets.
* `KeychainStorageTests` — set / get round-trip; replace behaviour; idempotent delete. Each test uses a per-test service name so it doesn't collide with the production entry.

### 10.3 Performance numbers

* 11 KB DOCX (compressed), 3209 paragraphs, ~1 MB raw text: import elapsed ~2.6s on a 2024 MacBook Pro (M-series, dev build, dry-run). The bottleneck is python-docx's XML parsing, not our pipeline.
* 7 KB DOCX (compressed), 6 paragraphs, 1 list: import elapsed ~0.2s.
* The HTML export for the dry-run mode is ~150 bytes; the DOCX export is ~37 KB (the python-docx scaffolding is large for a single-block document).
* The PDF export via weasyprint is ~3 KB for a single-block document.

### 10.4 Fixtures

`tools/tessera/importers/tests/fixtures/` and `tools/tessera/exporters/tests/fixtures/` contain the small, hand-crafted sample files (DOCX, XLSX, PPTX, PDF, EML, MBOX, HTML, Markdown). The fixtures are regenerated by `python3 -m tools.tessera.importers.tests._make_fixtures` and committed to the worktree so the tests can run offline.

---

## 11. Out of scope (v1 → v2)

* **OCR for scanned PDFs.** Punt to v2 (the architect's call).
* **Password-protected files.** The importer returns an error in v1; v2 will detect and prompt.
* **Real-time spreadsheet formulas.** Formulas are stored as text in v1; v2 will re-evaluate via a sandboxed Python engine.
* **Email replies that import back as receipts.** The user can export to email; replies don't auto-import. Punt to v2.
* **Real-time collaboration** (Google Docs-style multi-user editing). Punt to v2 per the spec's §11.3.
* **Slide master layouts** (PPTX). Not preserved; only text frames and basic shapes.
* **Image OCR** (extracted images from PDFs, DOCX, PPTX). Punt to v2; v1 writes the bytes to a sidecar directory.
* **Format-drift tracking.** The round-trip AST after an import → export → import has different UUIDs and slight format drift. v2 will track the drift explicitly so the audit trail can answer "what changed?".
* **Multi-part HTTP request bodies.** The Swift HTTP server reads the body as raw bytes; v2 will parse multipart bodies.
* **Streaming stdout / stderr from the Python subprocess.** v1 reads synchronously; v2 may switch to a streaming async reader for live progress UI.
* **Auth on the HTTP API.** Loopback only in v1; a token in v2 when a remote data layer is needed.
* **Per-service share-sheet integrations** (Gmail, Outlook, Fastmail, ...). The system share sheet covers all of these in v1; per-service APIs are out of scope.
* **Slack attachments (file uploads via `files.upload`).** Text-only posts in v1; attachments in v2.
