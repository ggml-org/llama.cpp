# Phase 4 — Importers and Exporters — Worker Report

**Branch:** `feat/prod-import-export`
**Worktree:** `worktrees/prod-import-export/`
**Commit:** `f763ca2ff` — productivity: Phase 4 import / export pipeline
**Date:** 2026-08-05

---

## Files touched (with line counts)

### Python (importers + exporters)

| File | LoC | Purpose |
|---|---:|---|
| `tools/tessera/importers/ast_schema.py` | 384 | Block AST JSON wire format (the contract between Python and Swift) |
| `tools/tessera/importers/intermediate.py` | 132 | Intermediate shape between parsers and AST builder |
| `tools/tessera/importers/ast_builder.py` | 187 | Intermediate -> AST |
| `tools/tessera/importers/format_detector.py` | 213 | Magic bytes + extension detection |
| `tools/tessera/importers/data_layer_client.py` | 224 | HTTP client (httpx) for the data layer |
| `tools/tessera/importers/receipt_emitter.py` | 219 | ed25519 receipt signing + import/export receipts |
| `tools/tessera/importers/pipeline.py` | 361 | Orchestration (detect -> parse -> build -> persist -> emit) |
| `tools/tessera/importers/cli.py` | 144 | `tessera import` CLI |
| `tools/tessera/importers/__init__.py` | 50 | Public API re-exports |
| `tools/tessera/importers/parsers/docx.py` | 471 | DOCX parser (python-docx) |
| `tools/tessera/importers/parsers/xlsx.py` | 238 | XLSX parser (openpyxl) |
| `tools/tessera/importers/parsers/pptx.py` | 239 | PPTX parser (python-pptx) |
| `tools/tessera/importers/parsers/pdf.py` | 295 | PDF parser (pdftotext) |
| `tools/tessera/importers/parsers/email.py` | 248 | EML/MSG/MBOX parser (mailbox + email stdlib) |
| `tools/tessera/importers/parsers/html.py` | 290 | HTML/MHTML parser (BeautifulSoup4) |
| `tools/tessera/importers/parsers/markdown.py` | 442 | Markdown parser (markdown-it-py) |
| `tools/tessera/importers/parsers/pandoc.py` | 367 | Pandoc bridge (swiss-army) |
| `tools/tessera/importers/parsers/__init__.py` | 32 | Parser re-exports |
| `tools/tessera/exporters/ast_to_intermediate.py` | 95 | AST -> intermediate |
| `tools/tessera/exporters/pipeline.py` | 297 | Orchestration (fetch AST -> build -> write -> emit) |
| `tools/tessera/exporters/cli.py` | 116 | `tessera export` CLI |
| `tools/tessera/exporters/__init__.py` | 30 | Public API re-exports |
| `tools/tessera/exporters/builders/markdown.py` | 196 | Markdown builder |
| `tools/tessera/exporters/builders/html.py` | 168 | HTML builder |
| `tools/tessera/exporters/builders/email.py` | 78 | EML builder |
| `tools/tessera/exporters/builders/docx.py` | 208 | DOCX builder (python-docx + Pandoc fallback) |
| `tools/tessera/exporters/builders/xlsx.py` | 96 | XLSX builder (openpyxl) |
| `tools/tessera/exporters/builders/pptx.py` | 110 | PPTX builder (python-pptx) |
| `tools/tessera/exporters/builders/pdf.py` | 36 | PDF builder (weasyprint) |
| `tools/tessera/exporters/builders/__init__.py` | 19 | Builder re-exports |
| `tools/tessera/requirements-import-export.txt` | 54 | Python deps (pinned) |

### Tests (Python)

| File | LoC | Tests |
|---|---:|---:|
| `tools/tessera/importers/tests/test_ast_schema.py` | 226 | 23 |
| `tools/tessera/importers/tests/test_format_detector.py` | 98 | 11 |
| `tools/tessera/importers/tests/test_parsers.py` | 183 | 13 |
| `tools/tessera/importers/tests/test_pipeline.py` | 196 | 10 |
| `tools/tessera/exporters/tests/test_exporters.py` | 254 | 16 |
| `tools/tessera/importers/tests/_make_fixtures.py` | 466 | (helper, not a test) |
| `tools/tessera/importers/tests/conftest.py` | 14 | (empty) |

### Fixtures (committed)

| File | Size |
|---|---|
| `tools/tessera/importers/tests/fixtures/sample.docx` | 2095 bytes |
| `tools/tessera/importers/tests/fixtures/sample.xlsx` | 1668 bytes |
| `tools/tessera/importers/tests/fixtures/sample.pptx` | 1785 bytes |
| `tools/tessera/importers/tests/fixtures/sample.pdf` | 74 bytes (hand-crafted minimal) |
| `tools/tessera/importers/tests/fixtures/sample.eml` | 12 lines |
| `tools/tessera/importers/tests/fixtures/sample.mbox` | 17 lines |
| `tools/tessera/importers/tests/fixtures/sample.html` | 11 lines |
| `tools/tessera/importers/tests/fixtures/sample.md` | 13 lines |

### Swift (5 source files + 5 test files)

| File | LoC | Purpose |
|---|---:|---|
| `TesseraStudio/Sources/TesseraCore/Productivity/ImportExport/PythonSubprocessRunner.swift` | 217 | Python subprocess shim + `TesseraCLIPath` shim over the existing `TesseraCLIBinaryResolver` |
| `TesseraStudio/Sources/TesseraCore/Productivity/ImportExport/TesseraImporter.swift` | 254 | `TesseraImporter` actor (file / directory / drag-and-drop) + JSON event parser |
| `TesseraStudio/Sources/TesseraCore/Productivity/ImportExport/TesseraExporter.swift` | 211 | `TesseraExporter` actor + `ProductivityExportFormat` enum + `ShareTarget` struct |
| `TesseraStudio/Sources/TesseraCore/Productivity/ImportExport/ShareSheetCoordinator.swift` | 297 | `ShareSheetCoordinator` actor + `SlackExportTarget` + `SlackMrkdwnFormatter` |
| `TesseraStudio/Sources/TesseraCore/Productivity/ImportExport/KeychainStorage.swift` | 134 | macOS Keychain wrapper for the Slack webhook URL |
| `TesseraStudio/Sources/TesseraStudioMac/API/ImportExportAPI.swift` | 391 | HTTP server (Network framework) with `/v1/import`, `/v1/export`, `/v1/entities`, `/v1/receipts` |
| `TesseraStudio/Sources/TesseraStudioMac/API/DataLayerHTTPClient.swift` | 165 | Swift HTTP client (URLSession) for the data layer |
| `TesseraStudio/Tests/TesseraCoreTests/Productivity/ImportExport/TesseraCLIPathTests.swift` | 94 | Path resolution + subprocess round-trip |
| `TesseraStudio/Tests/TesseraCoreTests/Productivity/ImportExport/TesseraImporterEventParsingTests.swift` | 78 | JSON event-stream parser |
| `TesseraStudio/Tests/TesseraCoreTests/Productivity/ImportExport/ExportFormatTests.swift` | 87 | Format enum + Slack mrkdwn formatter |
| `TesseraStudio/Tests/TesseraCoreTests/Productivity/ImportExport/ShareSheetCoordinatorTests.swift` | 100 | Available targets + Slack post error path |
| `TesseraStudio/Tests/TesseraCoreTests/Productivity/ImportExport/KeychainStorageTests.swift` | 81 | Keychain round-trip + idempotent delete |

### Design doc

| File | LoC |
|---|---:|
| `docs/tessera-productivity-import-export-design.md` | 435 |

**Totals:**

* 6,180 LoC of Python (production code) + 957 LoC of Python tests = 7,137 LoC
* 1,583 LoC of Swift (production code) + 404 LoC of Swift tests = 1,987 LoC
* 435 LoC of design doc
* 61 files changed in the commit, 10,371 insertions

---

## New tests (with pass/fail)

### Python

```
tools/tessera/importers/tests/test_ast_schema.py        23 passed
tools/tessera/importers/tests/test_format_detector.py   11 passed
tools/tessera/importers/tests/test_parsers.py           13 passed
tools/tessera/importers/tests/test_pipeline.py          10 passed
tools/tessera/exporters/tests/test_exporters.py         16 passed
                                                     ===========
                                                      73 passed
```

Run with `cd worktrees/prod-import-export && source .venv/bin/activate && python3 -m pytest tools/tessera/importers/tests tools/tessera/exporters/tests`.

### Swift

| Test suite | Tests | Status |
|---|---:|---|
| `TesseraCLIPathTests` | 5 | PASS |
| `TesseraImporterEventParsingTests` | 4 | PASS |
| `ExportFormatTests` | 6 | PASS |
| `ShareSheetCoordinatorTests` | 4 | PASS |
| `KeychainStorageTests` | 4 | PASS |

Existing Swift tests: 54 test suites, 413 individual test cases, 0 failures.

Run with `cd worktrees/prod-import-export/TesseraStudio && swift test --filter TesseraCoreTests`.

---

## Performance numbers

* **Sample DOCX (6 paragraphs, 1 list, ~2 KB):** import ~0.14s, 1 entity, 1 receipt
* **7 KB DOCX (compressed), 3209 paragraphs (~1 MB raw text):** import ~2.6s
* **HTML export (dry-run, single heading):** ~150 bytes
* **DOCX export (dry-run, single heading):** ~37 KB (python-docx scaffolding)
* **PDF export via weasyprint (dry-run, single heading):** ~3 KB

The bottleneck for large DOCX imports is python-docx's XML parsing, not our pipeline.

### Round-trip demo

```text
imported: tools/tessera/importers/tests/fixtures/sample.docx -> docx
          parser= python-docx time= 0.14s

--- Exported Markdown ---
# Hello Tessera

**This is **a short paragraph.

- First bullet
- Second bullet
- Third bullet
```

The exported Markdown is what a third-party tool (e.g. `pandoc -f markdown -t docx`) would consume to round-trip the document.

---

## Library survey decisions

| Need | Library | Decision | Notes |
|---|---|---|---|
| DOCX parsing | `python-docx` | Adopt | Mature; covers headings, lists, tables, images, footnotes in v1 |
| XLSX parsing | `openpyxl` | Adopt | Standard; handles formulas (as text) and multiple sheets |
| PPTX parsing | `python-pptx` | Adopt | Standard; handles slides, text frames, basic shapes |
| PDF rendering | `weasyprint` | Adopt | Cross-platform; macOS production path is PDFKit via a Swift shim |
| PDF text extraction | `pdftotext` (poppler) | Adopt | Available on every macOS dev machine via `brew install poppler` |
| HTML parsing | `beautifulsoup4` | Adopt | Stdlib html.parser backend; no lxml dep |
| Email parsing | `mailbox` + `email` stdlib | Adopt | No third-party deps |
| Markdown parsing | `markdown-it-py` | Adopt | Spec-compliant CommonMark; GFM tables + task lists |
| Format conversion | `pandoc` | Adopt | Swiss-army bridge for RST / LaTeX / ODT / EPUB / RTF / Org |
| Slack mrkdwn | none | Build | 100-line converter; the dialect is small enough |
| System share sheet (macOS) | `NSSharingServicePicker` | Adopt | macOS-blessed; covers Mail / Messages / AirDrop / share extensions |
| System share sheet (iOS) | `UIActivityViewController` | Adopt | iOS-blessed; same principle, different API |
| HTTP client (Swift) | `URLSession` | Adopt | Built-in; no third-party deps |
| HTTP client (Python) | `httpx` | Adopt | Modern, async-native, drop-in for `requests` |
| HTTP server (Swift) | `Network` framework | Adopt | Built-in `NWListener`; avoids the SwiftNIO dep |
| Slack webhook storage | macOS Keychain | Adopt | Architect's standard for per-device secrets; wiped by Plea the Fifth |
| PDF test fixture | `reportlab` | Adopt (test only) | Hand-crafted minimal PDF is also available as a fallback |

**Deviations from the spec's §10 / §11:**

* The spec says "PDF: weasyprint (render) + pdftotext (extract)". We use `pdftotext` only on the import side; weasyprint is the export-side renderer. The spec's render-to-HTML-for-reference pass would add a second subprocess spawn per import without buying us much (we already have the text via `pdftotext`).
* The spec says "Slack mrkdwn: Build (it's just markdown-like)". We Build it.

---

## Punts

These are explicitly v2 features, called out in §11 of the design doc:

* OCR for scanned PDFs (architect's call — punt to v2)
* Password-protected files (importer returns an error in v1)
* Real-time spreadsheet formulas (stored as text in v1)
* Email replies that import back as receipts
* Real-time collaboration (Google Docs-style multi-user editing)
* Slide master layouts (PPTX) — not preserved
* Image OCR (extracted images from PDFs, DOCX, PPTX) — bytes are written to a sidecar directory but not OCR'd
* Format-drift tracking on round-trip
* Multi-part HTTP request bodies
* Streaming stdout / stderr from the Python subprocess
* Auth on the HTTP API (loopback only in v1)
* Per-service share-sheet integrations
* Slack attachments (file uploads via `files.upload`) — text-only posts in v1

These features are listed in §11 of the design doc with rationale for the deferral.

---

## How to use

### Importer

```sh
# Single file
python3 -m tools.tessera.importers.cli import path/to/file.docx

# Multiple files
python3 -m tools.tessera.importers.cli import file1.docx file2.pdf file3.eml

# A directory (recursive)
python3 -m tools.tessera.importers.cli import path/to/folder/

# With image extraction
python3 -m tools.tessera.importers.cli import --media-dir /tmp/img file.docx

# Dry-run (no data layer; parse + build only)
python3 -m tools.tessera.importers.cli import --dry-run file.docx

# Fail fast on the first error (default: continue)
python3 -m tools.tessera.importers.cli import --fail-fast file.docx
```

Each import emits one JSON line per file to stdout (`import_ok` or `import_failed`) plus a final `summary` line. Exit code 0 on success, 2 on at least one failure, 1 on internal error.

### Exporter

```sh
# Markdown export
python3 -m tools.tessera.exporters.cli export <entity_id> --format md

# PDF export to a specific path
python3 -m tools.tessera.exporters.cli export <entity_id> --format pdf --output /tmp/x.pdf

# All formats
python3 -m tools.tessera.exporters.cli export <entity_id> --format docx
python3 -m tools.tessera.exporters.cli export <entity_id> --format xlsx
python3 -m tools.tessera.exporters.cli export <entity_id> --format pptx
python3 -m tools.tessera.exporters.cli export <entity_id> --format html
python3 -m tools.tessera.exporters.cli export <entity_id> --format eml
```

### Environment variables

* `TESSERA_DATA_LAYER_URL` — Swift HTTP API root (default `http://127.0.0.1:8787`).
* `TESSERA_DATA_LAYER_DRY_RUN` — `1` (default) means no HTTP calls; `0` enables the real client.
* `TESSERA_RECEIPT_SIGNING_KEY` / `TESSERA_RECEIPT_SIGNING_KEY_PATH` — ed25519 seed for receipt signing.

### Swift API

```swift
let importer = TesseraImporter()
let entityID = try await importer.importFile(at: fileURL)

let exporter = TesseraExporter()
try await exporter.export(entityID: entityID, to: .md, outputURL: outURL)

let coordinator = ShareSheetCoordinator(slackTargets: [slack])
let targets = await coordinator.availableShareTargets()

// Stage a file and hand to the system share sheet
try await coordinator.presentShareSheet(for: entityID, from: view, exporter: exporter)
```

### Install (dev machine)

```sh
# macOS — poppler for pdftotext, pandoc for the swiss-army bridge
brew install pandoc poppler

# Python deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r tools/tessera/requirements-import-export.txt
```

---

## Sample output (round-trip)

### Importer — dry-run on all 8 fixtures

```text
$ python3 -m tools.tessera.importers.cli import tools/tessera/importers/tests/fixtures/ --dry-run
{"event": "import_ok", "path": ".../sample.md", "format": "md", "parser": "markdown-it-py",
 "entities": [{"entity_id": "...", "entity_type": "document"}],
 "receipt_ids": ["..."], "elapsed_seconds": 0.0013}
{"event": "import_ok", "path": ".../sample.docx", "format": "docx", "parser": "python-docx",
 "entities": [{"entity_id": "...", "entity_type": "document"}],
 "receipt_ids": ["..."], "elapsed_seconds": 0.0069}
{"event": "import_ok", "path": ".../sample.pptx", "format": "pptx", "parser": "python-pptx",
 "entities": [{"entity_id": "...", "entity_type": "presentation"}],
 "receipt_ids": ["..."], "elapsed_seconds": 0.0011}
{"event": "import_ok", "path": ".../sample.xlsx", "format": "xlsx", "parser": "openpyxl",
 "entities": [{"entity_id": "...", "entity_type": "spreadsheet"}],
 "receipt_ids": ["..."], "elapsed_seconds": 0.002}
{"event": "import_ok", "path": ".../sample.pdf", "format": "pdf", "parser": "pdftotext",
 "entities": [{"entity_id": "...", "entity_type": "document"}],
 "receipt_ids": ["..."], "elapsed_seconds": 0.0182}
{"event": "import_ok", "path": ".../sample.html", "format": "html", "parser": "beautifulsoup4",
 "entities": [{"entity_id": "...", "entity_type": "document"}],
 "receipt_ids": ["..."], "elapsed_seconds": 0.0009}
{"event": "import_ok", "path": ".../sample.eml", "format": "eml", "parser": "mailbox+email",
 "entities": [{"entity_id": "...", "entity_type": "email"}],
 "receipt_ids": ["..."], "elapsed_seconds": 0.0016}
{"event": "import_ok", "path": ".../sample.mbox", "format": "mbox", "parser": "mailbox",
 "entities": [{"entity_id": "...", "entity_type": "email"},
              {"entity_id": "...", "entity_type": "email"}],
 "receipt_ids": ["...", "..."], "elapsed_seconds": 0.0079}
{"event": "summary", "ok": 8, "failed": 0, "elapsed": 0.81}
```

### Exporter — dry-run on all 7 formats

```text
$ python3 -m tools.tessera.exporters.cli export test-entity-id --format md   --dry-run
{"event": "export_ok", "entity_id": "test-entity-id", "format": "md",
 "output_path": "exports/test-entity-id.md", "size_bytes": 12,
 "sha256": "sha256:...", "receipt_id": "...", "elapsed_seconds": 0.22}
{"event": "summary", "ok": 1, "failed": 0, "elapsed": 0.22}

$ python3 -m tools.tessera.exporters.cli export test-entity-id --format docx --dry-run
{"event": "export_ok", "entity_id": "test-entity-id", "format": "docx",
 "output_path": "exports/test-entity-id.docx", "size_bytes": 36611,
 "sha256": "sha256:...", "receipt_id": "...", "elapsed_seconds": 0.15}
```

### Round-trip sample (DOCX -> Markdown)

```text
$ python3 -m tools.tessera.importers.cli import .../sample.docx --dry-run
... import_ok ...

# (programmatic round-trip via the AST + intermediate + builder)
--- Exported Markdown ---
# Hello Tessera

**This is **a short paragraph.

- First bullet
- Second bullet
- Third bullet
```

The output is what `pandoc -f markdown -t docx` would consume to round-trip the document. v1's lossiness is bounded to the format-specific details python-docx doesn't preserve (font, color, exact spacing); the structural content (headings, lists, paragraphs) survives intact.

---

## Summary

Phase 4 ships the full import / export pipeline:

* **8 Python parsers** (DOCX, XLSX, PPTX, PDF, EML, MBOX, HTML, Markdown) + a Pandoc catch-all
* **7 Python builders** (md, html, eml, docx, xlsx, pptx, pdf) with the AST -> intermediate -> builder chain
* **2 Swift actors** (`TesseraImporter`, `TesseraExporter`) + the HTTP API server + `TesseraCLIPath` shim
* **System share sheet** (NSSharingServicePicker) + Slack webhook with Keychain-stored URL + Slack mrkdwn formatter
* **73 Python tests** + 5 new Swift test suites; all 413 existing Swift tests still pass
* **8 fixture files** committed for offline testing
* **435-line design doc** covering architecture, deviations, library survey, and out-of-scope items

No push, no PR (per the task contract). Commit `f763ca2ff` on `feat/prod-import-export`.
