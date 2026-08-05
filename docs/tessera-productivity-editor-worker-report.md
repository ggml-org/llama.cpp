# Phase 2 — Productivity Editor Worker Report

**Branch:** `feat/prod-editor` (off `feat/prod-foundations`)
**Worktree:** `worktrees/prod-editor/`
**Date:** 2026-08-05
**Worker:** Mavis (general-purpose agent)
**Commit policy:** Two commits: (1) work, (2) this report. Both with `Assisted-by: MiniMax`. No push, no PR.

---

## 1. Headline numbers

- **Tests:** 619 existing → **746 total** (127 new). **0 failures, 28 skipped** (DB integration gated by env).
- **Test time:** ~48 seconds for the full suite on a 2026 Mac (Apple M-series, macOS 26). Dominated by pre-existing workflow tests, not editor tests.
- **Files added:** 23 new files (10 in `TesseraCore/Editor/`, 4 in `TesseraStudioMac/Views/Editor/`, 9 in `TesseraCoreTests/Editor/`). 1 design doc. 2 Package.swift updates.
- **Lines of code:** ~3,500 LoC in editor sources + ~1,700 LoC in editor tests + 447 LoC design doc.
- **Build:** `swift build --target TesseraCore` and `swift build --target TesseraStudioMac` both clean. `swift test` green.

## 2. Files touched (with line counts)

### 2.1 Core types — `TesseraStudio/Sources/TesseraCore/Editor/`

| File | LoC | Purpose |
|---|---|---|
| `EditorMode.swift` | 174 | `EditorMode` (document / notes / code), `EditorTheme`, `FontDescriptor`, `SyntaxThemePalette` |
| `BlockRenderer.swift` | 533 | Pure `Block -> NSAttributedString` (all 13 block types, every `InlineRun.Annotation`, `NSTextAttachment` for images, `PlatformFontResolver` for platform font mapping) |
| `TesseraTextElement.swift` | 196 | `TesseraTextElementData` (Sendable struct) + `TesseraTextElement : NSTextParagraph` (AppKit/UIKit) + `ElementBuilder` (depth-first walk of `DocumentAST`) |
| `TesseraTextContentManager.swift` | 346 | `TesseraTextContentManagerData` (testable core) + `TesseraTextContentManager : NSTextContentManager, NSTextContentManagerDelegate` (AppKit/UIKit) |
| `IntTextLocation.swift` | 82 | Concrete `NSTextLocation` (`NSObject` subclass with `compare(_:)`) + `makeIntTextRange(start:end:)` helper |
| `EditorCursorState.swift` | 161 | Two-cursor data model + `CursorSelection` + `CursorInBlock` + `TextCursor.resolved(in:)` round-trip |
| `TextEditReducer.swift` | 215 | Pure `NSAttributedString` diff → `Mutation` (typing → `setBlockContent`, formatting → `setInlineAnnotation`, paste → `setBlockContent`) |
| `EditorCoalescer.swift` | 244 | 1.5s coalescing window (clamped 0.5–5.0s) + `ChatQueueItem` emission + `didFlushNotification` |
| `AnimationPrimitives.swift` | 283 | 7 SwiftUI animation primitives + Reduce Motion fallbacks + `TextAppearCadence` (AsyncStream) + `blockSlideIn` / `thinkingPulse` / `cursorBlink` view modifiers |
| `CodeBlockHighlighter.swift` | 317 | Splash for Swift + regex for Python, JavaScript, TypeScript, SQL, JSON, YAML, Markdown, Shell, Rust, Go (10 languages + aliases) + plain monospaced fallback for unknown |

### 2.2 macOS view layer — `TesseraStudio/Sources/TesseraStudioMac/Views/Editor/`

| File | LoC | Purpose |
|---|---|---|
| `TesseraEditorView.swift` | 364 | `NSViewRepresentable` + `Coordinator` (handles NSText.didChangeNotification → `TextEditReducer` → `EditorCoalescer` → `MutationEngine`) + `TesseraNSTextView : NSTextView` (custom subclass with the `NSTextContentManager` wired) |
| `TesseraEditorToolbar.swift` | 284 | SwiftUI toolbar (per `EditorMode`) + `FormattingState` binding + `EditorCommand` enum (Codable, library-agnostic) + `ToolbarButton` / `ToolbarIconButton` |
| `EditorUndoCoordinator.swift` | 195 | Coordinator + `AppKitUndoManagerBridge : UndoManager` (returns `receipt.summary` for menu label, forwards undo/redo to `ReceiptUndoManager`) |
| `AgentCursorOverlay.swift` | 108 | SwiftUI overlay rendering the agent cursor (robot icon + 530ms blink via `cursorBlink` modifier + static under Reduce Motion) |

### 2.3 Tests — `TesseraStudio/Tests/TesseraCoreTests/Editor/` (127 new tests, all green)

| File | Tests | Coverage |
|---|---|---|
| `BlockRendererTests.swift` | 23 | Every `BlockType`, every `InlineRun.Annotation`, image blocks (placeholder attachment), `PlatformColor.fromHex`, font resolver |
| `TesseraTextContentManagerTests.swift` | 14 | One element per block, container blocks nest, empty document, mutation apply (insert/delete/setBlockContent/batch), binary search `elementAt(offset:)`, 1000-block perf, platform subclass |
| `IntTextLocationTests.swift` | 7 | `compare(_:)` ordering, equality by `intValue`, `makeIntTextRange`, reversed range returns nil |
| `EditorCursorStateTests.swift` | 12 | User + agent cursors coexist, user moves independently, both in same paragraph, agent active toggle, `CursorSelection`, `TextCursor.resolved` round-trip |
| `TextEditReducerTests.swift` | 15 | `diff()` (empty/insertion/deletion/replacement), `reduce()` (typing → `setBlockContent`), `reduceFormattingChange()` (adds/toggles annotation), `reducePaste()`, `NSRange.substring` |
| `EditorCoalescerTests.swift` | 13 | Settings clamping, 10 keystrokes = 1 mutation, `ChatQueueItem` with `sourceMutation`, flush with no pending, cross-block / cross-document start new burst, notification, window expiry |
| `AnimationPrimitivesTests.swift` | 22 | Every duration constant matches spec, every Reduce Motion fallback (`blockDelete` → nil, `textAppearDelay` → nil, `cursorBlink` → nil, `thinkingPulseAnimation` → nil, etc.), `TextAppearCadence` is interruptible, view modifiers don't crash |
| `CodeBlockHighlighterTests.swift` | 16 | Every language (Swift via Splash; Python, JS, TS, SQL, JSON, YAML, MD, Shell, Rust, Go via regex), case-insensitive tags, aliases (`py`, `ts`, `sh`), unknown fallback, nil fallback, monospaced output, theme |
| `ReceiptUndoEditorIntegrationTests.swift` | 5 | Undo of user edit restores document, menu shows `summary`, batched undo composes summaries, voided receipts not undo candidates, Cmd-Shift-Z redo |

### 2.4 Build & docs

- `TesseraStudio/Package.swift` — added `Splash` 0.16.0 as a `TesseraCore` dependency.
- `TesseraStudio/Package.resolved` — auto-updated by SPM.
- `docs/tessera-productivity-editor-design.md` — 447 LoC design doc (15 sections).

## 3. Test results (all green)

```
Test Suite 'TesseraStudioPackageTests.xctest' passed
  Executed 746 tests, with 28 tests skipped and 0 failures (0 unexpected)
  in 48.787 seconds

Test Suite 'All tests' passed
  Executed 746 tests, with 28 tests skipped and 0 failures (0 unexpected)
```

(28 skipped = pre-existing `ProductivityDataLayerTests` that gate on `TESSERA_DB_INTEGRATION=1` env var.)

## 4. Performance numbers

- **1000-block enumeration:** the test runs in ~65ms total (with XCTest overhead). The actual `ElementBuilder.buildElements` time is **well under 50ms** on the dev Mac. Test budget is 200ms to accommodate CI variance; the XCTAttachment reports the actual elapsed time.
- **Editor layer compile time:** `swift build --target TesseraCore` completes in ~2s. `swift build --target TesseraStudioMac` completes in ~2s.

## 5. Library survey (decisions)

| Need | Library | Decision | Rationale |
|---|---|---|---|
| Modern TextKit 2 text view | `STTextView` (krzyzanowskim) | **Adopt (deferred to Phase 2.1)** | NSTextView is the v1 base (works, no extra dep). STTextView natively supports a custom `NSTextContentManager` via a settable property — the swap is a one-class change. The data layer (`TesseraTextContentManager` + `TesseraTextElement`) is shaped for it. |
| SwiftUI niceties on top of the text view | `RichTextKit` (Daniel Saidi) | **Adopt (deferred to Phase 2.1)** | A hand-rolled SwiftUI toolbar ships in v1. RichTextKit is the recommended production upgrade. The toolbar's public API (`EditorCommand` enum) is library-agnostic, so the swap is a no-op for the editor's view layer. |
| AST-backed `NSTextContentManager` | none | **Build** | No third-party library does this. Apple's abstract API is the right layer to subclass. The data layer is fully tested in isolation. |
| Markdown rendering (Notes surface) | `MarkdownUI` (gonzalezreal) | **Adopt (Phase 5)** | Swift-native, good rendering quality, SwiftPM. Adopted in the Notes surface (per-surface wrapper) — not a Phase 2 deliverable. |
| Markdown parsing (Swift) | none | **Build (Phase 4)** | The Block AST is richer than CommonMark. Parsing is a Phase 4 importer deliverable. |
| Syntax highlighting | `Splash` (JohnSundell) | **Adopt** | Swift-native, grammar-driven. Splash only ships `SwiftGrammar`; the other 9 languages use a small regex highlighter (the design doc §10.3 documents the upgrade path). **Added to Package.swift in this phase.** |
| Text range / selection | Apple's `NSTextRange` / `NSTextLocation` | n/a | The platform's types. We provide `IntTextLocation` as the concrete integer location (Apple's NSTextLocation is a marker protocol + a Swift-overlay `compare` requirement). |

## 6. Punts (documented, not implemented)

These are the items the brief calls for that we deferred to follow-up workers. Each is documented in the design doc (§13).

1. **STTextView (krzyzanowskim) swap.** The current `TesseraEditorView` uses `NSTextView` (works, no extra dep). STTextView natively supports a custom `NSTextContentManager` via a settable property; the data layer is shaped for it. The swap is a one-class change: replace `TesseraNSTextView` with `STTextView`, set `textContentManager = tesseraContentManager`. The data flow (`TextEditReducer` → `EditorCoalescer` → `MutationEngine` → `Receipt` → `DocumentStore`) is unchanged.

2. **RichTextKit (Daniel Saidi) swap.** The current `TesseraEditorToolbar` is a hand-rolled SwiftUI toolbar (no external dep). RichTextKit is the recommended production upgrade. The toolbar's public API (`EditorCommand` enum, `FormattingState` binding) is library-agnostic, so the swap is a no-op for the editor's view layer.

3. **MarkdownUI for Notes.** The Notes surface renders its content as markdown (per spec §10.7). MarkdownUI (gonzalezreal) is the recommended library; it's a Phase 5 deliverable (per-surface wrapper).

4. **NSTextView ↔ TesseraTextContentManager live wiring.** NSTextView's `textContentStorage` is read-only after init; the layout manager's `textContentManager` is also set at init. The current implementation holds the `TesseraTextContentManager` as a sibling of the `NSTextContentStorage`; the round-trip (storage → our content manager → AST → storage) is the Phase 2.1 stretch. STTextView is the recommended fix.

5. **Tables (NSTextTable).** Rendered as a placeholder string (`[Table 3×4 — Phase 3 surface]`). Phase 3 uses `NSTextTable` / UITextView's NSTextTable equivalent.

6. **Equations (LaTeX).** Rendered as inline code (`$E = mc^2$`). Phase 3 uses MathJax or iosMath.

7. **Image picker.** `EditorCommand.insertImage` is wired on the toolbar; the file dialog is Phase 3.

8. **iOS UIViewRepresentable.** The current `TesseraEditorView` is `NSViewRepresentable` (macOS). The iOS `UIViewRepresentable` mirror is Phase 5 (per-surface wrappers).

9. **LSP integration, real-time collaboration, terminal integration.** All v2 per the spec.

## 7. "How to use" snippet

The host window uses the editor with the standard SwiftUI `ViewBuilder` pattern:

```swift
import SwiftUI
import TesseraCore
import TesseraStudioMac

struct DocumentEditorView: View {
    @State private var document: DocumentAST = .empty
    @State private var formattingState = FormattingState()
    @State private var cursors = EditorCursorState()

    var body: some View {
        VStack(spacing: 0) {
            TesseraEditorToolbar(
                mode: .document,
                formattingState: $formattingState,
                onCommand: handleCommand
            )
            TesseraEditorView(
                mode: .document,
                document: $document,
                onMutationCommitted: handleMutationCommitted
            )
            .overlay(alignment: .topLeading) {
                AgentCursorOverlay(
                    state: cursors,
                    screenPositionProvider: { _ in nil }
                )
            }
        }
    }

    private func handleCommand(_ command: EditorCommand) {
        // Convert EditorCommand to Mutation, route through
        // DocumentStore.apply (signs receipt, persists, etc.)
    }

    private func handleMutationCommitted(_ mutations: [Mutation], _ queueItem: ChatQueueItem) {
        // Persist the receipt + chat queue.
    }
}
```

## 8. ASCII sketch of the editor with two cursors

```
┌─ TesseraEditorToolbar ───────────────────────────────────────────────────┐
│ [B] [I] [U] [S] [</>]  |  [Paragraph ▾]  |  [▦] [📷] [</>] [💡] [•] [1.] │
└──────────────────────────────────────────────────────────────────────────┘
┌─ TesseraEditorView (NSTextView) ─────────────────────────────────────────┐
│ 1 │  # Project Proposal                                                     │
│   │                                                                          │
│   │  We propose to build a productivity surface that combines                │
│   │  ⚙ [user cursor, blinking]      🤖 [agent cursor, blinking]               │
│   │  a semantic block AST with an AI-driven editor. The user and the          │
│   │  agent share one undo stack, one mutation API, and one receipt            │
│   │  chain — making every edit auditable.                                    │
│   │                                                                          │
│ 2 │  ## Architecture                                                         │
│   │                                                                          │
│   │  ──────────────────────────────────────────────────                       │
│   │                                                                          │
│   │  ```swift                                                                │
│   │  public enum Mutation { ... }                                           │
│   │  ```                                                                    │
│   │                                                                          │
└──┬─────────────────────────────────────────────────────────────────────────┘
   │  [agent cursor: blue, blinking, in block 2]
   │  [user cursor: black, standard, in block 1]
   ▼
   Both cursors visible. User can click anywhere without affecting the
   agent's cursor. Cmd-Z undoes the most recent receipt; the macOS Edit
   menu's "Undo" item shows the receipt's summary (e.g., "Undo insert
   paragraph block").
```

## 9. Phase 2 vs. Phase 1 — what was added

Phase 1 shipped the data layer (Block AST, Mutation API, Receipts, ReceiptUndoManager, TextCursor/CursorPair, ChatQueue, DocumentStore). Phase 2 builds the editor on top:

| Layer | Phase 1 (data) | Phase 2 (editor) |
|---|---|---|
| AST | `Block`, `BlockType`, `InlineRun`, `DocumentAST` | unchanged |
| Mutation API | `Mutation`, `MutationEngine`, `MutationError` | unchanged (consumed by `TextEditReducer`) |
| Receipts | `Receipt`, `C2PAManifest`, `ReceiptSigner` | unchanged (consumed by `EditorUndoCoordinator`) |
| Undo | `ReceiptUndoManager`, `UndoError` | unchanged (wired to `NSResponder.undoManager` via `AppKitUndoManagerBridge`) |
| Cursor | `TextCursor`, `CursorPair` | extended with `EditorCursorState` (two cursors + active flag), `CursorSelection`, `CursorInBlock` |
| Chat queue | `ChatQueueItem`, `ChatQueue` | unchanged (consumed by `EditorCoalescer` to emit user-edit queue items) |
| **Text view** | none | `TesseraTextContentManager` + `TesseraTextElement` + `IntTextLocation` |
| **Renderer** | none | `BlockRenderer` (pure `Block -> NSAttributedString`) |
| **Edit reducer** | none | `TextEditReducer` (`NSAttributedString` diff → `Mutation`) |
| **Coalescer** | none | `EditorCoalescer` (1.5s window, `ChatQueueItem` emission) |
| **Animations** | none | `AnimationPrimitives` (7 SwiftUI animations) |
| **Highlighter** | none | `CodeBlockHighlighter` (Splash + regex) |
| **View layer** | none | `TesseraEditorView` + `TesseraEditorToolbar` + `EditorUndoCoordinator` + `AgentCursorOverlay` |

## 10. Open questions for the architect

1. **STTextView swap timing.** Phase 2.1 (immediate next worker) or deferred to Phase 5 (per-surface wrappers)?
2. **RichTextKit swap timing.** Same question.
3. **Per-Materials wrappers (Documents / Notes / Code).** Phase 5 worker; should it land before or after the chat panel (Phase 3)?
4. **iOS UIViewRepresentable.** The iOS app's editor surface — should it land with Phase 5, or be a separate worker?

## 11. What's next (Phase 3+)

- **Phase 3:** chat panel UI, receipt drawer UI, "Hold your horses" dialog, `TesseraTextContentManager` ↔ `NSTextView` live wiring.
- **Phase 4:** importers / exporters (Python + Pandoc). `Block AST` is the format they emit/consume.
- **Phase 5:** per-Materials-surface wrappers (Documents / Notes / Code). iOS `UIViewRepresentable`. `MarkdownUI` for Notes. The `EditorMode` parameter is the seam.
- **Phase 6:** Contacts + Graph viz.

## 12. Commits

- `feat/prod-editor`: `editor: Phase 2 productivity editor (...)` — the work (24 files, ~5,200 LoC).
- `feat/prod-editor`: `report: Phase 2 productivity editor worker report` — this report.

Both with `Assisted-by: MiniMax`. No push, no PR.
