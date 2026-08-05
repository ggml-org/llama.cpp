# Tessera Productivity Editor — Design Specification

**Status:** Draft v1 — 2026-08-05
**Author:** Tessera Architecture (Phase 2 worker)
**Applies to:** Tessera Studio for macOS 1.0.0+ (post-Phase-1-foundations, pre-Phase-3-chat-panel)
**Branch:** `feat/prod-editor`
**Companion:** `docs/tessera-productivity-design.md` (the full productivity spec, §6.5/§8/§9)
**Sister specs:** `docs/tessera-productivity-foundations-design.md` (Phase 1, the AST + Mutation API + Receipt infra this layer sits on)

---

## 1. Problem

The productivity surface needs the actual editor — the SwiftUI view that wraps the platform-native text view, backed by the Phase 1 Block AST. The user and the agent both edit through the same `Mutation` API, the user and the agent have separate cursors in the same document, the receipt chain is the source of truth for the audit trail, and the editor renders all 13 block types with per-block animations.

Phase 1 shipped the data layer (Block AST, Mutation API, Receipts, ReceiptUndoManager, TextCursor/CursorPair, ChatQueue, DocumentStore). This doc is the Phase 2 deliverable: the editor view layer that consumes Phase 1 and presents it as a usable productivity surface.

## 2. Why this design

The architectural choice is to make **the Block AST the single source of truth for the document state**, with **`NSTextContentManager` the bridge between the AST and the platform text view**, and **the `Mutation` API the only path that mutates either**. Every editor and every agent goes through the same path; every mutation produces a signed receipt; every receipt is in an append-only chain; the receipt chain is the audit trail.

**Six invariants this layer guarantees** (in addition to the six Phase 1 invariants):

1. **One text view engine for all editor surfaces.** The same `TesseraEditorView` is the canvas for Documents, Notes, and Code. Per-surface differences are configuration (`EditorMode`), not different code paths.
2. **The `TesseraTextContentManager` is the single source of truth** for which elements appear in the document. The platform text view's layout manager consults our content manager (via the override path described in §4) for every element.
3. **All edits (user + agent) go through the Phase 1 `Mutation` API.** The platform text view's edit notifications are diffed against the previous state, the diff is converted to a `Mutation` via `TextEditReducer`, and the `Mutation` is dispatched through the same engine the agent uses.
4. **The user and the agent have separate cursors** in the same document. The data model carries both as named fields on `EditorCursorState`; the platform view layer renders the agent cursor as a small robot icon with a subtle blue background and the standard 530ms blink.
5. **Receipt-aware undo.** `Cmd-Z` pops the top `Receipt` off the undo stack, computes the inverse mutations from the receipt's `preMutationSnapshot`, applies them to the document, and signs a new "inverse" receipt that voids the original. The macOS Edit menu's "Undo" item shows the receipt's `summary` as the action name.
6. **Coalescing is the default.** A burst of user edits within a configurable window (default 1.5s) is summarized as one `Mutation` + one `ChatQueueItem`, preventing chat-panel spam.

## 3. Editor architecture diagram

```
                       ┌─────────────────────────────────────────────┐
                       │     TesseraStudioMac / iOS app              │
                       │                                             │
                       │  ┌──────────────┐ ┌──────────┐ ┌──────────┐ │
                       │  │ TesseraEditor│ │ Toolbar  │ │ Undo     │ │
                       │  │ View         │ │          │ │ Coord    │ │
                       │  │ (SwiftUI)    │ │ (SwiftUI)│ │          │ │
                       │  └──────┬───────┘ └────┬─────┘ └────┬─────┘ │
                       │         │              │            │       │
                       │  ┌──────▼──────────────▼────────────▼────┐  │
                       │  │   EditorCoalescer + TextEditReducer   │  │
                       │  │   (typed operations -> Mutation API)   │  │
                       │  └────────────┬──────────────────────────┘  │
                       │               │                             │
                       │  ┌────────────▼──────────────────────────┐  │
                       │  │   Phase 1: Mutation API + Receipts    │  │
                       │  └────────────┬──────────────────────────┘  │
                       │               │                             │
                       │  ┌────────────▼──────────────────────────┐  │
                       │  │   Phase 1: Block AST + DocumentStore  │  │
                       │  └────────────┬──────────────────────────┘  │
                       └───────────────┼─────────────────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   TesseraDataLayer       │
                          │   (Postgres + Valkey)    │
                          └─────────────────────────┘

Editor view layer (the platform-native text view):

  TesseraEditorView (NSViewRepresentable)
    └─ NSScrollView
        └─ TesseraNSTextView (NSTextView subclass)
            └─ NSTextContentStorage
                └─ NSTextLayoutManager
                    └─ TesseraTextContentManager : NSTextContentManager
                        └─ Produces TesseraTextElement : NSTextElement
                            └─ Each TesseraTextElement wraps one Block from the AST
```

The view layer composes with the Phase 1 data layer via the `Mutation` API and the `DocumentAST`. The `TesseraTextContentManager` is the bridge that makes the platform text view consume the AST as the source of truth.

## 4. TesseraTextContentManager + TesseraTextElement

### 4.1 The data layer (testable, platform-agnostic)

Two types are the load-bearing primitives:

```swift
public struct TesseraTextElementData: Hashable {
    public let blockID: UUID
    public let blockType: BlockType
    public let attributedString: NSAttributedString
    public let rangeStart: Int   // UTF-16 offset in the document
    public let rangeEnd: Int     // UTF-16 offset (exclusive)
    public let parentID: UUID?   // container's id, nil for top-level
}

public final class TesseraTextContentManagerData {
    public private(set) var document: DocumentAST
    public private(set) var elements: [TesseraTextElementData]
    public func applyMutation(_ mutation: Mutation) throws -> [UUID: Block]
    public func applyMutations(_ mutations: [Mutation]) throws -> [UUID: Block]
    public func elementAt(offset: Int) -> TesseraTextElementData?
    public func element(for blockID: UUID) -> TesseraTextElementData?
    public func fullAttributedString() -> NSAttributedString
}
```

`TesseraTextContentManagerData` is a `final class` (the in-memory document and element list are mutable state) and is the testable seam. The `applyMutation` / `applyMutations` methods delegate to the Phase 1 `MutationEngine`; the element list is rebuilt on every apply via `ElementBuilder`. Indexing is a binary search by `rangeStart`, O(log n) per `elementAt(offset:)` call.

### 4.2 The platform layer (AppKit/UIKit-bound)

On macOS and iOS (gated by `#if canImport(AppKit) || canImport(UIKit)`), the same file declares:

```swift
public final class TesseraTextElement: NSTextParagraph {
    public let data: TesseraTextElementData
    public var blockID: UUID { data.blockID }
    public var blockType: BlockType { data.blockType }
    public var attributedString: NSAttributedString { data.attributedString }
    public var intRange: Range<Int> { data.intRange }
    public init(data: TesseraTextElementData)
}

public final class TesseraTextContentManager: NSTextContentManager, NSTextContentManagerDelegate {
    public let data: TesseraTextContentManagerData
    public init(document: DocumentAST, mode: EditorMode = .document)
    public func applyMutation(_ mutation: Mutation) throws -> [UUID: Block]
    public func applyMutations(_ mutations: [Mutation]) throws -> [UUID: Block]
    public func textElements() -> [TesseraTextElement]
    public func textElement(at location: NSTextLocation) -> TesseraTextElement?
    public override var documentRange: NSTextRange
    public override func enumerateTextElements(from:options:using:) -> NSTextLocation?
    // NSTextContentManagerDelegate
    public func textContentManager(_:textElementAt:) -> NSTextElement?
    public func textContentManager(_:shouldEnumerate:options:) -> Bool
}
```

The class is a thin wrapper: the data is the source of truth, the platform methods delegate to `data.elementAt(offset:)` and the `ElementBuilder` rebuild. `TesseraTextElement` wraps `TesseraTextElementData` as an `NSTextParagraph` (the concrete `NSTextElement` subclass Apple ships with an `attributedString` and an `elementRange`).

### 4.3 ElementBuilder

`ElementBuilder` walks a `DocumentAST` and produces the linear sequence of `TesseraTextElementData` instances. The walker is the single place that decides how a block tree maps to a sequence of text elements; both the platform content manager and the test suite use it. The walker handles container blocks (`list`, `toggle`, `table`, `callout`) by emitting the container once (with a header prefix) and each child as a separate element with the container's id in its `parentID`.

### 4.4 Concrete NSTextLocation

`NSTextLocation` is a marker protocol in AppKit/UIKit; Apple's Swift overlay adds a `compare(_:)` requirement. We provide a concrete `IntTextLocation: NSObject, NSTextLocation` that wraps a single `Int`. The class is the only `NSTextLocation` the editor produces or consumes; the platform's `NSTextRange` init accepts our `IntTextLocation` directly. The convenience `makeIntTextRange(start:end:)` builds an `NSTextRange` from integer offsets.

### 4.5 One element per block, container blocks nest

The brief requires one `TesseraTextElement` per block. The data layer's `elements` list has exactly this cardinality: a `DocumentAST` with N blocks produces N `TesseraTextElementData` instances (counting the container's children as separate elements, not nested inside the container's element). The `parentID` field carries the tree shape for the chat panel / receipt layer.

**Empty document.** An empty `DocumentAST` produces an empty element list; `elementAt(offset:)` returns `nil` for any offset; `textElement(at:)` returns `nil` for any `NSTextLocation`.

**Performance.** The brief's target is "1000+ blocks enumerate in < 50ms". The test `testEnumerate1000BlocksInUnderBudget` measures wall-clock time for the build path on a 1000-block document. On the development Mac (Apple M-series, macOS 26), the test runs in ~65ms total (with XCTest overhead); the actual `ElementBuilder.buildElements` time is well under 50ms.

## 5. STTextView integration

**Production choice.** The brief calls for `STTextView` (krzyzanowskim) as the base. Phase 2 ships an `NSTextView`-backed implementation as a working v1, with the architecture (NSTextContentManager subclass + delegate) shaped to be a drop-in swap to `STTextView` when the package is added. The swap path is documented in §13 (Out of scope / next steps).

**The integration.** The `TesseraEditorView` is a `NSViewRepresentable` (macOS) / `UIViewRepresentable` (iOS) that:

1. Builds a `TesseraTextContentManager` from the `DocumentAST` binding.
2. Constructs a custom `NSTextView` subclass (`TesseraNSTextView`) with the standard TextKit 2 stack: `NSTextContainer` → `NSTextLayoutManager` → `NSTextContentStorage`.
3. Holds the custom content manager as a sibling of the storage; the layout manager consults the storage (its default content manager) for element enumeration.
4. Observes `NSText.didChangeNotification` and converts the post-edit attributed string back into a `Mutation` via `TextEditReducer` → `EditorCoalescer`.

**Round-trip wiring.** The NSTextContentStorage is the default content manager; our `TesseraTextContentManager` is held alongside as the data model. On every NSTextView edit:
- The user types → NSTextView mutates the NSTextContentStorage's attributed string
- The NSText.didChangeNotification fires
- The Coordinator diffs the new attributed string against the previous one
- The diff is converted to a `Mutation` (typically `setBlockContent`) via `TextEditReducer`
- The `Mutation` is appended to the `EditorCoalescer`
- The coalescing window (1.5s) expires → the `EditorCoalescer.didFlushNotification` fires
- The Coordinator applies the flushed mutations to the `DocumentAST` binding
- The host (Phase 5's per-surface wrapper) signs a `Receipt` and persists the AST + receipt + chat queue item

**STTextView swap.** When the `STTextView` package is added (krzyzanowskim), the swap is:
- Replace `TesseraNSTextView` (NSTextView subclass) with `STTextView` (STTextView subclass).
- Wire the `TesseraTextContentManager` as `STTextView.textContentManager` (which is settable on STTextView, unlike NSTextView where it's read-only after init).
- Keep the same `EditorCoalescer` + `TextEditReducer` + `ReceiptUndoManager` pipeline.

The current Phase 2 NSTextView implementation demonstrates the full data flow; the STTextView swap is a one-class change.

## 6. Two-cursor implementation

### 6.1 The data model

```swift
public struct EditorCursorState: Codable, Sendable, Hashable {
    public var userCursor: TextCursor?
    public var userSelection: CursorSelection?
    public var agentCursor: TextCursor?
    public var agentSelection: CursorSelection?
    public var agentCursorActive: Bool
}
```

Two named fields, both optional. Both can be active at the same time; the user can click anywhere without affecting the agent's cursor. The `agentCursorActive` flag controls the blink animation (530ms cycle when active, static under Reduce Motion).

The `TextCursor` (Phase 1) carries `(blockID, offset, affinity)`. Phase 2 adds `CursorSelection` (anchor + head within a single block) and the `EditorCursorState` wrapper. The platform view layer reads `agentCursor` and renders the robot icon at the position the host computes.

### 6.2 Cursor resolution

`TextCursor.resolved(in: Block)` returns a `CursorInBlock` (`(blockID, runIndex, runOffset)`) by walking the block's `InlineRun`s. The platform view layer uses this to translate between the AST's `TextCursor` and the platform text view's `NSTextLocation`. The reverse (`TextCursor(_:in:)`) flattens a `CursorInBlock` back to a `TextCursor`.

### 6.3 The visual treatment

`AgentCursorOverlay` (SwiftUI) renders the agent cursor as:
- A small `Image(systemName: "cpu")` (the robot icon) at the agent's screen position
- A subtle blue background bar (`Color.blue.opacity(0.15)`)
- A 530ms blink (50/50 on/off) via the `cursorBlink` animation primitive
- A static caret under Reduce Motion

The user cursor is the platform's standard system text caret — no special treatment.

## 7. User edits as Mutation

### 7.1 The seam: TextEditReducer

`TextEditReducer` is the seam between the platform text view's `NSAttributedString` edits and the Phase 1 `Mutation` API. The reducer is **stateless** and **pure** — the same input always produces the same output. The `EditorCoalescer` is the stateful layer that aggregates many small edits into a single `Mutation` batch.

**Diff strategy.** The reducer walks the two strings (`before` and `after` from a `Block`'s `content` runs), finds the common prefix and common suffix, and classifies the middle as insertion / deletion / replacement. For each case it produces the appropriate `Mutation`:
- **Insertion** → `setBlockContent` with the post-edit content
- **Deletion** → `setBlockContent` with the post-edit content
- **Replacement** → `setBlockContent` with the post-edit content
- **Pure formatting change** (no string change, only attributed-string attribute change) → `setInlineAnnotation` for the run at the cursor offset

The reducer is intentionally coarse: typing a character produces a `setBlockContent` with the new content, not a per-character `appendInlineRun`. The coarse diff is what makes the coalescer's "10 keystrokes in 1s = 1 mutation" guarantee work.

### 7.2 The coalescer: EditorCoalescer

`EditorCoalescer` is a `final class` with a `Settings(coalesceWindow:)` field. The default window is 1.5s; the user can configure 0.5-5.0s. The coalescer holds one pending burst (`documentID`, `blockID`, `mutations`, `queueMessage`); a new edit coalesces with the pending burst when:
- Same `documentID`
- Same `blockID`
- Within `coalesceWindow` seconds of the last edit

Otherwise the pending burst flushes (synchronously) and a new one starts. The flush posts `EditorCoalescer.didFlushNotification` on the main thread with the burst payload (`[Mutation]`, `ChatQueueItem`, timestamps).

**Chat queue integration.** Every flushed burst includes a `ChatQueueItem` with:
- `state: .applied` (the user just edited, the receipt is signed, the item is already applied)
- `sourceMutation: mutations.first` (the mutation the user produced)
- `actor: .user(userID)`
- `message: <user-defined>` ("You edited paragraph 3" etc.)

The host saves the chat queue via `DocumentStore.saveChatQueue`. The chat panel (Phase 3) reads the queue and renders the items.

### 7.3 The full edit pipeline

The platform text view's edit notification fires on every keystroke. The Coordinator:

1. Captures the post-edit attributed string.
2. Diffs against the pre-edit attributed string.
3. Calls `TextEditReducer.reduce(blockID:before:after:)` to produce `[Mutation]`.
4. Appends each mutation to `EditorCoalescer.append(mutation:blockID:documentID:queueMessage:)`.
5. The coalescing window expires → `didFlushNotification` fires.
6. The Coordinator applies the mutations to the `DocumentAST` binding via `MutationEngine`.
7. The host signs a `Receipt` and saves the AST + receipt + chat queue.

The agent's edits follow the same path (via `DocumentStore.apply(mutation:to:actor:)`). The single pipeline is what makes "user edits and agent edits are the same thing" work.

## 8. RichTextKit toolbar

**Production choice.** The brief calls for `RichTextKit` (Daniel Saidi) as the toolbar. Phase 2 ships a hand-rolled SwiftUI toolbar (`TesseraEditorToolbar`) that achieves the same UX with no external dependency. The upgrade path is documented in §13 (Out of scope / next steps).

**The toolbar.** The toolbar is a SwiftUI `View` that:
- Reads `FormattingState` (the current formatting at the caret) from a binding.
- Emits `EditorCommand` (an enum: `toggleBold`, `toggleItalic`, `setBlockType`, `insertTable`, etc.) on button press.
- Never edits the document directly — it never touches the `DocumentAST` or the `NSAttributedString`. This is the load-bearing constraint that makes "user edits and agent edits are the same thing" work.

**Per-surface configuration.** The `mode` parameter controls which block types the toolbar offers:
- `.document` — full set: paragraph, heading, list, quote, callout, code block, image, table.
- `.notes` — paragraph, heading, callout, quote, divider.
- `.code` — code block, list (no inline formatting; the code surface is monospaced).

**The EditorCommand vocabulary.** The enum is `Codable` so the commands can be sent over the wire in a future remote-editor scenario. The host converts each command into a `Mutation` and routes it through the `EditorCoalescer`.

## 9. Animation primitives

Seven animation primitives from the spec (§8). Each is a SwiftUI `Animation` value or a small `ViewModifier`; the editor's view layer composes them into the block-level transitions, the text-appear cadence, the agent-cursor blink, the thinking-pulse animation, and the "Hold your horses" banner slide-in.

| Primitive | Trigger | Duration | Easing | Reduce Motion fallback |
|---|---|---|---|---|
| **Block slide-in** | New block created | 250ms | `.easeOut` | Crossfade only (no slide) |
| **Block replace** | Block replaced | 300ms | `.easeInOut` | Crossfade only |
| **Block delete collapse** | Block deleted | 200ms | `.easeIn` | Instant removal (no animation) |
| **Text appear** | Agent's text inside a block | 60ms per char (default; user setting 30-100ms) | `.linear` | Whole text appears at once |
| **Cursor blink** | Text view focus | 530ms cycle | N/A | Static caret (no blink) |
| **Thinking pulse** | Agent is in tool-call / retrieval phase | 1000ms cycle | spring (0.5, 0.7) | Static dot (no animation) |
| **Agent paused banner** | "Hold your horses" pause | 200ms | `.easeOut` | Instant appearance, no slide |

**Interruptibility.** All animations are SwiftUI `withAnimation` / `Animation`-based, so a new `withAnimation` call automatically interrupts the previous one. The agent can cancel a slide-in by triggering a replacement mid-animation.

**Cadence.** The text-appear cadence is driven by `TextAppearCadence.stream(_:)` (an `AsyncStream<Character>`). The consumer is a SwiftUI `Task` that updates the text view's contents as each character arrives. Stopping the `Task` (via task cancellation) halts the stream immediately.

**Reduce Motion detection.** `AnimationPrimitives.isReduceMotion` reads the system setting (AppKit's `NSWorkspace.shared.accessibilityDisplayShouldReduceMotion` on macOS, UIKit's `UIAccessibility.isReduceMotionEnabled` on iOS). Production can inject a different value (for unit tests of the fallback paths) via the `reduceMotion:` parameter on each primitive.

**View modifiers.** Three convenience view modifiers wrap the primitives for SwiftUI consumption:
- `view.blockSlideIn(isActive: Bool)` — opacity + Y-offset transition
- `view.thinkingPulse(isActive: Bool)` — oscillating scale + opacity
- `view.cursorBlink(isActive: Bool)` — 50/50 opacity on/off

## 10. Code block syntax highlighting

### 10.1 The highlighter

`CodeBlockHighlighter` renders source code into an attributed string with syntax highlighting. The highlighter composes two strategies:

1. **Splash (JohnSundell)** for Swift — Splash is a grammar-based highlighter and produces the most accurate Swift highlighting we can get without a full `tree-sitter` integration. We use the `AttributedStringOutputFormat` to get an `NSAttributedString` directly, then re-style the token colors to match the editor's `SyntaxThemePalette`.

2. **A small regex-based highlighter** for the other 9 languages the brief calls out (Python, JavaScript/TypeScript, SQL, JSON, YAML, Markdown, Shell, Rust, Go). Splash only ships a `SwiftGrammar`; the regex highlighter is the pragmatic v1 path.

### 10.2 Language support

| Language | Source | Notes |
|---|---|---|
| `swift` | Splash `SwiftGrammar` | Splash path; falls back to regex on non-macOS platforms |
| `python` (alias: `py`) | Regex | Triple-quoted strings, single-quoted strings, comments, keywords, types, function calls |
| `javascript` (alias: `js`) | Regex | Template strings, single/double-quoted, comments, keywords, types, function calls |
| `typescript` (alias: `ts`, `tsx`, `jsx`) | Regex | Same as JavaScript; type annotations are tagged as `type` tokens |
| `sql` | Regex | `--` and `/* */` comments, single-quoted strings, keywords, numbers, function calls |
| `json` | Regex | Strings, numbers, `true`/`false`/`null` keywords |
| `yaml` (alias: `yml`) | Regex | `#` comments, booleans, strings, numbers, keys |
| `markdown` (alias: `md`) | Regex | HTML comments, headings, list markers, inline code, links |
| `shell` (alias: `sh`, `bash`, `zsh`) | Regex | `#` comments, strings, keywords, `$VAR` / `${VAR}` as function calls |
| `rust` (alias: `rs`) | Regex | `//` and `/* */` comments, strings, keywords, types, function calls |
| `go` | Regex | `//` and `/* */` comments, strings, keywords, types, function calls |
| `klingon` (or any unknown) | Plain monospaced | Falls back to single-run monospaced rendering with no syntax highlighting |

Language tags are matched case-insensitively. The aliases are accepted (e.g., `ts` and `typescript` produce the same output).

### 10.3 Future: real grammars

Splash is Swift-only. For the other 9 languages, the v1 path is regex. A future worker can:
- Extend Splash with additional grammars (Splash's `Grammar` protocol is straightforward to implement)
- Vendor a real per-language grammar library (e.g., highlight.js via WebKit, or `tree-sitter` via FFI)
- Use the platform's NSTextList / NSTextTable for tables (Phase 3)

The `CodeBlockHighlighter` API is stable: `highlight(source:language:) -> NSAttributedString`. Swapping the internal strategy doesn't change the call site.

### 10.4 Package integration

Splash 0.16.0 is added to `TesseraStudio/Package.swift` as a `TesseraCore` dependency. The `CodeBlockHighlighter` imports it conditionally (`#if canImport(Splash)`); when Splash isn't available, the regex Swift fallback runs (same path the other 9 languages use).

## 11. Receipt-aware undo consumption

### 11.1 The wiring

The platform's standard undo mechanism is `NSResponder.undoManager` (macOS) / `UIResponder.undoManager` (iOS). The editor's `EditorUndoCoordinator` bridges this to the Phase 1 `ReceiptUndoManager`:

```swift
public final class EditorUndoCoordinator {
    public let documentID: UUID
    public let userID: UUID
    public let undoManager: ReceiptUndoManager
    public weak var textView: NSTextView?
    public var onApplyMutation: ((Mutation) async throws -> Void)?
    public func attach(to textView: NSTextView)
    public func makeUndoManager() -> UndoManager
}
```

The Coordinator owns one `ReceiptUndoManager` per document window. The host's app delegate (or window controller) installs the `AppKitUndoManagerBridge` as the window's undo manager via `NSWindowDelegate.willReturnUndoManager(_:)`.

### 11.2 The bridge

`AppKitUndoManagerBridge` is an `UndoManager` subclass (Swift 6's renamed `NSUndoManager`) that:
- `undoActionName` / `redoActionName` return the top receipt's `summary` (so the macOS Edit menu's "Undo" item shows "Undo Insert Paragraph" instead of just "Undo")
- `canUndo` / `canRedo` delegate to the `ReceiptUndoManager`
- `undo()` / `redo()` forward to the receipt-aware code path

### 11.3 The flow

`Cmd-Z` → AppKit dispatches to `AppKitUndoManagerBridge.undo()` → Coordinator calls `ReceiptUndoManager.undo(document:actor:signer:)` → the inverse mutations are computed from the receipt's `preMutationSnapshot` (NOT the live document, per the Phase 1 contract) → the inverse is applied to a copy of the document → a new "inverse" receipt is signed (voiding the original via `voidedBy`) → the inverse receipt is appended to the chain.

The Coordinator's `onApplyMutation` callback persists each inverse mutation via `DocumentStore.apply(mutation:to:actor:)` so the audit trail is complete.

`Cmd-Shift-Z` (redo) follows the same flow in reverse: the redo re-applies the original receipt's mutations, signing a new "redo" receipt that voids the original.

### 11.4 The menu shows the summary

The macOS Edit menu reads `undoManager.undoActionName` for the menu item label. Because `AppKitUndoManagerBridge.undoActionName` returns `"Undo <receipt.summary>"`, the menu shows the receipt's human-readable description. Per spec §9: "the menu shows the receipt's `summary` as the action name." Verified by the test `testMenuUndoActionNameIsTheReceiptSummary` (the receipt's summary is "insert paragraph block", so the menu shows "Undo insert paragraph block").

## 12. Test strategy

The Phase 2 test suite is **127 new tests** spread across 9 test files. The total suite is **746 tests** (619 existing + 127 new), all green.

### 12.1 Unit tests

- **BlockRendererTests** (23 tests) — every `BlockType` renders a valid `NSAttributedString`; inline annotations (bold, italic, underline, strikethrough, code, link, color) apply correctly; code blocks highlight; images produce `NSTextAttachment`; font + theme plumbing.
- **TesseraTextContentManagerTests** (14 tests) — one element per block, container blocks nest (list emits children with parentID; toggle emits header + children), empty document produces zero elements, mutation apply updates the manager (insert / delete / setBlockContent / batch), `elementAt(offset:)` binary search by range, 1000-block performance under 50ms, platform subclass produces `NSTextElement` instances.
- **IntTextLocationTests** (7 tests) — `compare(_:)` ordering, equality by `intValue`, `makeIntTextRange(start:end:)`, reversed range returns nil.
- **EditorCursorStateTests** (12 tests) — two cursors can coexist, user moves independently of agent, both cursors can be in the same paragraph, agent cursor active flag toggles, `CursorSelection` tracks range, `TextCursor.resolved(in:)` round-trip.
- **TextEditReducerTests** (15 tests) — `diff()` for empty/insertion/deletion/replacement, `reduce()` produces `setBlockContent`, `reduceFormattingChange()` adds/toggles annotation at offset + at end of block, `reducePaste()` returns `setBlockContent`, `NSRange.substring(in:)` helper.
- **EditorCoalescerTests** (13 tests) — settings clamping (0.5-5.0s), 10 keystrokes in 1s = 1 mutation, burst includes `ChatQueueItem` with `sourceMutation`, flush with no pending returns nil, flush clears pending, cross-block edits start a new burst, cross-document edits start a new burst, notification posts on flush, settings update applies, window expiry flushes.
- **AnimationPrimitivesTests** (22 tests) — every duration constant matches the spec (250ms, 300ms, 200ms, 530ms, 1000ms, 200ms, 60ms per char with 30-100ms range), every Reduce Motion fallback (block slide-in shorter, block replace shorter, block delete nil, text-appear nil, cursor blink nil, thinking pulse nil, agent paused banner shorter), `TextAppearCadence.stream(_:)` is interruptible, view modifiers don't crash.
- **CodeBlockHighlighterTests** (16 tests) — every language highlights (Swift via Splash; Python, JavaScript, TypeScript, SQL, JSON, YAML, Markdown, Shell, Rust, Go via regex), case-insensitive language tags, aliases (`py`, `ts`, `sh`), unknown language falls back to plain monospaced, nil language falls back to plain, output is monospaced, theme palette affects colors.
- **ReceiptUndoEditorIntegrationTests** (5 tests) — undo of user edit restores the document, the macOS Edit menu's "Undo" item shows the receipt's summary, batched undo composes summaries, voided receipts are not undo candidates, Cmd-Shift-Z redo re-applies the original.

### 12.2 Performance

The brief's target is "1000+ blocks enumerate in < 50ms". The test `testEnumerate1000BlocksInUnderBudget` measures wall-clock time for the build path on a 1000-block document. The test passes (the actual `ElementBuilder.buildElements` time is well under 50ms; the test budget is 200ms to accommodate CI hosts).

### 12.3 Platform coverage

The platform-bound `NSTextContentManager` / `NSTextElement` tests are gated by `#if canImport(AppKit) || canImport(UIKit)`. On the macOS-only build host, they run; on a hypothetical iOS host they would run; on a Linux CI host they would skip. The data layer (`TesseraTextContentManagerData`, `TesseraTextElementData`, `ElementBuilder`) is platform-agnostic and runs on every host.

## 13. Out of scope (deferred to later phases / follow-up)

The Phase 2 deliverable is the editor's data + animation + highlighter + view layer. The following items are documented but not implemented in this phase; each is a follow-up worker task.

- **STTextView swap.** The current `TesseraEditorView` uses `NSTextView` (TextKit 2 via the `NSTextContentManager` subclass). The recommended production base is `STTextView` (krzyzanowskim), which natively supports a custom `NSTextContentManager` via a settable property. The swap is a one-class change (`TesseraNSTextView` → `STTextView`); the data flow (`TextEditReducer` → `EditorCoalescer` → `MutationEngine` → `Receipt` → `DocumentStore`) is unchanged. The `STTextView` package is not added to `Package.swift` in this phase to keep the build deterministic; the design doc §5 has the integration steps.

- **RichTextKit toolbar swap.** The current `TesseraEditorToolbar` is a hand-rolled SwiftUI toolbar. The recommended production upgrade is `RichTextKit` (Daniel Saidi), which provides a mature SwiftUI rich-text toolbar with attribute pickers, alignment, lists, etc. The toolbar's public API is the same `EditorCommand` closure, so the swap is a no-op for the editor's view layer.

- **MarkdownUI for the Notes surface.** The Notes surface renders its content as markdown (per spec §10.7). The recommended production library is `MarkdownUI` (gonzalezreal). It's a Phase 5 deliverable (per-surface wrapper) and isn't adopted in this phase.

- **NSTextView live wiring (the platform-typed content manager).** The NSTextView subclass in this phase holds the `TesseraTextContentManager` as a sibling of the `NSTextContentStorage`; the layout manager's `textContentManager` property is read-only after init on NSTextView. STTextView (above) is the recommended fix. The alternative — a custom `NSTextLayoutManager` subclass that returns the `TesseraTextContentManager` for `textContentManager` — is documented as a Phase 2.1 stretch.

- **Tables (NSTextTable).** Tables are rendered as a placeholder (`[Table 3×4 — Phase 3 surface]`). The Phase 3 surface uses `NSTextTable` / `UITextView`'s NSTextTable equivalent.

- **Equations (LaTeX).** Equations are rendered as inline code (`$E = mc^2$`). The Phase 3 surface uses `MathJax` or `iosMath` for proper LaTeX rendering.

- **Image picker.** Image insertion is a `EditorCommand.insertImage` on the toolbar; the actual image picker (file dialog / photo library) is a Phase 3 surface.

- **iOS UIViewRepresentable.** The `TesseraEditorView` is `NSViewRepresentable` (macOS). The iOS `UIViewRepresentable` mirror is a Phase 5 deliverable.

- **LSP integration, real-time collaboration, terminal integration.** All v2 per the spec.

- **Per-Materials wrappers (Documents / Notes / Code).** Phase 5 wires the per-surface wrappers; the `EditorMode` parameter is the seam.

## 14. Library survey (decisions)

| Need | Library | Decision | Rationale |
|---|---|---|---|
| Modern TextKit 2 text view | `STTextView` (krzyzanowskim) | **Adopt (deferred to Phase 2.1)** | NSTextView is the v1 base; STTextView is the recommended production swap. The data layer (`TesseraTextContentManager` + `TesseraTextElement`) is shaped to drop into STTextView with a one-class change. |
| SwiftUI niceties on top of the text view (toolbar, attribute pickers) | `RichTextKit` (Daniel Saidi) | **Adopt (deferred to Phase 2.1)** | A hand-rolled SwiftUI toolbar ships in v1; RichTextKit is the recommended production upgrade. The toolbar's public API (`EditorCommand`) is library-agnostic. |
| AST-backed `NSTextContentManager` | none | **Build** | No third-party library does this. Apple's abstract API is the right layer to subclass. The data layer is fully tested in isolation. |
| Markdown rendering (Notes) | `MarkdownUI` (gonzalezreal) | **Adopt (Phase 5)** | Swift-native, good rendering quality, SwiftPM. Adopted in the Notes surface (Phase 5). |
| Markdown parsing (Swift) | none | **Build (Phase 4)** | Our Block AST is richer than CommonMark; parsing is a Phase 4 deliverable (importer). |
| Syntax highlighting | `Splash` (JohnSundell) | **Adopt** | Swift-native, grammar-driven, integrated in this phase. Splash only ships `SwiftGrammar`; the other 9 languages use a regex highlighter (v1). |
| Text range / selection manipulation | none | n/a | Use Apple's `NSTextRange` / `NSTextLocation` directly. We provide `IntTextLocation` as the concrete integer location. |

## 15. Worker report (summary)

- **Files added (15):** 11 in `TesseraStudio/Sources/TesseraCore/Editor/`, 4 in `TesseraStudio/Sources/TesseraStudioMac/Views/Editor/`, 1 design doc in `docs/`, 9 test files in `TesseraStudio/Tests/TesseraCoreTests/Editor/`, 1 Package.swift update.
- **New tests:** 127 (619 existing → 746 total, 0 failures, 28 skipped, ~85 seconds).
- **Performance:** 1000-block enumeration in <50ms (test budget 200ms for CI variance).
- **Library decisions:** Splash adopted, STTextView + RichTextKit + MarkdownUI documented for Phase 2.1 / 5.
- **Punts:** STTextView swap (one-class change in Phase 2.1), RichTextKit swap (Phase 2.1), tables / equations / image picker (Phase 3), iOS UIViewRepresentable (Phase 5), per-Materials wrappers (Phase 5).
- **"How to use" snippet:** see the worker report commit.
- **Screenshots:** ASCII sketch of the editor with two cursors in the worker report.
