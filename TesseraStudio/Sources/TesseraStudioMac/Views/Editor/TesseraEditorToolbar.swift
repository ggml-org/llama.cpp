import SwiftUI
import AppKit
import TesseraCore

// MARK: - FormattingState

/// The current formatting state at the user's caret. The
/// toolbar reads this to highlight the active buttons
/// (bold, italic, etc.). The state is updated by the
/// platform text view's selection change; Phase 3 wires
/// the live update.
public struct FormattingState: Equatable {
    public var isBold: Bool = false
    public var isItalic: Bool = false
    public var isUnderline: Bool = false
    public var isStrikethrough: Bool = false
    public var isCode: Bool = false
    public var headingLevel: Int? = nil
    public var linkURL: URL? = nil
    public var blockType: BlockType = .paragraph

    public init() {}
}

// MARK: - TesseraEditorToolbar

/// The SwiftUI formatting toolbar that sits above (or
/// below) the editor. The toolbar composes the
/// platform-agnostic primitives (text style buttons,
/// block-type pickers) with custom buttons for the
/// block types the productivity surface promotes
/// (callout, table, image, code block).
///
/// **Per-surface configuration.** The `mode` parameter
/// controls which block types the toolbar offers as
/// one-click inserts:
///   * `.document` — full set: paragraph, heading,
///     list, quote, callout, code block, image, table.
///   * `.notes` — paragraph, heading, callout, quote,
///     divider (no tables, no images, no code blocks).
///   * `.code` — code block, list (no inline
///     formatting; the code surface is monospaced).
///
/// **Actions.** Every action calls into the
/// `EditorCommand` closure with a typed intent; the host
/// (the `TesseraEditorView`'s coordinator) converts the
/// intent into a `Mutation` and routes it through the
/// `EditorCoalescer`. The toolbar never edits the
/// document directly — it never touches the
/// `DocumentAST` or the `NSAttributedString`. This is
/// the load-bearing constraint that makes "user edits
/// and agent edits are the same thing" work.
///
/// **RichTextKit upgrade path.** Phase 2 ships a
/// hand-rolled SwiftUI toolbar. The recommended
/// production upgrade is `RichTextKit` (Daniel Saidi),
/// which provides a mature SwiftUI rich-text toolbar
/// with attribute pickers, alignment, lists, etc. The
/// toolbar's public API is the same `EditorCommand`
/// closure, so the swap is a no-op for the editor's
/// view layer.
public struct TesseraEditorToolbar: View {
    public let mode: EditorMode
    @Binding public var formattingState: FormattingState
    public let onCommand: (EditorCommand) -> Void

    public init(
        mode: EditorMode,
        formattingState: Binding<FormattingState>,
        onCommand: @escaping (EditorCommand) -> Void
    ) {
        self.mode = mode
        self._formattingState = formattingState
        self.onCommand = onCommand
    }

    public var body: some View {
        HStack(spacing: 8) {
            // Inline formatting group
            inlineFormattingGroup
            Divider().frame(height: 18)
            // Block type group
            blockTypeGroup
            if mode != .code {
                Divider().frame(height: 18)
                // Insert group (callout, table, image, code block)
                insertGroup
            }
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.thinMaterial)
    }

    // MARK: - Inline formatting

    @ViewBuilder
    private var inlineFormattingGroup: some View {
        if mode != .code {
            HStack(spacing: 4) {
                ToolbarButton(
                    label: "B",
                    weight: .bold,
                    isActive: formattingState.isBold,
                    shortcut: "⌘B"
                ) { onCommand(.toggleBold) }
                ToolbarButton(
                    label: "I",
                    italic: true,
                    isActive: formattingState.isItalic,
                    shortcut: "⌘I"
                ) { onCommand(.toggleItalic) }
                ToolbarButton(
                    label: "U",
                    underline: true,
                    isActive: formattingState.isUnderline,
                    shortcut: "⌘U"
                ) { onCommand(.toggleUnderline) }
                ToolbarButton(
                    label: "S",
                    strikethrough: true,
                    isActive: formattingState.isStrikethrough
                ) { onCommand(.toggleStrikethrough) }
                ToolbarButton(
                    label: "</>",
                    monospaced: true,
                    isActive: formattingState.isCode,
                    shortcut: "⌘E"
                ) { onCommand(.toggleCode) }
            }
        }
    }

    // MARK: - Block type

    @ViewBuilder
    private var blockTypeGroup: some View {
        HStack(spacing: 4) {
            Picker("Block", selection: blockTypeBinding) {
                Text("Paragraph").tag(BlockType.paragraph)
                Text("Heading 1").tag(BlockType.heading)
                if mode != .code {
                    Text("Heading 2").tag(BlockType.heading)
                    Text("Heading 3").tag(BlockType.heading)
                }
                if mode == .document || mode == .notes {
                    Text("Quote").tag(BlockType.quote)
                    Text("Callout").tag(BlockType.callout)
                }
                if mode == .document {
                    Text("List").tag(BlockType.list)
                    Text("Divider").tag(BlockType.divider)
                }
                if mode == .code {
                    Text("Code Block").tag(BlockType.codeBlock)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 160)
        }
    }

    private var blockTypeBinding: Binding<BlockType> {
        Binding(
            get: { formattingState.blockType },
            set: { newValue in
                formattingState.blockType = newValue
                onCommand(.setBlockType(newValue))
            }
        )
    }

    // MARK: - Insert

    @ViewBuilder
    private var insertGroup: some View {
        HStack(spacing: 4) {
            ToolbarIconButton(systemName: "tablecells") {
                onCommand(.insertTable)
            }
            ToolbarIconButton(systemName: "photo") {
                onCommand(.insertImage)
            }
            ToolbarIconButton(systemName: "chevron.left.forwardslash.chevron.right") {
                onCommand(.insertCodeBlock)
            }
            if mode == .document || mode == .notes {
                ToolbarIconButton(systemName: "exclamationmark.bubble") {
                    onCommand(.insertCallout)
                }
                ToolbarIconButton(systemName: "list.bullet") {
                    onCommand(.toggleUnorderedList)
                }
                ToolbarIconButton(systemName: "list.number") {
                    onCommand(.toggleOrderedList)
                }
            }
        }
    }
}

// MARK: - ToolbarButton

/// A simple text-based toolbar button with an active state.
struct ToolbarButton: View {
    let label: String
    var weight: Font.Weight = .regular
    var italic: Bool = false
    var underline: Bool = false
    var strikethrough: Bool = false
    var monospaced: Bool = false
    var isActive: Bool = false
    var shortcut: String? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 0) {
                Text(label)
                    .font(.system(
                        size: 13,
                        weight: weight,
                        design: monospaced ? .monospaced : .default
                    ))
                    .italic(italic)
                    .underline(underline)
                    .strikethrough(strikethrough)
                if let shortcut {
                    Text(shortcut)
                        .font(.system(size: 8))
                        .foregroundStyle(.secondary)
                }
            }
            .frame(minWidth: 28, minHeight: 22)
            .padding(.horizontal, 4)
            .background(isActive ? Color.accentColor.opacity(0.2) : Color.clear)
            .cornerRadius(4)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - ToolbarIconButton

/// An SF Symbol toolbar button.
struct ToolbarIconButton: View {
    let systemName: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemName)
                .frame(minWidth: 24, minHeight: 22)
                .padding(.horizontal, 4)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - EditorCommand

/// The toolbar's command vocabulary. The toolbar emits
/// one of these for every action; the host converts the
/// command into a `Mutation` and routes it through the
/// `EditorCoalescer`. The enum is `Codable` so the
/// commands can be sent over the wire in a future
/// remote-editor scenario.
public enum EditorCommand: Codable, Equatable, Hashable {
    case toggleBold
    case toggleItalic
    case toggleUnderline
    case toggleStrikethrough
    case toggleCode
    case setBlockType(BlockType)
    case insertTable
    case insertImage
    case insertCodeBlock
    case insertCallout
    case toggleUnorderedList
    case toggleOrderedList
    case insertLink(URL)
    case removeLink
}
