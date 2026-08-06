import SwiftUI
import TesseraCore

// MARK: - CodeOutlineView

/// The outline panel: a flat list of `CodeOutlineItem`s.
/// Clicking a row jumps the editor to the line (the
/// editor pane observes the view model's
/// `pendingScrollLine` and scrolls to it).
///
/// **Why flat, not nested.** The `CodeOutlineItem`s
/// have a `parentID` field, but the v1 panel renders
/// the list flat (with indentation by depth). A nested
/// `OutlineGroup` would give the user a proper
/// collapse/expand experience, but it costs another
/// disclosure state and the user typically wants to
/// scan the whole outline anyway.
public struct CodeOutlineView: View {

    public let outline: CodeOutline
    @State private var kindFilter: CodeOutlineItem.Kind?

    public init(outline: CodeOutline) {
        self.outline = outline
    }

    public var body: some View {
        VStack(spacing: 0) {
            filterBar
            Divider()
            if outline.isEmpty {
                emptyState
            } else {
                outlineList
            }
        }
    }

    private var filterBar: some View {
        HStack(spacing: 4) {
            Image(systemName: "list.bullet.indent")
                .foregroundStyle(.secondary)
            Picker("Kind", selection: $kindFilter) {
                Text("All").tag(CodeOutlineItem.Kind?.none)
                ForEach(CodeOutlineItem.Kind.allCases, id: \.self) { kind in
                    Text(kind.rawValue).tag(CodeOutlineItem.Kind?.some(kind))
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            Spacer()
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
    }

    private var emptyState: some View {
        VStack(spacing: 6) {
            Image(systemName: "list.bullet.rectangle")
                .font(.system(size: 24))
                .foregroundStyle(.secondary)
            Text("No outline available")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            if outline.language == "plain" {
                Text("This file's language is unknown. The outline is empty for unrecognized languages.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 16)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var outlineList: some View {
        let rows: [OutlineRow] = filtered().map { OutlineRow(item: $0) }
        return ScrollView {
            LazyVStack(alignment: .leading, spacing: 2) {
                ForEach(rows) { row in
                    outlineRow(row)
                }
            }
            .padding(.vertical, 4)
        }
    }

    /// A view-shaped wrapper that avoids the
    /// `ForEach(items, id: \.id)` inference trap
    /// (the compiler sometimes tries to interpret
    /// the items as a binding when the closure
    /// body contains a Text on the item's `line`
    /// field).
    private struct OutlineRow: Identifiable {
        let item: CodeOutlineItem
        var id: UUID { item.id }
        var label: String { item.label }
        var kind: CodeOutlineItem.Kind { item.kind }
        var line: Int { item.line }
        var depth: Int { item.depth }
    }

    @ViewBuilder
    private func outlineRow(_ row: OutlineRow) -> some View {
        HStack(spacing: 6) {
            Image(systemName: iconName(for: row.kind))
                .foregroundStyle(.tint)
                .frame(width: 14)
            Text(row.label)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            Text("L\(row.line)")
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
        }
        .padding(.leading, CGFloat(row.depth) * 8)
        .padding(.vertical, 2)
    }

    private func filtered() -> [CodeOutlineItem] {
        guard let kindFilter else { return outline.items }
        return outline.items.filter { $0.kind == kindFilter }
    }

    private func iconName(for kind: CodeOutlineItem.Kind) -> String {
        switch kind {
        case .function, .method: return "function"
        case .class: return "cube"
        case .struct: return "square.stack.3d.up"
        case .enum: return "list.bullet.indent"
        case .proto: return "link"
        case .extension: return "arrow.triangle.branch"
        case .interface: return "circle.dashed"
        case .namespace: return "folder"
        case .property: return "v.square"
        case .constant: return "equal.square"
        case .typealiasKind: return "t.square"
        case .macro: return "wand.and.stars"
        }
    }
}
