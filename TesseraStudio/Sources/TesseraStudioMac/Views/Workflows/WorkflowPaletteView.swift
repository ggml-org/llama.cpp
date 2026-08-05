import SwiftUI
import TesseraCore

/// The left-hand palette of available node types. Driven directly
/// off ``WorkflowNodeRegistry.allTypeIds`` / ``paletteEntry(for:)``;
/// adding a new ``WorkflowNodeType`` to the registry shows up
/// here without a UI change.
///
/// Rows are drag sources for drag-onto-canvas: the payload is
/// the node type id as a plain String, matched by the canvas's
/// `dropDestination(for: String.self)`. The view also exposes a
/// search field so the palette scales to the 18+ shipped
/// TesseraTools once we wrap more of them as workflow nodes.
struct WorkflowPaletteView: View {
    let registry: WorkflowNodeRegistry
    @State private var query: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Nodes")
                .font(.headline)
                .padding(.horizontal, 12)
                .padding(.top, 10)
                .padding(.bottom, 4)
                .accessibilityAddTraits(.isHeader)
            List(filteredEntries, id: \.typeId) { entry in
                row(entry)
            }
            .listStyle(.sidebar)
            .accessibilityLabel("Node palette")
            .accessibilityHint("Drag a node from here onto the canvas to add it to the workflow")
        }
        // HIG 2.9: use the system search field instead of a
        // hand-rolled TextField; the sidebar placement puts it
        // where macOS users expect list filtering to live.
        .searchable(text: $query, placement: .automatic, prompt: "Filter nodes")
        .frame(minWidth: 220)
    }

    private var filteredEntries: [WorkflowNodePaletteEntry] {
        let entries = registry.allTypeIds.compactMap { registry.paletteEntry(for: $0) }
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if q.isEmpty { return entries }
        return entries.filter {
            $0.typeId.lowercased().contains(q)
                || $0.displayName.lowercased().contains(q)
        }
    }

    private func row(_ entry: WorkflowNodePaletteEntry) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(entry.displayName)
                .font(.system(.body, design: .rounded).weight(.medium))
            Text(entry.typeId)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(entry.summary)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
        .padding(.vertical, 2)
        // The whole row drags, not just the label text; the
        // payload is the type id string the canvas drop target
        // inserts a node for.
        .contentShape(Rectangle())
        .draggable(entry.typeId)
    }
}
