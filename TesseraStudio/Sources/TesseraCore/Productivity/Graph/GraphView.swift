import Foundation
import SwiftUI
import Grape

// MARK: - GraphView

/// The main graph view. Renders the visible node + edge
/// set from ``GraphViewModel`` using Grape's
/// `ForceDirectedGraph` (SwiftUI-native, 2D simd, KDTree-
/// accelerated many-body force).
///
/// **Layout:**
///   * **Sidebar (left)**: type filter chips + search box +
///     visibility radius slider. Built on top of the view
///     model.
///   * **Canvas (center)**: the force-directed graph. The
///     Grape view handles pan + zoom via its built-in
///     gesture set; the spec's pan / pinch / Cmd-+/- on
///     macOS is wired via `graphStates.modelTransform`.
///   * **Detail (right)**: the focused node's metadata
///     (label, type, subtype, links to other visible nodes).
///
/// **Performance:** the visible set is computed by the view
/// model and capped at a few hundred nodes. Grape's
/// `BufferedKDTree` keeps the many-body force under 100ms
/// for that size (per the Grape README's M1 Max benchmark:
/// 0.005s for 77 nodes / 254 edges in release).
///
/// **Privacy:** node labels come from `graph_entities.label`
/// which is the user's own data. No third-party network
/// call is made by this view. The "open in native surface"
/// action (double-click) emits a `contact_opened` / `doc_opened`
/// receipt via the data layer; that receipt is wired in the
/// per-surface view that owns the open action.
public struct GraphView: View {

    public init(viewModel: GraphViewModel) {
        self.viewModel = viewModel
    }

    @Bindable var viewModel: GraphViewModel
    @State private var graphStates = ForceDirectedGraphState()
    @State private var showFilters: Bool = true

    public var body: some View {
        NavigationSplitView {
            GraphSidebar(viewModel: viewModel, showFilters: $showFilters)
                .navigationSplitViewColumnWidth(min: 240, ideal: 280)
        } detail: {
            HStack(spacing: 0) {
                canvas
                if viewModel.focusedNode != nil {
                    GraphDetailPanel(viewModel: viewModel)
                        .frame(minWidth: 280, idealWidth: 320)
                        .background(.background)
                        .overlay(alignment: .leading) {
                            Divider()
                        }
                }
            }
        }
        .navigationTitle("Graph")
        .toolbar {
            GraphToolbar(viewModel: viewModel, graphStates: $graphStates)
        }
        .onAppear {
            if viewModel.snapshot.nodeCount == 0 && !viewModel.isLoading {
                Task { await viewModel.load() }
            }
        }
        .overlay {
            if viewModel.isLoading {
                ProgressView("Loading graph…")
                    .controlSize(.large)
                    .padding()
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
            }
        }
        .overlay(alignment: .top) {
            if let err = viewModel.loadError {
                Text(err)
                    .font(.caption)
                    .padding(8)
                    .background(.red.opacity(0.2), in: RoundedRectangle(cornerRadius: 4))
                    .padding()
            }
        }
    }

    @ViewBuilder
    private var canvas: some View {
        if viewModel.visibleNodes.isEmpty {
            emptyState
        } else {
            ForceDirectedGraph(states: graphStates) {
                Series(viewModel.visibleNodes) { node in
                    NodeMark(id: node.id)
                        .symbolSize(radius: 4.0 + 8.0 * node.importance)
                        .foregroundStyle(nodeColor(node))
                }
                Series(viewModel.visibleEdges) { edge in
                    LinkMark(from: edge.sourceID, to: edge.targetID)
                        .foregroundStyle(edgeColor(edge))
                }
            } force: {
                .manyBody(strength: -25)
                .center(strength: 0.05)
                .link(originalLength: 50.0, stiffness: .weightedByDegree { _, _ in 1.0 })
                .collide(radius: 6.0)
            }
            .background(Color(NSColor.controlBackgroundColor))
        }
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "circle.dotted")
                .font(.system(size: 48))
                .foregroundStyle(.tertiary)
            Text("No graph data yet")
                .font(.headline)
            Text("Add a contact, document, or task to populate the graph.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            if let err = viewModel.loadError {
                Text(err)
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Style helpers

    private func nodeColor(_ node: GraphNode) -> Color {
        if viewModel.selectedNodeIDs.contains(node.id) {
            return .accentColor
        }
        if viewModel.findMatches.contains(node.id) {
            return .yellow
        }
        if viewModel.anchorNodeIDs.contains(node.id) {
            return GraphNode.color(for: node.entityType).opacity(0.8)
        }
        return GraphNode.color(for: node.entityType)
    }

    /// Color for an edge. The base color comes from
    /// `GraphEdge.color`; we layer opacity on top so
    /// heavier links are more visible. Style
    /// (superseded / voided) is communicated through
    /// opacity so the eye can pick them out.
    private func edgeColor(_ edge: GraphEdge) -> Color {
        let base = edge.color
        let weightOpacity = min(1.0, 0.4 + Double(edge.weight) * 0.3)
        switch edge.style {
        case .normal:
            return base.opacity(weightOpacity)
        case .superseded:
            return .orange.opacity(weightOpacity)
        case .voided:
            return .red.opacity(weightOpacity * 0.5)
        }
    }
}

// MARK: - Sidebar

private struct GraphSidebar: View {
    @Bindable var viewModel: GraphViewModel
    @Binding var showFilters: Bool

    private let typeChips: [(String, String)] = [
        ("document", "doc.text"),
        ("task", "checkmark.square"),
        ("contact", "person.crop.circle"),
        ("email", "envelope"),
        ("reminder", "bell"),
        ("calendar_event", "calendar"),
        ("note", "note.text"),
        ("code", "chevron.left.forwardslash.chevron.right"),
    ]

    var body: some View {
        List {
            Section("Search") {
                TextField("Find (⌘F)", text: $viewModel.searchQuery)
                    .textFieldStyle(.roundedBorder)
                    .onChange(of: viewModel.searchQuery) { _, _ in
                        viewModel.recomputeVisible()
                    }
            }
            Section("Visibility") {
                Picker("Radius", selection: $viewModel.radius) {
                    ForEach(GraphViewModel.VisibilityRadius.allCases) { r in
                        Text(r.displayName).tag(r)
                    }
                }
                .pickerStyle(.menu)
                .onChange(of: viewModel.radius) { _, _ in
                    viewModel.recomputeVisible()
                }
            }
            Section("Filter by type") {
                ForEach(typeChips, id: \.0) { type, icon in
                    Button {
                        viewModel.toggleType(type)
                    } label: {
                        Label(type, systemImage: icon)
                            .foregroundStyle(
                                viewModel.typeFilter.contains(type) ? Color.accentColor : .primary
                            )
                    }
                    .buttonStyle(.plain)
                }
                Button("Clear filters") {
                    viewModel.clearFilters()
                }
                .disabled(viewModel.typeFilter.isEmpty && viewModel.searchQuery.isEmpty)
            }
            Section("Stats") {
                HStack {
                    Text("Nodes")
                    Spacer()
                    Text("\(viewModel.visibleNodes.count) / \(viewModel.snapshot.nodeCount)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                HStack {
                    Text("Edges")
                    Spacer()
                    Text("\(viewModel.visibleEdges.count) / \(viewModel.snapshot.edgeCount)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
        }
        .listStyle(.sidebar)
    }
}

// MARK: - Toolbar

private struct GraphToolbar: ToolbarContent {
    @Bindable var viewModel: GraphViewModel
    @Binding var graphStates: ForceDirectedGraphState

    var body: some ToolbarContent {
        ToolbarItemGroup(placement: .primaryAction) {
            Button {
                Task { await viewModel.load() }
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .help("Reload graph")
            Button {
                graphStates.isRunning.toggle()
            } label: {
                Image(systemName: graphStates.isRunning ? "pause.fill" : "play.fill")
            }
            .help(graphStates.isRunning ? "Pause simulation" : "Resume simulation")
        }
    }
}

// MARK: - Detail panel

private struct GraphDetailPanel: View {
    @Bindable var viewModel: GraphViewModel

    var body: some View {
        if let node = viewModel.focusedNode {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .center, spacing: 8) {
                    Image(systemName: node.iconName)
                        .font(.title2)
                        .foregroundStyle(GraphNode.color(for: node.entityType))
                    VStack(alignment: .leading) {
                        Text(node.label)
                            .font(.headline)
                            .lineLimit(2)
                        Text(node.entityType)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button {
                        viewModel.clearSelection()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
                // "Open in <native surface>" — only shown
                // when a surface wired an open handler
                // (the calendar surface does; see
                // CalendarGraphConnector).
                if viewModel.openEntityHandler != nil {
                    Button {
                        viewModel.open(node)
                    } label: {
                        Label(openLabel(for: node), systemImage: "arrow.up.forward.square")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
                Divider()
                relatedEdgesSection(node: node)
                Spacer()
            }
            .padding()
        } else {
            EmptyView()
        }
    }

    private func openLabel(for node: GraphNode) -> String {
        switch node.entityType {
        case "calendar_event", "event": return "Open in Calendar"
        case "contact": return "Open in Contacts"
        case "document", "note", "doc": return "Open in Editor"
        default: return "Open"
        }
    }

    @ViewBuilder
    private func relatedEdgesSection(node: GraphNode) -> some View {
        let related = viewModel.visibleEdges.filter {
            $0.sourceID == node.id || $0.targetID == node.id
        }
        if related.isEmpty {
            Text("No related entities in the current view.")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            Text("Related")
                .font(.subheadline)
                .fontWeight(.medium)
            ForEach(related) { edge in
                let otherID = edge.sourceID == node.id ? edge.targetID : edge.sourceID
                let other = viewModel.snapshot.nodes.first { $0.id == otherID }
                HStack {
                    Image(systemName: other?.iconName ?? "circle")
                        .foregroundStyle(GraphNode.color(for: other?.entityType ?? ""))
                    VStack(alignment: .leading) {
                        Text(other?.label ?? "Unknown")
                            .font(.caption)
                            .lineLimit(1)
                        Text(edge.linkType)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                }
            }
        }
    }
}
