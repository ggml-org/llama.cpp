import Foundation
import SwiftUI
import Observation

// MARK: - GraphViewModel

/// Drives the graph view. The view binds to this @Observable
/// type; the model's mutations trigger SwiftUI invalidation.
///
/// **Progressive disclosure.** The full graph can have
/// thousands of nodes; the view renders at most 50 in the
/// initial state (pinned + recently-accessed). The "visible
/// radius" slider expands the window to 1 / 2 / 3 hops, or
/// "all". The view model holds the full snapshot in memory
/// (one snapshot per session is small) and computes the
/// visible node set on each radius change.
///
/// **Selection.** The view model owns the selection set
/// (one node for "click to select", multiple for
/// "drag-to-select"). The view reads `selectedNodes` and
/// drives the detail panel from it.
@Observable
@MainActor
public final class GraphViewModel {

    // MARK: - State

    public enum VisibilityRadius: Int, Sendable, CaseIterable, Identifiable {
        case initial = 0
        case oneHop = 1
        case twoHops = 2
        case threeHops = 3
        case all = -1

        public var id: Int { rawValue }
        public var displayName: String {
            switch self {
            case .initial: return "Pinned + recent"
            case .oneHop: return "1 hop"
            case .twoHops: return "2 hops"
            case .threeHops: return "3 hops"
            case .all: return "All"
            }
        }
    }

    /// The full snapshot loaded at startup. The visible set
    /// is derived from this.
    public private(set) var snapshot: GraphSnapshot = .empty

    /// The subset of `snapshot` the view should render. The
    /// view model recomputes this when the radius or the
    /// filter changes.
    public private(set) var visibleNodes: [GraphNode] = []
    public private(set) var visibleEdges: [GraphEdge] = []

    /// The current filter: a list of entity types the user
    /// wants to see. Empty means "all types".
    public var typeFilter: Set<String> = []
    /// The current search query. The view highlights nodes
    /// whose label contains the query (case-insensitive).
    public var searchQuery: String = ""
    /// The current visibility radius.
    public var radius: VisibilityRadius = .initial
    /// The current selection. The view reads this to drive
    /// the detail panel.
    public var selectedNodeIDs: Set<UUID> = []

    /// Set to a non-nil error string when the loader fails.
    /// The view shows it in a banner.
    public var loadError: String?

    /// True while the initial load is in flight.
    public var isLoading: Bool = false

    /// The "anchor" set for the initial view: nodes that
    /// are pinned OR in the top-N most-recently-updated.
    public private(set) var anchorNodeIDs: Set<UUID> = []

    /// The anchor set the user has selected. The view
    /// draws a halo around anchor nodes.
    public var anchorSet: Set<UUID> = []

    /// Called when the user opens a node in its native
    /// surface (the detail panel's Open button; the
    /// spec's "double-click to open" gesture routes here
    /// too once canvas hit-testing lands). Materials
    /// surfaces wire this at construction time — the
    /// calendar surface sets it via
    /// ``CalendarGraphConnector``; nil leaves open a
    /// no-op.
    public var openEntityHandler: (@MainActor (GraphNode) -> Void)?

    private let store: GraphStore
    private let initialNodeCount: Int

    /// `initialNodeCount` is the number of nodes the
    /// "Pinned + recent" view shows by default. The spec
    /// says 50.
    public init(
        store: GraphStore,
        initialNodeCount: Int = 50
    ) {
        self.store = store
        self.initialNodeCount = initialNodeCount
    }

    // MARK: - Loading

    /// Load the full graph from the data layer. Runs on the
    /// main actor; the data layer's actor serializes the
    /// actual query. The view shows a spinner via
    /// `isLoading`.
    public func load() async {
        isLoading = true
        loadError = nil
        defer { isLoading = false }
        do {
            let raw = try await store.loadAll()
            // Second pass: combine degree + recency into
            // the final importance score.
            let rescored = GraphStore.recomputeImportance(
                for: raw.nodes,
                edges: raw.edges
            )
            let nodesByID = Dictionary(uniqueKeysWithValues: rescored.map { ($0.id, $0) })
            let nodes = Array(nodesByID.values)
            snapshot = GraphSnapshot(nodes: nodes, edges: raw.edges)
            recomputeVisible()
        } catch {
            loadError = String(describing: error)
        }
    }

    // MARK: - Visible-set recomputation

    /// Recompute `visibleNodes` and `visibleEdges` from the
    /// current `radius` / `typeFilter` / `searchQuery`.
    /// Called on any of those changes AND after `load()`.
    public func recomputeVisible() {
        // Step 1: apply the type filter.
        let typeFiltered: [GraphNode] = typeFilter.isEmpty
            ? snapshot.nodes
            : snapshot.nodes.filter { typeFilter.contains($0.entityType) }

        // Step 2: apply the search query (label contains,
        // case-insensitive). An empty query means no
        // search filter.
        let searchFiltered: [GraphNode] = searchQuery.isEmpty
            ? typeFiltered
            : typeFiltered.filter {
                $0.label.localizedCaseInsensitiveContains(searchQuery)
            }

        // Step 3: apply the visibility radius.
        let byRadius: [GraphNode]
        switch radius {
        case .initial:
            byRadius = initialSlice(from: searchFiltered)
        case .oneHop, .twoHops, .threeHops:
            byRadius = hopSlice(
                from: searchFiltered,
                hops: radius.rawValue
            )
        case .all:
            byRadius = searchFiltered
        }
        visibleNodes = byRadius

        // Step 4: filter the edges to those whose endpoints
        // are both in the visible node set.
        let visibleIDs = Set(byRadius.map(\.id))
        visibleEdges = snapshot.edges.filter {
            visibleIDs.contains($0.sourceID) && visibleIDs.contains($0.targetID)
        }

        // Step 5: update the anchor set.
        anchorNodeIDs = Set(initialSlice(from: searchFiltered).map(\.id))
    }

    /// The "pinned + recent" initial slice: top
    /// `initialNodeCount` nodes by importance (which
    /// already weights recency 0.5 + degree 0.5).
    private func initialSlice(from nodes: [GraphNode]) -> [GraphNode] {
        let pinned = nodes.filter(\.isPinned)
        let rest = nodes.filter { !$0.isPinned }
        let sorted = rest.sorted { $0.importance > $1.importance }
        let budget = max(0, initialNodeCount - pinned.count)
        return pinned + Array(sorted.prefix(budget))
    }

    /// The "1 / 2 / 3 hops" slice: union of every anchor
    /// node's k-hop neighborhood. Anchors are the initial
    /// slice. The result is a set of node IDs; the view
    /// re-materializes the visible node list from that.
    private func hopSlice(
        from nodes: [GraphNode],
        hops: Int
    ) -> [GraphNode] {
        let nodesByID = Dictionary(uniqueKeysWithValues: nodes.map { ($0.id, $0) })
        let anchors = initialSlice(from: nodes)
        var visited: Set<UUID> = []
        for anchor in anchors {
            visited.formUnion(snapshot.neighbors(of: anchor.id, hops: hops))
        }
        return nodes.filter { visited.contains($0.id) }
    }

    // MARK: - Selection

    /// Select one node (clears the prior selection).
    public func select(_ node: GraphNode, additive: Bool = false) {
        if additive {
            selectedNodeIDs.insert(node.id)
        } else {
            selectedNodeIDs = [node.id]
        }
    }

    /// Clear the selection.
    public func clearSelection() {
        selectedNodeIDs.removeAll()
    }

    /// Open a node in its native surface via
    /// ``openEntityHandler``. No-op when no handler is
    /// wired (the graph is browsable without any surface
    /// attached).
    public func open(_ node: GraphNode) {
        openEntityHandler?(node)
    }

    /// The single selected node, or nil when zero or many
    /// are selected. The view uses this for the detail
    /// panel header.
    public var focusedNode: GraphNode? {
        guard selectedNodeIDs.count == 1,
              let id = selectedNodeIDs.first else { return nil }
        return snapshot.nodes.first { $0.id == id }
    }

    // MARK: - Filter

    /// Toggle a type in the type filter. Empty set means
    /// "all types". The view binds the chip UI to this.
    public func toggleType(_ entityType: String) {
        if typeFilter.contains(entityType) {
            typeFilter.remove(entityType)
        } else {
            typeFilter.insert(entityType)
        }
        recomputeVisible()
    }

    /// Reset every filter. The "clear" toolbar button.
    public func clearFilters() {
        typeFilter.removeAll()
        searchQuery = ""
        recomputeVisible()
    }

    // MARK: - Find (Cmd-F)

    /// The set of node ids whose label matches the current
    /// search query. The view uses this for the find
    /// highlight + the "pan to first match" affordance.
    public var findMatches: Set<UUID> {
        guard !searchQuery.isEmpty else { return [] }
        return Set(
            snapshot.nodes
                .filter { $0.label.localizedCaseInsensitiveContains(searchQuery) }
                .map(\.id)
        )
    }

    /// The first match of the current search, in label
    /// order. The view pans to this node.
    public var firstFindMatch: GraphNode? {
        guard !searchQuery.isEmpty else { return nil }
        return snapshot.nodes
            .filter { $0.label.localizedCaseInsensitiveContains(searchQuery) }
            .sorted { $0.label < $1.label }
            .first
    }
}
