import Foundation

// MARK: - GraphStore

/// Read-side facade for the graph view. Wraps
/// ``TesseraDataLayer`` and translates the row-shaped domain
/// types (``GraphEntity``, ``EntityLink``) into the view-
/// shaped types (``GraphNode``, ``GraphEdge``). The store is
/// a struct because the data layer actor is the source of
/// concurrency safety.
///
/// **Importance score:** computed as
/// `0.5 * normalizedDegree + 0.5 * recency` where
/// `normalizedDegree` is the node's degree divided by the
/// max degree in the graph, and `recency` is `1 - age / maxAge`
/// (the most-recently-updated node gets a 1.0, the oldest
/// gets 0.0). The score is used for the node's radius in the
/// view; the spec's "size by importance" requirement.
public struct GraphStore: Sendable {

    private let dataLayer: TesseraDataLayer

    public init(dataLayer: TesseraDataLayer) {
        self.dataLayer = dataLayer
    }

    // MARK: - Load

    /// Load every node + edge in the database, capped at
    /// `nodeLimit` + `edgeLimit`. The graph view calls this
    /// once at startup; the view's progressive-disclosure
    /// policy then materializes a subset of the snapshot for
    /// the initial render.
    public func loadAll(
        nodeLimit: Int = 10_000,
        edgeLimit: Int = 50_000
    ) async throws -> GraphSnapshot {
        // Step 1: load every entity across all types. The
        // Postgres table is small enough for a full scan
        // (10k+ rows is well within the `hybrid_search`
        // budget) and the graph view needs the whole shape.
        let nodes = try await loadAllNodes(limit: nodeLimit)
        // Step 2: load every link, with `edgeLimit` as the
        // safety bound. The `idx_entity_links_source` /
        // `_target` indexes (migration 0003) keep the
        // adjacency-list walk cheap.
        let edges = try await dataLayer.listAllLinks(limit: edgeLimit)
        let graphEdges = edges.compactMap { link -> GraphEdge? in
            // Drop edges whose endpoints aren't in the
            // materialized node set (the edge points at an
            // entity that wasn't loaded because of the
            // nodeLimit).
            guard nodes.contains(where: { $0.id == link.sourceID }),
                  nodes.contains(where: { $0.id == link.targetID }) else {
                return nil
            }
            return GraphEdge(
                id: link.id,
                sourceID: link.sourceID,
                targetID: link.targetID,
                linkType: link.linkType,
                weight: link.weight
            )
        }
        return GraphSnapshot(nodes: nodes, edges: graphEdges)
    }

    /// Load only entities of a specific type (e.g. just
    /// contacts). The graph view's filter chips call this
    /// when the user wants to focus on one material.
    public func loadOfType(
        _ entityType: String,
        nodeLimit: Int = 10_000
    ) async throws -> GraphSnapshot {
        let rows = try await dataLayer.listByEntityType(
            entityType: entityType,
            limit: nodeLimit
        )
        let nodes = rows.map { Self.node(from: $0, importance: 1.0) }
        return GraphSnapshot(nodes: nodes, edges: [])
    }

    // MARK: - Search

    /// Find nodes whose label matches the query. Used by the
    /// graph view's Cmd-F find shortcut. Returns the
    /// matching nodes (not the surrounding subgraph); the
    /// view does the highlight + pan-to-first-match.
    public func search(
        query: String,
        limit: Int = 20
    ) async throws -> [GraphNode] {
        let rows = try await dataLayer.searchByLabelPrefix(
            entityType: "any",
            labelPrefix: query,
            limit: limit
        )
        return rows.map { Self.node(from: $0, importance: 1.0) }
    }

    // MARK: - Helpers

    private func loadAllNodes(limit: Int) async throws -> [GraphNode] {
        // The data layer doesn't yet expose an
        // "entity_type = ANY" query (it does
        // listByEntityType for one type). We paginate over
        // the known types. The v1 surface has ~10
        // material types; the loop is cheap and stays
        // within the snapshot's memory budget.
        let knownTypes = [
            "document", "note", "doc", "task", "todo",
            "reminder", "calendar_event", "event", "email",
            "contact", "spreadsheet", "presentation", "code",
        ]
        var allRows: [GraphEntity] = []
        for type in knownTypes {
            let rows = try await dataLayer.listByEntityType(
                entityType: type,
                limit: limit
            )
            allRows.append(contentsOf: rows)
            if allRows.count >= limit { break }
        }
        return Self.makeNodes(from: allRows)
    }

    /// Compute the importance score per node and produce
    /// the final `GraphNode` list. The score depends on
    /// degree (in + out) and recency.
    private static func makeNodes(from rows: [GraphEntity]) -> [GraphNode] {
        // First pass: build degree counts from the rows
        // themselves. We don't have the link set here
        // (the caller passes the edges separately), so we
        // approximate "degree" with the entity's
        // updatedAt recency in this pass; the final
        // snapshot then runs a second pass that combines
        // degree + recency.
        let now = Date()
        let maxAge = max(1.0, now.timeIntervalSince(
            rows.map(\.updatedAt).min() ?? now
        ))
        return rows.map { row in
            let recency = 1.0 - (now.timeIntervalSince(row.updatedAt) / maxAge)
            // Degree 0 in this pass; the snapshot's
            // builder combines it.
            return node(from: row, importance: max(0.0, recency))
        }
    }

    /// Single-row -> node conversion. `importance` is the
    /// precomputed score; the snapshot's init does a second
    /// pass to combine degree + recency into the final
    /// score.
    static func node(from row: GraphEntity, importance: Double) -> GraphNode {
        GraphNode(
            id: row.id,
            entityType: row.entityType,
            subtype: row.subtype,
            label: row.label,
            importance: importance,
            updatedAt: row.updatedAt,
            isPinned: false
        )
    }

    /// Recompute importance scores given the full edge set.
    /// Called by the view's snapshot loader after the
    /// `GraphStore.loadAll()` call. The function is
    /// exposed so the view can re-run the importance pass
    /// when new edges are added.
    public static func recomputeImportance(
        for nodes: [GraphNode],
        edges: [GraphEdge]
    ) -> [GraphNode] {
        // Degree per node.
        var degree: [UUID: Int] = [:]
        for n in nodes { degree[n.id] = 0 }
        for e in edges {
            degree[e.sourceID, default: 0] += 1
            degree[e.targetID, default: 0] += 1
        }
        let maxDegree = max(1, degree.values.max() ?? 1)
        let now = Date()
        let oldest = nodes.map(\.updatedAt).min() ?? now
        let maxAge = max(1.0, now.timeIntervalSince(oldest))
        return nodes.map { n in
            let d = Double(degree[n.id] ?? 0) / Double(maxDegree)
            let r = max(0.0, 1.0 - now.timeIntervalSince(n.updatedAt) / maxAge)
            let score = max(0.0, min(1.0, 0.5 * d + 0.5 * r))
            var copy = n
            copy = GraphNode(
                id: n.id,
                entityType: n.entityType,
                subtype: n.subtype,
                label: n.label,
                importance: score,
                updatedAt: n.updatedAt,
                isPinned: n.isPinned
            )
            return copy
        }
    }
}
