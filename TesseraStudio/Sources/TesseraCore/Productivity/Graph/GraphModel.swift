import Foundation
import SwiftUI

// MARK: - GraphNode

/// One node in the graph view. Wraps a `graph_entity` row plus
/// the view-layer state (display string, color, layout
/// position). The `entityType` field drives the node's color
/// in the view; the `size` field is computed from
/// degree-centrality (the number of incoming + outgoing
/// edges) and recency.
public struct GraphNode: Identifiable, Sendable, Equatable, Hashable {

    /// The entity id (matches `graph_entities.id`).
    public let id: UUID
    /// The entity type ("document", "task", "contact", ...).
    public let entityType: String
    /// The entity subtype, if any ("person" / "organization" /
    /// "group" for contacts).
    public let subtype: String?
    /// The display label, capped at 30 characters in the
    /// view to keep the canvas uncluttered.
    public let label: String
    /// Cached short label (first 30 chars of `label`).
    public let shortLabel: String
    /// SF Symbol name for the entity type (e.g. `"doc.text"`
    /// for documents, `"person.crop.circle"` for contacts).
    public let iconName: String
    /// Importance score in [0, 1]. Used for the node's
    /// radius. Computed at load time from degree centrality
    /// and recency.
    public let importance: Double
    /// Most-recent access timestamp (for the "recent" sort).
    public let updatedAt: Date
    /// True when the user has pinned this node. Pinned
    /// nodes are always in the initial view regardless of
    /// the visibility slider.
    public let isPinned: Bool

    public init(
        id: UUID,
        entityType: String,
        subtype: String? = nil,
        label: String,
        importance: Double,
        updatedAt: Date,
        isPinned: Bool = false
    ) {
        self.id = id
        self.entityType = entityType
        self.subtype = subtype
        self.label = label
        self.shortLabel = String(label.prefix(30))
        self.iconName = Self.iconName(for: entityType, subtype: subtype)
        self.importance = importance
        self.updatedAt = updatedAt
        self.isPinned = isPinned
    }

    /// Map an entity type to an SF Symbol. The mapping is
    /// exhaustive enough for the materials the v1 surface
    /// knows about; unknown types get the generic "circle"
    /// placeholder.
    public static func iconName(for entityType: String, subtype: String? = nil) -> String {
        switch entityType {
        case "document", "note", "doc": return "doc.text"
        case "task", "todo": return "checkmark.square"
        case "reminder": return "bell"
        case "calendar_event", "event": return "calendar"
        case "email": return "envelope"
        case "contact":
            switch subtype {
            case "organization": return "building.2"
            case "group": return "person.3"
            default: return "person.crop.circle"
            }
        case "spreadsheet": return "tablecells"
        case "presentation": return "rectangle.on.rectangle"
        case "code": return "chevron.left.forwardslash.chevron.right"
        default: return "circle"
        }
    }

    /// Map an entity type to a base color (in RGB) for the
    /// graph view. The view uses this to set the node's
    /// foreground style.
    public static func color(for entityType: String) -> Color {
        switch entityType {
        case "document", "note", "doc": return .blue
        case "task", "todo": return .green
        case "reminder": return .yellow
        case "calendar_event", "event": return .purple
        case "email": return .pink
        case "contact": return .orange
        case "spreadsheet": return .teal
        case "presentation": return .indigo
        case "code": return .gray
        default: return .secondary
        }
    }
}

// MARK: - GraphEdge

/// One edge in the graph view. Mirrors an `entity_links` row
/// plus the view-layer state (style, thickness). The
/// `linkStyle` field drives the line drawing (solid for
/// normal links, dashed for superseded, dotted for voided).
public struct GraphEdge: Identifiable, Sendable, Equatable, Hashable {

    public let id: UUID
    public let sourceID: UUID
    public let targetID: UUID
    public let linkType: String
    public let weight: Float

    public init(
        id: UUID,
        sourceID: UUID,
        targetID: UUID,
        linkType: String,
        weight: Float
    ) {
        self.id = id
        self.sourceID = sourceID
        self.targetID = targetID
        self.linkType = linkType
        self.weight = weight
    }

    public enum Style: String, Sendable, Equatable, Hashable, CaseIterable {
        case normal
        case superseded
        case voided

        public static func from(linkType: String) -> Style {
            if linkType.hasPrefix("superseded_") { return .superseded }
            if linkType.hasPrefix("voided_") { return .voided }
            return .normal
        }
    }

    public var style: Style { Style.from(linkType: linkType) }

    /// Visual line width in points. The view maps this to a
    /// SwiftUI stroke width; the min / max bounds keep the
    /// graph readable when the user has edges with very
    /// different weights.
    public var lineWidth: Double {
        let base = 0.5 + Double(weight) * 1.5
        return min(max(base, 0.5), 3.0)
    }

    /// Color for the edge. The view layers opacity on top of
    /// the linkType color so heavier links are more visible.
    public var color: Color {
        switch linkType {
        case "authored", "owns", "created": return .blue
        case "mentioned_in", "references": return .purple
        case "assigned_to", "responsible": return .green
        case "attendee_of", "invitee": return .pink
        case "part_of", "member_of": return .orange
        case "related_to", "links_to": return .gray
        default: return .secondary
        }
    }
}

// MARK: - GraphSnapshot

/// A point-in-time snapshot of the graph: the set of nodes
/// the view should render, the set of edges among them, and
/// the adjacency lists the layout algorithm needs. The
/// snapshot is value-typed so the SwiftUI view can recompute
/// derived state without re-querying the data layer.
public struct GraphSnapshot: Sendable, Equatable {

    public let nodes: [GraphNode]
    public let edges: [GraphEdge]
    /// `adjacency[nodeID]` is the list of neighbor ids
    /// (incoming + outgoing). Used by the layout's link
    /// force and by the hop-radius expansion in the view
    /// model.
    public let adjacency: [UUID: Set<UUID>]

    public init(nodes: [GraphNode], edges: [GraphEdge]) {
        self.nodes = nodes
        self.edges = edges
        var adj: [UUID: Set<UUID>] = [:]
        for n in nodes { adj[n.id] = [] }
        for e in edges {
            if adj[e.sourceID] != nil { adj[e.sourceID, default: []].insert(e.targetID) }
            if adj[e.targetID] != nil { adj[e.targetID, default: []].insert(e.sourceID) }
        }
        self.adjacency = adj
    }

    public static let empty = GraphSnapshot(nodes: [], edges: [])

    public var nodeCount: Int { nodes.count }
    public var edgeCount: Int { edges.count }

    public func neighbors(of nodeID: UUID, hops: Int = 1) -> Set<UUID> {
        var visited: Set<UUID> = [nodeID]
        var frontier: Set<UUID> = [nodeID]
        for _ in 0..<hops {
            var next: Set<UUID> = []
            for id in frontier {
                for n in adjacency[id] ?? [] where !visited.contains(n) {
                    next.insert(n)
                    visited.insert(n)
                }
            }
            frontier = next
            if frontier.isEmpty { break }
        }
        return visited
    }
}
