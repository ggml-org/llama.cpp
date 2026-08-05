import XCTest
@testable import TesseraCore

/// Unit tests for the graph model layer (``GraphNode``,
/// ``GraphEdge``, ``GraphSnapshot``). These are pure-value
/// tests; they don't touch the data layer.
final class GraphModelTests: XCTestCase {

    // MARK: - GraphNode

    func testGraphNodeIdentity() {
        let id = UUID()
        let node = GraphNode(
            id: id, entityType: "contact", subtype: "person",
            label: "Ada Lovelace", importance: 0.5,
            updatedAt: Date()
        )
        XCTAssertEqual(node.id, id)
        XCTAssertEqual(node.entityType, "contact")
    }

    func testShortLabelCapsAt30() {
        let long = "This is a very long label that should be truncated to 30 characters"
        let node = GraphNode(
            id: UUID(), entityType: "document", label: long,
            importance: 0.0, updatedAt: Date()
        )
        XCTAssertEqual(node.shortLabel.count, 30)
        XCTAssertTrue(long.hasPrefix(node.shortLabel))
    }

    func testIconForKnownTypes() {
        XCTAssertEqual(
            GraphNode.iconName(for: "document"),
            "doc.text"
        )
        XCTAssertEqual(
            GraphNode.iconName(for: "contact"),
            "person.crop.circle"
        )
        XCTAssertEqual(
            GraphNode.iconName(for: "contact", subtype: "organization"),
            "building.2"
        )
        XCTAssertEqual(
            GraphNode.iconName(for: "contact", subtype: "group"),
            "person.3"
        )
        XCTAssertEqual(
            GraphNode.iconName(for: "unknown_type"),
            "circle"
        )
    }

    // MARK: - GraphEdge

    func testEdgeStyleFromLinkType() {
        XCTAssertEqual(
            GraphEdge(id: UUID(), sourceID: UUID(), targetID: UUID(), linkType: "authored", weight: 1.0).style,
            .normal
        )
        XCTAssertEqual(
            GraphEdge(id: UUID(), sourceID: UUID(), targetID: UUID(), linkType: "superseded_by", weight: 1.0).style,
            .superseded
        )
        XCTAssertEqual(
            GraphEdge(id: UUID(), sourceID: UUID(), targetID: UUID(), linkType: "voided_link", weight: 1.0).style,
            .voided
        )
    }

    func testEdgeLineWidthBounded() {
        let light = GraphEdge(id: UUID(), sourceID: UUID(), targetID: UUID(), linkType: "x", weight: 0.0)
        let heavy = GraphEdge(id: UUID(), sourceID: UUID(), targetID: UUID(), linkType: "x", weight: 100.0)
        XCTAssertEqual(light.lineWidth, 0.5, accuracy: 0.01)
        XCTAssertEqual(heavy.lineWidth, 3.0, accuracy: 0.01)
    }

    // MARK: - GraphSnapshot

    func testEmptySnapshot() {
        let s = GraphSnapshot.empty
        XCTAssertEqual(s.nodeCount, 0)
        XCTAssertEqual(s.edgeCount, 0)
    }

    func testSnapshotBuildsAdjacency() {
        let a = GraphNode(id: UUID(), entityType: "doc", label: "A", importance: 1, updatedAt: Date())
        let b = GraphNode(id: UUID(), entityType: "doc", label: "B", importance: 1, updatedAt: Date())
        let c = GraphNode(id: UUID(), entityType: "doc", label: "C", importance: 1, updatedAt: Date())
        let e1 = GraphEdge(id: UUID(), sourceID: a.id, targetID: b.id, linkType: "x", weight: 1)
        let e2 = GraphEdge(id: UUID(), sourceID: b.id, targetID: c.id, linkType: "x", weight: 1)
        let s = GraphSnapshot(nodes: [a, b, c], edges: [e1, e2])
        XCTAssertEqual(s.adjacency[a.id]?.count, 1)
        XCTAssertTrue(s.adjacency[a.id]?.contains(b.id) == true)
        XCTAssertEqual(s.adjacency[b.id]?.count, 2)
        XCTAssertTrue(s.adjacency[b.id]?.contains(a.id) == true)
        XCTAssertTrue(s.adjacency[b.id]?.contains(c.id) == true)
        XCTAssertEqual(s.adjacency[c.id]?.count, 1)
        XCTAssertTrue(s.adjacency[c.id]?.contains(b.id) == true)
    }

    func testSnapshotNeighbors() {
        let a = GraphNode(id: UUID(), entityType: "doc", label: "A", importance: 1, updatedAt: Date())
        let b = GraphNode(id: UUID(), entityType: "doc", label: "B", importance: 1, updatedAt: Date())
        let c = GraphNode(id: UUID(), entityType: "doc", label: "C", importance: 1, updatedAt: Date())
        let e1 = GraphEdge(id: UUID(), sourceID: a.id, targetID: b.id, linkType: "x", weight: 1)
        let e2 = GraphEdge(id: UUID(), sourceID: b.id, targetID: c.id, linkType: "x", weight: 1)
        let s = GraphSnapshot(nodes: [a, b, c], edges: [e1, e2])
        let oneHop = s.neighbors(of: a.id, hops: 1)
        XCTAssertTrue(oneHop.contains(a.id))
        XCTAssertTrue(oneHop.contains(b.id))
        XCTAssertFalse(oneHop.contains(c.id))
        let twoHops = s.neighbors(of: a.id, hops: 2)
        XCTAssertTrue(twoHops.contains(c.id))
    }

    // MARK: - Performance: 1000+ nodes

    func test1000NodeSnapshotBuildsFast() {
        // The spec asks for 1000+ nodes < 100ms initial
        // layout. We measure the in-memory snapshot
        // build (the layout itself is Grape's job).
        let count = 1000
        var nodes: [GraphNode] = []
        nodes.reserveCapacity(count)
        for i in 0..<count {
            nodes.append(GraphNode(
                id: UUID(), entityType: "doc",
                label: "Node \(i)",
                importance: Double(i) / Double(count),
                updatedAt: Date()
            ))
        }
        var edges: [GraphEdge] = []
        edges.reserveCapacity(count * 2)
        for i in 0..<(count - 1) {
            edges.append(GraphEdge(
                id: UUID(),
                sourceID: nodes[i].id, targetID: nodes[i + 1].id,
                linkType: "next", weight: 1.0
            ))
        }
        let start = Date()
        let s = GraphSnapshot(nodes: nodes, edges: edges)
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertEqual(s.nodeCount, count)
        XCTAssertEqual(s.edgeCount, count - 1)
        XCTAssertLessThan(elapsed, 0.100, "1000-node snapshot took \(elapsed)s")
    }

    func test5000NodeSnapshotBuildsFast() {
        // The spec's progressive-disclosure target:
        // 5000-node layout < 1s. We measure the snapshot
        // build (the layout is Grape's, separately
        // benchmarked).
        let count = 5000
        var nodes: [GraphNode] = []
        nodes.reserveCapacity(count)
        for i in 0..<count {
            nodes.append(GraphNode(
                id: UUID(), entityType: "doc",
                label: "N\(i)",
                importance: 0.5,
                updatedAt: Date()
            ))
        }
        var edges: [GraphEdge] = []
        edges.reserveCapacity(count * 4)
        for i in 0..<(count - 1) {
            for j in 0..<4 {
                let target = (i + j + 1) % count
                edges.append(GraphEdge(
                    id: UUID(),
                    sourceID: nodes[i].id, targetID: nodes[target].id,
                    linkType: "edge", weight: 1.0
                ))
            }
        }
        let start = Date()
        let s = GraphSnapshot(nodes: nodes, edges: edges)
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertEqual(s.nodeCount, count)
        XCTAssertLessThan(elapsed, 1.0, "5000-node snapshot took \(elapsed)s")
    }
}
