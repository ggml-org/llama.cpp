import SwiftUI
import TesseraCore

/// The canvas. Phase 2.1 ships a static, read-only canvas:
/// positions are stored in a `[nodeId: CGPoint]` map owned by
/// the parent, the canvas renders one rounded rectangle per
/// declared node and one bezier curve per declared edge, and
/// that's it. Drag-to-move is Phase 2.2; port hit-testing +
/// wire-on-drag is Phase 2.3; selection + parameter side panel
/// is Phase 2.4.
///
/// The canvas reads ``workflow`` directly because the value type
/// is small and Codable; the binding back to the document (for
/// save/load) is wired by the parent `WorkflowsView` via
/// ``WorkflowPositionMap``.
///
/// The bezier-connection rendering is intentionally minimal:
/// a quadratic curve from `(sourceNode.outputPort).center` to
/// `(targetNode.inputPort).center`, with the control point
/// pulled horizontally by 40pt. The exact control-point math
/// is the same one the editor will use in Phase 2.3, so the
/// later connection-during-drag preview will not have to
/// re-tune the curvature.
struct WorkflowCanvasView: View {
    let workflow: Workflow
    let registry: WorkflowNodeRegistry
    @Binding var positions: WorkflowPositionMap

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                gridBackground
                edgeLayer
                nodeLayer
            }
            .frame(width: geo.size.width, height: geo.size.height)
            .clipped()
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var gridBackground: some View {
        Canvas { ctx, size in
            let spacing: CGFloat = 24
            let color = Color.secondary.opacity(0.08)
            var path = Path()
            var x: CGFloat = 0
            while x < size.width {
                path.move(to: CGPoint(x: x, y: 0))
                path.addLine(to: CGPoint(x: x, y: size.height))
                x += spacing
            }
            var y: CGFloat = 0
            while y < size.height {
                path.move(to: CGPoint(x: 0, y: y))
                path.addLine(to: CGPoint(x: size.width, y: y))
                y += spacing
            }
            ctx.stroke(path, with: .color(color), lineWidth: 0.5)
        }
    }

    private var nodeLayer: some View {
        ForEach(workflow.nodes, id: \.id) { node in
            if let type = registry.nodeType(for: node.type) {
                WorkflowNodeView(
                    node: node,
                    type: type,
                    position: positions[node.id] ?? defaultPosition(for: node)
                )
            }
        }
    }

    private var edgeLayer: some View {
        Canvas { ctx, _ in
            for edge in workflow.edges {
                let from = portCenter(
                    nodeId: edge.fromNode, portId: edge.fromPort, side: .right)
                let to = portCenter(
                    nodeId: edge.toNode, portId: edge.toPort, side: .left)
                guard let from, let to else { continue }
                let dx = max(40, abs(to.x - from.x) * 0.5)
                var path = Path()
                path.move(to: from)
                path.addCurve(
                    to: to,
                    control1: CGPoint(x: from.x + dx, y: from.y),
                    control2: CGPoint(x: to.x - dx, y: to.y)
                )
                ctx.stroke(path, with: .color(.secondary), lineWidth: 1.5)
            }
        }
    }

    private enum Side { case left, right }

    private func portCenter(
        nodeId: String, portId: String, side: Side
    ) -> CGPoint? {
        guard let center = positions[nodeId],
              let type = registry.nodeType(for: workflow.node(id: nodeId)?.type ?? "")
        else { return nil }
        let ports = (side == .left) ? type.inputs : type.outputs
        guard let idx = ports.firstIndex(where: { $0.id == portId }) else {
            return nil
        }
        // Node rect: 200x(40 + N*20) approximately. Ports are at
        // y = header(40) + 8 + idx*20 + 10 (half of port row).
        let portCount = ports.count
        let approxHeight: CGFloat = 40 + 8 + CGFloat(portCount) * 20 + 8
        let xOffset: CGFloat = (side == .left) ? 14 : 200 - 14
        let yOffset: CGFloat = 40 + 8 + CGFloat(idx) * 20 + 10
        return CGPoint(
            x: center.x + xOffset - 100, // -100 because position is center
            y: center.y + yOffset - approxHeight / 2
        )
    }

    private func defaultPosition(for node: WorkflowNode) -> CGPoint {
        positions[node.id] ?? CGPoint(x: 200, y: 200)
    }
}

/// Position store for the canvas. Keyed by node id. The map is
/// `@Binding`-driven so the parent `WorkflowsView` can persist
/// it (eventually) to the workflow document alongside the
/// graph itself. Phase 2.1 keeps it in-memory only; Phase 2.5
/// will fold positions into the workflow JSON.
typealias WorkflowPositionMap = [String: CGPoint]

extension Workflow {
    /// Look up a node by id. Returns nil for unknown ids (the
    /// editor treats unknown ids as stale UI state, not as a
    /// crash).
    func node(id: String) -> WorkflowNode? {
        nodes.first { $0.id == id }
    }
}
