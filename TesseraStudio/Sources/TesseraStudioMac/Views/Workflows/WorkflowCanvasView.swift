import SwiftUI
import TesseraCore

/// The canvas. Renders the workflow's nodes, edges, and any
/// in-flight wire (drag-to-wire preview). Phase 2.2 added
/// drag-to-move; Phase 2.3 added port hit-testing +
/// drag-to-wire. The hit test lives in ``WorkflowsView`` (it
/// has access to the workflow model and the connection error
/// banner); this view just renders and reports.
///
/// The bezier-connection rendering uses the same control math
/// for real edges, the drag preview, and the hit test, so
/// visual + behavioral + hit-test stay in sync.
struct WorkflowCanvasView: View {
    let workflow: Workflow
    let registry: WorkflowNodeRegistry
    @Binding var positions: WorkflowPositionMap
    @Binding var pendingConnection: PendingConnection?
    @Binding var selectedNodeId: String?
    /// Type of the output port an in-flight wire was dragged
    /// from, if any. The parent sets it while a pending
    /// connection is live so input ports can show compatibility
    /// feedback before the drop; nil renders ports normally.
    var pendingSourceType: WorkflowPortType? = nil
    /// Zoom factor and pan offset for the canvas content. The
    /// parent owns them because the palette drop handler (which
    /// lives in the parent) has to undo the transform when it
    /// converts its drop location into canvas coordinates.
    @Binding var zoom: CGFloat
    @Binding var pan: CGSize
    let onConnectionCompleted: (CGPoint, CGSize) -> Void
    /// Fired by `WorkflowNodeView` after a drag-to-move gesture
    /// completes, with `(nodeId, startPosition, endPosition)`.
    /// The parent uses this to register an undo entry.
    let onPositionDragEnded: (String, CGPoint, CGPoint) -> Void

    /// Named coordinate space declared on the (untransformed)
    /// canvas content. Port drags report their locations in
    /// this space, so wire drawing and drop hit-testing stay in
    /// canvas coordinates regardless of zoom + pan.
    static let coordinateSpaceName = "tesseraCanvas"

    /// The canvas bounds captured from the GeometryReader, so
    /// the drop callback reports the real size instead of a
    /// constant. Zero until the first layout pass; a wire can
    /// only complete after that.
    @State private var canvasSize: CGSize = .zero
    /// Anchor values for the in-flight pan / magnify gestures,
    /// so each change is applied relative to the value the
    /// gesture started at (the bindings are live state).
    @State private var panDragStart: CGSize?
    @State private var magnifyStartZoom: CGFloat?

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                gridBackground
                ZStack(alignment: .topLeading) {
                    edgeLayer
                    pendingConnectionLayer
                    nodeLayer
                }
                .frame(width: geo.size.width, height: geo.size.height)
                .coordinateSpace(name: Self.coordinateSpaceName)
                .scaleEffect(zoom, anchor: .topLeading)
                .offset(pan)
            }
            .onAppear { canvasSize = geo.size }
            .onChange(of: geo.size) { _, newSize in
                canvasSize = newSize
            }
        }
        // Tap on empty canvas clears an in-flight wire (lets
        // the user abort a drag without dropping on a port)
        // and clears selection. Attached outside the transform
        // so the whole viewport stays tappable at any zoom.
        .contentShape(Rectangle())
        .onTapGesture {
            if pendingConnection != nil {
                pendingConnection = nil
            }
            selectedNodeId = nil
        }
        // Drag on empty canvas pans the view. Node and port
        // drags are deeper in the hierarchy, so they win where
        // they exist; a drag that starts on empty space pans.
        .gesture(
            DragGesture(minimumDistance: 8)
                .onChanged { value in
                    if panDragStart == nil { panDragStart = pan }
                    if let start = panDragStart {
                        pan = CGSize(
                            width: start.width + value.translation.width,
                            height: start.height + value.translation.height
                        )
                    }
                }
                .onEnded { _ in panDragStart = nil }
        )
        // Trackpad pinch zooms. The anchor stays top-leading;
        // cursor-anchored zoom is a follow-on.
        .simultaneousGesture(
            MagnifyGesture()
                .onChanged { value in
                    if magnifyStartZoom == nil { magnifyStartZoom = zoom }
                    if let start = magnifyStartZoom {
                        zoom = WorkflowGeometry.clampedZoom(
                            start * value.magnification)
                    }
                }
                .onEnded { _ in magnifyStartZoom = nil }
        )
        .clipped()
        .overlay(alignment: .bottomTrailing) {
            zoomControls.padding(10)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    /// Zoom in / out / reset cluster, pinned to the canvas
    /// corner. The keyboard shortcuts give the same actions a
    /// keyboard path (Cmd+= / Cmd+- / Cmd+0).
    private var zoomControls: some View {
        HStack(spacing: 4) {
            Button {
                zoom = WorkflowGeometry.clampedZoom(zoom / 1.25)
            } label: {
                Image(systemName: "minus.magnifyingglass")
            }
            .keyboardShortcut("-", modifiers: .command)
            .help("Zoom out")
            Text("\(Int(round(zoom * 100)))%")
                .monospacedDigit()
                .frame(minWidth: 44)
            Button {
                zoom = WorkflowGeometry.clampedZoom(zoom * 1.25)
            } label: {
                Image(systemName: "plus.magnifyingglass")
            }
            .keyboardShortcut("=", modifiers: .command)
            .help("Zoom in")
            Button {
                zoom = 1
                pan = .zero
            } label: {
                Image(systemName: "arrow.up.left.and.down.right.magnifyingglass")
            }
            .keyboardShortcut("0", modifiers: .command)
            .help("Reset zoom and pan")
        }
        .buttonStyle(.borderless)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
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
                    position: bindingFor(nodeId: node.id, type: type),
                    isSelected: selectedNodeId == node.id,
                    pendingSourceType: pendingSourceType,
                    onSelect: { selectedNodeId = node.id },
                    onPortDragStarted: { endpoint in
                        let pos = positions[endpoint.nodeId] ?? .zero
                        let center = WorkflowGeometry.portCenter(
                            nodeCenter: pos,
                            portIndex: indexOfPort(
                                in: type.outputs, id: endpoint.portId
                            ),
                            isLeft: false,
                            portCount: type.outputs.count
                        )
                        pendingConnection = PendingConnection(
                            source: endpoint, currentPoint: center
                        )
                    },
                    onPortDragChanged: { location in
                        pendingConnection?.currentPoint = location
                    },
                    onPortDragEnded: { location in
                        guard let pending = pendingConnection else { return }
                        let size = currentCanvasSize()
                        pendingConnection = nil
                        onConnectionCompleted(
                            location,
                            pending.source.portType == .toolResult
                                ? locationOnly(location, size: size)
                                : size
                        )
                    },
                    onPositionDragEnded: { start, end in
                        onPositionDragEnded(node.id, start, end)
                    }
                )
            }
        }
    }

    /// Helper for the size we pass back on drop: the real
    /// canvas bounds captured from the GeometryReader. The
    /// drop handler doesn't actually need the size today, but
    /// the signature is forward-compatible with a future check
    /// against canvas bounds.
    private func currentCanvasSize() -> CGSize {
        canvasSize
    }

    private func locationOnly(_ point: CGPoint, size: CGSize) -> CGSize {
        size
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

    /// Draw the in-flight wire from the source port to the
    /// current drag position. Dashed so it's visually distinct
    /// from real edges.
    private var pendingConnectionLayer: some View {
        Canvas { ctx, _ in
            guard let pending = pendingConnection else { return }
            guard let from = portCenter(
                nodeId: pending.source.nodeId,
                portId: pending.source.portId,
                side: .right
            ) else { return }
            let to = pending.currentPoint
            let dx = max(40, abs(to.x - from.x) * 0.5)
            var path = Path()
            path.move(to: from)
            path.addCurve(
                to: to,
                control1: CGPoint(x: from.x + dx, y: from.y),
                control2: CGPoint(x: to.x - dx, y: to.y)
            )
            ctx.stroke(
                path,
                with: .color(.accentColor.opacity(0.6)),
                style: StrokeStyle(lineWidth: 1.5, lineCap: .round, dash: [4, 3])
            )
            // A small filled dot at the cursor end so the
            // user can see where the drop will land.
            let dot = Path(ellipseIn: CGRect(
                x: to.x - 4, y: to.y - 4, width: 8, height: 8
            ))
            ctx.fill(dot, with: .color(.accentColor))
        }
    }

    /// Build a binding that updates `positions[nodeId]` in the
    /// parent's map. Reads go through the map (so the bezier
    /// layer sees the same position); writes go through the
    /// map (so the drag actually persists). Nodes that have
    /// no entry yet start at a default position.
    private func bindingFor(
        nodeId: String, type: any WorkflowNodeType.Type
    ) -> Binding<CGPoint> {
        Binding(
            get: {
                positions[nodeId] ?? defaultPosition(
                    forId: nodeId, type: type
                )
            },
            set: { newValue in
                positions[nodeId] = newValue
            }
        )
    }

    private func portCenter(
        nodeId: String, portId: String, side: WorkflowPortView.Side
    ) -> CGPoint? {
        guard let node = workflow.node(id: nodeId),
              let type = registry.nodeType(for: node.type),
              let pos = positions[nodeId] else { return nil }
        let ports = (side == .left) ? type.inputs : type.outputs
        guard let idx = ports.firstIndex(where: { $0.id == portId }) else {
            return nil
        }
        return WorkflowGeometry.portCenter(
            nodeCenter: pos, portIndex: idx,
            isLeft: side == .left,
            portCount: ports.count
        )
    }

    private func indexOfPort(
        in ports: [WorkflowPort], id: String
    ) -> Int {
        ports.firstIndex(where: { $0.id == id }) ?? 0
    }

    private func defaultPosition(
        forId nodeId: String, type: any WorkflowNodeType.Type
    ) -> CGPoint {
        // Lay out new nodes in a grid starting at (200, 200),
        // offset by 240pt horizontally and 120pt vertically per
        // index. The first time a node shows up on the canvas
        // it lands here; subsequent drags overwrite it.
        let idx = workflow.nodes.firstIndex { $0.id == nodeId } ?? 0
        let col = idx % 3
        let row = idx / 3
        return CGPoint(
            x: 200 + CGFloat(col) * 260,
            y: 200 + CGFloat(row) * 140
        )
    }
}

extension Workflow {
    /// Look up a node by id. Returns nil for unknown ids (the
    /// editor treats unknown ids as stale UI state, not as a
    /// crash).
    func node(id: String) -> WorkflowNode? {
        nodes.first { $0.id == id }
    }
}

/// Position store for the canvas. Keyed by node id. The map is
/// `@Binding`-driven so the parent `WorkflowsView` can persist
/// it (eventually) to the workflow document alongside the
/// graph itself. Phase 2.4 will fold positions into the
/// workflow JSON.
typealias WorkflowPositionMap = [String: CGPoint]
