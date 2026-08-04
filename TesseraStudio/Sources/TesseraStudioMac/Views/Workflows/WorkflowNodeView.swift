import SwiftUI
import TesseraCore

/// One node on the workflow canvas. Renders a rounded rectangle
/// with the node's display name, its input ports on the left
/// and its output ports on the right. Phase 2.2 added
/// drag-to-move (the header bar is the grab handle); Phase
/// 2.3 added port hit-testing (the port dot is the grab
/// handle for drag-to-wire).
///
/// The view binds to a ``CGPoint`` via the parent canvas
/// (positions are stored centrally in ``WorkflowPositionMap``,
/// not in the view). The port-drag callbacks
/// (``onPortDragStarted`` / ``onPortDragChanged`` /
/// ``onPortDragEnded``) are wired by the parent
/// `WorkflowsView` so the canvas can build the in-flight
/// connection and run the drop test.
struct WorkflowNodeView: View {
    let node: WorkflowNode
    let type: any WorkflowNodeType.Type
    @Binding var position: CGPoint
    let isSelected: Bool
    let onSelect: () -> Void
    let onPortDragStarted: (PendingPortEndpoint) -> Void
    let onPortDragChanged: (CGPoint) -> Void
    let onPortDragEnded: (CGPoint) -> Void

    @State private var dragStart: CGPoint?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            HStack(alignment: .top, spacing: 0) {
                portColumn(ports: type.inputs, side: .left)
                Spacer(minLength: 12)
                portColumn(ports: type.outputs, side: .right)
            }
            .padding(.vertical, 8)
            .padding(.horizontal, 4)
        }
        .frame(width: 200)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(
                    isSelected ? Color.accentColor : Color.secondary.opacity(0.4),
                    lineWidth: isSelected ? 2 : 1
                )
        )
        .shadow(color: .black.opacity(0.08), radius: 3, x: 0, y: 1)
        .position(position)
        .onTapGesture(count: 1) { onSelect() }
        .gesture(
            DragGesture(minimumDistance: 1, coordinateSpace: .local)
                .onChanged { value in
                    if dragStart == nil {
                        dragStart = position
                    }
                    if let start = dragStart {
                        position = CGPoint(
                            x: start.x + value.translation.width,
                            y: start.y + value.translation.height
                        )
                    }
                }
                .onEnded { _ in
                    dragStart = nil
                }
        )
    }

    private var header: some View {
        HStack(spacing: 6) {
            Image(systemName: "square.dashed")
                .foregroundStyle(.secondary)
            Text(type.displayName)
                .font(.system(.subheadline, design: .rounded).weight(.semibold))
                .lineLimit(1)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .contentShape(Rectangle())
    }

    private func portColumn(
        ports: [WorkflowPort], side: WorkflowPortView.Side
    ) -> some View {
        VStack(alignment: side == .left ? .leading : .trailing, spacing: 4) {
            ForEach(ports, id: \.id) { port in
                WorkflowPortView(
                    port: port,
                    side: side,
                    onDragStarted: { _ in
                        onPortDragStarted(PendingPortEndpoint(
                            nodeId: node.id, portId: port.id, portType: port.type
                        ))
                    },
                    onDragChanged: { location in
                        onPortDragChanged(location)
                    },
                    onDragEnded: { location in
                        onPortDragEnded(location)
                    }
                )
            }
            if ports.isEmpty {
                Color.clear.frame(height: 1)
            }
        }
    }
}

/// A single port on a node. Renders a small dot + the port
/// label. The dot is the grab handle for drag-to-wire: a
/// `DragGesture` on the dot emits the three callbacks so the
/// canvas can build / update / complete the in-flight
/// connection. The dot is also a `contentShape` of `Circle`
/// so the hit area is round, not square.
struct WorkflowPortView: View {
    enum Side { case left, right }

    let port: WorkflowPort
    let side: Side
    let onDragStarted: (CGPoint) -> Void
    let onDragChanged: (CGPoint) -> Void
    let onDragEnded: (CGPoint) -> Void

    var body: some View {
        HStack(spacing: 6) {
            if side == .right {
                Spacer(minLength: 0)
                Text(port.label)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
                .overlay(Circle().stroke(Color.primary.opacity(0.2), lineWidth: 0.5))
                .contentShape(Circle())
                .gesture(
                    DragGesture(minimumDistance: 0, coordinateSpace: .global)
                        .onChanged { value in
                            onDragStarted(value.startLocation)
                            onDragChanged(value.location)
                        }
                        .onEnded { value in
                            onDragEnded(value.location)
                        }
                )
            if side == .left {
                Text(port.label)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer(minLength: 0)
            }
        }
    }

    private var color: Color {
        switch port.type {
        case .string, .number, .boolean: return .gray
        case .path: return .orange
        case .gguf: return .blue
        case .json: return .purple
        case .toolResult: return .green
        case .bag: return .pink
        }
    }
}
