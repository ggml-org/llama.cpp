import SwiftUI
import AppKit
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
    /// Type of the port an in-flight wire was dragged from, if
    /// any. Input ports use it to show live compatibility
    /// feedback (accent ring vs dim + slash) while the drag is
    /// active; nil renders ports normally.
    var pendingSourceType: WorkflowPortType? = nil
    let onSelect: () -> Void
    let onPortDragStarted: (PendingPortEndpoint) -> Void
    let onPortDragChanged: (CGPoint) -> Void
    let onPortDragEnded: (CGPoint) -> Void
    /// Fired on drag-end with the original and final positions
    /// so the parent can register an undo entry. Optional; when
    /// nil, the view still drags normally but no undo is
    /// registered. Equal start and end positions don't fire.
    var onPositionDragEnded: ((CGPoint, CGPoint) -> Void)? = nil

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
                    let start = dragStart ?? position
                    dragStart = nil
                    if start != position {
                        onPositionDragEnded?(start, position)
                    }
                }
        )
        // Grab cursor over the node body: the whole node is
        // the drag handle. pointerStyle(.grabIdle) needs
        // macOS 15 and the package targets macOS 14, so use
        // the push-based NSCursor stack instead.
        .onHover { hovering in
            if hovering {
                NSCursor.openHand.push()
            } else {
                NSCursor.pop()
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(type.displayName) node")
        .accessibilityHint("Drag to move. Use the action menu to delete.")
        .accessibilityAddTraits(.isButton)
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
                    pendingSourceType: side == .left ? pendingSourceType : nil,
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

    /// Live feedback state while a wire drag is in flight.
    /// `.none` outside a drag (and on output ports); input
    /// ports compare their type against the dragged source.
    private enum DragFeedback { case none, compatible, incompatible }

    let port: WorkflowPort
    let side: Side
    /// Type of the port an in-flight wire was dragged from.
    /// Set on input ports only; drives the live compatibility
    /// feedback while the drag is active.
    var pendingSourceType: WorkflowPortType? = nil
    let onDragStarted: (CGPoint) -> Void
    let onDragChanged: (CGPoint) -> Void
    let onDragEnded: (CGPoint) -> Void

    private var dragFeedback: DragFeedback {
        guard side == .left, let pendingSourceType else { return .none }
        return WorkflowGeometry.isWireCompatible(
            source: pendingSourceType, target: port.type
        ) ? .compatible : .incompatible
    }

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
                .overlay(
                    Circle().stroke(
                        dragFeedback == .compatible
                            ? Color.accentColor
                            : Color.primary.opacity(0.2),
                        lineWidth: dragFeedback == .compatible ? 1.5 : 0.5
                    )
                )
                .opacity(dragFeedback == .incompatible ? 0.35 : 1.0)
                .overlay {
                    if dragFeedback == .incompatible {
                        Image(systemName: "circle.slash")
                            .font(.system(size: 10))
                            .foregroundStyle(.secondary)
                            .accessibilityHidden(true)
                    }
                }
                .contentShape(Circle())
                .gesture(
                    // Canvas space, not global: the wire preview
                    // and the drop hit-test both work in canvas
                    // coordinates, and the named space is declared
                    // inside the zoom/pan transform so these values
                    // stay correct at any zoom.
                    DragGesture(
                        minimumDistance: 0,
                        coordinateSpace: .named(WorkflowCanvasView.coordinateSpaceName)
                    )
                        .onChanged { value in
                            onDragStarted(value.startLocation)
                            onDragChanged(value.location)
                        }
                        .onEnded { value in
                            onDragEnded(value.location)
                        }
                )
                .accessibilityElement()
                // The dot's color encodes the port type; the
                // label spells the type out so the encoding is
                // not color-only. The drag state is spelled
                // out too, so the ring / slash feedback is not
                // visual-only.
                .accessibilityLabel("\(port.label), \(portTypeName) \(side == .left ? "input" : "output") port\(accessibilityDragState)")
                .accessibilityHint(side == .left
                    ? "Drag to this port to wire an output to it"
                    : "Drag from this port to an input port to wire them")
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

    /// Compatibility state during an in-flight wire drag, as a
    /// label suffix. Empty outside a drag.
    private var accessibilityDragState: String {
        switch dragFeedback {
        case .none: return ""
        case .compatible: return ", compatible drop target"
        case .incompatible: return ", incompatible drop target"
        }
    }

    /// Human-readable port type for the accessibility label, so
    /// the type encoding is not carried by the dot color alone.
    private var portTypeName: String {
        switch port.type {
        case .string: return "string"
        case .number: return "number"
        case .boolean: return "boolean"
        case .path: return "path"
        case .gguf: return "GGUF"
        case .json: return "JSON"
        case .toolResult: return "tool result"
        case .bag: return "bag"
        }
    }
}
