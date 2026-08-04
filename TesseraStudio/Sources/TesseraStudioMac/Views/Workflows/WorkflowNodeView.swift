import SwiftUI
import TesseraCore

/// One node on the workflow canvas. Renders a rounded rectangle
/// with the node's display name, its input ports on the left and
/// its output ports on the right. Phase 2.1 ships a read-only
/// view (no drag, no port hit-testing); the drag/wire/edit
/// behaviour lives in the later Phase 2 sub-steps.
///
/// The view binds to a ``WorkflowNodePosition`` via the parent
/// canvas (positions are stored centrally, not in the view, so
/// the canvas can hand them to the bezier-connection renderer).
struct WorkflowNodeView: View {
    let node: WorkflowNode
    let type: any WorkflowNodeType.Type
    let position: CGPoint

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
                .stroke(Color.secondary.opacity(0.4), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.08), radius: 3, x: 0, y: 1)
        .position(position)
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
    }

    private func portColumn(
        ports: [WorkflowPort],
        side: WorkflowPortView.Side
    ) -> some View {
        VStack(alignment: side == .left ? .leading : .trailing, spacing: 4) {
            ForEach(ports, id: \.id) { port in
                WorkflowPortView(port: port, side: side)
            }
            if ports.isEmpty {
                Color.clear.frame(height: 1)
            }
        }
    }
}

/// A single port on a node. Renders a small dot + the port label;
/// hit-testing (Phase 2.3) will be added to the same view without
/// changing its public shape.
struct WorkflowPortView: View {
    enum Side { case left, right }

    let port: WorkflowPort
    let side: Side

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
