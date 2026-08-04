import Foundation
import CoreGraphics

/// Geometry helpers for the workflow editor. The math is
/// shared between the canvas (renders bezier connections +
/// hit-test bounds) and the drop test (finds the target port
/// at a cursor position). Keeping the math here means the
/// renderer and the validator can't drift apart.
///
/// All positions are in canvas-local coordinates (origin at
/// the top-left of the canvas; +x right, +y down). A node's
/// ``position`` is the centre of its rounded rect; port
/// positions are computed relative to that centre.
public enum WorkflowGeometry {
    /// The visual width of a node. The editor renders nodes
    /// at this width; the math assumes it.
    public static let nodeWidth: CGFloat = 200

    /// Approximate height of a node given its port count.
    /// Used only for hit-testing - the rendered height tracks
    /// the real SwiftUI layout, but the hit-test only needs
    /// the port row positions which depend on this estimate.
    public static func nodeHeight(portCount: Int) -> CGFloat {
        40 + 8 + CGFloat(portCount) * 20 + 8
    }

    /// Centre of a port dot, in canvas coordinates. `isLeft`
    /// is true for input ports (rendered on the left edge) and
    /// false for output ports (rendered on the right). The
    /// math is the same as the SwiftUI renderer's, mirrored.
    public static func portCenter(
        nodeCenter: CGPoint, portIndex: Int, isLeft: Bool, portCount: Int
    ) -> CGPoint {
        let xOffset: CGFloat = isLeft ? 14 : nodeWidth - 14
        let yOffset: CGFloat = 40 + 8 + CGFloat(portIndex) * 20 + 10
        return CGPoint(
            x: nodeCenter.x + xOffset - nodeWidth / 2,
            y: nodeCenter.y + yOffset - nodeHeight(portCount: portCount) / 2
        )
    }
}
