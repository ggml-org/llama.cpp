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

    /// Whether a wire dragged from an output port of type
    /// ``source`` may land on an input port of type ``target``.
    /// Applies the same rule as the drop-time validation
    /// (``WorkflowPortType/canFlowInto``); lifted here so the
    /// live drag feedback and the drop check can't drift apart.
    public static func isWireCompatible(
        source: WorkflowPortType, target: WorkflowPortType
    ) -> Bool {
        source.canFlowInto(target)
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

    /// Smallest / largest zoom factor the canvas allows.
    public static let minZoom: CGFloat = 0.25
    public static let maxZoom: CGFloat = 4.0

    /// Clamp a zoom factor to the supported range.
    public static func clampedZoom(_ zoom: CGFloat) -> CGFloat {
        min(max(zoom, minZoom), maxZoom)
    }

    /// Convert a point reported in the canvas viewport (palette
    /// drops, overlay controls) into canvas content coordinates
    /// under the current zoom + pan. The canvas renders as
    /// `screen = canvas * zoom + pan` with a top-leading anchor.
    public static func canvasPoint(
        fromViewport point: CGPoint, zoom: CGFloat, pan: CGSize
    ) -> CGPoint {
        CGPoint(
            x: (point.x - pan.width) / zoom,
            y: (point.y - pan.height) / zoom
        )
    }

    /// Inverse of ``canvasPoint(fromViewport:zoom:pan:)``.
    public static func viewportPoint(
        fromCanvas point: CGPoint, zoom: CGFloat, pan: CGSize
    ) -> CGPoint {
        CGPoint(
            x: point.x * zoom + pan.width,
            y: point.y * zoom + pan.height
        )
    }
}
