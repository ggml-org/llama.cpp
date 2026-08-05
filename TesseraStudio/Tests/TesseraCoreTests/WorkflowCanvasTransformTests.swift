import XCTest
import CoreGraphics
@testable import TesseraCore

/// Viewport <-> canvas coordinate math for the zoom/pan
/// transform. The canvas renders as `screen = canvas * zoom +
/// pan` (top-leading anchor); drops reported in viewport space
/// must round-trip through these helpers to land where the
/// cursor was, at any zoom.
final class WorkflowCanvasTransformTests: XCTestCase {
    func testCanvasPointUndoesZoomAndPan() {
        let viewport = CGPoint(x: 110, y: 120)
        let canvas = WorkflowGeometry.canvasPoint(
            fromViewport: viewport, zoom: 2, pan: CGSize(width: 10, height: 20))
        XCTAssertEqual(canvas.x, 50, accuracy: 0.001)
        XCTAssertEqual(canvas.y, 50, accuracy: 0.001)
    }

    func testIdentityAtZoomOneNoPan() {
        let p = CGPoint(x: 321.5, y: 654.25)
        let canvas = WorkflowGeometry.canvasPoint(
            fromViewport: p, zoom: 1, pan: .zero)
        XCTAssertEqual(canvas.x, p.x, accuracy: 0.001)
        XCTAssertEqual(canvas.y, p.y, accuracy: 0.001)
    }

    func testRoundTripThroughBothDirections() {
        let original = CGPoint(x: 480, y: 300)
        let zoom: CGFloat = 0.75
        let pan = CGSize(width: -120, height: 45)
        let screen = WorkflowGeometry.viewportPoint(
            fromCanvas: original, zoom: zoom, pan: pan)
        let back = WorkflowGeometry.canvasPoint(
            fromViewport: screen, zoom: zoom, pan: pan)
        XCTAssertEqual(back.x, original.x, accuracy: 0.001)
        XCTAssertEqual(back.y, original.y, accuracy: 0.001)
    }

    func testClampedZoomStaysInsideBounds() {
        XCTAssertEqual(WorkflowGeometry.clampedZoom(0.01), WorkflowGeometry.minZoom)
        XCTAssertEqual(WorkflowGeometry.clampedZoom(99), WorkflowGeometry.maxZoom)
        XCTAssertEqual(WorkflowGeometry.clampedZoom(1.5), 1.5)
    }
}
