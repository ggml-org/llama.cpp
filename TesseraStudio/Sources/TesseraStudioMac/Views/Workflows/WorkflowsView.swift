import SwiftUI
import TesseraCore

/// The "Workflows" surface. Phase 2.1 ships the read-only
/// layout: a palette on the left, a canvas on the right, a
/// toolbar on top, and a hard-coded `calibrate -> quantize`
/// example workflow on the canvas to prove the rendering
/// primitives. Drag-to-move, port hit-testing, parameter side
/// panel, document persistence, and the run-from-toolbar
/// behaviour land in later sub-steps.
struct WorkflowsView: View {
    @State private var workflow: Workflow = WorkflowsView.exampleWorkflow
    @State private var positions: WorkflowPositionMap = WorkflowsView.examplePositions
    @State private var registry: WorkflowNodeRegistry = .default

    var body: some View {
        VStack(spacing: 0) {
            WorkflowToolbarView(
                onNew: { workflow = Workflow(name: "untitled", nodes: [], edges: []) },
                onOpen: {},
                onSave: {},
                onRun: {}
            )
            HSplitView {
                WorkflowPaletteView(registry: registry)
                WorkflowCanvasView(
                    workflow: workflow,
                    registry: registry,
                    positions: $positions
                )
            }
        }
    }

    /// The same `calibrate -> quantize` example the design doc
    /// shows in section 16.4. The positions are hand-picked so
    /// the rendered graph looks balanced (the calibrate node on
    /// the left, the quantize node on the right, the bezier
    /// reaching across the middle).
    static let exampleWorkflow = Workflow(
        name: "calibrate-and-quantize",
        nodes: [
            WorkflowNode(
                id: "calib",
                type: CalibrateNode.typeId,
                parameters: ["n_tokens": .number(8000)]
            ),
            WorkflowNode(
                id: "q",
                type: QuantizeNode.typeId,
                parameters: [:]
            ),
        ],
        edges: [
            WorkflowEdge(
                fromNode: "calib", fromPort: "result",
                toNode: "q", toPort: "policy_path"
            ),
        ]
    )

    static let examplePositions: WorkflowPositionMap = [
        "calib": CGPoint(x: 220, y: 220),
        "q":     CGPoint(x: 540, y: 220),
    ]
}
