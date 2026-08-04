import SwiftUI
import UniformTypeIdentifiers
import TesseraCore

/// The "Workflows" surface. Holds the in-flight workflow +
/// positions + selection, routes toolbar / palette / canvas /
/// port gestures, and uses `WorkflowDocument` for the Open /
/// Save round-trip.
struct WorkflowsView: View {
    @State private var workflow: Workflow = WorkflowsView.exampleWorkflow
    @State private var positions: WorkflowPositionMap = WorkflowsView.examplePositions
    @State private var registry: WorkflowNodeRegistry = .default
    @State private var pendingConnection: PendingConnection?
    @State private var connectionError: String?
    @State private var selectedNodeId: String?
    @State private var document: WorkflowDocument = WorkflowDocument(
        workflow: WorkflowsView.exampleWorkflow,
        positions: WorkflowsView.examplePositions
    )
    @State private var documentName: String = "calibrate-and-quantize.tessera-workflow"
    @State private var isExporting = false
    @State private var isImporting = false
    @State private var isRunning = false
    @State private var runEvents: [WorkflowEvent] = []
    @State private var showRunProgress = false

    @Environment(\.undoManager) private var undoManager

    var body: some View {
        VStack(spacing: 0) {
            WorkflowToolbarView(
                onNew: newDocument,
                onOpen: { isImporting = true },
                onSave: { isExporting = true },
                onRun: { runWorkflow() },
                isRunning: isRunning
            )
            HSplitView {
                WorkflowPaletteView(registry: registry)
                ZStack {
                    WorkflowCanvasView(
                        workflow: workflow,
                        registry: registry,
                        positions: $positions,
                        pendingConnection: $pendingConnection,
                        selectedNodeId: $selectedNodeId,
                        onConnectionCompleted: { dropPoint, canvasSize in
                            completeConnection(at: dropPoint, in: canvasSize)
                        },
                        onPositionDragEnded: { nodeId, start, end in
                            recordNodeMove(nodeId: nodeId, start: start, end: end)
                        }
                    )
                    if let err = connectionError {
                        connectionErrorBanner(err)
                    }
                }
                parameterPanel
            }
        }
        .fileExporter(
            isPresented: $isExporting,
            document: document,
            contentType: .json,
            defaultFilename: documentName
        ) { result in
            if case .failure(let err) = result {
                connectionError = "Save failed: \(err.localizedDescription)"
            }
        }
        .fileImporter(
            isPresented: $isImporting,
            allowedContentTypes: [.json],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    loadDocument(from: url)
                }
            case .failure(let err):
                connectionError = "Open failed: \(err.localizedDescription)"
            }
        }
        .sheet(isPresented: $showRunProgress) {
            runProgressSheet
        }
        // Publish File > New / Open / Save actions to the
        // focused scene so the App-level menu commands
        // (Cmd-N, Cmd-O, Cmd-S, Shift-Cmd-S) can reach us.
        .focusedSceneValue(\.workflowMenuActions, WorkflowMenuActions(
            new: newDocument,
            open: { isImporting = true },
            save: { isExporting = true },
            canSave: { !workflow.nodes.isEmpty || !workflow.edges.isEmpty }
        ))
    }

    // MARK: - Undo registration

    /// Register a single undo / redo pair on the env UndoManager.
    /// The action name is shown in the Edit menu ("Undo Connect
    /// Nodes", "Undo Move Node", etc.).
    private func registerUndoPair(
        name: String,
        undo: @escaping (WorkflowsView) -> Void,
        redo: @escaping (WorkflowsView) -> Void
    ) {
        guard let mgr = undoManager else { return }
        mgr.setActionName(name)
        // `registerUndo(withTarget:handler:)` requires an
        // `AnyObject` target; `WorkflowsView` is a struct. The
        // `UndoPair` helper holds the closures and re-registers
        // itself for redo after each undo (and vice versa).
        // The captured `self` is a struct copy, but writes through
        // `@State` go to the same backing storage as the live
        // view, so the undo actually mutates the live state.
        let pair = UndoPair(
            undo: { undo(self) },
            redo: { redo(self) },
            manager: mgr
        )
        mgr.registerUndo(withTarget: pair) { $0.performUndo() }
    }

    private func recordNodeMove(nodeId: String, start: CGPoint, end: CGPoint) {
        guard start != end else { return }
        let nodeId = nodeId
        let start = start, end = end
        registerUndoPair(
            name: "Move Node",
            undo: { vc in
                var p = vc.positions
                p[nodeId] = start
                vc.positions = p
                vc.document = WorkflowDocument(workflow: vc.workflow, positions: vc.positions)
            },
            redo: { vc in
                var p = vc.positions
                p[nodeId] = end
                vc.positions = p
                vc.document = WorkflowDocument(workflow: vc.workflow, positions: vc.positions)
            }
        )
    }

    private func recordConnectionAddition(
        newEdge: WorkflowEdge,
        oldEdges: [WorkflowEdge]
    ) {
        registerUndoPair(
            name: "Connect Nodes",
            undo: { vc in
                vc.workflow = Workflow(
                    schema: vc.workflow.schema,
                    name: vc.workflow.name,
                    nodes: vc.workflow.nodes,
                    edges: oldEdges
                )
                vc.document = WorkflowDocument(workflow: vc.workflow, positions: vc.positions)
            },
            redo: { vc in
                vc.workflow = Workflow(
                    schema: vc.workflow.schema,
                    name: vc.workflow.name,
                    nodes: vc.workflow.nodes,
                    edges: oldEdges + [newEdge]
                )
                vc.document = WorkflowDocument(workflow: vc.workflow, positions: vc.positions)
            }
        )
    }

    // MARK: - Run progress sheet

    /// The run-progress sheet. Shows the live stream of
    /// ``WorkflowEvent``s from the executor. Closes when the
    /// run finishes (success or failure).
    private var runProgressSheet: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Run progress")
                    .font(.headline)
                Spacer()
                Button("Close") { showRunProgress = false }
                    .disabled(isRunning)
            }
            .padding(12)
            Divider()
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 4) {
                        ForEach(Array(runEvents.enumerated()), id: \.offset) { (idx, ev) in
                            runEventRow(ev)
                                .id(idx)
                        }
                    }
                    .padding(12)
                }
                .onChange(of: runEvents.count) { _, newCount in
                    if newCount > 0 {
                        proxy.scrollTo(newCount - 1, anchor: .bottom)
                    }
                }
            }
        }
        .frame(minWidth: 520, minHeight: 360)
    }

    @ViewBuilder
    private func runEventRow(_ event: WorkflowEvent) -> some View {
        switch event {
        case .started(let name, let total):
            HStack {
                Image(systemName: "play.circle.fill")
                    .foregroundStyle(.green)
                Text("Started \"\(name)\" — \(total) nodes")
            }
        case .nodeStarted(let id, let typeId):
            HStack {
                Image(systemName: "arrow.right.circle")
                    .foregroundStyle(.blue)
                Text("Node \(id) (\(typeId)) started")
            }
        case .nodeFinished(let id, let success, let msg):
            HStack {
                Image(systemName: success ? "checkmark.circle.fill" : "xmark.octagon.fill")
                    .foregroundStyle(success ? .green : .red)
                Text("Node \(id) \(success ? "finished" : "failed")\(msg.map { ": \($0)" } ?? "")")
            }
        case .log(_, let level, let message):
            HStack {
                Text(level.rawValue.uppercased())
                    .font(.caption2.monospaced())
                    .foregroundStyle(levelColor(level))
                    .frame(width: 50, alignment: .leading)
                Text(message)
                    .font(.callout)
            }
        case .finished(let success, let message):
            HStack {
                Image(systemName: success ? "checkmark.seal.fill" : "xmark.seal.fill")
                    .foregroundStyle(success ? .green : .red)
                Text(success ? "Workflow finished" : "Workflow failed")
                if let message {
                    Text("(\(message))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .font(.body.weight(.medium))
        }
    }

    private func levelColor(_ level: WorkflowLogLevel) -> Color {
        switch level {
        case .debug: return .secondary
        case .info:  return .blue
        case .warn:  return .orange
        case .error: return .red
        }
    }

    /// Drive the executor. Builds a per-run context (silent
    /// logger so the progress sheet is the only event
    /// surface), runs to completion, accumulates the events
    /// for the sheet, and clears the running flag on the
    /// final event.
    private func runWorkflow() {
        guard !isRunning else { return }
        runEvents.removeAll()
        isRunning = true
        showRunProgress = true
        let context = WorkflowExecutionContext(
            fileSystem: LocalTesseraFileSystem(),
            logger: SilentWorkflowLogger()
        )
        let executor = WorkflowExecutor(registry: registry)
        Task {
            for await event in await executor.run(workflow, context: context) {
                await MainActor.run { runEvents.append(event) }
                if case .finished = event {
                    await MainActor.run { isRunning = false }
                }
            }
        }
    }

    /// The right-hand parameter panel. Hidden visually but kept
    /// in the layout when no node is selected (so the HSplit
    /// divider doesn't jump around).
    @ViewBuilder
    private var parameterPanel: some View {
        if let id = selectedNodeId,
           let node = workflow.node(id: id),
           let type = registry.nodeType(for: node.type) {
            WorkflowParameterPanelView(
                node: node,
                type: type,
                parameters: parametersBinding(for: id)
            )
        } else {
            VStack {
                ContentUnavailableView(
                    "No selection",
                    systemImage: "cursorarrow.click",
                    description: Text("Click a node to edit its parameters.")
                )
            }
            .frame(minWidth: 240)
        }
    }

    /// Build a binding into `workflow.nodes[i].parameters`
    /// keyed by node id. Reads return the current dict;
    /// writes replace the node with a new copy (Swift value
    /// type) so the workflow change republishes.
    ///
    /// The `TextField` inside `WorkflowParameterPanelView` uses
    /// the same UndoManager from the environment, so per-keystroke
    /// undo is handled by the text field's own undo registration.
    /// We don't wrap parameter changes here to avoid creating
    /// one undo entry per keystroke.
    private func parametersBinding(
        for nodeId: String
    ) -> Binding<[String: JSONValue]> {
        Binding(
            get: {
                workflow.node(id: nodeId)?.parameters ?? [:]
            },
            set: { newParams in
                guard let idx = workflow.nodes.firstIndex(where: { $0.id == nodeId }) else { return }
                let old = workflow.nodes[idx]
                workflow = Workflow(
                    schema: workflow.schema,
                    name: workflow.name,
                    nodes: workflow.nodes.enumerated().map { (i, n) in
                        i == idx
                            ? WorkflowNode(
                                id: n.id, type: n.type, parameters: newParams
                            )
                            : n
                    },
                    edges: workflow.edges
                )
                _ = old // silence unused-let warning if any
            }
        )
    }

    private func connectionErrorBanner(_ message: String) -> some View {
        VStack {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                Text(message)
                    .font(.callout)
                Spacer()
                Button("Dismiss") { connectionError = nil }
                    .buttonStyle(.borderless)
            }
            .padding(10)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
            .padding(.top, 12)
            .padding(.horizontal, 16)
            Spacer()
        }
        .transition(.opacity)
    }

    private func newDocument() {
        workflow = Workflow(name: "untitled", nodes: [], edges: [])
        positions = [:]
        selectedNodeId = nil
        pendingConnection = nil
        documentName = "untitled.tessera-workflow"
        document = WorkflowDocument(workflow: workflow, positions: positions)
        // A "New" can't be undone; clear the stack.
        undoManager?.removeAllActions()
    }

    private func loadDocument(from url: URL) {
        // `fileImporter` gives us a URL with the security
        // scoped resource flag set; start access so we can
        // read it. End access on completion.
        let didStart = url.startAccessingSecurityScopedResource()
        defer { if didStart { url.stopAccessingSecurityScopedResource() } }
        do {
            let data = try Data(contentsOf: url)
            let envelope = try JSONDecoder().decode(
                WorkflowDocument.Envelope.self, from: data
            )
            workflow = envelope.workflow
            positions = envelope.positions ?? [:]
            selectedNodeId = nil
            document = WorkflowDocument(workflow: workflow, positions: positions)
            documentName = url.lastPathComponent
            // Loading a new file replaces the undo stack.
            undoManager?.removeAllActions()
        } catch {
            connectionError = "Open failed: \(error.localizedDescription)"
        }
    }

    private func completeConnection(
        at dropPoint: CGPoint, in canvasSize: CGSize
    ) {
        guard let pending = pendingConnection else { return }
        let target = nearestPort(
            to: dropPoint, except: pending.source.nodeId, side: .left
        )
        pendingConnection = nil
        guard let target else {
            connectionError = "Drop on an input port to create a connection."
            return
        }
        if target.nodeId == pending.source.nodeId {
            connectionError = "A node cannot connect to itself."
            return
        }
        if !pending.source.portType.canFlowInto(target.portType) {
            connectionError = "Type mismatch: \(pending.source.portType.rawValue) cannot flow into \(target.portType.rawValue)."
            return
        }
        if workflow.edges.contains(where: {
            $0.toNode == target.nodeId && $0.toPort == target.portId
        }) {
            connectionError = "Input port \"\(target.portId)\" already has a source."
            return
        }
        let oldEdges = workflow.edges
        let newEdge = WorkflowEdge(
            fromNode: pending.source.nodeId,
            fromPort: pending.source.portId,
            toNode: target.nodeId,
            toPort: target.portId
        )
        workflow = Workflow(
            schema: workflow.schema,
            name: workflow.name,
            nodes: workflow.nodes,
            edges: oldEdges + [newEdge]
        )
        document = WorkflowDocument(workflow: workflow, positions: positions)
        recordConnectionAddition(newEdge: newEdge, oldEdges: oldEdges)
    }

    private func nearestPort(
        to point: CGPoint, except sourceNodeId: String, side: WorkflowPortView.Side
    ) -> PendingPortEndpoint? {
        var best: (PendingPortEndpoint, CGFloat)?
        let threshold: CGFloat = 20
        for node in workflow.nodes where node.id != sourceNodeId {
            guard let type = registry.nodeType(for: node.type),
                  let pos = positions[node.id] else { continue }
            let ports = (side == .left) ? type.inputs : type.outputs
            for (idx, port) in ports.enumerated() {
                let center = WorkflowGeometry.portCenter(
                    nodeCenter: pos, portIndex: idx,
                    isLeft: side == .left,
                    portCount: ports.count
                )
                let dx = center.x - point.x
                let dy = center.y - point.y
                let d = (dx * dx + dy * dy).squareRoot()
                if d < threshold && (best == nil || d < best!.1) {
                    best = (PendingPortEndpoint(
                        nodeId: node.id, portId: port.id, portType: port.type
                    ), d)
                }
            }
        }
        return best?.0
    }

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

struct PendingConnection {
    var source: PendingPortEndpoint
    var currentPoint: CGPoint
}

struct PendingPortEndpoint: Equatable {
    let nodeId: String
    let portId: String
    let portType: WorkflowPortType
}

/// Pair of undo / redo closures used as the `AnyObject` target
/// for `UndoManager.registerUndo(withTarget:handler:)` (which
/// requires a class target, not a SwiftUI View struct). After
/// each invocation the helper re-registers itself in the
/// opposite direction, so the UndoManager always has one
/// pending entry that alternates between undo and redo.
@MainActor
private final class UndoPair {
    let undo: () -> Void
    let redo: () -> Void
    weak var manager: UndoManager?

    init(undo: @escaping () -> Void, redo: @escaping () -> Void, manager: UndoManager?) {
        self.undo = undo
        self.redo = redo
        self.manager = manager
    }

    func performUndo() {
        undo()
        guard let mgr = manager else { return }
        mgr.registerUndo(withTarget: self) { $0.performRedo() }
    }

    func performRedo() {
        redo()
        guard let mgr = manager else { return }
        mgr.registerUndo(withTarget: self) { $0.performUndo() }
    }
}
