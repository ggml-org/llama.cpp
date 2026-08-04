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
    /// The document as of the last New / Open / Save. Comparing
    /// it against `document` yields the "Edited" indicator - a
    /// derived value, not a flag that could desync.
    @State private var savedDocument: WorkflowDocument = WorkflowDocument(
        workflow: WorkflowsView.exampleWorkflow,
        positions: WorkflowsView.examplePositions
    )
    @State private var documentName: String = "calibrate-and-quantize"
    @State private var isExporting = false
    @State private var isImporting = false
    @State private var runPhase: WorkflowRunPhase = .idle

    @Environment(\.undoManager) private var undoManager

    /// True when the live document differs from the last saved /
    /// loaded snapshot. Drives the "Edited" marker in the window
    /// title (macOS documents show `<name> - Edited`).
    private var isEdited: Bool {
        document != savedDocument
    }

    var body: some View {
        VStack(spacing: 0) {
            WorkflowToolbarView(
                onNew: newDocument,
                onOpen: { isImporting = true },
                onSave: { isExporting = true },
                onRun: { runWorkflow() },
                isRunning: runPhase.isRunning
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
            contentType: .tesseraWorkflow,
            defaultFilename: documentName
        ) { result in
            switch result {
            case .success(let url):
                // Save succeeded: the current document is now the
                // saved baseline, and the title drops "- Edited".
                savedDocument = document
                documentName = url.deletingPathExtension().lastPathComponent
            case .failure(let err):
                connectionError = "Save failed: \(err.localizedDescription)"
            }
        }
        .fileImporter(
            isPresented: $isImporting,
            allowedContentTypes: [.tesseraWorkflow],
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
        .sheet(isPresented: runSheetPresented) {
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
        // The window title carries the document name plus the
        // standard "- Edited" marker while there are unsaved
        // changes (HIG: documents surface modification state in
        // the title bar, not just a dot somewhere).
        .navigationTitle(isEdited ? "\(documentName) - Edited" : documentName)
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
        // The drag gesture already wrote `end` into `positions`;
        // sync the save snapshot so the move both flips the
        // "Edited" marker and survives the next Cmd-S.
        document = WorkflowDocument(workflow: workflow, positions: positions)
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

    /// Drives the run-progress sheet off ``runPhase``. The sheet
    /// is presented for any non-idle phase; dismissing it returns
    /// the phase to `.idle`. While a run is in flight the Close
    /// button is disabled and only Cancel can end the run.
    private var runSheetPresented: Binding<Bool> {
        Binding(
            get: {
                if case .idle = runPhase { return false }
                return true
            },
            set: { presented in
                if !presented { runPhase = .idle }
            }
        )
    }

    /// The run-progress sheet. Shows the live stream of
    /// ``WorkflowEvent``s from the executor. While the run is in
    /// flight the sheet offers a Cancel button (the only way to
    /// stop a run); once the run reaches a terminal ``WorkflowRunOutcome``
    /// the footer states the outcome and Close becomes enabled.
    private var runProgressSheet: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Run progress")
                    .font(.headline)
                Spacer()
                switch runPhase {
                case .running(let task, _):
                    Button("Cancel", role: .cancel) { cancelRun(task) }
                        .accessibilityHint("Stop the workflow run")
                default:
                    Button("Close") { runPhase = .idle }
                }
            }
            .padding(12)
            Divider()
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 4) {
                        ForEach(Array(runPhase.events.enumerated()), id: \.offset) { (idx, ev) in
                            runEventRow(ev)
                                .id(idx)
                        }
                    }
                    .padding(12)
                }
                .onChange(of: runPhase.events.count) { _, newCount in
                    if newCount > 0 {
                        proxy.scrollTo(newCount - 1, anchor: .bottom)
                    }
                }
            }
            if case .finished(let outcome, _) = runPhase {
                Divider()
                runOutcomeFooter(outcome)
            }
        }
        .frame(minWidth: 520, minHeight: 360)
    }

    /// Terminal-outcome footer. One row per outcome case, each
    /// pairing a symbol with the text so the state is not
    /// color-only.
    @ViewBuilder
    private func runOutcomeFooter(_ outcome: WorkflowRunOutcome) -> some View {
        HStack(spacing: 6) {
            switch outcome {
            case .succeeded:
                Image(systemName: "checkmark.seal.fill").foregroundStyle(.green)
                Text("Workflow finished")
            case .failed(let message):
                Image(systemName: "xmark.seal.fill").foregroundStyle(.red)
                Text("Workflow failed")
                if let message {
                    Text("(\(message))").foregroundStyle(.secondary)
                }
            case .cancelled(let completedNodes):
                Image(systemName: "xmark.circle").foregroundStyle(.orange)
                Text("Cancelled after \(completedNodes) node(s)")
            }
            Spacer()
        }
        .font(.body.weight(.medium))
        .padding(12)
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
            HStack(spacing: 6) {
                // Symbol + text carry the severity; the color is
                // redundant emphasis, not the only signal.
                HStack(spacing: 3) {
                    Image(systemName: levelSymbol(level))
                    Text(level.rawValue.uppercased())
                        .font(.caption2.monospaced())
                }
                .foregroundStyle(levelColor(level))
                .frame(width: 70, alignment: .leading)
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

    private func levelSymbol(_ level: WorkflowLogLevel) -> String {
        switch level {
        case .debug: return "terminal"
        case .info:  return "info.circle"
        case .warn:  return "exclamationmark.triangle"
        case .error: return "xmark.octagon"
        }
    }

    /// Drive the executor. Builds a per-run context (silent
    /// logger so the progress sheet is the only event surface),
    /// runs to completion, and folds every event into
    /// ``runPhase``. The terminal `.finished` event is parsed
    /// exactly once into a ``WorkflowRunOutcome`` so the sheet
    /// footer switches on a structured result rather than the
    /// raw success flag.
    private func runWorkflow() {
        guard !runPhase.isRunning else { return }
        let context = WorkflowExecutionContext(
            fileSystem: LocalTesseraFileSystem(),
            logger: SilentWorkflowLogger()
        )
        let executor = WorkflowExecutor(registry: registry)
        let workflow = self.workflow
        let task = Task {
            for await event in await executor.run(workflow, context: context) {
                if Task.isCancelled { return }
                if let outcome = WorkflowRunOutcome(finishedEvent: event) {
                    await MainActor.run {
                        self.finishRun(
                            outcome: outcome, appending: event,
                            workflowName: workflow.name
                        )
                    }
                    return
                }
                await MainActor.run { self.appendRunEvent(event) }
            }
        }
        runPhase = .running(task: task, events: [])
    }

    /// Cancel the in-flight run. Cancels the driving task and
    /// transitions straight to a terminal `.cancelled` outcome;
    /// `completedNodes` counts the nodes that already finished.
    private func cancelRun(_ task: Task<Void, Never>) {
        guard case .running(_, let events) = runPhase else { return }
        task.cancel()
        runPhase = .finished(outcome: .cancelled(events: events), events: events)
    }

    private func appendRunEvent(_ event: WorkflowEvent) {
        guard case .running(let task, var events) = runPhase else { return }
        events.append(event)
        runPhase = .running(task: task, events: events)
    }

    private func finishRun(
        outcome: WorkflowRunOutcome,
        appending event: WorkflowEvent,
        workflowName: String
    ) {
        var events = runPhase.events
        events.append(event)
        runPhase = .finished(outcome: outcome, events: events)
        // Pull the user back only if they wandered off mid-run
        // (the notifier no-ops while the app is frontmost and
        // for cancellations).
        WorkflowRunNotifier.post(outcome: outcome, workflowName: workflowName)
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
                // Keep the save snapshot in sync so a parameter
                // edit both flips the "Edited" marker and is
                // actually included by the next Cmd-S.
                document = WorkflowDocument(workflow: workflow, positions: positions)
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
        documentName = "untitled"
        document = WorkflowDocument(workflow: workflow, positions: positions)
        // A fresh document starts clean, not "Edited".
        savedDocument = document
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
            // The freshly loaded file is the saved baseline.
            savedDocument = document
            // fileExporter appends the extension itself (driven
            // by contentType), so the suggested filename must be
            // the bare base name without the `.tessera-workflow`
            // suffix.
            documentName = url.deletingPathExtension().lastPathComponent
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

/// The run-lifecycle state for the editor. A sum type replaces
/// the old `(isRunning, showRunProgress, runEvents)` flag soup:
/// a run is either not started, in flight (with its driving task
/// and the events so far), or terminal (with a structured
/// ``WorkflowRunOutcome`` and the full event trail). Every UI
/// question - "show the sheet?", "offer Cancel?", "enable
/// Close?" - becomes an exhaustive switch instead of a flag
/// combination that could disagree with itself.
enum WorkflowRunPhase {
    case idle
    case running(task: Task<Void, Never>, events: [WorkflowEvent])
    case finished(outcome: WorkflowRunOutcome, events: [WorkflowEvent])

    var isRunning: Bool {
        if case .running = self { return true }
        return false
    }

    var events: [WorkflowEvent] {
        switch self {
        case .idle: return []
        case .running(_, let events): return events
        case .finished(_, let events): return events
        }
    }
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
