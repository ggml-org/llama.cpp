import SwiftUI
import TesseraCore

/// Scene-lived state for the Workflows surface.
///
/// The shell rebuilds the detail column on every sidebar
/// destination switch, which used to destroy `WorkflowsView` and
/// its `@State` - throwing away in-flight edits and orphaning a
/// running run. `ContentView` owns this store instead (one per
/// window, lifetime = the scene), so switching destinations is a
/// presentation-only event: the document, its saved baseline, the
/// selection, and any in-flight run survive, and the run sheet
/// re-presents from ``runPhase`` when the user returns.
///
/// Hydration discipline: the whole-content state (workflow +
/// positions + name + saved baseline) is written in exactly four
/// places - `init`, ``hydrateNewDocument()``, ``hydrate(from:)``,
/// and ``markSaved(at:)`` - so the store is always fully formed.
/// ``document`` is DERIVED from the live workflow + positions
/// instead of a stored snapshot, so it cannot lag behind the
/// edits; the stale-snapshot save bugs are ruled out by
/// construction rather than by remembering to resync.
@MainActor
@Observable
final class WorkflowEditorStore {
    var workflow: Workflow
    var positions: WorkflowPositionMap
    var selectedNodeId: String?
    var documentName: String
    var runPhase: WorkflowRunPhase = .idle
    /// Whether this window currently shows the Workflows
    /// destination. Pushed in by the owning scene (ContentView)
    /// - the store itself stays navigation-agnostic. Read at run
    /// completion to decide whether the user needs a ping: a run
    /// that finishes with its own surface on screen needs none.
    /// Defaults to visible, the direction that suppresses pings.
    var workflowsSurfaceVisible = true
    /// The document as of the last New / Open / Save. Comparing
    /// it against the derived ``document`` yields the "Edited"
    /// indicator - a derived value, not a flag that could desync.
    private(set) var savedDocument: WorkflowDocument

    let registry: WorkflowNodeRegistry = .default

    init(
        workflow: Workflow,
        positions: WorkflowPositionMap,
        documentName: String
    ) {
        self.workflow = workflow
        self.positions = positions
        self.documentName = documentName
        self.savedDocument = WorkflowDocument(workflow: workflow, positions: positions)
    }

    /// The standard seed content (the two-node
    /// calibrate -> quantize example).
    convenience init() {
        self.init(
            workflow: WorkflowEditorStore.exampleWorkflow,
            positions: WorkflowEditorStore.examplePositions,
            documentName: "calibrate-and-quantize"
        )
    }

    /// The live document. Derived, not stored: every Save /
    /// export reads exactly what the canvas is editing.
    var document: WorkflowDocument {
        WorkflowDocument(workflow: workflow, positions: positions)
    }

    /// True when the live document differs from the last saved /
    /// loaded baseline. Drives the "Edited" marker in the window
    /// title (macOS documents show `<name> - Edited`).
    var isEdited: Bool { document != savedDocument }

    /// Save needs at least one node or edge to write.
    var canSave: Bool { !workflow.nodes.isEmpty || !workflow.edges.isEmpty }

    // MARK: - Hydration

    /// Replace the content with an empty document.
    func hydrateNewDocument() {
        workflow = Workflow(name: "untitled", nodes: [], edges: [])
        positions = [:]
        selectedNodeId = nil
        documentName = "untitled"
        // A fresh document starts clean, not "Edited".
        savedDocument = document
    }

    /// Replace the content with the file at `url`. Throws on
    /// read / decode failure; the caller presents the error.
    func hydrate(from url: URL) throws {
        // `fileImporter` gives us a URL with the security
        // scoped resource flag set; start access so we can
        // read it. End access on completion.
        let didStart = url.startAccessingSecurityScopedResource()
        defer { if didStart { url.stopAccessingSecurityScopedResource() } }
        let data = try Data(contentsOf: url)
        let envelope = try JSONDecoder().decode(
            WorkflowDocument.Envelope.self, from: data
        )
        workflow = envelope.workflow
        positions = envelope.positions ?? [:]
        selectedNodeId = nil
        // fileExporter appends the extension itself (driven
        // by contentType), so the suggested filename must be
        // the bare base name without the `.tessera-workflow`
        // suffix.
        documentName = url.deletingPathExtension().lastPathComponent
        // The freshly loaded file is the saved baseline.
        savedDocument = document
    }

    /// Record a successful save: the written file becomes the
    /// saved baseline and the title drops "- Edited".
    func markSaved(at url: URL) {
        savedDocument = document
        documentName = url.deletingPathExtension().lastPathComponent
    }

    // MARK: - Mutations

    /// Replace one node's parameters (Swift value semantics:
    /// rebuild the node and the workflow so the change publishes).
    /// No-op if the node is gone.
    func setParameters(for nodeId: String, values: [String: JSONValue]) {
        guard let idx = workflow.nodes.firstIndex(where: { $0.id == nodeId }) else { return }
        workflow = Workflow(
            schema: workflow.schema,
            name: workflow.name,
            nodes: workflow.nodes.enumerated().map { (i, n) in
                i == idx
                    ? WorkflowNode(id: n.id, type: n.type, parameters: values)
                    : n
            },
            edges: workflow.edges
        )
    }

    /// Replace the edge list, preserving schema / name / nodes.
    func setEdges(_ edges: [WorkflowEdge]) {
        workflow = Workflow(
            schema: workflow.schema,
            name: workflow.name,
            nodes: workflow.nodes,
            edges: edges
        )
    }

    /// Replace the node list, preserving schema / name / edges.
    func setNodes(_ nodes: [WorkflowNode]) {
        workflow = Workflow(
            schema: workflow.schema,
            name: workflow.name,
            nodes: nodes,
            edges: workflow.edges
        )
    }

    // MARK: - Run lifecycle

    /// Drive the executor. Builds a per-run context (silent
    /// logger so the progress sheet is the only event surface),
    /// runs to completion, and folds every event into
    /// ``runPhase``. The terminal `.finished` event is parsed
    /// exactly once into a ``WorkflowRunOutcome`` so the sheet
    /// footer switches on a structured result rather than the
    /// raw success flag.
    func runWorkflow() {
        guard !runPhase.isRunning else { return }
        let context = WorkflowExecutionContext(
            fileSystem: LocalTesseraFileSystem(),
            logger: SilentWorkflowLogger()
        )
        let executor = WorkflowExecutor(registry: registry)
        let snapshot = workflow
        let task = Task {
            for await event in await executor.run(snapshot, context: context) {
                if Task.isCancelled { return }
                if let outcome = WorkflowRunOutcome(finishedEvent: event) {
                    self.finishRun(
                        outcome: outcome, appending: event,
                        workflowName: snapshot.name
                    )
                    return
                }
                self.appendRunEvent(event)
            }
        }
        runPhase = .running(task: task, events: [])
    }

    /// Cancel the in-flight run. Cancels the driving task and
    /// transitions straight to a terminal `.cancelled` outcome;
    /// `completedNodes` counts the nodes that already finished.
    func cancelRun(_ task: Task<Void, Never>) {
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
        // Pull the user back only if they wandered off mid-run:
        // the notifier posts only when the outcome is not on
        // screen (app backgrounded, or the window showing another
        // destination) and never for cancellations.
        WorkflowRunNotifier.post(
            outcome: outcome, workflowName: workflowName,
            runSurfaceVisible: workflowsSurfaceVisible
        )
    }

    // MARK: - Seed content

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
