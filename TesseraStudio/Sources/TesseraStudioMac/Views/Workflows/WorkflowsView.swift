import SwiftUI
import UniformTypeIdentifiers
import TesseraCore

/// The "Workflows" surface. A presentation shell over the
/// scene-lived ``WorkflowEditorStore`` owned by `ContentView`:
/// routes toolbar / palette / canvas / port gestures into the
/// store and renders its state. The document, its saved baseline,
/// the selection, and any in-flight run live in the store, so
/// they survive sidebar destination switches; what stays here is
/// presentation-only state that may safely reset on the way back
/// (the in-flight wire, the error banner, the file pickers, and
/// column visibility).
struct WorkflowsView: View {
    @Bindable var editor: WorkflowEditorStore

    @State private var pendingConnection: PendingConnection?
    @State private var connectionError: String?
    @State private var isExporting = false
    /// Save As presents its own exporter so the two save paths
    /// stay distinct presentation states (Save may later write
    /// back to a remembered URL without touching Save As).
    @State private var isSavingAs = false
    /// Save launched from the unsaved-changes alert (New / Open
    /// guard). Distinct from Save / Save As so the follow-up
    /// replacement only runs on THIS save's success.
    @State private var isSavingBeforeReplace = false
    /// The New / Open that triggered the unsaved-changes alert,
    /// held until Save succeeds, Discard is chosen, or Cancel
    /// drops it.
    @State private var pendingDocumentAction: PendingDocumentAction?
    @State private var showUnsavedChangesAlert = false
    @State private var isImporting = false
    /// Palette column visibility for the NavigationSplitView
    /// (View > Show/Hide Node Palette toggles it).
    @State private var paletteVisibility: NavigationSplitViewVisibility = .all
    /// The parameter panel is presented as an inspector (HIG:
    /// supplementary detail about the current selection). On by
    /// default so the editor reads the same as before.
    @State private var inspectorVisible = true
    /// Canvas zoom + pan. Owned here (not by the canvas) because
    /// the palette drop handler has to undo the transform when it
    /// converts its drop location into canvas coordinates.
    @State private var canvasZoom: CGFloat = 1
    @State private var canvasPan: CGSize = .zero
    /// Canvas keyboard focus. Arrow keys nudge the selected
    /// node (HIG T3-2); the canvas takes focus on open and
    /// whenever a node is selected.
    @FocusState private var canvasFocused: Bool

    @Environment(\.undoManager) private var undoManager
    /// HIG 2.7 / 3.6: under Reduce Motion the banner appears and
    /// disappears instantly instead of fading.
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        NavigationSplitView(columnVisibility: $paletteVisibility) {
            WorkflowPaletteView(registry: editor.registry)
                .navigationSplitViewColumnWidth(min: 200, ideal: 240, max: 320)
        } detail: {
            ZStack {
                WorkflowCanvasView(
                    workflow: editor.workflow,
                    registry: editor.registry,
                    positions: $editor.positions,
                    pendingConnection: $pendingConnection,
                    selectedNodeId: $editor.selectedNodeId,
                    pendingSourceType: pendingConnection?.source.portType,
                    zoom: $canvasZoom,
                    pan: $canvasPan,
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
            // Scope the animation to the banner state so the
            // canvas itself never animates; nil under Reduce
            // Motion makes the banner appear / disappear instantly.
            .animation(reduceMotion ? nil : .default, value: connectionError)
            // HIG 14.6: the palette drags a node-type id string;
            // add the node at the drop point, rejecting ids the
            // registry doesn't know.
            .dropDestination(for: String.self) { items, location in
                guard let typeId = items.first else { return false }
                // The drop location is in this (untransformed) stack's
                // space; undo zoom + pan before inserting the node.
                let canvasLocation = WorkflowGeometry.canvasPoint(
                    fromViewport: location, zoom: canvasZoom, pan: canvasPan)
                return addNode(typeId: typeId, at: canvasLocation)
            }
            // HIG T3-2: arrow keys move the selected node (1pt,
            // Shift for 10pt) - the keyboard alternative to
            // drag-to-move. The canvas holds keyboard focus so
            // the presses reach it.
            .focusable()
            .focused($canvasFocused)
            .focusEffectDisabled()
            .onKeyPress(
                keys: [.upArrow, .downArrow, .leftArrow, .rightArrow],
                phases: [.down, .repeat]
            ) { press in
                nudgeSelectedNode(press)
            }
            .onChange(of: editor.selectedNodeId) { _, newValue in
                if newValue != nil { canvasFocused = true }
            }
            .defaultFocus($canvasFocused, true)
            // The parameter panel is a HIG inspector: supplementary
            // detail about the canvas selection, toggleable from
            // View > Show/Hide Inspector.
            .inspector(isPresented: $inspectorVisible) {
                parameterPanel
                    .inspectorColumnWidth(min: 240, ideal: 300, max: 380)
            }
            .toolbar { workflowToolbarItems }
            // The outer shell already contributes the destination
            // sidebar toggle; the palette is toggled from the View
            // menu instead of a second, identical toolbar button.
            .toolbar(removing: .sidebarToggle)
        }
        .fileExporter(
            isPresented: $isExporting,
            document: editor.document,
            contentType: .tesseraWorkflow,
            defaultFilename: editor.documentName
        ) { result in
            switch result {
            case .success(let url):
                // Save succeeded: the store records the new
                // baseline and the title drops "- Edited".
                editor.markSaved(at: url)
            case .failure(let err):
                connectionError = "Save failed: \(err.localizedDescription)"
            }
        }
        // Save As: same live document, prefilled with the current
        // name, but its own panel presentation; the chosen URL
        // becomes the new saved baseline.
        .fileExporter(
            isPresented: $isSavingAs,
            document: editor.document,
            contentType: .tesseraWorkflow,
            defaultFilename: editor.documentName
        ) { result in
            switch result {
            case .success(let url):
                editor.markSaved(at: url)
            case .failure(let err):
                connectionError = "Save failed: \(err.localizedDescription)"
            }
        }
        // The New / Open guard's save: on success the pending
        // replacement runs; on failure it is dropped and the
        // document stays put.
        .fileExporter(
            isPresented: $isSavingBeforeReplace,
            document: editor.document,
            contentType: .tesseraWorkflow,
            defaultFilename: editor.documentName
        ) { result in
            switch result {
            case .success(let url):
                editor.markSaved(at: url)
                performPendingDocumentAction()
            case .failure(let err):
                pendingDocumentAction = nil
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
                    requestOpenDocument(from: url)
                }
            case .failure(let err):
                connectionError = "Open failed: \(err.localizedDescription)"
            }
        }
        .sheet(isPresented: runSheetPresented) {
            runProgressSheet
        }
        // HIG 14.7: New / Open replace the whole document. When
        // the current one has unsaved edits, confirm first -
        // Save finishes before the replacement, Discard replaces
        // directly, Cancel keeps everything as is.
        .alert(
            "Save changes to \"\(editor.documentName)\"?",
            isPresented: $showUnsavedChangesAlert
        ) {
            Button("Save…") { isSavingBeforeReplace = true }
                .keyboardShortcut(.defaultAction)
            Button("Discard", role: .destructive) { performPendingDocumentAction() }
            Button("Cancel", role: .cancel) { pendingDocumentAction = nil }
        } message: {
            Text("Your changes will be lost if you don't save them.")
        }
        // Publish File > New / Open / Save actions to the
        // focused scene so the App-level menu commands
        // (Cmd-N, Cmd-O, Cmd-S, Shift-Cmd-S) can reach us.
        .focusedSceneValue(\.workflowMenuActions, WorkflowMenuActions(
            new: requestNewDocument,
            open: { isImporting = true },
            save: { isExporting = true },
            saveAs: { isSavingAs = true },
            canSave: { editor.canSave },
            togglePalette: togglePalette,
            paletteVisible: { paletteVisibility != .detailOnly },
            toggleInspector: { inspectorVisible.toggle() },
            inspectorVisible: { inspectorVisible }
        ))
        // The window title carries the document name plus the
        // standard "- Edited" marker while there are unsaved
        // changes (HIG: documents surface modification state in
        // the title bar, not just a dot somewhere).
        .navigationTitle(editor.isEdited ? "\(editor.documentName) - Edited" : editor.documentName)
    }

    // MARK: - Window toolbar

    /// The window-toolbar items for the Workflows surface
    /// (replaces the old custom HStack toolbar). New / Open /
    /// Save sit in the secondary group; Run is the primary
    /// action, paired with the running spinner while a run is
    /// in flight.
    @ToolbarContentBuilder
    private var workflowToolbarItems: some ToolbarContent {
        ToolbarItemGroup(placement: .secondaryAction) {
            Button(action: requestNewDocument) {
                Label("New", systemImage: "doc.badge.plus")
            }
            .help("Create an empty workflow")
            .disabled(editor.runPhase.isRunning)
            .accessibilityLabel("New workflow")
            .accessibilityHint("Replace the current workflow with an empty one")

            Button(action: { isImporting = true }) {
                Label("Open", systemImage: "folder")
            }
            .help("Open a workflow file from disk")
            .disabled(editor.runPhase.isRunning)
            .accessibilityLabel("Open workflow")
            .accessibilityHint("Choose a workflow file to open")

            Button(action: { isExporting = true }) {
                Label("Save", systemImage: "square.and.arrow.down")
            }
            .help("Save the current workflow to disk")
            .disabled(editor.runPhase.isRunning)
            .accessibilityLabel("Save workflow")
            .accessibilityHint("Save the current workflow to a file")
        }
        ToolbarItemGroup(placement: .primaryAction) {
            if editor.runPhase.isRunning {
                ProgressView()
                    .controlSize(.small)
                    .accessibilityLabel("Workflow running")
            }
            Button(action: { editor.runWorkflow() }) {
                Label("Run", systemImage: "play.fill")
            }
            .help("Run the current workflow")
            .disabled(editor.runPhase.isRunning)
            .accessibilityLabel("Run workflow")
            .accessibilityHint("Execute the current workflow and show progress")
        }
    }

    private func togglePalette() {
        paletteVisibility = (paletteVisibility == .all) ? .detailOnly : .all
    }

    // MARK: - Undo registration

    /// Register a single undo / redo pair on the env UndoManager.
    /// The action name is shown in the Edit menu ("Undo Connect
    /// Nodes", "Undo Move Node", etc.). The closures capture the
    /// scene-lived store (a class), so an entry stays valid even
    /// when the view is recreated by a destination switch.
    private func registerUndoPair(
        name: String,
        undo: @escaping () -> Void,
        redo: @escaping () -> Void
    ) {
        guard let mgr = undoManager else { return }
        mgr.setActionName(name)
        // `registerUndo(withTarget:handler:)` requires an
        // `AnyObject` target; the `UndoPair` helper holds the
        // closures and re-registers itself for redo after each
        // undo (and vice versa).
        let pair = UndoPair(undo: undo, redo: redo, manager: mgr)
        mgr.registerUndo(withTarget: pair) { $0.performUndo() }
    }

    private func recordNodeMove(nodeId: String, start: CGPoint, end: CGPoint) {
        guard start != end else { return }
        let editor = self.editor
        registerUndoPair(
            name: "Move Node",
            undo: { editor.positions[nodeId] = start },
            redo: { editor.positions[nodeId] = end }
        )
    }

    private func recordConnectionAddition(
        newEdge: WorkflowEdge,
        oldEdges: [WorkflowEdge]
    ) {
        let editor = self.editor
        registerUndoPair(
            name: "Connect Nodes",
            undo: { editor.setEdges(oldEdges) },
            redo: { editor.setEdges(oldEdges + [newEdge]) }
        )
    }

    /// Insert a node of a registered type at the drop point and
    /// register undo. Returns false (drop rejected) for ids the
    /// registry doesn't know.
    private func addNode(typeId: String, at position: CGPoint) -> Bool {
        guard editor.registry.nodeType(for: typeId) != nil else { return false }
        let oldNodes = editor.workflow.nodes
        let oldPositions = editor.positions
        let node = WorkflowNode(id: makeNodeId(for: typeId), type: typeId)
        var newPositions = oldPositions
        newPositions[node.id] = position
        editor.setNodes(oldNodes + [node])
        editor.positions = newPositions
        let editor = self.editor
        registerUndoPair(
            name: "Add Node",
            undo: {
                editor.setNodes(oldNodes)
                editor.positions = oldPositions
            },
            redo: {
                editor.setNodes(oldNodes + [node])
                editor.positions = newPositions
            }
        )
        return true
    }

    /// First free id of the form <typeId>, <typeId>-2, ...
    private func makeNodeId(for typeId: String) -> String {
        var id = typeId
        var suffix = 2
        while editor.workflow.node(id: id) != nil {
            id = "\(typeId)-\(suffix)"
            suffix += 1
        }
        return id
    }

    // MARK: - Run progress sheet

    /// Drives the run-progress sheet off the store's run phase.
    /// The sheet is presented for any non-idle phase; dismissing
    /// it returns the phase to `.idle`. While a run is in flight
    /// the Close button is disabled and only Cancel can end the
    /// run.
    private var runSheetPresented: Binding<Bool> {
        Binding(
            get: {
                if case .idle = editor.runPhase { return false }
                return true
            },
            set: { presented in
                if !presented { editor.runPhase = .idle }
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
                switch editor.runPhase {
                case .running(let task, _):
                    Button("Cancel", role: .cancel) { editor.cancelRun(task) }
                        .accessibilityHint("Stop the workflow run")
                default:
                    Button("Close") { editor.runPhase = .idle }
                        .keyboardShortcut(.cancelAction)
                }
            }
            .padding(12)
            Divider()
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 4) {
                        ForEach(Array(editor.runPhase.events.enumerated()), id: \.offset) { (idx, ev) in
                            runEventRow(ev)
                                .id(idx)
                        }
                    }
                    .padding(12)
                }
                .onChange(of: editor.runPhase.events.count) { _, newCount in
                    if newCount > 0 {
                        proxy.scrollTo(newCount - 1, anchor: .bottom)
                    }
                }
            }
            if case .finished(let outcome, _) = editor.runPhase {
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

    /// The right-hand parameter panel. Hidden visually but kept
    /// in the layout when no node is selected (so the HSplit
    /// divider doesn't jump around).
    @ViewBuilder
    private var parameterPanel: some View {
        if let id = editor.selectedNodeId,
           let node = editor.workflow.node(id: id),
           let type = editor.registry.nodeType(for: node.type) {
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

    /// Build a binding into the store's workflow keyed by node
    /// id. Reads return the current dict; writes replace the
    /// node through the store (Swift value type), so the change
    /// republishes and the derived document picks it up.
    ///
    /// The `TextField` inside `WorkflowParameterPanelView` uses
    /// the same UndoManager from the environment, so per-keystroke
    /// undo is handled by the text field's own undo registration.
    /// We don't wrap parameter changes here to avoid creating
    /// one undo entry per keystroke.
    private func parametersBinding(
        for nodeId: String
    ) -> Binding<[String: JSONValue]> {
        let editor = self.editor
        return Binding(
            get: { editor.workflow.node(id: nodeId)?.parameters ?? [:] },
            set: { editor.setParameters(for: nodeId, values: $0) }
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
                    .keyboardShortcut(.cancelAction)
            }
            .padding(10)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
            .padding(.top, 12)
            .padding(.horizontal, 16)
            Spacer()
        }
        .transition(bannerTransition)
    }

    /// Fade under normal motion, instant (identity) under Reduce
    /// Motion.
    private var bannerTransition: AnyTransition {
        reduceMotion ? .identity : .opacity
    }

    private func newDocument() {
        editor.hydrateNewDocument()
        pendingConnection = nil
        // A "New" can't be undone; clear the stack.
        undoManager?.removeAllActions()
    }

    private func openDocument(from url: URL) {
        do {
            try editor.hydrate(from: url)
            pendingConnection = nil
            // Loading a new file replaces the undo stack.
            undoManager?.removeAllActions()
        } catch {
            connectionError = "Open failed: \(error.localizedDescription)"
        }
    }

    /// New with the unsaved-edits guard: clean documents hydrate
    /// directly, edited ones confirm first.
    private func requestNewDocument() {
        if editor.isEdited {
            pendingDocumentAction = .new
            showUnsavedChangesAlert = true
        } else {
            newDocument()
        }
    }

    /// Open of an already-picked file with the unsaved-edits
    /// guard. The importer always presents; the guard decides
    /// what happens with the chosen URL.
    private func requestOpenDocument(from url: URL) {
        if editor.isEdited {
            pendingDocumentAction = .open(url)
            showUnsavedChangesAlert = true
        } else {
            openDocument(from: url)
        }
    }

    /// Run the pending New / Open after the alert resolves via
    /// Discard or a successful save. No-op when nothing is
    /// pending (Cancel already dropped it).
    private func performPendingDocumentAction() {
        guard let action = pendingDocumentAction else { return }
        pendingDocumentAction = nil
        switch action {
        case .new:
            newDocument()
        case .open(let url):
            openDocument(from: url)
        }
    }

    /// Nudge the selected node with the arrow keys (HIG T3-2):
    /// 1pt per press, 10pt with Shift, one undoable step each.
    /// Returns .ignored with no selection so the keys fall
    /// through to whatever else handles them.
    private func nudgeSelectedNode(_ press: KeyPress) -> KeyPress.Result {
        guard let id = editor.selectedNodeId,
              let current = editor.positions[id] else { return .ignored }
        let step: CGFloat = press.modifiers.contains(.shift) ? 10 : 1
        let delta: CGSize
        switch press.key {
        case .upArrow: delta = CGSize(width: 0, height: -step)
        case .downArrow: delta = CGSize(width: 0, height: step)
        case .leftArrow: delta = CGSize(width: -step, height: 0)
        case .rightArrow: delta = CGSize(width: step, height: 0)
        default: return .ignored
        }
        let moved = CGPoint(x: current.x + delta.width, y: current.y + delta.height)
        let editor = self.editor
        editor.positions[id] = moved
        registerUndoPair(
            name: "Move Node",
            undo: { editor.positions[id] = current },
            redo: { editor.positions[id] = moved }
        )
        return .handled
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
        if editor.workflow.edges.contains(where: {
            $0.toNode == target.nodeId && $0.toPort == target.portId
        }) {
            connectionError = "Input port \"\(target.portId)\" already has a source."
            return
        }
        let oldEdges = editor.workflow.edges
        let newEdge = WorkflowEdge(
            fromNode: pending.source.nodeId,
            fromPort: pending.source.portId,
            toNode: target.nodeId,
            toPort: target.portId
        )
        editor.setEdges(oldEdges + [newEdge])
        recordConnectionAddition(newEdge: newEdge, oldEdges: oldEdges)
    }

    private func nearestPort(
        to point: CGPoint, except sourceNodeId: String, side: WorkflowPortView.Side
    ) -> PendingPortEndpoint? {
        var best: (PendingPortEndpoint, CGFloat)?
        let threshold: CGFloat = 20
        for node in editor.workflow.nodes where node.id != sourceNodeId {
            guard let type = editor.registry.nodeType(for: node.type),
                  let pos = editor.positions[node.id] else { continue }
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
}

/// A New / Open request held while the unsaved-changes alert is
/// up. `open` carries the picked URL so the replacement can run
/// after a successful save without re-presenting the importer.
enum PendingDocumentAction {
    case new
    case open(URL)
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
