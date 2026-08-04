import SwiftUI
import SwiftData
import TesseraCore

enum Destination: String, CaseIterable, Identifiable {
    case library = "Library"
    case playground = "Playground"
    case runs = "Runs"
    case learning = "Learning"
    case workflows = "Workflows"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .library: "books.vertical"
        case .playground: "bubble.left.and.text.bubble.right"
        case .runs: "clock.arrow.circlepath"
        case .learning: "chart.bar.doc.horizontal"
        case .workflows: "rectangle.connected.to.line.below"
        }
    }
}

/// macOS Studio shell: 3-destination split view with a leading chat-history
/// drawer, a bottom telemetry drawer, first-run onboarding, and export.
struct ContentView: View {
    @AppStorage(TesseraSettingsKey.onboardingComplete) private var onboardingComplete = false
    @Environment(\.modelContext) private var modelContext

    // Per-window scene restoration (HIG 2.5). SwiftUI persists
    // these when the window closes or the app quits and restores
    // them on relaunch; every window keeps its own copy. The
    // destination is stored as its raw string because
    // @SceneStorage needs a RawRepresentable/String-compatible
    // value, and Optional<Destination> is not one.
    @SceneStorage("ContentView.destination") private var storedDestinationRaw: String?
    @SceneStorage("ContentView.telemetryExpanded") private var telemetryExpanded = false

    @State private var agentLoop = TesseraAgentLoop(
        registry: TesseraToolRegistry.default,
        approvalEngine: TesseraApprovalEngine(),
        llmProvider: TesseraLLMProviderFactory.makeFromSettings(),
        maxIterations: TesseraSettings.maxIterations,
        tokenLimit: TesseraSettings.tokenBudget
    )
    @State private var showHistory = false
    @State private var telemetryMonitor = TelemetryMonitor(
        bridge: TesseraEngineBridgeFactory.makeInferenceBridge()
    )
    @State private var restoredMessages: [ChatMessage] = []
    @State private var playgroundSession = UUID()
    @State private var exportItem: ExportItem?

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            detail
        }
        .overlay(alignment: .leading) {
            if showHistory {
                historyDrawer
                    .transition(.move(edge: .leading))
            }
        }
        .sheet(item: $exportItem) { item in
            ExportView(item: item)
        }
        .sheet(isPresented: Binding(
            get: { !onboardingComplete },
            set: { onboardingComplete = !$0 }
        )) {
            OnboardingView()
        }
        .onChange(of: agentLoop.isRunning) { _, running in
            if running {
                telemetryMonitor.start()
            } else {
                telemetryMonitor.stop()
            }
        }
        // Publish the telemetry drawer toggle to the focused scene
        // so View > Show/Hide Telemetry can reach it from any
        // destination.
        .focusedSceneValue(\.telemetryMenuActions, TelemetryMenuActions(
            toggle: { withAnimation { telemetryExpanded.toggle() } },
            isExpanded: { telemetryExpanded }
        ))
    }

    /// Current sidebar destination. Falls back to the Playground
    /// on first launch (nothing stored yet) and whenever a stored
    /// value no longer parses (a destination renamed or removed
    /// between versions degrades to the default, not a crash).
    private var selection: Destination? {
        storedDestinationRaw.flatMap(Destination.init(rawValue:)) ?? .playground
    }

    private var selectionBinding: Binding<Destination?> {
        Binding(
            get: { selection },
            set: { storedDestinationRaw = $0?.rawValue }
        )
    }

    private var sidebar: some View {
        List(Destination.allCases, selection: selectionBinding) { dest in
            Label(dest.rawValue, systemImage: dest.icon)
                .tag(dest)
        }
        .navigationTitle("Tessera Studio")
        .frame(minWidth: 180)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button("History", systemImage: "sidebar.left") {
                    withAnimation { showHistory.toggle() }
                }
                .accessibilityHint("Shows or hides the chat history drawer")
            }
        }
    }

    @ViewBuilder
    private var detail: some View {
        VStack(spacing: 0) {
            detailContent
            TelemetryDrawer(monitor: telemetryMonitor, isExpanded: $telemetryExpanded)
        }
    }

    @ViewBuilder
    private var detailContent: some View {
        switch selection {
        case .library:
            LibraryView()
        case .playground:
            PlaygroundView(agentLoop: agentLoop, restoredMessages: restoredMessages)
                .id(playgroundSession)
        case .runs:
            RunsView()
        case .learning:
            LearningDashboardView()
        case .workflows:
            WorkflowsView()
        case nil:
            ContentUnavailableView(
                "Select a destination",
                systemImage: "sidebar.left",
                description: Text("Choose Library, Playground, Runs, Learning, or Workflows from the sidebar.")
            )
        }
    }

    private var historyDrawer: some View {
        ChatHistoryDrawer(
            isPresented: $showHistory,
            onRestore: { convo in restore(convo) },
            onExport: { convo, format in exportConversation(convo, format: format) }
        )
        .frame(width: 300)
        .clipShape(RoundedRectangle(cornerRadius: 0))
        .shadow(radius: 8)
    }

    private func restore(_ convo: Conversation) {
        restoredMessages = ConversationStore.messages(for: convo.id, in: modelContext)
        playgroundSession = UUID()
        storedDestinationRaw = Destination.playground.rawValue
        withAnimation { showHistory = false }
    }

    private func exportConversation(_ convo: Conversation, format: ExportFormat) {
        let messages = ConversationStore.messages(for: convo.id, in: modelContext)
        let base = slug(convo.title)
        switch format {
        case .markdown:
            let md = ConversationExporter.markdown(title: convo.title, messages: messages)
            exportItem = ExportItem(title: convo.title, filename: "\(base).md", data: Data(md.utf8))
        case .json:
            let js = ConversationExporter.json(title: convo.title, messages: messages)
            exportItem = ExportItem(title: convo.title, filename: "\(base).json", data: Data(js.utf8))
        case .pdf, .png:
            break
        }
    }

    private func slug(_ title: String) -> String {
        let cleaned = title.lowercased()
            .replacingOccurrences(of: " ", with: "-")
            .filter { $0.isLetter || $0.isNumber || $0 == "-" }
        return cleaned.isEmpty ? "conversation" : String(cleaned.prefix(48))
    }
}
