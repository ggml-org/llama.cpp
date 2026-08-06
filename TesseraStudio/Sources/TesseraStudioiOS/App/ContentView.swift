#if os(iOS)
import SwiftUI
import SwiftData
import TesseraCore

/// iOS Studio shell: tab bar (Library / Playground / Runs / Settings) with
/// first-run onboarding, a chat-history sheet, and a telemetry drawer.
struct ContentView: View {
    @AppStorage(TesseraSettingsKey.onboardingComplete) private var onboardingComplete = false
    @Environment(\.modelContext) private var modelContext

    @State private var agentLoop = TesseraAgentLoop(
        registry: TesseraToolRegistry.default,
        approvalEngine: TesseraApprovalEngine(),
        llmProvider: TesseraLLMProviderFactory.makeFromSettings(),
        maxIterations: TesseraSettings.maxIterations,
        tokenLimit: TesseraSettings.tokenBudget
    )
    @State private var telemetryMonitor = TelemetryMonitor(
        bridge: TesseraEngineBridgeFactory.makeInferenceBridge()
    )
    @State private var telemetryExpanded = false
    @State private var showHistory = false
    @State private var restoredMessages: [ChatMessage] = []
    @State private var playgroundSession = UUID()
    @State private var exportItem: ExportItem?
    // Email surface state (Phase 5). Same
    // lazy-bootstrap pattern as the macOS
    // ``EmailSurfaceBootstrap``; the
    // iOS-side ``EmailView_iOS`` is
    // constructed once the user opens the
    // Email tab.
    @State private var emailSurface = EmailSurfaceBootstrap_iOS()

    var body: some View {
        TabView {
            NavigationStack {
                LibraryView()
            }
            .tabItem { Label("Library", systemImage: "books.vertical") }

            NavigationStack {
                VStack(spacing: 0) {
                    PlaygroundView(agentLoop: agentLoop, restoredMessages: restoredMessages)
                        .id(playgroundSession)
                    TelemetryDrawer(monitor: telemetryMonitor, isExpanded: $telemetryExpanded)
                }
                .toolbar {
                    ToolbarItem(placement: .primaryAction) {
                        Button("History", systemImage: "sidebar.left") { showHistory = true }
                    }
                }
            }
            .tabItem { Label("Playground", systemImage: "bubble.left.and.text.bubble.right") }

            NavigationStack {
                RunsView()
            }
            .tabItem { Label("Runs", systemImage: "clock.arrow.circlepath") }

            NavigationStack {
                emailSurface.installIfNeeded()
                EmailView_iOS(
                    store: emailSurface.store,
                    sender: emailSurface.sender,
                    importer: emailSurface.importer,
                    identity: emailSurface.identity
                )
            }
            .tabItem { Label("Email", systemImage: "envelope") }

            NavigationStack {
                SettingsView()
            }
            .tabItem { Label("Settings", systemImage: "gearshape") }
        }
        .sheet(isPresented: $showHistory) {
            NavigationStack {
                ChatHistoryDrawer(
                    isPresented: $showHistory,
                    onRestore: { convo in restore(convo) },
                    onExport: { convo, format in exportConversation(convo, format: format) }
                )
            }
        }
        .sheet(item: $exportItem) { item in
            ExportView(item: item)
        }
        .fullScreenCover(isPresented: Binding(
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
    }

    private func restore(_ convo: Conversation) {
        restoredMessages = ConversationStore.messages(for: convo.id, in: modelContext)
        playgroundSession = UUID()
        showHistory = false
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
#endif
