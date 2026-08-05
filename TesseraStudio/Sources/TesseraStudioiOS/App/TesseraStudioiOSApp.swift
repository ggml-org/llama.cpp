#if os(iOS)
import SwiftUI
import SwiftData
import TesseraCore

@main
struct TesseraStudioiOSApp: App {
    let container: ModelContainer

    init() {
        TesseraSettings.registerDefaults()
        TesseraLearningServices.installDefaults()
        // "Plea the Fifth" composition root (phase 3). Same
        // shape as the macOS app: load the phrase from the
        // Keychain, install the UIKit text-input swizzle. The
        // actual wipe executor is wired in by phase 2; for now
        // the trigger is silent.
        Task { await CovertTriggerMonitor.shared.loadFromKeychain() }
        TextInputInterceptor.install()
        do {
            let schema = Schema([ChatMessage.self, RunRecord.self, Conversation.self])
            let config = ModelConfiguration("TesseraStudio", schema: schema)
            container = try ModelContainer(for: schema, configurations: [config])
        } catch {
            fatalError("Failed to create ModelContainer: \(error)")
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(container)
    }
}
#endif
