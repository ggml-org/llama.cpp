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
