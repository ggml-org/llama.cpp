import SwiftUI
import SwiftData

@main
struct TesseraStudioApp: App {
    let container: ModelContainer

    init() {
        do {
            let schema = Schema([ChatMessage.self, RunRecord.self])
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
        #if os(macOS)
        .defaultSize(width: 1200, height: 800)
        #endif
    }
}
