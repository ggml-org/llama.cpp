import SwiftUI

enum Destination: String, CaseIterable, Identifiable {
    case library = "Library"
    case playground = "Playground"
    case runs = "Runs"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .library: "books.vertical"
        case .playground: "bubble.left.and.text.bubble.right"
        case .runs: "clock.arrow.circlepath"
        }
    }
}

struct ContentView: View {
    @State private var selection: Destination? = .playground
    @State private var agentLoop = TesseraAgentLoop(
        registry: TesseraToolRegistry.default,
        approvalEngine: TesseraApprovalEngine()
    )

    var body: some View {
        NavigationSplitView {
            List(Destination.allCases, selection: $selection) { dest in
                Label(dest.rawValue, systemImage: dest.icon)
                    .tag(dest)
            }
            .navigationTitle("Tessera Studio")
            #if os(macOS)
            .frame(minWidth: 180)
            #endif
        } detail: {
            switch selection {
            case .library:
                LibraryView()
            case .playground:
                PlaygroundView(agentLoop: agentLoop)
            case .runs:
                RunsView()
            case nil:
                ContentUnavailableView(
                    "Select a destination",
                    systemImage: "sidebar.left",
                    description: Text("Choose Library, Playground, or Runs from the sidebar.")
                )
            }
        }
    }
}
