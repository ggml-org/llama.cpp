import SwiftUI
import TesseraCore

// MARK: - CodeSurfaceView

/// The macOS Code surface host. Composes the file tree
/// sidebar, the code editor pane, the outline panel,
/// the search bar, and the git panel into a
/// `NavigationSplitView`.
///
/// **Layout:**
///   * Three columns: sidebar (tree) | editor | detail
///     (outline + git).
///   * The sidebar shows the watched root as a tree;
///     clicking a file opens it in the editor.
///   * The editor uses the same `TesseraEditorView` as
///     the Documents surface (Phase 2), configured for
///     code via `EditorMode.code` +
///     `CodeEditorConfiguration`. The current Phase 2
///     text view is `NSTextView`; the production swap
///     to `STTextView` is a follow-up.
///   * The detail column has a tab picker (Outline /
///     Git / Search); each tab owns its own sub-view.
///
/// **Threading.** The view observes the
/// `CodeSurfaceViewModel` (a `@MainActor`
/// `ObservableObject`). File events from the watcher
/// come in on the actor's mailbox; the view model
/// marshals them to the main actor for SwiftUI.
public struct CodeSurfaceView: View {

    public init(viewModel: CodeSurfaceViewModel) {
        self._viewModel = StateObject(wrappedValue: viewModel)
    }

    @StateObject private var viewModel: CodeSurfaceViewModel
    @State private var detailTab: DetailTab = .outline

    private enum DetailTab: String, CaseIterable, Identifiable {
        case outline, git, search
        var id: String { rawValue }
        var label: String {
            switch self {
            case .outline: return "Outline"
            case .git: return "Git"
            case .search: return "Search"
            }
        }
    }

    public var body: some View {
        NavigationSplitView {
            CodeFileTreeView(viewModel: viewModel)
                .frame(minWidth: 220, idealWidth: 280)
        } content: {
            CodeEditorPaneView(viewModel: viewModel)
                .frame(minWidth: 360)
        } detail: {
            VStack(spacing: 0) {
                Picker("Detail", selection: $detailTab) {
                    ForEach(DetailTab.allCases) { tab in
                        Text(tab.label).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
                .padding(8)
                Divider()
                Group {
                    switch detailTab {
                    case .outline:
                        CodeOutlineView(outline: viewModel.currentOutline)
                    case .git:
                        CodeGitPanelView(
                            viewModel: viewModel,
                            commits: viewModel.recentCommits,
                            blame: viewModel.currentBlame
                        )
                    case .search:
                        CodeSearchPanelView(viewModel: viewModel)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .frame(minWidth: 280, idealWidth: 360)
        }
        .navigationTitle("Code")
        .toolbar {
            ToolbarItem(placement: .status) {
                Text(viewModel.statusMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .onAppear {
            Task { await viewModel.start() }
        }
        .onDisappear {
            viewModel.stop()
        }
        .alert(
            "Code surface error",
            isPresented: Binding(
                get: { viewModel.lastError != nil },
                set: { if !$0 { viewModel.lastError = nil } }
            ),
            presenting: viewModel.lastError
        ) { _ in
            Button("OK") { viewModel.lastError = nil }
        } message: { error in
            Text(error)
        }
    }
}
