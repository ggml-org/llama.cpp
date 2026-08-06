import SwiftUI
import AppKit
import TesseraCore

// MARK: - CodeFileTreeView

/// The sidebar file tree. The view is a SwiftUI
/// `List` over the flattened `CodeFileTree`; the
/// `DisclosureGroup` per directory gives the
/// expand/collapse behavior.
///
/// **Why `OutlineGroup`, not `List(children:)`.** The
/// `OutlineGroup` API is the SwiftUI idiomatic way to
/// render a tree of `Identifiable` nodes; it manages
/// the disclosure state and the keyboard navigation
/// for free. The `CodeFileTreeNode`'s `id` is stable
/// across rebuilds (it's the absolute path), so the
/// `OutlineGroup` keeps the user's expansion state
/// even when the tree is rebuilt after a file change.
public struct CodeFileTreeView: View {

    @ObservedObject public var viewModel: CodeSurfaceViewModel
    @State private var searchText: String = ""

    public init(viewModel: CodeSurfaceViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        VStack(spacing: 0) {
            searchBar
            Divider()
            if viewModel.tree.root.children?.isEmpty ?? true {
                emptyState
            } else {
                fileList
            }
        }
    }

    private var searchBar: some View {
        HStack(spacing: 4) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField("Filter files", text: $searchText)
                .textFieldStyle(.plain)
            if !searchText.isEmpty {
                Button {
                    searchText = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
    }

    private var fileList: some View {
        List {
            OutlineGroup(
                viewModel.tree.root.children ?? [],
                id: \.id,
                children: \.optionalChildren
            ) { node in
                if node.isDirectory {
                    directoryRow(node)
                } else {
                    fileRow(node)
                }
            }
        }
        .listStyle(.sidebar)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "folder.badge.questionmark")
                .font(.system(size: 32))
                .foregroundStyle(.secondary)
            Text("No files yet")
                .font(.headline)
            if viewModel.watchedRoot == nil {
                Text("Pick a root directory to start watching.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            } else {
                Text("The watched root has no supported source files.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func directoryRow(_ node: CodeFileTreeNode) -> some View {
        HStack(spacing: 4) {
            Image(systemName: node.iconName)
                .foregroundStyle(.tint)
                .frame(width: 16)
            Text(node.name)
                .lineLimit(1)
        }
        .contentShape(Rectangle())
    }

    private func fileRow(_ node: CodeFileTreeNode) -> some View {
        HStack(spacing: 4) {
            Image(systemName: node.iconName)
                .foregroundStyle(.secondary)
                .frame(width: 16)
            Text(node.name)
                .lineLimit(1)
                .foregroundStyle(filtered(node) ? .primary : .secondary)
        }
        .contentShape(Rectangle())
        .onTapGesture {
            if let file = node.file {
                Task { await viewModel.open(file: file) }
            }
        }
        .contextMenu {
            if let file = node.file {
                Button("Open") {
                    Task { await viewModel.open(file: file) }
                }
                Button("Add tag…") {
                    Task { await viewModel.addTag("TODO", to: file.id) }
                }
            }
        }
    }

    private func filtered(_ node: CodeFileTreeNode) -> Bool {
        guard !searchText.isEmpty else { return true }
        return node.name.localizedCaseInsensitiveContains(searchText)
    }
}

// MARK: - CodeFileTreeNode convenience

extension CodeFileTreeNode {
    /// The `children` as an optional, for use with
    /// `OutlineGroup(children: \.optionalChildren)`.
    /// The base `children` is `[CodeFileTreeNode]?` --
    /// `OutlineGroup` wants a non-optional list with
    /// a `nil` for leaves, so we wrap it.
    var optionalChildren: [CodeFileTreeNode]? {
        return children
    }
}
