import SwiftUI
import TesseraCore

// MARK: - CodeGitPanelView

/// The Git panel: recent commits + diff + blame for the
/// current file. The view is read-only in v1 (no
/// `git commit` / `git push`); the toolbar shows a
/// refresh button that triggers a `git fetch` + a
/// re-read of the recent commits.
///
/// **v1 read-only.** The panel renders the
/// `GitCommit` list (newest first), the diff for the
/// latest commit (if the file is in a dirty state vs.
/// HEAD), and a per-line blame view when the user
/// expands a commit.
public struct CodeGitPanelView: View {

    @ObservedObject public var viewModel: CodeSurfaceViewModel
    public let commits: [GitCommit]?
    public let blame: [GitBlame]?

    public init(
        viewModel: CodeSurfaceViewModel,
        commits: [GitCommit]?,
        blame: [GitBlame]?
    ) {
        self.viewModel = viewModel
        self.commits = commits
        self.blame = blame
    }

    public var body: some View {
        if let commits {
            List(commits) { commit in
                commitRow(commit)
            }
            .listStyle(.plain)
        } else if let blame {
            List(blame.indices, id: \.self) { i in
                blameRow(blame[i])
            }
            .listStyle(.plain)
        } else {
            emptyState
        }
    }

    private func commitRow(_ commit: GitCommit) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(String(commit.hash.prefix(7)))
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                Text(commit.message)
                    .lineLimit(2)
                Spacer()
            }
            HStack(spacing: 6) {
                Text(commit.authorName)
                    .font(.caption)
                Text(commit.date, style: .relative)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if !commit.filesChanged.isEmpty {
                Text(commit.filesChanged.joined(separator: ", "))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .padding(.vertical, 2)
    }

    private func blameRow(_ line: GitBlame) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Text(String(line.line))
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .frame(width: 36, alignment: .trailing)
            Text(String(line.commit.hash.prefix(7)))
                .font(.caption.monospaced())
                .foregroundStyle(.tint)
            Text(line.originalLine)
                .font(.caption.monospaced())
                .lineLimit(1)
                .truncationMode(.tail)
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "clock.arrow.circlepath")
                .font(.system(size: 32))
                .foregroundStyle(.secondary)
            Text("No git history")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Text("Open a file inside a git repository to see its history.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 16)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
