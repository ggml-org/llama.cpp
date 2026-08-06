import Foundation

// MARK: - GitCommit

/// One commit returned by ``GitReadOnly/recentCommits(file:limit:)``.
/// The struct is what the Code surface's git panel renders;
/// the per-commit "files changed" list is best-effort
/// (we run `git show --name-only` per commit; for very
/// large commits we cap the list at 100 files).
public struct GitCommit: Codable, Sendable, Identifiable, Hashable {

    /// Full 40-char SHA-1 (or SHA-256 if the repo is
    /// `git config --shallow` aware). The Code surface
    /// uses the short SHA (first 7 chars) for display;
    /// the full SHA is for the receipt chain.
    public var hash: String

    /// Author name. Parsed from `git log --format=%an`.
    public var authorName: String

    /// Author email. Parsed from `git log --format=%ae`.
    public var authorEmail: String

    /// Commit date (committer date, not author date;
    /// matches what `git log` shows by default).
    public var date: Date

    /// Commit subject line (the first line of the
    /// commit message, no body).
    public var message: String

    /// Paths changed in this commit. The list is
    /// RELATIVE to the repo root; the consumer
    /// resolves them against the file's own path.
    /// Empty for the synthetic `deleted` events.
    public var filesChanged: [String]

    public var id: String { hash }

    public init(
        hash: String,
        authorName: String,
        authorEmail: String,
        date: Date,
        message: String,
        filesChanged: [String] = []
    ) {
        self.hash = hash
        self.authorName = authorName
        self.authorEmail = authorEmail
        self.date = date
        self.message = message
        self.filesChanged = filesChanged
    }
}

// MARK: - GitDiff

/// A diff between two revisions, as returned by
/// ``GitReadOnly/diff(file:since:)``. The struct
/// mirrors the `git diff` output: the `file` is the
/// path relative to the repo root; the `hunks` are
/// the per-hunk changes.
public struct GitDiff: Codable, Sendable, Hashable {
    public var file: String
    public var hunks: [DiffHunk]

    public init(file: String, hunks: [DiffHunk]) {
        self.file = file
        self.hunks = hunks
    }
}

/// One hunk of a diff. The hunk carries the line range
/// in the pre- and post-revision plus the per-line
/// content (`+`, `-`, or ` ` for context). The struct
/// is content-agnostic; the Code surface renders it in
/// a monospaced `Text` view.
public struct DiffHunk: Codable, Sendable, Hashable {
    /// `@@ -oldStart,oldCount +newStart,newCount @@`
    /// header, parsed. The struct holds the parsed
    /// numbers (not the raw header) so the view can
    /// format them itself.
    public var oldStart: Int
    public var oldCount: Int
    public var newStart: Int
    public var newCount: Int
    /// Each line is `" "` (context), `"+"` (added), or
    /// `"-"` (removed). The first character of each
    /// line is the marker; the rest is the file content.
    public var lines: [String]

    public init(
        oldStart: Int,
        oldCount: Int,
        newStart: Int,
        newCount: Int,
        lines: [String]
    ) {
        self.oldStart = oldStart
        self.oldCount = oldCount
        self.newStart = newStart
        self.newCount = newCount
        self.lines = lines
    }
}

// MARK: - GitBlame

/// One line of `git blame` output, as returned by
/// ``GitReadOnly/blame(file:)``. The struct pairs the
/// line number with the commit that introduced the line.
public struct GitBlame: Codable, Sendable, Hashable {
    public var line: Int
    public var commit: GitCommit
    public var originalLine: String

    public init(line: Int, commit: GitCommit, originalLine: String) {
        self.line = line
        self.commit = commit
        self.originalLine = originalLine
    }
}

// MARK: - GitReadOnly

/// Read-only Git integration for the Code surface. The
/// actor spawns `git` as a subprocess (`Process` +
/// `Pipe`), parses the output into the typed structs
/// above, and exposes the result to the SwiftUI view
/// through `async throws` methods.
///
/// **Why `Process`, not a Swift lib.** The mature Swift
/// Git libraries (SwiftGit, GitSwift) are either
/// unmaintained or carry a large surface for what we
/// need (we don't need remotes, branches, worktrees).
/// `Process` is a 100-line wrapper around `git log`,
/// `git diff`, and `git blame`; the parsing is
/// straightforward (the `--format=...` flags give us
/// machine-readable output). v2 can swap to a vendored
/// libgit2 if the surface grows (push, PR, conflict
/// resolution).
///
/// **v1 is read-only.** `git log`, `git diff`, `git blame`
/// are the only operations. v2 adds `git commit`,
/// `git push`, branch operations, and PR workflows.
///
/// **Repository discovery.** The actor's `init` takes
/// a `repoURL` (the working copy root). The actor
/// resolves every git command to that root; relative
/// paths in the output are relative to the root.
///
/// **Error model.** Errors are `GitReadOnlyError`; the
/// common case is "not a git repository" (the actor
/// runs `git rev-parse --show-toplevel` at construction
/// and surfaces the failure on every subsequent call).
public actor GitReadOnly {

    /// The repo's working-copy root. The actor validates
    /// this is a git repository in `init`; the property
    /// is the canonicalized path the actor uses for every
    /// subprocess call.
    public let repoURL: URL

    /// The git binary on `PATH`. The actor uses
    /// `/usr/bin/env` to find `git` so the user's
    /// PATH-driven git (Homebrew, asdf, ...) is honored.
    /// The override is for tests (a fixture that prints
    /// canned output).
    private let gitExecutable: (String, [String]) -> Process

    /// The repo's top-level directory (the output of
    /// `git rev-parse --show-toplevel`). Pre-resolved at
    /// `init` time so the per-call subprocess can `cd`
    /// to it without re-resolving.
    private let resolvedTopLevel: URL

    /// `true` when the repo at `repoURL` is a git repo.
    /// Computed at `init`; the actor throws
    /// `GitReadOnlyError.notARepository` on every call
    /// if false. The `isValidRepository` flag is set
    /// by the constructor (default + test-only);
    /// `resolvedTopLevel` is the source of truth for
    /// the default constructor, but the test-only
    /// constructor can pass `false` to simulate a
    /// non-repo without supplying an empty path.
    private let isValid: Bool

    public var isValidRepository: Bool { isValid }

    /// The default initializer. The actor shells out to
    /// `git` (via `/usr/bin/env`) at construction time
    /// to resolve the top-level. If the path is not a
    /// git repo, `isValidRepository` is `false` and
    /// every call throws.
    public init(repoURL: URL) {
        self.repoURL = repoURL.standardizedFileURL
        self.gitExecutable = { gitPath, args in
            // The default launcher uses /usr/bin/env
            // so the user's PATH-driven git is used.
            // v1 doesn't support a custom git binary;
            // the override is for tests.
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            p.arguments = [gitPath] + args
            return p
        }
        // Best-effort top-level resolution. We can't
        // `await` from `init` synchronously; the
        // resolution happens lazily on the first call
        // (or via `validate()` which the caller can
        // `await`).
        let resolved = Self.syncResolveTopLevel(
            for: self.repoURL,
            launcher: self.gitExecutable
        )
        self.resolvedTopLevel = resolved ?? URL(fileURLWithPath: "/")
        self.isValid = resolved != nil
    }

    /// Test-only initializer: the caller provides a
    /// synchronous "launch a process" closure. Tests
    /// use a fixture that prints canned output (no
    /// real `git` binary needed). Pass `isValid: false`
    /// to simulate a non-repo.
    init(
        repoURL: URL,
        resolvedTopLevel: URL?,
        isValid: Bool = true,
        launcher: @escaping (String, [String]) -> Process
    ) {
        self.repoURL = repoURL.standardizedFileURL
        self.resolvedTopLevel = resolvedTopLevel ?? URL(fileURLWithPath: "/")
        self.isValid = isValid
        self.gitExecutable = launcher
    }

    /// Validate the repo. The method is a no-op for a
    /// valid repo and throws otherwise. The caller
    /// typically calls this once at the start of a
    /// git-panel render and surfaces the error in the
    /// UI as a "Not a git repository" banner.
    public func validate() async throws {
        guard isValidRepository else {
            throw GitReadOnlyError.notARepository(url: repoURL)
        }
    }

    // MARK: - recentCommits

    /// The most recent `limit` commits that touched the
    /// given `file`. The result is in reverse-chrono
    /// order (newest first).
    ///
    /// **The git call.** `git log --format=%H|%an|%ae|%at|%s
    /// --date=unix --follow -- <path>`. The `--follow`
    /// flag walks the file's history through renames;
    /// the `--format` flag emits a parseable line per
    /// commit. The actor reads the output line-by-line
    /// and splits on the delimiter.
    public func recentCommits(
        file: CodeFile,
        limit: Int = 50
    ) async throws -> [GitCommit] {
        try await validate()
        let relPath = relativePath(of: file)
        let args = [
            "log",
            "--follow",
            "--no-merges",
            "--pretty=format:%H|%an|%ae|%at|%s",
            "--date=unix",
            "--name-only",
            "-n", String(limit),
            "--", relPath,
        ]
        let output = try await runGit(args, in: resolvedTopLevel)
        return Self.parseCommits(output: output)
    }

    /// Parse the `git log --pretty=format:... --name-only`
    /// output. Each commit is a header line + a list of
    /// file names (separated by blank lines).
    static func parseCommits(output: String) -> [GitCommit] {
        // The output is a sequence of blocks. Each block:
        //   <hash>|<authorName>|<authorEmail>|<unix-timestamp>|<subject>
        //   <file1>
        //   <file2>
        //   ...
        //   <blank line>
        // The block separator is a blank line; we split
        // on the block boundary and parse each block
        // independently.
        var commits: [GitCommit] = []
        // Normalize line endings (git on Windows
        // sometimes emits CRLF) and split into
        // non-empty blocks.
        let lines = output
            .replacingOccurrences(of: "\r\n", with: "\n")
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        var i = 0
        while i < lines.count {
            // Skip leading blank lines.
            if lines[i].isEmpty { i += 1; continue }
            let header = lines[i]
            let parts = header.split(separator: "|", maxSplits: 4,
                                     omittingEmptySubsequences: false)
                .map(String.init)
            guard parts.count == 5 else { i += 1; continue }
            let hash = parts[0]
            let authorName = parts[1]
            let authorEmail = parts[2]
            let unix = TimeInterval(parts[3]) ?? 0
            let message = parts[4]
            // Collect the file list (everything after the
            // header, until the next blank line).
            i += 1
            var files: [String] = []
            while i < lines.count && !lines[i].isEmpty {
                files.append(lines[i])
                i += 1
            }
            // The commit's date is the committer date (the
            // `%at` placeholder in the format). We use
            // `Date(timeIntervalSince1970:)`; the unix
            // timestamp is in seconds.
            commits.append(GitCommit(
                hash: hash,
                authorName: authorName,
                authorEmail: authorEmail,
                date: Date(timeIntervalSince1970: unix),
                message: message,
                filesChanged: files
            ))
        }
        return commits
    }

    // MARK: - diff

    /// The diff between the file's current contents and
    /// its state at `since` (a revision -- a SHA, a
    /// branch name, `HEAD~1`, etc.). The diff is parsed
    /// into a list of hunks; the file path is relative
    /// to the repo root.
    public func diff(
        file: CodeFile,
        since: String
    ) async throws -> GitDiff {
        try await validate()
        let relPath = relativePath(of: file)
        let args = [
            "diff",
            "--no-color",
            "--no-ext-diff",
            since,
            "--", relPath,
        ]
        let output = try await runGit(args, in: resolvedTopLevel)
        return GitDiff(file: relPath, hunks: Self.parseHunks(output: output))
    }

    /// Parse the `git diff` output. The format is
    /// sequence of hunks, each starting with
    /// `@@ -oldStart,oldCount +newStart,newCount @@`
    /// followed by the per-line content. The function
    /// returns an empty list for an empty diff (no
    /// changes between the two revisions).
    static func parseHunks(output: String) -> [DiffHunk] {
        let lines = output
            .replacingOccurrences(of: "\r\n", with: "\n")
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        var hunks: [DiffHunk] = []
        var i = 0
        while i < lines.count {
            let line = lines[i]
            if line.hasPrefix("@@") {
                // Parse the header.
                if let hunk = parseHunkHeader(line) {
                    i += 1
                    var content: [String] = []
                    while i < lines.count && !lines[i].hasPrefix("@@") &&
                          !lines[i].hasPrefix("diff ") {
                        content.append(lines[i])
                        i += 1
                    }
                    var parsed = hunk
                    parsed.lines = content
                    hunks.append(parsed)
                    continue
                }
            }
            i += 1
        }
        return hunks
    }

    private static func parseHunkHeader(_ line: String) -> DiffHunk? {
        // Format: @@ -oldStart,oldCount +newStart,newCount @@
        // The trailing ` @@` may carry the function name
        // (e.g. `@@ -1,4 +1,5 @@ func foo()`); we strip
        // everything after the second `@@`.
        guard line.hasPrefix("@@") else { return nil }
        // The header line is `@@ -X,Y +A,B @@`. The
        // commas may be omitted (e.g. `@@ -1 +1 @@` for
        // a single line).
        let stripped = line.dropFirst("@@".count)
        // Split on the next `@@`.
        let parts = stripped
            .split(separator: "@@", maxSplits: 1, omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
        guard parts.count >= 1 else { return nil }
        let header = parts[0]
        // The header is two halves separated by a space.
        // Each half is `-[oldspec]` and `+[newspec]`.
        let halves = header.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: false)
            .map(String.init)
        guard halves.count == 2 else { return nil }
        guard let (oldStart, oldCount) = parseRange(String(halves[0].dropFirst())),
              let (newStart, newCount) = parseRange(String(halves[1].dropFirst()))
        else { return nil }
        return DiffHunk(
            oldStart: oldStart,
            oldCount: oldCount,
            newStart: newStart,
            newCount: newCount,
            lines: []
        )
    }

    private static func parseRange(_ s: String) -> (Int, Int)? {
        // Format: "start[,count]". If `count` is
        // omitted, it's 1 by default.
        let parts = s.split(separator: ",")
        guard let start = Int(parts[0]) else { return nil }
        let count = parts.count > 1 ? (Int(parts[1]) ?? 1) : 1
        return (start, count)
    }

    // MARK: - blame

    /// `git blame` for the file. The result is one
    /// `GitBlame` per line in the file; the order is
    /// the file's line order (1-indexed).
    public func blame(file: CodeFile) async throws -> [GitBlame] {
        try await validate()
        let relPath = relativePath(of: file)
        let args = [
            "blame",
            "--line-porcelain",
            relPath,
        ]
        let output = try await runGit(args, in: resolvedTopLevel)
        return Self.parseBlame(output: output)
    }

    /// Parse `git blame --line-porcelain` output. The
    /// porcelain format is one block per line; each
    /// block is a header line (`<sha> <origLine> <finalLine> [<count>]`)
    /// followed by key/value pairs and the line content.
    /// The function groups the blocks by their final
    /// line number (the line number in the post-blame
    /// file).
    static func parseBlame(output: String) -> [GitBlame] {
        // The format is a sequence of blocks. Each block:
        //   <sha> <origLine> <finalLine> [<count>]
        //   author <name>
        //   author-mail <email>
        //   author-time <unix>
        //   ...
        //   \t<content>
        // We track the most recent commit across blocks
        // (the per-line commit is the same in many
        // consecutive lines). The function returns
        // one entry per line.
        let lines = output
            .replacingOccurrences(of: "\r\n", with: "\n")
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        var blame: [GitBlame] = []
        var currentSHA: String?
        var currentAuthorName: String = "Unknown"
        var currentAuthorEmail: String = ""
        var currentDate: Date = Date()
        var currentMessage: String = ""
        var currentFinalLine: Int = 0
        var currentContentLine: String?
        var i = 0
        while i < lines.count {
            let line = lines[i]
            if line.isEmpty { i += 1; continue }
            // The header line: 40-char hex SHA, space,
            // original line, space, final line, space,
            // optional count.
            let parts = line.split(separator: " ", maxSplits: 3)
                .map(String.init)
            if parts.count >= 3,
               parts[0].count >= 8,
               parts[0].allSatisfy({ $0.isHexDigit }) {
                // Emit the PREVIOUS block's blame
                // entry (if any) before starting the
                // new one. The previous block's
                // content line was captured by the
                // last iteration's metadata parse.
                if let prev = currentSHA,
                   let contentLine = currentContentLine {
                    let commit = GitCommit(
                        hash: prev,
                        authorName: currentAuthorName,
                        authorEmail: currentAuthorEmail,
                        date: currentDate,
                        message: currentMessage
                    )
                    blame.append(GitBlame(
                        line: currentFinalLine,
                        commit: commit,
                        originalLine: contentLine
                    ))
                }
                currentSHA = parts[0]
                currentAuthorName = "Unknown"
                currentAuthorEmail = ""
                currentDate = Date()
                currentMessage = ""
                currentFinalLine = Int(parts[2]) ?? 0
                currentContentLine = nil
                // Parse the rest of the block.
                var j = i + 1
                while j < lines.count && !lines[j].isEmpty &&
                      !lines[j].hasPrefix("\t") {
                    let kv = lines[j].split(separator: " ", maxSplits: 1)
                        .map(String.init)
                    if kv.count == 2 {
                        switch kv[0] {
                        case "author": currentAuthorName = kv[1]
                        case "author-mail":
                            // git wraps the email in `<>`;
                            // strip them.
                            currentAuthorEmail = kv[1]
                                .trimmingCharacters(in: CharacterSet(charactersIn: "<>"))
                        case "author-time":
                            if let unix = TimeInterval(kv[1]) {
                                currentDate = Date(timeIntervalSince1970: unix)
                            }
                        case "summary":
                            currentMessage = kv[1]
                        default: break
                        }
                    }
                    j += 1
                }
                // The content line is the `\t`-prefixed
                // line immediately after the metadata.
                if j < lines.count, lines[j].hasPrefix("\t") {
                    currentContentLine = String(lines[j].dropFirst())
                }
                i = j + 1
                continue
            }
            i += 1
        }
        // Emit the final block's blame entry (no
        // subsequent block to trigger it).
        if let prev = currentSHA,
           let contentLine = currentContentLine {
            let commit = GitCommit(
                hash: prev,
                authorName: currentAuthorName,
                authorEmail: currentAuthorEmail,
                date: currentDate,
                message: currentMessage
            )
            blame.append(GitBlame(
                line: currentFinalLine,
                commit: commit,
                originalLine: contentLine
            ))
        }
        return blame
    }

    private static func readContentLine(after index: Int, in lines: [String]) -> String? {
        // The content line is the `\t`-prefixed line
        // immediately after the per-line metadata. We
        // scan forward from `index + 1` for the first
        // line that starts with `\t` and return the
        // rest.
        var j = index + 1
        while j < lines.count {
            if lines[j].hasPrefix("\t") {
                return String(lines[j].dropFirst())
            }
            if lines[j].isEmpty { break }
            j += 1
        }
        return nil
    }

    // MARK: - Subprocess

    /// Run a git command in `workdir` and return its
    /// stdout as a `String`. The stderr is captured
    /// for diagnostics; a non-zero exit throws
    /// `GitReadOnlyError.gitFailed(...)`.
    private func runGit(_ args: [String], in workdir: URL) async throws -> String {
        let process = gitExecutable("git", args)
        process.currentDirectoryURL = workdir
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe
        do {
            try process.run()
        } catch {
            throw GitReadOnlyError.gitBinaryUnavailable(
                reason: "failed to launch git: \(error)"
            )
        }
        // Read both pipes concurrently to avoid a
        // deadlock on a large stderr (the subprocess
        // blocks waiting for the pipe to drain). The
        // `readDataToEndOfFile()` variant doesn't
        // throw (the throwing `readToEnd()` requires
        // a try that conflicts with the `async let`
        // here).
        let outHandle = outPipe.fileHandleForReading
        let errHandle = errPipe.fileHandleForReading
        async let outData: Data = outHandle.readDataToEndOfFile()
        async let errData: Data = errHandle.readDataToEndOfFile()
        process.waitUntilExit()
        let stdout = await outData
        let stderr = await errData
        if process.terminationStatus != 0 {
            let errStr = String(data: stderr, encoding: .utf8) ?? ""
            throw GitReadOnlyError.gitFailed(
                command: "git " + args.joined(separator: " "),
                exitCode: process.terminationStatus,
                stderr: errStr
            )
        }
        return String(data: stdout, encoding: .utf8) ?? ""
    }

    /// Compute the path of `file` relative to the repo
    /// root. The function preserves the leading `./` if
    /// git expects it (git on macOS accepts both).
    private func relativePath(of file: CodeFile) -> String {
        let fileURL = URL(fileURLWithPath: file.path).standardizedFileURL
        let rootPath = resolvedTopLevel.path
        let filePath = fileURL.path
        if filePath.hasPrefix(rootPath + "/") {
            return String(filePath.dropFirst(rootPath.count + 1))
        }
        if filePath == rootPath {
            return "."
        }
        return filePath
    }

    /// Synchronous best-effort `git rev-parse --show-toplevel`
    /// resolver. Used at `init` time (we can't `await`
    /// from a non-isolated init). On any error, the
    /// function returns nil and `resolvedTopLevel`
    /// stays empty; the actor throws on subsequent
    /// calls.
    private static func syncResolveTopLevel(
        for repoURL: URL,
        launcher: (String, [String]) -> Process
    ) -> URL? {
        let process = launcher("git", ["rev-parse", "--show-toplevel"])
        process.currentDirectoryURL = repoURL
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
        } catch {
            return nil
        }
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let path = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return path.isEmpty ? nil : URL(fileURLWithPath: path)
    }
}

// MARK: - Errors

public enum GitReadOnlyError: Error, Sendable, Equatable {
    case notARepository(url: URL)
    case gitBinaryUnavailable(reason: String)
    case gitFailed(command: String, exitCode: Int32, stderr: String)
    case fileNotInRepository(path: String)
    case binaryFileSkipped(path: String)
    case parseError(line: String, reason: String)
}
