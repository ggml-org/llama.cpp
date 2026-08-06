import XCTest
@testable import TesseraCore

/// Tests for ``GitReadOnly`` and the git output
/// parsers. The tests use the test-only init that
/// takes a `launcher` closure, so no real `git`
/// binary is required; the launcher prints canned
/// output to the test's pipes.
final class GitReadOnlyTests: XCTestCase {

    // MARK: - parseCommits

    func testParseCommitsSingleCommit() {
        let output = """
        4d2b1f8c1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b|John Doe|john@example.com|1700000000|Initial commit
        main.swift
        README.md

        """
        let commits = GitReadOnly.parseCommits(output: output)
        XCTAssertEqual(commits.count, 1)
        let c = commits[0]
        XCTAssertEqual(c.hash, "4d2b1f8c1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b")
        XCTAssertEqual(c.authorName, "John Doe")
        XCTAssertEqual(c.authorEmail, "john@example.com")
        XCTAssertEqual(c.message, "Initial commit")
        XCTAssertEqual(c.filesChanged, ["main.swift", "README.md"])
        XCTAssertEqual(c.date.timeIntervalSince1970, 1700000000, accuracy: 1.0)
    }

    func testParseCommitsMultipleCommits() {
        let output = """
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa|Alice|alice@a.com|1700000000|First
        file1.swift
        file2.swift

        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb|Bob|bob@b.com|1700001000|Second
        file3.swift

        """
        let commits = GitReadOnly.parseCommits(output: output)
        XCTAssertEqual(commits.count, 2)
        XCTAssertEqual(commits[0].hash, "a" + String(repeating: "a", count: 39))
        XCTAssertEqual(commits[1].hash, "b" + String(repeating: "b", count: 39))
        XCTAssertEqual(commits[0].message, "First")
        XCTAssertEqual(commits[1].message, "Second")
        XCTAssertEqual(commits[1].filesChanged, ["file3.swift"])
    }

    func testParseCommitsEmptyOutput() {
        let commits = GitReadOnly.parseCommits(output: "")
        XCTAssertEqual(commits.count, 0)
    }

    func testParseCommitsHandlesCRLF() {
        let output = "hash123|Alice|a@a.com|1700000000|msg\r\nfile.swift\r\n\r\n"
        let commits = GitReadOnly.parseCommits(output: output)
        XCTAssertEqual(commits.count, 1)
        XCTAssertEqual(commits[0].filesChanged, ["file.swift"])
    }

    // MARK: - parseHunks

    func testParseHunksSingleHunk() {
        let output = """
        @@ -1,4 +1,5 @@
         line1
        -old line 2
        +new line 2
         line3
         line4
        +added line
        """
        let hunks = GitReadOnly.parseHunks(output: output)
        XCTAssertEqual(hunks.count, 1)
        XCTAssertEqual(hunks[0].oldStart, 1)
        XCTAssertEqual(hunks[0].oldCount, 4)
        XCTAssertEqual(hunks[0].newStart, 1)
        XCTAssertEqual(hunks[0].newCount, 5)
        XCTAssertEqual(hunks[0].lines.count, 6)
    }

    func testParseHunksMultipleHunks() {
        let output = """
        @@ -1,3 +1,4 @@
         a
        +b
         c
        @@ -10,3 +11,4 @@
         x
        +y
         z
        """
        let hunks = GitReadOnly.parseHunks(output: output)
        XCTAssertEqual(hunks.count, 2)
        XCTAssertEqual(hunks[0].oldStart, 1)
        XCTAssertEqual(hunks[1].oldStart, 10)
        XCTAssertEqual(hunks[1].newStart, 11)
    }

    func testParseHunksEmpty() {
        let hunks = GitReadOnly.parseHunks(output: "")
        XCTAssertEqual(hunks.count, 0)
    }

    func testParseRangeWithComma() {
        let hunk = GitReadOnly.parseHunks(output: "@@ -10,5 +20,7 @@\n a\n").first
        XCTAssertNotNil(hunk)
        XCTAssertEqual(hunk?.oldStart, 10)
        XCTAssertEqual(hunk?.oldCount, 5)
        XCTAssertEqual(hunk?.newStart, 20)
        XCTAssertEqual(hunk?.newCount, 7)
    }

    func testParseRangeWithoutComma() {
        // The header may omit the count when count = 1.
        let hunk = GitReadOnly.parseHunks(output: "@@ -10 +20 @@\n a\n").first
        XCTAssertNotNil(hunk)
        XCTAssertEqual(hunk?.oldStart, 10)
        XCTAssertEqual(hunk?.oldCount, 1)
        XCTAssertEqual(hunk?.newStart, 20)
        XCTAssertEqual(hunk?.newCount, 1)
    }

    // MARK: - parseBlame

    func testParseBlameSingleLine() {
        let output = """
        4d2b1f8c1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b 1 1 1
        author John
        author-mail <john@example.com>
        author-time 1700000000
        author-tz +0000
        committer John
        committer-mail <john@example.com>
        committer-time 1700000000
        summary Initial commit
        filename main.swift
        \tlet x = 1
        """
        let blame = GitReadOnly.parseBlame(output: output)
        XCTAssertEqual(blame.count, 1)
        XCTAssertEqual(blame[0].line, 1)
        XCTAssertEqual(blame[0].commit.hash, "4d2b1f8c1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b")
        XCTAssertEqual(blame[0].commit.authorName, "John")
        XCTAssertEqual(blame[0].commit.authorEmail, "john@example.com")
        XCTAssertEqual(blame[0].originalLine, "let x = 1")
    }

    func testParseBlameMultipleLines() {
        let output = """
        4d2b1f8c1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b 1 1 2
        author John
        author-mail <john@example.com>
        author-time 1700000000
        summary Initial commit
        filename main.swift
        \tlet x = 1
        5e3c2a0f1e2d3c4b5a69788796a5b4c3d2e1f0a9b 2 2
        author John
        author-mail <john@example.com>
        author-time 1700000000
        summary Initial commit
        filename main.swift
        \tlet y = 2
        """
        let blame = GitReadOnly.parseBlame(output: output)
        XCTAssertEqual(blame.count, 2)
        XCTAssertEqual(blame[0].line, 1)
        XCTAssertEqual(blame[0].originalLine, "let x = 1")
        XCTAssertEqual(blame[1].line, 2)
        XCTAssertEqual(blame[1].originalLine, "let y = 2")
    }

    func testParseBlameStripsEmailBrackets() {
        let output = """
        4d2b1f8c1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b 1 1 1
        author John
        author-mail <john@example.com>
        author-time 1700000000
        summary msg
        filename x.swift
        \tlet x = 1
        """
        let blame = GitReadOnly.parseBlame(output: output)
        XCTAssertEqual(blame[0].commit.authorEmail, "john@example.com")
    }

    // MARK: - actor

    func testValidateFailsForNonRepo() async throws {
        let actor = GitReadOnly(
            repoURL: URL(fileURLWithPath: "/nonexistent"),
            resolvedTopLevel: nil,
            isValid: false,
            launcher: { _, _ in Process() }
        )
        do {
            try await actor.validate()
            XCTFail("expected notARepository")
        } catch GitReadOnlyError.notARepository {
            // expected
        }
    }

    // MARK: - real git (if available)

    func testRealGitLogOnFixture() async throws {
        // The test creates a real temp git repo with
        // a single commit, then runs `git log` against
        // it. The test is skipped when git is not on
        // PATH (CI on a stripped image). For dev
        // machines the test is the integration check.
        try await XCTSkipIf(!gitAvailable(), "git not on PATH")
        let dir = makeTempGitRepo()
        defer { try? FileManager.default.removeItem(at: dir) }
        // Initialize the temp dir as a git repo.
        try await runShell(
            "/bin/sh",
            ["-c", "cd \(dir.path) && git init -q && git config user.email test@test && git config user.name Test"],
            in: dir
        )
        let file = CodeFile(
            path: dir.appendingPathComponent("hello.swift").path,
            body: "let x = 1\n"
        )
        FileManager.default.createFile(
            atPath: file.path,
            contents: file.body.data(using: .utf8)
        )
        // Commit the file in the temp repo.
        try await runShell(
            "/bin/sh",
            ["-c", "cd \(dir.path) && git add hello.swift && git commit -q -m 'Initial'"],
            in: dir
        )
        let actor = GitReadOnly(
            repoURL: dir,
            resolvedTopLevel: dir,
            isValid: true,
            launcher: { git, args in
                let p = Process()
                p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
                p.arguments = [git] + args
                return p
            }
        )
        let commits = try await actor.recentCommits(file: file, limit: 5)
        XCTAssertGreaterThanOrEqual(commits.count, 1)
        XCTAssertEqual(commits[0].message, "Initial")
        XCTAssertTrue(commits[0].filesChanged.contains("hello.swift"))
    }

    // MARK: - helpers

    private func gitAvailable() -> Bool {
        // Use a synchronous process check: `/usr/bin/env git --version`
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        p.arguments = ["git", "--version"]
        p.standardOutput = Pipe()
        p.standardError = Pipe()
        do {
            try p.run()
            p.waitUntilExit()
            return p.terminationStatus == 0
        } catch {
            return false
        }
    }

    private func makeTempGitRepo() -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("git-test-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true
        )
        return dir
    }

    private func runShell(
        _ exe: String, _ args: [String], in cwd: URL
    ) async throws {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: exe)
        p.arguments = args
        p.currentDirectoryURL = cwd
        p.standardOutput = Pipe()
        p.standardError = Pipe()
        try p.run()
        p.waitUntilExit()
        if p.terminationStatus != 0 {
            throw NSError(domain: "test", code: Int(p.terminationStatus))
        }
    }
}
