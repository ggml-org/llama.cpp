import XCTest
@testable import TesseraCore

/// Tests for ``CodeSearchIndex``. The index is
/// in-memory; the tests build a small fixture set +
/// run queries + assert the hit structure.
final class CodeSearchIndexTests: XCTestCase {

    private func makeIndex() -> CodeSearchIndex {
        let files = [
            CodeFile(
                path: "/tmp/project/main.swift",
                body: """
                import Foundation

                let greeting = "hello"
                func greet(name: String) -> String {
                    return "\\(greeting), \\(name)!"
                }
                """
            ),
            CodeFile(
                path: "/tmp/project/helper.py",
                body: """
                def greet(name):
                    return f"hi {name}"
                """
            ),
            CodeFile(
                path: "/tmp/project/data.json",
                body: """
                { "name": "greeting", "value": "hello" }
                """
            ),
        ]
        return CodeSearchIndex(files: files)
    }

    // MARK: - Basic search

    func testLiteralSearchFindsMatch() {
        let index = makeIndex()
        let hits = index.search(CodeSearchQuery(pattern: "greet"))
        XCTAssertGreaterThanOrEqual(hits.count, 2)
        // At least one hit per file containing "greet"
        let paths = Set(hits.map(\.file.path))
        XCTAssertTrue(paths.contains("/tmp/project/main.swift"))
        XCTAssertTrue(paths.contains("/tmp/project/helper.py"))
    }

    func testLiteralSearchIsCaseInsensitiveByDefault() {
        let index = makeIndex()
        let hits = index.search(CodeSearchQuery(pattern: "GREETING"))
        XCTAssertGreaterThan(hits.count, 0)
    }

    func testLiteralSearchIsCaseSensitiveWhenAsked() {
        let index = makeIndex()
        let hits = index.search(CodeSearchQuery(
            pattern: "GREETING", caseSensitive: true
        ))
        XCTAssertEqual(hits.count, 0)
    }

    // MARK: - Regex

    func testRegexSearch() {
        let index = makeIndex()
        // Look for a function declaration (matches both
        // `func greet` in Swift and `def greet` in Python).
        let hits = index.search(CodeSearchQuery(
            pattern: "(func|def)\\s+greet", isRegex: true
        ))
        XCTAssertGreaterThanOrEqual(hits.count, 2)
    }

    func testRegexWithCaptureGroups() {
        let index = makeIndex()
        // Look for "greeting" or "greet" with the `gr` prefix.
        let hits = index.search(CodeSearchQuery(
            pattern: "\\bgr(ee|eat)\\w*", isRegex: true
        ))
        XCTAssertGreaterThan(hits.count, 0)
    }

    // MARK: - Hit structure

    func testHitCarriesFileAndLine() {
        let index = makeIndex()
        // Use "import" which appears only on line 1 of main.swift.
        let hits = index.search(CodeSearchQuery(pattern: "import"))
        let firstHit = hits.first
        XCTAssertNotNil(firstHit)
        XCTAssertEqual(firstHit?.column, 1)
        XCTAssertEqual(firstHit?.line, 1)
        XCTAssertTrue(firstHit?.lineText.contains("import") == true)
    }

    func testHitCarriesMatchRange() {
        let index = makeIndex()
        let hits = index.search(CodeSearchQuery(pattern: "import"))
        let firstHit = hits.first
        XCTAssertNotNil(firstHit)
        let range = firstHit?.matchRange
        XCTAssertNotNil(range)
        // The range should match the search term length.
        XCTAssertEqual(range?.count, "import".count)
    }

    // MARK: - Empty query

    func testEmptyQueryReturnsNoHits() {
        let index = makeIndex()
        let hits = index.search(CodeSearchQuery(pattern: ""))
        XCTAssertEqual(hits.count, 0)
    }

    // MARK: - Max results

    func testMaxResultsLimitsOutput() {
        var index = CodeSearchIndex()
        var files: [CodeFile] = []
        for i in 0..<10 {
            files.append(CodeFile(
                path: "/tmp/file\(i).swift",
                body: "let token = \(i)\n"
            ))
        }
        index.setFiles(files)
        let hits = index.search(CodeSearchQuery(pattern: "token", maxResults: 5))
        XCTAssertEqual(hits.count, 5)
    }

    // MARK: - Upsert + remove

    func testUpsertReplacesExisting() {
        // Start with a single-file index (no other
        // "hello" content that would mask the test).
        var index = CodeSearchIndex(files: [
            CodeFile(path: "/tmp/main.swift", body: "let hello = 1\n")
        ])
        let updated = CodeFile(
            path: "/tmp/main.swift",
            body: "let bonjour = 1\n"
        )
        index.upsert(updated)
        let hits = index.search(CodeSearchQuery(pattern: "hello"))
        XCTAssertEqual(hits.count, 0)
        let hits2 = index.search(CodeSearchQuery(pattern: "bonjour"))
        XCTAssertEqual(hits2.count, 1)
    }

    func testRemoveDropsFile() {
        var index = makeIndex()
        index.remove(path: "/tmp/project/main.swift")
        let hits = index.search(CodeSearchQuery(pattern: "greet"))
        XCTAssertEqual(hits.filter { $0.file.path == "/tmp/project/main.swift" }.count, 0)
    }

    // MARK: - Group by file

    func testGroupByFile() {
        let index = makeIndex()
        let hits = index.search(CodeSearchQuery(pattern: "greet"))
        let grouped = CodeSearchIndex.groupByFile(hits)
        XCTAssertGreaterThanOrEqual(grouped.count, 2)
        // Each group is sorted by file path.
        let paths = grouped.map(\.file.path)
        XCTAssertEqual(paths, paths.sorted())
    }

    // MARK: - File count

    func testFileCount() {
        let index = makeIndex()
        XCTAssertEqual(index.fileCount, 3)
    }
}
