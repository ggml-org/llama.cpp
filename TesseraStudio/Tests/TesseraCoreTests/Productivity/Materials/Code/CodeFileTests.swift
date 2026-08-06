import XCTest
@testable import TesseraCore

/// Tests for the ``CodeFile`` model: JSON round-trip,
/// language detection from file extension, checksum
/// computation, and the entity type / subtype
/// conventions the data layer relies on.
final class CodeFileTests: XCTestCase {

    // MARK: - JSON round-trip

    func testCodeFileRoundTripsJSON() throws {
        let date = Date(timeIntervalSince1970: 1_500_000)
        let original = CodeFile(
            id: UUID(),
            path: "/Users/test/MyProject/Sources/Foo.swift",
            filename: "Foo.swift",
            language: "swift",
            body: """
            import Foundation

            struct Foo {
                let value: Int
            }
            """,
            size: 80,
            modifiedAt: date,
            checksum: "sha256:abc123",
            linkedEntityIDs: [UUID()],
            tags: ["core", "stable"],
            createdAt: date,
            updatedAt: date
        )
        let data = try original.jsonData()
        let decoded = try CodeFile.from(jsonData: data)
        XCTAssertEqual(decoded.id, original.id)
        XCTAssertEqual(decoded.path, original.path)
        XCTAssertEqual(decoded.filename, original.filename)
        XCTAssertEqual(decoded.language, original.language)
        XCTAssertEqual(decoded.body, original.body)
        XCTAssertEqual(decoded.size, original.size)
        XCTAssertEqual(decoded.modifiedAt, original.modifiedAt)
        XCTAssertEqual(decoded.checksum, original.checksum)
        XCTAssertEqual(decoded.linkedEntityIDs, original.linkedEntityIDs)
        XCTAssertEqual(decoded.tags, original.tags)
    }

    func testCodeFileJSONStringRoundTrips() throws {
        let file = CodeFile(
            path: "/tmp/x.py",
            body: "x = 1\n"
        )
        let str = try file.jsonDataString()
        let decoded = try CodeFile.from(jsonDataString: str)
        XCTAssertEqual(decoded.path, file.path)
        XCTAssertEqual(decoded.body, file.body)
    }

    // MARK: - Filename derivation

    func testFilenameIsDerivedFromPath() {
        let file = CodeFile(path: "/Users/test/MyProject/Sources/Foo.swift", body: "")
        XCTAssertEqual(file.filename, "Foo.swift")
    }

    func testFilenameFallsBackForEmptyPath() {
        let file = CodeFile(path: "/", body: "")
        // "/" has no lastPathComponent beyond itself
        // in Foundation's URL semantics.
        XCTAssertFalse(file.filename.isEmpty)
    }

    // MARK: - Language detection

    func testDetectLanguageFromSwiftExtension() {
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/Foo.swift"),
            "swift"
        )
    }

    func testDetectLanguageFromPythonExtension() {
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/script.py"),
            "python"
        )
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/stub.pyi"),
            "python"
        )
    }

    func testDetectLanguageFromTypeScriptExtensions() {
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/Foo.ts"),
            "typescript"
        )
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/Foo.tsx"),
            "typescript"
        )
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/Foo.jsx"),
            "typescript"
        )
    }

    func testDetectLanguageFromShellExtension() {
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/build.sh"),
            "shell"
        )
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/deploy.bash"),
            "shell"
        )
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/setup.zsh"),
            "shell"
        )
    }

    func testDetectLanguageFromCppAndC() {
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/foo.cpp"),
            "cpp"
        )
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/foo.cxx"),
            "cpp"
        )
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/foo.h"),
            "c"
        )
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/foo.c"),
            "c"
        )
    }

    func testDetectLanguageFromRustAndGo() {
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/foo.rs"),
            "rust"
        )
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/foo.go"),
            "go"
        )
    }

    func testDetectLanguageForUnknownExtension() {
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/foo.xyz"),
            CodeFile.unknownLanguage
        )
    }

    func testDetectLanguageIsCaseInsensitive() {
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/FOO.SWIFT"),
            "swift"
        )
        XCTAssertEqual(
            CodeFile.detectLanguage(forPath: "/tmp/Bar.PY"),
            "python"
        )
    }

    // MARK: - Checksum

    func testComputeChecksumIsStableForSameBody() {
        let body = "let x = 1\n"
        let a = CodeFile.computeChecksum(of: body)
        let b = CodeFile.computeChecksum(of: body)
        XCTAssertEqual(a, b)
    }

    func testComputeChecksumIsSHA256() {
        let body = "let x = 1\n"
        let checksum = CodeFile.computeChecksum(of: body)
        XCTAssertTrue(checksum.hasPrefix("sha256:"))
        // SHA-256 hex = 64 chars
        let hex = String(checksum.dropFirst("sha256:".count))
        XCTAssertEqual(hex.count, 64)
    }

    func testComputeChecksumChangesWithBody() {
        let a = CodeFile.computeChecksum(of: "let x = 1\n")
        let b = CodeFile.computeChecksum(of: "let x = 2\n")
        XCTAssertNotEqual(a, b)
    }

    func testBodyMatchesChecksum() {
        let file = CodeFile(
            path: "/tmp/x.swift",
            body: "let x = 1\n",
            checksum: CodeFile.computeChecksum(of: "let x = 1\n")
        )
        XCTAssertTrue(file.bodyMatches(checksum: file.checksum))
        XCTAssertFalse(file.bodyMatches(checksum: "sha256:bad"))
    }

    // MARK: - Entity type / subtype

    func testEntityTypeIsCode() {
        XCTAssertEqual(CodeFile.entityType, "code")
    }

    func testSubtypeStringMatchesLanguage() {
        let file = CodeFile(path: "/tmp/x.swift", body: "")
        XCTAssertEqual(file.subtypeString, "swift")
    }

    func testUnknownLanguageSubtype() {
        let file = CodeFile(path: "/tmp/x.unknown", body: "")
        XCTAssertEqual(file.subtypeString, "plain")
        XCTAssertFalse(file.hasKnownLanguage)
    }

    func testKnownLanguageSubtype() {
        let file = CodeFile(path: "/tmp/x.py", body: "")
        XCTAssertTrue(file.hasKnownLanguage)
    }
}
