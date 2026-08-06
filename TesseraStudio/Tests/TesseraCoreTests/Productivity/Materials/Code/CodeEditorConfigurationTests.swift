import XCTest
@testable import TesseraCore

/// Tests for the ``CodeEditorConfiguration`` and the
/// ``EditorMode`` extension that exposes
/// `isCodeMode`. The configuration is a plain value
/// type; the tests cover its Codable round-trip +
/// the per-case behavior of the enum.
final class CodeEditorConfigurationTests: XCTestCase {

    // MARK: - defaults

    func testDefaultConfiguration() {
        let cfg = CodeEditorConfiguration.default
        XCTAssertTrue(cfg.showLineNumbers)
        XCTAssertTrue(cfg.codeFolding)
        XCTAssertTrue(cfg.multiCursor)
        XCTAssertTrue(cfg.findInFile)
        XCTAssertFalse(cfg.minimap)
        XCTAssertNil(cfg.syntaxHighlightingLanguage)
    }

    // MARK: - JSON round-trip

    func testConfigurationRoundTripsJSON() throws {
        let cfg = CodeEditorConfiguration(
            showLineNumbers: true,
            syntaxHighlightingLanguage: "swift",
            codeFolding: false,
            multiCursor: true,
            findInFile: true,
            minimap: true
        )
        let data = try JSONEncoder().encode(cfg)
        let decoded = try JSONDecoder().decode(CodeEditorConfiguration.self, from: data)
        XCTAssertEqual(decoded.showLineNumbers, cfg.showLineNumbers)
        XCTAssertEqual(decoded.syntaxHighlightingLanguage, cfg.syntaxHighlightingLanguage)
        XCTAssertEqual(decoded.codeFolding, cfg.codeFolding)
        XCTAssertEqual(decoded.multiCursor, cfg.multiCursor)
        XCTAssertEqual(decoded.findInFile, cfg.findInFile)
        XCTAssertEqual(decoded.minimap, cfg.minimap)
    }

    func testConfigurationHashable() {
        let a = CodeEditorConfiguration.default
        let b = CodeEditorConfiguration.default
        XCTAssertEqual(a, b)
        XCTAssertEqual(a.hashValue, b.hashValue)
    }

    // MARK: - EditorMode

    func testEditorModeIsCodeForCodeCases() {
        XCTAssertTrue(EditorMode.code.isCodeMode)
        XCTAssertTrue(EditorMode.codeWithConfig.isCodeMode)
        XCTAssertFalse(EditorMode.document.isCodeMode)
        XCTAssertFalse(EditorMode.notes.isCodeMode)
    }

    func testEditorModeStaticDefault() {
        XCTAssertTrue(EditorMode.codeDefault.showLineNumbers)
        XCTAssertTrue(EditorMode.codeDefault.codeFolding)
    }
}
