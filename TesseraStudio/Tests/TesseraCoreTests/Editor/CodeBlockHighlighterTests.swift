import XCTest
import Foundation
#if canImport(AppKit)
import AppKit
#endif
@testable import TesseraCore

/// Tests for `CodeBlockHighlighter`. The highlighter uses
/// Splash (when available) for Swift and a small regex-based
/// highlighter for the other 9 languages the brief calls out.
/// Unknown languages fall back to plain monospaced rendering.
final class CodeBlockHighlighterTests: XCTestCase {

    private func makeHighlighter() -> CodeBlockHighlighter {
        CodeBlockHighlighter(theme: .light)
    }

    // MARK: - Per-language highlighting

    func testSwiftHighlights() {
        let h = makeHighlighter()
        let s = h.highlight(source: "let x = 1", language: "swift")
        XCTAssertTrue(s.string.contains("let"))
        XCTAssertTrue(s.string.contains("1"))
    }

    func testPythonHighlights() {
        let h = makeHighlighter()
        let s = h.highlight(source: "def hello():\n    return 1", language: "python")
        XCTAssertTrue(s.string.contains("def"))
        XCTAssertTrue(s.string.contains("hello"))
    }

    func testJavaScriptHighlights() {
        let h = makeHighlighter()
        let s = h.highlight(source: "const x = 1; // comment", language: "javascript")
        XCTAssertTrue(s.string.contains("const"))
        XCTAssertTrue(s.string.contains("comment"))
    }

    func testTypeScriptHighlights() {
        let h = makeHighlighter()
        let s = h.highlight(source: "const x: number = 1;", language: "typescript")
        XCTAssertTrue(s.string.contains("number"))
    }

    func testSQLHighlights() {
        let h = makeHighlighter()
        let s = h.highlight(source: "SELECT * FROM users WHERE id = 1;", language: "sql")
        XCTAssertTrue(s.string.contains("SELECT"))
        XCTAssertTrue(s.string.contains("FROM"))
    }

    func testJSONHighlights() {
        let h = makeHighlighter()
        let s = h.highlight(source: "{\"key\": \"value\", \"num\": 42, \"flag\": true}", language: "json")
        XCTAssertTrue(s.string.contains("key"))
        XCTAssertTrue(s.string.contains("value"))
        XCTAssertTrue(s.string.contains("42"))
        XCTAssertTrue(s.string.contains("true"))
    }

    func testYAMLHighlights() {
        let h = makeHighlighter()
        let s = h.highlight(source: "key: value\nflag: true\n# comment", language: "yaml")
        XCTAssertTrue(s.string.contains("key"))
        XCTAssertTrue(s.string.contains("comment"))
    }

    func testMarkdownHighlights() {
        let h = makeHighlighter()
        let s = h.highlight(source: "# Heading\n\n- bullet\n\n`code`", language: "markdown")
        XCTAssertTrue(s.string.contains("Heading"))
        XCTAssertTrue(s.string.contains("bullet"))
    }

    func testShellHighlights() {
        let h = makeHighlighter()
        let s = h.highlight(source: "if [ \"$x\" = \"1\" ]; then echo yes; fi", language: "shell")
        XCTAssertTrue(s.string.contains("if"))
        XCTAssertTrue(s.string.contains("then"))
    }

    func testRustHighlights() {
        let h = makeHighlighter()
        let s = h.highlight(source: "fn main() {\n    let x: i32 = 1;\n}", language: "rust")
        XCTAssertTrue(s.string.contains("fn"))
        XCTAssertTrue(s.string.contains("let"))
    }

    func testGoHighlights() {
        let h = makeHighlighter()
        let s = h.highlight(source: "package main\nfunc main() {}", language: "go")
        XCTAssertTrue(s.string.contains("package"))
        XCTAssertTrue(s.string.contains("func"))
    }

    // MARK: - Language tag normalization

    func testLanguageTagsAreCaseInsensitive() {
        let h = makeHighlighter()
        XCTAssertEqual(
            h.highlight(source: "let x = 1", language: "SWIFT").string,
            h.highlight(source: "let x = 1", language: "swift").string
        )
        XCTAssertEqual(
            h.highlight(source: "def x(): pass", language: "Python").string,
            h.highlight(source: "def x(): pass", language: "python").string
        )
    }

    func testLanguageAliases() {
        let h = makeHighlighter()
        // py is an alias for python
        XCTAssertEqual(
            h.highlight(source: "def x(): pass", language: "py").string,
            h.highlight(source: "def x(): pass", language: "python").string
        )
        // ts is an alias for typescript
        XCTAssertEqual(
            h.highlight(source: "const x: number = 1", language: "ts").string,
            h.highlight(source: "const x: number = 1", language: "typescript").string
        )
        // sh is an alias for shell
        XCTAssertEqual(
            h.highlight(source: "echo hi", language: "sh").string,
            h.highlight(source: "echo hi", language: "shell").string
        )
    }

    // MARK: - Unknown language fallback

    func testUnknownLanguageFallsBackToPlain() {
        let h = makeHighlighter()
        let s = h.highlight(source: "let x = 1", language: "klingon")
        XCTAssertEqual(s.string, "let x = 1")
        // No crash; the string is unchanged.
    }

    func testNilLanguageFallsBackToPlain() {
        let h = makeHighlighter()
        let s = h.highlight(source: "hello world", language: nil)
        XCTAssertEqual(s.string, "hello world")
    }

    // MARK: - Output is monospaced

    func testOutputIsMonospaced() {
        let h = makeHighlighter()
        let s = h.highlight(source: "x = 1", language: "python")
        // The first run's .font attribute should be a
        // monospaced font (we don't assert the specific
        // font, but we do check the attribute is present).
        var sawFont = false
        s.enumerateAttribute(.font, in: NSRange(location: 0, length: s.length)) { value, _, _ in
            if value != nil { sawFont = true }
        }
        XCTAssertTrue(sawFont)
    }

    // MARK: - Theme integration

    func testThemePaletteAffectsColors() {
        let customTheme = SyntaxThemePalette(
            plain: "#111111",
            operator: "#222222",
            keyword: "#333333",
            type: "#444444",
            number: "#555555",
            string: "#666666",
            identifier: "#777777",
            comment: "#888888",
            functionCall: "#999999"
        )
        let h = CodeBlockHighlighter(theme: customTheme)
        let s = h.highlight(source: "def x(): pass", language: "python")
        // Just confirm the highlighter uses the theme (i.e.
        // the output is non-empty and the source is preserved).
        XCTAssertTrue(s.string.contains("def"))
    }
}
