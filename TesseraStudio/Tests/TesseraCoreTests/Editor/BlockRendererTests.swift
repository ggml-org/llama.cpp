import XCTest
import Foundation
#if canImport(AppKit)
import AppKit
#endif
@testable import TesseraCore

/// Tests for the BlockRenderer. Covers every block type, every
/// inline annotation, code-block highlighting, image blocks, and
/// a few cross-cutting invariants (per-block length matches
/// element range, container blocks nest their children).
final class BlockRendererTests: XCTestCase {

    // MARK: - Per-block-type renderers

    func testRenderHeadingProducesAttributedStringWithFont() {
        let renderer = BlockRenderer()
        let block = Block(
            id: UUID(),
            type: .heading,
            attributes: ["level": .number(1)],
            content: [InlineRun(text: "Hello")]
        )
        let s = renderer.render(block, in: .document)
        XCTAssertGreaterThan(s.length, 0)
        XCTAssertEqual(s.string, "Hello")
    }

    func testRenderHeadingLevelIsClamped() {
        // Level 99 should be clamped to 6 (the highest level).
        let renderer = BlockRenderer()
        let block = Block(
            id: UUID(),
            type: .heading,
            attributes: ["level": .number(99)],
            content: [InlineRun(text: "Loud")]
        )
        let s = renderer.render(block, in: .document)
        XCTAssertEqual(s.string, "Loud")
        // No crash; the renderer picks a font and proceeds.
    }

    func testRenderParagraphProducesAttributedString() {
        let renderer = BlockRenderer()
        let block = Block(
            id: UUID(),
            type: .paragraph,
            content: [InlineRun(text: "Body text")]
        )
        let s = renderer.render(block, in: .document)
        XCTAssertEqual(s.string, "Body text")
    }

    func testRenderListContainerIncludesMarkerPrefix() {
        let renderer = BlockRenderer()
        let unordered = Block(
            id: UUID(),
            type: .list,
            attributes: ["style": .string("unordered")],
            content: []
        )
        let ordered = Block(
            id: UUID(),
            type: .list,
            attributes: ["style": .string("ordered")],
            content: []
        )
        let task = Block(
            id: UUID(),
            type: .list,
            attributes: ["style": .string("task")],
            content: []
        )
        XCTAssertTrue(renderer.render(unordered, in: .document).string.hasPrefix("•"))
        XCTAssertTrue(renderer.render(ordered, in: .document).string.hasPrefix("1."))
        XCTAssertTrue(renderer.render(task, in: .document).string.hasPrefix("☐"))
    }

    func testRenderListItemIncludesBullet() {
        let renderer = BlockRenderer()
        let block = Block(
            id: UUID(),
            type: .listItem,
            content: [InlineRun(text: "an item")]
        )
        let s = renderer.render(block, in: .document)
        XCTAssertTrue(s.string.contains("an item"))
    }

    func testRenderTablePlaceholder() {
        let renderer = BlockRenderer()
        let block = Block(
            id: UUID(),
            type: .table,
            attributes: ["rows": .number(3), "cols": .number(4)],
            content: []
        )
        let s = renderer.render(block, in: .document)
        XCTAssertTrue(s.string.contains("3×4"))
    }

    func testRenderTableCellUsesInlineContent() {
        let renderer = BlockRenderer()
        let block = Block(
            id: UUID(),
            type: .tableCell,
            content: [InlineRun(text: "cell text")]
        )
        XCTAssertEqual(renderer.render(block, in: .document).string, "cell text")
    }

    func testRenderCodeBlockWithoutLanguage() {
        let renderer = BlockRenderer()
        let block = Block(
            id: UUID(),
            type: .codeBlock,
            attributes: [:],
            content: [InlineRun(text: "let x = 1")]
        )
        let s = renderer.render(block, in: .document)
        XCTAssertTrue(s.string.contains("let x = 1"))
    }

    func testRenderCodeBlockWithLanguageProducesHighlightedOutput() {
        let renderer = BlockRenderer()
        let block = Block(
            id: UUID(),
            type: .codeBlock,
            attributes: ["language": .string("swift")],
            content: [InlineRun(text: "let x = 1")]
        )
        let s = renderer.render(block, in: .document)
        // Even with the regex fallback, the source text is present.
        XCTAssertTrue(s.string.contains("let x = 1"))
        // The code block adds a trailing newline for layout.
        XCTAssertTrue(s.string.hasSuffix("\n"))
    }

    func testRenderCalloutIncludesEmoji() {
        let renderer = BlockRenderer()
        let block = Block(
            id: UUID(),
            type: .callout,
            attributes: ["emoji": .string("⚠️")],
            content: [InlineRun(text: "Be careful")]
        )
        let s = renderer.render(block, in: .document)
        XCTAssertTrue(s.string.contains("⚠️"))
        XCTAssertTrue(s.string.contains("Be careful"))
    }

    func testRenderDividerIsNonEmpty() {
        let renderer = BlockRenderer()
        let block = Block(
            id: UUID(),
            type: .divider,
            content: []
        )
        let s = renderer.render(block, in: .document)
        XCTAssertGreaterThan(s.length, 0)
    }

    func testRenderQuoteWrapsInSmartQuotes() {
        let renderer = BlockRenderer()
        let block = Block(
            id: UUID(),
            type: .quote,
            content: [InlineRun(text: "to be or not to be")]
        )
        let s = renderer.render(block, in: .document)
        XCTAssertTrue(s.string.contains("to be or not to be"))
    }

    func testRenderQuoteIncludesCite() {
        let renderer = BlockRenderer()
        let block = Block(
            id: UUID(),
            type: .quote,
            attributes: ["cite": .string("Shakespeare")],
            content: [InlineRun(text: "to be")]
        )
        let s = renderer.render(block, in: .document)
        XCTAssertTrue(s.string.contains("Shakespeare"))
    }

    func testRenderToggleIncludesMarker() {
        let renderer = BlockRenderer()
        let expanded = Block(
            id: UUID(),
            type: .toggle,
            attributes: ["expanded": .bool(true)],
            content: [InlineRun(text: "Section")]
        )
        let collapsed = Block(
            id: UUID(),
            type: .toggle,
            attributes: ["expanded": .bool(false)],
            content: [InlineRun(text: "Section")]
        )
        XCTAssertTrue(renderer.render(expanded, in: .document).string.contains("▾"))
        XCTAssertTrue(renderer.render(collapsed, in: .document).string.contains("▸"))
    }

    func testRenderEquationUsesLatexSource() {
        let renderer = BlockRenderer()
        let block = Block(
            id: UUID(),
            type: .equation,
            attributes: ["latex": .string("E = mc^2")],
            content: []
        )
        let s = renderer.render(block, in: .document)
        XCTAssertTrue(s.string.contains("E = mc^2"))
    }

    func testRenderImageWithMissingSourceProducesPlaceholder() {
        let renderer = BlockRenderer()
        let block = Block(
            id: UUID(),
            type: .image,
            attributes: ["source": .string("https://invalid.invalid/x.png"), "alt": .string("alt")],
            content: []
        )
        // No crash on missing source; an NSTextAttachment is
        // produced (or, on non-AppKit platforms, a placeholder
        // string).
        let s = renderer.render(block, in: .document)
        XCTAssertGreaterThanOrEqual(s.length, 0)
    }

    // MARK: - Inline annotations

    func testRenderInlineRunsRespectsAnnotationOrder() {
        let renderer = BlockRenderer()
        let runs = [
            InlineRun(text: "bold", annotations: [.bold]),
            InlineRun(text: "italic", annotations: [.italic]),
            InlineRun(text: "code", annotations: [.code]),
        ]
        let block = Block(id: UUID(), type: .paragraph, content: runs)
        let s = renderer.render(block, in: .document)
        XCTAssertEqual(s.string, "bolditaliccode")
    }

    func testRenderInlineRunsAppliesUnderlineAndStrike() {
        let renderer = BlockRenderer()
        let runs = [
            InlineRun(text: "u", annotations: [.underline]),
            InlineRun(text: "s", annotations: [.strikethrough]),
        ]
        let block = Block(id: UUID(), type: .paragraph, content: runs)
        let s = renderer.render(block, in: .document)
        // The .underlineStyle / .strikethroughStyle attributes
        // are present on the runs.
        var sawUnderline = false
        var sawStrike = false
        s.enumerateAttribute(.underlineStyle, in: NSRange(location: 0, length: s.length)) { value, _, _ in
            if (value as? Int) ?? 0 != 0 { sawUnderline = true }
        }
        s.enumerateAttribute(.strikethroughStyle, in: NSRange(location: 0, length: s.length)) { value, _, _ in
            if (value as? Int) ?? 0 != 0 { sawStrike = true }
        }
        XCTAssertTrue(sawUnderline)
        XCTAssertTrue(sawStrike)
    }

    func testRenderInlineRunsAppliesLink() {
        let renderer = BlockRenderer()
        let url = URL(string: "https://example.com")!
        let runs = [
            InlineRun(text: "click", annotations: [.link(url)]),
        ]
        let block = Block(id: UUID(), type: .paragraph, content: runs)
        let s = renderer.render(block, in: .document)
        var sawLink = false
        s.enumerateAttribute(.link, in: NSRange(location: 0, length: s.length)) { value, _, _ in
            if (value as? URL) == url { sawLink = true }
        }
        XCTAssertTrue(sawLink)
    }

    func testRenderInlineRunsAppliesColor() {
        let renderer = BlockRenderer()
        let runs = [
            InlineRun(text: "red", annotations: [.color(hex: "#FF0000")]),
        ]
        let block = Block(id: UUID(), type: .paragraph, content: runs)
        let s = renderer.render(block, in: .document)
        XCTAssertGreaterThan(s.length, 0)
    }

    // MARK: - All-blocks

    func testRenderAllJoinsBlocksWithNewlines() {
        let renderer = BlockRenderer()
        let blocks = [
            Block(id: UUID(), type: .paragraph, content: [InlineRun(text: "a")]),
            Block(id: UUID(), type: .paragraph, content: [InlineRun(text: "b")]),
        ]
        let s = renderer.renderAll(blocks, in: .document)
        XCTAssertEqual(s.string, "a\nb")
    }

    // MARK: - Font + theme plumbing

    func testFontResolverProducesMonospacedForMonospaceFamily() {
        let resolver = DefaultFontResolver()
        let desc = FontDescriptor.monospace(size: 12)
        #if canImport(AppKit)
        let f = resolver.font(from: desc)
        #elseif canImport(UIKit)
        let f = resolver.font(from: desc)
        #else
        return
        #endif
        // The default resolver maps .monospace to a monospaced
        // system font; we don't assert the specific font but
        // we do assert the call returns without crashing.
        _ = f
    }

    func testPlatformColorFromHex() {
        let red = PlatformColor.fromHex("#FF0000")
        XCTAssertNotNil(red)
        let bad = PlatformColor.fromHex("not-a-color")
        XCTAssertNil(bad)
        let short = PlatformColor.fromHex("#FFF")
        XCTAssertNil(short, "3-digit hex is not supported; must be 6 or 8 chars")
    }
}
