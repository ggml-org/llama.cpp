import XCTest
@testable import TesseraCore

/// Tests for the export format enum and the Slack mrkdwn
/// formatter.
final class ExportFormatTests: XCTestCase {

    /// Every ``ProductivityExportFormat`` case has a non-empty
    /// display name and a file extension matching the raw
    /// value.
    func testFormatMetadata() {
        for f in ProductivityExportFormat.allCases {
            XCTAssertFalse(
                f.displayName.isEmpty,
                "\(f.rawValue) must have a display name"
            )
            XCTAssertEqual(
                f.fileExtension, f.rawValue,
                "\(f.rawValue) fileExtension should match rawValue"
            )
        }
    }

    /// Slack mrkdwn conversion: bold and links are the most
    /// user-visible transformations.
    func testSlackMrkdwnBold() {
        let out = SlackMrkdwnFormatter.format("**bold text**")
        XCTAssertEqual(out, "*bold text*", "**bold** should become *bold* in mrkdwn")
    }

    func testSlackMrkdwnLink() {
        let out = SlackMrkdwnFormatter.format("[click here](https://example.com)")
        XCTAssertEqual(
            out, "<https://example.com|click here>",
            "[text](url) should become <url|text> in mrkdwn"
        )
    }

    func testSlackMrkdwnHeading() {
        let out = SlackMrkdwnFormatter.format("# Heading 1")
        XCTAssertEqual(out, "*Heading 1*", "# heading should become *bold* in mrkdwn")
    }

    func testSlackMrkdwnBullet() {
        let out = SlackMrkdwnFormatter.format("- one\n- two")
        XCTAssertTrue(
            out.contains("• one") && out.contains("• two"),
            "bullets should be replaced with the bullet char; got \(out)"
        )
    }

    func testSlackMrkdwnStrikethrough() {
        let out = SlackMrkdwnFormatter.format("~~gone~~")
        XCTAssertEqual(out, "~gone~", "~~strike~~ should become ~strike~ in mrkdwn")
    }

    /// Round-trip: a small Markdown doc with mixed
    /// constructs converts cleanly to mrkdwn.
    func testSlackMrkdwnRoundTrip() {
        let md = """
        # Title

        A paragraph with **bold** and *italic* and a
        [link](https://example.com).

        - bullet one
        - bullet two
        """
        let out = SlackMrkdwnFormatter.format(md)
        XCTAssertTrue(out.contains("*Title*"), "heading promoted to bold")
        XCTAssertTrue(out.contains("*bold*"), "bold converted")
        XCTAssertTrue(out.contains("_italic_"), "italic converted to underscores")
        XCTAssertTrue(out.contains("<https://example.com|link>"), "link converted")
        XCTAssertTrue(out.contains("• bullet"), "bullet converted")
    }
}
