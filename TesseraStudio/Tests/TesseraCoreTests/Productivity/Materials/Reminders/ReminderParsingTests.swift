import XCTest
@testable import TesseraCore

/// Tests for the natural-language ``ReminderCommandParser``.
/// The parser is the bridge between the chat panel's free-
/// text user input and the structured agent tool calls;
/// the parsing is the load-bearing step.
final class ReminderParsingTests: XCTestCase {

    private let parser = ReminderCommandParser()

    // MARK: - Create intent

    func testCreateBeforeMinutes() {
        let cmd = parser.parse("remind me 15 min before the Q3 review meeting")
        XCTAssertNotNil(cmd)
        XCTAssertEqual(cmd?.kind, .create)
        XCTAssertEqual(cmd?.offsetMinutes, -15)
        // The leading "the" is stripped (the agent's
        // fuzzy-match step would treat it as noise).
        XCTAssertEqual(cmd?.eventTitleFragment, "Q3 review meeting")
    }

    func testCreateBeforeHours() {
        let cmd = parser.parse("remind me 2 hours before the all-hands")
        XCTAssertEqual(cmd?.kind, .create)
        XCTAssertEqual(cmd?.offsetMinutes, -120)
        XCTAssertEqual(cmd?.eventTitleFragment, "all-hands")
    }

    func testCreateAfter() {
        let cmd = parser.parse("remind me 30 min after the standup")
        XCTAssertEqual(cmd?.kind, .create)
        XCTAssertEqual(cmd?.offsetMinutes, 30)
        XCTAssertEqual(cmd?.eventTitleFragment, "standup")
    }

    func testCreateSingularHour() {
        let cmd = parser.parse("remind me 1 hour before the Q3 review")
        XCTAssertEqual(cmd?.kind, .create)
        XCTAssertEqual(cmd?.offsetMinutes, -60)
    }

    func testCreateWithReminderSuffixStripped() {
        // "remind me 5 min before the X reminder" — the
        // "the" prefix and the "reminder" suffix on the
        // title should be stripped.
        let cmd = parser.parse("remind me 5 min before the X reminder")
        XCTAssertEqual(cmd?.eventTitleFragment, "X")
    }

    func testCreateUsesToPrefix() {
        let cmd = parser.parse("remind me to 5 min before standup")
        XCTAssertEqual(cmd?.kind, .create)
        XCTAssertEqual(cmd?.offsetMinutes, -5)
    }

    // MARK: - List intent

    func testListIntent() {
        for phrase in [
            "list my reminders",
            "show my reminders",
            "what are my reminders",
            "reminders",
        ] {
            let cmd = parser.parse(phrase)
            XCTAssertNotNil(cmd, "expected list for: \(phrase)")
            XCTAssertEqual(cmd?.kind, .list, "expected list for: \(phrase)")
        }
    }

    // MARK: - Dismiss intent

    func testDismissIntent() {
        let cmd = parser.parse("dismiss the Q3 review reminder")
        XCTAssertEqual(cmd?.kind, .dismiss)
        XCTAssertEqual(cmd?.eventTitleFragment, "Q3 review")
    }

    func testDismissWithoutThePrefix() {
        let cmd = parser.parse("dismiss Q3 review reminder")
        XCTAssertEqual(cmd?.kind, .dismiss)
        XCTAssertEqual(cmd?.eventTitleFragment, "Q3 review")
    }

    func testAcknowledgeAlias() {
        let cmd = parser.parse("acknowledge the Q3 review reminder")
        XCTAssertEqual(cmd?.kind, .dismiss)
    }

    // MARK: - Snooze intent

    func testSnoozeIntent() {
        let cmd = parser.parse("snooze the Q3 review reminder for 10 min")
        XCTAssertEqual(cmd?.kind, .snooze)
        XCTAssertEqual(cmd?.eventTitleFragment, "Q3 review")
        XCTAssertEqual(cmd?.snoozeMinutes, 10)
    }

    func testSnoozeDefaultMinutes() {
        // "snooze the X reminder" without a duration
        // defaults to 10 min per the parser's heuristic.
        let cmd = parser.parse("snooze the Q3 review reminder")
        XCTAssertEqual(cmd?.kind, .snooze)
        XCTAssertEqual(cmd?.snoozeMinutes, 10)
    }

    func testSnoozeHours() {
        let cmd = parser.parse("snooze the Q3 review reminder for 1 hour")
        XCTAssertEqual(cmd?.kind, .snooze)
        XCTAssertEqual(cmd?.snoozeMinutes, 60)
    }

    // MARK: - Unmatched inputs

    func testUnmatchedReturnsNil() {
        XCTAssertNil(parser.parse(""))
        XCTAssertNil(parser.parse("hello there"))
        XCTAssertNil(parser.parse("remind"))
        XCTAssertNil(parser.parse("remind me"))
    }

    // MARK: - Raw input preservation

    func testRawInputPreserved() {
        let cmd = parser.parse("  remind me 15 min before the X  ")
        XCTAssertEqual(cmd?.rawInput, "remind me 15 min before the X")
    }
}
