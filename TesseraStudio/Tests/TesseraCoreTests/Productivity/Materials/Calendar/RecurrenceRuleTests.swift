import XCTest
@testable import TesseraCore

/// Tests for ``RecurrenceRule``: RRULE parsing (the
/// supported subset), loud rejection of everything
/// outside it, serialization round-trip, and occurrence
/// expansion. Expansion tests pin a fixed time zone so
/// they are deterministic regardless of where CI runs.
final class RecurrenceRuleTests: XCTestCase {

    // MARK: - Fixtures

    private let calendar: Calendar = {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "America/Los_Angeles")!
        c.firstWeekday = 1
        return c
    }()

    private func date(
        _ year: Int, _ month: Int, _ day: Int,
        _ hour: Int = 0, _ minute: Int = 0
    ) -> Date {
        calendar.date(from: DateComponents(
            year: year, month: month, day: day, hour: hour, minute: minute
        ))!
    }

    // MARK: - Parsing

    func testParseDaily() throws {
        let rule = try RecurrenceRule(rrule: "FREQ=DAILY")
        XCTAssertEqual(rule.frequency, .daily)
        XCTAssertEqual(rule.interval, 1)
        XCTAssertNil(rule.count)
        XCTAssertNil(rule.until)
        XCTAssertTrue(rule.byDay.isEmpty)
    }

    func testParseWeeklyWithMultipleByDay() throws {
        let rule = try RecurrenceRule(rrule: "FREQ=WEEKLY;BYDAY=MO,WE,FR")
        XCTAssertEqual(rule.frequency, .weekly)
        XCTAssertEqual(rule.byDay, [.monday, .wednesday, .friday])
    }

    func testParseIntervalCountUntil() throws {
        let rule = try RecurrenceRule(rrule: "FREQ=DAILY;INTERVAL=2;COUNT=5;UNTIL=20260131T090000Z")
        XCTAssertEqual(rule.interval, 2)
        XCTAssertEqual(rule.count, 5)
        XCTAssertNotNil(rule.until)
    }

    func testParseUntilDateOnly() throws {
        let rule = try RecurrenceRule(rrule: "FREQ=WEEKLY;UNTIL=20260131")
        XCTAssertNotNil(rule.until)
        // Date-only UNTIL is local midnight.
        XCTAssertEqual(rule.until, date(2026, 1, 31))
    }

    func testParseMonthlyByMonthDay() throws {
        let rule = try RecurrenceRule(rrule: "FREQ=MONTHLY;BYMONTHDAY=15")
        XCTAssertEqual(rule.frequency, .monthly)
        XCTAssertEqual(rule.byMonthDay, [15])
    }

    func testParseYearlyByMonth() throws {
        let rule = try RecurrenceRule(rrule: "FREQ=YEARLY;BYMONTH=3")
        XCTAssertEqual(rule.frequency, .yearly)
        XCTAssertEqual(rule.byMonth, [3])
    }

    func testParseAcceptsRRULEPrefixAndWKST() throws {
        let rule = try RecurrenceRule(rrule: "RRULE:FREQ=DAILY;WKST=SU")
        XCTAssertEqual(rule.frequency, .daily)
    }

    func testParseIsCaseTolerant() throws {
        let rule = try RecurrenceRule(rrule: "freq=weekly;byday=mo")
        XCTAssertEqual(rule.frequency, .weekly)
        XCTAssertEqual(rule.byDay, [.monday])
    }

    // MARK: - Parse errors

    func testParseEmptyThrows() {
        XCTAssertThrowsError(try RecurrenceRule(rrule: "")) { error in
            XCTAssertEqual(error as? RecurrenceRule.ParseError, .empty)
        }
        XCTAssertThrowsError(try RecurrenceRule(rrule: "   "))
    }

    func testParseMissingFrequencyThrows() {
        XCTAssertThrowsError(try RecurrenceRule(rrule: "INTERVAL=2")) { error in
            XCTAssertEqual(error as? RecurrenceRule.ParseError, .missingFrequency)
        }
    }

    func testParseUnknownFrequencyThrows() {
        XCTAssertThrowsError(try RecurrenceRule(rrule: "FREQ=HOURLY")) { error in
            XCTAssertEqual(error as? RecurrenceRule.ParseError, .unknownFrequency("HOURLY"))
        }
    }

    func testParseUnknownPartThrows() {
        XCTAssertThrowsError(try RecurrenceRule(rrule: "FREQ=DAILY;BYSETPOS=1")) { error in
            guard case .unknownPart = error as? RecurrenceRule.ParseError else {
                return XCTFail("expected unknownPart, got \(error)")
            }
        }
    }

    func testParseOrdinalWeekdayThrows() {
        // "1MO" / "-1FR" ordinals are out of scope for v1;
        // reject loudly instead of half-interpreting.
        XCTAssertThrowsError(try RecurrenceRule(rrule: "FREQ=MONTHLY;BYDAY=1MO")) { error in
            guard case .unsupportedOrdinal = error as? RecurrenceRule.ParseError else {
                return XCTFail("expected unsupportedOrdinal, got \(error)")
            }
        }
        XCTAssertThrowsError(try RecurrenceRule(rrule: "FREQ=MONTHLY;BYDAY=-1FR"))
    }

    func testParseNegativeMonthDayThrows() {
        XCTAssertThrowsError(try RecurrenceRule(rrule: "FREQ=MONTHLY;BYMONTHDAY=-1")) { error in
            guard case .invalidValue = error as? RecurrenceRule.ParseError else {
                return XCTFail("expected invalidValue, got \(error)")
            }
        }
    }

    func testParseInvalidNumbersThrow() {
        XCTAssertThrowsError(try RecurrenceRule(rrule: "FREQ=DAILY;COUNT=0"))
        XCTAssertThrowsError(try RecurrenceRule(rrule: "FREQ=DAILY;INTERVAL=0"))
        XCTAssertThrowsError(try RecurrenceRule(rrule: "FREQ=YEARLY;BYMONTH=13"))
        XCTAssertThrowsError(try RecurrenceRule(rrule: "FREQ=DAILY;UNTIL=notadate"))
    }

    // MARK: - Serialization

    func testRRuleStringRoundTrip() throws {
        let rules = [
            "FREQ=DAILY",
            "FREQ=WEEKLY;BYDAY=MO",
            "FREQ=WEEKLY;INTERVAL=2;BYDAY=MO,WE,FR",
            "FREQ=MONTHLY;BYMONTHDAY=15",
            "FREQ=YEARLY;BYMONTH=3",
            "FREQ=DAILY;COUNT=5",
        ]
        for rrule in rules {
            let rule = try RecurrenceRule(rrule: rrule)
            XCTAssertEqual(rule.rruleString, rrule)
            let reparsed = try RecurrenceRule(rrule: rule.rruleString)
            XCTAssertEqual(reparsed, rule)
        }
    }

    func testWeekdayCalendarMapping() {
        XCTAssertEqual(RecurrenceRule.Weekday.monday.calendarWeekday, 2)
        XCTAssertEqual(RecurrenceRule.Weekday.sunday.calendarWeekday, 1)
        XCTAssertEqual(RecurrenceRule.Weekday.from(calendarWeekday: 7), .saturday)
    }

    // MARK: - Occurrence expansion

    func testWeeklyByDayExpansion() {
        // Anchor Wednesday 2026-01-07; rule lands on
        // Mondays. January 2026 Mondays after the anchor:
        // 12th, 19th, 26th.
        let rule = try! RecurrenceRule(rrule: "FREQ=WEEKLY;BYDAY=MO")
        let anchor = date(2026, 1, 7, 10, 30)
        let range = date(2026, 1, 1)...date(2026, 2, 1)
        let occurrences = rule.occurrences(anchor: anchor, in: range, calendar: calendar)
        XCTAssertEqual(occurrences, [date(2026, 1, 12, 10, 30), date(2026, 1, 19, 10, 30), date(2026, 1, 26, 10, 30)])
    }

    func testWeeklyDefaultsToAnchorWeekday() {
        let rule = try! RecurrenceRule(rrule: "FREQ=WEEKLY")
        let anchor = date(2026, 1, 7, 9, 0) // Wednesday
        let range = date(2026, 1, 7)...date(2026, 1, 29)
        let occurrences = rule.occurrences(anchor: anchor, in: range, calendar: calendar)
        XCTAssertEqual(occurrences, [
            date(2026, 1, 7, 9, 0), date(2026, 1, 14, 9, 0),
            date(2026, 1, 21, 9, 0), date(2026, 1, 28, 9, 0),
        ])
    }

    func testCountLimitsOccurrences() {
        let rule = try! RecurrenceRule(rrule: "FREQ=DAILY;COUNT=3")
        let anchor = date(2026, 1, 1, 8, 0)
        let range = date(2026, 1, 1)...date(2026, 12, 31)
        let occurrences = rule.occurrences(anchor: anchor, in: range, calendar: calendar)
        XCTAssertEqual(occurrences.count, 3)
        XCTAssertEqual(occurrences.first, anchor)
    }

    func testUntilLimitsOccurrences() {
        let rule = try! RecurrenceRule(rrule: "FREQ=DAILY;UNTIL=20260105T080000")
        let anchor = date(2026, 1, 1, 8, 0)
        let range = date(2026, 1, 1)...date(2026, 12, 31)
        let occurrences = rule.occurrences(anchor: anchor, in: range, calendar: calendar)
        XCTAssertEqual(occurrences.count, 5)
        XCTAssertEqual(occurrences.last, date(2026, 1, 5, 8, 0))
    }

    func testBiweeklyIntervalAlignsWithCalendarWeeks() {
        // Anchor Wednesday 2026-01-07, BYDAY=MO,FR,
        // INTERVAL=2. The anchor's week (week 0) emits
        // Friday Jan 9; week 1 is skipped entirely; week
        // 2 emits Monday Jan 19 + Friday Jan 23. This
        // pins the "weeks counted between week starts"
        // semantics - a naive day-count interval would
        // drift when BYDAY names a day before the anchor.
        let rule = try! RecurrenceRule(rrule: "FREQ=WEEKLY;INTERVAL=2;BYDAY=MO,FR")
        let anchor = date(2026, 1, 7, 15, 0)
        let range = date(2026, 1, 7)...date(2026, 1, 31)
        let occurrences = rule.occurrences(anchor: anchor, in: range, calendar: calendar)
        XCTAssertEqual(occurrences, [
            date(2026, 1, 9, 15, 0),
            date(2026, 1, 19, 15, 0),
            date(2026, 1, 23, 15, 0),
        ])
    }

    func testMonthlyByMonthDayFromLateAnchor() {
        // Anchor Jan 31 with BYMONTHDAY=15: the first
        // occurrence is Feb 15 (the expander jumps to the
        // next candidate month instead of walking 31 days).
        let rule = try! RecurrenceRule(rrule: "FREQ=MONTHLY;BYMONTHDAY=15")
        let anchor = date(2026, 1, 31, 10, 0)
        let range = date(2026, 1, 31)...date(2026, 4, 30)
        let occurrences = rule.occurrences(anchor: anchor, in: range, calendar: calendar)
        XCTAssertEqual(occurrences, [
            date(2026, 2, 15, 10, 0), date(2026, 3, 15, 10, 0), date(2026, 4, 15, 10, 0),
        ])
    }

    func testMonthlyDefaultsToAnchorDay() {
        let rule = try! RecurrenceRule(rrule: "FREQ=MONTHLY")
        let anchor = date(2026, 1, 3, 12, 0)
        let range = date(2026, 1, 1)...date(2026, 4, 30)
        let occurrences = rule.occurrences(anchor: anchor, in: range, calendar: calendar)
        XCTAssertEqual(occurrences, [
            date(2026, 1, 3, 12, 0), date(2026, 2, 3, 12, 0),
            date(2026, 3, 3, 12, 0), date(2026, 4, 3, 12, 0),
        ])
    }

    func testYearlyExpansion() {
        let rule = try! RecurrenceRule(rrule: "FREQ=YEARLY")
        let anchor = date(2025, 3, 14, 9, 0)
        let range = date(2025, 1, 1)...date(2028, 12, 31)
        let occurrences = rule.occurrences(anchor: anchor, in: range, calendar: calendar)
        XCTAssertEqual(occurrences, [
            date(2025, 3, 14, 9, 0), date(2026, 3, 14, 9, 0),
            date(2027, 3, 14, 9, 0), date(2028, 3, 14, 9, 0),
        ])
    }

    func testExpansionPreservesTimeOfDayAcrossDST() {
        // US DST starts 2026-03-08: a daily 9:30 event
        // keeps wall-clock 9:30 on both sides (the
        // calendar, not raw seconds, does the day math).
        // The range runs past the 9th's 9:30 so all three
        // occurrences land inside it.
        let rule = try! RecurrenceRule(rrule: "FREQ=DAILY")
        let anchor = date(2026, 3, 7, 9, 30)
        let range = date(2026, 3, 7)...date(2026, 3, 9, 23, 0)
        let occurrences = rule.occurrences(anchor: anchor, in: range, calendar: calendar)
        XCTAssertEqual(occurrences.count, 3)
        for occurrence in occurrences {
            XCTAssertEqual(calendar.component(.hour, from: occurrence), 9)
            XCTAssertEqual(calendar.component(.minute, from: occurrence), 30)
        }
    }

    func testUnboundedDailyRuleIsCapped() {
        let rule = try! RecurrenceRule(rrule: "FREQ=DAILY")
        let anchor = date(2026, 1, 1)
        let range = anchor...date(2030, 12, 31)
        let occurrences = rule.occurrences(anchor: anchor, in: range, calendar: calendar)
        XCTAssertEqual(occurrences.count, RecurrenceRule.maxOccurrences)
    }

    func testRangeBoundsAreInclusive() {
        let rule = try! RecurrenceRule(rrule: "FREQ=DAILY")
        let anchor = date(2026, 1, 1, 10, 0)
        // Range ends exactly at the occurrence instant.
        let occurrences = rule.occurrences(
            anchor: anchor,
            in: anchor...date(2026, 1, 2, 10, 0),
            calendar: calendar
        )
        XCTAssertEqual(occurrences.count, 2)
    }
}
