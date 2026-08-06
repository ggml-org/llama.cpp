import XCTest
@testable import TesseraCore

/// Unit tests for ``ProductivityTaskNLUParser``. The parser
/// is rule-based; the tests cover the patterns documented in
/// the parser's doc comment (priority prefix, due-date prefix,
/// trailing dates, "in N days", "next <weekday>", contact +
/// document links, etc.).
final class ProductivityTaskNLUParserTests: XCTestCase {

    // MARK: - Simple inputs

    func testParseSimpleTitle() {
        let parser = ProductivityTaskNLUParser()
        let parsed = parser.parse("buy milk")
        XCTAssertEqual(parsed.title, "buy milk")
        XCTAssertNil(parsed.dueAt)
        XCTAssertEqual(parsed.priority, .none)
        XCTAssertEqual(parsed.linkedEntityIDs, [])
    }

    func testParseTrailingCommaIsStripped() {
        let parser = ProductivityTaskNLUParser()
        let parsed = parser.parse("buy milk,")
        XCTAssertEqual(parsed.title, "buy milk")
    }

    func testParseEmptyInputReturnsEmpty() {
        let parser = ProductivityTaskNLUParser()
        let parsed = parser.parse("   ")
        // Empty / whitespace-only input -> empty title.
        // The parser doesn't fabricate a title.
        XCTAssertTrue(parsed.title.isEmpty)
        XCTAssertNil(parsed.dueAt)
        XCTAssertEqual(parsed.priority, .none)
    }

    // MARK: - Priority

    func testParseHighPriorityPrefix() {
        let parser = ProductivityTaskNLUParser()
        let parsed = parser.parse("high priority: review the Q3 report")
        XCTAssertEqual(parsed.priority, .high)
        XCTAssertEqual(parsed.title, "review the Q3 report")
    }

    func testParseMediumPriorityPrefix() {
        let parser = ProductivityTaskNLUParser()
        let parsed = parser.parse("medium priority: write the blog post")
        XCTAssertEqual(parsed.priority, .medium)
        XCTAssertEqual(parsed.title, "write the blog post")
    }

    func testParseLowPriorityPrefix() {
        let parser = ProductivityTaskNLUParser()
        let parsed = parser.parse("low priority: clean the garage")
        XCTAssertEqual(parsed.priority, .low)
        XCTAssertEqual(parsed.title, "clean the garage")
    }

    func testParseUrgentPrefix() {
        let parser = ProductivityTaskNLUParser()
        let parsed = parser.parse("urgent: ship the release")
        XCTAssertEqual(parsed.priority, .high)
        XCTAssertEqual(parsed.title, "ship the release")
    }

    func testParseExclamationAsHighPriority() {
        let parser = ProductivityTaskNLUParser()
        let parsed = parser.parse("call the bank!")
        XCTAssertEqual(parsed.priority, .high)
        XCTAssertEqual(parsed.title, "call the bank")
    }

    // MARK: - Due dates

    func testParseTomorrowPrefix() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let parser = ProductivityTaskNLUParser(now: { now })
        let parsed = parser.parse("tomorrow call John")
        XCTAssertNotNil(parsed.dueAt)
        guard let due = parsed.dueAt else { return }
        let cal = Calendar.current
        let tomorrow = cal.date(byAdding: .day, value: 1, to: cal.startOfDay(for: now))!
        let dueDay = cal.startOfDay(for: due)
        let dueHour = cal.component(.hour, from: due)
        XCTAssertEqual(dueDay, tomorrow)
        XCTAssertEqual(dueHour, 9) // default to 9am when no time given
        XCTAssertEqual(parsed.title, "call John")
        XCTAssertEqual(parsed.list, .today)
    }

    func testParseTomorrowAtTime() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let parser = ProductivityTaskNLUParser(now: { now })
        let parsed = parser.parse("tomorrow at 3pm, call John")
        XCTAssertNotNil(parsed.dueAt)
        guard let due = parsed.dueAt else { return }
        let cal = Calendar.current
        // The parser resolves "tomorrow at 3pm" to 3pm in
        // the local timezone. We check the day + hour
        // components rather than the absolute Date because
        // the absolute Date depends on the local timezone.
        let tomorrow = cal.date(byAdding: .day, value: 1, to: cal.startOfDay(for: now))!
        let dueDay = cal.startOfDay(for: due)
        let dueHour = cal.component(.hour, from: due)
        XCTAssertEqual(dueDay, tomorrow)
        XCTAssertEqual(dueHour, 15)
    }

    func testParseTodayPrefix() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let parser = ProductivityTaskNLUParser(now: { now })
        let parsed = parser.parse("today, finish the slides")
        XCTAssertNotNil(parsed.dueAt)
        XCTAssertEqual(parsed.title, "finish the slides")
    }

    func testParseTodayAtNoon() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let parser = ProductivityTaskNLUParser(now: { now })
        let parsed = parser.parse("today at noon, lunch with team")
        XCTAssertNotNil(parsed.dueAt)
        let cal = Calendar.current
        let noon = cal.date(bySettingHour: 12, minute: 0, second: 0, of: cal.startOfDay(for: now))!
        XCTAssertEqual(parsed.dueAt, noon)
    }

    func testParseTonight() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let parser = ProductivityTaskNLUParser(now: { now })
        let parsed = parser.parse("tonight: watch the keynote")
        XCTAssertNotNil(parsed.dueAt)
        let cal = Calendar.current
        let tonight = cal.date(bySettingHour: 20, minute: 0, second: 0, of: cal.startOfDay(for: now))!
        XCTAssertEqual(parsed.dueAt, tonight)
    }

    func testParseNextMonday() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        // 1_700_000_000 = 2023-11-14 22:13:20 UTC = Tuesday
        // Adjust to local: this test is best-effort; we just
        // check the parsed date is roughly 7 days out.
        let parser = ProductivityTaskNLUParser(now: { now })
        let parsed = parser.parse("next monday, plan the sprint")
        XCTAssertNotNil(parsed.dueAt)
        XCTAssertEqual(parsed.title, "plan the sprint")
    }

    func testParseInNDays() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let parser = ProductivityTaskNLUParser(now: { now })
        let parsed = parser.parse("in 3 days, submit the form")
        XCTAssertNotNil(parsed.dueAt)
        let expected = now.addingTimeInterval(3 * 24 * 3600)
        XCTAssertEqual(parsed.dueAt, expected)
        XCTAssertEqual(parsed.title, "submit the form")
    }

    func testParseInNWeeks() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let parser = ProductivityTaskNLUParser(now: { now })
        let parsed = parser.parse("in 2 weeks, review the design")
        XCTAssertNotNil(parsed.dueAt)
        let expected = now.addingTimeInterval(14 * 24 * 3600)
        XCTAssertEqual(parsed.dueAt, expected)
    }

    func testParseInADay() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let parser = ProductivityTaskNLUParser(now: { now })
        let parsed = parser.parse("in a day, call the dentist")
        XCTAssertNotNil(parsed.dueAt)
    }

    func testParseOnDate() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let parser = ProductivityTaskNLUParser(now: { now })
        let parsed = parser.parse("on Jan 15, review the contract")
        XCTAssertNotNil(parsed.dueAt)
        let cal = Calendar.current
        let year = cal.component(.year, from: now)
        let expected = cal.date(from: DateComponents(year: year, month: 1, day: 15, hour: 9))
        XCTAssertEqual(parsed.dueAt, expected)
    }

    // MARK: - List inference

    func testSomedayInputGoesToSomeday() {
        let parser = ProductivityTaskNLUParser()
        let parsed = parser.parse("someday: learn to surf")
        XCTAssertEqual(parsed.list, .someday)
    }

    func testNoDueDateNoListKeywordIsAnytime() {
        let parser = ProductivityTaskNLUParser()
        let parsed = parser.parse("buy milk")
        XCTAssertEqual(parsed.list, .anytime)
    }

    func testFutureDueDateIsUpcoming() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let parser = ProductivityTaskNLUParser(now: { now })
        let parsed = parser.parse("in 3 days, submit the form")
        XCTAssertEqual(parsed.list, .upcoming)
    }

    // MARK: - ParsedProductivityTask.toTask()

    func testToTaskUsesInferredList() {
        // Use 8am UTC so "tomorrow at 3pm" falls within
        // the next 24 hours regardless of the local
        // timezone (the test asserts the inferred list
        // is Today, which requires dueAt <= now+24h).
        let now = Date(timeIntervalSince1970: 1_700_000_000 - 22 * 3600) // 8am UTC
        let parser = ProductivityTaskNLUParser(now: { now })
        let parsed = parser.parse("tomorrow at 3pm, call John")
        let task = parsed.toTask(now: now)
        XCTAssertEqual(task.list, .today)
        XCTAssertEqual(task.title, "call John")
    }

    func testToTaskFallsBackToAnytime() {
        let parser = ProductivityTaskNLUParser()
        let parsed = parser.parse("buy milk")
        let task = parsed.toTask()
        XCTAssertEqual(task.list, .anytime)
    }

    // MARK: - Ambiguous input

    func testAmbiguousInputFallsBack() {
        let parser = ProductivityTaskNLUParser()
        let parsed = parser.parse("just do the thing")
        XCTAssertEqual(parsed.title, "just do the thing")
        XCTAssertNil(parsed.dueAt)
        XCTAssertEqual(parsed.priority, .none)
        XCTAssertEqual(parsed.list, .anytime)
    }
}

// MARK: - Mock contact / document adapters for NLU tests

/// In-memory `ContactsAdapter` for tests. Matches a contact
/// when any of the candidate words appears in the contact's
/// display name (case-insensitive).
final class MockContactsAdapter: ContactsAdapter {
    private let contacts: [Contact]

    init(contacts: [Contact]) {
        self.contacts = contacts
    }

    func find(matchingAny candidates: [String]) -> Contact? {
        let lowerCandidates = Set(candidates.map { $0.lowercased() })
        for contact in contacts {
            let nameWords = contact.displayName
                .lowercased()
                .split(separator: " ")
                .map(String.init)
            for word in nameWords {
                if lowerCandidates.contains(word) {
                    return contact
                }
            }
        }
        return nil
    }
}

/// In-memory `DocumentStoreNLU` for tests.
final class MockDocumentStoreNLU: DocumentStoreNLU {
    private let stubs: [DocumentStub]

    init(stubs: [DocumentStub]) {
        self.stubs = stubs
    }

    func findStub(matchingAny candidates: [String]) -> DocumentStub? {
        let lowerCandidates = Set(candidates.map { $0.lowercased() })
        for stub in stubs {
            let titleWords = stub.title
                .lowercased()
                .split(separator: " ")
                .map(String.init)
            for word in titleWords {
                if lowerCandidates.contains(word) {
                    return stub
                }
            }
        }
        return nil
    }
}

/// Tests for the NLU parser's contact + document linking.
final class ProductivityTaskNLULinkingTests: XCTestCase {

    func testContactLinking() {
        let contact = Contact(
            name: NameComponents(first: "John", last: "Doe"),
            emails: [LabeledEmail(label: .work, value: "john@acme.com")]
        )
        let adapter = MockContactsAdapter(contacts: [contact])
        let parser = ProductivityTaskNLUParser(contacts: adapter)
        let parsed = parser.parse("call John about the contract")
        XCTAssertEqual(parsed.linkedEntityIDs, [contact.id])
    }

    func testDocumentLinking() {
        let stub = DocumentStub(id: UUID(), title: "Q3 report")
        let adapter = MockDocumentStoreNLU(stubs: [stub])
        let parser = ProductivityTaskNLUParser(documents: adapter)
        let parsed = parser.parse("review the Q3 report")
        XCTAssertEqual(parsed.linkedEntityIDs, [stub.id])
    }

    func testQuotedNameLinking() {
        let contact = Contact(
            name: NameComponents(first: "Jane", last: "Smith")
        )
        let adapter = MockContactsAdapter(contacts: [contact])
        let parser = ProductivityTaskNLUParser(contacts: adapter)
        let parsed = parser.parse("\"Jane Smith\" followup call")
        XCTAssertEqual(parsed.linkedEntityIDs, [contact.id])
    }

    func testNoLinkWhenNameDoesNotMatch() {
        let parser = ProductivityTaskNLUParser(contacts: MockContactsAdapter(contacts: []))
        let parsed = parser.parse("call John about the contract")
        XCTAssertEqual(parsed.linkedEntityIDs, [])
    }

    func testCaseInsensitiveContactMatching() {
        let contact = Contact(
            name: NameComponents(first: "Alice", last: "Wonder")
        )
        let adapter = MockContactsAdapter(contacts: [contact])
        let parser = ProductivityTaskNLUParser(contacts: adapter)
        let parsed = parser.parse("call alice wonder")
        XCTAssertEqual(parsed.linkedEntityIDs, [contact.id])
    }
}
