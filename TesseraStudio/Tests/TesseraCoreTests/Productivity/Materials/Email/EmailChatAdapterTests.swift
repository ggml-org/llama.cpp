import XCTest
@testable import TesseraCore

/// Tests for the email chat panel adapter.
/// Covers both the intent parser (pure
/// string-based) and the run handlers
/// (which exercise the in-memory lookup
/// closure with a fake email list).
final class EmailChatAdapterTests: XCTestCase {

    // MARK: - Intent parsing

    func testParseReplyWithNamedRecipient() {
        let adapter = makeAdapter(emails: [])
        let intent = adapter.parseIntent("reply to John's email with: thanks, will do")
        guard case let .reply(emailID, body) = intent else {
            XCTFail("expected .reply, got \(intent)")
            return
        }
        XCTAssertEqual(body, "thanks, will do")
        XCTAssertEqual(emailID, EmailChatAdapter.sentinelID)
    }

    func testParseReplyWithEmailRecipient() {
        let adapter = makeAdapter(emails: [])
        let intent = adapter.parseIntent("reply to alice@example.com's email with: ok")
        guard case let .reply(_, body) = intent else {
            XCTFail("expected .reply")
            return
        }
        XCTAssertEqual(body, "ok")
    }

    func testParseReplyWithoutWithMarker() {
        let adapter = makeAdapter(emails: [])
        let intent = adapter.parseIntent("reply to John about lunch")
        XCTAssertEqual(intent, .unknown)
    }

    func testParseSummarizeThisThread() {
        let adapter = makeAdapter(emails: [])
        let intent = adapter.parseIntent("summarize this thread")
        XCTAssertEqual(intent, .summarize(threadID: nil))
    }

    func testParseSummarizeNamedThread() {
        let adapter = makeAdapter(emails: [])
        let intent = adapter.parseIntent("summarize thread abc@def.com")
        XCTAssertEqual(intent, .summarize(threadID: "abc@def.com"))
    }

    func testParseFindFromSender() {
        let adapter = makeAdapter(emails: [])
        let intent = adapter.parseIntent("find emails from John about Q3 planning")
        XCTAssertEqual(intent, .find(sender: "John", topic: "Q3 planning"))
    }

    func testParseFindOnlySender() {
        let adapter = makeAdapter(emails: [])
        let intent = adapter.parseIntent("find emails from John")
        XCTAssertEqual(intent, .find(sender: "John", topic: nil))
    }

    func testParseFindNoFilters() {
        let adapter = makeAdapter(emails: [])
        let intent = adapter.parseIntent("find emails")
        XCTAssertEqual(intent, .unknown)
    }

    func testParseUnknown() {
        let adapter = makeAdapter(emails: [])
        let intent = adapter.parseIntent("what's the weather")
        XCTAssertEqual(intent, .unknown)
    }

    // MARK: - Run handlers (with fake in-memory email list)

    /// Reply with the sentinel: the adapter
    /// resolves to the first email in the
    /// lookup list (the chat panel's
    /// "current" email) and the
    /// `openReplyComposer` context
    /// callback fires. The list ordering
    /// is the store's responsibility
    /// (``EmailStore.list`` sorts by
    /// receivedAt DESC); the adapter
    /// takes the first element of
    /// whatever the lookup returns.
    func testRunReplyOpensComposerForMostRecent() async {
        let mostRecentID = UUID()
        let emails = [
            makeEmail(id: mostRecentID, subject: "newer"),
            makeEmail(id: UUID(), subject: "older"),
        ]
        let adapter = makeAdapter(emails: emails)
        var opened: (UUID, String)?
        let ctx = EmailChatAdapter.RunContext(
            openReplyComposer: { id, body in
                opened = (id, body)
            }
        )
        let result = await adapter.run(
            intent: .reply(emailID: EmailChatAdapter.sentinelID, body: "thanks"),
            context: ctx
        )
        guard case .composerOpened(let id) = result else {
            XCTFail("expected .composerOpened, got \(result)")
            return
        }
        XCTAssertEqual(id, mostRecentID)
        XCTAssertEqual(opened?.0, mostRecentID)
        XCTAssertEqual(opened?.1, "thanks")
    }

    /// Reply with a specific email id:
    /// the adapter uses that id directly
    /// (no lookup).
    func testRunReplyOpensComposerForSpecificID() async {
        let targetID = UUID()
        let emails = [
            makeEmail(id: targetID, subject: "target"),
        ]
        let adapter = makeAdapter(emails: emails)
        var opened: (UUID, String)?
        let ctx = EmailChatAdapter.RunContext(
            openReplyComposer: { id, body in
                opened = (id, body)
            }
        )
        let result = await adapter.run(
            intent: .reply(emailID: targetID, body: "ok"),
            context: ctx
        )
        guard case .composerOpened(let id) = result else {
            XCTFail("expected .composerOpened")
            return
        }
        XCTAssertEqual(id, targetID)
        XCTAssertEqual(opened?.1, "ok")
    }

    /// Summarize with no emails returns
    /// `.noAction`.
    func testRunSummarizeNoEmails() async {
        let adapter = makeAdapter(emails: [])
        let ctx = EmailChatAdapter.RunContext()
        let result = await adapter.run(
            intent: .summarize(threadID: "x"),
            context: ctx
        )
        if case .noAction = result { /* expected */ } else {
            XCTFail("expected .noAction, got \(result)")
        }
    }

    /// Summarize with a named thread id
    /// returns the matching emails in
    /// `noteCreated`. The note title
    /// includes the count.
    func testRunSummarizeWithThreadID() async {
        let emails = [
            makeEmail(id: UUID(), subject: "First", threadID: "T1"),
            makeEmail(id: UUID(), subject: "Second", threadID: "T1"),
            makeEmail(id: UUID(), subject: "Other", threadID: "T2"),
        ]
        let adapter = makeAdapter(emails: emails)
        var created: (String, String)?
        let ctx = EmailChatAdapter.RunContext(
            createNote: { title, body in
                created = (title, body)
            }
        )
        let result = await adapter.run(
            intent: .summarize(threadID: "T1"),
            context: ctx
        )
        guard case .noteCreated(let title, _) = result else {
            XCTFail("expected .noteCreated, got \(result)")
            return
        }
        XCTAssertTrue(title.contains("2 messages"), "title should mention 2 messages; got \(title)")
        XCTAssertTrue(created?.0.contains("2 messages") == true)
        XCTAssertNotNil(created?.1)
    }

    /// "summarize this thread" uses the
    /// most recent email's thread.
    func testRunSummarizeThisThreadUsesMostRecent() async {
        let mostRecentID = UUID()
        let emails = [
            makeEmail(id: mostRecentID, subject: "newer", threadID: "T-B"),
            makeEmail(id: UUID(), subject: "older", threadID: "T-A"),
        ]
        let adapter = makeAdapter(emails: emails)
        let ctx = EmailChatAdapter.RunContext()
        let result = await adapter.run(
            intent: .summarize(threadID: nil),
            context: ctx
        )
        guard case .noteCreated(_, let body) = result else {
            XCTFail("expected .noteCreated")
            return
        }
        // The body should contain the
        // most recent email's subject
        // (the "newer" one).
        XCTAssertTrue(body.contains("newer"), "body should mention 'newer'; got: \(body.prefix(200))")
    }

    /// Find returns the matching emails.
    func testRunFindBySender() async {
        let aliceID = UUID()
        let bobID = UUID()
        let emails = [
            makeEmail(id: aliceID, subject: "Q3 plan", from: EmailAddress(name: "Alice", email: "alice@example.com")),
            makeEmail(id: bobID, subject: "Lunch?", from: EmailAddress(name: "Bob", email: "bob@example.com")),
        ]
        let adapter = makeAdapter(emails: emails)
        var shown: [EmailMessage] = []
        let ctx = EmailChatAdapter.RunContext(
            showInlineResults: { results in
                shown = results
            }
        )
        let result = await adapter.run(
            intent: .find(sender: "Alice", topic: nil),
            context: ctx
        )
        guard case .inlineResults(let ids) = result else {
            XCTFail("expected .inlineResults")
            return
        }
        XCTAssertEqual(ids, [aliceID])
        XCTAssertEqual(shown.count, 1)
    }

    /// Find with both sender + topic.
    func testRunFindBySenderAndTopic() async {
        let matchID = UUID()
        let emails = [
            makeEmail(id: UUID(), subject: "Q3 plan", from: EmailAddress(name: "Alice", email: "alice@example.com")),
            makeEmail(id: matchID, subject: "Q3 review", from: EmailAddress(name: "Alice", email: "alice@example.com")),
        ]
        let adapter = makeAdapter(emails: emails)
        let ctx = EmailChatAdapter.RunContext()
        let result = await adapter.run(
            intent: .find(sender: "Alice", topic: "review"),
            context: ctx
        )
        guard case .inlineResults(let ids) = result else {
            XCTFail("expected .inlineResults")
            return
        }
        XCTAssertEqual(ids, [matchID])
    }

    /// Find with no matches returns an
    /// empty list.
    func testRunFindNoMatches() async {
        let emails = [
            makeEmail(id: UUID(), subject: "x", from: EmailAddress(email: "a@b")),
        ]
        let adapter = makeAdapter(emails: emails)
        let ctx = EmailChatAdapter.RunContext()
        let result = await adapter.run(
            intent: .find(sender: "nobody", topic: nil),
            context: ctx
        )
        guard case .inlineResults(let ids) = result else {
            XCTFail("expected .inlineResults")
            return
        }
        XCTAssertTrue(ids.isEmpty)
    }

    /// Unknown intent: no action.
    func testRunUnknown() async {
        let adapter = makeAdapter(emails: [])
        let ctx = EmailChatAdapter.RunContext()
        let result = await adapter.run(
            intent: .unknown,
            context: ctx
        )
        if case .noAction(let reason) = result {
            XCTAssertFalse(reason.isEmpty)
        } else {
            XCTFail("expected .noAction")
        }
    }

    // MARK: - Stub

    /// Build an adapter backed by a
    /// fixed in-memory list. The closure
    /// runs on the actor; the test awaits
    /// the result.
    private func makeAdapter(emails: [EmailMessage]) -> EmailChatAdapter {
        let captured = emails
        return EmailChatAdapter(lookup: { captured })
    }

    private func makeEmail(
        id: UUID,
        subject: String,
        threadID: String? = nil,
        from: EmailAddress? = nil
    ) -> EmailMessage {
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        return EmailMessage(
            id: id,
            messageID: "msg-\(id.uuidString)@x",
            from: from ?? EmailAddress(email: "alice@example.com"),
            subject: subject,
            bodyPlain: "body of \(subject)",
            receivedAt: date,
            folder: .inbox,
            threadID: threadID,
            createdAt: date,
            updatedAt: date
        )
    }
}
