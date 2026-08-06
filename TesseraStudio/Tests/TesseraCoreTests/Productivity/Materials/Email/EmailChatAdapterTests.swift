import XCTest
@testable import TesseraCore

/// Tests for the email chat panel adapter. Pure
/// unit tests; the adapter's `parseIntent` and
/// `run` helpers are deterministic and don't
/// touch a real data layer.
final class EmailChatAdapterTests: XCTestCase {

    // MARK: - Intent parsing

    func testParseReplyWithNamedRecipient() {
        let adapter = EmailChatAdapter(store: makeStoreStub())
        let intent = adapter.parseIntent("reply to John's email with: thanks, will do")
        guard case let .reply(emailID, body) = intent else {
            XCTFail("expected .reply, got \(intent)")
            return
        }
        XCTAssertEqual(body, "thanks, will do")
        // The parser doesn't resolve the name
        // (the run handler does); the emailID
        // is the sentinel.
        XCTAssertEqual(emailID, EmailChatAdapter.sentinelID)
    }

    func testParseReplyWithEmailRecipient() {
        let adapter = EmailChatAdapter(store: makeStoreStub())
        let intent = adapter.parseIntent("reply to alice@example.com's email with: ok")
        guard case let .reply(_, body) = intent else {
            XCTFail("expected .reply")
            return
        }
        XCTAssertEqual(body, "ok")
    }

    func testParseReplyWithoutWithMarker() {
        let adapter = EmailChatAdapter(store: makeStoreStub())
        let intent = adapter.parseIntent("reply to John about lunch")
        XCTAssertEqual(intent, .unknown)
    }

    func testParseSummarizeThisThread() {
        let adapter = EmailChatAdapter(store: makeStoreStub())
        let intent = adapter.parseIntent("summarize this thread")
        XCTAssertEqual(intent, .summarize(threadID: nil))
    }

    func testParseSummarizeNamedThread() {
        let adapter = EmailChatAdapter(store: makeStoreStub())
        let intent = adapter.parseIntent("summarize thread abc@def.com")
        XCTAssertEqual(intent, .summarize(threadID: "abc@def.com"))
    }

    func testParseFindFromSender() {
        let adapter = EmailChatAdapter(store: makeStoreStub())
        let intent = adapter.parseIntent("find emails from John about Q3 planning")
        XCTAssertEqual(intent, .find(sender: "John", topic: "Q3 planning"))
    }

    func testParseFindOnlySender() {
        let adapter = EmailChatAdapter(store: makeStoreStub())
        let intent = adapter.parseIntent("find emails from John")
        XCTAssertEqual(intent, .find(sender: "John", topic: nil))
    }

    func testParseFindNoFilters() {
        let adapter = EmailChatAdapter(store: makeStoreStub())
        let intent = adapter.parseIntent("find emails")
        XCTAssertEqual(intent, .unknown)
    }

    func testParseUnknown() {
        let adapter = EmailChatAdapter(store: makeStoreStub())
        let intent = adapter.parseIntent("what's the weather")
        XCTAssertEqual(intent, .unknown)
    }

    // MARK: - Run handlers (synchronous, fake context)

    func testRunReplyNoEmailNoAction() async {
        let adapter = EmailChatAdapter(store: makeStoreStub())
        let ctx = EmailChatAdapter.RunContext()
        let result = await adapter.run(
            intent: .reply(emailID: EmailChatAdapter.sentinelID, body: "x"),
            context: ctx
        )
        if case .noAction = result { /* expected */ } else {
            XCTFail("expected .noAction, got \(result)")
        }
    }

    func testRunSummarizeNoEmails() async {
        let adapter = EmailChatAdapter(store: makeStoreStub())
        let ctx = EmailChatAdapter.RunContext()
        let result = await adapter.run(
            intent: .summarize(threadID: "x"),
            context: ctx
        )
        if case .noAction = result { /* expected */ } else {
            XCTFail("expected .noAction, got \(result)")
        }
    }

    func testRunFindEmptyStore() async {
        let adapter = EmailChatAdapter(store: makeStoreStub())
        let ctx = EmailChatAdapter.RunContext()
        let result = await adapter.run(
            intent: .find(sender: "anyone", topic: nil),
            context: ctx
        )
        if case .inlineResults(let ids) = result {
            XCTAssertTrue(ids.isEmpty)
        } else {
            XCTFail("expected .inlineResults, got \(result)")
        }
    }

    // MARK: - Stub store

    /// A real-but-un-started `TesseraDataLayer`
    /// for parser tests. The parser is pure and
    /// doesn't call into the store; we just need
    /// a non-nil facade. The run-handler tests
    /// that DO call into the store (`.reply`,
    /// `.summarize`, `.find`) verify the
    /// no-op fallbacks, which work even when
    /// the facade throws on every call.
    private func makeStoreStub() -> EmailStore {
        let dl = TesseraDataLayer(configuration: .init())
        return EmailStore(dataLayer: dl)
    }
}
