import XCTest
@testable import TesseraCore

/// Tests for ``EmailSender``'s pure helpers
/// (the EML staging logic, the .eml shape
/// produced by ``DraftEmail/emlData()``). The
/// actual share sheet presentation is
/// `@MainActor`-bound and not exercised in the
/// unit tests; the integration test is via the
/// macOS UI smoke (manually verified).
final class EmailSenderTests: XCTestCase {

    func testSenderErrorEquality() {
        let a = EmailSender.SenderError.emptyDraft
        let b = EmailSender.SenderError.emptyDraft
        XCTAssertEqual(a, b)
        let c = EmailSender.SenderError.stageFailed(reason: "x")
        XCTAssertNotEqual(a, c)
    }

    func testSendResultHashable() {
        let url = URL(fileURLWithPath: "/tmp/a.eml")
        let r1 = EmailSender.SendResult.routedToSystemShare(url)
        let r2 = EmailSender.SendResult.routedToSystemShare(url)
        XCTAssertEqual(r1, r2)
        let r3 = EmailSender.SendResult.savedAsDraft
        XCTAssertNotEqual(r1, r3)
    }

    // MARK: - EML output

    func testDraftEMLDataHasRequiredHeaders() {
        let draft = DraftEmail(
            from: EmailAddress(name: "Me", email: "me@example.com"),
            to: [EmailAddress(name: "Bob", email: "bob@example.com")],
            subject: "Hello",
            bodyPlain: "Hi Bob,\n\nHow are you?"
        )
        let data = draft.emlData()
        let s = String(data: data, encoding: .utf8) ?? ""
        XCTAssertTrue(s.contains("From: Me <me@example.com>"))
        XCTAssertTrue(s.contains("To: Bob <bob@example.com>"))
        XCTAssertTrue(s.contains("Subject: Hello"))
        XCTAssertTrue(s.contains("Hi Bob,"))
        XCTAssertTrue(s.contains("Message-ID:"))
    }

    func testDraftEMLDataQuotedPrintableSafe() {
        // The .eml body is UTF-8. The header
        // values pass through unmodified; the
        // body lines are not encoded-word
        // because modern Apple Mail handles
        // UTF-8 .eml natively. This test
        // documents the assumption.
        let draft = DraftEmail(
            from: EmailAddress(email: "a@b.com"),
            to: [EmailAddress(email: "c@d.com")],
            subject: "Test",
            bodyPlain: "Hello \u{4e16}\u{754c}"
        )
        let data = draft.emlData()
        let s = String(data: data, encoding: .utf8) ?? ""
        XCTAssertTrue(s.contains("\u{4e16}\u{754c}"))
    }

    func testDraftEMLMultipartWhenHTMLPresent() {
        let draft = DraftEmail(
            from: EmailAddress(email: "a@b.com"),
            to: [EmailAddress(email: "c@d.com")],
            subject: "HTML test",
            bodyPlain: "Plain version",
            bodyHTML: "<p>HTML version</p>"
        )
        let data = draft.emlData()
        let s = String(data: data, encoding: .utf8) ?? ""
        XCTAssertTrue(s.contains("multipart/alternative"))
        XCTAssertTrue(s.contains("boundary="))
        XCTAssertTrue(s.contains("text/plain"))
        XCTAssertTrue(s.contains("text/html"))
        XCTAssertTrue(s.contains("Plain version"))
        XCTAssertTrue(s.contains("HTML version"))
    }

    func testDraftEMLInReplyToWhenSet() {
        let draft = DraftEmail(
            inReplyToMessageID: "orig@example.com",
            threadID: "orig@example.com",
            from: EmailAddress(email: "a@b.com"),
            to: [EmailAddress(email: "c@d.com")],
            subject: "Re: hi",
            bodyPlain: "Yes"
        )
        let data = draft.emlData()
        let s = String(data: data, encoding: .utf8) ?? ""
        XCTAssertTrue(s.contains("In-Reply-To: <orig@example.com>"))
        XCTAssertTrue(s.contains("References: <orig@example.com>"))
    }

    func testDraftToEmailMessageUsesDraftsFolder() {
        let draft = DraftEmail(
            from: EmailAddress(email: "a@b.com"),
            to: [EmailAddress(email: "c@d.com")],
            subject: "x",
            bodyPlain: "y"
        )
        let message = draft.toEmailMessage()
        XCTAssertEqual(message.folder, .drafts)
        XCTAssertEqual(message.id, draft.id)
        XCTAssertEqual(message.subject, "x")
    }
}
