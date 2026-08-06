import XCTest
@testable import TesseraCore

/// Tests for the ``EmailComposer`` reply/forward
/// logic. Pure unit tests; no I/O.
final class EmailComposerTests: XCTestCase {

    // MARK: - Reply

    func testReplyPreservesSenderAsTo() {
        let original = makeOriginal()
        let composer = EmailComposer(mode: .reply(to: original, all: false), from: makeIdentity())
        XCTAssertEqual(composer.to.count, 1)
        XCTAssertEqual(composer.to.first?.email, original.from.email)
        XCTAssertTrue(composer.cc.isEmpty)
    }

    func testReplySetsSubjectWithRePrefix() {
        let original = makeOriginal(subject: "Lunch tomorrow?")
        let composer = EmailComposer(mode: .reply(to: original, all: false), from: makeIdentity())
        XCTAssertEqual(composer.subject, "Re: Lunch tomorrow?")
    }

    func testReplySubjectIdempotent() {
        let original = makeOriginal(subject: "Re: hello")
        let composer = EmailComposer(mode: .reply(to: original, all: false), from: makeIdentity())
        XCTAssertEqual(composer.subject, "Re: hello")
    }

    func testReplyQuotesBody() {
        let original = makeOriginal(bodyPlain: "Hi there\nHow are you?")
        let composer = EmailComposer(mode: .reply(to: original, all: false), from: makeIdentity())
        XCTAssertTrue(composer.bodyPlain.contains("> Hi there"))
        XCTAssertTrue(composer.bodyPlain.contains("> How are you?"))
        XCTAssertTrue(composer.bodyPlain.contains("wrote:"))
    }

    func testReplySetsInReplyTo() {
        let original = makeOriginal()
        let composer = EmailComposer(mode: .reply(to: original, all: false), from: makeIdentity())
        XCTAssertEqual(composer.inReplyTo, original.id)
        XCTAssertEqual(composer.inReplyToMessageID, original.messageID)
        XCTAssertEqual(composer.threadID, original.messageID)
    }

    // MARK: - Reply all

    func testReplyAllIncludesOtherRecipients() {
        let original = makeOriginal(
            from: EmailAddress(email: "alice@example.com"),
            to: [EmailAddress(email: "bob@example.com"), EmailAddress(email: "carol@example.com")],
            cc: [EmailAddress(email: "dave@example.com")]
        )
        let composer = EmailComposer(mode: .reply(to: original, all: true), from: makeIdentity())
        // The to: line is the original sender (alice).
        XCTAssertEqual(composer.to.first?.email, "alice@example.com")
        // The cc: line is everyone else.
        let ccEmails = Set(composer.cc.map { $0.email })
        XCTAssertTrue(ccEmails.contains("bob@example.com"))
        XCTAssertTrue(ccEmails.contains("carol@example.com"))
        XCTAssertTrue(ccEmails.contains("dave@example.com"))
    }

    func testReplyAllExcludesSender() {
        let original = makeOriginal(
            from: EmailAddress(email: "alice@example.com"),
            to: [EmailAddress(email: "bob@example.com")]
        )
        let composer = EmailComposer(mode: .reply(to: original, all: true), from: makeIdentity())
        // The sender of the reply (the user) is excluded
        // from the cc list; the test fixture's "me" is
        // the composer `from`, not the original from.
        // Alice (the original sender) is in `to`, not cc.
        XCTAssertFalse(composer.cc.contains { $0.email == "alice@example.com" })
    }

    // MARK: - Forward

    func testForwardSetsSubjectWithFwdPrefix() {
        let original = makeOriginal(subject: "Original subject")
        let composer = EmailComposer(mode: .forward(original), from: makeIdentity())
        XCTAssertEqual(composer.subject, "Fwd: Original subject")
    }

    func testForwardSubjectIdempotent() {
        let original = makeOriginal(subject: "Fwd: x")
        let composer = EmailComposer(mode: .forward(original), from: makeIdentity())
        XCTAssertEqual(composer.subject, "Fwd: x")
    }

    func testForwardIncludesHeader() {
        let original = makeOriginal(
            from: EmailAddress(name: "Alice", email: "alice@example.com"),
            to: [EmailAddress(email: "bob@example.com")]
        )
        let composer = EmailComposer(mode: .forward(original), from: makeIdentity())
        XCTAssertTrue(composer.bodyPlain.contains("Forwarded message"))
        XCTAssertTrue(composer.bodyPlain.contains("From: Alice <alice@example.com>"))
        XCTAssertTrue(composer.bodyPlain.contains("Subject:"))
    }

    func testForwardPreservesAttachments() {
        let attachment = Attachment(filename: "doc.pdf", mimeType: "application/pdf", size: 1024)
        let original = makeOriginal(attachments: [attachment])
        let composer = EmailComposer(mode: .forward(original), from: makeIdentity())
        XCTAssertEqual(composer.attachments.count, 1)
        XCTAssertEqual(composer.attachments.first?.filename, "doc.pdf")
    }

    func testForwardInheritsThreadID() {
        let original = makeOriginal()
        let composer = EmailComposer(mode: .forward(original), from: makeIdentity())
        XCTAssertEqual(composer.threadID, original.messageID)
    }

    // MARK: - New

    func testNewComposerEmpty() {
        let composer = EmailComposer(mode: .new, from: makeIdentity())
        XCTAssertTrue(composer.to.isEmpty)
        XCTAssertTrue(composer.cc.isEmpty)
        XCTAssertTrue(composer.subject.isEmpty)
        XCTAssertTrue(composer.bodyPlain.isEmpty)
    }

    // MARK: - Build

    func testBuildNewDraft() {
        let composer = EmailComposer(mode: .new, from: makeIdentity())
            .setTo([EmailAddress(email: "x@y")])
            .setSubject("Hi")
            .setBody("Hello")
        let draft = composer.build()
        XCTAssertEqual(draft.composeMode, .new)
        XCTAssertEqual(draft.to.first?.email, "x@y")
        XCTAssertEqual(draft.subject, "Hi")
        XCTAssertEqual(draft.bodyPlain, "Hello")
        XCTAssertEqual(draft.from.email, makeIdentity().email)
    }

    func testBuildReplyDraft() {
        let original = makeOriginal()
        let composer = EmailComposer(mode: .reply(to: original, all: false), from: makeIdentity())
        let draft = composer.build()
        XCTAssertEqual(draft.composeMode, .reply)
    }

    func testBuildReplyAllDraft() {
        let original = makeOriginal()
        let composer = EmailComposer(mode: .reply(to: original, all: true), from: makeIdentity())
        let draft = composer.build()
        XCTAssertEqual(draft.composeMode, .replyAll)
    }

    func testBuildForwardDraft() {
        let original = makeOriginal()
        let composer = EmailComposer(mode: .forward(original), from: makeIdentity())
        let draft = composer.build()
        XCTAssertEqual(draft.composeMode, .forward)
    }

    // MARK: - EML

    func testEMLDataIncludesRequiredHeaders() {
        let original = makeOriginal()
        let composer = EmailComposer(mode: .reply(to: original, all: false), from: makeIdentity())
        let draft = composer.build()
        let data = draft.emlData()
        let str = String(data: data, encoding: .utf8) ?? ""
        XCTAssertTrue(str.contains("From:"))
        XCTAssertTrue(str.contains("To:"))
        XCTAssertTrue(str.contains("Subject: Re:"))
        XCTAssertTrue(str.contains("Message-ID:"))
        XCTAssertTrue(str.contains("In-Reply-To:"))
    }

    // MARK: - Helpers

    private func makeIdentity() -> EmailAddress {
        EmailAddress(name: "Me", email: "me@example.com")
    }

    private func makeOriginal(
        messageID: String = "orig-msg@example.com",
        subject: String = "Hello",
        bodyPlain: String = "Body",
        from: EmailAddress? = nil,
        to: [EmailAddress] = [],
        cc: [EmailAddress] = [],
        attachments: [Attachment] = []
    ) -> EmailMessage {
        EmailMessage(
            id: UUID(),
            messageID: messageID,
            from: from ?? EmailAddress(name: "Alice", email: "alice@example.com"),
            to: to,
            cc: cc,
            subject: subject,
            bodyPlain: bodyPlain,
            receivedAt: Date(timeIntervalSince1970: 1_700_000_000),
            folder: .inbox,
            threadID: messageID,
            attachments: attachments
        )
    }
}
