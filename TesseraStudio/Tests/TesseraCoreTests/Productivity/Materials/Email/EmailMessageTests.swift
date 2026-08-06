import XCTest
@testable import TesseraCore

/// Unit tests for ``EmailMessage`` and its
/// supporting types. These tests are pure
/// (no data layer) and run in every environment.
final class EmailMessageTests: XCTestCase {

    // MARK: - JSON round-trip

    func testRoundTripJSON() throws {
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let original = EmailMessage(
            messageID: "abc123@example.com",
            from: EmailAddress(name: "Alice", email: "alice@example.com"),
            to: [EmailAddress(email: "bob@example.com")],
            cc: [],
            subject: "Hello",
            bodyPlain: "Body line 1\nBody line 2",
            bodyHTML: "<p>Body</p>",
            receivedAt: date,
            sentAt: date,
            isRead: true,
            folder: .inbox,
            threadID: "thread-anchor@example.com",
            attachments: [Attachment(
                filename: "report.pdf",
                mimeType: "application/pdf",
                size: 12345
            )],
            createdAt: date,
            updatedAt: date
        )
        let data = try original.jsonData()
        let decoded = try EmailMessage.from(jsonData: data)
        XCTAssertEqual(decoded, original)
    }

    func testJSONStringRoundTrip() throws {
        let date = Date(timeIntervalSince1970: 1_000_000)
        let original = EmailMessage(
            messageID: "x@y",
            from: EmailAddress(email: "a@b"),
            subject: "Test",
            bodyPlain: "hi",
            receivedAt: date,
            createdAt: date,
            updatedAt: date
        )
        let body = try original.jsonDataString()
        let decoded = try EmailMessage.from(jsonDataString: body)
        XCTAssertEqual(decoded, original)
    }

    func testInvalidUTF8Throws() {
        XCTAssertThrowsError(try EmailMessage.from(jsonDataString: "not json")) { _ in }
    }

    // MARK: - Display helpers

    func testDisplaySubjectEmpty() {
        let e = EmailMessage(messageID: "x", from: EmailAddress(email: "a@b"), subject: "  ")
        XCTAssertEqual(e.displaySubject, "(no subject)")
    }

    func testDisplaySubjectTrimmed() {
        let e = EmailMessage(messageID: "x", from: EmailAddress(email: "a@b"), subject: "  hi  ")
        XCTAssertEqual(e.displaySubject, "hi")
    }

    func testSnippetStripsNewlines() {
        let e = EmailMessage(
            messageID: "x",
            from: EmailAddress(email: "a@b"),
            bodyPlain: "Line 1\nLine 2\nLine 3"
        )
        XCTAssertEqual(e.snippet, "Line 1 Line 2 Line 3")
    }

    func testSnippetTruncates() {
        let long = String(repeating: "a", count: 200)
        let e = EmailMessage(messageID: "x", from: EmailAddress(email: "a@b"), bodyPlain: long)
        XCTAssertEqual(e.snippet.count, 80)
    }

    func testSenderDisplayPrefersName() {
        let e = EmailMessage(
            messageID: "x",
            from: EmailAddress(name: "Alice Example", email: "alice@example.com")
        )
        XCTAssertEqual(e.senderDisplay, "Alice Example")
    }

    func testSenderDisplayFallsBackToEmail() {
        let e = EmailMessage(
            messageID: "x",
            from: EmailAddress(email: "alice@example.com")
        )
        XCTAssertEqual(e.senderDisplay, "alice@example.com")
    }

    // MARK: - EmailAddress

    func testMailboxStringWithName() {
        let a = EmailAddress(name: "Alice", email: "alice@example.com")
        XCTAssertEqual(a.mailboxString, "Alice <alice@example.com>")
    }

    func testMailboxStringBare() {
        let a = EmailAddress(email: "alice@example.com")
        XCTAssertEqual(a.mailboxString, "alice@example.com")
    }

    func testCanonicalEmailLowercases() {
        let a = EmailAddress(email: "Alice@Example.COM")
        XCTAssertEqual(a.canonicalEmail, "alice@example.com")
    }

    // MARK: - Folder

    func testFolderDisplayNames() {
        XCTAssertEqual(Folder.inbox.displayName, "Inbox")
        XCTAssertEqual(Folder.sent.displayName, "Sent")
        XCTAssertEqual(Folder.drafts.displayName, "Drafts")
        XCTAssertEqual(Folder.archive.displayName, "Archive")
        XCTAssertEqual(Folder.trash.displayName, "Trash")
        XCTAssertEqual(Folder.custom("Work").displayName, "Work")
    }

    func testFolderSystemID() {
        XCTAssertEqual(Folder.inbox.systemID, "inbox")
        XCTAssertEqual(Folder.custom("Work").systemID, "label:work")
    }

    func testFolderIsSystem() {
        XCTAssertTrue(Folder.inbox.isSystem)
        XCTAssertTrue(Folder.trash.isSystem)
        XCTAssertFalse(Folder.custom("Work").isSystem)
    }

    // MARK: - Threading

    func testThreadingFromReferences() {
        let anchor = Threading.normalize(
            messageID: "msg@x",
            inReplyTo: "child@x",
            references: ["anchor@x", "middle@x"]
        )
        XCTAssertEqual(anchor, "anchor@x")
    }

    func testThreadingFromInReplyTo() {
        let anchor = Threading.normalize(
            messageID: "msg@x",
            inReplyTo: "parent@x",
            references: []
        )
        XCTAssertEqual(anchor, "parent@x")
    }

    func testThreadingFallsBackToMessageID() {
        let anchor = Threading.normalize(
            messageID: "self@x",
            inReplyTo: nil,
            references: []
        )
        XCTAssertEqual(anchor, "self@x")
    }

    func testThreadingSkipsEmptyReferences() {
        let anchor = Threading.normalize(
            messageID: "self@x",
            inReplyTo: nil,
            references: ["", "real@x"]
        )
        XCTAssertEqual(anchor, "real@x")
    }

    func testThreadingStripBrackets() {
        XCTAssertEqual(Threading.stripBrackets("<a@b>"), "a@b")
        XCTAssertEqual(Threading.stripBrackets("a@b"), "a@b")
        XCTAssertEqual(Threading.stripBrackets("  <a@b>  "), "a@b")
    }

    func testThreadingSplitReferences() {
        let refs = Threading.splitReferences("<a@b> <c@d> <e@f>")
        XCTAssertEqual(refs, ["a@b", "c@d", "e@f"])
    }

    func testThreadingSplitReferencesEmpty() {
        XCTAssertEqual(Threading.splitReferences(""), [])
    }

    // MARK: - Attachment

    func testAttachmentHashable() {
        let id = UUID()
        let a1 = Attachment(id: id, filename: "f", mimeType: "text/plain", size: 10, dataReference: "x")
        let a2 = Attachment(id: id, filename: "f", mimeType: "text/plain", size: 10, dataReference: "x")
        XCTAssertEqual(a1, a2)
    }

    func testAttachmentIdentifiable() {
        let a1 = Attachment(id: UUID(), filename: "f", mimeType: "text/plain", size: 10)
        let a2 = a1  // same id
        XCTAssertEqual(a1.id, a2.id)
    }
}
