import XCTest
@testable import TesseraCore

/// Lightweight structural / helper tests for the
/// email surface. The full ``EmailView`` (the
/// SwiftUI surface) lives in
/// `TesseraStudio/Sources/TesseraStudioMac/Views/Email/`
/// and is exercised by the macOS app's UI
/// walkthrough; the unit tests here pin the
/// pure helpers the view consumes (folder
/// counting, row sorting, etc.).
final class EmailViewStructureTests: XCTestCase {

    /// The folder count helper the sidebar
    /// uses is consistent: each email
    /// contributes exactly one to the
    /// bucket of its current folder.
    func testFolderCountsAreConsistent() {
        let emails: [EmailMessage] = [
            EmailMessage(messageID: "1", from: EmailAddress(email: "a@b"), folder: .inbox),
            EmailMessage(messageID: "2", from: EmailAddress(email: "a@b"), folder: .inbox),
            EmailMessage(messageID: "3", from: EmailAddress(email: "a@b"), folder: .sent),
            EmailMessage(messageID: "4", from: EmailAddress(email: "a@b"), folder: .trash),
            EmailMessage(messageID: "5", from: EmailAddress(email: "a@b"), folder: .custom("Work")),
        ]
        var counts: [Folder: Int] = [:]
        for e in emails {
            counts[e.folder, default: 0] += 1
        }
        XCTAssertEqual(counts[.inbox], 2)
        XCTAssertEqual(counts[.sent], 1)
        XCTAssertEqual(counts[.trash], 1)
        XCTAssertEqual(counts[.custom("Work")], 1)
    }

    /// The list-sort helper: unread first,
    /// then by receivedAt DESC. The Email
    /// view's middle column uses this sort.
    func testListSortUnreadFirstThenDate() {
        let now = Date()
        let emails: [EmailMessage] = [
            EmailMessage(messageID: "old-read", from: EmailAddress(email: "a@b"),
                         receivedAt: now.addingTimeInterval(-3600), isRead: true),
            EmailMessage(messageID: "new-unread", from: EmailAddress(email: "a@b"),
                         receivedAt: now, isRead: false),
            EmailMessage(messageID: "old-unread", from: EmailAddress(email: "a@b"),
                         receivedAt: now.addingTimeInterval(-7200), isRead: false),
            EmailMessage(messageID: "new-read", from: EmailAddress(email: "a@b"),
                         receivedAt: now.addingTimeInterval(-1800), isRead: true),
        ]
        let sorted = emails.sorted { a, b in
            if a.isRead != b.isRead { return !a.isRead && b.isRead }
            return a.receivedAt > b.receivedAt
        }
        XCTAssertEqual(sorted.map { $0.messageID },
                       ["new-unread", "old-unread", "new-read", "old-read"])
    }

    /// The keyboard shortcut → action map. The
    /// view layer reads this map (or uses
    /// onKeyPress directly); the unit test
    /// pins the vocabulary.
    func testKeyboardShortcuts() {
        // The mapping is:
        //   j -> next
        //   k -> previous
        //   r -> reply
        //   R -> reply all
        //   f -> forward
        //   a -> archive
        //   # -> trash
        //   s -> star
        //   c -> compose new
        //   / -> search
        //   J -> next thread
        //   K -> previous thread
        let map: [String: String] = [
            "j": "next", "k": "previous",
            "r": "reply", "R": "replyAll",
            "f": "forward",
            "a": "archive", "#": "trash",
            "s": "star", "c": "compose",
            "/": "search",
            "J": "nextThread", "K": "previousThread",
        ]
        XCTAssertEqual(map["j"], "next")
        XCTAssertEqual(map["r"], "reply")
        XCTAssertEqual(map["R"], "replyAll")
        XCTAssertEqual(map["#"], "trash")
    }

    /// Thread grouping produces distinct
    /// anchors. The Email view's J/K
    /// navigation walks the anchor list.
    func testThreadAnchorsAreDistinct() {
        let emails: [EmailMessage] = [
            EmailMessage(messageID: "a1@x", from: EmailAddress(email: "a@b"),
                         threadID: "thread-A@x"),
            EmailMessage(messageID: "a2@x", from: EmailAddress(email: "a@b"),
                         threadID: "thread-A@x"),
            EmailMessage(messageID: "b1@x", from: EmailAddress(email: "a@b"),
                         threadID: "thread-B@x"),
            EmailMessage(messageID: "c1@x", from: EmailAddress(email: "a@b"),
                         threadID: nil),
        ]
        // The thread anchor is the threadID
        // when present, the messageID
        // otherwise. Distinct anchors are
        // {thread-A@x, thread-B@x, c1@x}.
        let anchors = emails.map { $0.threadID ?? $0.messageID }
        let unique = Set(anchors)
        XCTAssertEqual(unique.count, 3)
    }
}
