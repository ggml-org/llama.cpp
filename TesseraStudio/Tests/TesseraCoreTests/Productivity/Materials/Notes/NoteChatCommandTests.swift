import XCTest
@testable import TesseraCore

/// Tests for ``NoteChatCommand``: parsing of the canonical
/// chat-panel phrasings, the `apply(to:)` dispatch.
final class NoteChatCommandTests: XCTestCase {

    // MARK: - Parsing: create note

    func testParseCreateNoteTitled() {
        let parsed = NoteChatCommand.parse(
            message: "create a new note titled 'Meeting notes for Q3 review'"
        )
        XCTAssertNotNil(parsed)
        if case .createNote(let title, let tags) = parsed?.command {
            XCTAssertEqual(title, "Meeting notes for Q3 review")
            XCTAssertTrue(tags.isEmpty)
        } else {
            XCTFail("expected createNote command")
        }
        XCTAssertEqual(parsed?.requiresAgentConfirmation, false)
    }

    func testParseCreateNoteTitledWithoutQuotes() {
        let parsed = NoteChatCommand.parse(
            message: "create a new note titled Q3 retrospective"
        )
        if case .createNote(let title, _) = parsed?.command {
            XCTAssertEqual(title, "Q3 retrospective")
        } else {
            XCTFail("expected createNote command")
        }
    }

    func testParseCreateNoteShorterForm() {
        // The "create" verb is required (matches the spec's
        // canonical phrasings); the shorter "new note titled"
        // variant is supported via the same pattern.
        let parsed = NoteChatCommand.parse(message: "create a note titled Sprint planning")
        if case .createNote(let title, _) = parsed?.command {
            XCTAssertEqual(title, "Sprint planning")
        } else {
            XCTFail("expected createNote command")
        }
    }

    // MARK: - Parsing: tags

    func testParseAddTag() {
        let noteID = UUID()
        let parsed = NoteChatCommand.parse(
            message: "add a tag 'q3-2026' to this note",
            activeNoteID: noteID
        )
        if case .addTag(let id, let tag) = parsed?.command {
            XCTAssertEqual(id, noteID)
            XCTAssertEqual(tag, "q3-2026")
        } else {
            XCTFail("expected addTag command")
        }
    }

    func testParseAddTagWithoutQuotes() {
        let noteID = UUID()
        let parsed = NoteChatCommand.parse(
            message: "add tag urgent to this note",
            activeNoteID: noteID
        )
        if case .addTag(let id, let tag) = parsed?.command {
            XCTAssertEqual(id, noteID)
            XCTAssertEqual(tag, "urgent")
        } else {
            XCTFail("expected addTag command")
        }
    }

    func testParseAddTagRequiresActiveNote() {
        let parsed = NoteChatCommand.parse(
            message: "add a tag urgent to this note",
            activeNoteID: nil
        )
        XCTAssertNil(parsed)
    }

    func testParseRemoveTag() {
        let noteID = UUID()
        let parsed = NoteChatCommand.parse(
            message: "remove tag 'q3-2026' from this note",
            activeNoteID: noteID
        )
        if case .removeTag(let id, let tag) = parsed?.command {
            XCTAssertEqual(id, noteID)
            XCTAssertEqual(tag, "q3-2026")
        } else {
            XCTFail("expected removeTag command")
        }
    }

    // MARK: - Parsing: pin / archive

    func testParsePinThisNote() {
        let noteID = UUID()
        let parsed = NoteChatCommand.parse(
            message: "pin this note",
            activeNoteID: noteID
        )
        if case .setPinned(let id, let pinned) = parsed?.command {
            XCTAssertEqual(id, noteID)
            XCTAssertTrue(pinned)
        } else {
            XCTFail("expected setPinned command")
        }
    }

    func testParseUnpinThisNote() {
        let noteID = UUID()
        let parsed = NoteChatCommand.parse(
            message: "unpin this note",
            activeNoteID: noteID
        )
        if case .setPinned(let id, let pinned) = parsed?.command {
            XCTAssertEqual(id, noteID)
            XCTAssertFalse(pinned)
        } else {
            XCTFail("expected setPinned command")
        }
    }

    func testParseArchiveThisNote() {
        let noteID = UUID()
        let parsed = NoteChatCommand.parse(
            message: "archive this note",
            activeNoteID: noteID
        )
        if case .setArchived(let id, let archived) = parsed?.command {
            XCTAssertEqual(id, noteID)
            XCTAssertTrue(archived)
        } else {
            XCTFail("expected setArchived command")
        }
    }

    func testParseUnarchiveThisNote() {
        let noteID = UUID()
        let parsed = NoteChatCommand.parse(
            message: "unarchive this note",
            activeNoteID: noteID
        )
        if case .setArchived(let id, let archived) = parsed?.command {
            XCTAssertEqual(id, noteID)
            XCTAssertFalse(archived)
        } else {
            XCTFail("expected setArchived command")
        }
    }

    func testParsePinRequiresActiveNote() {
        let parsed = NoteChatCommand.parse(
            message: "pin this note",
            activeNoteID: nil
        )
        XCTAssertNil(parsed)
    }

    // MARK: - Parsing: link

    func testParseLinkToOtherEntity() {
        let noteID = UUID()
        let targetID = UUID()
        let parsed = NoteChatCommand.parse(
            message: "link this note to that document",
            activeNoteID: noteID,
            targetEntityID: targetID
        )
        if case .link(let id, let target, let linkType) = parsed?.command {
            XCTAssertEqual(id, noteID)
            XCTAssertEqual(target, targetID)
            XCTAssertEqual(linkType, "related_to")
        } else {
            XCTFail("expected link command")
        }
    }

    func testParseLinkWithoutTargetReturnsNil() {
        let noteID = UUID()
        let parsed = NoteChatCommand.parse(
            message: "link this note to that document",
            activeNoteID: noteID,
            targetEntityID: nil
        )
        XCTAssertNil(parsed)
    }

    // MARK: - Parsing: summarize (requires confirmation)

    func testParseSummarize() {
        let parsed = NoteChatCommand.parse(message: "summarize this article")
        XCTAssertNotNil(parsed)
        XCTAssertEqual(parsed?.requiresAgentConfirmation, true)
    }

    // MARK: - Parsing: empty / no match

    func testParseEmptyMessage() {
        XCTAssertNil(NoteChatCommand.parse(message: ""))
        XCTAssertNil(NoteChatCommand.parse(message: "   "))
    }

    func testParseNonMatchingMessage() {
        XCTAssertNil(NoteChatCommand.parse(message: "what's the weather?"))
    }
}
