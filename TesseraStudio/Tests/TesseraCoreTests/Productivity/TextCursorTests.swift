import XCTest
@testable import TesseraCore

/// Tests for the two-cursor data model: TextCursor + CursorPair.
final class TextCursorTests: XCTestCase {

    func testCursorEquality() {
        let id = UUID()
        let a = TextCursor(blockID: id, offset: 5, affinity: .downstream)
        let b = TextCursor(blockID: id, offset: 5, affinity: .downstream)
        XCTAssertEqual(a, b)
    }

    func testCursorInequalityDifferentBlock() {
        let a = TextCursor(blockID: UUID(), offset: 5, affinity: .downstream)
        let b = TextCursor(blockID: UUID(), offset: 5, affinity: .downstream)
        XCTAssertNotEqual(a, b)
    }

    func testCursorInequalityDifferentOffset() {
        let id = UUID()
        let a = TextCursor(blockID: id, offset: 5)
        let b = TextCursor(blockID: id, offset: 6)
        XCTAssertNotEqual(a, b)
    }

    func testCursorInequalityDifferentAffinity() {
        let id = UUID()
        let a = TextCursor(blockID: id, offset: 5, affinity: .upstream)
        let b = TextCursor(blockID: id, offset: 5, affinity: .downstream)
        XCTAssertNotEqual(a, b)
    }

    func testCursorSerialization() throws {
        let id = UUID()
        let original = TextCursor(blockID: id, offset: 42, affinity: .upstream)
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(TextCursor.self, from: data)
        XCTAssertEqual(original, decoded)
    }

    func testDefaultAffinityIsDownstream() {
        let cursor = TextCursor(blockID: UUID(), offset: 0)
        XCTAssertEqual(cursor.affinity, .downstream)
    }

    // MARK: - CursorPair

    func testEmptyPair() {
        let pair = CursorPair()
        XCTAssertTrue(pair.isEmpty)
        XCTAssertEqual(pair.count, 0)
        XCTAssertNil(pair.user)
        XCTAssertNil(pair.agent)
    }

    func testPairWithUserOnly() {
        let cursor = TextCursor(blockID: UUID(), offset: 0)
        let pair = CursorPair(user: cursor)
        XCTAssertFalse(pair.isEmpty)
        XCTAssertEqual(pair.count, 1)
        XCTAssertNotNil(pair.user)
        XCTAssertNil(pair.agent)
    }

    func testPairWithBothCursors() {
        let user = TextCursor(blockID: UUID(), offset: 5)
        let agent = TextCursor(blockID: UUID(), offset: 10)
        let pair = CursorPair(user: user, agent: agent)
        XCTAssertFalse(pair.isEmpty)
        XCTAssertEqual(pair.count, 2)
        XCTAssertEqual(pair.user, user)
        XCTAssertEqual(pair.agent, agent)
    }

    func testPairSerialization() throws {
        let pair = CursorPair(
            user: TextCursor(blockID: UUID(), offset: 5),
            agent: TextCursor(blockID: UUID(), offset: 10, affinity: .upstream)
        )
        let data = try JSONEncoder().encode(pair)
        let decoded = try JSONDecoder().decode(CursorPair.self, from: data)
        XCTAssertEqual(pair, decoded)
    }

    func testTwoCursorsSameBlockDifferentOffsets() throws {
        let blockID = UUID()
        let user = TextCursor(blockID: blockID, offset: 5)
        let agent = TextCursor(blockID: blockID, offset: 10)
        let pair = CursorPair(user: user, agent: agent)
        let data = try JSONEncoder().encode(pair)
        let decoded = try JSONDecoder().decode(CursorPair.self, from: data)
        XCTAssertEqual(decoded.user?.offset, 5)
        XCTAssertEqual(decoded.agent?.offset, 10)
        XCTAssertEqual(decoded.user?.blockID, blockID)
        XCTAssertEqual(decoded.agent?.blockID, blockID)
    }
}
