import XCTest
import Foundation
@testable import TesseraCore

/// Tests for `EditorCoalescer` (per spec §5.5). The coalescer
/// aggregates a burst of user edits into a single `Mutation`
/// batch + a single `ChatQueueItem`. The default window is
/// 1.5s, configurable 0.5-5.0s.
final class EditorCoalescerTests: XCTestCase {

    // MARK: - Settings clamping

    func testSettingsClampsToMin() {
        let s = EditorCoalescer.Settings(coalesceWindow: 0.1)
        XCTAssertEqual(s.coalesceWindow, 0.5)
    }

    func testSettingsClampsToMax() {
        let s = EditorCoalescer.Settings(coalesceWindow: 100)
        XCTAssertEqual(s.coalesceWindow, 5.0)
    }

    func testSettingsAcceptsValidRange() {
        let s = EditorCoalescer.Settings(coalesceWindow: 2.0)
        XCTAssertEqual(s.coalesceWindow, 2.0)
    }

    func testDefaultSettingsIs1500ms() {
        let s = EditorCoalescer.Settings.default
        XCTAssertEqual(s.coalesceWindow, 1.5)
    }

    // MARK: - Coalescing a burst

    func testTenKeystrokesIn1sProduceOneBurst() {
        let c = EditorCoalescer(settings: .default)
        let docID = UUID()
        let blockID = UUID()
        for _ in 0..<10 {
            c.append(
                mutation: .setBlockContent(blockID: blockID, content: [InlineRun(text: "x")]),
                blockID: blockID,
                documentID: docID,
                queueMessage: "you edited"
            )
        }
        XCTAssertTrue(c.hasPending)
        // The first flush should have a single mutation
        // (the most recent one; intermediate states are
        // overwritten by the reducer-style coalesce).
        let burst = c.flush()
        XCTAssertNotNil(burst)
        XCTAssertEqual(burst?.mutations.count, 1)
        XCTAssertEqual(burst?.queueItem.documentID, docID)
        XCTAssertEqual(burst?.queueItem.state, .applied)
    }

    func testBurstIncludesQueueItemWithSourceMutation() {
        let c = EditorCoalescer(settings: .default)
        let docID = UUID()
        let blockID = UUID()
        c.append(
            mutation: .setBlockContent(blockID: blockID, content: [InlineRun(text: "x")]),
            blockID: blockID,
            documentID: docID,
            queueMessage: "you edited paragraph 3"
        )
        let burst = c.flush()
        XCTAssertNotNil(burst?.queueItem.sourceMutation)
        XCTAssertEqual(burst?.queueItem.message, "you edited paragraph 3")
    }

    func testFlushWithNoPendingReturnsNil() {
        let c = EditorCoalescer()
        XCTAssertNil(c.flush())
    }

    func testFlushClearsPendingState() {
        let c = EditorCoalescer(settings: EditorCoalescer.Settings(coalesceWindow: 0.5))
        let docID = UUID()
        let blockID = UUID()
        c.append(
            mutation: .setBlockContent(blockID: blockID, content: [InlineRun(text: "x")]),
            blockID: blockID,
            documentID: docID,
            queueMessage: "edit"
        )
        XCTAssertTrue(c.hasPending)
        _ = c.flush()
        XCTAssertFalse(c.hasPending)
        XCTAssertNil(c.flush())
    }

    // MARK: - Cross-block edits start a new burst

    func testEditsInDifferentBlocksStartNewBursts() {
        let c = EditorCoalescer(settings: EditorCoalescer.Settings(coalesceWindow: 5.0))
        let docID = UUID()
        let blockA = UUID()
        let blockB = UUID()
        c.append(
            mutation: .setBlockContent(blockID: blockA, content: [InlineRun(text: "a")]),
            blockID: blockA,
            documentID: docID,
            queueMessage: "edit A"
        )
        // A second edit in a different block flushes the
        // first burst and starts a new one.
        c.append(
            mutation: .setBlockContent(blockID: blockB, content: [InlineRun(text: "b")]),
            blockID: blockB,
            documentID: docID,
            queueMessage: "edit B"
        )
        // Now flush: we get the second burst.
        let burst = c.flush()
        XCTAssertEqual(burst?.blockID, blockB)
        XCTAssertEqual(burst?.queueItem.message, "edit B")
    }

    func testEditsInDifferentDocumentsStartNewBursts() {
        let c = EditorCoalescer(settings: EditorCoalescer.Settings(coalesceWindow: 5.0))
        let docA = UUID()
        let docB = UUID()
        let blockID = UUID()
        c.append(
            mutation: .setBlockContent(blockID: blockID, content: [InlineRun(text: "a")]),
            blockID: blockID,
            documentID: docA,
            queueMessage: "edit A"
        )
        c.append(
            mutation: .setBlockContent(blockID: blockID, content: [InlineRun(text: "b")]),
            blockID: blockID,
            documentID: docB,
            queueMessage: "edit B"
        )
        let burst = c.flush()
        XCTAssertEqual(burst?.documentID, docB)
    }

    // MARK: - Notification on flush

    func testFlushPostsNotification() {
        let c = EditorCoalescer(settings: EditorCoalescer.Settings(coalesceWindow: 0.5))
        let docID = UUID()
        let blockID = UUID()
        let expectation = self.expectation(forNotification: EditorCoalescer.didFlushNotification, object: nil)
        c.append(
            mutation: .setBlockContent(blockID: blockID, content: [InlineRun(text: "x")]),
            blockID: blockID,
            documentID: docID,
            queueMessage: "edit"
        )
        _ = c.flush()
        wait(for: [expectation], timeout: 1.0)
    }

    // MARK: - Settings update

    func testUpdateSettingsApplies() {
        let c = EditorCoalescer(settings: EditorCoalescer.Settings(coalesceWindow: 1.5))
        c.updateSettings(EditorCoalescer.Settings(coalesceWindow: 2.0))
        XCTAssertEqual(c.coalesceWindow, 2.0)
    }

    // MARK: - Window expiry

    func testWindowExpiryFlushes() {
        // Use a short window (0.5s minimum) and wait for it
        // to elapse before checking hasPending flips to false.
        let c = EditorCoalescer(settings: EditorCoalescer.Settings(coalesceWindow: 0.5))
        let docID = UUID()
        let blockID = UUID()
        c.append(
            mutation: .setBlockContent(blockID: blockID, content: [InlineRun(text: "x")]),
            blockID: blockID,
            documentID: docID,
            queueMessage: "edit"
        )
        XCTAssertTrue(c.hasPending)
        // Wait for the window to elapse.
        let expectation = XCTestExpectation(description: "flush timer fired")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 2.0)
        XCTAssertFalse(c.hasPending)
    }
}
