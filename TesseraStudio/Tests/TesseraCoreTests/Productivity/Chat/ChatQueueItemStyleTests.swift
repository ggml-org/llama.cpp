import XCTest
@testable import TesseraCore

/// Tests for the per-state visual treatment of
/// `ChatQueueItem`. The treatment is encoded in
/// `ChatQueueItemStyle`; the SwiftUI view consumes the
/// style. The tests cover all five states
/// (pending / inProgress / applied / failed / superseded)
/// and the supersession-takes-precedence rule.
final class ChatQueueItemStyleTests: XCTestCase {

    private func makeItem(
        state: ChatQueueItem.State = .pending,
        supersededByID: UUID? = nil
    ) -> ChatQueueItem {
        ChatQueueItem(
            documentID: UUID(),
            order: 0,
            message: "x",
            state: state,
            actor: .user(UUID()),
            supersededByID: supersededByID
        )
    }

    // MARK: - Per-state treatment

    func testPendingStyle() {
        let item = makeItem(state: .pending)
        let style = ChatQueueItemStyle.style(for: item, in: [item])
        XCTAssertEqual(style.state, .pending)
        XCTAssertEqual(style.icon, .clock)
        XCTAssertTrue(style.isItalic)
        XCTAssertEqual(style.opacity, 0.6, accuracy: 0.001)
        XCTAssertEqual(style.backgroundStyle, .clear)
        XCTAssertFalse(style.showsProgress)
        XCTAssertFalse(style.showsRetry)
        XCTAssertFalse(style.pulseAnimation)
    }

    func testInProgressStyle() {
        let item = makeItem(state: .inProgress)
        let style = ChatQueueItemStyle.style(for: item, in: [item])
        XCTAssertEqual(style.state, .inProgress)
        XCTAssertEqual(style.icon, .progress)
        XCTAssertFalse(style.isItalic)
        XCTAssertEqual(style.opacity, 1.0)
        XCTAssertEqual(style.backgroundStyle, .subtleHighlight)
        XCTAssertTrue(style.showsProgress)
        XCTAssertTrue(style.pulseAnimation)
    }

    func testAppliedStyle() {
        let item = makeItem(state: .applied)
        let style = ChatQueueItemStyle.style(for: item, in: [item])
        XCTAssertEqual(style.state, .applied)
        XCTAssertEqual(style.icon, .checkmark)
        XCTAssertFalse(style.isItalic)
        XCTAssertEqual(style.opacity, 1.0)
        XCTAssertEqual(style.backgroundStyle, .clear)
    }

    func testFailedStyle() {
        let item = makeItem(state: .failed)
        let style = ChatQueueItemStyle.style(for: item, in: [item])
        XCTAssertEqual(style.state, .failed)
        XCTAssertEqual(style.icon, .warning)
        XCTAssertEqual(style.backgroundStyle, .redFlash)
        XCTAssertTrue(style.showsRetry)
    }

    func testSupersededStyle() {
        let item = makeItem(state: .applied, supersededByID: UUID())
        let style = ChatQueueItemStyle.style(for: item, in: [item])
        XCTAssertEqual(style.state, .superseded)
        XCTAssertEqual(style.icon, .superseded)
        XCTAssertEqual(style.opacity, 0.5, accuracy: 0.001)
        XCTAssertNotNil(style.replaceBadge)
    }

    func testSupersessionTakesPrecedenceOverState() {
        // A pending item that is superseded is rendered
        // as superseded, not pending.
        let item = makeItem(state: .pending, supersededByID: UUID())
        let style = ChatQueueItemStyle.style(for: item, in: [item])
        XCTAssertEqual(style.state, .superseded)
    }

    // MARK: - Display position

    func testDisplayPosition() {
        let a = makeItem()
        let b = makeItem()
        let c = makeItem()
        XCTAssertEqual(a.displayPosition(among: [a, b, c]), 1)
        XCTAssertEqual(b.displayPosition(among: [a, b, c]), 2)
        XCTAssertEqual(c.displayPosition(among: [a, b, c]), 3)
    }

    func testDisplayPositionAbsent() {
        let a = makeItem()
        let b = makeItem()
        let c = makeItem()
        let absent = makeItem()
        XCTAssertNil(absent.displayPosition(among: [a, b, c]))
    }

    // MARK: - Display builder

    func testDisplayBuilderWiresPosition() {
        let a = makeItem()
        let b = makeItem()
        let display = ChatQueueItemDisplay.display(
            for: a,
            in: [a, b],
            meta: .empty
        )
        XCTAssertEqual(display.position, 1)
        XCTAssertEqual(display.message, "x")
    }

    // MARK: - Icon system names

    func testIconSystemNames() {
        XCTAssertEqual(ChatQueueItemStyle(state: .pending, icon: .clock, isItalic: false, opacity: 0, backgroundStyle: .clear).iconSystemName, "clock")
        XCTAssertEqual(ChatQueueItemStyle(state: .inProgress, icon: .progress, isItalic: false, opacity: 0, backgroundStyle: .clear).iconSystemName, "circle.dotted")
        XCTAssertEqual(ChatQueueItemStyle(state: .applied, icon: .checkmark, isItalic: false, opacity: 0, backgroundStyle: .clear).iconSystemName, "checkmark.circle.fill")
        XCTAssertEqual(ChatQueueItemStyle(state: .failed, icon: .warning, isItalic: false, opacity: 0, backgroundStyle: .clear).iconSystemName, "exclamationmark.triangle.fill")
        XCTAssertEqual(ChatQueueItemStyle(state: .superseded, icon: .superseded, isItalic: false, opacity: 0, backgroundStyle: .clear).iconSystemName, "arrow.uturn.backward")
    }
}
