import XCTest
@testable import TesseraCore

/// Tests for the `HoldMode` enum. The hold mode is the
/// pause state of a document's chat queue (per spec
/// §6.8). The transitions are exercised in
/// `ChatPanelStateMachineTests`; these tests cover the
/// enum's computed properties.
final class HoldModeTests: XCTestCase {

    func testIsPaused() {
        XCTAssertFalse(HoldMode.running.isPaused)
        XCTAssertTrue(HoldMode.holdRequested.isPaused)
        XCTAssertTrue(HoldMode.hold.isPaused)
        XCTAssertTrue(HoldMode.resuming.isPaused)
    }

    func testIsUserPaused() {
        XCTAssertFalse(HoldMode.running.isUserPaused)
        XCTAssertFalse(HoldMode.holdRequested.isUserPaused)
        XCTAssertTrue(HoldMode.hold.isUserPaused)
        XCTAssertFalse(HoldMode.resuming.isUserPaused)
    }

    func testFooterButtonLabel() {
        XCTAssertEqual(HoldMode.running.footerButtonLabel, "Hold your horses")
        XCTAssertEqual(HoldMode.holdRequested.footerButtonLabel, "Resume")
        XCTAssertEqual(HoldMode.hold.footerButtonLabel, "Resume")
        XCTAssertEqual(HoldMode.resuming.footerButtonLabel, "Resume")
    }

    func testAllCases() {
        let cases: Set<HoldMode> = [.running, .holdRequested, .hold, .resuming]
        XCTAssertEqual(Set(HoldMode.allCases), cases)
    }
}
