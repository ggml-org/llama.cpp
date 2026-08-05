import XCTest
@testable import TesseraCore

/// Tests for ``ReceiptsCoordinator``. The coordinator
/// is the cross-surface navigation state between the
/// chat panel, the receipt drawer, and (eventually) the
/// Graph view.
final class ReceiptsCoordinatorTests: XCTestCase {

    // MARK: - Open receipt in drawer

    func testOpenReceiptInDrawerSetsFocus() async {
        let coord = ReceiptsCoordinator()
        let receiptID = UUID()
        await coord.openReceiptInDrawer(receiptID, fromChatItem: nil)
        let focus = await coord.currentFocus
        if case .receipt(let id) = focus {
            XCTAssertEqual(id, receiptID)
        } else {
            XCTFail("expected .receipt focus")
        }
    }

    func testOpenReceiptSetsDrawerVisible() async {
        let coord = ReceiptsCoordinator()
        let receiptID = UUID()
        await coord.setDrawerVisible(false)
        await coord.openReceiptInDrawer(receiptID, fromChatItem: nil)
        let visible = await coord.isDrawerVisible
        XCTAssertTrue(visible)
    }

    func testOpenReceiptProducesOpenRequest() async {
        let coord = ReceiptsCoordinator()
        let receiptID = UUID()
        let chatItemID = UUID()
        await coord.openReceiptInDrawer(receiptID, fromChatItem: chatItemID)
        let request = await coord.consumeOpenRequest()
        XCTAssertEqual(request?.receiptID, receiptID)
        XCTAssertEqual(request?.fromChatItemID, chatItemID)
    }

    func testConsumeOpenRequestClearsIt() async {
        let coord = ReceiptsCoordinator()
        let receiptID = UUID()
        await coord.openReceiptInDrawer(receiptID, fromChatItem: nil)
        _ = await coord.consumeOpenRequest()
        let second = await coord.consumeOpenRequest()
        XCTAssertNil(second)
    }

    // MARK: - Show in chat

    func testShowInChatWithNoLookupReturnsNil() async {
        let coord = ReceiptsCoordinator()
        let receiptID = UUID()
        let result = await coord.showInChat(receiptID: receiptID)
        XCTAssertNil(result)
    }

    func testShowInChatWithLookupResolvesID() async {
        let coord = ReceiptsCoordinator()
        let receiptID = UUID()
        let chatItemID = UUID()
        await coord.setChatItemLookup { inputReceiptID in
            if inputReceiptID == receiptID { return chatItemID }
            return nil
        }
        let result = await coord.showInChat(receiptID: receiptID)
        XCTAssertEqual(result, chatItemID)
        let target = await coord.currentScrollTarget
        XCTAssertEqual(target, chatItemID)
    }

    func testClearScrollTarget() async {
        let coord = ReceiptsCoordinator()
        let receiptID = UUID()
        let chatItemID = UUID()
        await coord.setChatItemLookup { _ in chatItemID }
        _ = await coord.showInChat(receiptID: receiptID)
        await coord.clearScrollTarget()
        let target = await coord.currentScrollTarget
        XCTAssertNil(target)
    }

    // MARK: - Show in graph

    func testShowInGraphSetsFocus() async {
        let coord = ReceiptsCoordinator()
        let entityID = UUID()
        await coord.showInGraph(entityID: entityID)
        let focus = await coord.currentFocus
        if case .graphEntity(let id) = focus {
            XCTAssertEqual(id, entityID)
        } else {
            XCTFail("expected .graphEntity focus")
        }
    }

    // MARK: - Drawer visibility

    func testToggleDrawer() async {
        let coord = ReceiptsCoordinator()
        let initial = await coord.isDrawerVisible
        await coord.toggleDrawerVisibility()
        let after = await coord.isDrawerVisible
        XCTAssertNotEqual(initial, after)
    }

    func testClearFocusResetsState() async {
        let coord = ReceiptsCoordinator()
        let receiptID = UUID()
        await coord.openReceiptInDrawer(receiptID, fromChatItem: nil)
        await coord.clearFocus()
        let focus = await coord.currentFocus
        if case .none = focus { } else {
            XCTFail("expected .none focus")
        }
        let request = await coord.consumeOpenRequest()
        XCTAssertNil(request)
    }
}
