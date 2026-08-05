import XCTest
import CryptoKit
@testable import TesseraCore

/// Tests for the `AgentContext` data model. The context
/// is the data the agent's `LLMProvider.complete(...)`
/// call sees; the prompt builder is also tested.
final class AgentContextTests: XCTestCase {

    func testEmptyContext() {
        let ctx = AgentContext(
            documentID: UUID(),
            pending: [],
            recentReceipts: [],
            documentAST: .empty
        )
        XCTAssertFalse(ctx.hasPending)
        XCTAssertNil(ctx.frontPending)
    }

    func testFrontPendingIsFirst() {
        let pending = [
            ChatQueueItem(documentID: UUID(), order: 0, message: "first", actor: .user(UUID())),
            ChatQueueItem(documentID: UUID(), order: 1, message: "second", actor: .user(UUID()))
        ]
        let ctx = AgentContext(
            documentID: UUID(),
            pending: pending,
            recentReceipts: [],
            documentAST: .empty
        )
        XCTAssertTrue(ctx.hasPending)
        XCTAssertEqual(ctx.frontPending?.message, "first")
    }

    func testPromptSectionIncludesAllFields() {
        let docID = UUID()
        let pending = [
            ChatQueueItem(documentID: docID, order: 0, message: "summarize section 2", actor: .user(UUID()))
        ]
        let key = Curve25519.Signing.PrivateKey()
        let signer = ReceiptSigner(signingKey: key)
        let receipt = (try? signer.sign(
            documentID: docID,
            mutations: [],
            priorReceiptID: nil,
            actor: .user(UUID()),
            preMutationSnapshot: [:]
        ))!
        let ctx = AgentContext(
            documentID: docID,
            pending: pending,
            recentReceipts: [receipt],
            documentAST: .empty
        )
        let prompt = ctx.asPromptSection()
        XCTAssertTrue(prompt.contains("<agent_context>"))
        XCTAssertTrue(prompt.contains(docID.uuidString))
        XCTAssertTrue(prompt.contains("summarize section 2"))
        XCTAssertTrue(prompt.contains("recent_receipts"))
    }

    func testPendingCapIsRespected() {
        // The state machine applies the cap; the context
        // struct just carries what was given.
        let pending = (0..<10).map { i in
            ChatQueueItem(documentID: UUID(), order: i, message: "\(i)", actor: .user(UUID()))
        }
        let ctx = AgentContext(
            documentID: UUID(),
            pending: Array(pending.prefix(5)),
            recentReceipts: [],
            documentAST: .empty
        )
        XCTAssertEqual(ctx.pending.count, 5)
    }
}
