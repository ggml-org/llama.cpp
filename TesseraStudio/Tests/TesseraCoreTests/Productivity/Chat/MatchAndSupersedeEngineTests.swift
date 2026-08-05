import XCTest
import CryptoKit
@testable import TesseraCore

/// Tests for ``MatchAndSupersedeEngine``. The engine's
/// LLM call is mocked; the heuristic fallback is exercised
/// by the tests that pass a nil provider.
final class MatchAndSupersedeEngineTests: XCTestCase {

    // MARK: - Heuristic fallback

    func testHeuristicMatchesLexicallySimilar() async throws {
        let engine = MatchAndSupersedeEngine(llmProvider: nil)
        let new = ChatQueueItem(
            documentID: UUID(),
            order: 0,
            message: "summarize section two of the report",
            actor: .user(UUID())
        )
        let existing = ChatQueueItem(
            documentID: UUID(),
            order: 1,
            message: "summarize section two of the report thoroughly",
            actor: .user(UUID())
        )
        let decision = try await engine.evaluate(
            newFront: new,
            existingQueue: [existing]
        )
        XCTAssertEqual(decision.supersededItemIDs, [existing.id])
    }

    func testHeuristicIgnoresUnrelated() async throws {
        let engine = MatchAndSupersedeEngine(llmProvider: nil)
        let new = ChatQueueItem(
            documentID: UUID(),
            order: 0,
            message: "summarize section 2",
            actor: .user(UUID())
        )
        let unrelated = ChatQueueItem(
            documentID: UUID(),
            order: 1,
            message: "translate the introduction to French",
            actor: .user(UUID())
        )
        let decision = try await engine.evaluate(
            newFront: new,
            existingQueue: [unrelated]
        )
        XCTAssertTrue(decision.supersededItemIDs.isEmpty)
    }

    func testHeuristicSkipsSupersededItems() async throws {
        let engine = MatchAndSupersedeEngine(llmProvider: nil)
        let new = ChatQueueItem(
            documentID: UUID(),
            order: 0,
            message: "summarize section two",
            actor: .user(UUID())
        )
        let existing = ChatQueueItem(
            documentID: UUID(),
            order: 1,
            message: "summarize section two",
            actor: .user(UUID()),
            supersededByID: UUID()  // already superseded
        )
        let decision = try await engine.evaluate(
            newFront: new,
            existingQueue: [existing]
        )
        // Already-superseded items are not candidates.
        XCTAssertTrue(decision.supersededItemIDs.isEmpty)
    }

    func testHeuristicEmptyQueue() async throws {
        let engine = MatchAndSupersedeEngine(llmProvider: nil)
        let new = ChatQueueItem(
            documentID: UUID(),
            order: 0,
            message: "x",
            actor: .user(UUID())
        )
        let decision = try await engine.evaluate(
            newFront: new,
            existingQueue: []
        )
        XCTAssertTrue(decision.supersededItemIDs.isEmpty)
    }

    // MARK: - LLM path

    func testLLMResponseIsParsed() async throws {
        // The LLM is mocked to return a valid JSON response
        // listing one superseded id.
        let supersededID = UUID()
        let provider: MatchAndSupersedeEngine.LLMProvider = { _, _ in
            """
            {
              "superseded_ids": ["\(supersededID.uuidString)"],
              "reasoning": "test reasoning"
            }
            """
        }
        let engine = MatchAndSupersedeEngine(llmProvider: provider)
        let new = ChatQueueItem(
            documentID: UUID(),
            order: 0,
            message: "new instruction",
            actor: .user(UUID())
        )
        let existing = ChatQueueItem(
            id: supersededID,
            documentID: UUID(),
            order: 1,
            message: "old instruction",
            actor: .user(UUID())
        )
        let decision = try await engine.evaluate(
            newFront: new,
            existingQueue: [existing]
        )
        XCTAssertEqual(decision.supersededItemIDs, [supersededID])
        XCTAssertEqual(decision.reasoning, "test reasoning")
    }

    func testLLMResponseWithProseIsTolerated() async throws {
        let supersededID = UUID()
        let provider: MatchAndSupersedeEngine.LLMProvider = { _, _ in
            """
            The model thinks: {"superseded_ids": ["\(supersededID.uuidString)"], "reasoning": "x"}
            """
        }
        let engine = MatchAndSupersedeEngine(llmProvider: provider)
        let new = ChatQueueItem(
            documentID: UUID(),
            order: 0,
            message: "new",
            actor: .user(UUID())
        )
        let existing = ChatQueueItem(
            id: supersededID,
            documentID: UUID(),
            order: 1,
            message: "old",
            actor: .user(UUID())
        )
        let decision = try await engine.evaluate(
            newFront: new,
            existingQueue: [existing]
        )
        XCTAssertEqual(decision.supersededItemIDs, [supersededID])
    }

    func testLLMResponseWithUnknownIDIsFiltered() async throws {
        let realID = UUID()
        let fakeID = UUID()
        let provider: MatchAndSupersedeEngine.LLMProvider = { _, _ in
            """
            {
              "superseded_ids": ["\(realID.uuidString)", "\(fakeID.uuidString)"],
              "reasoning": "x"
            }
            """
        }
        let engine = MatchAndSupersedeEngine(llmProvider: provider)
        let new = ChatQueueItem(
            documentID: UUID(),
            order: 0,
            message: "new",
            actor: .user(UUID())
        )
        let real = ChatQueueItem(
            id: realID,
            documentID: UUID(),
            order: 1,
            message: "old",
            actor: .user(UUID())
        )
        let decision = try await engine.evaluate(
            newFront: new,
            existingQueue: [real]
        )
        XCTAssertEqual(decision.supersededItemIDs, [realID])
    }

    func testLLMFailureFallsBackToHeuristic() async throws {
        let provider: MatchAndSupersedeEngine.LLMProvider = { _, _ in
            throw NSError(domain: "test", code: 1)
        }
        let engine = MatchAndSupersedeEngine(llmProvider: provider)
        let new = ChatQueueItem(
            documentID: UUID(),
            order: 0,
            message: "summarize section two",
            actor: .user(UUID())
        )
        let existing = ChatQueueItem(
            documentID: UUID(),
            order: 1,
            message: "summarize section two thoroughly",
            actor: .user(UUID())
        )
        let decision = try await engine.evaluate(
            newFront: new,
            existingQueue: [existing]
        )
        // Heuristic kicked in; the similar message is superseded.
        XCTAssertEqual(decision.supersededItemIDs, [existing.id])
    }

    func testLLMUnparseableFallsBackToHeuristic() async throws {
        let provider: MatchAndSupersedeEngine.LLMProvider = { _, _ in
            "not json at all"
        }
        let engine = MatchAndSupersedeEngine(llmProvider: provider)
        let new = ChatQueueItem(
            documentID: UUID(),
            order: 0,
            message: "summarize section two",
            actor: .user(UUID())
        )
        let existing = ChatQueueItem(
            documentID: UUID(),
            order: 1,
            message: "summarize section two thoroughly",
            actor: .user(UUID())
        )
        let decision = try await engine.evaluate(
            newFront: new,
            existingQueue: [existing]
        )
        XCTAssertEqual(decision.supersededItemIDs, [existing.id])
    }

    // MARK: - Caching

    func testDecisionIsCached() async throws {
        let counter = CallCounter()
        let provider: MatchAndSupersedeEngine.LLMProvider = { _, _ in
            await counter.increment()
            return """
            { "superseded_ids": [], "reasoning": "none" }
            """
        }
        let engine = MatchAndSupersedeEngine(llmProvider: provider)
        // Use a fixed id so the cache key is stable across
        // calls (the cache is keyed by the new-front id).
        let newID = UUID(uuidString: "11111111-1111-1111-1111-111111111111")!
        let new = ChatQueueItem(
            id: newID,
            documentID: UUID(),
            order: 0,
            message: "x",
            actor: .user(UUID())
        )
        // Use a non-empty queue so the LLM is actually
        // called the first time (an empty queue skips
        // the LLM and just caches `.none`).
        let existing = ChatQueueItem(
            documentID: UUID(),
            order: 1,
            message: "y",
            actor: .user(UUID())
        )
        _ = try await engine.evaluate(newFront: new, existingQueue: [existing])
        _ = try await engine.evaluate(newFront: new, existingQueue: [existing])
        let count = await counter.value
        XCTAssertEqual(count, 1, "second call should hit the cache")
    }

    actor CallCounter {
        var value: Int = 0
        func increment() { value += 1 }
    }

    // MARK: - Tokenization helper

    func testTokenizeStripsPunctuation() {
        let tokens = MatchAndSupersedeEngine.tokenize("Hello, World! This is a test.")
        XCTAssertEqual(Set(tokens), Set(["hello", "world", "this", "is", "a", "test"]))
    }

    func testTokenizeLowercases() {
        let tokens = MatchAndSupersedeEngine.tokenize("UPPER lower")
        XCTAssertEqual(tokens, ["upper", "lower"])
    }
}
