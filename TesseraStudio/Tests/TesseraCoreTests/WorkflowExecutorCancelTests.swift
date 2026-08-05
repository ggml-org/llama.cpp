import XCTest
@testable import TesseraCore

// MARK: - Stub nodes for cancellation tests

/// Records node starts and finishes outside the event stream,
/// so tests can assert what the executor actually ran even after
/// a consumer cancellation cuts the stream off (a cancelled
/// iterator stops seeing buffered events).
final class NodeRunRecorder: @unchecked Sendable {
    static let shared = NodeRunRecorder()
    private let lock = NSLock()
    private var entries: [String] = []

    func reset() {
        lock.lock(); entries = []; lock.unlock()
    }
    func record(_ entry: String) {
        lock.lock(); entries.append(entry); lock.unlock()
    }
    var all: [String] {
        lock.lock(); defer { lock.unlock() }
        return entries
    }
}

/// Flags when the consuming `for await` loop exits; this SDK's
/// `Task` has no `isDone`, so tests observe termination here.
final class ConsumerFinished: @unchecked Sendable {
    private let lock = NSLock()
    private var flagged = false
    func mark() {
        lock.lock(); flagged = true; lock.unlock()
    }
    var isMarked: Bool {
        lock.lock(); defer { lock.unlock() }
        return flagged
    }
}

/// Test-only node: records its tag, sleeps for the `seconds`
/// parameter, then records completion. `Task.sleep` throws on
/// cancellation, so a cancelled run never records the finish.
struct SleepyNode: WorkflowNodeType {
    static let typeId = "test_sleepy"
    static let displayName = "Sleepy"
    static let summary = "Test-only node that sleeps for a parameter-controlled duration."
    static let inputs: [WorkflowPort] = [
        WorkflowPort(id: "value", label: "Value", type: .string),
    ]
    static let outputs: [WorkflowPort] = [
        WorkflowPort(id: "value", label: "Value", type: .string),
    ]
    static let parameterSchema = JSONSchema()

    static func execute(
        parameters: [String: JSONValue],
        inputs: [String: WorkflowPortValue],
        context: WorkflowExecutionContext
    ) async throws -> [String: WorkflowPortValue] {
        let tag = parameters["tag"]?.stringValue ?? "?"
        let seconds = parameters["seconds"]?.numberValue ?? 0.05
        NodeRunRecorder.shared.record("start:\(tag)")
        try await Task.sleep(for: .seconds(seconds))
        NodeRunRecorder.shared.record("finish:\(tag)")
        return ["value": .string(tag)]
    }
}

// MARK: - Cancellation

final class WorkflowExecutorCancelTests: XCTestCase {
    private func registry() -> WorkflowNodeRegistry {
        WorkflowNodeRegistry(types: [SleepyNode.self])
    }

    /// Linear chain a -> b -> ... with per-tag sleep durations.
    private func chain(_ tags: [String], seconds: [String: Double] = [:]) -> Workflow {
        let nodes = tags.map {
            WorkflowNode(id: $0, type: SleepyNode.typeId,
                         parameters: ["tag": .string($0),
                                      "seconds": .number(seconds[$0] ?? 0.05)])
        }
        let edges = zip(tags, tags.dropFirst()).map {
            WorkflowEdge(fromNode: $0, fromPort: "value", toNode: $1, toPort: "value")
        }
        return Workflow(name: "cancel-chain", nodes: nodes, edges: edges)
    }

    private func waitFor(timeout: TimeInterval, _ predicate: () -> Bool) async -> Bool {
        let deadline = Date(timeIntervalSinceNow: timeout)
        while Date() < deadline {
            if predicate() { return true }
            try? await Task.sleep(for: .milliseconds(5))
        }
        return predicate()
    }

    override func setUp() {
        super.setUp()
        NodeRunRecorder.shared.reset()
    }

    func testRunWithoutCancelCompletesAllNodes() async {
        let wf = chain(["a", "b"])
        let ex = WorkflowExecutor(registry: registry())
        var events: [WorkflowEvent] = []
        for await ev in await ex.run(wf) {
            events.append(ev)
        }
        guard case .finished(let success, _) = events.last else {
            XCTFail("missing finished event; events=\(events)")
            return
        }
        XCTAssertTrue(success)
        XCTAssertEqual(NodeRunRecorder.shared.all,
                       ["start:a", "finish:a", "start:b", "finish:b"])
    }

    func testCancelMidRunPreventsRemainingNodes() async {
        let wf = chain(["a", "b", "c"], seconds: ["b": 0.3])
        let ex = WorkflowExecutor(registry: registry())
        let stream = await ex.run(wf)
        let finished = ConsumerFinished()
        let consumer = Task {
            for await _ in stream {}
            finished.mark()
        }
        let bStarted = await waitFor(timeout: 2) {
            NodeRunRecorder.shared.all.contains("start:b")
        }
        XCTAssertTrue(bStarted)
        consumer.cancel()
        // far longer than b's remaining sleep; an executor that
        // ignored cancellation would run c almost immediately
        try? await Task.sleep(for: .seconds(1.5))
        let entries = NodeRunRecorder.shared.all
        XCTAssertFalse(entries.contains("start:c"),
                       "cancelled run kept scheduling: \(entries)")
        XCTAssertTrue(finished.isMarked)
    }

    func testCancelInterruptsInFlightNode() async {
        let wf = chain(["solo"], seconds: ["solo": 2.0])
        let ex = WorkflowExecutor(registry: registry())
        let stream = await ex.run(wf)
        let consumer = Task {
            for await _ in stream {}
        }
        let started = await waitFor(timeout: 2) {
            NodeRunRecorder.shared.all.contains("start:solo")
        }
        XCTAssertTrue(started)
        consumer.cancel()
        // outlast the full sleep: an uninterrupted node records
        // finish:solo after 2s
        try? await Task.sleep(for: .seconds(2.5))
        XCTAssertFalse(NodeRunRecorder.shared.all.contains("finish:solo"))
    }

    func testCancelTerminatesConsumerPromptly() async {
        let wf = chain(["long"], seconds: ["long": 30.0])
        let ex = WorkflowExecutor(registry: registry())
        let stream = await ex.run(wf)
        let finished = ConsumerFinished()
        let consumer = Task {
            for await _ in stream {}
            finished.mark()
        }
        let started = await waitFor(timeout: 2) {
            NodeRunRecorder.shared.all.contains("start:long")
        }
        XCTAssertTrue(started)
        consumer.cancel()
        let done = await waitFor(timeout: 2) { finished.isMarked }
        XCTAssertTrue(done, "consumer task still iterating 2s after cancel")
    }
}
