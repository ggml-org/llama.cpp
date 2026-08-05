import Foundation

/// Errors surfaced by the workflow executor. All are
/// ``Equatable`` so headless runs and tests can assert against
/// them without matching on the error description.
public enum WorkflowExecutorError: Error, Equatable {
    /// The workflow references a node ``type`` that no
    /// registered ``WorkflowNodeType`` provides.
    case unknownNodeType(nodeId: String, typeId: String)
    /// The workflow declares a port that the node type
    /// doesn't (anymore?). A schema-version mismatch; the
    /// user should migrate or pin the workflow to an older
    /// runtime.
    case unknownPort(nodeId: String, portId: String, side: String)
    /// An edge connects two ports whose types are
    /// incompatible (see ``WorkflowPortType.canFlowInto``).
    case typeMismatch(edge: WorkflowEdge, fromType: String, toType: String)
    /// The workflow's graph has a cycle. The executor
    /// cannot topologically sort a cyclic graph.
    case cycle(detectedAt: String)
    /// A node threw during ``execute``. The ``nodeId`` and
    /// ``message`` are surfaced to the caller; the workflow
    /// is aborted.
    case nodeFailed(nodeId: String, message: String)
}

/// Executes a workflow. The executor:
///   1. Validates the workflow (every node type is
///      registered; every edge connects declared ports with
///      type-compatible ends; the graph is acyclic).
///   2. Topologically sorts the nodes.
///   3. Runs each node in order, threading outputs into the
///      next node's inputs.
///   4. Surfaces ``WorkflowEvent``s so the editor can render
///      progress and headless runs can log to stderr.
///
/// The executor is ``Sendable`` and the run is async; tests
/// can run multiple workflows concurrently. Each node
/// implementation is responsible for its own thread / actor
/// isolation; the executor does not serialise node executions
/// (a workflow with no data dependencies between branches
/// can run them in parallel — though the current implementation
/// runs them sequentially, topologically; parallel execution
/// is a follow-on).
public actor WorkflowExecutor {
    private let registry: WorkflowNodeRegistry

    public init(registry: WorkflowNodeRegistry) {
        self.registry = registry
    }

    /// Run the workflow to completion. The returned stream
    /// emits ``WorkflowEvent``s; the final event is always
    /// ``.finished``. The function itself returns the
    /// final outputs (one entry per output port on the last
    /// node in the topological order, so the caller can
    /// inspect what came out of the workflow).
    ///
    /// The task iterating the stream owns the run: cancelling
    /// it stops the executor - no further nodes are scheduled
    /// and the in-flight node observes Task cancellation.
    public func run(
        _ workflow: Workflow,
        context: WorkflowExecutionContext = WorkflowExecutionContext()
    ) -> AsyncStream<WorkflowEvent> {
        AsyncStream { continuation in
            let producer = Task {
                let outputs = await self.runInternal(workflow, context: context, continuation: continuation)
                continuation.finish()
                _ = outputs
            }
            // the consumer's cancellation must reach the producer; the
            // unstructured task above does not inherit it on its own
            continuation.onTermination = { termination in
                if case .cancelled = termination {
                    producer.cancel()
                }
            }
        }
    }

    /// Same as ``run`` but returns the outputs alongside the
    /// stream. The stream is iterated by the caller's
    /// ``for await`` loop; the function returns the final
    /// per-node output map once the stream is finished.
    public func runCollecting(
        _ workflow: Workflow,
        context: WorkflowExecutionContext = WorkflowExecutionContext()
    ) -> (events: AsyncStream<WorkflowEvent>, finalOutputs: Task<[String: [String: WorkflowPortValue]], Never>) {
        let events = AsyncStream<WorkflowEvent>.makeStream()
        let finalOutputs = Task {
            await self.runAndCollectInternal(workflow, context: context, eventSink: events.continuation)
        }
        return (events.stream, finalOutputs)
    }

    private func runAndCollectInternal(
        _ workflow: Workflow,
        context: WorkflowExecutionContext,
        eventSink: AsyncStream<WorkflowEvent>.Continuation
    ) async -> [String: [String: WorkflowPortValue]] {
        let outputs = await runInternal(workflow, context: context, continuation: eventSink)
        return outputs
    }

    private func runInternal(
        _ workflow: Workflow,
        context: WorkflowExecutionContext,
        continuation: AsyncStream<WorkflowEvent>.Continuation
    ) async -> [String: [String: WorkflowPortValue]] {
        continuation.yield(.started(workflowName: workflow.name, totalNodes: workflow.nodes.count))
        // 1. Validate.
        do {
            try validate(workflow)
        } catch let e as WorkflowExecutorError {
            continuation.yield(.finished(success: false, message: String(describing: e)))
            return [:]
        } catch {
            continuation.yield(.finished(success: false, message: "\(error)"))
            return [:]
        }
        // 2. Topological sort.
        let order: [WorkflowNode]
        do {
            order = try topologicalSort(workflow)
        } catch let e as WorkflowExecutorError {
            continuation.yield(.finished(success: false, message: String(describing: e)))
            return [:]
        } catch {
            continuation.yield(.finished(success: false, message: "\(error)"))
            return [:]
        }
        // 3. Execute in order.
        var outputs: [String: [String: WorkflowPortValue]] = [:]
        var inputs: [String: [String: WorkflowPortValue]] = [:]
        for node in order {
            if Task.isCancelled {
                continuation.yield(.finished(success: false, message: "cancelled"))
                return outputs
            }
            guard let type = registry.nodeType(for: node.type) else {
                let msg = "internal: validator accepted \(node.type) but registry lost it"
                continuation.yield(.nodeFinished(nodeId: node.id, success: false, message: msg))
                continuation.yield(.finished(success: false, message: msg))
                return outputs
            }
            continuation.yield(.nodeStarted(nodeId: node.id, typeId: type.typeId))
            let nodeInputs = inputs[node.id] ?? [:]
            // Per-node scope: the logger is a fan-in from the
            // context logger + a node-id-tagged prefix so the
            // editor can route lines back to the right node.
            let nodeLogger = NodeTagLogger(
                base: context.logger, nodeId: node.id, sink: continuation)
            let nodeContext = WorkflowExecutionContext(
                fileSystem: context.fileSystem, logger: nodeLogger)
            do {
                let result = try await type.execute(
                    parameters: node.parameters,
                    inputs: nodeInputs,
                    context: nodeContext)
                outputs[node.id] = result
                // Hand the outputs to downstream consumers.
                let outgoing = workflow.edges.filter { $0.fromNode == node.id }
                for edge in outgoing {
                    if let value = result[edge.fromPort] {
                        inputs[edge.toNode, default: [:]][edge.toPort] = value
                    }
                }
                continuation.yield(.nodeFinished(nodeId: node.id, success: true, message: nil))
            } catch {
                let msg = "\(error)"
                continuation.yield(.nodeFinished(nodeId: node.id, success: false, message: msg))
                continuation.yield(.finished(success: false, message: "node \(node.id) failed: \(msg)"))
                return outputs
            }
        }
        continuation.yield(.finished(success: true, message: nil))
        return outputs
    }

    /// Validate the workflow: every node type is registered;
    /// every edge references declared ports with compatible
    /// types; the graph is acyclic. Throws on the first error.
    /// Cycle detection uses the same Kahn's algorithm as
    /// ``topologicalSort(_:)`` so what ``validate`` accepts
    /// ``topologicalSort`` will accept.
    public func validate(_ workflow: Workflow) throws {
        var declaredPorts: [String: (inputs: [String: WorkflowPort], outputs: [String: WorkflowPort])] = [:]
        for node in workflow.nodes {
            guard let type = registry.nodeType(for: node.type) else {
                throw WorkflowExecutorError.unknownNodeType(nodeId: node.id, typeId: node.type)
            }
            var inputsById: [String: WorkflowPort] = [:]
            for port in type.inputs { inputsById[port.id] = port }
            var outputsById: [String: WorkflowPort] = [:]
            for port in type.outputs { outputsById[port.id] = port }
            declaredPorts[node.id] = (inputsById, outputsById)
        }
        for edge in workflow.edges {
            guard let fromDecls = declaredPorts[edge.fromNode] else {
                throw WorkflowExecutorError.unknownPort(nodeId: edge.fromNode, portId: edge.fromPort, side: "output")
            }
            guard let toDecls = declaredPorts[edge.toNode] else {
                throw WorkflowExecutorError.unknownPort(nodeId: edge.toNode, portId: edge.toPort, side: "input")
            }
            guard let fromPort = fromDecls.outputs[edge.fromPort] else {
                throw WorkflowExecutorError.unknownPort(nodeId: edge.fromNode, portId: edge.fromPort, side: "output")
            }
            guard let toPort = toDecls.inputs[edge.toPort] else {
                throw WorkflowExecutorError.unknownPort(nodeId: edge.toNode, portId: edge.toPort, side: "input")
            }
            if !fromPort.type.canFlowInto(toPort.type) {
                throw WorkflowExecutorError.typeMismatch(
                    edge: edge, fromType: fromPort.type.rawValue, toType: toPort.type.rawValue)
            }
        }
        // Cycle check: run Kahn's and let it throw if anything
        // remains. The result is discarded; we only care that
        // the algorithm accepts the graph.
        _ = try topologicalSort(workflow)
    }

    /// Kahn's algorithm: peel off nodes with zero in-degree
    /// until the graph is empty. If any node remains, the
    /// graph has a cycle; throw with the first remaining node.
    public func topologicalSort(_ workflow: Workflow) throws -> [WorkflowNode] {
        var inDegree: [String: Int] = [:]
        var adjacency: [String: [String]] = [:]
        for node in workflow.nodes {
            inDegree[node.id] = 0
            adjacency[node.id] = []
        }
        for edge in workflow.edges {
            adjacency[edge.fromNode, default: []].append(edge.toNode)
            inDegree[edge.toNode, default: 0] += 1
        }
        var ready: [String] = workflow.nodes
            .filter { (inDegree[$0.id] ?? 0) == 0 }
            .map { $0.id }
        var result: [WorkflowNode] = []
        while let next = ready.first {
            ready.removeFirst()
            guard let node = workflow.nodes.first(where: { $0.id == next }) else {
                continue
            }
            result.append(node)
            for successor in adjacency[next] ?? [] {
                inDegree[successor, default: 0] -= 1
                if inDegree[successor] == 0 {
                    ready.append(successor)
                }
            }
        }
        if result.count != workflow.nodes.count {
            let remaining = workflow.nodes
                .filter { !result.contains(where: { $0.id == $0.id }) && (inDegree[$0.id] ?? 0) > 0 }
                .map { $0.id }
            throw WorkflowExecutorError.cycle(detectedAt: remaining.first ?? "?")
        }
        return result
    }
}

/// Logger that wraps a base logger + a node id, and also
/// forwards to an ``AsyncStream.Continuation`` so the editor
/// can render per-node log lines in real time.
private struct NodeTagLogger: WorkflowLogger {
    let base: any WorkflowLogger
    let nodeId: String
    let sink: AsyncStream<WorkflowEvent>.Continuation

    func log(_ message: String, level: WorkflowLogLevel) {
        base.log("[\(nodeId)] \(message)", level: level)
        sink.yield(.log(nodeId: nodeId, level: level, message: message))
    }
}
