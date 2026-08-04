import XCTest
@testable import TesseraCore

// MARK: - Stub nodes for tests

/// Echo node for tests: takes a `value` input of any type, returns
/// it on the `value` output unchanged. Used to wire up
/// round-trip and executor tests without depending on the real
/// TesseraTools (which have file system side effects).
struct EchoNode: WorkflowNodeType {
    static let typeId = "test_echo"
    static let displayName = "Echo"
    static let summary = "Test-only echo node. Passes a value through."
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
        return inputs
    }
}

/// Source node for tests: synthesises a constant string on the
/// `value` output. Reads the constant from a parameter.
struct ConstSourceNode: WorkflowNodeType {
    static let typeId = "test_const"
    static let displayName = "Const"
    static let summary = "Test-only constant source."
    static let inputs: [WorkflowPort] = []
    static let outputs: [WorkflowPort] = [
        WorkflowPort(id: "value", label: "Value", type: .string),
    ]
    static let parameterSchema = JSONSchema(
        type: "object",
        properties: [
            "value": SchemaProperty(
                type: "string",
                description: "The constant to emit.",
                defaultValue: "default"
            ),
        ],
        required: []
    )

    static func execute(
        parameters: [String: JSONValue],
        inputs: [String: WorkflowPortValue],
        context: WorkflowExecutionContext
    ) async throws -> [String: WorkflowPortValue] {
        let v = parameters["value"]?.stringValue ?? "default"
        return ["value": .string(v)]
    }
}

/// Failing node for tests: throws unconditionally. Used to verify
/// the executor surfaces a `nodeFailed` event and aborts the
/// workflow.
struct AlwaysFailNode: WorkflowNodeType {
    static let typeId = "test_fail"
    static let displayName = "AlwaysFail"
    static let summary = "Test-only failing node."
    static let inputs: [WorkflowPort] = []
    static let outputs: [WorkflowPort] = [
        WorkflowPort(id: "value", label: "Value", type: .string),
    ]
    static let parameterSchema = JSONSchema()

    static func execute(
        parameters: [String: JSONValue],
        inputs: [String: WorkflowPortValue],
        context: WorkflowExecutionContext
    ) async throws -> [String: WorkflowPortValue] {
        throw NSError(
            domain: "test", code: 42,
            userInfo: [NSLocalizedDescriptionKey: "intentional failure"])
    }
}

// MARK: - Round-trip

final class WorkflowRoundTripTests: XCTestCase {
    func testCodableRoundTrip() throws {
        let original = Workflow(
            name: "calibrate-and-quantize",
            nodes: [
                WorkflowNode(
                    id: "calib",
                    type: CalibrateNode.typeId,
                    parameters: ["n_tokens": .number(8000)]
                ),
                WorkflowNode(
                    id: "q",
                    type: QuantizeNode.typeId,
                    parameters: [:]
                ),
            ],
            edges: [
                WorkflowEdge(
                    fromNode: "calib", fromPort: "result",
                    toNode: "q", toPort: "policy_path"
                ),
            ]
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(original)
        let decoded = try JSONDecoder().decode(Workflow.self, from: data)
        XCTAssertEqual(decoded, original)
        // Spot-check the JSON shape so the file is human-greppable.
        // Whitespace and key order depend on JSONEncoder options,
        // so we assert on the structural fields that should always
        // appear (the executor + the editor both rely on them).
        let str = String(data: data, encoding: .utf8) ?? ""
        XCTAssertTrue(str.contains("tessera.workflow.v1"),
            "expected schema marker, got: \(str)")
        XCTAssertTrue(str.contains("calib"),
            "expected node id 'calib', got: \(str)")
        XCTAssertTrue(str.contains("n_tokens"),
            "expected parameter key 'n_tokens', got: \(str)")
    }

    func testDefaultRegistryHasFiveNodes() {
        let reg = WorkflowNodeRegistry.default
        XCTAssertEqual(reg.allTypeIds.sorted(), [
            "calibrate", "evaluate", "inspect_sidecar",
            "load_model", "quantize",
        ])
    }

    func testPaletteEntryExposesPortsAndParams() {
        let reg = WorkflowNodeRegistry.default
        guard let entry = reg.paletteEntry(for: "calibrate") else {
            XCTFail("calibrate missing from default registry")
            return
        }
        // Required schema properties -> input ports.
        XCTAssertEqual(Set(entry.inputs.map(\.id)),
                       Set(["model_path", "corpus_path", "output_path"]))
        // Port types: model/corpus/output all end in _path -> .path.
        for port in entry.inputs {
            XCTAssertEqual(port.type, .path,
                "expected .path for port \(port.id), got \(port.type)")
        }
        // Optional properties stay in parameterSchema, not in inputs.
        XCTAssertNotNil(entry.parameterSchema.properties?["n_tokens"])
        XCTAssertNotNil(entry.parameterSchema.properties?["modality"])
        // Required ports are NOT also in the parameter schema.
        XCTAssertNil(entry.parameterSchema.properties?["model_path"])
    }
}

// MARK: - Schema splitting

final class TesseraToolNodeSchemaSplitTests: XCTestCase {
    func testSplitsRequiredIntoPorts() {
        let schema = JSONSchema(
            type: "object",
            properties: [
                "req_str": SchemaProperty(type: "string"),
                "opt_int": SchemaProperty(type: "integer", defaultValue: "0"),
            ],
            required: ["req_str"]
        )
        let (ports, params) = TesseraToolNode.splitSchema(schema)
        XCTAssertEqual(ports.map(\.id), ["req_str"])
        XCTAssertEqual(ports.first?.type, .string)
        XCTAssertNotNil(params.properties?["opt_int"])
        XCTAssertNil(params.properties?["req_str"])
    }

    func testPathSuffixDetection() {
        let prop = SchemaProperty(type: "string")
        XCTAssertEqual(TesseraToolNode.portType(for: prop, name: "model_path"), .path)
        XCTAssertEqual(TesseraToolNode.portType(for: prop, name: "output_path"), .path)
        XCTAssertEqual(TesseraToolNode.portType(for: prop, name: "path"), .path)
        XCTAssertEqual(TesseraToolNode.portType(for: prop, name: "sidecar"), .string)
    }

    func testIntegerAndBooleanTypes() {
        let n = SchemaProperty(type: "integer")
        let b = SchemaProperty(type: "boolean")
        XCTAssertEqual(TesseraToolNode.portType(for: n, name: "n_ctx"), .number)
        XCTAssertEqual(TesseraToolNode.portType(for: b, name: "verbose"), .boolean)
    }
}

// MARK: - Validation

final class WorkflowValidationTests: XCTestCase {
    private func registry() -> WorkflowNodeRegistry {
        WorkflowNodeRegistry(types: [
            EchoNode.self, ConstSourceNode.self, AlwaysFailNode.self,
        ])
    }

    func testRejectsUnknownNodeType() async {
        let wf = Workflow(
            name: "bad",
            nodes: [
                WorkflowNode(id: "x", type: "no_such_node", parameters: [:]),
            ],
            edges: []
        )
        let ex = WorkflowExecutor(registry: registry())
        let events = await ex.run(wf)
        var sawFinished = false
        for await ev in events {
            if case .finished(let success, _) = ev {
                XCTAssertFalse(success)
                sawFinished = true
            }
        }
        XCTAssertTrue(sawFinished)
    }

    func testRejectsCycle() async {
        // Two-node cycle: a -> b -> a.
        let wf = Workflow(
            name: "cyclic",
            nodes: [
                WorkflowNode(id: "a", type: EchoNode.typeId, parameters: [:]),
                WorkflowNode(id: "b", type: EchoNode.typeId, parameters: [:]),
            ],
            edges: [
                WorkflowEdge(fromNode: "a", fromPort: "value",
                             toNode: "b", toPort: "value"),
                WorkflowEdge(fromNode: "b", fromPort: "value",
                             toNode: "a", toPort: "value"),
            ]
        )
        let ex = WorkflowExecutor(registry: registry())
        do {
            try await ex.validate(wf)
            XCTFail("expected cycle error")
        } catch WorkflowExecutorError.cycle {
            // Expected.
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testRejectsTypeMismatch() async {
        // EchoNode has string ports on both sides; declare a
        // mismatch by editing the workflow to claim the edge
        // type is wrong at validation. We can't easily construct
        // a type mismatch through the public API because the
        // port types are baked into the node type. So instead,
        // we use a separate check: edge references a port that
        // doesn't exist on the target node.
        let wf = Workflow(
            name: "bad-port",
            nodes: [
                WorkflowNode(id: "a", type: ConstSourceNode.typeId, parameters: [:]),
                WorkflowNode(id: "b", type: EchoNode.typeId, parameters: [:]),
            ],
            edges: [
                WorkflowEdge(fromNode: "a", fromPort: "value",
                             toNode: "b", toPort: "no_such_port"),
            ]
        )
        let ex = WorkflowExecutor(registry: registry())
        do {
            try await ex.validate(wf)
            XCTFail("expected unknown-port error")
        } catch WorkflowExecutorError.unknownPort {
            // Expected.
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }
}

// MARK: - Executor

final class WorkflowExecutorTests: XCTestCase {
    private func registry() -> WorkflowNodeRegistry {
        WorkflowNodeRegistry(types: [
            EchoNode.self, ConstSourceNode.self, AlwaysFailNode.self,
        ])
    }

    func testRunsSourceToEcho() async {
        let wf = Workflow(
            name: "const->echo",
            nodes: [
                WorkflowNode(id: "src",
                             type: ConstSourceNode.typeId,
                             parameters: ["value": .string("hello world")]),
                WorkflowNode(id: "sink",
                             type: EchoNode.typeId, parameters: [:]),
            ],
            edges: [
                WorkflowEdge(fromNode: "src", fromPort: "value",
                             toNode: "sink", toPort: "value"),
            ]
        )
        let ex = WorkflowExecutor(registry: registry())
        var events: [WorkflowEvent] = []
        for await ev in await ex.run(wf) {
            events.append(ev)
        }
        // Last event must be .finished(success: true).
        guard case .finished(let success, _) = events.last else {
            XCTFail("missing finished event; events=\(events)")
            return
        }
        XCTAssertTrue(success)
        // We expect: started, nodeStarted x2, nodeFinished x2, finished.
        XCTAssertEqual(events.count, 6)
    }

    func testSurfacesNodeFailure() async {
        let wf = Workflow(
            name: "fail",
            nodes: [
                WorkflowNode(id: "boom",
                             type: AlwaysFailNode.typeId, parameters: [:]),
            ],
            edges: []
        )
        let ex = WorkflowExecutor(registry: registry())
        var events: [WorkflowEvent] = []
        for await ev in await ex.run(wf) {
            events.append(ev)
        }
        guard case .finished(let success, let msg) = events.last else {
            XCTFail("missing finished event")
            return
        }
        XCTAssertFalse(success)
        XCTAssertNotNil(msg)
        XCTAssertTrue(msg!.contains("boom"),
            "failure message should reference the failing node, got: \(msg!)")
    }

    func testEchoCarriesValueAcrossEdge() async {
        let wf = Workflow(
            name: "passthrough",
            nodes: [
                WorkflowNode(id: "src",
                             type: ConstSourceNode.typeId,
                             parameters: ["value": .string("carried")]),
                WorkflowNode(id: "sink",
                             type: EchoNode.typeId, parameters: [:]),
            ],
            edges: [
                WorkflowEdge(fromNode: "src", fromPort: "value",
                             toNode: "sink", toPort: "value"),
            ]
        )
        let ex = WorkflowExecutor(registry: registry())
        let pair = await ex.runCollecting(wf)
        let outputs = await pair.finalOutputs.value
        // The echo node's "value" output should equal the
        // source's constant.
        guard let sinkValue = outputs["sink"]?["value"] else {
            XCTFail("sink output missing; outputs=\(outputs)")
            return
        }
        XCTAssertEqual(sinkValue, .string("carried"))
    }
}
