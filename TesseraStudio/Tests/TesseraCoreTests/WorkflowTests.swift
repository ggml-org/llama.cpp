import XCTest
import UniformTypeIdentifiers
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

// MARK: - Position math (used by the canvas hit-test)

final class WorkflowPositionMathTests: XCTestCase {
    /// Phase 2.3 makes the canvas's port-center math visible
    /// outside the canvas (so the drop test can find the
    /// target port). This test pins the math so the canvas
    /// and the drop test can't drift.
    func testPortCenterMatchesExpectedRight() {
        let p = WorkflowGeometry.portCenter(
            nodeCenter: CGPoint(x: 200, y: 200),
            portIndex: 0,
            isLeft: false,
            portCount: 3
        )
        // nodeWidth is 200; xOffset for right = 200-14=186;
        // nodeCenter.x + 186 - 100 = 286.
        XCTAssertEqual(p.x, 286, accuracy: 0.5)
        // nodeHeight(3) = 40 + 8 + 60 + 8 = 116; yOffset for
        // the first port = 40 + 8 + 0 + 10 = 58; nodeCenter.y
        // + 58 - 58 = 200.
        XCTAssertEqual(p.y, 200, accuracy: 0.5)
    }

    func testPortCenterMatchesExpectedLeft() {
        let p = WorkflowGeometry.portCenter(
            nodeCenter: CGPoint(x: 200, y: 200),
            portIndex: 0,
            isLeft: true,
            portCount: 3
        )
        // xOffset for left = 14; nodeCenter.x + 14 - 100 = 114.
        XCTAssertEqual(p.x, 114, accuracy: 0.5)
    }

    func testNodeHeightScalesWithPorts() {
        XCTAssertEqual(WorkflowGeometry.nodeHeight(portCount: 0), 56, accuracy: 0.5)
        XCTAssertEqual(WorkflowGeometry.nodeHeight(portCount: 1), 76, accuracy: 0.5)
        XCTAssertEqual(WorkflowGeometry.nodeHeight(portCount: 5), 156, accuracy: 0.5)
    }
}

// MARK: - Custom UTI contract (HIG 1.4 / 2.1)

/// The `com.tessera.workflow` UTI is the contract between
/// `WorkflowDocument` (Mac), `fileExporter` / `fileImporter`
/// modifiers in `WorkflowsView`, the Info.plist's
/// `UTExportedTypeDeclarations`, and Launch Services. These
/// tests pin the Swift-side declaration so the constant can't
/// drift from the Info.plist entry (`com.tessera.workflow`,
/// extension `tessera-workflow`).
///
/// Conformance queries (`conforms(to: .json)`) resolve through
/// Launch Services, which only knows the type once the app
/// bundle is registered — SwiftPM tests run without the bundle,
/// so conformance is declared in the Info.plist and covered by
/// a launch-time / manual Finder check instead of unit tests.
final class TesseraWorkflowUTTypeTests: XCTestCase {
    func testIdentifierMatchesInfoPlistDeclaration() {
        // Must match UTTypeIdentifier in
        // Support/Mac/Info.plist UTExportedTypeDeclarations.
        XCTAssertEqual(UTType.tesseraWorkflow.identifier, "com.tessera.workflow")
    }

    func testIsNotJSONItself() {
        // Sanity: the custom type must NOT be the same as
        // plain public.json — that would mean the declaration
        // collapsed to no UTI at all and file pickers would
        // show every JSON file as a workflow.
        XCTAssertNotEqual(UTType.tesseraWorkflow.identifier, UTType.json.identifier)
    }
}

/// `WorkflowRunOutcome` is the single structured result of a
/// run. These tests pin the parse boundary: terminal `.finished`
/// events map to exactly one outcome case, non-terminal events
/// yield nil (the run is still going), and `cancelled` counts
/// only the nodes that actually finished successfully.
final class WorkflowRunOutcomeTests: XCTestCase {
    func testFinishedSuccessParsesToSucceeded() {
        let event = WorkflowEvent.finished(success: true, message: nil)
        let outcome = WorkflowRunOutcome(finishedEvent: event)
        XCTAssertEqual(outcome, .succeeded(summary: nil))
        XCTAssertTrue(outcome?.isSucceeded ?? false)
    }

    func testFinishedSuccessCarriesSummary() {
        let event = WorkflowEvent.finished(success: true, message: "done")
        XCTAssertEqual(WorkflowRunOutcome(finishedEvent: event), .succeeded(summary: "done"))
    }

    func testFinishedFailureParsesToFailed() {
        let event = WorkflowEvent.finished(success: false, message: "node q failed: boom")
        XCTAssertEqual(
            WorkflowRunOutcome(finishedEvent: event),
            .failed(message: "node q failed: boom")
        )
    }

    func testNonTerminalEventYieldsNil() {
        // A mid-run event must not be mistaken for a terminal
        // outcome — the editor keeps waiting in that case.
        XCTAssertNil(WorkflowRunOutcome(finishedEvent: .started(workflowName: "w", totalNodes: 2)))
        XCTAssertNil(WorkflowRunOutcome(finishedEvent: .nodeStarted(nodeId: "a", typeId: "t")))
        XCTAssertNil(WorkflowRunOutcome(finishedEvent: .nodeFinished(nodeId: "a", success: true, message: nil)))
        XCTAssertNil(WorkflowRunOutcome(finishedEvent: .log(nodeId: nil, level: .info, message: "hi")))
    }

    func testCancelledCountsOnlySuccessfulNodes() {
        let events: [WorkflowEvent] = [
            .started(workflowName: "w", totalNodes: 3),
            .nodeStarted(nodeId: "a", typeId: "t"),
            .nodeFinished(nodeId: "a", success: true, message: nil),
            .nodeStarted(nodeId: "b", typeId: "t"),
            .nodeFinished(nodeId: "b", success: false, message: "boom"),
        ]
        XCTAssertEqual(WorkflowRunOutcome.cancelled(events: events), .cancelled(completedNodes: 1))
    }

    func testCancelledOnEmptyTrailIsZero() {
        XCTAssertEqual(WorkflowRunOutcome.cancelled(events: []), .cancelled(completedNodes: 0))
    }
}

/// Notification copy is a pure function of the outcome, so the
/// notifier cannot drift from what the run sheet reports.
final class WorkflowRunNotificationContentTests: XCTestCase {
    func testSucceededCopy() {
        let content = WorkflowRunNotificationContent(
            outcome: .succeeded(summary: nil), workflowName: "calibrate-and-quantize"
        )
        XCTAssertEqual(content.title, "Workflow finished")
        XCTAssertEqual(content.body, "\"calibrate-and-quantize\" completed successfully.")
    }

    func testFailedCopyCarriesMessage() {
        let content = WorkflowRunNotificationContent(
            outcome: .failed(message: "node q failed: out of disk"), workflowName: "w"
        )
        XCTAssertEqual(content.title, "Workflow failed")
        XCTAssertEqual(content.body, "\"w\": node q failed: out of disk")
    }

    func testFailedCopyWithoutMessage() {
        let content = WorkflowRunNotificationContent(
            outcome: .failed(message: nil), workflowName: "w"
        )
        XCTAssertEqual(content.body, "\"w\" did not complete.")
    }

    func testCancelledCopyReportsProgress() {
        let content = WorkflowRunNotificationContent(
            outcome: .cancelled(completedNodes: 2), workflowName: "w"
        )
        XCTAssertEqual(content.title, "Workflow cancelled")
        XCTAssertEqual(content.body, "\"w\" stopped after 2 node(s).")
    }
}

// MARK: - Numeric parameter fields (HIG T1-4)

/// The parameter panel renders integer/number schema properties as
/// text fields, but must STORE numeric JSONValues - a node's
/// `samples = 100` has to reach the executor as 100, not "100".
/// These tests drive the same write-through rule the panel applies
/// on every keystroke (`WorkflowNumericInput.edit`).
final class WorkflowNumericParameterTests: XCTestCase {
    /// Apply one text-field edit the way the panel does.
    private func apply(
        _ text: String, integer: Bool,
        to parameters: inout [String: JSONValue],
        key: String = "samples"
    ) {
        switch WorkflowNumericInput.edit(text: text, integer: integer) {
        case .clear:
            parameters.removeValue(forKey: key)
        case .store(let number):
            parameters[key] = .number(number)
        case .reject:
            break
        }
    }

    func testIntegerTextStoresNumberNotString() {
        var parameters: [String: JSONValue] = [:]
        apply("100", integer: true, to: &parameters)
        XCTAssertEqual(parameters["samples"], .number(100))
        if case .string = parameters["samples"] {
            XCTFail("integer field stored a JSON string")
        }
    }

    func testNumberTextKeepsDecimals() {
        var parameters: [String: JSONValue] = [:]
        apply("4.5", integer: false, to: &parameters)
        XCTAssertEqual(parameters["samples"], .number(4.5))
    }

    func testIntegerTextRoundsToWhole() {
        var parameters: [String: JSONValue] = [:]
        apply("4.7", integer: true, to: &parameters)
        XCTAssertEqual(parameters["samples"], .number(5))
    }

    func testMidTypingKeepsPreviousValue() {
        var parameters: [String: JSONValue] = ["samples": .number(100)]
        // "-" does not parse yet; the stored value must survive
        // the in-progress edit.
        apply("-", integer: true, to: &parameters)
        XCTAssertEqual(parameters["samples"], .number(100))
        // And the in-progress text is not clobbered by a re-sync.
        XCTAssertNil(WorkflowNumericInput.syncedText(
            current: "-", value: parameters["samples"], isEditing: true))
    }

    func testClearRemovesTheKey() {
        var parameters: [String: JSONValue] = ["samples": .number(100)]
        apply("", integer: true, to: &parameters)
        XCTAssertNil(parameters["samples"])
    }

    func testDocumentWithNumericParameterLoadsAndDisplays() throws {
        let json = """
        {
          "schema": "tessera.workflow.v1",
          "name": "w",
          "nodes": [
            { "id": "calib", "type": "calibrate",
              "parameters": { "n_tokens": 8000 } }
          ],
          "edges": []
        }
        """
        let wf = try JSONDecoder().decode(Workflow.self, from: Data(json.utf8))
        let stored = wf.nodes.first?.parameters["n_tokens"]
        XCTAssertEqual(stored, .number(8000))
        // The panel shows integral numbers without a trailing ".0".
        XCTAssertEqual(WorkflowNumericInput.displayText(for: stored), "8000")
    }

    func testLegacyStringNumberStillDisplays() {
        // Documents saved by the old string binding keep working:
        // shown as-is, healed to .number on the next valid edit.
        let legacy = JSONValue.string("100")
        XCTAssertEqual(WorkflowNumericInput.displayText(for: legacy), "100")
        var parameters: [String: JSONValue] = ["samples": legacy]
        apply("200", integer: true, to: &parameters)
        XCTAssertEqual(parameters["samples"], .number(200))
    }

    func testExternalChangeSyncsWhenNotEditing() {
        XCTAssertEqual(
            WorkflowNumericInput.syncedText(
                current: "100", value: .number(250), isEditing: false),
            "250")
        // Nothing changed -> no rewrite.
        XCTAssertNil(WorkflowNumericInput.syncedText(
            current: "250", value: .number(250), isEditing: false))
    }
}

// MARK: - Stepper pairing for bounded integers (HIG 13.16)

/// A bounded integer gets a Stepper paired with its text field;
/// open or half-known ranges stay a plain numeric field. The
/// decision lives in `WorkflowNumericInput` so it is testable
/// without rendering the panel.
final class WorkflowNumericStepperTests: XCTestCase {
    func testFullyBoundedIntegerGetsStepper() {
        let prop = SchemaProperty(type: "integer", minimum: 1, maximum: 100)
        XCTAssertTrue(WorkflowNumericInput.usesStepper(for: prop))
        XCTAssertEqual(WorkflowNumericInput.stepperBounds(for: prop), 1...100)
    }

    func testUnboundedIntegerStaysTextField() {
        XCTAssertFalse(WorkflowNumericInput.usesStepper(
            for: SchemaProperty(type: "integer")))
        // A half-known range is not bounded enough for a stepper.
        XCTAssertFalse(WorkflowNumericInput.usesStepper(
            for: SchemaProperty(type: "integer", minimum: 1)))
        XCTAssertFalse(WorkflowNumericInput.usesStepper(
            for: SchemaProperty(type: "integer", maximum: 100)))
    }

    func testNumberAndStringNeverGetStepper() {
        XCTAssertFalse(WorkflowNumericInput.usesStepper(
            for: SchemaProperty(type: "number", minimum: 0, maximum: 1)))
        XCTAssertFalse(WorkflowNumericInput.usesStepper(
            for: SchemaProperty(type: "string", minimum: 0, maximum: 1)))
    }

    func testInvertedBoundsYieldNoStepper() {
        XCTAssertNil(WorkflowNumericInput.stepperBounds(
            for: SchemaProperty(type: "integer", minimum: 10, maximum: 1)))
    }

    func testSchemaPropertyBoundsRoundTrip() throws {
        let prop = SchemaProperty(type: "integer", minimum: 0, maximum: 64)
        let data = try JSONEncoder().encode(prop)
        let decoded = try JSONDecoder().decode(SchemaProperty.self, from: data)
        XCTAssertEqual(decoded.minimum, 0)
        XCTAssertEqual(decoded.maximum, 64)
        // Documents written before the keys existed decode with
        // nil bounds.
        let old = try JSONDecoder().decode(
            SchemaProperty.self, from: Data(#"{"type": "integer"}"#.utf8))
        XCTAssertNil(old.minimum)
        XCTAssertNil(old.maximum)
    }

    func testWrappedToolSchemasAnnotateObviousMinimums() {
        // Sample counts floor at 1...
        XCTAssertEqual(CalibrateTool().parameters.properties?["n_tokens"]?.minimum, 1)
        XCTAssertEqual(EvaluateTool().parameters.properties?["n_tokens"]?.minimum, 1)
        // ...and thread counts at 0 (0 = all cores).
        XCTAssertEqual(QuantizeTool().parameters.properties?["n_threads"]?.minimum, 0)
        // The annotations survive the node schema split.
        XCTAssertEqual(CalibrateNode.parameterSchema.properties?["n_tokens"]?.minimum, 1)
    }
}
