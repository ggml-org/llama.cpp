import XCTest
@testable import TesseraCore

/// Captures the argv of every `run` call and returns a canned result. Used
/// to assert what the engine tools shell out to without spawning a real
/// process. Each test stages a response via `enqueue`; if no result is
/// queued, the mock returns a zero-exit empty result. When the call's
/// argv contains a `--config <path>` pair, the mock also reads the file
/// and stashes the parsed JSON so the test can assert on it before the
/// tool's `defer` block deletes the temp file.
final class MockProcessShell: TesseraProcessShell, @unchecked Sendable {
    struct Call {
        let executable: String
        let arguments: [String]
        let environment: [String: String]?
        let workingDirectory: String?
        let configJSON: [String: Any]?
    }

    private let lock = NSLock()
    private var _calls: [Call] = []
    private var _nextResults: [ProcessResult] = []

    func enqueue(_ result: ProcessResult) {
        lock.withLock { _nextResults.append(result) }
    }

    var calls: [Call] {
        lock.withLock { _calls }
    }

    func run(
        executable: String,
        arguments: [String],
        environment: [String: String]?,
        workingDirectory: String?
    ) async throws -> ProcessResult {
        var configJSON: [String: Any]?
        if let i = arguments.firstIndex(of: "--config"), i + 1 < arguments.count {
            let path = arguments[i + 1]
            if let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                configJSON = obj
            }
        }
        lock.withLock {
            _calls.append(Call(
                executable: executable,
                arguments: arguments,
                environment: environment,
                workingDirectory: workingDirectory,
                configJSON: configJSON
            ))
        }
        return lock.withLock {
            if _nextResults.isEmpty {
                return ProcessResult(exitCode: 0, stdout: "", stderr: "")
            }
            return _nextResults.removeFirst()
        }
    }
}

final class EngineToolRetargetTests: XCTestCase {

    private var mockShell: MockProcessShell!
    private var fakeBinary: String!

    override func setUp() async throws {
        try await super.setUp()
        mockShell = MockProcessShell()
        fakeBinary = try makeExecutableTempFile()
        // Stage the binary via the settings key so the resolver finds it
        // regardless of override semantics.
        UserDefaults.standard.set(fakeBinary, forKey: TesseraSettingsKey.tesseraCLIPath)
    }

    override func tearDown() async throws {
        UserDefaults.standard.removeObject(forKey: TesseraSettingsKey.tesseraCLIPath)
        try? FileManager.default.removeItem(atPath: fakeBinary)
        mockShell = nil
        fakeBinary = nil
        try await super.tearDown()
    }

    // MARK: per-tool assertions

    func testQuantizeToolShellsOutToCLI() async throws {
        mockShell.enqueue(ProcessResult(exitCode: 0, stdout: "ok", stderr: ""))
        let tool = QuantizeTool(shell: mockShell)
        let result = try await tool.execute(arguments: [
            "model_path": .string("/tmp/m.gguf"),
            "output_path": .string("/tmp/m.tq.gguf"),
            "policy_path": .string("/tmp/policy.json"),
        ])
        XCTAssertTrue(result.success, "expected success, got \(result.error ?? result.output)")
        XCTAssertEqual(mockShell.calls.count, 1)
        let call = mockShell.calls[0]
        XCTAssertEqual(call.executable, fakeBinary)
        XCTAssertEqual(call.arguments[0], "quantize")
        XCTAssertEqual(call.arguments[1], "/tmp/m.gguf")
        XCTAssertEqual(call.arguments[2], "/tmp/m.tq.gguf")
        XCTAssertEqual(call.arguments[3], "--config")
        // Config was read by the mock at call time (before the tool's
        // `defer` block deletes the temp file).
        let config = try XCTUnwrap(call.configJSON)
        XCTAssertEqual(config["policy_path"] as? String, "/tmp/policy.json")
    }

    func testCalibrateToolShellsOutToCLI() async throws {
        mockShell.enqueue(ProcessResult(exitCode: 0, stdout: "ok", stderr: ""))
        let tool = CalibrateTool(shell: mockShell)
        let result = try await tool.execute(arguments: [
            "model_path": .string("/tmp/m.gguf"),
            "corpus_path": .string("/tmp/corpus"),
            "output_path": .string("/tmp/m.imatrix"),
            "n_tokens": .number(1000),
            "modality": .string("text"),
        ])
        XCTAssertTrue(result.success, "expected success, got \(result.error ?? result.output)")
        XCTAssertEqual(mockShell.calls.count, 1)
        let call = mockShell.calls[0]
        XCTAssertEqual(call.executable, fakeBinary)
        XCTAssertEqual(call.arguments[0], "calibrate")
        // corpus is the positional per the spec; model + output go in JSON.
        XCTAssertEqual(call.arguments[1], "/tmp/corpus")
        XCTAssertEqual(call.arguments[2], "--config")
        let config = try XCTUnwrap(call.configJSON)
        XCTAssertEqual(config["model_path"] as? String, "/tmp/m.gguf")
        XCTAssertEqual(config["output_path"] as? String, "/tmp/m.imatrix")
        XCTAssertEqual(config["n_tokens"] as? Int, 1000)
        XCTAssertEqual(config["modality"] as? String, "text")
    }

    func testEvolveToolShellsOutToCLI() async throws {
        mockShell.enqueue(ProcessResult(exitCode: 0, stdout: "ok", stderr: ""))
        let tool = EvolveTool(shell: mockShell)
        let result = try await tool.execute(arguments: [
            "model_path": .string("/tmp/m.gguf"),
            "imatrix_path": .string("/tmp/m.imatrix"),
            "output_path": .string("/tmp/policy.json"),
            "target_bits": .number(3.5),
        ])
        XCTAssertTrue(result.success, "expected success, got \(result.error ?? result.output)")
        XCTAssertEqual(mockShell.calls.count, 1)
        let call = mockShell.calls[0]
        XCTAssertEqual(call.executable, fakeBinary)
        XCTAssertEqual(call.arguments[0], "evolve")
        XCTAssertEqual(call.arguments[1], "/tmp/m.gguf")
        XCTAssertEqual(call.arguments[2], "--config")
    }

    func testEvaluateToolShellsOutToCLIAndParsesJSON() async throws {
        // Evaluate prints a JSON result on stdout; the tool should
        // surface it as the agent's data payload when it parses.
        let json = """
        {"perplexity": 7.2, "latency_ms": 18.3, "power_mw": 5400}
        """
        mockShell.enqueue(ProcessResult(exitCode: 0, stdout: json, stderr: ""))
        let tool = EvaluateTool(shell: mockShell)
        let result = try await tool.execute(arguments: [
            "model_path": .string("/tmp/m.gguf"),
            "n_tokens": .number(256),
        ])
        XCTAssertTrue(result.success, "expected success, got \(result.error ?? result.output)")
        XCTAssertEqual(mockShell.calls.count, 1)
        let call = mockShell.calls[0]
        XCTAssertEqual(call.executable, fakeBinary)
        XCTAssertEqual(call.arguments[0], "evaluate")
        XCTAssertEqual(call.arguments[1], "/tmp/m.gguf")
        // JSON-parsed fields are merged into the data payload.
        let data = try XCTUnwrap(result.data)
        XCTAssertEqual(data["perplexity"], JSONValue.number(7.2))
        XCTAssertEqual(data["latency_ms"], JSONValue.number(18.3))
        XCTAssertEqual(data["power_mw"], JSONValue.number(5400))
        // Stable fields the tool always sets.
        XCTAssertEqual(data["backend"], JSONValue.string("cli"))
    }

    func testConvertToolShellsOutToCLI() async throws {
        mockShell.enqueue(ProcessResult(exitCode: 0, stdout: "ok", stderr: ""))
        let tool = ConvertTool(shell: mockShell)
        let result = try await tool.execute(arguments: [
            "model_path": .string("/tmp/m.gguf"),
            "output_path": .string("/tmp/m.mlmodelc"),
            "compute_units": .string("cpuAndNeuralEngine"),
            "precision": .string("float16"),
        ])
        XCTAssertTrue(result.success, "expected success, got \(result.error ?? result.output)")
        XCTAssertEqual(mockShell.calls.count, 1)
        let call = mockShell.calls[0]
        XCTAssertEqual(call.executable, fakeBinary)
        XCTAssertEqual(call.arguments[0], "convert")
        XCTAssertEqual(call.arguments[1], "/tmp/m.gguf")
        XCTAssertEqual(call.arguments[2], "/tmp/m.mlmodelc")
        XCTAssertEqual(call.arguments[3], "--format")
        XCTAssertEqual(call.arguments[4], "coreml")
        XCTAssertEqual(call.arguments[5], "--config")
    }

    func testInspectSidecarToolShellsOutToCLI() async throws {
        let sidecarPath = try writeTempSidecar()
        defer { try? FileManager.default.removeItem(atPath: sidecarPath) }
        let sidecarJSON = """
        {"schema_version": 1, "tessera_profile": "tq-3.5",
         "effective_bits": 3.5, "kernel_version": "v1"}
        """
        mockShell.enqueue(ProcessResult(exitCode: 0, stdout: sidecarJSON, stderr: ""))
        let tool = InspectSidecarTool(shell: mockShell)
        let result = try await tool.execute(arguments: ["path": .string(sidecarPath)])
        XCTAssertTrue(result.success, "expected success, got \(result.error ?? result.output)")
        XCTAssertEqual(mockShell.calls.count, 1)
        let call = mockShell.calls[0]
        XCTAssertEqual(call.executable, fakeBinary)
        XCTAssertEqual(call.arguments[0], "inspect-sidecar")
        XCTAssertEqual(call.arguments[1], sidecarPath)
        let data = try XCTUnwrap(result.data)
        XCTAssertEqual(data["tessera_profile"], JSONValue.string("tq-3.5"))
        XCTAssertEqual(data["effective_bits"], JSONValue.number(3.5))
    }

    func testListModelsToolShellsOutToCLIAndParsesArray() async throws {
        // The CLI is expected to print a JSON array of {name, kind, size}
        // objects; the tool should surface the count + sorted list.
        let json = """
        [{"name":"a.gguf","kind":"GGUF","size":1234},
         {"name":"b.mlmodelc","kind":"CoreML","size":5678}]
        """
        mockShell.enqueue(ProcessResult(exitCode: 0, stdout: json, stderr: ""))
        let tool = ListModelsTool(shell: mockShell)
        let result = try await tool.execute(arguments: [
            "directory": .string(NSTemporaryDirectory()),
        ])
        XCTAssertTrue(result.success, "expected success, got \(result.error ?? result.output)")
        XCTAssertEqual(mockShell.calls.count, 1)
        let call = mockShell.calls[0]
        XCTAssertEqual(call.executable, fakeBinary)
        XCTAssertEqual(call.arguments[0], "list-models")
        let data = try XCTUnwrap(result.data)
        XCTAssertEqual(data["count"], JSONValue.number(2))
        XCTAssertEqual(data["backend"], JSONValue.string("cli"))
    }

    func testLoadModelToolDoesNotInvokeShell() async throws {
        let modelPath = try writeTempModel()
        defer { try? FileManager.default.removeItem(atPath: modelPath) }
        let tool = LoadModelTool()
        let result = try await tool.execute(arguments: [
            "model_path": .string(modelPath),
            "n_ctx": .number(2048),
        ])
        XCTAssertTrue(result.success, "expected success, got \(result.error ?? result.output)")
        // load_model is a Swift state op, not a subprocess.
        XCTAssertEqual(mockShell.calls.count, 0)
        let data = try XCTUnwrap(result.data)
        XCTAssertEqual(data["status"], JSONValue.string("loaded"))
        XCTAssertEqual(data["n_ctx"], JSONValue.number(2048))
    }

    // MARK: failure surface

    func testToolReturnsBinaryNotFoundWhenResolverFails() async throws {
        // Override the settings with a path that does not exist; the
        // resolver returns nil and the tool should surface a clear error
        // that names the checked locations. Clear `knownLocations` so
        // a developer-machine install of tessera-cli at e.g.
        // ~/Developer/GitHub/tessera/build/bin/tessera-cli doesn't
        // short-circuit the test (the same hook the W3 worker exposed
        // for the resolver's own unit tests).
        let originalLocations = TesseraCLIBinaryResolver.knownLocations
        TesseraCLIBinaryResolver.knownLocations = []
        defer { TesseraCLIBinaryResolver.knownLocations = originalLocations }

        UserDefaults.standard.set("/nope/tessera-cli", forKey: TesseraSettingsKey.tesseraCLIPath)
        let tool = QuantizeTool(shell: mockShell)
        let result = try await tool.execute(arguments: [
            "model_path": .string("/tmp/m.gguf"),
            "output_path": .string("/tmp/m.tq.gguf"),
            "policy_path": .string("/tmp/policy.json"),
        ])
        XCTAssertFalse(result.success, "expected failure when binary is missing")
        let err = try XCTUnwrap(result.error)
        XCTAssertTrue(err.contains("tessera-cli"), "msg should name the binary: \(err)")
        XCTAssertEqual(mockShell.calls.count, 0, "no subprocess should be spawned")
    }

    // MARK: helpers

    private func makeExecutableTempFile() throws -> String {
        let dir = NSTemporaryDirectory()
        let path = (dir as NSString).appendingPathComponent("tessera-cli-fake-\(UUID().uuidString)")
        try "#!/bin/sh\nexit 0\n".write(toFile: path, atomically: true, encoding: .utf8)
        var attrs = try FileManager.default.attributesOfItem(atPath: path)
        attrs[.posixPermissions] = 0o755
        try FileManager.default.setAttributes(attrs, ofItemAtPath: path)
        return path
    }

    private func writeTempSidecar() throws -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("sidecar-\(UUID().uuidString).json")
        try "{}".write(toFile: path, atomically: true, encoding: .utf8)
        return path
    }

    private func writeTempModel() throws -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("model-\(UUID().uuidString).gguf")
        try "GGUF".write(toFile: path, atomically: true, encoding: .utf8)
        return path
    }
}
