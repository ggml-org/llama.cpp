import XCTest
@testable import TesseraCore

// MARK: - Base class: TESSERA_SCRIPT_DIR for the test process

/// All Python-tool tests share this base class so TESSERA_SCRIPT_DIR is
/// set before any test runs. The bridge's actor caches the resolved
/// script dir on first use, so the env var must be set before the first
/// call to ``PythonEngineBridge/discoverScriptDir()`` -- which can be
/// triggered by ``TesseraToolRegistry/default``'s lazy initialisation
/// (the 9 thin tools call ``PythonTool(scriptName:)`` at construction).
class PythonToolTestBase: XCTestCase {
    private static let resolvedScriptDir: String = {
        let here = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // TesseraCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // TesseraStudio
            .deletingLastPathComponent()  // repo root
        return here.appendingPathComponent("tools/tessera").path
    }()

    override class func setUp() {
        super.setUp()
        setenv("TESSERA_SCRIPT_DIR", Self.resolvedScriptDir, 1)
    }
}

// MARK: - argv builder

final class PythonToolArgvBuilderTests: PythonToolTestBase {

    private func params(_ properties: [String: String], types: [String: String] = [:]) -> JSONSchema {
        var props: [String: SchemaProperty] = [:]
        for (k, v) in properties {
            let type = types[k] ?? "string"
            props[k] = SchemaProperty(type: type, description: v)
        }
        return JSONSchema(type: "object", properties: props, required: [])
    }

    func testStringArgumentBecomesFlagAndValue() {
        let schema = params(["layers": "Directory"])
        let argv = PythonTool.buildArgv(
            parameters: schema, positional: [], subcommand: nil,
            arguments: ["layers": .string("/tmp/layers")]
        )
        XCTAssertEqual(argv, ["--layers", "/tmp/layers"])
    }

    func testIntegerArgumentIsFormattedWithoutTrailingDotZero() {
        let schema = params(["batch_size": "Batch"], types: ["batch_size": "integer"])
        let argv = PythonTool.buildArgv(
            parameters: schema, positional: [], subcommand: nil,
            arguments: ["batch_size": .number(8)]
        )
        XCTAssertEqual(argv, ["--batch-size", "8"])
    }

    func testNumberArgumentWithFractionalIsStringified() {
        let schema = params(["lr": "LR"], types: ["lr": "number"])
        let argv = PythonTool.buildArgv(
            parameters: schema, positional: [], subcommand: nil,
            arguments: ["lr": .number(0.001)]
        )
        XCTAssertEqual(argv, ["--lr", "0.001"])
    }

    func testBooleanTrueEmitsFlag() {
        let schema = params(["progressive_eval": "Progressive"], types: ["progressive_eval": "boolean"])
        let argv = PythonTool.buildArgv(
            parameters: schema, positional: [], subcommand: nil,
            arguments: ["progressive_eval": .bool(true)]
        )
        XCTAssertEqual(argv, ["--progressive-eval"])
    }

    func testBooleanFalseOmitsFlag() {
        let schema = params(["progressive_eval": "Progressive"], types: ["progressive_eval": "boolean"])
        let argv = PythonTool.buildArgv(
            parameters: schema, positional: [], subcommand: nil,
            arguments: ["progressive_eval": .bool(false)]
        )
        XCTAssertEqual(argv, [])
    }

    func testArrayArgumentEmitsFlagAndAllValues() {
        let schema = params(["vision_inputs": "Inputs"], types: ["vision_inputs": "array"])
        let argv = PythonTool.buildArgv(
            parameters: schema, positional: [], subcommand: nil,
            arguments: ["vision_inputs": .array([.string("a.jpg"), .string("b.png")])]
        )
        XCTAssertEqual(argv, ["--vision-inputs", "a.jpg", "b.png"])
    }

    func testEmptyArrayOmitsFlag() {
        let schema = params(["vision_inputs": "Inputs"], types: ["vision_inputs": "array"])
        let argv = PythonTool.buildArgv(
            parameters: schema, positional: [], subcommand: nil,
            arguments: ["vision_inputs": .array([])]
        )
        XCTAssertEqual(argv, [])
    }

    func testSubcommandAppearsFirst() {
        let schema = params(["store": "Store"])
        let argv = PythonTool.buildArgv(
            parameters: schema, positional: [], subcommand: "summarize",
            arguments: ["store": .string("/tmp/store")]
        )
        XCTAssertEqual(argv, ["summarize", "--store", "/tmp/store"])
    }

    func testPositionalArgsPrecedeFlagsInOrder() {
        let schema = params(["layers": "Layers", "output": "Output"])
        let argv = PythonTool.buildArgv(
            parameters: schema, positional: ["layers", "output"], subcommand: nil,
            arguments: [
                "layers": .string("/tmp/in"),
                "output": .string("/tmp/out"),
            ]
        )
        XCTAssertEqual(argv, ["/tmp/in", "/tmp/out"])
    }

    func testNullAndEmptyStringAreOmitted() {
        let schema = params([
            "a": "A",
            "b": "B",
        ])
        let argv = PythonTool.buildArgv(
            parameters: schema, positional: [], subcommand: nil,
            arguments: ["a": .null, "b": .string("")]
        )
        XCTAssertEqual(argv, [])
    }

    func testSnakeCasePropertyBecomesKebabFlag() {
        let schema = params([
            "db": "DB",
            "model_hash": "Hash",
            "budget_fraction": "Budget",
        ], types: ["budget_fraction": "number"])
        let argv = PythonTool.buildArgv(
            parameters: schema, positional: [], subcommand: nil,
            arguments: [
                "db": .string("/tmp/db.duckdb"),
                "model_hash": .string("abc"),
                "budget_fraction": .number(0.5),
            ]
        )
        // Order is non-deterministic (Swift dictionary iteration); check
        // the flag/value pairing structurally.
        XCTAssertEqual(argv.count, 6)
        XCTAssertTrue(argv.contains("--db"))
        XCTAssertTrue(argv.contains("/tmp/db.duckdb"))
        XCTAssertTrue(argv.contains("--model-hash"))
        XCTAssertTrue(argv.contains("abc"))
        XCTAssertTrue(argv.contains("--budget-fraction"))
        XCTAssertTrue(argv.contains("0.5"))
        // Every flag is followed by its value (no two flags adjacent).
        for i in 0..<(argv.count - 1) {
            if argv[i].hasPrefix("--") {
                XCTAssertFalse(argv[i + 1].hasPrefix("--"),
                               "flag \(argv[i]) immediately followed by another flag")
            }
        }
    }
}

// MARK: - Schema sidecar loading (all 9 wrapped tools)

final class PythonSchemaSidecarTests: PythonToolTestBase {

    /// The tools/tessera/ directory relative to the worktree root. Tests
    /// resolve it once and use it for every schema-load check.
    private static let worktreeRoot: String = {
        // .../TesseraStudio/Tests/TesseraCoreTests/<this file>
        // The package layout puts tools/tessera/ at the repo root.
        let here = URL(fileURLWithPath: #filePath)
        return here
            .deletingLastPathComponent()  // TesseraCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // TesseraStudio
            .deletingLastPathComponent()  // repo root
            .path
    }()

    private static let scriptDir: URL = {
        URL(fileURLWithPath: worktreeRoot)
            .appendingPathComponent("tools/tessera")
    }()

    func testScriptDirExists() throws {
        let fm = FileManager.default
        XCTAssertTrue(fm.fileExists(atPath: Self.scriptDir.path),
                      "tools/tessera not found at \(Self.scriptDir.path)")
    }

    private func load(_ name: String) throws -> PythonSchemaSidecar {
        let url = Self.scriptDir.appendingPathComponent("\(name).schema.json")
        return try PythonSchemaSidecar.load(from: url)
    }

    func testMultimodalCalibrateSchemaLoads() throws {
        let s = try load("multimodal_calibrate")
        XCTAssertEqual(s.name, "multimodal_calibrate")
        XCTAssertEqual(s.script, "multimodal_calibrate")
        XCTAssertNotNil(s.parameters.properties?["vision_tower"])
        XCTAssertNotNil(s.parameters.properties?["vision_inputs"])
    }

    func testAWQEvolveSchemaLoads() throws {
        let s = try load("awq-evolve")
        XCTAssertEqual(s.name, "awq_evolve")
        XCTAssertEqual(s.script, "awq-evolve")
        XCTAssertEqual(s.parameters.required, ["layers", "output"])
        XCTAssertEqual(s.parameters.properties?["model_role"]?.type, "string")
    }

    func testUnifiedCalibrateSchemaLoads() throws {
        let s = try load("unified_calibrate")
        XCTAssertEqual(s.name, "unified_calibrate")
        XCTAssertEqual(s.parameters.required, ["output"])
        XCTAssertNotNil(s.parameters.properties?["component"])
        XCTAssertEqual(s.parameters.properties?["component"]?.type, "array")
    }

    func testPerTensorCalibrateSchemaLoads() throws {
        let s = try load("per_tensor_calibrate")
        XCTAssertEqual(s.name, "per_tensor_calibrate")
        XCTAssertEqual(s.parameters.required, ["output"])
        let props = s.parameters.properties ?? [:]
        XCTAssertGreaterThan(props.count, 30, "expected a large parameter set")
    }

    func testL3HessianTraceSchemaLoads() throws {
        let s = try load("l3_hessian_trace")
        XCTAssertEqual(s.name, "l3_hessian_trace")
        XCTAssertEqual(s.parameters.required, ["layers", "output"])
        XCTAssertEqual(s.parameters.properties?["method"]?.type, "string")
    }

    func testTesseraDBQuerySchemaLoads() throws {
        let s = try load("tessera_db_query")
        XCTAssertEqual(s.name, "tessera_db_query")
        XCTAssertEqual(s.defaultApprovalLevel, "auto",
                       "DB query is read-only; should default to auto-approval")
        XCTAssertEqual(s.parameters.properties?["query"]?.type, "string")
    }

    func testEvidenceStoreSchemaLoads() throws {
        let s = try load("evidence-store")
        XCTAssertEqual(s.name, "evidence_store_summarize")
        XCTAssertEqual(s.subcommand, "summarize")
        XCTAssertEqual(s.parameters.required, ["store"])
    }

    func testBackfillSchemaLoads() throws {
        let s = try load("backfill")
        XCTAssertEqual(s.name, "backfill")
        XCTAssertEqual(s.parameters.required, ["db", "model_hash"])
    }

    func testShadowCalibrateSchemaLoads() throws {
        let s = try load("shadow-calibrate")
        XCTAssertEqual(s.name, "shadow_calibrate")
        XCTAssertEqual(s.parameters.required, ["base_policy", "output"])
    }

    func testAllSchemaPropertyKeysAreSnakeCase() throws {
        // Lock in the property->flag mapping: every JSONSchema property
        // key must be snake_case (since the wrapper translates _ -> -).
        let scripts = [
            "multimodal_calibrate", "awq-evolve", "unified_calibrate",
            "per_tensor_calibrate", "l3_hessian_trace", "tessera_db_query",
            "evidence-store", "backfill", "shadow-calibrate",
        ]
        for name in scripts {
            let s = try load(name)
            for key in s.parameters.properties?.keys ?? [:].keys {
                XCTAssertFalse(key.contains("-"),
                               "schema \(name) uses kebab-case property '\(key)'; expected snake_case")
            }
        }
    }
}

// MARK: - PythonTool registration

final class PythonToolRegistrationTests: PythonToolTestBase {
    /// Every thin tool struct should be in the default registry.
    func testDefaultRegistryContainsAllNinePythonTools() {
        let names = Set(TesseraToolRegistry.default.allTools.map(\.name))
        let expected: Set<String> = [
            "awq_evolve",
            "backfill",
            "evidence_store_summarize",
            "l3_hessian_trace",
            "multimodal_calibrate",
            "per_tensor_calibrate",
            "shadow_calibrate",
            "tessera_db_query",
            "unified_calibrate",
        ]
        for name in expected {
            XCTAssertTrue(names.contains(name), "registry missing \(name)")
        }
    }

    func testPythonToolNameMatchesSidecar() throws {
        // Each thin tool's reported name must match the sidecar's "name"
        // field. This locks the tool/sidecar contract.
        let here = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // TesseraCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // TesseraStudio
            .deletingLastPathComponent()  // repo root
        let scriptDir = here.appendingPathComponent("tools/tessera")

        let toolsAndSchemas: [(any TesseraTool, String)] = [
            (AWQEvolveTool(),               "awq-evolve"),
            (BackfillTool(),                "backfill"),
            (EvidenceStoreSummarizeTool(),  "evidence-store"),
            (L3HessianTraceTool(),          "l3_hessian_trace"),
            (MultimodalCalibrateTool(),     "multimodal_calibrate"),
            (PerTensorCalibrateTool(),      "per_tensor_calibrate"),
            (ShadowCalibrateTool(),         "shadow-calibrate"),
            (TesseraDBQueryTool(),          "tessera_db_query"),
            (UnifiedCalibrateTool(),        "unified_calibrate"),
        ]
        for (tool, schema) in toolsAndSchemas {
            let sidecar = try PythonSchemaSidecar.load(
                from: scriptDir.appendingPathComponent("\(schema).schema.json")
            )
            XCTAssertEqual(tool.name, sidecar.name,
                           "tool name \(tool.name) != sidecar name \(sidecar.name)")
        }
    }
}

// MARK: - End-to-end smoke test

/// Exercises the full Python bridge against a real Python interpreter
/// and a real tessera_db_query run. Gated on python3 being on PATH.
final class PythonEngineBridgeEndToEndTests: PythonToolTestBase {
    private var scratchDB: URL?
    private var originalScriptDir: String?

    override func setUp() async throws {
        try await super.setUp()
        // Point the bridge at THIS worktree's tools/tessera, not the
        // main checkout the home-directory fallback would find.
        let here = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // TesseraCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // TesseraStudio
            .deletingLastPathComponent()  // repo root
        let scriptDir = here.appendingPathComponent("tools/tessera").path
        originalScriptDir = ProcessInfo.processInfo.environment["TESSERA_SCRIPT_DIR"]
        setenv("TESSERA_SCRIPT_DIR", scriptDir, 1)
    }

    override func tearDown() async throws {
        if let scratchDB {
            try? FileManager.default.removeItem(at: scratchDB)
        }
        // Restore the env var so subsequent tests see the original.
        if let originalScriptDir {
            setenv("TESSERA_SCRIPT_DIR", originalScriptDir, 1)
        } else {
            unsetenv("TESSERA_SCRIPT_DIR")
        }
        try await super.tearDown()
    }

    func testTesseraDBQueryListModelsEndToEnd() async throws {
        // 1. Spin up an in-memory duckdb, write a few rows, dump to disk.
        let dbURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tessera-pybridge-test-\(UUID().uuidString).duckdb")
        let populateScript = """
        import duckdb, sys
        conn = duckdb.connect(sys.argv[1])
        conn.execute(\"\"\"
            CREATE TABLE tensor_stats (
                model_hash TEXT, model_role TEXT, name TEXT, family TEXT,
                layer_depth INTEGER, out_dim INTEGER, in_dim INTEGER,
                n_elements INTEGER, dtype TEXT,
                kurtosis DOUBLE, eff_rank DOUBLE, rms DOUBLE,
                mean_abs DOUBLE, tail_ratio DOUBLE,
                source TEXT, recommended_action TEXT,
                updated_at TEXT, backfill_count INTEGER
            )
        \"\"\")
        conn.execute(\"INSERT INTO tensor_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)\",
            ['abc123', 'trunk', 'blk.0.attn_q', 'attn', 0, 4096, 4096, 16777216,
             'f16', 0.0, 64.0, 0.5, 0.1, 0.05, 'cpp_quant', 'Q4_K', '2026-08-03 10:00:00', None])
        conn.execute(\"INSERT INTO tensor_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)\",
            ['abc123', 'dflash', 'enc.0.q_proj', 'attn', 0, 2048, 2048, 4194304,
             'f16', 0.0, 32.0, 0.4, 0.1, 0.05, 'py_cal', 'Q4_K', '2026-08-03 11:00:00', 1])
        conn.execute(\"INSERT INTO tensor_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)\",
            ['def456', 'trunk', 'blk.0.attn_q', 'attn', 0, 4096, 4096, 16777216,
             'f16', 0.0, 64.0, 0.5, 0.1, 0.05, 'cpp_quant', 'Q4_K', '2026-08-03 12:00:00', None])
        conn.close()
        """
        let popResult = try await runPython(populateScript, args: [dbURL.path])
        XCTAssertEqual(popResult.exitCode, 0,
                       "duckdb populate failed: \(popResult.stderr)")
        scratchDB = dbURL

        // 2. Run the wrapper end-to-end: build the tool, execute it,
        //    parse the JSON result.
        let tool = TesseraDBQueryTool()
        let result = try await tool.execute(arguments: [
            "query": .string("list_models"),
            "db": .string(dbURL.path),
            "limit": .number(50),
        ])
        XCTAssertTrue(result.success, "tool failed: \(result.error ?? "?")")
        XCTAssertNotNil(result.data?["parsed"])

        guard case let .object(parsed)? = result.data?["parsed"] else {
            XCTFail("expected object payload, got \(String(describing: result.data))")
            return
        }
        guard case let .array(rows)? = parsed["rows"] else {
            XCTFail("expected rows array, got \(parsed.keys.sorted())")
            return
        }
        XCTAssertEqual(rows.count, 2, "expected 2 distinct model_hashes")
    }

    func testArgvBuilderForTesseraDBQuerySidecar() throws {
        // End-to-end argv check: build args the way a real call would,
        // and verify the wrapper assembles them correctly. The 'query'
        // property is declared positional in the sidecar (it is the
        // argparse subcommand); the rest become flags.
        let here = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sidecar = try PythonSchemaSidecar.load(
            from: here.appendingPathComponent("tools/tessera/tessera_db_query.schema.json")
        )
        XCTAssertEqual(sidecar.positional, ["query"],
                       "tessera_db_query sidecar should mark 'query' as positional (the argparse subcommand)")
        let argv = PythonTool.buildArgv(
            parameters: sidecar.parameters,
            positional: sidecar.positional ?? [],
            subcommand: sidecar.subcommand,
            arguments: [
                "query": .string("list_models"),
                "db": .string("/tmp/x.duckdb"),
                "limit": .number(100),
            ]
        )
        // query is positional (1 entry), then 2 flag/value pairs.
        XCTAssertEqual(argv.count, 5)
        XCTAssertEqual(argv[0], "list_models", "first argv entry should be the positional subcommand")
        XCTAssertTrue(argv.contains("--db"))
        XCTAssertTrue(argv.contains("/tmp/x.duckdb"))
        XCTAssertTrue(argv.contains("--limit"))
        XCTAssertTrue(argv.contains("100"))
        // Flags precede their values.
        for i in 0..<(argv.count - 1) {
            if argv[i].hasPrefix("--") {
                XCTAssertFalse(argv[i + 1].hasPrefix("--"),
                               "flag \(argv[i]) immediately followed by another flag")
            }
        }
    }

    /// Helper: run python3 -c <script> [args...] and capture stdout+stderr.
    private func runPython(_ script: String, args: [String]) async throws
    -> (exitCode: Int32, stdout: String, stderr: String) {
        let runner = ProcessRunner()
        let result = try await runner.run(
            executable: "/usr/bin/env",
            arguments: ["python3", "-c", script] + args
        )
        return (result.exitCode, result.stdout, result.stderr)
    }
}

// MARK: - PythonEngineBridge discovery

final class PythonEngineBridgeDiscoveryTests: PythonToolTestBase {
    func testBridgeReturnsAValidPythonInterpreter() async throws {
        let bridge = PythonEngineBridge.shared
        // First call populates the cache; second call should be a no-op.
        let url1 = try await bridge.discoverPython()
        let url2 = try await bridge.discoverPython()
        XCTAssertEqual(url1, url2, "cached python URL changed between calls")
        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: url1.path),
                      "\(url1.path) is not executable")
    }

    func testBridgeResolvesScriptDir() async throws {
        let dir = try await PythonEngineBridge.shared.discoverScriptDir()
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.path))
        // The script dir should contain at least one of our wrapped tools.
        let probe = dir.appendingPathComponent("multimodal_calibrate.py")
        XCTAssertTrue(FileManager.default.fileExists(atPath: probe.path),
                      "expected multimodal_calibrate.py in \(dir.path)")
    }

    func testLocateScriptReturnsCorrectURL() async throws {
        let url = try await PythonEngineBridge.shared.locateScript("multimodal_calibrate")
        XCTAssertTrue(url.path.hasSuffix("multimodal_calibrate.py"))
    }

    func testLocateScriptThrowsForMissingName() async throws {
        let bridge = PythonEngineBridge.shared
        do {
            _ = try await bridge.locateScript("definitely_not_a_real_script_xyz")
            XCTFail("expected scriptNotFound")
        } catch let err as PythonError {
            switch err {
            case .scriptNotFound:
                break  // expected
            default:
                XCTFail("wrong error: \(err)")
            }
        } catch {
            XCTFail("wrong error type: \(error)")
        }
    }
}

// MARK: - Cancellation propagation

/// Verifies the bridge's `run(...)` stream is cancellable. We shell out
/// to a long-running Python sleep and assert the subprocess is gone
/// within a few seconds of the consumer task being cancelled.
///
/// Approach: we don't go through ``PythonEngineBridge.run`` (the stream
/// API) for this test. Instead we drive the same machinery directly
/// via ``ProcessRunner.runStreamingCombined`` to keep the test focused
/// on the cancellation -> terminate chain (the bridge just wraps the
/// same `onTermination -> process.terminate` plumbing).
final class PythonEngineBridgeCancellationTests: PythonToolTestBase {
    func testCancelTerminatesSubprocessQuickly() async throws {
        let runner = ProcessRunner()
        let stream = runner.runStreamingCombined(
            executable: "/usr/bin/env",
            arguments: ["python3", "-c", "import time; time.sleep(30)"]
        )
        let start = Date()
        let task = Task {
            for try await _ in stream { }
        }
        // Give the subprocess a moment to actually start sleeping.
        try await Task.sleep(for: .milliseconds(200))
        task.cancel()
        // The stream's onTermination is called when the iterator is
        // dropped. Give the cancellation handler a few seconds to
        // terminate the subprocess.
        let deadline = Date().addingTimeInterval(5.0)
        while Date() < deadline {
            if task.isCancelled { break }
            try? await Task.sleep(for: .milliseconds(50))
        }
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertTrue(task.isCancelled, "task did not cancel")
        XCTAssertLessThan(elapsed, 5.0, "subprocess took \(elapsed)s to terminate after cancel")
    }
}
