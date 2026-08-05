import XCTest
@testable import TesseraCore

final class TesseraEngineBridgeFactoryTests: XCTestCase {
    func testFactoryReturnsCLIBridge() {
        let bridge = TesseraEngineBridgeFactory.makeInferenceBridge()
        XCTAssertTrue(bridge is CLIEngineBridge)
    }
}

// Locks in the no-xcframework (SwiftPM stub) contract: when the package is
// built by swift build / swift test the CTesseraFFI stub is linked, so the
// bridge reports unavailable and every operation falls back to the CLI. When
// tessera.xcframework is linked in the Xcode app these assertions flip and
// the real C++ engine takes over.
final class TesseraFFIBridgeStubTests: XCTestCase {
    func testStubReportsUnavailable() {
        // The SwiftPM stub returns 0 from tessera_ffi_is_available().
        XCTAssertFalse(TesseraFFIBridge.isAvailable)
    }

    func testStubVersionIsTheStubMarker() {
        XCTAssertEqual(TesseraFFIBridge.version, "tessera-ffi-stub")
    }

    func testStubQuantizeFallsBackToCLI() {
        let outcome = TesseraFFIBridge.quantize(
            model: "/tmp/model.gguf", output: "/tmp/out.gguf", config: [:]
        )
        XCTAssertEqual(outcome, .fallbackToCLI)
    }

    func testStubEvolveFallsBackToCLI() {
        let outcome = TesseraFFIBridge.evolve(model: "/tmp/model.gguf", config: [:])
        XCTAssertEqual(outcome, .fallbackToCLI)
    }

    func testStubConvertFallsBackToCLI() {
        let outcome = TesseraFFIBridge.convert(
            model: "/tmp/model.gguf", output: "/tmp/out.mlmodelc", format: "coreml"
        )
        XCTAssertEqual(outcome, .fallbackToCLI)
    }

    func testStubEvaluateFallsBackToCLI() {
        let outcome = TesseraFFIBridge.evaluate(model: "/tmp/model.gguf", config: [:])
        XCTAssertEqual(outcome, .fallbackToCLI)
    }

    func testCapabilitiesReflectStub() {
        let caps = TesseraEngineBridgeFactory.capabilities
        XCTAssertFalse(caps.ffiAvailable)
        XCTAssertEqual(caps.ffiVersion, "tessera-ffi-stub")
    }

    // Model-context stubs: the SwiftPM stub returns NULL from
    // tessera_load_model and the "use the CLI" code from the *_model
    // entry points. The bridge exposes that as TesseraEngineError
    // (loadModel) and .fallbackToCLI (the model-context ops).
    func testStubModelOpsFallBackToCLI() async {
        // loadModel throws engineUnavailable because the stub returns NULL.
        // (The check `!TesseraFFIBridge.isAvailable` runs before throwing
        // modelLoadFailed, so the SwiftPM stub path is the engineUnavailable
        // case rather than the generic "bad path" case.)
        do {
            _ = try await TesseraFFIBridge.loadModel(path: "/tmp/nonexistent.gguf")
            XCTFail("expected throw from stub loadModel")
        } catch TesseraEngineError.engineUnavailable {
            // expected
        } catch {
            XCTFail("expected TesseraEngineError.engineUnavailable, got \(error)")
        }

        // Without a handle we still exercise the static *_model methods to
        // confirm they handle a nil handle by returning the expected
        // outcome. The bridge forwards to the actor; the actor's
        // evolve/evaluate/convert call the C entry points, which the stub
        // resolves to the fallback code. We synthesise a nil handle to
        // exercise the early-return path inside the actor (the C stub
        // also tolerates a NULL handle and returns 1 / NULL).
        let nilHandle = TesseraModelHandle(raw: nil)
        let evolveOutcome = await TesseraFFIBridge.evolveModel(handle: nilHandle)
        XCTAssertEqual(evolveOutcome, .fallbackToCLI)
        let evaluateOutcome = await TesseraFFIBridge.evaluateModel(handle: nilHandle)
        XCTAssertEqual(evaluateOutcome, .fallbackToCLI)
        let convertOutcome = await TesseraFFIBridge.convertModel(
            handle: nilHandle, output: "/tmp/out.mlmodelc", format: "coreml"
        )
        XCTAssertEqual(convertOutcome, .fallbackToCLI)
    }

    // Exercise the actor's lifecycle: loadModel throws on the stub
    // (engineUnavailable) and free is a no-op even with a nil handle.
    // The actor's Box bookkeeping is verified by the test, which would
    // crash if we double-freed (the stub is a no-op so it cannot tell).
    func testEngineContextLifecycle() async {
        // The actor is a singleton; the test is single-async so there is
        // no contention on `live` from concurrent callers.
        do {
            _ = try await TesseraEngineContext.shared.loadModel(path: "/tmp/x.gguf")
            XCTFail("expected throw from loadModel on stub")
        } catch TesseraEngineError.engineUnavailable {
            // expected on the SwiftPM stub
        } catch {
            XCTFail("expected TesseraEngineError.engineUnavailable, got \(error)")
        }

        // free is safe with a nil handle and a never-tracked handle -
        // both must be silent no-ops, not crashes.
        let nilHandle = TesseraModelHandle(raw: nil)
        await TesseraEngineContext.shared.free(handle: nilHandle)
        let randomHandle = TesseraModelHandle(raw: OpaquePointer(bitPattern: 0xDEADBEEF))
        await TesseraEngineContext.shared.free(handle: randomHandle)
    }

    // TesseraModelHandle must be Sendable so it can cross the actor
    // boundary. This is a compile-time check; the test exists to lock
    // the surface so a future refactor does not silently drop the
    // @unchecked Sendable conformance.
    func testTesseraModelHandleIsSendable() {
        let h: any Sendable = TesseraModelHandle(raw: nil)
        XCTAssertNotNil(h)
    }
}

final class QuantizationReceiptTests: XCTestCase {
    private let sampleJSON = """
    {
      "schema_version": "llama.tessera.calibration-receipt.v1",
      "model": {"name":"gemma","family":"Gemma","parameter_count":"4B","source_bits":16,"output_bits":3.5,"file_size_bytes":2000000000},
      "tensors": [
        {"name":"blk.0.attn_q","bits":4.0,"mse":0.001,"snr_db":30.0},
        {"name":"blk.1.attn_q","bits":3.0,"mse":0.003,"snr_db":28.0}
      ],
      "calibration": {"corpus":"wiki","token_count":5000,"modality":"text","dequant_mode":"T640_3D"},
      "ga_archive": {"generations":50,"population":32,"best_fitness":0.95,"archive_size":10},
      "duration_seconds": 120.5
    }
    """

    func testDecodesReceipt() throws {
        let receipt = try JSONDecoder().decode(QuantizationReceipt.self, from: Data(sampleJSON.utf8))
        XCTAssertEqual(receipt.model.name, "gemma")
        XCTAssertEqual(receipt.model.outputBits, 3.5)
        XCTAssertEqual(receipt.tensors.count, 2)
        XCTAssertEqual(receipt.calibration.tokenCount, 5000)
        XCTAssertEqual(receipt.gaArchive?.generations, 50)
        XCTAssertEqual(receipt.durationSeconds, 120.5)
    }

    func testMeanMSE() throws {
        let receipt = try JSONDecoder().decode(QuantizationReceipt.self, from: Data(sampleJSON.utf8))
        XCTAssertEqual(receipt.meanMSE ?? 0, 0.002, accuracy: 1e-9)
    }

    func testToleratesMissingOptionalFields() throws {
        let minimal = """
        {"model": {"name":"x"}, "calibration": {"corpus":"c"}}
        """
        let receipt = try JSONDecoder().decode(QuantizationReceipt.self, from: Data(minimal.utf8))
        XCTAssertEqual(receipt.model.name, "x")
        XCTAssertNil(receipt.gaArchive)
        XCTAssertTrue(receipt.tensors.isEmpty)
        XCTAssertNil(receipt.meanMSE)
    }
}

final class MarkdownBlockParserTests: XCTestCase {
    func testSplitsCodeFenceFromProse() {
        let text = "intro line\n```swift\nlet x = 1\n```\noutro line"
        let blocks = MarkdownBlockParser.parse(text)
        XCTAssertEqual(blocks.count, 3)

        guard case .prose(let intro) = blocks[0] else { return XCTFail("expected prose first") }
        XCTAssertEqual(intro, ["intro line"])

        guard case .code(let language, let code) = blocks[1] else { return XCTFail("expected code second") }
        XCTAssertEqual(language, "swift")
        XCTAssertEqual(code, "let x = 1")

        guard case .prose(let outro) = blocks[2] else { return XCTFail("expected prose third") }
        XCTAssertEqual(outro, ["outro line"])
    }

    func testUnterminatedFenceBecomesPendingCodeBlock() {
        let blocks = MarkdownBlockParser.parse("```python\nprint('hi')")
        XCTAssertEqual(blocks.count, 1)
        guard case .code(let language, let code) = blocks[0] else { return XCTFail("expected code") }
        XCTAssertEqual(language, "python")
        XCTAssertEqual(code, "print('hi')")
    }
}

final class ConversationExporterTests: XCTestCase {
    func testMarkdownExport() {
        let messages = [
            ChatMessage(role: .user, content: "Hello"),
            ChatMessage(role: .assistant, content: "Hi there"),
        ]
        let md = ConversationExporter.markdown(title: "My Chat", messages: messages)
        XCTAssertTrue(md.contains("# My Chat"))
        XCTAssertTrue(md.contains("## User"))
        XCTAssertTrue(md.contains("## Tessera Agent"))
        XCTAssertTrue(md.contains("Hello"))
    }

    func testJSONExportRoundTrips() throws {
        let messages = [ChatMessage(role: .user, content: "Hello")]
        let json = ConversationExporter.json(title: "My Chat", messages: messages)
        let data = Data(json.utf8)
        let decoded = try JSONDecoder().decode([String: JSONValue].self, from: data)
        XCTAssertEqual(decoded["title"]?.stringValue, "My Chat")
        if case .array(let msgs)? = decoded["messages"] {
            XCTAssertEqual(msgs.count, 1)
        } else {
            XCTFail("expected messages array")
        }
    }
}

final class JSONValueTests: XCTestCase {
    func testShortDescription() {
        XCTAssertEqual(JSONValue.string("hi").shortDescription, "hi")
        XCTAssertEqual(JSONValue.bool(true).shortDescription, "true")
        XCTAssertEqual(JSONValue.null.shortDescription, "null")
        XCTAssertEqual(JSONValue.array([.null, .null]).shortDescription, "[2 items]")
        XCTAssertEqual(JSONValue.object(["a": .null]).shortDescription, "{1 keys}")
    }
}
