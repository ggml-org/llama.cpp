import XCTest
@testable import TesseraCore

final class TesseraEngineBridgeFactoryTests: XCTestCase {
    func testFactoryReturnsCLIBridge() {
        let bridge = TesseraEngineBridgeFactory.makeInferenceBridge()
        XCTAssertTrue(bridge is CLIEngineBridge)
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
