import XCTest
import CryptoKit
@testable import TesseraCore

/// Tests for the Block AST data model: BlockType, InlineRun,
/// Block, DocumentAST. Round-trip JSON serialization for every
/// block type and every inline annotation variant. Empty
/// document, deep nesting, content-hash stability.
final class BlockTests: XCTestCase {

    // MARK: - BlockType

    func testEveryBlockTypeRoundTrips() throws {
        for type in BlockType.allCases {
            let block = Block(
                id: UUID(),
                type: type,
                attributes: ["k": .string("v")],
                content: [InlineRun(text: "hi")]
            )
            let data = try JSONEncoder().encode(block)
            let decoded = try JSONDecoder().decode(Block.self, from: data)
            XCTAssertEqual(decoded.type, type, "round-trip failed for \(type)")
            XCTAssertEqual(decoded.attributes["k"]?.stringValue, "v")
        }
    }

    func testAllCasesPresent() {
        // Lock in the BlockType set. If a new case is added
        // intentionally, this test must be updated.
        let expected: Set<BlockType> = [
            .heading, .paragraph, .list, .listItem, .table, .tableCell,
            .image, .codeBlock, .callout, .divider, .quote, .toggle, .equation
        ]
        XCTAssertEqual(Set(BlockType.allCases), expected)
    }

    // MARK: - InlineRun

    func testPlainInlineRun() throws {
        let run = InlineRun(text: "hello")
        let data = try JSONEncoder().encode(run)
        let decoded = try JSONDecoder().decode(InlineRun.self, from: data)
        XCTAssertEqual(decoded.text, "hello")
        XCTAssertTrue(decoded.annotations.isEmpty)
    }

    func testBoldAnnotation() throws {
        let run = InlineRun(text: "bold", annotations: [.bold])
        let data = try JSONEncoder().encode(run)
        let decoded = try JSONDecoder().decode(InlineRun.self, from: data)
        XCTAssertEqual(decoded.annotations, [.bold])
    }

    func testItalicAnnotation() throws {
        let run = InlineRun(text: "italic", annotations: [.italic])
        let data = try JSONEncoder().encode(run)
        let decoded = try JSONDecoder().decode(InlineRun.self, from: data)
        XCTAssertEqual(decoded.annotations, [.italic])
    }

    func testUnderlineAnnotation() throws {
        let run = InlineRun(text: "u", annotations: [.underline])
        let data = try JSONEncoder().encode(run)
        let decoded = try JSONDecoder().decode(InlineRun.self, from: data)
        XCTAssertEqual(decoded.annotations, [.underline])
    }

    func testStrikethroughAnnotation() throws {
        let run = InlineRun(text: "x", annotations: [.strikethrough])
        let data = try JSONEncoder().encode(run)
        let decoded = try JSONDecoder().decode(InlineRun.self, from: data)
        XCTAssertEqual(decoded.annotations, [.strikethrough])
    }

    func testCodeAnnotation() throws {
        let run = InlineRun(text: "code", annotations: [.code])
        let data = try JSONEncoder().encode(run)
        let decoded = try JSONDecoder().decode(InlineRun.self, from: data)
        XCTAssertEqual(decoded.annotations, [.code])
    }

    func testSubscriptAnnotation() throws {
        let run = InlineRun(text: "x", annotations: [.subscript])
        let data = try JSONEncoder().encode(run)
        let decoded = try JSONDecoder().decode(InlineRun.self, from: data)
        XCTAssertEqual(decoded.annotations, [.subscript])
    }

    func testSuperscriptAnnotation() throws {
        let run = InlineRun(text: "x", annotations: [.superscript])
        let data = try JSONEncoder().encode(run)
        let decoded = try JSONDecoder().decode(InlineRun.self, from: data)
        XCTAssertEqual(decoded.annotations, [.superscript])
    }

    func testLinkAnnotation() throws {
        let url = URL(string: "https://tessera.example")!
        let run = InlineRun(text: "link", annotations: [.link(url)])
        let data = try JSONEncoder().encode(run)
        let decoded = try JSONDecoder().decode(InlineRun.self, from: data)
        XCTAssertEqual(decoded.annotations.count, 1)
        if case .link(let decodedURL) = decoded.annotations[0] {
            XCTAssertEqual(decodedURL, url)
        } else {
            XCTFail("expected .link annotation")
        }
    }

    func testColorAnnotation() throws {
        let run = InlineRun(text: "c", annotations: [.color(hex: "#FF00FF")])
        let data = try JSONEncoder().encode(run)
        let decoded = try JSONDecoder().decode(InlineRun.self, from: data)
        XCTAssertEqual(decoded.annotations.count, 1)
        if case .color(let hex) = decoded.annotations[0] {
            XCTAssertEqual(hex, "#FF00FF")
        } else {
            XCTFail("expected .color annotation")
        }
    }

    func testMultipleAnnotationsOnOneRun() throws {
        let url = URL(string: "https://x")!
        let run = InlineRun(
            text: "combo",
            annotations: [.bold, .italic, .link(url), .color(hex: "#000")]
        )
        let data = try JSONEncoder().encode(run)
        let decoded = try JSONDecoder().decode(InlineRun.self, from: data)
        XCTAssertEqual(decoded.annotations.count, 4)
        XCTAssertTrue(decoded.annotations.contains(.bold))
        XCTAssertTrue(decoded.annotations.contains(.italic))
    }

    // MARK: - DocumentAST round-trip

    func testEmptyDocumentRoundTrip() throws {
        let doc = DocumentAST.empty
        let data = try doc.jsonData()
        let decoded = try DocumentAST.from(jsonData: data)
        XCTAssertTrue(decoded.blocks.isEmpty)
        XCTAssertTrue(decoded.rootChildren.isEmpty)
    }

    func testSingleBlockDocument() throws {
        let block = Block(
            type: .paragraph,
            content: [InlineRun(text: "hello world")]
        )
        let doc = DocumentAST(
            blocks: [block.id: block],
            rootChildren: [block.id]
        )
        let decoded = try DocumentAST.from(jsonData: try doc.jsonData())
        XCTAssertEqual(decoded.rootChildren, [block.id])
        XCTAssertEqual(decoded.blocks[block.id]?.type, .paragraph)
        XCTAssertEqual(decoded.blocks[block.id]?.content.first?.text, "hello world")
    }

    func testDeepNestingPreserved() throws {
        // Build a 100-deep nested list.
        var blocks: [UUID: Block] = [:]
        var currentParent: UUID? = nil
        var lastID: UUID? = nil
        for depth in 0..<100 {
            let block = Block(
                type: .listItem,
                content: [InlineRun(text: "level \(depth)")],
                parentID: currentParent
            )
            blocks[block.id] = block
            if let pid = currentParent {
                blocks[pid]?.children.append(block.id)
            }
            currentParent = block.id
            lastID = block.id
        }
        let doc = DocumentAST(blocks: blocks, rootChildren: [blocks.first(where: { $0.value.parentID == nil })!.key])
        let decoded = try DocumentAST.from(jsonData: try doc.jsonData())

        // Walk the depth-first order; it should be exactly 100.
        XCTAssertEqual(decoded.depthFirstOrder().count, 100)
        // The deepest block's text should be preserved.
        XCTAssertEqual(
            decoded.blocks[lastID!]?.content.first?.text,
            "level 99"
        )
    }

    func testContentHashIsStable() throws {
        let block = Block(
            type: .paragraph,
            content: [InlineRun(text: "stable")]
        )
        let doc = DocumentAST(
            blocks: [block.id: block],
            rootChildren: [block.id]
        )
        // Two hashes of the same document should be byte-equal.
        let h1 = try doc.contentHash()
        let h2 = try doc.contentHash()
        XCTAssertEqual(h1, h2)
        XCTAssertTrue(h1.hasPrefix("sha256:"))
        XCTAssertEqual(h1.count, "sha256:".count + 64)  // 32 bytes hex
    }

    func testContentHashChangesWhenContentChanges() throws {
        let b1 = Block(
            id: UUID(),
            type: .paragraph,
            content: [InlineRun(text: "v1")]
        )
        let b2 = Block(
            id: b1.id,
            type: .paragraph,
            content: [InlineRun(text: "v2")]
        )
        let doc1 = DocumentAST(blocks: [b1.id: b1], rootChildren: [b1.id])
        let doc2 = DocumentAST(blocks: [b2.id: b2], rootChildren: [b2.id])
        XCTAssertNotEqual(try doc1.contentHash(), try doc2.contentHash())
    }

    func testDepthFirstOrderEmpty() {
        let doc = DocumentAST.empty
        XCTAssertEqual(doc.depthFirstOrder(), [])
    }

    func testDepthFirstOrderSingleRoot() {
        let block = Block(type: .paragraph)
        let doc = DocumentAST(
            blocks: [block.id: block],
            rootChildren: [block.id]
        )
        XCTAssertEqual(doc.depthFirstOrder(), [block.id])
    }

    func testDepthFirstOrderSiblingOrder() {
        let a = Block(type: .paragraph)
        let b = Block(type: .paragraph)
        let c = Block(type: .paragraph)
        let doc = DocumentAST(
            blocks: [a.id: a, b.id: b, c.id: c],
            rootChildren: [a.id, b.id, c.id]
        )
        XCTAssertEqual(doc.depthFirstOrder(), [a.id, b.id, c.id])
    }

    func testChildrenOf() {
        let parent = Block(type: .list)
        let child1 = Block(type: .listItem, parentID: parent.id)
        let child2 = Block(type: .listItem, parentID: parent.id)
        let doc = DocumentAST(
            blocks: [
                parent.id: parent,
                child1.id: child1,
                child2.id: child2
            ],
            rootChildren: [parent.id]
        )
        // Update parent's children after construction.
        var updatedDoc = doc
        updatedDoc.blocks[parent.id]?.children = [child1.id, child2.id]
        XCTAssertEqual(updatedDoc.children(of: nil), [parent.id])
        XCTAssertEqual(updatedDoc.children(of: parent.id), [child1.id, child2.id])
    }

    func testContains() {
        let block = Block(type: .paragraph)
        let doc = DocumentAST(blocks: [block.id: block], rootChildren: [block.id])
        XCTAssertTrue(doc.contains(block.id))
        XCTAssertFalse(doc.contains(UUID()))
    }

    // MARK: - AnyCodable

    func testAnyCodableTypealiasIsJSONValue() {
        // The brief writes `AnyCodable`; we typealias to JSONValue.
        // Lock the alias in.
        let v: AnyCodable = .string("hello")
        XCTAssertEqual(v.stringValue, "hello")
    }
}
