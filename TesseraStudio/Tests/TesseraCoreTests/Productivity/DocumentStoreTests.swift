import XCTest
@testable import TesseraCore

/// Tests for the DocumentStore: JSON encoding/decoding of ASTs,
/// error handling for missing documents. The DB integration
/// tests (env-gated) live in ProductivityDataLayerTests.
final class DocumentStoreTests: XCTestCase {

    // MARK: - DocumentAST JSON round-trip (DocumentStore shape)

    func testDocumentJSONEncoding() throws {
        let block = Block(
            type: .paragraph,
            content: [InlineRun(text: "hello world", annotations: [.bold])]
        )
        let doc = DocumentAST(blocks: [block.id: block], rootChildren: [block.id])
        let data = try doc.jsonData()
        let json = String(data: data, encoding: .utf8)!
        XCTAssertTrue(json.contains("paragraph"))
        XCTAssertTrue(json.contains("hello world"))
    }

    func testDocumentJSONDecoding() throws {
        let json = """
        {
            "blocks": {
                "11111111-1111-1111-1111-111111111111": {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "type": "paragraph",
                    "attributes": {},
                    "content": [{"text": "hi", "annotations": []}],
                    "children": [],
                    "parentID": null
                }
            },
            "rootChildren": ["11111111-1111-1111-1111-111111111111"]
        }
        """
        let data = json.data(using: .utf8)!
        let doc = try DocumentAST.from(jsonData: data)
        XCTAssertEqual(doc.rootChildren.count, 1)
        XCTAssertEqual(doc.blocks.count, 1)
    }

    // MARK: - DocumentStoreError

    func testDocumentStoreErrorEquality() {
        let id = UUID()
        XCTAssertEqual(
            DocumentStoreError.documentNotFound(id: id),
            DocumentStoreError.documentNotFound(id: id)
        )
        XCTAssertNotEqual(
            DocumentStoreError.documentNotFound(id: id),
            DocumentStoreError.documentNotFound(id: UUID())
        )
        XCTAssertEqual(
            DocumentStoreError.emptyMutationBatch,
            DocumentStoreError.emptyMutationBatch
        )
    }
}
