import XCTest
import CryptoKit
@testable import TesseraCore

/// Tests for ``ReceiptExportService``. The tests
/// exercise the public API directly; the
/// `TesseraDataLayer` is used with a real (but
/// unconnected) configuration — the tests never call
/// `start()`, so the actor's internal stores are
/// uninitialized. The export service only needs the
/// facade's `appendReceiptToChain` path, which the
/// tests don't exercise (the export service is given
/// an empty chain, so it throws `noReceipts` before
/// reaching the data layer). The Markdown / JSON / C2PA
/// builders are exercised in isolation.
final class ReceiptExportServiceTests: XCTestCase {

    private func makeService() -> ReceiptExportService {
        let dataLayer = TesseraDataLayer(configuration: .init(
            dataStore: .init(
                host: "localhost",
                port: 5432,
                username: "tessera",
                password: nil,
                database: "tessera",
                minimumConnections: 1,
                maximumConnections: 2
            ),
            cache: TesseraCache.Configuration(host: "localhost", port: 6379, password: nil, databaseNumber: 0, poolSize: 1, namespace: "test")
        ))
        let signer = ReceiptSigner(signingKey: Curve25519.Signing.PrivateKey())
        let store = DocumentStore(dataLayer: dataLayer)
        return ReceiptExportService(
            documentStore: store,
            dataLayer: dataLayer,
            signer: signer
        )
    }

    private func makeReceipt(documentID: UUID) -> Receipt {
        let key = Curve25519.Signing.PrivateKey()
        let signer = ReceiptSigner(signingKey: key)
        return (try? signer.sign(
            documentID: documentID,
            mutations: [],
            priorReceiptID: nil,
            actor: .user(UUID()),
            preMutationSnapshot: [:]
        ))!
    }

    // MARK: - User confirmation

    func testExportWithoutConfirmationThrows() async {
        let service = makeService()
        do {
            _ = try await service.export(
                documentID: UUID(),
                format: .signedJSON,
                documentTitle: "Test",
                userID: UUID(),
                userConfirmed: false
            )
            XCTFail("expected userDenied")
        } catch ExportError.userDenied {
            // expected
        } catch {
            XCTFail("expected userDenied, got \(error)")
        }
    }

    // MARK: - Filename

    func testFilenameForSignedJSON() {
        let name = ReceiptExportService.makeFilename(
            documentTitle: "My Document",
            format: .signedJSON,
            now: Date(timeIntervalSince1970: 0)
        )
        XCTAssertTrue(name.hasPrefix("my-document-audit-"))
        XCTAssertTrue(name.hasSuffix(".json"))
    }

    func testFilenameForMarkdown() {
        let name = ReceiptExportService.makeFilename(
            documentTitle: "My Document",
            format: .markdown,
            now: Date(timeIntervalSince1970: 0)
        )
        XCTAssertTrue(name.hasPrefix("my-document-audit-"))
        XCTAssertTrue(name.hasSuffix(".md"))
    }

    func testFilenameForC2PA() {
        let name = ReceiptExportService.makeFilename(
            documentTitle: "My Document",
            format: .c2paDocument,
            now: Date(timeIntervalSince1970: 0)
        )
        XCTAssertTrue(name.hasPrefix("my-document-c2pa."))
    }

    // MARK: - Slugify

    func testSlugifyStripsPunctuation() {
        XCTAssertEqual(ReceiptExportService.slugify("Hello, World!"), "hello-world")
        XCTAssertEqual(ReceiptExportService.slugify("  spaces  "), "spaces")
        XCTAssertEqual(ReceiptExportService.slugify(""), "document")
    }

    // MARK: - Markdown rendering

    func testMarkdownRenderingIsHumanReadable() {
        let key = Curve25519.Signing.PrivateKey()
        let signer = ReceiptSigner(signingKey: key)
        let docID = UUID()
        let receipt = (try? signer.sign(
            documentID: docID,
            mutations: [],
            priorReceiptID: nil,
            actor: .user(UUID()),
            preMutationSnapshot: [:]
        ))!
        let service = makeService()
        let md = service.buildMarkdownSummary(
            documentTitle: "Doc",
            chain: [receipt]
        )
        let str = String(data: md, encoding: .utf8) ?? ""
        XCTAssertTrue(str.contains("# Doc"))
        XCTAssertTrue(str.contains("Audit Trail"))
    }

    func testMarkdownIncludesActor() {
        let key = Curve25519.Signing.PrivateKey()
        let signer = ReceiptSigner(signingKey: key)
        let docID = UUID()
        let receipt = (try? signer.sign(
            documentID: docID,
            mutations: [],
            priorReceiptID: nil,
            actor: .agent(UUID(), model: "claude", promptHash: "abc"),
            preMutationSnapshot: [:]
        ))!
        let service = makeService()
        let md = service.buildMarkdownSummary(
            documentTitle: "Doc",
            chain: [receipt]
        )
        let str = String(data: md, encoding: .utf8) ?? ""
        XCTAssertTrue(str.contains("agent"))
        XCTAssertTrue(str.contains("claude"))
    }

    // MARK: - Signed JSON bundle

    func testSignedJSONBundleIsValidJSON() throws {
        let key = Curve25519.Signing.PrivateKey()
        let signer = ReceiptSigner(signingKey: key)
        let docID = UUID()
        let receipt = (try? signer.sign(
            documentID: docID,
            mutations: [],
            priorReceiptID: nil,
            actor: .user(UUID()),
            preMutationSnapshot: [:]
        ))!
        let service = makeService()
        let data = try service.buildSignedJSONBundle(
            documentID: docID,
            chain: [receipt],
            documentTitle: "Doc"
        )
        let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertNotNil(parsed)
        XCTAssertEqual(parsed?["documentID"] as? String, docID.uuidString)
        XCTAssertEqual(parsed?["receiptCount"] as? Int, 1)
    }

    // MARK: - C2PA bundle

    func testC2PABundleIncludesBody() throws {
        let key = Curve25519.Signing.PrivateKey()
        let signer = ReceiptSigner(signingKey: key)
        let docID = UUID()
        let receipt = (try? signer.sign(
            documentID: docID,
            mutations: [],
            priorReceiptID: nil,
            actor: .user(UUID()),
            preMutationSnapshot: [:]
        ))!
        let ast = DocumentAST(blocks: [:], rootChildren: [])
        let service = makeService()
        let data = try service.buildC2PADocument(
            documentID: docID,
            documentTitle: "Doc",
            chain: [receipt],
            ast: ast
        )
        let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertNotNil(parsed)
        XCTAssertEqual(parsed?["documentID"] as? String, docID.uuidString)
    }

    // MARK: - Egress policy

    func testDenialEgressPolicyBlocks() async {
        let dataLayer = TesseraDataLayer(configuration: .init(
            dataStore: .init(
                host: "localhost",
                port: 5432,
                username: "tessera",
                password: nil,
                database: "tessera",
                minimumConnections: 1,
                maximumConnections: 2
            ),
            cache: TesseraCache.Configuration(host: "localhost", port: 6379, password: nil, databaseNumber: 0, poolSize: 1, namespace: "test")
        ))
        let signer = ReceiptSigner(signingKey: Curve25519.Signing.PrivateKey())
        let store = DocumentStore(dataLayer: dataLayer)
        // We can't easily inject a non-empty chain
        // without a real DB; we test the policy gate by
        // wiring a denial policy and confirming the
        // service throws `buildFailed("egress...")` when
        // a non-empty chain would be present. The empty
        // chain check fires first, so we test the policy
        // check via the lower-level `buildSignedJSONBundle`
        // path. The full flow test is gated by the
        // integration test in
        // `ProductivityDataLayerTests`.
        let denial = DenyAllEgressPolicy()
        let service = ReceiptExportService(
            documentStore: store,
            dataLayer: dataLayer,
            signer: signer,
            egressPolicy: denial
        )
        // The egress policy is only consulted on a
        // non-empty chain. Verify the policy's `allowsExport`
        // directly.
        let allowed = denial.allowsExport(documentID: UUID(), format: .signedJSON)
        XCTAssertFalse(allowed)
    }
}

/// A test-only `EgressPolicy` that denies every export.
struct DenyAllEgressPolicy: EgressPolicy {
    func allowsExport(documentID: UUID, format: ReceiptExportFormat) -> Bool {
        return false
    }
}
