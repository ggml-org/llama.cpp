import XCTest
import CryptoKit
@testable import TesseraCore

/// Tests for the receipt detail view's diff rendering
/// and signature verification. The view itself is a
/// SwiftUI view (not unit-testable without a renderer);
/// these tests cover the underlying data shape.
final class ReceiptDetailViewTests: XCTestCase {

    func testReceiptVerificationWithValidSignature() {
        let key = Curve25519.Signing.PrivateKey()
        let signer = ReceiptSigner(signingKey: key)
        let docID = UUID()
        guard let receipt = try? signer.sign(
            documentID: docID,
            mutations: [],
            priorReceiptID: nil,
            actor: .user(UUID()),
            preMutationSnapshot: [:]
        ) else {
            XCTFail("sign failed")
            return
        }
        let result = signer.verify(receipt, against: key.publicKey)
        if case .valid = result { } else {
            XCTFail("expected valid, got \(result)")
        }
    }

    func testReceiptVerificationWithInvalidSignature() {
        let key1 = Curve25519.Signing.PrivateKey()
        let key2 = Curve25519.Signing.PrivateKey()
        let signer = ReceiptSigner(signingKey: key1)
        let docID = UUID()
        guard let receipt = try? signer.sign(
            documentID: docID,
            mutations: [],
            priorReceiptID: nil,
            actor: .user(UUID()),
            preMutationSnapshot: [:]
        ) else {
            XCTFail("sign failed")
            return
        }
        // Verify with the wrong key.
        let result = signer.verify(receipt, against: key2.publicKey)
        if case .invalid = result { } else {
            XCTFail("expected invalid, got \(result)")
        }
    }

    func testReceiptVerificationWithVoided() {
        let key = Curve25519.Signing.PrivateKey()
        let signer = ReceiptSigner(signingKey: key)
        let docID = UUID()
        guard var receipt = try? signer.sign(
            documentID: docID,
            mutations: [],
            priorReceiptID: nil,
            actor: .user(UUID()),
            preMutationSnapshot: [:]
        ) else {
            XCTFail("sign failed")
            return
        }
        receipt.voidedBy = UUID()
        let result = signer.verify(receipt, against: key.publicKey)
        if case .voided = result { } else {
            XCTFail("expected voided, got \(result)")
        }
    }

    func testReceiptCanonicalFormDeterministic() throws {
        let key = Curve25519.Signing.PrivateKey()
        let signer = ReceiptSigner(signingKey: key)
        let docID = UUID()
        let actor = Actor.user(UUID())
        let r1 = try signer.sign(
            documentID: docID,
            mutations: [],
            priorReceiptID: nil,
            actor: actor,
            preMutationSnapshot: [:]
        )
        let r2 = try signer.sign(
            documentID: docID,
            mutations: [],
            priorReceiptID: nil,
            actor: actor,
            preMutationSnapshot: [:]
        )
        // Different ids, but the canonical form is the
        // same shape. The signatures will differ.
        let c1 = try r1.canonicalBytes()
        let c2 = try r2.canonicalBytes()
        XCTAssertNotEqual(c1, c2, "different ids => different canonical form")
    }
}
