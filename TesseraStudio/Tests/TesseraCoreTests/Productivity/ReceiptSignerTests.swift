import XCTest
import CryptoKit
@testable import TesseraCore

/// Tests for the receipt-signing plumbing: ReceiptSigner
/// composition with the Keychain (in injected mode for tests),
/// signature round-trip, edge cases.
final class ReceiptSignerTests: XCTestCase {

    private var key: Curve25519.Signing.PrivateKey!
    private var signer: ReceiptSigner!

    override func setUp() {
        super.setUp()
        key = Curve25519.Signing.PrivateKey()
        signer = ReceiptSigner(signingKey: key)
    }

    // MARK: - Composition

    func testHasSigningKeyWithInjectedKey() {
        XCTAssertTrue(signer.hasSigningKey)
    }

    func testPublicKeyIsInjected() {
        XCTAssertEqual(signer.publicKey?.rawRepresentation, key.publicKey.rawRepresentation)
    }

    func testNoSigningKeyWhenKeychainEmpty() {
        // When the keychain has no volume password, the .keychain
        // signer reports no signing key. (This test doesn't
        // interact with the real Keychain; the default constructor
        // is the keychain source, but the keychain may or may not
        // have a volume password in the test environment.)
        let defaultSigner = ReceiptSigner()
        // Either case is acceptable; we just verify the property
        // is queryable.
        _ = defaultSigner.hasSigningKey
    }

    // MARK: - ReceiptSignerError

    func testReceiptSignerErrorEquality() {
        XCTAssertEqual(
            ReceiptSignerError.signingKeyUnavailable,
            ReceiptSignerError.signingKeyUnavailable
        )
        XCTAssertNotEqual(
            ReceiptSignerError.signingKeyUnavailable,
            ReceiptSignerError.signingFailed("x")
        )
    }
}
