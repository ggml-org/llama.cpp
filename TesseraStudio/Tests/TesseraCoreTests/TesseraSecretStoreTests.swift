import XCTest
@testable import TesseraCore

/// ``TesseraSecretStore`` keeps API keys in the Keychain instead
/// of UserDefaults. These tests exercise the real Keychain path
/// (verified to work in a bare process) using throwaway accounts
/// so they never touch the app's real stored key, and clean up
/// after themselves.
final class TesseraSecretStoreTests: XCTestCase {
    private var account: String!

    override func setUp() {
        super.setUp()
        account = "unit-test-\(UUID().uuidString)"
    }

    override func tearDown() {
        _ = TesseraSecretStore.setSecret(nil, account: account)
        account = nil
        super.tearDown()
    }

    func testRoundTripAndState() {
        XCTAssertEqual(TesseraSecretStore.state(account: account), .missing)
        XCTAssertTrue(TesseraSecretStore.setSecret("sk-abc123", account: account))
        XCTAssertEqual(TesseraSecretStore.secret(account: account), "sk-abc123")
        XCTAssertEqual(TesseraSecretStore.state(account: account), .stored)
    }

    func testOverwriteReplacesValue() {
        _ = TesseraSecretStore.setSecret("first", account: account)
        _ = TesseraSecretStore.setSecret("second", account: account)
        XCTAssertEqual(TesseraSecretStore.secret(account: account), "second")
    }

    func testNilDeletes() {
        _ = TesseraSecretStore.setSecret("to-delete", account: account)
        XCTAssertEqual(TesseraSecretStore.state(account: account), .stored)
        XCTAssertTrue(TesseraSecretStore.setSecret(nil, account: account))
        XCTAssertEqual(TesseraSecretStore.state(account: account), .missing)
        XCTAssertNil(TesseraSecretStore.secret(account: account))
    }

    func testEmptyStringDeletes() {
        _ = TesseraSecretStore.setSecret("to-delete", account: account)
        XCTAssertTrue(TesseraSecretStore.setSecret("", account: account))
        XCTAssertEqual(TesseraSecretStore.state(account: account), .missing)
    }

    func testMigrateLegacyUserDefaultsIntoKeychain() {
        // Throwaway defaults suite so we never touch the app's
        // real remoteAPIKey preference.
        let defaults = UserDefaults(suiteName: "tessera-secret-migration-test")!
        let defaultsKey = "legacy-\(UUID().uuidString)"
        defer {
            defaults.removeObject(forKey: defaultsKey)
            defaults.removePersistentDomain(forName: "tessera-secret-migration-test")
        }
        defaults.set("legacy-secret", forKey: defaultsKey)

        // Nothing in the Keychain yet; migration should move the
        // UserDefaults value over and delete the plaintext copy.
        let migrated = TesseraSecretStore.migrateLegacy(
            defaultsKey: defaultsKey, account: account, defaults: defaults
        )
        XCTAssertEqual(migrated, "legacy-secret")
        XCTAssertEqual(TesseraSecretStore.secret(account: account), "legacy-secret")
        XCTAssertNil(defaults.string(forKey: defaultsKey), "plaintext copy must be deleted")
    }

    func testMigrateLegacyNoValueReturnsNil() {
        let defaults = UserDefaults(suiteName: "tessera-secret-migration-test")!
        let defaultsKey = "absent-\(UUID().uuidString)"
        XCTAssertNil(TesseraSecretStore.migrateLegacy(
            defaultsKey: defaultsKey, account: account, defaults: defaults
        ))
    }
}
