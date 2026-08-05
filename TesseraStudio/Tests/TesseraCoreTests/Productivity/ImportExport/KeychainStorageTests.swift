import XCTest
@testable import TesseraCore

/// Tests for the Keychain storage wrapper. These tests
/// require the test process to have access to the macOS
/// Keychain; they are skipped when running in a sandbox
/// that denies Keychain access (e.g. some CI environments).
final class KeychainStorageTests: XCTestCase {

    /// Use a unique service per test so we don't collide
    /// with other tests in the same process.
    private var storage: KeychainStorage!
    private var uniqueService: String!

    override func setUp() {
        super.setUp()
        uniqueService = "com.tessera.test.import-export.\(UUID().uuidString)"
        storage = KeychainStorage()
        storage.service = uniqueService
        storage.account = "test-user"
    }

    override func tearDown() {
        // Best-effort cleanup.
        try? storage.deleteWebhookURL()
        storage = nil
        super.tearDown()
    }

    /// Setting and getting a webhook URL round-trips.
    func testSetGetRoundTrip() throws {
        let url = URL(string: "https://hooks.slack.com/services/T0/B0/abcdef")!
        try storage.setWebhookURL(url)
        let read = try storage.getWebhookURL()
        XCTAssertEqual(read, url, "set and then get should return the same URL")
    }

    /// ``getWebhookURL`` returns nil when nothing has been
    /// stored.
    func testGetReturnsNilWhenEmpty() throws {
        // tearDown already removed any leftover from a prior
        // test; assert that.
        let read = try storage.getWebhookURL()
        XCTAssertNil(read, "expected nil before any set; got \(String(describing: read))")
    }

    /// Setting a webhook URL twice replaces the first
    /// value (the implementation deletes before adding).
    func testSetReplaces() throws {
        let url1 = URL(string: "https://hooks.slack.com/services/T0/B0/one")!
        let url2 = URL(string: "https://hooks.slack.com/services/T0/B0/two")!
        try storage.setWebhookURL(url1)
        try storage.setWebhookURL(url2)
        let read = try storage.getWebhookURL()
        XCTAssertEqual(read, url2, "second set should replace the first")
    }

    /// ``deleteWebhookURL`` is a no-op when nothing has
    /// been stored.
    func testDeleteIsIdempotent() throws {
        XCTAssertNoThrow(try storage.deleteWebhookURL())
        XCTAssertNoThrow(try storage.deleteWebhookURL())
    }
}
