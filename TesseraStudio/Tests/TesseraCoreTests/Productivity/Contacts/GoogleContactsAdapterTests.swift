import XCTest
@testable import TesseraCore

/// Tests for ``GoogleContactsAdapter``. We exercise the
/// non-OAuth surface (token parsing, person translation,
/// URL construction). The OAuth web flow is skipped;
/// ``init(configuration:session:initialToken:)`` lets the
/// test fixture pre-build a token.
final class GoogleContactsAdapterTests: XCTestCase {

    func testTokenIsExpiringSoon() {
        let token = GoogleContactsAdapter.GoogleOAuthToken(
            accessToken: "a",
            refreshToken: "r",
            expiresAt: Date().addingTimeInterval(30)
        )
        XCTAssertTrue(token.isExpiringSoon)
    }

    func testTokenNotExpiring() {
        let token = GoogleContactsAdapter.GoogleOAuthToken(
            accessToken: "a",
            refreshToken: "r",
            expiresAt: Date().addingTimeInterval(3600)
        )
        XCTAssertFalse(token.isExpiringSoon)
    }

    func testPersonTranslation() {
        let person = GoogleContactsAdapter.GooglePerson(
            resourceName: "people/c123",
            etag: "etag",
            names: [
                GoogleContactsAdapter.GoogleName(
                    displayName: "Marie Curie",
                    givenName: "Marie",
                    familyName: "Curie"
                )
            ],
            emailAddresses: [
                GoogleContactsAdapter.GoogleEmail(value: "marie@radium.org", type: "work")
            ],
            phoneNumbers: [
                GoogleContactsAdapter.GooglePhone(value: "+1-555-0100", type: "mobile")
            ],
            organizations: [
                GoogleContactsAdapter.GoogleOrganization(
                    name: "Sorbonne",
                    title: "Professor"
                )
            ],
            birthdays: nil,
            photos: nil
        )
        let c = GoogleContactsAdapter.contact(from: person)
        XCTAssertEqual(c.subtype, Contact.Subtype.person)
        XCTAssertEqual(c.name.first, "Marie")
        XCTAssertEqual(c.name.last, "Curie")
        XCTAssertEqual(c.organization, "Sorbonne")
        XCTAssertEqual(c.title, "Professor")
        XCTAssertEqual(c.emails.first?.value, "marie@radium.org")
        XCTAssertEqual(c.phones.first?.value, "+1-555-0100")
        XCTAssertEqual(c.sourceURL, "people/c123")
    }

    func testPersonTranslationWithBirthday() {
        let person = GoogleContactsAdapter.GooglePerson(
            resourceName: "people/c1",
            names: nil,
            emailAddresses: nil,
            phoneNumbers: nil,
            organizations: nil,
            birthdays: [
                GoogleContactsAdapter.GoogleBirthday(
                    date: GoogleContactsAdapter.GoogleDate(year: 1980, month: 4, day: 12)
                )
            ],
            photos: nil
        )
        let c = GoogleContactsAdapter.contact(from: person)
        let comps = Calendar(identifier: .gregorian).dateComponents(
            [.year, .month, .day], from: c.birthday ?? Date()
        )
        XCTAssertEqual(comps.year, 1980)
        XCTAssertEqual(comps.month, 4)
        XCTAssertEqual(comps.day, 12)
    }

    func testPersonTranslationWithMissingOptionalFields() {
        let person = GoogleContactsAdapter.GooglePerson(resourceName: "people/c1")
        let c = GoogleContactsAdapter.contact(from: person)
        XCTAssertEqual(c.displayName, "Unnamed")
        XCTAssertTrue(c.emails.isEmpty)
        XCTAssertTrue(c.phones.isEmpty)
    }

    func testAuthorizationURLContainsRequiredParams() async throws {
        let adapter = try GoogleContactsAdapter(
            configuration: .init(
                clientID: "client-id",
                clientSecret: "secret",
                redirectURI: "https://app/callback"
            )
        )
        let url = await adapter.makeAuthorizationURL(
            state: "test-state",
            scopes: ["https://www.googleapis.com/auth/contacts.readonly"]
        )
        let s = url.absoluteString
        XCTAssertTrue(s.contains("client_id=client-id"))
        XCTAssertTrue(s.contains("state=test-state"))
        XCTAssertTrue(s.contains("contacts.readonly"))
    }

    func testAdapterConstructionWithInitialToken() async throws {
        let adapter = try GoogleContactsAdapter(
            configuration: .init(
                clientID: "id",
                clientSecret: "secret",
                redirectURI: "https://app/callback"
            ),
            initialToken: .init(
                accessToken: "a",
                refreshToken: "r",
                expiresAt: Date().addingTimeInterval(3600)
            )
        )
        // Token stays valid; refresh should be a no-op
        // (no network call made because the expiry is
        // an hour out).
        try await adapter.refreshTokenIfNeeded()
    }
}
