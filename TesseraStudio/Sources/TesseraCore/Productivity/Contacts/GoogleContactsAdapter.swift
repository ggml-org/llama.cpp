import Foundation
import Security

// MARK: - GoogleContactsAdapter

/// Read contacts from Google People API. The adapter is opt-in
/// (the user explicitly grants OAuth access in Settings); it
/// stores the OAuth token via the existing ``TesseraKeychainVolume``
/// infra so the token doesn't end up in the encrypted-volume
/// plaintext.
///
/// **Auth flow:** the user pastes a Google OAuth client ID +
/// secret into Settings. The adapter opens an
/// `ASWebAuthenticationSession` against Google's OAuth
/// endpoint, the user signs in + grants the
/// `https://www.googleapis.com/auth/contacts.readonly` scope,
/// and the resulting authorization code is exchanged for a
/// refresh + access token. The refresh token is stored in
/// Keychain; the access token is in-memory only.
///
/// **Why the spec uses `ASWebAuthenticationSession` and not a
/// custom URL scheme:** the system browser session is the
/// only path that doesn't require a private-use URL scheme
/// registered in Info.plist, and the spec's "no keychain
/// entitlements beyond the system volume" constraint means
/// custom URL schemes are out.
///
/// **Test path:** the URL session is injected so tests can
/// supply a `URLProtocol` subclass that returns canned
/// responses. The OAuth flow itself is skipped in tests —
/// the constructor takes a pre-built ``GoogleOAuthToken`` when
/// the caller already has one (the test fixture).
public actor GoogleContactsAdapter {

    public struct Configuration: Sendable, Equatable {
        public var clientID: String
        public var clientSecret: String
        public var redirectURI: String

        public init(clientID: String, clientSecret: String, redirectURI: String) {
            self.clientID = clientID
            self.clientSecret = clientSecret
            self.redirectURI = redirectURI
        }
    }

    public struct GoogleOAuthToken: Codable, Sendable, Equatable {
        public var accessToken: String
        public var refreshToken: String
        public var expiresAt: Date
        public var tokenType: String

        public init(
            accessToken: String,
            refreshToken: String,
            expiresAt: Date,
            tokenType: String = "Bearer"
        ) {
            self.accessToken = accessToken
            self.refreshToken = refreshToken
            self.expiresAt = expiresAt
            self.tokenType = tokenType
        }

        /// True when the access token has less than 60 seconds of
        /// validity remaining. ``refreshTokenIfNeeded()`` uses
        /// this to decide whether a refresh is required.
        public var isExpiringSoon: Bool {
            expiresAt.timeIntervalSinceNow < 60
        }
    }

    public struct GooglePerson: Codable, Sendable {
        public let resourceName: String
        public let etag: String?
        public let names: [GoogleName]?
        public let emailAddresses: [GoogleEmail]?
        public let phoneNumbers: [GooglePhone]?
        public let organizations: [GoogleOrganization]?
        public let birthdays: [GoogleBirthday]?
        public let photos: [GooglePhoto]?

        public init(
            resourceName: String,
            etag: String? = nil,
            names: [GoogleName]? = nil,
            emailAddresses: [GoogleEmail]? = nil,
            phoneNumbers: [GooglePhone]? = nil,
            organizations: [GoogleOrganization]? = nil,
            birthdays: [GoogleBirthday]? = nil,
            photos: [GooglePhoto]? = nil
        ) {
            self.resourceName = resourceName
            self.etag = etag
            self.names = names
            self.emailAddresses = emailAddresses
            self.phoneNumbers = phoneNumbers
            self.organizations = organizations
            self.birthdays = birthdays
            self.photos = photos
        }
    }

    public struct GooglePhoto: Codable, Sendable {
        public let url: String?
        public let default_: Bool?

        public init(url: String? = nil, default_: Bool? = nil) {
            self.url = url
            self.default_ = default_
        }

        enum CodingKeys: String, CodingKey {
            case url
            case default_ = "default"
        }
    }

    public struct GoogleName: Codable, Sendable {
        public let displayName: String?
        public let givenName: String?
        public let middleName: String?
        public let familyName: String?
        public let honorificPrefix: String?
        public let honorificSuffix: String?

        public init(
            displayName: String? = nil,
            givenName: String? = nil,
            middleName: String? = nil,
            familyName: String? = nil,
            honorificPrefix: String? = nil,
            honorificSuffix: String? = nil
        ) {
            self.displayName = displayName
            self.givenName = givenName
            self.middleName = middleName
            self.familyName = familyName
            self.honorificPrefix = honorificPrefix
            self.honorificSuffix = honorificSuffix
        }
    }

    public struct GoogleEmail: Codable, Sendable {
        public let value: String
        public let type: String?
        public let formattedType: String?

        public init(value: String, type: String? = nil, formattedType: String? = nil) {
            self.value = value
            self.type = type
            self.formattedType = formattedType
        }
    }

    public struct GooglePhone: Codable, Sendable {
        public let value: String
        public let type: String?
        public let formattedType: String?

        public init(value: String, type: String? = nil, formattedType: String? = nil) {
            self.value = value
            self.type = type
            self.formattedType = formattedType
        }
    }

    public struct GoogleOrganization: Codable, Sendable {
        public let name: String?
        public let title: String?

        public init(name: String? = nil, title: String? = nil) {
            self.name = name
            self.title = title
        }
    }

    public struct GoogleBirthday: Codable, Sendable {
        public let date: GoogleDate?

        public init(date: GoogleDate? = nil) {
            self.date = date
        }
    }

    public struct GoogleDate: Codable, Sendable {
        public let year: Int?
        public let month: Int?
        public let day: Int?

        public init(year: Int? = nil, month: Int? = nil, day: Int? = nil) {
            self.year = year
            self.month = month
            self.day = day
        }
    }

    /// The top-level "list connections" response.
    public struct ConnectionsResponse: Codable, Sendable {
        public let connections: [GooglePerson]?
        public let nextPageToken: String?
        public let nextSyncToken: String?
    }

    // MARK: - Stored state

    private let configuration: Configuration
    private let session: URLSession
    private var token: GoogleOAuthToken?
    /// The Keychain account name under which the refresh token
    /// is persisted. Namespaced per-Google-account so the user
    /// can connect multiple Google accounts.
    private let keychainAccount: String

    public init(
        configuration: Configuration,
        session: URLSession = .shared,
        keychainAccount: String = "google-contacts-default"
    ) throws {
        self.configuration = configuration
        self.session = session
        self.keychainAccount = keychainAccount
    }

    /// Construct with a pre-built token (test path). The token
    /// is NOT persisted to Keychain; the caller is responsible
    /// for storing it.
    public init(
        configuration: Configuration,
        session: URLSession = .shared,
        initialToken: GoogleOAuthToken
    ) throws {
        self.configuration = configuration
        self.session = session
        self.token = initialToken
        self.keychainAccount = "google-contacts-default"
    }

    // MARK: - Authentication

    /// Begin the OAuth flow. The actual browser session is
    /// driven by `ASWebAuthenticationSession` on the main
    /// actor; this method returns the URL the browser should
    /// open (the caller wires the session to it).
    ///
    /// In tests this method is skipped; the test fixture
    /// constructs the adapter with a pre-built token.
    public func makeAuthorizationURL(
        state: String = UUID().uuidString,
        scopes: [String] = ["https://www.googleapis.com/auth/contacts.readonly"]
    ) -> URL {
        var components = URLComponents(string: "https://accounts.google.com/o/oauth2/v2/auth")!
        components.queryItems = [
            URLQueryItem(name: "client_id", value: configuration.clientID),
            URLQueryItem(name: "redirect_uri", value: configuration.redirectURI),
            URLQueryItem(name: "response_type", value: "code"),
            URLQueryItem(name: "scope", value: scopes.joined(separator: " ")),
            URLQueryItem(name: "state", value: state),
            URLQueryItem(name: "access_type", value: "offline"),
            URLQueryItem(name: "prompt", value: "consent"),
        ]
        return components.url!
    }

    /// Exchange an authorization code for an OAuth token. The
    /// caller (the `ASWebAuthenticationSession` completion
    /// handler) passes the `code` from the redirect URL. The
    /// returned token is stored in memory and the refresh
    /// token is persisted to Keychain.
    public func authenticate(authorizationCode code: String) async throws -> GoogleOAuthToken {
        var request = URLRequest(url: URL(string: "https://oauth2.googleapis.com/token")!)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        let bodyParams: [(String, String)] = [
            ("code", code),
            ("client_id", configuration.clientID),
            ("client_secret", configuration.clientSecret),
            ("redirect_uri", configuration.redirectURI),
            ("grant_type", "authorization_code"),
        ]
        request.httpBody = bodyParams
            .map { "\($0.0)=\(Self.urlEncode($0.1))" }
            .joined(separator: "&")
            .data(using: .utf8)
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw GoogleContactsError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw GoogleContactsError.tokenExchangeFailed(status: http.statusCode, body: body)
        }
        let token = try Self.parseTokenResponse(data: data)
        self.token = token
        Self.storeRefreshToken(token.refreshToken, account: keychainAccount)
        return token
    }

    /// Refresh the access token using the stored refresh token.
    /// Idempotent: if the access token is still valid, returns
    /// the existing token without making a network call.
    public func refreshTokenIfNeeded() async throws {
        guard let current = token else {
            throw GoogleContactsError.notAuthenticated
        }
        if !current.isExpiringSoon { return }
        var request = URLRequest(url: URL(string: "https://oauth2.googleapis.com/token")!)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        let bodyParams: [(String, String)] = [
            ("refresh_token", current.refreshToken),
            ("client_id", configuration.clientID),
            ("client_secret", configuration.clientSecret),
            ("grant_type", "refresh_token"),
        ]
        request.httpBody = bodyParams
            .map { "\($0.0)=\(Self.urlEncode($0.1))" }
            .joined(separator: "&")
            .data(using: .utf8)
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            throw GoogleContactsError.tokenRefreshFailed
        }
        let refreshed = try Self.parseTokenResponse(data: data, prior: current)
        self.token = refreshed
    }

    // MARK: - Fetch

    /// Fetch every contact in the user's Google account. Uses
    /// the `people.connections.list` endpoint with the
    /// `personFields=names,emailAddresses,phoneNumbers,organizations,birthdays,photos`
    /// field mask (the People API requires an explicit field
    /// mask; without it the response is empty).
    public func fetchAllContacts() async throws -> [Contact] {
        try await refreshTokenIfNeeded()
        guard let current = token else {
            throw GoogleContactsError.notAuthenticated
        }
        var allContacts: [Contact] = []
        var pageToken: String? = nil
        repeat {
            var components = URLComponents(string: "https://people.googleapis.com/v1/people/me/connections")!
            var items: [URLQueryItem] = [
                URLQueryItem(
                    name: "personFields",
                    value: "names,emailAddresses,phoneNumbers,organizations,birthdays,photos"
                ),
                URLQueryItem(name: "pageSize", value: "1000"),
            ]
            if let pageToken {
                items.append(URLQueryItem(name: "pageToken", value: pageToken))
            }
            components.queryItems = items
            var request = URLRequest(url: components.url!)
            request.setValue("\(current.tokenType) \(current.accessToken)", forHTTPHeaderField: "Authorization")
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse,
                  (200..<300).contains(http.statusCode) else {
                throw GoogleContactsError.fetchFailed(
                    status: (response as? HTTPURLResponse)?.statusCode ?? -1
                )
            }
            let decoded = try JSONDecoder().decode(ConnectionsResponse.self, from: data)
            for person in decoded.connections ?? [] {
                allContacts.append(Self.contact(from: person))
            }
            pageToken = decoded.nextPageToken
        } while pageToken != nil
        return allContacts
    }

    // MARK: - Translation (Google -> Contact)

    /// Translate a Google People API person into our `Contact`.
    /// The Google model is rich (covers every field VCard does
    /// plus social profiles, IM handles, ...) but we map only
    /// the fields the productivity surface cares about; the
    /// `sourceURL` is set to the resource name so a re-import
    /// can detect the existing contact.
    public static func contact(from person: GooglePerson) -> Contact {
        let primaryName = person.names?.first
        let components = NameComponents(
            prefix: primaryName?.honorificPrefix,
            first: primaryName?.givenName,
            middle: primaryName?.middleName,
            last: primaryName?.familyName,
            suffix: primaryName?.honorificSuffix
        )
        let emails: [LabeledEmail] = (person.emailAddresses ?? []).map { e in
            LabeledEmail(
                label: map(emailType: e.type),
                value: e.value,
                isPrimary: false
            )
        }
        let phones: [LabeledPhone] = (person.phoneNumbers ?? []).map { p in
            LabeledPhone(
                label: map(phoneType: p.type),
                value: p.value,
                isPrimary: false
            )
        }
        let organization = person.organizations?.first?.name
        let title = person.organizations?.first?.title
        let birthday: Date? = {
            guard let date = person.birthdays?.first?.date else { return nil }
            guard let year = date.year, let month = date.month, let day = date.day else {
                return nil
            }
            var c = DateComponents()
            c.year = year
            c.month = month
            c.day = day
            return Calendar(identifier: .gregorian).date(from: c)
        }()
        return Contact(
            subtype: .person,
            name: components,
            emails: emails,
            phones: phones,
            addresses: [],
            organization: organization,
            title: title,
            birthday: birthday,
            photo: nil,
            notes: nil,
            sourceURL: person.resourceName,
            linkedEntityIDs: []
        )
    }

    private static func map(emailType: String?) -> LabeledEmail.Label {
        switch emailType {
        case "home": return .home
        case "work": return .work
        case "other": return .other
        case .some(let s): return .custom(s)
        case nil: return .other
        }
    }

    private static func map(phoneType: String?) -> LabeledPhone.Label {
        switch phoneType {
        case "mobile": return .mobile
        case "work": return .work
        case "home": return .home
        case "main": return .main
        case "fax": return .fax
        case "other": return .other
        case .some(let s): return .custom(s)
        case nil: return .other
        }
    }

    // MARK: - Helpers

    private static func parseTokenResponse(
        data: Data,
        prior: GoogleOAuthToken? = nil
    ) throws -> GoogleOAuthToken {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw GoogleContactsError.tokenParseFailed
        }
        guard let accessToken = json["access_token"] as? String else {
            throw GoogleContactsError.tokenParseFailed
        }
        let refreshToken = (json["refresh_token"] as? String) ?? prior?.refreshToken ?? ""
        let expiresIn = (json["expires_in"] as? Int) ?? 3600
        let tokenType = (json["token_type"] as? String) ?? "Bearer"
        return GoogleOAuthToken(
            accessToken: accessToken,
            refreshToken: refreshToken,
            expiresAt: Date().addingTimeInterval(TimeInterval(expiresIn)),
            tokenType: tokenType
        )
    }

    /// URL-encode a string for a form body. We don't pull in a
    /// full URL encoder for one call site; this is the minimum
    /// we need to encode the token-exchange form fields.
    private static func urlEncode(_ s: String) -> String {
        var allowed = CharacterSet.urlQueryAllowed
        allowed.remove(charactersIn: "+&=?")
        return s.addingPercentEncoding(withAllowedCharacters: allowed) ?? s
    }

    private static func storeRefreshToken(_ token: String, account: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: TesseraSecretStore.service,
            kSecAttrAccount as String: account,
        ]
        let data = Data(token.utf8)
        let updateStatus = SecItemUpdate(
            query as CFDictionary,
            [kSecValueData as String: data] as CFDictionary
        )
        if updateStatus == errSecSuccess { return }
        guard updateStatus == errSecItemNotFound else { return }
        var addQuery = query
        addQuery[kSecValueData as String] = data
        addQuery[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        addQuery[kSecAttrSynchronizable as String] = kCFBooleanFalse
        SecItemAdd(addQuery as CFDictionary, nil)
    }
}

// MARK: - Errors

public enum GoogleContactsError: Error, Sendable, Equatable {
    case invalidResponse
    case notAuthenticated
    case tokenExchangeFailed(status: Int, body: String)
    case tokenRefreshFailed
    case tokenParseFailed
    case fetchFailed(status: Int)
}
