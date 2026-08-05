import Foundation

// MARK: - CardDAVImporter

/// Read contacts from a CardDAV server (iCloud, Fastmail,
/// Nextcloud, ...). CardDAV is "just" WebDAV with a contact
/// payload (VCard) and a sync-token model for incremental
/// updates. The protocol is documented in RFC 6352; the
/// sync-token flow is in RFC 6578.
///
/// **Why we implement it directly:** there's no mature Swift
/// CardDAV library and the surface we need is small:
///   * PROPFIND against the user's principal URL to discover
///     the address-book URL.
///   * PROPFIND against the address-book URL to enumerate
///     the contact hrefs + their etags.
///   * REPORT (addressbook-query) with a time-range filter
///     for sync-token deltas.
///   * GET on each contact href to fetch the VCard body.
///
/// All four operations are HTTP requests with XML request
/// bodies and XML responses; we use Foundation's `URLSession`
/// + `XMLParser` to avoid pulling in a heavyweight XML
/// library.
///
/// **Auth:** Basic with an app-specific password (the same
/// shape iCloud / Fastmail / Nextcloud all use). The
/// password is supplied via ``Configuration`` and stored in
/// Keychain via ``TesseraSecretStore`` (not in
/// ``Configuration`` directly) once the user enters it.
public actor CardDAVImporter {

    public struct Configuration: Sendable, Equatable {
        public var serverURL: URL
        public var username: String
        public var password: String
        /// Initial path the importer should probe for the
        /// principal. iCloud uses `/.well-known/carddav`
        /// and most self-hosted servers use `/principals/<user>`.
        public var principalPath: String

        public init(
            serverURL: URL,
            username: String,
            password: String,
            principalPath: String = "/.well-known/carddav"
        ) {
            self.serverURL = serverURL
            self.username = username
            self.password = password
            self.principalPath = principalPath
        }
    }

    /// A single address-book contact as returned by the
    /// server: href (the path the importer should GET to
    /// fetch the VCard), etag (for sync), vcard (the
    /// parsed body, populated by ``fetchAllContacts()``).
    public struct CardDAVContact: Sendable, Equatable {
        public let href: String
        public let etag: String
        public var contact: Contact?

        public init(href: String, etag: String, contact: Contact? = nil) {
            self.href = href
            self.etag = etag
            self.contact = contact
        }
    }

    /// The result of a sync-token delta query. The importer
    /// stores the `newSyncToken` and passes it on the next
    /// call to ``fetchChanges(since:)`` to receive only
    /// the rows that changed since the token was issued.
    public struct ContactDelta: Sendable, Equatable {
        public var upserted: [CardDAVContact]
        public var removedHrefs: [String]
        public var newSyncToken: String?

        public init(
            upserted: [CardDAVContact] = [],
            removedHrefs: [String] = [],
            newSyncToken: String? = nil
        ) {
            self.upserted = upserted
            self.removedHrefs = removedHrefs
            self.newSyncToken = newSyncToken
        }
    }

    // MARK: - State

    private let configuration: Configuration
    private let session: URLSession
    private var addressBookURL: URL?
    private var lastSyncToken: String?

    public init(
        configuration: Configuration,
        session: URLSession = .shared
    ) throws {
        self.configuration = configuration
        self.session = session
    }

    /// Set the address book URL explicitly (skips the
    /// discovery flow). The UI uses this when the user has
    /// already configured their account and the importer
    /// has cached the address book URL.
    public func setAddressBookURL(_ url: URL) {
        self.addressBookURL = url
    }

    /// Set the sync token from a prior session (loaded from
    /// the database by the caller).
    public func setSyncToken(_ token: String?) {
        self.lastSyncToken = token
    }

    // MARK: - Discovery

    /// Discover the user's principal URL. The flow is:
    ///   1. PROPFIND the well-known principal URL with
    ///      Depth=0 asking for the `current-user-principal`
    ///      property.
    ///   2. Follow the returned href.
    ///   3. PROPFIND again with Depth=0 asking for the
    ///      `addressbook-home-set` or `calendar-home-set`
    ///      (some servers use the calendar-home-set for
    ///      the CardDAV root, which is the original RFC
    ///      4791 behaviour).
    public func discoverPrincipal() async throws -> URL {
        let url = configuration.serverURL
            .appendingPathComponent(configuration.principalPath)
        let body = Self.propfindBody(props: ["DAV: current-user-principal"])
        let (data, response) = try await sendRequest(
            url: url,
            method: "PROPFIND",
            depth: "0",
            body: body
        )
        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            throw CardDAVError.discoveryFailed(
                status: (response as? HTTPURLResponse)?.statusCode ?? -1,
                body: String(data: data, encoding: .utf8) ?? ""
            )
        }
        let parser = CardDAVXMLParser()
        parser.parse(data: data)
        guard let principalHref = parser.firstHref else {
            throw CardDAVError.principalNotFound
        }
        return URL(string: principalHref, relativeTo: configuration.serverURL) ?? url
    }

    /// Discover the address-book URL for the principal.
    /// Issues PROPFIND asking for the `addressbook-home-set`
    /// property.
    public func discoverAddressBookURL() async throws -> URL {
        let principal = try await discoverPrincipal()
        let body = Self.propfindBody(props: [
            "DAV: addressbook-home-set",
            "urn:ietf:params:xml:ns:carddav addressbook-home-set",
        ])
        let (data, response) = try await sendRequest(
            url: principal,
            method: "PROPFIND",
            depth: "0",
            body: body
        )
        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            throw CardDAVError.discoveryFailed(
                status: (response as? HTTPURLResponse)?.statusCode ?? -1,
                body: String(data: data, encoding: .utf8) ?? ""
            )
        }
        let parser = CardDAVXMLParser()
        parser.parse(data: data)
        guard let href = parser.firstHref else {
            throw CardDAVError.addressBookNotFound
        }
        let resolved = URL(string: href, relativeTo: configuration.serverURL) ?? principal
        self.addressBookURL = resolved
        return resolved
    }

    // MARK: - Fetch

    /// Fetch every contact. The flow is:
    ///   1. Discover the address-book URL (or reuse the
    ///      cached value if the caller set one).
    ///   2. PROPFIND Depth=1 to enumerate hrefs + etags.
    ///   3. GET each href in turn, parse the VCard body
    ///      via the ``VCardImporter``.
    public func fetchAllContacts() async throws -> [Contact] {
        let bookURL: URL
        if let cached = addressBookURL {
            bookURL = cached
        } else {
            bookURL = try await discoverAddressBookURL()
        }
        // PROPFIND for href + etag at Depth=1.
        let propfindBody = Self.propfindBody(props: [
            "DAV: getetag",
            "DAV: resourcetype",
        ])
        let (data, response) = try await sendRequest(
            url: bookURL,
            method: "PROPFIND",
            depth: "1",
            body: propfindBody
        )
        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            throw CardDAVError.listFailed(
                status: (response as? HTTPURLResponse)?.statusCode ?? -1,
                body: String(data: data, encoding: .utf8) ?? ""
            )
        }
        let parser = CardDAVXMLParser()
        parser.parse(data: data)
        let entries = parser.responses.filter { $0.href.hasSuffix(".vcf") || $0.href.contains("/") }
        var contacts: [Contact] = []
        for entry in entries {
            let contactURL = URL(string: entry.href, relativeTo: bookURL) ?? bookURL
            let (vcardData, vcardResponse) = try await sendRequest(
                url: contactURL,
                method: "GET",
                depth: nil,
                body: nil
            )
            guard let vhttp = vcardResponse as? HTTPURLResponse,
                  (200..<300).contains(vhttp.statusCode) else {
                continue
            }
            let importer = VCardImporter()
            do {
                let parsed = try await importer.parse(data: vcardData)
                if var c = parsed.first {
                    c.sourceURL = entry.href
                    contacts.append(c)
                }
            } catch {
                // Skip malformed entries; don't abort the
                // whole import on one bad contact.
                continue
            }
        }
        return contacts
    }

    /// Incremental sync. Issues a `REPORT` (addressbook-query)
    /// with a `sync-token` filter to receive only the rows
    /// that changed since the last sync. The response carries
    /// a new `syncToken`; the caller persists it for the next
    /// delta.
    public func fetchChanges(since syncToken: String) async throws -> ContactDelta {
        let bookURL: URL
        if let cached = addressBookURL {
            bookURL = cached
        } else {
            bookURL = try await discoverAddressBookURL()
        }
        let body = Self.syncCollectionBody(syncToken: syncToken)
        let (data, response) = try await sendRequest(
            url: bookURL,
            method: "REPORT",
            depth: "1",
            body: body
        )
        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            throw CardDAVError.syncFailed(
                status: (response as? HTTPURLResponse)?.statusCode ?? -1,
                body: String(data: data, encoding: .utf8) ?? ""
            )
        }
        let parser = CardDAVXMLParser()
        parser.parse(data: data)
        var upserted: [CardDAVContact] = []
        for entry in parser.responses {
            let contactURL = URL(string: entry.href, relativeTo: bookURL) ?? bookURL
            if let vcardData = try? await fetchVCard(at: contactURL) {
                let importer = VCardImporter()
                if let parsed = try? await importer.parse(data: vcardData),
                   var c = parsed.first {
                    c.sourceURL = entry.href
                    upserted.append(CardDAVContact(
                        href: entry.href,
                        etag: entry.etag ?? "",
                        contact: c
                    ))
                }
            }
        }
        return ContactDelta(
            upserted: upserted,
            removedHrefs: parser.removedHrefs,
            newSyncToken: parser.syncToken
        )
    }

    // MARK: - HTTP

    private func sendRequest(
        url: URL,
        method: String,
        depth: String?,
        body: Data?
    ) async throws -> (Data, URLResponse) {
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/xml; charset=utf-8", forHTTPHeaderField: "Content-Type")
        request.setValue("text/xml", forHTTPHeaderField: "Accept")
        if let depth {
            request.setValue(depth, forHTTPHeaderField: "Depth")
        }
        if let body {
            request.httpBody = body
        }
        // Basic auth header.
        let credentials = "\(configuration.username):\(configuration.password)"
        if let credData = credentials.data(using: .utf8) {
            let base64 = credData.base64EncodedString()
            request.setValue("Basic \(base64)", forHTTPHeaderField: "Authorization")
        }
        return try await session.data(for: request)
    }

    private func fetchVCard(at url: URL) async throws -> Data {
        let (data, response) = try await sendRequest(
            url: url,
            method: "GET",
            depth: nil,
            body: nil
        )
        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            throw CardDAVError.fetchFailed
        }
        return data
    }

    // MARK: - XML bodies

    /// Build a PROPFIND request body. `props` is the list of
    /// property names; the formatter writes the proper
    /// `<d:prop>` element with namespace declarations.
    static func propfindBody(props: [String]) -> Data {
        var s = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        s += "<d:propfind xmlns:d=\"DAV:\" xmlns:cr=\"urn:ietf:params:xml:ns:carddav\">\n"
        s += "  <d:prop>\n"
        for prop in props {
            // Accept both "DAV: current-user-principal" and
            // "urn:ietf:params:xml:ns:carddav addressbook-home-set"
            // (the form is "namespace localName" with the
            // namespace binding declared on the root).
            let parts = prop.split(separator: " ", maxSplits: 1).map(String.init)
            let (ns, localName) = (parts.first ?? "DAV:", parts.count > 1 ? parts[1] : parts[0])
            let prefix: String
            switch ns {
            case "DAV:": prefix = "d"
            case "urn:ietf:params:xml:ns:carddav": prefix = "cr"
            default: prefix = "d"
            }
            s += "    <\(prefix):\(localName)/>\n"
        }
        s += "  </d:prop>\n"
        s += "</d:propfind>\n"
        return Data(s.utf8)
    }

    /// Build a sync-collection REPORT body (RFC 6578).
    static func syncCollectionBody(syncToken: String) -> Data {
        var s = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        s += "<d:sync-collection xmlns:d=\"DAV:\">\n"
        s += "  <d:sync-token>\(syncToken)</d:sync-token>\n"
        s += "  <d:prop>\n"
        s += "    <d:getetag/>\n"
        s += "  </d:prop>\n"
        s += "</d:sync-collection>\n"
        return Data(s.utf8)
    }
}

// MARK: - XML parser

/// Minimal SAX-style parser for CardDAV multistatus
/// responses. Extracts hrefs, etags, sync tokens, and
/// removed-href lists. The parser is stateful (one instance
/// per response); call ``parse(data:)`` and then read the
/// `responses` / `syncToken` / `removedHrefs` fields.
final class CardDAVXMLParser: NSObject, XMLParserDelegate {

    struct Response: Equatable {
        var href: String
        var etag: String?
    }

    var responses: [Response] = []
    var removedHrefs: [String] = []
    var syncToken: String?
    var firstHref: String? { responses.first?.href }

    private var currentHref: String?
    private var currentEtag: String?
    private var currentText: String = ""
    private var elementStack: [String] = []
    private var isRemoved: Bool = false

    func parse(data: Data) {
        responses = []
        removedHrefs = []
        syncToken = nil
        currentHref = nil
        currentEtag = nil
        currentText = ""
        elementStack = []
        isRemoved = false
        let parser = XMLParser(data: data)
        parser.delegate = self
        parser.parse()
    }

    func parser(_ parser: XMLParser, didStartElement elementName: String,
                namespaceURI: String?, qualifiedName qName: String?,
                attributes attributeDict: [String : String] = [:]) {
        elementStack.append(elementName)
        currentText = ""
        // Some XML parsers pass the qualified name with
        // the namespace prefix; match either form for
        // element comparison.
        let localName = elementName.contains(":")
            ? String(elementName.split(separator: ":").last ?? "")
            : elementName
        if localName == "response" {
            currentHref = nil
            currentEtag = nil
            isRemoved = false
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        currentText += string
    }

    func parser(_ parser: XMLParser, didEndElement elementName: String,
                namespaceURI: String?, qualifiedName qName: String?) {
        // The element is still on the stack at this
        // point; remove it AFTER we read the text and
        // check the parent context.
        let parentStack = elementStack
        let trimmed = currentText.trimmingCharacters(in: .whitespacesAndNewlines)
        // Some XML parsers pass the qualified name with
        // the namespace prefix (e.g. "d:href"). Match
        // either form.
        let localName = elementName.contains(":")
            ? String(elementName.split(separator: ":").last ?? "")
            : elementName
        switch localName {
        case "href":
            if parentStack.contains("response") || parentStack.contains("d:response") {
                if currentHref == nil {
                    currentHref = trimmed
                } else {
                    responses.append(Response(href: trimmed, etag: nil))
                }
            } else if isRemoved {
                removedHrefs.append(trimmed)
            }
        case "getetag":
            if !isRemoved {
                currentEtag = trimmed
            }
        case "response":
            if let href = currentHref {
                if isRemoved {
                    // The response was deleted on the
                    // server (the propstat was 404 or
                    // similar). Record the href in
                    // `removedHrefs` so the caller can
                    // delete the local copy.
                    if !removedHrefs.contains(href) {
                        removedHrefs.append(href)
                    }
                } else {
                    responses.append(Response(href: href, etag: currentEtag))
                }
            }
        case "status":
            if !trimmed.contains("200") {
                isRemoved = true
            }
        case "sync-token":
            syncToken = trimmed
        default:
            break
        }
        if !elementStack.isEmpty {
            elementStack.removeLast()
        }
        currentText = ""
    }
}

// MARK: - Errors

public enum CardDAVError: Error, Sendable, Equatable {
    case discoveryFailed(status: Int, body: String)
    case principalNotFound
    case addressBookNotFound
    case listFailed(status: Int, body: String)
    case syncFailed(status: Int, body: String)
    case fetchFailed
}
