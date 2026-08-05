import Foundation

// MARK: - Contact

/// A first-class material in the Tessera productivity surface.
///
/// Contacts are stored as `graph_entity` rows with `entity_type = 'contact'`,
/// matching the universal "one row per thing" pattern documented in
/// `docs/tessera-data-layer-design.md` §3. The `body` column carries
/// the contact fields as JSON (the `name`, `emails`, `phones`, etc.).
/// `label` is the display string ("Jane Doe" for a person, "Acme Inc."
/// for an organization); `subtype` discriminates the three shapes
/// (`.person` / `.organization` / `.group`).
///
/// The struct is intentionally self-contained: no references to the
/// data layer, no framework imports beyond `Foundation`. Importers
/// (Apple Contacts, VCard, Google, CardDAV) build `Contact` values
/// from their source format; the ``ContactStore`` writes them to
/// Postgres and produces the constitutional receipt.
public struct Contact: Codable, Sendable, Identifiable, Hashable {

    /// Stable identifier. New contacts get a fresh UUID at creation;
    /// importer round-trips preserve the existing id so re-imports
    /// upsert in place rather than duplicating.
    public let id: UUID

    /// The three shapes a contact can take. Drives default field
    /// interpretation: an `.organization` contact may have many
    /// members; a `.group` contact typically has no phone numbers.
    public var subtype: Subtype

    /// Display name parts. For `.person` the `first` + `last`
    /// combination drives the contact's display string; for
    /// `.organization` the `last` field carries the org name.
    public var name: NameComponents

    public var emails: [LabeledEmail]
    public var phones: [LabeledPhone]
    public var addresses: [LabeledAddress]

    /// For `.person`: the employer. For `.organization`: the parent
    /// organization. For `.group`: usually nil.
    public var organization: String?

    /// For `.person`: job title ("Staff Engineer", "CEO"). For
    /// `.organization`: typically nil.
    public var title: String?

    public var birthday: Date?
    public var photo: Data?
    public var notes: String?

    /// Where this contact came from (Apple Contacts URL, Google
    /// resource name, CardDAV href, file path for VCard import).
    /// Preserved for round-trip and auditability.
    public var sourceURL: String?

    /// Other graph entities this contact is linked to (recent
    /// emails, calendar events, tasks, documents). The graph view
    /// uses these as edges; the contact view lists them in the
    /// "related" section.
    public var linkedEntityIDs: [UUID]

    public var createdAt: Date
    public var updatedAt: Date

    public enum Subtype: String, Codable, Sendable, Hashable, CaseIterable {
        case person
        case organization
        case group
    }

    public init(
        id: UUID = UUID(),
        subtype: Subtype = .person,
        name: NameComponents,
        emails: [LabeledEmail] = [],
        phones: [LabeledPhone] = [],
        addresses: [LabeledAddress] = [],
        organization: String? = nil,
        title: String? = nil,
        birthday: Date? = nil,
        photo: Data? = nil,
        notes: String? = nil,
        sourceURL: String? = nil,
        linkedEntityIDs: [UUID] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.subtype = subtype
        self.name = name
        self.emails = emails
        self.phones = phones
        self.addresses = addresses
        self.organization = organization
        self.title = title
        self.birthday = birthday
        self.photo = photo
        self.notes = notes
        self.sourceURL = sourceURL
        self.linkedEntityIDs = linkedEntityIDs
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    // MARK: - Convenience display string

    /// Human-readable display string. The graph view uses the first
    /// 30 characters of this string as the node label; the contact
    /// list uses the full string.
    public var displayName: String {
        switch subtype {
        case .person:
            let parts = [name.first, name.middle, name.last].compactMap { $0 }
            let joined = parts.joined(separator: " ").trimmingCharacters(in: .whitespaces)
            if !joined.isEmpty { return joined }
            if let nickname = name.nickname, !nickname.isEmpty { return nickname }
            if let org = organization, !org.isEmpty { return org }
            return "Unnamed"
        case .organization:
            if let org = name.last, !org.isEmpty { return org }
            if let org = organization, !org.isEmpty { return org }
            return "Unnamed organization"
        case .group:
            if let g = name.last, !g.isEmpty { return g }
            return "Unnamed group"
        }
    }

    /// The `entity_type` value persisted in `graph_entities.entity_type`.
    /// Pinned so the contact view's `hybrid_search` filter and the
    /// graph view's "by type" grouping all agree.
    public static let entityType = "contact"

    /// The `subtype` value persisted in `graph_entities.subtype`.
    public var subtypeString: String { subtype.rawValue }
}

// MARK: - NameComponents

/// Parts of a person's name, modeled after Apple's `CNContact.namePrefix`
/// / `givenName` / `middleName` / `familyName` / `nameSuffix` fields.
/// For `.organization` and `.group` subtypes the `last` field carries
/// the org/group name (matching how macOS Contacts and Google People
/// both store an organization as a single "name" string).
public struct NameComponents: Codable, Sendable, Hashable {
    public var prefix: String?
    public var first: String?
    public var middle: String?
    public var last: String?
    public var suffix: String?
    public var nickname: String?

    public init(
        prefix: String? = nil,
        first: String? = nil,
        middle: String? = nil,
        last: String? = nil,
        suffix: String? = nil,
        nickname: String? = nil
    ) {
        self.prefix = prefix
        self.first = first
        self.middle = middle
        self.last = last
        self.suffix = suffix
        self.nickname = nickname
    }

    /// True when no name field has a value. Used by importers to
    /// decide whether a VCard's `N:` line was empty.
    public var isEmpty: Bool {
        [prefix, first, middle, last, suffix, nickname]
            .allSatisfy { ($0 ?? "").isEmpty }
    }
}

// MARK: - Labeled contacts

/// A value + a label + a primary flag. The label uses the same
/// vocabulary as Apple Contacts and VCard 3.0/4.0: `home`, `work`,
/// `other`, plus a free-form `custom` field for user-defined labels
/// (Apple's `CNLabeledValue` allows arbitrary labels).
///
/// `isPrimary` is the "preferred" flag. The agent's contact lookup
/// returns the primary email / phone by default; the UI shows a
/// "primary" badge on the labeled row.
public struct LabeledEmail: Codable, Sendable, Hashable {
    public enum Label: Codable, Sendable, Hashable {
        case home
        case work
        case other
        case custom(String)
    }
    public var label: Label
    public var value: String
    public var isPrimary: Bool

    public init(label: Label, value: String, isPrimary: Bool = false) {
        self.label = label
        self.value = value
        self.isPrimary = isPrimary
    }
}

public struct LabeledPhone: Codable, Sendable, Hashable {
    public enum Label: Codable, Sendable, Hashable {
        case mobile
        case work
        case home
        case main
        case fax
        case other
        case custom(String)
    }
    public var label: Label
    public var value: String
    public var isPrimary: Bool

    public init(label: Label, value: String, isPrimary: Bool = false) {
        self.label = label
        self.value = value
        self.isPrimary = isPrimary
    }
}

public struct LabeledAddress: Codable, Sendable, Hashable {
    public enum Label: Codable, Sendable, Hashable {
        case home
        case work
        case billing
        case other
        case custom(String)
    }
    public var label: Label
    public var street: String
    public var city: String?
    public var region: String?
    public var postalCode: String?
    public var country: String?

    public init(
        label: Label,
        street: String,
        city: String? = nil,
        region: String? = nil,
        postalCode: String? = nil,
        country: String? = nil
    ) {
        self.label = label
        self.street = street
        self.city = city
        self.region = region
        self.postalCode = postalCode
        self.country = country
    }

    /// Single-line address for display in the contact list / graph detail.
    public var oneLine: String {
        var parts: [String] = [street]
        if let city, !city.isEmpty { parts.append(city) }
        if let region, !region.isEmpty { parts.append(region) }
        if let postalCode, !postalCode.isEmpty { parts.append(postalCode) }
        if let country, !country.isEmpty { parts.append(country) }
        return parts.joined(separator: ", ")
    }
}

// MARK: - JSON helpers

extension Contact {
    /// Encode the contact to JSON. The encoded form is what the
    /// `body` column of `graph_entities` stores. Uses sorted keys
    /// + ISO-8601 dates so the bytes are deterministic.
    public func jsonData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    /// Decode a contact from JSON. The inverse of ``jsonData()``.
    public static func from(jsonData: Data) throws -> Contact {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(Contact.self, from: jsonData)
    }
}
