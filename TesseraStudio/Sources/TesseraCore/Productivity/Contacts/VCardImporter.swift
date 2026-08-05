import Foundation
#if canImport(Contacts)
import Contacts
#endif

// MARK: - VCardImporter

/// Parse + serialize VCard data into the app's `Contact` model.
///
/// Backed by Apple's `CNContactVCardSerialization` on platforms
/// that ship the `Contacts` framework (macOS, iOS). The
/// `Contacts` framework is available on Linux via the
/// `swift-corelibs-foundation` shim but `CNContactVCardSerialization`
/// is Apple-only; the importer therefore degrades to a no-op
/// parser on Linux and reports a typed error on the first
/// attempt. Tests on Linux skip the framework-dependent paths
/// (see ``VCardImporterTests``).
///
/// The importer is an `actor` because the VCard parse can be
/// expensive on large address books (a few thousand contacts).
/// Multiple imports in flight serialize through the actor.
public actor VCardImporter {

    public init() {}

    // MARK: - Parse

    /// Parse VCard data into a list of `Contact` values. The
    /// data may contain one VCard or many (VCard 3.0 supports
    /// `BEGIN:VCARD` ... `END:VCARD` blocks separated by a blank
    /// line; VCard 4.0 is the same shape). Empty data returns
    /// an empty array.
    public func parse(data: Data) throws -> [Contact] {
        guard !data.isEmpty else { return [] }
        #if canImport(Contacts) && (os(macOS) || os(iOS))
        do {
            let cnContacts = try CNContactVCardSerialization.contacts(with: data)
            return cnContacts.map { Self.contact(from: $0) }
        } catch {
            throw VCardError.parseFailed(underlying: String(describing: error))
        }
        #else
        throw VCardError.frameworkUnavailable
        #endif
    }

    /// Convenience: read the file at `fileURL` and parse it.
    public func parse(fileURL: URL) throws -> [Contact] {
        let data = try Data(contentsOf: fileURL)
        let contacts = try parse(data: data)
        // Stamp the source URL on each contact so the receipt
        // chain can trace the import back to the file.
        let urlString = fileURL.absoluteString
        return contacts.map { c in
            var stamped = c
            stamped.sourceURL = urlString
            return stamped
        }
    }

    // MARK: - Serialize

    /// Serialize a list of contacts to VCard 3.0 data. The
    /// `CNContactVCardSerialization.data(with:)` API produces
    /// the standard 3.0 format which is widely supported
    /// (macOS Contacts, Google Contacts import, Fastmail, ...).
    public func serialize(contacts: [Contact]) throws -> Data {
        #if canImport(Contacts) && (os(macOS) || os(iOS))
        let cnContacts = contacts.map { Self.cnContact(from: $0) }
        do {
            return try CNContactVCardSerialization.data(with: cnContacts)
        } catch {
            throw VCardError.serializeFailed(underlying: String(describing: error))
        }
        #else
        throw VCardError.frameworkUnavailable
        #endif
    }

    /// Convenience: serialize and write to `fileURL`. Creates
    /// intermediate directories if needed.
    public func write(contacts: [Contact], to fileURL: URL) throws {
        let data = try serialize(contacts: contacts)
        let directory = fileURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        try data.write(to: fileURL, options: .atomic)
    }

    // MARK: - CN bridge

    #if canImport(Contacts) && (os(macOS) || os(iOS))

    /// Convert a `CNContact` (the Apple Contacts framework model)
    /// into our app's `Contact`. Field selection mirrors the
    /// VCard 3.0 surface; the `CNContactStore` provides more
    /// keys (social profiles, related names, ...) but those are
    /// v2 work and would only be exercised by a UI that surfaces
    /// them.
    static func contact(from cn: CNContact) -> Contact {
        let name = NameComponents(
            prefix: cn.namePrefix,
            first: cn.givenName,
            middle: cn.middleName,
            last: cn.familyName,
            suffix: cn.nameSuffix,
            nickname: cn.nickname
        )
        let emails: [LabeledEmail] = cn.emailAddresses.map { labeled in
            LabeledEmail(
                label: map(emailLabel: labeled.label),
                value: String(labeled.value),
                isPrimary: false
            )
        }
        let phones: [LabeledPhone] = cn.phoneNumbers.map { labeled in
            LabeledPhone(
                label: map(phoneLabel: labeled.label),
                value: String(labeled.value.stringValue),
                isPrimary: false
            )
        }
        let addresses: [LabeledAddress] = cn.postalAddresses.map { labeled in
            let v = labeled.value
            return LabeledAddress(
                label: map(addressLabel: labeled.label),
                street: v.street,
                city: v.city,
                region: v.state,
                postalCode: v.postalCode,
                country: v.country
            )
        }
        let organization: String? = cn.organizationName.isEmpty ? nil : cn.organizationName
        let title: String? = cn.jobTitle.isEmpty ? nil : cn.jobTitle
        let photo: Data? = cn.imageDataAvailable ? cn.imageData : nil
        let birthday: Date? = cn.birthday?.date
        return Contact(
            subtype: .person,
            name: name,
            emails: emails,
            phones: phones,
            addresses: addresses,
            organization: organization,
            title: title,
            birthday: birthday,
            photo: photo,
            notes: nil,
            sourceURL: nil,
            linkedEntityIDs: []
        )
    }

    /// Convert our `Contact` back into a `CNContact` for
    /// serialization. We build a `CNMutableContact` and stamp
    /// each field, then return the immutable form.
    static func cnContact(from contact: Contact) -> CNContact {
        let m = CNMutableContact()
        m.namePrefix = contact.name.prefix ?? ""
        m.givenName = contact.name.first ?? ""
        m.middleName = contact.name.middle ?? ""
        m.familyName = contact.name.last ?? ""
        m.nameSuffix = contact.name.suffix ?? ""
        m.nickname = contact.name.nickname ?? ""
        m.organizationName = contact.organization ?? (contact.subtype == .organization ? (contact.name.last ?? "") : "")
        m.jobTitle = contact.title ?? ""
        m.emailAddresses = contact.emails.map { e in
            CNLabeledValue(
                label: emailLabelString(e.label),
                value: e.value as NSString
            )
        }
        m.phoneNumbers = contact.phones.map { p in
            let phone = CNPhoneNumber(stringValue: p.value)
            return CNLabeledValue(
                label: phoneLabelString(p.label),
                value: phone
            )
        }
        m.postalAddresses = contact.addresses.map { a in
            let addr = CNMutablePostalAddress()
            addr.street = a.street
            addr.city = a.city ?? ""
            addr.state = a.region ?? ""
            addr.postalCode = a.postalCode ?? ""
            addr.country = a.country ?? ""
            return CNLabeledValue(
                label: addressLabelString(a.label),
                value: addr as CNPostalAddress
            )
        }
        if let birthday = contact.birthday {
            m.birthday = Calendar(identifier: .gregorian).dateComponents(
                [.year, .month, .day],
                from: birthday
            )
        }
        if let photo = contact.photo {
            m.imageData = photo
        }
        return m
    }

    /// Map Apple's `CNLabeledValue` email label (an `NSString?`
    /// that's either a constant like `CNLabelHome` or a custom
    /// user label) to our `LabeledEmail.Label`.
    private static func map(emailLabel: String?) -> LabeledEmail.Label {
        guard let raw = emailLabel, !raw.isEmpty else { return .other }
        switch raw {
        case CNLabelHome: return .home
        case CNLabelWork: return .work
        default: return .custom(raw)
        }
    }

    private static func map(phoneLabel: String?) -> LabeledPhone.Label {
        guard let raw = phoneLabel, !raw.isEmpty else { return .other }
        switch raw {
        case CNLabelPhoneNumberMobile: return .mobile
        case CNLabelWork: return .work
        case CNLabelHome: return .home
        case CNLabelPhoneNumberMain: return .main
        case CNLabelPhoneNumberPager: return .fax
        default: return .custom(raw)
        }
    }

    private static func map(addressLabel: String?) -> LabeledAddress.Label {
        guard let raw = addressLabel, !raw.isEmpty else { return .other }
        switch raw {
        case CNLabelHome: return .home
        case CNLabelWork: return .work
        default: return .custom(raw)
        }
    }

    private static func emailLabelString(_ label: LabeledEmail.Label) -> String {
        switch label {
        case .home: return CNLabelHome
        case .work: return CNLabelWork
        case .other: return CNLabelOther
        case .custom(let s): return s
        }
    }

    private static func phoneLabelString(_ label: LabeledPhone.Label) -> String {
        switch label {
        case .mobile: return CNLabelPhoneNumberMobile
        case .work: return CNLabelWork
        case .home: return CNLabelHome
        case .main: return CNLabelPhoneNumberMain
        case .fax: return CNLabelPhoneNumberPager
        case .other: return CNLabelOther
        case .custom(let s): return s
        }
    }

    private static func addressLabelString(_ label: LabeledAddress.Label) -> String {
        switch label {
        case .home: return CNLabelHome
        case .work: return CNLabelWork
        case .billing: return CNLabelWork
        case .other: return CNLabelOther
        case .custom(let s): return s
        }
    }
    #endif
}

// MARK: - Errors

public enum VCardError: Error, Sendable, Equatable {
    case parseFailed(underlying: String)
    case serializeFailed(underlying: String)
    /// The platform doesn't ship the Apple Contacts framework
    /// (Linux). The importer is a no-op in that environment.
    case frameworkUnavailable
    case fileNotFound(URL)
}
