import Foundation
#if canImport(Contacts)
import Contacts
#endif

// MARK: - AppleContactsAdapter

/// Read contacts from the system Address Book on macOS via
/// `CNContactStore`. The adapter is the one place the app
/// touches the `Contacts` framework for read access; the rest
/// of the productivity surface consumes `Contact` values and
/// has no dependency on `CNContactStore`.
///
/// **Entitlement.** Production builds need
/// `com.apple.developer.contacts` to read the user's address
/// book. The dev-preview build runs without the entitlement
/// by falling back to the VCard path (Apple exports the
/// entire address book to a `.vcf` file via
/// `NSWorkspace.shared.open(_:)` and the user drops it onto
/// the import panel). The adapter's `init` does NOT request
/// access — that's deferred to ``requestAccess()`` so the
/// constructor is cheap and side-effect free.
///
/// **Observation.** ``startObservingChanges()`` returns an
/// `AsyncStream<ContactChange>` that emits every CNContactStore
/// change. The macOS Contacts framework delivers change
/// notifications via `CNContactStoreDidChangeNotification`;
/// we adapt that to an async stream so the contact view can
/// react without holding a notification observer token
/// across actor boundaries.
public actor AppleContactsAdapter {

    public struct ContactChange: Sendable, Equatable {
        public let kind: Kind
        public let contactID: UUID?

        public enum Kind: String, Sendable, Equatable {
            case inserted
            case updated
            case deleted
        }

        public init(kind: Kind, contactID: UUID? = nil) {
            self.kind = kind
            self.contactID = contactID
        }
    }

    public enum AccessStatus: String, Sendable, Equatable {
        case authorized
        case denied
        case restricted
        case notDetermined
    }

    #if canImport(Contacts) && (os(macOS) || os(iOS))
    private let store: CNContactStore
    #endif

    /// Construct the adapter. Does NOT request access (the
    /// entitlement is checked at request time).
    public init() throws {
        #if canImport(Contacts) && (os(macOS) || os(iOS))
        self.store = CNContactStore()
        #endif
    }

    // MARK: - Permission

    /// Request read access to the user's contacts. Returns true
    /// when the user grants access; false on deny / restriction.
    /// On Linux / non-Apple platforms this returns false because
    /// the `Contacts` framework isn't available.
    public func requestAccess() async throws -> Bool {
        #if canImport(Contacts) && (os(macOS) || os(iOS))
        return try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Bool, Error>) in
            self.store.requestAccess(for: .contacts) { granted, error in
                if let error = error {
                    cont.resume(throwing: error)
                } else {
                    cont.resume(returning: granted)
                }
            }
        }
        #else
        return false
        #endif
    }

    /// The current access status. Useful for the UI to decide
    /// whether to show a "permission needed" banner.
    public var accessStatus: AccessStatus {
        get async {
            #if canImport(Contacts) && (os(macOS) || os(iOS))
            switch CNContactStore.authorizationStatus(for: .contacts) {
            case .authorized: return .authorized
            case .denied: return .denied
            case .restricted: return .restricted
            case .notDetermined: return .notDetermined
            @unknown default: return .notDetermined
            }
            #else
            return .denied
            #endif
        }
    }

    // MARK: - Fetch

    /// Fetch every contact in the address book. On a large
    /// address book this can be slow (10k+ contacts); the
    /// caller should expect to dispatch this to a background
    /// task. The contact keys requested are the standard set
    /// (name, organization, emails, phones, addresses,
    /// birthday, image) — no social profiles / related names
    /// (v2 work).
    public func fetchAllContacts() async throws -> [Contact] {
        #if canImport(Contacts) && (os(macOS) || os(iOS))
        let keys: [CNKeyDescriptor] = [
            CNContactNamePrefixKey as CNKeyDescriptor,
            CNContactGivenNameKey as CNKeyDescriptor,
            CNContactMiddleNameKey as CNKeyDescriptor,
            CNContactFamilyNameKey as CNKeyDescriptor,
            CNContactNameSuffixKey as CNKeyDescriptor,
            CNContactNicknameKey as CNKeyDescriptor,
            CNContactOrganizationNameKey as CNKeyDescriptor,
            CNContactJobTitleKey as CNKeyDescriptor,
            CNContactEmailAddressesKey as CNKeyDescriptor,
            CNContactPhoneNumbersKey as CNKeyDescriptor,
            CNContactPostalAddressesKey as CNKeyDescriptor,
            CNContactBirthdayKey as CNKeyDescriptor,
            CNContactImageDataKey as CNKeyDescriptor,
            CNContactImageDataAvailableKey as CNKeyDescriptor,
        ]
        let request = CNContactFetchRequest(keysToFetch: keys)
        request.sortOrder = .userDefault
        var out: [Contact] = []
        try self.store.enumerateContacts(with: request) { cn, _ in
            out.append(VCardImporter.contact(from: cn))
        }
        return out
        #else
        return []
        #endif
    }

    /// Fetch one contact by the Apple Contacts identifier.
    /// The identifier is the string CNContactStore hands out
    /// via `CNContact.identifier`; we don't use it as our
    /// primary id because identifiers can change across
    /// address book re-imports.
    public func fetchContact(identifier: String) async throws -> Contact? {
        #if canImport(Contacts) && (os(macOS) || os(iOS))
        let keys: [CNKeyDescriptor] = [
            CNContactNamePrefixKey as CNKeyDescriptor,
            CNContactGivenNameKey as CNKeyDescriptor,
            CNContactMiddleNameKey as CNKeyDescriptor,
            CNContactFamilyNameKey as CNKeyDescriptor,
            CNContactNameSuffixKey as CNKeyDescriptor,
            CNContactNicknameKey as CNKeyDescriptor,
            CNContactOrganizationNameKey as CNKeyDescriptor,
            CNContactJobTitleKey as CNKeyDescriptor,
            CNContactEmailAddressesKey as CNKeyDescriptor,
            CNContactPhoneNumbersKey as CNKeyDescriptor,
            CNContactPostalAddressesKey as CNKeyDescriptor,
            CNContactBirthdayKey as CNKeyDescriptor,
            CNContactImageDataKey as CNKeyDescriptor,
            CNContactImageDataAvailableKey as CNKeyDescriptor,
        ]
        let predicate = CNContact.predicateForContacts(withIdentifiers: [identifier])
        do {
            let matches = try self.store.unifiedContacts(matching: predicate, keysToFetch: keys)
            if let first = matches.first {
                return VCardImporter.contact(from: first)
            }
            return nil
        } catch {
            throw AppleContactsError.fetchFailed(underlying: String(describing: error))
        }
        #else
        return nil
        #endif
    }

    // MARK: - Observation

    /// Begin observing address-book changes. The returned
    /// `AsyncStream` emits one `ContactChange` per change
    /// notification. The stream completes when the consumer
    /// cancels the consuming task.
    ///
    /// The implementation is best-effort: on Linux / non-Apple
    /// platforms the stream emits no events and stays open
    /// until cancelled.
    public func startObservingChanges() -> AsyncStream<ContactChange> {
        AsyncStream { continuation in
            #if canImport(Contacts) && (os(macOS) || os(iOS))
            let observer = NotificationCenter.default.addObserver(
                forName: .CNContactStoreDidChange,
                object: nil,
                queue: nil
            ) { _ in
                continuation.yield(ContactChange(kind: .updated, contactID: nil))
            }
            continuation.onTermination = { _ in
                NotificationCenter.default.removeObserver(observer)
            }
            #else
            continuation.onTermination = { _ in }
            #endif
        }
    }
}

// MARK: - Errors

public enum AppleContactsError: Error, Sendable, Equatable {
    case fetchFailed(underlying: String)
    case permissionDenied
    case frameworkUnavailable
}
