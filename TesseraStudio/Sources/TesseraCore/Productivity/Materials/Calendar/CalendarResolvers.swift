import Foundation
#if canImport(CoreLocation)
import CoreLocation
#endif

// MARK: - GeocodingLocationResolver

/// A ``LocationResolver`` backed by `CLGeocoder`. Geocoding
/// is async and touches the system geocoder (on-device, no
/// API key), so the resolver separates the two paths:
///
///  * ``geocode(_:)`` — async; resolves a free-form place
///    name to a coordinate and fills the cache. The view
///    model calls this after a quick-add lands, and again
///    when an event detail opens with an ungeocoded
///    location.
///  * ``coordinate(for:)`` — sync; reads the cache. This is
///    what the NLU parser consults during parsing so the
///    parse path never blocks.
///
/// The cache is memory-only (v1): geocodes are cheap and
/// the privacy posture is "don't persist more location
/// history than the events themselves carry".
public final class GeocodingLocationResolver: LocationResolver, @unchecked Sendable {

    private let lock = NSLock()
    private var cache: [String: CalendarEvent.Coordinate] = [:]

    public init() {}

    /// Sync cache read (``LocationResolver``).
    public func coordinate(for location: String) -> CalendarEvent.Coordinate? {
        lock.lock()
        defer { lock.unlock() }
        return cache[Self.normalize(location)]
    }

    /// Seed the cache directly (the event editor uses this
    /// when the user picks a search result).
    public func prime(_ location: String, with coordinate: CalendarEvent.Coordinate) {
        lock.lock()
        defer { lock.unlock() }
        cache[Self.normalize(location)] = coordinate
    }

    /// Resolve a place name via CLGeocoder and cache the
    /// result. Returns nil when geocoding fails or the
    /// platform has no CoreLocation.
    @discardableResult
    public func geocode(_ location: String) async -> CalendarEvent.Coordinate? {
        #if canImport(CoreLocation)
        let geocoder = CLGeocoder()
        do {
            let placemarks = try await geocoder.geocodeAddressString(location)
            if let loc = placemarks.first?.location {
                let coordinate = CalendarEvent.Coordinate(
                    latitude: loc.coordinate.latitude,
                    longitude: loc.coordinate.longitude
                )
                prime(location, with: coordinate)
                return coordinate
            }
        } catch {
            // Geocoding is best-effort: an ungeocoded
            // location keeps its free-form string, which is
            // still shown verbatim in the event detail.
        }
        return nil
        #else
        return nil
        #endif
    }

    private static func normalize(_ s: String) -> String {
        s.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - ContactSnapshotAdapter

/// A ``ContactsAdapter`` snapshot of the ``ContactStore``.
/// The view model loads it once (and refreshes it when the
/// contact surface changes) so the NLU parser's lookups
/// stay synchronous.
public struct ContactSnapshotAdapter: ContactsAdapter {
    public var contacts: [Contact]

    public init(contacts: [Contact]) {
        self.contacts = contacts
    }

    /// Load every contact from the store into a snapshot.
    public static func load(from store: ContactStore) async throws -> ContactSnapshotAdapter {
        ContactSnapshotAdapter(contacts: try await store.list())
    }

    public func contacts(matching name: String) -> [Contact] {
        StaticContactsAdapter(contacts: contacts).contacts(matching: name)
    }
}

// MARK: - DocumentSnapshotAdapter

/// A ``DocumentResolver`` snapshot of the document entity
/// labels. Built from the data layer's
/// `listByEntityType("document")` — the label column is the
/// document title, so no AST load is needed.
public struct DocumentSnapshotAdapter: DocumentResolver {
    public var documents: [ResolvedDocument]

    public init(documents: [ResolvedDocument]) {
        self.documents = documents
    }

    /// Load every document's (id, title) from the data
    /// layer. Entities without a label are skipped.
    public static func load(from dataLayer: TesseraDataLayer, limit: Int = 1000) async throws -> DocumentSnapshotAdapter {
        let rows = try await dataLayer.listByEntityType(entityType: "document", limit: limit)
        let docs = rows
            .filter { !$0.label.isEmpty }
            .map { ResolvedDocument(id: $0.id, title: $0.label) }
        return DocumentSnapshotAdapter(documents: docs)
    }

    public func documents(matching title: String) -> [ResolvedDocument] {
        StaticDocumentResolver(documents: documents).documents(matching: title)
    }
}
