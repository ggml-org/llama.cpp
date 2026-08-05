import Foundation

// MARK: - ActiveDocumentInfo

/// Information about a document whose chat queue is
/// currently active in the app. The `CrossDocumentChatRegistry`
/// emits a list of these; the chat panel uses the list to
/// render the "Working in background" chip (per spec §6.9).
public struct ActiveDocumentInfo: Sendable, Hashable, Identifiable {
    public let documentID: UUID
    public let title: String
    public let inFlightItemCount: Int
    public let isCurrent: Bool
    public var id: UUID { documentID }

    public init(
        documentID: UUID,
        title: String,
        inFlightItemCount: Int = 0,
        isCurrent: Bool = false
    ) {
        self.documentID = documentID
        self.title = title
        self.inFlightItemCount = inFlightItemCount
        self.isCurrent = isCurrent
    }
}

// MARK: - CrossDocumentChatRegistry

/// Tracks the set of active chat queues across all open
/// documents (per spec §6.9). The registry exposes:
///
/// - `register(_:for:title:)`: called by the host view
///   when a document's `ChatPanelStateMachine` is created.
/// - `unregister(documentID:)`: called when the document
///   closes.
/// - `setCurrent(documentID:)`: called when the user
///   switches documents.
/// - `activeDocuments()`: returns the list of registered
///   documents with their in-flight item counts.
/// - `pauseAll()`: calls `forceHold()` on every registered
///   state machine, pausing every agent run across every
///   document.
///
/// The registry is the seam between the per-document state
/// machines and the cross-document UI affordances (the
/// "Working in background" chip in the chat panel of the
/// current document, and the "Pause all" button on the
/// chip).
public actor CrossDocumentChatRegistry {

    /// A registered state machine. The registry holds a
    /// weak reference is not possible across actor boundaries;
    /// the registry holds a strong reference and the
    /// `unregister` path is the only way to drop it. The
    /// host view is expected to call `unregister` on
    /// document close.
    private struct Registration {
        let machine: ChatPanelStateMachine
        let title: String
        var inFlightCount: Int
    }

    private var registrations: [UUID: Registration] = [:]
    private var currentDocumentID: UUID?

    public init() {}

    // MARK: - Register / unregister

    /// Register a state machine. A document can only have
    /// one registered machine at a time; re-registering
    /// replaces the old registration.
    public func register(
        _ machine: ChatPanelStateMachine,
        for documentID: UUID,
        title: String
    ) {
        registrations[documentID] = Registration(
            machine: machine,
            title: title,
            inFlightCount: 0
        )
    }

    /// Unregister a state machine. Idempotent: a second
    /// call is a no-op.
    public func unregister(documentID: UUID) {
        registrations.removeValue(forKey: documentID)
        if currentDocumentID == documentID {
            currentDocumentID = nil
        }
    }

    /// Set the current document id. The registry's
    /// `activeDocuments()` will mark this one as
    /// `isCurrent: true`.
    public func setCurrent(documentID: UUID?) {
        currentDocumentID = documentID
    }

    // MARK: - Read

    /// The list of registered documents, in registration
    /// order, with their in-flight item counts. The current
    /// document is marked `isCurrent: true`.
    public func activeDocuments() -> [ActiveDocumentInfo] {
        registrations.map { (id, reg) in
            ActiveDocumentInfo(
                documentID: id,
                title: reg.title,
                inFlightItemCount: reg.inFlightCount,
                isCurrent: id == currentDocumentID
            )
        }.sorted { lhs, rhs in
            // Current first, then by title for stability.
            if lhs.isCurrent != rhs.isCurrent { return lhs.isCurrent && !rhs.isCurrent }
            return lhs.title < rhs.title
        }
    }

    /// The number of registered documents.
    public var registrationCount: Int { registrations.count }

    /// True iff the registry has no registrations.
    public var isEmpty: Bool { registrations.isEmpty }

    // MARK: - In-flight tracking

    /// Update the in-flight count for a document. The host
    /// view typically calls this from the state machine's
    /// change observer; the chat panel header shows the
    /// count, the registry's "Working in background" chip
    /// shows the cross-document version.
    public func setInFlightCount(_ count: Int, for documentID: UUID) {
        guard var reg = registrations[documentID] else { return }
        reg.inFlightCount = max(0, count)
        registrations[documentID] = reg
    }

    // MARK: - Pause all

    /// Pause every registered state machine. The
    /// "Pause all" button on the "Working in background"
    /// chip calls this. The receipt chain serializes the
    /// pause requests (the data layer's per-document
    /// `chat_queues` table is updated for each document
    /// in turn).
    public func pauseAll() async {
        for (_, reg) in registrations {
            try? await reg.machine.forceHold()
        }
    }

    // MARK: - Current lookup

    /// The currently-active document id. nil when no
    /// document is current.
    public var currentDocument: UUID? { currentDocumentID }

    /// Look up the title of a document by id. nil when the
    /// document isn't registered.
    public func title(for documentID: UUID) -> String? {
        registrations[documentID]?.title
    }
}
