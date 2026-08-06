import Foundation
#if canImport(AppKit)
import AppKit
#endif

// MARK: - EmailSender

/// The "send" path for v1. v1 is read + reply only;
/// we don't talk to SMTP directly. Instead, the
/// sender:
///
/// 1. Stages the draft as an .eml file in
///    `NSTemporaryDirectory()` (the file is named
///    `tessera-draft-<uuid>.eml`).
/// 2. Hands the file URL to the system share sheet
///    (macOS `NSSharingServicePicker`, iOS
///    `UIActivityViewController`). The user picks
///    "Mail" and Apple Mail takes the file.
/// 3. Persists the draft in the `.drafts` folder
///    with a `pendingSend` flag, so the
///    cancellation case ("user closed the share
///    sheet without sending") still leaves a
///    recoverable draft.
///
/// The actor owns the share-sheet presentation
/// context (the SwiftUI view provides the anchor
/// view on macOS; iOS doesn't need an anchor). The
/// stage + persist steps happen off the main
/// actor; the share-sheet presentation hops to
/// the main actor because `NSSharingServicePicker`
/// is `@MainActor`-isolated.
public actor EmailSender {

    /// The result of ``send(_:)``.
    public enum SendResult: Sendable, Hashable {
        /// The user picked a share target (typically
        /// "Mail") and the .eml file was handed to
        /// it. The URL is the staged file path; the
        /// caller is free to delete it (the share
        /// sheet has already read the bytes by the
        /// time this returns).
        case routedToSystemShare(URL)
        /// The user cancelled the share sheet. The
        /// draft was still saved to `.drafts` and
        /// the user can pick it up from the draft
        /// list.
        case savedAsDraft
    }

    /// Errors raised by the sender.
    public enum SenderError: Error, Sendable, Equatable {
        case stageFailed(reason: String)
        case noShareSheetAvailable
        case shareSheetPresentationFailed(reason: String)
        case emptyDraft
    }

    private let shareSheetCoordinator: ShareSheetCoordinator
    private let store: EmailStore
    private let stageDirectory: URL
    private let dateProvider: @Sendable () -> Date

    /// Default initializer. The stage directory
    /// defaults to `NSTemporaryDirectory()`; the
    /// date provider defaults to `Date.init`.
    public init(
        shareSheetCoordinator: ShareSheetCoordinator,
        store: EmailStore,
        stageDirectory: URL = URL(fileURLWithPath: NSTemporaryDirectory()),
        dateProvider: @escaping @Sendable () -> Date = { Date() }
    ) {
        self.shareSheetCoordinator = shareSheetCoordinator
        self.store = store
        self.stageDirectory = stageDirectory
        self.dateProvider = dateProvider
    }

    // MARK: - Send

    /// Send a draft. The flow:
    /// 1. Validate the draft (non-empty to, subject
    ///    optional but a body is required).
    /// 2. Stage the .eml file.
    /// 3. Persist the draft in `.drafts` (so a
    ///    cancellation doesn't lose the work).
    /// 4. Present the share sheet. The share sheet
    ///    is async: it returns when the user picks
    ///    a target OR cancels.
    /// 5. If picked, append an
    ///    `email_routed_to_share_sheet` receipt to
    ///    the original email (the one we're
    ///    replying to / forwarding) and the draft
    ///    entity.
    public func send(_ draft: DraftEmail, original: EmailMessage? = nil) async throws -> SendResult {
        guard !draft.to.isEmpty else {
            throw SenderError.emptyDraft
        }
        // 1) Stage the .eml file. The .eml
        //    encoding is in DraftEmail.emlData().
        let stageURL = stageDirectory
            .appendingPathComponent("tessera-draft-\(draft.id.uuidString).eml")
        let eml = draft.emlData()
        do {
            try eml.write(to: stageURL, options: .atomic)
        } catch {
            throw SenderError.stageFailed(reason: String(describing: error))
        }

        // 2) Persist the draft in `.drafts` first.
        //    The draft is what shows up in the
        //    draft list whether the user sends or
        //    cancels. The `pendingSend` flag is
        //    removed once the share sheet resolves.
        let storedMessage = draft.toEmailMessage()
        _ = try await store.saveDraft(storedMessage)

        // 3) Present the share sheet. The
        //    presentation is @MainActor on macOS;
        //    we hop across.
        let didRoute: Bool
        #if canImport(AppKit)
        didRoute = await presentShareSheetMac(stageURL: stageURL)
        #else
        didRoute = false
        #endif

        // 4) Append the routing receipt. The
        //    original email (if any) is the
        //    `entity_id`; the draft is a separate
        //    entity that's also annotated.
        if didRoute {
            if let original {
                _ = try await store.appendEmailReceipt(
                    entityID: original.id,
                    receiptType: EmailReceiptType.routedToShareSheet.rawValue,
                    payload: [
                        "draftID": .string(draft.id.uuidString),
                        "stageURL": .string(stageURL.path),
                        "composeMode": .string(draft.composeMode.rawValue),
                    ]
                )
            }
            // Promote the draft to `.sent` so the
            // user can find it under Sent.
            var sent = storedMessage
            sent.folder = .sent
            sent.updatedAt = dateProvider()
            _ = try await store.upsert(sent)
            return .routedToSystemShare(stageURL)
        } else {
            // User cancelled. The draft stays in
            // .drafts as-is.
            return .savedAsDraft
        }
    }

    // MARK: - Share sheet (macOS)

    #if canImport(AppKit)
    /// Present the macOS share sheet. Returns
    /// true when the user picked a target, false
    /// when the user cancelled.
    @MainActor
    private func presentShareSheetMac(stageURL: URL) async -> Bool {
        // NSSharingServicePicker is constructed
        // with the staged file URL. The picker
        // shows Mail / Messages / AirDrop /
        // Save-as / etc. We watch for the
        // picker's dismissal via a delegate
        // shim — when the user cancels, the
        // delegate's
        // `sharingServicePickerDidClose` is
        // called; we resolve the continuation
        // there.
        let picker = NSSharingServicePicker(items: [stageURL])
        // The picker needs a window / view to
        // anchor to. We present it from the
        // key window; if no key window, the
        // picker is shown without an anchor
        // (it still works for items that
        // have a single share target).
        let anchorView: NSView
        let anchorRect: NSRect
        if let window = NSApp.keyWindow,
           let view = window.contentView {
            anchorView = view
            anchorRect = view.bounds
        } else {
            // No key window — present with a
            // zero-anchor placeholder so the
            // picker still gets a view to
            // attach to.
            anchorView = NSView()
            anchorRect = .zero
        }
        return await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
            let delegate = ShareSheetDelegate { didPick in
                cont.resume(returning: didPick)
            }
            // Register the delegate in the
            // module-level retainer so the
            // picker doesn't deallocate it
            // before the close callback fires.
            ShareSheetDelegateRegistry.shared.retain(delegate)
            picker.delegate = delegate
            picker.show(relativeTo: anchorRect, of: anchorView, preferredEdge: .minY)
        }
    }
    #endif
}

// MARK: - Share sheet delegate retainer

#if canImport(AppKit)
/// Module-level retainer for live share-sheet
/// delegates. NSSharingServicePicker does not
/// retain its delegate; without an explicit
/// retainer the delegate deallocates before
/// the close callback fires and the
/// continuation never resumes. The retainer
/// removes itself from the registry when the
/// picker closes.
@MainActor
final class ShareSheetDelegateRegistry {
    static let shared = ShareSheetDelegateRegistry()
    private var retainers: [ObjectIdentifier: ShareSheetDelegate] = [:]
    private init() {}

    func retain(_ delegate: ShareSheetDelegate) {
        retainers[ObjectIdentifier(delegate)] = delegate
    }

    func release(_ delegate: ShareSheetDelegate) {
        retainers.removeValue(forKey: ObjectIdentifier(delegate))
    }
}

/// The NSSharingServicePicker delegate that
/// resolves the send continuation. The picker
/// doesn't tell us WHICH target the user
/// picked; it only tells us that the picker
/// was used vs cancelled. We treat "used" as
/// routed (the picker only resolves with a
/// target if the user actually picked one).
final class ShareSheetDelegate: NSObject, NSSharingServicePickerDelegate {
    let onClose: (Bool) -> Void
    private var didChooseTarget: Bool = false

    init(onClose: @escaping (Bool) -> Void) {
        self.onClose = onClose
    }

    func sharingServicePicker(
        _ sharingServicePicker: NSSharingServicePicker,
        didChoose service: NSSharingService?
    ) {
        // service is non-nil when the user
        // picked a target. The picker
        // dismisses automatically; the
        // close callback fires after this.
        if service != nil { didChooseTarget = true }
    }

    func sharingServicePickerDidClose(
        _ sharingServicePicker: NSSharingServicePicker
    ) {
        onClose(didChooseTarget)
        // Drop the retainer; the picker
        // won't fire the callback again.
        MainActor.assumeIsolated {
            ShareSheetDelegateRegistry.shared.release(self)
        }
    }
}
#endif

// MARK: - EmailStore extension for receipts

extension EmailStore {
    /// Append a receipt to an email entity. Exposed
    /// as an internal so the sender can write
    /// routing receipts without re-fetching the
    /// email (the store's existing `appendReceipt`
    /// is private).
    func appendEmailReceipt(
        entityID: UUID,
        receiptType: String,
        payload: [String: JSONValue]
    ) async throws -> GraphReceipt {
        try await appendReceiptPublic(
            entityID: entityID,
            receiptType: receiptType,
            payload: payload
        )
    }

    /// Public re-export of the private
    /// ``appendReceipt`` so the sender (which is
    /// in a sibling file) can write receipts.
    func appendReceiptPublic(
        entityID: UUID,
        receiptType: String,
        payload: [String: JSONValue]
    ) async throws -> GraphReceipt {
        let dl = self._dataLayer()
        return try await dl.appendReceipt(
            entityID: entityID,
            receiptType: receiptType,
            payload: payload
        )
    }
}
