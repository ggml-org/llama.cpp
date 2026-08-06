#if os(iOS)
import SwiftUI
import TesseraCore

/// The iOS-side lazy bootstrap for the
/// Email surface. Mirrors the macOS
/// ``EmailSurfaceBootstrap``; the only
/// difference is the type name
/// (``_iOS``) so the two bootstrap
/// helpers don't collide when the app
/// builds for both platforms.
///
/// See ``EmailSurfaceBootstrap`` for the
/// full design rationale.
@MainActor
final class EmailSurfaceBootstrap_iOS: ObservableObject {

    private(set) var dataLayer: TesseraDataLayer
    private(set) var store: EmailStore
    private(set) var sender: EmailSender
    private(set) var importer: EmailImporter
    let identity: EmailAddress

    private var didInstall: Bool = false

    init() {
        let dl = TesseraDataLayer(configuration: .init())
        self.dataLayer = dl
        let store = EmailStore(dataLayer: dl)
        self.store = store
        let coordinator = ShareSheetCoordinator()
        let sender = EmailSender(shareSheetCoordinator: coordinator, store: store)
        self.sender = sender
        let importer = TesseraImporter()
        let emailImporter = EmailImporter(importer: importer, store: store)
        self.importer = emailImporter
        self.identity = EmailAddress(name: "Me", email: "me@local")
    }

    /// Start the data layer on first use. Idempotent.
    func installIfNeeded() {
        guard !didInstall else { return }
        didInstall = true
        Task { [dataLayer] in
            _ = await dataLayer.start()
        }
    }
}
#endif
