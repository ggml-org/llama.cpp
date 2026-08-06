import SwiftUI
import TesseraCore

/// The lazy bootstrap for the Email surface's
/// dependencies. The Email surface needs a
/// ``TesseraDataLayer`` + an ``EmailStore`` + an
/// ``EmailSender`` + an ``EmailImporter``; the
/// bootstrap holds them as a single
/// `@MainActor` `ObservableObject` so the
/// SwiftUI view can observe identity changes
/// without re-creating the store on every
/// render.
///
/// The bootstrap is intentionally cheap: it
/// holds references but doesn't open any
/// connections. The data layer is a Swift
/// actor; calling `start()` is the step that
/// connects to Postgres. The bootstrap calls
/// `start()` on first use and the view shows
/// the data as it arrives (the EmailStore's
/// `list(limit:)` returns whatever the data
/// layer has at that moment; the view reloads
/// when the data layer signals a change in a
/// follow-up).
///
/// **Why lazy.** ContentView has many
/// destinations. Creating the data layer +
/// importer + sender at app launch would
/// block on Postgres even for users who never
/// open the Email surface. The lazy
/// `installIfNeeded()` pattern defers the work
/// to first use.
@MainActor
final class EmailSurfaceBootstrap: ObservableObject {

    private(set) var dataLayer: TesseraDataLayer
    private(set) var store: EmailStore
    private(set) var sender: EmailSender
    private(set) var importer: EmailImporter
    let identity: EmailAddress

    private var didInstall: Bool = false

    init() {
        // The data layer is created with a
        // default configuration. The real
        // config (Postgres host/port, Valkey
        // namespace) is read from
        // TesseraSettings in the production
        // app; the dev preview uses the
        // defaults which point at localhost.
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
        // The "from" address is a placeholder
        // for the dev preview. The production
        // app reads the user's primary email
        // from the Contacts surface (Phase 6)
        // or from the system Mail settings.
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
