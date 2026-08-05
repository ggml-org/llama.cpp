#if canImport(AppKit)
import AppKit
import Foundation

/// App-wide notifications for the "Plea the Fifth" feature. Centralised
/// so the settings view (which posts) and the menu item (which listens)
/// agree on the contract without a direct dependency.
extension Notification.Name {
    /// Posted by the macOS Settings view when the user clicks
    /// "View last wipe report...". The PleadTheFifthMenuItem (which
    /// owns the actual report window) observes this and presents it.
    static let openLastWipeReport = Notification.Name("com.tessera.studio.openLastWipeReport")
}
#endif
