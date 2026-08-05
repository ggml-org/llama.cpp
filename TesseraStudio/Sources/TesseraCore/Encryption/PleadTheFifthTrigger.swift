import Foundation

/// The surface that fired the wipe. Recorded in the wipe report so
/// the user (auditing their own action) can tell whether a wipe
/// came from the hot-key, the menu bar, or the covert trigger.
///
/// Defined here in Phase 3 (the trigger surface) so the protocol
/// and the test mock can both reference it without depending on
/// Phase 2's `PleadTheFifthExecutor` concrete type. Phase 2's
/// executor exposes a `TriggerSource` with the same cases; the
/// composition root maps between the two on the call site.
public enum PleadTheFifthTriggerSource: String, Sendable, Equatable {
    /// `Cmd+Shift+Backspace`. No confirmation.
    case hotkey
    /// Menu bar "Plead the Fifth..." with the typed phrase.
    case menu
    /// The covert trigger phrase appeared in a text input.
    case covert
}

/// Contract a wipe executor exposes to its trigger surfaces.
///
/// Phase 3 (covert trigger) needs to call into the wipe executor
/// without depending on its concrete type. Phase 2 builds
/// `PleadTheFifthExecutor`; until it lands, the trigger surface
/// only needs the protocol. The composition root in
/// `TesseraStudioMac` wires the real executor in once both
/// branches merge.
///
/// The real executor's `destroyAll(trigger:)` conforms to
/// `fire(trigger:)`. The test mock `RecordingPleadTheFifthTrigger`
/// also conforms so the trigger logic is unit-testable in
/// isolation.
public protocol PleadTheFifthTrigger: Sendable {
    /// Begin the wipe. Returns when the wipe has been initiated
    /// (not necessarily completed - the covert trigger must not
    /// block the UI thread waiting for the wipe).
    func fire(trigger: PleadTheFifthTriggerSource) async throws
}
