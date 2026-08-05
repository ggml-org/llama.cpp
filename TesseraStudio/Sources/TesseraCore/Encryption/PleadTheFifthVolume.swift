import Foundation

/// Volume layer that the wipe executor needs to talk to. The Phase 1
/// `TesseraEncryptedVolume` actor conforms to this; tests use a mock.
///
/// Kept narrow on purpose - the executor only needs three things to run
/// the documented 9-step wipe: a way to ask the OS to unmount, a way to
/// check whether the unmount already happened, and the list of encrypted
/// artifacts on disk that need to be overwritten and unlinked.
public protocol PleadTheFifthVolume: Sendable {
    /// Ask the OS to unmount the volume. After the key has been destroyed
    /// (the crypto-shred event in step 3), this is expected to fail; the
    /// executor records the failure and continues.
    func unmount() async throws

    /// Whether the volume is currently mounted. Used by the executor's
    /// step 5 to decide whether an unmount failure is unexpected.
    func isMounted() async -> Bool

    /// Filesystem paths the executor must overwrite and delete. Typically
    /// the sparsebundle itself and any side files (Info.plist inside
    /// `.sparsebundle/`, `token`, `Lock`, etc.).
    var encryptedArtifacts: [URL] { get async }
}
