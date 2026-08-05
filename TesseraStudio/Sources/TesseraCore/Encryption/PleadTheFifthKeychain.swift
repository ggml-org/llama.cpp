import Foundation
import Security

/// The volume password (and the optional DAK / WK) live in the
/// macOS Keychain. Destroying the entry is the crypto-shred event -
/// once `SecItemDelete` returns success, the encrypted data on disk
/// is cryptographically unrecoverable, regardless of what happens
/// to the overwrite / delete steps that follow.
///
/// This enum is the single seam for key destruction so callers do
/// not have to know the Keychain query layout.
public enum PleadTheFifthKeychain {
    /// Keychain service name (the "kind" of credential). The full
    /// query combines this with an account name (volume password,
    /// DAK, WK).
    public static let service = "com.tessera.studio.encryption"

    public static let volumePasswordAccount = "volume-password"
    public static let dataAccessKeyAccount = "data-access-key"
    public static let wrappingKeyAccount = "wrapping-key"

    /// Delete a single Keychain entry. Returns `true` if the entry
    /// was deleted or was already missing (`errSecItemNotFound`),
    /// `false` on any other failure.
    ///
    /// Idempotent on purpose: re-running the wipe (e.g. the user
    /// double-tapped the hot-key) must not stall on a "not found"
    /// from a prior run. Per the design's fail-closed note, a
    /// repeated wipe is a no-op at the Keychain level.
    @discardableResult
    public static func deleteEntry(account: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let status = SecItemDelete(query as CFDictionary)
        switch status {
        case errSecSuccess, errSecItemNotFound:
            return true
        default:
            return false
        }
    }
}
