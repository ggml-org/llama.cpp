import Foundation
import Security

/// What the UI needs to know about a stored secret, WITHOUT the
/// secret itself: either there is one on file or there isn't.
/// Views switch on this instead of holding a "did the user type
/// a key?" flag that could disagree with the Keychain.
public enum TesseraSecretState: Equatable, Sendable {
    case stored
    case missing
}

/// Keychain-backed storage for API keys. Secrets never live in
/// UserDefaults (a plaintext plist readable by any process with
/// access to the preferences domain); the only other in-memory
/// copies are the transient ones an editing field or an HTTP
/// client needs.
public enum TesseraSecretStore {
    /// Keychain service namespace for all Tessera Studio secrets.
    public static let service = "com.tessera.studio"

    /// Account name for the remote LLM provider API key.
    public static let remoteAPIKeyAccount = "remoteAPIKey"

    /// The stored/missing state for an account, for UI display.
    public static func state(account: String) -> TesseraSecretState {
        secret(account: account) != nil ? .stored : .missing
    }

    /// Read a secret. Returns nil when nothing is stored.
    ///
    /// One-shot migration: builds before the Keychain move kept
    /// the remote API key in UserDefaults. The first read after
    /// upgrade moves that value into the Keychain and deletes
    /// the plaintext copy, so existing setups keep working.
    public static func secret(account: String) -> String? {
        if let value = keychainString(account: account) {
            return value
        }
        if account == remoteAPIKeyAccount {
            return migrateLegacy(
                defaultsKey: TesseraSettingsKey.remoteAPIKey,
                account: account
            )
        }
        return nil
    }

    /// Move a legacy UserDefaults secret into the Keychain and
    /// delete the plaintext copy. Parameterized so tests can run
    /// it against throwaway keys instead of the app's real ones.
    /// Returns the migrated value, or nil when there was nothing
    /// to migrate.
    static func migrateLegacy(
        defaultsKey: String,
        account: String,
        defaults: UserDefaults = .standard
    ) -> String? {
        guard let legacy = defaults.string(forKey: defaultsKey), !legacy.isEmpty else {
            return nil
        }
        if setSecret(legacy, account: account) {
            defaults.removeObject(forKey: defaultsKey)
        }
        return legacy
    }

    /// Store a secret, or delete it when nil / empty. Returns
    /// false when the Keychain refused (missing entitlements in
    /// a bare test process, locked keychain, ...); callers treat
    /// that as "not stored" rather than crashing.
    @discardableResult
    public static func setSecret(_ value: String?, account: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        guard let value, !value.isEmpty else {
            let status = SecItemDelete(query as CFDictionary)
            return status == errSecSuccess || status == errSecItemNotFound
        }
        let data = Data(value.utf8)
        let updateStatus = SecItemUpdate(
            query as CFDictionary,
            [kSecValueData as String: data] as CFDictionary
        )
        if updateStatus == errSecSuccess {
            return true
        }
        guard updateStatus == errSecItemNotFound else {
            return false
        }
        var addQuery = query
        addQuery[kSecValueData as String] = data
        return SecItemAdd(addQuery as CFDictionary, nil) == errSecSuccess
    }

    private static func keychainString(account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess, let data = item as? Data else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }
}
