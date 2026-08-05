import Foundation
import Security

// MARK: - KeychainStorage

/// Thin wrapper around the macOS Keychain Services API for
/// storing the Slack webhook URL.
///
/// The webhook URL is the only third-party-secret in v1
/// (the spec §11.2: "no per-target API keys" - the Slack
/// webhook URL is the exception because Slack's share-sheet
/// integration is unreliable; we accept a single webhook
/// URL per user).
///
/// We store the URL as a generic password keyed by
/// ``keychainAccount`` and ``keychainService``. The service
/// is the existing ``TesseraSecretStore.service``
/// (``"com.tessera.studio"``) so the entry is wiped by the
/// same crypto-shred event the volume password is. The
/// accessibility class is
/// ``kSecAttrAccessibleWhenUnlockedThisDeviceOnly`` so the
/// value is unavailable after a reboot until the user
/// unlocks the keychain.
///
/// The class is `final` and `Sendable` because the only
/// state is the service / account constants, which are
/// immutable.
public final class KeychainStorage: Sendable {
    public static let shared = KeychainStorage()

    /// The Keychain service identifier. We reuse the
    /// existing ``TesseraSecretStore.service`` so the
    /// PleaTheFifth 9-step wipe (which clears every entry
    /// under that service) takes the webhook URL with it.
    public var service: String = "com.tessera.studio"
    /// The Keychain account identifier (per-target). The
    /// default is the macOS user name + the target kind;
    /// tests can override.
    public var account: String = "slack-webhook.\(NSUserName())"

    public init() {}

    // MARK: - Webhook URL

    /// Persist the Slack webhook URL to the Keychain.
    /// Replaces any existing entry.
    public func setWebhookURL(_ url: URL) throws {
        guard let data = url.absoluteString.data(using: .utf8) else {
            throw KeychainError.encodingFailed
        }
        try? deleteWebhookURL()
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            kSecAttrSynchronizable as String: kCFBooleanFalse! as Any,
            kSecValueData as String: data,
        ]
        let status = SecItemAdd(query as CFDictionary, nil)
        if status != errSecSuccess {
            throw KeychainError.osStatus(status)
        }
    }

    /// Return the Slack webhook URL, or nil if none has been
    /// stored.
    public func getWebhookURL() throws -> URL? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound {
            return nil
        }
        if status != errSecSuccess {
            throw KeychainError.osStatus(status)
        }
        guard let data = result as? Data,
              let str = String(data: data, encoding: .utf8),
              let url = URL(string: str) else {
            return nil
        }
        return url
    }

    /// Delete the Slack webhook URL. No-op if none is stored.
    public func deleteWebhookURL() throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let status = SecItemDelete(query as CFDictionary)
        if status != errSecSuccess && status != errSecItemNotFound {
            throw KeychainError.osStatus(status)
        }
    }
}

public enum KeychainError: Error, LocalizedError {
    case osStatus(OSStatus)
    case encodingFailed
    public var errorDescription: String? {
        switch self {
        case .osStatus(let status):
            if let s = SecCopyErrorMessageString(status, nil) as String? {
                return "Keychain error (\(status)): \(s)"
            }
            return "Keychain error (\(status))"
        case .encodingFailed:
            return "Keychain: failed to encode the value as UTF-8"
        }
    }
}
