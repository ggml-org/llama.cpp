import Foundation

/// One deterministic scrubber rule (usage dataset spec section 8, phase 1:
/// versioned, reviewable pattern rules for secrets, keys, credentials,
/// paths, addresses, phone numbers, account identifiers).
public struct TesseraScrubRule: Sendable {
    /// Stable id reported by the probe and the quarantine list (never the
    /// matched content itself).
    public let id: String
    public let pattern: String
    /// Replacement template for the scrub wall; the read-only probe ignores it.
    public let replacement: String

    public init(id: String, pattern: String, replacement: String) {
        self.id = id
        self.pattern = pattern
        self.replacement = replacement
    }
}

/// The shared rule set behind BOTH the anonymization wall
/// (`TesseraCurating.scrub`) and the curation stage's read-only sensitivity
/// probe (runtime-traces spec section 12.3). One list, so what the wall
/// catches and what curation flags can never drift apart.
public enum TesseraScrubRules {
    /// Rule set version. Stamped into curation ledger entries as
    /// `anonymizer_required_version`; bump when rules change so quarantined
    /// sessions can be re-analyzed under the newer set.
    public static let version = 1

    /// The ledger field format for the current rule set.
    public static var requiredVersionStamp: String { ">=\(version)" }

    public static let all: [TesseraScrubRule] = [
        // PEM private-key blocks (span lines).
        TesseraScrubRule(
            id: "pem-key",
            pattern: "-----BEGIN [A-Z ]*PRIVATE KEY-----[\\s\\S]*?-----END [A-Z ]*PRIVATE KEY-----",
            replacement: "[REDACTED PRIVATE KEY]"),
        // Bearer tokens.
        TesseraScrubRule(
            id: "bearer-token",
            pattern: "(?i)Bearer\\s+[A-Za-z0-9._\\-]+",
            replacement: "Bearer [REDACTED]"),
        // OpenAI-style secret keys.
        TesseraScrubRule(
            id: "secret-key",
            pattern: "\\bsk-[A-Za-z0-9_\\-]{8,}",
            replacement: "sk-[REDACTED]"),
        // KEY=VALUE lines whose name looks sensitive.
        TesseraScrubRule(
            id: "credential-assignment",
            pattern: "(?im)^(\\s*(?:export\\s+)?[A-Za-z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD)[A-Za-z0-9_]*\\s*[:=]\\s*).*$",
            replacement: "$1[REDACTED]"),
        // Email addresses (contact info).
        TesseraScrubRule(
            id: "email",
            pattern: "\\b[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}\\b",
            replacement: "[REDACTED EMAIL]"),
        // Phone numbers, NANP and international shapes (contact info).
        // Conservative on purpose: a false positive quarantines (reviewable,
        // local-only), a false negative leaks.
        TesseraScrubRule(
            id: "phone",
            pattern: "(?<![0-9])(?:\\+?[0-9]{1,3}[ \\-]?)?(?:\\(?[0-9]{3}\\)?[ \\-]?)[0-9]{3}[ \\-]?[0-9]{4}(?![0-9])",
            replacement: "[REDACTED PHONE]"),
        // Absolute filesystem paths (user-home shapes on macOS/Linux/Windows).
        TesseraScrubRule(
            id: "fs-path",
            pattern: "(?:/Users/|/home/|C:\\\\Users\\\\)[^\\s\"'`,)]+",
            replacement: "[REDACTED PATH]"),
    ]

    /// The scrub wall: apply every rule's replacement, in order.
    public static func scrub(_ text: String) -> String {
        var out = text
        for rule in all {
            out = replace(out, pattern: rule.pattern, with: rule.replacement)
        }
        return out
    }

    /// Read-only probe: ids of the rules with at least one match, in rule
    /// order. This is the curation stage's quarantine signal; it never
    /// reports the matched content itself.
    public static func probe(_ text: String) -> [String] {
        all.compactMap { rule in
            guard let regex = try? NSRegularExpression(pattern: rule.pattern) else { return nil }
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            return regex.firstMatch(in: text, options: [], range: range) != nil ? rule.id : nil
        }
    }

    private static func replace(_ input: String, pattern: String, with template: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return input }
        let range = NSRange(input.startIndex..<input.endIndex, in: input)
        return regex.stringByReplacingMatches(in: input, options: [], range: range, withTemplate: template)
    }
}

/// Display classes for sensitivity probes (runtime-traces spec section 10):
/// the quarantine list names the CLASS that quarantined a session, never the
/// matched content. The mapping is keyed on the rule ids from
/// ``TesseraScrubRules/all`` so the quarantine surface and the scrub wall
/// share one vocabulary and cannot drift.
public enum TesseraProbeClass {
    public static let secrets = "Secrets"
    public static let contactInfo = "Contact info"
    public static let paths = "Paths"
    public static let modelMismatch = "Model mismatch"

    /// Human class for one scrub rule id. Unknown ids (a future rule the
    /// mapping has not caught up with) surface their own id: honest, and
    /// still never the matched content.
    public static func label(forRule id: String) -> String {
        switch id {
        case "pem-key", "bearer-token", "secret-key", "credential-assignment":
            return secrets
        case "email", "phone":
            return contactInfo
        case "fs-path":
            return paths
        default:
            return id
        }
    }

    /// Classes for one ledger entry's reasons: `probe:<id>` entries map
    /// through ``label(forRule:)``, `model-mismatch` maps to its own class,
    /// anything else is not a probe signal. Deduplicated in first-seen
    /// order.
    public static func classes(forLedgerReasons reasons: [String]) -> [String] {
        var seen = Set<String>()
        var out: [String] = []
        for reason in reasons {
            let label: String
            if reason.hasPrefix("probe:") {
                let id = String(reason.dropFirst("probe:".count))
                guard id != "none" else { continue }
                label = self.label(forRule: id)
            } else if reason == "model-mismatch" {
                label = modelMismatch
            } else {
                continue
            }
            if seen.insert(label).inserted {
                out.append(label)
            }
        }
        return out
    }
}
