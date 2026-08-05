import Foundation
#if canImport(AppKit)
import AppKit
#endif

// MARK: - ShareSheetCoordinator

/// Presents the system share sheet on macOS via
/// ``NSSharingServicePicker`` (and iOS via
/// ``UIActivityViewController``, behind an ``#if canImport(UIKit)``
/// guard). The coordinator owns the list of available share
/// targets; the picker and the iOS activity view controller
/// are constructed on demand from that list.
///
/// The coordinator is an actor so concurrent invocations from
/// the UI don't race. The ``availableShareTargets()`` method
/// is the source of truth for what the user can pick from;
/// the v1 list is the system share sheet (which itself
/// discovers Mail, Messages, AirDrop, etc.) plus the
/// in-process Slack / Discord / Teams webhooks the user
/// has configured.
public actor ShareSheetCoordinator {
    private let slackTargets: [SlackExportTarget]
    private let customTargets: [ShareTarget]

    public init(
        slackTargets: [SlackExportTarget] = [],
        customTargets: [ShareTarget] = []
    ) {
        self.slackTargets = slackTargets
        self.customTargets = customTargets
    }

    /// Returns the share targets the user can currently pick
    /// from. The system share sheet is always present; the
    /// in-process targets (Slack, etc.) are appended when
    /// configured.
    public func availableShareTargets() -> [ShareTarget] {
        var out: [ShareTarget] = []
        #if canImport(AppKit)
        // The system share sheet is a single "target" in the
        // v1 model: it represents ``NSSharingServicePicker``,
        // which is the macOS way to expose Mail / Messages /
        // AirDrop / etc. without per-service integration.
        out.append(
            ShareTarget(
                id: "system.sharing-service-picker",
                name: "Share via…",
                accepts: Set(ProductivityExportFormat.allCases),
                handler: { _ in
                    // The picker is shown via
                    // ``presentShareSheet``; this handler is a
                    // no-op because the picker handles its
                    // own routing.
                }
            )
        )
        #endif
        for slack in slackTargets {
            out.append(slack.shareTarget)
        }
        out += customTargets
        return out
    }

    /// Present the macOS system share sheet anchored to the
    /// given view. The picker shows the user all installed
    /// share services (Mail, Messages, AirDrop, etc.) that
    /// can accept the entity's formats; the user picks one
    /// and the system handles the handoff.
    ///
    /// On iOS this is a thin wrapper around
    /// ``UIActivityViewController`` (compiled in the
    /// TesseraStudioiOS target's ``#if os(iOS)``-guarded
    /// sources). On macOS it uses ``NSSharingServicePicker``.
    @MainActor
    public func presentShareSheet(
        for entityID: UUID,
        from view: NSView,
        exporter: TesseraExporter
    ) async throws {
        // Stage the file once so the picker has a real URL
        // to share. We use the markdown export (the most
        // universal); the picker lets the user pick another
        // format on its own.
        let staged = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tessera-shared-\(entityID.uuidString).md")
        try await exporter.export(entityID: entityID, to: .md, outputURL: staged)
        #if canImport(AppKit)
        let picker = NSSharingServicePicker(
            items: [staged]
        )
        picker.show(relativeTo: view.bounds, of: view, preferredEdge: .minY)
        #endif
    }
}

// MARK: - SlackExportTarget

/// A Slack webhook target. The webhook URL is stored in
/// macOS Keychain (via ``TesseraKeychainVolume``'s standard
/// entry) so it never lands in ``UserDefaults`` or in the
/// data layer's Postgres. The mrkdwn formatter is a small
/// purpose-built converter (Slack's mrkdwn is markdown-like
/// but not CommonMark: ``*bold*`` instead of ``**bold**``,
/// no ``# heading`` syntax, etc.).
///
/// v1 supports text-only posts. Attachments (file uploads
/// via Slack's files.upload API) are punted to v2.
public struct SlackExportTarget: Sendable {
    public var webhookURL: URL
    public var channel: String?
    public var username: String?

    public init(
        webhookURL: URL,
        channel: String? = nil,
        username: String? = nil
    ) {
        self.webhookURL = webhookURL
        self.channel = channel
        self.username = username
    }

    /// The ``ShareTarget`` for this Slack target. The
    /// handler exports the document to Markdown (Slack's
    /// preferred format for ``mrkdwn``) and POSTs the
    /// formatted payload to the webhook.
    public var shareTarget: ShareTarget {
        let me = self
        return ShareTarget(
            id: "slack.\(webhookURL.absoluteString.hashValue)",
            name: channel.map { "Slack — \($0)" } ?? "Slack",
            accepts: [.md, .html, .pdf],
            handler: { url in
                try await me.post(document: url)
            }
        )
    }

    /// Format the document and POST it to the webhook.
    /// The handler reads the file at ``document``, formats
    /// it as Slack mrkdwn, and posts. The HTTP call uses
    /// ``URLSession`` (no third-party deps).
    public func post(document fileURL: URL) async throws {
        let raw = try String(contentsOf: fileURL, encoding: String.Encoding.utf8)
        let formatted = SlackMrkdwnFormatter.format(raw)
        let payload = SlackPayload(
            text: formatted,
            channel: channel,
            username: username
        )
        let data = try JSONEncoder().encode(payload)
        var request = URLRequest(url: webhookURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = data
        let (_, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            throw SlackExportError.postFailed
        }
    }
}

public enum SlackExportError: Error, LocalizedError {
    case postFailed
    case invalidWebhookURL
    public var errorDescription: String? {
        switch self {
        case .postFailed: return "Slack webhook post failed"
        case .invalidWebhookURL: return "Invalid Slack webhook URL"
        }
    }
}

/// The Slack POST payload (only the fields we set in v1).
struct SlackPayload: Codable, Sendable {
    let text: String
    let channel: String?
    let username: String?
}

// MARK: - SlackMrkdwnFormatter

/// Convert a Markdown string to Slack mrkdwn.
///
/// Slack's mrkdwn is documented at
/// <https://api.slack.com/reference/surfaces/formatting>.
/// The differences from CommonMark that matter for our
/// documents:

/// * ``**bold**`` → ``*bold*``
/// * ``*italic*`` → ``_italic_`` (italic is underscores in
///   mrkdwn; ``*`` is reserved for bold)
/// * ``~~strike~~`` → ``~strike~``
/// * ``# heading`` → ``*heading*`` (Slack has no heading
///   syntax; we promote to bold)
/// * ``[text](url)`` → ``<url|text>``
/// * ``- item`` / ``1. item`` → ``• item`` / ``1. item``
///   (Slack has a ``-`` for list-like; we use the
///   bullet char so the lists look right in the chat
///   client)
public enum SlackMrkdwnFormatter {
    public static func format(_ markdown: String) -> String {
        var out = markdown
        // 1) links: [text](url) -> <url|text>
        out = _linkReplacing(in: out)
        // 2) bold: **text** -> *text*
        out = _boldReplacing(in: out)
        // 3) italic: *text* -> _text_  (but only the single-asterisk
        //    form; double-asterisks are already converted)
        out = _italicReplacing(in: out)
        // 4) strikethrough: ~~text~~ -> ~text~
        out = _strikeReplacing(in: out)
        // 5) headings: # ... ###### -> *...*
        out = _headingReplacing(in: out)
        // 6) bullets: - item or * item -> • item
        out = _bulletReplacing(in: out)
        return out
    }

    private static func _linkReplacing(in s: String) -> String {
        // Simple non-nested replacement. A real implementation
        // would tokenize; v1's documents don't have nested
        // markdown.
        return _replace(in: s, pattern: #"\[([^\]]+)\]\(([^)]+)\)"#) { groups in
            guard groups.count >= 3 else { return "" }
            return "<\(groups[2])|\(groups[1])>"
        }
    }

    private static func _boldReplacing(in s: String) -> String {
        return _replace(in: s, pattern: #"\*\*([^*]+)\*\*"#) { groups in
            groups.count >= 2 ? "*\(groups[1])*" : ""
        }
    }

    private static func _italicReplacing(in s: String) -> String {
        // Single asterisk, not double (already handled) and not
        // adjacent to alphanumeric boundary characters that
        // would make it a bullet.
        return _replace(in: s, pattern: #"(?<!\*)\*([^*]+)\*(?!\*)"#) { groups in
            groups.count >= 2 ? "_\(groups[1])_" : ""
        }
    }

    private static func _strikeReplacing(in s: String) -> String {
        return _replace(in: s, pattern: #"~~([^~]+)~~"#) { groups in
            groups.count >= 2 ? "~\(groups[1])~" : ""
        }
    }

    private static func _headingReplacing(in s: String) -> String {
        // Match a line that starts with 1-6 hashes followed by
        // a space; promote to bold.
        return _replace(in: s, pattern: #"(?m)^#{1,6}\s+(.+)$"#) { groups in
            groups.count >= 2 ? "*\(groups[1])*" : ""
        }
    }

    private static func _bulletReplacing(in s: String) -> String {
        return _replace(in: s, pattern: #"(?m)^[\s]*[-*]\s+(.+)$"#) { groups in
            groups.count >= 2 ? "• \(groups[1])" : ""
        }
    }

    private static func _replace(
        in s: String,
        pattern: String,
        with replacement: (_ groups: [String]) -> String
    ) -> String {
        guard let re = try? NSRegularExpression(pattern: pattern, options: []) else {
            return s
        }
        let ns = s as NSString
        var result = ""
        var lastEnd = 0
        let matches = re.matches(in: s, options: [], range: NSRange(location: 0, length: ns.length))
        for m in matches {
            let r = m.range
            if r.location > lastEnd {
                result += ns.substring(with: NSRange(location: lastEnd, length: r.location - lastEnd))
            }
            var groups: [String] = []
            for i in 0..<m.numberOfRanges {
                let groupRange = m.range(at: i)
                if groupRange.location == NSNotFound {
                    groups.append("")
                } else {
                    groups.append(ns.substring(with: groupRange))
                }
            }
            result += replacement(groups)
            lastEnd = r.location + r.length
        }
        if lastEnd < ns.length {
            result += ns.substring(with: NSRange(location: lastEnd, length: ns.length - lastEnd))
        }
        return result
    }
}
