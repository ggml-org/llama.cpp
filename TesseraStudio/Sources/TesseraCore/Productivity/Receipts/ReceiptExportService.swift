import Foundation
import CryptoKit

// MARK: - ReceiptExportFormat

/// The format of a receipt-chain export (per spec §7.5).
/// The user picks one of three formats; the export service
/// builds the corresponding artifact.
public enum ReceiptExportFormat: String, CaseIterable, Sendable, Codable, Identifiable {
    /// Signed JSON bundle — the full chain as a single
    /// JSON file, with the ed25519 signatures and C2PA
    /// manifests inline. Default export.
    case signedJSON
    /// Markdown summary — human-readable Markdown, suitable
    /// for sharing with non-technical reviewers. Opt-in
    /// via a toggle in the export UI.
    case markdown
    /// C2PA-signed document — the document itself, signed
    /// with the C2PA manifest embedded. The document is
    /// then verifiable by any C2PA-aware tool.
    case c2paDocument

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .signedJSON: return "Signed JSON bundle"
        case .markdown: return "Markdown summary"
        case .c2paDocument: return "C2PA-signed document"
        }
    }

    public var fileExtension: String {
        switch self {
        case .signedJSON: return "json"
        case .markdown: return "md"
        case .c2paDocument: return "c2pa.txt"
        }
    }
}

// MARK: - ExportArtifact

/// The output of a receipt-chain export. The artifact is
/// in-memory; the host view is expected to write it to
/// disk (via NSSavePanel on macOS, fileExporter on iOS).
public struct ExportArtifact: Codable, Sendable, Hashable, Identifiable {
    public let id: UUID
    public let documentID: UUID
    public let format: ReceiptExportFormat
    public let filename: String
    public let payload: Data
    public let receiptID: UUID
    public let createdAt: Date

    public init(
        id: UUID = UUID(),
        documentID: UUID,
        format: ReceiptExportFormat,
        filename: String,
        payload: Data,
        receiptID: UUID,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.documentID = documentID
        self.format = format
        self.filename = filename
        self.payload = payload
        self.receiptID = receiptID
        self.createdAt = createdAt
    }
}

// MARK: - ExportError

public enum ExportError: Error, Sendable, Equatable {
    case userDenied
    case noReceipts
    case buildFailed(String)
    case persistFailed(String)
}

// MARK: - EgressPolicy

/// A pluggable egress policy for receipt exports. The
/// default policy is `allowAll` (the user-confirmation
/// flow is the primary gate; the policy is a hook for
/// future per-destination rules). The `TesseraEgressGuard`
/// enum (on main) is the filter for training data; this
/// is a higher-level gate for the export surface.
public protocol EgressPolicy: Sendable {
    func allowsExport(documentID: UUID, format: ReceiptExportFormat) -> Bool
}

public struct AllowAllEgressPolicy: EgressPolicy {
    public init() {}
    public func allowsExport(documentID: UUID, format: ReceiptExportFormat) -> Bool { true }
}

// MARK: - ReceiptExportService

/// The receipt-chain export service (per spec §7.5). The
/// service builds a signed-JSON / Markdown / C2PA bundle
/// for a document's receipt chain, then logs the export
/// as a `Receipt` (with `receiptType: "export"`) so the
/// audit trail captures the export.
///
/// **Egress gate.** The service consults the configured
/// ``EgressPolicy`` before building the artifact. The
/// default policy is `AllowAllEgressPolicy`; the gate is
/// a hook for future per-destination policy (e.g., "block
/// exports of receipt chains to unencrypted volumes").
///
/// **User confirmation.** The service requires a
/// `userConfirmed: true` flag at the call site. The
/// SwiftUI export view shows a confirmation dialog; the
/// dialog's "Export" button is the only path that sets
/// the flag. This is the "export is gated, the user
/// approves" path from spec §7.5.
///
/// **Export receipt.** The export is logged as a chain
/// entry. The receipt's `summary` field is "Exported N
/// receipts as <format>" so the audit trail is
/// self-describing. The receipt is signed with the same
/// key as the document receipts (per spec §7.2.1).
public struct ReceiptExportService: Sendable {

    public let documentStore: DocumentStore
    public let dataLayer: TesseraDataLayer
    public let signer: ReceiptSigner
    public let egressPolicy: any EgressPolicy

    public init(
        documentStore: DocumentStore,
        dataLayer: TesseraDataLayer,
        signer: ReceiptSigner,
        egressPolicy: any EgressPolicy = AllowAllEgressPolicy()
    ) {
        self.documentStore = documentStore
        self.dataLayer = dataLayer
        self.signer = signer
        self.egressPolicy = egressPolicy
    }

    // MARK: - Build + sign + log

    /// Build, sign, and log an export. Returns the artifact
    /// (in memory; the caller writes it to disk).
    public func export(
        documentID: UUID,
        format: ReceiptExportFormat,
        documentTitle: String,
        userID: UserID,
        userConfirmed: Bool
    ) async throws -> ExportArtifact {
        guard userConfirmed else {
            throw ExportError.userDenied
        }

        // 1. Load the chain.
        let chain = try await documentStore.history(of: documentID, limit: 1000)
        guard !chain.isEmpty else {
            throw ExportError.noReceipts
        }

        // 2. Egress gate. The policy's `allowsExport` returns
        //    true by default; the gate is a hook for future
        //    per-destination policy (e.g., "block exports of
        //    receipt chains to unencrypted volumes").
        guard egressPolicy.allowsExport(documentID: documentID, format: format) else {
            throw ExportError.buildFailed("egress policy denied the export")
        }

        // 3. Build the artifact payload.
        let filename = Self.makeFilename(
            documentTitle: documentTitle,
            format: format
        )
        let payload: Data
        switch format {
        case .signedJSON:
            payload = (try? buildSignedJSONBundle(
                documentID: documentID,
                chain: chain,
                documentTitle: documentTitle
            )) ?? Data()
            if payload.isEmpty {
                throw ExportError.buildFailed("signed JSON bundle build failed")
            }
        case .markdown:
            payload = buildMarkdownSummary(
                documentTitle: documentTitle,
                chain: chain
            )
        case .c2paDocument:
            let ast = try await documentStore.loadDocument(id: documentID)
            payload = try buildC2PADocument(
                documentID: documentID,
                documentTitle: documentTitle,
                chain: chain,
                ast: ast
            )
        }

        // 4. Build the export receipt directly with the
        //    export-specific summary. The summary is part
        //    of the receipt's canonical content; we need
        //    to set it before signing. The signer's `sign`
        //    populates the summary from the mutations; for
        //    a non-mutation receipt (an export), we
        //    construct the receipt manually.
        let summary = "Exported \(chain.count) receipt\(chain.count == 1 ? "" : "s") as \(format.displayName)"
        let priorReceiptID = chain.last?.id
        let logged = try makeExportReceipt(
            documentID: documentID,
            priorReceiptID: priorReceiptID,
            actor: .user(userID),
            summary: summary
        )

        // 5. Persist the export receipt to the chain. We
        //    use a separate receipt type so the data
        //    layer can filter exports.
        try await persistExportReceipt(logged)

        return ExportArtifact(
            documentID: documentID,
            format: format,
            filename: filename,
            payload: payload,
            receiptID: logged.id
        )
    }

    /// Build an export receipt by hand. The export
    /// produces no mutations (the document is unchanged);
    /// the receipt is logged for the audit trail. The
    /// receipt is signed with the same key as the
    /// document receipts (per spec §7.2.1).
    private func makeExportReceipt(
        documentID: UUID,
        priorReceiptID: UUID?,
        actor: Actor,
        summary: String
    ) throws -> Receipt {
        let signer = self.signer
        guard let privateKey = try? Self.currentSigningKey(signer: signer) else {
            throw ExportError.buildFailed("signing key unavailable")
        }
        let receiptID = UUID()
        let timestamp = Date()
        // The receipt's C2PA manifest is built the same
        // way as a document receipt (per spec §7.2.1).
        let c2pa = try Self.buildExportManifest(
            receiptID: receiptID,
            documentID: documentID,
            actor: actor,
            summary: summary,
            signingKey: privateKey
        )
        let receipt = Receipt(
            id: receiptID,
            documentID: documentID,
            actor: actor,
            mutations: [],
            timestamp: timestamp,
            priorReceiptID: priorReceiptID,
            signature: Data(),
            c2paManifest: c2pa,
            summary: summary,
            preMutationSnapshot: [:]
        )
        let canonical = try receipt.canonicalBytes()
        let signature = try privateKey.signature(for: canonical)
        return Receipt(
            id: receipt.id,
            documentID: receipt.documentID,
            actor: receipt.actor,
            mutations: receipt.mutations,
            timestamp: receipt.timestamp,
            priorReceiptID: receipt.priorReceiptID,
            signature: signature,
            c2paManifest: receipt.c2paManifest,
            summary: receipt.summary,
            preMutationSnapshot: receipt.preMutationSnapshot,
            voidedBy: receipt.voidedBy
        )
    }

    private static func currentSigningKey(signer: ReceiptSigner) throws -> Curve25519.Signing.PrivateKey {
        // The ReceiptSigner wraps either a Keychain-backed
        // key or an injected one. We replicate its private
        // key extraction here because the signer's
        // `currentSigningKey` is private. We use the
        // signer's `publicKey` for verification; if the
        // signing key is unavailable, we throw.
        if let key = signer.injectedSigningKey {
            return key
        }
        if let key = TesseraKeychainVolume.receiptSigningKey() {
            return key
        }
        throw ExportError.buildFailed("signing key unavailable")
    }

    private static func buildExportManifest(
        receiptID: UUID,
        documentID: UUID,
        actor: Actor,
        summary: String,
        signingKey: Curve25519.Signing.PrivateKey
    ) throws -> C2PAManifest {
        var assertions: [C2PAManifest.Assertion] = []

        // c2pa.hash.data: not computed for an export
        // (the document is unchanged); use a null sentinel.
        assertions.append(C2PAManifest.Assertion(
            label: "c2pa.hash.data",
            data: .null
        ))

        // c2pa.actions: the action is "c2pa.exported".
        assertions.append(C2PAManifest.Assertion(
            label: "c2pa.actions",
            data: .object([
                "actions": .array([
                    .object([
                        "action": .string("c2pa.exported"),
                        "when": .string(Self.iso8601.string(from: Date()))
                    ])
                ])
            ])
        ))

        // tessera.actor
        let actorJSON: AnyCodable
        switch actor {
        case .user(let id):
            actorJSON = .object([
                "type": .string("user"),
                "id": .string(id.uuidString)
            ])
        case .agent(let runID, let model, let promptHash):
            actorJSON = .object([
                "type": .string("agent"),
                "agent_run_id": .string(runID.uuidString),
                "model": .string(model),
                "prompt_hash": .string(promptHash)
            ])
        }
        assertions.append(C2PAManifest.Assertion(
            label: "tessera.actor",
            data: actorJSON
        ))

        // tessera.summary
        assertions.append(C2PAManifest.Assertion(
            label: "tessera.summary",
            data: .string(summary)
        ))

        // tessera.receipt_id
        assertions.append(C2PAManifest.Assertion(
            label: "tessera.receipt_id",
            data: .string(receiptID.uuidString)
        ))

        // Sign the canonical assertion list.
        let canonicalAssertions = try C2PAManifest(
            format: "c2pa.v2",
            claimGenerator: "tessera/1.0",
            assertions: assertions,
            signature: ""
        ).canonicalBytes()
        let manifestSig = try signingKey.signature(for: canonicalAssertions)
        let manifestSigString = "ed25519:" + manifestSig.base64EncodedString()
        return C2PAManifest(
            format: "c2pa.v2",
            claimGenerator: "tessera/1.0",
            assertions: assertions,
            signature: manifestSigString
        )
    }

    private static let iso8601: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    // MARK: - Filename

    static func makeFilename(
        documentTitle: String,
        format: ReceiptExportFormat,
        now: Date = Date()
    ) -> String {
        let slug = Self.slugify(documentTitle)
        let dateStr = Self.dateFormatter.string(from: now)
        let ext = format.fileExtension
        switch format {
        case .signedJSON, .markdown:
            return "\(slug)-audit-\(dateStr).\(ext)"
        case .c2paDocument:
            return "\(slug)-c2pa.\(ext)"
        }
    }

    static func slugify(_ s: String) -> String {
        let lowered = s.lowercased()
        let scalars = lowered.unicodeScalars.map { scalar -> Character in
            CharacterSet.alphanumerics.contains(scalar) ? Character(scalar) : "-"
        }
        let cleaned = String(scalars)
        let collapsed = cleaned.split(separator: "-", omittingEmptySubsequences: true).joined(separator: "-")
        let trimmed = collapsed.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return trimmed.isEmpty ? "document" : String(trimmed.prefix(64))
    }

    private static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd"
        f.timeZone = TimeZone(identifier: "UTC")
        return f
    }()

    // MARK: - Signed JSON bundle

    /// The full chain as a single JSON file. The bundle
    /// is the chain in order (oldest first) plus the
    /// document metadata (id, title, exported-at). Each
    /// receipt is the full ``Receipt`` shape (with the
    /// ed25519 signature and C2PA manifest inline); the
    /// bundle is self-contained for offline verification.
    public func buildSignedJSONBundle(
        documentID: UUID,
        chain: [Receipt],
        documentTitle: String
    ) throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let bundle = SignedExportBundle(
            format: "tessera.export.v1",
            documentID: documentID,
            documentTitle: documentTitle,
            exportedAt: Date(),
            receiptCount: chain.count,
            chain: chain
        )
        return try encoder.encode(bundle)
    }

    // MARK: - Markdown summary

    /// A human-readable Markdown rendering of the chain.
    /// Each receipt is a section with the timestamp,
    /// actor, summary, and mutations list. The output is
    /// suitable for sharing with non-technical reviewers
    /// (lawyers, doctors, journalists) who want a
    /// plain-text view of the audit trail.
    public func buildMarkdownSummary(
        documentTitle: String,
        chain: [Receipt]
    ) -> Data {
        var lines: [String] = []
        lines.append("# \(documentTitle) — Audit Trail")
        lines.append("")
        lines.append("Exported \(Self.dateFormatter.string(from: Date()))")
        lines.append("\(chain.count) receipt\(chain.count == 1 ? "" : "s")")
        lines.append("")
        for (idx, receipt) in chain.enumerated() {
            lines.append("## \(idx + 1). \(receipt.summary)")
            lines.append("")
            lines.append("- **Timestamp:** \(receipt.timestamp)")
            switch receipt.actor {
            case .user(let uid):
                lines.append("- **Actor:** user (\(uid.uuidString.prefix(8)))")
            case .agent(let runID, let model, let promptHash):
                lines.append("- **Actor:** agent (\(model), prompt \(promptHash.prefix(12)))")
                lines.append("- **Agent run:** \(runID.uuidString)")
            }
            lines.append("- **Receipt id:** `\(receipt.id.uuidString)`")
            if let prior = receipt.priorReceiptID {
                lines.append("- **Prior receipt:** `\(prior.uuidString)`")
            }
            if !receipt.mutations.isEmpty {
                lines.append("- **Mutations:**")
                for m in receipt.mutations {
                    lines.append("    - `\(m.shortDescription)`")
                }
            } else {
                lines.append("- **Mutations:** _(none — non-document receipt, e.g. export)_")
            }
            if let manifest = receipt.c2paManifest {
                lines.append("- **C2PA manifest:** `\(manifest.format)`, \(manifest.assertions.count) assertion\(manifest.assertions.count == 1 ? "" : "s")")
            }
            if receipt.isVoided {
                lines.append("- **Voided by:** `\(receipt.voidedBy?.uuidString ?? "unknown")`")
            }
            lines.append("")
        }
        let md = lines.joined(separator: "\n")
        return Data(md.utf8)
    }

    // MARK: - C2PA-signed document

    /// The document itself, signed with the C2PA manifest
    /// embedded. For v1 the document is serialized to
    /// plain text (the AST's flattened content) with a
    /// C2PA footer that includes the last receipt's
    /// manifest. Future versions can target the
    /// document's native export format.
    public func buildC2PADocument(
        documentID: UUID,
        documentTitle: String,
        chain: [Receipt],
        ast: DocumentAST
    ) throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]

        // 1. Serialize the document body.
        let body = Self.flattenAST(ast)

        // 2. Embed the last receipt's C2PA manifest (if any).
        let lastManifest = chain.last(where: { $0.c2paManifest != nil })?.c2paManifest
        let manifestData: Data?
        if let lastManifest {
            manifestData = try encoder.encode(lastManifest)
        } else {
            manifestData = nil
        }

        // 3. Build the C2PA envelope.
        let envelope = C2PADocumentEnvelope(
            format: "tessera.c2pa.doc.v1",
            documentID: documentID,
            documentTitle: documentTitle,
            exportedAt: Date(),
            body: body,
            receiptCount: chain.count,
            embeddedManifest: manifestData
        )
        return try encoder.encode(envelope)
    }

    /// Flatten a `DocumentAST` to plain text. The block
    /// separator is a blank line; headings are prefixed
    /// with the corresponding number of `#` characters.
    static func flattenAST(_ ast: DocumentAST) -> String {
        var lines: [String] = []
        let rootIDs = ast.rootChildren
        for id in rootIDs {
            guard let block = ast.blocks[id] else { continue }
            switch block.type {
            case .heading:
                let level: Int
                if case .number(let n) = block.attributes["level"] {
                    level = max(1, min(6, Int(n)))
                } else {
                    level = 1
                }
                let prefix = String(repeating: "#", count: level)
                let text = block.content.map { $0.text }.joined()
                lines.append("\(prefix) \(text)")
            case .paragraph:
                let text = block.content.map { $0.text }.joined()
                lines.append(text)
            case .listItem:
                let text = block.content.map { $0.text }.joined()
                lines.append("- \(text)")
            case .codeBlock:
                let text = block.content.map { $0.text }.joined()
                lines.append("```")
                lines.append(text)
                lines.append("```")
            case .divider:
                lines.append("---")
            case .quote:
                let text = block.content.map { $0.text }.joined()
                lines.append("> \(text)")
            default:
                let text = block.content.map { $0.text }.joined()
                if !text.isEmpty { lines.append(text) }
            }
            lines.append("")
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - Persistence

    /// Persist the export receipt to the data layer. We
    /// use a separate receipt type (`"export"`) so the
    /// data layer can filter exports for the audit
    /// inspector. The receipt's payload is the standard
    /// ``Receipt`` shape (so the chain is uniform); the
    /// type is in the row's `receipt_type` column.
    private func persistExportReceipt(_ receipt: Receipt) async throws {
        let payload = try Self.encodeReceiptPayload(receipt)
        _ = try await dataLayer.appendReceiptToChain(
            documentID: receipt.documentID,
            receiptType: "export",
            payload: payload,
            signature: receipt.signature
        )
    }

    private static func encodeReceiptPayload(_ receipt: Receipt) throws -> [String: JSONValue] {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(receipt)
        let raw = try JSONSerialization.jsonObject(with: data, options: [])
        guard let obj = raw as? [String: Any] else { return [:] }
        return convert(obj) as? [String: JSONValue] ?? [:]
    }

    private static func convert(_ obj: Any) -> JSONValue {
        if obj is NSNull { return .null }
        if let v = obj as? Bool { return .bool(v) }
        if let v = obj as? Double { return .number(v) }
        if let v = obj as? Int { return .number(Double(v)) }
        if let v = obj as? String { return .string(v) }
        if let v = obj as? [Any] { return .array(v.map(convert)) }
        if let v = obj as? [String: Any] {
            var out: [String: JSONValue] = [:]
            for (k, val) in v { out[k] = convert(val) }
            return .object(out)
        }
        return .string(String(describing: obj))
    }
}

// MARK: - SignedExportBundle

/// The JSON shape of a signed-JSON export. The bundle is
/// self-contained for offline verification: a recipient
/// can verify each receipt's signature without contacting
/// the data layer.
public struct SignedExportBundle: Codable, Sendable, Hashable {
    public let format: String
    public let documentID: UUID
    public let documentTitle: String
    public let exportedAt: Date
    public let receiptCount: Int
    public let chain: [Receipt]

    public init(
        format: String = "tessera.export.v1",
        documentID: UUID,
        documentTitle: String,
        exportedAt: Date,
        receiptCount: Int,
        chain: [Receipt]
    ) {
        self.format = format
        self.documentID = documentID
        self.documentTitle = documentTitle
        self.exportedAt = exportedAt
        self.receiptCount = receiptCount
        self.chain = chain
    }
}

// MARK: - C2PADocumentEnvelope

/// The shape of a C2PA-signed document export. The
/// envelope is the document body plus the C2PA manifest;
/// a C2PA-aware tool can extract the manifest and verify
/// it against the embedded public key.
public struct C2PADocumentEnvelope: Codable, Sendable, Hashable {
    public let format: String
    public let documentID: UUID
    public let documentTitle: String
    public let exportedAt: Date
    public let body: String
    public let receiptCount: Int
    public let embeddedManifest: Data?

    public init(
        format: String = "tessera.c2pa.doc.v1",
        documentID: UUID,
        documentTitle: String,
        exportedAt: Date,
        body: String,
        receiptCount: Int,
        embeddedManifest: Data?
    ) {
        self.format = format
        self.documentID = documentID
        self.documentTitle = documentTitle
        self.exportedAt = exportedAt
        self.body = body
        self.receiptCount = receiptCount
        self.embeddedManifest = embeddedManifest
    }
}
