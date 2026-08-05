import Foundation
import CryptoKit

// MARK: - ReceiptSigner

/// Signs and verifies ``Receipt`` instances using the device's
/// ed25519 key (derived from the volume password by
/// ``TesseraKeychainVolume``). Single point of contact for the
/// receipt signing path: every receipt that's persisted goes
/// through here.
///
/// The signer composes with ``TesseraKeychainVolume`` rather than
/// introducing a new Keychain dependency. The volume password
/// doubles as the signing-key seed (per
/// `docs/tessera-productivity-foundations-design.md` §5). When the
/// ``PleadTheFifthExecutor`` step 3 destroys the volume password,
/// the signing key disappears with it -- prior signed receipts are
/// unverifiable on the device that produced them, by design.
///
/// For testing, the signer accepts an injected private key via
/// ``init(signingKey:)`` so unit tests don't need a real Keychain
/// entry.
public struct ReceiptSigner: Sendable {

    /// The signing key source. In production this is the Keychain
    /// (via ``TesseraKeychainVolume``); in tests it can be an
    /// injected ephemeral key.
    private let keySource: KeySource

    public init() {
        self.keySource = .keychain
    }

    public init(signingKey: Curve25519.Signing.PrivateKey) {
        self.keySource = .injected(signingKey)
    }

    /// True iff a signing key is currently available. When false,
    /// ``sign(_:priorReceipt:actor:)`` throws
    /// ``ReceiptSignerError/signingKeyUnavailable``.
    public var hasSigningKey: Bool {
        switch keySource {
        case .keychain: return TesseraKeychainVolume.receiptSigningKey() != nil
        case .injected: return true
        }
    }

    /// The current public key (for verification, embedded in the
    /// chain). Nil when the signing key is unavailable.
    public var publicKey: Curve25519.Signing.PublicKey? {
        switch keySource {
        case .keychain: return TesseraKeychainVolume.receiptVerificationKey()
        case .injected(let key): return key.publicKey
        }
    }

    /// The injected signing key, when this signer was
    /// initialized with one. Used by the export service to
    /// build export receipts with a custom summary (the
    /// regular `sign(_:)` path doesn't expose the summary
    /// override). Nil for the production (Keychain-backed)
    /// signer.
    public var injectedSigningKey: Curve25519.Signing.PrivateKey? {
        switch keySource {
        case .keychain: return nil
        case .injected(let key): return key
        }
    }

    /// Sign a new receipt. The receipt's `id`, `timestamp`, and
    /// `signature` are populated by this call; the `summary` is
    /// composed from the mutations. The `c2paManifest` is generated
    /// here too (so the manifest's signature covers the same
    /// content the receipt signature covers).
    ///
    /// The `preMutationSnapshot` is the blocks-affected snapshot
    /// captured at apply-time (see `MutationEngine.apply`). It is
    /// embedded in the receipt so the receipt is self-contained
    /// for undo: the inverse mutations are computed from this
    /// snapshot, not from the current document state.
    public func sign(
        documentID: UUID,
        mutations: [Mutation],
        priorReceiptID: UUID?,
        actor: Actor,
        timestamp: Date = Date(),
        documentContentHash: String? = nil,
        preMutationSnapshot: [UUID: Block] = [:]
    ) throws -> Receipt {
        let privateKey = try currentSigningKey()
        let publicKey = privateKey.publicKey

        let receiptID = UUID()
        let summary = composeSummary(for: mutations)
        let c2pa = try makeC2PAManifest(
            receiptID: receiptID,
            documentID: documentID,
            actor: actor,
            mutations: mutations,
            summary: summary,
            contentHash: documentContentHash,
            signingKey: privateKey
        )
        var receipt = Receipt(
            id: receiptID,
            documentID: documentID,
            actor: actor,
            mutations: mutations,
            timestamp: timestamp,
            priorReceiptID: priorReceiptID,
            signature: Data(),  // placeholder, set below
            c2paManifest: c2pa,
            summary: summary,
            preMutationSnapshot: preMutationSnapshot
        )
        let canonical = try receipt.canonicalBytes()
        let signature = try privateKey.signature(for: canonical)
        receipt = Receipt(
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
        // Sanity check: the signature we just made MUST verify.
        if !receipt.verify(against: publicKey) {
            throw ReceiptSignerError.signingFailed("signature failed self-verification")
        }
        return receipt
    }

    /// Verify a receipt against a public key. Convenience wrapper
    /// around ``Receipt/verify(against:)`` that also returns a
    /// discriminated result (valid / invalid / voided) for the
    /// receipt drawer UI.
    public func verify(
        _ receipt: Receipt,
        against publicKey: Curve25519.Signing.PublicKey
    ) -> ReceiptVerification {
        if receipt.isVoided { return .voided(by: receipt.voidedBy!) }
        return receipt.verify(against: publicKey) ? .valid : .invalid
    }

    // MARK: - Internals

    private func currentSigningKey() throws -> Curve25519.Signing.PrivateKey {
        switch keySource {
        case .keychain:
            guard let key = TesseraKeychainVolume.receiptSigningKey() else {
                throw ReceiptSignerError.signingKeyUnavailable
            }
            return key
        case .injected(let key):
            return key
        }
    }

    private func composeSummary(for mutations: [Mutation]) -> String {
        if mutations.isEmpty { return "no-op" }
        if mutations.count == 1 { return mutations[0].shortDescription }
        // Group by shortDescription to compress repeated edits.
        var groups: [String: Int] = [:]
        for m in mutations { groups[m.shortDescription, default: 0] += 1 }
        return groups
            .sorted { $0.key < $1.key }
            .map { entry in entry.value == 1 ? entry.key : "\(entry.value) × \(entry.key)" }
            .joined(separator: ", ")
    }

    private func makeC2PAManifest(
        receiptID: UUID,
        documentID: UUID,
        actor: Actor,
        mutations: [Mutation],
        summary: String,
        contentHash: String?,
        signingKey: Curve25519.Signing.PrivateKey
    ) throws -> C2PAManifest {
        var assertions: [C2PAManifest.Assertion] = []

        // Standard c2pa.hash.data assertion (always present, even if
        // the caller didn't pass a content hash -- the field is set
        // to nil-sentinel in that case so the verifier knows the
        // hash wasn't computed at sign-time).
        let hashData: AnyCodable = contentHash.map { .string($0) } ?? .null
        assertions.append(C2PAManifest.Assertion(
            label: "c2pa.hash.data",
            data: hashData
        ))

        // Standard c2pa.actions assertion: the action this receipt
        // represents is `c2pa.edited` (per the spec §5.3).
        assertions.append(C2PAManifest.Assertion(
            label: "c2pa.actions",
            data: .object([
                "actions": .array([
                    .object([
                        "action": .string("c2pa.edited"),
                        "when": .string(Self.iso8601.string(from: Date()))
                    ])
                ])
            ])
        ))

        // Tessera-specific: the actor. Stored as a tagged object so
        // a future verifier can tell user from agent without the
        // full receipt payload.
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

        // Tessera-specific: the human-readable summary.
        assertions.append(C2PAManifest.Assertion(
            label: "tessera.summary",
            data: .string(summary)
        ))

        // Tessera-specific: the receipt id (so a verifier can
        // cross-reference the manifest with the receipt in the chain).
        assertions.append(C2PAManifest.Assertion(
            label: "tessera.receipt_id",
            data: .string(receiptID.uuidString)
        ))

        // Sign the canonical assertion list. The signature is over
        // the assertions only (not the metadata fields), so a
        // future verifier with the assertions can re-verify without
        // knowing which Tessera build produced the manifest.
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

    // MARK: - KeySource

    private enum KeySource: Sendable {
        case keychain
        case injected(Curve25519.Signing.PrivateKey)
    }
}

// MARK: - Verification result

public enum ReceiptVerification: Sendable, Equatable {
    case valid
    case invalid
    case voided(by: UUID)

    public var isAcceptable: Bool {
        switch self {
        case .valid: return true
        case .invalid, .voided: return false
        }
    }
}

// MARK: - ReceiptSignerError

public enum ReceiptSignerError: Error, Sendable, Equatable {
    case signingKeyUnavailable
    case signingFailed(String)
}
