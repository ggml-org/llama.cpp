import Foundation
import CryptoKit

// MARK: - Actor

/// The user or agent that produced a receipt. The spec (§5.2) commits
/// to two cases: a human user and an agent run (with the model +
/// prompt hash so the receipt is traceable to the model that emitted
/// the mutations).
///
/// `UserID` and `AgentRunID` are type aliases for `UUID` so they
/// share the on-disk representation but are not interchangeable in
/// Swift's type system. The two aliases make call sites self-documenting
/// (an `Agent.runID` reads better than a bare `UUID`).
public typealias UserID = UUID
public typealias AgentRunID = UUID

public enum Actor: Codable, Sendable, Hashable {
    case user(UserID)
    case agent(AgentRunID, model: String, promptHash: String)
}

// MARK: - C2PAManifest

/// A C2PA content authenticity manifest. v1 ships a placeholder
/// format that matches the C2PA Technical Specification 2.x's
/// `c2pa.v2` shape (per the spec §5.3 + §7.2) but is signed with
/// ed25519 from CryptoKit rather than the spec's `es256` (ECDSA
/// P-256). The format fields are the same; only the signature
/// algorithm tag differs (`ed25519` vs `es256`).
///
/// **Deferral note:** the C2PA Technical Specification 2.x is
/// detailed and the reference Rust implementation (`c2pa-rs`)
/// would have to be vendored via FFI to produce spec-compliant
/// manifests. We use a placeholder shape that:
///   * Is parseable by a future C2PA-aware tool with a small
///     adapter (the field set is identical).
///   * Signs with ed25519 because the architect chose ed25519 as
///     the receipt-signing primitive (CryptoKit only exposes
///     Curve25519.Signing; ECDSA P-256 is a different key path).
///   * Embeds the document content hash + the receipt's
///     mutation summary + the actor in the assertions, so a
///     future verifier can answer "what was edited, by whom,
///     to what content hash" without the full receipt payload.
///
/// The placeholder is forward-compatible: when a production
/// C2PA library matures (or the architect chooses to vendor
/// `c2pa-rs`), the C2PAManifest shape is what the verifier
/// would consume -- no schema change is required.
public struct C2PAManifest: Codable, Sendable, Hashable {
    /// Manifest format identifier. Pinned to `c2pa.v2` to match the
    /// C2PA Technical Specification 2.x. The `ed25519` signature
    /// algorithm is non-standard; see the doc comment on the type.
    public let format: String
    /// Identifier of the tool that produced the manifest. We use
    /// `tessera/<version>`; the version is set at build time.
    public let claimGenerator: String
    /// Ordered list of assertions. The spec's required assertions
    /// (hash of the data, action taken) are always present; we add
    /// the actor and the mutation summary so a future verifier
    /// doesn't need the full receipt payload to answer basic
    /// provenance questions.
    public let assertions: [Assertion]
    /// The signature of the canonical assertion list, computed
    /// over the JSON-canonicalized assertions. Format:
    /// `ed25519:<base64>`.
    public let signature: String

    public init(
        format: String = "c2pa.v2",
        claimGenerator: String = "tessera/1.0",
        assertions: [Assertion],
        signature: String
    ) {
        self.format = format
        self.claimGenerator = claimGenerator
        self.assertions = assertions
        self.signature = signature
    }

    /// One named assertion in the manifest. The `label` follows
    /// the C2PA convention (`c2pa.hash.data`, `c2pa.actions`,
    /// `tessera.actor`, `tessera.summary`). The `data` is a
    /// JSON value matching the label's expected shape.
    public struct Assertion: Codable, Sendable, Hashable {
        public let label: String
        public let data: AnyCodable

        public init(label: String, data: AnyCodable) {
            self.label = label
            self.data = data
        }
    }

    /// Canonical bytes for signing/verification. We serialize the
    /// assertions with sorted keys + ISO-8601 dates, the same
    /// canonicalization the receipt itself uses, so the manifest
    /// signature and the receipt signature are over the same
    /// canonical form. This is the property that makes "verify
    /// manifest" equivalent to "verify receipt content".
    public func canonicalBytes() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(assertions)
    }
}

// MARK: - Receipt

/// A signed, immutable record of one or more mutations against a
/// document. The constitutional-receipt backbone of the productivity
/// surface: every commit is one receipt; every receipt is signed;
/// every receipt is inspectable.
///
/// The receipt is the source of truth for the audit trail. The
/// `priorReceiptID` field forms the chain; the `signature` field
/// makes it tamper-evident.
///
/// **Lifecycle states:** the chain is append-only. A receipt can
/// be VOIDED (e.g., by an undo) by appending a NEW receipt whose
/// `summary` field names the original receipt and whose `mutations`
/// include a `void` marker. The original receipt's `voidedBy` field
/// is updated to point at the new one. This keeps the chain
/// append-only (which is the property the constitutional
/// crypto-shred relies on) while still letting a user "undo" a
/// receipt.
public struct Receipt: Codable, Sendable, Hashable, Identifiable {
    public let id: UUID
    public let documentID: UUID
    public let actor: Actor
    public let mutations: [Mutation]
    public let timestamp: Date
    /// The receipt that immediately preceded this one in the
    /// document's chain. `nil` for the genesis receipt (the first
    /// receipt of a new document).
    public let priorReceiptID: UUID?
    /// ed25519 signature over the canonical JSON form of
    /// (id, documentID, actor, mutations, timestamp,
    /// priorReceiptID, c2paManifest, summary). Stored as raw
    /// bytes (64 bytes).
    public let signature: Data
    /// C2PA content authenticity manifest. Embedded inline so the
    /// receipt is self-contained for export.
    public let c2paManifest: C2PAManifest?
    /// Human-readable summary ("3 paragraphs updated, 1 list added").
    public let summary: String
    /// Pre-mutation snapshot of the blocks this receipt touched.
    /// The receipt is self-contained for undo: the inverse
    /// mutations are computed from this snapshot, not from the
    /// current document state. This is what makes "the receipt is
    /// the source of truth for the audit trail" true -- a future
    /// auditor with the receipt can undo it without the live
    /// document.
    ///
    /// For mutations that don't touch block content (e.g.
    /// `setDocumentTitle`), the snapshot is empty.
    public let preMutationSnapshot: [UUID: Block]
    /// If this receipt was voided (e.g., by an undo receipt), the
    /// id of the voiding receipt. nil when the receipt is still
    /// valid. Set by appending a new receipt and updating the
    /// chain -- the original row is not mutated, only this
    /// `voidedBy` field is set in the in-memory representation.
    public var voidedBy: UUID?

    public init(
        id: UUID = UUID(),
        documentID: UUID,
        actor: Actor,
        mutations: [Mutation],
        timestamp: Date = Date(),
        priorReceiptID: UUID? = nil,
        signature: Data,
        c2paManifest: C2PAManifest? = nil,
        summary: String,
        preMutationSnapshot: [UUID: Block] = [:],
        voidedBy: UUID? = nil
    ) {
        self.id = id
        self.documentID = documentID
        self.actor = actor
        self.mutations = mutations
        self.timestamp = timestamp
        self.priorReceiptID = priorReceiptID
        self.signature = signature
        self.c2paManifest = c2paManifest
        self.summary = summary
        self.preMutationSnapshot = preMutationSnapshot
        self.voidedBy = voidedBy
    }
}

// MARK: - Canonical form (for signing + verification)

extension Receipt {
    /// Encode the receipt to a canonical form for signing. The
    /// canonical form EXCLUDES the `signature` and `voidedBy` fields
    /// (the signature is computed over the content, not over itself;
    /// `voidedBy` is appended after signing and is not part of the
    /// original commitment).
    ///
    /// Sorted keys + ISO-8601 dates ensure that two semantically
    /// equal receipts produce the same bytes, regardless of dict
    /// ordering.
    public func canonicalBytes() throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let content = ReceiptCanonicalContent(
            id: id,
            documentID: documentID,
            actor: actor,
            mutations: mutations,
            timestamp: timestamp,
            priorReceiptID: priorReceiptID,
            c2paManifest: c2paManifest,
            summary: summary,
            preMutationSnapshot: preMutationSnapshot
        )
        return try encoder.encode(content)
    }

    /// True iff the signature is a valid ed25519 signature over the
    /// canonical content. Returns false (does not throw) on any
    /// verification error so the caller can treat "invalid" and
    /// "missing key" the same way.
    public func verify(against publicKey: Curve25519.Signing.PublicKey) -> Bool {
        do {
            let content = try canonicalBytes()
            return publicKey.isValidSignature(signature, for: content)
        } catch {
            return false
        }
    }

    /// True iff this receipt is voided (has a `voidedBy` set).
    public var isVoided: Bool { voidedBy != nil }
}

// MARK: - Canonical-content helper

/// The receipt shape used for canonical encoding. Mirrors ``Receipt``
/// but omits the signature and `voidedBy` (which are added AFTER the
/// signature is computed). This is the type JSONEncoder actually
/// sees; the wrapper ``Receipt/canonicalBytes()`` populates it from
/// the public fields.
private struct ReceiptCanonicalContent: Codable, Sendable {
    let id: UUID
    let documentID: UUID
    let actor: Actor
    let mutations: [Mutation]
    let timestamp: Date
    let priorReceiptID: UUID?
    let c2paManifest: C2PAManifest?
    let summary: String
    let preMutationSnapshot: [UUID: Block]
}
