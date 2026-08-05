import SwiftUI
import CryptoKit
import TesseraCore

// MARK: - ReceiptDetailView

/// The receipt details view (per spec §7.4). Five
/// sections: header, mutations, diff, signature, C2PA.
/// Tapping "Show in chat" calls the coordinator; tapping
/// "Show in graph" navigates to the Graph view (Phase 6
/// dependency; the navigation is wired but the Graph
/// surface itself is a later phase).
public struct ReceiptDetailView: View {

    public let receipt: Receipt
    public let documentTitle: String
    public let signer: ReceiptSigner
    public let onShowInChat: () -> Void
    public let onShowInGraph: () -> Void
    public let onClose: () -> Void

    @State private var verificationResult: VerificationDisplay?
    @State private var c2paSheetShown: Bool = false

    public init(
        receipt: Receipt,
        documentTitle: String,
        signer: ReceiptSigner,
        onShowInChat: @escaping () -> Void,
        onShowInGraph: @escaping () -> Void,
        onClose: @escaping () -> Void
    ) {
        self.receipt = receipt
        self.documentTitle = documentTitle
        self.signer = signer
        self.onShowInChat = onShowInChat
        self.onShowInGraph = onShowInGraph
        self.onClose = onClose
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                headerSection
                Divider()
                mutationsSection
                Divider()
                diffSection
                Divider()
                signatureSection
                Divider()
                c2paSection
                Spacer(minLength: 16)
            }
            .padding(16)
        }
        .frame(minWidth: 320, idealWidth: 380)
        .background(Color(NSColor.textBackgroundColor))
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button(action: onClose) {
                    Image(systemName: "xmark")
                }
                .help("Close receipt")
            }
        }
        .sheet(isPresented: $c2paSheetShown) {
            if let manifest = receipt.c2paManifest {
                C2PAManifestSheet(manifest: manifest)
            }
        }
    }

    // MARK: - Sections

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(receipt.summary)
                    .font(.system(size: 14, weight: .semibold))
                    .lineLimit(2)
                Spacer()
            }
            HStack(spacing: 6) {
                Image(systemName: actorIcon)
                    .font(.system(size: 11))
                    .foregroundStyle(actorTint)
                Text(actorLabel)
                    .font(.system(size: 11))
                Text("·")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
                Text(timestampText(receipt.timestamp))
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
            Text("Document: \(documentTitle)")
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
            HStack(spacing: 12) {
                Button {
                    onShowInChat()
                } label: {
                    Label("Show in chat", systemImage: "bubble.left")
                }
                .buttonStyle(.borderless)
                .font(.system(size: 11))

                Button {
                    onShowInGraph()
                } label: {
                    Label("Show in graph", systemImage: "rectangle.connected.to.line.below")
                }
                .buttonStyle(.borderless)
                .font(.system(size: 11))
            }
        }
    }

    private var mutationsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionHeader("Mutations", count: receipt.mutations.count)
            if receipt.mutations.isEmpty {
                Text("No mutations (e.g., this is an export receipt).")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 2)
            } else {
                ForEach(Array(receipt.mutations.enumerated()), id: \.offset) { idx, mutation in
                    HStack(alignment: .top, spacing: 4) {
                        Text("\(idx + 1).")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(.secondary)
                        Text(mutation.shortDescription)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(.primary)
                    }
                }
            }
        }
    }

    private var diffSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionHeader("Diff", count: nil)
            if receipt.mutations.isEmpty {
                Text("No content changes.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            } else {
                ReceiptDiffView(
                    receipt: receipt
                )
            }
        }
    }

    private var signatureSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionHeader("Signature", count: nil)
            Text("ed25519: " + receipt.signature.prefix(16).map { String(format: "%02x", $0) }.joined() + "…")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
            HStack {
                Button {
                    runVerification()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "checkmark.shield")
                            .font(.system(size: 11))
                        Text("Verify")
                            .font(.system(size: 11))
                    }
                }
                if let result = verificationResult {
                    HStack(spacing: 4) {
                        Image(systemName: result.icon)
                            .foregroundStyle(result.tint)
                        Text(result.label)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(result.tint)
                    }
                }
            }
        }
    }

    private var c2paSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionHeader("C2PA manifest", count: nil)
            if let manifest = receipt.c2paManifest {
                HStack(spacing: 6) {
                    Text(manifest.format)
                        .font(.system(size: 10, design: .monospaced))
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(Capsule().fill(Color.purple.opacity(0.15)))
                        .foregroundStyle(.purple)
                    Text(manifest.claimGenerator)
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button {
                        c2paSheetShown = true
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "doc.text.magnifyingglass")
                                .font(.system(size: 10))
                            Text("View")
                                .font(.system(size: 10))
                        }
                    }
                    .buttonStyle(.borderless)
                }
                Text("\(manifest.assertions.count) assertion\(manifest.assertions.count == 1 ? "" : "s")")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            } else {
                Text("No C2PA manifest.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Helpers

    private func sectionHeader(_ title: String, count: Int?) -> some View {
        HStack {
            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            if let count {
                Text("\(count)")
                    .font(.system(size: 10))
                    .padding(.horizontal, 4)
                    .padding(.vertical, 1)
                    .background(Capsule().fill(Color.secondary.opacity(0.12)))
            }
            Spacer()
        }
    }

    private func runVerification() {
        guard let publicKey = signer.publicKey else {
            verificationResult = .init(
                label: "no public key",
                icon: "questionmark.circle",
                tint: .secondary
            )
            return
        }
        let result = signer.verify(receipt, against: publicKey)
        switch result {
        case .valid:
            verificationResult = .init(
                label: "valid",
                icon: "checkmark.circle.fill",
                tint: .green
            )
        case .invalid:
            verificationResult = .init(
                label: "invalid",
                icon: "xmark.octagon.fill",
                tint: .red
            )
        case .voided:
            verificationResult = .init(
                label: "voided",
                icon: "exclamationmark.triangle.fill",
                tint: .orange
            )
        }
    }

    private var actorIcon: String {
        switch receipt.actor {
        case .user: return "person.fill"
        case .agent: return "cpu"
        }
    }

    private var actorTint: Color {
        switch receipt.actor {
        case .user: return .accentColor
        case .agent: return .purple
        }
    }

    private var actorLabel: String {
        switch receipt.actor {
        case .user(let id): return "user \(id.uuidString.prefix(8))"
        case .agent(_, let model, let promptHash):
            return "agent · \(model) · \(promptHash.prefix(8))"
        }
    }

    private func timestampText(_ date: Date) -> String {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .medium
        return f.string(from: date)
    }
}

private struct VerificationDisplay: Sendable {
    let label: String
    let icon: String
    let tint: Color
}

// MARK: - ReceiptDiffView

/// A simple before/after diff of the receipt's affected
/// blocks. The receipt's `preMutationSnapshot` carries
/// the pre-state; the live document is queried for the
/// post-state when the detail view renders. The diff is
/// text-only (the spec calls for red strikethrough for
/// deletions and green underline for additions).
public struct ReceiptDiffView: View {

    public let receipt: Receipt
    public let postDocument: DocumentAST?

    public init(receipt: Receipt, postDocument: DocumentAST? = nil) {
        self.receipt = receipt
        self.postDocument = postDocument
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(diffEntries(), id: \.id) { entry in
                diffLine(entry)
            }
            if diffEntries().isEmpty {
                Text("No content changes.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
        }
    }

    private struct DiffEntry: Identifiable {
        let id: String
        let kind: Kind
        let text: String
        enum Kind { case same, deletion, addition }
    }

    private func diffEntries() -> [DiffEntry] {
        // For each affected block, compare the pre and
        // post flattened text. v1 uses a simple
        // line-by-line longest-common-subsequence.
        var entries: [DiffEntry] = []
        for mutation in receipt.mutations {
            let (blockID, _) = extractBlockInfo(mutation)
            guard let blockID else { continue }
            let preText = receipt.preMutationSnapshot[blockID].map(flattenBlock) ?? ""
            let postText = postDocument?.blocks[blockID].map(flattenBlock) ?? ""
            let lines = lineDiff(pre: preText, post: postText)
            for line in lines {
                entries.append(line)
            }
        }
        return entries
    }

    private func extractBlockInfo(_ mutation: Mutation) -> (UUID?, String?) {
        switch mutation {
        case .setBlockContent(let id, _),
             .setBlockAttribute(let id, _, _),
             .appendInlineRun(let id, _),
             .replaceInlineRun(let id, _, _),
             .deleteInlineRun(let id, _),
             .replaceBlock(let id, _),
             .deleteBlock(let id):
            return (id, nil)
        default:
            return (nil, nil)
        }
    }

    private func flattenBlock(_ block: Block) -> String {
        block.content.map { $0.text }.joined()
    }

    private func lineDiff(pre: String, post: String) -> [DiffEntry] {
        let preLines = pre.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        let postLines = post.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        var result: [DiffEntry] = []
        // Naive LCS-based diff: walk both lists, mark
        // matches and edits.
        let m = preLines.count
        let n = postLines.count
        var i = 0, j = 0
        while i < m || j < n {
            if i < m && j < n && preLines[i] == postLines[j] {
                result.append(DiffEntry(id: "same-\(i)-\(j)", kind: .same, text: preLines[i]))
                i += 1
                j += 1
            } else if j < n && (i >= m || preLines[i] != postLines[j]) {
                result.append(DiffEntry(id: "add-\(j)", kind: .addition, text: postLines[j]))
                j += 1
            } else if i < m {
                result.append(DiffEntry(id: "del-\(i)", kind: .deletion, text: preLines[i]))
                i += 1
            }
        }
        return result
    }

    @ViewBuilder
    private func diffLine(_ entry: DiffEntry) -> some View {
        switch entry.kind {
        case .same:
            Text(entry.text)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.primary)
        case .deletion:
            Text(entry.text)
                .font(.system(size: 11, design: .monospaced))
                .strikethrough(true, color: .red)
                .foregroundStyle(.red)
        case .addition:
            Text(entry.text)
                .font(.system(size: 11, design: .monospaced))
                .underline(true, color: .green)
                .foregroundStyle(.green)
        }
    }
}
