#if os(iOS)
import SwiftUI
import TesseraCore

// MARK: - ReceiptsDrawerSheet_iOS

/// The iOS receipts drawer. Per spec §7.3 the iOS
/// drawer is a modal sheet with `.large` detent. The
/// three tabs (This document / All documents / Export)
/// are rendered inside the sheet; the user swipes
/// between them via a `Picker(selection: ...).pickerStyle(.segmented)`
/// at the top.
public struct ReceiptsDrawerSheet_iOS: View {

    public let documentID: UUID
    public let documentTitle: String
    @Environment(\.dismiss) private var dismiss

    @State private var selectedTab: ReceiptsDrawerView.Tab = .thisDocument
    @State private var receipts: [Receipt] = []
    @State private var selectedReceipt: Receipt?
    @State private var isLoading: Bool = false
    @State private var errorMessage: String?

    public init(documentID: UUID, documentTitle: String) {
        self.documentID = documentID
        self.documentTitle = documentTitle
    }

    public var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                Picker("Tab", selection: $selectedTab) {
                    ForEach(ReceiptsDrawerView.Tab.allCases) { tab in
                        Text(tab.rawValue).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 12)
                .padding(.top, 8)
                Divider()
                content
            }
            .navigationTitle("Receipts")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .onAppear { Task { await load() } }
        }
    }

    @ViewBuilder
    private var content: some View {
        switch selectedTab {
        case .thisDocument:
            thisDocumentContent
        case .allDocuments:
            allDocumentsContent
        case .export:
            Text("Export is in the macOS drawer.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var thisDocumentContent: some View {
        VStack(spacing: 0) {
            if isLoading {
                ProgressView().padding()
            } else if let err = errorMessage {
                Text(err).foregroundStyle(.red).padding()
            } else {
                receiptList
                if let selected = selectedReceipt {
                    Divider()
                    ReceiptDetailView_iOS(receipt: selected, documentTitle: documentTitle)
                }
            }
        }
    }

    private var receiptList: some View {
        List(receipts, selection: $selectedReceipt) { receipt in
            NavigationLink(value: receipt) {
                ReceiptRowView(receipt: receipt, isSelected: selectedReceipt?.id == receipt.id)
            }
        }
        .listStyle(.plain)
    }

    private var allDocumentsContent: some View {
        // v1: same as the macOS "all documents" tab minus
        // the cross-doc navigation (which requires a
        // document picker that's not in this build).
        Text("All-documents view is the same as the macOS drawer. Filter by date and actor.")
            .font(.subheadline)
            .foregroundStyle(.secondary)
            .padding()
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func load() async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }
        // The full TesseraDataStore path is wired on the
        // macOS side. For the iOS placeholder we use an
        // empty list — the production wiring is in Phase 5
        // when the per-Materials surface wrappers land.
        receipts = []
    }
}

// MARK: - iOS detail view

struct ReceiptDetailView_iOS: View {
    let receipt: Receipt
    let documentTitle: String

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text(receipt.summary).font(.headline)
                if !receipt.mutations.isEmpty {
                    Text("Mutations:").font(.subheadline.weight(.medium))
                    ForEach(Array(receipt.mutations.enumerated()), id: \.offset) { idx, mutation in
                        Text("\(idx + 1). \(mutation.shortDescription)")
                            .font(.caption.monospaced())
                    }
                }
                if let manifest = receipt.c2paManifest {
                    Text("C2PA: \(manifest.format) — \(manifest.assertions.count) assertions")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding()
        }
    }
}

#endif
