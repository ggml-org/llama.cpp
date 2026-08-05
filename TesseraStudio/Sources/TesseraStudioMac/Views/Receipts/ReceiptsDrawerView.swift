import SwiftUI
import TesseraCore

// MARK: - ReceiptsDrawerView

/// The macOS receipts drawer (per spec §7.3). A
/// right-side inspector pane with three tabs:
///   * "This document" — the receipt chain for the open
///     document, newest first.
///   * "All documents" — the same chain view but across
///     all documents, filterable.
///   * "Export" — the export UI.
///
/// **Always available.** The drawer is always rendered
/// (visibility toggled by Cmd-Option-2). The SwiftUI
/// `NavigationSplitView` column that hosts the drawer
/// keeps the column around even when hidden; toggling
/// just animates the column's width.
///
/// **Coordination.** The drawer reads its focus from
/// `ReceiptsCoordinatorBridge` and forwards user actions
/// (open receipt, show in chat, show in graph) through
/// the coordinator. The chat panel observes the same
/// bridge and updates its highlight.
public struct ReceiptsDrawerView: View {

    public let documentID: UUID
    public let documentTitle: String
    public let documentStore: DocumentStore
    public let service: ReceiptExportService
    public let userID: UserID
    @ObservedObject public var bridge: ReceiptsCoordinatorBridge

    @State private var selectedTab: Tab = .thisDocument
    @State private var receipts: [Receipt] = []
    @State private var allDocuments: [AllDocumentsEntry] = []
    @State private var allFilter = AllDocumentsFilter()
    @State private var selectedReceipt: Receipt?
    @State private var isLoading: Bool = false
    @State private var errorMessage: String?
    @State private var signer: ReceiptSigner = ReceiptSigner()

    public enum Tab: String, CaseIterable, Identifiable, Sendable {
        case thisDocument = "This document"
        case allDocuments = "All documents"
        case export = "Export"
        public var id: String { rawValue }
    }

    public init(
        documentID: UUID,
        documentTitle: String,
        documentStore: DocumentStore,
        service: ReceiptExportService,
        userID: UserID,
        bridge: ReceiptsCoordinatorBridge
    ) {
        self.documentID = documentID
        self.documentTitle = documentTitle
        self.documentStore = documentStore
        self.service = service
        self.userID = userID
        self.bridge = bridge
    }

    public var body: some View {
        VStack(spacing: 0) {
            tabBar
            Divider()
            content
        }
        .frame(minWidth: 280, idealWidth: 360, maxWidth: 520, maxHeight: .infinity)
        .background(Color(NSColor.textBackgroundColor).opacity(0.5))
        .onAppear {
            Task {
                await loadReceipts()
                await bridge.refresh()
            }
        }
        .onChange(of: bridge.focus) { _, newFocus in
            if case .receipt(let id) = newFocus {
                if let match = receipts.first(where: { $0.id == id }) {
                    selectedReceipt = match
                    selectedTab = .thisDocument
                }
            }
        }
    }

    // MARK: - Tab bar

    private var tabBar: some View {
        HStack(spacing: 0) {
            ForEach(Tab.allCases) { tab in
                Button {
                    selectedTab = tab
                } label: {
                    VStack(spacing: 2) {
                        Text(tab.rawValue)
                            .font(.system(size: 11, weight: selectedTab == tab ? .semibold : .regular))
                            .foregroundStyle(selectedTab == tab ? Color.primary : .secondary)
                        Rectangle()
                            .fill(selectedTab == tab ? Color.accentColor : .clear)
                            .frame(height: 2)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
            Spacer()
        }
        .background(.bar)
    }

    // MARK: - Content

    @ViewBuilder
    private var content: some View {
        switch selectedTab {
        case .thisDocument:
            thisDocumentTab
        case .allDocuments:
            allDocumentsTab
        case .export:
            ReceiptExportView(
                documentID: documentID,
                documentTitle: documentTitle,
                service: service,
                userID: userID,
                onExported: { _ in
                    Task { await loadReceipts() }
                }
            )
        }
    }

    private var thisDocumentTab: some View {
        VStack(spacing: 0) {
            if isLoading {
                ProgressView()
                    .controlSize(.small)
                    .padding(20)
            } else if let err = errorMessage {
                Text(err)
                    .font(.system(size: 11))
                    .foregroundStyle(.red)
                    .padding()
            } else {
                chainList
            }
        }
    }

    private var chainList: some View {
        HSplitView {
            receiptList
                .frame(minWidth: 200, idealWidth: 240)
            if let selected = selectedReceipt {
                ReceiptDetailView(
                    receipt: selected,
                    documentTitle: documentTitle,
                    signer: signer,
                    onShowInChat: {
                        Task {
                            _ = await bridge.showInChat(receiptID: selected.id)
                        }
                    },
                    onShowInGraph: {
                        Task {
                            // The graph entity id is the
                            // document's id for v1. Phase 6
                            // navigates to a more specific
                            // entity.
                            await bridge.showInGraph(entityID: documentID)
                        }
                    },
                    onClose: {
                        selectedReceipt = nil
                    }
                )
                .frame(minWidth: 280)
            } else {
                Color.clear
            }
        }
    }

    private var receiptList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 0) {
                if receipts.isEmpty {
                    Text("No receipts yet")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                        .padding()
                } else {
                    ForEach(receipts) { r in
                        ReceiptRowView(receipt: r, isSelected: selectedReceipt?.id == r.id)
                            .onTapGesture {
                                selectedReceipt = r
                            }
                    }
                }
            }
        }
    }

    private var allDocumentsTab: some View {
        VStack(alignment: .leading, spacing: 0) {
            allDocumentsFilter
            Divider()
            allDocumentsList
        }
    }

    private var allDocumentsFilter: some View {
        HStack(spacing: 6) {
            Picker("Date", selection: $allFilter.dateRange) {
                ForEach(AllDocumentsFilter.DateRange.allCases) { r in
                    Text(r.displayName).tag(r)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 110)
            Picker("Actor", selection: $allFilter.actorKind) {
                ForEach(AllDocumentsFilter.ActorKind.allCases) { k in
                    Text(k.displayName).tag(k)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 110)
            Spacer()
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    private var allDocumentsList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 0) {
                let filtered = allFilter.apply(to: allDocuments)
                if filtered.isEmpty {
                    Text("No matching receipts")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                        .padding()
                } else {
                    ForEach(filtered) { entry in
                        Button {
                            // v1: only this document's
                            // receipts are openable from the
                            // all-documents tab. Cross-doc
                            // navigation is a Phase 6 surface.
                            if entry.documentID == documentID {
                                selectedTab = .thisDocument
                                selectedReceipt = entry.receipt
                            }
                        } label: {
                            allDocumentsRow(entry)
                        }
                        .buttonStyle(.plain)
                        .disabled(entry.documentID != documentID)
                    }
                }
            }
        }
    }

    private func allDocumentsRow(_ entry: AllDocumentsEntry) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: entry.documentID == documentID ? "doc.text" : "doc")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.receipt.summary)
                    .font(.system(size: 11, weight: .medium))
                    .lineLimit(1)
                Text("\(entry.documentTitle) · \(entry.timestampText)")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 8)
        .contentShape(Rectangle())
    }

    // MARK: - Loading

    private func loadReceipts() async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }
        do {
            receipts = try await documentStore.history(of: documentID, limit: 200)
            allDocuments = receipts.map { r in
                AllDocumentsEntry(
                    documentID: documentID,
                    documentTitle: documentTitle,
                    receipt: r
                )
            }
        } catch {
            errorMessage = "Failed to load receipts: \(error)"
        }
    }
}

// MARK: - All documents filter

struct AllDocumentsEntry: Identifiable, Hashable {
    let documentID: UUID
    let documentTitle: String
    let receipt: Receipt
    var id: UUID { receipt.id }
    var timestamp: Date { receipt.timestamp }
    var timestampText: String {
        let f = DateFormatter()
        f.dateStyle = .short
        f.timeStyle = .short
        return f.string(from: receipt.timestamp)
    }
}

struct AllDocumentsFilter: Hashable {
    enum DateRange: String, CaseIterable, Identifiable, Hashable {
        case all
        case last24h
        case last7d
        case last30d
        var id: String { rawValue }
        var displayName: String {
            switch self {
            case .all: return "All time"
            case .last24h: return "Last 24h"
            case .last7d: return "Last 7d"
            case .last30d: return "Last 30d"
            }
        }
    }
    enum ActorKind: String, CaseIterable, Identifiable, Hashable {
        case all
        case user
        case agent
        var id: String { rawValue }
        var displayName: String {
            switch self {
            case .all: return "All actors"
            case .user: return "User only"
            case .agent: return "Agent only"
            }
        }
    }
    var dateRange: DateRange = .all
    var actorKind: ActorKind = .all

    func apply(to entries: [AllDocumentsEntry]) -> [AllDocumentsEntry] {
        let cutoff: Date
        let now = Date()
        switch dateRange {
        case .all: cutoff = .distantPast
        case .last24h: cutoff = now.addingTimeInterval(-86400)
        case .last7d: cutoff = now.addingTimeInterval(-604800)
        case .last30d: cutoff = now.addingTimeInterval(-2592000)
        }
        return entries.filter { entry in
            guard entry.timestamp >= cutoff else { return false }
            switch actorKind {
            case .all: return true
            case .user:
                if case .user = entry.receipt.actor { return true }
                return false
            case .agent:
                if case .agent = entry.receipt.actor { return true }
                return false
            }
        }
    }
}
