import SwiftUI
import SwiftData

/// Date-range filter for the history drawer.
public enum HistoryDateFilter: String, CaseIterable, Sendable {
    case all = "All"
    case today = "Today"
    case week = "This Week"
    case month = "This Month"

    public func contains(_ date: Date) -> Bool {
        let cal = Calendar.current
        switch self {
        case .all: return true
        case .today: return cal.isDateInToday(date)
        case .week: return cal.isDate(date, equalTo: Date(), toGranularity: .weekOfYear)
        case .month: return cal.isDate(date, equalTo: Date(), toGranularity: .month)
        }
    }
}

/// Slide-out leading drawer listing past conversations from SwiftData.
/// Search/filter by text, date, model name, and tool used. Tap a row to
/// restore it; swipe to delete; context menu for rename/export.
public struct ChatHistoryDrawer: View {
    @Environment(\.modelContext) private var modelContext

    @Query(sort: [SortDescriptor(\Conversation.updatedAt, order: .reverse)])
    private var conversations: [Conversation]

    @Binding var isPresented: Bool
    public var onRestore: (Conversation) -> Void
    public var onExport: (Conversation, ExportFormat) -> Void

    @State private var searchText = ""
    @State private var dateFilter: HistoryDateFilter = .all
    @State private var renaming: Conversation?
    @State private var renameText = ""
    @State private var confirmingDelete: Conversation?

    public init(
        isPresented: Binding<Bool>,
        onRestore: @escaping (Conversation) -> Void,
        onExport: @escaping (Conversation, ExportFormat) -> Void
    ) {
        self._isPresented = isPresented
        self.onRestore = onRestore
        self.onExport = onExport
    }

    private var filtered: [Conversation] {
        conversations.filter { convo in
            guard dateFilter.contains(convo.updatedAt) else { return false }
            guard !searchText.isEmpty else { return true }
            let q = searchText.localizedLowercase
            return convo.title.localizedLowercase.contains(q)
                || convo.modelName.localizedLowercase.contains(q)
                || convo.toolNames.contains { $0.localizedLowercase.contains(q) }
        }
    }

    public var body: some View {
        VStack(spacing: 0) {
            header
            filters
            Divider()
            list
        }
        .frame(minWidth: 260)
        .background(.bar)
        .alert("Rename Conversation", isPresented: Binding(
            get: { renaming != nil },
            set: { if !$0 { renaming = nil } }
        )) {
            TextField("Title", text: $renameText)
            Button("Cancel", role: .cancel) { renaming = nil }
            Button("Save") { commitRename() }
        }
        // HIG 14.1 / 13.5: deleting a conversation is irreversible
        // user data, so both delete entry points route through this
        // confirmation instead of firing directly.
        .confirmationDialog("Delete Conversation?", isPresented: Binding(
            get: { confirmingDelete != nil },
            set: { if !$0 { confirmingDelete = nil } }
        ), titleVisibility: .visible) {
            Button("Delete", role: .destructive) {
                if let convo = confirmingDelete {
                    delete(convo)
                }
                confirmingDelete = nil
            }
        } message: {
            if let convo = confirmingDelete {
                Text("\"\(convo.title)\" and its messages will be permanently removed.")
            }
        }
    }

    private var header: some View {
        HStack {
            Text("History")
                .font(.headline)
            Spacer()
            Button(action: { withAnimation { isPresented = false } }) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding()
    }

    private var filters: some View {
        VStack(spacing: 8) {
            TextField("Search title, model, tool", text: $searchText)
                .textFieldStyle(.roundedBorder)
            Picker("Date", selection: $dateFilter) {
                ForEach(HistoryDateFilter.allCases, id: \.self) { filter in
                    Text(filter.rawValue).tag(filter)
                }
            }
            .pickerStyle(.segmented)
        }
        .padding(.horizontal)
        .padding(.bottom, 8)
    }

    private var list: some View {
        Group {
            if filtered.isEmpty {
                ContentUnavailableView(
                    "No Conversations",
                    systemImage: "bubble.left.and.text.bubble.right",
                    description: Text("Past chats appear here.")
                )
            } else {
                List(filtered) { convo in
                    row(convo)
                        .contentShape(Rectangle())
                        .onTapGesture { onRestore(convo) }
                        .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                            Button(role: .destructive) {
                                confirmingDelete = convo
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                            Button {
                                beginRename(convo)
                            } label: {
                                Label("Rename", systemImage: "pencil")
                            }
                            .tint(.blue)
                        }
                        .contextMenu {
                            Button("Rename") { beginRename(convo) }
                            Menu("Export") {
                                Button("Markdown") { onExport(convo, .markdown) }
                                Button("JSON") { onExport(convo, .json) }
                            }
                            Divider()
                            Button("Delete", role: .destructive) { confirmingDelete = convo }
                        }
                }
            }
        }
    }

    private func row(_ convo: Conversation) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(convo.title)
                .font(.subheadline.weight(.medium))
                .lineLimit(1)
            HStack(spacing: 6) {
                if !convo.modelName.isEmpty {
                    Text(convo.modelName)
                        .font(.caption2)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(.quaternary, in: Capsule())
                }
                Text(convo.updatedAt, style: .relative)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                if !convo.toolNames.isEmpty {
                    Text("· \(convo.toolNames.count) tools")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
        }
        .padding(.vertical, 2)
    }

    private func beginRename(_ convo: Conversation) {
        renameText = convo.title
        renaming = convo
    }

    private func commitRename() {
        guard let convo = renaming else { return }
        let trimmed = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            convo.title = trimmed
            convo.updatedAt = Date()
        }
        renaming = nil
    }

    private func delete(_ convo: Conversation) {
        ConversationStore.deleteConversation(convo, in: modelContext)
    }
}

/// Helpers for reading and deleting conversations and their messages.
public enum ConversationStore {
    /// Fetch the messages for a conversation, ordered by timestamp.
    public static func messages(for conversationID: UUID, in context: ModelContext) -> [ChatMessage] {
        let descriptor = FetchDescriptor<ChatMessage>(
            predicate: #Predicate { $0.conversationID == conversationID },
            sortBy: [SortDescriptor(\.timestamp)]
        )
        return (try? context.fetch(descriptor)) ?? []
    }

    /// Delete a conversation and all of its messages.
    public static func deleteConversation(_ convo: Conversation, in context: ModelContext) {
        for message in messages(for: convo.id, in: context) {
            context.delete(message)
        }
        context.delete(convo)
    }
}
