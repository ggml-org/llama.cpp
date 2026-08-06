import SwiftUI
import TesseraCore

// MARK: - TasksView (macOS)

/// The macOS Tasks surface.
///
/// **Layout.** `NavigationSplitView` with three columns:
///   * Sidebar: the five lists (Inbox / Today / Upcoming /
///     Anytime / Someday) with their counts.
///   * Middle: the tasks in the selected list, with the
///     NLU input bar at the top and a per-row checkbox +
///     title + priority badge.
///   * Detail: the selected task's metadata (notes, due
///     date, priority, linked entities, receipt chain).
///
/// **Data.** The view reads from ``ProductivityTaskStore``
/// (which wraps ``TesseraDataLayer``). Mutations go
/// through the same store so every change is a
/// constitutional receipt.
///
/// **Natural language.** The input bar at the top of the
/// middle column parses free-form text via
/// ``ProductivityTaskNLUParser``; the parsed
/// ``ParsedProductivityTask`` is then routed through
/// ``ProductivityTaskStore/upsert(_:actor:)``. v1 is
/// rule-based; v2 may add LLM-based NLU behind the same
/// interface.
public struct TasksView: View {

    public init(
        store: ProductivityTaskStore,
        userID: UserID = UUID(),
        contacts: ContactsAdapter? = nil,
        documents: DocumentStoreNLU? = nil
    ) {
        self.store = store
        self.userID = userID
        self.contacts = contacts
        self.documents = documents
    }

    private let store: ProductivityTaskStore
    private let userID: UserID
    private let contacts: ContactsAdapter?
    private let documents: DocumentStoreNLU?

    @State private var allTasks: [ProductivityTask] = []
    @State private var selectedList: ProductivityTask.List = .today
    @State private var selectedTaskID: UUID?
    @State private var searchText: String = ""
    @State private var isLoading: Bool = false
    @State private var loadError: String?
    @State private var showCompleted: Bool = false

    @State private var inputText: String = ""
    @State private var isParsing: Bool = false
    @State private var lastParseError: String?

    public var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 200, ideal: 240)
        } content: {
            list
                .navigationSplitViewColumnWidth(min: 360, ideal: 480)
        } detail: {
            if let id = selectedTaskID,
               let task = allTasks.first(where: { $0.id == id }) {
                TaskDetailView(task: task, store: store, userID: userID)
            } else {
                emptyState
            }
        }
        .navigationTitle("Tasks")
        .searchable(text: $searchText, prompt: "Search tasks")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task { await load() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Reload tasks")
            }
            ToolbarItem(placement: .primaryAction) {
                Toggle("Completed", isOn: $showCompleted)
                    .help("Show completed tasks")
            }
        }
        .onAppear {
            if allTasks.isEmpty && !isLoading {
                Task { await load() }
            }
        }
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        List(selection: $selectedList) {
            Section("Smart lists") {
                ForEach([ProductivityTask.List.today, .upcoming], id: \.self) { list in
                    Label {
                        HStack {
                            Text(list.displayName)
                            Spacer()
                            Text("\(count(in: list))")
                                .foregroundStyle(.secondary)
                        }
                    } icon: {
                        Image(systemName: list.systemImageName)
                            .foregroundStyle(GraphNode.color(for: ProductivityTask.entityType))
                    }
                    .tag(list as ProductivityTask.List?)
                }
            }
            Section("Manual lists") {
                ForEach([ProductivityTask.List.inbox, .anytime, .someday], id: \.self) { list in
                    Label {
                        HStack {
                            Text(list.displayName)
                            Spacer()
                            Text("\(count(in: list))")
                                .foregroundStyle(.secondary)
                        }
                    } icon: {
                        Image(systemName: list.systemImageName)
                            .foregroundStyle(.secondary)
                    }
                    .tag(list as ProductivityTask.List?)
                }
            }
        }
        .listStyle(.sidebar)
    }

    // MARK: - List

    private var list: some View {
        VStack(spacing: 0) {
            nluInput
            Divider()
            tasksList
        }
    }

    private var nluInput: some View {
        HStack(alignment: .center, spacing: 8) {
            Image(systemName: "plus.circle")
                .font(.title3)
                .foregroundStyle(.secondary)
            TextField(
                "Type a task — try \"tomorrow at 3pm, call John\"",
                text: $inputText
            )
            .textFieldStyle(.plain)
            .onSubmit { submitInput() }
            Button("Add") { submitInput() }
                .keyboardShortcut(.defaultAction)
                .disabled(inputText.trimmingCharacters(in: .whitespaces).isEmpty || isParsing)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.background.secondary)
    }

    private var tasksList: some View {
        List(selection: $selectedTaskID) {
            ForEach(filteredTasks) { task in
                TaskRow(
                    task: task,
                    store: store,
                    userID: userID
                )
                .tag(task.id as UUID?)
                .contextMenu {
                    contextMenu(for: task)
                }
            }
        }
        .overlay {
            if isLoading {
                ProgressView().controlSize(.large)
            } else if filteredTasks.isEmpty {
                ContentUnavailableView(
                    emptyTitle,
                    systemImage: "checkmark.square",
                    description: Text(emptyDescription)
                )
            } else if let err = loadError {
                ContentUnavailableView(
                    "Couldn't load tasks",
                    systemImage: "exclamationmark.triangle",
                    description: Text(err)
                )
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "checkmark.square")
                .font(.system(size: 64))
                .foregroundStyle(.tertiary)
            Text("Select a task")
                .font(.title3)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Actions

    private func submitInput() {
        let raw = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return }
        let parser = ProductivityTaskNLUParser(contacts: contacts, documents: documents)
        let parsed = parser.parse(raw)
        let task = parsed.toTask()
        isParsing = true
        Task {
            do {
                _ = try await store.upsert(task, actor: .user(userID))
                inputText = ""
                isParsing = false
                await load()
            } catch {
                lastParseError = String(describing: error)
                isParsing = false
            }
        }
    }

    @ViewBuilder
    private func contextMenu(for task: ProductivityTask) -> some View {
        Button("Open in editor") {
            // Wired in a follow-up worker (the editor
            // surface's "open in editor" gesture is
            // covered by Phase 2's TesseraTextContentManager).
        }
        .disabled(true)
        Divider()
        Menu("Move to") {
            ForEach(ProductivityTask.List.allCases, id: \.self) { target in
                Button(target.displayName) {
                    Task {
                        _ = try? await store.move(id: task.id, to: target, actor: .user(userID))
                        await load()
                    }
                }
                .disabled(target == task.list)
            }
        }
        Menu("Priority") {
            ForEach(ProductivityTask.Priority.allCases, id: \.self) { p in
                Button(p.displayName) {
                    Task {
                        _ = try? await store.setPriority(id: task.id, to: p, actor: .user(userID))
                        await load()
                    }
                }
                .disabled(p == task.priority)
            }
        }
        Divider()
        if task.isCompleted {
            Button("Reopen") {
                Task {
                    _ = try? await store.reopen(id: task.id, actor: .user(userID))
                    await load()
                }
            }
        } else {
            Button("Mark Completed") {
                Task {
                    _ = try? await store.complete(id: task.id, actor: .user(userID))
                    await load()
                }
            }
        }
        Divider()
        Button("Delete", role: .destructive) {
            Task {
                _ = try? await store.delete(id: task.id, actor: .user(userID))
                await load()
            }
        }
    }

    // MARK: - Data

    private var filteredTasks: [ProductivityTask] {
        let byList = allTasks.filter { task in
            ProductivityTaskFilter.isIn(task, list: selectedList, asOf: Date())
        }
        let visible = byList.filter { task in
            showCompleted ? true : !task.isCompleted
        }
        let sorted = ProductivityTaskFilter.sortForList(visible, list: selectedList)
        guard !searchText.isEmpty else { return sorted }
        let q = searchText.lowercased()
        return sorted.filter { task in
            task.title.lowercased().contains(q) ||
            task.notes.lowercased().contains(q) ||
            task.tags.contains(where: { $0.lowercased().contains(q) })
        }
    }

    private func count(in list: ProductivityTask.List) -> Int {
        allTasks.filter { ProductivityTaskFilter.isIn($0, list: list, asOf: Date()) }
            .filter { !$0.isCompleted }
            .count
    }

    private var emptyTitle: String {
        switch selectedList {
        case .inbox: return "Inbox is clear"
        case .today: return "Nothing due today"
        case .upcoming: return "Nothing upcoming this week"
        case .anytime: return "No anytime tasks"
        case .someday: return "Someday is empty"
        }
    }

    private var emptyDescription: String {
        switch selectedList {
        case .inbox: return "Type a new task in the box above to add to your Inbox."
        default: return "Type a new task above — try \"tomorrow at 3pm, call John\"."
        }
    }

    private func load() async {
        isLoading = true
        loadError = nil
        defer { isLoading = false }
        do {
            let rows = try await store.list(limit: 1000)
            allTasks = rows
        } catch {
            loadError = String(describing: error)
        }
    }
}

// MARK: - TaskRow

private struct TaskRow: View {
    let task: ProductivityTask
    let store: ProductivityTaskStore
    let userID: UserID

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            Button {
                Task { await toggleCompletion() }
            } label: {
                Image(systemName: task.isCompleted ? "checkmark.circle.fill" : "circle")
                    .font(.title3)
                    .foregroundStyle(task.isCompleted ? .green : .secondary)
            }
            .buttonStyle(.plain)
            VStack(alignment: .leading, spacing: 2) {
                Text(task.title)
                    .strikethrough(task.isCompleted)
                    .font(.body)
                HStack(spacing: 8) {
                    if let due = task.dueAt {
                        Label {
                            Text(due.formatted(date: .abbreviated, time: .shortened))
                        } icon: {
                            Image(systemName: "calendar")
                        }
                        .font(.caption)
                        .foregroundStyle(due < Date() && !task.isCompleted ? .red : .secondary)
                    }
                    if !task.tags.isEmpty {
                        Text(task.tags.joined(separator: ", "))
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                    if !task.notes.isEmpty {
                        Text(task.notesPreview)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
            Spacer()
            if task.priority != .none {
                Image(systemName: task.prioritySystemImageName)
                    .font(.caption)
                    .foregroundStyle(priorityColor(task.priority))
            }
        }
        .padding(.vertical, 2)
    }

    private func priorityColor(_ p: ProductivityTask.Priority) -> Color {
        switch p {
        case .none: return .secondary
        case .low: return .blue
        case .medium: return .orange
        case .high: return .red
        }
    }

    private func toggleCompletion() async {
        if task.isCompleted {
            _ = try? await store.reopen(id: task.id, actor: .user(userID))
        } else {
            _ = try? await store.complete(id: task.id, actor: .user(userID))
        }
    }
}

// MARK: - TaskDetailView

private struct TaskDetailView: View {
    let task: ProductivityTask
    let store: ProductivityTaskStore
    let userID: UserID

    @State private var title: String = ""
    @State private var notes: String = ""
    @State private var priority: ProductivityTask.Priority = .none
    @State private var dueAt: Date = Date()
    @State private var hasDueDate: Bool = false
    @State private var receipts: [GraphReceipt] = []
    @State private var isSaving: Bool = false
    @State private var loaded: Bool = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                Divider()
                metadataEditor
                Divider()
                linkedSection
                Divider()
                receiptsSection
            }
            .padding()
        }
        .task {
            if !loaded {
                loadFromTask()
                loaded = true
                await loadReceipts()
            }
        }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task { await save() }
                } label: {
                    Text(isSaving ? "Saving…" : "Save")
                }
                .disabled(isSaving)
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            TextField("Title", text: $title)
                .font(.title2)
                .textFieldStyle(.plain)
            HStack(spacing: 8) {
                Label(task.list.displayName, systemImage: task.list.systemImageName)
                    .font(.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.quaternary, in: Capsule())
                if let due = task.dueAt, !task.isCompleted {
                    Text("Due \(due.formatted(date: .abbreviated, time: .shortened))")
                        .font(.caption)
                        .foregroundStyle(due < Date() ? .red : .secondary)
                }
                if task.isCompleted {
                    Label("Completed", systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.green)
                }
            }
        }
    }

    private var metadataEditor: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Priority")
                    .frame(width: 80, alignment: .leading)
                Picker("", selection: $priority) {
                    ForEach(ProductivityTask.Priority.allCases, id: \.self) { p in
                        Text(p.displayName).tag(p)
                    }
                }
                .pickerStyle(.segmented)
            }
            HStack {
                Text("Due")
                    .frame(width: 80, alignment: .leading)
                Toggle("Has due date", isOn: $hasDueDate)
                if hasDueDate {
                    DatePicker("", selection: $dueAt)
                        .labelsHidden()
                }
            }
            VStack(alignment: .leading) {
                Text("Notes")
                    .font(.subheadline)
                TextEditor(text: $notes)
                    .frame(minHeight: 80, maxHeight: 200)
                    .border(.separator, width: 1)
            }
        }
    }

    private var linkedSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Linked")
                .font(.subheadline)
                .fontWeight(.medium)
            if task.linkedEntityIDs.isEmpty {
                Text("No linked entities.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(task.linkedEntityIDs, id: \.self) { id in
                    Label(id.uuidString, systemImage: "link")
                        .font(.caption)
                }
            }
        }
    }

    private var receiptsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Receipts")
                .font(.subheadline)
                .fontWeight(.medium)
            if receipts.isEmpty {
                Text("No receipts yet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(receipts) { r in
                    HStack {
                        Image(systemName: "doc.text")
                            .foregroundStyle(.tertiary)
                        VStack(alignment: .leading) {
                            Text(r.receiptType)
                                .font(.caption)
                            Text(r.witnessedAt.formatted(date: .abbreviated, time: .shortened))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                    }
                }
            }
        }
    }

    private func loadFromTask() {
        title = task.title
        notes = task.notes
        priority = task.priority
        if let due = task.dueAt {
            hasDueDate = true
            dueAt = due
        } else {
            hasDueDate = false
            dueAt = Date()
        }
    }

    private func loadReceipts() async {
        do {
            receipts = try await store.receipts(forTask: task.id)
        } catch {
            receipts = []
        }
    }

    private func save() async {
        isSaving = true
        defer { isSaving = false }
        var updated = task
        updated.title = title
        updated.notes = notes
        updated.priority = priority
        updated.dueAt = hasDueDate ? dueAt : nil
        _ = try? await store.upsert(updated, actor: .user(userID))
        await loadReceipts()
    }
}
