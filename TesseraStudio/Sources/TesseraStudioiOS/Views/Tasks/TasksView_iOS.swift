#if os(iOS)
import SwiftUI
import UIKit
import TesseraCore

// MARK: - TasksView_iOS

/// The iOS Tasks surface. The layout is a `NavigationStack`
/// with the five lists in a horizontal scrollable tab strip
/// at the top, the tasks in a `List`, and a navigation push
/// to the task detail. The NLU input is a sheet (a single
/// text field with a "Add" button) so the user can type a
/// free-form task without leaving the list.
public struct TasksView_iOS: View {

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
    @State private var isLoading: Bool = false
    @State private var loadError: String?
    @State private var showInputSheet: Bool = false

    public var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                listTabStrip
                Divider()
                tasksList
            }
            .navigationTitle("Tasks")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showInputSheet = true
                    } label: {
                        Image(systemName: "plus")
                    }
                }
            }
            .sheet(isPresented: $showInputSheet) {
                TaskInputSheet(
                    store: store,
                    userID: userID,
                    contacts: contacts,
                    documents: documents,
                    onAdded: { Task { await load() } }
                )
            }
            .onAppear {
                if allTasks.isEmpty && !isLoading {
                    Task { await load() }
                }
            }
        }
    }

    // MARK: - Tab strip

    private var listTabStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(ProductivityTask.List.allCases, id: \.self) { list in
                    Button {
                        selectedList = list
                    } label: {
                        VStack(spacing: 4) {
                            HStack(spacing: 4) {
                                Image(systemName: list.systemImageName)
                                Text(list.displayName)
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(
                                selectedList == list
                                    ? Color.accentColor.opacity(0.15)
                                    : Color.clear,
                                in: Capsule()
                            )
                            .foregroundStyle(
                                selectedList == list ? Color.accentColor : .primary
                            )
                            Text("\(count(in: list))")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
    }

    // MARK: - Tasks list

    private var tasksList: some View {
        List {
            ForEach(filteredTasks) { task in
                NavigationLink(value: task) {
                    TaskRow_iOS(task: task)
                }
                .swipeActions(edge: .trailing) {
                    Button(role: .destructive) {
                        Task {
                            _ = try? await store.delete(id: task.id, actor: .user(userID))
                            await load()
                        }
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                    Button {
                        Task {
                            _ = try? await store.complete(id: task.id, actor: .user(userID))
                            await load()
                        }
                    } label: {
                        Label("Complete", systemImage: "checkmark")
                    }
                    .tint(.green)
                }
            }
        }
        .navigationDestination(for: ProductivityTask.self) { task in
            TaskDetailView_iOS(task: task, store: store, userID: userID)
        }
        .overlay {
            if isLoading {
                ProgressView()
            } else if filteredTasks.isEmpty {
                ContentUnavailableView(
                    "No tasks in \(selectedList.displayName)",
                    systemImage: "checkmark.square",
                    description: Text("Tap + to add one.")
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

    // MARK: - Data

    private var filteredTasks: [ProductivityTask] {
        let byList = allTasks.filter { task in
            ProductivityTaskFilter.isIn(task, list: selectedList, asOf: Date())
        }
        return ProductivityTaskFilter.sortForList(byList, list: selectedList)
    }

    private func count(in list: ProductivityTask.List) -> Int {
        allTasks.filter { ProductivityTaskFilter.isIn($0, list: list, asOf: Date()) }
            .filter { !$0.isCompleted }
            .count
    }

    private func load() async {
        isLoading = true
        loadError = nil
        defer { isLoading = false }
        do {
            allTasks = try await store.list(limit: 1000)
        } catch {
            loadError = String(describing: error)
        }
    }
}

// MARK: - Row

private struct TaskRow_iOS: View {
    let task: ProductivityTask

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            Image(systemName: task.isCompleted ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(task.isCompleted ? .green : .secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text(task.title)
                    .strikethrough(task.isCompleted)
                if let due = task.dueAt, !task.isCompleted {
                    Text(due.formatted(date: .abbreviated, time: .omitted))
                        .font(.caption)
                        .foregroundStyle(due < Date() ? .red : .secondary)
                }
            }
            Spacer()
            if task.priority != .none {
                Image(systemName: task.prioritySystemImageName)
                    .foregroundStyle(priorityColor)
            }
        }
    }

    private var priorityColor: Color {
        switch task.priority {
        case .none: return .secondary
        case .low: return .blue
        case .medium: return .orange
        case .high: return .red
        }
    }
}

// MARK: - Input sheet

private struct TaskInputSheet: View {
    let store: ProductivityTaskStore
    let userID: UserID
    let contacts: ContactsAdapter?
    let documents: DocumentStoreNLU?
    let onAdded: () -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var text: String = ""
    @State private var isWorking: Bool = false
    @State private var error: String?

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                Text("Type a task — try \"tomorrow at 3pm, call John\".")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextEditor(text: $text)
                    .border(.separator, width: 1)
                if let error {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
                Spacer()
            }
            .padding()
            .navigationTitle("New task")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Add") { submit() }
                        .disabled(text.trimmingCharacters(in: .whitespaces).isEmpty || isWorking)
                }
            }
        }
    }

    private func submit() {
        let raw = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return }
        let parser = ProductivityTaskNLUParser(contacts: contacts, documents: documents)
        let parsed = parser.parse(raw)
        let task = parsed.toTask()
        isWorking = true
        Task {
            do {
                _ = try await store.upsert(task, actor: .user(userID))
                isWorking = false
                onAdded()
                dismiss()
            } catch {
                self.error = String(describing: error)
                isWorking = false
            }
        }
    }
}

// MARK: - Detail

private struct TaskDetailView_iOS: View {
    let task: ProductivityTask
    let store: ProductivityTaskStore
    let userID: UserID

    var body: some View {
        List {
            Section {
                Text(task.title)
                    .font(.title3)
                HStack {
                    Image(systemName: task.list.systemImageName)
                    Text(task.list.displayName)
                    if let due = task.dueAt, !task.isCompleted {
                        Spacer()
                        Text(due.formatted(date: .abbreviated, time: .shortened))
                            .foregroundStyle(due < Date() ? .red : .secondary)
                    }
                }
                if !task.notes.isEmpty {
                    Text(task.notes)
                }
            }
            Section("Priority") {
                Picker("Priority", selection: priorityBinding) {
                    ForEach(ProductivityTask.Priority.allCases, id: \.self) { p in
                        Text(p.displayName).tag(p)
                    }
                }
                .pickerStyle(.segmented)
            }
            Section("List") {
                Picker("List", selection: listBinding) {
                    ForEach(ProductivityTask.List.allCases, id: \.self) { l in
                        Text(l.displayName).tag(l)
                    }
                }
            }
        }
        .navigationTitle("Task")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var priorityBinding: Binding<ProductivityTask.Priority> {
        Binding(
            get: { task.priority },
            set: { newValue in
                Task {
                    _ = try? await store.setPriority(id: task.id, to: newValue, actor: .user(userID))
                }
            }
        )
    }

    private var listBinding: Binding<ProductivityTask.List> {
        Binding(
            get: { task.list },
            set: { newValue in
                Task {
                    _ = try? await store.move(id: task.id, to: newValue, actor: .user(userID))
                }
            }
        )
    }
}
#endif
