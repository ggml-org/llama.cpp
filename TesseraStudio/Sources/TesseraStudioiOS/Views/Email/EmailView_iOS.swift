#if os(iOS)
import SwiftUI
import UniformTypeIdentifiers
import TesseraCore

/// The iOS Email surface. v1 is a thin
/// wrapper around the macOS surface's
/// helpers; the iOS-specific UX (swipe
/// gestures, sheet-based compose) is
/// implemented here. The keyboard-first
/// vocabulary (`j/k/r/R/f/a/#/s/c`) is
/// the macOS pattern; on iOS the
/// equivalent affordances are:
/// * swipe-left: archive
/// * swipe-right: mark read
/// * tap-and-hold: context menu (the
///   same options as the macOS list
///   context menu)
/// * toolbar buttons for reply / forward
/// * + for new compose
///
/// The iOS surface uses `NavigationStack`
/// (per the spec's iOS section §13.2);
/// the macOS surface uses
/// `NavigationSplitView`. The shared
/// helpers (``EmailStore``,
/// ``EmailSender``, ``EmailImporter``,
/// ``EmailComposer``) are platform-
/// neutral.
///
/// **Layout:**
/// * **NavigationStack root**: list of
///   emails for the current folder.
/// * **Push to detail**: the email's
///   full body + actions.
/// * **Sheet for compose**: same
///   ``EmailComposerSheet`` shape as
///   macOS, presented modally.
public struct EmailView_iOS: View {

    public init(
        store: EmailStore,
        sender: EmailSender,
        importer: EmailImporter,
        identity: EmailAddress
    ) {
        self.store = store
        self.sender = sender
        self.importer = importer
        self.identity = identity
    }

    private let store: EmailStore
    private let sender: EmailSender
    private let importer: EmailImporter
    private let identity: EmailAddress

    @State private var emails: [EmailMessage] = []
    @State private var selectedFolder: Folder = .inbox
    @State private var searchText: String = ""
    @State private var isLoading: Bool = false
    @State private var loadError: String?
    @State private var composerDraft: EmailComposer?
    @State private var showComposer: Bool = false

    private var filteredEmails: [EmailMessage] {
        // Folder filter.
        let inFolder = emails.filter { $0.folder == selectedFolder }
        // Search filter.
        guard !searchText.isEmpty else { return inFolder }
        let q = searchText.lowercased()
        return inFolder.filter { e in
            e.subject.lowercased().contains(q) ||
            e.from.email.lowercased().contains(q) ||
            (e.from.name ?? "").lowercased().contains(q) ||
            e.bodyPlain.lowercased().contains(q)
        }
    }

    public var body: some View {
        NavigationStack {
            list
                .navigationTitle(selectedFolder.displayName)
                .searchable(text: $searchText, prompt: "Search")
                .toolbar {
                    ToolbarItem(placement: .primaryAction) {
                        Button {
                            startNewCompose()
                        } label: {
                            Image(systemName: "square.and.pencil")
                        }
                    }
                }
                .navigationDestination(for: EmailMessage.self) { email in
                    detail(email: email)
                }
        }
        .onAppear {
            if emails.isEmpty && !isLoading {
                Task { await load() }
            }
        }
        .onChange(of: selectedFolder) { _, _ in
            Task { await load() }
        }
        .sheet(isPresented: $showComposer) {
            if let draft = composerDraft {
                EmailComposerSheet_iOS(
                    composer: draft,
                    sender: sender,
                    store: store,
                    onClose: {
                        showComposer = false
                        composerDraft = nil
                        Task { await load() }
                    }
                )
            }
        }
    }

    private var list: some View {
        List {
            Section {
                Picker("Folder", selection: $selectedFolder) {
                    ForEach([.inbox, .sent, .drafts, .archive, .trash], id: \.self) { folder in
                        Text(folder.displayName).tag(folder)
                    }
                }
                .pickerStyle(.segmented)
            }
            ForEach(filteredEmails) { email in
                NavigationLink(value: email) {
                    EmailRow_iOS(email: email)
                }
                .swipeActions(edge: .leading) {
                    Button {
                        Task { _ = try? await store.setStarred(email.id, starred: !email.isStarred); await load() }
                    } label: {
                        Image(systemName: email.isStarred ? "star.slash" : "star")
                    }
                    .tint(.yellow)
                }
                .swipeActions(edge: .trailing) {
                    Button(role: .destructive) {
                        Task { _ = try? await store.setFolder(email.id, folder: .trash); await load() }
                    } label: {
                        Label("Trash", systemImage: "trash")
                    }
                    Button {
                        Task { _ = try? await store.setFolder(email.id, folder: .archive); await load() }
                    } label: {
                        Label("Archive", systemImage: "archivebox")
                    }
                    .tint(.gray)
                }
            }
        }
        .navigationDestination(for: EmailMessage.self) { email in
            detail(email: email)
        }
        .overlay {
            if isLoading {
                ProgressView().controlSize(.large)
            } else if filteredEmails.isEmpty {
                ContentUnavailableView(
                    "No emails",
                    systemImage: "envelope",
                    description: Text("Tap + to compose, or use the toolbar to import an .eml / .mbox file.")
                )
            } else if let err = loadError {
                ContentUnavailableView(
                    "Couldn't load emails",
                    systemImage: "exclamationmark.triangle",
                    description: Text(err)
                )
            }
        }
    }

    private func detail(email: EmailMessage) -> some View {
        EmailDetailView_iOS(
            email: email,
            store: store,
            onReply: { startReply(all: false, to: email) },
            onReplyAll: { startReply(all: true, to: email) },
            onForward: { startForward(email) },
            onArchive: {
                Task { _ = try? await store.setFolder(email.id, folder: .archive); await load() }
            },
            onTrash: {
                Task { _ = try? await store.setFolder(email.id, folder: .trash); await load() }
            },
            onToggleStar: {
                Task { _ = try? await store.setStarred(email.id, starred: !email.isStarred); await load() }
            }
        )
    }

    // MARK: - Actions

    private func load() async {
        isLoading = true
        loadError = nil
        defer { isLoading = false }
        do {
            emails = try await store.list(limit: 1000)
        } catch {
            loadError = String(describing: error)
        }
    }

    private func startNewCompose() {
        composerDraft = EmailComposer(mode: .new, from: identity)
        showComposer = true
    }

    private func startReply(all: Bool, to email: EmailMessage) {
        composerDraft = EmailComposer(mode: .reply(to: email, all: all), from: identity)
        showComposer = true
    }

    private func startForward(_ email: EmailMessage) {
        composerDraft = EmailComposer(mode: .forward(email), from: identity)
        showComposer = true
    }
}

// MARK: - Row

private struct EmailRow_iOS: View {
    let email: EmailMessage

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            if email.isStarred {
                Image(systemName: "star.fill")
                    .foregroundStyle(.yellow)
                    .font(.caption)
            }
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(email.senderDisplay)
                        .font(.body)
                        .fontWeight(email.isRead ? .regular : .semibold)
                        .lineLimit(1)
                    Spacer()
                    Text(email.receivedAt, style: .date)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Text(email.displaySubject)
                    .font(.subheadline)
                    .lineLimit(1)
                Text(email.snippet)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Detail

private struct EmailDetailView_iOS: View {
    let email: EmailMessage
    let store: EmailStore
    let onReply: () -> Void
    let onReplyAll: () -> Void
    let onForward: () -> Void
    let onArchive: () -> Void
    let onTrash: () -> Void
    let onToggleStar: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(email.displaySubject).font(.title2).fontWeight(.medium)
                    HStack {
                        Text(email.senderDisplay).font(.subheadline)
                        Spacer()
                        Text(email.receivedAt, style: .date).font(.caption)
                    }
                }
                Divider()
                Text(email.bodyPlain)
                    .font(.body)
                    .textSelection(.enabled)
                if !email.attachments.isEmpty {
                    Divider()
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Attachments (\(email.attachments.count))")
                            .font(.subheadline).fontWeight(.medium)
                        ForEach(email.attachments) { a in
                            HStack {
                                Image(systemName: "paperclip")
                                Text(a.filename)
                                Spacer()
                                Text("\(a.size) bytes").font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            .padding()
        }
        .toolbar {
            ToolbarItemGroup(placement: .bottomBar) {
                Button(action: onReply) { Image(systemName: "arrowshape.turn.up.left") }
                Button(action: onForward) { Image(systemName: "arrowshape.turn.up.right") }
                Button(action: onToggleStar) {
                    Image(systemName: email.isStarred ? "star.fill" : "star")
                }
                Button(action: onArchive) { Image(systemName: "archivebox") }
                Button(action: onTrash) { Image(systemName: "trash") }
            }
        }
    }
}

// MARK: - Composer sheet (iOS variant)

/// The iOS composer sheet. The
/// implementation mirrors the macOS
/// ``EmailComposerSheet``; v1 keeps them
/// as parallel implementations to avoid
/// platform-specific view modifiers. A
/// follow-up could share a single sheet
/// view via ``ViewThatFits``.
private struct EmailComposerSheet_iOS: View {
    let composer: EmailComposer
    let sender: EmailSender
    let store: EmailStore
    let onClose: () -> Void

    @State private var bodyText: String
    @State private var toText: String
    @State private var ccText: String
    @State private var subjectText: String
    @State private var isSending: Bool = false
    @State private var sendError: String?

    init(
        composer: EmailComposer,
        sender: EmailSender,
        store: EmailStore,
        onClose: @escaping () -> Void
    ) {
        self.composer = composer
        self.sender = sender
        self.store = store
        self.onClose = onClose
        self._bodyText = State(initialValue: composer.bodyPlain)
        self._toText = State(initialValue: composer.to.map { $0.mailboxString }.joined(separator: ", "))
        self._ccText = State(initialValue: composer.cc.map { $0.mailboxString }.joined(separator: ", "))
        self._subjectText = State(initialValue: composer.subject)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("To") { TextField("", text: $toText) }
                Section("Cc") { TextField("", text: $ccText) }
                Section("Subject") { TextField("", text: $subjectText) }
                Section("Body") {
                    TextEditor(text: $bodyText).frame(minHeight: 200)
                }
                if let err = sendError {
                    Section { Text(err).foregroundStyle(.red) }
                }
            }
            .navigationTitle("Compose")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { onClose() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Send") { send() }
                        .disabled(isSending)
                }
            }
        }
    }

    private func send() {
        isSending = true
        sendError = nil
        Task {
            do {
                let addresses = parseAddresses(toText)
                let ccs = parseAddresses(ccText)
                let final = composer
                    .setTo(addresses)
                    .setCC(ccs)
                    .setSubject(subjectText)
                    .setBody(bodyText)
                let draft = final.build()
                let result = try await sender.send(draft, original: nil)
                switch result {
                case .routedToSystemShare: onClose()
                case .savedAsDraft: sendError = "Cancelled; draft saved."; isSending = false
                }
            } catch {
                sendError = String(describing: error)
                isSending = false
            }
        }
    }

    private func parseAddresses(_ text: String) -> [EmailAddress] {
        let entries = text.split(separator: ",").map(String.init)
        return entries.compactMap { entry in
            let trimmed = entry.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return nil }
            if let open = trimmed.firstIndex(of: "<"),
               let close = trimmed.firstIndex(of: ">"),
               close > open {
                let name = String(trimmed[trimmed.startIndex..<open])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let email = String(trimmed[trimmed.index(after: open)..<close])
                return EmailAddress(name: name.isEmpty ? nil : name, email: email)
            }
            if trimmed.contains("@") {
                return EmailAddress(email: trimmed)
            }
            return nil
        }
    }
}
#endif
